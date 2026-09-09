//! Alloca promotion (mem2reg): lifts a non-escaping stack slot into SSA values so every
//! downstream pass (gvn, simplify, strength, licm) can see through it, instead of leaving
//! the value opaque behind a load/store pair. The promotion target is block parameters,
//! which this IR already speaks natively, so a slot's stored value threads through block
//! params (the block-param analog of phi insertion) rather than through memory.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const cfg_mod = @import("cfg.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "mem2reg", .run = run };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;

    const promotable = try findPromotable(allocator, func);
    defer allocator.free(promotable);

    // Index the promotable slots 0..num_slots and record each slot's element type.
    var slot_of_value = try allocator.alloc(?u32, func.valueCount());
    defer allocator.free(slot_of_value);
    @memset(slot_of_value, null);
    var slot_elem: std.ArrayList(ir.types.Type) = .empty;
    defer slot_elem.deinit(allocator);
    // Iterate LIVE block instructions, not the whole instruction pool: a prior mem2reg run
    // (an earlier fixpoint iteration) leaves its promoted allocas in the pool but removes them
    // from every block. Seeding from the pool would re-discover such a dead alloca and re-promote
    // it forever, so the pass would never report "no change" and the fixpoint would not converge.
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .alloca => |al| {
                    const r = func.instResult(inst).?;
                    if (promotable[@intFromEnum(r)]) {
                        slot_of_value[@intFromEnum(r)] = @intCast(slot_elem.items.len);
                        try slot_elem.append(allocator, al.elem);
                    }
                },
                else => {},
            }
        }
    }
    const num_slots = slot_elem.items.len;
    if (num_slots == 0) return false;

    var b = Builder{
        .allocator = allocator,
        .func = func,
        .cfg = try cfg_mod.build(allocator, func),
        .slot_of_value = slot_of_value,
        .slot_elem = slot_elem.items,
        .num_slots = num_slots,
    };
    defer b.deinit();
    try b.buildTables();
    try b.rewrite();
    try b.collapseTrivialPhis();
    return true;
}

/// SSA construction over the promotable slots, adapted from Braun et al. "Simple and Efficient
/// Construction of SSA Form" (CC 2013) onto this IR's block parameters. `local_end`/`writes` are
/// precomputed per (slot, block) in `buildTables`, so `readEndOfBlock` never depends on the order
/// blocks are visited, which is what lets loops resolve without an explicit dominance-frontier pass.
const Builder = struct {
    allocator: std.mem.Allocator,
    func: *Function,
    cfg: cfg_mod.Cfg,
    slot_of_value: []const ?u32,
    slot_elem: []const ir.types.Type,
    num_slots: usize,

    /// The last value stored to (slot, block) within that block, valid only where `writes` is set.
    local_end: []Value = &.{},
    writes: []bool = &.{},
    /// The value flowing into a block for a slot (a fresh block param or a forwarded value), memoized.
    entry_memo: []?Value = &.{},
    /// A defensive backstop only. `readEntry` now memoizes a block parameter for every block with a
    /// predecessor before it recurses, so a control-flow cycle re-entering a block always finds that
    /// memoized parameter rather than this guard. It can therefore only fire on a truly pathological
    /// input (a block reached before its own memo is installed), where it recovers as an undef zero.
    computing: []bool = &.{},
    undef_zero: []?Value = &.{},

    fn deinit(self: *Builder) void {
        self.cfg.deinit(self.allocator);
        self.allocator.free(self.local_end);
        self.allocator.free(self.writes);
        self.allocator.free(self.entry_memo);
        self.allocator.free(self.computing);
        self.allocator.free(self.undef_zero);
    }

    fn idx(self: *const Builder, slot: usize, block: usize) usize {
        return slot * self.cfg.blockCount() + block;
    }

    /// The promotable slot index a value names, or null. `slot_of_value` is sized to the value
    /// count when the pass started. The SSA construction then SYNTHESIZES new values (block
    /// parameters), whose indices fall at or beyond that length. Such a value is never a
    /// promotable alloca, so it names no slot. A promoted POINTER slot produces one of these
    /// values, and a later load or store THROUGH that pointer reads it here, so this bounds
    /// check is load-bearing, not defensive.
    fn slotOf(self: *const Builder, v: Value) ?u32 {
        const i = @intFromEnum(v);
        if (i >= self.slot_of_value.len) return null;
        return self.slot_of_value[i];
    }

    /// Phase 1: record each block's last store per slot. Independent of any traversal order.
    fn buildTables(self: *Builder) pass.Error!void {
        const n = self.num_slots * self.cfg.blockCount();
        self.local_end = try self.allocator.alloc(Value, n);
        // Cells without a store are never read (gated on `writes`), but `repointCaches` scans the
        // whole array, so initialize to a valid Value to avoid reading uninitialized memory.
        @memset(self.local_end, @enumFromInt(0));
        self.writes = try self.allocator.alloc(bool, n);
        @memset(self.writes, false);
        self.entry_memo = try self.allocator.alloc(?Value, n);
        @memset(self.entry_memo, null);
        self.computing = try self.allocator.alloc(bool, n);
        @memset(self.computing, false);
        self.undef_zero = try self.allocator.alloc(?Value, self.num_slots);
        @memset(self.undef_zero, null);

        for (0..self.cfg.blockCount()) |bi| {
            for (self.func.blockInsts(@enumFromInt(bi))) |inst| {
                switch (self.func.opcode(inst)) {
                    .store => |st| if (self.slotOf(st.ptr)) |s| {
                        self.local_end[self.idx(s, bi)] = st.value;
                        self.writes[self.idx(s, bi)] = true;
                    },
                    else => {},
                }
            }
        }
    }

    /// The slot's value on exit from `block`: its last local store, else the value flowing in.
    fn readEndOfBlock(self: *Builder, slot: usize, block: usize) pass.Error!Value {
        if (self.writes[self.idx(slot, block)]) return self.local_end[self.idx(slot, block)];
        return self.readEntry(slot, block);
    }

    /// The slot's value on entry to `block`: a forwarded single-predecessor value, or a fresh block
    /// parameter merging the predecessors (the block-param analog of phi insertion). Memoized.
    ///
    /// Every block with at least one predecessor introduces its block parameter and memoizes it
    /// *before* reading the predecessors. This is the Braun et al. cycle break: a control-flow cycle
    /// re-entering this block (a single-predecessor chain around a loop, not only a diamond join)
    /// then resolves to this parameter instead of hitting the `computing` guard and fabricating an
    /// undef zero. After every incoming edge carries its argument, `removeTrivialPhi` collapses the
    /// parameter when it has a single distinct incoming value, which restores plain single-pred
    /// forwarding for acyclic code and leaves only genuine merges (and loop-carried parameters).
    fn readEntry(self: *Builder, slot: usize, block: usize) pass.Error!Value {
        if (self.entry_memo[self.idx(slot, block)]) |v| return v;
        if (self.computing[self.idx(slot, block)]) return self.undefZero(slot);
        self.computing[self.idx(slot, block)] = true;
        defer self.computing[self.idx(slot, block)] = false;

        const preds = self.cfg.predecessors(block);
        if (preds.len == 0) {
            const z = try self.undefZero(slot);
            self.entry_memo[self.idx(slot, block)] = z;
            return z;
        }
        const phi = try self.func.appendBlockParam(@enumFromInt(block), self.slot_elem[slot]);
        self.entry_memo[self.idx(slot, block)] = phi;
        try self.addEdgeArgs(slot, @enumFromInt(block));
        const resolved = try self.removeTrivialPhi(slot, block, phi);
        self.entry_memo[self.idx(slot, block)] = resolved;
        return resolved;
    }

    /// Braun et al. trivial-phi elimination. The block parameter `phi` on `block` is trivial when,
    /// across its incoming edge arguments, it has at most one distinct value other than itself (a
    /// self-reference is a back edge feeding the parameter its own value and does not count). A
    /// trivial phi equals that unique value `v` (or an undef zero when it had no non-self operand at
    /// all, e.g. an unreachable self-cycle): every use is forwarded to `v`, the parameter is deleted
    /// and its now-orphaned argument dropped from every incoming edge, and `v` is returned. Two or
    /// more distinct non-self operands make it a genuine merge, returned unchanged.
    ///
    /// This handles a phi trivial at the moment it is built. A phi that becomes trivial only LATER
    /// (its back-edge argument turned into a self-reference after an inner single-pred phi collapsed
    /// onto it) is a loop-carried identity parameter threading a loop-invariant slot around the back
    /// edge. `collapseTrivialPhis` sweeps those away after construction, so the function does not
    /// carry thousands of identity parameters into the backend (each one is otherwise an edge move
    /// across every loop iteration).
    fn removeTrivialPhi(self: *Builder, slot: usize, block: usize, phi: Value) pass.Error!Value {
        const blk: Block = @enumFromInt(block);
        const params = self.func.blockParams(blk);
        const pidx = for (params, 0..) |p, i| {
            if (p == phi) break i;
        } else return phi; // no longer a parameter (already collapsed): nothing to do

        const c = self.classifyPhi(blk, pidx, phi);
        if (c.distinct_two) return phi; // a genuine merge of >=2 values: keep the block parameter

        const v: Value = c.same orelse try self.undefZero(slot);
        self.func.replaceAllUses(phi, v);
        self.repointCaches(phi, v);
        try self.dropParamAndEdgeArgs(blk, pidx);
        return v;
    }

    /// The classification of a block parameter's incoming edge arguments: `same` is the single
    /// distinct value other than the parameter itself (null when every operand is a self-reference),
    /// and `distinct_two` is set once two different non-self values appear. Only PREDECESSORS carry
    /// an edge into `blk`, so scan those, not every block. The predecessor list is built in ascending
    /// source order with a block that has two edges to `blk` appearing on consecutive entries, so
    /// skipping a repeat visits each source once (its inner scan then handles both of its edges),
    /// matching the argument order `addEdgeArgs` wrote.
    fn classifyPhi(self: *Builder, blk: Block, pidx: usize, phi: Value) struct { same: ?Value, distinct_two: bool } {
        var same: ?Value = null;
        var distinct_two = false;
        const consider = struct {
            fn f(op: Value, self_phi: Value, s: *?Value, two: *bool) void {
                if (op == self_phi) return; // self-reference: does not count toward triviality
                if (s.*) |prev| {
                    if (prev != op) two.* = true;
                } else s.* = op;
            }
        }.f;
        var prev_si: i64 = -1;
        for (self.cfg.predecessors(@intFromEnum(blk))) |si| {
            if (@as(i64, si) == prev_si) continue;
            prev_si = si;
            const source: Block = @enumFromInt(si);
            for (self.func.blockInsts(source)) |inst| {
                switch (self.func.opcode(inst)) {
                    .@"if" => |cf| {
                        if (cf.then.target == blk) consider(self.func.blockArgs(cf.then)[pidx], phi, &same, &distinct_two);
                        if (cf.@"else".target == blk) consider(self.func.blockArgs(cf.@"else")[pidx], phi, &same, &distinct_two);
                    },
                    else => {},
                }
            }
            if (self.func.terminator(source)) |term| switch (term) {
                .jump => |j| if (j.target == blk) consider(self.func.blockArgs(j)[pidx], phi, &same, &distinct_two),
                .ret => {},
            };
        }
        return .{ .same = same, .distinct_two = distinct_two };
    }

    /// After SSA construction, collapse every block parameter that has become trivial - one whose
    /// incoming edges carry at most one distinct value other than itself. `readEntry` only removes a
    /// phi that is trivial at the moment it is built; a phi that becomes trivial LATER (an inner phi
    /// it merged collapsed onto one value) is left in place, and a large function with many loops
    /// accumulates thousands of these identity parameters. They bloat every later pass and, since the
    /// backend threads each one across every edge, the generated code. This runs to a fixpoint:
    /// collapsing one parameter can make a parameter that referenced it trivial, so it repeats until a
    /// full sweep collapses nothing. A parameter with no non-self operand (an unreachable self-cycle)
    /// is left alone. `replaceAllUses` fixes every use, including the edge arguments feeding other
    /// parameters, so a later sweep sees the exposed triviality.
    fn collapseTrivialPhis(self: *Builder) pass.Error!void {
        var changed = true;
        while (changed) {
            changed = false;
            for (0..self.cfg.blockCount()) |bi| {
                const blk: Block = @enumFromInt(bi);
                var pi: usize = 0;
                while (pi < self.func.blockParams(blk).len) {
                    const param = self.func.blockParams(blk)[pi];
                    const c = self.classifyPhi(blk, pi, param);
                    if (c.distinct_two or c.same == null) {
                        pi += 1; // a real merge, or an unreachable self-cycle: keep it
                        continue;
                    }
                    self.func.replaceAllUses(param, c.same.?);
                    try self.dropParamAndEdgeArgs(blk, pi);
                    changed = true;
                    // The parameter at `pi` is gone and the next one shifted into its place, so do
                    // not advance `pi`.
                }
            }
        }
    }

    /// Repoint every cached reference to `from` (a value just folded into `to`, its defining
    /// instruction dropped and every use existing *at this moment* already fixed by
    /// `replaceAllUses`) across both of `Builder`'s value caches: `entry_memo` (a block's resolved
    /// entry value, which `removeTrivialPhi` can collapse straight to a raw predecessor value) and
    /// `local_end` (Phase 1's snapshot of each block's last store, which can itself be a promotable
    /// load's result). Neither cache is part of the IR, so `replaceAllUses` cannot see into it - a
    /// cache entry equal to `from` is only patched here. This matters because both caches are read
    /// lazily and repeatedly as `rewrite` walks blocks in index order: `readEntry`/`readEndOfBlock`
    /// can hand a cached value back out to build a *new* edge argument or replacement long after
    /// `from`'s own promotion, and that new reference must land on the live `to`, not the now-dead
    /// `from`, regardless of which block order exposed the read. Called both when `removeTrivialPhi`
    /// collapses a block parameter and when `rewrite` promotes a load.
    fn repointCaches(self: *Builder, from: Value, to: Value) void {
        for (self.entry_memo) |*m| {
            if (m.*) |mv| {
                if (mv == from) m.* = to;
            }
        }
        for (self.local_end) |*m| {
            if (m.* == from) m.* = to;
        }
    }

    /// Delete parameter `pidx` from `block` and drop the argument at that same position from every
    /// incoming edge, keeping every edge's arity matched to the block's parameter count. The edge
    /// scan mirrors `addEdgeArgs`/`appendArg` so positions stay consistent.
    fn dropParamAndEdgeArgs(self: *Builder, block: Block, pidx: usize) pass.Error!void {
        const params = self.func.blockParams(block);
        var np: std.ArrayList(Value) = .empty;
        defer np.deinit(self.allocator);
        for (params, 0..) |p, i| if (i != pidx) try np.append(self.allocator, p);
        try self.func.setBlockParams(block, np.items);

        var prev_si: i64 = -1;
        for (self.cfg.predecessors(@intFromEnum(block))) |si| {
            if (@as(i64, si) == prev_si) continue;
            prev_si = si;
            const source: Block = @enumFromInt(si);
            for (self.func.blockInsts(source)) |inst| {
                switch (self.func.opcode(inst)) {
                    .@"if" => |cf| {
                        if (cf.then.target == block) try self.dropEdgeArg(.{ .if_then = inst }, pidx);
                        if (cf.@"else".target == block) try self.dropEdgeArg(.{ .if_else = inst }, pidx);
                    },
                    else => {},
                }
            }
            if (self.func.terminator(source)) |term| switch (term) {
                .jump => |j| if (j.target == block) try self.dropEdgeArg(.{ .term = source }, pidx),
                .ret => {},
            };
        }
    }

    /// Remove the argument at position `pidx` from a single edge's argument list, mirroring the
    /// per-edge write in `appendArg`.
    fn dropEdgeArg(self: *Builder, edge: EdgeRef, pidx: usize) pass.Error!void {
        const old = switch (edge) {
            .term => self.func.blockArgs(self.func.terminator(edge.term).?.jump),
            .if_then => self.func.blockArgs(self.func.opcode(edge.if_then).@"if".then),
            .if_else => self.func.blockArgs(self.func.opcode(edge.if_else).@"if".@"else"),
        };
        var buf: std.ArrayList(Value) = .empty;
        defer buf.deinit(self.allocator);
        for (old, 0..) |a, i| if (i != pidx) try buf.append(self.allocator, a);
        const list = try self.func.internValues(buf.items);
        switch (edge) {
            .term => self.func.terminatorPtr(edge.term).*.?.jump.args = list,
            .if_then => self.func.opcodeMut(edge.if_then).@"if".then.args = list,
            .if_else => self.func.opcodeMut(edge.if_else).@"if".@"else".args = list,
        }
    }

    /// Append, to every control-flow edge targeting `block`, the source block's exit value for the
    /// slot, matching the block parameter just appended. Arity stays consistent because each edge
    /// gains exactly one argument per new parameter.
    fn addEdgeArgs(self: *Builder, slot: usize, block: Block) pass.Error!void {
        var prev_si: i64 = -1;
        for (self.cfg.predecessors(@intFromEnum(block))) |si| {
            if (@as(i64, si) == prev_si) continue;
            prev_si = si;
            const source: Block = @enumFromInt(si);
            for (self.func.blockInsts(source)) |inst| {
                switch (self.func.opcode(inst)) {
                    .@"if" => |cf| {
                        if (cf.then.target == block) try self.appendArg(.{ .if_then = inst }, slot, si);
                        if (cf.@"else".target == block) try self.appendArg(.{ .if_else = inst }, slot, si);
                    },
                    else => {},
                }
            }
            if (self.func.terminator(source)) |term| switch (term) {
                .jump => |j| if (j.target == block) try self.appendArg(.{ .term = source }, slot, si),
                .ret => {},
            };
        }
    }

    const EdgeRef = union(enum) { term: Block, if_then: Inst, if_else: Inst };

    fn appendArg(self: *Builder, edge: EdgeRef, slot: usize, source: usize) pass.Error!void {
        const val = try self.readEndOfBlock(slot, source);
        const old = switch (edge) {
            .term => self.func.blockArgs(self.func.terminator(edge.term).?.jump),
            .if_then => self.func.blockArgs(self.func.opcode(edge.if_then).@"if".then),
            .if_else => self.func.blockArgs(self.func.opcode(edge.if_else).@"if".@"else"),
        };
        var buf: std.ArrayList(Value) = .empty;
        defer buf.deinit(self.allocator);
        try buf.appendSlice(self.allocator, old);
        try buf.append(self.allocator, val);
        const list = try self.func.internValues(buf.items);
        switch (edge) {
            .term => self.func.terminatorPtr(edge.term).*.?.jump.args = list,
            .if_then => self.func.opcodeMut(edge.if_then).@"if".then.args = list,
            .if_else => self.func.opcodeMut(edge.if_else).@"if".@"else".args = list,
        }
    }

    /// A typed zero standing in for a read of an uninitialized slot (undefined behavior in the
    /// source). Materialized once per slot at the entry block and cached.
    fn undefZero(self: *Builder, slot: usize) pass.Error!Value {
        if (self.undef_zero[slot]) |v| return v;
        const ty = self.slot_elem[slot];
        const op: ir.function.Opcode = switch (self.func.types.type_kind(ty)) {
            .float => .{ .fconst = 0 },
            else => .{ .iconst = 0 },
        };
        const v = try self.func.appendInst(@enumFromInt(0), ty, op);
        self.undef_zero[slot] = v;
        return v;
    }

    /// Phase 2: rewrite loads to the resolved SSA value and drop every promoted alloca/load/store.
    fn rewrite(self: *Builder) pass.Error!void {
        var current = try self.allocator.alloc(?Value, self.num_slots);
        defer self.allocator.free(current);

        for (0..self.cfg.blockCount()) |bi| {
            @memset(current, null); // slot values reset at each block entry, computed lazily
            var kept: std.ArrayList(Inst) = .empty;
            defer kept.deinit(self.allocator);
            for (self.func.blockInsts(@enumFromInt(bi))) |inst| {
                switch (self.func.opcode(inst)) {
                    .alloca => {
                        const r = self.func.instResult(inst).?;
                        if (self.slotOf(r) == null) try kept.append(self.allocator, inst);
                    },
                    .store => |st| if (self.slotOf(st.ptr)) |s| {
                        current[s] = st.value;
                    } else try kept.append(self.allocator, inst),
                    .load => |ld| if (self.slotOf(ld.ptr)) |s| {
                        const val = current[s] orelse blk: {
                            const v = try self.readEntry(s, bi);
                            current[s] = v;
                            break :blk v;
                        };
                        const result = self.func.instResult(inst).?;
                        self.func.replaceAllUses(result, val);
                        // `local_end`/`entry_memo` may have snapshotted this very load's result
                        // (e.g. `store (load a_slot), r_slot`, Phase 1's `local_end[r_slot, block]`)
                        // before this promotion dropped it. Repoint so a later cache read (an
                        // `addEdgeArgs` for a merge block visited after this one) sources the
                        // resolved value instead of a now-dead load, independent of block order.
                        self.repointCaches(result, val);
                    } else try kept.append(self.allocator, inst),
                    else => try kept.append(self.allocator, inst),
                }
            }
            try self.func.setBlockInsts(@enumFromInt(bi), kept.items);
        }
    }
};

/// A bitmap over values: true for each `alloca` result that is promotable, i.e. its address
/// never escapes (only ever a load/store `ptr`) and its element is a scalar (int/float/bool/ptr),
/// so a load/store maps to exactly one SSA value. The caller owns the slice.
fn findPromotable(allocator: std.mem.Allocator, func: *const Function) pass.Error![]bool {
    const promotable = try allocator.alloc(bool, func.valueCount());
    errdefer allocator.free(promotable);
    @memset(promotable, false);

    // Seed with every scalar-element alloca. Iterate LIVE block instructions, not the pool: a
    // promoted alloca left over from an earlier fixpoint iteration still sits in the pool but is
    // gone from every block, and re-seeding it would make the pass re-promote a dead slot forever.
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .alloca => |al| if (isScalar(func, al.elem)) {
                    promotable[@intFromEnum(func.instResult(inst).?)] = true;
                },
                else => {},
            }
        }
    }
    // Clear any whose address escapes: used anywhere but as a load/store `ptr`.
    markEscapes(func, promotable);
    return promotable;
}

/// Clear `promotable` for any slot whose address is used as anything but the `ptr` of a
/// load/store: a store value, a call/return operand, arithmetic, an aggregate field, or a
/// branch argument all let the address (and thus aliasing) escape this analysis.
fn markEscapes(func: *const Function, promotable: []bool) void {
    const esc = struct {
        fn hit(p: []bool, v: Value) void {
            if (@intFromEnum(v) < p.len) p[@intFromEnum(v)] = false;
        }
    }.hit;
    for (0..func.instCount()) |i| {
        switch (func.opcode(@enumFromInt(i))) {
            // ld.ptr is the sanctioned use - UNLESS the load is `volatile` (SM9 Plan 2 Task 4):
            // a volatile access must observably hit memory, so its alloca cannot be promoted
            // to an SSA value and the load must stay in the IR.
            .load => |ld| if (ld.@"volatile") esc(promotable, ld.ptr),
            // ptr is fine, value escapes; a `volatile` store likewise pins its alloca unpromotable.
            .store => |st| {
                esc(promotable, st.value);
                if (st.@"volatile") esc(promotable, st.ptr);
            },
            .alloca, .iconst, .fconst, .fconst128, .global_addr => {},
            .arith => |a| {
                esc(promotable, a.lhs);
                esc(promotable, a.rhs);
            },
            .arith_imm => |a| esc(promotable, a.lhs),
            .icmp => |c| {
                esc(promotable, c.lhs);
                esc(promotable, c.rhs);
            },
            .select => |s| {
                esc(promotable, s.cond);
                esc(promotable, s.then);
                esc(promotable, s.@"else");
            },
            .extract => |e| esc(promotable, e.aggregate),
            .convert => |cv| esc(promotable, cv.value),
            .unary => |u| esc(promotable, u.value),
            .prefetch => |pf| esc(promotable, pf.ptr),
            // SM12 T3: unlike `load`/`store`'s `ptr` (a sanctioned dereference), `list` is the
            // `va_list` OBJECT's raw address, captured for a later backend expansion to do its
            // own pointer arithmetic on (e.g. writing the next-argument pointer into it) - the
            // same reason `.addrof`-taking uses always escape. An alloca feeding one of these
            // must keep its real stack storage, never be promoted away as a bare SSA value.
            .va_start => |vs| esc(promotable, vs.list),
            .va_arg => |va| esc(promotable, va.list),
            .va_end => |ve| esc(promotable, ve.list),
            .dot => |d| {
                esc(promotable, d.acc);
                esc(promotable, d.a);
                esc(promotable, d.b);
            },
            .matmul => |mm| {
                esc(promotable, mm.a);
                esc(promotable, mm.b);
                esc(promotable, mm.c);
            },
            .struct_new => |sn| for (func.valueList(sn.fields)) |f| esc(promotable, f),
            .call => |c| {
                for (func.valueList(c.args)) |arg| esc(promotable, arg);
                if (c.ret_dest) |rd| esc(promotable, rd); // SM14 M4d-c T1: the call writes memory through the dest
            },
            .call_indirect => |c| {
                esc(promotable, c.target);
                for (func.valueList(c.args)) |arg| esc(promotable, arg);
                if (c.ret_dest) |rd| esc(promotable, rd); // SM14 M4d-c T1: the call writes memory through the dest
            },
            .@"if" => |cf| {
                esc(promotable, cf.cond);
                for (func.blockArgs(cf.then)) |arg| esc(promotable, arg);
                for (func.blockArgs(cf.@"else")) |arg| esc(promotable, arg);
            },
        }
    }
    for (0..func.blockCount()) |bi| {
        if (func.terminator(@enumFromInt(bi))) |term| switch (term) {
            .ret => |r| for (r.slice()) |vv| esc(promotable, vv),
            .jump => |j| for (func.blockArgs(j)) |arg| esc(promotable, arg),
        };
    }
}

/// True for the scalar types a single load/store round-trips as one SSA value.
fn isScalar(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .int, .float, .bool, .ptr => true,
        .vector, .array, .slice, .@"struct" => false,
    };
}

const testing = std.testing;

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

fn intTy(func: *Function, bits: u16, signedness: std.builtin.Signedness) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = signedness, .bits = bits } });
}

/// The promoted function is still well-formed SSA: every edge passes the right number and type of
/// arguments to its target's parameters, and every use is dominated by its definition.
fn expectVerifies(allocator: std.mem.Allocator, func: *const Function) !void {
    var diags = try ir.verify.verify(allocator, func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

/// No load, store, or alloca instruction survives in `block` (they were all promoted away).
fn expectNoMemoryInsts(func: *const Function, block: Block) !void {
    for (func.blockInsts(block)) |inst| {
        switch (func.opcode(inst)) {
            .alloca, .load, .store => return error.MemoryInstSurvived,
            else => {},
        }
    }
}

test "single-block store then load forwards the stored value" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(b, x, slot);
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(y) });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(x, func.terminator(b).?.ret.values[0]); // load became x
    try expectNoMemoryInsts(&func, b);
}

test "store in entry forwards across a jump to its single successor" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const ptr_t = try func.types.ptrGlobal();
    const b0 = try func.appendBlock();
    const b1 = try func.appendBlock();
    const x = try func.appendBlockParam(b0, t);
    const slot = try func.appendInst(b0, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(b0, x, slot);
    try func.setJump(b0, b1, &.{});
    const y = try func.appendInst(b1, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b1, .{ .ret = ir.function.Ret.one(y) });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(x, func.terminator(b1).?.ret.values[0]); // load forwarded across the edge
    try expectNoMemoryInsts(&func, b0);
    try expectNoMemoryInsts(&func, b1);
}

test "diamond store on each arm merges into a block parameter at the join" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const b0 = try func.appendBlock();
    const b1 = try func.appendBlock();
    const b2 = try func.appendBlock();
    const b3 = try func.appendBlock();
    const c = try func.appendBlockParam(b0, bool_t);
    const slot = try func.appendInst(b0, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendIf(b0, c, .{ .target = b1 }, .{ .target = b2 });
    const a = try func.appendInst(b1, t, .{ .iconst = 10 });
    try func.appendStore(b1, a, slot);
    try func.setJump(b1, b3, &.{});
    const bb = try func.appendInst(b2, t, .{ .iconst = 20 });
    try func.appendStore(b2, bb, slot);
    try func.setJump(b2, b3, &.{});
    const y = try func.appendInst(b3, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b3, .{ .ret = ir.function.Ret.one(y) });

    try testing.expect(try runOnce(allocator, &func));
    // A single new parameter on the join carries the merged value, and the ret returns it.
    const params = func.blockParams(b3);
    try testing.expectEqual(@as(usize, 1), params.len);
    try testing.expectEqual(params[0], func.terminator(b3).?.ret.values[0]);
    // Each arm passes its stored constant along its edge to the join.
    try testing.expectEqual(a, func.blockArgs(func.terminator(b1).?.jump)[0]);
    try testing.expectEqual(bb, func.blockArgs(func.terminator(b2).?.jump)[0]);
    for ([_]Block{ b0, b1, b2, b3 }) |blk| try expectNoMemoryInsts(&func, blk);
    try expectVerifies(allocator, &func);
}

test "loop-carried slot becomes a header parameter threaded around the back edge" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();
    // entry: slot = alloca; store 0; jump header
    const slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = t } });
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.appendStore(entry, zero, slot);
    try func.setJump(entry, header, &.{});
    // header: i = load slot; cond = i < 10; if cond -> body else exit
    const i = try func.appendInst(header, t, .{ .load = .{ .ptr = slot } });
    const ten = try func.appendInst(header, t, .{ .iconst = 10 });
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = ten } });
    try func.appendIf(header, cond, .{ .target = body }, .{ .target = exit });
    // body: iv = load slot; iv2 = iv + 1; store iv2; jump header
    const iv = try func.appendInst(body, t, .{ .load = .{ .ptr = slot } });
    const iv2 = try func.appendInst(body, t, .{ .arith_imm = .{ .op = .add, .lhs = iv, .imm = 1 } });
    try func.appendStore(body, iv2, slot);
    try func.setJump(body, header, &.{});
    // exit: r = load slot; ret r
    const r = try func.appendInst(exit, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(exit, .{ .ret = ir.function.Ret.one(r) });

    try testing.expect(try runOnce(allocator, &func));
    // The header carries the loop value as its one parameter.
    const params = func.blockParams(header);
    try testing.expectEqual(@as(usize, 1), params.len);
    const p = params[0];
    // The header comparison and the body increment both read that parameter, not memory.
    try testing.expectEqual(p, func.opcode(func.definingInst(cond).?).icmp.lhs);
    try testing.expectEqual(p, func.opcode(func.definingInst(iv2).?).arith_imm.lhs);
    // entry seeds it with 0, the back edge threads the incremented value, exit returns it.
    try testing.expectEqual(zero, func.blockArgs(func.terminator(entry).?.jump)[0]);
    try testing.expectEqual(iv2, func.blockArgs(func.terminator(body).?.jump)[0]);
    try testing.expectEqual(p, func.terminator(exit).?.ret.values[0]);
    for ([_]Block{ entry, header, body, exit }) |blk| try expectNoMemoryInsts(&func, blk);
    try expectVerifies(allocator, &func);
}

test "a slot whose address escapes to a call is left in memory" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(b, x, slot);
    try func.appendVoidCall(b, "escape", &.{slot}); // address leaves the function
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(y) });

    try testing.expect(!try runOnce(allocator, &func)); // not promotable, nothing changes
    var loads: usize = 0;
    for (func.blockInsts(b)) |inst| {
        if (func.opcode(inst) == .load) loads += 1;
    }
    try testing.expectEqual(@as(usize, 1), loads); // the load still reads memory
}

test "a slot with an aggregate element is not promoted" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const arr_t = try func.types.intern(.{ .array = .{ .len = 4, .elem = t } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = arr_t } });
    _ = try func.appendInst(b, arr_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    try testing.expect(!try runOnce(allocator, &func)); // aggregate slot stays in memory
}

test "loop-invariant slot through a single-pred chain re-entering the header is not corrupted to zero" {
    // int f(int n) { int i = 0; while (1) { if (i >= n) break; i += 1; } return i; }
    //
    // header has two predecessors (entry, cont) so mem2reg makes it a join and memoizes the
    // new block param for a slot BEFORE recursing into its predecessors. But `n` is read only
    // through the single-pred chain body -> cont (neither writes n_slot), so resolving n's
    // value on the cont->header back edge walks cont (single-pred: body) -> body (single-pred:
    // header) -> header again. That inner revisit of body re-enters readEntry(n_slot, body)
    // while the outer call for the very same (slot, block) is still marked `computing`, so it
    // hits the cycle guard and fabricates a fresh `iconst 0` instead of resolving to header's
    // own n parameter. That spurious zero then gets threaded onto the back edge, corrupting the
    // invariant slot.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const cont = try func.appendBlock();
    const exit = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);

    // entry: i_slot = alloca; n_slot = alloca; store 0 -> i_slot; store n -> n_slot; jump header
    const i_slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = t } });
    const n_slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = t } });
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.appendStore(entry, zero, i_slot);
    try func.appendStore(entry, n, n_slot);
    try func.setJump(entry, header, &.{});

    // header: while (1) -> body, else exit (the else edge is never taken at runtime, but it is
    // a real CFG edge so header has exactly one join-worthy predecessor pair: entry and cont).
    const one_h = try func.appendInst(header, t, .{ .iconst = 1 });
    try func.appendIf(header, one_h, .{ .target = body }, .{ .target = exit });

    // body: iv = load i_slot; nv = load n_slot; if (iv >= nv) exit else cont
    const iv = try func.appendInst(body, t, .{ .load = .{ .ptr = i_slot } });
    const nv = try func.appendInst(body, t, .{ .load = .{ .ptr = n_slot } });
    const ge = try func.appendInst(body, bool_t, .{ .icmp = .{ .op = .ge, .lhs = iv, .rhs = nv } });
    try func.appendIf(body, ge, .{ .target = exit }, .{ .target = cont });

    // cont: iv2 = load i_slot; inc = iv2 + 1; store inc -> i_slot; jump header (the back edge)
    const iv2 = try func.appendInst(cont, t, .{ .load = .{ .ptr = i_slot } });
    const inc = try func.appendInst(cont, t, .{ .arith_imm = .{ .op = .add, .lhs = iv2, .imm = 1 } });
    try func.appendStore(cont, inc, i_slot);
    try func.setJump(cont, header, &.{});

    // exit: r = load i_slot; ret r
    const r = try func.appendInst(exit, t, .{ .load = .{ .ptr = i_slot } });
    func.setTerminator(exit, .{ .ret = ir.function.Ret.one(r) });

    try testing.expect(try runOnce(allocator, &func));

    // The crux: `nv` (the loaded value of the invariant slot, compared in `iv >= nv`) must resolve
    // to the function's own `n` parameter, NOT a freshly materialized zero constant that the
    // cycle-guard bug would fabricate on the back edge. `collapseTrivialPhis` folds the loop-carried
    // identity parameter that construction threaded n through straight back to n, so the comparison
    // reads n directly. A `0` here (or any other value) would prove the invariant was corrupted.
    try testing.expectEqual(n, func.opcode(func.definingInst(ge).?).icmp.rhs);
    try expectVerifies(allocator, &func);
}

/// Every argument on every control-flow edge (`.jump` and `.@"if"` then/else) must reference a
/// value that is still *live*: a surviving instruction result (its defining instruction is still
/// in some block's instruction list) or a parameter still present on its block. A promotion that
/// drops an instruction but leaves an edge argument pointing at its now-orphaned result produces a
/// dangling SSA edge, which this catches directly (independent of the full verifier).
fn assertNoDanglingEdgeArgs(allocator: std.mem.Allocator, func: *const Function) !void {
    var live = try allocator.alloc(bool, func.valueCount());
    defer allocator.free(live);
    @memset(live, false);
    for (0..func.blockCount()) |bi| {
        const blk: Block = @enumFromInt(bi);
        for (func.blockParams(blk)) |p| live[@intFromEnum(p)] = true;
        for (func.blockInsts(blk)) |inst| {
            if (func.instResult(inst)) |r| live[@intFromEnum(r)] = true;
        }
    }
    for (0..func.blockCount()) |bi| {
        const blk: Block = @enumFromInt(bi);
        for (func.blockInsts(blk)) |inst| {
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                for (func.blockArgs(cf.then)) |a| if (!live[@intFromEnum(a)]) return error.DanglingEdgeArg;
                for (func.blockArgs(cf.@"else")) |a| if (!live[@intFromEnum(a)]) return error.DanglingEdgeArg;
            }
        }
        if (func.terminator(blk)) |term| switch (term) {
            .jump => |j| for (func.blockArgs(j)) |a| if (!live[@intFromEnum(a)]) return error.DanglingEdgeArg,
            .ret => {},
        };
    }
}

test "diamond storing a promoted load into a slot then reading it after the merge does not dangle" {
    // int f(int a) { int r = 0; if (a > 0) { r = a; } else { r = -a; } return r; }
    //
    // The then arm's `r = a` is `t = load a_slot; store t -> r_slot`, i.e. it stores the RESULT
    // of a promotable load. mem2reg's Phase 1 records local_end[r_slot, then] = t (the load
    // result). Because the then block has a LOWER block index than the merge block, Phase 2
    // promotes and drops the load `t` (running replaceAllUses(t -> a's value)) BEFORE the merge
    // block's join parameter and its incoming edge arguments are ever materialized. When the
    // merge param is finally built, addEdgeArgs sources the then->merge argument straight from the
    // stale local_end[r_slot, then] = t, a value that is by now dead (its load was dropped and the
    // covering replaceAllUses already ran and cannot reach this not-yet-created argument). The
    // else arm is unaffected: it stores `nt = 0 - a`, a surviving sub, not a promoted-away load.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();

    // Creation order is load-bearing: then (block 1) MUST precede merge (block 3) so Phase 2
    // visits and promotes the then-arm load before the merge join parameter is constructed.
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();

    // entry: store a -> a_slot; store 0 -> r_slot; if a > 0 then->then_b else->else_b
    const a = try func.appendBlockParam(entry, t);
    const a_slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = t } });
    const r_slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(entry, a, a_slot);
    const zero0 = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.appendStore(entry, zero0, r_slot);
    const zero_c = try func.appendInst(entry, t, .{ .iconst = 0 });
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = zero_c } });
    try func.appendIf(entry, cond, .{ .target = then_b }, .{ .target = else_b });

    // then: t = load a_slot; store t -> r_slot; jump merge
    const tl = try func.appendInst(then_b, t, .{ .load = .{ .ptr = a_slot } });
    try func.appendStore(then_b, tl, r_slot);
    try func.setJump(then_b, merge, &.{});

    // else: t2 = load a_slot; nt = 0 - t2; store nt -> r_slot; jump merge
    const t2 = try func.appendInst(else_b, t, .{ .load = .{ .ptr = a_slot } });
    const z2 = try func.appendInst(else_b, t, .{ .iconst = 0 });
    const nt = try func.appendInst(else_b, t, .{ .arith = .{ .op = .sub, .lhs = z2, .rhs = t2 } });
    try func.appendStore(else_b, nt, r_slot);
    try func.setJump(else_b, merge, &.{});

    // merge: r = load r_slot; ret r
    const r = try func.appendInst(merge, t, .{ .load = .{ .ptr = r_slot } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(r) });

    try testing.expect(try runOnce(allocator, &func));

    // The crux: no edge argument may reference a dropped (promoted-away) value. This catches the
    // dangling then->merge argument directly.
    try assertNoDanglingEdgeArgs(allocator, &func);
    try expectVerifies(allocator, &func);

    // The merge join carries one parameter; the ret returns it.
    const params = func.blockParams(merge);
    try testing.expectEqual(@as(usize, 1), params.len);
    try testing.expectEqual(params[0], func.terminator(merge).?.ret.values[0]);
    // The then->merge edge must forward `a` (the promoted load's resolved value), not a dead value.
    try testing.expectEqual(a, func.blockArgs(func.terminator(then_b).?.jump)[0]);
    // The else->merge edge forwards the surviving negation.
    const else_arg = func.blockArgs(func.terminator(else_b).?.jump)[0];
    try testing.expect(std.meta.activeTag(func.opcode(func.definingInst(else_arg).?)) == .arith);
    for ([_]Block{ entry, then_b, else_b, merge }) |blk| try expectNoMemoryInsts(&func, blk);
}

test "two independent scalar slots both promote" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const sa = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    const sb = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(b, x, sa);
    try func.appendStore(b, y, sb);
    const la = try func.appendInst(b, t, .{ .load = .{ .ptr = sa } });
    const lb = try func.appendInst(b, t, .{ .load = .{ .ptr = sb } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = la, .rhs = lb } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    try testing.expect(try runOnce(allocator, &func));
    // sum now adds the two stored values directly.
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(x, add.lhs);
    try testing.expectEqual(y, add.rhs);
    try expectNoMemoryInsts(&func, b);
}
