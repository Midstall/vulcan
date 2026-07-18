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
    for (0..func.instCount()) |i| {
        const inst: Inst = @enumFromInt(i);
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
    /// Guards `readEntry` against re-entering a block whose entry is still being computed (a cycle
    /// of single-predecessor blocks, only reachable on malformed input): recover as an undef zero.
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

    /// Phase 1: record each block's last store per slot. Independent of any traversal order.
    fn buildTables(self: *Builder) pass.Error!void {
        const n = self.num_slots * self.cfg.blockCount();
        self.local_end = try self.allocator.alloc(Value, n);
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
                    .store => |st| if (self.slot_of_value[@intFromEnum(st.ptr)]) |s| {
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
    fn readEntry(self: *Builder, slot: usize, block: usize) pass.Error!Value {
        if (self.entry_memo[self.idx(slot, block)]) |v| return v;
        if (self.computing[self.idx(slot, block)]) return self.undefZero(slot);
        self.computing[self.idx(slot, block)] = true;
        defer self.computing[self.idx(slot, block)] = false;

        const preds = self.cfg.predecessors(block);
        var result: Value = undefined;
        if (preds.len == 0) {
            result = try self.undefZero(slot);
            self.entry_memo[self.idx(slot, block)] = result;
        } else if (preds.len == 1) {
            result = try self.readEndOfBlock(slot, preds[0]);
            self.entry_memo[self.idx(slot, block)] = result;
        } else {
            // A join: introduce a block parameter, break cycles by memoizing it before reading the
            // predecessors, then thread the slot's exit value onto every incoming edge.
            const phi = try self.func.appendBlockParam(@enumFromInt(block), self.slot_elem[slot]);
            self.entry_memo[self.idx(slot, block)] = phi;
            try self.addEdgeArgs(slot, @enumFromInt(block));
            result = phi;
        }
        return result;
    }

    /// Append, to every control-flow edge targeting `block`, the source block's exit value for the
    /// slot, matching the block parameter just appended. Arity stays consistent because each edge
    /// gains exactly one argument per new parameter.
    fn addEdgeArgs(self: *Builder, slot: usize, block: Block) pass.Error!void {
        for (0..self.cfg.blockCount()) |si| {
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
                        if (self.slot_of_value[@intFromEnum(r)] == null) try kept.append(self.allocator, inst);
                    },
                    .store => |st| if (self.slot_of_value[@intFromEnum(st.ptr)]) |s| {
                        current[s] = st.value;
                    } else try kept.append(self.allocator, inst),
                    .load => |ld| if (self.slot_of_value[@intFromEnum(ld.ptr)]) |s| {
                        const val = current[s] orelse blk: {
                            const v = try self.readEntry(s, bi);
                            current[s] = v;
                            break :blk v;
                        };
                        self.func.replaceAllUses(self.func.instResult(inst).?, val);
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

    // Seed with every scalar-element alloca.
    for (0..func.instCount()) |i| {
        const inst: Inst = @enumFromInt(i);
        switch (func.opcode(inst)) {
            .alloca => |al| if (isScalar(func, al.elem)) {
                promotable[@intFromEnum(func.instResult(inst).?)] = true;
            },
            else => {},
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
            .load => {}, // ld.ptr is the sanctioned use
            .store => |st| esc(promotable, st.value), // ptr is fine, value escapes
            .alloca, .iconst, .fconst, .global_addr => {},
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
            .call => |c| for (func.valueList(c.args)) |arg| esc(promotable, arg),
            .call_indirect => |c| {
                esc(promotable, c.target);
                for (func.valueList(c.args)) |arg| esc(promotable, arg);
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
            .ret => |v| if (v) |vv| esc(promotable, vv),
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
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(b, x, slot);
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = y });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(x, func.terminator(b).?.ret.?); // load became x
    try expectNoMemoryInsts(&func, b);
}

test "store in entry forwards across a jump to its single successor" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const ptr_t = try func.types.intern(.ptr);
    const b0 = try func.appendBlock();
    const b1 = try func.appendBlock();
    const x = try func.appendBlockParam(b0, t);
    const slot = try func.appendInst(b0, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(b0, x, slot);
    try func.setJump(b0, b1, &.{});
    const y = try func.appendInst(b1, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b1, .{ .ret = y });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(x, func.terminator(b1).?.ret.?); // load forwarded across the edge
    try expectNoMemoryInsts(&func, b0);
    try expectNoMemoryInsts(&func, b1);
}

test "diamond store on each arm merges into a block parameter at the join" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
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
    func.setTerminator(b3, .{ .ret = y });

    try testing.expect(try runOnce(allocator, &func));
    // A single new parameter on the join carries the merged value, and the ret returns it.
    const params = func.blockParams(b3);
    try testing.expectEqual(@as(usize, 1), params.len);
    try testing.expectEqual(params[0], func.terminator(b3).?.ret.?);
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
    const ptr_t = try func.types.intern(.ptr);
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
    func.setTerminator(exit, .{ .ret = r });

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
    try testing.expectEqual(p, func.terminator(exit).?.ret.?);
    for ([_]Block{ entry, header, body, exit }) |blk| try expectNoMemoryInsts(&func, blk);
    try expectVerifies(allocator, &func);
}

test "a slot whose address escapes to a call is left in memory" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    try func.appendStore(b, x, slot);
    try func.appendVoidCall(b, "escape", &.{slot}); // address leaves the function
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = y });

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
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = arr_t } });
    _ = try func.appendInst(b, arr_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = null });

    try testing.expect(!try runOnce(allocator, &func)); // aggregate slot stays in memory
}

test "two independent scalar slots both promote" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const ptr_t = try func.types.intern(.ptr);
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
    func.setTerminator(b, .{ .ret = sum });

    try testing.expect(try runOnce(allocator, &func));
    // sum now adds the two stored values directly.
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(x, add.lhs);
    try testing.expectEqual(y, add.rhs);
    try expectNoMemoryInsts(&func, b);
}
