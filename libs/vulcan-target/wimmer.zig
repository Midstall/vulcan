//! Shared, target-independent Wimmer-Franz register allocator (Wimmer & Franz, CGO 2010).
//! This module owns the ALGORITHM; each backend owns ENCODING and a `RegDescription` that
//! describes its register model to the allocator. This first task defines only the target
//! abstraction TYPES (no allocation algorithm yet); the scan and resolution follow in later tasks.
//!
//! Physical registers are an ABSTRACTION here: the allocator only ever sees a `u16` register
//! INDEX within a class. The backend chooses a stable numbering and maps the index back to its own
//! register enum. See each backend's `*RegDescription` for the numbering it picked (aarch64 uses
//! the register's own enum integer value, so gpr class index n names x_n and fpr class index n
//! names v_n).

const std = @import("std");
const ir = @import("vulcan-ir");

const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;
const Function = ir.function.Function;
const Terminator = ir.function.Terminator;
const Jump = ir.function.Jump;

/// The only failure mode of interval construction is running out of memory.
pub const Error = std.mem.Allocator.Error;

/// Where a value lives at a program point: a physical register (a class-relative INDEX) or a spill
/// slot. The class is implied by the value's `classOf`.
pub const Location = union(enum) { reg: u16, slot: u32 };

/// Whether an operand use requires a register. `must_have_register` operands cannot read from a
/// spill slot on this target (the safe conservative default); `should_have_register` operands may
/// fold/reload from a slot (a target that can, e.g. x86 memory operands, relaxes specific opcodes).
pub const UseKind = enum { must_have_register, should_have_register };

/// One register class (e.g. gpr vs fpr). `allocatable` is the set of physical register INDICES the
/// scan may hand out for this class; `callee_saved` is the subset that needs prologue save/restore
/// when used; `slot_bytes` is the spill-slot size for a value of this class.
pub const RegClass = struct {
    name: []const u8,
    allocatable: []const u16,
    callee_saved: []const u16,
    slot_bytes: u16,
};

/// A per-class set of register indices clobbered at some point (used by `CallSite`).
pub const ClassRegs = struct { class: u16, regs: []const u16 };

/// A call at position `pos` clobbers `clobbered` (one `ClassRegs` per affected class). The allocator
/// turns each into a fixed interval so no value survives the call in a clobbered register.
pub const CallSite = struct { pos: u32, clobbered: []const ClassRegs };

/// An entry parameter (or any value) pre-colored to a fixed physical register of a class (the ABI
/// argument registers at function entry).
pub const FixedAssign = struct { value: Value, class: u16, reg: u16 };

/// A backend's description of its register model for one function. Built per-function because the
/// allocatable sets can differ by function (aarch64 leaf vs non-leaf pools). `classOf` and `useKind`
/// receive the backend `ctx` so they can consult backend helpers. `aarch64RegDescription` (and each
/// future backend's builder) allocates the owned slices; call `deinit` to free them.
pub const RegDescription = struct {
    classes: []const RegClass,
    classOf: *const fn (ctx: *const anyopaque, func: *const Function, v: Value) u16,
    useKind: *const fn (ctx: *const anyopaque, func: *const Function, inst: Inst, operand: Value) UseKind,
    entry_fixed: []const FixedAssign,
    call_sites: []const CallSite,
    scratch: []const u16,
    ctx: *const anyopaque,

    /// Free every owned slice the backend builder allocated: each class's `allocatable`/`callee_saved`,
    /// the `classes` slice, each call site's per-class `regs` and its `clobbered` slice, the
    /// `call_sites` slice, `entry_fixed`, and `scratch`. Class names and `ctx` are not owned (static).
    pub fn deinit(self: *RegDescription, allocator: std.mem.Allocator) void {
        for (self.classes) |c| {
            allocator.free(c.allocatable);
            allocator.free(c.callee_saved);
        }
        allocator.free(self.classes);
        for (self.call_sites) |cs| {
            for (cs.clobbered) |cr| allocator.free(cr.regs);
            allocator.free(cs.clobbered);
        }
        allocator.free(self.call_sites);
        allocator.free(self.entry_fixed);
        allocator.free(self.scratch);
        self.* = undefined;
    }
};

// ===========================================================================
// Task 2: lifetime intervals (BUILDINTERVALS, Wimmer & Franz Fig 4).
//
// A value's lifetime is a set of half-open live RANGES with HOLES between the
// regions where it is dead, plus the positions it is USED at. Physical
// registers are constrained by FIXED intervals: one per call-clobbered
// register (blocking it over each call) and one per entry parameter (pinning
// its ABI register at function entry). No allocation happens here; the scan
// (Task 3) consumes these.
//
// Position numbering matches the aarch64 backend's `linearize` EXACTLY, so the
// `RegDescription.call_sites` positions (built with that same numbering) line
// up: blocks are walked in block-index order 0..nblocks (NOT reverse-post
// order), a block's parameter row shares the block's start position, then +1
// per instruction and +1 for the terminator slot. Blocks are numbered
// contiguously: `block_from[bi+1] == block_to[bi]`.
// ===========================================================================

/// A half-open live range `[from, to)`. A value live over disjoint ranges has HOLES between them.
pub const Range = struct { from: u32, to: u32 };

/// A use of a value at position `pos` with the register requirement the backend reported.
pub const UsePos = struct { pos: u32, kind: UseKind };

/// One lifetime interval. A VALUE interval (`fixed_reg == null`) describes where an SSA value is
/// live and used. A FIXED interval (`fixed_reg != null`) blocks a physical register over its ranges:
/// call-clobber fixed intervals carry `value == null`, while an entry-parameter fixed interval keeps
/// `value` set to the pinned parameter so the scan can honor the ABI hint.
pub const Interval = struct {
    value: ?Value,
    class: u16,
    fixed_reg: ?u16,
    ranges: []Range, // ascending, disjoint, merged
    uses: []UsePos, // ascending by `pos`
    location: ?Location = null, // filled by the scan (Task 3+); null here

    /// The interval's first live position. Programmer error to call on an empty interval.
    pub fn start(self: *const Interval) u32 {
        std.debug.assert(self.ranges.len > 0);
        return self.ranges[0].from;
    }

    /// The interval's half-open end (one past its last live position).
    pub fn end(self: *const Interval) u32 {
        std.debug.assert(self.ranges.len > 0);
        return self.ranges[self.ranges.len - 1].to;
    }

    /// Whether `pos` falls inside one of the interval's live ranges.
    pub fn covers(self: *const Interval, pos: u32) bool {
        for (self.ranges) |r| {
            if (r.from <= pos and pos < r.to) return true;
        }
        return false;
    }

    /// The first use at position `>= pos` (Wimmer's `>=` convention), or null if none remain.
    pub fn firstUseAfter(self: *const Interval, pos: u32) ?u32 {
        for (self.uses) |u| {
            if (u.pos >= pos) return u.pos;
        }
        return null;
    }

    /// The first position both intervals cover (their earliest overlap), or null if they never
    /// overlap. Used by the scan's `freeUntilPos`. Both range lists are ascending and disjoint, so a
    /// two-pointer merge finds the earliest intersection.
    pub fn nextIntersection(self: *const Interval, other: *const Interval) ?u32 {
        var i: usize = 0;
        var j: usize = 0;
        while (i < self.ranges.len and j < other.ranges.len) {
            const a = self.ranges[i];
            const b = other.ranges[j];
            const lo = @max(a.from, b.from);
            const hi = @min(a.to, b.to);
            if (lo < hi) return lo;
            // Advance whichever range ends first; it cannot intersect any later range of the other.
            if (a.to < b.to) i += 1 else j += 1;
        }
        return null;
    }
};

/// Free the interval slice and every interval's owned `ranges`/`uses`.
pub fn freeIntervals(allocator: std.mem.Allocator, intervals: []Interval) void {
    for (intervals) |iv| {
        allocator.free(iv.ranges);
        allocator.free(iv.uses);
    }
    allocator.free(intervals);
}

fn rangeLessThan(_: void, a: Range, b: Range) bool {
    return a.from < b.from;
}

/// Sort `ranges` ascending and merge overlapping OR touching ranges in place, leaving disjoint
/// ranges with holes where the value is dead. `[a, b)` and `[b, c)` touch and merge into `[a, c)`.
fn normalizeRanges(ranges: *std.ArrayList(Range)) void {
    if (ranges.items.len == 0) return;
    std.mem.sort(Range, ranges.items, {}, rangeLessThan);
    var w: usize = 0;
    for (ranges.items) |r| {
        if (w > 0 and r.from <= ranges.items[w - 1].to) {
            // Overlap or adjacency: extend the previous range.
            if (r.to > ranges.items[w - 1].to) ranges.items[w - 1].to = r.to;
        } else {
            ranges.items[w] = r;
            w += 1;
        }
    }
    ranges.shrinkRetainingCapacity(w);
}

/// Visit every operand VALUE of `inst`, calling `f(ctx, value, is_edge_arg)`. Edge arguments (the
/// values passed to a successor's block parameters by an `if`) are flagged so callers can treat them
/// as uses in the PREDECESSOR. Independent reimplementation of the backend's operand walk over the
/// target-independent `Opcode` set (exhaustive, so a new opcode forces an update here).
fn visitOperands(func: *const Function, inst: Inst, ctx: anytype, comptime f: fn (@TypeOf(ctx), Value, bool) void) void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .alloca, .global_addr => {},
        .arith => |a| {
            f(ctx, a.lhs, false);
            f(ctx, a.rhs, false);
        },
        .arith_imm => |a| f(ctx, a.lhs, false),
        .icmp => |c| {
            f(ctx, c.lhs, false);
            f(ctx, c.rhs, false);
        },
        .select => |s| {
            f(ctx, s.cond, false);
            f(ctx, s.then, false);
            f(ctx, s.@"else", false);
        },
        .extract => |e| f(ctx, e.aggregate, false),
        .convert => |cv| f(ctx, cv.value, false),
        .unary => |u| f(ctx, u.value, false),
        .load => |l| f(ctx, l.ptr, false),
        .store => |st| {
            f(ctx, st.value, false);
            f(ctx, st.ptr, false);
        },
        .prefetch => |pf| f(ctx, pf.ptr, false),
        .dot => |d| {
            f(ctx, d.acc, false);
            f(ctx, d.a, false);
            f(ctx, d.b, false);
        },
        .matmul => |mm| {
            f(ctx, mm.a, false);
            f(ctx, mm.b, false);
            f(ctx, mm.c, false);
        },
        .struct_new => |sn| for (func.valueList(sn.fields)) |fld| f(ctx, fld, false),
        .call => |c| for (func.valueList(c.args)) |a| f(ctx, a, false),
        .call_indirect => |c| {
            f(ctx, c.target, false);
            for (func.valueList(c.args)) |a| f(ctx, a, false);
        },
        .@"if" => |cf| {
            f(ctx, cf.cond, false);
            for (func.blockArgs(cf.then)) |a| f(ctx, a, true);
            for (func.blockArgs(cf.@"else")) |a| f(ctx, a, true);
        },
    }
}

/// Visit every operand VALUE of a terminator. A `.ret` value is an ordinary operand, a `.jump`'s
/// arguments are edge arguments (uses in this block, flowing into the successor's parameters).
fn visitTermOperands(func: *const Function, term: Terminator, ctx: anytype, comptime f: fn (@TypeOf(ctx), Value, bool) void) void {
    switch (term) {
        .ret => |v| if (v) |vv| f(ctx, vv, false),
        .jump => |j| for (func.blockArgs(j)) |a| f(ctx, a, true),
    }
}

/// Build the lifetime intervals the scan consumes: one interval per value that is ever live (with
/// ranges, holes, and use positions), plus fixed intervals for physical registers (call clobbers
/// from `desc.call_sites` and entry parameters from `desc.entry_fixed`). The caller owns the result
/// and must release it with `freeIntervals`.
pub fn buildIntervals(allocator: std.mem.Allocator, func: *const Function, desc: *const RegDescription) Error![]Interval {
    const nblocks = func.blockCount();
    const nval = func.valueCount();

    // --- Per-block numbering + per-value scratch ---
    const block_from = try allocator.alloc(u32, nblocks);
    defer allocator.free(block_from);
    const block_to = try allocator.alloc(u32, nblocks);
    defer allocator.free(block_to);
    const def_pos = try allocator.alloc(u32, nval);
    defer allocator.free(def_pos);
    const is_def = try allocator.alloc(bool, nval);
    defer allocator.free(is_def);
    @memset(def_pos, 0);
    @memset(is_def, false);

    // Per-value range/use builders. Ranges are appended raw (possibly overlapping) and normalized
    // once at the end; a raw-append plus a single sort/merge is simpler than an incremental
    // insert-and-merge and gives the same disjoint result.
    const range_lists = try allocator.alloc(std.ArrayList(Range), nval);
    defer allocator.free(range_lists);
    for (range_lists) |*rl| rl.* = .empty;
    defer for (range_lists) |*rl| rl.deinit(allocator);
    const use_lists = try allocator.alloc(std.ArrayList(UsePos), nval);
    defer allocator.free(use_lists);
    for (use_lists) |*ul| ul.* = .empty;
    defer for (use_lists) |*ul| ul.deinit(allocator);

    // Liveness bitsets, indexed `bi * nval + vi`. `defined`/`used` are the block-local gen/kill sets.
    const defined = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(defined);
    const used = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(used);
    @memset(defined, false);
    @memset(used, false);

    // Successor lists, driving the liveness fixpoint (from `if` instructions and `jump` terminators).
    const succ = try allocator.alloc(std.ArrayList(u32), nblocks);
    defer allocator.free(succ);
    for (succ) |*s| s.* = .empty;
    defer for (succ) |*s| s.deinit(allocator);

    // The context threaded through the operand walk while gathering uses and ranges.
    const Gather = struct {
        allocator: std.mem.Allocator,
        func: *const Function,
        desc: *const RegDescription,
        used_row: []bool,
        range_lists: []std.ArrayList(Range),
        use_lists: []std.ArrayList(UsePos),
        block_from: u32,
        pos: u32,
        inst: Inst,
        term_kind: bool, // true when visiting a terminator (no `inst`, default kind)
        err: ?Error = null,

        fn visit(self: *@This(), v: Value, is_edge_arg: bool) void {
            _ = is_edge_arg;
            const vi = @intFromEnum(v);
            self.used_row[vi] = true;
            const kind: UseKind = if (self.term_kind)
                // A ret value or block-param move is a register move at the block boundary; no target
                // folds a spill slot there, so it needs a register.
                .must_have_register
            else
                self.desc.useKind(self.desc.ctx, self.func, self.inst, v);
            self.use_lists[vi].append(self.allocator, .{ .pos = self.pos, .kind = kind }) catch |e| {
                self.err = e;
            };
            // A use makes the value live from its block's start up to and including the use position.
            self.range_lists[vi].append(self.allocator, .{ .from = self.block_from, .to = self.pos + 1 }) catch |e| {
                self.err = e;
            };
        }
    };

    // --- Pass A: number positions, record defs, gather uses + use-ranges, build the CFG. ---
    var pos: u32 = 0;
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        block_from[bi] = pos;
        for (func.blockParams(block)) |p| {
            const pi = @intFromEnum(p);
            def_pos[pi] = pos;
            is_def[pi] = true;
            defined[bi * nval + pi] = true;
        }
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            var g = Gather{
                .allocator = allocator,
                .func = func,
                .desc = desc,
                .used_row = used[bi * nval ..][0..nval],
                .range_lists = range_lists,
                .use_lists = use_lists,
                .block_from = block_from[bi],
                .pos = pos,
                .inst = inst,
                .term_kind = false,
            };
            visitOperands(func, inst, &g, Gather.visit);
            if (g.err) |e| return e;
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                try succ[bi].append(allocator, @intFromEnum(cf.then.target));
                try succ[bi].append(allocator, @intFromEnum(cf.@"else".target));
            }
            if (func.instResult(inst)) |r| {
                const ri = @intFromEnum(r);
                def_pos[ri] = pos;
                is_def[ri] = true;
                defined[bi * nval + ri] = true;
            }
            pos += 1;
        }
        const term_pos = pos;
        if (func.terminator(block)) |term| {
            var g = Gather{
                .allocator = allocator,
                .func = func,
                .desc = desc,
                .used_row = used[bi * nval ..][0..nval],
                .range_lists = range_lists,
                .use_lists = use_lists,
                .block_from = block_from[bi],
                .pos = term_pos,
                .inst = undefined,
                .term_kind = true,
            };
            visitTermOperands(func, term, &g, Gather.visit);
            if (g.err) |e| return e;
            if (term == .jump) try succ[bi].append(allocator, @intFromEnum(term.jump.target));
        }
        block_to[bi] = term_pos + 1;
        pos += 1;
    }

    // --- Liveness fixpoint: live_out[b] = union of successors' live_in; live_in[b] = used[b] union
    // (live_out[b] minus defined[b]). Back-edges make a loop header's live-in flow into the body's
    // live-out, so loop-carried values stay live across the whole body with no separate loop pass.
    // A block parameter is `defined` in its block, so it is never live-in from an edge; the edge
    // ARGUMENT that feeds it is `used` in the predecessor. Monotonic, so the fixpoint terminates. ---
    const live_in = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(live_in);
    const live_out = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(live_out);
    @memset(live_in, false);
    @memset(live_out, false);
    if (nblocks > 0 and nval > 0) {
        var changed = true;
        while (changed) {
            changed = false;
            var b: usize = nblocks;
            while (b > 0) {
                b -= 1;
                for (succ[b].items) |s| {
                    for (0..nval) |v| {
                        if (live_in[@as(usize, s) * nval + v] and !live_out[b * nval + v]) {
                            live_out[b * nval + v] = true;
                            changed = true;
                        }
                    }
                }
                for (0..nval) |v| {
                    const new_in = (used[b * nval + v] or live_out[b * nval + v]) and !defined[b * nval + v];
                    if (new_in and !live_in[b * nval + v]) {
                        live_in[b * nval + v] = true;
                        changed = true;
                    }
                }
            }
        }
    }

    // --- Pass B: every value live-out of a block is live across that whole block. ---
    for (0..nblocks) |bi| {
        for (0..nval) |v| {
            if (!live_out[bi * nval + v]) continue;
            try range_lists[v].append(allocator, .{ .from = block_from[bi], .to = block_to[bi] });
        }
    }

    // --- Pass C: normalize, clamp each range set to start at the value's single def, and ensure a
    // dead def (defined, never used, not live-out) still gets a minimal [def, def+1) range. ---
    for (0..nval) |v| {
        if (is_def[v] and range_lists[v].items.len == 0) {
            try range_lists[v].append(allocator, .{ .from = def_pos[v], .to = def_pos[v] + 1 });
        }
        normalizeRanges(&range_lists[v]);
        if (range_lists[v].items.len == 0) continue;
        if (is_def[v]) {
            // In SSA a value is not live before its single def, which lies in its earliest range.
            const first = &range_lists[v].items[0];
            std.debug.assert(first.from <= def_pos[v] and def_pos[v] < first.to);
            first.from = def_pos[v];
        }
    }

    // --- Assemble the result: value intervals first, then fixed intervals. ---
    var result: std.ArrayList(Interval) = .empty;
    errdefer {
        for (result.items) |iv| {
            allocator.free(iv.ranges);
            allocator.free(iv.uses);
        }
        result.deinit(allocator);
    }

    for (0..nval) |v| {
        if (range_lists[v].items.len == 0) continue;
        // Assert the ascending+disjoint invariant and ascending uses (programmer errors otherwise).
        assertRangesSorted(range_lists[v].items);
        assertUsesSorted(use_lists[v].items);
        const rs = try range_lists[v].toOwnedSlice(allocator);
        errdefer allocator.free(rs);
        const us = try use_lists[v].toOwnedSlice(allocator);
        errdefer allocator.free(us);
        const value: Value = @enumFromInt(v);
        try result.append(allocator, .{
            .value = value,
            .class = desc.classOf(desc.ctx, func, value),
            .fixed_reg = null,
            .ranges = rs,
            .uses = us,
        });
    }

    try appendFixedIntervals(allocator, func, desc, &result);

    return result.toOwnedSlice(allocator);
}

/// Assert `ranges` is ascending and disjoint (holes allowed).
fn assertRangesSorted(ranges: []const Range) void {
    var prev_to: u32 = 0;
    for (ranges, 0..) |r, idx| {
        std.debug.assert(r.from < r.to);
        if (idx > 0) std.debug.assert(r.from > prev_to);
        prev_to = r.to;
    }
}

/// Assert `uses` is ascending by position (equal positions allowed for a doubly-used operand).
fn assertUsesSorted(uses: []const UsePos) void {
    var prev: u32 = 0;
    for (uses, 0..) |u, idx| {
        if (idx > 0) std.debug.assert(u.pos >= prev);
        prev = u.pos;
    }
}

/// A per-(class, register) accumulator for call-clobber fixed intervals.
const FixedKey = struct { class: u16, reg: u16 };

/// Append fixed intervals: one merged interval per call-clobbered physical register (blocking it
/// over each call position) and one per entry parameter (pinning its ABI register at `[0, 1)`, with
/// the parameter value kept as an allocation hint for the scan).
fn appendFixedIntervals(allocator: std.mem.Allocator, func: *const Function, desc: *const RegDescription, result: *std.ArrayList(Interval)) Error!void {
    _ = func;

    // Merge call clobbers per (class, reg): each clobbered register gets a [pos, pos+1) range at
    // every call it is live across, collected into a single interval with multiple ranges + holes.
    var keys: std.ArrayList(FixedKey) = .empty;
    defer keys.deinit(allocator);
    var builders: std.ArrayList(std.ArrayList(Range)) = .empty;
    defer {
        for (builders.items) |*b| b.deinit(allocator);
        builders.deinit(allocator);
    }

    for (desc.call_sites) |cs| {
        for (cs.clobbered) |cr| {
            for (cr.regs) |reg| {
                const key = FixedKey{ .class = cr.class, .reg = reg };
                var idx: ?usize = null;
                for (keys.items, 0..) |k, i| {
                    if (k.class == key.class and k.reg == key.reg) {
                        idx = i;
                        break;
                    }
                }
                if (idx == null) {
                    try keys.append(allocator, key);
                    try builders.append(allocator, .empty);
                    idx = builders.items.len - 1;
                }
                try builders.items[idx.?].append(allocator, .{ .from = cs.pos, .to = cs.pos + 1 });
            }
        }
    }

    for (keys.items, 0..) |key, i| {
        normalizeRanges(&builders.items[i]);
        assertRangesSorted(builders.items[i].items);
        const rs = try builders.items[i].toOwnedSlice(allocator);
        errdefer allocator.free(rs);
        const us = try allocator.alloc(UsePos, 0);
        errdefer allocator.free(us);
        try result.append(allocator, .{
            .value = null,
            .class = key.class,
            .fixed_reg = key.reg,
            .ranges = rs,
            .uses = us,
        });
    }

    // Entry parameters: each arrives in a fixed ABI register at function entry. Represented as a
    // fixed interval over `[0, 1)` on that register, tagged with the parameter value so the scan can
    // honor the pre-color. `fixed_reg != null` marks it as a fixed interval regardless of `value`.
    for (desc.entry_fixed) |ef| {
        const rs = try allocator.alloc(Range, 1);
        errdefer allocator.free(rs);
        rs[0] = .{ .from = 0, .to = 1 };
        const us = try allocator.alloc(UsePos, 0);
        errdefer allocator.free(us);
        try result.append(allocator, .{
            .value = ef.value,
            .class = ef.class,
            .fixed_reg = ef.reg,
            .ranges = rs,
            .uses = us,
        });
    }
}

// ===========================================================================
// Task 3: the linear scan (LINEARSCAN, Wimmer & Franz Fig 5) plus
// TRYALLOCATEFREEREG (Fig 6), extended by Task 4 with ALLOCATEBLOCKEDREG
// (Fig 7), SPLITINTERVAL, and spill-slot assignment. The scan now handles
// register pressure: when no whole register is free it splits live ranges and
// spills, so a value may span MULTIPLE intervals with different locations. The
// output is the full `Allocation` the later resolution/emission tasks consume:
// its per-value multi-segment map, per-class slot counts, and
// `used_callee_saved`.
// ===========================================================================

/// One placement of a value: from position `from`, the value lives at `loc`. A value that never
/// splits has a single segment. Task 4's splitter produces multi-segment values.
pub const Segment = struct { from: u32, loc: Location };

/// A data move the resolver emits (Task 5/7): move `src` to `dst` within `class`. `value` is the IR
/// value whose bits this move transfers, carried so a backend can look up its type and pick the
/// width-appropriate move/store/load (e.g. x86 movups for a 128-bit vector vs vmovups for 256-bit).
/// Populated for EVERY move, including the scratch cycle-break and slot<->slot routing steps, each of
/// which routes one specific value's bits through the class scratch. aarch64/riscv64 emit their
/// vector moves at a fixed width, so they ignore this field.
pub const Move = struct { src: Location, dst: Location, class: u16, value: Value };

/// An intra-block spill/reload/move the resolver emits at position `at` (Task 5). Unused here.
pub const Action = struct {
    at: u32,
    kind: enum { store, reload, move },
    class: u16,
    src: Location,
    dst: Location,
};

/// The parallel move set on a control-flow edge (Task 7): resolution and block-param moves. Unused
/// here.
pub const EdgeMoves = struct { pred: Block, succ: Block, moves: []Move };

/// A callee-saved physical register that the allocation actually used, so the prologue must save it.
pub const UsedSaved = struct { class: u16, reg: u16 };

/// The register allocation result. Task 3 fills `segments` (one register segment per value),
/// `slot_count_per_class` (all zero, nothing spills yet), and `used_callee_saved`. The remaining
/// fields belong to later tasks and stay empty here.
pub const Allocation = struct {
    segments: std.AutoHashMapUnmanaged(Value, []Segment) = .empty,
    actions: []Action = &.{},
    edge_moves: []EdgeMoves = &.{},
    slot_count_per_class: []u32 = &.{},
    used_callee_saved: []UsedSaved = &.{},
    /// True when some value's location CHANGES across a block boundary (a segment transition whose
    /// two sides fall in different blocks). Realizing that change needs a control-flow-edge move, the
    /// job of the cross-block resolution task (Task 7); the intra-block `actions` here do NOT cover
    /// it. A backend emitting from this allocation before that task exists must bail when this is set
    /// rather than silently drop the edge move. False for a single-block function or any function
    /// whose every value keeps one location per block.
    needs_resolution: bool = false,

    /// Free every owned slice: each value's segment slice and the map itself, the action slice, each
    /// edge's move slice and the edge slice, the per-class slot counts, and the used-saved slice.
    pub fn deinit(self: *Allocation, allocator: std.mem.Allocator) void {
        var it = self.segments.iterator();
        while (it.next()) |e| allocator.free(e.value_ptr.*);
        self.segments.deinit(allocator);
        allocator.free(self.actions);
        for (self.edge_moves) |em| allocator.free(em.moves);
        allocator.free(self.edge_moves);
        allocator.free(self.slot_count_per_class);
        allocator.free(self.used_callee_saved);
        self.* = undefined;
    }
};

/// The only failure mode of `allocate`: out of memory. Task 4's splitter handles every register
/// pressure case, so the earlier `error.Unsupported` bail is gone.
pub const AllocateError = Error;

/// A free-until position meaning "never conflicts". Program positions never reach it.
const infinity: u32 = std.math.maxInt(u32);

/// The widest physical register index the freeUntilPos bookkeeping supports. aarch64 uses 0..31.
const max_phys_regs: usize = 64;

fn intervalStartLessThan(_: void, a: *Interval, b: *Interval) bool {
    return a.start() < b.start();
}

/// The physical register an interval currently occupies: a fixed interval blocks its `fixed_reg`, a
/// placed value interval lives in its assigned register. Programmer error to call on an unplaced or
/// spilled value interval (Task 3 keeps every active/inactive value in a register).
fn assignedReg(it: *const Interval) u16 {
    if (it.fixed_reg) |fr| return fr;
    return switch (it.location.?) {
        .reg => |r| r,
        .slot => unreachable,
    };
}

/// True iff `set` contains `reg`.
fn containsReg(set: []const u16, reg: u16) bool {
    for (set) |x| if (x == reg) return true;
    return false;
}

/// The earliest position at which the call-clobber `fixed` interval genuinely forces `current` out
/// of its register: a clobber point `c` that `current` holds a register ACROSS, i.e. it `covers(c)`
/// and `covers(c + 1)`. A value that merely READS an operand at the call (`covers(c)` but dead at
/// `c + 1`) is not forced out, because the operand read happens before the call clobbers registers.
/// This is the half-open-numbering analogue of the backend's `spansCall`, and it is what keeps a
/// call ARGUMENT in a caller-saved register from being spuriously spilled. Null if the clobber never
/// cuts across `current`. Programmer error unless `fixed` is a call-clobber interval (`value` null).
fn fixedClobberConflict(current: *const Interval, fixed: *const Interval) ?u32 {
    std.debug.assert(fixed.value == null);
    for (fixed.ranges) |r| {
        var c = r.from;
        while (c < r.to) : (c += 1) {
            if (current.covers(c) and current.covers(c + 1)) return c;
        }
    }
    return null;
}

/// The ABI register `v` is hinted to hold at entry, or null if `v` is not an entry parameter of
/// class `class`. The entry-param fixed interval is a HINT, not a hard block, so the scan may hand
/// the parameter its own ABI register.
fn entryHint(desc: *const RegDescription, v: Value, class: u16) ?u16 {
    for (desc.entry_fixed) |ef| {
        if (ef.class == class and ef.value == v) return ef.reg;
    }
    return null;
}

/// TRYALLOCATEFREEREG (Wimmer & Franz Fig 6), without the splitting tail. Compute `freeUntilPos` for
/// every candidate register of `current`'s class (its class pool plus its entry-param hint register),
/// clamp it by the active and inactive intervals that occupy those registers, then pick the register
/// free the longest (ties broken toward the hint). Return that register when it covers `current`'s
/// whole lifetime, otherwise null. A null result means either no register is free at all or a
/// register is free for a prefix only, both of which require a split this task defers to Task 4.
fn tryAllocateFreeReg(
    current: *const Interval,
    active: []const *Interval,
    inactive: []const *Interval,
    desc: *const RegDescription,
) ?u16 {
    const class_idx = current.class;
    const class = desc.classes[class_idx];

    var free_until = [_]u32{0} ** max_phys_regs;
    var is_candidate = [_]bool{false} ** max_phys_regs;
    for (class.allocatable) |r| {
        std.debug.assert(r < max_phys_regs);
        is_candidate[r] = true;
        free_until[r] = infinity;
    }
    const hint = if (current.value) |v| entryHint(desc, v, class_idx) else null;
    if (hint) |h| {
        std.debug.assert(h < max_phys_regs);
        is_candidate[h] = true;
        free_until[h] = infinity;
    }

    // An active interval of this class occupies its register right now (free until position 0). An
    // entry-param fixed interval for `current`'s own value is the hint, not a block, so skip it.
    for (active) |it| {
        if (it.class != class_idx) continue;
        if (sameValue(it, current)) continue;
        const r = assignedReg(it);
        if (r < max_phys_regs and is_candidate[r]) free_until[r] = 0;
    }
    // An inactive interval of this class has a hole here, so its register is free only until the two
    // ranges next intersect. A call-clobber fixed interval (seeded into `inactive`) uses the
    // read-before-clobber rule so a call ARGUMENT is not treated as conflicting. Same hint exception.
    for (inactive) |it| {
        if (it.class != class_idx) continue;
        if (sameValue(it, current)) continue;
        const r = assignedReg(it);
        if (r < max_phys_regs and is_candidate[r]) {
            const x = if (it.fixed_reg != null) fixedClobberConflict(current, it) else current.nextIntersection(it);
            if (x) |xx| {
                if (xx < free_until[r]) free_until[r] = xx;
            }
        }
    }

    // Pick the register free the longest, preferring the hint on a tie so a parameter keeps its ABI
    // register.
    var best_free: u32 = 0;
    for (0..max_phys_regs) |r| {
        if (is_candidate[r] and free_until[r] > best_free) best_free = free_until[r];
    }
    if (best_free == 0) return null;
    var chosen: ?u16 = null;
    if (hint) |h| {
        if (is_candidate[h] and free_until[h] == best_free) chosen = h;
    }
    if (chosen == null) {
        for (0..max_phys_regs) |r| {
            if (is_candidate[r] and free_until[r] == best_free) {
                chosen = @intCast(r);
                break;
            }
        }
    }
    const reg = chosen.?;

    // A register free at least to `current.end()` (half-open) covers the whole interval. Anything
    // less would need a split, which Task 3 does not do.
    if (free_until[reg] >= current.end()) return reg;
    return null;
}

/// True iff both intervals describe the same value (used to skip a value's own entry-param hint
/// interval while computing `freeUntilPos`).
fn sameValue(a: *const Interval, b: *const Interval) bool {
    const av = a.value orelse return false;
    const bv = b.value orelse return false;
    return av == bv;
}

/// Allocate registers for `func` using the shared linear scan with live-range splitting. Builds
/// intervals, seeds the call-clobber fixed intervals into `inactive` so they are visible before
/// their first range (CHANGE 1, the fixed-interval visibility crux), runs LINEARSCAN with
/// `allocateBlockedReg` (Fig 7) + `splitInterval` under register pressure, then collapses each
/// value's (possibly multi-interval) placement into a multi-segment `Allocation`. The caller owns
/// the result and releases it with `Allocation.deinit`.
pub fn allocate(allocator: std.mem.Allocator, func: *const Function, desc: *const RegDescription) AllocateError!Allocation {
    const intervals = try buildIntervals(allocator, func, desc);
    defer freeIntervals(allocator, intervals);

    // Split children are heap-allocated intervals born during the scan; tracked here so their owned
    // `ranges`/`uses` and the interval box itself are freed even on an error path.
    var children: std.ArrayList(*Interval) = .empty;
    defer {
        for (children.items) |c| {
            allocator.free(c.ranges);
            allocator.free(c.uses);
            allocator.destroy(c);
        }
        children.deinit(allocator);
    }

    // One spill slot per spilled interval, counted per class (no slot coloring yet).
    const slots = try allocator.alloc(u32, desc.classes.len);
    defer allocator.free(slots);
    @memset(slots, 0);

    // The worklist (`unhandled`) is a priority queue: value intervals and entry-param fixed
    // intervals sorted ascending by start, popped from the front, with split children re-inserted in
    // sorted position. CHANGE 1: a CALL-CLOBBER fixed interval (`value == null`) is taken OUT of the
    // worklist and seeded directly into `inactive` (its first range is at a call, a hole at position
    // 0), so `tryAllocateFreeReg`/`allocateBlockedReg` see the clobber at every EARLIER position. An
    // entry-param fixed interval stays in the worklist (a hint realized when popped, skip-clamped for
    // its own value), never a hard block.
    var unhandled: std.ArrayList(*Interval) = .empty;
    defer unhandled.deinit(allocator);
    var active: std.ArrayList(*Interval) = .empty;
    defer active.deinit(allocator);
    var inactive: std.ArrayList(*Interval) = .empty;
    defer inactive.deinit(allocator);

    for (intervals) |*iv| {
        if (iv.fixed_reg != null and iv.value == null) {
            iv.location = .{ .reg = iv.fixed_reg.? };
            try inactive.append(allocator, iv);
        } else {
            try unhandled.append(allocator, iv);
        }
    }
    std.mem.sort(*Interval, unhandled.items, {}, intervalStartLessThan);

    // The scan is a worklist loop. Every split child starts strictly after the interval it came from,
    // so total intervals are bounded by (values x positions); the guard asserts that bound to catch a
    // splitting bug that would otherwise loop forever.
    const max_pos = maxEndPosition(intervals);
    const iter_bound: usize = intervals.len + intervals.len * (@as(usize, max_pos) + 1);
    var iters: usize = 0;
    while (unhandled.items.len > 0) {
        iters += 1;
        std.debug.assert(iters <= iter_bound);
        const current = unhandled.orderedRemove(0);
        const position = current.start();

        // Expire or deactivate active intervals. Ranges are half-open, so `end() <= position` means
        // the interval's last live position is behind us (handled); a live interval that does not
        // cover `position` sits in a hole (inactive). `swapRemove` reorders, which is fine here.
        var ai: usize = 0;
        while (ai < active.items.len) {
            const it = active.items[ai];
            if (it.end() <= position) {
                _ = active.swapRemove(ai);
            } else if (!it.covers(position)) {
                try inactive.append(allocator, active.swapRemove(ai));
            } else {
                ai += 1;
            }
        }
        // Expire or reactivate inactive intervals: an expired one is handled, one that now covers
        // `position` returns to active.
        var ii: usize = 0;
        while (ii < inactive.items.len) {
            const it = inactive.items[ii];
            if (it.end() <= position) {
                _ = inactive.swapRemove(ii);
            } else if (it.covers(position)) {
                try active.append(allocator, inactive.swapRemove(ii));
            } else {
                ii += 1;
            }
        }

        // An entry-param fixed interval realizes its ABI pin: occupy its register over `[0, 1)`.
        // (Call-clobber fixed intervals never enter the worklist; they were seeded into `inactive`.)
        if (current.fixed_reg != null) {
            current.location = .{ .reg = current.fixed_reg.? };
            try active.append(allocator, current);
            continue;
        }

        // A value interval: take a free register covering its whole lifetime, or fall back to the
        // blocked-register path that splits and spills to make room.
        if (tryAllocateFreeReg(current, active.items, inactive.items, desc)) |reg| {
            current.location = .{ .reg = reg };
            try active.append(allocator, current);
        } else {
            try allocateBlockedReg(allocator, current, &active, &inactive, &unhandled, &children, slots, desc);
        }
    }

    // Task 9: in a test/debug build, verify the completed allocation before lowering it. A firing
    // assert here means the scan produced an UNSOUND allocation (a real bug), caught now instead of as
    // a downstream miscompile. Gated on `runtime_safety`, so the ReleaseFast production JIT is not
    // slowed. The verifier reasons over the final intervals (the originals plus the split children),
    // flattened into one read-only slice.
    if (std.debug.runtime_safety) {
        const all = try allocator.alloc(Interval, intervals.len + children.items.len);
        defer allocator.free(all);
        @memcpy(all[0..intervals.len], intervals);
        for (children.items, 0..) |c, ci| all[intervals.len + ci] = c.*;
        const violations = try verifyIntervals(allocator, all);
        defer allocator.free(violations);
        std.debug.assert(violations.len == 0);
    }

    var result = try buildAllocation(allocator, func, intervals, children.items, slots, desc);
    errdefer result.deinit(allocator);
    // Task 7: RESOLVEDATAFLOW. Compute the control-flow-edge moves while the intervals (needed for
    // liveness) are still alive, and order each edge's moves as a valid parallel move.
    try resolveDataFlow(allocator, func, desc, intervals, children.items, &result);
    return result;
}

/// The per-block half-open position bounds `[from, to)`, numbered EXACTLY as `buildIntervals` does
/// (block start row, one position per instruction, one terminator slot), so a position looked up
/// here lands in the same block the intervals were built against. Blocks are contiguous
/// (`from[bi+1] == to[bi]`). The caller owns both slices.
fn computeBlockBounds(allocator: std.mem.Allocator, func: *const Function) Error!struct { from: []u32, to: []u32 } {
    const nblocks = func.blockCount();
    const from = try allocator.alloc(u32, nblocks);
    errdefer allocator.free(from);
    const to = try allocator.alloc(u32, nblocks);
    errdefer allocator.free(to);
    var pos: u32 = 0;
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        from[bi] = pos;
        pos += 1; // the block-parameter row
        pos += @intCast(func.blockInsts(block).len);
        // The terminator shares the block-end position, then one final increment lands on the next
        // block, so the block's half-open end is one past the terminator slot.
        to[bi] = pos + 1;
        pos += 1;
    }
    return .{ .from = from, .to = to };
}

/// The index of the block containing `pos`. Programmer error if `pos` lies outside every block
/// (the numbering is dense and contiguous, so every valid position belongs to exactly one block).
fn blockOfPos(block_from: []const u32, block_to: []const u32, pos: u32) usize {
    for (0..block_from.len) |bi| {
        if (block_from[bi] <= pos and pos < block_to[bi]) return bi;
    }
    unreachable;
}

/// The largest half-open `end()` across all intervals (the position count), or 0 if there are none.
fn maxEndPosition(intervals: []const Interval) u32 {
    var m: u32 = 0;
    for (intervals) |*iv| {
        if (iv.ranges.len == 0) continue;
        if (iv.end() > m) m = iv.end();
    }
    return m;
}

/// The first `must_have_register` use of `it`, or null if it has none (then the interval may live in
/// memory over its whole lifetime).
fn firstMustHaveUse(it: *const Interval) ?u32 {
    for (it.uses) |u| {
        if (u.kind == .must_have_register) return u.pos;
    }
    return null;
}

/// Insert `iv` into the sorted worklist `list`, keeping it ascending by `start()` (a split child is
/// placed after any interval that starts no later than it).
fn insertSorted(allocator: std.mem.Allocator, list: *std.ArrayList(*Interval), iv: *Interval) Error!void {
    var idx: usize = 0;
    while (idx < list.items.len and list.items[idx].start() <= iv.start()) : (idx += 1) {}
    try list.insert(allocator, idx, iv);
}

/// SPLITINTERVAL (Wimmer & Franz): split `parent` at `pos` into a head the parent keeps (ranges and
/// uses at positions `< pos`) and a freshly allocated CHILD carrying everything at positions `>= pos`
/// (a range straddling `pos` is cut into `[from, pos)` for the head and `[pos, to)` for the child).
/// Both reference the same `value`; the child's `location` is unset (the caller re-inserts it into
/// the worklist or assigns it a slot). The child is appended to `children` so it is freed with the
/// rest. Programmer error unless `parent.start() < pos < parent.end()` (both halves non-empty).
fn splitInterval(allocator: std.mem.Allocator, parent: *Interval, pos: u32, children: *std.ArrayList(*Interval)) Error!*Interval {
    std.debug.assert(parent.fixed_reg == null);
    std.debug.assert(pos > parent.start());
    std.debug.assert(pos < parent.end());

    var head_ranges: std.ArrayList(Range) = .empty;
    errdefer head_ranges.deinit(allocator);
    var tail_ranges: std.ArrayList(Range) = .empty;
    errdefer tail_ranges.deinit(allocator);
    for (parent.ranges) |r| {
        if (r.to <= pos) {
            try head_ranges.append(allocator, r);
        } else if (r.from >= pos) {
            try tail_ranges.append(allocator, r);
        } else {
            try head_ranges.append(allocator, .{ .from = r.from, .to = pos });
            try tail_ranges.append(allocator, .{ .from = pos, .to = r.to });
        }
    }
    var head_uses: std.ArrayList(UsePos) = .empty;
    errdefer head_uses.deinit(allocator);
    var tail_uses: std.ArrayList(UsePos) = .empty;
    errdefer tail_uses.deinit(allocator);
    for (parent.uses) |u| {
        if (u.pos < pos) {
            try head_uses.append(allocator, u);
        } else {
            try tail_uses.append(allocator, u);
        }
    }

    // The split point lies strictly inside the parent's live span, so both halves keep a range.
    std.debug.assert(head_ranges.items.len > 0);
    std.debug.assert(tail_ranges.items.len > 0);

    const child = try allocator.create(Interval);
    errdefer allocator.destroy(child);
    const tr = try tail_ranges.toOwnedSlice(allocator);
    errdefer allocator.free(tr);
    const tu = try tail_uses.toOwnedSlice(allocator);
    errdefer allocator.free(tu);
    const hr = try head_ranges.toOwnedSlice(allocator);
    errdefer allocator.free(hr);
    const hu = try head_uses.toOwnedSlice(allocator);
    errdefer allocator.free(hu);

    child.* = .{
        .value = parent.value,
        .class = parent.class,
        .fixed_reg = null,
        .ranges = tr,
        .uses = tu,
        .location = null,
    };
    try children.append(allocator, child);

    // Commit: the child now owns the tail; replace the parent's ranges/uses with the head.
    allocator.free(parent.ranges);
    parent.ranges = hr;
    allocator.free(parent.uses);
    parent.uses = hu;
    return child;
}

/// Spill `current`: it is the cheapest interval to move to memory (its own first use is further off
/// than any evictable register's next use, or no register can be evicted here at all). A
/// `must_have_register` use forces the spilled part to end at that use so the use lands back in a
/// register (the head, which holds no must_have use, is safe in memory and the register-needing tail
/// is re-queued). Otherwise the whole interval goes to a slot. Shared by both spill-current paths.
fn spillCurrent(
    allocator: std.mem.Allocator,
    current: *Interval,
    unhandled: *std.ArrayList(*Interval),
    children: *std.ArrayList(*Interval),
    slots: []u32,
    class_idx: u16,
) Error!void {
    if (firstMustHaveUse(current)) |u| {
        // A must_have use at `current.start()` would mean more values need a register at one position
        // than the class has registers, an unsatisfiable target model (a programmer error, not a
        // spill we can make). For every reachable case the first must_have use is strictly later.
        std.debug.assert(u > current.start());
        const tail = try splitInterval(allocator, current, u, children);
        current.location = .{ .slot = slots[class_idx] };
        slots[class_idx] += 1;
        try insertSorted(allocator, unhandled, tail);
    } else {
        current.location = .{ .slot = slots[class_idx] };
        slots[class_idx] += 1;
    }
}

/// ALLOCATEBLOCKEDREG (Wimmer & Franz Fig 7). `current` could not get a free register, so either it
/// or an interval occupying a register must be split. Compute `nextUsePos[r]` for every allocatable
/// register `r` of `current`'s class (the earliest future use of whatever holds `r`, or a hard block
/// where a fixed interval clobbers `r`), pick the register whose next use is FURTHEST away, then:
/// spill `current` if its own first use is even further off, otherwise take the register and split
/// the intervals it displaces. A fixed clobber of the chosen register before `current` ends also
/// splits `current` before the clobber. Honors `must_have_register`: a spilled head is cut before the
/// first must_have use so that use lands back in a register.
fn allocateBlockedReg(
    allocator: std.mem.Allocator,
    current: *Interval,
    active: *std.ArrayList(*Interval),
    inactive: *std.ArrayList(*Interval),
    unhandled: *std.ArrayList(*Interval),
    children: *std.ArrayList(*Interval),
    slots: []u32,
    desc: *const RegDescription,
) Error!void {
    const class_idx = current.class;
    const class = desc.classes[class_idx];
    const p = current.start();

    var next_use = [_]u32{infinity} ** max_phys_regs;
    var block_pos = [_]u32{infinity} ** max_phys_regs;
    var is_candidate = [_]bool{false} ** max_phys_regs;
    // A register is EVICTABLE only if the active value interval occupying it can be legally split at
    // `p`, i.e. that interval starts strictly before `p`. An occupant starting AT `p` (a same-start
    // same-class value, e.g. one of a block's params when there are more of them than the pool) cannot
    // be split there (`splitInterval` requires `pos > start`), so its register is not evictable here.
    var evictable = [_]bool{true} ** max_phys_regs;
    for (class.allocatable) |r| {
        std.debug.assert(r < max_phys_regs);
        is_candidate[r] = true;
    }

    // An active VALUE interval occupies its register from here; the register is wanted again at that
    // interval's next use (`>= p`), or is spillable cheaply until its end if it has no further use.
    for (active.items) |it| {
        if (it.class != class_idx) continue;
        if (it.fixed_reg != null) continue;
        const r = assignedReg(it);
        if (r >= max_phys_regs or !is_candidate[r]) continue;
        const u = it.firstUseAfter(p) orelse it.end();
        if (u < next_use[r]) next_use[r] = u;
        if (it.start() == p) evictable[r] = false;
    }
    // An inactive VALUE interval only reclaims its register where it next intersects `current`; its
    // next use bounds how soon that register is genuinely wanted.
    for (inactive.items) |it| {
        if (it.class != class_idx) continue;
        if (it.fixed_reg != null) continue;
        const r = assignedReg(it);
        if (r >= max_phys_regs or !is_candidate[r]) continue;
        if (current.nextIntersection(it) == null) continue;
        const u = it.firstUseAfter(p) orelse it.end();
        if (u < next_use[r]) next_use[r] = u;
    }
    // A fixed interval is a HARD block on its register: record it in both `next_use` (the register
    // cannot be chosen past there) and `block_pos` (the point `current` must be split before). A
    // call-clobber interval (`value == null`) uses the read-before-clobber rule so a call ARGUMENT is
    // not spuriously forced out. An entry-param fixed interval (`value != null`) OWNS its ABI register
    // over `[0, 1)`, so any overlap is a hard block computed by plain intersection. This mirrors
    // `tryAllocateFreeReg` so a register held ONLY by another value's entry pin is not read as
    // `infinity` and wrongly taken with no split, which would overlap two values on one ABI register.
    // The value's own entry-param pin is its hint, not a block, so it is skipped.
    for ([_][]const *Interval{ active.items, inactive.items }) |list| {
        for (list) |it| {
            if (it.fixed_reg == null) continue;
            if (it.class != class_idx) continue;
            if (sameValue(it, current)) continue;
            const r = it.fixed_reg.?;
            if (r >= max_phys_regs or !is_candidate[r]) continue;
            const x = if (it.value == null) fixedClobberConflict(current, it) else current.nextIntersection(it);
            if (x) |xx| {
                if (xx < next_use[r]) next_use[r] = xx;
                if (xx < block_pos[r]) block_pos[r] = xx;
            }
        }
    }

    // Pick the EVICTABLE register whose next use is furthest away (the least costly to steal). A
    // register whose active occupant starts at `p` is skipped: it cannot be split at `p` to make room.
    var chosen: ?u16 = null;
    var best: u32 = 0;
    for (0..max_phys_regs) |r| {
        if (!is_candidate[r]) continue;
        if (!evictable[r]) continue;
        if (chosen == null or next_use[r] > best) {
            best = next_use[r];
            chosen = @intCast(r);
        }
    }

    // No register is evictable: every candidate is held by a same-start same-class interval (e.g. more
    // params than the pool). None can be split at `p`, so spill `current` instead. It belongs to that
    // same-start group, so spilling it is valid and makes progress.
    const reg = chosen orelse {
        try spillCurrent(allocator, current, unhandled, children, slots, class_idx);
        return;
    };

    // If `current`'s own first use is later than the chosen register's next use, `current` is the
    // cheapest to spill: keep it in memory over the head and re-allocate the register-needing tail.
    const current_first_use = current.firstUseAfter(p);
    if (current_first_use == null or current_first_use.? > next_use[reg]) {
        try spillCurrent(allocator, current, unhandled, children, slots, class_idx);
        return;
    }

    // Otherwise take `reg` for `current` and split whatever occupies it.
    current.location = .{ .reg = reg };

    // The single active value interval on `reg` is split at `p`: its head keeps the register up to
    // here (and expires next step), its tail is re-allocated elsewhere.
    for (active.items) |it| {
        if (it.class != class_idx) continue;
        if (it.fixed_reg != null) continue;
        if (assignedReg(it) != reg) continue;
        std.debug.assert(it.start() < p);
        const tail = try splitInterval(allocator, it, p, children);
        try insertSorted(allocator, unhandled, tail);
        break;
    }
    // Each inactive value interval on `reg` that would reclaim it inside `current`'s life is split at
    // that intersection; its tail is re-allocated elsewhere.
    for (inactive.items) |it| {
        if (it.class != class_idx) continue;
        if (it.fixed_reg != null) continue;
        if (assignedReg(it) != reg) continue;
        const x = current.nextIntersection(it) orelse continue;
        std.debug.assert(x > it.start() and x < it.end());
        const tail = try splitInterval(allocator, it, x, children);
        try insertSorted(allocator, unhandled, tail);
    }
    // A fixed interval clobbers `reg` before `current` ends: `current` cannot hold `reg` across the
    // clobber, so split it before the block and re-allocate the far side.
    if (block_pos[reg] < current.end()) {
        const bp = block_pos[reg];
        std.debug.assert(bp > p);
        const tail = try splitInterval(allocator, current, bp, children);
        try insertSorted(allocator, unhandled, tail);
    }
    try active.append(allocator, current);
}

fn intervalStartLessThanConst(_: void, a: *const Interval, b: *const Interval) bool {
    return a.start() < b.start();
}

/// True iff two locations name the same register or the same slot (for merging adjacent segments).
fn locEql(a: Location, b: Location) bool {
    return switch (a) {
        .reg => |ra| switch (b) {
            .reg => |rb| ra == rb,
            .slot => false,
        },
        .slot => |sa| switch (b) {
            .reg => false,
            .slot => |sb| sa == sb,
        },
    };
}

/// The move kind that realizes a `src -> dst` location change: a register drop to a slot is a
/// `store`, a slot lift back to a register is a `reload`, and a register-to-register or (backend
/// scratch-realized) slot-to-slot shuffle is a `move`.
fn actionKind(src: Location, dst: Location) @FieldType(Action, "kind") {
    return switch (src) {
        .reg => switch (dst) {
            .reg => .move,
            .slot => .store,
        },
        .slot => switch (dst) {
            .reg => .reload,
            .slot => .move,
        },
    };
}

fn actionAtLessThan(_: void, a: Action, b: Action) bool {
    return a.at < b.at;
}

/// Invoke `f(iv)` for every placed VALUE interval, both the originals and the split children (fixed
/// intervals and unplaced intervals are skipped). Keeps the two storage lists in one walk.
fn forEachPlacedValue(originals: []const Interval, children: []const *Interval, ctx: anytype, comptime f: fn (@TypeOf(ctx), *const Interval) Error!void) Error!void {
    for (originals) |*iv| {
        if (iv.fixed_reg != null) continue;
        if (iv.value == null) continue;
        // The scan places every value interval it processes; a null here would silently drop a
        // segment and miscompile, so it is a programmer error, not a skip.
        std.debug.assert(iv.location != null);
        try f(ctx, iv);
    }
    for (children) |iv| {
        std.debug.assert(iv.fixed_reg == null);
        std.debug.assert(iv.value != null);
        std.debug.assert(iv.location != null);
        try f(ctx, iv);
    }
}

/// Collapse the placed intervals into an `Allocation`. A value now owns MULTIPLE intervals (a parent
/// plus split children) with different locations, so gather every interval per value, sort by
/// `start()`, and emit one ascending segment per interval (merging adjacent identical-location
/// segments). Also records the per-class spill-slot counts and the callee-saved registers used.
fn buildAllocation(allocator: std.mem.Allocator, func: *const Function, intervals: []const Interval, children: []const *Interval, slots: []const u32, desc: *const RegDescription) AllocateError!Allocation {
    var result = Allocation{};
    errdefer result.deinit(allocator);

    // Block bounds drive the intra- vs cross-block classification of every segment transition below.
    const bounds = try computeBlockBounds(allocator, func);
    defer allocator.free(bounds.from);
    defer allocator.free(bounds.to);

    // Intra-block data moves (spill/reload/reg-move) accumulated across every value, sorted ascending
    // by `at` at the end. A cross-block transition is NOT an action here (Task 7 emits it as an edge
    // move); it only flips `needs_resolution` so the backend bails instead of miscompiling.
    var actions: std.ArrayList(Action) = .empty;
    errdefer actions.deinit(allocator);

    // Group every placed value interval (originals + children) by its value.
    var groups: std.AutoHashMapUnmanaged(Value, std.ArrayList(*const Interval)) = .empty;
    defer {
        var git = groups.iterator();
        while (git.next()) |e| e.value_ptr.deinit(allocator);
        groups.deinit(allocator);
    }
    const Grouper = struct {
        allocator: std.mem.Allocator,
        groups: *std.AutoHashMapUnmanaged(Value, std.ArrayList(*const Interval)),
        fn add(self: @This(), iv: *const Interval) Error!void {
            const gop = try self.groups.getOrPut(self.allocator, iv.value.?);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(self.allocator, iv);
        }
    };
    try forEachPlacedValue(intervals, children, Grouper{ .allocator = allocator, .groups = &groups }, Grouper.add);

    // Emit each value's ascending multi-segment list.
    var git = groups.iterator();
    while (git.next()) |e| {
        const list = e.value_ptr;
        std.mem.sort(*const Interval, list.items, {}, intervalStartLessThanConst);
        var segs: std.ArrayList(Segment) = .empty;
        errdefer segs.deinit(allocator);
        for (list.items) |iv| {
            const loc = iv.location.?;
            if (segs.items.len > 0 and locEql(segs.items[segs.items.len - 1].loc, loc)) continue;
            try segs.append(allocator, .{ .from = iv.start(), .loc = loc });
        }
        // Every consecutive segment pair is a location change the emitter must realize. An INTRA-block
        // change (both sides in one block) becomes an `Action` at the later segment's `from`; a
        // CROSS-block change is an edge move Task 7 owns, so it only records `needs_resolution`.
        const class = list.items[0].class;
        var i: usize = 0;
        while (i + 1 < segs.items.len) : (i += 1) {
            const a = segs.items[i];
            const b = segs.items[i + 1];
            const at = b.from;
            // A transition is cross-block (resolved on the edge by Task 7) ONLY when the later segment
            // begins EXACTLY on a block-entry position. Any other transition happens mid-block and is an
            // intra-block action, even when the earlier segment began in an earlier block: a value held
            // in a register across a block boundary and evicted mid-block must be stored HERE, not on the
            // edge (the edge sees the same register on both sides, so it emits no move). Classifying such
            // a mid-block spill as cross-block drops the store, a silent miscompile.
            const at_block = blockOfPos(bounds.from, bounds.to, at);
            const is_cross_block = at == bounds.from[at_block];
            if (is_cross_block) {
                result.needs_resolution = true;
                continue;
            }
            try actions.append(allocator, .{ .at = at, .kind = actionKind(a.loc, b.loc), .class = class, .src = a.loc, .dst = b.loc });
        }

        const owned = try segs.toOwnedSlice(allocator);
        errdefer allocator.free(owned);
        try result.segments.put(allocator, e.key_ptr.*, owned);
    }

    // Actions land in ascending-`at` order for the emitter's single-cursor drain.
    std.mem.sort(Action, actions.items, {}, actionAtLessThan);
    result.actions = try actions.toOwnedSlice(allocator);

    // Copy the accumulated per-class slot counts into an owned slice for the result.
    const slot_counts = try allocator.alloc(u32, desc.classes.len);
    errdefer allocator.free(slot_counts);
    std.debug.assert(slots.len == desc.classes.len);
    @memcpy(slot_counts, slots);
    result.slot_count_per_class = slot_counts;

    // Record the callee-saved registers any value segment landed in, so the prologue saves them.
    const Saver = struct {
        allocator: std.mem.Allocator,
        used: *std.ArrayList(UsedSaved),
        desc: *const RegDescription,
        fn add(self: @This(), iv: *const Interval) Error!void {
            const reg = switch (iv.location.?) {
                .reg => |r| r,
                .slot => return,
            };
            if (!containsReg(self.desc.classes[iv.class].callee_saved, reg)) return;
            for (self.used.items) |u| {
                if (u.class == iv.class and u.reg == reg) return;
            }
            try self.used.append(self.allocator, .{ .class = iv.class, .reg = reg });
        }
    };
    var used: std.ArrayList(UsedSaved) = .empty;
    errdefer used.deinit(allocator);
    try forEachPlacedValue(intervals, children, Saver{ .allocator = allocator, .used = &used, .desc = desc }, Saver.add);
    result.used_callee_saved = try used.toOwnedSlice(allocator);

    return result;
}

// ===========================================================================
// Task 7: RESOLVEDATAFLOW (Wimmer & Franz Fig 8) plus the standard parallel-move
// ordering. After the scan, a value may live in DIFFERENT locations on the two
// sides of a control-flow edge (the splitter placed it in a register in one
// block and a slot in another, or a block parameter simply lands in a different
// register than the argument that feeds it). Each such difference becomes a MOVE
// on that edge. The raw move set of one edge may contain conflicts (one move's
// destination is another's source) and cycles (a register swap), so each edge's
// moves are ordered into a valid sequence: every source is read before it is
// overwritten, cycles are broken through the class scratch register, and a
// slot->slot shuffle is routed through the class scratch too (no target moves
// memory to memory in one op). The backend (Task 8) emits the ordered list
// op-by-op with no further reordering.
//
// PRECONDITION: the function has NO critical edge. Resolution places moves at an
// edge, and a critical edge (a multi-successor `if` block feeding a
// multi-predecessor block) has no block that can host them without corrupting
// the sibling edge. The Task-8 wiring calls `splitCriticalEdges` before building
// the RegDescription, so numbering stays consistent; `assertNoCriticalEdges`
// fails loudly on a wiring mistake. `allocate` never splits edges itself (that
// would invalidate the already-built positions).
// ===========================================================================

/// The location a value occupies at position `pos`, read from its ascending segment list: the
/// location of the last segment that starts at or before `pos`. Programmer error if `pos` precedes
/// the value's first segment (the caller only looks up positions the value is defined at or past).
fn locationAt(segs: []const Segment, pos: u32) Location {
    std.debug.assert(segs.len > 0);
    std.debug.assert(segs[0].from <= pos);
    var loc = segs[0].loc;
    for (segs) |s| {
        if (s.from <= pos) loc = s.loc;
    }
    return loc;
}

/// True iff some VALUE interval of `v` (an original or a split child, never a fixed interval) covers
/// `pos`. A value's lifetime is the union of its intervals, so this is its true liveness at `pos`.
fn valueLiveAt(intervals: []const Interval, children: []const *Interval, v: Value, pos: u32) bool {
    for (intervals) |*iv| {
        if (iv.fixed_reg != null) continue;
        const iv_v = iv.value orelse continue;
        if (iv_v == v and iv.covers(pos)) return true;
    }
    for (children) |iv| {
        std.debug.assert(iv.fixed_reg == null);
        if (iv.value.? == v and iv.covers(pos)) return true;
    }
    return false;
}

/// True iff `v` is a block parameter of `block`.
fn isParamOf(func: *const Function, block: Block, v: Value) bool {
    for (func.blockParams(block)) |p| {
        if (p == v) return true;
    }
    return false;
}

/// RESOLVEDATAFLOW: fill `result.edge_moves` with one ordered `EdgeMoves` per control-flow edge that
/// actually needs a shuffle. Consumes the (post-scan) intervals for liveness and `result.segments`
/// for locations, so it must run before `freeIntervals`.
fn resolveDataFlow(
    allocator: std.mem.Allocator,
    func: *const Function,
    desc: *const RegDescription,
    intervals: []const Interval,
    children: []const *Interval,
    result: *Allocation,
) Error!void {
    if (std.debug.runtime_safety) try assertNoCriticalEdges(allocator, func);

    const bounds = try computeBlockBounds(allocator, func);
    defer allocator.free(bounds.from);
    defer allocator.free(bounds.to);

    var edges: std.ArrayList(EdgeMoves) = .empty;
    errdefer {
        for (edges.items) |em| allocator.free(em.moves);
        edges.deinit(allocator);
    }

    const nblocks = func.blockCount();
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        // An `if` instruction contributes its two edges (then, else); each carries its own args.
        for (func.blockInsts(block)) |inst| {
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                try addEdgeMoves(allocator, func, desc, intervals, children, result, bounds.from, bounds.to, &edges, block, cf.then);
                try addEdgeMoves(allocator, func, desc, intervals, children, result, bounds.from, bounds.to, &edges, block, cf.@"else");
            }
        }
        // A `jump` terminator contributes one edge; a `ret` contributes none.
        if (func.terminator(block)) |term| switch (term) {
            .jump => |j| try addEdgeMoves(allocator, func, desc, intervals, children, result, bounds.from, bounds.to, &edges, block, j),
            .ret => {},
        };
    }

    result.edge_moves = try edges.toOwnedSlice(allocator);
}

/// Compute and store the ordered move list for one edge `pred -> edge.target`. The raw move set is
/// (1) one move per successor PARAMETER whose location differs from the location of the ARGUMENT the
/// predecessor passes for it, and (2) one move per NON-parameter value that is live-in to the
/// successor and whose location differs across the edge. An edge with no differing location adds no
/// `EdgeMoves`.
fn addEdgeMoves(
    allocator: std.mem.Allocator,
    func: *const Function,
    desc: *const RegDescription,
    intervals: []const Interval,
    children: []const *Interval,
    result: *const Allocation,
    block_from: []const u32,
    block_to: []const u32,
    edges: *std.ArrayList(EdgeMoves),
    pred: Block,
    edge: Jump,
) Error!void {
    const succ = edge.target;
    // The predecessor's branch executes at its last position; the successor is entered at its
    // parameter row. A location looked up at `pt` is the value's placement as control leaves `pred`,
    // and at `ss` its placement as control enters `succ`.
    const pt = block_to[@intFromEnum(pred)] - 1;
    const ss = block_from[@intFromEnum(succ)];

    var raw: std.ArrayList(Move) = .empty;
    defer raw.deinit(allocator);

    // (1) Block-parameter moves: the argument's location at pred exit into the parameter's location
    // at succ entry. Same length is an IR invariant (verification guarantees arg/param arity).
    const params = func.blockParams(succ);
    const args = func.blockArgs(edge);
    std.debug.assert(params.len == args.len);
    for (params, args) |p, a| {
        // Every used value has a segment by invariant (a def, even a dead one, gets a minimal interval).
        // A missing segment is an invariant break, not a move to skip: skipping it would silently drop
        // the move (a miscompile), so fail loudly instead.
        const a_segs = result.segments.get(a) orelse {
            std.debug.assert(false);
            continue;
        };
        const p_segs = result.segments.get(p) orelse {
            std.debug.assert(false);
            continue;
        };
        const from = locationAt(a_segs, pt);
        const to = locationAt(p_segs, ss);
        if (!locEql(from, to)) {
            // The move transfers the parameter's bits (same IR type as the argument), so `p` names the
            // width for a width-aware backend.
            try raw.append(allocator, .{ .src = from, .dst = to, .class = desc.classOf(desc.ctx, func, p), .value = p });
        }
    }

    // (2) Non-parameter live-in moves: a value that flows THROUGH the edge (live-in to succ, not a
    // succ parameter) and whose location changes across the edge. A value only live-out of pred but
    // dead at succ entry is not moved.
    var it = result.segments.iterator();
    while (it.next()) |e| {
        const v = e.key_ptr.*;
        if (isParamOf(func, succ, v)) continue;
        if (!valueLiveAt(intervals, children, v, ss)) continue;
        const segs = e.value_ptr.*;
        // Live-in to succ across this edge implies defined before the edge, so both lookups are valid.
        std.debug.assert(segs[0].from <= ss);
        std.debug.assert(segs[0].from <= pt);
        const from = locationAt(segs, pt);
        const to = locationAt(segs, ss);
        if (!locEql(from, to)) {
            try raw.append(allocator, .{ .src = from, .dst = to, .class = desc.classOf(desc.ctx, func, v), .value = v });
        }
    }

    if (raw.items.len == 0) return;

    const ordered = try orderMoves(allocator, raw.items, desc);
    errdefer allocator.free(ordered);
    try edges.append(allocator, .{ .pred = pred, .succ = succ, .moves = ordered });
}

/// True iff `loc` is a register (rather than a spill slot).
fn locIsReg(loc: Location) bool {
    return switch (loc) {
        .reg => true,
        .slot => false,
    };
}

/// A stable u64 key for a `Location`, disjoint across the reg/slot spaces (register keys in the low
/// half, slot keys in the high half). Used by the ordering validator's content simulation.
fn locKey(loc: Location) u64 {
    return switch (loc) {
        .reg => |r| r,
        .slot => |s| (@as(u64, 1) << 32) | s,
    };
}

/// Order a raw parallel-move set into a valid emission sequence, grouping by class (a move never
/// conflicts with one of another class, since the register/slot spaces are per class) and ordering
/// each class independently through `orderClassMoves`. The caller owns the returned slice. Exposed
/// for white-box testing of the cycle/swap and slot->slot routing.
pub fn orderMoves(allocator: std.mem.Allocator, raw: []const Move, desc: *const RegDescription) Error![]Move {
    std.debug.assert(desc.scratch.len == desc.classes.len);
    var out: std.ArrayList(Move) = .empty;
    errdefer out.deinit(allocator);
    for (0..desc.classes.len) |ci| {
        const class_idx: u16 = @intCast(ci);
        try orderClassMoves(allocator, class_idx, raw, desc.scratch[ci], &out);
    }
    return out.toOwnedSlice(allocator);
}

/// True iff some pending move OTHER than `self_i` still reads `loc` as its source (so writing `loc`
/// now would clobber a value not yet moved).
fn readByOther(pending: []const Move, self_i: usize, loc: Location) bool {
    for (pending, 0..) |m, j| {
        if (j == self_i) continue;
        if (locEql(m.src, loc)) return true;
    }
    return false;
}

/// The standard parallel-move sequencing for one class. Repeatedly emit a move whose destination no
/// other pending move still reads (safe to overwrite). When only cycles remain, break one by routing
/// a node's value through the class scratch register: emit `scratch <- src`, retarget that move to
/// read the scratch, and continue (the location it used to read is now free, unblocking the rest of
/// the cycle). A slot->slot move is expanded to `scratch <- slot` then `slot <- scratch` at emit time
/// (memory cannot move to memory in one op). Appends the ordered primitive moves to `out`.
///
/// Correctness rests on two structural facts about THIS allocator's edge moves: a spill slot names a
/// unique interval, so a slot is never both a source and a destination on one edge. Therefore every
/// cycle consists only of reg->reg moves (the scratch save/restore is reg->reg), and a slot->slot
/// move is always independent (its slot destination is read by no other move), so it is emitted in
/// the first drain while the scratch is free, never nested inside a scratch-held cycle.
fn orderClassMoves(
    allocator: std.mem.Allocator,
    class_idx: u16,
    raw: []const Move,
    scratch_reg: u16,
    out: *std.ArrayList(Move),
) Error!void {
    var pending: std.ArrayList(Move) = .empty;
    defer pending.deinit(allocator);
    for (raw) |m| {
        if (m.class == class_idx) try pending.append(allocator, m);
    }
    if (pending.items.len == 0) return;

    const scratch_loc: Location = .{ .reg = scratch_reg };
    // The scratch register is reserved, so no value ever lives there: no raw move may touch it.
    for (pending.items) |m| {
        std.debug.assert(!locEql(m.src, scratch_loc));
        std.debug.assert(!locEql(m.dst, scratch_loc));
    }

    const out_start = out.items.len;
    var scratch_busy = false;
    // Each loop iteration either emits one pending move (shrinking `pending`) or breaks one cycle
    // (which unblocks at least one emit next), so the count is bounded by twice the move count.
    const bound: usize = pending.items.len * 2 + 4;
    var iters: usize = 0;
    while (pending.items.len > 0) {
        iters += 1;
        std.debug.assert(iters <= bound);

        var free_idx: ?usize = null;
        for (pending.items, 0..) |m, i| {
            if (readByOther(pending.items, i, m.dst)) continue;
            free_idx = i;
            break;
        }

        if (free_idx) |i| {
            const m = pending.orderedRemove(i);
            try emitPrimitive(allocator, out, m, scratch_loc, class_idx, &scratch_busy);
            continue;
        }

        // Only cycles remain, and every cycle node is reg->reg. Break one by saving its source into
        // the scratch, then reading the scratch in its place.
        std.debug.assert(!scratch_busy);
        const m0 = &pending.items[0];
        std.debug.assert(locIsReg(m0.src) and locIsReg(m0.dst));
        // The save routes m0's value through the scratch, so it carries m0's value for the width.
        try out.append(allocator, .{ .src = m0.src, .dst = scratch_loc, .class = class_idx, .value = m0.value });
        scratch_busy = true;
        m0.src = scratch_loc;
    }
    std.debug.assert(!scratch_busy);

    if (std.debug.runtime_safety) {
        try assertOrderingValid(allocator, class_idx, raw, out.items[out_start..]);
    }
}

/// Emit one ordered move as backend-primitive op(s): a reg source (reg->reg move or reg->slot store)
/// and a slot->reg load pass through unchanged, while a slot->slot shuffle expands to a load into the
/// scratch then a store out of it. A move that READS the scratch closes a broken cycle, so the
/// scratch is free again after it.
fn emitPrimitive(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(Move),
    m: Move,
    scratch_loc: Location,
    class_idx: u16,
    scratch_busy: *bool,
) Error!void {
    const closes = locEql(m.src, scratch_loc);
    switch (m.src) {
        .reg => try out.append(allocator, m),
        .slot => switch (m.dst) {
            .reg => try out.append(allocator, m),
            .slot => {
                // Memory-to-memory needs the scratch, which is free outside a cycle break. Both halves
                // route `m`'s value through the scratch, so they carry its value for the width.
                std.debug.assert(!scratch_busy.*);
                try out.append(allocator, .{ .src = m.src, .dst = scratch_loc, .class = class_idx, .value = m.value });
                try out.append(allocator, .{ .src = scratch_loc, .dst = m.dst, .class = class_idx, .value = m.value });
            },
        },
    }
    if (closes) scratch_busy.* = false;
}

/// Validate an ordered class sequence realizes the raw parallel move: simulate each location's
/// contents (a token per original source) through the ordered ops, then assert every raw move's
/// destination ends holding its source's original value. Catches any read-after-overwrite ordering
/// bug (including a mishandled cycle or scratch clobber). Debug-only (allocates a scratch map).
fn assertOrderingValid(
    allocator: std.mem.Allocator,
    class_idx: u16,
    raw: []const Move,
    ordered: []const Move,
) Error!void {
    var content: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    defer content.deinit(allocator);
    for (raw) |m| {
        if (m.class != class_idx) continue;
        const k = locKey(m.src);
        try content.put(allocator, k, k);
    }
    for (ordered) |m| {
        // A source must have been initialized (an original source, or the scratch written earlier).
        const val = content.get(locKey(m.src)).?;
        try content.put(allocator, locKey(m.dst), val);
    }
    for (raw) |m| {
        if (m.class != class_idx) continue;
        const got = content.get(locKey(m.dst)).?;
        std.debug.assert(got == locKey(m.src));
    }
}

/// Debug assertion that the function has NO critical edge (a `>1`-successor `if` block feeding a
/// `>1`-predecessor block). Resolution assumes Task-8 wiring split them first; a violation is a
/// wiring bug, surfaced here rather than as a silent miscompile.
fn assertNoCriticalEdges(allocator: std.mem.Allocator, func: *const Function) Error!void {
    const nblocks = func.blockCount();
    if (nblocks == 0) return;

    const pred_count = try allocator.alloc(u32, nblocks);
    defer allocator.free(pred_count);
    @memset(pred_count, 0);
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                pred_count[@intFromEnum(cf.then.target)] += 1;
                pred_count[@intFromEnum(cf.@"else".target)] += 1;
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .jump => |j| pred_count[@intFromEnum(j.target)] += 1,
            .ret => {},
        };
    }

    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                // The `if` block already has two successors, so either target with more than one
                // predecessor makes that edge critical.
                std.debug.assert(pred_count[@intFromEnum(cf.then.target)] <= 1);
                std.debug.assert(pred_count[@intFromEnum(cf.@"else".target)] <= 1);
            }
        }
    }
}

// ===========================================================================
// Task 9: the debug verifier (VERIFYINTERVALS).
//
// A white-box checker that validates a COMPLETED allocation and runs inside
// `allocate` whenever `std.debug.runtime_safety` is on (every test and debug
// build), so an allocation bug fails loudly at an assert instead of silently
// miscompiling. It never runs under ReleaseFast, so the production JIT keeps
// its speed. It operates on the final intervals (the originals AND the split
// children flattened into one slice) and checks three soundness properties:
//
//   1. REGISTER EXCLUSIVITY: no two same-class intervals in the same physical
//      register have overlapping live ranges (the core soundness property).
//   2. MUST_HAVE_REGISTER: every `must_have_register` use is covered by a
//      register-located interval of its value, never only a spill slot.
//   3. ASSIGNMENT: every value interval with a use was placed somewhere.
//
// The one legitimate same-(class, reg) overlap, a value interval and its OWN
// entry-parameter fixed interval at entry, is exempt from check 1.
// ===========================================================================

/// One thing the allocation got wrong, found by `verifyIntervals`. `a` and `b` index the intervals
/// slice passed to the verifier (`b == a` for the single-interval checks); `pos` is the program
/// position the violation manifests at.
pub const Violation = struct {
    kind: enum { reg_overlap, must_have_spilled, unassigned },
    a: usize,
    b: usize,
    pos: u32,
};

/// The physical register an interval OCCUPIES, or null when it holds none (a spilled value interval,
/// or a value interval the scan never placed). A fixed interval occupies its `fixed_reg`; a placed
/// value interval occupies its assigned `.reg`. The nullable analogue of `assignedReg`.
fn occupiedReg(it: *const Interval) ?u16 {
    if (it.fixed_reg) |fr| return fr;
    const loc = it.location orelse return null;
    return switch (loc) {
        .reg => |r| r,
        .slot => null,
    };
}

/// True iff the pair is a value interval and its OWN entry-parameter fixed interval, the single
/// legitimate same-(class, reg) overlap (the ABI register they share at entry over `[0, 1)`). Exactly
/// one of the pair is a fixed interval (`fixed_reg != null`) and both carry the same pinned value.
fn isEntryParamHintPair(a: *const Interval, b: *const Interval) bool {
    const av = a.value orelse return false;
    const bv = b.value orelse return false;
    if (av != bv) return false;
    return (a.fixed_reg != null) != (b.fixed_reg != null);
}

/// The earliest position two same-(class, reg) intervals genuinely conflict at, or null. When one is a
/// CALL-CLOBBER fixed interval (`value == null`) the read-before-clobber rule applies (a value may READ
/// an operand in a caller-saved register AT the call and die before the clobber takes effect), so the
/// conflict is `fixedClobberConflict`, EXACTLY the rule the scan itself allocated by. Any other pair
/// conflicts wherever their ranges intersect.
fn occupancyConflict(a: *const Interval, b: *const Interval) ?u32 {
    if (a.fixed_reg != null and a.value == null) return fixedClobberConflict(b, a);
    if (b.fixed_reg != null and b.value == null) return fixedClobberConflict(a, b);
    return a.nextIntersection(b);
}

/// True iff some `.reg`-located interval of value `v` covers `pos` (i.e. `v` is in a register there).
fn valueInRegAt(intervals: []const Interval, v: Value, pos: u32) bool {
    for (intervals) |*it| {
        if (it.fixed_reg != null) continue;
        const iv = it.value orelse continue;
        if (iv != v) continue;
        const loc = it.location orelse continue;
        switch (loc) {
            .reg => if (it.covers(pos)) return true,
            .slot => {},
        }
    }
    return false;
}

/// Verify a completed allocation, returning every soundness `Violation` (empty slice = valid). The
/// caller owns and frees the returned slice. See the section header for the three checks. The input is
/// the flattened final intervals (originals plus split children); it is read-only and not freed here.
pub fn verifyIntervals(allocator: std.mem.Allocator, intervals: []const Interval) Error![]Violation {
    var violations: std.ArrayList(Violation) = .empty;
    errdefer violations.deinit(allocator);

    // CHECK 1: register exclusivity. Every pair of same-class intervals occupying the same physical
    // register must have disjoint live ranges, except a value interval and its own entry-param fixed
    // interval sharing the ABI register at entry.
    for (intervals, 0..) |*ia, i| {
        const ra = occupiedReg(ia) orelse continue;
        for (intervals[i + 1 ..], i + 1..) |*ib, j| {
            if (ia.class != ib.class) continue;
            const rb = occupiedReg(ib) orelse continue;
            if (ra != rb) continue;
            if (isEntryParamHintPair(ia, ib)) continue;
            if (occupancyConflict(ia, ib)) |pos| {
                try violations.append(allocator, .{ .kind = .reg_overlap, .a = i, .b = j, .pos = pos });
            }
        }
    }

    // CHECK 2: must_have_register satisfaction. Every `must_have_register` use of a value must fall in a
    // `.reg`-located interval of that same value; a use covered ONLY by a `.slot` interval would read an
    // operand from memory where the target forbids it.
    for (intervals, 0..) |*ia, i| {
        if (ia.fixed_reg != null) continue;
        const va = ia.value orelse continue;
        for (ia.uses) |u| {
            if (u.kind != .must_have_register) continue;
            if (!valueInRegAt(intervals, va, u.pos)) {
                try violations.append(allocator, .{ .kind = .must_have_spilled, .a = i, .b = i, .pos = u.pos });
            }
        }
    }

    // CHECK 3: assignment. Every value interval that has at least one use must have been placed.
    for (intervals, 0..) |*ia, i| {
        if (ia.fixed_reg != null) continue;
        if (ia.value == null) continue;
        if (ia.uses.len == 0) continue;
        if (ia.location == null) {
            try violations.append(allocator, .{ .kind = .unassigned, .a = i, .b = i, .pos = ia.start() });
        }
    }

    return violations.toOwnedSlice(allocator);
}
