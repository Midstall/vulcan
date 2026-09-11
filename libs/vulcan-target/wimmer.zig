//! Shared, target-independent Wimmer-Franz register allocator (Wimmer & Franz, CGO 2010).
//! This module owns the ALGORITHM. Each backend owns ENCODING and supplies a `RegDescription`
//! that describes its register model to the allocator.
//!
//! Physical registers are an ABSTRACTION here. The allocator only ever sees a `u16` register
//! INDEX within a class. The backend picks a stable numbering and maps the index back to its
//! own register enum. See each backend's `*RegDescription` for the numbering it picked. For
//! example, aarch64 uses the register's own enum integer value. So gpr class index n names x_n,
//! and fpr class index n names v_n.
//!
//! A value normally occupies ONE register. A backend that supplies `RegDescription.regWidth` may
//! give a value a span of N CONSECUTIVE registers under an alignment, which is what an NVIDIA 64-bit
//! address pair needs. `Location.reg` then names the BASE of the span, and every place the allocator
//! picks, blocks, or compares a register works over the whole span. See `RegWidth`.

const std = @import("std");
const ir = @import("vulcan-ir");

const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;
const Function = ir.function.Function;
const Terminator = ir.function.Terminator;
const Jump = ir.function.Jump;

/// Interval construction can fail only with an out-of-memory error.
pub const Error = std.mem.Allocator.Error;

/// Where a value lives at a program point: a physical register (a class-relative INDEX) or a
/// spill slot. The value's `classOf` sets the class.
pub const Location = union(enum) { reg: u16, slot: u32 };

/// How many CONSECUTIVE physical registers ONE value occupies, and the alignment its BASE register
/// index must satisfy. The default, one register at any index, is the model every backend used
/// before this type existed.
///
/// A value with `regs = 2, alignment = 2` occupies `R(n)` and `R(n + 1)` with `n` even, which is how
/// an NVIDIA 64-bit address is held. `Location.reg` always names the BASE of the span, and the
/// backend derives the other registers of the span from that base.
///
/// A width above one puts TWO obligations on the backend:
///   1. `RegClass.slot_bytes` must be large enough for the WIDEST value of the class, because a
///      spilled value still gets exactly ONE slot. This is the rule aarch64 already follows to keep
///      a 128-bit vector in a 16-byte slot.
///   2. The class `scratch`, and `scratch2` when it is supplied, must reserve a whole ALIGNED span
///      of that width, because a move the resolver routes through the scratch carries a whole value.
/// `allocate` checks obligation 2 while safety checks are on. Obligation 1 is a byte count the
/// allocator cannot see, so it stays a documented contract.
pub const RegWidth = struct {
    regs: u16 = 1,
    alignment: u16 = 1,
};

/// Whether an operand use requires a register. A `must_have_register` operand cannot read from a
/// spill slot on this target. This is the safe, conservative default. A `should_have_register`
/// operand may fold or reload from a slot. A target that supports this, for example x86 memory
/// operands, relaxes specific opcodes.
pub const UseKind = enum { must_have_register, should_have_register };

/// One register class, for example gpr or fpr. `allocatable` is the set of physical register
/// INDICES the scan may hand out for this class. `callee_saved` is the subset that needs a
/// prologue save and restore when used. `slot_bytes` is the spill-slot size for a value of this
/// class.
pub const RegClass = struct {
    name: []const u8,
    allocatable: []const u16,
    callee_saved: []const u16,
    slot_bytes: u16,
    // OPTIONAL subset of registers whose CALL clobber only destroys a WIDE value, preserving a
    // NARROW one (a value the backend's `RegDescription.isNarrow` flags). This models an ABI that
    // saves only the low half of certain callee-saved registers across a call: aarch64 AAPCS64
    // preserves the low 64 bits of v8..v15, so a scalar float there survives a call while a 128-bit
    // vector loses its upper half. A value flagged `narrow` therefore does NOT conflict with a call
    // clobber of a register in this set, so it may stay resident across the call. Empty (the default)
    // means every clobber destroys every value, byte-identical to the prior behavior.
    narrow_preserved: []const u16 = &.{},
};

/// A per-class set of register indices clobbered at some point. Used by `CallSite`.
pub const ClassRegs = struct { class: u16, regs: []const u16 };

/// A call at position `pos` clobbers `clobbered`, one `ClassRegs` per affected class. The
/// allocator turns each into a fixed interval, so no value survives the call in a clobbered
/// register.
pub const CallSite = struct { pos: u32, clobbered: []const ClassRegs };

/// An entry parameter, or any value, pre-colored to a fixed physical register of a class. This
/// covers the ABI argument registers at function entry.
pub const FixedAssign = struct { value: Value, class: u16, reg: u16 };

/// A backend's description of its register model for one function. It is built per function
/// because the allocatable sets can differ by function, for example aarch64 leaf vs non-leaf
/// pools. `classOf` and `useKind` receive the backend `ctx`, so they can consult backend helpers.
/// `aarch64RegDescription`, and each backend's builder, allocates the owned slices. Call `deinit`
/// to free them.
pub const RegDescription = struct {
    classes: []const RegClass,
    classOf: *const fn (ctx: *const anyopaque, func: *const Function, v: Value) u16,
    useKind: *const fn (ctx: *const anyopaque, func: *const Function, inst: Inst, operand: Value) UseKind,
    entry_fixed: []const FixedAssign,
    call_sites: []const CallSite,
    scratch: []const u16,
    // OPTIONAL second per-class scratch register, one per class, reserved exactly like `scratch`
    // (never allocatable, no raw edge move touches it). It exists ONLY to break a parallel-move CYCLE
    // that involves a spill slot both a source and a destination on one edge: the cycle value is held
    // here while `scratch` stays free to realize any slot-to-slot memory move drained during the hold.
    // A backend provides it (a spare reserved temp) to let `coalesce_spill_slots` commit on high-
    // pressure functions instead of reverting. Empty (the default) means "not provided": the resolver
    // keeps its single-scratch reg-only cycle break and `coalesceSpillSlots` keeps its revert guard, so
    // a backend without a spare temp (x86-32) is unaffected and byte-identical.
    scratch2: []const u16 = &.{},
    ctx: *const anyopaque,
    // OPTIONAL interference-aware copy coalescing hook. Given a value `v`, return the source value
    // when the instruction that defines `v` is a PURE same-class register copy of that source (a
    // plain `mov`/`fmov` the backend emits with no bit change), else null. The allocator uses it to
    // place a copy destination and its source on one register when the source dies at the copy, so
    // the copy becomes a no-op. Null (the default) keeps the allocator's behavior byte-identical, so
    // a backend that does not opt in is unaffected. Only a case the backend lowers to an EXACT plain
    // copy is safe to report. A widening or narrowing that changes bits must NOT be reported.
    copySource: ?*const fn (ctx: *const anyopaque, func: *const Function, v: Value) ?Value = null,
    // OPTIONAL narrow-value predicate. Return true when value `v` survives a call clobber of a
    // register in its class's `narrow_preserved` set (i.e. it fits in the preserved low half). On
    // aarch64 this is a SCALAR float (not a 128-bit vector), which AAPCS64 preserves in v8..v15
    // across a call. Consulted once per value at interval-build time to set `Interval.narrow`. Null
    // (the default) flags nothing narrow, byte-identical to the prior behavior.
    isNarrow: ?*const fn (ctx: *const anyopaque, func: *const Function, v: Value) bool = null,
    // OPTIONAL block-argument coalescing. When true, the scan hints a block parameter toward the
    // register of an incoming argument that is already placed, so the edge move that feeds the
    // parameter becomes a same-register no-op the edge resolver drops. A parameter and its incoming
    // argument live in different blocks and never interfere (the argument dies on the edge, before
    // the parameter is born), so this is a pure PREFERENCE, never a correctness change. Combined with
    // `copySource`, it lets a copy chain (source -> convert -> block parameter) collapse onto one
    // register. False (the default) keeps the allocator byte-identical for a backend that does not
    // opt in.
    coalesce_block_params: bool = false,
    // OPTIONAL spill-slot coalescing. When true, after the scan a block parameter that SPILLED shares
    // ONE spill slot with an incoming argument that also spilled, when the two do not interfere. The
    // edge move that fed the parameter then becomes a same-slot no-op the edge resolver drops, so a
    // `ldr scratch,[arg_slot]; str scratch,[param_slot]` pair disappears. Coalescing is interference-
    // checked (two simultaneously live values never share a slot) and guarded so the parallel-move
    // ordering precondition still holds (no slot is both a source and a destination on one edge); if
    // the guard ever fails the whole coalescing is dropped and the byte-identical distinct-slot
    // placement stands. False (the default) keeps the allocator byte-identical for a non-opting backend.
    coalesce_spill_slots: bool = false,
    // OPTIONAL multi-register width hook. Return how many CONSECUTIVE registers value `v` occupies
    // and the alignment its base register index must satisfy. It exists because an NVIDIA 64-bit
    // address lives in an EVEN-ALIGNED GPR pair `R(n):R(n + 1)`, the first value in this project that
    // needs more than one architectural register. aarch64 and x86_64 hold a 128-bit vector in ONE
    // register, so they never need it. The hook is read once per value while the intervals are built,
    // and once per entry-parameter pin, so its answer must not change for a value inside one
    // function. Null (the default) gives every value exactly one register at any index, byte-identical
    // to the prior behavior for a backend that does not opt in. Read the `RegWidth` doc comment for
    // the two obligations a width above one puts on the backend.
    regWidth: ?*const fn (ctx: *const anyopaque, func: *const Function, v: Value) RegWidth = null,

    /// Free every owned slice the backend builder allocated: each class's `allocatable` and
    /// `callee_saved`, the `classes` slice, each call site's per-class `regs` and its `clobbered`
    /// slice, the `call_sites` slice, `entry_fixed`, and `scratch`. Class names and `ctx` are
    /// static, so they are not owned.
    pub fn deinit(self: *RegDescription, allocator: std.mem.Allocator) void {
        for (self.classes) |c| {
            allocator.free(c.allocatable);
            allocator.free(c.callee_saved);
            if (c.narrow_preserved.len > 0) allocator.free(c.narrow_preserved);
        }
        allocator.free(self.classes);
        for (self.call_sites) |cs| {
            for (cs.clobbered) |cr| allocator.free(cr.regs);
            allocator.free(cs.clobbered);
        }
        allocator.free(self.call_sites);
        allocator.free(self.entry_fixed);
        allocator.free(self.scratch);
        if (self.scratch2.len > 0) allocator.free(self.scratch2);
        self.* = undefined;
    }
};

// ===========================================================================
// Lifetime intervals (BUILDINTERVALS, Wimmer & Franz Fig 4).
//
// A value's lifetime is a set of half-open live RANGES with HOLES between the
// regions where it is dead, plus the positions it is USED at. Physical
// registers are constrained by FIXED intervals: one per call-clobbered
// register (blocking it over each call) and one per entry parameter (pinning
// its ABI register at function entry). No allocation happens here. The scan
// step that follows consumes these intervals.
//
// Position numbering matches the aarch64 backend's `linearize` EXACTLY, so the
// `RegDescription.call_sites` positions, built with that same numbering, line
// up. Blocks are walked in block-index order 0..nblocks (NOT reverse-post
// order). A block's parameter row shares the block's start position, then adds
// 1 per instruction and 1 for the terminator slot. Blocks are numbered
// contiguously: `block_from[bi+1] == block_to[bi]`.
// ===========================================================================

/// A half-open live range `[from, to)`. A value live over disjoint ranges has HOLES between them.
pub const Range = struct { from: u32, to: u32 };

/// A use of a value at position `pos` with the register requirement the backend reported.
pub const UsePos = struct { pos: u32, kind: UseKind };

/// One lifetime interval. A VALUE interval (`fixed_reg == null`) describes where an SSA value is
/// live and used. A FIXED interval (`fixed_reg != null`) blocks a physical register over its
/// ranges. A call-clobber fixed interval carries `value == null`. An entry-parameter fixed
/// interval keeps `value` set to the pinned parameter, so the scan can honor the ABI hint.
pub const Interval = struct {
    value: ?Value,
    class: u16,
    fixed_reg: ?u16,
    ranges: []Range, // ascending, disjoint, merged
    uses: []UsePos, // ascending by `pos`
    location: ?Location = null, // filled by the scan, null here
    // For a value interval that is a COALESCABLE COPY DESTINATION, the source value the backend's
    // `copySource` reported. It means: the instruction that defines this value is a pure register
    // copy of `copy_src` in the same class (a plain `mov`/`fmov` that changes no bits). When the
    // source dies AT the copy (its last live position is this value's def), the allocator may place
    // both on ONE register, which turns the copy into a no-op the backend elides. Null for a
    // non-copy value, and for every backend that does not set `RegDescription.copySource`.
    copy_src: ?Value = null,
    // For a VALUE interval, true when the backend's `RegDescription.isNarrow` flagged this value as
    // one that survives a clobber of a `RegClass.narrow_preserved` register (e.g. a scalar float in
    // aarch64 v8..v15, whose low 64 bits AAPCS64 preserves across a call). A narrow value may stay
    // register-resident across such a clobber. Inherited by split children. False by default.
    narrow: bool = false,
    // For a FIXED call-clobber interval, true when its register is in the class's `narrow_preserved`
    // set, so the clobber spares a `narrow` value. False by default (a full clobber, or a value
    // interval).
    preserves_narrow: bool = false,
    // How many CONSECUTIVE registers this interval occupies, and the alignment its base register
    // must meet. Both come from the backend's `RegDescription.regWidth` for the interval's value.
    // They stay 1 and 1 for every backend that does not opt in, and for a CALL-CLOBBER fixed
    // interval, which always blocks exactly one register. An ENTRY-PARAMETER fixed interval carries
    // the width of the parameter it pins, so the pin blocks the whole span. A split child inherits
    // both from its parent, because a split cuts a lifetime and never changes a value's width.
    regs: u16 = 1,
    reg_align: u16 = 1,

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
            // Advance whichever range ends first. It cannot intersect any later range of the other.
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
        .atomic_rmw => |a| {
            f(ctx, a.ptr, false);
            f(ctx, a.value, false);
            if (a.compare) |c| f(ctx, c, false);
        },
        // A barrier carries no Value operand to visit.
        .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
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
        .va_start => |vs| f(ctx, vs.list, false),
        .va_arg => |va| f(ctx, va.list, false),
        .va_end => |ve| f(ctx, ve.list, false),
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
        .call => |c| {
            for (func.valueList(c.args)) |a| f(ctx, a, false);
            if (c.ret_dest) |rd| f(ctx, rd, false); // the register-return dest, read post-call
        },
        .call_indirect => |c| {
            f(ctx, c.target, false);
            for (func.valueList(c.args)) |a| f(ctx, a, false);
            if (c.ret_dest) |rd| f(ctx, rd, false); // the register-return dest, read post-call
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
        .ret => |r| for (r.slice()) |vv| f(ctx, vv, false),
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

    // Per-value range and use builders. Ranges are appended raw, and possibly overlapping, then
    // normalized once at the end. A raw append plus a single sort and merge is simpler than an
    // incremental insert-and-merge step, and it gives the same disjoint result.
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
            const vi = @intFromEnum(v);
            self.used_row[vi] = true;
            const kind: UseKind = if (is_edge_arg)
                // An edge argument feeds a successor block parameter through the parallel move the
                // resolver realizes on the edge. That move can load from or store to a spill slot
                // (orderMoves routes a slot end through the class scratch), so an edge argument does
                // NOT need a register. Marking it should_have lets a high-pressure loop back-edge
                // leave a carried value in a slot instead of demanding a register for EVERY carried
                // value at the single terminator position, a demand no register-poor target (i386,
                // x86_64) can satisfy once the value count passes the register file size.
                .should_have_register
            else if (self.term_kind)
                // A ret value normally goes to its ABI return register at the block boundary, so it
                // needs a register. The exception is a value whose class has NO allocatable register
                // (a memory-resident class, for example riscv64/i386 binary128, which lives in a
                // stack slot and moves into its ABI location, a GPR pair or the stack, only at the
                // ret): it can never occupy a register, so a `must_have_register` demand would be
                // unsatisfiable. The terminator emitter reads it straight from its slot, so mark it
                // `should_have_register` instead. Non-empty classes are unaffected (byte-identical).
                (if (self.desc.classes[self.desc.classOf(self.desc.ctx, self.func, v)].allocatable.len == 0) UseKind.should_have_register else .must_have_register)
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

    // --- Liveness fixpoint: live_out[b] is the union of successors' live_in. live_in[b] is used[b]
    // union (live_out[b] minus defined[b]). Back-edges make a loop header's live-in flow into the
    // body's live-out, so loop-carried values stay live across the whole body with no separate
    // loop pass. A block parameter is `defined` in its block, so it is never live-in from an edge.
    // The edge ARGUMENT that feeds it is `used` in the predecessor. The computation is monotonic,
    // so the fixpoint terminates. ---
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
            if (first.from <= def_pos[v] and def_pos[v] < first.to) {
                first.from = def_pos[v];
            } else {
                // Defensive path: the def is reachable but every use of the value lies in an
                // unreachable block, so its ranges were built entirely from the dead region and the
                // earliest one does not contain the def. This is only reachable when a caller skipped
                // `ir.reachable.neutralizeUnreachable` (which empties dead blocks so no such range is
                // ever built). Degrade the value to a single dead-def range instead of crashing: drop
                // the dead-region ranges and keep only [def, def+1). For an all-reachable function the
                // invariant always holds, so this branch is never taken and the output is unchanged.
                range_lists[v].clearRetainingCapacity();
                try range_lists[v].append(allocator, .{ .from = def_pos[v], .to = def_pos[v] + 1 });
            }
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
        const class = desc.classOf(desc.ctx, func, value);
        // Record the copy source when the backend opts into coalescing and reports this value as a
        // pure same-class copy. The same-class guard is defensive: a cross-class copy shares no
        // register pool, so it can never coalesce. Whether the source actually DIES at the copy is
        // checked later, at the allocation and verification sites, from the source's live range.
        var copy_src: ?Value = null;
        if (desc.copySource) |hook| {
            if (hook(desc.ctx, func, value)) |s| {
                if (desc.classOf(desc.ctx, func, s) == class) copy_src = s;
            }
        }
        const narrow = if (desc.isNarrow) |hook| hook(desc.ctx, func, value) else false;
        // The register width is read ONCE here and carried on the interval, so every later site
        // (the scan, the splitter, the resolver, the verifier) reads one answer.
        const width = widthOf(desc, func, value);
        try result.append(allocator, .{
            .value = value,
            .class = class,
            .fixed_reg = null,
            .ranges = rs,
            .uses = us,
            .copy_src = copy_src,
            .narrow = narrow,
            .regs = width.regs,
            .reg_align = width.alignment,
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
        const preserves_narrow = containsReg(desc.classes[key.class].narrow_preserved, key.reg);
        try result.append(allocator, .{
            .value = null,
            .class = key.class,
            .fixed_reg = key.reg,
            .ranges = rs,
            .uses = us,
            .preserves_narrow = preserves_narrow,
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
        // The pin covers the parameter's WHOLE register span. A two-register parameter pinned at
        // `R(n)` owns `R(n)` and `R(n + 1)` at entry, so a pin that blocked only the base would let
        // the scan hand `R(n + 1)` to another value that is live at entry.
        const width = widthOf(desc, func, ef.value);
        try result.append(allocator, .{
            .value = ef.value,
            .class = ef.class,
            .fixed_reg = ef.reg,
            .ranges = rs,
            .uses = us,
            .regs = width.regs,
            .reg_align = width.alignment,
        });
    }
}

// ===========================================================================
// The linear scan (LINEARSCAN, Wimmer & Franz Fig 5) plus TRYALLOCATEFREEREG
// (Fig 6), extended with ALLOCATEBLOCKEDREG (Fig 7), SPLITINTERVAL, and
// spill-slot assignment. The scan now handles register pressure. When no
// whole register is free, it splits live ranges and spills, so a value may
// span MULTIPLE intervals with different locations. The output is the full
// `Allocation` the later resolution and emission steps consume: its
// per-value multi-segment map, per-class slot counts, and
// `used_callee_saved`.
// ===========================================================================

/// One placement of a value: from position `from`, the value lives at `loc`. A value that never
/// splits has a single segment. The splitter produces multi-segment values.
pub const Segment = struct { from: u32, loc: Location };

/// A data move the resolver emits: move `src` to `dst` within `class`. `value` is the IR value
/// whose bits this move transfers. It is carried so a backend can look up its type and pick the
/// width-appropriate move, store, or load, for example x86 movups for a 128-bit vector vs vmovups
/// for 256-bit. It is populated for EVERY move, including the scratch cycle-break and
/// slot-to-slot routing steps. Each of these routes one specific value's bits through the class
/// scratch. aarch64 and riscv64 emit their vector moves at a fixed width, so they ignore this
/// field.
pub const Move = struct { src: Location, dst: Location, class: u16, value: Value };

/// An intra-block spill, reload, or move the resolver emits at position `at`. A same-position
/// CLUSTER of these is a parallel move: all sources read from the pre-instruction state, and all
/// destinations are written. So `buildAllocation` orders each cluster through the SAME routine the
/// control-flow edges use, `orderMoves`. Every source is read before it is overwritten, register
/// cycles are broken through the class scratch, and a slot-to-slot shuffle is expanded through the
/// scratch too. `value` carries the IR value whose bits the action transfers, so a width-aware
/// backend can pick the move, store, or load form. It survives the scratch routing, since each
/// step carries the routed value. Draining the resulting list in order is hazard-free.
pub const Action = struct {
    at: u32,
    kind: enum { store, reload, move },
    class: u16,
    src: Location,
    dst: Location,
    value: Value,
};

/// The parallel move set on a control-flow edge: resolution and block-param moves. Unused here.
pub const EdgeMoves = struct { pred: Block, succ: Block, moves: []Move };

/// A callee-saved physical register that the allocation actually used, so the prologue must save it.
pub const UsedSaved = struct { class: u16, reg: u16 };

/// The register allocation result. The scan fills `segments`, one register segment per value,
/// `slot_count_per_class`, all zero until something spills, and `used_callee_saved`. The
/// remaining fields belong to later processing steps and stay empty here.
pub const Allocation = struct {
    segments: std.AutoHashMapUnmanaged(Value, []Segment) = .empty,
    actions: []Action = &.{},
    edge_moves: []EdgeMoves = &.{},
    slot_count_per_class: []u32 = &.{},
    used_callee_saved: []UsedSaved = &.{},
    /// True when some value's location CHANGES across a block boundary. This is a segment
    /// transition whose two sides fall in different blocks. Realizing that change needs a
    /// control-flow-edge move, the job of cross-block resolution. The intra-block `actions` here
    /// do NOT cover it. A backend that emits from this allocation before resolution runs must bail
    /// when this flag is set, rather than silently drop the edge move. This is false for a
    /// single-block function, or any function whose every value keeps one location per block.
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

/// `allocate`'s failure modes: out of memory, or `error.Unsupported` when a single position demands
/// more simultaneous must-have registers than the class has. This bail is deliberate. It matches
/// the OLD allocator's "too many live params" rejection of the same shape. The splitter handles
/// every OTHER register-pressure case by splitting and spilling. This is the residual case where
/// nothing is left to split.
pub const AllocateError = Error || error{Unsupported};

/// A free-until position meaning "never conflicts". Program positions never reach it.
const infinity: u32 = std.math.maxInt(u32);

/// The widest physical register index the freeUntilPos bookkeeping supports. aarch64 uses 0..31.
const max_phys_regs: usize = 64;

fn intervalStartLessThan(_: void, a: *Interval, b: *Interval) bool {
    return a.start() < b.start();
}

/// The physical register an interval currently occupies. A fixed interval blocks its `fixed_reg`.
/// A placed value interval lives in its assigned register. It is a programmer error to call this
/// on an unplaced or spilled value interval, since the scan keeps every active or inactive value
/// in a register.
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

/// The register width value `v` occupies, from the backend's optional `regWidth` hook. A backend
/// that does not supply the hook gets one register at any index, which is what every backend but
/// NVIDIA needs.
fn widthOf(desc: *const RegDescription, func: *const Function, v: Value) RegWidth {
    const hook = desc.regWidth orelse return .{};
    const w = hook(desc.ctx, func, v);
    // A width of zero holds no bits, and an alignment of zero divides nothing. Both are programmer
    // errors in the backend, not runtime faults, so they assert.
    std.debug.assert(w.regs >= 1);
    std.debug.assert(w.alignment >= 1);
    return w;
}

/// True iff the register spans `[base_a, base_a + regs_a)` and `[base_b, base_b + regs_b)` share at
/// least one register. This is the WHOLE-SPAN test that replaces a base-only equality compare. A
/// check that tests only the base lets a two-register value sit on top of its neighbor, which is a
/// silent wrong-answer bug, so every interference site uses this.
fn spansOverlap(base_a: u16, regs_a: u16, base_b: u16, regs_b: u16) bool {
    return base_a < base_b +| regs_b and base_b < base_a +| regs_a;
}

/// True iff `[base, base + regs)` is a legal placement for a value of this alignment: the base meets
/// the alignment, the span fits inside the per-register bookkeeping, and every register of the span
/// is in the candidate set. When `evictable` is given, every register of the span must be evictable
/// too. For a one-register value with alignment 1 this reduces to the candidate (and evictable) flag
/// of the single register, which is what the scan tested before multi-register values existed.
fn spanPlaceable(
    is_candidate: *const [max_phys_regs]bool,
    evictable: ?*const [max_phys_regs]bool,
    base: usize,
    regs: u16,
    alignment: u16,
) bool {
    if (base % alignment != 0) return false;
    if (base + regs > max_phys_regs) return false;
    for (base..base + regs) |r| {
        if (!is_candidate[r]) return false;
        if (evictable) |e| {
            if (!e[r]) return false;
        }
    }
    return true;
}

/// The minimum of `arr` over the register span `[base, base + regs)`. A span is only as free, or as
/// unblocked, as its most constrained register. The caller checks `spanPlaceable` first, so the span
/// is in range.
fn spanMin(arr: *const [max_phys_regs]u32, base: usize, regs: u16) u32 {
    var m: u32 = infinity;
    for (base..base + regs) |r| {
        if (arr[r] < m) m = arr[r];
    }
    return m;
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
    // A NARROW value survives a clobber that only preserves the low half (aarch64: a scalar float in
    // a v8..v15 whose low 64 bits AAPCS64 keeps across a call). Such a value stays register-resident
    // across the call, so the clobber never forces it out. A WIDE value (e.g. a 128-bit vector) is
    // NOT flagged narrow and still conflicts, since the call destroys its upper half.
    if (fixed.preserves_narrow and current.narrow) return null;
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
/// every candidate register of `current`'s class, its class pool plus its entry-param hint
/// register. Clamp it by the active and inactive intervals that occupy those registers, then pick
/// the register free the longest, with ties broken toward a hint. `param_hint`, when set, is the
/// register an incoming argument holds for a block parameter, so the tie breaks toward eliding the
/// edge move. Return that register when it covers `current`'s whole lifetime, otherwise null. A null
/// result means either no register is free at all, or a register is free for a prefix only. Both
/// cases require a split, which this function defers to the blocked-register path.
fn tryAllocateFreeReg(
    current: *const Interval,
    active: []const *Interval,
    inactive: []const *Interval,
    desc: *const RegDescription,
    param_hint: ?u16,
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
        // The pin names the BASE of the parameter's span, so every register of that span joins the
        // candidate set. For a one-register value this is the single hinted register, as before.
        var k: u16 = 0;
        while (k < current.regs) : (k += 1) {
            std.debug.assert(h + k < max_phys_regs);
            is_candidate[h + k] = true;
            free_until[h + k] = infinity;
        }
    }

    // An active interval of this class occupies its register right now (free until position 0). An
    // entry-param fixed interval for `current`'s own value is the hint, not a block, so skip it.
    // When `current` is a copy of an active interval that DIES at the copy, that active interval's
    // register is NOT a block: the one instruction reads it and writes `current`, so sharing the
    // register makes the copy a no-op. Keep the register free for `current` and record it as the
    // preferred (coalesce) register, so the backend elides the move.
    var coalesce_hint: ?u16 = null;
    for (active) |it| {
        if (it.class != class_idx) continue;
        if (sameValue(it, current)) continue;
        const r = assignedReg(it);
        if (copyDiesAt(current, it)) {
            if (r < max_phys_regs and is_candidate[r]) coalesce_hint = r;
            continue;
        }
        // The occupant holds its WHOLE span right now, so every register of it is busy. Marking only
        // the base would leave the second register of a pair looking free.
        var k: u16 = 0;
        while (k < it.regs) : (k += 1) {
            const rr = r +| k;
            if (rr < max_phys_regs and is_candidate[rr]) free_until[rr] = 0;
        }
    }
    // An inactive interval of this class has a hole here, so its register is free only until the two
    // ranges next intersect. A call-clobber fixed interval (seeded into `inactive`) uses the
    // read-before-clobber rule so a call ARGUMENT is not treated as conflicting. Same hint exception.
    for (inactive) |it| {
        if (it.class != class_idx) continue;
        if (sameValue(it, current)) continue;
        const r = assignedReg(it);
        if (r >= max_phys_regs) continue;
        const x = if (it.fixed_reg != null) fixedClobberConflict(current, it) else current.nextIntersection(it);
        if (x) |xx| {
            // The reclaim takes back the occupant's WHOLE span, so clamp every register of it.
            var k: u16 = 0;
            while (k < it.regs) : (k += 1) {
                const rr = r +| k;
                if (rr < max_phys_regs and is_candidate[rr] and xx < free_until[rr]) free_until[rr] = xx;
            }
        }
    }

    // Pick the register free the longest, preferring a hint on a tie. Three hints can apply: the
    // copy-coalesce hint (a copy destination keeps its source's register, eliding the move), the
    // block-parameter hint (a parameter keeps an incoming argument's register, eliding the edge
    // move), and the entry-parameter hint (a parameter keeps its ABI register). The coalesce hint
    // wins, then the block-parameter hint, then the entry hint. These three never apply to the same
    // value (a copy destination is not a parameter, and only a block-0 parameter has an entry hint).
    // The candidate now is a BASE register, and its span is free only as long as its most
    // constrained register. For the one-register default this is exactly the old per-register scan,
    // in the same ascending order, so the pick is byte-identical.
    const pref_hint = coalesce_hint orelse param_hint orelse hint;
    var best_free: u32 = 0;
    for (0..max_phys_regs) |b| {
        if (!spanPlaceable(&is_candidate, null, b, current.regs, current.reg_align)) continue;
        const f = spanMin(&free_until, b, current.regs);
        if (f > best_free) best_free = f;
    }
    if (best_free == 0) return null;
    var chosen: ?u16 = null;
    if (pref_hint) |h| {
        if (spanPlaceable(&is_candidate, null, h, current.regs, current.reg_align) and
            spanMin(&free_until, h, current.regs) == best_free) chosen = h;
    }
    if (chosen == null) {
        for (0..max_phys_regs) |b| {
            if (!spanPlaceable(&is_candidate, null, b, current.regs, current.reg_align)) continue;
            if (spanMin(&free_until, b, current.regs) == best_free) {
                chosen = @intCast(b);
                break;
            }
        }
    }
    const reg = chosen.?;

    // A span free at least to `current.end()` (half-open) covers the whole interval. Anything less
    // would need a split, which this function does not do.
    if (best_free >= current.end()) return reg;
    return null;
}

/// True iff `dst` is a coalescable copy of `src`, and `src` DIES at the copy. That is, `dst`'s
/// defining instruction is a pure same-class copy of `src`'s value (the backend reported it through
/// `copySource`), and `src`'s last live position is `dst`'s def position. The single instruction
/// reads `src` and writes `dst`, and their live ranges touch at ONLY that one position (`src` covers
/// `[.., P + 1)`, `dst` covers `[P, ..)`). So placing both on one register makes the copy a no-op the
/// backend elides, with no loss of `src`, which is dead right after. `src` must be a real value
/// interval, not a fixed one. This is the single safety test both the scan and the verifier key off.
fn copyDiesAt(dst: *const Interval, src: *const Interval) bool {
    const cs = dst.copy_src orelse return false;
    if (src.fixed_reg != null) return false;
    const sv = src.value orelse return false;
    if (cs != sv) return false;
    // Two values can share ONE register span only when they have the same shape. A width or an
    // alignment difference means the two cover different register sets, so placing one on the other
    // would leave part of the wider one on a register the narrower one does not own. Both fields are
    // 1 for a backend with no `regWidth` hook, so this is byte-identical there.
    if (src.regs != dst.regs or src.reg_align != dst.reg_align) return false;
    return src.end() == dst.start() + 1;
}

/// True iff the two intervals are a legitimate copy-coalesce pair: one is a copy destination, the
/// other is its source that dies at the copy. Their only same-register overlap is the single copy
/// position, which is a no-op move, so the verifier accepts them sharing one register. This is the
/// coalescing analogue of `isEntryParamHintPair`.
fn isCopyCoalescePair(a: *const Interval, b: *const Interval) bool {
    return copyDiesAt(a, b) or copyDiesAt(b, a);
}

/// True iff both intervals describe the same value (used to skip a value's own entry-param hint
/// interval while computing `freeUntilPos`).
fn sameValue(a: *const Interval, b: *const Interval) bool {
    const av = a.value orelse return false;
    const bv = b.value orelse return false;
    return av == bv;
}

/// The physical register value `v` is placed in, or null when `v` has no placed register interval
/// yet. Among `v`'s intervals (the original plus any split children), it returns the register of the
/// one placed in a register that lives LATEST (the greatest `end`), which is the register `v` holds
/// as it flows out toward a block edge. It is used only to seed a block-parameter register HINT, so
/// an imprecise pick can cost optimality but never correctness. Fixed intervals are skipped.
fn placedRegOf(intervals: []const Interval, children: []const *Interval, v: Value) ?u16 {
    var best: ?u16 = null;
    var best_end: u32 = 0;
    for (intervals) |*it| {
        if (it.fixed_reg != null) continue;
        if (it.value != v) continue;
        const loc = it.location orelse continue;
        switch (loc) {
            .reg => |r| if (best == null or it.end() >= best_end) {
                best = r;
                best_end = it.end();
            },
            .slot => {},
        }
    }
    for (children) |it| {
        if (it.fixed_reg != null) continue;
        if (it.value != v) continue;
        const loc = it.location orelse continue;
        switch (loc) {
            .reg => |r| if (best == null or it.end() >= best_end) {
                best = r;
                best_end = it.end();
            },
            .slot => {},
        }
    }
    return best;
}

/// Maps each block-parameter value to the incoming ARGUMENT values its predecessors pass for it, one
/// per in-edge. Built only when `coalesce_block_params` is on, and read to seed a parameter's
/// register hint from an argument that is already placed.
const ParamArgs = std.AutoHashMapUnmanaged(Value, std.ArrayListUnmanaged(Value));

/// Record, for edge `edge`, that each successor parameter receives the argument at the same index.
/// An arity mismatch is an IR invariant break, so this skips it defensively rather than pairing a
/// parameter with the wrong argument.
fn addParamArgEdge(allocator: std.mem.Allocator, func: *const Function, map: *ParamArgs, edge: Jump) Error!void {
    const params = func.blockParams(edge.target);
    const args = func.blockArgs(edge);
    if (params.len != args.len) return;
    for (params, args) |p, a| {
        const gop = try map.getOrPut(allocator, p);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(allocator, a);
    }
}

/// Build the block-parameter to incoming-argument map by walking every edge (both `if` arms and each
/// `jump`). The caller owns the result and frees it with `freeParamArgs`.
fn buildParamArgs(allocator: std.mem.Allocator, func: *const Function) Error!ParamArgs {
    var map: ParamArgs = .empty;
    errdefer freeParamArgs(allocator, &map);
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                try addParamArgEdge(allocator, func, &map, cf.then);
                try addParamArgEdge(allocator, func, &map, cf.@"else");
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .jump => |j| try addParamArgEdge(allocator, func, &map, j),
            .ret => {},
        };
    }
    return map;
}

/// Free the per-parameter argument lists and the map itself.
fn freeParamArgs(allocator: std.mem.Allocator, map: *ParamArgs) void {
    var it = map.valueIterator();
    while (it.next()) |list| list.deinit(allocator);
    map.deinit(allocator);
}

/// The register to hint block parameter `current` toward: the register of the FIRST of its incoming
/// arguments that is already placed in a register. The scan processes intervals by ascending start,
/// so an argument from an earlier predecessor is already placed and yields a hint, while a not-yet-
/// placed argument (for example one on a loop back-edge) is simply skipped. A parameter and its
/// argument never interfere, so this is a pure preference. Null when `current` is not a parameter, or
/// no incoming argument is placed yet.
fn computeParamHint(map: *const ParamArgs, intervals: []const Interval, children: []const *Interval, current: *const Interval) ?u16 {
    const v = current.value orelse return null;
    const list = map.getPtr(v) orelse return null;
    for (list.items) |arg| {
        if (placedRegOf(intervals, children, arg)) |r| return r;
    }
    return null;
}

// ===========================================================================
// SPILL-SLOT COALESCING. The scan hands every spilled interval its own fresh
// slot, so a spilled block parameter and the spilled argument that feeds it
// land on DIFFERENT slots. The edge move for the pair is then a slot-to-slot
// copy, `ldr scratch,[arg_slot]; str scratch,[param_slot]`, two memory ops per
// phi per in-edge. In a high-pressure function with many spilled loop-carried
// phis this dominates the memory traffic. Giving the parameter and its argument
// ONE slot makes the edge move same-slot, and the edge resolver drops a
// same-location move, so the copy vanishes. This mirrors what a graph-coloring
// allocator gets from phi coalescing, extended onto the stack.
//
// Two soundness properties are enforced:
//   1. INTERFERENCE: two values that are simultaneously live never share a
//      slot. A union is rejected unless every interval on one slot is disjoint
//      from every interval on the other. A loop phi and its own back-edge
//      argument, which overlap in the loop body, are correctly NOT coalesced.
//   2. PARALLEL-MOVE PRECONDITION: `orderClassMoves` relies on no spill slot
//      being both a source and a destination on one edge (its cycle-break
//      routes only registers through the scratch). After coalescing this is
//      re-verified over every edge's full move set; if any edge would break it,
//      the WHOLE coalescing is dropped and the byte-identical distinct-slot
//      placement stands.
// Both checks run under the same numbering the scan and resolver use, so the
// coalesced placement is exactly what `buildAllocation` and `resolveDataFlow`
// then consume, with the redundant same-slot moves naturally absent.
// ===========================================================================

/// Union-find representative of `i` with path halving, over the flat per-class slot space.
fn ufFind(parent: []u32, i: u32) u32 {
    var x = i;
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

/// The location value `v` occupies at position `pos`, read from its intervals the way `locationAt`
/// reads a segment list: the location of the interval with the greatest `start()` at or before `pos`.
/// Null when `v` has no interval starting at or before `pos`. `all` holds every value interval
/// (originals plus split children), each with a filled `location`.
fn valueLocAt(all: []const *Interval, v: Value, pos: u32) ?Location {
    var best: ?*const Interval = null;
    for (all) |iv| {
        if (iv.value.? != v) continue;
        if (iv.start() > pos) continue;
        if (best == null or iv.start() > best.?.start()) best = iv;
    }
    if (best) |b| return b.location.?;
    return null;
}

/// True iff the two spill-slot groups (all class-`class` intervals whose slot's representative is
/// `rep_a`, versus `rep_b`) contain a pair of intervals whose live ranges overlap. Such a pair
/// cannot share a slot: they would hold two different live values at once. Reads the pre-rewrite
/// slot numbers still stored in each interval's `location`.
fn slotGroupsInterfere(all: []const *Interval, parent: []u32, class_off: []const u32, class: u16, rep_a: u32, rep_b: u32) bool {
    for (all) |ia| {
        if (ia.class != class) continue;
        const sa = switch (ia.location.?) {
            .slot => |s| s,
            .reg => continue,
        };
        if (ufFind(parent, class_off[class] + sa) != rep_a) continue;
        for (all) |ib| {
            if (ib.class != class) continue;
            const sb = switch (ib.location.?) {
                .slot => |s| s,
                .reg => continue,
            };
            if (ufFind(parent, class_off[class] + sb) != rep_b) continue;
            if (ia.nextIntersection(ib) != null) return true;
        }
    }
    return false;
}

/// Attempt to coalesce, for one edge `pred -> edge.target`, each spilled successor parameter with the
/// spilled argument the predecessor passes for it. `pt` is the predecessor's branch position and `ss`
/// the successor's parameter row, so a location read at `pt` is the argument's placement leaving
/// `pred` and at `ss` the parameter's placement entering the successor. A union is taken only when the
/// two slots are distinct and their groups do not interfere.
fn coalesceEdgeParams(all: []const *Interval, parent: []u32, class_off: []const u32, func: *const Function, desc: *const RegDescription, block_from: []const u32, block_to: []const u32, pred: Block, edge: Jump) void {
    const succ = edge.target;
    const pt = block_to[@intFromEnum(pred)] - 1;
    const ss = block_from[@intFromEnum(succ)];
    const params = func.blockParams(succ);
    const args = func.blockArgs(edge);
    if (params.len != args.len) return;
    for (params, args) |p, a| {
        const p_loc = valueLocAt(all, p, ss) orelse continue;
        const a_loc = valueLocAt(all, a, pt) orelse continue;
        const p_slot = switch (p_loc) {
            .slot => |s| s,
            .reg => continue,
        };
        const a_slot = switch (a_loc) {
            .slot => |s| s,
            .reg => continue,
        };
        const c = desc.classOf(desc.ctx, func, p);
        // A parameter and its argument carry the same IR type, so the same class. A mismatch would be
        // an invariant break; skip it rather than union across classes.
        if (desc.classOf(desc.ctx, func, a) != c) continue;
        const rep_p = ufFind(parent, class_off[c] + p_slot);
        const rep_a = ufFind(parent, class_off[c] + a_slot);
        if (rep_p == rep_a) continue;
        if (slotGroupsInterfere(all, parent, class_off, c, rep_p, rep_a)) continue;
        parent[rep_p] = rep_a;
    }
}

/// Accumulate, into `mark` (bit0 = used as a move SOURCE, bit1 = used as a move DESTINATION), every
/// spill slot that one edge's move set reads or writes, and report whether any slot ends up BOTH. That
/// is exactly the condition `orderClassMoves` forbids. Covers both parameter moves and the live-in
/// (through) values whose location changes across the edge, matching `addEdgeMoves`. Reads the
/// post-rewrite slot numbers now in each interval's `location`.
fn edgeSlotBothSrcAndDst(all: []const *Interval, intervals: []const Interval, children: []const *Interval, func: *const Function, desc: *const RegDescription, class_off: []const u32, mark: []u8, block_from: []const u32, block_to: []const u32, pred: Block, edge: Jump) bool {
    const succ = edge.target;
    const pt = block_to[@intFromEnum(pred)] - 1;
    const ss = block_from[@intFromEnum(succ)];

    const Marker = struct {
        fn go(m: []u8, coff: []const u32, cls: u16, from: Location, to: Location) bool {
            if (locEql(from, to)) return false;
            switch (from) {
                .slot => |s| m[coff[cls] + s] |= 1,
                .reg => {},
            }
            switch (to) {
                .slot => |s| m[coff[cls] + s] |= 2,
                .reg => {},
            }
            return false;
        }
    };

    // (1) Parameter moves.
    const params = func.blockParams(succ);
    const args = func.blockArgs(edge);
    if (params.len == args.len) {
        for (params, args) |p, a| {
            const from = valueLocAt(all, a, pt) orelse continue;
            const to = valueLocAt(all, p, ss) orelse continue;
            _ = Marker.go(mark, class_off, desc.classOf(desc.ctx, func, p), from, to);
        }
    }
    // (2) Live-through moves: a value live-in to `succ`, not a parameter, whose location changes.
    for (all) |iv| {
        const v = iv.value.?;
        if (isParamOf(func, succ, v)) continue;
        if (!valueLiveAt(intervals, children, v, ss)) continue;
        const from = valueLocAt(all, v, pt) orelse continue;
        const to = valueLocAt(all, v, ss) orelse continue;
        _ = Marker.go(mark, class_off, desc.classOf(desc.ctx, func, v), from, to);
    }

    for (mark) |x| {
        if (x == 3) return true;
    }
    return false;
}

/// Coalesce spilled block parameters with their spilled incoming arguments onto shared slots, so the
/// feeding edge moves become same-slot no-ops the resolver drops. Interference-checked (property 1)
/// and guarded by the parallel-move precondition (property 2); an unsafe result reverts to the
/// distinct-slot placement, keeping the allocation byte-identical to the non-coalescing path. A no-op
/// unless the backend set `RegDescription.coalesce_spill_slots`. Rewrites the `location` of the
/// spilled intervals in place and compacts `slots` to the reduced per-class count.
///
/// Register WIDTH never enters this code, and that is correct rather than an oversight: a spilled
/// value takes exactly ONE slot whatever its register width, because a slot is `RegClass.slot_bytes`
/// wide and the backend sizes that for its widest value (see `RegWidth`). Sharing is decided by live
/// ranges alone, so a two-register value coalesces with a non-interfering partner exactly as a
/// one-register value does, and the slot renumbering stays a plain dense remap.
fn coalesceSpillSlots(allocator: std.mem.Allocator, func: *const Function, intervals: []Interval, children: []const *Interval, slots: []u32, desc: *const RegDescription) Error!void {
    if (!desc.coalesce_spill_slots) return;
    const nclasses: u16 = @intCast(desc.classes.len);

    var total: u32 = 0;
    for (slots) |s| total += s;
    if (total == 0) return; // nothing spilled

    // Flatten every value interval (originals + split children); each has a filled location.
    var all_list: std.ArrayList(*Interval) = .empty;
    defer all_list.deinit(allocator);
    for (intervals) |*iv| {
        if (iv.fixed_reg != null) continue;
        if (iv.value == null) continue;
        try all_list.append(allocator, iv);
    }
    for (children) |iv| try all_list.append(allocator, iv);
    const all = all_list.items;

    // Per-class slot offsets and a union-find over the flat slot space (unions stay within a class).
    const class_off = try allocator.alloc(u32, nclasses + 1);
    defer allocator.free(class_off);
    class_off[0] = 0;
    for (0..nclasses) |c| class_off[c + 1] = class_off[c] + slots[c];
    const parent = try allocator.alloc(u32, total);
    defer allocator.free(parent);
    for (0..total) |i| parent[i] = @intCast(i);

    const bounds = try computeBlockBounds(allocator, func);
    defer allocator.free(bounds.from);
    defer allocator.free(bounds.to);

    // PASS 0: union all spilled pieces of the SAME value onto one slot. The splitter hands each split
    // child its own fresh slot, so a value split under pressure and spilled on both sides of a split
    // lands on two slots and pays a slot-to-slot move where its location "changes" across an edge, even
    // though it is one value. Its pieces have DISJOINT lifetimes (a split partitions a lifetime), so
    // they never hold two live values at once; coalescing them is always interference-safe and removes
    // that self-move. The interference check still runs as a guard.
    var value_slot: std.AutoHashMapUnmanaged(Value, u32) = .empty;
    defer value_slot.deinit(allocator);
    for (all) |ia| {
        const sa = switch (ia.location.?) {
            .slot => |s| s,
            .reg => continue,
        };
        const c = ia.class;
        const gop = try value_slot.getOrPut(allocator, ia.value.?);
        if (!gop.found_existing) {
            gop.value_ptr.* = sa;
            continue;
        }
        const rep_a = ufFind(parent, class_off[c] + sa);
        const rep_b = ufFind(parent, class_off[c] + gop.value_ptr.*);
        if (rep_a == rep_b) continue;
        if (slotGroupsInterfere(all, parent, class_off, c, rep_a, rep_b)) continue;
        parent[rep_a] = rep_b;
    }

    // PASS 1: union each spilled parameter with a non-interfering spilled incoming argument.
    for (0..func.blockCount()) |bi| {
        const pred: Block = @enumFromInt(bi);
        for (func.blockInsts(pred)) |inst| {
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                coalesceEdgeParams(all, parent, class_off, func, desc, bounds.from, bounds.to, pred, cf.then);
                coalesceEdgeParams(all, parent, class_off, func, desc, bounds.from, bounds.to, pred, cf.@"else");
            }
        }
        if (func.terminator(pred)) |term| switch (term) {
            .jump => |j| coalesceEdgeParams(all, parent, class_off, func, desc, bounds.from, bounds.to, pred, j),
            .ret => {},
        };
    }

    // Compact each class's live slots to a dense 0.. numbering (a rep gets the next free index).
    const remap = try allocator.alloc(u32, total);
    defer allocator.free(remap);
    @memset(remap, std.math.maxInt(u32));
    const new_counts = try allocator.alloc(u32, nclasses);
    defer allocator.free(new_counts);
    for (0..nclasses) |c| {
        var next: u32 = 0;
        for (0..slots[c]) |s| {
            const flat = class_off[c] + @as(u32, @intCast(s));
            const r = ufFind(parent, flat);
            if (remap[r] == std.math.maxInt(u32)) {
                remap[r] = next;
                next += 1;
            }
            remap[flat] = remap[r];
        }
        new_counts[c] = next;
    }

    // Snapshot the original locations so an unsafe result can revert bit-for-bit.
    const orig_loc = try allocator.alloc(Location, all.len);
    defer allocator.free(orig_loc);
    for (all, 0..) |iv, i| orig_loc[i] = iv.location.?;

    // Apply the remap to every spilled interval.
    for (all) |iv| {
        switch (iv.location.?) {
            .slot => |s| iv.location = .{ .slot = remap[class_off[iv.class] + s] },
            .reg => {},
        }
    }

    // PASS 2: re-verify the parallel-move precondition over every edge, on the rewritten locations.
    // When the backend gives EVERY class a second scratch, the resolver breaks slot-involving cycles
    // through it (`orderClassMoves`), so a slot being both a source and a destination on an edge is
    // fully handled and this guard is not needed; the coalescing always commits. Without full scratch2
    // (x86-32), keep the conservative all-or-nothing revert so the single-scratch resolver never faces
    // a slot cycle it cannot break.
    const full_scratch2 = desc.scratch2.len == nclasses;
    if (!full_scratch2) {
        const mark = try allocator.alloc(u8, total);
        defer allocator.free(mark);
        var unsafe = false;
        walk: for (0..func.blockCount()) |bi| {
            const pred: Block = @enumFromInt(bi);
            for (func.blockInsts(pred)) |inst| {
                if (func.opcode(inst) == .@"if") {
                    const cf = func.opcode(inst).@"if";
                    for ([_]Jump{ cf.then, cf.@"else" }) |e| {
                        @memset(mark, 0);
                        if (edgeSlotBothSrcAndDst(all, intervals, children, func, desc, class_off, mark, bounds.from, bounds.to, pred, e)) {
                            unsafe = true;
                            break :walk;
                        }
                    }
                }
            }
            if (func.terminator(pred)) |term| switch (term) {
                .jump => |j| {
                    @memset(mark, 0);
                    if (edgeSlotBothSrcAndDst(all, intervals, children, func, desc, class_off, mark, bounds.from, bounds.to, pred, j)) {
                        unsafe = true;
                        break :walk;
                    }
                },
                .ret => {},
            };
        }

        if (unsafe) {
            // Revert to the distinct-slot placement; leave `slots` untouched.
            for (all, 0..) |iv, i| iv.location = orig_loc[i];
            return;
        }
    }

    // Commit the reduced per-class slot counts.
    for (0..nclasses) |c| slots[c] = new_counts[c];
}

/// Allocate registers for `func` using the shared linear scan with live-range splitting. It
/// builds intervals, then seeds the call-clobber fixed intervals into `inactive` so they are
/// visible before their first range. This is the fixed-interval visibility rule the scan depends
/// on. It runs LINEARSCAN with `allocateBlockedReg` (Fig 7) and `splitInterval` under register
/// pressure, then collapses each value's placement, possibly across multiple intervals, into a
/// multi-segment `Allocation`. The caller owns the result and releases it with
/// `Allocation.deinit`.
pub fn allocate(allocator: std.mem.Allocator, func: *const Function, desc: *const RegDescription) AllocateError!Allocation {
    const intervals = try buildIntervals(allocator, func, desc);
    defer freeIntervals(allocator, intervals);

    // A move the resolver routes through a class scratch carries a WHOLE value, so a class that
    // holds a multi-register value needs an aligned, fully reserved scratch span of that width.
    // Checking it here, against the widths this function actually built, beats a comment that says
    // the backend must remember. It costs nothing for a backend with no `regWidth` hook.
    if (std.debug.runtime_safety) assertScratchFitsWidth(intervals, desc);

    // Split children are heap-allocated intervals born during the scan. They are tracked here, so
    // their owned `ranges`, `uses`, and the interval box itself are freed even on an error path.
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

    // The worklist (`unhandled`) is a priority queue. It holds value intervals sorted ascending
    // by start, popped from the front, with split children re-inserted in sorted position. Fixed
    // intervals never enter it. A CALL-CLOBBER fixed interval (`value == null`) is seeded into
    // `inactive`, since its first range is at a call, a hole at position 0. This lets
    // `tryAllocateFreeReg` and `allocateBlockedReg` see the clobber at every EARLIER position. An
    // entry-param fixed interval is seeded into `active`: it lives at position 0, and it must
    // block its ABI register from the first value pop. In the worklist it popped LAST at position
    // 0 (buildIntervals appends fixed intervals after every value interval), so a value whose own
    // hint was clamped, by a div or shift clobber for example, could take another parameter's pin
    // before the pin was ever seen. The verifier caught exactly that overlap.
    var unhandled: std.ArrayList(*Interval) = .empty;
    defer unhandled.deinit(allocator);
    var active: std.ArrayList(*Interval) = .empty;
    defer active.deinit(allocator);
    var inactive: std.ArrayList(*Interval) = .empty;
    defer inactive.deinit(allocator);

    for (intervals) |*iv| {
        if (iv.fixed_reg != null) {
            iv.location = .{ .reg = iv.fixed_reg.? };
            if (iv.value == null) {
                try inactive.append(allocator, iv);
            } else {
                try active.append(allocator, iv);
            }
        } else {
            try unhandled.append(allocator, iv);
        }
    }
    std.mem.sort(*Interval, unhandled.items, {}, intervalStartLessThan);

    // Block-argument coalescing map. Empty (and unused) unless the backend opted in. It hints a
    // parameter toward an incoming argument's register so the edge move becomes a no-op.
    var param_args: ParamArgs = if (desc.coalesce_block_params) try buildParamArgs(allocator, func) else .empty;
    defer freeParamArgs(allocator, &param_args);

    // The scan is a worklist loop. Every split child starts strictly after the interval it came
    // from, so the total interval count is bounded by (values x positions). The guard asserts that
    // bound, to catch a splitting bug that would otherwise loop forever.
    const max_pos = maxEndPosition(intervals);
    const iter_bound: usize = intervals.len + intervals.len * (@as(usize, max_pos) + 1);
    var iters: usize = 0;
    while (unhandled.items.len > 0) {
        iters += 1;
        std.debug.assert(iters <= iter_bound);
        const current = unhandled.orderedRemove(0);
        const position = current.start();

        // Expire or deactivate active intervals. Ranges are half-open, so `end() <= position` means
        // the interval's last live position is behind us, and it is handled. A live interval that
        // does not cover `position` sits in a hole, so it becomes inactive. `swapRemove` reorders
        // the list, which is fine here.
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

        // Every fixed interval was seeded at the start: call clobbers into `inactive`,
        // entry-param pins into `active`. The worklist holds value intervals only.
        std.debug.assert(current.fixed_reg == null);

        // A value interval: take a free register covering its whole lifetime, or fall back to the
        // blocked-register path that splits and spills to make room. A block-parameter hint (from an
        // already-placed incoming argument) biases the free-register pick toward eliding the edge move.
        const param_hint: ?u16 = if (desc.coalesce_block_params)
            computeParamHint(&param_args, intervals, children.items, current)
        else
            null;
        if (tryAllocateFreeReg(current, active.items, inactive.items, desc, param_hint)) |reg| {
            current.location = .{ .reg = reg };
            try active.append(allocator, current);
        } else {
            try allocateBlockedReg(allocator, current, &active, &inactive, &unhandled, &children, slots, desc);
        }
    }

    // Coalesce spilled block parameters with their spilled incoming arguments onto shared slots, so
    // the feeding edge moves collapse to same-slot no-ops the resolver drops. A no-op unless the
    // backend opted in; interference-checked and guarded so an unsafe result reverts to the
    // distinct-slot placement. Runs before the verifier, so the verified intervals are the final ones.
    try coalesceSpillSlots(allocator, func, intervals, children.items, slots, desc);

    // In a test or debug build, verify the completed allocation before lowering it. A firing assert
    // here means the scan produced an UNSOUND allocation, a real bug, caught now instead of as a
    // downstream miscompile. This check is gated on `runtime_safety`, so the ReleaseFast production
    // JIT is not slowed. The verifier reasons over the final intervals, the originals plus the split
    // children, flattened into one read-only slice.
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
    // RESOLVEDATAFLOW: compute the control-flow-edge moves while the intervals, needed for
    // liveness, are still alive. Order each edge's moves as a valid parallel move.
    try resolveDataFlow(allocator, func, desc, intervals, children.items, &result);
    return result;
}

/// Assert the reserved scratch registers of every class can hold that class's WIDEST value. The
/// resolver routes a slot-to-slot move, and a broken move cycle, through the class scratch, and that
/// transfer carries a whole value. So a class whose widest value takes N registers needs the N
/// registers from the scratch base reserved, and the base must meet the alignment that width demands.
/// A backend that reserved only the base would silently destroy the register beside its scratch.
/// A backend with no `regWidth` hook cannot report a width above one, so this returns at once and
/// costs such a backend nothing, even in a safety build.
fn assertScratchFitsWidth(intervals: []const Interval, desc: *const RegDescription) void {
    if (desc.regWidth == null) return;
    for (0..desc.classes.len) |ci| {
        var regs: u16 = 1;
        var alignment: u16 = 1;
        for (intervals) |*iv| {
            if (iv.class != ci) continue;
            if (iv.regs > regs) regs = iv.regs;
            if (iv.reg_align > alignment) alignment = iv.reg_align;
        }
        if (regs == 1 and alignment == 1) continue;
        const bases = [_]?u16{
            if (ci < desc.scratch.len) desc.scratch[ci] else null,
            if (ci < desc.scratch2.len) desc.scratch2[ci] else null,
        };
        for (bases) |maybe_base| {
            const base = maybe_base orelse continue;
            std.debug.assert(base % alignment == 0);
            // Every register the scratch span covers must be outside the allocatable pool, so no
            // value is ever placed where a routed move will overwrite it.
            var k: u16 = 1;
            while (k < regs) : (k += 1) {
                std.debug.assert(!containsReg(desc.classes[ci].allocatable, base + k));
            }
        }
    }
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

/// SPLITINTERVAL (Wimmer & Franz): split `parent` at `pos` into a head the parent keeps, with
/// ranges and uses at positions `< pos`, and a freshly allocated CHILD carrying everything at
/// positions `>= pos`. A range straddling `pos` is cut into `[from, pos)` for the head and
/// `[pos, to)` for the child. Both reference the same `value`. The child's `location` is unset,
/// since the caller re-inserts it into the worklist or assigns it a slot. The child is appended
/// to `children`, so it is freed with the rest. It is a programmer error unless
/// `parent.start() < pos < parent.end()`, so both halves stay non-empty.
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
        .narrow = parent.narrow,
        // A split cuts a LIFETIME, never a value's shape, so the child holds the same number of
        // registers under the same alignment as its parent.
        .regs = parent.regs,
        .reg_align = parent.reg_align,
    };
    try children.append(allocator, child);

    // Commit: the child now owns the tail. Replace the parent's ranges and uses with the head.
    allocator.free(parent.ranges);
    parent.ranges = hr;
    allocator.free(parent.uses);
    parent.uses = hu;
    return child;
}

/// Spill `current`. It is the cheapest interval to move to memory: its own first use is further
/// off than any evictable register's next use, or no register can be evicted here at all. A
/// `must_have_register` use forces the spilled part to end at that use, so the use lands back in
/// a register. The head, which holds no must_have use, is safe in memory, and the register-needing
/// tail is re-queued. Otherwise the whole interval goes to a slot. Both spill-current paths share
/// this function. It returns `error.Unsupported` when a same-position must-have demand exceeds
/// the register pool, the unsatisfiable case the old allocator also rejects as "too many live
/// params".
///
/// A MULTI-REGISTER value spills the same way a one-register value does, into ONE slot. The slot is
/// `RegClass.slot_bytes` wide and the backend must size that for the class's widest value, so the
/// store and the reload the resolver emits carry the whole span. `Move.value` names the IR value, so
/// a width-aware backend picks the wide store and the wide load from it, the same mechanism aarch64
/// uses for a 128-bit vector in a 16-byte slot. Nothing here counts registers.
fn spillCurrent(
    allocator: std.mem.Allocator,
    current: *Interval,
    unhandled: *std.ArrayList(*Interval),
    children: *std.ArrayList(*Interval),
    slots: []u32,
    class_idx: u16,
) AllocateError!void {
    if (firstMustHaveUse(current)) |u| {
        // A must_have use AT `current.start()` means `current` needs a register the instant it
        // becomes live. But it is the interval being spilled BECAUSE nothing here can be freed for
        // it. A same-position group, for example a loop header with more live params than the
        // class has registers, can reach this exact split child through repeated re-eviction,
        // where the only remaining must-have use coincides with the child's own start. That is
        // more simultaneous must-have demand than the class can ever satisfy. It is the SAME "too
        // many live params" limit the old allocator (aarch64 isel.zig `allocate`) rejects for this
        // shape. So this bails to match, rather than asserting a programmer error.
        if (u <= current.start()) return error.Unsupported;
        const tail = try splitInterval(allocator, current, u, children);
        current.location = .{ .slot = slots[class_idx] };
        slots[class_idx] += 1;
        try insertSorted(allocator, unhandled, tail);
    } else {
        current.location = .{ .slot = slots[class_idx] };
        slots[class_idx] += 1;
    }
}

/// ALLOCATEBLOCKEDREG (Wimmer & Franz Fig 7). `current` could not get a free register, so either
/// it, or an interval occupying a register, must be split. Compute `nextUsePos[r]` for every
/// allocatable register `r` of `current`'s class: the earliest future use of whatever holds `r`,
/// or a hard block where a fixed interval clobbers `r`. Pick the register whose next use is
/// FURTHEST away. Then spill `current` if its own first use is even further off. Otherwise take
/// the register and split the intervals it displaces. A fixed clobber of the chosen register
/// before `current` ends also splits `current` before the clobber. This honors
/// `must_have_register`: a spilled head is cut before the first must_have use, so that use lands
/// back in a register. It propagates `error.Unsupported` from `spillCurrent` when a same-position
/// must-have demand exceeds the register pool.
fn allocateBlockedReg(
    allocator: std.mem.Allocator,
    current: *Interval,
    active: *std.ArrayList(*Interval),
    inactive: *std.ArrayList(*Interval),
    unhandled: *std.ArrayList(*Interval),
    children: *std.ArrayList(*Interval),
    slots: []u32,
    desc: *const RegDescription,
) AllocateError!void {
    const class_idx = current.class;
    const class = desc.classes[class_idx];
    const p = current.start();

    var next_use = [_]u32{infinity} ** max_phys_regs;
    var block_pos = [_]u32{infinity} ** max_phys_regs;
    var is_candidate = [_]bool{false} ** max_phys_regs;
    // A register is EVICTABLE only if the active value interval occupying it can be legally split
    // at `p`. That is, that interval starts strictly before `p`. An occupant starting AT `p`, a
    // same-start same-class value, for example one of a block's params when there are more of
    // them than the pool, cannot be split there, since `splitInterval` requires `pos > start`. So
    // its register is not evictable here.
    var evictable = [_]bool{true} ** max_phys_regs;
    for (class.allocatable) |r| {
        std.debug.assert(r < max_phys_regs);
        is_candidate[r] = true;
    }

    // An active VALUE interval occupies its register from here. The register is wanted again at
    // that interval's next use (`>= p`), or is spillable cheaply until its end if it has no
    // further use.
    for (active.items) |it| {
        if (it.class != class_idx) continue;
        if (it.fixed_reg != null) continue;
        const r = assignedReg(it);
        if (r >= max_phys_regs) continue;
        const u = it.firstUseAfter(p) orelse it.end();
        // The occupant holds its WHOLE span, so every register of it is wanted again at that use,
        // and every register of it is unevictable when the occupant starts here. Recording the base
        // only would let the scan steal the second half of a two-register value.
        var k: u16 = 0;
        while (k < it.regs) : (k += 1) {
            const rr = r +| k;
            if (rr >= max_phys_regs or !is_candidate[rr]) continue;
            if (u < next_use[rr]) next_use[rr] = u;
            if (it.start() == p) evictable[rr] = false;
        }
    }
    // An inactive VALUE interval only reclaims its register where it next intersects `current`.
    // Its next use bounds how soon that register is genuinely wanted.
    for (inactive.items) |it| {
        if (it.class != class_idx) continue;
        if (it.fixed_reg != null) continue;
        const r = assignedReg(it);
        if (r >= max_phys_regs) continue;
        if (current.nextIntersection(it) == null) continue;
        const u = it.firstUseAfter(p) orelse it.end();
        var k: u16 = 0;
        while (k < it.regs) : (k += 1) {
            const rr = r +| k;
            if (rr >= max_phys_regs or !is_candidate[rr]) continue;
            if (u < next_use[rr]) next_use[rr] = u;
        }
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
            if (r >= max_phys_regs) continue;
            const x = if (it.value == null) fixedClobberConflict(current, it) else current.nextIntersection(it);
            if (x) |xx| {
                // A call clobber is one register wide, but an entry pin covers its parameter's whole
                // span, so block every register the fixed interval owns.
                var k: u16 = 0;
                while (k < it.regs) : (k += 1) {
                    const rr = r +| k;
                    if (rr >= max_phys_regs or !is_candidate[rr]) continue;
                    if (xx < next_use[rr]) next_use[rr] = xx;
                    if (xx < block_pos[rr]) block_pos[rr] = xx;
                }
            }
        }
    }

    // Pick the EVICTABLE register whose next use is furthest away (the least costly to steal). A
    // register whose active occupant starts at `p` is skipped: it cannot be split at `p` to make room.
    // The candidate is a BASE register whose whole span is placeable and evictable, and the span is
    // wanted back as soon as its earliest-wanted register. For the one-register default this is the
    // old per-register loop in the same order, so the pick is byte-identical.
    var chosen: ?u16 = null;
    var best: u32 = 0;
    for (0..max_phys_regs) |b| {
        if (!spanPlaceable(&is_candidate, &evictable, b, current.regs, current.reg_align)) continue;
        const nu = spanMin(&next_use, b, current.regs);
        if (chosen == null or nu > best) {
            best = nu;
            chosen = @intCast(b);
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
    // `best` is the chosen span's next-use position, the value `next_use[reg]` held before spans
    // existed.
    const current_first_use = current.firstUseAfter(p);
    if (current_first_use == null or current_first_use.? > best) {
        try spillCurrent(allocator, current, unhandled, children, slots, class_idx);
        return;
    }

    // Otherwise take the span based at `reg` for `current` and split whatever occupies any register
    // of it.
    current.location = .{ .reg = reg };

    // Every active value interval whose span OVERLAPS the taken span is split at `p`: its head keeps
    // its registers up to here (and expires next step), its tail is re-allocated elsewhere. Register
    // exclusivity means a ONE-register span has at most one such occupant, so this walks the same
    // single interval the earlier base-equality test found. A WIDER span can displace more than one
    // occupant, so the loop no longer stops at the first.
    for (active.items) |it| {
        if (it.class != class_idx) continue;
        if (it.fixed_reg != null) continue;
        if (!spansOverlap(assignedReg(it), it.regs, reg, current.regs)) continue;
        std.debug.assert(it.start() < p);
        const tail = try splitInterval(allocator, it, p, children);
        try insertSorted(allocator, unhandled, tail);
    }
    // Each inactive value interval overlapping the taken span that would reclaim it inside
    // `current`'s life is split at that intersection. Its tail is re-allocated elsewhere.
    for (inactive.items) |it| {
        if (it.class != class_idx) continue;
        if (it.fixed_reg != null) continue;
        if (!spansOverlap(assignedReg(it), it.regs, reg, current.regs)) continue;
        const x = current.nextIntersection(it) orelse continue;
        std.debug.assert(x > it.start() and x < it.end());
        const tail = try splitInterval(allocator, it, x, children);
        try insertSorted(allocator, unhandled, tail);
    }
    // A fixed interval clobbers some register of the taken span before `current` ends: `current`
    // cannot hold the span across the clobber, so split it before the block and re-allocate the far
    // side.
    const span_block_pos = spanMin(&block_pos, reg, current.regs);
    if (span_block_pos < current.end()) {
        const bp = span_block_pos;
        // `bp <= p` means the clobber falls AT `current`'s very start, which is not a legal split
        // point, since `splitInterval` needs pos > start. This arises when `current` is a value used
        // AT `p` and ALSO live past a clobber it cannot escape, for example an xmm value that is a
        // call ARGUMENT and live across the call, in a class with no callee-saved register. It
        // cannot bridge the clobber in `reg`, so it must live in a slot across it. Spill `current`
        // instead of splitting. The occupants evicted above are simply re-allocated, which is
        // harmless. `spillCurrent` still bails `error.Unsupported` if `current` has a MUST_HAVE use
        // at its start, a demand no slot can satisfy. A target whose uses may read from a slot
        // (`should_have_register`) spills cleanly.
        if (bp <= p) {
            try spillCurrent(allocator, current, unhandled, children, slots, class_idx);
            return;
        }
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

/// Order the intra-block actions so every same-position cluster drains hazard-free. The input is
/// the actions already sorted ascending by `at`. For each maximal same-`at` run, this treats the
/// cluster as a parallel move, where each action is one `(src -> dst)` transfer at that position,
/// and runs it through `orderMoves`, the SAME routine and scratch cycle-break the control-flow
/// edges use. It then re-tags the ordered primitive moves as actions at that position. The result
/// is still ascending by `at`, and within a cluster every source is read before it is overwritten.
/// The caller owns the returned slice. Reusing the edge resolver is what lets the aarch64 bridge
/// drop its ad-hoc same-position hazard detector: the ordering makes the fixed drain order always
/// safe.
///
/// The parallel-move invariants `orderClassMoves` relies on hold for an intra-block cluster
/// exactly as for an edge. Every spill slot names a distinct interval, so within one position a
/// slot is never both a source and a destination. A store writes a fresh slot no other action
/// reads, and a reload reads a slot no other action writes. So every register cycle is reg-to-reg
/// and every slot-to-slot move is independent, and the scratch routing is never nested inside a
/// held cycle.
fn orderIntraActions(allocator: std.mem.Allocator, sorted: []const Action, desc: *const RegDescription) Error![]Action {
    var out: std.ArrayList(Action) = .empty;
    errdefer out.deinit(allocator);

    var moves: std.ArrayList(Move) = .empty;
    defer moves.deinit(allocator);

    var i: usize = 0;
    while (i < sorted.len) {
        const at = sorted[i].at;
        var j = i;
        while (j < sorted.len and sorted[j].at == at) : (j += 1) {}

        moves.clearRetainingCapacity();
        for (sorted[i..j]) |a| {
            try moves.append(allocator, .{ .src = a.src, .dst = a.dst, .class = a.class, .value = a.value });
        }
        const ordered = try orderMoves(allocator, moves.items, desc);
        defer allocator.free(ordered);
        for (ordered) |m| {
            try out.append(allocator, .{
                .at = at,
                .kind = actionKind(m.src, m.dst),
                .class = m.class,
                .src = m.src,
                .dst = m.dst,
                .value = m.value,
            });
        }
        i = j;
    }
    return out.toOwnedSlice(allocator);
}

/// Invoke `f(iv)` for every placed VALUE interval, both the originals and the split children (fixed
/// intervals and unplaced intervals are skipped). Keeps the two storage lists in one walk.
fn forEachPlacedValue(originals: []const Interval, children: []const *Interval, ctx: anytype, comptime f: fn (@TypeOf(ctx), *const Interval) Error!void) Error!void {
    for (originals) |*iv| {
        if (iv.fixed_reg != null) continue;
        if (iv.value == null) continue;
        // The scan places every value interval it processes. A null here would silently drop a
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

    // Intra-block data moves (spill, reload, reg-move) accumulated across every value, sorted
    // ascending by `at` at the end. A cross-block transition is NOT an action here. Resolution
    // emits it as an edge move instead, so this only flips `needs_resolution`, so the backend
    // bails instead of miscompiling.
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
        // Every consecutive segment pair is a location change the emitter must realize. An
        // INTRA-block change, where both sides fall in one block, becomes an `Action` at the later
        // segment's `from`. A CROSS-block change is an edge move resolution owns, so it only
        // records `needs_resolution`.
        const class = list.items[0].class;
        var i: usize = 0;
        while (i + 1 < segs.items.len) : (i += 1) {
            const a = segs.items[i];
            const b = segs.items[i + 1];
            const at = b.from;
            // A transition is cross-block, resolved on the edge, ONLY when the later segment begins
            // EXACTLY on a block-entry position. Any other transition happens mid-block and is an
            // intra-block action, even when the earlier segment began in an earlier block. A value
            // held in a register across a block boundary and evicted mid-block must be stored HERE,
            // not on the edge, since the edge sees the same register on both sides, so it emits no
            // move. Classifying such a mid-block spill as cross-block drops the store, a silent
            // miscompile.
            const at_block = blockOfPos(bounds.from, bounds.to, at);
            const is_cross_block = at == bounds.from[at_block];
            if (is_cross_block) {
                result.needs_resolution = true;
                continue;
            }
            try actions.append(allocator, .{ .at = at, .kind = actionKind(a.loc, b.loc), .class = class, .src = a.loc, .dst = b.loc, .value = e.key_ptr.* });
        }

        const owned = try segs.toOwnedSlice(allocator);
        errdefer allocator.free(owned);
        try result.segments.put(allocator, e.key_ptr.*, owned);
    }

    // Actions land in ascending-`at` order for the emitter's single-cursor drain, and every
    // same-position cluster is ordered into a hazard-free parallel-move sequence. Draining the
    // result in order can never clobber a live value: a store's source is read before a reload
    // overwrites that register, a register cycle is broken through the class scratch, and a
    // slot-to-slot shuffle is expanded through it. This reuses `orderMoves`, the edge-move
    // ordering, not a second resolver.
    std.mem.sort(Action, actions.items, {}, actionAtLessThan);
    const raw_actions = try actions.toOwnedSlice(allocator);
    defer allocator.free(raw_actions);
    result.actions = try orderIntraActions(allocator, raw_actions, desc);

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
            // A multi-register value occupies its WHOLE span, so EVERY callee-saved register of the
            // span needs a prologue save. Recording the base only would leave the second half of a
            // pair unsaved, and the function would destroy its caller's copy.
            var k: u16 = 0;
            while (k < iv.regs) : (k += 1) {
                const rr = reg +| k;
                if (!containsReg(self.desc.classes[iv.class].callee_saved, rr)) continue;
                var seen = false;
                for (self.used.items) |u| {
                    if (u.class == iv.class and u.reg == rr) seen = true;
                }
                if (seen) continue;
                try self.used.append(self.allocator, .{ .class = iv.class, .reg = rr });
            }
        }
    };
    var used: std.ArrayList(UsedSaved) = .empty;
    errdefer used.deinit(allocator);
    try forEachPlacedValue(intervals, children, Saver{ .allocator = allocator, .used = &used, .desc = desc }, Saver.add);
    result.used_callee_saved = try used.toOwnedSlice(allocator);

    return result;
}

// ===========================================================================
// RESOLVEDATAFLOW (Wimmer & Franz Fig 8) plus the standard parallel-move
// ordering. After the scan, a value may live in DIFFERENT locations on the two
// sides of a control-flow edge. The splitter placed it in a register in one
// block and a slot in another, or a block parameter simply lands in a different
// register than the argument that feeds it. Each such difference becomes a MOVE
// on that edge. The raw move set of one edge may contain conflicts, where one
// move's destination is another's source, and cycles, a register swap. So each
// edge's moves are ordered into a valid sequence. Every source is read before
// it is overwritten, cycles are broken through the class scratch register, and
// a slot-to-slot shuffle is routed through the class scratch too, since no
// target moves memory to memory in one op. The backend emits the ordered list
// op by op with no further reordering.
//
// PRECONDITION: the function has NO critical edge. Resolution places moves at
// an edge, and a critical edge, a multi-successor `if` block feeding a
// multi-predecessor block, has no block that can host them without corrupting
// the sibling edge. The driver calls `splitCriticalEdges` before building the
// RegDescription, so numbering stays consistent. `assertNoCriticalEdges` fails
// loudly on a wiring mistake. `allocate` never splits edges itself, since that
// would invalidate the already-built positions.
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

/// True iff some VALUE interval of `v`, an original or a split child, never a fixed interval,
/// covers `pos`. A value's lifetime is the union of its intervals, so this is its true liveness
/// at `pos`.
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
        // An `if` instruction contributes its two edges, then and else. Each carries its own args.
        for (func.blockInsts(block)) |inst| {
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                try addEdgeMoves(allocator, func, desc, intervals, children, result, bounds.from, bounds.to, &edges, block, cf.then);
                try addEdgeMoves(allocator, func, desc, intervals, children, result, bounds.from, bounds.to, &edges, block, cf.@"else");
            }
        }
        // A `jump` terminator contributes one edge. A `ret` contributes none.
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
    // The predecessor's branch executes at its last position. The successor is entered at its
    // parameter row. A location looked up at `pt` is the value's placement as control leaves
    // `pred`, and at `ss` its placement as control enters `succ`.
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
        const s2: ?u16 = if (ci < desc.scratch2.len) desc.scratch2[ci] else null;
        try orderClassMoves(allocator, class_idx, raw, desc.scratch[ci], s2, &out);
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
/// Cycle handling depends on whether any spill slot is BOTH a source and a destination among this
/// class's moves. Without a slot conflict (the historical case: each slot names a unique interval, so
/// slots never both send and receive), every cycle is reg->reg and the single class `scratch` breaks
/// it, byte-identical to before. WITH a slot conflict (spill-slot coalescing merged a parameter and
/// its argument, so a value in a slot and the same slot receiving another value can sit in one cycle),
/// the cycle value is routed through the SECOND scratch `scratch2` instead, leaving `scratch` free to
/// realize any slot->slot memory move drained while the cycle is held. A slot conflict only reaches
/// here when the backend provided `scratch2` (coalescing commits such a placement only then), so the
/// `scratch2 == null` branch is unreachable and fails closed.
fn orderClassMoves(
    allocator: std.mem.Allocator,
    class_idx: u16,
    raw: []const Move,
    scratch_reg: u16,
    scratch2_reg: ?u16,
    out: *std.ArrayList(Move),
) Error!void {
    var pending: std.ArrayList(Move) = .empty;
    defer pending.deinit(allocator);
    for (raw) |m| {
        if (m.class == class_idx) try pending.append(allocator, m);
    }
    if (pending.items.len == 0) return;

    const scratch_loc: Location = .{ .reg = scratch_reg };
    const scratch2_loc: ?Location = if (scratch2_reg) |r| .{ .reg = r } else null;
    // Both scratch registers are reserved, so no value ever lives there: no raw move may touch either.
    for (pending.items) |m| {
        std.debug.assert(!locEql(m.src, scratch_loc));
        std.debug.assert(!locEql(m.dst, scratch_loc));
        if (scratch2_loc) |s2| {
            std.debug.assert(!locEql(m.src, s2));
            std.debug.assert(!locEql(m.dst, s2));
        }
    }

    // A slot both source and destination means a cycle can contain a slot; break such cycles through
    // scratch2. `cycle_loc` is where a broken cycle node's value is parked: scratch2 under a conflict,
    // else the ordinary scratch (byte-identical single-scratch path).
    // A slot is both a source and a destination only after spill-slot coalescing merged a slot, which
    // `coalesceSpillSlots` commits ONLY when every class has a scratch2. So a conflict here guarantees
    // scratch2 is present; the `orelse` is an unreachable invariant, not a runtime path.
    const has_slot_conflict = slotBothSrcAndDst(pending.items);
    const cycle_loc: Location = if (has_slot_conflict) (scratch2_loc orelse unreachable) else scratch_loc;

    const out_start = out.items.len;
    var scratch_busy = false; // scratch held: a slot->slot expansion, or a no-conflict cycle break
    var scratch2_busy = false; // scratch2 held: a slot-conflict cycle break
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
            try emitPrimitive(allocator, out, m, scratch_loc, scratch2_loc, class_idx, &scratch_busy, &scratch2_busy);
            continue;
        }

        // Only cycles remain. Break one by saving its source into the cycle scratch, then reading the
        // scratch in its place; the location it used to read is now free, unblocking the rest.
        const m0 = &pending.items[0];
        if (has_slot_conflict) {
            std.debug.assert(!scratch2_busy);
            scratch2_busy = true;
        } else {
            std.debug.assert(!scratch_busy);
            // Without a slot conflict every cycle node is reg->reg (the classic invariant).
            std.debug.assert(locIsReg(m0.src) and locIsReg(m0.dst));
            scratch_busy = true;
        }
        // The save routes m0's value through the cycle scratch, so it carries m0's value for the width.
        // m0.src may be a slot under a conflict, making this a load rather than a reg move.
        try out.append(allocator, .{ .src = m0.src, .dst = cycle_loc, .class = class_idx, .value = m0.value });
        m0.src = cycle_loc;
    }
    std.debug.assert(!scratch_busy and !scratch2_busy);

    if (std.debug.runtime_safety) {
        try assertOrderingValid(allocator, class_idx, raw, out.items[out_start..]);
    }
}

/// True iff some spill slot in `moves` is the source of one move and the destination of a DIFFERENT
/// move. That is the one situation the single-scratch cycle break cannot serve (a cycle may then
/// contain a slot both sent and received), so it selects the scratch2 routing. A self-move (a slot to
/// itself) is not a conflict: it moves nothing.
fn slotBothSrcAndDst(moves: []const Move) bool {
    for (moves, 0..) |a, i| {
        const s = switch (a.src) {
            .slot => |x| x,
            .reg => continue,
        };
        for (moves, 0..) |b, j| {
            if (i == j) continue;
            switch (b.dst) {
                .slot => |y| if (y == s and a.class == b.class) return true,
                .reg => {},
            }
        }
    }
    return false;
}

/// Emit one ordered move as backend-primitive op(s): a reg source (reg->reg move or reg->slot store)
/// and a slot->reg load pass through unchanged, while a slot->slot shuffle expands to a load into the
/// scratch then a store out of it. A move that READS either scratch closes a broken cycle, so that
/// scratch is free again after it. A move retargeted onto scratch2 reads a register, so it never
/// re-enters the slot->slot path and never needs `scratch` while scratch2 is held.
fn emitPrimitive(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(Move),
    m: Move,
    scratch_loc: Location,
    scratch2_loc: ?Location,
    class_idx: u16,
    scratch_busy: *bool,
    scratch2_busy: *bool,
) Error!void {
    const closes = locEql(m.src, scratch_loc);
    const closes2 = if (scratch2_loc) |s2| locEql(m.src, s2) else false;
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
    if (closes2) scratch2_busy.* = false;
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

/// Debug assertion that the function has NO critical edge: a `>1`-successor `if` block feeding a
/// `>1`-predecessor block. Resolution assumes the driver split them first. A violation is a
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
// The debug verifier (VERIFYINTERVALS).
//
// A white-box checker that validates a COMPLETED allocation and runs inside
// `allocate` whenever `std.debug.runtime_safety` is on, in every test and
// debug build. So an allocation bug fails loudly at an assert instead of
// silently miscompiling. It never runs under ReleaseFast, so the production
// JIT keeps its speed. It operates on the final intervals, the originals AND
// the split children flattened into one slice, and checks three soundness
// properties:
//
//   1. REGISTER EXCLUSIVITY: no two same-class intervals whose register SPANS
//      overlap have overlapping live ranges. This is the core soundness
//      property. A value that occupies N consecutive registers is compared over
//      its whole span, not its base alone, so a two-register value sitting on
//      top of its neighbor is caught here.
//   2. MUST_HAVE_REGISTER: every `must_have_register` use is covered by a
//      register-located interval of its value, never only a spill slot.
//   3. ASSIGNMENT: every value interval with a use was placed somewhere.
//   4. SPAN LEGALITY: a placed interval's base register meets the alignment its
//      width demands, and its span fits the register-index space.
//
// The two legitimate same-span overlaps, a value interval and its OWN
// entry-parameter fixed interval at entry, and a coalesced copy pair, are
// exempt from check 1. Both share a WHOLE span. A PARTIAL span overlap is never
// legitimate, so it stays a violation even for such a pair.
// ===========================================================================

/// One thing the allocation got wrong, found by `verifyIntervals`. `a` and `b` index the intervals
/// slice passed to the verifier, and `b == a` for the single-interval checks. `pos` is the program
/// position the violation manifests at.
pub const Violation = struct {
    kind: enum { reg_overlap, must_have_spilled, unassigned, misaligned_span },
    a: usize,
    b: usize,
    pos: u32,
};

/// The physical register an interval OCCUPIES, or null when it holds none: a spilled value
/// interval, or a value interval the scan never placed. A fixed interval occupies its
/// `fixed_reg`. A placed value interval occupies its assigned `.reg`. This is the nullable
/// analogue of `assignedReg`.
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

/// True iff the two intervals occupy the SAME register span: same base, same width. Both legitimate
/// same-register overlaps, an entry-parameter pin and a coalesced copy pair, share a whole span, so
/// the exemptions in check 1 are gated on this. A PARTIAL overlap, for example a two-register value
/// based one register into another two-register value, is never legitimate and stays a violation.
/// For a one-register model this is exactly the `base_a == base_b` the check already required before
/// reaching an exemption.
fn sameSpan(a: *const Interval, base_a: u16, b: *const Interval, base_b: u16) bool {
    return base_a == base_b and a.regs == b.regs;
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

/// Verify a completed allocation, returning every soundness `Violation`. An empty slice means the
/// allocation is valid. The caller owns and frees the returned slice. See the section header for
/// the three checks. The input is the flattened final intervals, originals plus split children.
/// It is read-only and not freed here.
pub fn verifyIntervals(allocator: std.mem.Allocator, intervals: []const Interval) Error![]Violation {
    var violations: std.ArrayList(Violation) = .empty;
    errdefer violations.deinit(allocator);

    // CHECK 1: register exclusivity. Every pair of same-class intervals whose register SPANS overlap
    // must have disjoint live ranges, except a value interval and its own entry-param fixed interval
    // sharing the ABI register span at entry, or a coalesced copy pair.
    for (intervals, 0..) |*ia, i| {
        const ra = occupiedReg(ia) orelse continue;
        for (intervals[i + 1 ..], i + 1..) |*ib, j| {
            if (ia.class != ib.class) continue;
            const rb = occupiedReg(ib) orelse continue;
            // The FULL span, not just the base. A two-register value based at R4 occupies R4 AND R5,
            // so it conflicts with anything that holds R5 as well. A base-only compare misses that,
            // and a missed conflict is a silent wrong-answer bug.
            if (!spansOverlap(ra, ia.regs, rb, ib.regs)) continue;
            if (sameSpan(ia, ra, ib, rb)) {
                if (isEntryParamHintPair(ia, ib)) continue;
                // A coalesced copy destination and its source share one register span over the
                // single copy position, a no-op move. That is their only overlap (the source dies at
                // the copy), so it is not a conflict.
                if (isCopyCoalescePair(ia, ib)) continue;
            }
            if (occupancyConflict(ia, ib)) |pos| {
                try violations.append(allocator, .{ .kind = .reg_overlap, .a = i, .b = j, .pos = pos });
            }
        }
    }

    // CHECK 2: must_have_register satisfaction. Every `must_have_register` use of a value must
    // fall in a `.reg`-located interval of that same value. A use covered ONLY by a `.slot`
    // interval would read an operand from memory where the target forbids it.
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

    // CHECK 4: span legality. A placed interval's BASE must meet the alignment its width demands,
    // and the whole span must fit the register-index space the scan reasons over. A misaligned base
    // is the first shape of bug a width-aware backend hits, for example an odd base handed to a
    // 64-bit address pair, and the hardware would then read the wrong register. An alignment of 1
    // divides every index, so a one-register model never reaches this.
    for (intervals, 0..) |*ia, i| {
        const ra = occupiedReg(ia) orelse continue;
        std.debug.assert(ia.reg_align >= 1);
        if (ra % ia.reg_align == 0 and @as(usize, ra) + ia.regs <= max_phys_regs) continue;
        try violations.append(allocator, .{ .kind = .misaligned_span, .a = i, .b = i, .pos = ia.start() });
    }

    return violations.toOwnedSlice(allocator);
}
