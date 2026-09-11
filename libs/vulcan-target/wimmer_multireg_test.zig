//! Unit tests for the shared Wimmer-Franz allocator's MULTI-REGISTER value support
//! (`RegDescription.regWidth` and `RegWidth`), driven by a synthetic backend rather than a real one.
//!
//! No shipping backend needs a multi-register value yet. aarch64 and x86_64 hold a 128-bit vector in
//! ONE architectural register, so they never opt in, and a test built on them could not exercise the
//! feature at all. The NVIDIA backend is the first case that needs it, because a 64-bit address
//! lives in an EVEN-ALIGNED GPR pair. The synthetic description below models exactly that rule: a
//! global pointer takes two consecutive registers on an even base, and every other value takes one.
//!
//! The tests cover the four places a width above one changes an answer: the free-register scan finds
//! an ALIGNED RUN, interference conflicts on EVERY register of a span and not the base alone, the
//! prologue saves every callee-saved register of a span, and the verifier rejects both a partial
//! span overlap and a misaligned base.

const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");

const wimmer = target.wimmer;
const Function = ir.function.Function;
const Value = ir.function.Value;
const Location = wimmer.Location;

// ---------------------------------------------------------------------------
// The synthetic backend.
//
// ONE register class ("gpr"). The class scratch is register 2, which reserves
// the aligned pair 2:3, because a move the resolver routes through the scratch
// carries a whole value. The allocatable pool is a test parameter, so a test can
// make a run of two aligned registers the only placement, or none at all.
// ---------------------------------------------------------------------------

/// The backend context. `pool` is the allocatable register set and `saved` the callee-saved subset.
const Synth = struct {
    pool: []const u16,
    saved: []const u16,
    /// What every operand use reports. `must_have_register` is the conservative default aarch64
    /// also reports. A test that wants the scan to SPILL rather than bail sets
    /// `should_have_register`, which lets a value read from its slot, the way x86 memory operands do.
    use_kind: wimmer.UseKind = .must_have_register,
    /// An optional entry-parameter pin, the ABI pre-color a real backend reports for an argument
    /// register. Null means the function takes no argument in a fixed register.
    pin: ?wimmer.FixedAssign = null,

    /// One class, so every value is class 0.
    fn classOf(_: *const anyopaque, _: *const Function, _: Value) u16 {
        return 0;
    }

    fn useKind(ctx: *const anyopaque, _: *const Function, _: ir.function.Inst, _: Value) wimmer.UseKind {
        const self: *const Synth = @ptrCast(@alignCast(ctx));
        return self.use_kind;
    }

    /// The NVIDIA pointer rule, in miniature: a GLOBAL pointer is a 64-bit device address, so it
    /// takes an even-aligned pair of registers. Every other value takes one register at any index.
    fn regWidth(_: *const anyopaque, func: *const Function, v: Value) wimmer.RegWidth {
        return switch (func.types.type_kind(func.valueType(v))) {
            .ptr => .{ .regs = 2, .alignment = 2 },
            else => .{},
        };
    }
};

/// Build the synthetic `RegDescription`. Every owned slice is allocated here, so `deinit` frees it
/// the same way a real backend's builder does. `entry_fixed` and `call_sites` stay empty: this
/// feature is about widths, and an ABI pin or a call clobber would only add noise.
fn synthDescription(allocator: std.mem.Allocator, ctx: *const Synth) !wimmer.RegDescription {
    const classes = try allocator.alloc(wimmer.RegClass, 1);
    errdefer allocator.free(classes);
    classes[0] = .{
        .name = "gpr",
        .allocatable = try allocator.dupe(u16, ctx.pool),
        .callee_saved = try allocator.dupe(u16, ctx.saved),
        // Wide enough for the widest value of the class (a two-register address), which is the
        // obligation `RegWidth` puts on a backend that reports a width above one.
        .slot_bytes = 8,
    };
    const scratch = try allocator.alloc(u16, 1);
    errdefer allocator.free(scratch);
    scratch[0] = 2; // reserves the aligned pair 2:3
    const entry_fixed = try allocator.alloc(wimmer.FixedAssign, if (ctx.pin == null) 0 else 1);
    errdefer allocator.free(entry_fixed);
    if (ctx.pin) |pin| entry_fixed[0] = pin;
    return .{
        .classes = classes,
        .classOf = Synth.classOf,
        .useKind = Synth.useKind,
        .entry_fixed = entry_fixed,
        .call_sites = try allocator.alloc(wimmer.CallSite, 0),
        .scratch = scratch,
        .ctx = ctx,
        .regWidth = Synth.regWidth,
    };
}

/// The single register a value was placed in, or null when it spilled or has no segment. Every test
/// function here keeps each value in one place, so one segment is the whole story.
fn soleReg(alloc: *const wimmer.Allocation, v: Value) ?u16 {
    const segs = alloc.segments.get(v) orelse return null;
    if (segs.len != 1) return null;
    return switch (segs[0].loc) {
        .reg => |r| r,
        .slot => null,
    };
}

/// True iff some segment of `v` lives in a spill slot, so the value left the registers at least once.
fn hasSlotSegment(alloc: *const wimmer.Allocation, v: Value) bool {
    const segs = alloc.segments.get(v) orelse return false;
    for (segs) |seg| {
        switch (seg.loc) {
            .slot => return true,
            .reg => {},
        }
    }
    return false;
}

/// Assert every REGISTER segment of `v` sits on a base that meets `alignment`. A split can hand a
/// value several register pieces, and each one has to be legal on its own.
fn expectAlignedRegSegments(alloc: *const wimmer.Allocation, v: Value, alignment: u16) !void {
    const segs = alloc.segments.get(v) orelse return error.ValueHasNoSegments;
    for (segs) |seg| {
        switch (seg.loc) {
            .reg => |r| try std.testing.expectEqual(@as(u16, 0), r % alignment),
            .slot => {},
        }
    }
}

/// True iff `set` records the callee-saved register `reg` of class 0.
fn savedContains(set: []const wimmer.UsedSaved, reg: u16) bool {
    for (set) |u| {
        if (u.class == 0 and u.reg == reg) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Allocation: aligned runs, whole-span interference, alignment, exhaustion.
// ---------------------------------------------------------------------------

test "multireg: a two-register value takes an aligned run and a narrow value skips its second half" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptrt = try func.types.ptrGlobal();

    // b0(p: ptr, x: i32):
    //   l = load p     ; p is read here
    //   s = add l, x   ; x is read here
    //   ret s
    // `p` and `x` are both live at position 0, so they cannot share a register.
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptrt);
    const x = try func.appendBlockParam(b, i32t);
    const l = try func.appendInst(b, i32t, .{ .load = .{ .ptr = p } });
    const s = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = l, .rhs = x } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });

    // The pool is exactly 4, 5, 6, 7. `p` needs an even base with its neighbor free, so 4 is the
    // only fit, and it then OWNS both 4 and 5. `x` must therefore land on 6. A scan that blocked
    // only the BASE register of `p` would hand `x` register 5 and corrupt the high half of the
    // address, which is the silent wrong-answer bug this test exists to catch.
    const ctx = Synth{ .pool = &.{ 4, 5, 6, 7 }, .saved = &.{} };
    var desc = try synthDescription(allocator, &ctx);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    try std.testing.expectEqual(@as(?u16, 4), soleReg(&alloc, p));
    try std.testing.expectEqual(@as(?u16, 6), soleReg(&alloc, x));
}

test "multireg: an odd-based pool still gives the pair an even base" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptrt = try func.types.ptrGlobal();

    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptrt);
    const l = try func.appendInst(b, i32t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(l) });

    // The pool starts at the ODD register 5. A run of two exists at 5:6, but the alignment forbids
    // an odd base, so the only legal placement is 6:7. An allocator that ignored the alignment would
    // answer 5.
    const ctx = Synth{ .pool = &.{ 5, 6, 7 }, .saved = &.{} };
    var desc = try synthDescription(allocator, &ctx);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    try std.testing.expectEqual(@as(?u16, 6), soleReg(&alloc, p));
}

test "multireg: a pool with no aligned run of two refuses the address" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptrt = try func.types.ptrGlobal();

    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptrt);
    const l = try func.appendInst(b, i32t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(l) });

    // A single odd register holds no aligned run of two. The scan finds no free span, the blocked
    // path finds nothing to evict, and the spill path cannot satisfy the address's own
    // `must_have_register` use at the start of its remaining piece. So the allocation bails the same
    // `error.Unsupported` a one-register class bails on a too-large same-position demand, rather
    // than placing the pair somewhere it does not fit.
    const ctx = Synth{ .pool = &.{5}, .saved = &.{} };
    var desc = try synthDescription(allocator, &ctx);
    defer desc.deinit(allocator);

    try std.testing.expectError(error.Unsupported, wimmer.allocate(allocator, &func, &desc));
}

test "multireg: a two-register value under pressure spills to ONE slot" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptrt = try func.types.ptrGlobal();

    // Two addresses live at once, and the pool holds exactly one aligned run of two. One of them has
    // to leave the registers.
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptrt);
    const q = try func.appendBlockParam(b, ptrt);
    const l = try func.appendInst(b, i32t, .{ .load = .{ .ptr = p } });
    const m = try func.appendInst(b, i32t, .{ .load = .{ .ptr = q } });
    const s = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = l, .rhs = m } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });

    // The pool holds ONE aligned run of two (4:5). Register 6 cannot start a pair, because 7 is
    // outside the pool, so the second address has nowhere to go and must leave the registers. The
    // reads may come from a slot here, so the scan spills instead of bailing.
    const ctx = Synth{ .pool = &.{ 4, 5, 6 }, .saved = &.{}, .use_kind = .should_have_register };
    var desc = try synthDescription(allocator, &ctx);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // Something did spill, and the spill went to a slot of the ONE class. A slot is `slot_bytes`
    // wide and the class sized that for its widest value, so a two-register value takes a single
    // slot exactly as a one-register value does.
    try std.testing.expectEqual(@as(u32, 1), alloc.slot_count_per_class[0]);
    try std.testing.expect(hasSlotSegment(&alloc, q));
    // Every register piece of BOTH addresses still keeps a legal even base after the split.
    try expectAlignedRegSegments(&alloc, p, 2);
    try expectAlignedRegSegments(&alloc, q, 2);
}

test "multireg: the blocked-register path never evicts half of a same-start pair" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptrt = try func.types.ptrGlobal();

    // b0(p: ptr, x: i32, y: i32):
    //   a  = add y, y
    //   b2 = add a, a
    //   l  = load p
    //   s  = add b2, l
    //   t  = add s, x
    //   ret t
    // The address, `x` and `y` are all born at position 0 and the pool holds three registers, so the
    // free-register scan runs out and `y` reaches the blocked-register path while the address is
    // still active and STARTS at the same position.
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptrt);
    const x = try func.appendBlockParam(b, i32t);
    const y = try func.appendBlockParam(b, i32t);
    const a = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = y } });
    const b2 = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
    const l = try func.appendInst(b, i32t, .{ .load = .{ .ptr = p } });
    const s = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = b2, .rhs = l } });
    const t = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = x } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(t) });

    // An interval that STARTS at the current position cannot be split there, so neither register of
    // the address is evictable. A blocked-register path that recorded only the BASE would read the
    // second register as free and evictable, take it, and then try to split the address at its own
    // start position, which is not a legal split point.
    const ctx = Synth{ .pool = &.{ 4, 5, 6 }, .saved = &.{} };
    var desc = try synthDescription(allocator, &ctx);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // The address is placed first and the pool holds exactly one aligned run, so it starts on 4.
    const p_segs = alloc.segments.get(p).?;
    try std.testing.expect(p_segs.len >= 1);
    try std.testing.expectEqual(Location{ .reg = 4 }, p_segs[0].loc);
    // Every register piece of the address, after any split, still sits on an even base.
    try expectAlignedRegSegments(&alloc, p, 2);
}

test "multireg: an entry-parameter pin blocks BOTH registers of the pinned pair" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptrt = try func.types.ptrGlobal();

    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptrt);
    const x = try func.appendBlockParam(b, i32t);
    const l = try func.appendInst(b, i32t, .{ .load = .{ .ptr = p } });
    const s = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = l, .rhs = x } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });

    // The address arrives in the ABI register pair 4:5, so 4 is pinned and the free pool starts at
    // 5. A pin that owned only its BASE would leave 5 in play, and `x`, which is live at entry, would
    // take the high half of an incoming address. The pin covers the parameter's whole span, so `x`
    // lands on 6.
    // Register 5 stays in the free pool on purpose. It is the half of the pinned pair the allocator
    // has to keep to itself without being told, which is exactly what the pin's width is for.
    const ctx = Synth{
        .pool = &.{ 5, 6, 7, 8 },
        .saved = &.{},
        .pin = .{ .value = p, .class = 0, .reg = 4 },
    };
    var desc = try synthDescription(allocator, &ctx);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    try std.testing.expectEqual(@as(?u16, 4), soleReg(&alloc, p));
    try std.testing.expectEqual(@as(?u16, 6), soleReg(&alloc, x));
}

test "multireg: the prologue saves BOTH callee-saved registers of a pair" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptrt = try func.types.ptrGlobal();

    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptrt);
    const l = try func.appendInst(b, i32t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(l) });

    // Registers 4 and 5 are BOTH callee-saved, and the address takes both. Recording only the base
    // would leave register 5 unsaved, so the function would destroy its caller's copy of it.
    const ctx = Synth{ .pool = &.{ 4, 5, 6, 7 }, .saved = &.{ 4, 5 } };
    var desc = try synthDescription(allocator, &ctx);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    try std.testing.expectEqual(@as(?u16, 4), soleReg(&alloc, p));
    try std.testing.expect(savedContains(alloc.used_callee_saved, 4));
    try std.testing.expect(savedContains(alloc.used_callee_saved, 5));
}

test "multireg: buildIntervals carries the reported width onto every interval" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptrt = try func.types.ptrGlobal();

    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptrt);
    const l = try func.appendInst(b, i32t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(l) });

    const ctx = Synth{ .pool = &.{ 4, 5, 6, 7 }, .saved = &.{} };
    var desc = try synthDescription(allocator, &ctx);
    defer desc.deinit(allocator);

    const intervals = try wimmer.buildIntervals(allocator, &func, &desc);
    defer wimmer.freeIntervals(allocator, intervals);

    for (intervals) |*iv| {
        const v = iv.value orelse continue;
        if (v == p) {
            try std.testing.expectEqual(@as(u16, 2), iv.regs);
            try std.testing.expectEqual(@as(u16, 2), iv.reg_align);
        } else {
            try std.testing.expectEqual(@as(u16, 1), iv.regs);
            try std.testing.expectEqual(@as(u16, 1), iv.reg_align);
        }
    }
}

test "multireg: a description with no width hook leaves every interval one register wide" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptrt = try func.types.ptrGlobal();

    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptrt);
    const l = try func.appendInst(b, i32t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(l) });

    const ctx = Synth{ .pool = &.{ 4, 5, 6, 7 }, .saved = &.{} };
    var desc = try synthDescription(allocator, &ctx);
    defer desc.deinit(allocator);
    // Drop the hook: this is the shape every shipping backend hands the allocator today.
    desc.regWidth = null;

    const intervals = try wimmer.buildIntervals(allocator, &func, &desc);
    defer wimmer.freeIntervals(allocator, intervals);

    for (intervals) |*iv| {
        try std.testing.expectEqual(@as(u16, 1), iv.regs);
        try std.testing.expectEqual(@as(u16, 1), iv.reg_align);
    }

    // And the pointer then takes ONE register, so the second half of the pool stays free.
    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);
    try std.testing.expectEqual(@as(?u16, 4), soleReg(&alloc, p));
}

// ---------------------------------------------------------------------------
// The verifier over a multi-register span. These build interval sets by hand,
// the way the existing verifier tests do, so a deliberately corrupted
// assignment can be handed straight to `verifyIntervals`.
// ---------------------------------------------------------------------------

/// Build an `Interval` with freshly owned `ranges`/`uses`, a register width, and a location. The
/// verifier never frees its input, so the test releases it with `freeOwned`.
fn ownedWide(
    allocator: std.mem.Allocator,
    value: ?Value,
    ranges: []const wimmer.Range,
    regs: u16,
    alignment: u16,
    location: ?Location,
) !wimmer.Interval {
    return .{
        .value = value,
        .class = 0,
        .fixed_reg = null,
        .ranges = try allocator.dupe(wimmer.Range, ranges),
        .uses = try allocator.alloc(wimmer.UsePos, 0),
        .location = location,
        .regs = regs,
        .reg_align = alignment,
    };
}

fn freeOwned(allocator: std.mem.Allocator, iv: wimmer.Interval) void {
    allocator.free(iv.ranges);
    allocator.free(iv.uses);
}

test "verify: a narrow value on the SECOND register of a live pair is flagged" {
    const allocator = std.testing.allocator;
    const v0: Value = @enumFromInt(0);
    const v1: Value = @enumFromInt(1);
    // A deliberately corrupted assignment: `v0` owns the pair 4:5, and `v1` sits on 5 while `v0` is
    // still live. The two BASE registers differ, so a base-only exclusivity check sees nothing. The
    // span check sees the real conflict.
    var ivs = [_]wimmer.Interval{
        try ownedWide(allocator, v0, &.{.{ .from = 0, .to = 8 }}, 2, 2, .{ .reg = 4 }),
        try ownedWide(allocator, v1, &.{.{ .from = 2, .to = 6 }}, 1, 1, .{ .reg = 5 }),
    };
    defer for (ivs) |iv| freeOwned(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 1), violations.len);
    try std.testing.expect(violations[0].kind == .reg_overlap);
    try std.testing.expectEqual(@as(u32, 2), violations[0].pos);
}

test "verify: a narrow value just PAST a live pair is not flagged" {
    const allocator = std.testing.allocator;
    const v0: Value = @enumFromInt(0);
    const v1: Value = @enumFromInt(1);
    // The other direction of the same guard: `v1` on register 6 is outside the pair 4:5, so the two
    // do not conflict. A span check that always conflicted would reject this valid allocation.
    var ivs = [_]wimmer.Interval{
        try ownedWide(allocator, v0, &.{.{ .from = 0, .to = 8 }}, 2, 2, .{ .reg = 4 }),
        try ownedWide(allocator, v1, &.{.{ .from = 2, .to = 6 }}, 1, 1, .{ .reg = 6 }),
    };
    defer for (ivs) |iv| freeOwned(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 0), violations.len);
}

test "verify: two pairs that OVERLAP by one register are flagged" {
    const allocator = std.testing.allocator;
    const v0: Value = @enumFromInt(0);
    const v1: Value = @enumFromInt(1);
    // 4:5 and 5:6 share register 5. Neither base equals the other, so only the span test finds it.
    // The second pair is also misaligned, so the span-legality check fires as well.
    var ivs = [_]wimmer.Interval{
        try ownedWide(allocator, v0, &.{.{ .from = 0, .to = 8 }}, 2, 2, .{ .reg = 4 }),
        try ownedWide(allocator, v1, &.{.{ .from = 3, .to = 8 }}, 2, 2, .{ .reg = 5 }),
    };
    defer for (ivs) |iv| freeOwned(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 2), violations.len);
    try std.testing.expect(violations[0].kind == .reg_overlap);
    try std.testing.expectEqual(@as(u32, 3), violations[0].pos);
    try std.testing.expect(violations[1].kind == .misaligned_span);
}

test "verify: a pair on an odd base is flagged as a misaligned span" {
    const allocator = std.testing.allocator;
    const v0: Value = @enumFromInt(0);
    // The single-interval corruption: nothing else is live, but the base breaks the alignment the
    // width demands, so the hardware would read the wrong register.
    var ivs = [_]wimmer.Interval{
        try ownedWide(allocator, v0, &.{.{ .from = 0, .to = 4 }}, 2, 2, .{ .reg = 7 }),
    };
    defer for (ivs) |iv| freeOwned(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 1), violations.len);
    try std.testing.expect(violations[0].kind == .misaligned_span);
}

test "verify: two pairs on the same aligned base with disjoint ranges are not flagged" {
    const allocator = std.testing.allocator;
    const v0: Value = @enumFromInt(0);
    const v1: Value = @enumFromInt(1);
    // Reuse of one pair after the first value dies is the normal, valid case.
    var ivs = [_]wimmer.Interval{
        try ownedWide(allocator, v0, &.{.{ .from = 0, .to = 4 }}, 2, 2, .{ .reg = 4 }),
        try ownedWide(allocator, v1, &.{.{ .from = 4, .to = 8 }}, 2, 2, .{ .reg = 4 }),
    };
    defer for (ivs) |iv| freeOwned(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 0), violations.len);
}
