//! Unit tests for the shared Wimmer-Franz allocator's target abstraction, exercised through
//! aarch64's `aarch64RegDescription`. These assert the DESCRIPTION only (no allocation runs yet):
//! that the per-function register model the aarch64 backend hands the shared allocator matches how
//! the existing aarch64 `allocate` actually builds its pools, pins entry params, and clobbers calls.

const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");

const aarch64 = target.aarch64.isel;
const riscv64 = target.riscv64.isel;
const x86_64 = target.x86_64.isel;
const wimmer = target.wimmer;
const Function = ir.function.Function;
const Value = ir.function.Value;

/// True iff `set` contains `idx`.
fn contains(set: []const u16, idx: u16) bool {
    for (set) |x| if (x == idx) return true;
    return false;
}

test "aarch64 description: leaf gpr pool, entry param pinning, no call sites" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // Two classes: 0 = gpr, 1 = fpr.
    try std.testing.expectEqual(@as(usize, 2), desc.classes.len);
    try std.testing.expectEqualStrings("gpr", desc.classes[0].name);
    try std.testing.expectEqualStrings("fpr", desc.classes[1].name);

    // Leaf GPR pool contains the caller-saved temporaries x9..x12 (indices 9..12). Two i32 params
    // pin x0..x1, so x2..x7 are also available, but x9..x12 must always be present.
    for ([_]u16{ 9, 10, 11, 12 }) |idx| {
        try std.testing.expect(contains(desc.classes[0].allocatable, idx));
    }
    // The two unused integer arg registers x2..x7 are in the leaf pool (n_gpr = 2).
    for ([_]u16{ 2, 3, 4, 5, 6, 7 }) |idx| {
        try std.testing.expect(contains(desc.classes[0].allocatable, idx));
    }
    // x0/x1 are pinned entry params, so they are NOT in the free pool.
    try std.testing.expect(!contains(desc.classes[0].allocatable, 0));
    try std.testing.expect(!contains(desc.classes[0].allocatable, 1));

    // Callee-saved gpr set is x19..x28.
    try std.testing.expectEqual(@as(usize, 10), desc.classes[0].callee_saved.len);
    for (19..29) |idx| try std.testing.expect(contains(desc.classes[0].callee_saved, @intCast(idx)));

    // Uniform 16-byte spill slots for both classes.
    try std.testing.expectEqual(@as(u16, 16), desc.classes[0].slot_bytes);
    try std.testing.expectEqual(@as(u16, 16), desc.classes[1].slot_bytes);

    // Entry params: the first gpr param maps to x0 (class 0, reg 0), the second to x1.
    try std.testing.expectEqual(@as(usize, 2), desc.entry_fixed.len);
    try std.testing.expectEqual(x, desc.entry_fixed[0].value);
    try std.testing.expectEqual(@as(u16, 0), desc.entry_fixed[0].class);
    try std.testing.expectEqual(@as(u16, 0), desc.entry_fixed[0].reg);
    try std.testing.expectEqual(y, desc.entry_fixed[1].value);
    try std.testing.expectEqual(@as(u16, 0), desc.entry_fixed[1].class);
    try std.testing.expectEqual(@as(u16, 1), desc.entry_fixed[1].reg);

    // A leaf function makes no calls, so there are no call-clobber sites.
    try std.testing.expectEqual(@as(usize, 0), desc.call_sites.len);

    // classOf: an integer value is gpr (0). useKind: always must_have_register on aarch64.
    try std.testing.expectEqual(@as(u16, 0), desc.classOf(desc.ctx, &func, prod));
    const prod_inst = func.blockInsts(b)[0];
    try std.testing.expectEqual(wimmer.UseKind.must_have_register, desc.useKind(desc.ctx, &func, prod_inst, x));

    // Scratch registers, indexed by class: gpr scratch x17 (index 17), fpr scratch v27 (index 27).
    try std.testing.expectEqual(@as(usize, 2), desc.scratch.len);
    try std.testing.expectEqual(@as(u16, 17), desc.scratch[0]);
    try std.testing.expectEqual(@as(u16, 27), desc.scratch[1]);
}

test "aarch64 description: fpr class pool and float param pinning" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f);
    const bp = try func.appendBlockParam(b, f);
    const sum = try func.appendInst(b, f, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // Leaf FPR pool contains the caller-saved v16..v23 plus the unused fp arg registers v2..v7.
    for (16..24) |idx| try std.testing.expect(contains(desc.classes[1].allocatable, @intCast(idx)));
    for ([_]u16{ 2, 3, 4, 5, 6, 7 }) |idx| {
        try std.testing.expect(contains(desc.classes[1].allocatable, idx));
    }
    // v0/v1 are pinned float params, not free.
    try std.testing.expect(!contains(desc.classes[1].allocatable, 0));
    try std.testing.expect(!contains(desc.classes[1].allocatable, 1));

    // Callee-saved fpr set is v8..v15.
    for (8..16) |idx| try std.testing.expect(contains(desc.classes[1].callee_saved, @intCast(idx)));

    // A float value classes into the fpr class (1).
    try std.testing.expectEqual(@as(u16, 1), desc.classOf(desc.ctx, &func, sum));

    // The two float params pin v0, v1 (class 1).
    try std.testing.expectEqual(@as(usize, 2), desc.entry_fixed.len);
    try std.testing.expectEqual(@as(u16, 1), desc.entry_fixed[0].class);
    try std.testing.expectEqual(@as(u16, 0), desc.entry_fixed[0].reg);
    try std.testing.expectEqual(@as(u16, 1), desc.entry_fixed[1].class);
    try std.testing.expectEqual(@as(u16, 1), desc.entry_fixed[1].reg);
}

test "aarch64 description: a non-leaf function with a call records a call-clobber site" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    // A call makes the function non-leaf and records a clobber site at the call position.
    const called = try func.appendCall(b, t, "callee", &.{x});
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // Non-leaf GPR pool is the callee-saved x19..x28.
    try std.testing.expectEqual(@as(usize, 10), desc.classes[0].allocatable.len);
    for (19..29) |idx| try std.testing.expect(contains(desc.classes[0].allocatable, @intCast(idx)));
    // Non-leaf FPR pool is the callee-saved v8..v15.
    for (8..16) |idx| try std.testing.expect(contains(desc.classes[1].allocatable, @intCast(idx)));

    // At least one call site.
    try std.testing.expect(desc.call_sites.len >= 1);
    const cs = desc.call_sites[0];

    // Its clobbered list carries both classes.
    var gpr_regs: ?[]const u16 = null;
    var fpr_regs: ?[]const u16 = null;
    for (cs.clobbered) |cr| {
        if (cr.class == 0) gpr_regs = cr.regs;
        if (cr.class == 1) fpr_regs = cr.regs;
    }
    try std.testing.expect(gpr_regs != null);
    try std.testing.expect(fpr_regs != null);

    // Class 0: the caller-saved gpr set x0..x17.
    for (0..18) |idx| try std.testing.expect(contains(gpr_regs.?, @intCast(idx)));
    // The callee-saved x19..x28 are NOT clobbered by a call.
    for (19..29) |idx| try std.testing.expect(!contains(gpr_regs.?, @intCast(idx)));

    // Class 1: the caller-saved fp regs PLUS the callee-saved v8..v15 (the vector-across-call
    // quirk over-clobbers all fp registers), so v0..v15 are all present.
    for (0..16) |idx| try std.testing.expect(contains(fpr_regs.?, @intCast(idx)));
}

// ---------------------------------------------------------------------------
// riscv64 RegDescription: the four-class (int/float/RVV-vector/VPU-vector),
// vpu-aware register model the shared allocator consumes. These assert the
// DESCRIPTION only (no allocation runs), mirroring the aarch64 tests above.
// ---------------------------------------------------------------------------

/// The clobbered register set for `class` at call site `cs`, or null when the call does not touch
/// that class (used by the riscv64 call-site tests).
fn clobberOf(cs: wimmer.CallSite, class: u16) ?[]const u16 {
    for (cs.clobbered) |cr| {
        if (cr.class == class) return cr.regs;
    }
    return null;
}

test "riscv64 description: non-vpu int + float pools and entry param pinning" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const ip = try func.appendBlockParam(b, i32t);
    const fp = try func.appendBlockParam(b, f32t);
    const isum = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = ip, .rhs = ip } });
    const fsum = try func.appendInst(b, f32t, .{ .arith = .{ .op = .add, .lhs = fp, .rhs = fp } });
    _ = fsum;
    func.setTerminator(b, .{ .ret = isum });

    var desc = try riscv64.riscv64RegDescription(allocator, &func, false, false);
    defer desc.deinit(allocator);

    // Four classes: 0 int, 1 float, 2 RVV-vector, 3 VPU-vector.
    try std.testing.expectEqual(@as(usize, 4), desc.classes.len);
    try std.testing.expectEqualStrings("int", desc.classes[0].name);
    try std.testing.expectEqualStrings("float", desc.classes[1].name);
    try std.testing.expectEqualStrings("vector", desc.classes[2].name);
    try std.testing.expectEqualStrings("vpu_vector", desc.classes[3].name);

    // Class 0 (int): allocatable is the caller-saved temps x5/x7/x28..x31 plus the callee-saved
    // x9/x18..x27. Slot size 8 bytes.
    for ([_]u16{ 5, 7, 28, 29, 30, 31 }) |idx| try std.testing.expect(contains(desc.classes[0].allocatable, idx));
    for ([_]u16{ 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 }) |idx| try std.testing.expect(contains(desc.classes[0].allocatable, idx));
    try std.testing.expectEqual(@as(usize, 11), desc.classes[0].callee_saved.len);
    for ([_]u16{ 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 }) |idx| try std.testing.expect(contains(desc.classes[0].callee_saved, idx));
    try std.testing.expectEqual(@as(u16, 8), desc.classes[0].slot_bytes);

    // Class 1 (float): allocatable is the caller-saved f0..f7/f28/f29 plus the callee-saved
    // f8/f9/f18..f27. Slot size 8 bytes.
    for ([_]u16{ 0, 1, 2, 3, 4, 5, 6, 7, 28, 29 }) |idx| try std.testing.expect(contains(desc.classes[1].allocatable, idx));
    for ([_]u16{ 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 }) |idx| try std.testing.expect(contains(desc.classes[1].allocatable, idx));
    try std.testing.expectEqual(@as(usize, 12), desc.classes[1].callee_saved.len);
    try std.testing.expectEqual(@as(u16, 8), desc.classes[1].slot_bytes);

    // Class 2 (RVV vector): v1..v27 allocatable, all caller-saved (callee_saved empty), 16-byte slot.
    try std.testing.expectEqual(@as(usize, 27), desc.classes[2].allocatable.len);
    for (1..28) |idx| try std.testing.expect(contains(desc.classes[2].allocatable, @intCast(idx)));
    try std.testing.expectEqual(@as(usize, 0), desc.classes[2].callee_saved.len);
    try std.testing.expectEqual(@as(u16, 16), desc.classes[2].slot_bytes);

    // Class 3 (VPU vector): empty in non-vpu mode, 32-byte slot.
    try std.testing.expectEqual(@as(usize, 0), desc.classes[3].allocatable.len);
    try std.testing.expectEqual(@as(u16, 32), desc.classes[3].slot_bytes);

    // Entry params: the int param pins a0 (class 0, reg x10 = index 10), the float param pins fa0
    // (class 1, reg f10 = index 10).
    try std.testing.expectEqual(@as(usize, 2), desc.entry_fixed.len);
    try std.testing.expectEqual(ip, desc.entry_fixed[0].value);
    try std.testing.expectEqual(@as(u16, 0), desc.entry_fixed[0].class);
    try std.testing.expectEqual(@as(u16, 10), desc.entry_fixed[0].reg);
    try std.testing.expectEqual(fp, desc.entry_fixed[1].value);
    try std.testing.expectEqual(@as(u16, 1), desc.entry_fixed[1].class);
    try std.testing.expectEqual(@as(u16, 10), desc.entry_fixed[1].reg);

    // A leaf function makes no calls, so there are no call-clobber sites.
    try std.testing.expectEqual(@as(usize, 0), desc.call_sites.len);

    // classOf: an int value is class 0, a float value is class 1. useKind: always must_have_register.
    try std.testing.expectEqual(@as(u16, 0), desc.classOf(desc.ctx, &func, isum));
    try std.testing.expectEqual(@as(u16, 1), desc.classOf(desc.ctx, &func, fp));
    const first_inst = func.blockInsts(b)[0];
    try std.testing.expectEqual(wimmer.UseKind.must_have_register, desc.useKind(desc.ctx, &func, first_inst, ip));

    // Scratch per class: int x6 (6), float f31 (31), RVV v31 (31), VPU f31 (31).
    try std.testing.expectEqual(@as(usize, 4), desc.scratch.len);
    try std.testing.expectEqual(@as(u16, 6), desc.scratch[0]);
    try std.testing.expectEqual(@as(u16, 31), desc.scratch[1]);
    try std.testing.expectEqual(@as(u16, 31), desc.scratch[2]);
    try std.testing.expectEqual(@as(u16, 31), desc.scratch[3]);
}

test "riscv64 description: a call clobbers caller-saved of every class incl all vector regs" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32t);
    const called = try func.appendCall(b, i32t, "callee", &.{x});
    const sum = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try riscv64.riscv64RegDescription(allocator, &func, false, false);
    defer desc.deinit(allocator);

    try std.testing.expect(desc.call_sites.len >= 1);
    const cs = desc.call_sites[0];

    // Class 0 clobbers exactly the caller-saved int temps x5/x7/x28..x31; the callee-saved
    // x9/x18..x27 survive a call and are NOT clobbered.
    const int_clob = clobberOf(cs, 0).?;
    for ([_]u16{ 5, 7, 28, 29, 30, 31 }) |idx| try std.testing.expect(contains(int_clob, idx));
    for ([_]u16{ 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 }) |idx| try std.testing.expect(!contains(int_clob, idx));

    // Class 1 clobbers the caller-saved float temps f0..f7/f28/f29; the callee-saved floats survive.
    const float_clob = clobberOf(cs, 1).?;
    for ([_]u16{ 0, 1, 2, 3, 4, 5, 6, 7, 28, 29 }) |idx| try std.testing.expect(contains(float_clob, idx));
    for ([_]u16{ 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 }) |idx| try std.testing.expect(!contains(float_clob, idx));

    // Class 2 clobbers ALL of v1..v27: every RVV vector register is caller-saved, so a vector cannot
    // survive a call. This is the mechanism that lets the shared allocator SPILL a vector across a
    // call instead of the old error.Unsupported bail.
    const vec_clob = clobberOf(cs, 2).?;
    try std.testing.expectEqual(@as(usize, 27), vec_clob.len);
    for (1..28) |idx| try std.testing.expect(contains(vec_clob, @intCast(idx)));

    // Class 3 (VPU vector) is inactive in non-vpu mode: no VPU registers to clobber.
    const vpu_clob = clobberOf(cs, 3).?;
    try std.testing.expectEqual(@as(usize, 0), vpu_clob.len);
}

test "riscv64 description: software-f16 shrinks class 0 to x5/x7 and drops x28..x31 from the clobber list" {
    // SP3 Task 2, Gap B: `uses_f16 = true` mirrors the OLD `compileFunction`'s `reserve_f16_scratch`
    // gate (x28..x31 are the software f16 convert scratch, unconditionally clobbered by
    // `emitHalfToFloat`/`emitFloatToHalf` at every f16 boundary). Both the allocatable set and the
    // per-call clobber list must exclude them, so the shared allocator never places a live value there.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32t);
    const called = try func.appendCall(b, i32t, "callee", &.{x});
    const sum = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try riscv64.riscv64RegDescription(allocator, &func, false, true);
    defer desc.deinit(allocator);

    // Class 0 allocatable: x5/x7 (the shrunk temp slice) + the 11 callee-saved (x9/x18..x27) = 13.
    try std.testing.expectEqual(@as(usize, 13), desc.classes[0].allocatable.len);
    for ([_]u16{ 5, 7 }) |idx| try std.testing.expect(contains(desc.classes[0].allocatable, idx));
    for ([_]u16{ 28, 29, 30, 31 }) |idx| try std.testing.expect(!contains(desc.classes[0].allocatable, idx));
    for ([_]u16{ 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 }) |idx| try std.testing.expect(contains(desc.classes[0].allocatable, idx));

    // The per-call clobber list matches: x5/x7 clobbered, x28..x31 NOT (nothing is ever allocated
    // there, so treating them as call-clobbered would be a stray, meaningless fixed interval).
    try std.testing.expect(desc.call_sites.len >= 1);
    const cs = desc.call_sites[0];
    const int_clob = clobberOf(cs, 0).?;
    for ([_]u16{ 5, 7 }) |idx| try std.testing.expect(contains(int_clob, idx));
    for ([_]u16{ 28, 29, 30, 31 }) |idx| try std.testing.expect(!contains(int_clob, idx));
}

test "riscv64 description: vpu mode swaps the vector class and narrows the float pool" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32t);
    const called = try func.appendCall(b, i32t, "callee", &.{x});
    const sum = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try riscv64.riscv64RegDescription(allocator, &func, true, false);
    defer desc.deinit(allocator);

    // Class 2 (RVV vector) is empty in vpu mode: there is no RVV under the VPU.
    try std.testing.expectEqual(@as(usize, 0), desc.classes[2].allocatable.len);

    // Class 3 (VPU vector) takes over: f16..f27 (12 registers), all caller-saved.
    try std.testing.expectEqual(@as(usize, 12), desc.classes[3].allocatable.len);
    for (16..28) |idx| try std.testing.expect(contains(desc.classes[3].allocatable, @intCast(idx)));
    try std.testing.expectEqual(@as(usize, 0), desc.classes[3].callee_saved.len);
    try std.testing.expectEqual(@as(u16, 32), desc.classes[3].slot_bytes);

    // Class 1 (float) narrows to f0..f7 only, with an empty callee-saved set (f8/f9 are reserved as
    // vpu spill scratch, f16..f31 belong to the VPU vector partition).
    try std.testing.expectEqual(@as(usize, 8), desc.classes[1].allocatable.len);
    for (0..8) |idx| try std.testing.expect(contains(desc.classes[1].allocatable, @intCast(idx)));
    try std.testing.expectEqual(@as(usize, 0), desc.classes[1].callee_saved.len);

    // The call now clobbers ALL of the VPU vector partition f16..f27 (class 3), while class 2 is
    // empty. This is the vpu analogue of the RVV vector-crosses-call-by-spilling mechanism.
    const cs = desc.call_sites[0];
    const vpu_clob = clobberOf(cs, 3).?;
    try std.testing.expectEqual(@as(usize, 12), vpu_clob.len);
    for (16..28) |idx| try std.testing.expect(contains(vpu_clob, @intCast(idx)));
    try std.testing.expectEqual(@as(usize, 0), clobberOf(cs, 2).?.len);

    // A vector value now classes into class 3 (VPU), and the float scratch is f8 in vpu mode.
    try std.testing.expectEqual(@as(u16, 8), desc.scratch[1]);
}

// ---------------------------------------------------------------------------
// x86_64 RegDescription: the two-class (gpr incl callee-saved / xmm) register
// model with per-position fixed-register-op clobbers (call / div / shift). These
// assert the DESCRIPTION only (no allocation runs), mirroring the tests above.
// ---------------------------------------------------------------------------

/// The call/clobber site at position `pos`, or null when no site sits there (used by the x86_64
/// fixed-register-op tests, where div/shift positions each carry their own clobber site).
fn siteAtPos(desc: *const wimmer.RegDescription, pos: u32) ?wimmer.CallSite {
    for (desc.call_sites) |cs| {
        if (cs.pos == pos) return cs;
    }
    return null;
}

test "x86_64 description: gpr pool includes callee-saved, xmm pool, entry param arg-reg hints" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const ip = try func.appendBlockParam(b, i32t);
    const fp = try func.appendBlockParam(b, f32t);
    const isum = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = ip, .rhs = ip } });
    const fsum = try func.appendInst(b, f32t, .{ .arith = .{ .op = .add, .lhs = fp, .rhs = fp } });
    _ = fsum;
    func.setTerminator(b, .{ .ret = isum });

    var desc = try x86_64.x86_64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // Two classes: 0 = gpr, 1 = xmm.
    try std.testing.expectEqual(@as(usize, 2), desc.classes.len);
    try std.testing.expectEqualStrings("gpr", desc.classes[0].name);
    try std.testing.expectEqualStrings("xmm", desc.classes[1].name);

    // Class 0 (gpr): allocatable is the caller-saved set {rax,rcx,rdx,rsi,rdi,r8,r9} PLUS the
    // callee-saved set {rbx,r12,r13,r14,r15}. Index = the register's own enum value.
    for ([_]u16{ 0, 1, 2, 6, 7, 8, 9 }) |idx| try std.testing.expect(contains(desc.classes[0].allocatable, idx));
    for ([_]u16{ 3, 12, 13, 14, 15 }) |idx| try std.testing.expect(contains(desc.classes[0].allocatable, idx));
    // rbp/rsp (frame/stack) and r10/r11 (scratch) are excluded.
    for ([_]u16{ 4, 5, 10, 11 }) |idx| try std.testing.expect(!contains(desc.classes[0].allocatable, idx));
    try std.testing.expectEqual(@as(usize, 12), desc.classes[0].allocatable.len);
    try std.testing.expectEqual(@as(u16, 8), desc.classes[0].slot_bytes);

    // Callee-saved gpr set is exactly {rbx,r12,r13,r14,r15}.
    try std.testing.expectEqual(@as(usize, 5), desc.classes[0].callee_saved.len);
    for ([_]u16{ 3, 12, 13, 14, 15 }) |idx| try std.testing.expect(contains(desc.classes[0].callee_saved, idx));

    // Class 1 (xmm): allocatable is xmm0..xmm12 (xmm13/14/15 are reserved scratch), no callee-saved.
    try std.testing.expectEqual(@as(usize, 13), desc.classes[1].allocatable.len);
    for (0..13) |idx| try std.testing.expect(contains(desc.classes[1].allocatable, @intCast(idx)));
    try std.testing.expectEqual(@as(usize, 0), desc.classes[1].callee_saved.len);
    try std.testing.expectEqual(@as(u16, 16), desc.classes[1].slot_bytes);

    // Entry params: the first gpr param pins rdi (index 7), the first xmm param pins xmm0 (index 0).
    try std.testing.expectEqual(@as(usize, 2), desc.entry_fixed.len);
    try std.testing.expectEqual(ip, desc.entry_fixed[0].value);
    try std.testing.expectEqual(@as(u16, 0), desc.entry_fixed[0].class);
    try std.testing.expectEqual(@as(u16, 7), desc.entry_fixed[0].reg);
    try std.testing.expectEqual(fp, desc.entry_fixed[1].value);
    try std.testing.expectEqual(@as(u16, 1), desc.entry_fixed[1].class);
    try std.testing.expectEqual(@as(u16, 0), desc.entry_fixed[1].reg);

    // A leaf function with no fixed-register op makes no clobber sites.
    try std.testing.expectEqual(@as(usize, 0), desc.call_sites.len);

    // classOf: an int value is gpr (0), a float value is xmm (1). useKind: always must_have_register.
    try std.testing.expectEqual(@as(u16, 0), desc.classOf(desc.ctx, &func, isum));
    try std.testing.expectEqual(@as(u16, 1), desc.classOf(desc.ctx, &func, fp));
    const first_inst = func.blockInsts(b)[0];
    try std.testing.expectEqual(wimmer.UseKind.must_have_register, desc.useKind(desc.ctx, &func, first_inst, ip));

    // Scratch per class: gpr r11 (index 11), xmm xmm15 (index 15).
    try std.testing.expectEqual(@as(usize, 2), desc.scratch.len);
    try std.testing.expectEqual(@as(u16, 11), desc.scratch[0]);
    try std.testing.expectEqual(@as(u16, 15), desc.scratch[1]);
}

test "x86_64 description: a call clobbers caller-saved gpr but NOT callee-saved" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32t);
    const called = try func.appendCall(b, i32t, "callee", &.{x});
    const sum = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try x86_64.x86_64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    try std.testing.expect(desc.call_sites.len >= 1);
    const cs = desc.call_sites[0];

    // Class 0 clobbers exactly the caller-saved gpr set {rax,rcx,rdx,rsi,rdi,r8,r9}; the callee-saved
    // {rbx,r12,r13,r14,r15} survive a call and are NOT clobbered.
    const gpr_clob = clobberOf(cs, 0).?;
    for ([_]u16{ 0, 1, 2, 6, 7, 8, 9 }) |idx| try std.testing.expect(contains(gpr_clob, idx));
    for ([_]u16{ 3, 12, 13, 14, 15 }) |idx| try std.testing.expect(!contains(gpr_clob, idx));

    // Class 1 clobbers ALL of xmm0..xmm12 (every xmm is caller-saved under System V), so an xmm cannot
    // survive a call.
    const xmm_clob = clobberOf(cs, 1).?;
    try std.testing.expectEqual(@as(usize, 13), xmm_clob.len);
    for (0..13) |idx| try std.testing.expect(contains(xmm_clob, @intCast(idx)));
}

test "x86_64 description: a div clobbers rax+rdx and a shift clobbers rcx at their positions" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32t);
    const y = try func.appendBlockParam(b, i32t);
    // Position numbering: block-param row = 0, div = 1, shift = 2, terminator = 3.
    const quot = try func.appendInst(b, i32t, .{ .arith = .{ .op = .div, .lhs = x, .rhs = y } });
    const shifted = try func.appendInst(b, i32t, .{ .arith = .{ .op = .shl, .lhs = quot, .rhs = y } });
    func.setTerminator(b, .{ .ret = shifted });

    var desc = try x86_64.x86_64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // A div position (1) carries a class-0 clobber of {rax,rdx} only (index 0 and 2), no xmm.
    const div_site = siteAtPos(&desc, 1).?;
    const div_clob = clobberOf(div_site, 0).?;
    try std.testing.expectEqual(@as(usize, 2), div_clob.len);
    try std.testing.expect(contains(div_clob, 0));
    try std.testing.expect(contains(div_clob, 2));
    try std.testing.expect(clobberOf(div_site, 1) == null);

    // A shift position (2) carries a class-0 clobber of {rcx} only (index 1), no xmm.
    const shift_site = siteAtPos(&desc, 2).?;
    const shift_clob = clobberOf(shift_site, 0).?;
    try std.testing.expectEqual(@as(usize, 1), shift_clob.len);
    try std.testing.expect(contains(shift_clob, 1));
    try std.testing.expect(clobberOf(shift_site, 1) == null);
}

// ---------------------------------------------------------------------------
// Task 2: interval builder (buildIntervals). These exercise range/hole/use and
// fixed-interval construction, still driven through the aarch64 RegDescription.
// ---------------------------------------------------------------------------

const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

/// The value interval for `v` (fixed-register intervals are excluded), or null.
fn findValueInterval(intervals: []const wimmer.Interval, v: Value) ?*const wimmer.Interval {
    for (intervals) |*iv| {
        if (iv.fixed_reg == null and iv.value != null and iv.value.? == v) return iv;
    }
    return null;
}

/// The call-clobber fixed interval blocking physical register `reg` of `class`, or null.
fn findFixed(intervals: []const wimmer.Interval, class: u16, reg: u16) ?*const wimmer.Interval {
    for (intervals) |*iv| {
        if (iv.value == null and iv.fixed_reg != null and iv.class == class and iv.fixed_reg.? == reg) return iv;
    }
    return null;
}

test "intervals: a value dead between two uses has a hole" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const bool_t = try func.types.intern(.bool);

    // b0 defines and uses V, then conditionally branches to b2 (uses V) or b1 (does not). Block b1
    // sits BETWEEN b0 and b2 in block-index order but never references V, so V is dead across it.
    const b0 = try func.appendBlock();
    const b1 = try func.appendBlock();
    const b2 = try func.appendBlock();
    const v = try func.appendBlockParam(b0, t);
    const cond = try func.appendBlockParam(b0, bool_t);
    _ = try func.appendInst(b0, t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = v } });
    try func.appendIf(b0, cond, .{ .target = b2 }, .{ .target = b1 });

    const zero = try func.appendInst(b1, t, .{ .iconst = 0 });
    func.setTerminator(b1, .{ .ret = zero });

    const use2 = try func.appendInst(b2, t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = v } });
    func.setTerminator(b2, .{ .ret = use2 });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    const intervals = try wimmer.buildIntervals(allocator, &func, &desc);
    defer wimmer.freeIntervals(allocator, intervals);

    const iv = findValueInterval(intervals, v) orelse return error.MissingInterval;
    // At least two disjoint ranges: [def..end of b0) and [start of b2..use2), with a hole over b1.
    try std.testing.expect(iv.ranges.len >= 2);
    // The hole means there is a position the interval does not cover between its two live regions.
    try std.testing.expect(iv.ranges[0].to < iv.ranges[1].from);
}

test "intervals: a loop-carried value stays live across the whole loop body" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const bool_t = try func.types.intern(.bool);

    // Counted loop summing 0..n. `n` is loop-invariant: defined in entry, tested every iteration,
    // so it must stay live contiguously across the loop header and body (no hole inside the loop).
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const acc = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bacc = try func.appendBlockParam(body, t);
    const racc = try func.appendBlockParam(done, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    const nacc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
    try func.setJump(body, loop, &.{ ni, nacc });
    func.setTerminator(done, .{ .ret = racc });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    const intervals = try wimmer.buildIntervals(allocator, &func, &desc);
    defer wimmer.freeIntervals(allocator, intervals);

    const iv = findValueInterval(intervals, n) orelse return error.MissingInterval;
    // The fixpoint liveness carries `n` live-out of entry, loop, and body, so its ranges merge into
    // one contiguous range with no hole across the loop body.
    try std.testing.expectEqual(@as(usize, 1), iv.ranges.len);
    try std.testing.expectEqual(@as(u32, 0), iv.start());
    // A single contiguous range covers its whole span with no hole across the loop body.
    try std.testing.expect(iv.covers(iv.start()));
    try std.testing.expect(iv.covers(iv.end() - 1));
}

test "intervals: a call clobbers every caller-saved gpr via a fixed interval at the call position" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);

    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const called = try func.appendCall(b, t, "callee", &.{x});
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);
    try std.testing.expect(desc.call_sites.len >= 1);
    const call_pos = desc.call_sites[0].pos;

    const intervals = try wimmer.buildIntervals(allocator, &func, &desc);
    defer wimmer.freeIntervals(allocator, intervals);

    // x0 (caller-saved gpr, index 0) is blocked by a value-less fixed interval over the call.
    const fx = findFixed(intervals, 0, 0) orelse return error.MissingFixedInterval;
    try std.testing.expect(fx.value == null);
    try std.testing.expect(fx.covers(call_pos));
}

test "intervals: use positions carry must_have_register on aarch64" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);

    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    const intervals = try wimmer.buildIntervals(allocator, &func, &desc);
    defer wimmer.freeIntervals(allocator, intervals);

    const iv = findValueInterval(intervals, x) orelse return error.MissingInterval;
    try std.testing.expect(iv.uses.len >= 1);
    for (iv.uses) |u| try std.testing.expectEqual(wimmer.UseKind.must_have_register, u.kind);
}

// ---------------------------------------------------------------------------
// Task 3: the linear scan skeleton + tryAllocateFreeReg (no splitting yet).
// These run `allocate` end to end on aarch64 register descriptions.
// ---------------------------------------------------------------------------

/// The single segment reg index assigned to value `v`, or null if `v` has no register segment.
fn soleSegmentReg(alloc: *const wimmer.Allocation, v: Value) ?u16 {
    const segs = alloc.segments.get(v) orelse return null;
    if (segs.len != 1) return null;
    return switch (segs[0].loc) {
        .reg => |r| r,
        .slot => null,
    };
}

/// The segment active at `pos` in `segs` (ascending by `from`). Programmer error if `pos` precedes
/// the first segment or `segs` is empty.
fn segmentAt(segs: []const wimmer.Segment, pos: u32) wimmer.Segment {
    std.debug.assert(segs.len > 0);
    std.debug.assert(segs[0].from <= pos);
    var chosen = segs[0];
    for (segs) |s| {
        if (s.from <= pos) chosen = s;
    }
    return chosen;
}

/// Build the 12-live-value integer pressure kernel: twelve constants all kept live to a trailing
/// reduction. This exceeds the ten-register leaf gpr pool, so the scan must split and spill.
fn buildPressureKernel(func: *Function, t: ir.types.Type) !Value {
    const b = try func.appendBlock();
    var consts: [12]Value = undefined;
    for (&consts, 0..) |*c, idx| c.* = try func.appendInst(b, t, .{ .iconst = @intCast(idx) });
    var acc = consts[0];
    for (consts[1..]) |c| acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = c } });
    func.setTerminator(b, .{ .ret = acc });
    return acc;
}

test "scan: a low-pressure straight-line function assigns every value a register with a single segment" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // Every live value lands in exactly one register segment.
    for ([_]Value{ x, y, prod, sum }) |v| {
        const segs = alloc.segments.get(v) orelse return error.MissingSegment;
        try std.testing.expectEqual(@as(usize, 1), segs.len);
        try std.testing.expect(segs[0].loc == .reg);
    }

    // A low-pressure leaf spills nothing, so every class slot count is zero.
    try std.testing.expectEqual(desc.classes.len, alloc.slot_count_per_class.len);
    for (alloc.slot_count_per_class) |c| try std.testing.expectEqual(@as(u32, 0), c);
}

test "scan: an entry parameter is assigned its ABI argument register via the hint" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // The first integer parameter arrives in x0 and the hint keeps it there.
    try std.testing.expectEqual(@as(?u16, 0), soleSegmentReg(&alloc, x));
}

test "scan: a value live across a call gets a callee-saved register (not a caller-saved one)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    // `pre` is defined before the call and read after it, so it lives across the call.
    const pre = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    const called = try func.appendCall(b, t, "callee", &.{x});
    const after = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = pre, .rhs = called } });
    func.setTerminator(b, .{ .ret = after });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    const reg = soleSegmentReg(&alloc, pre) orelse return error.MissingSegment;
    // A caller-saved gpr would be clobbered by the call, so `pre` must land in x19..x28.
    try std.testing.expect(contains(desc.classes[0].callee_saved, reg));
}

// ---------------------------------------------------------------------------
// Task 4: register pressure handling (allocateBlockedReg + splitting + spill).
// These exercise `allocate` on functions whose pressure forces a split.
// ---------------------------------------------------------------------------

test "scan: register pressure splits an interval into a reg segment then a slot segment" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    _ = try buildPressureKernel(&func, t);

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // The pressure exceeds the pool, so `allocate` no longer bails: it splits and spills.
    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // At least one gpr spill slot was handed out.
    try std.testing.expect(alloc.slot_count_per_class[0] > 0);

    // Some value now lives in a register for one segment and a spill slot for another.
    var found = false;
    var it = alloc.segments.iterator();
    while (it.next()) |e| {
        const segs = e.value_ptr.*;
        if (segs.len < 2) continue;
        var has_slot = false;
        var has_reg = false;
        for (segs) |s| switch (s.loc) {
            .slot => has_slot = true,
            .reg => has_reg = true,
        };
        if (has_slot and has_reg) found = true;
    }
    try std.testing.expect(found);
}

test "scan: a must_have_register use is never left in a spilled segment" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    _ = try buildPressureKernel(&func, t);

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // Rebuild the intervals to recover every value's use positions and requirements.
    const intervals = try wimmer.buildIntervals(allocator, &func, &desc);
    defer wimmer.freeIntervals(allocator, intervals);

    // The key correctness invariant: at every must_have_register use, the value is in a register.
    for (intervals) |*iv| {
        if (iv.fixed_reg != null) continue;
        const v = iv.value orelse continue;
        const segs = alloc.segments.get(v) orelse continue;
        for (iv.uses) |u| {
            if (u.kind != .must_have_register) continue;
            const seg = segmentAt(segs, u.pos);
            try std.testing.expect(seg.loc == .reg);
        }
    }
}

test "scan: more same-class params than the register pool spills without crashing" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const b = try func.appendBlock();

    // Fourteen integer params all live simultaneously exceed the ten-register non-leaf gpr pool, so
    // more same-class values start at the block start than the pool can hold.
    const nparams = 14;
    var params: [nparams]Value = undefined;
    for (&params) |*p| p.* = try func.appendBlockParam(b, t);

    // A call makes the function non-leaf (pool = callee-saved x19..x28) and is the barrier the params
    // live across: every param is used AFTER it, so all fourteen are simultaneously live at once.
    const called = try func.appendCall(b, t, "callee", &.{params[0]});

    // Combine every param in REVERSE declaration order so the furthest-next-use victim the blocked
    // path picks is an EARLY-declared param whose interval starts at the block start. That is the
    // degenerate split-at-own-start (pos == victim.start()) that Finding 1 must not crash on.
    var acc = called;
    var idx: usize = nparams;
    while (idx > 0) {
        idx -= 1;
        acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = params[idx] } });
    }
    func.setTerminator(b, .{ .ret = acc });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // The allocation SUCCEEDS (no crash, no assert-abort): the pressure is resolved by spilling.
    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // Something spilled to a gpr slot: fourteen values cannot all fit in ten registers.
    try std.testing.expect(alloc.slot_count_per_class[0] > 0);

    // The must_have_register invariant still holds: at every must_have use the value is in a register.
    const intervals = try wimmer.buildIntervals(allocator, &func, &desc);
    defer wimmer.freeIntervals(allocator, intervals);
    for (intervals) |*iv| {
        if (iv.fixed_reg != null) continue;
        const v = iv.value orelse continue;
        const segs = alloc.segments.get(v) orelse continue;
        for (iv.uses) |u| {
            if (u.kind != .must_have_register) continue;
            const seg = segmentAt(segs, u.pos);
            try std.testing.expect(seg.loc == .reg);
        }
    }
}

test "scan: a float value live across a call is split at the call (the vector-quirk clobber forces it out of fp regs)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, f);
    // `pre` is a float defined before the call and read after it, so it lives across the call. The
    // vector-across-call quirk clobbers every fp register, so it cannot stay in one across the call.
    const pre = try func.appendInst(b, f, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    const called = try func.appendCall(b, f, "callee", &.{x});
    const after = try func.appendInst(b, f, .{ .arith = .{ .op = .add, .lhs = pre, .rhs = called } });
    func.setTerminator(b, .{ .ret = after });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);
    try std.testing.expect(desc.call_sites.len >= 1);
    const call_pos = desc.call_sites[0].pos;

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    const segs = alloc.segments.get(pre) orelse return error.MissingSegment;
    // `pre` was split around the call: it has more than one segment.
    try std.testing.expect(segs.len >= 2);
    // At the call it is NOT in a callee-saved fp register (v8..v15): the clobber forced it out. This
    // proves the fixed call-clobber interval was VISIBLE when `pre` was allocated (CHANGE 1).
    const at_call = segmentAt(segs, call_pos);
    const in_fp_reg = switch (at_call.loc) {
        .reg => |r| r >= 8 and r <= 15,
        .slot => false,
    };
    try std.testing.expect(!in_fp_reg);
}

// ---------------------------------------------------------------------------
// Task 7: RESOLVEDATAFLOW (edge_moves) + parallel-move ordering. The ordering is
// unit-tested directly through `wimmer.orderMoves` (deterministic), and the edge
// computation is tested end to end through `wimmer.allocate`.
// ---------------------------------------------------------------------------

const Move = wimmer.Move;
const Location = wimmer.Location;

/// True iff two locations are the same register or the same slot.
fn locEq(a: Location, b: Location) bool {
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

/// A u64 key for a location, disjoint across the reg/slot spaces (mirrors the allocator's `locKey`).
fn keyOf(loc: Location) u64 {
    return switch (loc) {
        .reg => |r| r,
        .slot => |s| (@as(u64, 1) << 32) | s,
    };
}

/// Simulate an ordered primitive move list and assert it realizes the raw parallel move `raw` for
/// `class`: every raw destination must end holding the value originally at its source. This is the
/// ground-truth validity check for the ordering (cycles, scratch routing, and all).
fn expectRealizesMoves(allocator: std.mem.Allocator, raw: []const Move, ordered: []const Move, class: u16) !void {
    var content: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    defer content.deinit(allocator);
    for (raw) |m| {
        if (m.class != class) continue;
        try content.put(allocator, keyOf(m.src), keyOf(m.src));
    }
    for (ordered) |m| {
        const v = content.get(keyOf(m.src)) orelse return error.ReadUninitializedLocation;
        try content.put(allocator, keyOf(m.dst), v);
    }
    for (raw) |m| {
        if (m.class != class) continue;
        const got = content.get(keyOf(m.dst)) orelse return error.DestinationNeverWritten;
        try std.testing.expectEqual(keyOf(m.src), got);
    }
}

test "resolve: a two-register cycle is ordered via the class scratch and realizes the swap" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // A minimal function just to build a real aarch64 RegDescription (its scratch/classes drive the
    // ordering). The moves below are hand-crafted, not derived from this function.
    const t = try func.types.intern(i32k);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    func.setTerminator(b, .{ .ret = x });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);
    const scratch0 = desc.scratch[0];

    // A pure two-register swap: x9 <- x10 and x10 <- x9 (class 0 = gpr). No ordering of these two
    // moves alone is correct, so the class scratch must break the cycle. `value` is unused by the
    // ordering (only a width-aware backend reads it), so any real value serves.
    const raw = [_]Move{
        .{ .src = .{ .reg = 9 }, .dst = .{ .reg = 10 }, .class = 0, .value = x },
        .{ .src = .{ .reg = 10 }, .dst = .{ .reg = 9 }, .class = 0, .value = x },
    };

    const ordered = try wimmer.orderMoves(allocator, &raw, &desc);
    defer allocator.free(ordered);

    // The swap cannot be done in two moves without a temporary, so the scratch adds a third.
    try std.testing.expectEqual(@as(usize, 3), ordered.len);
    // The scratch register is both written (saving the cycle value) and read (restoring it).
    var writes_scratch = false;
    var reads_scratch = false;
    for (ordered) |m| {
        if (locEq(m.dst, .{ .reg = scratch0 })) writes_scratch = true;
        if (locEq(m.src, .{ .reg = scratch0 })) reads_scratch = true;
    }
    try std.testing.expect(writes_scratch);
    try std.testing.expect(reads_scratch);
    // And applying the ordered sequence performs the swap.
    try expectRealizesMoves(allocator, &raw, ordered, 0);
}

test "resolve: a slot-to-slot move is routed through the class scratch" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    func.setTerminator(b, .{ .ret = x });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);
    const scratch0 = desc.scratch[0];

    // A single slot->slot move: no target moves memory to memory in one op, so it expands into a load
    // into the scratch then a store out of it.
    const raw = [_]Move{.{ .src = .{ .slot = 0 }, .dst = .{ .slot = 1 }, .class = 0, .value = x }};

    const ordered = try wimmer.orderMoves(allocator, &raw, &desc);
    defer allocator.free(ordered);

    try std.testing.expectEqual(@as(usize, 2), ordered.len);
    // First a load slot0 -> scratch, then a store scratch -> slot1.
    try std.testing.expect(locEq(ordered[0].src, .{ .slot = 0 }));
    try std.testing.expect(locEq(ordered[0].dst, .{ .reg = scratch0 }));
    try std.testing.expect(locEq(ordered[1].src, .{ .reg = scratch0 }));
    try std.testing.expect(locEq(ordered[1].dst, .{ .slot = 1 }));
    try expectRealizesMoves(allocator, &raw, ordered, 0);
}

test "resolve: independent reg and slot moves pass through as single ops in a valid order" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    func.setTerminator(b, .{ .ret = x });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // A conflict chain (not a cycle): x9 must be read before x10 overwrites it, plus a store and a
    // reload. The ordering must read x9 first (into x10) before writing x9 from x11.
    const raw = [_]Move{
        .{ .src = .{ .reg = 9 }, .dst = .{ .reg = 10 }, .class = 0, .value = x },
        .{ .src = .{ .reg = 11 }, .dst = .{ .reg = 9 }, .class = 0, .value = x },
        .{ .src = .{ .reg = 12 }, .dst = .{ .slot = 3 }, .class = 0, .value = x },
        .{ .src = .{ .slot = 4 }, .dst = .{ .reg = 13 }, .class = 0, .value = x },
    };

    const ordered = try wimmer.orderMoves(allocator, &raw, &desc);
    defer allocator.free(ordered);

    // No cycle, so no scratch is introduced: exactly four ops.
    try std.testing.expectEqual(@as(usize, 4), ordered.len);
    try expectRealizesMoves(allocator, &raw, ordered, 0);
}

test "resolve: a store and a reload on the same register at one position order store-first (gap #6)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    func.setTerminator(b, .{ .ret = x });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // The exact same-position drain hazard the retired aarch64 `hasSamePosRegHazard` guarded against:
    // one value is RELOADED into x9 while a DIFFERENT value is STORED out of x9, both at one position.
    // A fixed drain order (reload before store) would reload x9 first, clobbering the value the store
    // still has to save. The parallel-move ordering must therefore emit the STORE first (its source x9
    // is read by the reload's destination, so the reload is blocked until x9 is free), then the reload.
    // This is the primitive `wimmer.orderIntraActions` runs on every intra-block same-position cluster.
    const raw = [_]Move{
        .{ .src = .{ .slot = 0 }, .dst = .{ .reg = 9 }, .class = 0, .value = x }, // reload into x9
        .{ .src = .{ .reg = 9 }, .dst = .{ .slot = 1 }, .class = 0, .value = x }, // store out of x9
    };

    const ordered = try wimmer.orderMoves(allocator, &raw, &desc);
    defer allocator.free(ordered);

    // No cycle (a slot is never both a source and a destination), so no scratch: exactly two ops, the
    // store strictly before the reload.
    try std.testing.expectEqual(@as(usize, 2), ordered.len);
    try std.testing.expect(locEq(ordered[0].src, .{ .reg = 9 }) and locEq(ordered[0].dst, .{ .slot = 1 }));
    try std.testing.expect(locEq(ordered[1].src, .{ .slot = 0 }) and locEq(ordered[1].dst, .{ .reg = 9 }));
    try expectRealizesMoves(allocator, &raw, ordered, 0);
}

test "resolve: a block-param arg in a different register than the param becomes an edge move" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);

    // entry(x): jump next(x) ; next(p): ret p+p. `x` pins x0 (ABI arg reg). `p` is a fresh param, so
    // it lands in a pool register other than x0. The entry->next edge must move x0 into p's register.
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const next = try func.appendBlock();
    const p = try func.appendBlockParam(next, t);
    try func.setJump(entry, next, &.{x});
    const sum = try func.appendInst(next, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = p } });
    func.setTerminator(next, .{ .ret = sum });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // x sits in x0, p in some other register.
    const x_reg = soleSegmentReg(&alloc, x) orelse return error.MissingSegment;
    const p_reg = soleSegmentReg(&alloc, p) orelse return error.MissingSegment;
    try std.testing.expectEqual(@as(u16, 0), x_reg);
    try std.testing.expect(p_reg != x_reg);

    // Exactly one edge (entry -> next) carries a single move x0 -> p_reg.
    try std.testing.expectEqual(@as(usize, 1), alloc.edge_moves.len);
    const em = alloc.edge_moves[0];
    try std.testing.expectEqual(entry, em.pred);
    try std.testing.expectEqual(next, em.succ);
    try std.testing.expectEqual(@as(usize, 1), em.moves.len);
    try std.testing.expect(locEq(em.moves[0].src, .{ .reg = x_reg }));
    try std.testing.expect(locEq(em.moves[0].dst, .{ .reg = p_reg }));
}

test "resolve: an argument feeding a spilled successor parameter becomes a reg-to-slot store move" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);

    // entry(seed): jump body(seed x14). body(p0..p13): a call makes the function non-leaf (pool =
    // callee-saved x19..x28, ten registers), and all fourteen params are live across it, so several
    // MUST spill to slots at body's parameter row. Every argument is `seed`, which sits in x0 at
    // entry's exit, so each move into a spilled param is a reg->slot store on the entry->body edge.
    const entry = try func.appendBlock();
    const seed = try func.appendBlockParam(entry, t);
    const body = try func.appendBlock();

    const nparams = 14;
    var params: [nparams]Value = undefined;
    for (&params) |*p| p.* = try func.appendBlockParam(body, t);
    const args = [_]Value{seed} ** nparams;
    try func.setJump(entry, body, &args);

    const called = try func.appendCall(body, t, "callee", &.{params[0]});
    var acc = called;
    for (params) |p| acc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
    func.setTerminator(body, .{ .ret = acc });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // Every edge's move list is a valid emission order (no destination overwritten before it is read).
    try expectAllEdgesValid(allocator, &alloc);

    // Some param spilled to a slot, so its edge move is a reg->slot store.
    try std.testing.expect(alloc.slot_count_per_class[0] > 0);
    var found_store = false;
    for (alloc.edge_moves) |em| {
        if (em.pred != entry or em.succ != body) continue;
        for (em.moves) |m| {
            const is_store = m.src == .reg and m.dst == .slot;
            if (is_store) found_store = true;
        }
    }
    try std.testing.expect(found_store);
}

/// Assert every edge's ordered move list is a valid emission sequence: replaying it never reads a
/// location after an earlier op in the same list overwrote it (the class scratch, written before it
/// is read, is exempt as the intended temporary).
fn expectAllEdgesValid(allocator: std.mem.Allocator, alloc: *const wimmer.Allocation) !void {
    for (alloc.edge_moves) |em| {
        var written: std.AutoHashMapUnmanaged(u64, void) = .empty;
        defer written.deinit(allocator);
        for (em.moves) |m| {
            // Reading a location an earlier op already overwrote would be a miscompile.
            try std.testing.expect(!written.contains(keyOf(m.src)));
            try written.put(allocator, keyOf(m.dst), {});
        }
    }
}

test "buildAllocation: a value split mid-successor-block gets an intra-block action, not a dropped edge move" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);

    // entry(seed): v = seed + seed ; jump body. `v` is defined in entry, live-in to body, and used only
    // at the VERY END of body. Twelve fresh constants in body create register pressure that exceeds the
    // leaf gpr pool, and `v`'s next use is the furthest, so the blocked-register path evicts `v` MID-body:
    // its register is stolen at a mid-block position `p` and `v` is stored to a slot there. That store is
    // an INTRA-block action at `p`, NOT a cross-block edge move (the edge sees `v` in a register on both
    // sides). Before the predicate fix, `p`'s block differs from the first segment's block, so the store
    // was misclassified as cross-block and silently dropped: a miscompile.
    const entry = try func.appendBlock();
    const seed = try func.appendBlockParam(entry, t);
    const v = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = seed, .rhs = seed } });
    const body = try func.appendBlock();
    try func.setJump(entry, body, &.{});

    var consts: [12]Value = undefined;
    for (&consts, 0..) |*c, idx| c.* = try func.appendInst(body, t, .{ .iconst = @intCast(idx) });
    var acc = consts[0];
    for (consts[1..]) |c| acc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = c } });
    // `v`'s single body use is last, giving it the furthest next use so the blocked path evicts it.
    acc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(body, .{ .ret = acc });

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // `entry` is block 0: param row (pos 0), `v` (pos 1), terminator (pos 2), so body starts at pos 3.
    const body_start: u32 = 3;

    // Find `v`'s reg->slot transition. It must land strictly inside body (a mid-block spill), and an
    // action realizing that store must exist at that position.
    const segs = alloc.segments.get(v) orelse return error.MissingSegment;
    var store_at: ?u32 = null;
    var i: usize = 0;
    while (i + 1 < segs.len) : (i += 1) {
        const is_store = segs[i].loc == .reg and segs[i + 1].loc == .slot;
        if (is_store) {
            store_at = segs[i + 1].from;
            break;
        }
    }
    const at = store_at orelse return error.NoMidBlockSpill;
    // The spill is genuinely mid-body (not on body's parameter/start row), so it is an intra-block action.
    try std.testing.expect(at > body_start);

    // The store must be realized by an intra-block action at `at`. Before the fix this action was dropped
    // (the transition was misclassified cross-block), so `v` would be read from the wrong place after `p`.
    var found_action = false;
    for (alloc.actions) |action| {
        if (action.at == at and action.kind == .store) found_action = true;
    }
    try std.testing.expect(found_action);
}

test "resolve: a split CFG (diamond) allocates with consistent, valid edge moves and no critical-edge trip" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const bool_t = try func.types.intern(.bool);

    // entry(x) -[if]-> a, b ; a -> m(x) ; b -> m(x+1) ; m(r): ret r. The two edges into m come from
    // single-successor jumps, so no edge is critical; splitCriticalEdges leaves it unchanged. Both
    // arms carry an argument into m, exercising cross-block param moves.
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const a = try func.appendBlock();
    const bb = try func.appendBlock();
    const m = try func.appendBlock();
    const r = try func.appendBlockParam(m, t);
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = x } });
    try func.appendIf(entry, cond, .{ .target = a }, .{ .target = bb });
    try func.setJump(a, m, &.{x});
    const xp1 = try func.appendArithImm(bb, t, .add, x, 1);
    try func.setJump(bb, m, &.{xp1});
    func.setTerminator(m, .{ .ret = r });

    // Task-8 wiring splits critical edges before allocation; here there are none, but run it anyway.
    try ir.critical_edge.splitCriticalEdges(allocator, &func);

    var desc = try aarch64.aarch64RegDescription(allocator, &func);
    defer desc.deinit(allocator);

    // allocate succeeds and its no-critical-edge assertion does not trip.
    var alloc = try wimmer.allocate(allocator, &func, &desc);
    defer alloc.deinit(allocator);

    // Every produced edge move list is internally valid.
    try expectAllEdgesValid(allocator, &alloc);
}

// ---------------------------------------------------------------------------
// Task 9: the debug verifier (verifyIntervals). White-box tests build interval
// sets directly (with owned ranges/uses) and check the three soundness
// properties: register exclusivity, must_have_register satisfaction, and
// assignment. The verifier also runs inside every test-build `allocate`, so the
// whole suite above exercises it on real allocations.
// ---------------------------------------------------------------------------

/// Build an `Interval` whose `ranges`/`uses` are freshly owned copies of `ranges`/`uses`, so the
/// verifier (which never frees its input) can be handed a hand-built set that the test then releases
/// with `freeOwnedInterval`.
fn ownedInterval(
    allocator: std.mem.Allocator,
    value: ?Value,
    class: u16,
    fixed_reg: ?u16,
    ranges: []const wimmer.Range,
    uses: []const wimmer.UsePos,
    location: ?Location,
) !wimmer.Interval {
    return .{
        .value = value,
        .class = class,
        .fixed_reg = fixed_reg,
        .ranges = try allocator.dupe(wimmer.Range, ranges),
        .uses = try allocator.dupe(wimmer.UsePos, uses),
        .location = location,
    };
}

/// Free the owned `ranges`/`uses` of an interval built by `ownedInterval`.
fn freeOwnedInterval(allocator: std.mem.Allocator, iv: wimmer.Interval) void {
    allocator.free(iv.ranges);
    allocator.free(iv.uses);
}

test "verify: a correct allocation has no violations" {
    const allocator = std.testing.allocator;
    const v0: Value = @enumFromInt(0);
    const v1: Value = @enumFromInt(1);
    // Two same-class values in DIFFERENT registers over the same range, one with a must_have use it
    // covers in a register. Nothing conflicts, nothing is spilled, both are assigned.
    var ivs = [_]wimmer.Interval{
        try ownedInterval(allocator, v0, 0, null, &.{.{ .from = 0, .to = 4 }}, &.{.{ .pos = 2, .kind = .must_have_register }}, .{ .reg = 3 }),
        try ownedInterval(allocator, v1, 0, null, &.{.{ .from = 0, .to = 4 }}, &.{}, .{ .reg = 5 }),
    };
    defer for (ivs) |iv| freeOwnedInterval(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 0), violations.len);
}

test "verify: two value intervals sharing a register over overlapping ranges is flagged" {
    const allocator = std.testing.allocator;
    const v0: Value = @enumFromInt(0);
    const v1: Value = @enumFromInt(1);
    // Both values are placed in x5 (class 0) with ranges that overlap at [2, 4): the core soundness
    // violation.
    var ivs = [_]wimmer.Interval{
        try ownedInterval(allocator, v0, 0, null, &.{.{ .from = 0, .to = 4 }}, &.{}, .{ .reg = 5 }),
        try ownedInterval(allocator, v1, 0, null, &.{.{ .from = 2, .to = 6 }}, &.{}, .{ .reg = 5 }),
    };
    defer for (ivs) |iv| freeOwnedInterval(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 1), violations.len);
    try std.testing.expect(violations[0].kind == .reg_overlap);
    try std.testing.expectEqual(@as(u32, 2), violations[0].pos);
}

test "verify: a must_have_register use covered only by a slot interval is flagged" {
    const allocator = std.testing.allocator;
    const v0: Value = @enumFromInt(0);
    // The only interval of v0 lives in a slot, yet carries a must_have_register use at pos 2. No
    // register-located interval of v0 covers pos 2, so the use is spilled where it may not be.
    var ivs = [_]wimmer.Interval{
        try ownedInterval(allocator, v0, 0, null, &.{.{ .from = 0, .to = 4 }}, &.{.{ .pos = 2, .kind = .must_have_register }}, .{ .slot = 0 }),
    };
    defer for (ivs) |iv| freeOwnedInterval(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 1), violations.len);
    try std.testing.expect(violations[0].kind == .must_have_spilled);
    try std.testing.expectEqual(@as(u32, 2), violations[0].pos);
}

test "verify: the entry-param fixed interval sharing its own param's register is NOT flagged" {
    const allocator = std.testing.allocator;
    const p: Value = @enumFromInt(0);
    // The legitimate [0, 1) overlap: a parameter's value interval in x0 and that same param's
    // entry-param fixed interval pinning x0 at entry. Same value, same reg, but not a real conflict.
    var ivs = [_]wimmer.Interval{
        try ownedInterval(allocator, p, 0, null, &.{.{ .from = 0, .to = 4 }}, &.{.{ .pos = 2, .kind = .must_have_register }}, .{ .reg = 0 }),
        try ownedInterval(allocator, p, 0, 0, &.{.{ .from = 0, .to = 1 }}, &.{}, .{ .reg = 0 }),
    };
    defer for (ivs) |iv| freeOwnedInterval(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 0), violations.len);
}

test "verify: a used value interval with no location is flagged unassigned" {
    const allocator = std.testing.allocator;
    const v0: Value = @enumFromInt(0);
    // A value with a real use that the scan never placed (location stays null) is an unassigned bug.
    var ivs = [_]wimmer.Interval{
        try ownedInterval(allocator, v0, 0, null, &.{.{ .from = 0, .to = 4 }}, &.{.{ .pos = 2, .kind = .should_have_register }}, null),
    };
    defer for (ivs) |iv| freeOwnedInterval(allocator, iv);

    const violations = try wimmer.verifyIntervals(allocator, &ivs);
    defer allocator.free(violations);
    try std.testing.expectEqual(@as(usize, 1), violations.len);
    try std.testing.expect(violations[0].kind == .unassigned);
}
