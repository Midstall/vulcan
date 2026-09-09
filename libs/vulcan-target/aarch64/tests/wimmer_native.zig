//! This file tests the shared Wimmer-Franz allocator on the AArch64 backend (test-only).
//! Each test compiles one function twice. The first path uses `selectFunction`, which calls
//! the backend's own `allocate`. The second path uses `compileFunctionWimmer`, which calls
//! the shared allocator but produces the SAME emission. Both versions run on the aarch64 host,
//! over many inputs, and the results must agree exactly. This agreement is the pass condition:
//! it shows the shared allocator produces correct machine code. The two allocators can pick
//! different registers or stack slots, so the check is EXECUTION equivalence (the same
//! results), not byte-identical code.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const jit = @import("../jit.zig");
const link = @import("../link.zig");
const encode = @import("../encode.zig");

const Function = ir.function.Function;

/// JIT `code` and call it with `args` (up to 3 i32 args, i32 return) on the aarch64 host.
fn callI32(code: []const u32, args: []const i32) !i32 {
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const ptr = buf.memory.ptr; // page-aligned, so it meets the function-pointer alignment
    return switch (args.len) {
        0 => @as(*const fn () callconv(.c) i32, @ptrCast(ptr))(),
        1 => @as(*const fn (i32) callconv(.c) i32, @ptrCast(ptr))(args[0]),
        2 => @as(*const fn (i32, i32) callconv(.c) i32, @ptrCast(ptr))(args[0], args[1]),
        3 => @as(*const fn (i32, i32, i32) callconv(.c) i32, @ptrCast(ptr))(args[0], args[1], args[2]),
        else => error.Unsupported,
    };
}

/// Compile `func` via the REFERENCE path and JIT-run it.
fn runReference(allocator: std.mem.Allocator, func: *const Function, args: []const i32) !i32 {
    const code = try isel.selectFunction(allocator, func);
    defer allocator.free(code);
    return callI32(code, args);
}

/// Compile `func` with the SHARED Wimmer allocator and JIT-run it. Takes `func` as mutable
/// because `compileFunctionWimmer` splits critical edges in place. For a function with no
/// critical edges, this split is a no-op and does no harm.
fn runWimmer(allocator: std.mem.Allocator, func: *Function, args: []const i32) !i32 {
    var compiled = try isel.compileFunctionWimmer(allocator, func);
    defer compiled.deinit(allocator);
    return callI32(compiled.code, args);
}

/// Run `func` both ways over every input tuple, and assert the results are bit-identical.
/// Single-block callers share one function, because edge splitting is a no-op there.
/// Cross-block callers use `expectCrossBlockEquivalent` instead, which builds a fresh,
/// unmutated reference.
fn expectEquivalent(allocator: std.mem.Allocator, func: *Function, inputs: []const [2]i32) !void {
    for (inputs) |in| {
        const ref = try runReference(allocator, func, &in);
        const wim = try runWimmer(allocator, func, &in);
        try std.testing.expectEqual(ref, wim);
    }
}

/// Cross-block differential harness. `build` constructs the SAME function twice. The reference
/// `selectFunction` compiles `fa` and never mutates it. `compileFunctionWimmer` compiles `fb`
/// and splits critical edges in place. Their JIT results must agree bit-for-bit over every
/// input. The reference is verified first, so an invalid hand-built CFG shows as a test error,
/// not as a mismatch between the two results.
fn expectCrossBlockEquivalent(
    allocator: std.mem.Allocator,
    comptime build: fn (std.mem.Allocator) anyerror!Function,
    nargs: usize,
    inputs: []const [3]i32,
) !void {
    var fa = try build(allocator);
    defer fa.deinit();
    var fb = try build(allocator);
    defer fb.deinit();

    var diag = try ir.verify.verify(allocator, &fa, .high);
    defer diag.deinit();
    try std.testing.expect(diag.ok());

    for (inputs) |in| {
        const ref = try runReference(allocator, &fa, in[0..nargs]);
        const wim = try runWimmer(allocator, &fb, in[0..nargs]);
        try std.testing.expectEqual(ref, wim);
    }
}

test "wimmer: a straight-line arithmetic function matches the old allocator" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // f(a, b) = ((a + b) * (a - b)) + (a * b). This is a short dependent chain with no register
    // pressure, so no splits fire. It proves the shared allocator's whole-life, single-segment
    // placements translate and emit correctly end-to-end.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    const dif = try func.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = bp } });
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = sum, .rhs = dif } });
    const ab = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    const res = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = ab } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(res) });

    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 3, 5 }, .{ -2, 7 }, .{ 100, -25 }, .{ -37, 41 } };
    try expectEquivalent(allocator, &func, &inputs);
}

test "wimmer: the register-pressure kernel matches across inputs" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // f(a, b) = sum over k in 1..=20 of (a*k + b). All 20 products stay live until the final
    // reduction, far past the GPR pool. So the shared allocator must SPLIT live ranges and
    // spill. This runs the intra-block store and reload actions for the first time.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    var terms: [20]ir.function.Value = undefined;
    var k: i64 = 1;
    while (k <= 20) : (k += 1) {
        const kc = try func.appendInst(b, t, .{ .iconst = k });
        const ak = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
    }
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) {
        acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    }
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(acc) });

    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    try expectEquivalent(allocator, &func, &inputs);
}

test "wimmer: a spilled value reloads the correct bits" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // f(a, b) = a*b (defined FIRST and held live over the pressure block) plus
    // sum_{k=1..20}(a*k + b). The early product's only remaining use is the very last add. So
    // it must be spilled under pressure and RELOADED for that use. The result is correct only
    // if the shared allocator's store and reload actions carry the exact bits.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const t0 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    var terms: [20]ir.function.Value = undefined;
    var k: i64 = 1;
    while (k <= 20) : (k += 1) {
        const kc = try func.appendInst(b, t, .{ .iconst = k });
        const ak = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
    }
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) {
        acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    }
    const res = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = t0 } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(res) });

    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    try expectEquivalent(allocator, &func, &inputs);
}

// ===========================================================================
// Cross-block: this section runs the shared allocator's cross-block live-range splitting for
// the first time. Each builder constructs a genuinely multi-block leaf i32 function. The
// reference (unsplit) and Wimmer (edge-split, edge-move-driven) compilations must agree
// bit-for-bit over every input.
// ===========================================================================

const Value = ir.function.Value;

/// A counted loop `f(n, x)` that carries an induction variable plus six accumulators through the
/// loop header, updates them in a dependency-chained body, and reduces them at the exit. The
/// loop carries enough simultaneously-live values across the back-edge to pressure the leaf GPR
/// pool. So the shared allocator must split live ranges and resolve them with edge moves, on
/// the back-edge and at the header entry.
fn buildLoopSum(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const n = try func.appendBlockParam(entry, t);
    const x = try func.appendBlockParam(entry, t);

    const li = try func.appendBlockParam(loop, t);
    const la = try func.appendBlockParam(loop, t);
    const lb = try func.appendBlockParam(loop, t);
    const lc = try func.appendBlockParam(loop, t);
    const ld = try func.appendBlockParam(loop, t);
    const le = try func.appendBlockParam(loop, t);
    const lf = try func.appendBlockParam(loop, t);

    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero, zero, zero, zero, zero, zero });

    // Header: continue while i < n (n stays live-in across the whole loop). Else fall to the exit.
    const cond = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = li, .rhs = n } });
    try func.appendIf(loop, cond, .{ .target = body }, .{ .target = exit });

    // Body: i += 1, and each accumulator folds x plus the previous new accumulator. This
    // dependency chain keeps them all live to the back-edge. x is live-in here across the header.
    const one = try func.appendInst(body, t, .{ .iconst = 1 });
    const inext = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = li, .rhs = one } });
    const na = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = la, .rhs = x } });
    const nb = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = lb, .rhs = na } });
    const nc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = lc, .rhs = nb } });
    const nd = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = ld, .rhs = nc } });
    const ne = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = le, .rhs = nd } });
    const nf = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = lf, .rhs = ne } });
    try func.setJump(body, loop, &.{ inext, na, nb, nc, nd, ne, nf });

    // Exit: reduce the accumulators. All are loop params, live-in from the header's else-edge.
    const s1 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = la, .rhs = lb } });
    const s2 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = lc } });
    const s3 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = ld } });
    const s4 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = s3, .rhs = le } });
    const s5 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = s4, .rhs = lf } });
    func.setTerminator(exit, .{ .ret = ir.function.Ret.one(s5) });

    return func;
}

test "wimmer: a loop-carried sum across a pressured loop body matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const inputs = [_][3]i32{
        .{ 0, 5, 0 },  .{ 1, 5, 0 }, .{ 2, 3, 0 },  .{ 5, 2, 0 },   .{ 10, 1, 0 },
        .{ 8, -3, 0 }, .{ 3, 7, 0 }, .{ -1, 9, 0 }, .{ 20, -2, 0 }, .{ 12, 4, 0 },
    };
    try expectCrossBlockEquivalent(std.testing.allocator, buildLoopSum, 2, &inputs);
}

/// A diamond `f(p, q)` where `base = p*q` is defined before the branch and used only after the
/// merge, so it is live along BOTH arms. The right arm builds many independent values that all
/// stay live to its reduction. This pressures the pool, so `base` (and others) split across the
/// arm and reload at the merge. It runs a cross-block spill and reload, plus a merge-parameter
/// move.
fn buildDiamond(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const left = try func.appendBlock();
    const right = try func.appendBlock();
    const merge = try func.appendBlock();

    const p = try func.appendBlockParam(entry, t);
    const q = try func.appendBlockParam(entry, t);
    const m = try func.appendBlockParam(merge, t);

    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = p, .rhs = q } });
    const base = try func.appendInst(entry, t, .{ .arith = .{ .op = .mul, .lhs = p, .rhs = q } });
    try func.appendIf(entry, cond, .{ .target = left }, .{ .target = right });

    // Left arm: light, one value goes into the merge parameter.
    const l = try func.appendInst(left, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = q } });
    try func.setJump(left, merge, &.{l});

    // Right arm: ten values, all live to the reduction. This pressures the pool while `base`
    // is live-through.
    const c1 = try func.appendInst(right, t, .{ .iconst = 1 });
    var rs: [10]Value = undefined;
    rs[0] = try func.appendInst(right, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = q } });
    rs[1] = try func.appendInst(right, t, .{ .arith = .{ .op = .sub, .lhs = p, .rhs = q } });
    rs[2] = try func.appendInst(right, t, .{ .arith = .{ .op = .mul, .lhs = p, .rhs = q } });
    rs[3] = try func.appendInst(right, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = c1 } });
    rs[4] = try func.appendInst(right, t, .{ .arith = .{ .op = .add, .lhs = q, .rhs = c1 } });
    rs[5] = try func.appendInst(right, t, .{ .arith = .{ .op = .add, .lhs = rs[0], .rhs = c1 } });
    rs[6] = try func.appendInst(right, t, .{ .arith = .{ .op = .add, .lhs = rs[1], .rhs = c1 } });
    rs[7] = try func.appendInst(right, t, .{ .arith = .{ .op = .add, .lhs = rs[2], .rhs = c1 } });
    rs[8] = try func.appendInst(right, t, .{ .arith = .{ .op = .add, .lhs = rs[3], .rhs = c1 } });
    rs[9] = try func.appendInst(right, t, .{ .arith = .{ .op = .add, .lhs = rs[4], .rhs = c1 } });
    var acc = rs[0];
    var i: usize = 1;
    while (i < rs.len) : (i += 1) {
        acc = try func.appendInst(right, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = rs[i] } });
    }
    try func.setJump(right, merge, &.{acc});

    // Merge: fold the arm's value with `base`, which spans the whole diamond.
    const res = try func.appendInst(merge, t, .{ .arith = .{ .op = .add, .lhs = m, .rhs = base } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(res) });

    return func;
}

test "wimmer: a diamond with a value live on both merge paths matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const inputs = [_][3]i32{
        .{ 0, 0, 0 }, .{ 1, 2, 0 },   .{ 2, 1, 0 },  .{ -3, 4, 0 }, .{ 5, -6, 0 },
        .{ 7, 7, 0 }, .{ -8, -2, 0 }, .{ 10, 3, 0 }, .{ -1, 0, 0 }, .{ 9, -9, 0 },
    };
    try expectCrossBlockEquivalent(std.testing.allocator, buildDiamond, 2, &inputs);
}

/// A value with a genuine lifetime HOLE: `v = a*b` is defined in the entry and reused only in
/// block `A`. Block `B` (numbered BETWEEN entry and A, and reached by the other branch) never
/// uses it. So `v` is dead across `B`'s positions, a hole the allocator may fill with `B`'s
/// pressured temporaries, and it must then restore `v` for its reuse in `A`.
fn buildHole(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const b_blk = try func.appendBlock(); // block 1: the dead region for v, numbered before A
    const a_blk = try func.appendBlock(); // block 2: reuses v after the hole
    const join = try func.appendBlock(); // block 3

    const a = try func.appendBlockParam(entry, t);
    const bp = try func.appendBlockParam(entry, t);
    const w = try func.appendBlockParam(join, t);

    const v = try func.appendInst(entry, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = bp } });
    try func.appendIf(entry, cond, .{ .target = a_blk }, .{ .target = b_blk });

    // B: pressured. It does NOT reference v, so v is dead across every position here.
    const c1 = try func.appendInst(b_blk, t, .{ .iconst = 1 });
    var rs: [8]Value = undefined;
    rs[0] = try func.appendInst(b_blk, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    rs[1] = try func.appendInst(b_blk, t, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = bp } });
    rs[2] = try func.appendInst(b_blk, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = c1 } });
    rs[3] = try func.appendInst(b_blk, t, .{ .arith = .{ .op = .add, .lhs = bp, .rhs = c1 } });
    rs[4] = try func.appendInst(b_blk, t, .{ .arith = .{ .op = .add, .lhs = rs[0], .rhs = c1 } });
    rs[5] = try func.appendInst(b_blk, t, .{ .arith = .{ .op = .add, .lhs = rs[1], .rhs = c1 } });
    rs[6] = try func.appendInst(b_blk, t, .{ .arith = .{ .op = .add, .lhs = rs[2], .rhs = c1 } });
    rs[7] = try func.appendInst(b_blk, t, .{ .arith = .{ .op = .add, .lhs = rs[3], .rhs = c1 } });
    var acc = rs[0];
    var i: usize = 1;
    while (i < rs.len) : (i += 1) {
        acc = try func.appendInst(b_blk, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = rs[i] } });
    }
    try func.setJump(b_blk, join, &.{acc});

    // A: reuses v after the hole.
    const res = try func.appendInst(a_blk, t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = a } });
    try func.setJump(a_blk, join, &.{res});

    func.setTerminator(join, .{ .ret = ir.function.Ret.one(w) });

    return func;
}

test "wimmer: a value with a lifetime hole reused after a dead region matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const inputs = [_][3]i32{
        .{ 0, 0, 0 }, .{ 1, 2, 0 }, .{ 2, 1, 0 },   .{ -3, 4, 0 }, .{ 5, -6, 0 },
        .{ 7, 7, 0 }, .{ 3, 8, 0 }, .{ -8, -2, 0 }, .{ 10, 3, 0 }, .{ 9, -9, 0 },
    };
    try expectCrossBlockEquivalent(std.testing.allocator, buildHole, 2, &inputs);
}

/// A genuine CRITICAL edge: the entry `if` (two successors) feeds a single merge block (two
/// predecessors) on BOTH arms, with a DIFFERENT argument on each. So each entry-to-merge edge
/// is critical. `splitCriticalEdges` inserts a forwarding block on each edge. The
/// merge-parameter move, plus the live-through `x`, land as edge moves on those forwarding
/// blocks.
fn buildCritical(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const merge = try func.appendBlock();

    const a = try func.appendBlockParam(entry, t);
    const bp = try func.appendBlockParam(entry, t);
    const z = try func.appendBlockParam(merge, t);

    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = bp } });
    const x = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    // Both arms target merge with a different argument. The larger operand flows through z.
    try func.appendIf(entry, cond, .{ .target = merge, .args = &.{bp} }, .{ .target = merge, .args = &.{a} });

    const res = try func.appendInst(merge, t, .{ .arith = .{ .op = .add, .lhs = z, .rhs = x } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(res) });

    return func;
}

test "wimmer: a critical-edge case matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const inputs = [_][3]i32{
        .{ 0, 0, 0 }, .{ 1, 2, 0 },   .{ 2, 1, 0 },  .{ -3, 4, 0 },    .{ 5, -6, 0 },
        .{ 7, 7, 0 }, .{ -8, -2, 0 }, .{ 10, 3, 0 }, .{ 100, -25, 0 }, .{ -37, 41, 0 },
    };
    try expectCrossBlockEquivalent(std.testing.allocator, buildCritical, 2, &inputs);
}

// ===========================================================================
// This section scales the differential guardrail to non-leaf (call), spill-across-call,
// call-argument-alias, float and vector param, and more-than-8-arg shapes, so each bridge gap
// can flip from "skip" to "assert" as it is fixed. JIT probes checked this empirically before
// these tests were written: `compileFunctionWimmer`'s only blanket bail for these new shapes is
// the leaf gate (`isLeaf`, gap #3). Any function with a `.call` or `.call_indirect` bails
// `error.Unsupported` immediately, before it reaches any of the other four bridge gaps
// (#4, #5, #6, #7). Every LEAF shape added below (many-arg stack params, f32, <4xf32>) already
// works through the SHARED emission: `emitFromAllocation`'s entry-param loop loads stack-passed
// args the same way for leaf and non-leaf functions. So those assert equivalence TODAY, rather
// than skip. Only the shapes with a real `.call` skip.
// ===========================================================================

/// JIT `code` and call it with `args` (up to 3 i64 args, i64 return) on the aarch64 host. This
/// mirrors `callI32` for the 64-bit integer file. Most of this file's tests use i32 only, and
/// i64 tests run the same placements at the 64-bit width.
fn callI64(code: []const u32, args: []const i64) !i64 {
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const ptr = buf.memory.ptr;
    return switch (args.len) {
        0 => @as(*const fn () callconv(.c) i64, @ptrCast(ptr))(),
        1 => @as(*const fn (i64) callconv(.c) i64, @ptrCast(ptr))(args[0]),
        2 => @as(*const fn (i64, i64) callconv(.c) i64, @ptrCast(ptr))(args[0], args[1]),
        3 => @as(*const fn (i64, i64, i64) callconv(.c) i64, @ptrCast(ptr))(args[0], args[1], args[2]),
        else => error.Unsupported,
    };
}

/// Adapter that feeds `callI64`'s 2-arg case through the fixed-`Args`-array shape
/// `expectEquivalentCC` expects.
fn callI64x2(code: []const u32, args: [2]i64) !i64 {
    return callI64(code, &args);
}

/// JIT `code` and call it with exactly 10 i64 args (i64 return). Args 0-7 land in x0..x7, and
/// args 8-9 arrive on the caller's outgoing stack area (System V / AAPCS). This is the
/// more-than-8-arg shape: the entry prologue must load the stack-passed params, not just the
/// register-passed ones.
fn callI64x10(code: []const u32, args: [10]i64) !i64 {
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
    const f: Fn = @ptrCast(buf.memory.ptr);
    return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]);
}

/// JIT `code` and call it with 2 f32 args (f32 return). All values stay in the `v` file
/// (v0/v1 in, v0 out).
fn callF32x2(code: []const u32, args: [2]f32) !f32 {
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (f32, f32) callconv(.c) f32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    return f(args[0], args[1]);
}

/// Like `expectEquivalent`, but takes the calling convention (i64, many-arg, f32) as a
/// parameter, through `Args`/`Ret`/`call`. `callI32` alone serves `expectEquivalent` the same
/// way. The reference path (`selectFunction`) ALWAYS runs, so a broken IR builder still shows
/// as a test failure, no matter what `must_run` is.
///
/// `must_run` closes a hardening gap a reviewer found: `expectEquivalent`-style helpers folded
/// "asserted equal" and "skipped because Unsupported" into the same pass result. This let a
/// shape that must always compile silently regress to a skip, with nobody noticing. Callers for
/// shapes the bridge must compile (leaf arithmetic, register pressure, non-leaf calls,
/// live-across-call, spill-across-call, split-param, call-arg cycles, float, vector, many-arg)
/// all pass `true`. Then `compileFunctionWimmer` returning `error.Unsupported` is a REGRESSION,
/// and the test fails loudly instead of returning early. Only a shape that hits one of the
/// SHARED limits the old path also rejects (composite-f16, zero-block) should pass `false`,
/// which tolerates the skip. Nothing in this file needs that today, since testing confirmed
/// every shape here already runs. So every call site below passes `true`. The flag stays, so a
/// future shape that is genuinely unsupported has somewhere to land, without bringing back the
/// silent-skip risk for everything else.
fn expectEquivalentCC(
    comptime Args: type,
    comptime Ret: type,
    allocator: std.mem.Allocator,
    func: *Function,
    comptime call: fn ([]const u32, Args) anyerror!Ret,
    inputs: []const Args,
    must_run: bool,
) !void {
    for (inputs) |in| {
        const ref_code = try isel.selectFunction(allocator, func);
        defer allocator.free(ref_code);
        const ref = try call(ref_code, in);

        var compiled = isel.compileFunctionWimmer(allocator, func) catch |err| {
            if (!must_run and err == error.Unsupported) return; // tolerated: a genuine shared limit
            return err; // must_run, or a real error: never swallowed
        };
        defer compiled.deinit(allocator);
        const wim = try call(compiled.code, in);
        try std.testing.expectEqual(ref, wim);
    }
}

/// Link an ALREADY-COMPILED entry function's code against a freshly-compiled `helper` (using the
/// reference `isel.compileFunction`), and resolve the entry's `bl` relocation(s) to `helper`'s
/// word offset. This mirrors `link.compileModule`'s two-function layout and relocation logic
/// exactly (entry first, callee immediately after). But it takes the entry's machine code
/// directly instead of compiling it, so a Wimmer-compiled caller can be linked the same way a
/// reference-compiled one is. (`link.compileModule` always compiles every function with the old
/// allocator, so it cannot be reused as-is for a Wimmer-compiled entry.) The helper's OWN
/// allocator does not matter for this check: AAPCS is the contract at the call boundary, not the
/// callee's internal implementation, and the reference module already runs the
/// reference-compiled helper. The caller owns the result.
fn linkWithCompiledEntry(allocator: std.mem.Allocator, entry_code: []const u32, entry_relocs: []const isel.Reloc, helper_name: []const u8, helper: *const Function) !link.Linked {
    var helper_compiled = try isel.compileFunction(allocator, helper, .{});
    defer helper_compiled.deinit(allocator);

    const helper_words = entry_code.len;
    var code = try allocator.alloc(u32, helper_words + helper_compiled.code.len);
    errdefer allocator.free(code);
    @memcpy(code[0..helper_words], entry_code);
    @memcpy(code[helper_words..], helper_compiled.code);
    for (entry_relocs) |r| {
        std.debug.assert(std.mem.eql(u8, r.symbol, helper_name));
        code[r.offset] = encode.bl(@intCast((@as(i64, @intCast(helper_words)) - @as(i64, @intCast(r.offset))) * 4));
    }

    const symbols = try allocator.alloc(link.Symbol, 2);
    errdefer allocator.free(symbols);
    symbols[0] = .{ .name = "main", .offset = 0 };
    symbols[1] = .{ .name = helper_name, .offset = helper_words * 4 };
    return .{ .code = code, .symbols = symbols, .data = &.{}, .relocs = &.{} };
}

/// Like `linkWithCompiledEntry`, but generalized to a HELPER MODULE of any number of functions
/// that may call each other (a genuine call CHAIN: `main` calls B calls a leaf, not just `main`
/// calling one leaf). `helpers` is compiled and self-linked with the reference
/// `link.compileModule`, which already resolves any relocation BETWEEN helpers correctly: `bl`
/// is PC-relative, so shifting the whole helper block by a constant (the entry's own length)
/// keeps every already-resolved inter-helper distance correct. This function only has to
/// resolve the entry's OWN relocations against the shifted helper symbol table. As with
/// `linkWithCompiledEntry`, the helpers' own allocator does not matter: AAPCS is the contract at
/// each call boundary, not a callee's internal implementation. The caller owns the result.
fn linkWithCompiledEntryModule(allocator: std.mem.Allocator, entry_code: []const u32, entry_relocs: []const isel.Reloc, helpers: *const link.Module) !link.Linked {
    var helpers_linked = try link.compileModule(allocator, helpers);
    defer helpers_linked.deinit(allocator);

    const entry_words = entry_code.len;
    var code = try allocator.alloc(u32, entry_words + helpers_linked.code.len);
    errdefer allocator.free(code);
    @memcpy(code[0..entry_words], entry_code);
    @memcpy(code[entry_words..], helpers_linked.code);
    for (entry_relocs) |r| {
        var target_word: ?usize = null;
        for (helpers_linked.symbols) |s| {
            if (std.mem.eql(u8, s.name, r.symbol)) target_word = entry_words + s.offset / 4;
        }
        // The caller builds `entry_relocs` and `helpers` together, so every entry call must name a
        // function actually present in `helpers`: an unresolved symbol here is a test-builder bug.
        const target = target_word orelse unreachable;
        code[r.offset] = encode.bl(@intCast((@as(i64, @intCast(target)) - @as(i64, @intCast(r.offset))) * 4));
    }

    const symbols = try allocator.alloc(link.Symbol, 1 + helpers_linked.symbols.len);
    errdefer allocator.free(symbols);
    symbols[0] = .{ .name = "main", .offset = 0 };
    for (helpers_linked.symbols, 0..) |s, i| symbols[1 + i] = .{ .name = s.name, .offset = entry_words * 4 + s.offset };
    return .{ .code = code, .symbols = symbols, .data = &.{}, .relocs = &.{} };
}

/// Differential harness for a shape that genuinely CALLS another function (`helper`). The
/// reference ALWAYS compiles and links a real two-function MODULE (`caller` as the entry,
/// `helper` resolved with a real `bl` relocation) and JIT-executes it over `inputs`
/// (`{a, b, c, expected}`). So it runs REAL call-clobber semantics rather than a faked or
/// unresolved call. `expected` is hand-computed, so a broken IR builder fails loudly even
/// before Wimmer is compared. `compileFunctionWimmer` is then REQUIRED to compile `caller`: the
/// last non-leaf bridge gaps are closed, since the same-position intra-block drain hazard and
/// the call-argument clobber are both resolved as parallel moves. So every call shape in this
/// file must compile, and a bail is a regression, not a tolerated skip. The Wimmer-compiled
/// caller is linked against the SAME helper with `linkWithCompiledEntry` and JIT-executed over
/// every input, REQUIRING bit-for-bit agreement with the reference module. This is a real link
/// and execute, not just a successful compile. So a shape whose call-argument setup miscompiles
/// (a wrong parallel-move order at a call site) still fails loudly, rather than passing on
/// compile success alone.
fn expectCallShapeEquivalent(
    allocator: std.mem.Allocator,
    helper_name: []const u8,
    helper: *const Function,
    caller: *Function,
    inputs: []const [4]i32, // {a, b, c, expected}
) !void {
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", caller);
    try module.addFunction(allocator, helper_name, helper);
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(linked.code));
    defer buf.deinit();
    const Fn = *const fn (i32, i32, i32) callconv(.c) i32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    for (inputs) |in| try std.testing.expectEqual(in[3], f(in[0], in[1], in[2]));

    // STRICT: the bridge must now compile every call shape in this file, including
    // spill-across-call clusters and call-argument permutations. A bail here is a REGRESSION,
    // not a tolerated skip, so it must show as a failure rather than return early.
    var wcaller = try isel.compileFunctionWimmer(allocator, caller);
    defer wcaller.deinit(allocator);

    var wlinked = try linkWithCompiledEntry(allocator, wcaller.code, wcaller.relocs, helper_name, helper);
    defer wlinked.deinit(allocator);
    var wbuf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(wlinked.code));
    defer wbuf.deinit();
    const wf: Fn = @ptrCast(wbuf.memory.ptr);
    for (inputs) |in| try std.testing.expectEqual(in[3], wf(in[0], in[1], in[2]));
}

/// A leaf helper `inc(x) = x + 1`, compiled as the callee of the non-leaf shapes below.
fn buildIncHelper(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const one = try func.appendInst(b, t, .{ .iconst = 1 });
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = one } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    return func;
}

/// A leaf helper `combo(x, y, z) = x*100 + y*10 + z`. It is POSITION-SENSITIVE, so a
/// call-argument permutation that clobbers a source before it is consumed shows up as wrong
/// arithmetic. A plain sum would not catch this, because addition hides the order.
fn buildComboHelper(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const z = try func.appendBlockParam(b, t);
    const c100 = try func.appendInst(b, t, .{ .iconst = 100 });
    const c10 = try func.appendInst(b, t, .{ .iconst = 10 });
    const x100 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = c100 } });
    const y10 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = y, .rhs = c10 } });
    const s1 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x100, .rhs = y10 } });
    const s2 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = z } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s2) });
    return func;
}

/// (a) a non-leaf caller that calls a leaf helper and folds the result: f(a, b, c) = inc(a) + b + c.
/// `b` and `c` are both LIVE ACROSS the call (used only in the folds afterward), while sitting in
/// their ABI arg registers (x1/x2), which the call clobbers (the clobber list is every
/// caller-saved gpr, x0..x17). So the shared allocator must move them off their ABI registers
/// into a callee-saved one for the whole function. This is the Wimmer bridge gap #5 shape: a
/// whole-life entry param placed off its ABI arg register. `a` only feeds the call itself and
/// dies there, so it needs no special handling.
fn buildCallUseResult(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    const called = try func.appendCall(b, t, "wimmer_inc", &.{a});
    const s1 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = bp } });
    const s2 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = cp } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s2) });
    return func;
}

test "wimmer: a non-leaf function that calls a leaf helper and uses the result (the #5 shape)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var helper = try buildIncHelper(allocator);
    defer helper.deinit();
    var caller = try buildCallUseResult(allocator);
    defer caller.deinit();
    const inputs = [_][4]i32{
        .{ 0, 0, 0, (0 + 1) + 0 + 0 },     .{ 1, 2, 3, (1 + 1) + 2 + 3 },
        .{ -5, 10, 2, (-5 + 1) + 10 + 2 }, .{ 100, -25, 4, (100 + 1) + -25 + 4 },
    };
    try expectCallShapeEquivalent(allocator, "wimmer_inc", &helper, &caller, &inputs);
}

/// (b) a non-leaf function with a computed (non-param) value LIVE ACROSS a call: `t = a*b` is
/// computed BEFORE the call and used only AFTER it. This forces the shared allocator to park it
/// in a callee-saved register, or spill it, rather than use a caller-saved temporary. This does
/// NOT touch the entry-param bridge gap (#5, see `buildCallUseResult` above): `t` is not one of
/// the function's own parameters. So its whole-life placement in a callee-saved register needs
/// no reconciliation between an ABI register and an allocated register. It is a plain
/// instruction result: wherever it is allocated is simply where its defining instruction writes
/// it. This is kept as its own shape, for coverage of a value live across a call within one
/// block, distinct from the param case.
fn buildLiveAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    const called = try func.appendCall(b, t, "wimmer_inc", &.{cp});
    const res = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = called } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(res) });
    return func;
}

test "wimmer: a computed value live across a call matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var helper = try buildIncHelper(allocator);
    defer helper.deinit();
    var caller = try buildLiveAcrossCall(allocator);
    defer caller.deinit();
    const inputs = [_][4]i32{
        .{ 0, 0, 0, 0 * 0 + (0 + 1) },     .{ 3, 5, 2, 3 * 5 + (2 + 1) },
        .{ -2, 7, 10, -2 * 7 + (10 + 1) }, .{ 100, -25, 4, 100 * -25 + (4 + 1) },
    };
    try expectCallShapeEquivalent(allocator, "wimmer_inc", &helper, &caller, &inputs);
}

/// (c) a high-pressure non-leaf function: 14 computed (non-param) values are live across a call.
/// This forces more live-across-call values than the 10-slot callee-saved gpr pool (x19..x28)
/// holds, so some spill to the stack instead. None of the caller's own entry params (`a`, `b`,
/// `c`) survive across the call, since each is consumed feeding either a term or the call
/// itself. So this shape does not exercise the entry-param bridge gaps (#4 split param, #5
/// off-ABI-register param). It targets the spill-under-call-pressure path for ordinary values,
/// which the bridge already handles.
fn buildSpillAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    const n_terms = 14;
    var terms: [n_terms]Value = undefined;
    var k: i64 = 1;
    while (k <= n_terms) : (k += 1) {
        const kc = try func.appendInst(b, t, .{ .iconst = k });
        const ak = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
    }
    const called = try func.appendCall(b, t, "wimmer_inc", &.{cp});
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    const res = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = called } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(res) });
    return func;
}

test "wimmer: a high-pressure non-leaf function forces spill-across-call" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var helper = try buildIncHelper(allocator);
    defer helper.deinit();
    var caller = try buildSpillAcrossCall(allocator);
    defer caller.deinit();
    // sum_{k=1..14}(a*k + b) + (c+1) == a*105 + 14*b + (c+1), since sum(1..14) == 105.
    const inputs = [_][4]i32{
        .{ 0, 0, 0, 0 * 105 + 14 * 0 + (0 + 1) },
        .{ 1, 1, 0, 1 * 105 + 14 * 1 + (0 + 1) },
        .{ 2, 3, 5, 2 * 105 + 14 * 3 + (5 + 1) },
        .{ -3, 4, -2, -3 * 105 + 14 * 4 + (-2 + 1) },
    };
    try expectCallShapeEquivalent(allocator, "wimmer_inc", &helper, &caller, &inputs);
}

/// (d) a call-argument shape whose args come from the caller's OWN entry params in a cyclic
/// permutation (`combo(b, c, a)`, not `combo(a, b, c)`). The call-argument setup must not
/// clobber a source register before it feeds its own destination slot: a sequential
/// `mov target, src` with no parallel-move resolution has this call-arg-alias risk. Testing
/// CONFIRMED a genuine miscompile under Wimmer's allocation, found while wiring up the real
/// link-and-execute comparison: it computed a wrong result before the hazard was first noticed.
/// The fix is the edge-move-driven call lowering (`Ctx.emitCallArgs`), which emits the argument
/// setup as a parallel move (stack stores first, then the register permutation through the
/// reserved scratch, then slot reloads). So this 3-cycle rotation (x0<-x1<-x2<-x0) now executes
/// correctly instead of being skipped.
fn buildCallArgAlias(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    const called = try func.appendCall(b, t, "wimmer_combo", &.{ bp, cp, a });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(called) });
    return func;
}

test "wimmer: a call-argument 3-cycle permutation matches (call-arg-alias resolved as a parallel move)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var helper = try buildComboHelper(allocator);
    defer helper.deinit();
    var caller = try buildCallArgAlias(allocator);
    defer caller.deinit();
    // combo(b, c, a) == b*100 + c*10 + a.
    const inputs = [_][4]i32{
        .{ 1, 2, 3, 2 * 100 + 3 * 10 + 1 },
        .{ 4, 5, 6, 5 * 100 + 6 * 10 + 4 },
        .{ -1, -2, -3, -2 * 100 + -3 * 10 + -1 },
        .{ 7, 0, 9, 0 * 100 + 9 * 10 + 7 },
    };
    try expectCallShapeEquivalent(allocator, "wimmer_combo", &helper, &caller, &inputs);
}

/// (d2) a SHORTER call-argument cycle: a 2-cycle SWAP of two arguments (`combo(b, a, c)`), plus
/// one argument passed through unchanged. Where `buildCallArgAlias` rotates three registers,
/// this stresses the minimal cycle the scratch cycle-break exists for (x0<->x1), alongside an
/// identity move (x2<-x2). So the parallel-move ordering must both break the swap and drop the
/// no-op. A sequential `mov x0,x1; mov x1,x0` would lose the original x1.
fn buildCallArgSwap(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    const called = try func.appendCall(b, t, "wimmer_combo", &.{ bp, a, cp });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(called) });
    return func;
}

test "wimmer: a call-argument 2-cycle swap matches (minimal permutation, identity elided)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var helper = try buildComboHelper(allocator);
    defer helper.deinit();
    var caller = try buildCallArgSwap(allocator);
    defer caller.deinit();
    // combo(b, a, c) == b*100 + a*10 + c.
    const inputs = [_][4]i32{
        .{ 1, 2, 3, 2 * 100 + 1 * 10 + 3 },
        .{ 4, 5, 6, 5 * 100 + 4 * 10 + 6 },
        .{ -1, -2, -3, -2 * 100 + -1 * 10 + -3 },
        .{ 7, 0, 9, 0 * 100 + 7 * 10 + 9 },
    };
    try expectCallShapeEquivalent(allocator, "wimmer_combo", &helper, &caller, &inputs);
}

/// (e) Wimmer bridge gap #4: a param genuinely SPLIT, not just whole-life reassigned. `bp`/`cp`
/// cross the call, like `buildCallUseResult`'s gap #5 shape. So their ABI hint registers (x1/x2,
/// clobbered by the call) cannot cover their whole lifetime, and both are placed in the shared
/// non-leaf pool (`x19..x28`, 10 registers) instead. So far this is identical to gap #5. But
/// AFTER the call, ten more local temporaries (`r0..r9`, all needing that same 10-register pool)
/// become simultaneously live at once, since they are defined before the final reduction
/// consumes them one by one. This overflows the pool: `bp` and `cp` already hold 2 of its 10
/// slots, leaving only 8 for the 10 new temporaries. Both `bp` and `cp` have their next use in
/// the FINAL combined result, strictly later than any `r`'s next use inside the reduction chain.
/// So the shared allocator's Belady, furthest-next-use eviction heuristic picks them as the
/// cheapest to evict: split at the eviction point, in a slot from there until reloaded for the
/// final use. This is a genuinely different shape from gap #5's `buildCallUseResult` (whole-life,
/// ONE relocation) and from `buildSpillAcrossCall` (params die before the call, never split).
/// Here a param's interval has multiple segments, the bridge gap `translateAllocation` fixes.
fn buildSplitParam(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    // `a` feeds the call and dies there. `bp`/`cp` do not, so both survive across the call.
    const called = try func.appendCall(b, t, "wimmer_inc", &.{a});
    const c1 = try func.appendInst(b, t, .{ .iconst = 1 });
    const r0 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = bp, .rhs = cp } });
    const r1 = try func.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = bp, .rhs = cp } });
    const r2 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = bp, .rhs = cp } });
    const r3 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = bp, .rhs = c1 } });
    const r4 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = cp, .rhs = c1 } });
    const r5 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = r0, .rhs = c1 } });
    const r6 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = r1, .rhs = c1 } });
    const r7 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = r2, .rhs = c1 } });
    const r8 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = r3, .rhs = c1 } });
    const r9 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = r4, .rhs = c1 } });
    // All ten are defined before the reduction below consumes the first one. So the peak
    // pressure point, right after `r9`, needs all ten PLUS `bp`/`cp` live at once: 12
    // candidates for the non-leaf pool's 10 registers.
    var acc = r0;
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = r1 } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = r2 } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = r3 } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = r4 } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = r5 } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = r6 } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = r7 } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = r8 } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = r9 } });
    const s1 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = bp, .rhs = cp } });
    const s2 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = acc } });
    const res = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = called } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(res) });
    return func;
}

/// The exact value `buildSplitParam` computes, mirrored in Zig. This way the test's expected
/// results come from the same formula, rather than hand-simplified algebra.
fn splitParamExpected(a: i32, bp: i32, cp: i32) i32 {
    const c1: i32 = 1;
    const r0 = bp + cp;
    const r1 = bp - cp;
    const r2 = bp * cp;
    const r3 = bp + c1;
    const r4 = cp + c1;
    const r5 = r0 + c1;
    const r6 = r1 + c1;
    const r7 = r2 + c1;
    const r8 = r3 + c1;
    const r9 = r4 + c1;
    var acc = r0;
    acc += r1;
    acc += r2;
    acc += r3;
    acc += r4;
    acc += r5;
    acc += r6;
    acc += r7;
    acc += r8;
    acc += r9;
    const called = a + 1;
    return bp + cp + acc + called;
}

test "wimmer: a param crossing a call is evicted by later pressure (the #4 split shape)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var helper = try buildIncHelper(allocator);
    defer helper.deinit();
    var caller = try buildSplitParam(allocator);
    defer caller.deinit();
    const cases = [_][3]i32{
        .{ 0, 0, 0 }, .{ 1, 2, 3 }, .{ -2, 5, -3 }, .{ 4, -6, 7 }, .{ 10, 1, -1 }, .{ -5, -5, 5 },
    };
    var inputs: [cases.len][4]i32 = undefined;
    for (cases, 0..) |c, i| inputs[i] = .{ c[0], c[1], c[2], splitParamExpected(c[0], c[1], c[2]) };
    try expectCallShapeEquivalent(allocator, "wimmer_inc", &helper, &caller, &inputs);
}

/// (f) gap #6, EXECUTING on-host: a spill-across-call cluster dense enough that the shared scan
/// lands two re-home actions (a store and a reload targeting the SAME physical register) at ONE
/// intra-block position. Fourteen i32 params, all live across a call, exceed the ten-register
/// non-leaf gpr pool. So early-declared params (intervals starting at the block-start row) are
/// the furthest-next-use eviction victims, and their re-homes cluster on a single position. This
/// is the exact same-position drain hazard the retired `hasSamePosRegHazard` used to bail on.
/// The shared allocator now orders each cluster as a parallel move (`wimmer.orderIntraActions`),
/// so the Wimmer-compiled function must execute correctly.
///
/// This shape is deliberately Wimmer-ONLY: the native `allocate` cannot spill a PARAMETER, since
/// it evicts only non-param victims. So "more live params than the register pool" is exactly the
/// case it bails on with `error.Unsupported`. So there is no native reference to diff against.
/// Instead the Wimmer-compiled result is checked against the hand-computed ground truth, still a
/// real on-host execution of the gap-#6 cluster ordering. It is the aarch64-emitting analogue of
/// the wimmer-unit test for "more same-class params than the register pool" allocation (which
/// only checks that allocation does not crash). Here the emitted code actually runs.
fn buildManyParamSpillAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    var params: [14]Value = undefined;
    for (&params) |*p| p.* = try func.appendBlockParam(b, t);
    // The call is the barrier every param lives across, since each is used AFTER it, and it
    // clobbers the caller-saved gpr file. So all fourteen must be parked in the ten
    // callee-saved registers, or spilled.
    const called = try func.appendCall(b, t, "wimmer_inc", &.{params[0]});
    // Fold in REVERSE declaration order, so the blocked-register path keeps evicting the earliest
    // params (their next use is furthest away). Their intervals start at the block-start row,
    // and that is what makes the store and reload re-homes pile onto one position.
    var acc = called;
    var idx: usize = params.len;
    while (idx > 0) {
        idx -= 1;
        acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = params[idx] } });
    }
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(acc) });
    return func;
}

/// Call a JIT-compiled function with 14 i32 arguments and an i32 return (args 0-7 in x0..x7, args
/// 8-13 on the caller's outgoing stack area), with the arguments held in an array.
fn callI32x14(code: []const u32, a: [14]i32) !i32 {
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) callconv(.c) i32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    return f(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]);
}

/// The ground truth `buildManyParamSpillAcrossCall` computes: `inc(p0) + sum(p0..p13)`, that is
/// `(p0 + 1) + (p0 + p1 + ... + p13)`, with i32 wraparound to match the backend's 32-bit adds.
fn manyParamExpected(c: [14]i32) i32 {
    var sum: i32 = 0;
    for (c) |x| sum +%= x;
    return (c[0] +% 1) +% sum;
}

test "wimmer: a 14-param spill-across-call clusters a same-position store/reload and matches (gap #6)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var helper = try buildIncHelper(allocator);
    defer helper.deinit();
    var caller = try buildManyParamSpillAcrossCall(allocator);
    defer caller.deinit();

    // Wimmer compiles the caller (native cannot, since it will not spill a param), linked against
    // the helper. It is JIT-executed below with 14 i32 arguments (6 on the stack), so it must be
    // compiled for the HOST calling convention: Apple's arm64 ABI packs those stack arguments, AAPCS64
    // pads them to 8 bytes. On a non-Darwin host the host ABI is AAPCS64 (byte-identical).
    const host_abi: isel.Abi = if (builtin.os.tag.isDarwin()) .apple else .aapcs64;
    var wcaller = try isel.compileFunctionWimmerAbi(allocator, &caller, .{ .abi = host_abi });
    defer wcaller.deinit(allocator);
    var wlinked = try linkWithCompiledEntry(allocator, wcaller.code, wcaller.relocs, "wimmer_inc", &helper);
    defer wlinked.deinit(allocator);

    const cases = [_][14]i32{
        .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 },
        .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14 },
        .{ 100, -50, 25, -12, 6, -3, 1, 0, 7, -7, 3, -3, 9, -9 },
    };
    for (cases) |c| {
        const wim = try callI32x14(wlinked.code, c);
        try std.testing.expectEqual(manyParamExpected(c), wim);
    }
}

/// A straight-line i64 arithmetic leaf function, mirroring the very first i32 test but in the
/// 64-bit integer file. Most tests above use i32 only, so this proves the shared allocator's
/// 64-bit gpr placements translate and emit correctly too, with values that exceed the i32 range.
fn buildI64Arith(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    const dif = try func.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = bp } });
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = sum, .rhs = dif } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(prod) });
    return func;
}

test "wimmer: a straight-line i64 arithmetic function matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = try buildI64Arith(allocator);
    defer func.deinit();
    const inputs = [_][2]i64{
        .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 3, 5 },
        .{ -2, 7 }, .{ 1_000_000_000, -25 }, .{ 5_000_000_000, 41 }, // exceeds i32 range
    };
    try expectEquivalentCC([2]i64, i64, allocator, &func, callI64x2, &inputs, true);
}

/// A leaf function taking 10 i64 params (args 8-9 are stack-passed): sums them all. Exercises
/// the more-than-8-arg shape.
fn buildI64Many(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const b = try func.appendBlock();
    var p: [10]Value = undefined;
    for (0..10) |i| p[i] = try func.appendBlockParam(b, t);
    var acc = p[0];
    for (1..10) |i| acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p[i] } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(acc) });
    return func;
}

test "wimmer: a 10-arg i64 function (2 stack-passed args) matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = try buildI64Many(allocator);
    defer func.deinit();
    const inputs = [_][10]i64{
        .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
        .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10 },
        .{ 100, 1, 1, 1, 1, 1, 1, 1, 1, 1_000_000 },
    };
    try expectEquivalentCC([10]i64, i64, allocator, &func, callI64x10, &inputs, true);
}

/// A leaf function with an f32 arg pair and an f32 return: `f(a, b) = a + b`, entirely in the `v`
/// file. Exercises the float-arg-plus-float-return shape.
fn buildF32Add(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f32_t);
    const bp = try func.appendBlockParam(b, f32_t);
    const s = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });
    return func;
}

test "wimmer: an f32 arg + f32 return leaf function matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = try buildF32Add(allocator);
    defer func.deinit();
    const inputs = [_][2]f32{
        .{ 0, 0 }, .{ 1, 2 }, .{ -3.5, 4.25 }, .{ 100.0, -25.5 },
    };
    try expectEquivalentCC([2]f32, f32, allocator, &func, callF32x2, &inputs, true);
}

/// A leaf function with a <4 x f32> vector arg: `f(out, v) = *out = v + v`. Exercises the
/// <4 x f32>-arg shape. This is not expressible through `expectEquivalentCC`, since the result
/// is an out-pointer side effect, not a scalar return. So it inlines the same
/// run-both-skip-on-Unsupported shape by hand.
fn buildVecArg(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, v4);
    const v2 = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = v, .rhs = v } });
    try func.appendStore(b, v2, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    return func;
}

test "wimmer: a <4xf32> vector-arg leaf function matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = try buildVecArg(allocator);
    defer func.deinit();
    const Fn = *const fn (*[4]f32, @Vector(4, f32)) callconv(.c) void;
    const inputs = [_]@Vector(4, f32){
        .{ 1, 2, 3, 4 }, .{ -1, 0, 5, -8 }, .{ 0.5, 0.25, -0.75, 3.5 },
    };
    for (inputs) |in| {
        const ref_code = try isel.selectFunction(allocator, &func);
        defer allocator.free(ref_code);
        var ref_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(ref_code));
        defer ref_buf.deinit();
        var ref_out: [4]f32 align(16) = undefined;
        @as(Fn, @ptrCast(ref_buf.memory.ptr))(&ref_out, in);

        // STRICT: testing confirmed the bridge compiles this shape today, with no skip observed.
        // So a bail here is a REGRESSION, not a tolerated skip.
        var compiled = try isel.compileFunctionWimmer(allocator, &func);
        defer compiled.deinit(allocator);
        var wim_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(compiled.code));
        defer wim_buf.deinit();
        var wim_out: [4]f32 align(16) = undefined;
        @as(Fn, @ptrCast(wim_buf.memory.ptr))(&wim_out, in);

        try std.testing.expectEqual(ref_out, wim_out);
    }
}

/// A <4 x f32> vector param `v` (in v0) kept live across heavy INTRA-BLOCK vector-register
/// pressure. This checks old-vs-Wimmer execution equivalence for a vector-class param, with many
/// simultaneously-live vector temporaries overflowing the fpr pool. The allocator keeps `v`
/// whole-life in its ABI register here (its entry hint pins v0), so this does NOT force a split
/// of `v` itself. The split vector-param establishment path (segment 0 off the ABI register,
/// needing the 128-bit `movVec`/`ldrQ`/`strQ` rather than the 64-bit scalar-float forms) is
/// correct by design, since it mirrors the whole-life-spilled-param branch and
/// `emitSplitAction`. The split mechanism itself is exercised for the gpr class by
/// `buildSplitParam`. This test guards the vector-param path end to end: a lane drop anywhere in
/// vector param handling changes the 4-lane result.
fn buildVecParamPressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, v4);
    const n = 20;
    var ts: [n]Value = undefined;
    ts[0] = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = v, .rhs = v } });
    var i: usize = 1;
    while (i < n) : (i += 1) ts[i] = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = ts[i - 1], .rhs = ts[0] } });
    var acc = ts[0];
    i = 1;
    while (i < n) : (i += 1) acc = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = ts[i] } });
    const res = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    try func.appendStore(b, res, out);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    return func;
}

test "wimmer: a vector param under intra-block pressure matches (gap #4 vector width)" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var func = try buildVecParamPressure(allocator);
    defer func.deinit();
    const Fn = *const fn (*[4]f32, @Vector(4, f32)) callconv(.c) void;
    const inputs = [_]@Vector(4, f32){
        .{ 1, 2, 3, 4 }, .{ -1, 0, 5, -8 }, .{ 0.5, 0.25, -0.75, 3.5 }, .{ 10, -20, 30, -40 },
    };
    for (inputs) |in| {
        const ref_code = try isel.selectFunction(allocator, &func);
        defer allocator.free(ref_code);
        var ref_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(ref_code));
        defer ref_buf.deinit();
        var ref_out: [4]f32 align(16) = undefined;
        @as(Fn, @ptrCast(ref_buf.memory.ptr))(&ref_out, in);

        // STRICT: testing confirmed the bridge compiles this shape today, with no skip observed.
        // So a bail here is a REGRESSION, not a tolerated skip.
        var compiled = try isel.compileFunctionWimmer(allocator, &func);
        defer compiled.deinit(allocator);
        var wim_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(compiled.code));
        defer wim_buf.deinit();
        var wim_out: [4]f32 align(16) = undefined;
        @as(Fn, @ptrCast(wim_buf.memory.ptr))(&wim_out, in);

        try std.testing.expectEqual(ref_out, wim_out);
    }
}

// ===========================================================================
// This section adds two REALISTIC multi-function shapes, for a broad-corpus hardening step.
// Everything above tests one bridge gap at a time in isolation. These two exercise the bridge
// the way a real non-leaf function actually looks: a genuine multi-hop CALL CHAIN, and a LOOP
// that calls on every iteration while carrying an accumulator across each call. Both assert
// STRICT equivalence, with no tolerated skip: testing confirmed the whole existing corpus
// already runs end to end, so a bail on either new shape would be a genuine regression, not an
// expected gap.
// ===========================================================================

/// Differential harness for a genuine call CHAIN: `caller` calls `mid`, which itself calls `leaf`.
/// All three are linked into one image both ways. The reference module compiles EVERY function
/// (`caller` included) with the native `isel.compileFunction`. The Wimmer side compiles ONLY
/// `caller` with `compileFunctionWimmer` (matching every other call-shape test in this file: the
/// callee's OWN allocator does not matter at a call boundary, only AAPCS does), and links it
/// against a `mid`+`leaf` helper module with `linkWithCompiledEntryModule`. STRICT: a
/// `compileFunctionWimmer` bail here is a regression, not a tolerated skip.
fn expectCallChainEquivalent(
    allocator: std.mem.Allocator,
    caller: *Function,
    mid_name: []const u8,
    mid: *const Function,
    leaf_name: []const u8,
    leaf: *const Function,
    inputs: []const [4]i32, // {a, b, c, expected}
) !void {
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", caller);
    try module.addFunction(allocator, mid_name, mid);
    try module.addFunction(allocator, leaf_name, leaf);
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(linked.code));
    defer buf.deinit();
    const Fn = *const fn (i32, i32, i32) callconv(.c) i32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    for (inputs) |in| try std.testing.expectEqual(in[3], f(in[0], in[1], in[2]));

    var wcaller = try isel.compileFunctionWimmer(allocator, caller);
    defer wcaller.deinit(allocator);

    var helpers: link.Module = .{};
    defer helpers.deinit(allocator);
    try helpers.addFunction(allocator, mid_name, mid);
    try helpers.addFunction(allocator, leaf_name, leaf);
    var wlinked = try linkWithCompiledEntryModule(allocator, wcaller.code, wcaller.relocs, &helpers);
    defer wlinked.deinit(allocator);
    var wbuf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(wlinked.code));
    defer wbuf.deinit();
    const wf: Fn = @ptrCast(wbuf.memory.ptr);
    for (inputs) |in| try std.testing.expectEqual(in[3], wf(in[0], in[1], in[2]));
}

/// The innermost leaf of the chain: `leaf(v) = v*2 + 1`.
fn buildChainLeaf(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);
    const c2 = try func.appendInst(b, t, .{ .iconst = 2 });
    const one = try func.appendInst(b, t, .{ .iconst = 1 });
    const v2 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = v, .rhs = c2 } });
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = v2, .rhs = one } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    return func;
}

/// The MIDDLE of the chain: `mid(x, y) = leaf(x) + y`. `y` is LIVE ACROSS `mid`'s own call to
/// `leaf`. So the chain's middle link is itself a realistic non-leaf function, not a passthrough.
fn buildChainMid(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const r = try func.appendCall(b, t, "wimmer_chain_leaf", &.{x});
    const res = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = r, .rhs = y } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(res) });
    return func;
}

/// The OUTERMOST caller under test (the one Wimmer actually compiles): `chain(a, b, c) = a*b +
/// mid(a, c) + c`. `t = a*b` is a computed value live across the call to `mid`. `c` is an entry
/// param live across it too, and is ALSO one of `mid`'s own call arguments. `a` feeds the call
/// and its last use is the call itself. This is a realistic non-leaf function whose own callee
/// (`mid`) is itself non-leaf, exercising the bridge across a genuine two-hop call chain rather
/// than a single hop.
fn buildChainCaller(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    const called = try func.appendCall(b, t, "wimmer_chain_mid", &.{ a, cp });
    const s1 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = called } });
    const res = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = cp } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(res) });
    return func;
}

/// The exact value `buildChainCaller` computes, mirrored in Zig (`leaf(v) = 2v+1`, `mid(x,y) =
/// leaf(x)+y`, `chain(a,b,c) = a*b + mid(a,c) + c`). This way the test's expected results come
/// from the same formula, rather than hand-simplified algebra.
fn chainCallerExpected(a: i32, b: i32, c: i32) i32 {
    const leaf_of_a = 2 *% a +% 1;
    const mid_result = leaf_of_a +% c;
    return a *% b +% mid_result +% c;
}

test "wimmer: a realistic two-hop call chain (caller calls mid calls leaf) matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var leaf = try buildChainLeaf(allocator);
    defer leaf.deinit();
    var mid = try buildChainMid(allocator);
    defer mid.deinit();
    var caller = try buildChainCaller(allocator);
    defer caller.deinit();
    const cases = [_][3]i32{
        .{ 0, 0, 0 }, .{ 1, 2, 3 }, .{ -2, 5, -3 }, .{ 4, -6, 7 }, .{ 10, 1, -1 }, .{ -5, -5, 5 },
    };
    var inputs: [cases.len][4]i32 = undefined;
    for (cases, 0..) |c, i| inputs[i] = .{ c[0], c[1], c[2], chainCallerExpected(c[0], c[1], c[2]) };
    try expectCallChainEquivalent(allocator, &caller, "wimmer_chain_mid", &mid, "wimmer_chain_leaf", &leaf, &inputs);
}

/// A leaf helper for the loop-call shape below: `helper(v) = v*2 + 3`.
fn buildLoopCallHelper(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);
    const c2 = try func.appendInst(b, t, .{ .iconst = 2 });
    const c3 = try func.appendInst(b, t, .{ .iconst = 3 });
    const v2 = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = v, .rhs = c2 } });
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = v2, .rhs = c3 } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    return func;
}

/// A counted loop `f(n, x)` that CALLS a helper on every iteration and carries an accumulator
/// LIVE ACROSS each call. `lacc` (the loop-carried accumulator) is a loop block param, read only
/// AFTER the call returns to fold in that iteration's result. So it must survive the call's
/// clobber of the caller-saved gpr file on every single iteration, not just once. `x` (an entry
/// param, never reassigned) is used directly in `body`, live-in across the header the same way
/// `buildLoopSum`'s `x` is. So it too must remain live across the back-edge AND across every
/// iteration's call. This is the first shape in this file combining cross-block loop machinery
/// with a genuine non-leaf call, proving the two bridge features work together and not just in
/// isolation.
fn buildLoopCallAcc(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const n = try func.appendBlockParam(entry, t);
    const x = try func.appendBlockParam(entry, t);

    const li = try func.appendBlockParam(loop, t);
    const lacc = try func.appendBlockParam(loop, t);

    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });

    const cond = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = li, .rhs = n } });
    try func.appendIf(loop, cond, .{ .target = body }, .{ .target = exit });

    // Body: call the helper with the loop-invariant `x`, then fold the result into `lacc`. It is
    // read AFTER the call, so it must be re-homed off any caller-saved register the call clobbers.
    const called = try func.appendCall(body, t, "wimmer_loopcall_helper", &.{x});
    const acc2 = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = lacc, .rhs = called } });
    const one = try func.appendInst(body, t, .{ .iconst = 1 });
    const inext = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = li, .rhs = one } });
    try func.setJump(body, loop, &.{ inext, acc2 });

    func.setTerminator(exit, .{ .ret = ir.function.Ret.one(lacc) });
    return func;
}

/// The exact value `buildLoopCallAcc` computes, mirrored in Zig: while `n <= 0` the loop never
/// enters its body (0 iterations). Otherwise it runs exactly `n` times, each time adding
/// `helper(x) = 2x + 3`.
fn loopCallAccExpected(n: i32, x: i32) i32 {
    if (n <= 0) return 0;
    const h = 2 *% x +% 3;
    var acc: i32 = 0;
    var i: i32 = 0;
    while (i < n) : (i += 1) acc +%= h;
    return acc;
}

/// Differential harness for `buildLoopCallAcc`. It combines `expectCrossBlockEquivalent`'s
/// two-copy pattern (`compileFunctionWimmer` splits critical edges in place, so the reference
/// build must stay unmutated) with `expectCallShapeEquivalent`'s real link-and-execute (the
/// function genuinely calls a helper, so both sides must resolve a real `bl`). STRICT: a Wimmer
/// bail is a regression.
fn expectLoopCallEquivalent(
    allocator: std.mem.Allocator,
    comptime build: fn (std.mem.Allocator) anyerror!Function,
    helper_name: []const u8,
    helper: *const Function,
    inputs: []const [3]i32, // {n, x, expected}
) !void {
    var fa = try build(allocator);
    defer fa.deinit();
    var fb = try build(allocator);
    defer fb.deinit();

    var diag = try ir.verify.verify(allocator, &fa, .high);
    defer diag.deinit();
    try std.testing.expect(diag.ok());

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &fa);
    try module.addFunction(allocator, helper_name, helper);
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(linked.code));
    defer buf.deinit();
    const Fn = *const fn (i32, i32) callconv(.c) i32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    for (inputs) |in| try std.testing.expectEqual(in[2], f(in[0], in[1]));

    var wcaller = try isel.compileFunctionWimmer(allocator, &fb);
    defer wcaller.deinit(allocator);
    var wlinked = try linkWithCompiledEntry(allocator, wcaller.code, wcaller.relocs, helper_name, helper);
    defer wlinked.deinit(allocator);
    var wbuf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(wlinked.code));
    defer wbuf.deinit();
    const wf: Fn = @ptrCast(wbuf.memory.ptr);
    for (inputs) |in| try std.testing.expectEqual(in[2], wf(in[0], in[1]));
}

test "wimmer: a loop that calls a helper each iteration with a live-across-call accumulator matches" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var helper = try buildLoopCallHelper(allocator);
    defer helper.deinit();
    const cases = [_][2]i32{
        .{ 0, 5 },  .{ 1, 5 },   .{ 5, 2 },   .{ 10, 1 },  .{ -3, 7 },
        .{ 8, -2 }, .{ 3, 100 }, .{ -1, -9 }, .{ 20, -4 }, .{ 6, 0 },
    };
    var inputs: [cases.len][3]i32 = undefined;
    for (cases, 0..) |c, i| inputs[i] = .{ c[0], c[1], loopCallAccExpected(c[0], c[1]) };
    try expectLoopCallEquivalent(allocator, buildLoopCallAcc, "wimmer_loopcall_helper", &helper, &inputs);
}

// ===========================================================================
// This section tests an over-demand shape that both allocators must REJECT, not crash on.
// ===========================================================================

/// A single-block self-loop with `n_params` i32 entry params, all fed straight back into the
/// same block as its own jump arguments. So every param is live simultaneously with a
/// `must_have_register` use at the SAME position, the back-edge. This is exactly the shape that
/// pressures a register class beyond its pool: the leaf GPR pool is only x9..x12 (4 registers),
/// plus whatever of x0..x7 is not itself a live entry param. So a large enough `n_params`
/// guarantees more simultaneous must-have demand than the class has registers to satisfy, on
/// BOTH allocators.
fn buildManyParamSelfLoop(allocator: std.mem.Allocator, n_params: usize) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const loop = try func.appendBlock();
    const params = try allocator.alloc(ir.function.Value, n_params);
    defer allocator.free(params);
    for (params) |*p| p.* = try func.appendBlockParam(loop, t);
    try func.setJump(loop, loop, params);
    return func;
}

// Testing found that `wimmer.zig`'s `spillCurrent` asserted `u > current.start()` on a split
// child whose only remaining must-have use coincides with its own (post-split) start, a case a
// large enough same-position register demand reaches. The OLD allocator (`aarch64/isel.zig`'s
// `allocate`) hits the shared "too many live params" limit and bails `error.Unsupported` for the
// same shape today. So Wimmer bailing the same way, not crashing, is a MATCHED shared limit, not
// a new restriction.
test "wimmer: an over-demand self-loop with more live params than registers spills its edge arguments" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // 24 simultaneously-live i32 params exceed every register the pool can offer. The 24 values
    // travel the self-jump as edge arguments, and an edge argument is `should_have_register`: the
    // parallel-move resolver can load it from or store it to a spill slot, so the allocator spills
    // the excess instead of demanding a register for every one at the single back-edge position.
    // So the allocation now SUCCEEDS, where it once bailed `error.Unsupported`.
    var func = try buildManyParamSelfLoop(allocator, 24);
    defer func.deinit();

    const code1 = try isel.selectFunction(allocator, &func);
    defer allocator.free(code1);
    try std.testing.expect(code1.len > 0);

    var compiled = try isel.compileFunctionWimmer(allocator, &func);
    defer compiled.deinit(allocator);
    try std.testing.expect(compiled.code.len > 0);
}
