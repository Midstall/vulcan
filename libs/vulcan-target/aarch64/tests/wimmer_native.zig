//! First-execution validation of the shared Wimmer-Franz allocator on the AArch64 backend
//! (test-only). Each test compiles one function TWICE: once through the existing reference path
//! (`selectFunction` -> the backend's own `allocate`) and once through `compileFunctionWimmer` (the
//! shared allocator -> the SAME emission). Both are JIT-run on the aarch64 host across many inputs
//! and required to agree exactly. Agreement is the gate: the shared allocator produces real,
//! correct machine code. The two allocators may pick different registers/slots, so the check is
//! EXECUTION-equivalence (identical results), not byte-identical code.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const jit = @import("../jit.zig");

const Function = ir.function.Function;

/// JIT `code` and call it with `args` (up to 3 i32 args, i32 return) on the aarch64 host.
fn callI32(code: []const u32, args: []const i32) !i32 {
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const ptr = buf.memory.ptr; // page-aligned, satisfies the function-pointer alignment
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

/// Compile `func` via the SHARED Wimmer allocator and JIT-run it. Takes `func` mutable because
/// `compileFunctionWimmer` splits critical edges in place (a no-op, hence harmless, for a function
/// with none).
fn runWimmer(allocator: std.mem.Allocator, func: *Function, args: []const i32) !i32 {
    var compiled = try isel.compileFunctionWimmer(allocator, func);
    defer compiled.deinit(allocator);
    return callI32(compiled.code, args);
}

/// Run `func` both ways over every input tuple and assert bit-identical results. Single-block
/// callers share one function (edge splitting is a no-op there); cross-block callers use
/// `expectCrossBlockEquivalent`, which builds a fresh unmutated reference.
fn expectEquivalent(allocator: std.mem.Allocator, func: *Function, inputs: []const [2]i32) !void {
    for (inputs) |in| {
        const ref = try runReference(allocator, func, &in);
        const wim = try runWimmer(allocator, func, &in);
        try std.testing.expectEqual(ref, wim);
    }
}

/// Cross-block differential harness. `build` constructs the SAME function twice: `fa` is compiled by
/// the reference `selectFunction` (never mutated), `fb` by `compileFunctionWimmer` (which splits
/// critical edges in place), and their JIT results must agree bit-for-bit over every input. The
/// reference is verified first so an invalid hand-built CFG surfaces as a test error, not a divergence.
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
    // f(a, b) = ((a + b) * (a - b)) + (a * b): a short dependent chain, no register pressure, so no
    // splits fire. Proves the shared allocator's whole-life single-segment placements translate and
    // emit correctly end-to-end.
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
    func.setTerminator(b, .{ .ret = res });

    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 3, 5 }, .{ -2, 7 }, .{ 100, -25 }, .{ -37, 41 } };
    try expectEquivalent(allocator, &func, &inputs);
}

test "wimmer: the register-pressure kernel matches across inputs" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // f(a, b) = sum over k in 1..=20 of (a*k + b). All 20 products stay live until the final
    // reduction, far past the GPR pool, so the shared allocator must SPLIT live ranges and spill,
    // exercising the intra-block actions (stores + reloads) for the first time under execution.
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
    func.setTerminator(b, .{ .ret = acc });

    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    try expectEquivalent(allocator, &func, &inputs);
}

test "wimmer: a spilled value reloads the correct bits" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    // f(a, b) = a*b (defined FIRST, held live over the pressure block) plus sum_{k=1..20}(a*k + b).
    // The early product's only remaining use is the very last add, so it must be spilled under
    // pressure and RELOADED for that use. The result is correct only if the shared allocator's
    // store/reload actions round-trip the exact bits.
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
    func.setTerminator(b, .{ .ret = res });

    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    try expectEquivalent(allocator, &func, &inputs);
}

// ===========================================================================
// Cross-block (Task 8): the first EXECUTION of the shared allocator's cross-block
// live-range splitting. Each builder constructs a genuinely multi-block leaf i32
// function; the reference (unsplit) and Wimmer (edge-split, edge-move-driven)
// compilations must agree bit-for-bit over every input.
// ===========================================================================

const Value = ir.function.Value;

/// A counted loop `f(n, x)` that carries an induction variable plus six accumulators through the
/// loop header, updates them in a dependency-chained body, and reduces them at the exit. The loop
/// carries enough simultaneously-live values across the back-edge to pressure the leaf GPR pool, so
/// the shared allocator must split live ranges and resolve them with edge moves on the back-edge and
/// the header entry.
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

    // Header: continue while i < n (n stays live-in across the whole loop), else fall to the exit.
    const cond = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = li, .rhs = n } });
    try func.appendIf(loop, cond, .{ .target = body }, .{ .target = exit });

    // Body: i += 1 and each accumulator folds x plus the previous new accumulator (a dependency
    // chain that keeps them all live to the back-edge). x is live-in here across the header.
    const one = try func.appendInst(body, t, .{ .iconst = 1 });
    const inext = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = li, .rhs = one } });
    const na = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = la, .rhs = x } });
    const nb = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = lb, .rhs = na } });
    const nc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = lc, .rhs = nb } });
    const nd = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = ld, .rhs = nc } });
    const ne = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = le, .rhs = nd } });
    const nf = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = lf, .rhs = ne } });
    try func.setJump(body, loop, &.{ inext, na, nb, nc, nd, ne, nf });

    // Exit: reduce the accumulators (all loop params, live-in from the header's else-edge).
    const s1 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = la, .rhs = lb } });
    const s2 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = lc } });
    const s3 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = ld } });
    const s4 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = s3, .rhs = le } });
    const s5 = try func.appendInst(exit, t, .{ .arith = .{ .op = .add, .lhs = s4, .rhs = lf } });
    func.setTerminator(exit, .{ .ret = s5 });

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

/// A diamond `f(p, q)` where `base = p*q` is defined before the branch and used only after the merge,
/// so it is live along BOTH arms. The right arm builds many independent values that all stay live to
/// its reduction, pressuring the pool so `base` (and others) split across the arm and reload at the
/// merge, exercising a cross-block spill/reload plus a merge-parameter move.
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

    // Left arm: light, one value into the merge parameter.
    const l = try func.appendInst(left, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = q } });
    try func.setJump(left, merge, &.{l});

    // Right arm: ten values all live to the reduction, pressuring the pool while `base` is live-through.
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

    // Merge: fold the arm's value with the diamond-spanning `base`.
    const res = try func.appendInst(merge, t, .{ .arith = .{ .op = .add, .lhs = m, .rhs = base } });
    func.setTerminator(merge, .{ .ret = res });

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

/// A value with a genuine lifetime HOLE: `v = a*b` is defined in the entry and reused only in block
/// `A`, while block `B` (numbered BETWEEN entry and A, and reached by the other branch) never uses it.
/// `v` is therefore dead across `B`'s positions, a hole the allocator may fill with `B`'s pressured
/// temporaries, then must restore `v` for its reuse in `A`.
fn buildHole(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const b_blk = try func.appendBlock(); // block 1: the dead region for v (numbered before A)
    const a_blk = try func.appendBlock(); // block 2: reuses v after the hole
    const join = try func.appendBlock(); // block 3

    const a = try func.appendBlockParam(entry, t);
    const bp = try func.appendBlockParam(entry, t);
    const w = try func.appendBlockParam(join, t);

    const v = try func.appendInst(entry, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = bp } });
    try func.appendIf(entry, cond, .{ .target = a_blk }, .{ .target = b_blk });

    // B: pressured, does NOT reference v, so v is dead across every position here.
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

    // A: reuse v after the hole.
    const res = try func.appendInst(a_blk, t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = a } });
    try func.setJump(a_blk, join, &.{res});

    func.setTerminator(join, .{ .ret = w });

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
/// predecessors) on BOTH arms with a DIFFERENT argument, so each entry->merge edge is critical.
/// `splitCriticalEdges` inserts a forwarding block on each, and the merge-parameter move plus the
/// live-through `x` land as edge moves on those forwarding blocks.
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
    // Both arms target merge with a different argument: the larger operand flows through z.
    try func.appendIf(entry, cond, .{ .target = merge, .args = &.{bp} }, .{ .target = merge, .args = &.{a} });

    const res = try func.appendInst(merge, t, .{ .arith = .{ .op = .add, .lhs = z, .rhs = x } });
    func.setTerminator(merge, .{ .ret = res });

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
