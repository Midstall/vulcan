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
const link = @import("../link.zig");
const encode = @import("../encode.zig");

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

// ===========================================================================
// Wimmer cutover SP1 Task 1: scale the differential guardrail to non-leaf (call), spill-across-
// call, call-argument-alias, float/vector-param, and >8-arg shapes, so Tasks 2-6 can flip each
// bridge gap from "skip" to "assert" as it lands. Investigated empirically (JIT probes) before
// writing these: `compileFunctionWimmer`'s ONLY blanket bail for these new shapes is the leaf gate
// (`isLeaf`, gap #3): any function containing a `.call`/`.call_indirect` bails `error.Unsupported`
// immediately, before reaching any of the other 4 bridge gaps (#4/#5/#6/#7). Every LEAF shape added
// below (many-arg stack params, f32, <4xf32>) is already handled correctly by the SHARED emission
// (`emitFromAllocation`'s entry-param loop loads stack-passed args generically, leaf or not), so
// those assert equivalence TODAY rather than skip. Only the shapes with a real `.call` skip.
// ===========================================================================

/// JIT `code` and call it with `args` (up to 3 i64 args, i64 return) on the aarch64 host. Mirrors
/// `callI32` for the 64-bit integer file (today's corpus is i32-only, and i64 exercises the same
/// placements at the 64-bit width).
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

/// Adapter: feeds `callI64`'s 2-arg case through the fixed-`Args`-array shape
/// `expectEquivalentCC` expects.
fn callI64x2(code: []const u32, args: [2]i64) !i64 {
    return callI64(code, &args);
}

/// JIT `code` and call it with exactly 10 i64 args (i64 return): args 0-7 land in x0..x7, args 8-9
/// arrive on the caller's outgoing stack area (System V/AAPCS). The `>8-arg` shape (Task 1 Step 4):
/// the entry prologue must load the stack-passed params, not just the register-passed ones.
fn callI64x10(code: []const u32, args: [10]i64) !i64 {
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) callconv(.c) i64;
    const f: Fn = @ptrCast(buf.memory.ptr);
    return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]);
}

/// JIT `code` and call it with 2 f32 args (f32 return), all in the `v` file (v0/v1 in, v0 out).
fn callF32x2(code: []const u32, args: [2]f32) !f32 {
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (f32, f32) callconv(.c) f32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    return f(args[0], args[1]);
}

/// Like `expectEquivalent`, but parameterized over the calling convention (i64, many-arg, f32) via
/// `Args`/`Ret`/`call`, the way `callI32` alone serves `expectEquivalent`. The reference path
/// (`selectFunction`) ALWAYS runs, so a malformed IR builder still surfaces as a test failure
/// regardless of `must_run`.
///
/// `must_run` is the Task 6 hardening (a Task 3 reviewer finding): `expectEquivalent`-style helpers
/// that silently fold "asserted equal" and "skipped because Unsupported" into the same green let a
/// shape that is MEANT to be total regress to a skip with nobody noticing. Callers for shapes the
/// bridge is now required to compile (leaf arithmetic, register pressure, non-leaf calls,
/// live-across-call, spill-across-call, split-param, call-arg cycles, float/vector/many-arg all pass
/// `true`: `compileFunctionWimmer` returning `error.Unsupported` is then a REGRESSION and fails the
/// test loudly rather than returning early. Only a shape that hits one of the SHARED limits the old
/// path also rejects (composite-f16, zero-block) would pass `false`, tolerating the skip. Nothing in
/// this corpus needs that today (Task 6 confirmed empirically that every shape here already runs), so
/// every call site below passes `true`. The flag stays so a future genuinely-unsupported shape has
/// somewhere to land without reintroducing the silent-skip hazard for everything else.
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

/// Link an ALREADY-COMPILED entry function's code against a freshly-compiled `helper` (via the
/// reference `isel.compileFunction`), resolving the entry's `bl` relocation(s) to `helper`'s word
/// offset. Mirrors `link.compileModule`'s two-function layout/relocation logic exactly (entry first,
/// callee immediately after), but takes the entry's machine code directly instead of compiling it
/// itself, so a Wimmer-compiled caller can be linked the same way a reference-compiled one is
/// (`link.compileModule` always compiles every function via the old allocator, so it cannot be
/// reused as-is for a Wimmer-compiled entry). The helper's OWN allocator does not matter for this
/// check: AAPCS is the contract at the call boundary, not the callee's internal implementation, and
/// the reference module already exercises the reference-compiled helper. The caller owns the result.
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
    return .{ .code = code, .symbols = symbols };
}

/// Like `linkWithCompiledEntry`, but generalized to a HELPER MODULE of arbitrarily many functions
/// that may call each other (a genuine call CHAIN: `main` -> B -> a leaf, not just `main` -> one
/// leaf). `helpers` is compiled and self-linked via the reference `link.compileModule` (which already
/// resolves any relocation BETWEEN helpers correctly: `bl` is PC-relative, so shifting the whole
/// helper block by a constant (the entry's own length) preserves every already-resolved inter-helper
/// distance). This function only has to resolve the entry's OWN relocations against the shifted
/// helper symbol table. As with `linkWithCompiledEntry`, the helpers' own allocator does not matter:
/// AAPCS is the contract at each call boundary, not a callee's internal implementation. The caller
/// owns the result.
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
    return .{ .code = code, .symbols = symbols };
}

/// Differential harness for a shape that genuinely CALLS another function (`helper`). The
/// reference ALWAYS compiles+links a real two-function MODULE (`caller` as the entry, `helper`
/// resolved via a real `bl` relocation) and JIT-executes it across `inputs` (`{a, b, c, expected}`),
/// so it exercises REAL call-clobber semantics rather than a faked/unresolved call. `expected` is
/// hand-computed, so a broken IR builder fails loudly even though Wimmer is not compared yet.
/// `compileFunctionWimmer` is then REQUIRED to compile `caller` (Task 5 closed the last non-leaf
/// bridge gaps: the same-position intra-block drain hazard and the call-argument clobber are both
/// resolved as parallel moves, so every call shape in this file must compile, and a bail is a
/// regression, not a tolerated skip). The Wimmer-compiled caller is linked against the SAME helper via
/// `linkWithCompiledEntry` and JIT-executed across every input, REQUIRING bit-for-bit agreement with
/// the reference module. This is a real link+execute, not just a successful compile, so a shape whose
/// call-argument setup miscompiles (a wrong parallel-move order at a call site) still fails loudly
/// rather than passing on compile-success alone.
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

    // STRICT: the bridge must now compile every call shape in this file (spill-across-call clusters
    // and call-argument permutations included). A bail here is a REGRESSION, not a tolerated skip, so
    // surface it as a failure rather than returning early.
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
    func.setTerminator(b, .{ .ret = r });
    return func;
}

/// A leaf helper `combo(x, y, z) = x*100 + y*10 + z`: POSITION-SENSITIVE, so a call-argument
/// permutation that clobbers a source before it is consumed shows up as wrong arithmetic (not
/// masked by commutativity, the way a plain sum would be).
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
    func.setTerminator(b, .{ .ret = s2 });
    return func;
}

/// (a) a non-leaf caller that calls a leaf helper and folds the result: f(a, b, c) = inc(a) + b + c.
/// `b` and `c` are both LIVE ACROSS the call (used only in the folds afterward) while sitting in
/// their ABI arg registers (x1/x2), which the call clobbers (the clobber list is every caller-saved
/// gpr, x0..x17). The shared allocator must therefore move them off their ABI registers into a
/// callee-saved one for the whole function - this is the actual Wimmer bridge gap #5 shape (a
/// whole-life entry param placed off its ABI arg register); `a` merely feeds the call itself and
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
    func.setTerminator(b, .{ .ret = s2 });
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
/// computed BEFORE the call and used only AFTER it, forcing the shared allocator to park it in a
/// callee-saved register (or spill it) rather than a caller-saved temporary. This does NOT touch the
/// entry-param bridge gap (#5, see `buildCallUseResult` above): `t` is not one of the function's own
/// parameters, so its whole-life placement in a callee-saved register needs no ABI-register-vs-
/// allocated-register reconciliation - it is a plain instruction result, wherever it is allocated is
/// simply where its defining instruction writes it. Kept as its own shape for cross-block-style
/// (intra-block-across-a-call) coverage distinct from the param case.
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
    func.setTerminator(b, .{ .ret = res });
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

/// (c) a high-pressure non-leaf function: 14 computed (non-param) values are live across a call,
/// forcing more live-across-call values than the 10-slot callee-saved gpr pool (x19..x28) holds, so
/// some spill to the stack instead. None of the caller's own entry params (`a`, `b`, `c`) survive
/// across the call (each is consumed feeding either a term or the call itself), so this shape does
/// not exercise the entry-param bridge gaps (#4 split param, #5 off-ABI-register param); it targets
/// the spill-under-call-pressure path for ordinary values, which the bridge already handles.
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
    func.setTerminator(b, .{ .ret = res });
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
/// permutation (`combo(b, c, a)`, not `combo(a, b, c)`): the call-argument setup must not clobber a
/// source register before it feeds its own destination slot (the latent call-arg-alias risk noted
/// in the cutover spec: sequential `mov target, src` with no parallel-move resolution). CONFIRMED a
/// genuine miscompile under Wimmer's allocation (verified empirically while wiring up the real
/// link+execute comparison: it computed a wrong result before the call-argument hazard was first
/// merely detected). Task 5 FIXES it: the edge-move-driven call lowering (`Ctx.emitCallArgs`) emits
/// the argument setup as a parallel move (stack stores first, then the register permutation through
/// the reserved scratch, then slot reloads), so this 3-cycle rotation (x0<-x1<-x2<-x0) executes
/// correctly rather than being skipped.
fn buildCallArgAlias(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    const called = try func.appendCall(b, t, "wimmer_combo", &.{ bp, cp, a });
    func.setTerminator(b, .{ .ret = called });
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

/// (d2) a SHORTER call-argument cycle: a 2-cycle SWAP of two arguments (`combo(b, a, c)`), plus one
/// argument passed through unchanged. Where `buildCallArgAlias` rotates three registers, this stresses
/// the minimal cycle the scratch cycle-break exists for (x0<->x1) alongside an identity move (x2<-x2),
/// so the parallel-move ordering must both break the swap and elide the no-op. A sequential
/// `mov x0,x1; mov x1,x0` would lose the original x1.
fn buildCallArgSwap(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    const called = try func.appendCall(b, t, "wimmer_combo", &.{ bp, a, cp });
    func.setTerminator(b, .{ .ret = called });
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

/// (e) Wimmer bridge gap #4: a param genuinely SPLIT, not just whole-life reassigned. `bp`/`cp` cross
/// the call (like `buildCallUseResult`'s gap #5 shape), so their ABI hint registers (x1/x2, clobbered
/// by the call) cannot cover their whole lifetime and both are placed in the shared non-leaf pool
/// (`x19..x28`, 10 registers) instead, so far identical to gap #5. But AFTER the call, ten more local
/// temporaries (`r0..r9`, all needing that same 10-register pool) become simultaneously live at once
/// (defined before the final reduction consumes them one by one), overflowing the pool: `bp` and `cp`
/// already hold 2 of its 10 slots, leaving only 8 for the 10 new temporaries. Both `bp` and `cp` have
/// their next use in the FINAL combined result, strictly later than any `r`'s next use inside the
/// reduction chain, so the shared allocator's Belady/furthest-next-use eviction heuristic picks them
/// as the cheapest to evict: split at the eviction point (a slot from there until reloaded for the
/// final use). This is a genuinely different shape from gap #5's `buildCallUseResult` (whole-life,
/// ONE relocation) and from `buildSpillAcrossCall` (params die before the call, never split): here a
/// param's interval has multiple segments, `translateAllocation`'s bridge gap this task fixes.
fn buildSplitParam(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const cp = try func.appendBlockParam(b, t);
    // `a` feeds the call and dies there. `bp`/`cp` do not, so both survive across it.
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
    // All ten are defined before the reduction below consumes the first one, so the peak pressure
    // point (right after `r9`) needs all ten PLUS `bp`/`cp` live at once: 12 candidates for the
    // non-leaf pool's 10 registers.
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
    func.setTerminator(b, .{ .ret = res });
    return func;
}

/// The exact value `buildSplitParam` computes, mirrored in Zig so the test's expected results are
/// derived from the same formula rather than hand-simplified algebra.
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

/// (f) gap #6, EXECUTING on-host: a spill-across-call cluster dense enough that the shared scan lands
/// two re-home actions (a store and a reload targeting the SAME physical register) at ONE intra-block
/// position. Fourteen i32 params, all live across a call, exceed the ten-register non-leaf gpr pool, so
/// early-declared params (intervals starting at the block-start row) are the furthest-next-use eviction
/// victims and their re-homes cluster on a single position, the exact same-position drain hazard the
/// retired `hasSamePosRegHazard` used to bail on. The shared allocator now orders each cluster as a
/// parallel move (`wimmer.orderIntraActions`), so the Wimmer-compiled function must execute correctly.
///
/// This shape is deliberately Wimmer-ONLY: the native `allocate` cannot spill a PARAMETER (it evicts
/// only non-param victims), so "more live params than the register pool" is exactly the case it bails
/// on with `error.Unsupported`. So there is no native reference to diff against; instead the
/// Wimmer-compiled result is checked against the hand-computed ground truth, still a real on-host
/// execution of the gap-#6 cluster ordering. It is the aarch64-emitting analogue of the wimmer-unit
/// "more same-class params than the register pool" allocation test (which only checks the allocation
/// does not crash); here the emitted code actually runs.
fn buildManyParamSpillAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    var params: [14]Value = undefined;
    for (&params) |*p| p.* = try func.appendBlockParam(b, t);
    // The call is the barrier every param lives across (each is used AFTER it), and it clobbers the
    // caller-saved gpr file, so all fourteen must be parked in the ten callee-saved registers or spilled.
    const called = try func.appendCall(b, t, "wimmer_inc", &.{params[0]});
    // Fold in REVERSE declaration order so the blocked-register path keeps evicting the earliest params
    // (their next use is furthest away), whose intervals start at the block-start row: that is what
    // makes the store/reload re-homes pile onto one position.
    var acc = called;
    var idx: usize = params.len;
    while (idx > 0) {
        idx -= 1;
        acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = params[idx] } });
    }
    func.setTerminator(b, .{ .ret = acc });
    return func;
}

/// Call a JIT-compiled 14-i32-argument, i32-returning function (args 0-7 in x0..x7, args 8-13 on the
/// caller's outgoing stack area) with the arguments held in an array.
fn callI32x14(code: []const u32, a: [14]i32) !i32 {
    var buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(code));
    defer buf.deinit();
    const Fn = *const fn (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) callconv(.c) i32;
    const f: Fn = @ptrCast(buf.memory.ptr);
    return f(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]);
}

/// The ground truth `buildManyParamSpillAcrossCall` computes: `inc(p0) + sum(p0..p13)`, i.e.
/// `(p0 + 1) + (p0 + p1 + ... + p13)`, with i32 wraparound (matching the backend's 32-bit adds).
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

    // Wimmer compiles the caller (native cannot: it will not spill a param), linked against the helper.
    var wcaller = try isel.compileFunctionWimmer(allocator, &caller);
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
/// 64-bit integer file: today's corpus was i32-only, so this proves the shared allocator's 64-bit
/// gpr placements translate and emit correctly too (values exceed the i32 range).
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
    func.setTerminator(b, .{ .ret = prod });
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

/// A leaf function taking 10 i64 params (args 8-9 are stack-passed): sums them all. Exercises the
/// `>8-arg` shape (Task 1 Step 4).
fn buildI64Many(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const b = try func.appendBlock();
    var p: [10]Value = undefined;
    for (0..10) |i| p[i] = try func.appendBlockParam(b, t);
    var acc = p[0];
    for (1..10) |i| acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p[i] } });
    func.setTerminator(b, .{ .ret = acc });
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
/// file. Exercises the `float arg + float return` shape (Task 1 Step 4).
fn buildF32Add(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f32_t);
    const bp = try func.appendBlockParam(b, f32_t);
    const s = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    func.setTerminator(b, .{ .ret = s });
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
/// `<4 x f32> arg` shape (Task 1 Step 4). Not expressible through `expectEquivalentCC`
/// (the result is an out-pointer side effect, not a scalar return), so it inlines the same
/// run-both-skip-on-Unsupported shape by hand.
fn buildVecArg(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const b = try func.appendBlock();
    const out = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, v4);
    const v2 = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = v, .rhs = v } });
    try func.appendStore(b, v2, out);
    func.setTerminator(b, .{ .ret = null });
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

        // STRICT (Task 6): the bridge is confirmed to compile this shape today (no skip was observed
        // empirically), so a bail here is a REGRESSION, not a tolerated skip.
        var compiled = try isel.compileFunctionWimmer(allocator, &func);
        defer compiled.deinit(allocator);
        var wim_buf = try jit.CodeBuffer.map(std.mem.sliceAsBytes(compiled.code));
        defer wim_buf.deinit();
        var wim_out: [4]f32 align(16) = undefined;
        @as(Fn, @ptrCast(wim_buf.memory.ptr))(&wim_out, in);

        try std.testing.expectEqual(ref_out, wim_out);
    }
}

/// A <4 x f32> vector param `v` (in v0) kept live across heavy INTRA-BLOCK vector-register pressure,
/// checking old-vs-Wimmer execution-equivalence for a vector-class param with many simultaneously-live
/// vector temporaries overflowing the fpr pool. Note the allocator keeps `v` whole-life in its ABI
/// register here (its entry hint pins v0), so this does NOT force a split of `v` itself. The split
/// vector-param ESTABLISHMENT path (segment 0 off the ABI register, needing the 128-bit
/// `movVec`/`ldrQ`/`strQ` rather than the 64-bit scalar-float forms) is defensively correct by
/// mirroring the whole-life-spilled-param branch and `emitSplitAction`, and the split MECHANISM itself
/// is exercised for the gpr class by `buildSplitParam`. This test guards the vector-param path end to
/// end (a lane drop anywhere in vector param handling diverges the 4-lane result).
fn buildVecParamPressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
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
    func.setTerminator(b, .{ .ret = null });
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

        // STRICT (Task 6): the bridge is confirmed to compile this shape today (no skip was observed
        // empirically), so a bail here is a REGRESSION, not a tolerated skip.
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
// Wimmer cutover Task 6: two REALISTIC multi-function shapes requested by the broad-corpus
// hardening step. Everything above tests one bridge gap at a time in isolation, and these two exercise
// the bridge the way a real non-leaf function actually looks: a genuine multi-hop CALL CHAIN, and a
// LOOP that calls on every iteration while carrying an accumulator across each call. Both assert
// STRICT equivalence (no tolerated skip): Task 6 confirmed empirically that the whole existing
// corpus already runs end to end, so a bail on either new shape would be a genuine regression, not
// an expected gap.
// ===========================================================================

/// Differential harness for a genuine call CHAIN: `caller` calls `mid`, which itself calls `leaf`.
/// All three are linked into one image both ways. The reference module compiles EVERY function
/// (`caller` included) via the native `isel.compileFunction`. The Wimmer side compiles ONLY `caller`
/// via `compileFunctionWimmer` (matching every other call-shape test in this file: the callee's OWN
/// allocator does not matter at a call boundary, only AAPCS does) and links it against a `mid`+`leaf`
/// helper module via `linkWithCompiledEntryModule`. STRICT: a `compileFunctionWimmer` bail here is a
/// regression, not a tolerated skip.
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
    func.setTerminator(b, .{ .ret = r });
    return func;
}

/// The MIDDLE of the chain: `mid(x, y) = leaf(x) + y`. `y` is LIVE ACROSS `mid`'s own call to
/// `leaf`, so the chain's middle link is itself a realistic non-leaf function, not a passthrough.
fn buildChainMid(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const r = try func.appendCall(b, t, "wimmer_chain_leaf", &.{x});
    const res = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = r, .rhs = y } });
    func.setTerminator(b, .{ .ret = res });
    return func;
}

/// The OUTERMOST caller under test (the one Wimmer actually compiles): `chain(a, b, c) = a*b +
/// mid(a, c) + c`. `t = a*b` is a computed value live across the call to `mid`. `c` is an entry param
/// live across it too (and is ALSO one of `mid`'s own call arguments), and `a` feeds the call and its
/// last use is the call itself. A realistic non-leaf function whose own callee (`mid`) is itself
/// non-leaf, exercising the bridge across a genuine two-hop call chain rather than a single hop.
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
    func.setTerminator(b, .{ .ret = res });
    return func;
}

/// The exact value `buildChainCaller` computes, mirrored in Zig (`leaf(v) = 2v+1`, `mid(x,y) =
/// leaf(x)+y`, `chain(a,b,c) = a*b + mid(a,c) + c`), so the test's expected results come from the
/// same formula rather than hand-simplified algebra.
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
    func.setTerminator(b, .{ .ret = r });
    return func;
}

/// A counted loop `f(n, x)` that CALLS a helper on every iteration and carries an accumulator
/// LIVE ACROSS each call: `lacc` (the loop-carried accumulator) is a loop block param, read only
/// AFTER the call returns to fold in that iteration's result, so it must survive the call's clobber
/// of the caller-saved gpr file on every single iteration, not just once. `x` (an entry param, never
/// reassigned) is used directly in `body`, live-in across the header the same way `buildLoopSum`'s
/// `x` is, so it too must remain live across the back-edge AND across every iteration's call. This is
/// the first shape in this file combining cross-block loop machinery with a genuine non-leaf call,
/// proving the two bridge features compose rather than being sound only in isolation.
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

    // Body: call the helper with the loop-invariant `x`, then fold the result into `lacc` (read
    // AFTER the call, so it must be re-homed off any caller-saved register the call clobbers).
    const called = try func.appendCall(body, t, "wimmer_loopcall_helper", &.{x});
    const acc2 = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = lacc, .rhs = called } });
    const one = try func.appendInst(body, t, .{ .iconst = 1 });
    const inext = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = li, .rhs = one } });
    try func.setJump(body, loop, &.{ inext, acc2 });

    func.setTerminator(exit, .{ .ret = lacc });
    return func;
}

/// The exact value `buildLoopCallAcc` computes, mirrored in Zig: while `n <= 0` the loop never
/// enters its body (0 iterations), otherwise it runs exactly `n` times, each adding
/// `helper(x) = 2x + 3`.
fn loopCallAccExpected(n: i32, x: i32) i32 {
    if (n <= 0) return 0;
    const h = 2 *% x +% 3;
    var acc: i32 = 0;
    var i: i32 = 0;
    while (i < n) : (i += 1) acc +%= h;
    return acc;
}

/// Differential harness for `buildLoopCallAcc`: combines `expectCrossBlockEquivalent`'s two-copy
/// pattern (`compileFunctionWimmer` splits critical edges in place, so the reference build must stay
/// unmutated) with `expectCallShapeEquivalent`'s real link+execute (the function genuinely calls a
/// helper, so both sides must resolve a real `bl`). STRICT: a Wimmer bail is a regression.
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
// Wimmer cutover Task 6b: an over-demand shape that both allocators must REJECT, not crash on.
// ===========================================================================

/// A single-block self-loop with `n_params` i32 entry params, all fed straight back into the same
/// block as its own jump arguments. Every param is therefore live simultaneously with a
/// `must_have_register` use at the SAME position (the back-edge), which is exactly the shape that
/// pressures a register class beyond its pool: the leaf GPR pool is only x9..x12 (4 registers) plus
/// whatever of x0..x7 is not itself a live entry param, so a large enough `n_params` guarantees more
/// simultaneous must-have demand than the class has registers to satisfy, on BOTH allocators.
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

// Task 2 found: `wimmer.zig`'s `spillCurrent` asserted `u > current.start()` on a split child whose
// only remaining must-have use coincides with its own (post-split) start, a case a large enough
// same-position register demand reaches. The OLD allocator (`aarch64/isel.zig`'s `allocate`) hits
// the shared "too many live params" limit and bails `error.Unsupported` for the same shape today, so
// Wimmer bailing the same way (not crashing) is a MATCHED shared limit, not a new restriction.
test "wimmer: an over-demand self-loop with more live params than registers bails on both allocators" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // 24 simultaneously-live i32 params vastly exceeds every register-supply story the leaf pool can
    // offer (8 arg registers + 4 scratch registers, and even that generous upper bound assumes zero
    // contention), so both allocators must reject this function rather than mis-schedule it.
    var func = try buildManyParamSelfLoop(allocator, 24);
    defer func.deinit();

    try std.testing.expectError(error.Unsupported, isel.selectFunction(allocator, &func));
    try std.testing.expectError(error.Unsupported, isel.compileFunctionWimmer(allocator, &func));
}
