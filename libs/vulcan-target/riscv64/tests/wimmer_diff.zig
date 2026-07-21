//! riscv64 adoption Task 2: the SHARED Wimmer-Franz allocator produces EXECUTABLE riscv64 code. Each
//! test builds TWO identical functions, compiles one through the backend's own `selectFunction` (the
//! reference) and the other through `isel.compileFunctionWimmerRiscv` (the shared allocator + the
//! same emission), runs BOTH under qemu-riscv64, and asserts the results are bit-identical across many
//! inputs. Scope is the INT, scalar-FLOAT, and RVV VECTOR classes (RV-T3 added vectors; the et-soc VPU
//! class is RV-T4). qemu is the execution oracle: a divergence means the shared allocation was
//! translated or emitted wrong. The RVV cases execute real vle32/vse32/vfadd/vfmul under qemu's
//! default RV64 CPU (which enables the V extension), the same way `fma_vector.zig` does.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const encode = @import("../encode.zig");
const link = @import("../link.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

/// Compile `func` through the shared Wimmer allocator (mutates `func`: it splits critical edges) and
/// run it under qemu with integer args, returning a0. Legalizes first so the IR matches what the
/// reference pipeline (`harness.runFunc`) feeds its own `selectFunction`.
fn runWimmer(io: std.Io, allocator: std.mem.Allocator, func: *Function, args: []const i64) !i64 {
    try ir.legalize.legalize(allocator, func);
    var compiled = try isel.compileFunctionWimmerRiscv(allocator, func, false);
    defer compiled.deinit(allocator);
    return harness.runCode(io, allocator, compiled.code, args, harness.qemu_user);
}

/// The float-ABI analogue of `runWimmer` (fa0.. args, fa0 result bits).
fn runWimmerFloat(io: std.Io, allocator: std.mem.Allocator, func: *Function, fargs: []const u64) !u64 {
    try ir.legalize.legalize(allocator, func);
    var compiled = try isel.compileFunctionWimmerRiscv(allocator, func, false);
    defer compiled.deinit(allocator);
    return harness.runCompiledFloat(io, allocator, compiled.code, false, fargs, harness.qemu_user);
}

/// Run the reference (`selectFunction`) and the Wimmer path on two freshly-built copies of the same
/// function for every input, asserting the integer results match. `build` takes only the allocator so
/// each side gets its own untouched function (both pipelines mutate the IR in place). `expected` is a
/// hand-derived ground-truth oracle mirroring the function's own semantics in plain Zig (see each call
/// site), independent of both compiled pipelines: `compileFunction`/`selectFunction` now IS the Wimmer
/// path, so a bare `ref == got` check alone would only prove the two entry points agree with each
/// other, not that either is right. `expected == got` catches a miscompile shared by both entries, and
/// `ref == got` stays as a belt-and-suspenders check that the two entry points did not diverge.
fn expectIntMatch(io: std.Io, comptime build: fn (std.mem.Allocator) anyerror!Function, comptime expected: fn ([]const i64) i64, inputs: []const []const i64) !void {
    const allocator = std.testing.allocator;
    for (inputs) |args| {
        var ref_func = try build(allocator);
        defer ref_func.deinit();
        var wim_func = try build(allocator);
        defer wim_func.deinit();

        const ref = harness.runFunc(io, allocator, &ref_func, args, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got = runWimmer(io, allocator, &wim_func, args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(expected(args), got);
        try std.testing.expectEqual(ref, got);
    }
}

// ---------------------------------------------------------------------------
// 1. Straight-line integer arithmetic.
// ---------------------------------------------------------------------------

/// f(a, b, c) = (a + b) * (b + c) - (a * c). A handful of simultaneously-live temps, no spilling: the
/// baseline that the prologue, the ABI-register param moves, and the int arithmetic translate right.
fn buildStraightLine(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i64_t);
    const b = try func.appendBlockParam(entry, i64_t);
    const c = try func.appendBlockParam(entry, i64_t);
    const ab = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const bc = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = b, .rhs = c } });
    const ac = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const prod = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = ab, .rhs = bc } });
    const res = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .sub, .lhs = prod, .rhs = ac } });
    func.setTerminator(entry, .{ .ret = res });
    return func;
}

/// The exact value `buildStraightLine` computes, mirrored in Zig: (a+b)*(b+c) - (a*c), i64 wrapping.
fn straightLineExpected(args: []const i64) i64 {
    const a = args[0];
    const b = args[1];
    const c = args[2];
    const ab = a +% b;
    const bc = b +% c;
    const ac = a *% c;
    return (ab *% bc) -% ac;
}

test "wimmer-rv: straight-line int arithmetic matches" {
    const inputs = [_][]const i64{ &.{ 1, 2, 3 }, &.{ 0, 0, 0 }, &.{ -5, 7, -9 }, &.{ 100, -200, 300 }, &.{ 123456, -1, 2 } };
    try expectIntMatch(std.testing.io, buildStraightLine, straightLineExpected, &inputs);
}

// ---------------------------------------------------------------------------
// 2. Integer register-pressure kernel (forces spilling / live-range splitting).
// ---------------------------------------------------------------------------

const n_fan = 30;

/// f(n) = sum_k (n*(k+1) + k) for k in 0..30. All 30 terms are created before any is consumed, so far
/// more integer values are live at once than the 17 allocatable registers: the shared allocator must
/// spill and tail-split. The reduction reloads every operand, so a wrong split/spill diverges.
fn buildIntPressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);
    var a: [n_fan]Value = undefined;
    for (0..n_fan) |k| {
        const coeff = try func.appendInst(entry, i64_t, .{ .iconst = @intCast(k + 1) });
        const prod = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = n, .rhs = coeff } });
        a[k] = try func.appendArithImm(entry, i64_t, .add, prod, @intCast(k));
    }
    var sum = a[n_fan - 1];
    var k: usize = n_fan - 1;
    while (k > 0) {
        k -= 1;
        sum = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = a[k] } });
    }
    func.setTerminator(entry, .{ .ret = sum });
    return func;
}

/// The exact value `buildIntPressure` computes, mirrored in Zig. Two's-complement addition and
/// multiplication distribute exactly mod 2^64, so sum_{k=0}^{29}(n*(k+1)+k) collapses to
/// 465*n + 435, since sum_{k=1}^{30}k = 465 and sum_{k=0}^{29}k = 435 (the reduction order in
/// `buildIntPressure` is irrelevant: wrapping addition is associative and commutative). i64 wrapping.
fn intPressureExpected(args: []const i64) i64 {
    const n = args[0];
    return (465 *% n) +% 435;
}

test "wimmer-rv: an int register-pressure kernel matches" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{7}, &.{-3}, &.{100}, &.{-1000}, &.{123456} };
    try expectIntMatch(std.testing.io, buildIntPressure, intPressureExpected, &inputs);
}

// ---------------------------------------------------------------------------
// 3. Float arithmetic + pressure kernel (f32 values, some spilling).
// ---------------------------------------------------------------------------

const n_flive = 30;

/// f(a) = sum_i (a + i) for i in 0..30, f32. All 30 sums are live at once, exceeding the allocatable
/// float file, so the shared allocator spills/splits scalar floats. Every intermediate is an exact
/// small integer in float, so the sum is order-independent and must match the reference bit-for-bit.
fn buildFloatPressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);
    var vals: [n_flive]Value = undefined;
    for (0..n_flive) |i| {
        const ci = try func.appendInst(entry, i32_t, .{ .iconst = @intCast(i) });
        const cf = try func.appendInst(entry, f32_t, .{ .convert = .{ .value = ci } });
        vals[i] = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = cf } });
    }
    var acc = vals[0];
    for (vals[1..]) |v| acc = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(entry, .{ .ret = acc });
    return func;
}

test "wimmer-rv: a float arithmetic + pressure kernel matches" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const a_vals = [_]f32{ 0.0, 1.0, 100.0, -7.0, 1234.0 };
    for (a_vals) |a_val| {
        var ref_func = try buildFloatPressure(allocator);
        defer ref_func.deinit();
        var wim_func = try buildFloatPressure(allocator);
        defer wim_func.deinit();

        const a_bits = [_]u64{@as(u32, @bitCast(a_val))};
        const ref = harness.runFuncFloat(io, allocator, &ref_func, false, &a_bits, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got = runWimmerFloat(io, allocator, &wim_func, &a_bits) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(ref, got);
    }
}

// ---------------------------------------------------------------------------
// 4. A loop-carried int sum across a pressured body (cross-block edge moves).
// ---------------------------------------------------------------------------

const n_body = 24;

/// f(n): acc=0; for i in 0..n: { t[k] = (i+1)*(k+1) for k in 0..24 (all live at once); acc += sum(t) };
/// return acc. Only three integers cross the loop as block params (acc, i, n), so the back-edge jump
/// is a narrow reg->reg parallel move (the cross-block Wimmer edge-move path), while the body creates
/// 24 simultaneously-live temporaries - far past the 17 allocatable integer registers - so the shared
/// allocator spills and tail-splits INSIDE the body. A wrong intra-block split or a wrong back-edge
/// move diverges from the reference.
fn buildLoopSum(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i64_t);
    const iv0 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const acc0 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ iv0, n, acc0 });

    const h_i = try func.appendBlockParam(header, i64_t);
    const h_n = try func.appendBlockParam(header, i64_t);
    const h_acc = try func.appendBlockParam(header, i64_t);
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = h_i, .rhs = h_n } });
    try func.appendIf(header, cond, .{ .target = body, .args = &.{ h_i, h_n, h_acc } }, .{ .target = exit, .args = &.{h_acc} });

    const b_i = try func.appendBlockParam(body, i64_t);
    const b_n = try func.appendBlockParam(body, i64_t);
    const b_acc = try func.appendBlockParam(body, i64_t);
    const ip1 = try func.appendArithImm(body, i64_t, .add, b_i, 1);
    var t: [n_body]Value = undefined;
    for (0..n_body) |k| {
        const coeff = try func.appendInst(body, i64_t, .{ .iconst = @intCast(k + 1) });
        t[k] = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .mul, .lhs = ip1, .rhs = coeff } });
    }
    var s = t[n_body - 1];
    var k: usize = n_body - 1;
    while (k > 0) {
        k -= 1;
        s = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = t[k] } });
    }
    const next_acc = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = b_acc, .rhs = s } });
    const next_i = try func.appendArithImm(body, i64_t, .add, b_i, 1);
    try func.setJump(body, header, &.{ next_i, b_n, next_acc });

    const e_acc = try func.appendBlockParam(exit, i64_t);
    func.setTerminator(exit, .{ .ret = e_acc });
    return func;
}

/// The exact value `buildLoopSum` computes, mirrored in Zig. Iteration i (0-indexed) contributes
/// ip1 * sum_{k=1}^{24}k = ip1*300 (n_body=24, sum_{1..24} = 300), so iters iterations (ip1 = 1..iters)
/// sum to 300*(1+..+iters) = 150*iters*(iters+1). The header test `h_i < h_n` starting at 0 runs zero
/// iterations for any n <= 0. i64 wrapping.
fn loopSumExpected(args: []const i64) i64 {
    const n = args[0];
    const iters: i64 = if (n > 0) n else 0;
    return 150 *% iters *% (iters +% 1);
}

test "wimmer-rv: a loop-carried int sum across a pressured body matches" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{3}, &.{7}, &.{12} };
    try expectIntMatch(std.testing.io, buildLoopSum, loopSumExpected, &inputs);
}

// ---------------------------------------------------------------------------
// 5. A diamond with an int value live on both paths.
// ---------------------------------------------------------------------------

/// f(n): c = n*3 (live on BOTH arms and the join); if n > 0 -> a else b; a: va = c + 10; b: vb = c +
/// 20; m(p): return p + c. The join `m` takes a phi `p` (va from a, vb from b) resolved by moves on
/// the jump edges a->m / b->m, while `c` is a value defined in the entry that stays live across both
/// arms into `m`. Exercises a cross-block value live on both paths plus a jump-edge phi.
fn buildDiamond(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const a_blk = try func.appendBlock();
    const b_blk = try func.appendBlock();
    const m_blk = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i64_t);
    const three = try func.appendInst(entry, i64_t, .{ .iconst = 3 });
    const c = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = n, .rhs = three } });
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = n, .rhs = zero } });
    try func.appendIf(entry, cond, .{ .target = a_blk }, .{ .target = b_blk });

    const va = try func.appendArithImm(a_blk, i64_t, .add, c, 10);
    try func.setJump(a_blk, m_blk, &.{va});

    const vb = try func.appendArithImm(b_blk, i64_t, .add, c, 20);
    try func.setJump(b_blk, m_blk, &.{vb});

    const p = try func.appendBlockParam(m_blk, i64_t);
    const r = try func.appendInst(m_blk, i64_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = c } });
    func.setTerminator(m_blk, .{ .ret = r });
    return func;
}

/// The exact value `buildDiamond` computes, mirrored in Zig: c = 3n, and the join adds c to whichever
/// arm ran (c+10 or c+20), so the result is 2c+10 = 6n+10 when n > 0, else 2c+20 = 6n+20. i64 wrapping.
fn diamondExpected(args: []const i64) i64 {
    const n = args[0];
    const c = n *% 3;
    return if (n > 0) (c +% 10) +% c else (c +% 20) +% c;
}

test "wimmer-rv: a diamond with an int value live on both paths matches" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{-1}, &.{5}, &.{-9}, &.{100} };
    try expectIntMatch(std.testing.io, buildDiamond, diamondExpected, &inputs);
}

// ---------------------------------------------------------------------------
// 6. RVV VECTOR class (RV-T3): a <4 x f32> value flows through the shared allocator, exercising the
//    class-2 translation, the vsetivli preamble, and (under pressure / across a call) the vector
//    spill/split path that GENERALIZES the old `error.Unsupported` bail on a vector live across a call.
//    Each function returns a scalar lane (fa0) so the result is observable through the float ABI.
// ---------------------------------------------------------------------------

/// Run the reference (`selectFunction`, via the full scheduler pipeline) and the shared Wimmer path on
/// two freshly-built copies of the same float-ABI function for every input, asserting the raw `fa0`
/// bits match. `build` takes only the allocator (each side gets an untouched function, both pipelines
/// mutate the IR in place). `fa_sets` are the fa0.. argument-bit vectors to try. `expected` is a
/// hand-derived ground-truth oracle over the f32 result lane (see each call site): `compileFunction`/
/// `selectFunction` now IS the Wimmer path, so `expected == got` catches a miscompile shared by both
/// entries, while `ref == got` stays as a belt-and-suspenders check that the two entry points agree.
fn expectFloatMatch(io: std.Io, comptime build: fn (std.mem.Allocator) anyerror!Function, comptime expected: fn ([]const u64) u32, fa_sets: []const []const u64) !void {
    const allocator = std.testing.allocator;
    for (fa_sets) |fargs| {
        var ref_func = try build(allocator);
        defer ref_func.deinit();
        var wim_func = try build(allocator);
        defer wim_func.deinit();

        const ref = harness.runFuncFloat(io, allocator, &ref_func, false, fargs, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got = runWimmerFloat(io, allocator, &wim_func, fargs) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(expected(fargs), @as(u32, @truncate(got)));
        try std.testing.expectEqual(ref, got);
    }
}

/// f(a) = ((a, 2, 3, 4) + (5, 6, 7, 8)) * (2, 2, 2, 2), returning lane 0 = (a + 5) * 2. A handful of
/// simultaneously-live <4 x f32> vectors, no spilling: the baseline that the class-2 register
/// translation, the one-shot vsetivli preamble, and the RVV vfadd/vfmul arithmetic translate right.
/// The lanes are built from `fconst`s and the fa0 param via `struct_new`; RVV has no vector ABI to
/// carry a whole vector in, and nothing in this backend constant-folds float arithmetic, so this
/// exercises the real hardware ops at qemu-execution time.
fn buildVecArith(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);

    const c2 = try func.appendInst(entry, f32_t, .{ .fconst = 2.0 });
    const c3 = try func.appendInst(entry, f32_t, .{ .fconst = 3.0 });
    const c4 = try func.appendInst(entry, f32_t, .{ .fconst = 4.0 });
    var a_lanes = [_]Value{ a, c2, c3, c4 };
    const va = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&a_lanes) } });

    const c5 = try func.appendInst(entry, f32_t, .{ .fconst = 5.0 });
    const c6 = try func.appendInst(entry, f32_t, .{ .fconst = 6.0 });
    const c7 = try func.appendInst(entry, f32_t, .{ .fconst = 7.0 });
    const c8 = try func.appendInst(entry, f32_t, .{ .fconst = 8.0 });
    var b_lanes = [_]Value{ c5, c6, c7, c8 };
    const vb = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&b_lanes) } });

    const vsum = try func.appendInst(entry, v4, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });

    const two = try func.appendInst(entry, f32_t, .{ .fconst = 2.0 });
    var c_lanes = [_]Value{ two, two, two, two };
    const vc = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&c_lanes) } });

    const vprod = try func.appendInst(entry, v4, .{ .arith = .{ .op = .mul, .lhs = vsum, .rhs = vc } });
    const lane0 = try func.appendInst(entry, f32_t, .{ .extract = .{ .aggregate = vprod, .index = 0 } });
    func.setTerminator(entry, .{ .ret = lane0 });
    return func;
}

/// The exact value `buildVecArith` computes, mirrored in Zig: lane 0 = (a + 5) * 2, f32.
fn vecArithExpected(fargs: []const u64) u32 {
    const a: f32 = @bitCast(@as(u32, @truncate(fargs[0])));
    const want: f32 = (a + 5.0) * 2.0;
    return @bitCast(want);
}

test "wimmer-rv: an RVV vector arithmetic function matches" {
    const a_vals = [_]f32{ 0.0, 1.0, 3.0, -7.0, 100.0 };
    var sets: [a_vals.len][1]u64 = undefined;
    var ptrs: [a_vals.len][]const u64 = undefined;
    for (a_vals, 0..) |v, i| {
        sets[i] = .{@as(u32, @bitCast(v))};
        ptrs[i] = &sets[i];
    }
    try expectFloatMatch(std.testing.io, buildVecArith, vecArithExpected, &ptrs);
}

const n_vlive = 30;

/// f(a) = sum_i splat(a + i) for i in 0..30, then lane 0 = sum_i (a + i) = 30*a + 435. All 30 <4 x f32>
/// vectors are created (each a splat of the scalar `a + i`) before any is consumed, so far more vectors
/// are live at once than the 27 allocatable RVV registers (v1..v27): the shared allocator must SPILL a
/// vector to a 16-byte slot and reload it. Every reduction step reloads an operand, so a wrong
/// spill/reload diverges. Each `a + i` is an exact small integer in f32, so the reduction (a strict
/// dependency chain, unreorderable by either pipeline's scheduler) is bit-identical to the reference.
fn buildVecPressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);

    var vals: [n_vlive]Value = undefined;
    for (0..n_vlive) |i| {
        const fi = try func.appendInst(entry, f32_t, .{ .fconst = @floatFromInt(i) });
        const ci = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = fi } });
        var lanes = [_]Value{ ci, ci, ci, ci };
        vals[i] = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&lanes) } });
    }
    var acc = vals[0];
    for (vals[1..]) |v| acc = try func.appendInst(entry, v4, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    const lane0 = try func.appendInst(entry, f32_t, .{ .extract = .{ .aggregate = acc, .index = 0 } });
    func.setTerminator(entry, .{ .ret = lane0 });
    return func;
}

/// The exact value `buildVecPressure` computes, mirrored in Zig: lane 0 = sum_{i=0}^{29}(a + i) =
/// 30*a + 435 (sum_{0..29} = 435), f32. Every term is an exact small integer offset from a, so the sum
/// is order-independent and exact for these inputs.
fn vecPressureExpected(fargs: []const u64) u32 {
    const a: f32 = @bitCast(@as(u32, @truncate(fargs[0])));
    const want: f32 = 30.0 * a + 435.0;
    return @bitCast(want);
}

test "wimmer-rv: RVV vector register pressure spills a vector and reloads it" {
    const a_vals = [_]f32{ 0.0, 1.0, 10.0, -4.0 };
    var sets: [a_vals.len][1]u64 = undefined;
    var ptrs: [a_vals.len][]const u64 = undefined;
    for (a_vals, 0..) |v, i| {
        sets[i] = .{@as(u32, @bitCast(v))};
        ptrs[i] = &sets[i];
    }
    try expectFloatMatch(std.testing.io, buildVecPressure, vecPressureExpected, &ptrs);
}

/// The leaf callee `g(x) = x + 1` (scalar f32). Built as its own function so the caller's `call` is a
/// real inter-function call that clobbers every caller-saved register (all of v1..v27 among them).
fn buildAddOneCallee(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const x = try func.appendBlockParam(blk, f32_t);
    const one = try func.appendInst(blk, f32_t, .{ .fconst = 1.0 });
    const r = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = one } });
    func.setTerminator(blk, .{ .ret = r });
    return func;
}

/// The caller `f(a)`: build vv = (a, 2, 3, 4) + (5, 6, 7, 8) BEFORE calling g(10), then AFTER the call
/// read lane 0 of vv and add the call result, returning (a + 5) + 11 = a + 16. `vv` (a <4 x f32>) is
/// defined before the call and used after it, so it is LIVE ACROSS the call. Every RVV register is
/// caller-saved, so the shared allocator cannot keep vv in a register across the call: it must spill/
/// split it to a 16-byte slot and reload it afterward. The OLD riscv64 Wimmer path bailed
/// `error.Unsupported` on exactly this shape; the test asserts it now compiles and computes right.
fn buildVecAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);

    const c2 = try func.appendInst(entry, f32_t, .{ .fconst = 2.0 });
    const c3 = try func.appendInst(entry, f32_t, .{ .fconst = 3.0 });
    const c4 = try func.appendInst(entry, f32_t, .{ .fconst = 4.0 });
    var a_lanes = [_]Value{ a, c2, c3, c4 };
    const va = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&a_lanes) } });

    const c5 = try func.appendInst(entry, f32_t, .{ .fconst = 5.0 });
    const c6 = try func.appendInst(entry, f32_t, .{ .fconst = 6.0 });
    const c7 = try func.appendInst(entry, f32_t, .{ .fconst = 7.0 });
    const c8 = try func.appendInst(entry, f32_t, .{ .fconst = 8.0 });
    var b_lanes = [_]Value{ c5, c6, c7, c8 };
    const vb = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&b_lanes) } });

    const vv = try func.appendInst(entry, v4, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });

    const ten = try func.appendInst(entry, f32_t, .{ .fconst = 10.0 });
    const cr = try func.appendCall(entry, f32_t, "callee", &.{ten});

    const lane0 = try func.appendInst(entry, f32_t, .{ .extract = .{ .aggregate = vv, .index = 0 } });
    const r = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = lane0, .rhs = cr } });
    func.setTerminator(entry, .{ .ret = r });
    return func;
}

/// Concatenate `caller_c` (entry, at word 0) and `callee_c`, resolve every call relocation in the
/// caller (all target "callee") to an intra-image `jal`, and run under qemu with float args, returning
/// the raw `fa0` bits. Mirrors `link.compileModule`'s call-resolution, but lets the caller be compiled
/// through whichever pipeline the test chose (the shared Wimmer path vs the native reference).
fn linkRunFloat(io: std.Io, allocator: std.mem.Allocator, caller_c: *const isel.Compiled, callee_c: *const isel.Compiled, fargs: []const u64) !u64 {
    const code = try allocator.alloc(u32, caller_c.code.len + callee_c.code.len);
    defer allocator.free(code);
    @memcpy(code[0..caller_c.code.len], caller_c.code);
    @memcpy(code[caller_c.code.len..], callee_c.code);
    const callee_start = caller_c.code.len;
    for (caller_c.relocs) |reloc| {
        std.debug.assert(reloc.kind == .call);
        std.debug.assert(std.mem.eql(u8, reloc.symbol, "callee"));
        const delta = (@as(i64, @intCast(callee_start)) - @as(i64, @intCast(reloc.offset))) * 4;
        code[reloc.offset] = encode.jal(.x1, @intCast(delta));
    }
    return harness.runCompiledFloat(io, allocator, code, false, fargs, harness.qemu_user);
}

// ---------------------------------------------------------------------------
// 7. SP3 Task 1: a same-position store/reload cluster at a block-param boundary, exercising the
//    bridge's retired `hasSamePosRegHazard` bail (replaced by consuming the shared allocator's
//    already-ordered `walloc.actions`).
//
// `n_pre_rv` values stay ACTIVE (live, in registers) from `entry` across every iteration of a loop
// BODY, while the body's OWN `n_hdr_rv` carried params arrive at its shared param-row position on
// both the first entry (a plain jump) and the back edge (also a plain jump). `n_pre_rv + n_hdr_rv`
// exceeds the 17-register allocatable int pool (`temp_regs` + `saved_regs`), so the accumulate loop
// deep in the body - which reads every `pre[k]` and every `b[k]` together - forces the shared
// allocator to evict some still-live `pre[k]` right where an arriving `b[k]` needs a register, an
// intra-block same-position store/reload pair.
//
// The loop TEST (`if`) sits at the body's OWN TAIL, after `pre[]`'s last use, so `pre[]` is fully
// DEAD by the time control reaches the `if` and never crosses an if-edge. Only the body's own
// loop-carried params (`b[]`, `total`, `iter`) cross the if-edge, and only as EXPLICIT args that
// `splitCriticalEdges` moves onto a landing block's jump - the same shape `buildLoopSum` already
// exercises successfully. This deliberately avoids putting `pre[]` across an `if`: riscv64's
// `.@"if"` emission can only branch (it cannot host an edge move), so a value live across an
// if-edge with a genuine location change is a SEPARATE, pre-existing bridge limitation
// (`edgeMoveOnIfEdge`) this task does not touch. An earlier draft of this test put the loop test at
// the block's TOP (a `header`/`body` split, mirroring x86_64 SP2's `buildHeaderEvictsPreLoop`
// directly) and confirmed exactly that: `pre[]` stayed live across the header's own if-edge with a
// location change, tripping `edgeMoveOnIfEdge` instead of (or in addition to) the intended hazard.
// ---------------------------------------------------------------------------

const n_pre_rv = 16;
const n_hdr_rv = 8;

/// `pre[]`: defined in `entry`, used only deep inside `body`'s accumulate loop (so they are ACTIVE,
/// occupying registers, across every entry into `body` and right up to that read point). `body`'s OWN
/// `n_hdr_rv` carried params (`b[]`) plus a running total and iteration counter arrive at its shared
/// param-row position on both the first entry (`entry -> body`) and the back edge
/// (`landing_continue -> body`). `n_pre_rv + n_hdr_rv + 2 > 17` (the int pool), so the accumulate loop
/// - which reads every `pre[k]` and every `b[k]` together - forces the shared allocator to evict some
/// still-live `pre[k]` right where an arriving `b[k]` needs a register, an intra-block same-position
/// store/reload pair. The loop test (`if`) sits at the TAIL, after `pre[]`'s last use, so `pre[]`
/// never crosses an if-edge (see the section comment above).
fn buildBodyEvictsPre(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const seed = try func.appendBlockParam(entry, i64_t);
    var pre: [n_pre_rv]Value = undefined;
    for (0..n_pre_rv) |k| pre[k] = try func.appendArithImm(entry, i64_t, .add, seed, @intCast(k));

    var hseed: [n_hdr_rv]Value = undefined;
    for (0..n_hdr_rv) |k| hseed[k] = try func.appendArithImm(entry, i64_t, .add, seed, @intCast(100 + k));
    const iter0 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const total0 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    var entry_args: [n_hdr_rv + 2]Value = undefined;
    for (0..n_hdr_rv) |k| entry_args[k] = hseed[k];
    entry_args[n_hdr_rv] = total0;
    entry_args[n_hdr_rv + 1] = iter0;
    try func.setJump(entry, body, &entry_args);

    var b: [n_hdr_rv]Value = undefined;
    for (0..n_hdr_rv) |k| b[k] = try func.appendBlockParam(body, i64_t);
    const b_total = try func.appendBlockParam(body, i64_t);
    const b_iter = try func.appendBlockParam(body, i64_t);
    // Consume the n_pre_rv pre-loop values AND the body-carried params together: this shared read
    // point is exactly where the n_pre_rv occupants (still active from the entry block) collide with
    // the freshly-arrived body params for the int register pool. Every `pre[k]` use ends here, well
    // before the tail `if` below.
    var acc = pre[0];
    for (1..n_pre_rv) |k| acc = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = pre[k] } });
    for (0..n_hdr_rv) |k| acc = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = b[k] } });
    const new_total = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = b_total, .rhs = acc } });
    const next_iter = try func.appendArithImm(body, i64_t, .add, b_iter, 1);
    const limit = try func.appendInst(body, i64_t, .{ .iconst = 3 });
    const cond = try func.appendInst(body, bool_t, .{ .icmp = .{ .op = .lt, .lhs = next_iter, .rhs = limit } });
    var back_args: [n_hdr_rv + 2]Value = undefined;
    for (0..n_hdr_rv) |k| back_args[k] = b[k];
    back_args[n_hdr_rv] = new_total;
    back_args[n_hdr_rv + 1] = next_iter;
    try func.appendIf(body, cond, .{ .target = body, .args = &back_args }, .{ .target = exit, .args = &.{new_total} });

    const e_total = try func.appendBlockParam(exit, i64_t);
    func.setTerminator(exit, .{ .ret = e_total });
    return func;
}

/// The exact value `buildBodyEvictsPre` computes, mirrored in Zig. The n_hdr_rv carried params stay =
/// hseed[k] = seed+100+k across every iteration (the back-edge forwards them unchanged), so each
/// iteration's acc = sum_{k=0..15}(seed+k) + sum_{k=0..7}(seed+100+k) = 24*seed + 948 (sum_{0..15} =
/// 120, 8*100 + sum_{0..7} = 800+28 = 828, 120+828 = 948) is IDENTICAL every time. iter runs
/// 0 -> 1 -> 2 -> 3, stopping once next_iter reaches the limit of 3, so the loop body executes exactly
/// 3 times and total = 3*acc. i64 wrapping.
fn bodyEvictsPreExpected(args: []const i64) i64 {
    const seed = args[0];
    const acc = (24 *% seed) +% 948;
    return 3 *% acc;
}

test "wimmer-rv: a body param cluster forces a same-position store/reload and matches" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{-1}, &.{5}, &.{-9}, &.{100}, &.{-1000} };
    try expectIntMatch(std.testing.io, buildBodyEvictsPre, bodyEvictsPreExpected, &inputs);
}

test "wimmer-rv: a vector live across a call spills across the call (not an error)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const a_vals = [_]f32{ 0.0, 1.0, 5.0, -3.0, 42.0 };

    // A vector live across a call has NO native reference to diff against (the retired native allocator
    // bailed `error.Unsupported` on it - every RVV register is caller-saved and it had no vector split -
    // and `compileFunction` is now itself the shared Wimmer path, so it no longer bails). The oracle is
    // therefore the mathematically-correct result, computed in f32 in the SAME op order the IR builds:
    // lane 0 of vv = a + 5, plus the call result g(10) = 11.
    for (a_vals) |a_val| {
        const fargs = [_]u64{@as(u32, @bitCast(a_val))};
        const lane0: f32 = a_val + 5.0;
        const want: f32 = lane0 + 11.0;

        // Wimmer: caller through the SHARED allocator (this MUST succeed, not error.Unsupported),
        // callee through the native path, linked and run. The vector `vv` spills across the call.
        var wim_caller = try buildVecAcrossCall(allocator);
        defer wim_caller.deinit();
        var wim_callee = try buildAddOneCallee(allocator);
        defer wim_callee.deinit();
        try ir.legalize.legalize(allocator, &wim_caller);
        try ir.legalize.legalize(allocator, &wim_callee);
        var wim_caller_c = try isel.compileFunctionWimmerRiscv(allocator, &wim_caller, false);
        defer wim_caller_c.deinit(allocator);
        var wim_callee_c = try isel.compileFunction(allocator, &wim_callee, .{});
        defer wim_callee_c.deinit(allocator);
        const got_bits = linkRunFloat(io, allocator, &wim_caller_c, &wim_callee_c, &fargs) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got: f32 = @bitCast(@as(u32, @truncate(got_bits)));
        try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
    }
}

// ---------------------------------------------------------------------------
// 8. SP3 Task 2, Gap A: the entry-param ABI hint. A param LIVE ACROSS A CALL lands off its ABI
//    register (a callee-saved register the shared allocator homes it in for the whole function), or -
//    under enough post-call pressure - genuinely SPLITS (one home across the call, possibly another
//    after it). Both bailed `error.Unsupported` before this task (translateAllocation's blanket
//    is_eparam bail, then the int/float ABI-register-equality check).
//
//    CRITICAL: for these tests to actually EXERCISE the fix, the callee must genuinely CLOBBER the
//    caller's argument registers a1/a2 (fa1/fa2). A leaf callee `g(x)=x+1` does NOT touch a1/a2, so
//    with the earlier (buggy) per-call clobber list - which omitted a0..a7/fa0..fa7 - the shared
//    allocator wrongly believed a1/a2 survived the call and left `bp`/`cp` there, yet the leaf never
//    overwrote them, so the wrong allocation still computed the right answer and the bug hid. Here the
//    callee `g(x) = m(x, x+1, x+2)` makes its OWN 3-arg nested call to `m`, so `g` writes a0/a1/a2
//    on the way in - clobbering exactly where `bp`/`cp` sit. With the clobber bug the differential
//    DIVERGES (Wimmer reads a garbage a1/a2); with the fix `bp`/`cp` are moved to callee-saved
//    registers in the prologue and the results match. Verified: reverting the arg-reg clobber makes
//    these tests FAIL, the fix makes them PASS.
// ---------------------------------------------------------------------------

/// The int leaf `m(p, q, r) = p + q + r`. Named "leaf" so `buildNestedCalleeInt` calls it.
fn buildAdd3LeafInt(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const blk = try func.appendBlock();
    const p = try func.appendBlockParam(blk, i64_t);
    const q = try func.appendBlockParam(blk, i64_t);
    const r = try func.appendBlockParam(blk, i64_t);
    const pq = try func.appendInst(blk, i64_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = q } });
    const s = try func.appendInst(blk, i64_t, .{ .arith = .{ .op = .add, .lhs = pq, .rhs = r } });
    func.setTerminator(blk, .{ .ret = s });
    return func;
}

/// The NON-LEAF int callee `g(x) = m(x, x+1, x+2) = 3x + 3`. Because it sets up a 3-argument call it
/// writes a0/a1/a2 on the way into `m`, so it CLOBBERS the caller's `bp`/`cp` argument registers
/// (a1/a2) - the property that makes the Gap A tests actually catch the missing-clobber bug. Named
/// "callee" so the Gap A callers call it; it calls "leaf".
fn buildNestedCalleeInt(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i64_t);
    const x1 = try func.appendArithImm(b, i64_t, .add, x, 1);
    const x2 = try func.appendArithImm(b, i64_t, .add, x, 2);
    const r = try func.appendCall(b, i64_t, "leaf", &.{ x, x1, x2 });
    func.setTerminator(b, .{ .ret = r });
    return func;
}

/// `g(x) = 3x + 3`, mirrored in Zig for the expected-value formulas below.
fn nestedCalleeInt(x: i64) i64 {
    return (x +% (x +% 1)) +% (x +% 2);
}

/// Link a caller's already-compiled code (entry, word 0) against a native, self-linked HELPERS module
/// (its internal `g -> m` call already resolved by `link.compileModule`, since `jal` is PC-relative
/// and shifting the whole helper block by the entry's length preserves that distance), resolving the
/// entry's own relocations against the shifted helper symbol table. Mirrors aarch64's
/// `linkWithCompiledEntryModule`. Runs under qemu with integer args, returning a0.
fn linkRunIntModule(io: std.Io, allocator: std.mem.Allocator, caller_c: *const isel.Compiled, helpers: *const link.Linked, args: []const i64) !i64 {
    const code = try allocator.alloc(u32, caller_c.code.len + helpers.code.len);
    defer allocator.free(code);
    @memcpy(code[0..caller_c.code.len], caller_c.code);
    @memcpy(code[caller_c.code.len..], helpers.code);
    const base = caller_c.code.len;
    std.debug.assert(helpers.relocs.len == 0); // helpers self-link fully (g -> m is intra-module)
    for (caller_c.relocs) |reloc| {
        std.debug.assert(reloc.kind == .call);
        var target_word: ?usize = null;
        for (helpers.symbols) |s| {
            if (std.mem.eql(u8, s.name, reloc.symbol)) target_word = base + s.offset;
        }
        const target = target_word orelse unreachable;
        const delta = (@as(i64, @intCast(target)) - @as(i64, @intCast(reloc.offset))) * 4;
        code[reloc.offset] = encode.jal(.x1, @intCast(delta));
    }
    return harness.runCode(io, allocator, code, args, harness.qemu_user);
}

/// Build the int helpers module (`callee` = g, `leaf` = m) and self-link it (native). The caller owns
/// the returned `Linked`.
fn buildIntHelpers(allocator: std.mem.Allocator) !link.Linked {
    var g = try buildNestedCalleeInt(allocator);
    defer g.deinit();
    var m = try buildAdd3LeafInt(allocator);
    defer m.deinit();
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "callee", &g);
    try module.addFunction(allocator, "leaf", &m);
    return link.compileModule(allocator, &module);
}

/// Compile `caller` (Wimmer or native), link it against a freshly-built int helpers module, and run.
fn compileLinkRunInt(io: std.Io, allocator: std.mem.Allocator, caller: *Function, wimmer_caller: bool, args: []const i64) !i64 {
    try ir.legalize.legalize(allocator, caller);
    var caller_c = if (wimmer_caller)
        try isel.compileFunctionWimmerRiscv(allocator, caller, false)
    else
        try isel.compileFunction(allocator, caller, .{});
    defer caller_c.deinit(allocator);
    var helpers = try buildIntHelpers(allocator);
    defer helpers.deinit(allocator);
    return linkRunIntModule(io, allocator, &caller_c, &helpers, args);
}

/// Gap A: a non-leaf caller `f(a, bp, cp) = g(a) + bp + cp`. `bp`/`cp` are entry params LIVE ACROSS
/// the call while sitting in their ABI arg registers (a1/a2), which the call clobbers (`g` writes
/// a0/a1/a2 to set up its own 3-arg call). The shared allocator must home them in a callee-saved
/// register (x9/x18..x27) for the whole function - a whole-life entry param placed off its ABI arg
/// register - or `bp`/`cp` are read back as garbage. `a` merely feeds the call and dies there.
fn buildIntParamAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, i64_t);
    const bp = try func.appendBlockParam(b, i64_t);
    const cp = try func.appendBlockParam(b, i64_t);
    const called = try func.appendCall(b, i64_t, "callee", &.{a});
    const s1 = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = bp } });
    const s2 = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = cp } });
    func.setTerminator(b, .{ .ret = s2 });
    return func;
}

test "wimmer-rv: a non-leaf function whose int param crosses a call is homed off its ABI register (Gap A)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // f(a,bp,cp) = g(a) + bp + cp = (3a+3) + bp + cp. `bp`/`cp` sit in a1/a2, which g clobbers.
    const inputs = [_][3]i64{ .{ 0, 0, 0 }, .{ 1, 2, 3 }, .{ -5, 10, 2 }, .{ 100, -25, 4 }, .{ 7, -3, 9 } };
    for (inputs) |in| {
        const args = [_]i64{ in[0], in[1], in[2] };
        const want = nestedCalleeInt(in[0]) +% in[1] +% in[2];

        // Reference: the native allocator already moves a cross-call entry param to a callee-saved
        // register (`allocateRegisters`'s `popSaved` branch), so the reference is correct and this is
        // a real differential.
        var ref_caller = try buildIntParamAcrossCall(allocator);
        defer ref_caller.deinit();
        const ref = compileLinkRunInt(io, allocator, &ref_caller, false, &args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(want, ref);

        // Wimmer: the caller through the SHARED allocator. With the arg-reg clobber fix, `bp`/`cp` are
        // moved off a1/a2 in the prologue and survive g's clobber; without it they read garbage.
        var wim_caller = try buildIntParamAcrossCall(allocator);
        defer wim_caller.deinit();
        const got = compileLinkRunInt(io, allocator, &wim_caller, true, &args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(ref, got);
    }
}

/// The f32 leaf `m(p, q, r) = p + q + r`. Named "leaf".
fn buildAdd3LeafFloat(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const p = try func.appendBlockParam(blk, f32_t);
    const q = try func.appendBlockParam(blk, f32_t);
    const r = try func.appendBlockParam(blk, f32_t);
    const pq = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = q } });
    const s = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = pq, .rhs = r } });
    func.setTerminator(blk, .{ .ret = s });
    return func;
}

/// The NON-LEAF f32 callee `g(x) = m(x, x+1, x+2)`. Writes fa0/fa1/fa2 for its 3-arg call, CLOBBERING
/// the caller's `bp`/`cp` float-argument registers (fa1/fa2). Named "callee"; calls "leaf".
fn buildNestedCalleeFloat(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, f32_t);
    const one = try func.appendInst(b, f32_t, .{ .fconst = 1.0 });
    const two = try func.appendInst(b, f32_t, .{ .fconst = 2.0 });
    const x1 = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = one } });
    const x2 = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = two } });
    const r = try func.appendCall(b, f32_t, "leaf", &.{ x, x1, x2 });
    func.setTerminator(b, .{ .ret = r });
    return func;
}

/// Build the f32 helpers module (`callee` = g, `leaf` = m) and self-link it (native).
fn buildFloatHelpers(allocator: std.mem.Allocator) !link.Linked {
    var g = try buildNestedCalleeFloat(allocator);
    defer g.deinit();
    var m = try buildAdd3LeafFloat(allocator);
    defer m.deinit();
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "callee", &g);
    try module.addFunction(allocator, "leaf", &m);
    return link.compileModule(allocator, &module);
}

/// The float analogue of `linkRunIntModule` (fa0.. args, fa0 result bits).
fn linkRunFloatModule(io: std.Io, allocator: std.mem.Allocator, caller_c: *const isel.Compiled, helpers: *const link.Linked, fargs: []const u64) !u64 {
    const code = try allocator.alloc(u32, caller_c.code.len + helpers.code.len);
    defer allocator.free(code);
    @memcpy(code[0..caller_c.code.len], caller_c.code);
    @memcpy(code[caller_c.code.len..], helpers.code);
    const base = caller_c.code.len;
    std.debug.assert(helpers.relocs.len == 0);
    for (caller_c.relocs) |reloc| {
        std.debug.assert(reloc.kind == .call);
        var target_word: ?usize = null;
        for (helpers.symbols) |s| {
            if (std.mem.eql(u8, s.name, reloc.symbol)) target_word = base + s.offset;
        }
        const target = target_word orelse unreachable;
        const delta = (@as(i64, @intCast(target)) - @as(i64, @intCast(reloc.offset))) * 4;
        code[reloc.offset] = encode.jal(.x1, @intCast(delta));
    }
    return harness.runCompiledFloat(io, allocator, code, false, fargs, harness.qemu_user);
}

/// Compile `caller` (Wimmer or native), link it against a fresh f32 helpers module, and run.
fn compileLinkRunFloat(io: std.Io, allocator: std.mem.Allocator, caller: *Function, wimmer_caller: bool, fargs: []const u64) !u64 {
    try ir.legalize.legalize(allocator, caller);
    var caller_c = if (wimmer_caller)
        try isel.compileFunctionWimmerRiscv(allocator, caller, false)
    else
        try isel.compileFunction(allocator, caller, .{});
    defer caller_c.deinit(allocator);
    var helpers = try buildFloatHelpers(allocator);
    defer helpers.deinit(allocator);
    return linkRunFloatModule(io, allocator, &caller_c, &helpers, fargs);
}

/// The FLOAT analogue of `buildIntParamAcrossCall`: `f(a, bp, cp) = g(a) + bp + cp`, all f32. `bp`/
/// `cp` sit in fa1/fa2 across the call to the CLOBBERING callee `g` (which writes fa0/fa1/fa2), so
/// the shared allocator must home them in a callee-saved float register for the whole function.
fn buildFloatParamAcrossCall(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f32_t);
    const bp = try func.appendBlockParam(b, f32_t);
    const cp = try func.appendBlockParam(b, f32_t);
    const called = try func.appendCall(b, f32_t, "callee", &.{a});
    const s1 = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = bp } });
    const s2 = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = cp } });
    func.setTerminator(b, .{ .ret = s2 });
    return func;
}

test "wimmer-rv: a non-leaf function whose float param crosses a call is homed off its ABI register (Gap A)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Small integer-valued f32s keep every intermediate exact, so the differential is bit-exact.
    const a_vals = [_][3]f32{
        .{ 0.0, 0.0, 0.0 }, .{ 1.0, 2.0, 3.0 }, .{ -5.0, 10.0, 2.0 }, .{ 100.0, -25.0, 4.0 },
    };
    for (a_vals) |v| {
        const fargs = [_]u64{ @as(u32, @bitCast(v[0])), @as(u32, @bitCast(v[1])), @as(u32, @bitCast(v[2])) };

        var ref_caller = try buildFloatParamAcrossCall(allocator);
        defer ref_caller.deinit();
        const ref_bits = compileLinkRunFloat(io, allocator, &ref_caller, false, &fargs) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const ref: f32 = @bitCast(@as(u32, @truncate(ref_bits)));

        var wim_caller = try buildFloatParamAcrossCall(allocator);
        defer wim_caller.deinit();
        const got_bits = compileLinkRunFloat(io, allocator, &wim_caller, true, &fargs) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got: f32 = @bitCast(@as(u32, @truncate(got_bits)));
        try std.testing.expectEqual(ref, got);
    }
}

const n_split_rv = 20;

/// Gap A, higher-pressure variant: `bp`/`cp` cross the call to the CLOBBERING callee `g` like
/// `buildIntParamAcrossCall`, but AFTER the call `n_split_rv` (20) more local temporaries become
/// simultaneously live at once (each `bp OP cp` or `bp/cp + k`, defined before the reduction below
/// consumes them one by one), overflowing the 17-register int pool. `bp`/`cp` have their next use in
/// the FINAL result, later than any temporary's next use, so the shared allocator's Belady eviction
/// heuristic is liable to evict them partway through - a genuinely SPLIT param (not just a whole-life
/// off-ABI reassignment), exercising `translateAllocation`'s segments branch and
/// `emitFromAllocation`'s first-segment establishment. Because `g` clobbers a1/a2, the FIRST segment
/// (across the call) must be off the arg registers or the result is garbage.
fn buildSplitParamRv(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, i64_t);
    const bp = try func.appendBlockParam(b, i64_t);
    const cp = try func.appendBlockParam(b, i64_t);
    const called = try func.appendCall(b, i64_t, "callee", &.{a});

    var r: [n_split_rv]Value = undefined;
    r[0] = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .add, .lhs = bp, .rhs = cp } });
    r[1] = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .sub, .lhs = bp, .rhs = cp } });
    r[2] = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .mul, .lhs = bp, .rhs = cp } });
    var k: usize = 3;
    while (k < n_split_rv) : (k += 1) r[k] = try func.appendArithImm(b, i64_t, .add, r[k - 3], @intCast(k));

    var acc = r[0];
    for (r[1..]) |term| acc = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = term } });
    const s1 = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .add, .lhs = bp, .rhs = cp } });
    const s2 = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = acc } });
    const res = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = called } });
    func.setTerminator(b, .{ .ret = res });
    return func;
}

/// The exact value `buildSplitParamRv` computes, mirrored in Zig (g(a) = 3a+3 via `nestedCalleeInt`).
fn splitParamRvExpected(a: i64, bp: i64, cp: i64) i64 {
    var r: [n_split_rv]i64 = undefined;
    r[0] = bp +% cp;
    r[1] = bp -% cp;
    r[2] = bp *% cp;
    var k: usize = 3;
    while (k < n_split_rv) : (k += 1) r[k] = r[k - 3] +% @as(i64, @intCast(k));
    var acc: i64 = r[0];
    for (r[1..]) |term| acc +%= term;
    return (bp +% cp) +% acc +% nestedCalleeInt(a);
}

test "wimmer-rv: a param crossing a call is evicted by later pressure (Gap A split shape)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const cases = [_][3]i64{ .{ 0, 0, 0 }, .{ 1, 2, 3 }, .{ -5, 10, 2 }, .{ 100, -25, 4 }, .{ 7, -3, 9 } };
    for (cases) |c| {
        const args = [_]i64{ c[0], c[1], c[2] };
        const want = splitParamRvExpected(c[0], c[1], c[2]);

        var ref_caller = try buildSplitParamRv(allocator);
        defer ref_caller.deinit();
        const ref = compileLinkRunInt(io, allocator, &ref_caller, false, &args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(want, ref);

        var wim_caller = try buildSplitParamRv(allocator);
        defer wim_caller.deinit();
        const got = compileLinkRunInt(io, allocator, &wim_caller, true, &args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(ref, got);
    }
}

// ---------------------------------------------------------------------------
// 9. SP3 Task 2, Gap B: SOFTWARE f16 (no Zfh). riscv64 has no hardware half, so an f16 is held as its
//    f32 widening in a float register and every boundary rounds via the inline software routines in
//    isel.zig (`emitHalfToFloat`/`emitFloatToHalf`), which need dedicated integer scratch (x28..x31).
//    The Wimmer entry bailed on ANY f16 function before this task (`functionUsesF16` at the top of
//    `compileFunctionWimmerRiscv`) because `riscv64RegDescription`'s class-0 pool always used the
//    full `temp_regs`, including x28..x31 - had the entry bail been lifted without also shrinking the
//    pool, a live value could have been placed in one of those registers and silently corrupted by
//    the very next f16 conversion. These shapes reuse the differential style of `tests/f16.zig`.
// ---------------------------------------------------------------------------

/// `f(a: f16, b: f16) -> f16`: a single f16 binary op (fa0/fa1 args, fa0 result, the held-as-f32
/// convention). Mirrors `f16.zig`'s `buildBinaryFn`.
fn buildF16BinaryRv(func: *Function, op: ir.function.BinOp) !void {
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f16_t);
    const c = try func.appendBlockParam(b, f16_t);
    const r = try func.appendInst(b, f16_t, .{ .arith = .{ .op = op, .lhs = a, .rhs = c } });
    func.setTerminator(b, .{ .ret = r });
}

fn widenF16(x: f16) u32 {
    return @bitCast(@as(f32, x));
}

test "wimmer-rv: f16 add/sub/mul/div (software emulation, no Zfh) matches the reference (Gap B)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const ops = [_]ir.function.BinOp{ .add, .sub, .mul, .div };
    const cases = [_]struct { a: f16, b: f16 }{
        .{ .a = 1.5, .b = 2.25 },
        .{ .a = -3.5, .b = 0.75 },
        .{ .a = 0.1, .b = 0.2 }, // not exact in f16, so results round
        .{ .a = 100.0, .b = 7.0 },
    };
    for (ops) |op| {
        for (cases) |c| {
            const fargs = [_]u64{ widenF16(c.a), widenF16(c.b) };

            var ref_func = Function.init(allocator);
            defer ref_func.deinit();
            try buildF16BinaryRv(&ref_func, op);
            const ref_bits = harness.runFuncFloat(io, allocator, &ref_func, false, &fargs, harness.qemu_user) catch |e| switch (e) {
                error.SkipZigTest => return error.SkipZigTest,
                else => return e,
            };

            var wim_func = Function.init(allocator);
            defer wim_func.deinit();
            try buildF16BinaryRv(&wim_func, op);
            const got_bits = runWimmerFloat(io, allocator, &wim_func, &fargs) catch |e| switch (e) {
                error.SkipZigTest => return error.SkipZigTest,
                else => return e,
            };
            try std.testing.expectEqual(@as(u32, @truncate(ref_bits)), @as(u32, @truncate(got_bits)));
        }
    }
}

const n_f16_live_rv = 30;

/// `f(a: f16) -> f16`: 30 simultaneously-live f16 values `a + i` (each an f16 add that rounds to
/// half), folded with f16 adds. 30 exceeds the caller-saved float temps, forcing several f16 values to
/// spill to the stack and reload. Mirrors `f16.zig`'s `buildSpillSumFn`.
fn buildF16SpillSumRv(func: *Function) !void {
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f16_t);
    var vals: [n_f16_live_rv]Value = undefined;
    for (0..n_f16_live_rv) |i| {
        const ci = try func.appendInst(b, i32_t, .{ .iconst = @intCast(i) });
        const cf = try func.appendInst(b, f16_t, .{ .convert = .{ .value = ci } });
        vals[i] = try func.appendInst(b, f16_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = cf } });
    }
    var acc = vals[0];
    for (vals[1..]) |v| acc = try func.appendInst(b, f16_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(b, .{ .ret = acc });
}

test "wimmer-rv: f16 register pressure spills and reloads bit-exact (Gap B)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const a_vals = [_]f16{ 100.0, 0.0, -7.0, 42.0 };
    for (a_vals) |a_val| {
        const fargs = [_]u64{widenF16(a_val)};

        var ref_func = Function.init(allocator);
        defer ref_func.deinit();
        try buildF16SpillSumRv(&ref_func);
        const ref_bits = harness.runFuncFloat(io, allocator, &ref_func, false, &fargs, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };

        var wim_func = Function.init(allocator);
        defer wim_func.deinit();
        try buildF16SpillSumRv(&wim_func);
        const got_bits = runWimmerFloat(io, allocator, &wim_func, &fargs) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(@as(u32, @truncate(ref_bits)), @as(u32, @truncate(got_bits)));
    }
}

const n_f16_int_mix = 20;

/// Gap B, the scratch-exclusion shape: `f(a: f16, b: f16) -> i64`. Builds `n_f16_int_mix` (20) live
/// integer temporaries derived from a seed (exceeding the f16-shrunk 13-register int pool, temp_regs_f16
/// + saved_regs), keeps them ALL live across an f16 load-then-store round trip (which forces an
/// `emitHalfToFloat` THEN an `emitFloatToHalf` through x28..x31 - see `f16_scratch_a`..`f16_scratch_d`),
/// then reduces the integers and combines with the f16 round-trip result. If the shared allocator ever
/// placed one of these 20 live integers in x28..x31 (the shrunk pool exists precisely to prevent that),
/// the f16 conversion mid-function would silently clobber it and this would diverge from the reference.
fn buildF16IntMixRv(func: *Function) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const n = try func.appendBlockParam(b, i64_t);
    const hbits = try func.appendBlockParam(b, i64_t); // half bit pattern in the low 16 bits

    var ints: [n_f16_int_mix]Value = undefined;
    for (0..n_f16_int_mix) |k| ints[k] = try func.appendArithImm(b, i64_t, .add, n, @intCast(k));

    // The f16 round trip: load the half from `hbits` (lhu + software extend), double it (f16 add,
    // rounds to half), and store the widened result back (software truncate). Both boundaries clobber
    // x28..x31 via `emitHalfToFloat`/`emitFloatToHalf` while every `ints[k]` above is still live.
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(b, hbits, slot); // low 16 bits = the half pattern
    const h = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot } });
    const h2 = try func.appendInst(b, f16_t, .{ .arith = .{ .op = .add, .lhs = h, .rhs = h } });
    const widened = try func.appendInst(b, f32_t, .{ .convert = .{ .value = h2 } });
    try func.appendStore(b, widened, slot);
    const hbits_out32 = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = slot } });
    const hbits_out = try func.appendInst(b, i64_t, .{ .convert = .{ .value = hbits_out32 } });

    var acc = ints[0];
    for (ints[1..]) |v| acc = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    const result = try func.appendInst(b, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = hbits_out } });
    func.setTerminator(b, .{ .ret = result });
}

test "wimmer-rv: software f16 mixed with 20 live integers matches (Gap B scratch exclusion)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const cases = [_]struct { n: i64, half: f16 }{
        .{ .n = 0, .half = 1.5 }, .{ .n = 7, .half = -3.5 }, .{ .n = -100, .half = 100.0 }, .{ .n = 12345, .half = 0.0 },
    };
    for (cases) |c| {
        const hbits: i64 = @as(u16, @bitCast(c.half));
        const args = [_]i64{ c.n, hbits };

        var ref_func = Function.init(allocator);
        defer ref_func.deinit();
        try buildF16IntMixRv(&ref_func);
        const ref = harness.runFunc(io, allocator, &ref_func, &args, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };

        var wim_func = Function.init(allocator);
        defer wim_func.deinit();
        try buildF16IntMixRv(&wim_func);
        const got = runWimmer(io, allocator, &wim_func, &args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(ref, got);
    }
}

// ---------------------------------------------------------------------------
// 10. SP3 Task 2 re-review: a 7th+ VPU float ENTRY param is a MATCHED SHARED LIMIT (bail, not
//     miscompile). In vpu (et-soc) mode fa6/fa7 (f16/f17) sit inside the VPU vector partition
//     (class 3), so pinning a 7th/8th scalar-float (class 1) param there would let the shared
//     allocator land a scalar float on top of a live VPU vector - a silent alias. Both the shared
//     Wimmer path (`riscv64RegDescription` / `translateAllocation`, `if (vpu and float_idx >= 6)`)
//     and the native `allocateRegisters` (isel.zig `if (vpu and float_arg >= 6)`) REJECT this shape
//     with `error.Unsupported`, so this is a feature-limit both paths decline identically, not a
//     regression. No qemu differential is needed (there is nothing to run); the assertion is simply
//     that BOTH compilers reject rather than emit.
// ---------------------------------------------------------------------------

/// `f(a..g) = a + b + c + d + e + f + g`, SEVEN f32 params. Under vpu mode the 7th param (index 6)
/// would pin fa6 = f16, inside the VPU vector partition, so both compilers must reject it.
fn buildSevenFloatParams(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    var ps: [7]Value = undefined;
    for (0..7) |i| ps[i] = try func.appendBlockParam(b, f32_t);
    var acc = ps[0];
    for (ps[1..]) |p| acc = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
    func.setTerminator(b, .{ .ret = acc });
    return func;
}

test "wimmer-rv: a 7th VPU float entry param bails (matched shared limit, not a miscompile)" {
    const allocator = std.testing.allocator;

    // Shared Wimmer path: must bail rather than alias fa6/fa7 onto a VPU vector.
    var wim = try buildSevenFloatParams(allocator);
    defer wim.deinit();
    try ir.legalize.legalize(allocator, &wim);
    try std.testing.expectError(error.Unsupported, isel.compileFunctionWimmerRiscv(allocator, &wim, true));

    // Native path rejects the SAME shape (isel.zig `if (vpu and float_arg >= 6)`), so this is a
    // matched feature-limit both paths decline, not a Wimmer-only gap.
    var nat = try buildSevenFloatParams(allocator);
    defer nat.deinit();
    try ir.legalize.legalize(allocator, &nat);
    try std.testing.expectError(error.Unsupported, isel.compileFunction(allocator, &nat, .{ .vpu = true }));

    // Sanity: with SIX vpu float params (fa0..fa5, all outside the vector partition) the Wimmer path
    // does NOT bail on this account - it compiles fine.
    var six = Function.init(allocator);
    defer six.deinit();
    {
        const f32_t = try six.types.intern(.{ .float = .f32 });
        const b = try six.appendBlock();
        var ps: [6]Value = undefined;
        for (0..6) |i| ps[i] = try six.appendBlockParam(b, f32_t);
        var acc = ps[0];
        for (ps[1..]) |p| acc = try six.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
        six.setTerminator(b, .{ .ret = acc });
    }
    try ir.legalize.legalize(allocator, &six);
    var compiled = try isel.compileFunctionWimmerRiscv(allocator, &six, true);
    compiled.deinit(allocator);
}

// ---------------------------------------------------------------------------
// N. Address-mode folding under register pressure (SP3 Task 3).
//    The FOLD-AWARE Wimmer entry (`compileFunctionWimmerRiscvFold`) runs the pre-allocation IR
//    rewrite (`applyFoldRewriteRiscv`) so folding is SOUND under the fold-blind shared allocator: it
//    repoints each folded mem op's ptr to the fold base and DCEs the dead address-add BEFORE the scan,
//    so `base` stays live to the load/store. Each shape keeps `base` live across a pressured region, so
//    a fold-blind allocation that let `base` die at the add would reuse its register and the folded
//    `off(base)` would read a stale register (the SP1/SP2 trap). We compile the same function through
//    the fold-on reference (`selectFunction` via `harness.runFunc`) AND the fold-aware Wimmer path, run
//    both under qemu-riscv64, assert bit-identical results, and prove the fold actually FIRED on the
//    Wimmer-fold output via disassembly. The plain `compileFunctionWimmerRiscv` differential above
//    stays on `empty_fold` (fold OFF) for byte-identity, so this is the only fold-on differential.
// ---------------------------------------------------------------------------

const fold_pressure = 24;

/// Count loads/stores in `code` that carry a NONZERO signed-12 displacement off a base other than sp
/// (x2): exactly a folded `ld rd, off(base)` / `sd rs2, off(base)` (and the fp/sub-word forms) with
/// off != 0 and base != sp. A spill addresses off sp, so filtering base != x2 isolates the fold. Zero
/// unless address folding fired. Mirrors `tests/addrfold.zig`'s `foldedMemOps` word decoder.
fn foldedMemOps(code: []const u32) usize {
    var count: usize = 0;
    for (code) |w| {
        const opcode = w & 0x7f;
        const rs1 = (w >> 15) & 0x1f;
        if (rs1 == 2) continue; // sp-relative: a spill slot, not a fold
        const is_load = opcode == 0x03 or opcode == 0x07; // LOAD (int) / LOAD-FP
        const is_store = opcode == 0x23 or opcode == 0x27; // STORE (int) / STORE-FP
        if (is_load) {
            const imm: i32 = @as(i32, @bitCast(w)) >> 20; // sign-extended I-immediate
            if (imm != 0) count += 1;
        } else if (is_store) {
            const hi: i32 = @as(i32, @bitCast(w)) >> 25; // sign-extended imm[11:5]
            const lo: i32 = @intCast((w >> 7) & 0x1f); // imm[4:0]
            if (((hi << 5) | lo) != 0) count += 1;
        }
    }
    return count;
}

/// Compile `func` through the FOLD-AWARE shared Wimmer path (address folding ON via the pre-allocation
/// rewrite) and run it under qemu, returning a0. Legalizes first, matching the reference pipeline.
fn runWimmerFold(io: std.Io, allocator: std.mem.Allocator, func: *Function, args: []const i64) !i64 {
    try ir.legalize.legalize(allocator, func);
    var compiled = try isel.compileFunctionWimmerRiscvFold(allocator, func, false);
    defer compiled.deinit(allocator);
    return harness.runCode(io, allocator, compiled.code, args, harness.qemu_user);
}

/// Compile `func` through the fold-aware Wimmer path and count the folded mem ops in its output, so a
/// test can prove the fold fired on the exact code it also executes.
fn wimmerFoldFolds(allocator: std.mem.Allocator, func: *Function) !usize {
    try ir.legalize.legalize(allocator, func);
    var compiled = try isel.compileFunctionWimmerRiscvFold(allocator, func, false);
    defer compiled.deinit(allocator);
    return foldedMemOps(compiled.code);
}

/// Run the fold-on reference and the fold-aware Wimmer path on two fresh copies of `build()` for every
/// input, asserting the integer results match, and first prove the fold FIRED on the Wimmer-fold output
/// (a zero would mean the analysis/rewrite silently stopped folding, hiding the very case this guards).
/// `expected` is a hand-derived ground-truth oracle mirroring the function's own semantics (see each
/// call site): `compileFunction`/`selectFunction` now IS the Wimmer path (fold included), so
/// `expected == got` catches a fold-rewrite miscompile shared by both entries, while `ref == got` stays
/// as a belt-and-suspenders check that the two entry points agree.
fn expectFoldMatch(io: std.Io, comptime build: fn (std.mem.Allocator) anyerror!Function, comptime expected: fn ([]const i64) i64, inputs: []const []const i64) !void {
    const allocator = std.testing.allocator;
    {
        // Compile-only (no qemu), so this never skips: it proves the fold fired on the exact code the
        // input loop below executes.
        var probe = try build(allocator);
        defer probe.deinit();
        try std.testing.expect(try wimmerFoldFolds(allocator, &probe) >= 1);
    }
    for (inputs) |args| {
        var ref_func = try build(allocator);
        defer ref_func.deinit();
        var wim_func = try build(allocator);
        defer wim_func.deinit();

        const ref = harness.runFunc(io, allocator, &ref_func, args, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        const got = runWimmerFold(io, allocator, &wim_func, args) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(expected(args), got);
        try std.testing.expectEqual(ref, got);
    }
}

/// f(arg, cond): ENTRY stores `arg` into buf1 and forms `p = buf0 + 8` (= buf1's address, a DEAD add
/// whose only use is the successor load). On cond > 0, then_b builds `fold_pressure` live temps BEFORE
/// loading [p] (which folds to `8(buf0)`), so a fold-blind allocation that let buf0 die at the add
/// would hand buf0's register to a temp and the folded load would read garbage. buf0's cross-branch
/// liveness flows ONLY through the folded load's base, so the rewrite must reroute it. Expected(cond>0)
/// = arg + sum_{k=1..P}(cond + k); Expected(cond<=0) = cond.
fn buildFoldLoadPressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const arg = try func.appendBlockParam(entry, i64_t);
    const cond = try func.appendBlockParam(entry, i64_t);

    const buf0 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    const buf1 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    try func.appendStore(entry, arg, buf1); // buf1 = arg (own pointer, references buf1)
    const p = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf0, .imm = 8 } }); // dead add = buf1 addr
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = cond, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });

    var vals: [fold_pressure]Value = undefined;
    for (0..fold_pressure) |k| {
        vals[k] = try func.appendInst(then_b, i64_t, .{ .arith_imm = .{ .op = .add, .lhs = cond, .imm = @intCast(k + 1) } });
    }
    const w = try func.appendInst(then_b, i64_t, .{ .load = .{ .ptr = p } }); // folds to 8(buf0) = arg
    var acc = w;
    for (0..fold_pressure) |k| acc = try func.appendInst(then_b, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vals[k] } });
    func.setTerminator(then_b, .{ .ret = acc });
    func.setTerminator(else_b, .{ .ret = cond });
    return func;
}

/// The exact value `buildFoldLoadPressure` computes, mirrored in Zig: sum_{k=1}^{24}(cond+k) =
/// 24*cond + 300 (sum_{1..24} = 300, fold_pressure=24), added to arg when cond > 0, else cond itself.
/// i64 wrapping.
fn foldLoadPressureExpected(args: []const i64) i64 {
    const arg = args[0];
    const cond = args[1];
    if (cond > 0) {
        const tail = (24 *% cond) +% 300;
        return arg +% tail;
    }
    return cond;
}

test "wimmer-rv: a folded load whose base is live across pressure matches the reference and folds" {
    const inputs = [_][]const i64{ &.{ 21, 1 }, &.{ 20, 0 }, &.{ 19, -1 }, &.{ 25, 5 }, &.{ 120, 100 }, &.{ 13, -7 }, &.{ 62, 42 } };
    try expectFoldMatch(std.testing.io, buildFoldLoadPressure, foldLoadPressureExpected, &inputs);
}

/// f(arg): ENTRY inits buf0, forms `p = buf0 + 8` (= buf1's address, a DEAD add whose only use is the
/// folded store), builds `fold_pressure` live temps from arg, reduces them to `sum`, then STORES sum
/// through the folded address `8(buf0)` (= buf1) and reads buf1 back (own pointer, off 0). The pressure
/// sits between the add and the folded store, so a fold-blind allocation that let buf0 die at the add
/// would reuse its register and the folded store would write to a garbage address (a wrong readback or
/// a fault). Expected = sum_{k=1..P}(arg + k) = P*arg + P(P+1)/2.
fn buildFoldStorePressure(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const arg = try func.appendBlockParam(e, i64_t);

    const buf0 = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    const buf1 = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    const z = try func.appendInst(e, i64_t, .{ .iconst = 0 });
    try func.appendStore(e, z, buf0); // init buf0 (own pointer, references buf0)
    const p = try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf0, .imm = 8 } }); // dead add = buf1 addr
    var vals: [fold_pressure]Value = undefined;
    for (0..fold_pressure) |k| vals[k] = try func.appendArithImm(e, i64_t, .add, arg, @intCast(k + 1));
    var sum = vals[0];
    for (1..fold_pressure) |k| sum = try func.appendInst(e, i64_t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = vals[k] } });
    try func.appendStore(e, sum, p); // folds to 8(buf0) = buf1
    const r = try func.appendInst(e, i64_t, .{ .load = .{ .ptr = buf1 } }); // reads back (own pointer, off 0)
    func.setTerminator(e, .{ .ret = r });
    return func;
}

/// The exact value `buildFoldStorePressure` computes, mirrored in Zig: sum_{k=1}^{24}(arg+k) =
/// 24*arg + 300 (sum_{1..24} = 300, fold_pressure=24). i64 wrapping.
fn foldStorePressureExpected(args: []const i64) i64 {
    const arg = args[0];
    return (24 *% arg) +% 300;
}

test "wimmer-rv: a folded store whose base is live across pressure matches the reference and folds" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{7}, &.{-3}, &.{100}, &.{-1000}, &.{123456} };
    try expectFoldMatch(std.testing.io, buildFoldStorePressure, foldStorePressureExpected, &inputs);
}
