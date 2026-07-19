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
/// each side gets its own untouched function (both pipelines mutate the IR in place).
fn expectIntMatch(io: std.Io, comptime build: fn (std.mem.Allocator) anyerror!Function, inputs: []const []const i64) !void {
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

test "wimmer-rv: straight-line int arithmetic matches" {
    const inputs = [_][]const i64{ &.{ 1, 2, 3 }, &.{ 0, 0, 0 }, &.{ -5, 7, -9 }, &.{ 100, -200, 300 }, &.{ 123456, -1, 2 } };
    try expectIntMatch(std.testing.io, buildStraightLine, &inputs);
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

test "wimmer-rv: an int register-pressure kernel matches" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{7}, &.{-3}, &.{100}, &.{-1000}, &.{123456} };
    try expectIntMatch(std.testing.io, buildIntPressure, &inputs);
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

test "wimmer-rv: a loop-carried int sum across a pressured body matches" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{3}, &.{7}, &.{12} };
    try expectIntMatch(std.testing.io, buildLoopSum, &inputs);
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

test "wimmer-rv: a diamond with an int value live on both paths matches" {
    const inputs = [_][]const i64{ &.{0}, &.{1}, &.{-1}, &.{5}, &.{-9}, &.{100} };
    try expectIntMatch(std.testing.io, buildDiamond, &inputs);
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
/// mutate the IR in place). `fa_sets` are the fa0.. argument-bit vectors to try.
fn expectFloatMatch(io: std.Io, comptime build: fn (std.mem.Allocator) anyerror!Function, fa_sets: []const []const u64) !void {
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

test "wimmer-rv: an RVV vector arithmetic function matches" {
    const a_vals = [_]f32{ 0.0, 1.0, 3.0, -7.0, 100.0 };
    var sets: [a_vals.len][1]u64 = undefined;
    var ptrs: [a_vals.len][]const u64 = undefined;
    for (a_vals, 0..) |v, i| {
        sets[i] = .{@as(u32, @bitCast(v))};
        ptrs[i] = &sets[i];
    }
    try expectFloatMatch(std.testing.io, buildVecArith, &ptrs);
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

test "wimmer-rv: RVV vector register pressure spills a vector and reloads it" {
    const a_vals = [_]f32{ 0.0, 1.0, 10.0, -4.0 };
    var sets: [a_vals.len][1]u64 = undefined;
    var ptrs: [a_vals.len][]const u64 = undefined;
    for (a_vals, 0..) |v, i| {
        sets[i] = .{@as(u32, @bitCast(v))};
        ptrs[i] = &sets[i];
    }
    try expectFloatMatch(std.testing.io, buildVecPressure, &ptrs);
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

test "wimmer-rv: a vector live across a call spills across the call (not an error)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const a_vals = [_]f32{ 0.0, 1.0, 5.0, -3.0, 42.0 };

    // The NATIVE allocator STILL bails `error.Unsupported` on a vector live across a call (every RVV
    // register is caller-saved and it has no vector split), so there is no native reference to diff
    // against - that limitation is exactly what the shared Wimmer path removes here. The oracle is
    // therefore the mathematically-correct result, computed in f32 in the SAME op order the IR builds:
    // lane 0 of vv = a + 5, plus the call result g(10) = 11.
    for (a_vals) |a_val| {
        const fargs = [_]u64{@as(u32, @bitCast(a_val))};
        const lane0: f32 = a_val + 5.0;
        const want: f32 = lane0 + 11.0;

        // Confirm the native path really cannot compile this shape (documents why there is no diff).
        var native_probe = try buildVecAcrossCall(allocator);
        defer native_probe.deinit();
        try ir.legalize.legalize(allocator, &native_probe);
        try std.testing.expectError(error.Unsupported, isel.compileFunction(allocator, &native_probe, .{}));

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
