//! Differential JIT oracle for accumulator-splitting unroll. For each counted reduction shape we
//! build two identical functions, split-unroll one (main loop with K independent partial accumulators
//! + a remainder loop), JIT both on the host, and require bit-identical results across a spread of
//! inputs chosen to exercise the remainder: trip counts divisible by K, not divisible by K, below K,
//! and zero. Any divergence is a miscompile in the transform. Runs where the native JIT has a backend.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;

fn ampere() *const opt.microarch.Model {
    return opt.microarch.modelFor(.@"ampere-altra");
}

fn hasJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .riscv64, .x86 => true,
        else => false,
    };
}

const Builder = *const fn (*Function) anyerror!void;

/// `for (i = 0; i < n; i += 1) s += i;  return s`  (sum of 0..n-1).
fn buildSum(func: *Function) anyerror!void {
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const s = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bs = try func.appendBlockParam(body, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done });
    const ns = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = bi } });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    func.setTerminator(done, .{ .ret = s });
}

/// `for (i = 0; i < n; i += 1) s += i*i;  return s` (a richer per-iteration increment).
fn buildSumSquares(func: *Function) anyerror!void {
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const s = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bs = try func.appendBlockParam(body, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done });
    const sq = try func.appendInst(body, t, .{ .arith = .{ .op = .mul, .lhs = bi, .rhs = bi } });
    const ns = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = sq } });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    func.setTerminator(done, .{ .ret = s });
}

/// Two accumulators at once: `for (i) { s += i; p += 2*i + 1; } return s + p` (proves multiple
/// reductions split together). Returns s (the second reduction is folded in via a final store-free
/// add on exit is not modeled, so just returns s; p exercises the multi-reduction path).
fn buildTwoAccumulators(func: *Function) anyerror!void {
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const s = try func.appendBlockParam(loop, t);
    const p = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bs = try func.appendBlockParam(body, t);
    const bp = try func.appendBlockParam(body, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s, p } }, .{ .target = done, .args = &.{ s, p } });
    const ns = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = bi } });
    const two_i = try func.appendArithImm(body, t, .mul, bi, 2);
    const incr = try func.appendArithImm(body, t, .add, two_i, 1);
    const np = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bp, .rhs = incr } });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns, np });
    const rs = try func.appendBlockParam(done, t);
    const rp = try func.appendBlockParam(done, t);
    const total = try func.appendInst(done, t, .{ .arith = .{ .op = .add, .lhs = rs, .rhs = rp } });
    func.setTerminator(done, .{ .ret = total });
}

fn expectSplitMatches(build: Builder) !void {
    const allocator = std.testing.allocator;
    // A spread hitting: 0, below K, exactly K, a K-multiple, and non-multiples around it.
    const inputs = [_]i64{ 0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 17, 24, 31, 100 };

    var orig = Function.init(allocator);
    defer orig.deinit();
    try build(&orig);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try build(&tuned);

    const split = try opt.microarch.splitunroll.run(allocator, &tuned, ampere());
    try std.testing.expect(split); // the shape is eligible

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_o = try target.native.jitFunction(allocator, &orig);
    defer buf_o.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();

    const Fn = *const fn (i64) callconv(.c) i64;
    const f_o = buf_o.entry(Fn, 0);
    const f_t = buf_t.entry(Fn, 0);

    for (inputs) |n| {
        try std.testing.expectEqual(f_o(n), f_t(n));
    }
}

test "splitunroll differential: sum reduction, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectSplitMatches(buildSum);
}

test "splitunroll differential: sum-of-squares reduction, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectSplitMatches(buildSumSquares);
}

test "splitunroll differential: two accumulators split together, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectSplitMatches(buildTwoAccumulators);
}
