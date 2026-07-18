//! Differential JIT oracle for the INT8 dot-product vectorizer. For a scalar `sum a[i]*b[i]` reduction
//! over an int8 array, we build two identical functions, run `dotprod.run` (recognize + vectorize to
//! SDOT/UDOT) on one under the ampere-altra model, JIT both on the host, and require bit-identical
//! results for a spread of lengths, INCLUDING non-multiples of 16 to exercise the remainder loop. The
//! vectorized function must compute exactly what the scalar one does; any divergence is a miscompile
//! in the transform, not a test to relax. A SIGNED (i8 -> SDOT) and an UNSIGNED (u8 -> UDOT) version
//! prove both lowerings end to end.
//!
//! The `dot` op only lowers on aarch64+dotprod, so these tests run on an aarch64 host and skip
//! elsewhere.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;
const Value = ir.function.Value;

/// The wide out-of-order aarch64 model with the dotprod feature, the one the pass transforms for.
fn ampere() *const opt.microarch.Model {
    return opt.microarch.modelFor(.@"ampere-altra");
}

/// Build `fn(a: ptr, b: ptr, n: i32) i32` computing `sum_{i<n} convert_i32(a[i]) * convert_i32(b[i])`
/// as the exact scalar dot-reduction the vectorizer recognizes: an int8 (signed) or u8 (unsigned)
/// unit-stride reduction with an i32 accumulator and two loop-carried element pointers advancing by
/// one byte per iteration.
fn buildDotLoop(func: *Function, sign: std.builtin.Signedness) !void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const e8 = try func.types.intern(.{ .int = .{ .signedness = sign, .bits = 8 } });

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const a_ptr = try func.appendBlockParam(entry, ptr_t);
    const b_ptr = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);

    const i = try func.appendBlockParam(header, i32_t);
    const acc = try func.appendBlockParam(header, i32_t);
    const pa = try func.appendBlockParam(header, ptr_t);
    const pb = try func.appendBlockParam(header, ptr_t);

    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const bpa = try func.appendBlockParam(body, ptr_t);
    const bpb = try func.appendBlockParam(body, ptr_t);

    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ zero, zero, a_ptr, b_ptr });

    const cmp = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(header, cmp, .{ .target = body, .args = &.{ i, acc, pa, pb } }, .{ .target = exit, .args = &.{} });

    const la = try func.appendInst(body, e8, .{ .load = .{ .ptr = bpa } });
    const lb = try func.appendInst(body, e8, .{ .load = .{ .ptr = bpb } });
    const ca = try func.appendInst(body, i32_t, .{ .convert = .{ .value = la } });
    const cb = try func.appendInst(body, i32_t, .{ .convert = .{ .value = lb } });
    const prod = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = ca, .rhs = cb } });
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = prod } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    const npa = try func.appendArithImm(body, ptr_t, .add, bpa, 1);
    const npb = try func.appendArithImm(body, ptr_t, .add, bpb, 1);
    try func.setJump(body, header, &.{ ni, nacc, npa, npb });

    func.setTerminator(exit, .{ .ret = acc });
}

/// The lengths exercised, chosen to cover an empty loop, a sub-vector remainder, exact vector
/// multiples, and remainders straddling one and several vector iterations.
const lengths = [_]i32{ 0, 1, 15, 16, 17, 31, 32, 33, 100 };

fn expectDotMatches(comptime Elem: type, sign: std.builtin.Signedness) !void {
    const allocator = std.testing.allocator;

    var scalar = Function.init(allocator);
    defer scalar.deinit();
    try buildDotLoop(&scalar, sign);

    var vectorized = Function.init(allocator);
    defer vectorized.deinit();
    try buildDotLoop(&vectorized, sign);

    const changed = try opt.microarch.dotprod.run(allocator, &vectorized, ampere());
    try std.testing.expect(changed); // the pass must actually transform, so the test is non-vacuous

    // The transform must preserve well-formedness for codegen.
    var diags = try ir.verify.verify(allocator, &vectorized, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_s = try target.native.jitFunction(allocator, &scalar);
    defer buf_s.deinit();
    var buf_v = try target.native.jitFunction(allocator, &vectorized);
    defer buf_v.deinit();

    const Fn = *const fn ([*]const Elem, [*]const Elem, i32) callconv(.c) i32;
    const f_s = buf_s.entry(Fn, 0);
    const f_v = buf_v.entry(Fn, 0);

    // A real backing array with a mix of magnitudes and (for the signed case) signs. 128 elements
    // comfortably covers the largest length plus a full 16-wide vector load.
    var a: [128]Elem = undefined;
    var b: [128]Elem = undefined;
    for (0..128) |k| {
        const ki: i32 = @intCast(k);
        if (Elem == i8) {
            a[k] = @intCast(@mod(ki * 7 - 61, 256) - 128);
            b[k] = @intCast(@mod(ki * 3 + 40, 256) - 128);
        } else {
            a[k] = @intCast(@mod(ki * 5 + 3, 256));
            b[k] = @intCast(@mod(ki * 11 + 7, 256));
        }
    }

    for (lengths) |n| {
        const rs = f_s(&a, &b, n);
        const rv = f_v(&a, &b, n);
        try std.testing.expectEqual(rs, rv);
    }
}

test "dotprod differential: signed i8 reduction (SDOT), remainder across all lengths" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    try expectDotMatches(i8, .signed);
}

test "dotprod differential: unsigned u8 reduction (UDOT), remainder across all lengths" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    try expectDotMatches(u8, .unsigned);
}
