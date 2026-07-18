//! f16 software emulation differentials, executed on qemu-riscv64 (the oracle). riscv64 has no
//! hardware half (no Zfh), so an f16 is held as its f32 WIDENING in a float register and every
//! boundary rounds via the inline software routines in isel.zig (`emitHalfToFloat` extends f16 ->
//! f32 exactly, `emitFloatToHalf` truncates f32 -> f16 with round-to-nearest-even). These tests
//! prove the generated code is bit-exact against Zig's own `@as(f16, ...)` / `@as(f32, @as(f16,
//! ...))` reference, which lowers f16 the same way (promote to f32, operate, round back).
//!
//! Coverage:
//!   - EXHAUSTIVE extend: all 65536 half bit patterns widened to f32, folded into a checksum and
//!     matched against the Zig reference over the same 65536 values (one qemu run).
//!   - COMPREHENSIVE truncate: ~1M structured f32 patterns (all exponents, all 10 kept mantissa
//!     bits, the guard bit, and the sign) run through f32 -> f16 -> f32, folded and matched. NaN
//!     inputs are excluded from both checksums (their exact half payload is target-defined) and
//!     covered by dedicated "maps to a NaN" assertions instead.
//!   - Explicit single-value assertions for the categorical edges (RNE ties both directions,
//!     subnormals, overflow to inf, the 65504/65520 boundary, +-0, inf, NaN, normals).
//!   - Full differentials: load/store round-trip, add/sub/mul/div (incl a non-half-representable
//!     product), convert both directions, int<->f16, and an f16 register-spill case.
//!
//! Skips (not fails) when qemu-riscv64 is not on PATH.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

fn f16Bits(x: f16) u16 {
    return @bitCast(x);
}

fn widen(x: f16) u32 {
    return @bitCast(@as(f32, x));
}

/// `f(in: i64) -> i64`: store the low 16 bits of `in` as an IEEE half in a stack slot, load it as
/// f16 (lhu + software extend), widen to f32 (identity, held-as-f32), then read the f32 bits back
/// out as an integer. Exercises the f16 LOAD and the f16->f32 convert. The returned value is the
/// f32 bits sign-extended (the reload is an `lw`), so compare the low 32 bits.
fn buildExtendFn(func: *Function) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const in = try func.appendBlockParam(b, i64_t);
    const slot = try func.appendInst(b, try func.types.intern(.ptr), .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(b, in, slot); // sd: low 16 bits are the half pattern
    const h = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot } });
    const f = try func.appendInst(b, f32_t, .{ .convert = .{ .value = h } });
    try func.appendStore(b, f, slot);
    const bits = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = bits });
}

/// `f(in: i64) -> i64`: reinterpret the low 32 bits of `in` as an f32, round it to f16 (software
/// truncate + extend, held-as-f32), then read the resulting f32-widening bits back as an integer.
/// Exercises the f32->f16 convert (the round-to-nearest-even truncate). Compare the low 32 bits.
fn buildTruncFn(func: *Function) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const in = try func.appendBlockParam(b, i64_t);
    const slot = try func.appendInst(b, try func.types.intern(.ptr), .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(b, in, slot);
    const x = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = slot } });
    const h = try func.appendInst(b, f16_t, .{ .convert = .{ .value = x } });
    const f = try func.appendInst(b, f32_t, .{ .convert = .{ .value = h } });
    try func.appendStore(b, f, slot);
    const bits = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = bits });
}

fn runIntFn(comptime buildFn: fn (*Function) anyerror!void, in: u32) !?u32 {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildFn(&func);
    // Sign-extend the 32-bit pattern through i32: the qemu integer stub materializes args as i32,
    // and the function stores only the low 32 bits, so a bit-31-set pattern must arrive this way.
    const arg: i64 = @as(i32, @bitCast(in));
    const got = harness.runFunc(std.testing.io, allocator, &func, &.{arg}, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return null,
        else => return e,
    };
    return @truncate(@as(u64, @bitCast(got)));
}

/// `f() -> i64`: loop i over 0..65536, widen the half whose bits are `i` to f32 (via the load
/// path), and fold the f32 bits into a rolling checksum `acc = acc*33 + bits`. NaN halves (exp 31,
/// mantissa != 0) fold 0 instead, so the checksum is independent of the target-defined NaN payload.
/// The host computes the same checksum over Zig's `@as(f32, @as(f16, ...))` and the two must match.
fn buildExtendSweepFn(func: *Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);

    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const i = try func.appendBlockParam(loop, i32_t);
    const acc = try func.appendBlockParam(loop, i32_t);
    // The stack scratch pointer is threaded through the loop as a block param. A value defined in
    // the entry and used only inside the loop body would be live across the back-edge, which the
    // riscv64 register allocator does not keep resident; threading it as a param avoids that.
    const lslot = try func.appendBlockParam(loop, ptr_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const slot = try func.appendBlockParam(body, ptr_t);
    const racc = try func.appendBlockParam(done, i32_t);

    const slot0 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero, slot0 });

    const n = try func.appendInst(loop, i32_t, .{ .iconst = 65536 });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc, lslot } }, .{ .target = done, .args = &.{acc} });

    try func.appendStore(body, bi, slot); // low 16 bits = half pattern
    const h = try func.appendInst(body, f16_t, .{ .load = .{ .ptr = slot } });
    const f = try func.appendInst(body, f32_t, .{ .convert = .{ .value = h } });
    try func.appendStore(body, f, slot);
    const bits = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = slot } });
    // NaN exclusion: absh = bi & 0x7fff; isnan = absh > 0x7c00.
    const mask = try func.appendInst(body, i32_t, .{ .iconst = 0x7FFF });
    const absh = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .bit_and, .lhs = bi, .rhs = mask } });
    const c7c00 = try func.appendInst(body, i32_t, .{ .iconst = 0x7C00 });
    const isnan = try func.appendInst(body, bool_t, .{ .icmp = .{ .op = .gt, .lhs = absh, .rhs = c7c00 } });
    const zero2 = try func.appendInst(body, i32_t, .{ .iconst = 0 });
    const contrib = try func.appendInst(body, i32_t, .{ .select = .{ .cond = isnan, .then = zero2, .@"else" = bits } });
    const c33 = try func.appendInst(body, i32_t, .{ .iconst = 33 });
    const m = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = bacc, .rhs = c33 } });
    const acc2 = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = m, .rhs = contrib } });
    const inext = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ inext, acc2, slot });

    func.setTerminator(done, .{ .ret = racc });
}

/// `f() -> i64`: loop i over 0..(1<<20), form the f32 pattern `i << 12` (sweeping all exponents,
/// all 10 kept mantissa bits, the guard bit, and the sign), round it to f16 and widen back, then
/// fold the widened bits into the same rolling checksum. NaN inputs (abs > 0x7f800000) fold 0.
fn buildTruncSweepFn(func: *Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);

    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const i = try func.appendBlockParam(loop, i32_t);
    const acc = try func.appendBlockParam(loop, i32_t);
    const lslot = try func.appendBlockParam(loop, ptr_t); // threaded (see the extend sweep)
    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const slot = try func.appendBlockParam(body, ptr_t);
    const racc = try func.appendBlockParam(done, i32_t);

    const slot0 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero, slot0 });

    const n = try func.appendInst(loop, i32_t, .{ .iconst = 1 << 20 });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc, lslot } }, .{ .target = done, .args = &.{acc} });

    const pat = try func.appendArithImm(body, i32_t, .shl, bi, 12);
    try func.appendStore(body, pat, slot);
    const x = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = slot } });
    const h = try func.appendInst(body, f16_t, .{ .convert = .{ .value = x } });
    const f = try func.appendInst(body, f32_t, .{ .convert = .{ .value = h } });
    try func.appendStore(body, f, slot);
    const bits = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = slot } });
    // NaN exclusion: absv = pat & 0x7fffffff; isnan = absv > 0x7f800000.
    const absmask = try func.appendInst(body, i32_t, .{ .iconst = 0x7FFFFFFF });
    const absv = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .bit_and, .lhs = pat, .rhs = absmask } });
    const cinf = try func.appendInst(body, i32_t, .{ .iconst = 0x7F800000 });
    const isnan = try func.appendInst(body, bool_t, .{ .icmp = .{ .op = .gt, .lhs = absv, .rhs = cinf } });
    const zero2 = try func.appendInst(body, i32_t, .{ .iconst = 0 });
    const contrib = try func.appendInst(body, i32_t, .{ .select = .{ .cond = isnan, .then = zero2, .@"else" = bits } });
    const c33 = try func.appendInst(body, i32_t, .{ .iconst = 33 });
    const m = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = bacc, .rhs = c33 } });
    const acc2 = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = m, .rhs = contrib } });
    const inext = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ inext, acc2, slot });

    func.setTerminator(done, .{ .ret = racc });
}

test "f16 extend: all 65536 half patterns widen to f32 bit-exact vs @as(f32, @as(f16, x)) (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildExtendSweepFn(&func);
    const got = harness.runFunc(std.testing.io, allocator, &func, &.{}, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    // Reference: the same rolling checksum over Zig's own f16->f32 widening. NaN halves fold 0, and
    // the device reload is an `lw` (sign-extension), so each contribution is a sign-extended i32.
    var acc: i64 = 0;
    var u: u32 = 0;
    while (u < 65536) : (u += 1) {
        const bits = widen(@bitCast(@as(u16, @intCast(u))));
        const is_nan = (u & 0x7FFF) > 0x7C00;
        const contrib: i64 = if (is_nan) 0 else @as(i32, @bitCast(bits));
        acc = acc *% 33 +% contrib;
    }
    try std.testing.expectEqual(acc, got);
}

test "f16 truncate: ~1M structured f32 patterns round to f16 bit-exact vs @as(f16, x) (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildTruncSweepFn(&func);
    const got = harness.runFunc(std.testing.io, allocator, &func, &.{}, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    var acc: i64 = 0;
    var i: u64 = 0;
    while (i < (1 << 20)) : (i += 1) {
        const pat: u32 = @truncate(i << 12);
        const x: f32 = @bitCast(pat);
        const is_nan = (pat & 0x7FFFFFFF) > 0x7F800000;
        var contrib: i64 = 0;
        if (!is_nan) {
            const h: f16 = @floatCast(x); // round-to-nearest-even truncate
            contrib = @as(i32, @bitCast(widen(h)));
        }
        acc = acc *% 33 +% contrib;
    }
    try std.testing.expectEqual(acc, got);
}

test "f16 extend: categorical single values widen exactly (qemu-riscv64)" {
    const cases = [_]f16{
        0.0, -0.0, 1.0, 1.5, -2.25,
        65504.0, 0.00006103515625, // smallest normal
        0.000000059604645, // smallest subnormal (2^-24)
        std.math.inf(f16),
        -std.math.inf(f16),
        3.140625,
    };
    for (cases) |x| {
        const got = (try runIntFn(buildExtendFn, f16Bits(x))) orelse return error.SkipZigTest;
        try std.testing.expectEqual(widen(x), got);
    }
}

test "f16 extend: a NaN half widens to a NaN f32 with nonzero mantissa (qemu-riscv64)" {
    const nan_half: u16 = 0x7E00; // quiet NaN half
    const got = (try runIntFn(buildExtendFn, nan_half)) orelse return error.SkipZigTest;
    const exp = (got >> 23) & 0xFF;
    const mant = got & 0x7FFFFF;
    try std.testing.expectEqual(@as(u32, 0xFF), exp); // all-ones exponent
    try std.testing.expect(mant != 0); // nonzero mantissa == NaN
}

test "f16 truncate: categorical single values round to nearest-even bit-exact (qemu-riscv64)" {
    // f32 inputs chosen to exercise every rounding path. Compared against Zig's own f32->f16.
    const inputs = [_]f32{
        0.0, -0.0, 1.0, -1.0,
        1.00048828125, // exactly halfway between 1.0 (even) and the next half -> rounds down to 1.0
        1.0009765625 + 0.00048828125, // halfway between two halves, odd side -> rounds up to even
        65504.0, // largest finite half (exact)
        65520.0, // exactly halfway 65504<->65536, ties to even == inf
        65519.0, // just below the tie, rounds to the max finite half
        65536.0, // overflows to +inf
        3.14159, // normal, rounds
        0.00006097555, // rounds into an f16 subnormal
        5.0e-8, // underflows toward zero
        std.math.inf(f32),
        -std.math.inf(f32),
    };
    for (inputs) |x| {
        const in: u32 = @bitCast(x);
        const got = (try runIntFn(buildTruncFn, in)) orelse return error.SkipZigTest;
        const want = widen(@as(f16, @floatCast(x)));
        try std.testing.expectEqual(want, got);
    }
}

test "f16 truncate: a NaN f32 maps to a NaN half with nonzero mantissa (qemu-riscv64)" {
    const nan_f32: u32 = 0x7FC00000;
    const got = (try runIntFn(buildTruncFn, nan_f32)) orelse return error.SkipZigTest; // f32-widening bits
    // The widened half must itself be a NaN: all-ones f32 exponent and nonzero mantissa.
    const exp = (got >> 23) & 0xFF;
    const mant = got & 0x7FFFFF;
    try std.testing.expectEqual(@as(u32, 0xFF), exp);
    try std.testing.expect(mant != 0);
}

/// `f(a: f16, b: f16) -> f16`: a single f16 binary op, both operands and the result in the
/// held-as-f32 convention (passed/returned as their f32 widening through fa0/fa1/fa0).
fn buildBinaryFn(func: *Function, op: ir.function.BinOp) !void {
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f16_t);
    const c = try func.appendBlockParam(b, f16_t);
    const r = try func.appendInst(b, f16_t, .{ .arith = .{ .op = op, .lhs = a, .rhs = c } });
    func.setTerminator(b, .{ .ret = r });
}

fn runBinary(op: ir.function.BinOp, a: f16, b: f16) !?f16 {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildBinaryFn(&func, op);
    const fargs = [_]u64{ widen(a), widen(b) };
    const got = harness.runFuncFloat(std.testing.io, allocator, &func, false, &fargs, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return null,
        else => return e,
    };
    const bits: u32 = @truncate(got);
    return @floatCast(@as(f32, @bitCast(bits))); // the widening back to f16 is exact
}

test "f16 add/sub/mul/div match Zig's per-op half rounding, bit-exact (qemu-riscv64)" {
    const cases = [_]struct { a: f16, b: f16 }{
        .{ .a = 1.5, .b = 2.25 },
        .{ .a = -3.5, .b = 0.75 },
        .{ .a = 0.1, .b = 0.2 }, // not exact in f16, so results round
        .{ .a = 100.0, .b = 7.0 },
    };
    for (cases) |c| {
        const add = (try runBinary(.add, c.a, c.b)) orelse return error.SkipZigTest;
        try std.testing.expectEqual(f16Bits(c.a + c.b), f16Bits(add));
        try std.testing.expectEqual(f16Bits(c.a - c.b), f16Bits((try runBinary(.sub, c.a, c.b)).?));
        try std.testing.expectEqual(f16Bits(c.a * c.b), f16Bits((try runBinary(.mul, c.a, c.b)).?));
        try std.testing.expectEqual(f16Bits(c.a / c.b), f16Bits((try runBinary(.div, c.a, c.b)).?));
    }
}

test "f16 multiply rounds its result to nearest-even half, not a raw f32 product (qemu-riscv64)" {
    // a*a whose exact f32 product is NOT representable in f16, so the half result must round. Proves
    // the arith path narrows to half per op rather than leaving the wider f32 product in place.
    const a: f16 = 1.0009765625; // 1 + 2^-10, itself exactly representable
    const got = (try runBinary(.mul, a, a)) orelse return error.SkipZigTest;
    try std.testing.expectEqual(f16Bits(a * a), f16Bits(got));
    const exact_f32: f32 = @as(f32, a) * @as(f32, a);
    try std.testing.expect(@as(f32, got) != exact_f32); // rounding to half changed the value
}

/// `f(a: f16) -> f16`: build 30 simultaneously-live f16 values `a + i` (each an f16 add that rounds
/// to half), then fold them with f16 adds. 30 exceeds the caller-saved float temps, forcing several
/// f16 values to spill to the stack and reload; the sum is only bit-exact if each spilled half
/// reloads with its held-as-f32 value intact. With `a` a small integer every intermediate is exact.
fn buildSpillSumFn(func: *Function) !void {
    const n_live = 30;
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f16_t);
    var vals: [n_live]Value = undefined;
    for (0..n_live) |i| {
        const ci = try func.appendInst(b, i32_t, .{ .iconst = @intCast(i) });
        const cf = try func.appendInst(b, f16_t, .{ .convert = .{ .value = ci } });
        vals[i] = try func.appendInst(b, f16_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = cf } });
    }
    var acc = vals[0];
    for (vals[1..]) |v| acc = try func.appendInst(b, f16_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(b, .{ .ret = acc });
}

test "f16 survives register spilling bit-exact (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSpillSumFn(&func);
    // Compile with selectFunction directly (no scheduler) so the wide live range - and thus the
    // spill - is preserved.
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);

    const a_val: f16 = 100.0;
    const fargs = [_]u64{widen(a_val)};
    const got_bits = harness.runCompiledFloat(std.testing.io, allocator, code, false, &fargs, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    const got: f16 = @floatCast(@as(f32, @bitCast(@as(u32, @truncate(got_bits)))));
    // Reference: the same left-fold with per-op half rounding (f16 add is not associative).
    var vals: [30]f16 = undefined;
    for (0..30) |i| vals[i] = a_val + @as(f16, @floatFromInt(i));
    var want: f16 = vals[0];
    for (vals[1..]) |v| want += v;
    try std.testing.expectEqual(f16Bits(want), f16Bits(got));
}

/// `f(x: i32) -> i64`: convert a 32-bit signed integer to f16 (int -> f32 then round to half), then
/// read the held-as-f32 widening bits back out (via a memory round-trip) so the integer ABI carries
/// the result. Returns the f32-widening bits; compare the low 32 bits.
fn buildIntToHalfFn(func: *Function) !void {
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const slot = try func.appendInst(b, try func.types.intern(.ptr), .{ .alloca = .{ .elem = i32_t } });
    const h = try func.appendInst(b, f16_t, .{ .convert = .{ .value = x } });
    const f = try func.appendInst(b, f32_t, .{ .convert = .{ .value = h } });
    try func.appendStore(b, f, slot);
    const bits = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = bits });
}

/// `f(in: i64) -> i64`: take a half bit pattern in the low 16 bits of `in`, load it as an f16 (lhu +
/// extend), then convert to a 32-bit signed integer (truncate toward zero). Integer ABI throughout.
fn buildHalfToIntFn(func: *Function) !void {
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const b = try func.appendBlock();
    const in = try func.appendBlockParam(b, i64_t);
    const slot = try func.appendInst(b, try func.types.intern(.ptr), .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(b, in, slot);
    const h = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot } });
    const r = try func.appendInst(b, i32_t, .{ .convert = .{ .value = h } });
    func.setTerminator(b, .{ .ret = r });
}

test "int <-> f16 conversions match Zig, bit-exact (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    // int -> f16: 2049 is not exactly representable and must round to 2048; -7 is exact.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildIntToHalfFn(&func);
        for ([_]i32{ 2049, -7, 0, 1000 }) |v| {
            const got = harness.runFunc(std.testing.io, allocator, &func, &.{@as(i64, v)}, harness.qemu_user) catch |e| switch (e) {
                error.SkipZigTest => return error.SkipZigTest,
                else => return e,
            };
            const want = widen(@as(f16, @floatFromInt(v)));
            try std.testing.expectEqual(want, @as(u32, @truncate(@as(u64, @bitCast(got)))));
        }
    }
    // f16 -> int: truncates toward zero.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildHalfToIntFn(&func);
        for ([_]f16{ 3.5, -3.9, 100.0, 0.0 }) |x| {
            const got = harness.runFunc(std.testing.io, allocator, &func, &.{@as(i64, @intCast(f16Bits(x)))}, harness.qemu_user) catch |e| switch (e) {
                error.SkipZigTest => return error.SkipZigTest,
                else => return e,
            };
            const want: i32 = @intFromFloat(x);
            try std.testing.expectEqual(@as(i64, want), got);
        }
    }
}
