//! f16 F16C differentials, executed on qemu-x86_64 (the oracle). x86_64 holds an f16 as its
//! f32 WIDENING in an xmm register and every boundary converts with the F16C hardware ops:
//! a load is `movzx word; movd; vcvtph2ps`, a store is `vcvtps2ph (RNE); movd; mov word`, and
//! every arithmetic result / narrowing convert rounds to nearest-even half with
//! `vcvtps2ph (RNE); vcvtph2ps`. These prove the generated code is bit-exact against Zig's own
//! `@as(f16, ...)` / `@as(f32, @as(f16, ...))` reference (which lowers f16 the same way: promote
//! to f32, operate, round back).
//!
//! F16C runs under qemu-x86_64 because the harness passes `-cpu max`, which exposes f16c (as it
//! already does for AVX). If a future qemu dropped it these tests would SIGILL rather than pass
//! silently.
//!
//! The process exit code is a single byte, so a full 16/32/64-bit result is recovered by running
//! a function once per byte index (a `sel` argument selecting `(bits >> sel*8) & 0xFF`) and
//! reassembling. Every reassembled value is then compared bit-exactly to the Zig reference.
//!
//! Skips (not fails) when qemu-x86_64 is not on PATH.

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

// Interned type shorthands for the builders below.
const Types = struct {
    f16: ir.types.Type,
    f32: ir.types.Type,
    f64: ir.types.Type,
    i32: ir.types.Type,
    u32: ir.types.Type,
    u64: ir.types.Type,
    ptr: ir.types.Type,
    fn init(func: *Function) !Types {
        return .{
            .f16 = try func.types.intern(.{ .float = .f16 }),
            .f32 = try func.types.intern(.{ .float = .f32 }),
            .f64 = try func.types.intern(.{ .float = .f64 }),
            .i32 = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }),
            .u32 = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } }),
            .u64 = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } }),
            .ptr = try func.types.intern(.ptr),
        };
    }
};

/// Emit the byte-select tail for a 32-bit `bits`: `ret (bits >> (sel*8)) & 0xFF`. Running the
/// function with sel = 0..3 recovers the whole 32-bit value under the one-byte exit oracle.
fn selectTail32(func: *Function, b: ir.function.Block, t: Types, bits: Value, sel: Value) !void {
    const sh = try func.appendArithImm(b, t.i32, .shl, sel, 3); // sel * 8
    const shifted = try func.appendInst(b, t.u32, .{ .arith = .{ .op = .shr, .lhs = bits, .rhs = sh } });
    const masked = try func.appendArithImm(b, t.u32, .bit_and, shifted, 0xFF);
    func.setTerminator(b, .{ .ret = masked });
}

/// The 64-bit counterpart, for an f64 result (sel = 0..7).
fn selectTail64(func: *Function, b: ir.function.Block, t: Types, bits: Value, sel: Value) !void {
    const sh = try func.appendArithImm(b, t.i32, .shl, sel, 3);
    const shifted = try func.appendInst(b, t.u64, .{ .arith = .{ .op = .shr, .lhs = bits, .rhs = sh } });
    const masked = try func.appendArithImm(b, t.u64, .bit_and, shifted, 0xFF);
    func.setTerminator(b, .{ .ret = masked });
}

/// Compile `func` and reassemble its `nbytes`-wide result by running it once per byte index with
/// args `{ leading..., sel }`. Returns null if qemu-x86_64 is unavailable.
fn reassemble(func: *Function, leading: []const i64, nbytes: usize) !?u64 {
    var acc: u64 = 0;
    var s: usize = 0;
    while (s < nbytes) : (s += 1) {
        var args: [3]i64 = undefined;
        @memcpy(args[0..leading.len], leading);
        args[leading.len] = @intCast(s);
        const byte = harness.runFunc(std.testing.io, std.testing.allocator, func, args[0 .. leading.len + 1], harness.qemu) catch |e| switch (e) {
            error.SkipZigTest => return null,
            else => return e,
        };
        acc |= @as(u64, byte) << @intCast(s * 8);
    }
    return acc;
}

/// `f(pattern: i32, sel: i32) -> i32`: take a half bit pattern in the low 16 bits, LOAD it as an
/// f16 from an i32 slot, STORE it to an f16 slot (2-byte vcvtps2ph store), LOAD it back, widen to
/// f32, and byte-select the f32-widening bits. Exercises the f16 load and store boundaries.
fn buildLoadStoreFn(func: *Function) !void {
    const t = try Types.init(func);
    const b = try func.appendBlock();
    const pat = try func.appendBlockParam(b, t.i32);
    const sel = try func.appendBlockParam(b, t.i32);
    const slot_i = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.i32 } });
    const slot_h = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.f16 } });
    try func.appendStore(b, pat, slot_i); // low 16 bits = the half pattern
    const h = try func.appendInst(b, t.f16, .{ .load = .{ .ptr = slot_i } });
    try func.appendStore(b, h, slot_h); // f16 store (vcvtps2ph + 16-bit write)
    const h2 = try func.appendInst(b, t.f16, .{ .load = .{ .ptr = slot_h } });
    const f = try func.appendInst(b, t.f32, .{ .convert = .{ .value = h2 } });
    try func.appendStore(b, f, slot_i);
    const bits = try func.appendInst(b, t.u32, .{ .load = .{ .ptr = slot_i } });
    try selectTail32(func, b, t, bits, sel);
}

test "f16 load/store round-trip is bit-exact vs @as(f16) (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildLoadStoreFn(&func);
    const cases = [_]f16{ 0.0, -0.0, 1.0, 1.5, -2.25, 65504.0, 0.00006103515625, 3.140625, std.math.inf(f16), -std.math.inf(f16) };
    for (cases) |x| {
        const got = (try reassemble(&func, &.{@as(i64, f16Bits(x))}, 4)) orelse return error.SkipZigTest;
        try std.testing.expectEqual(@as(u64, widen(x)), got);
    }
}

/// `f(pattern: i32, sel: i32) -> i32`: load the low-16 pattern as an f16, widen to f32 (identity,
/// held-as-f32), byte-select the f32 bits. Exercises the f16->f32 convert.
fn buildExtendFn(func: *Function) !void {
    const t = try Types.init(func);
    const b = try func.appendBlock();
    const pat = try func.appendBlockParam(b, t.i32);
    const sel = try func.appendBlockParam(b, t.i32);
    const slot = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.i32 } });
    try func.appendStore(b, pat, slot);
    const h = try func.appendInst(b, t.f16, .{ .load = .{ .ptr = slot } });
    const f = try func.appendInst(b, t.f32, .{ .convert = .{ .value = h } });
    try func.appendStore(b, f, slot);
    const bits = try func.appendInst(b, t.u32, .{ .load = .{ .ptr = slot } });
    try selectTail32(func, b, t, bits, sel);
}

/// `f(pattern: i32, sel: i32) -> i32`: reinterpret the low 32 bits as an f32, round it to f16
/// (vcvtps2ph RNE), widen back, and byte-select the resulting f32-widening bits. Exercises the
/// f32->f16 convert (proving it rounds, not a bare move, via the rounding cases below).
fn buildTruncFn(func: *Function) !void {
    const t = try Types.init(func);
    const b = try func.appendBlock();
    const pat = try func.appendBlockParam(b, t.i32);
    const sel = try func.appendBlockParam(b, t.i32);
    const slot = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.i32 } });
    try func.appendStore(b, pat, slot);
    const x = try func.appendInst(b, t.f32, .{ .load = .{ .ptr = slot } });
    const h = try func.appendInst(b, t.f16, .{ .convert = .{ .value = x } });
    const f = try func.appendInst(b, t.f32, .{ .convert = .{ .value = h } });
    try func.appendStore(b, f, slot);
    const bits = try func.appendInst(b, t.u32, .{ .load = .{ .ptr = slot } });
    try selectTail32(func, b, t, bits, sel);
}

test "f16 -> f32 widens exactly (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildExtendFn(&func);
    const cases = [_]f16{ 0.0, 1.0, -2.25, 65504.0, 0.000000059604645, 3.140625, std.math.inf(f16) };
    for (cases) |x| {
        const got = (try reassemble(&func, &.{@as(i64, f16Bits(x))}, 4)) orelse return error.SkipZigTest;
        try std.testing.expectEqual(@as(u64, widen(x)), got);
    }
}

test "f32 -> f16 rounds to nearest-even, not a bare move, bit-exact (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildTruncFn(&func);
    // f32 inputs chosen to exercise every rounding path (ties both ways, overflow, subnormal).
    const inputs = [_]f32{
        0.0, -0.0, 1.0, -1.0,
        1.00048828125, // halfway to the next half, rounds down to even (1.0)
        3.14159, // normal, rounds
        65504.0, // largest finite half
        65520.0, // exactly halfway 65504<->65536, ties to even == inf
        65519.0, // just below the tie, rounds to the max finite half
        65536.0, // overflows to +inf
        0.00006097555, // rounds into an f16 subnormal
        5.0e-8, // underflows toward zero
        std.math.inf(f32),
    };
    for (inputs) |x| {
        const in: u32 = @bitCast(x);
        const got = (try reassemble(&func, &.{@as(i64, @as(i32, @bitCast(in)))}, 4)) orelse return error.SkipZigTest;
        const want = widen(@as(f16, @floatCast(x))); // Zig's own round-to-nearest-even
        try std.testing.expectEqual(@as(u64, want), got);
        // A non-exactly-representable input must actually change under rounding (proving the
        // convert is not a bare copy of the f32).
        if (!std.math.isInf(x) and x != @as(f32, @as(f16, @floatCast(x)))) {
            try std.testing.expect(got != in);
        }
    }
}

/// `f(packed_ab: i32, sel: i32) -> i32`: unpack two half patterns (a in the low 16 bits, b in the
/// high 16 bits), load each as f16, apply `op`, widen the f16 result to f32, and byte-select the
/// f32-widening bits. The per-op half rounding must match Zig's f16 arithmetic bit-for-bit.
fn buildBinaryFn(func: *Function, op: ir.function.BinOp) !void {
    const t = try Types.init(func);
    const b = try func.appendBlock();
    const packed_ab = try func.appendBlockParam(b, t.u32);
    const sel = try func.appendBlockParam(b, t.i32);
    const slot_a = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.i32 } });
    const slot_b = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.i32 } });
    const a_pat = try func.appendArithImm(b, t.u32, .bit_and, packed_ab, 0xFFFF);
    const b_pat = try func.appendArithImm(b, t.u32, .shr, packed_ab, 16);
    try func.appendStore(b, a_pat, slot_a);
    try func.appendStore(b, b_pat, slot_b);
    const ha = try func.appendInst(b, t.f16, .{ .load = .{ .ptr = slot_a } });
    const hb = try func.appendInst(b, t.f16, .{ .load = .{ .ptr = slot_b } });
    const r = try func.appendInst(b, t.f16, .{ .arith = .{ .op = op, .lhs = ha, .rhs = hb } });
    const rf = try func.appendInst(b, t.f32, .{ .convert = .{ .value = r } });
    try func.appendStore(b, rf, slot_a);
    const bits = try func.appendInst(b, t.u32, .{ .load = .{ .ptr = slot_a } });
    try selectTail32(func, b, t, bits, sel);
}

fn runBinary(op: ir.function.BinOp, a: f16, b: f16) !?f16 {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildBinaryFn(&func, op);
    const packed_ab: u32 = @as(u32, f16Bits(a)) | (@as(u32, f16Bits(b)) << 16);
    const got = (try reassemble(&func, &.{@as(i64, @as(i32, @bitCast(packed_ab)))}, 4)) orelse return null;
    return @floatCast(@as(f32, @bitCast(@as(u32, @intCast(got))))); // the f32->f16 narrowing is exact
}

test "f16 add/sub/mul/div match Zig's per-op half rounding, bit-exact (qemu-x86_64)" {
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

test "f16 multiply rounds its result to nearest-even half, not a raw f32 product (qemu-x86_64)" {
    // a*a whose exact f32 product is NOT representable in f16, so the half result must round.
    const a: f16 = 1.0009765625; // 1 + 2^-10, itself exactly representable
    const got = (try runBinary(.mul, a, a)) orelse return error.SkipZigTest;
    try std.testing.expectEqual(f16Bits(a * a), f16Bits(got));
    const exact_f32: f32 = @as(f32, a) * @as(f32, a);
    try std.testing.expect(@as(f32, got) != exact_f32); // rounding to half changed the value
}

/// `f(x: i32, sel: i32) -> i32`: convert a 32-bit signed integer to f16 (int -> f32 then round to
/// half), widen back to f32, byte-select the f32-widening bits.
fn buildIntToHalfFn(func: *Function) !void {
    const t = try Types.init(func);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t.i32);
    const sel = try func.appendBlockParam(b, t.i32);
    const slot = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.i32 } });
    const h = try func.appendInst(b, t.f16, .{ .convert = .{ .value = x } });
    const f = try func.appendInst(b, t.f32, .{ .convert = .{ .value = h } });
    try func.appendStore(b, f, slot);
    const bits = try func.appendInst(b, t.u32, .{ .load = .{ .ptr = slot } });
    try selectTail32(func, b, t, bits, sel);
}

/// `f(pattern: i32, sel: i32) -> i32`: load the low-16 half pattern as f16, convert to a 32-bit
/// signed integer (truncate toward zero), byte-select. The result is the whole i32.
fn buildHalfToIntFn(func: *Function) !void {
    const t = try Types.init(func);
    const b = try func.appendBlock();
    const pat = try func.appendBlockParam(b, t.i32);
    const sel = try func.appendBlockParam(b, t.i32);
    const slot = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.i32 } });
    try func.appendStore(b, pat, slot);
    const h = try func.appendInst(b, t.f16, .{ .load = .{ .ptr = slot } });
    const r = try func.appendInst(b, t.u32, .{ .convert = .{ .value = h } });
    try selectTail32(func, b, t, r, sel);
}

test "int <-> f16 conversions match Zig, bit-exact (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    { // int -> f16: 2049 is not exactly representable and must round to 2048; -7 is exact.
        var func = Function.init(allocator);
        defer func.deinit();
        try buildIntToHalfFn(&func);
        for ([_]i32{ 2049, -7, 0, 1000 }) |v| {
            const got = (try reassemble(&func, &.{@as(i64, v)}, 4)) orelse return error.SkipZigTest;
            try std.testing.expectEqual(@as(u64, widen(@as(f16, @floatFromInt(v)))), got);
        }
    }
    { // f16 -> int: truncates toward zero.
        var func = Function.init(allocator);
        defer func.deinit();
        try buildHalfToIntFn(&func);
        for ([_]f16{ 3.5, -3.9, 100.0, 0.0 }) |x| {
            const got = (try reassemble(&func, &.{@as(i64, f16Bits(x))}, 4)) orelse return error.SkipZigTest;
            const want: i32 = @intFromFloat(x);
            try std.testing.expectEqual(@as(u64, @as(u32, @bitCast(want))), got);
        }
    }
}

/// `f(sel: i32) -> i32`: an f64 CONSTANT `c`, converted to f16 (cvtsd2ss then round to half),
/// widened back to f32, byte-select the f32-widening bits. The constant avoids passing a 64-bit
/// argument through the one-byte integer stub.
fn buildDoubleToHalfFn(func: *Function, c: f64) !void {
    const t = try Types.init(func);
    const b = try func.appendBlock();
    const sel = try func.appendBlockParam(b, t.i32);
    const slot = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.i32 } });
    const d = try func.appendInst(b, t.f64, .{ .fconst = c });
    const h = try func.appendInst(b, t.f16, .{ .convert = .{ .value = d } });
    const f = try func.appendInst(b, t.f32, .{ .convert = .{ .value = h } });
    try func.appendStore(b, f, slot);
    const bits = try func.appendInst(b, t.u32, .{ .load = .{ .ptr = slot } });
    try selectTail32(func, b, t, bits, sel);
}

/// `f(sel: i32) -> i64`: an f16 CONSTANT `v`, converted to f64 (cvtss2sd of the held f32), the
/// full 64-bit f64 pattern byte-selected (sel = 0..7).
fn buildHalfToDoubleFn(func: *Function, v: f16) !void {
    const t = try Types.init(func);
    const b = try func.appendBlock();
    const sel = try func.appendBlockParam(b, t.i32);
    const slot = try func.appendInst(b, t.ptr, .{ .alloca = .{ .elem = t.f64 } });
    const h = try func.appendInst(b, t.f16, .{ .fconst = @floatCast(v) });
    const d = try func.appendInst(b, t.f64, .{ .convert = .{ .value = h } });
    try func.appendStore(b, d, slot);
    const bits = try func.appendInst(b, t.u64, .{ .load = .{ .ptr = slot } });
    try selectTail64(func, b, t, bits, sel);
}

test "f64 -> f16 rounds bit-exact vs @as(f16) (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    // Include a value not representable in f16 (0.1) to prove the narrowing rounds.
    const cases = [_]f64{ 0.0, 1.0, -2.25, 0.1, 65504.0, 65536.0, 3.140625 };
    for (cases) |c| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildDoubleToHalfFn(&func, c);
        const got = (try reassemble(&func, &.{}, 4)) orelse return error.SkipZigTest;
        try std.testing.expectEqual(@as(u64, widen(@as(f16, @floatCast(c)))), got);
    }
}

test "f16 -> f64 widens bit-exact vs @as(f64, @as(f16)) (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    const cases = [_]f16{ 0.0, 1.0, -2.25, 0.1, 65504.0, 3.140625 };
    for (cases) |v| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildHalfToDoubleFn(&func, v);
        const got = (try reassemble(&func, &.{}, 8)) orelse return error.SkipZigTest;
        const want: u64 = @bitCast(@as(f64, v));
        try std.testing.expectEqual(want, got);
    }
}
