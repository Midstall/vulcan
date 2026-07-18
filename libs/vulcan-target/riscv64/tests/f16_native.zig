//! NATIVE f16 (Zfh) differentials, executed on `qemu-riscv64 -cpu max` (the oracle). A Zfh-capable
//! model (here river-rc1.f, which sets `features.riscv64.zfh`) makes the riscv64 backend hold an f16
//! natively in a float register (NaN-boxed into the low 16 bits) and lower every f16 op to a real
//! half instruction: `flh`/`fsh` for load/store, `fadd.h`/`fsub.h`/`fmul.h`/`fdiv.h` for arithmetic,
//! `fcvt.s.h`/`fcvt.h.s` (and the f64 / int siblings) for conversions, `feq.h`/`flt.h`/`fle.h` for
//! compares, and `fmv.h.x` for an f16 constant. This is the alternative to the software emulation
//! validated in f16.zig (held-as-f32 widening). Both must agree with Zig's own `@as(f16, ...)`.
//!
//! The default `qemu-riscv64` (RV64GC) has NO Zfh, so these run under `harness.qemu_user_cpumax`
//! (`-cpu max`). All test IR uses the INTEGER ABI: f16 values enter/leave through memory (an `sd`
//! of the raw pattern then an `flh`, and an `fsh` result reloaded as a zero-extended `u16`), so the
//! host never has to NaN-box a half into a float-argument register. Slots that receive an i64 store
//! are `alloca`d as i64 so the 8-byte `sd` stays in bounds.
//!
//! Skips (not fails) when qemu-riscv64 is not on PATH.

const std = @import("std");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const isel = @import("../isel.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

/// The Zfh-capable model: river-rc1.f carries `features.riscv64.zfh`, so `selectFunctionForModel`
/// flips the backend to the native half path.
fn zfhModel() *const opt.microarch.Model {
    return opt.microarch.modelFor(.@"river-rc1.f");
}

fn f16Bits(x: f16) u16 {
    return @bitCast(x);
}

fn widen(x: f16) u32 {
    return @bitCast(@as(f32, x));
}

/// Compile `func` through the NATIVE Zfh path and run it under `-cpu max` with integer args,
/// returning the raw a0. Skips (returns null) when qemu is absent.
fn runNative(func: *Function, args: []const i64) !?i64 {
    const allocator = std.testing.allocator;
    var words = try harness.compileFuncForModel(allocator, func, zfhModel());
    defer words.deinit(allocator);
    return harness.runCode(std.testing.io, allocator, words.items, args, harness.qemu_user_cpumax) catch |e| switch (e) {
        error.SkipZigTest => null,
        else => e,
    };
}

/// `f(a_bits: i64, b_bits: i64) -> i64`: load the low 16 bits of each arg as a native f16 (`flh`),
/// apply one f16 binary op (`fadd.h`/...), store the half result (`fsh`), and return it as a
/// zero-extended `u16`. Exercises native load, arithmetic, and store in one function.
fn buildBinaryFn(func: *Function, op: ir.function.BinOp) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const a_bits = try func.appendBlockParam(b, i64_t);
    const b_bits = try func.appendBlockParam(b, i64_t);
    const slot_a = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    const slot_b = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    try func.appendStore(b, a_bits, slot_a); // sd (8 bytes), low 2 = half a
    try func.appendStore(b, b_bits, slot_b);
    const ha = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot_a } });
    const hb = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot_b } });
    const r = try func.appendInst(b, f16_t, .{ .arith = .{ .op = op, .lhs = ha, .rhs = hb } });
    try func.appendStore(b, r, slot_a); // fsh: writes the low 2 bytes
    const bits = try func.appendInst(b, u16_t, .{ .load = .{ .ptr = slot_a } }); // lhu: zero-extended half
    func.setTerminator(b, .{ .ret = bits });
}

fn runBinary(op: ir.function.BinOp, a: f16, b: f16) !?u16 {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildBinaryFn(&func, op);
    const got = (try runNative(&func, &.{ @as(i64, f16Bits(a)), @as(i64, f16Bits(b)) })) orelse return null;
    return @truncate(@as(u64, @bitCast(got)));
}

test "native f16 add/sub/mul/div match Zig's per-op half rounding, bit-exact (qemu -cpu max)" {
    const cases = [_]struct { a: f16, b: f16 }{
        .{ .a = 1.5, .b = 2.25 },
        .{ .a = -3.5, .b = 0.75 },
        .{ .a = 0.1, .b = 0.2 }, // not exact in f16, so results round
        .{ .a = 100.0, .b = 7.0 },
        .{ .a = 1.0009765625, .b = 1.0009765625 }, // product not representable, must round to half
    };
    for (cases) |c| {
        const add = (try runBinary(.add, c.a, c.b)) orelse return error.SkipZigTest;
        try std.testing.expectEqual(f16Bits(c.a + c.b), add);
        try std.testing.expectEqual(f16Bits(c.a - c.b), (try runBinary(.sub, c.a, c.b)).?);
        try std.testing.expectEqual(f16Bits(c.a * c.b), (try runBinary(.mul, c.a, c.b)).?);
        try std.testing.expectEqual(f16Bits(c.a / c.b), (try runBinary(.div, c.a, c.b)).?);
    }
}

// P4 (aarch64) proved native single-rounded `fdiv.h` agrees with `@as(f16, a/b)` for every finite
// f16 pair, because f16's 10-bit mantissa is small enough that a divide through f32 never double-
// rounds wrong. Confirm the same for native `fdiv.h` here: sweep EVERY one of the 65536 half
// patterns as the dividend `a`, divide by a fixed `b`, fold each result's half bits into a rolling
// checksum, and match it against the host's `@as(f16, a/b)` over the same 65536 values. One qemu
// run per `b`. NaN/non-finite dividends fold 0, so the checksum is independent of the NaN payload.

/// `f(b_bits: i64) -> i64`: loop `ai` over 0..65536, load the half whose bits are `ai` (`flh`),
/// divide it by the half whose bits are `b_bits` (native `fdiv.h`), and fold the result's half bits
/// into `acc = acc*33 + bits`. A NaN or non-finite dividend (abs > 0x7c00) folds 0 instead.
fn buildDivSweepFn(func: *Function) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const ptr_t = try func.types.intern(.ptr);

    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const b_bits = try func.appendBlockParam(entry, i64_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const acc = try func.appendBlockParam(loop, i32_t);
    // Stack scratch and the divisor slot are threaded through the loop as block params (a value
    // defined in the entry and used only inside the loop body would be live across the back-edge,
    // which the riscv64 allocator does not keep resident - see f16.zig's sweep builders). `slot`
    // holds one 4-byte word: the dividend pattern for `flh`, later rewritten to hold the quotient.
    const lslot = try func.appendBlockParam(loop, ptr_t);
    const lbslot = try func.appendBlockParam(loop, ptr_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const slot = try func.appendBlockParam(body, ptr_t);
    const bslot = try func.appendBlockParam(body, ptr_t);
    const racc = try func.appendBlockParam(done, i32_t);

    const slot0 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const bslot0 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i64_t } }); // holds an i64 store
    try func.appendStore(entry, b_bits, bslot0); // the divisor half pattern, stored once
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero, slot0, bslot0 });

    const n = try func.appendInst(loop, i32_t, .{ .iconst = 65536 });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc, lslot, lbslot } }, .{ .target = done, .args = &.{acc} });

    try func.appendStore(body, bi, slot); // sw: low 16 bits = the dividend half pattern
    const ha = try func.appendInst(body, f16_t, .{ .load = .{ .ptr = slot } });
    const hb = try func.appendInst(body, f16_t, .{ .load = .{ .ptr = bslot } });
    const q = try func.appendInst(body, f16_t, .{ .arith = .{ .op = .div, .lhs = ha, .rhs = hb } });
    // Zero the word, then `fsh` the quotient's 2 bytes over it, so an i32 load reads the half bits
    // zero-extended (no int->int convert, which the backend does not lower yet).
    const zw = try func.appendInst(body, i32_t, .{ .iconst = 0 });
    try func.appendStore(body, zw, slot); // sw: clears all 4 bytes
    try func.appendStore(body, q, slot); // fsh: overwrites the low 2 bytes
    const qbits = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = slot } }); // [half | 0]
    // NaN/non-finite dividend exclusion: absa = bi & 0x7fff; skip = absa > 0x7c00.
    const mask = try func.appendInst(body, i32_t, .{ .iconst = 0x7FFF });
    const absa = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .bit_and, .lhs = bi, .rhs = mask } });
    const c7c00 = try func.appendInst(body, i32_t, .{ .iconst = 0x7C00 });
    const skip = try func.appendInst(body, bool_t, .{ .icmp = .{ .op = .gt, .lhs = absa, .rhs = c7c00 } });
    const zero2 = try func.appendInst(body, i32_t, .{ .iconst = 0 });
    const contrib = try func.appendInst(body, i32_t, .{ .select = .{ .cond = skip, .then = zero2, .@"else" = qbits } });
    const c33 = try func.appendInst(body, i32_t, .{ .iconst = 33 });
    const m = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = bacc, .rhs = c33 } });
    const acc2 = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = m, .rhs = contrib } });
    const inext = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ inext, acc2, slot, bslot });

    func.setTerminator(done, .{ .ret = racc });
}

test "native f16 fdiv.h matches @as(f16, a/b) for ALL 65536 dividends, several divisors (qemu -cpu max)" {
    const allocator = std.testing.allocator;
    // A spread of divisors: small integers, a not-f16-exact value, and a subnormal, so the divide
    // exercises normal, rounding, and tiny-magnitude cases across the whole dividend space.
    const divisors = [_]f16{ 3.0, 7.0, 0.1, 0.00006103515625, -2.5 };
    for (divisors) |bdiv| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildDivSweepFn(&func);
        const got = (try runNative(&func, &.{@as(i64, f16Bits(bdiv))})) orelse return error.SkipZigTest;
        // Host reference: the same rolling checksum over Zig's own `@as(f16, a/b)`.
        var acc: i64 = 0;
        var u: u32 = 0;
        while (u < 65536) : (u += 1) {
            const a: f16 = @bitCast(@as(u16, @intCast(u)));
            const skip = (u & 0x7FFF) > 0x7C00; // NaN/inf dividend
            const contrib: i64 = if (skip) 0 else @as(u16, @bitCast(a / bdiv));
            acc = acc *% 33 +% contrib;
        }
        // The device checksum is an i32 sign-extended through the i64 return; compare the low 32.
        const want_lo: u32 = @bitCast(@as(i32, @truncate(acc)));
        const got_lo: u32 = @truncate(@as(u64, @bitCast(got)));
        try std.testing.expectEqual(want_lo, got_lo);
    }
}

/// `f(a_bits: i64) -> i64`: load a native f16 (`flh`), widen to f32 (`fcvt.s.h`), return the f32
/// bits. Compare the low 32 bits.
fn buildF16ToF32Fn(func: *Function) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const in = try func.appendBlockParam(b, i64_t);
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    try func.appendStore(b, in, slot);
    const h = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot } });
    const f = try func.appendInst(b, f32_t, .{ .convert = .{ .value = h } });
    try func.appendStore(b, f, slot);
    const bits = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = bits });
}

/// `f(x_bits: i64) -> i64`: load an f32 (`flw`), narrow to a native f16 (`fcvt.h.s`), store (`fsh`),
/// return the zero-extended half bits.
fn buildF32ToF16Fn(func: *Function) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const in = try func.appendBlockParam(b, i64_t);
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    try func.appendStore(b, in, slot);
    const x = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = slot } });
    const h = try func.appendInst(b, f16_t, .{ .convert = .{ .value = x } });
    try func.appendStore(b, h, slot);
    const bits = try func.appendInst(b, u16_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = bits });
}

test "native f16 <-> f32 convert bit-exact vs Zig (qemu -cpu max)" {
    const allocator = std.testing.allocator;
    // f16 -> f32: exact widen for every category.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildF16ToF32Fn(&func);
        const cases = [_]f16{ 0.0, -0.0, 1.0, 1.5, -2.25, 65504.0, 0.00006103515625, 3.140625, std.math.inf(f16) };
        for (cases) |x| {
            const got = (try runNative(&func, &.{@as(i64, f16Bits(x))})) orelse return error.SkipZigTest;
            try std.testing.expectEqual(widen(x), @as(u32, @truncate(@as(u64, @bitCast(got)))));
        }
    }
    // f32 -> f16: single-rounded narrow.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildF32ToF16Fn(&func);
        const inputs = [_]f32{ 0.0, 1.0, 65520.0, 65519.0, 65536.0, 3.14159, 0.00006097555, 5.0e-8, std.math.inf(f32) };
        for (inputs) |x| {
            const got = (try runNative(&func, &.{@as(i64, @as(i32, @bitCast(x)))})) orelse return error.SkipZigTest;
            try std.testing.expectEqual(f16Bits(@as(f16, @floatCast(x))), @as(u16, @truncate(@as(u64, @bitCast(got)))));
        }
    }
}

/// `f(a_bits: i64) -> i64`: load a native f16, widen to f64 (`fcvt.d.h`), return the f64 bits.
fn buildF16ToF64Fn(func: *Function) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const in = try func.appendBlockParam(b, i64_t);
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    const dslot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = f64_t } });
    try func.appendStore(b, in, slot);
    const h = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot } });
    const d = try func.appendInst(b, f64_t, .{ .convert = .{ .value = h } });
    try func.appendStore(b, d, dslot);
    const bits = try func.appendInst(b, i64_t, .{ .load = .{ .ptr = dslot } });
    func.setTerminator(b, .{ .ret = bits });
}

test "native f16 -> f64 convert bit-exact vs Zig (qemu -cpu max)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildF16ToF64Fn(&func);
    for ([_]f16{ 1.5, -2.25, 0.00006103515625, 3.140625, 65504.0 }) |x| {
        const got = (try runNative(&func, &.{@as(i64, f16Bits(x))})) orelse return error.SkipZigTest;
        try std.testing.expectEqual(@as(u64, @bitCast(@as(f64, x))), @as(u64, @bitCast(got)));
    }
}

/// `f(x: i32) -> i64`: signed int -> native f16 (`fcvt.h.w`), store, return the half bits.
fn buildIntToF16Fn(func: *Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const h = try func.appendInst(b, f16_t, .{ .convert = .{ .value = x } });
    try func.appendStore(b, h, slot); // fsh (2 bytes); u16 load reads exactly those
    const bits = try func.appendInst(b, u16_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = bits });
}

/// `f(a_bits: i64) -> i64`: load a native f16, truncate to a signed int (`fcvt.w.h`, rtz).
fn buildF16ToIntFn(func: *Function) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const in = try func.appendBlockParam(b, i64_t);
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    try func.appendStore(b, in, slot);
    const h = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot } });
    const r = try func.appendInst(b, i32_t, .{ .convert = .{ .value = h } });
    func.setTerminator(b, .{ .ret = r });
}

test "native int <-> f16 conversions match Zig, bit-exact (qemu -cpu max)" {
    const allocator = std.testing.allocator;
    {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildIntToF16Fn(&func);
        for ([_]i32{ 2049, -7, 0, 1000 }) |v| { // 2049 not representable -> rounds to 2048
            const got = (try runNative(&func, &.{@as(i64, v)})) orelse return error.SkipZigTest;
            try std.testing.expectEqual(f16Bits(@as(f16, @floatFromInt(v))), @as(u16, @truncate(@as(u64, @bitCast(got)))));
        }
    }
    {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildF16ToIntFn(&func);
        for ([_]f16{ 3.5, -3.9, 100.0, 0.0 }) |x| {
            const got = (try runNative(&func, &.{@as(i64, @intCast(f16Bits(x)))})) orelse return error.SkipZigTest;
            try std.testing.expectEqual(@as(i64, @intFromFloat(x)), got);
        }
    }
}

/// `f() -> i64`: materialize an f16 constant (`fmv.h.x`), store it (`fsh`), return the half bits.
fn buildFconstFn(func: *Function, val: f16) !void {
    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    const c = try func.appendInst(b, f16_t, .{ .fconst = val });
    try func.appendStore(b, c, slot); // fsh
    const bits = try func.appendInst(b, u16_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(b, .{ .ret = bits });
}

test "native f16 fconst materializes the exact half bits (qemu -cpu max)" {
    const allocator = std.testing.allocator;
    for ([_]f16{ 3.140625, -2.25, 1.0, 0.0, 65504.0 }) |val| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildFconstFn(&func, val);
        const got = (try runNative(&func, &.{})) orelse return error.SkipZigTest;
        try std.testing.expectEqual(f16Bits(val), @as(u16, @truncate(@as(u64, @bitCast(got)))));
    }
}

/// `f(a_bits: i64, b_bits: i64) -> i64`: load two native halves, compare with `op`, return 0/1 (via
/// a `select` so the boolean lands in an integer register the ABI can carry).
fn buildCmpFn(func: *Function, op: ir.function.CmpOp) !void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const a_bits = try func.appendBlockParam(b, i64_t);
    const b_bits = try func.appendBlockParam(b, i64_t);
    const slot_a = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    const slot_b = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    try func.appendStore(b, a_bits, slot_a);
    try func.appendStore(b, b_bits, slot_b);
    const ha = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot_a } });
    const hb = try func.appendInst(b, f16_t, .{ .load = .{ .ptr = slot_b } });
    const r = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = op, .lhs = ha, .rhs = hb } });
    const one = try func.appendInst(b, i32_t, .{ .iconst = 1 });
    const zero = try func.appendInst(b, i32_t, .{ .iconst = 0 });
    const w = try func.appendInst(b, i32_t, .{ .select = .{ .cond = r, .then = one, .@"else" = zero } });
    func.setTerminator(b, .{ .ret = w });
}

test "native f16 compares (feq.h/flt.h/fle.h) match Zig (qemu -cpu max)" {
    const allocator = std.testing.allocator;
    const pairs = [_]struct { a: f16, b: f16 }{
        .{ .a = 1.5, .b = 2.25 }, .{ .a = 2.25, .b = 1.5 }, .{ .a = -1.0, .b = -1.0 }, .{ .a = 0.0, .b = -0.0 },
    };
    const ops = [_]ir.function.CmpOp{ .eq, .ne, .lt, .le, .gt, .ge };
    for (ops) |op| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildCmpFn(&func, op);
        for (pairs) |p| {
            const got = (try runNative(&func, &.{ @as(i64, f16Bits(p.a)), @as(i64, f16Bits(p.b)) })) orelse return error.SkipZigTest;
            const want: i64 = if (switch (op) {
                .eq => p.a == p.b,
                .ne => p.a != p.b,
                .lt => p.a < p.b,
                .le => p.a <= p.b,
                .gt => p.a > p.b,
                .ge => p.a >= p.b,
            }) 1 else 0;
            try std.testing.expectEqual(want, got);
        }
    }
}

test "native Zfh path emits fewer words than the software emulation for the same f16 op" {
    const allocator = std.testing.allocator;
    // Compile the SAME f16 add both ways: native (Zfh model) vs the default emulation
    // (selectFunction). The native path drops every software widen/round, so it must be strictly
    // shorter - proof the gate actually switches lowering rather than being a no-op.
    var func = Function.init(allocator);
    defer func.deinit();
    try buildBinaryFn(&func, .add);
    try ir.legalize.legalize(allocator, &func);
    try isel.splitCriticalEdges(allocator, &func);
    const native = try isel.selectFunctionForModel(allocator, &func, zfhModel());
    defer allocator.free(native);
    const soft = try isel.selectFunction(allocator, &func);
    defer allocator.free(soft);
    try std.testing.expect(native.len < soft.len);
}
