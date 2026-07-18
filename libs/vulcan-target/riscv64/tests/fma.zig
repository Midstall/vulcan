//! Scalar float FMA fusion, executed on qemu-riscv64 (the oracle). Builds `a*b+c` / `a*b-c` /
//! `c-a*b` IR functions - the multiply immediately preceding its single-use add/sub, exactly the
//! shape `fusesIntoNextArith` (isel.zig) fuses - and checks that the qemu-executed result is
//! bit-identical to the host's `@mulAdd`, proving both single-rounding and the RISC-V variant
//! mapping (fmadd/fmsub/fnmsub) are correct, not just that fusion "did something". Operand
//! triples are chosen (see gen_fma search, referenced in the FMA plan) so the naive
//! separately-rounded computation differs from the fused one: a wrong variant or a fusion that
//! silently failed to fire would show up as a mismatch here, not pass by coincidence.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const disasm = @import("../disasm.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;

/// The three fused shapes `fusesIntoNextArith` recognizes (see isel.zig's `.arith` add/sub
/// branch): `add`: a*b+c -> fmadd. `sub`: a*b-c -> fmsub. `csub`: c-a*b -> fnmsub. This mapping
/// is RISC-V's own (RISC-V FMSUB is rs1*rs2-rs3, unlike aarch64's FMSUB).
const FmaShape = enum { add, sub, csub };

/// `f(a, b, c)` computing `shape` in `dbl` precision (f64 if true, else f32), with the multiply
/// immediately preceding its single consuming add/sub - exactly the shape `fusesIntoNextArith`
/// fuses into one fmadd/fmsub/fnmsub.
fn buildFmaFunc(allocator: std.mem.Allocator, dbl: bool, shape: FmaShape) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ft = try func.types.intern(.{ .float = if (dbl) .f64 else .f32 });
    const b = try func.appendBlock();
    const a_p = try func.appendBlockParam(b, ft);
    const b_p = try func.appendBlockParam(b, ft);
    const c_p = try func.appendBlockParam(b, ft);
    const prod = try func.appendInst(b, ft, .{ .arith = .{ .op = .mul, .lhs = a_p, .rhs = b_p } });
    const r = switch (shape) {
        .add => try func.appendInst(b, ft, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = c_p } }),
        .sub => try func.appendInst(b, ft, .{ .arith = .{ .op = .sub, .lhs = prod, .rhs = c_p } }),
        .csub => try func.appendInst(b, ft, .{ .arith = .{ .op = .sub, .lhs = c_p, .rhs = prod } }),
    };
    func.setTerminator(b, .{ .ret = r });
    return func;
}

/// The unsigned integer type with T's bit width, for bit-exact (not `==`) float comparison.
fn Bits(comptime T: type) type {
    return std.meta.Int(.unsigned, @bitSizeOf(T));
}

/// Build, compile, and check the FMA `shape` in precision `T` on the operands given as raw bit
/// patterns (exact - avoids any decimal-literal round-trip drift):
///   1. Structural: the emitted code contains `mnemonic` and no separate fmul/fadd/fsub - proves
///      fusion actually fired for this shape (not just that the arithmetic happens to work out).
///   2. The qemu-executed result is bit-identical to the hardware FMA reference `@mulAdd` (proves
///      the variant mapping and single-rounding are both correct).
///   3. That result DIFFERS from the naive separately-rounded computation - the operands were
///      searched specifically so fused != separate, so a fusion that silently didn't fire (or
///      fired with the wrong variant, landing back on the separate-rounding value by coincidence)
///      would be caught here.
fn checkFma(io: std.Io, comptime T: type, shape: FmaShape, a_bits: Bits(T), b_bits: Bits(T), c_bits: Bits(T), mnemonic: []const u8) !void {
    const allocator = std.testing.allocator;
    const dbl = T == f64;

    var struct_func = try buildFmaFunc(allocator, dbl, shape);
    defer struct_func.deinit();
    const code = try isel.selectFunction(allocator, &struct_func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, mnemonic) != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fmul") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fadd") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fsub") == null);

    var run_func = try buildFmaFunc(allocator, dbl, shape);
    defer run_func.deinit();
    const fargs: [3]u64 = .{ a_bits, b_bits, c_bits };
    const got_bits = try harness.runFuncFloat(io, allocator, &run_func, dbl, &fargs, harness.qemu_user);
    const got: T = if (dbl) @bitCast(got_bits) else @bitCast(@as(u32, @truncate(got_bits)));

    const a: T = @bitCast(a_bits);
    const b: T = @bitCast(b_bits);
    const c: T = @bitCast(c_bits);
    const want = switch (shape) {
        .add => @mulAdd(T, a, b, c),
        .sub => @mulAdd(T, a, b, -c),
        .csub => @mulAdd(T, -a, b, c),
    };
    try std.testing.expectEqual(@as(Bits(T), @bitCast(want)), @as(Bits(T), @bitCast(got)));

    const naive = switch (shape) {
        .add => a * b + c,
        .sub => a * b - c,
        .csub => c - a * b,
    };
    try std.testing.expect(@as(Bits(T), @bitCast(naive)) != @as(Bits(T), @bitCast(got)));
}

// Operand triples below were found by random search specifically because the separately-
// rounded computation (a*b then +/-c as two instructions) differs from the fused one, so each
// test would fail if fusion did not fire, or fired with the wrong variant.

test "fma: scalar f32 a*b+c matches @mulAdd bit-exactly and fuses to fmadd (qemu-riscv64)" {
    try checkFma(std.testing.io, f32, .add, 0x38ae71f0, 0x58a7be01, 0xc96261bc, "fmadd");
}

test "fma: scalar f32 a*b-c matches @mulAdd bit-exactly and fuses to fmsub (qemu-riscv64)" {
    try checkFma(std.testing.io, f32, .sub, 0x27e9e63b, 0x78b47c95, 0x615c195b, "fmsub");
}

test "fma: scalar f32 c-a*b matches @mulAdd bit-exactly and fuses to fnmsub (qemu-riscv64)" {
    try checkFma(std.testing.io, f32, .csub, 0x3c3f1cf2, 0x5dab2467, 0xd175b0dc, "fnmsub");
}

test "fma: scalar f64 a*b+c matches @mulAdd bit-exactly and fuses to fmadd (qemu-riscv64)" {
    try checkFma(std.testing.io, f64, .add, 0x0662b334da310f42, 0x3ea7261930bfc4ff, 0x843a36e1f3cdd748, "fmadd");
}

test "fma: scalar f64 a*b-c matches @mulAdd bit-exactly and fuses to fmsub (qemu-riscv64)" {
    try checkFma(std.testing.io, f64, .sub, 0x467c4e95c120b668, 0x850e5cb558b81340, 0x08d3c37c210053c2, "fmsub");
}

test "fma: scalar f64 c-a*b matches @mulAdd bit-exactly and fuses to fnmsub (qemu-riscv64)" {
    try checkFma(std.testing.io, f64, .csub, 0x4f835e4d2daf6429, 0xe8a23b5aeaa309d1, 0x7502fbd4b839369b, "fnmsub");
}

test "fma: a multi-use mul does NOT fuse (separate fmul+fadd, correct result, qemu-riscv64)" {
    // The product feeds the fusible add AND is read again afterward, so it is not single-use:
    // fusion must be declined, and the multiply stays materialized.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ft = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const a = try func.appendBlockParam(blk, ft);
    const b = try func.appendBlockParam(blk, ft);
    const c = try func.appendBlockParam(blk, ft);
    const prod = try func.appendInst(blk, ft, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const s = try func.appendInst(blk, ft, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = c } }); // a*b+c, fusible shape...
    const r = try func.appendInst(blk, ft, .{ .arith = .{ .op = .add, .lhs = s, .rhs = prod } }); // ...but prod is reused here
    func.setTerminator(blk, .{ .ret = r });

    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);
    const text = try disasm.format(allocator, code);
    defer allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "fmul") != null); // the multiply is still materialized
    try std.testing.expect(std.mem.indexOf(u8, text, "fadd") != null); // and added separately, twice
    try std.testing.expect(std.mem.indexOf(u8, text, "fmadd") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fmsub") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fnmsub") == null);

    var run_func = Function.init(allocator);
    defer run_func.deinit();
    const ft2 = try run_func.types.intern(.{ .float = .f32 });
    const blk2 = try run_func.appendBlock();
    const a2 = try run_func.appendBlockParam(blk2, ft2);
    const b2 = try run_func.appendBlockParam(blk2, ft2);
    const c2 = try run_func.appendBlockParam(blk2, ft2);
    const prod2 = try run_func.appendInst(blk2, ft2, .{ .arith = .{ .op = .mul, .lhs = a2, .rhs = b2 } });
    const s2 = try run_func.appendInst(blk2, ft2, .{ .arith = .{ .op = .add, .lhs = prod2, .rhs = c2 } });
    const r2 = try run_func.appendInst(blk2, ft2, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = prod2 } });
    run_func.setTerminator(blk2, .{ .ret = r2 });

    const av: f32 = 2.5;
    const bv: f32 = 3.25;
    const cv: f32 = 1.5;
    const fargs: [3]u64 = .{ @as(u32, @bitCast(av)), @as(u32, @bitCast(bv)), @as(u32, @bitCast(cv)) };
    const got_bits = try harness.runFuncFloat(std.testing.io, allocator, &run_func, false, &fargs, harness.qemu_user);
    const got: f32 = @bitCast(@as(u32, @truncate(got_bits)));
    const prod_v = av * bv;
    const expected = (prod_v + cv) + prod_v;
    try std.testing.expectEqual(expected, got);
}
