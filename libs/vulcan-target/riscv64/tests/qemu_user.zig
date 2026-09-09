//! QEMU user-mode runner: `qemu-riscv64`. Runs the shared codegen/optimization case corpus as plain
//! Linux static ELFs (syscall write/exit), so the RISC-V backend executes on any machine with qemu
//! even when River and Spike are absent. Skips when qemu-riscv64 is not on PATH.

const std = @import("std");
const ir = @import("vulcan-ir");
const cases = @import("cases.zig");
const harness = @import("harness.zig");
const isel = @import("../isel.zig");

const Function = ir.function.Function;

test "qemu-user-riscv: shared codegen and optimization cases" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu_user);
}

// binary128 (f128) DATA MOVEMENT on lp64d: an f128 has no register form (it is 16 bytes in a GPR
// PAIR or memory), so it lives in a 16-byte stack slot and materializes into an aligned a-register
// pair only at ABI boundaries. f128 arithmetic/compare/convert are soft-fp libcalls (the shared
// softfp pass), so none appears here; these cases prove the 16-byte value survives the pair ABI and
// the memory paths intact. Each runs under qemu-riscv64 and asserts all 128 bits.

/// The two 64-bit halves (low, high) of an f128's bit pattern.
fn halves(v: f128) [2]u64 {
    const bits: u128 = @bitCast(v);
    return .{ @truncate(bits), @truncate(bits >> 64) };
}

test "qemu-user-riscv f128: identity carries all 128 bits through the a0:a1 pair" {
    const allocator = std.testing.allocator;
    const v: f128 = 0.1;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, t);
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(a) });
    const h = halves(v);
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{ h[0], h[1] }, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(v)), got);
}

test "qemu-user-riscv f128: return the second argument (a2:a3 -> a0:a1)" {
    const allocator = std.testing.allocator;
    const v0: f128 = 2.5;
    const v1: f128 = 1.0 / 3.0;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    _ = try f.appendBlockParam(b, t);
    const bb = try f.appendBlockParam(b, t);
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(bb) });
    const h0 = halves(v0);
    const h1 = halves(v1);
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{ h0[0], h0[1], h1[0], h1[1] }, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(v1)), got);
}

test "qemu-user-riscv f128: an int arg before an f128 forces the aligned pair (a2:a3, a1 skipped)" {
    const allocator = std.testing.allocator;
    const v: f128 = 3.141592653589793238462643383279502884;
    var f = Function.init(allocator);
    defer f.deinit();
    const i64_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    _ = try f.appendBlockParam(b, i64_t); // consumes a0; the f128 must then align to a2:a3
    const a = try f.appendBlockParam(b, t);
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(a) });
    const h = halves(v);
    // argHalves: a0 = x (0x1234), a1 = filler (skipped by alignment), a2:a3 = the f128.
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{ 0x1234, 0xdead, h[0], h[1] }, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(v)), got);
}

test "qemu-user-riscv f128: constant materialized from its two 64-bit halves" {
    const allocator = std.testing.allocator;
    const c: f128 = 2.718281828459045235360287471352662497;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    const k = try f.appendInst(b, t, .{ .fconst128 = @bitCast(c) });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(k) });
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{}, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(c)), got);
}

test "qemu-user-riscv f128: alloca store then load round-trips all 16 bytes" {
    const allocator = std.testing.allocator;
    const v: f128 = -0.7;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const ptr_t = try f.types.ptrGlobal();
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, t);
    const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    try f.appendStore(b, a, slot);
    const r = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    const h = halves(v);
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{ h[0], h[1] }, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(v)), got);
}

// An f128 add has no riscv64 instruction; the shared softfp pass lowers it to a call, and the
// backend must place the two f128 arguments in the a0:a1 and a2:a3 register pairs and emit a call
// relocation against the undefined `__addtf3`. Compile-only (no riscv64 libgcc in the harness to
// execute the arithmetic), but it exercises the real call-argument pair placement.
test "riscv64: an f128 add compiles to an undefined __addtf3 soft-fp call relocation" {
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const r = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    var compiled = try isel.compileFunction(allocator, &f, .{});
    defer compiled.deinit(allocator);
    var addtf3: usize = 0;
    for (compiled.relocs) |rel| {
        if (std.mem.eql(u8, rel.symbol, "__addtf3")) addtf3 += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), addtf3);
}
