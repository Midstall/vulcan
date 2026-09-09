//! qemu-i386 runner: execute the shared cases.zig under qemu-i386 user mode. The harness
//! wraps each function in a static ELF and QEMU runs it. Skips when qemu-i386 is not on
//! PATH.

const std = @import("std");
const ir = @import("vulcan-ir");
const cases = @import("cases.zig");
const harness = @import("harness.zig");
const link = @import("../link.zig");

const Function = ir.function.Function;

test "i386 cases run under qemu-i386" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu);
}

test "an unreachable block that uses a reachable value compiles and the reachable path runs" {
    // The exact shape that tripped the shared allocator's SSA def-in-range assert before
    // `neutralizeUnreachable` was adopted: a value DEFINED in the reachable entry is USED by a block
    // NO reachable block branches to. The production `compile`/`selectFunction` must neutralize the
    // orphan block, tolerate its emptied (no-instruction, null-terminator) form in emission, and still
    // return the reachable sum. Skips when qemu-i386 is not on PATH.
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    // entry: s = x + y ; ret s.
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const y = try func.appendBlockParam(entry, i32_t);
    const s = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(s) });

    // The unreachable block: it USES `s` (a reachable value) yet nothing branches to it.
    const dead = try func.appendBlock();
    const d = try func.appendInst(dead, i32_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = s } });
    func.setTerminator(dead, .{ .ret = ir.function.Ret.one(d) });

    // Compiles without crashing AND the reachable path executes correctly (5 + 3 == 8, checked by
    // the low result byte, which the i386 harness carries out through the process exit code).
    try harness.expectRun(std.testing.io, a, &func, &.{ 5, 3 }, 8, harness.qemu);
}

test "a rodata global read via global_addr returns its value under qemu-i386" {
    // main() -> *(&K), K a rodata i32 constant. Proves isel's `global_addr` arm
    // (`mov rd, imm32`) end to end: a real `.abs32` reloc, resolved (here, statically by
    // `runModuleData`) to K's ABSOLUTE runtime address, executed on real i386 machine
    // code under qemu-i386.
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var main_f = Function.init(allocator);
    defer main_f.deinit();
    {
        const t = try main_f.types.intern(i32k);
        const ptr_t = try main_f.types.ptrGlobal();
        const b = try main_f.appendBlock();
        const g = try main_f.appendGlobalAddr(b, ptr_t, "K");
        const v = try main_f.appendInst(b, t, .{ .load = .{ .ptr = g } });
        main_f.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }
    const k_bytes = [_]u8{ 42, 0, 0, 0 }; // i32 42, little-endian
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main_f);
    try module.addData(allocator, "K", &k_bytes);

    try std.testing.expectEqual(@as(u8, 42), try harness.runModuleData(io, allocator, &module, &.{}, harness.qemu));
}
