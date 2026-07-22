//! qemu-i386 runner: execute the shared cases.zig under qemu-i386 user mode. The harness
//! wraps each function in a static ELF and QEMU runs it. Skips when qemu-i386 is not on
//! PATH.

const std = @import("std");
const ir = @import("vulcan-ir");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

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
    func.setTerminator(entry, .{ .ret = s });

    // The unreachable block: it USES `s` (a reachable value) yet nothing branches to it.
    const dead = try func.appendBlock();
    const d = try func.appendInst(dead, i32_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = s } });
    func.setTerminator(dead, .{ .ret = d });

    // Compiles without crashing AND the reachable path executes correctly (5 + 3 == 8, checked by
    // the low result byte, which the i386 harness carries out through the process exit code).
    try harness.expectRun(std.testing.io, a, &func, &.{ 5, 3 }, 8, harness.qemu);
}
