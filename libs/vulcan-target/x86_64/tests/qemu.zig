//! qemu-x86_64 runner: execute the shared cases.zig under qemu-x86_64 user mode. The
//! harness wraps each function in a static Linux ELF and QEMU runs it. Skips when
//! qemu-x86_64 is not on PATH.

const std = @import("std");
const ir = @import("vulcan-ir");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;

test "x86-64 cases run under qemu-x86_64" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu);
}

test "an unreachable block that uses a reachable value compiles and the reachable path runs" {
    // The exact shape that tripped the shared allocator's SSA def-in-range assert before
    // `neutralizeUnreachable` was adopted: a value DEFINED in the reachable entry is USED by a block
    // NO reachable block branches to. The production `compile`/`selectFunction` must neutralize the
    // orphan block, tolerate its emptied (no-instruction, null-terminator) form in emission, and still
    // return the reachable sum. Skips when qemu-x86_64 is not on PATH.
    const a = std.testing.allocator;
    var func = Function.init(a);
    defer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });

    // entry: s = x + y ; ret s.
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i64_t);
    const y = try func.appendBlockParam(entry, i64_t);
    const s = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(entry, .{ .ret = s });

    // The unreachable block: it USES `s` (a reachable value) yet nothing branches to it.
    const dead = try func.appendBlock();
    const d = try func.appendInst(dead, i64_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = s } });
    func.setTerminator(dead, .{ .ret = d });

    // Compiles without crashing AND the reachable path executes correctly (5 + 3 == 8).
    try harness.expectRunFull(std.testing.io, a, &func, &.{ 5, 3 }, 8, harness.qemu);
}
