//! QEMU user-mode runner: `qemu-riscv64`. Runs the shared codegen/optimization case corpus as plain
//! Linux static ELFs (syscall write/exit), so the RISC-V backend executes on any machine with qemu
//! even when River and Spike are absent. Skips when qemu-riscv64 is not on PATH.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

test "qemu-user-riscv: shared codegen and optimization cases" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu_user);
}
