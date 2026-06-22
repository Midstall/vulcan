//! QEMU backend runner: `qemu-system-riscv64 -M virt`. Currently marked
//! incompatible (the firmware self-loops with no exit, so QEMU would run
//! forever), so these cases skip. Adding a shutdown device / SBI poweroff to the
//! entry stub would let them run.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

test "qemu-riscv: shared codegen and optimization cases" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu);
}
