//! QEMU user-mode runner with RVC compression: runs the shared case corpus as compressed RV64GC
//! static ELFs (every self-contained case is compressed before running). This turns the whole corpus
//! into an execution test of the RVC compressor on real, diverse codegen, not just hand-built cases.
//! Skips when qemu-riscv64 is not on PATH.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

test "qemu-user-riscv (RVC): shared codegen and optimization cases run compressed" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu_user_rvc);
}
