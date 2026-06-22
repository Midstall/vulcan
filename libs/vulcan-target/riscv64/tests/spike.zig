//! Spike backend runner: the RISC-V ISA simulator. Currently marked incompatible
//! (the firmware emits results over an MMIO 16550 UART at 0x10000000, whereas
//! spike uses the HTIF tohost mechanism), so these cases skip. Adapting the entry
//! stub to write tohost would let them run.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

test "spike: shared codegen and optimization cases" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.spike);
}
