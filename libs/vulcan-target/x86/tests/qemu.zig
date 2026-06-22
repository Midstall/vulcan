//! qemu-i386 runner: execute the shared cases.zig under qemu-i386 user mode. The harness
//! wraps each function in a static ELF and QEMU runs it. Skips when qemu-i386 is not on
//! PATH.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

test "i386 cases run under qemu-i386" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu);
}
