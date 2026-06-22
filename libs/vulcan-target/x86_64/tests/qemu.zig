//! qemu-x86_64 runner: execute the shared cases.zig under qemu-x86_64 user mode. The
//! harness wraps each function in a static Linux ELF and QEMU runs it. Skips when
//! qemu-x86_64 is not on PATH.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

test "x86-64 cases run under qemu-x86_64" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu);
}
