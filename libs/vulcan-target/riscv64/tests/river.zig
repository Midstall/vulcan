//! River backend runner: executes the shared RISC-V cases on Midstall's River
//! functional emulator (`river-emulator`). Skips if it is not on PATH.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

test "river: shared codegen and optimization cases" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.river);
}
