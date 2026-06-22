//! Native runner: execute the shared cases.zig in-process by mapping the compiled code
//! into W^X memory and calling it directly. Valid only when the host is x86-64. On any
//! other host every case skips, since x86 machine code cannot run natively there. On an
//! x86-64 host this also exercises the JIT (coherent_jit) path.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

test "x86-64 cases run natively in-process (skips off x86-64)" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.native);
}
