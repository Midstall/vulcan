//! Native runner: execute the shared cases.zig in-process by mapping the compiled code
//! into W^X memory and calling it. Valid only when the host is i386. On any other host
//! every case skips, since i386 machine code cannot run natively there. On an i386 host
//! it also exercises the JIT (coherent_jit) path.

const std = @import("std");
const cases = @import("cases.zig");
const harness = @import("harness.zig");

test "i386 cases run natively in-process (skips off i386)" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.native);
}
