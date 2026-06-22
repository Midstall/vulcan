//! The NVIDIA GPU target: SASS (the native GPU ISA) encoding and codegen. Unlike
//! the CPU backends, generated code runs as a compute kernel under a launch
//! descriptor (no call stack, abundant registers, predicated execution, scoreboard
//! scheduling for variable-latency ops). The encoding is shared Volta..Blackwell.
//! Validation is by encoding tests (the core ops match prism's hardware-verified
//! bit patterns). Live GPU execution is driven from prism's compute dispatch.

const std = @import("std");

pub const encode = @import("nvidia/encode.zig");
pub const isel = @import("nvidia/isel.zig");
pub const schedule = @import("nvidia/schedule.zig");

/// Cross-cutting tests (no GPU execution here, so structural only). The SPIR-V ->
/// IR -> SASS path validates the frontend-to-GPU pipeline.
const tests = struct {
    pub const spirv = @import("nvidia/tests/spirv.zig");
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(tests);
}
