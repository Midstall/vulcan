//! RISC-V 64-bit target: encoding, registers, calling convention, and codegen.
//! The first-class Vulcan backend.

const std = @import("std");

pub const encode = @import("riscv64/encode.zig");
pub const isel = @import("riscv64/isel.zig");
pub const emit = @import("riscv64/emit.zig");
pub const schedule = @import("riscv64/schedule.zig");
pub const link = @import("riscv64/link.zig");
pub const object = @import("riscv64/object.zig");
pub const ld = @import("riscv64/ld.zig");
pub const jit = @import("riscv64/jit.zig");

/// Execution-test runners, one per backend (see each file). Each skips when its
/// backend's tool is unavailable or its machine is incompatible.
const tests = struct {
    pub const harness = @import("riscv64/tests/harness.zig");
    pub const river = @import("riscv64/tests/river.zig");
    pub const spike = @import("riscv64/tests/spike.zig");
    pub const qemu = @import("riscv64/tests/qemu.zig");
    pub const native = @import("riscv64/tests/native.zig");
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(tests);
}
