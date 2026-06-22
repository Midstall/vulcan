//! AArch64 (A64) target: encoding and codegen. Bring-up in progress. Host is
//! aarch64, so generated code is validated by native in-process execution.

const std = @import("std");

pub const encode = @import("aarch64/encode.zig");
pub const isel = @import("aarch64/isel.zig");
pub const link = @import("aarch64/link.zig");
pub const object = @import("aarch64/object.zig");
pub const ld = @import("aarch64/ld.zig");
pub const jit = @import("aarch64/jit.zig");

/// Execution-test runners, one per backend (see each file). Skip when the
/// backend is unavailable or incompatible.
const tests = struct {
    pub const native = @import("aarch64/tests/native.zig");
    pub const qemu = @import("aarch64/tests/qemu.zig");
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(tests);
}
