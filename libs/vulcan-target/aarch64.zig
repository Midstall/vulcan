//! AArch64 (A64) target: encoding and codegen. Bring-up in progress. Host is
//! aarch64, so generated code is validated by native in-process execution.

const std = @import("std");

pub const encode = @import("aarch64/encode.zig");
pub const disasm = @import("aarch64/disasm.zig");
pub const isel = @import("aarch64/isel.zig");
pub const peephole = @import("aarch64/peephole.zig");
pub const link = @import("aarch64/link.zig");
pub const object = @import("aarch64/object.zig");
pub const jit = @import("aarch64/jit.zig");

/// Execution-test runners, one per backend (see each file). Skip when the
/// backend is unavailable or incompatible.
const tests = struct {
    pub const native = @import("aarch64/tests/native.zig");
    pub const wimmer_native = @import("aarch64/tests/wimmer_native.zig");
    pub const qemu = @import("aarch64/tests/qemu.zig");
    pub const global_addr = @import("aarch64/tests/global_addr.zig");
    pub const link_archive = @import("aarch64/tests/link_archive.zig");
    pub const link_script = @import("aarch64/tests/link_script.zig");
    pub const dynamic = @import("aarch64/tests/dynamic.zig");
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(tests);
}
