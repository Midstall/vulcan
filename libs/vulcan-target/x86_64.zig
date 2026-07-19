//! x86-64 target: encoding and codegen. Host is aarch64, so generated code is
//! validated by execution under `qemu-x86_64` (user-mode), not in-process.

const std = @import("std");

pub const encode = @import("x86_64/encode.zig");
pub const disasm = @import("x86_64/disasm.zig");
pub const isel = @import("x86_64/isel.zig");
pub const link = @import("x86_64/link.zig");
pub const object = @import("x86_64/object.zig");
pub const elf = @import("x86_64/elf.zig");
/// W^X JIT buffer (x86-64 is cache-coherent). On a non-x86-64 host the buffer maps
/// correctly but its code must not be called in-process (use qemu).
pub const jit = @import("coherent_jit.zig");

/// Execution-test runners over the shared cases (cases.zig + harness.zig): `qemu`
/// builds a static Linux ELF and runs it under qemu-x86_64. `native` calls the code
/// in-process via the JIT (skips off an x86-64 host).
const tests = struct {
    pub const harness = @import("x86_64/tests/harness.zig");
    pub const cases = @import("x86_64/tests/cases.zig");
    pub const qemu = @import("x86_64/tests/qemu.zig");
    pub const native = @import("x86_64/tests/native.zig");
    pub const f16_emulation = @import("x86_64/tests/f16.zig");
    pub const wimmer_diff = @import("x86_64/tests/wimmer_diff.zig");
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(tests);
}
