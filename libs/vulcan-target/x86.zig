//! x86 (32-bit / i386) target: encoding and codegen. Host is aarch64, so generated
//! code is validated by execution under `qemu-i386` (user-mode).

const std = @import("std");

pub const encode = @import("x86/encode.zig");
pub const disasm = @import("x86/disasm.zig");
pub const isel = @import("x86/isel.zig");
pub const link = @import("x86/link.zig");
pub const object = @import("x86/object.zig");
pub const elf = @import("x86/elf.zig");
/// W^X JIT buffer (i386 is cache-coherent). On a non-i386 host the buffer maps
/// correctly but its code must not be called in-process (use qemu).
pub const jit = @import("coherent_jit.zig");

/// Execution-test runners over the shared cases (cases.zig + harness.zig): `qemu`
/// builds a static i386 ELF and runs it under qemu-i386. `native` calls the code
/// in-process via the JIT (skips off an i386 host).
const tests = struct {
    pub const harness = @import("x86/tests/harness.zig");
    pub const cases = @import("x86/tests/cases.zig");
    pub const qemu = @import("x86/tests/qemu.zig");
    pub const native = @import("x86/tests/native.zig");
    pub const addrfold = @import("x86/tests/addrfold.zig");
    pub const wimmer_diff = @import("x86/tests/wimmer_diff.zig");
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(tests);
}
