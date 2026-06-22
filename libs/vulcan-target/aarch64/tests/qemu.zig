//! QEMU backend runner for AArch64 (`qemu-aarch64` user-mode). The development host
//! is aarch64, so the native runner already links a module and executes the standalone
//! ELF in-process (see "object+ld+exec" in native.zig). No emulator is needed for
//! execution coverage. Reserved for cross-checking on a non-aarch64 host, not yet
//! wired into this environment.

const std = @import("std");

test "qemu-aarch64: ELF execution is covered natively (see native.zig object+ld+exec)" {
    return error.SkipZigTest;
}
