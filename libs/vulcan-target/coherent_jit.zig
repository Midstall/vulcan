//! W^X executable-memory buffer for cache-coherent ISAs (x86 / x86-64), where the
//! instruction and data caches are coherent so no explicit icache sync is needed.
//! Memory comes from a pluggable `jit_platform.Provider` (posix by default, UEFI/etc.
//! via `CodeBuffer.mapWith`).
//!
//! When the host ISA differs from the code's ISA (e.g. building x86 code on an
//! aarch64 host), the buffer is mapped correctly but `entry`'s function pointer must
//! not be called: the bytes are for a different CPU. Run under an emulator (qemu).

const std = @import("std");
const platform = @import("jit_platform.zig");

pub const Error = platform.Error;
pub const Provider = platform.Provider;

fn noSync(_: []const u8) void {}

/// A page of executable memory holding generated machine code (no icache sync needed).
pub const CodeBuffer = platform.Buffer(noSync);

test "maps code into an executable buffer (W^X)" {
    // x86-64 `mov eax, 42` then `ret` (the bytes are not run here - host may differ).
    const code = [_]u8{ 0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3 };
    var buf = try CodeBuffer.map(&code);
    defer buf.deinit();
    try std.testing.expectEqualSlices(u8, &code, buf.memory[0..code.len]);
    _ = buf.entry(*const fn () callconv(.c) i32, 0); // a valid pointer, not called cross-ISA
}
