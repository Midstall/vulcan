//! Baremetal/freestanding executable images. A flat binary is the linked code and
//! data with no container: load it at the link base address and jump to the entry
//! point. For ROM/flash or a bootloader's "load this blob at address X" path. No OS,
//! no dynamic linking, no runtime. Self-contained machine code.
//!
//! Baremetal ELF executables live per-backend in `ld.writeElfExec` (a static
//! `ET_EXEC` entered at a fixed address). UEFI PE32+ applications are in `pe.zig`.
//! Freestanding-clean.

const std = @import("std");

pub const Error = std.mem.Allocator.Error;

/// Produce a flat binary from a linked code image. When `zero_fill_bss` is true the
/// result is `mem_size` bytes (the code followed by zero-initialized `.bss`), as a
/// ROM/flash image. Otherwise it is just the `code` bytes. Caller owns the result.
pub fn flatBinary(allocator: std.mem.Allocator, code: []const u8, mem_size: u64, zero_fill_bss: bool) Error![]u8 {
    const len: usize = if (zero_fill_bss) @intCast(@max(mem_size, code.len)) else code.len;
    const buf = try allocator.alloc(u8, len);
    @memset(buf, 0);
    @memcpy(buf[0..code.len], code);
    return buf;
}

test "flat binary is the raw code, optionally with a zeroed bss tail" {
    const allocator = std.testing.allocator;
    const code = [_]u8{ 0x11, 0x22, 0x33, 0x44 };

    // Without bss fill: exactly the code.
    const raw = try flatBinary(allocator, &code, 16, false);
    defer allocator.free(raw);
    try std.testing.expectEqualSlices(u8, &code, raw);

    // With bss fill: code followed by zeros up to mem_size.
    const rom = try flatBinary(allocator, &code, 8, true);
    defer allocator.free(rom);
    try std.testing.expectEqual(@as(usize, 8), rom.len);
    try std.testing.expectEqualSlices(u8, &code, rom[0..4]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, rom[4..8]);
}
