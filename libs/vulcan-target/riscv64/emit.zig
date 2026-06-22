//! Emission: turn selected machine words into a little-endian byte stream, the
//! actual RISC-V machine code.

const std = @import("std");

/// Encode machine words as little-endian bytes. The caller owns the result.
pub fn emitBytes(allocator: std.mem.Allocator, words: []const u32) std.mem.Allocator.Error![]u8 {
    const bytes = try allocator.alloc(u8, words.len * 4);
    for (words, 0..) |word, i| {
        std.mem.writeInt(u32, bytes[i * 4 ..][0..4], word, .little);
    }
    return bytes;
}

test "emits machine words as little-endian bytes" {
    const words = [_]u32{ 0x003100b3, 0x00008067 };
    const bytes = try emitBytes(std.testing.allocator, &words);
    defer std.testing.allocator.free(bytes);

    try std.testing.expectEqualSlices(u8, &.{
        0xb3, 0x00, 0x31, 0x00, // add x1, x2, x3
        0x67, 0x80, 0x00, 0x00, // ret
    }, bytes);
}
