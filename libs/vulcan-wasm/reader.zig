//! Reader for the WebAssembly binary format: a byte cursor with LEB128 decoding.

const std = @import("std");

pub const Error = error{ InvalidWasm, UnexpectedEof } || std.mem.Allocator.Error;

/// The Wasm magic (`\0asm`) and version 1, at the start of every module.
pub const magic = [_]u8{ 0x00, 0x61, 0x73, 0x6D };
pub const version = [_]u8{ 0x01, 0x00, 0x00, 0x00 };

/// A forward cursor over a byte slice.
pub const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,

    pub fn atEnd(self: *const Cursor) bool {
        return self.pos >= self.bytes.len;
    }

    pub fn byte(self: *Cursor) Error!u8 {
        if (self.pos >= self.bytes.len) return error.UnexpectedEof;
        defer self.pos += 1;
        return self.bytes[self.pos];
    }

    /// Take the next `n` bytes as a sub-slice (borrowed from the input).
    pub fn take(self: *Cursor, n: usize) Error![]const u8 {
        if (self.pos + n > self.bytes.len) return error.UnexpectedEof;
        defer self.pos += n;
        return self.bytes[self.pos .. self.pos + n];
    }

    /// An unsigned LEB128 integer.
    pub fn uleb(self: *Cursor) Error!u64 {
        var result: u64 = 0;
        var shift: u32 = 0;
        while (true) {
            const b = try self.byte();
            if (shift >= 64) return error.InvalidWasm;
            result |= @as(u64, b & 0x7F) << @intCast(shift);
            shift += 7;
            if (b & 0x80 == 0) break;
        }
        return result;
    }

    /// A signed LEB128 integer (sign-extended).
    pub fn sleb(self: *Cursor) Error!i64 {
        var result: u64 = 0;
        var shift: u32 = 0;
        var b: u8 = 0;
        while (true) {
            b = try self.byte();
            if (shift >= 64) return error.InvalidWasm;
            result |= @as(u64, b & 0x7F) << @intCast(shift);
            shift += 7;
            if (b & 0x80 == 0) break;
        }
        if (shift < 64 and (b & 0x40) != 0) result |= ~@as(u64, 0) << @intCast(shift);
        return @bitCast(result);
    }

    /// A `u32` LEB128 (the common index/count form).
    pub fn u32leb(self: *Cursor) Error!u32 {
        const v = try self.uleb();
        if (v > std.math.maxInt(u32)) return error.InvalidWasm;
        return @intCast(v);
    }

    /// A Wasm name: a LEB length followed by that many UTF-8 bytes (borrowed).
    pub fn name(self: *Cursor) Error![]const u8 {
        const n = try self.u32leb();
        return self.take(n);
    }
};

test "uleb decodes multi-byte values" {
    var c = Cursor{ .bytes = &.{ 0xE5, 0x8E, 0x26 } }; // 624485
    try std.testing.expectEqual(@as(u64, 624485), try c.uleb());
    try std.testing.expect(c.atEnd());
}

test "sleb decodes negative values" {
    var c = Cursor{ .bytes = &.{0x7F} }; // -1
    try std.testing.expectEqual(@as(i64, -1), try c.sleb());
    var c2 = Cursor{ .bytes = &.{ 0xC0, 0xBB, 0x78 } }; // -123456
    try std.testing.expectEqual(@as(i64, -123456), try c2.sleb());
}

test "single-byte small values" {
    var c = Cursor{ .bytes = &.{0x02} };
    try std.testing.expectEqual(@as(u64, 2), try c.uleb());
}
