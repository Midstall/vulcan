//! SPIR-V binary reader: validate the header and iterate the instruction stream. A
//! module is 32-bit words: a 5-word header (magic, version, generator, id-bound, schema)
//! then instructions. Each instruction's first word packs `(word_count << 16) | opcode`,
//! and the remaining `word_count - 1` words are operands.

const std = @import("std");

/// SPIR-V magic number (little-endian). A byte-swapped value is a big-endian module,
/// which this reader does not handle.
pub const magic: u32 = 0x07230203;

pub const Error = error{ NotSpirv, BigEndian, Truncated, MalformedInstruction };

/// A decoded SPIR-V module header.
pub const Header = struct {
    version: u32,
    generator: u32,
    id_bound: u32, // ids are in the range [0, id_bound)
};

/// One instruction: its opcode and a borrowed slice of operand words.
pub const Instruction = struct {
    opcode: u16,
    operands: []const u32,
};

/// Cursor over a SPIR-V word stream. `init` validates the header, `next` yields
/// instructions until the stream is exhausted.
pub const Reader = struct {
    words: []const u32,
    pos: usize, // index of the next instruction's first word
    header: Header,

    pub fn init(words: []const u32) Error!Reader {
        if (words.len < 5) return error.Truncated;
        if (words[0] != magic) {
            // Byte-swapped magic identifies a big-endian module.
            if (@byteSwap(words[0]) == magic) return error.BigEndian;
            return error.NotSpirv;
        }
        return .{
            .words = words,
            .pos = 5,
            .header = .{ .version = words[1], .generator = words[2], .id_bound = words[3] },
        };
    }

    /// Build a reader from raw bytes (length must be a multiple of 4 and 4-aligned).
    pub fn fromBytes(bytes: []const u8) Error!Reader {
        if (bytes.len % 4 != 0) return error.Truncated;
        if (@intFromPtr(bytes.ptr) % 4 != 0) return error.MalformedInstruction;
        const words = std.mem.bytesAsSlice(u32, bytes);
        return init(words);
    }

    /// The next instruction, or null at end of stream.
    pub fn next(self: *Reader) Error!?Instruction {
        if (self.pos >= self.words.len) return null;
        const head = self.words[self.pos];
        const word_count: usize = head >> 16;
        const opcode: u16 = @truncate(head & 0xffff);
        if (word_count == 0) return error.MalformedInstruction;
        if (self.pos + word_count > self.words.len) return error.Truncated;
        const operands = self.words[self.pos + 1 .. self.pos + word_count];
        self.pos += word_count;
        return .{ .opcode = opcode, .operands = operands };
    }
};

/// Builds SPIR-V word streams in tests: append a header, then instructions whose word
/// count is computed from the operands.
pub const Builder = struct {
    words: std.ArrayList(u32) = .empty,

    pub fn init(allocator: std.mem.Allocator, id_bound: u32) std.mem.Allocator.Error!Builder {
        var b: Builder = .{};
        try b.words.appendSlice(allocator, &.{ magic, 0x00010000, 0, id_bound, 0 });
        return b;
    }

    pub fn deinit(self: *Builder, allocator: std.mem.Allocator) void {
        self.words.deinit(allocator);
    }

    /// Emit one instruction: `(word_count << 16) | opcode` then the operands.
    pub fn emit(self: *Builder, allocator: std.mem.Allocator, opcode: u16, operands: []const u32) std.mem.Allocator.Error!void {
        const word_count: u32 = @intCast(1 + operands.len);
        try self.words.append(allocator, (word_count << 16) | opcode);
        try self.words.appendSlice(allocator, operands);
    }
};

test "reads a header and iterates instructions" {
    const allocator = std.testing.allocator;
    var b = try Builder.init(allocator, 3);
    defer b.deinit(allocator);
    try b.emit(allocator, 17, &.{1}); // OpCapability Shader
    try b.emit(allocator, 21, &.{ 1, 32, 1 }); // OpTypeInt %1 32 1

    var r = try Reader.init(b.words.items);
    try std.testing.expectEqual(@as(u32, 3), r.header.id_bound);

    const cap = (try r.next()).?;
    try std.testing.expectEqual(@as(u16, 17), cap.opcode);
    try std.testing.expectEqualSlices(u32, &.{1}, cap.operands);

    const tint = (try r.next()).?;
    try std.testing.expectEqual(@as(u16, 21), tint.opcode);
    try std.testing.expectEqualSlices(u32, &.{ 1, 32, 1 }, tint.operands);

    try std.testing.expectEqual(@as(?Instruction, null), try r.next());
}

test "rejects a non-SPIR-V and a big-endian stream" {
    try std.testing.expectError(error.NotSpirv, Reader.init(&.{ 0xdeadbeef, 0, 0, 0, 0 }));
    try std.testing.expectError(error.BigEndian, Reader.init(&.{ @byteSwap(magic), 0, 0, 0, 0 }));
    try std.testing.expectError(error.Truncated, Reader.init(&.{ magic, 0 }));
}
