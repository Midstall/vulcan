//! Entity references and the dense pools backing them. IR entities (values,
//! instructions, blocks) are typed `u32` indices into dense pools, never
//! pointers, keeping the IR cache-friendly, serializable, and freestanding.

const std = @import("std");

/// A dense, append-only pool of `T`, handing out typed references. `Ref` is
/// distinct per element type, so a reference into one pool cannot be mistaken
/// for a reference into another.
pub fn Pool(comptime T: type) type {
    return struct {
        const Self = @This();

        /// A typed index into this pool. Non-exhaustive so only the pool mints
        /// valid values.
        pub const Ref = enum(u32) { _ };

        allocator: std.mem.Allocator,
        items: std.ArrayList(T),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator, .items = .empty };
        }

        pub fn deinit(self: *Self) void {
            self.items.deinit(self.allocator);
        }

        /// Append an item, returning its reference.
        pub fn append(self: *Self, item: T) std.mem.Allocator.Error!Ref {
            const index: u32 = @intCast(self.items.items.len);
            try self.items.append(self.allocator, item);
            return @enumFromInt(index);
        }

        /// Borrow a mutable pointer to a stored item.
        pub fn get(self: *Self, ref: Ref) *T {
            return &self.items.items[@intFromEnum(ref)];
        }
    };
}

test "pool append returns distinct refs and get round-trips" {
    var pool = Pool(u32).init(std.testing.allocator);
    defer pool.deinit();

    const a = try pool.append(10);
    const b = try pool.append(20);

    try std.testing.expect(a != b);
    try std.testing.expectEqual(@as(u32, 10), pool.get(a).*);
    try std.testing.expectEqual(@as(u32, 20), pool.get(b).*);
}
