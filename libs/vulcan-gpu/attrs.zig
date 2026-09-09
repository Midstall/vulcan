//! Typed access to the `vulcan.gpu` attribute namespace. Every layer reads and writes kernel
//! metadata through this file, so the namespace and key strings appear once.
//!
//! The IR stores attributes in a flat list and `addAttr` appends without replacing, so a key
//! can appear more than once on the same target. Every getter here returns the FIRST match,
//! which is the value the first writer set. There is no remove operation in the IR.

const std = @import("std");
const ir = @import("vulcan-ir");
const builtin_mod = @import("builtin.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Builtin = builtin_mod.Builtin;

pub const Error = std.mem.Allocator.Error;

/// The attribute namespace every key below lives in.
pub const namespace = "vulcan.gpu";

const key_builtin = "builtin";
const key_local_size = [3][]const u8{ "local_size_x", "local_size_y", "local_size_z" };
const key_shared_bytes = "shared_bytes";

/// The first `vulcan.gpu` integer attribute with `key` on `target`, or null when absent.
fn intAttr(func: *const Function, target: ir.function.AttrTarget, key: []const u8) ?i64 {
    var it = func.attributesOf(target);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (!std.mem.eql(u8, c.namespace, namespace)) continue;
            if (!std.mem.eql(u8, c.key, key)) continue;
            return switch (c.value) {
                .int => |n| n,
                .flag, .string => null,
            };
        },
        .@"inline", .noreturn, .cold, .@"align", .endian => {},
    };
    return null;
}

fn addInt(func: *Function, target: ir.function.AttrTarget, key: []const u8, n: i64) Error!void {
    try func.addAttr(target, .{ .custom = .{ .namespace = namespace, .key = key, .value = .{ .int = n } } });
}

/// Tag `value` as sourced from hardware rather than from the parameter block.
pub fn setBuiltin(func: *Function, value: Value, b: Builtin) Error!void {
    try addInt(func, .{ .value = value }, key_builtin, @intFromEnum(b));
}

/// The builtin `value` carries, or null when it is an ordinary parameter.
///
/// The stored payload is UNTRUSTED: it survives a text round trip and any frontend can write
/// it, so an out-of-range number must not become an invalid enum. `std.enums.fromInt` returns
/// null instead, and the caller treats that as "not a builtin".
pub fn builtinOf(func: *const Function, value: Value) ?Builtin {
    const raw = intAttr(func, .{ .value = value }, key_builtin) orelse return null;
    if (raw < 0 or raw > std.math.maxInt(u16)) return null;
    return std.enums.fromInt(Builtin, @as(u16, @intCast(raw)));
}

/// Record the workgroup size the kernel declares.
pub fn setLocalSize(func: *Function, size: [3]u32) Error!void {
    for (size, key_local_size) |n, key| try addInt(func, .func, key, n);
}

/// The declared workgroup size. Each axis defaults to 1 when the frontend recorded none,
/// which is the correct identity for a kernel that declares no size.
pub fn localSize(func: *const Function) [3]u32 {
    var out: [3]u32 = .{ 1, 1, 1 };
    for (&out, key_local_size) |*slot, key| {
        const raw = intAttr(func, .func, key) orelse continue;
        if (raw > 0 and raw <= std.math.maxInt(u32)) slot.* = @intCast(raw);
    }
    return out;
}

/// Record the workgroup shared memory the kernel needs, in bytes.
pub fn setSharedBytes(func: *Function, bytes: u32) Error!void {
    try addInt(func, .func, key_shared_bytes, bytes);
}

/// The workgroup shared memory the kernel needs, in bytes. Zero when it needs none.
pub fn sharedBytes(func: *const Function) u32 {
    const raw = intAttr(func, .func, key_shared_bytes) orelse return 0;
    if (raw < 0 or raw > std.math.maxInt(u32)) return 0;
    return @intCast(raw);
}

test "a builtin tag round-trips through the attribute store" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try setBuiltin(&func, v, .block_id_y);
    try std.testing.expectEqual(@as(?Builtin, .block_id_y), builtinOf(&func, v));
}

test "an untagged parameter reports no builtin" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try std.testing.expectEqual(@as(?Builtin, null), builtinOf(&func, v));
}

test "an out-of-range builtin payload reports null instead of an invalid enum" {
    // Suspicious case: the payload is untrusted, so a bad number must not become a Builtin.
    // @enumFromInt here would be undefined behaviour.
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try func.addAttr(.{ .value = v }, .{ .custom = .{
        .namespace = namespace,
        .key = "builtin",
        .value = .{ .int = 9999 },
    } });
    try std.testing.expectEqual(@as(?Builtin, null), builtinOf(&func, v));
}

test "a negative builtin payload reports null" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try func.addAttr(.{ .value = v }, .{ .custom = .{
        .namespace = namespace,
        .key = "builtin",
        .value = .{ .int = -1 },
    } });
    try std.testing.expectEqual(@as(?Builtin, null), builtinOf(&func, v));
}

test "local size round-trips all three axes" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    try setLocalSize(&func, .{ 64, 2, 1 });
    try std.testing.expectEqual([3]u32{ 64, 2, 1 }, localSize(&func));
}

test "local size defaults to one on every axis when unset" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, localSize(&func));
}

test "shared bytes round-trips and defaults to zero" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    try std.testing.expectEqual(@as(u32, 0), sharedBytes(&func));
    try setSharedBytes(&func, 4096);
    try std.testing.expectEqual(@as(u32, 4096), sharedBytes(&func));
}

test "a foreign namespace with the same key is ignored" {
    // Suspicious case: the attribute bag is open, so another layer may use the key "builtin".
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try func.addAttr(.{ .value = v }, .{ .custom = .{
        .namespace = "other.layer",
        .key = "builtin",
        .value = .{ .int = 0 },
    } });
    try std.testing.expectEqual(@as(?Builtin, null), builtinOf(&func, v));
}
