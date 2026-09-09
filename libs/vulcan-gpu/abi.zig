//! The per-target ABI descriptor and the shared parameter layout algorithm. Every backend
//! places its parameter block through `layoutParams`, so the placement rule lives once.

const std = @import("std");
const ir = @import("vulcan-ir");
const attrs = @import("attrs.zig");
const kernel = @import("kernel.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Param = kernel.Param;
const Layout = kernel.Layout;

pub const Error = std.mem.Allocator.Error || error{ UnsupportedParamType, SharedMemoryTooLarge };

/// How a target lays out and delivers a kernel's parameter block.
pub const Abi = struct {
    /// The byte offset in the target's parameter memory where the block starts. On NVIDIA this
    /// is the constant-bank offset: 0x160 under the CUDA driver convention, which puts a
    /// driver-owned block in front of the parameters, and 0 for a runtime that binds its own
    /// parameter buffer as the base of the bank. It is a property of the RUNTIME's binding
    /// convention, not of the silicon, so it is data and not a constant.
    param_base: u32,
    /// The width of an address in bytes.
    pointer_bytes: u8,
    /// The minimum alignment of any parameter in bytes. A parameter wider than this aligns to
    /// its own size.
    param_align: u8,
    /// The most workgroup shared memory a kernel may request, in bytes.
    max_shared_bytes: u32,
};

/// Align `offset` up to `alignment`, which must be a power of two.
fn alignUp(offset: u32, alignment: u32) u32 {
    std.debug.assert(alignment != 0 and @popCount(alignment) == 1);
    return (offset + alignment - 1) & ~(alignment - 1);
}

/// The size in bytes of a scalar parameter, or null when the backend cannot place it.
/// Aggregates are rejected: a kernel signature flattens them before it reaches here.
fn scalarBytes(func: *const Function, v: Value) ?u8 {
    return switch (func.types.type_kind(func.valueType(v))) {
        .bool => 1,
        .int => |i| if (i.bits <= 64 and i.bits % 8 == 0) @intCast(i.bits / 8) else null,
        .float => |f| switch (f) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
            .f128 => 16,
        },
        .ptr, .vector, .@"struct", .array, .slice => null,
    };
}

/// Place the entry block's parameters into a parameter block.
///
/// Builtin-tagged parameters are SKIPPED: the hardware supplies them, so they take no space.
/// When `returns_value` is true an implicit output pointer is placed first, and the kernel
/// writes its result through it.
///
/// The caller OWNS the result and must release it with `Layout.deinit`.
pub fn layoutParams(
    allocator: std.mem.Allocator,
    func: *const Function,
    a: Abi,
    returns_value: bool,
) Error!Layout {
    if (attrs.sharedBytes(func) > a.max_shared_bytes) return error.SharedMemoryTooLarge;

    var placed: std.ArrayList(Param) = .empty;
    errdefer placed.deinit(allocator);

    var cursor: u32 = 0;
    var out_pointer: ?Param = null;
    if (returns_value) {
        cursor = alignUp(cursor, a.pointer_bytes);
        out_pointer = .{ .offset = cursor, .size = a.pointer_bytes, .kind = .{ .pointer = .global } };
        cursor += a.pointer_bytes;
    }

    for (func.blockParams(@enumFromInt(0))) |p| {
        if (attrs.builtinOf(func, p) != null) continue;

        const is_ptr = func.types.type_kind(func.valueType(p)) == .ptr;
        const size: u32 = if (is_ptr) a.pointer_bytes else scalarBytes(func, p) orelse
            return error.UnsupportedParamType;
        const alignment = @max(@as(u32, a.param_align), size);

        cursor = alignUp(cursor, alignment);
        try placed.append(allocator, .{
            .offset = cursor,
            .size = size,
            // M1 has no IR address space, so every pointer parameter is global. M1.5 reads
            // the space off the pointer type instead.
            .kind = if (is_ptr) .{ .pointer = .global } else .{ .scalar = @intCast(size) },
        });
        cursor += size;
    }

    return .{
        .params = try placed.toOwnedSlice(allocator),
        .bytes = cursor,
        .out_pointer = out_pointer,
    };
}

const nvidia_test_abi: Abi = .{
    .param_base = 0x160,
    .pointer_bytes = 8,
    .param_align = 4,
    .max_shared_bytes = 48 * 1024,
};

fn testFunc(allocator: std.mem.Allocator) !Function {
    return Function.init(allocator);
}

test "two i32 parameters pack at four-byte offsets" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, t);
    _ = try func.appendBlockParam(b, t);

    var layout = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, false);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), layout.params.len);
    try std.testing.expectEqual(@as(u32, 0), layout.params[0].offset);
    try std.testing.expectEqual(@as(u32, 4), layout.params[1].offset);
    try std.testing.expectEqual(@as(u32, 8), layout.bytes);
    try std.testing.expectEqual(@as(?Param, null), layout.out_pointer);
}

test "a pointer parameter aligns to eight even after a four-byte scalar" {
    // Suspicious case: the scalar leaves the cursor at 4, and a pointer must not straddle.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, i32_t);
    _ = try func.appendBlockParam(b, ptr_t);

    var layout = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, false);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 0), layout.params[0].offset);
    try std.testing.expectEqual(@as(u32, 8), layout.params[1].offset);
    try std.testing.expectEqual(@as(u32, 16), layout.bytes);
    try std.testing.expectEqual(kernel.AddressSpace.global, layout.params[1].kind.pointer);
}

test "a value-returning kernel places the output pointer first" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, t);

    var layout = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, true);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 0), layout.out_pointer.?.offset);
    try std.testing.expectEqual(@as(u32, 8), layout.out_pointer.?.size);
    try std.testing.expectEqual(@as(u32, 8), layout.params[0].offset);
    try std.testing.expectEqual(@as(u32, 12), layout.bytes);
}

test "a builtin parameter takes no space in the block" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const gid = try func.appendBlockParam(b, t);
    _ = try func.appendBlockParam(b, t);
    try attrs.setBuiltin(&func, gid, .global_id_x);

    var layout = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, false);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), layout.params.len);
    try std.testing.expectEqual(@as(u32, 0), layout.params[0].offset);
    try std.testing.expectEqual(@as(u32, 4), layout.bytes);
}

test "an empty parameter list produces a zero-byte block" {
    // Suspicious case: the boundary where there is nothing to place.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    _ = try func.appendBlock();

    var layout = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, false);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), layout.params.len);
    try std.testing.expectEqual(@as(u32, 0), layout.bytes);
}

test "an aggregate parameter is rejected rather than misplaced" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const arr_t = try func.types.intern(.{ .array = .{ .len = 4, .elem = i32_t } });
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, arr_t);

    try std.testing.expectError(
        error.UnsupportedParamType,
        layoutParams(std.testing.allocator, &func, nvidia_test_abi, false),
    );
}

test "a shared memory request above the target maximum is rejected" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    _ = try func.appendBlock();
    try attrs.setSharedBytes(&func, 64 * 1024);

    try std.testing.expectError(
        error.SharedMemoryTooLarge,
        layoutParams(std.testing.allocator, &func, nvidia_test_abi, false),
    );
}

test "an f64 parameter aligns to eight" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f64_t = try func.types.intern(.{ .float = .f64 });
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, i32_t);
    _ = try func.appendBlockParam(b, f64_t);

    var layout = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, false);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 8), layout.params[1].offset);
    try std.testing.expectEqual(@as(u32, 8), layout.params[1].size);
}
