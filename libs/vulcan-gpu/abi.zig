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
    /// Whether the hardware gives a thread ONE linear identifier instead of one per axis.
    ///
    /// NVIDIA sets false. `SR_TID_X/Y/Z` and `SR_CTAID_X/Y/Z` deliver each axis by itself, so
    /// only `grid_dim_*` has to come out of the launch-shape region.
    ///
    /// ET-SoC-1 sets true. A hart reads one `hartid` CSR, so the backend recovers the three
    /// axes from it with `kernel.axisIndex`, and that needs the extents. The workgroup extents
    /// are compile-time, but the GRID extents are not, so every WORKGROUP builtin on such a
    /// target needs the region as well as `grid_dim_*`.
    linear_thread_id: bool,
};

/// Whether `func` reads a builtin that the launch-shape region has to supply under `a`.
///
/// This is the whole presence rule for the region, and it is a function of the IR and the
/// target ABI alone, so a runtime reproduces it by reading `LaunchInfo.launch_shape` and
/// never has to guess.
fn needsLaunchShape(func: *const Function, a: Abi) bool {
    for (func.blockParams(@enumFromInt(0))) |p| {
        const b = attrs.builtinOf(func, p) orelse continue;
        switch (b) {
            // The grid size comes from the launch on every target. No hardware holds it.
            .grid_dim_x, .grid_dim_y, .grid_dim_z => return true,
            // A workgroup index is one linear number on a linear-id target, and splitting it
            // into three axes needs the grid extents. On a per-axis target the hardware
            // already holds the answer, so the region stays absent.
            .block_id_x,
            .block_id_y,
            .block_id_z,
            .global_id_x,
            .global_id_y,
            .global_id_z,
            => if (a.linear_thread_id) return true,
            // A thread index splits with the DECLARED workgroup extents, which
            // `attrs.localSize` already holds, so it never needs the region.
            .thread_id_x,
            .thread_id_y,
            .thread_id_z,
            .block_dim_x,
            .block_dim_y,
            .block_dim_z,
            .lane_id,
            .warp_id,
            .subgroup_size,
            => {},
            // A graphics builtin never reaches a parameter block. The backend refuses it.
            .vertex_index,
            .instance_index,
            .frag_coord,
            .point_coord,
            .front_facing,
            => {},
        }
    }
    return false;
}

/// Whether any edge branches back to the entry block.
///
/// The parameter contract is POSITIONAL: `layoutParams` walks the entry block's parameters to
/// decide both the slot of each one and whether the launch-shape region is present, and a
/// backend then walks the same list in the same order to emit the loads. That only holds while
/// the list stays fixed, and `mem2reg` breaks it when the entry has a predecessor: it appends
/// a phi parameter to any block with predecessors (`readEntry`), and `collapseTrivialPhis`
/// deletes one from any block whose phi is trivial. Either would move a slot the runtime has
/// already been told about.
///
/// The entry of a kernel has no predecessor in practice, so the hazard is latent. This makes
/// the invariant explicit rather than accidental: a caller that finds it true must refuse.
/// `offload.lowerToLoopNest` refuses the same shape for a different reason of its own.
pub fn entryIsBranchTarget(func: *const Function) bool {
    const entry: ir.function.Block = @enumFromInt(0);
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => |cf| {
                    if (cf.then.target == entry) return true;
                    if (cf.@"else".target == entry) return true;
                },
                else => {},
            }
        }
        const term = func.terminator(block) orelse continue;
        switch (term) {
            .jump => |j| if (j.target == entry) return true,
            .ret => {},
        }
    }
    return false;
}

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
/// The launch-shape region goes LAST, after every explicit parameter, and only when
/// `needsLaunchShape` says the kernel reads a builtin the grid decides. Placing it last is
/// what makes the conditional presence free: adding a grid builtin to a kernel appends bytes
/// and moves no existing parameter, so a runtime that builds the block from the declaration
/// order keeps working.
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

        const param_kind = func.types.type_kind(func.valueType(p));
        const size: u32 = if (param_kind == .ptr) a.pointer_bytes else scalarBytes(func, p) orelse
            return error.UnsupportedParamType;
        const alignment = @max(@as(u32, a.param_align), size);

        cursor = alignUp(cursor, alignment);
        // The address space comes off the pointer type, so a shared parameter reports `shared`
        // and the runtime does not have to guess. A non-pointer type reaches the scalar arm
        // only after `scalarBytes` accepted it, so the composite arms are unreachable here.
        const kind: kernel.ParamKind = switch (param_kind) {
            .ptr => |space| .{ .pointer = space },
            .bool, .int, .float, .vector, .@"struct", .array, .slice => .{ .scalar = @intCast(size) },
        };
        try placed.append(allocator, .{ .offset = cursor, .size = size, .kind = kind });
        cursor += size;
    }

    // The launch-shape region. Each extent is a 32-bit unsigned integer, so the region aligns
    // to four, raised by the target's own floor the same way a parameter is.
    const launch_shape: ?kernel.LaunchShape = if (needsLaunchShape(func, a)) shape: {
        cursor = alignUp(cursor, @max(@as(u32, a.param_align), kernel.LaunchShape.axis_bytes));
        const at = cursor;
        cursor += kernel.LaunchShape.bytes;
        break :shape .{ .offset = at };
    } else null;

    return .{
        .params = try placed.toOwnedSlice(allocator),
        .bytes = cursor,
        .out_pointer = out_pointer,
        .launch_shape = launch_shape,
    };
}

const nvidia_test_abi: Abi = .{
    .param_base = 0x160,
    .pointer_bytes = 8,
    .param_align = 4,
    .max_shared_bytes = 48 * 1024,
    .linear_thread_id = false,
};

/// A linear-id target, where a thread reads one identifier and the backend splits it.
const linear_test_abi: Abi = .{
    .param_base = 0,
    .pointer_bytes = 8,
    .param_align = 1,
    .max_shared_bytes = 48 * 1024,
    .linear_thread_id = true,
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
    const ptr_t = try func.types.ptrGlobal();
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

test "a shared pointer parameter reports its address space in the layout" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const global_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, global_t);
    _ = try func.appendBlockParam(b, shared_t);

    var layout = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, false);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(kernel.AddressSpace.global, layout.params[0].kind.pointer);
    try std.testing.expectEqual(kernel.AddressSpace.shared, layout.params[1].kind.pointer);
}

test "a kernel that reads no grid builtin reserves no launch-shape region" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, t);

    var layout = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, false);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(?kernel.LaunchShape, null), layout.launch_shape);
    try std.testing.expectEqual(@as(u32, 4), layout.bytes);
}

test "grid_dim reserves the launch-shape region after the explicit parameters" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, ptr_t);
    const g = try func.appendBlockParam(b, i32_t);
    try attrs.setBuiltin(&func, g, .grid_dim_y);

    var layout = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, false);
    defer layout.deinit(std.testing.allocator);

    // The pointer keeps offset 0, exactly as it would with no grid builtin at all.
    try std.testing.expectEqual(@as(usize, 1), layout.params.len);
    try std.testing.expectEqual(@as(u32, 0), layout.params[0].offset);
    // The region follows it, and the block grows by the region.
    try std.testing.expectEqual(@as(u32, 8), layout.launch_shape.?.offset);
    try std.testing.expectEqual(@as(u32, 12), layout.launch_shape.?.axisOffset(1));
    try std.testing.expectEqual(@as(u32, 20), layout.bytes);
}

test "adding a grid builtin moves no existing parameter" {
    // The property the placement decision buys. Two kernels with the same explicit
    // parameters must report the same offsets for them, whether or not one reads the grid.
    const allocator = std.testing.allocator;
    var plain = try testFunc(allocator);
    defer plain.deinit();
    var with_grid = try testFunc(allocator);
    defer with_grid.deinit();

    for ([_]*Function{ &plain, &with_grid }, 0..) |func, i| {
        const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const ptr_t = try func.types.ptrGlobal();
        const b = try func.appendBlock();
        _ = try func.appendBlockParam(b, ptr_t);
        _ = try func.appendBlockParam(b, i32_t);
        _ = try func.appendBlockParam(b, ptr_t);
        if (i == 1) {
            const g = try func.appendBlockParam(b, i32_t);
            try attrs.setBuiltin(func, g, .grid_dim_x);
        }
    }

    var a_layout = try layoutParams(allocator, &plain, nvidia_test_abi, true);
    defer a_layout.deinit(allocator);
    var b_layout = try layoutParams(allocator, &with_grid, nvidia_test_abi, true);
    defer b_layout.deinit(allocator);

    try std.testing.expectEqual(a_layout.out_pointer.?.offset, b_layout.out_pointer.?.offset);
    try std.testing.expectEqual(a_layout.params.len, b_layout.params.len);
    for (a_layout.params, b_layout.params) |x, y| try std.testing.expectEqual(x.offset, y.offset);
    try std.testing.expectEqual(a_layout.bytes + kernel.LaunchShape.bytes, b_layout.bytes);
}

test "a workgroup builtin reserves the region only on a linear-id target" {
    // Suspicious case: the same IR, two ABIs, two answers. A per-axis target reads its own
    // CTAID register and pays nothing; a linear-id target has to split one identifier and
    // cannot do it without the grid extents.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const bid = try func.appendBlockParam(b, i32_t);
    try attrs.setBuiltin(&func, bid, .block_id_y);

    var per_axis = try layoutParams(std.testing.allocator, &func, nvidia_test_abi, false);
    defer per_axis.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?kernel.LaunchShape, null), per_axis.launch_shape);
    try std.testing.expectEqual(@as(u32, 0), per_axis.bytes);

    var linear = try layoutParams(std.testing.allocator, &func, linear_test_abi, false);
    defer linear.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 0), linear.launch_shape.?.offset);
    try std.testing.expectEqual(@as(u32, 12), linear.bytes);
}

test "a thread builtin never reserves the region, even on a linear-id target" {
    // The workgroup extents are DECLARED, so splitting a thread identifier needs no launch
    // data. A region here would be twelve bytes no runtime ever writes.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const tid = try func.appendBlockParam(b, i32_t);
    try attrs.setBuiltin(&func, tid, .thread_id_z);

    var layout = try layoutParams(std.testing.allocator, &func, linear_test_abi, false);
    defer layout.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?kernel.LaunchShape, null), layout.launch_shape);
}

test "the launch-shape region aligns to four after an odd parameter block" {
    // Suspicious case: a target with a one-byte alignment floor can leave the cursor odd, and
    // an unaligned 32-bit extent is a fault or a silent tear depending on the hardware.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, i8_t);
    const g = try func.appendBlockParam(b, i32_t);
    try attrs.setBuiltin(&func, g, .grid_dim_z);

    var layout = try layoutParams(std.testing.allocator, &func, linear_test_abi, false);
    defer layout.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 0), layout.params[0].offset);
    try std.testing.expectEqual(@as(u32, 4), layout.launch_shape.?.offset);
    try std.testing.expectEqual(@as(u32, 16), layout.bytes);
}

test "entryIsBranchTarget finds an edge back to the entry and clears a straight-line kernel" {
    // The guard exists because the parameter contract is positional and `mem2reg` edits the
    // parameter list of any block a branch reaches. The negative control is the same shape
    // with the edge pointing forward instead of back.
    const allocator = std.testing.allocator;
    {
        var func = try testFunc(allocator);
        defer func.deinit();
        const b0 = try func.appendBlock();
        try func.setJump(b0, b0, &.{});
        try std.testing.expect(entryIsBranchTarget(&func));
    }
    {
        var func = try testFunc(allocator);
        defer func.deinit();
        const b0 = try func.appendBlock();
        const b1 = try func.appendBlock();
        try func.setJump(b0, b1, &.{});
        func.setTerminator(b1, .{ .ret = ir.function.Ret.none() });
        try std.testing.expect(!entryIsBranchTarget(&func));
    }
}
