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

pub const Error = std.mem.Allocator.Error || error{
    UnsupportedParamType,
    SharedMemoryTooLarge,
    /// The kernel declares `vulcan.gpu.shared_bytes`, it declares shared `alloca` slots, and
    /// the two numbers differ. See `layoutSharedFrame`.
    SharedFrameMismatch,
    /// The kernel declares BOTH a shared `alloca` and a `ptr(shared)` parameter. Both claim
    /// offset 0 of the same window. See `layoutSharedFrame`.
    SharedFrameAliasesParameter,
    /// A shared `alloca` holds a type the shared frame cannot place. See `sharedShape`.
    UnsupportedSharedType,
};

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

/// One workgroup-shared slot: the `alloca` that declares it, and where it sits.
pub const SharedSlot = struct {
    /// The `alloca` result. Its type is a `ptr(shared)`.
    value: Value,
    /// The byte offset of the slot from the start of the workgroup's shared window.
    offset: u32,
    /// The size of the slot in bytes.
    size: u32,
};

/// The placed workgroup-shared frame. The caller OWNS `slots` and must release it with
/// `deinit`.
pub const SharedFrame = struct {
    /// One entry per shared `alloca`, in the order the instructions appear.
    slots: []SharedSlot,
    /// How many bytes the slots occupy. Zero when the kernel declares no shared `alloca`.
    bytes: u32,
    /// The whole workgroup window the runtime must give the kernel, which is what
    /// `LaunchInfo.shared_bytes` reports. It is `bytes` for a kernel that declares slots, and
    /// the frontend's `attrs.sharedBytes` for one that declares none (the extern-shared model,
    /// where a `ptr(shared)` PARAMETER carries a window the host sized and placed).
    total: u32,

    /// Release the slot slice. `allocator` must be the one that placed the frame.
    pub fn deinit(self: *SharedFrame, allocator: std.mem.Allocator) void {
        allocator.free(self.slots);
    }

    /// The byte offset of the slot `value` declares, or null when `value` is not a shared
    /// `alloca`. A frame holds a handful of slots, so a scan is the right lookup.
    pub fn offsetOf(self: *const SharedFrame, value: Value) ?u32 {
        for (self.slots) |s| {
            if (s.value == value) return s.offset;
        }
        return null;
    }
};

/// The size and the alignment of one shared slot in bytes. See `sharedShape`.
const SharedShape = struct { size: u32, alignment: u32 };

/// The shape of one shared slot, or null when the frame cannot place the type.
///
/// A POINTER element is refused. The width of an address in shared memory is a target fact
/// this descriptor does not carry: a `global` address is `pointer_bytes` wide, while a
/// `shared` address is a window offset that is narrower on the hardware that has one. A
/// shared tile OF ADDRESSES has no caller today, so the frame refuses it instead of guessing
/// a width and writing the wrong number of bytes.
///
/// An integer of a width that is not 8, 16, 32 or 64 bits is refused for the same reason: it
/// has no natural alignment, and rounding one up would place the next slot where the frontend
/// did not put it.
fn sharedShape(func: *const Function, ty: ir.types.Type) ?SharedShape {
    return switch (func.types.type_kind(ty)) {
        .bool => .{ .size = 1, .alignment = 1 },
        .int => |i| switch (i.bits) {
            8 => .{ .size = 1, .alignment = 1 },
            16 => .{ .size = 2, .alignment = 2 },
            32 => .{ .size = 4, .alignment = 4 },
            64 => .{ .size = 8, .alignment = 8 },
            else => null,
        },
        .float => |f| switch (f) {
            .f16 => .{ .size = 2, .alignment = 2 },
            .f32 => .{ .size = 4, .alignment = 4 },
            .f64 => .{ .size = 8, .alignment = 8 },
            .f128 => .{ .size = 16, .alignment = 16 },
        },
        // An array or a vector is `len` elements laid end to end, and it aligns like ONE
        // element. The element shape recurses, so an array of arrays places correctly.
        .array => |arr| sharedRun(func, arr.elem, arr.len),
        .vector => |v| sharedRun(func, v.elem, v.len),
        .ptr, .@"struct", .slice => null,
    };
}

/// `count` elements of `elem` laid end to end. Null when the element has no shared shape or
/// the run does not fit in 32 bits.
fn sharedRun(func: *const Function, elem: ir.types.Type, count: u64) ?SharedShape {
    const e = sharedShape(func, elem) orelse return null;
    const n = std.math.cast(u32, count) orelse return null;
    const size = std.math.mul(u32, e.size, n) catch return null;
    return .{ .size = size, .alignment = e.alignment };
}

/// Whether the entry block takes a `ptr(shared)` parameter, which is the extern-shared model.
fn hasSharedParam(func: *const Function) bool {
    for (func.blockParams(@enumFromInt(0))) |p| {
        switch (func.types.type_kind(func.valueType(p))) {
            .ptr => |space| switch (space) {
                .shared => return true,
                .global, .constant, .private => {},
            },
            .bool, .int, .float, .vector, .array, .slice, .@"struct" => {},
        }
    }
    return false;
}

/// Place every workgroup-shared `alloca` of `func` into the workgroup's shared window.
///
/// ## Who owns the frame
///
/// THIS PLACEMENT OWNS IT, and `attrs.sharedBytes` is the frontend's DECLARATION of the total.
/// The two meet under one rule, stated below, and any disagreement is an error.
///
/// The offset of a slot is a code-generation fact: a backend materializes it as an immediate
/// and feeds it straight to a shared load or store, so it depends on the target's alignment
/// rules and on how wide an address in that window is. A frontend that wrote per-slot offsets
/// into the IR would be committing portable IR to one target's layout, and every other backend
/// would then have to honour a layout that does not fit its hardware or ignore attributes that
/// look authoritative.
///
/// `attrs.sharedBytes` cannot become a derived number, because the EXTERN-SHARED model has no
/// `alloca` at all: the kernel takes a `ptr(shared)` parameter and the host writes an offset
/// into it, so only the frontend knows how large that window is. The attribute therefore keeps
/// its meaning, "the whole workgroup window this kernel needs".
///
/// ## The rule
///
///   - No shared `alloca`: the frame is empty and `total` is the declared `attrs.sharedBytes`,
///     exactly as before this placement existed.
///   - Shared `alloca`s and NO declaration (`shared_bytes` absent, which reads as 0): the frame
///     is the whole window and `total` is the frame.
///   - Shared `alloca`s AND a declaration: the two must be equal, or `SharedFrameMismatch`. The
///     declaration is a checksum, never a budget. A budget would let a frontend over-declare
///     and lose occupancy with nothing to show for it.
///   - Shared `alloca`s AND a `ptr(shared)` parameter: `SharedFrameAliasesParameter`. Both
///     claim offset 0 of one window, and the IR carries no way to say that they do not overlap.
///     CUDA places its `extern __shared__` region after the static ones; expressing that needs
///     the parameter to carry an offset, which is a separate decision.
///
/// The frame is also checked against `Abi.max_shared_bytes`, exactly as the declared total is,
/// so a frame that overflows the hardware window is refused here and not at dispatch.
///
/// The caller OWNS the result and must release it with `SharedFrame.deinit`.
pub fn layoutSharedFrame(allocator: std.mem.Allocator, func: *const Function, a: Abi) Error!SharedFrame {
    var placed: std.ArrayList(SharedSlot) = .empty;
    errdefer placed.deinit(allocator);

    // A 64-bit cursor cannot overflow, so the window check below is the only bound and it is
    // the one the hardware actually imposes.
    var cursor: u64 = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            const al = switch (func.opcode(inst)) {
                .alloca => |al| al,
                .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .select => continue,
                .struct_new, .extract, .convert, .unary, .global_addr => continue,
                .call, .call_indirect, .load, .store, .prefetch, .@"if" => continue,
                .va_start, .va_arg, .va_end, .dot, .matmul, .barrier, .atomic_rmw => continue,
            };
            const result = func.instResult(inst) orelse continue;
            switch (func.types.type_kind(func.valueType(result))) {
                .ptr => |space| switch (space) {
                    .shared => {},
                    // A private, global or constant `alloca` is an ordinary storage slot and
                    // no business of the workgroup window.
                    .global, .constant, .private => continue,
                },
                .bool, .int, .float, .vector, .array, .slice, .@"struct" => continue,
            }
            const shape = sharedShape(func, al.elem) orelse return error.UnsupportedSharedType;
            const alignment: u64 = shape.alignment;
            const at = (cursor + alignment - 1) & ~(alignment - 1);
            cursor = at + shape.size;
            if (cursor > a.max_shared_bytes) return error.SharedMemoryTooLarge;
            try placed.append(allocator, .{ .value = result, .offset = @intCast(at), .size = shape.size });
        }
    }

    const declared = attrs.sharedBytes(func);
    if (placed.items.len == 0) {
        placed.deinit(allocator);
        return .{ .slots = &.{}, .bytes = 0, .total = declared };
    }
    if (hasSharedParam(func)) return error.SharedFrameAliasesParameter;
    const bytes: u32 = @intCast(cursor);
    if (declared != 0 and declared != bytes) return error.SharedFrameMismatch;
    return .{ .slots = try placed.toOwnedSlice(allocator), .bytes = bytes, .total = bytes };
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

/// A kernel that declares `slots` shared `alloca`s of the given element types, in order.
/// Returns the alloca results so a test can ask the frame where each one landed.
fn sharedKernel(func: *Function, elems: []const ir.types.Type, out: []Value) !void {
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    for (elems, 0..) |e, i| {
        out[i] = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = e } });
    }
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
}

test "a kernel with no shared alloca reports the declared total and an empty frame" {
    // The extern-shared model, which worked before this placement existed and must keep
    // working: the window is a `ptr(shared)` PARAMETER the host sizes and places.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, shared_t);
    try attrs.setSharedBytes(&func, 1024);

    var frame = try layoutSharedFrame(std.testing.allocator, &func, nvidia_test_abi);
    defer frame.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), frame.slots.len);
    try std.testing.expectEqual(@as(u32, 0), frame.bytes);
    try std.testing.expectEqual(@as(u32, 1024), frame.total);
}

test "two shared allocas get distinct, non-overlapping offsets and the frame is their sum" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const tile_t = try func.types.intern(.{ .array = .{ .len = 32, .elem = f32_t } });
    var vals: [2]Value = undefined;
    try sharedKernel(&func, &.{ tile_t, tile_t }, &vals);

    var frame = try layoutSharedFrame(std.testing.allocator, &func, nvidia_test_abi);
    defer frame.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), frame.slots.len);
    try std.testing.expectEqual(@as(?u32, 0), frame.offsetOf(vals[0]));
    try std.testing.expectEqual(@as(?u32, 128), frame.offsetOf(vals[1]));
    // The second slot starts where the first one ends, so the two never share a byte.
    try std.testing.expectEqual(frame.slots[0].offset + frame.slots[0].size, frame.slots[1].offset);
    try std.testing.expectEqual(@as(u32, 256), frame.bytes);
    try std.testing.expectEqual(@as(u32, 256), frame.total);
}

test "a shared slot aligns to its element, so a byte tile does not misalign the tile after it" {
    // Suspicious case: without the alignment step the f32 tile would start at offset 3 and
    // every LDS of it would split a word. The padding is what makes the total 4 + 128 and not
    // 3 + 128.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const three_t = try func.types.intern(.{ .array = .{ .len = 3, .elem = i8_t } });
    const tile_t = try func.types.intern(.{ .array = .{ .len = 32, .elem = f32_t } });
    var vals: [2]Value = undefined;
    try sharedKernel(&func, &.{ three_t, tile_t }, &vals);

    var frame = try layoutSharedFrame(std.testing.allocator, &func, nvidia_test_abi);
    defer frame.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?u32, 0), frame.offsetOf(vals[0]));
    try std.testing.expectEqual(@as(?u32, 4), frame.offsetOf(vals[1]));
    try std.testing.expectEqual(@as(u32, 132), frame.bytes);
}

test "a declared shared total that AGREES with the frame is accepted" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const tile_t = try func.types.intern(.{ .array = .{ .len = 16, .elem = f32_t } });
    var vals: [1]Value = undefined;
    try sharedKernel(&func, &.{tile_t}, &vals);
    try attrs.setSharedBytes(&func, 64);

    var frame = try layoutSharedFrame(std.testing.allocator, &func, nvidia_test_abi);
    defer frame.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 64), frame.total);
}

test "a declared shared total that DISAGREES with the frame is refused" {
    // The whole point of keeping the declaration: a frontend that thinks the tile is 64 bytes
    // while the frame places 128 must hear about it, never get a window sized from one of the
    // two numbers with no diagnostic.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const tile_t = try func.types.intern(.{ .array = .{ .len = 32, .elem = f32_t } });
    var vals: [1]Value = undefined;
    try sharedKernel(&func, &.{tile_t}, &vals);
    try attrs.setSharedBytes(&func, 64);

    try std.testing.expectError(
        error.SharedFrameMismatch,
        layoutSharedFrame(std.testing.allocator, &func, nvidia_test_abi),
    );
}

test "a shared alloca beside a shared PARAMETER is refused, because both claim offset zero" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    _ = try func.appendBlockParam(b, shared_t);
    _ = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = f32_t } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(
        error.SharedFrameAliasesParameter,
        layoutSharedFrame(std.testing.allocator, &func, nvidia_test_abi),
    );
}

test "a shared frame larger than the target window is refused" {
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const big_t = try func.types.intern(.{ .array = .{ .len = 64 * 1024, .elem = f32_t } });
    var vals: [1]Value = undefined;
    try sharedKernel(&func, &.{big_t}, &vals);

    try std.testing.expectError(
        error.SharedMemoryTooLarge,
        layoutSharedFrame(std.testing.allocator, &func, nvidia_test_abi),
    );
}

test "a shared alloca of a type the frame cannot size is refused, not guessed at" {
    // A tile OF ADDRESSES. The width of an address in the shared window is a target fact this
    // descriptor does not carry, so the frame refuses rather than pick one.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    var vals: [1]Value = undefined;
    try sharedKernel(&func, &.{ptr_t}, &vals);

    try std.testing.expectError(
        error.UnsupportedSharedType,
        layoutSharedFrame(std.testing.allocator, &func, nvidia_test_abi),
    );
}

test "a PRIVATE alloca takes no room in the shared frame" {
    // The negative control for the address-space test in the placement loop. An ordinary
    // stack slot is no business of the workgroup window, and counting it would inflate the
    // window the runtime allocates for every workgroup.
    var func = try testFunc(std.testing.allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const private_t = try func.types.intern(.{ .ptr = .private });
    const global_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    _ = try func.appendInst(b, private_t, .{ .alloca = .{ .elem = f32_t } });
    _ = try func.appendInst(b, global_t, .{ .alloca = .{ .elem = f32_t } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var frame = try layoutSharedFrame(std.testing.allocator, &func, nvidia_test_abi);
    defer frame.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), frame.slots.len);
    try std.testing.expectEqual(@as(u32, 0), frame.total);
}
