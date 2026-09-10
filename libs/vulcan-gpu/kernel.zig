//! The kernel launch contract. A runtime reads these types to build a launch descriptor, and
//! it never needs the instruction selector to do so.

const std = @import("std");
const ir = @import("vulcan-ir");

/// Where a pointer points. A backend maps this to its own memory spaces: on NVIDIA `global`
/// is LDG and `shared` is LDS, on ET-SoC `shared` is the shire scratchpad, and on the CPU
/// offload path `shared` is a stack buffer.
///
/// The IR owns this enum, because the IR pointer type carries it and `vulcan-ir` cannot depend
/// on this module. It is re-exported here so a runtime that reads a `LaunchInfo` does not need
/// to import the IR.
pub const AddressSpace = ir.types.AddressSpace;

/// What a parameter carries.
pub const ParamKind = union(enum) {
    /// A scalar of this width in BYTES.
    scalar: u8,
    /// An address in this space.
    pointer: AddressSpace,
};

/// One parameter inside the parameter block. `offset` and `size` are BYTES, and `offset` is
/// measured from the start of the block, NOT from the start of the target's constant memory.
/// A backend adds `Abi.param_base` when it emits the load.
pub const Param = struct {
    offset: u32,
    size: u32,
    kind: ParamKind,
};

/// The number of axes an index space has. Index 0 is x, index 1 is y and index 2 is z.
pub const axes: usize = 3;

/// The launch-shape region of the parameter block: where a runtime writes the GRID size.
///
/// A builtin such as `grid_dim_x` has no value until the launch chooses one, so no hardware
/// register holds it and no attribute of the kernel declares it. The region gives that value
/// a place in the parameter contract. It holds the grid size in workgroups as three
/// consecutive 32-bit unsigned integers, x first, then y, then z.
///
/// The region sits AFTER every explicit parameter and after `Layout.out_pointer`. That
/// placement is what lets `layoutParams` reserve it only when the kernel needs it: a kernel
/// that gains a grid builtin grows the block at the END, so every other parameter keeps the
/// offset it already had. A region in FRONT would move all of them.
pub const LaunchShape = struct {
    /// The byte offset of the x extent from the start of the block. The y extent follows at
    /// `offset + axis_bytes` and the z extent at `offset + 2 * axis_bytes`.
    offset: u32,

    /// The width of one grid extent in bytes.
    pub const axis_bytes: u32 = 4;
    /// The size of the whole region in bytes.
    pub const bytes: u32 = @as(u32, axes) * axis_bytes;

    /// The byte offset of the extent for `axis`, measured from the start of the block.
    pub fn axisOffset(self: LaunchShape, axis: u2) u32 {
        std.debug.assert(axis < axes);
        return self.offset + @as(u32, axis) * axis_bytes;
    }
};

/// Flatten a three-dimensional index into ONE linear index, x fastest and z slowest:
///
///     linear = (idx[2] * extent[1] + idx[1]) * extent[0] + idx[0]
///
/// This is the launch contract's linearization rule, and every target obeys it. It is the
/// order `offload.lowerToLoopNest` visits the grid in, because that nest puts the x loop
/// innermost, and it is the order GPU hardware numbers the threads of a workgroup in.
///
/// A target whose hardware gives each axis its own register never needs the rule. A LINEAR-ID
/// target does: an ET-SoC-1 hart reads one `hartid` CSR, so the backend recovers the three
/// axes with `axisIndex`, and it can only do so against an agreed rule.
pub fn linearIndex(idx: [axes]u32, extent: [axes]u32) u32 {
    return (idx[2] * extent[1] + idx[1]) * extent[0] + idx[0];
}

/// The inverse of `linearIndex`: recover the three axes from one linear index.
///
///     idx[0] = linear % extent[0]
///     idx[1] = (linear / extent[0]) % extent[1]
///     idx[2] = linear / (extent[0] * extent[1])
///
/// `linear` must be less than the product of the extents, which the launch contract
/// guarantees: a runtime starts exactly that many threads in a workgroup and exactly that
/// many workgroups in a grid.
pub fn axisIndex(linear: u32, extent: [axes]u32) [axes]u32 {
    std.debug.assert(extent[0] != 0 and extent[1] != 0 and extent[2] != 0);
    std.debug.assert(linear < extent[0] * extent[1] * extent[2]);
    return .{
        linear % extent[0],
        (linear / extent[0]) % extent[1],
        linear / (extent[0] * extent[1]),
    };
}

/// The placed parameter block. The caller OWNS `params` and must release it with `deinit`.
pub const Layout = struct {
    /// The explicit parameters, in declaration order. Builtin parameters are absent: the
    /// hardware supplies them, so they occupy no space in the block.
    params: []Param,
    /// The total size of the block in bytes, including `out_pointer` and `launch_shape` when
    /// they are present.
    bytes: u32,
    /// The implicit output pointer a value-returning kernel writes its result through. It is
    /// placed FIRST in the block, before every explicit parameter. Null for a void kernel.
    out_pointer: ?Param,
    /// The launch-shape region, placed LAST in the block. Null when the kernel reads no
    /// builtin the grid shape decides, in which case the kernel pays nothing for it.
    launch_shape: ?LaunchShape,

    /// Release the parameter slice. `allocator` must be the one that placed the layout.
    pub fn deinit(self: *Layout, allocator: std.mem.Allocator) void {
        allocator.free(self.params);
    }
};

/// Everything a runtime needs to launch a kernel. No target types appear here.
pub const LaunchInfo = struct {
    /// The explicit parameters, in declaration order. Borrowed from the `Kernel` that owns it.
    params: []const Param,
    /// The size of the parameter block in bytes. The runtime binds a buffer at least this
    /// large, and the backend's loads read at `Abi.param_base` plus each `Param.offset`.
    param_bytes: u32,
    /// Where the runtime writes the GRID size in workgroups, as three consecutive 32-bit
    /// unsigned integers inside the parameter block. Null when the kernel reads no builtin
    /// the grid shape decides, in which case the runtime writes nothing and the block holds
    /// the parameters alone. See `LaunchShape`.
    launch_shape: ?LaunchShape,
    /// The declared workgroup size.
    block: [3]u32,
    /// The workgroup shared memory the kernel needs, in bytes.
    shared_bytes: u32,
    /// Registers per thread. The backend writes this. It is meaningful ONLY for a target whose
    /// launch descriptor declares a register budget, such as the NVIDIA QMD. A target with no
    /// such budget, such as ET-SoC or the CPU offload path, writes 0 and its runtime ignores
    /// the field.
    reg_count: u32,
    /// How many hardware control barriers the kernel needs. The runtime writes this into the
    /// launch descriptor. It is meaningful ONLY for a target whose descriptor declares one,
    /// such as the NVIDIA QMD `BARRIER_COUNT` field: a dispatch that leaves BARRIER_COUNT at 0
    /// while the kernel runs a BAR.SYNC is UNDEFINED. Mesa NAK sets `num_control_barriers = 1`
    /// beside its `OpBar` for the same reason. A backend that emits no barrier writes 0, and a
    /// target with no such field writes 0 and its runtime ignores it.
    barrier_count: u32,
};

test "a scalar param kind reports its width in bytes" {
    const k: ParamKind = .{ .scalar = 4 };
    try std.testing.expectEqual(@as(u8, 4), k.scalar);
}

test "a pointer param kind reports its address space" {
    const k: ParamKind = .{ .pointer = .shared };
    try std.testing.expectEqual(AddressSpace.shared, k.pointer);
}

test "deinit releases the parameter slice" {
    const allocator = std.testing.allocator;
    const params = try allocator.alloc(Param, 2);
    params[0] = .{ .offset = 0, .size = 8, .kind = .{ .pointer = .global } };
    params[1] = .{ .offset = 8, .size = 4, .kind = .{ .scalar = 4 } };
    var layout: Layout = .{ .params = params, .bytes = 12, .out_pointer = null, .launch_shape = null };
    layout.deinit(allocator);
    // The testing allocator fails the test at deinit if this leaked.
}

test "the launch-shape region reports one offset per axis, four bytes apart" {
    const s: LaunchShape = .{ .offset = 24 };
    try std.testing.expectEqual(@as(u32, 24), s.axisOffset(0));
    try std.testing.expectEqual(@as(u32, 28), s.axisOffset(1));
    try std.testing.expectEqual(@as(u32, 32), s.axisOffset(2));
    // The region ends where the last axis ends, so a runtime writes exactly `bytes` bytes.
    try std.testing.expectEqual(s.offset + LaunchShape.bytes, s.axisOffset(2) + LaunchShape.axis_bytes);
}

test "linearIndex runs x fastest and z slowest" {
    const extent = [3]u32{ 4, 3, 2 };
    try std.testing.expectEqual(@as(u32, 0), linearIndex(.{ 0, 0, 0 }, extent));
    // One step on x moves one slot, one step on y moves four, one step on z moves twelve.
    try std.testing.expectEqual(@as(u32, 1), linearIndex(.{ 1, 0, 0 }, extent));
    try std.testing.expectEqual(@as(u32, 4), linearIndex(.{ 0, 1, 0 }, extent));
    try std.testing.expectEqual(@as(u32, 12), linearIndex(.{ 0, 0, 1 }, extent));
    try std.testing.expectEqual(@as(u32, 23), linearIndex(.{ 3, 2, 1 }, extent));
}

test "axisIndex inverts linearIndex over a whole index space" {
    // Suspicious case: the two must agree on EVERY point, not on the corners, because a
    // linear-id backend decomposes with one and the host nest builds with the other.
    const extent = [3]u32{ 4, 3, 2 };
    var z: u32 = 0;
    while (z < extent[2]) : (z += 1) {
        var y: u32 = 0;
        while (y < extent[1]) : (y += 1) {
            var x: u32 = 0;
            while (x < extent[0]) : (x += 1) {
                const linear = linearIndex(.{ x, y, z }, extent);
                try std.testing.expectEqual([3]u32{ x, y, z }, axisIndex(linear, extent));
            }
        }
    }
}

test "a one-dimensional index space leaves the linear index unchanged" {
    // Suspicious case: the degenerate shape every existing kernel launches with. The x axis
    // must come back as the linear index itself, or a flat launch would change meaning.
    const extent = [3]u32{ 64, 1, 1 };
    var i: u32 = 0;
    while (i < 64) : (i += 1) {
        try std.testing.expectEqual([3]u32{ i, 0, 0 }, axisIndex(i, extent));
        try std.testing.expectEqual(i, linearIndex(.{ i, 0, 0 }, extent));
    }
}
