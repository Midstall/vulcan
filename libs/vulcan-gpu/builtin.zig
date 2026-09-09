//! Values the hardware gives a kernel, rather than values the parameter block carries.
//! A frontend tags an entry-block parameter with one of these, and the backend then sources
//! that parameter from a special register or an equivalent hardware path.
//!
//! The numbering is vulcan's own. It is deliberately NOT the SPIR-V BuiltIn numbering: the
//! IR builder is an equally valid front door, so the backend must not depend on one frontend's
//! constants. `vulcan-spirv` owns the map from SPIR-V numbers to these.
//!
//! The compute group is the SIMT vocabulary that NVIDIA, AMD, Intel and ET-SoC share. A
//! non-SIMT accelerator ignores it and uses only the parameter block. The graphics group is
//! the fixed-function pipeline's inputs.

const std = @import("std");

/// A hardware-provided kernel or shader input.
pub const Builtin = enum(u16) {
    /// The thread index inside its workgroup. CUDA threadIdx, SPIR-V LocalInvocationId.
    thread_id_x = 0,
    thread_id_y = 1,
    thread_id_z = 2,
    /// The workgroup index inside the grid. CUDA blockIdx, SPIR-V WorkgroupId.
    block_id_x = 3,
    block_id_y = 4,
    block_id_z = 5,
    /// The size of a workgroup in threads. CUDA blockDim, SPIR-V WorkgroupSize.
    block_dim_x = 6,
    block_dim_y = 7,
    block_dim_z = 8,
    /// The size of the grid in workgroups. CUDA gridDim, SPIR-V NumWorkgroups.
    grid_dim_x = 9,
    grid_dim_y = 10,
    grid_dim_z = 11,
    /// block_id * block_dim + thread_id, fused. SPIR-V GlobalInvocationId.
    global_id_x = 12,
    global_id_y = 13,
    global_id_z = 14,
    /// The lane index inside a subgroup. NVIDIA calls a subgroup a warp.
    lane_id = 15,
    /// The subgroup index inside a workgroup.
    warp_id = 16,
    /// The number of lanes in a subgroup. 32 on NVIDIA, 8 on the ET-SoC VPU.
    subgroup_size = 17,
    /// The vertex index of a draw. SPIR-V VertexIndex.
    vertex_index = 32,
    /// The instance index of a draw. SPIR-V InstanceIndex.
    instance_index = 33,
    /// The window-space position of a fragment. SPIR-V FragCoord.
    frag_coord = 34,
    /// The point-sprite coordinate of a fragment, 0 to 1. SPIR-V PointCoord.
    point_coord = 35,
    /// Whether a fragment belongs to a front-facing primitive. SPIR-V FrontFacing.
    front_facing = 36,

    /// Whether this builtin belongs to the compute group. A graphics builtin on a compute
    /// kernel is a frontend error, and a backend uses this to reject it.
    pub fn isCompute(self: Builtin) bool {
        return @intFromEnum(self) <= @intFromEnum(Builtin.subgroup_size);
    }

    /// The axis this builtin selects, or null when it has no axis. A backend uses this to
    /// index a three-element table instead of writing out a 15-arm switch.
    pub fn axis(self: Builtin) ?u2 {
        return switch (self) {
            .thread_id_x, .block_id_x, .block_dim_x, .grid_dim_x, .global_id_x => 0,
            .thread_id_y, .block_id_y, .block_dim_y, .grid_dim_y, .global_id_y => 1,
            .thread_id_z, .block_id_z, .block_dim_z, .grid_dim_z, .global_id_z => 2,
            .lane_id, .warp_id, .subgroup_size => null,
            .vertex_index, .instance_index, .frag_coord, .point_coord, .front_facing => null,
        };
    }
};

test "isCompute separates the compute group from the graphics group" {
    try std.testing.expect(Builtin.thread_id_x.isCompute());
    try std.testing.expect(Builtin.subgroup_size.isCompute());
    try std.testing.expect(!Builtin.vertex_index.isCompute());
    try std.testing.expect(!Builtin.front_facing.isCompute());
}

test "axis reports the component for the three-axis builtins and null otherwise" {
    try std.testing.expectEqual(@as(?u2, 0), Builtin.thread_id_x.axis());
    try std.testing.expectEqual(@as(?u2, 1), Builtin.block_id_y.axis());
    try std.testing.expectEqual(@as(?u2, 2), Builtin.global_id_z.axis());
    try std.testing.expectEqual(@as(?u2, null), Builtin.lane_id.axis());
    try std.testing.expectEqual(@as(?u2, null), Builtin.vertex_index.axis());
}

test "the compute group boundary sits where isCompute expects it" {
    // Pins the numbering the isCompute comparison depends on. Renumbering the enum without
    // moving the boundary would silently reclassify a builtin.
    try std.testing.expectEqual(@as(u16, 17), @intFromEnum(Builtin.subgroup_size));
    try std.testing.expectEqual(@as(u16, 32), @intFromEnum(Builtin.vertex_index));
}
