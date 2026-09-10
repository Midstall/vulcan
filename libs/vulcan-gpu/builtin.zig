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

    /// Whether every thread of ONE WORKGROUP reads the same value from this builtin.
    ///
    /// The scope is the workgroup, because that is the scope a workgroup barrier synchronizes.
    /// A uniformity analysis asks this question to decide whether a branch can split the threads
    /// that a barrier waits for, and only the threads of one workgroup wait at that barrier.
    ///
    /// `block_id_*` is UNIFORM under this definition, although it changes from one workgroup to
    /// the next. Every thread of one workgroup reads the same workgroup index, so a branch on it
    /// sends all of them the same way. Two different workgroups can go different ways, but they
    /// never wait at the same workgroup barrier, so their disagreement cannot desynchronize one.
    /// A wider scope, for example a grid-wide barrier, must NOT use this answer.
    ///
    /// `subgroup_size` is a hardware constant, and `block_dim_*` and `grid_dim_*` come from the
    /// launch shape, so all three hold the same value in every thread of the grid.
    ///
    /// Every graphics builtin is per-invocation, so all of them are divergent.
    pub fn isWorkgroupUniform(self: Builtin) bool {
        return switch (self) {
            .thread_id_x, .thread_id_y, .thread_id_z => false,
            .global_id_x, .global_id_y, .global_id_z => false,
            .lane_id, .warp_id => false,
            .block_id_x, .block_id_y, .block_id_z => true,
            .block_dim_x, .block_dim_y, .block_dim_z => true,
            .grid_dim_x, .grid_dim_y, .grid_dim_z => true,
            .subgroup_size => true,
            .vertex_index, .instance_index, .frag_coord, .point_coord, .front_facing => false,
        };
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

test "isWorkgroupUniform splits the per-thread builtins from the per-workgroup ones" {
    try std.testing.expect(!Builtin.thread_id_x.isWorkgroupUniform());
    try std.testing.expect(!Builtin.thread_id_z.isWorkgroupUniform());
    try std.testing.expect(!Builtin.global_id_y.isWorkgroupUniform());
    try std.testing.expect(!Builtin.lane_id.isWorkgroupUniform());
    try std.testing.expect(!Builtin.warp_id.isWorkgroupUniform());
    try std.testing.expect(Builtin.block_dim_x.isWorkgroupUniform());
    try std.testing.expect(Builtin.grid_dim_z.isWorkgroupUniform());
    try std.testing.expect(Builtin.subgroup_size.isWorkgroupUniform());
}

test "block_id is uniform inside a workgroup, which is the scope of a workgroup barrier" {
    // Suspicious case: the workgroup index changes from one workgroup to the next, so it looks
    // divergent. It is not, at this scope. Every thread of one workgroup reads the same index,
    // and only the threads of one workgroup wait at a workgroup barrier.
    try std.testing.expect(Builtin.block_id_x.isWorkgroupUniform());
    try std.testing.expect(Builtin.block_id_y.isWorkgroupUniform());
    try std.testing.expect(Builtin.block_id_z.isWorkgroupUniform());
}

test "every graphics builtin is divergent" {
    try std.testing.expect(!Builtin.vertex_index.isWorkgroupUniform());
    try std.testing.expect(!Builtin.instance_index.isWorkgroupUniform());
    try std.testing.expect(!Builtin.frag_coord.isWorkgroupUniform());
    try std.testing.expect(!Builtin.point_coord.isWorkgroupUniform());
    try std.testing.expect(!Builtin.front_facing.isWorkgroupUniform());
}

test "the compute group boundary sits where isCompute expects it" {
    // Pins the numbering the isCompute comparison depends on. Renumbering the enum without
    // moving the boundary would silently reclassify a builtin.
    try std.testing.expectEqual(@as(u16, 17), @intFromEnum(Builtin.subgroup_size));
    try std.testing.expectEqual(@as(u16, 32), @intFromEnum(Builtin.vertex_index));
}
