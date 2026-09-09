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

/// The placed parameter block. The caller OWNS `params` and must release it with `deinit`.
pub const Layout = struct {
    /// The explicit parameters, in declaration order. Builtin parameters are absent: the
    /// hardware supplies them, so they occupy no space in the block.
    params: []Param,
    /// The total size of the block in bytes, including `out_pointer` when present.
    bytes: u32,
    /// The implicit output pointer a value-returning kernel writes its result through. It is
    /// placed FIRST in the block, before every explicit parameter. Null for a void kernel.
    out_pointer: ?Param,

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
    var layout: Layout = .{ .params = params, .bytes = 12, .out_pointer = null };
    layout.deinit(allocator);
    // The testing allocator fails the test at deinit if this leaked.
}
