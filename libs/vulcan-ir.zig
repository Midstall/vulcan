//! Target-independent intermediate representation shared by every later Vulcan
//! component. Freestanding: no libc, no OS, no globals. Allocation flows through
//! a caller-supplied allocator.

const std = @import("std");

pub const entity = @import("vulcan-ir/entity.zig");
pub const types = @import("vulcan-ir/types.zig");
pub const function = @import("vulcan-ir/function.zig");
pub const builder = @import("vulcan-ir/builder.zig");
pub const parser = @import("vulcan-ir/parser.zig");
pub const verify = @import("vulcan-ir/verify.zig");
pub const legalize = @import("vulcan-ir/legalize.zig");
pub const critical_edge = @import("vulcan-ir/critical_edge.zig");
pub const expand = @import("vulcan-ir/expand.zig");
pub const bitcode = @import("vulcan-ir/bitcode.zig");

test {
    std.testing.refAllDecls(@This());
}
