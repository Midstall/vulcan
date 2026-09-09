//! Target-neutral accelerator kernel ABI. A kernel is an entry function with a parameter block
//! of a known layout, a declared launch shape, and a scratch memory requirement. This module
//! owns that vocabulary so a frontend, a backend, and a runtime all agree on it without any
//! of them depending on the others.
//!
//! Freestanding-clean. Depends only on the IR.
//!
//! The module has two tiers. `kernel` and `abi` are universal: a dataflow accelerator such as
//! Tenstorrent uses them. `builtin` is the SIMT vocabulary that NVIDIA, AMD, Intel and ET-SoC
//! share, and a non-SIMT target ignores it. Divergence handling belongs to each backend and
//! is deliberately absent here: NVIDIA reconverges warps, ET-SoC minions branch independently.

const std = @import("std");

pub const builtin = @import("vulcan-gpu/builtin.zig");
pub const attrs = @import("vulcan-gpu/attrs.zig");

/// A value the hardware provides rather than the parameter block. See `builtin.Builtin`.
pub const Builtin = builtin.Builtin;

test {
    std.testing.refAllDecls(@This());
}
