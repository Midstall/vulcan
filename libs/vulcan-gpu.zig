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
pub const kernel = @import("vulcan-gpu/kernel.zig");
pub const abi = @import("vulcan-gpu/abi.zig");
pub const offload = @import("vulcan-gpu/offload.zig");

/// A value the hardware provides rather than the parameter block. See `builtin.Builtin`.
pub const Builtin = builtin.Builtin;
/// The per-target parameter delivery rules. See `abi.Abi`.
pub const Abi = abi.Abi;
/// The launch metadata a runtime reads. See `kernel.LaunchInfo`.
pub const LaunchInfo = kernel.LaunchInfo;
/// A placed parameter. See `kernel.Param`.
pub const Param = kernel.Param;
/// Where a pointer points. See `kernel.AddressSpace`.
pub const AddressSpace = kernel.AddressSpace;
/// Place a kernel's parameter block. See `abi.layoutParams`.
pub const layoutParams = abi.layoutParams;
/// Rewrite a kernel into a host-runnable loop nest. See `offload.lowerToLoopNest`.
pub const lowerToLoopNest = offload.lowerToLoopNest;

test {
    std.testing.refAllDecls(@This());
}
