//! Bidirectional SPIR-V layer. Lowers a SPIR-V binary to Vulcan IR (frontend) and emits
//! SPIR-V from a Vulcan IR function (backend). SPIR-V is SSA with basic blocks and phi
//! nodes, mapping onto Vulcan IR's block-parameter SSA. Freestanding, depends only on IR.

const std = @import("std");

pub const binary = @import("vulcan-spirv/binary.zig");
pub const opcodes = @import("vulcan-spirv/opcodes.zig");
pub const lower = @import("vulcan-spirv/lower.zig");
pub const emit = @import("vulcan-spirv/emit.zig");

/// Lower the first function of a SPIR-V module (words) to a Vulcan IR function. Caller
/// owns and must `deinit` the result.
pub const lowerModule = lower.lowerModule;

/// Emit a Vulcan IR function as a standalone SPIR-V module (words). Reverse of
/// `lowerModule`. Caller owns the result.
pub const emitModule = emit.emitModule;

/// Emit a Vulcan IR function as a SPIR-V shader entry point (OpEntryPoint + in/out
/// variables) described by a `ShaderInfo`. Caller owns the result.
pub const emitShader = emit.emitShader;
pub const ShaderInfo = emit.ShaderInfo;
pub const Stage = emit.Stage;
pub const InterfaceVar = emit.InterfaceVar;
pub const OutputVar = emit.OutputVar;

test {
    std.testing.refAllDecls(@This());
}
