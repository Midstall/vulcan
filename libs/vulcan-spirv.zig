//! Bidirectional SPIR-V layer. Lowers a SPIR-V binary to Vulcan IR (frontend) and emits
//! SPIR-V from a Vulcan IR function (backend). SPIR-V is SSA with basic blocks and phi
//! nodes, mapping onto Vulcan IR's block-parameter SSA. Freestanding, depends only on IR.

const std = @import("std");

pub const binary = @import("vulcan-spirv/binary.zig");
pub const opcodes = @import("vulcan-spirv/opcodes.zig");
pub const lower = @import("vulcan-spirv/lower.zig");
pub const emit = @import("vulcan-spirv/emit.zig");
pub const inline_calls = @import("vulcan-spirv/inline_calls.zig");
pub const widen = @import("vulcan-spirv/widen.zig");
pub const disasm = @import("vulcan-spirv/disasm.zig");

/// Disassemble a SPIR-V binary (words or bytes) into canonical spirv-dis-style text. A
/// reading aid for `.spv` files. Caller owns the result.
pub const disassemble = disasm.format;
pub const disassembleBytes = disasm.formatBytes;

/// Lower the first function of a SPIR-V module (words) to a Vulcan IR function. Caller
/// owns and must `deinit` the result.
pub const lowerModule = lower.lowerModule;

/// Inline every `OpFunctionCall` in a multi-function SPIR-V module into its entry function,
/// returning a single-function word stream with no calls. Vulkan shaders are non-recursive,
/// so this always terminates. Caller owns and must free the result.
pub const inlineCalls = inline_calls.inlineCalls;

/// Widen a straight-line, buffer-free fragment IR function to 4-wide SIMD (a 2x2 quad):
/// every f32 value becomes a `<4 x f32>` of the quad's 4 fragments, executed lane-wise by
/// the NEON vector backend. Returns error.NotWidenable for shaders outside the vectorizable
/// subset (the caller keeps the scalar per-fragment path).
pub const widenGraphics = widen.widenGraphics;

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
