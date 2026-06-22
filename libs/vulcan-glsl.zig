//! GLSL frontend: parse OpenGL Shading Language source and lower it to Vulcan IR, or
//! straight to SPIR-V. Covers scalar functions, vectors, control flow, and the
//! shader-stage model (in/out/uniform/builtins). Depends only on the IR.

const std = @import("std");
const spirv = @import("vulcan-spirv");

const lower = @import("vulcan-glsl/lower.zig");

pub const Error = lower.Error;
pub const Module = lower.Module;
pub const LoweredFunction = lower.LoweredFunction;
pub const SpirvError = lower.Error || spirv.emit.Error || error{NoFunction};
/// Shader pipeline stage for `compileShaderToSpirv` (re-exported so callers need only
/// import vulcan-glsl).
pub const Stage = spirv.Stage;

/// Compile GLSL source to a module of lowered IR functions. Caller owns it.
pub const compile = lower.compile;

/// Compile GLSL source to a SPIR-V binary (words): parse, lower to Vulcan IR, then emit
/// SPIR-V for the first function. Caller owns the result.
pub fn compileToSpirv(allocator: std.mem.Allocator, source: []const u8) SpirvError![]u32 {
    var module = try compile(allocator, source);
    defer module.deinit(allocator);
    if (module.functions.len == 0) return error.NoFunction;
    return spirv.emitModule(allocator, &module.functions[0].func, module.functions[0].name);
}

/// Compile a GLSL shader (with `in`/`out` globals and a `void main`) to a SPIR-V
/// entry-point shader for `stage`. `in` globals become input variables, the single
/// `out` global the output variable. Caller owns the result.
pub fn compileShaderToSpirv(allocator: std.mem.Allocator, source: []const u8, stage: spirv.Stage) SpirvError![]u32 {
    var shader = try lower.compileShader(allocator, source);
    defer shader.deinit();

    const input_vars = try allocator.alloc(spirv.InterfaceVar, shader.interface.inputs.len);
    defer allocator.free(input_vars);
    for (shader.interface.inputs, 0..) |iv, i| input_vars[i] = .{ .location = iv.location, .components = iv.components, .builtin = iv.builtin };

    const info = spirv.ShaderInfo{
        .stage = stage,
        .inputs = input_vars,
        .output = if (shader.interface.output) |o| spirv.OutputVar{ .location = o.location, .components = o.comps, .builtin = o.builtin } else null,
        .local_size = shader.interface.local_size,
        .uniform_count = shader.interface.uniform_count,
        .sampler_count = shader.interface.sampler_count,
    };
    return spirv.emitShader(allocator, &shader.func, info);
}

test {
    std.testing.refAllDecls(@import("vulcan-glsl/lexer.zig"));
    std.testing.refAllDecls(@import("vulcan-glsl/parser.zig"));
    std.testing.refAllDecls(lower);
}
