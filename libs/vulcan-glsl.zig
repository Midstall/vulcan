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

/// One default-uniform-block member (re-exported from the lowering): its name, float
/// offset, and float count. The GLES layer uses this to map glGetUniformLocation +
/// glUniform* writes to the right block offset.
pub const UniformMember = lower.UniformMember;

/// One `uniform sampler2D` member (re-exported from the lowering): its name and SPIR-V
/// binding. The GLES layer maps the sampler name -> binding so the texture unit set via
/// glUniform1i on that uniform binds to the HAL at the right binding.
pub const SamplerMember = lower.SamplerMember;
pub const UniformBlock = lower.UniformBlock;
pub const UniformBlockMember = lower.UniformBlockMember;

/// One vertex-attribute member: its name (heap-duped) and the pipeline location the lowering
/// assigned it (declaration order, or an explicit layout(location=)). The GLES layer resolves
/// glGetAttribLocation/glGetActiveAttrib + the draw-time vertex layout against this list (the
/// real attributes), instead of a name-substring heuristic.
pub const AttributeMember = struct {
    name: []const u8,
    location: u32,
    /// Component count (1 scalar, 2..4 vector) - reported by glGetActiveAttrib.
    components: u8,
};

/// One vertex-shader OUTPUT varying: its name (heap-duped) and the pipeline location the lowering
/// assigned it (declaration order, or an explicit layout(location=)), plus the component count.
/// The GLES layer uses this to map a glTransformFeedbackVaryings capture name to the VS output
/// slot (location*4 + component) the driver reads when capturing transform feedback. Builtins
/// (gl_Position/gl_PointSize) are NOT surfaced here (they have no varying name).
pub const OutputMember = struct {
    name: []const u8,
    location: u32,
    /// Component count (1 scalar, 2..4 vector).
    components: u8,
};

/// A compiled shader's SPIR-V plus the default-uniform-block layout. The caller owns
/// `spirv` (a word stream), `uniforms` (a member slice whose names are heap-duped),
/// `samplers` (likewise), and `attributes` (likewise). Free all via `deinit`.
pub const CompiledShader = struct {
    spirv: []u32,
    uniforms: []UniformMember,
    samplers: []SamplerMember,
    /// The named vertex inputs (attributes), empty for a fragment shader. Names heap-duped.
    attributes: []AttributeMember,
    /// The named vertex OUTPUT varyings (name -> location + components), empty for a fragment
    /// shader (and for a VS with no located `out`s). Names heap-duped. Used by transform feedback.
    outputs: []OutputMember,
    /// The named uniform interface blocks (glGetUniformBlockIndex). Names heap-duped.
    uniform_blocks: []UniformBlock,
    /// Total bytes the default uniform block occupies (the tight-packed float count * 4).
    block_size: u32,

    pub fn deinit(self: *CompiledShader, allocator: std.mem.Allocator) void {
        allocator.free(self.spirv);
        for (self.uniforms) |u| allocator.free(u.name);
        allocator.free(self.uniforms);
        for (self.samplers) |s| allocator.free(s.name);
        allocator.free(self.samplers);
        for (self.attributes) |a| allocator.free(a.name);
        allocator.free(self.attributes);
        for (self.outputs) |o| allocator.free(o.name);
        allocator.free(self.outputs);
        for (self.uniform_blocks) |b| {
            allocator.free(b.name);
            allocator.free(b.members);
        }
        allocator.free(self.uniform_blocks);
        self.* = undefined;
    }
};

/// Compile a GLSL shader for `stage` to SPIR-V AND surface its default-uniform-block
/// layout (name -> float offset). This is `compileShaderToSpirv` plus the uniform member
/// table. Callers that need glUniform* support use it. Caller owns the result (`deinit`).
pub fn compileShaderWithLayout(allocator: std.mem.Allocator, source: []const u8, stage: spirv.Stage) SpirvError!CompiledShader {
    const lower_stage: lower.ShaderStage = switch (stage) {
        .vertex => .vertex,
        .fragment => .fragment,
        .compute => .compute,
    };
    var shader = try lower.compileShaderStage(allocator, source, lower_stage);
    // Hand ownership of the duped uniform-member names to the caller: copy the member
    // table out, then null the shader's slice so its deinit does not free the names.
    const uniforms = try allocator.alloc(UniformMember, shader.interface.uniforms.len);
    var block_size: u32 = 0;
    for (shader.interface.uniforms, 0..) |u, i| {
        uniforms[i] = u; // name pointer transferred
        block_size = @max(block_size, (u.offset_floats + u.float_count * u.array_len) * 4);
    }
    errdefer allocator.free(uniforms);
    // Transfer the sampler members likewise (name pointers transferred to the caller).
    const samplers = try allocator.alloc(SamplerMember, shader.interface.samplers.len);
    for (shader.interface.samplers, 0..) |s, i| samplers[i] = s;
    errdefer allocator.free(samplers);
    const shader_uniforms = shader.interface.uniforms;
    const shader_samplers = shader.interface.samplers;
    shader.interface.uniforms = &.{}; // prevent double-free of the duped names
    shader.interface.samplers = &.{};

    // The named vertex inputs (attributes). Dupe each name (the shader's input names are freed
    // by its deinit below). Builtins (gl_FragCoord/gl_VertexIndex) have empty names - skipped.
    var attr_count: usize = 0;
    for (shader.interface.inputs) |iv| {
        if (iv.builtin == null and iv.name.len > 0) attr_count += 1;
    }
    const attributes = try allocator.alloc(AttributeMember, attr_count);
    for (attributes) |*a| a.* = .{ .name = "", .location = 0, .components = 0 }; // safe to free on error
    errdefer {
        for (attributes) |a| if (a.name.len > 0) allocator.free(a.name);
        allocator.free(attributes);
    }
    {
        var ai: usize = 0;
        for (shader.interface.inputs) |iv| {
            if (iv.builtin != null or iv.name.len == 0) continue;
            attributes[ai] = .{ .name = try allocator.dupe(u8, iv.name), .location = iv.location, .components = iv.components };
            ai += 1;
        }
    }

    // The named vertex OUTPUT varyings (name -> location + components). Builtins (gl_Position/
    // gl_PointSize) carry an empty name and are skipped. Names duped (the interface's copies
    // are freed by shader.deinit below). Used by the GLES transform-feedback capture path.
    var out_count: usize = 0;
    for (shader.interface.outputs) |ov| {
        if (ov.builtin == null and ov.name.len > 0) out_count += 1;
    }
    const outputs_members = try allocator.alloc(OutputMember, out_count);
    for (outputs_members) |*o| o.* = .{ .name = "", .location = 0, .components = 0 }; // safe to free on error
    errdefer {
        for (outputs_members) |o| if (o.name.len > 0) allocator.free(o.name);
        allocator.free(outputs_members);
    }
    {
        var oi: usize = 0;
        for (shader.interface.outputs) |ov| {
            if (ov.builtin != null or ov.name.len == 0) continue;
            outputs_members[oi] = .{ .name = try allocator.dupe(u8, ov.name), .location = ov.location, .components = @intCast(ov.comps.len) };
            oi += 1;
        }
    }

    defer {
        allocator.free(shader_uniforms);
        allocator.free(shader_samplers);
        shader.deinit();
    }

    const input_vars = try allocator.alloc(spirv.InterfaceVar, shader.interface.inputs.len);
    defer allocator.free(input_vars);
    for (shader.interface.inputs, 0..) |iv, i| input_vars[i] = .{ .location = iv.location, .components = iv.components, .builtin = iv.builtin };

    const output_vars = try allocator.alloc(spirv.OutputVar, shader.interface.outputs.len);
    defer allocator.free(output_vars);
    for (shader.interface.outputs, 0..) |o, i| output_vars[i] = .{ .location = o.location, .components = o.comps, .builtin = o.builtin };

    // NOTE: shader.interface.samplers was emptied above (ownership moved to `samplers`), so
    // read the sampler kinds from the local copy, not the now-empty interface slice.
    const sampler_is_cube = try allocator.alloc(bool, samplers.len);
    defer allocator.free(sampler_is_cube);
    const sampler_is_3d = try allocator.alloc(bool, samplers.len);
    defer allocator.free(sampler_is_3d);
    const sampler_is_2darray = try allocator.alloc(bool, samplers.len);
    defer allocator.free(sampler_is_2darray);
    for (samplers, 0..) |s, i| {
        sampler_is_cube[i] = s.cube;
        sampler_is_3d[i] = s.tex3d;
        sampler_is_2darray[i] = s.tex2darray;
    }

    const info = spirv.ShaderInfo{
        .stage = stage,
        .inputs = input_vars,
        .outputs = output_vars,
        .local_size = shader.interface.local_size,
        .uniform_count = shader.interface.uniform_count,
        .sampler_count = shader.interface.sampler_count,
        .sampler_is_cube = sampler_is_cube,
        .sampler_is_3d = sampler_is_3d,
        .sampler_is_2darray = sampler_is_2darray,
    };
    // The named uniform blocks (dupe each name: the interface's copies are freed by
    // shader.deinit above via the deferred cleanup).
    const uniform_blocks = try allocator.alloc(UniformBlock, shader.interface.uniform_blocks.len);
    for (uniform_blocks) |*b| b.* = .{ .name = "", .binding = 0, .byte_offset = 0, .byte_size = 0, .members = &.{} };
    errdefer {
        for (uniform_blocks) |b| {
            if (b.name.len > 0) allocator.free(b.name);
            allocator.free(b.members);
        }
        allocator.free(uniform_blocks);
    }
    for (shader.interface.uniform_blocks, 0..) |b, i| {
        uniform_blocks[i] = .{ .name = try allocator.dupe(u8, b.name), .binding = b.binding, .byte_offset = b.byte_offset, .byte_size = b.byte_size, .members = try allocator.dupe(UniformBlockMember, b.members) };
    }

    const words = try spirv.emitShader(allocator, &shader.func, info);
    return .{ .spirv = words, .uniforms = uniforms, .samplers = samplers, .attributes = attributes, .outputs = outputs_members, .uniform_blocks = uniform_blocks, .block_size = block_size };
}

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
    const lower_stage: lower.ShaderStage = switch (stage) {
        .vertex => .vertex,
        .fragment => .fragment,
        .compute => .compute,
    };
    var shader = try lower.compileShaderStage(allocator, source, lower_stage);
    defer shader.deinit();

    const input_vars = try allocator.alloc(spirv.InterfaceVar, shader.interface.inputs.len);
    defer allocator.free(input_vars);
    for (shader.interface.inputs, 0..) |iv, i| input_vars[i] = .{ .location = iv.location, .components = iv.components, .builtin = iv.builtin };

    const output_vars = try allocator.alloc(spirv.OutputVar, shader.interface.outputs.len);
    defer allocator.free(output_vars);
    for (shader.interface.outputs, 0..) |o, i| output_vars[i] = .{ .location = o.location, .components = o.comps, .builtin = o.builtin };

    const sampler_is_cube = try allocator.alloc(bool, shader.interface.samplers.len);
    defer allocator.free(sampler_is_cube);
    const sampler_is_3d = try allocator.alloc(bool, shader.interface.samplers.len);
    defer allocator.free(sampler_is_3d);
    const sampler_is_2darray = try allocator.alloc(bool, shader.interface.samplers.len);
    defer allocator.free(sampler_is_2darray);
    for (shader.interface.samplers, 0..) |s, i| {
        sampler_is_cube[i] = s.cube;
        sampler_is_3d[i] = s.tex3d;
        sampler_is_2darray[i] = s.tex2darray;
    }

    const info = spirv.ShaderInfo{
        .stage = stage,
        .inputs = input_vars,
        .outputs = output_vars,
        .local_size = shader.interface.local_size,
        .uniform_count = shader.interface.uniform_count,
        .sampler_count = shader.interface.sampler_count,
        .sampler_is_cube = sampler_is_cube,
        .sampler_is_3d = sampler_is_3d,
        .sampler_is_2darray = sampler_is_2darray,
    };
    return spirv.emitShader(allocator, &shader.func, info);
}

test {
    std.testing.refAllDecls(@import("vulcan-glsl/lexer.zig"));
    std.testing.refAllDecls(@import("vulcan-glsl/parser.zig"));
    std.testing.refAllDecls(@import("vulcan-glsl/preprocess.zig"));
    std.testing.refAllDecls(lower);
}

test "compileShaderWithLayout surfaces the es2gears default-uniform-block layout" {
    const gpa = std.testing.allocator;
    // The es2gears vertex shader's uniforms in declaration order: two mat4s then two vec4s.
    const vs_src =
        \\attribute vec3 position;
        \\attribute vec3 normal;
        \\uniform mat4 ModelViewProjectionMatrix;
        \\uniform mat4 NormalMatrix;
        \\uniform vec4 LightSourcePosition;
        \\uniform vec4 MaterialColor;
        \\varying vec4 Color;
        \\void main(void) {
        \\  vec3 N = normalize(vec3(NormalMatrix * vec4(normal, 1.0)));
        \\  vec3 L = normalize(LightSourcePosition.xyz);
        \\  float diffuse = max(dot(N, L), 0.0);
        \\  Color = vec4((0.2 + diffuse) * MaterialColor.xyz, MaterialColor.a);
        \\  gl_Position = ModelViewProjectionMatrix * vec4(position, 1.0);
        \\}
    ;
    var c = try compileShaderWithLayout(gpa, vs_src, .vertex);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 4), c.uniforms.len);
    // mat4 @ 0 (16 floats), mat4 @ 16, vec4 @ 32, vec4 @ 36. Block = 40*4 = 160 bytes.
    try std.testing.expectEqualStrings("ModelViewProjectionMatrix", c.uniforms[0].name);
    try std.testing.expectEqual(@as(u32, 0), c.uniforms[0].offset_floats);
    try std.testing.expectEqual(@as(u32, 16), c.uniforms[0].float_count);
    try std.testing.expectEqual(@as(u8, 4), c.uniforms[0].mat_dim);
    try std.testing.expectEqualStrings("NormalMatrix", c.uniforms[1].name);
    try std.testing.expectEqual(@as(u32, 16), c.uniforms[1].offset_floats);
    try std.testing.expectEqualStrings("LightSourcePosition", c.uniforms[2].name);
    try std.testing.expectEqual(@as(u32, 32), c.uniforms[2].offset_floats);
    try std.testing.expectEqual(@as(u32, 4), c.uniforms[2].float_count);
    try std.testing.expectEqualStrings("MaterialColor", c.uniforms[3].name);
    try std.testing.expectEqual(@as(u32, 36), c.uniforms[3].offset_floats);
    try std.testing.expectEqual(@as(u32, 160), c.block_size);
    // SPIR-V magic word present.
    try std.testing.expectEqual(@as(u32, 0x07230203), c.spirv[0]);
}

test "compileShaderWithLayout surfaces sampler2D members (name -> binding) alongside UBO uniforms" {
    const gpa = std.testing.allocator;
    // A textured fragment shader: a sampler uniform AND a default-block scalar uniform must
    // coexist - the samplers are separate descriptors (bindings 1 and 2, binding 0 is reserved
    // for the default uniform block), the scalar a UBO float at slot 0.
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex0;
        \\uniform sampler2D uTex1;
        \\uniform float uAlpha;
        \\varying vec2 vUV;
        \\void main(void) {
        \\  vec4 a = texture2D(uTex0, vUV);
        \\  vec4 b = texture2D(uTex1, vUV);
        \\  gl_FragColor = vec4((a.rgb + b.rgb) * uAlpha, 1.0);
        \\}
    ;
    var c = try compileShaderWithLayout(gpa, fs_src, .fragment);
    defer c.deinit(gpa);
    // Two samplers in declaration order -> bindings 2 and 3 (bindings 0/1 reserved for the
    // per-stage default uniform blocks: VS block at 0, FS block at 1).
    try std.testing.expectEqual(@as(usize, 2), c.samplers.len);
    try std.testing.expectEqualStrings("uTex0", c.samplers[0].name);
    try std.testing.expectEqual(@as(u32, 2), c.samplers[0].binding);
    try std.testing.expectEqualStrings("uTex1", c.samplers[1].name);
    try std.testing.expectEqual(@as(u32, 3), c.samplers[1].binding);
    // The non-sampler uniform is still a UBO float member (not a sampler).
    try std.testing.expectEqual(@as(usize, 1), c.uniforms.len);
    try std.testing.expectEqualStrings("uAlpha", c.uniforms[0].name);
    try std.testing.expectEqual(@as(u32, 0x07230203), c.spirv[0]);
}

test "samplerCube + textureCube(vec3) compiles to a Cube-dim image sampled by a vec3" {
    const gpa = std.testing.allocator;
    // A skybox-style fragment shader: a cube sampler addressed by a vec3 direction. This used
    // to fail (the texture lookup forced a vec2 coord), the wall that blocked cubemaps.
    const fs_src =
        \\precision mediump float;
        \\uniform samplerCube uSky;
        \\varying vec3 vDir;
        \\void main(void) {
        \\  gl_FragColor = textureCube(uSky, vDir);
        \\}
    ;
    var c = try compileShaderWithLayout(gpa, fs_src, .fragment);
    defer c.deinit(gpa);
    // The sampler surfaces as a cube sampler at the first sampler binding (slot 2).
    try std.testing.expectEqual(@as(usize, 1), c.samplers.len);
    try std.testing.expectEqualStrings("uSky", c.samplers[0].name);
    try std.testing.expectEqual(@as(u32, 2), c.samplers[0].binding);
    try std.testing.expect(c.samplers[0].cube);
    // Valid SPIR-V, and it declares an OpTypeImage with Dim = Cube (3). OpTypeImage is opcode
    // 25; its operands are [result_id, sampled_type, Dim, ...], so Dim is the 3rd operand word.
    try std.testing.expectEqual(@as(u32, 0x07230203), c.spirv[0]);
    var found_cube = false;
    var i: usize = 5; // skip the 5-word SPIR-V header
    while (i < c.spirv.len) {
        const word0 = c.spirv[i];
        const opcode = word0 & 0xFFFF;
        const wcount = word0 >> 16;
        if (wcount == 0) break; // malformed guard
        if (opcode == 25 and i + 3 < c.spirv.len and c.spirv[i + 3] == 3) found_cube = true;
        i += wcount;
    }
    try std.testing.expect(found_cube);
}

test "a sampler shader with an all-CONSTANT integer coord (texelFetch(s, ivec3(0,0,1), 0)) compiles" {
    // Regression: an all-constant fetch coord has no float VARYING param, so the sampler-image
    // f32 type must be found among the function's values (gl_FragColor = vec4), not just the params.
    const gpa = std.testing.allocator;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2DArray uArr;
        \\void main() { gl_FragColor = texelFetch(uArr, ivec3(0, 0, 1), 0); }
    ;
    var c = try compileShaderWithLayout(gpa, fs_src, .fragment);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), c.samplers.len);
    try std.testing.expect(c.samplers[0].tex2darray);
}

test "textureLod compiles to an explicit-LOD sample (OpImageSampleExplicitLod)" {
    const gpa = std.testing.allocator;
    const fs_src =
        \\precision mediump float;
        \\uniform sampler2D uTex;
        \\varying vec2 vUV;
        \\void main() { gl_FragColor = textureLod(uTex, vUV, 2.0); }
    ;
    const words = try compileShaderToSpirv(gpa, fs_src, .fragment);
    defer gpa.free(words);
    // OpImageSampleExplicitLod is opcode 88; the implicit form (87) must NOT be used here.
    var found_explicit = false;
    var found_implicit = false;
    var i: usize = 5;
    while (i < words.len) {
        const word0 = words[i];
        const opcode = word0 & 0xFFFF;
        const wcount = word0 >> 16;
        if (wcount == 0) break;
        if (opcode == 88) found_explicit = true;
        if (opcode == 87) found_implicit = true;
        i += wcount;
    }
    try std.testing.expect(found_explicit);
    try std.testing.expect(!found_implicit);
}

test "if/else with a live uniform matrix across the branch (glmark2 conditionals)" {
    // A uniform mat4 lives unchanged across an if/else that reassigns a float. The merge
    // must carry the matrix Val through (no phi needed when its components are identical).
    const src =
        \\attribute vec3 position;
        \\uniform mat4 ModelViewProjectionMatrix;
        \\varying vec4 dummy;
        \\void main(void){
        \\  dummy = vec4(1.0);
        \\  float d = fract(position.x);
        \\  if (d >= 0.5) d = fract(2.0 * d); else d = fract(3.0 * d);
        \\  vec4 pos = vec4(position.x, position.y + 0.1 * d * fract(position.x), position.z, 1.0);
        \\  gl_Position = ModelViewProjectionMatrix * pos;
        \\}
    ;
    const r = try compileShaderToSpirv(std.testing.allocator, src, .vertex);
    std.testing.allocator.free(r);
}
