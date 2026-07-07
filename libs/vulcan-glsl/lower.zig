//! Lower a parsed GLSL module to Vulcan IR. Handles scalar functions (float/int/uint/
//! bool) and float vectors (vec2/vec3/vec4), scalarized to per-component scalar values as
//! the SPIR-V frontend does (no backend vector support needed): a `vecN` value is N scalar
//! IR values, vector arithmetic is component-wise, swizzles select components, `dot` is a
//! sum of products. Bodies flow in SSA form (assignment rebinds a name), so no allocas.

const std = @import("std");
const ir = @import("vulcan-ir");
const parser = @import("parser.zig");
const preprocess = @import("preprocess.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = parser.Type;

pub const Error = parser.Error || preprocess.Error || error{ Unsupported, TypeMismatch, UndefinedName, MissingMain, TooManyOutputs, BadSwizzle } || std.mem.Allocator.Error;

pub const LoweredFunction = struct { name: []u8, func: Function };

pub const Module = struct {
    functions: []LoweredFunction,

    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        for (self.functions) |*lf| {
            lf.func.deinit();
            allocator.free(lf.name);
        }
        allocator.free(self.functions);
        self.functions = &.{};
    }

    pub fn find(self: *const Module, name: []const u8) ?*const Function {
        for (self.functions) |*lf| if (std.mem.eql(u8, lf.name, name)) return &lf.func;
        return null;
    }

    /// A mutable handle to a function, for passes that rewrite the IR (vectorization).
    pub fn findMut(self: *Module, name: []const u8) ?*Function {
        for (self.functions) |*lf| if (std.mem.eql(u8, lf.name, name)) return &lf.func;
        return null;
    }
};

/// Compile GLSL source to Vulcan IR. Caller owns the result.
pub fn compile(allocator: std.mem.Allocator, source: []const u8) Error!Module {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const pp = try preprocess.run(arena.allocator(), source);
    var p = try parser.Parser.init(arena.allocator(), pp);
    const ast = try p.parseModule();

    var list: std.ArrayList(LoweredFunction) = .empty;
    errdefer {
        for (list.items) |*lf| {
            lf.func.deinit();
            allocator.free(lf.name);
        }
        list.deinit(allocator);
    }
    for (ast.functions) |fn_ast| {
        // A function returning a vector/matrix cannot be a standalone IR function (the IR
        // `ret` is single-value). Such a function is only ever inlined at its call sites, so
        // skip emitting dead, unrepresentable standalone IR for it.
        if (vecLen(fn_ast.ret) != 0 or matDim(fn_ast.ret) != 0) continue;
        const name = try allocator.dupe(u8, fn_ast.name);
        errdefer allocator.free(name);
        var func = try lowerFunction(allocator, fn_ast, ast.functions, ast.structs);
        errdefer func.deinit();
        try list.append(allocator, .{ .name = name, .func = func });
    }
    return .{ .functions = try list.toOwnedSlice(allocator) };
}

/// An input interface variable: its location (or a SPIR-V BuiltIn number when `builtin`
/// is set, e.g. gl_FragCoord), its component count (1 scalar, 2..4 vector), and its source
/// attribute name (heap-duped, empty for a builtin). The GLES layer resolves
/// glGetAttribLocation against the named vertex inputs (a `varying`-derived fragment input
/// is also named, but only vertex inputs are attributes).
pub const ShaderVar = struct { location: u32, components: u8, builtin: ?u32 = null, name: []const u8 = "" };

/// The output interface variable: its location (or a builtin, e.g. gl_Position) and the
/// scalarized component IR values. `builtin` is the SPIR-V BuiltIn number (0 = Position)
/// when the output is a builtin rather than a located varying.
pub const ShaderOutput = struct { location: u32, comps: []Value, builtin: ?u32 = null, name: []const u8 = "" };

/// One member of the default uniform block (a bare `uniform <type> <name>;`). The
/// front end packs all default-block uniforms into ONE block of scalarized floats in
/// declaration order (the same packing the emitted PushConstant block uses, a tight
/// 4-byte stride per scalar). `offset_floats` is this member's first float index in the
/// block. `float_count` is how many floats it occupies (a mat4 = 16, a vec4 = 4, a
/// scalar = 1). This is the name->offset map a `glGetUniformLocation` + `glUniform*`
/// implementation needs to write each uniform's bytes at the right block offset.
pub const UniformMember = struct {
    name: []const u8,
    offset_floats: u32,
    /// Floats in ONE element (a scalar = 1, a vecN = N, a matN = N*N). For a non-array
    /// uniform this is the whole member. For an array it is the per-element stride and
    /// count (tight-packed, no std140 padding, matching this front end's block layout).
    float_count: u32,
    /// Matrix dimension (2/3/4) for a matN uniform, else 0. Vector length goes in
    /// `float_count` when this is 0.
    mat_dim: u8 = 0,
    /// Array length for an array uniform (`uniform vec3 c[3];` -> 3), else 1 for a scalar
    /// uniform. The element stride in floats is `float_count`. Element `i` lives at
    /// `offset_floats + i*float_count`.
    array_len: u32 = 1,
};

/// One `uniform sampler2D` declaration: its name plus its SPIR-V binding (descriptor set
/// 0). The binding is the declaration order among samplers, matching what the emitter
/// decorates and what `texture(name, uv)` lowers to (`tex.sample.<binding>`). The GLES
/// layer uses this to map a sampler-uniform name -> its binding so the texture bound to
/// the unit `glUniform1i` set on that uniform can be bound to the HAL at the right binding.
pub const SamplerMember = struct {
    name: []const u8,
    binding: u32,
    /// true for a `samplerCube` (a Cube-dim image sampled by a vec3 direction), false for a
    /// plain `sampler2D`. The emitter declares the image with the matching SPIR-V Dim.
    cube: bool = false,
    /// true for a `sampler3D` (a 3D-dim image sampled by a vec3 coordinate). A cube and a 3D
    /// sampler share the vec3-coordinate sample path; the host sampler dispatches on the bound
    /// descriptor (cube face-select vs trilinear 3D). Mutually exclusive with `cube`.
    tex3d: bool = false,
    /// true for a `sampler2DArray` (a 2D-Arrayed image sampled by a vec3 (u, v, layer)). Shares
    /// the vec3-coordinate sample path; the third coordinate is a raw LAYER index (not normalized)
    /// selecting one independent 2D layer (no cross-layer filtering). Mutually exclusive with the
    /// others. The emitter declares the image with SPIR-V Dim 2D + Arrayed.
    tex2darray: bool = false,
};

/// A named uniform interface block surfaced to the host. Its members are part of the default
/// uniform block (laid out by `uniforms` above); this record pins the block's name, its GL
/// binding point (`layout(binding=N)`, else 0 - the app may retarget it with
/// glUniformBlockBinding), and its byte size (GL_UNIFORM_BLOCK_DATA_SIZE). The GLES layer
/// resolves glGetUniformBlockIndex to this block and, at draw, feeds the glBindBufferBase'd
/// buffer to the block's descriptor instead of the program's glUniform* storage.
pub const UniformBlock = struct {
    name: []const u8,
    binding: u32,
    /// Byte offset of the block's first member in the default block (the block occupies a
    /// contiguous member range in declaration order).
    byte_offset: u32,
    /// Total bytes the block spans per the std140 layout rules (the GL_UNIFORM_BLOCK_DATA_SIZE
    /// an app sizes its glBindBufferBase'd buffer to), rounded up to 16.
    byte_size: u32,
    /// Per-member std140<->tight repack table (one entry per member, declaration order). The
    /// shader reads the block members TIGHT-packed (`byte_offset`-relative), but a
    /// glBindBufferBase'd user buffer is laid out per std140 (vec3 16-align, vec2 8-align, arrays
    /// 16-stride, mat columns 16-stride). The host uses this to gather each member from its std140
    /// offset in the user buffer into the tight offset the shader expects. For an all-16-byte-member
    /// block (vec4/mat4) every entry is an identity copy (std140 == tight). Empty only for a block
    /// with no members.
    members: []const UniformBlockMember = &.{},
};

/// One member of a named uniform block, precomputed so the host can repack a std140 user buffer
/// (glBindBufferBase) into the tight layout the shader reads. All offsets are block-relative.
/// The repack copies `unit_count` chunks of `copy_bytes`: chunk `u` moves the bytes at
/// `std140_offset + u*std140_stride` in the user buffer to `tight_offset + u*tight_stride` in
/// the shader's tight UBO. A unit is a whole scalar/vector (non-array), an array element, or a
/// matrix column - each a contiguous run whose std140 vs tight stride differs by the padding.
pub const UniformBlockMember = struct {
    std140_offset: u32,
    tight_offset: u32,
    copy_bytes: u32,
    unit_count: u32,
    std140_stride: u32,
    tight_stride: u32,
};

pub const ShaderInterface = struct {
    inputs: []ShaderVar,
    /// Every shader output: a vertex shader writing gl_Position AND varyings has more than
    /// one. A fragment shader has its single color output (`gl_FragColor` / the `out`).
    outputs: []ShaderOutput,
    local_size: [3]u32,
    uniform_count: u32 = 0,
    sampler_count: u32 = 0,
    /// The default-uniform-block members, in declaration order, with their float offsets.
    /// Empty when the shader declares no (non-sampler) uniforms.
    uniforms: []UniformMember = &.{},
    /// The `uniform sampler2D` members (name -> binding), in declaration order. Empty when
    /// the shader declares no samplers.
    samplers: []SamplerMember = &.{},
    /// The named uniform interface blocks, in declaration order. Empty when the shader
    /// declares no named blocks (only loose default-block uniforms).
    uniform_blocks: []UniformBlock = &.{},
};

pub const LoweredShader = struct {
    func: Function,
    interface: ShaderInterface,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LoweredShader) void {
        self.func.deinit();
        for (self.interface.inputs) |iv| if (iv.name.len > 0) self.allocator.free(iv.name);
        self.allocator.free(self.interface.inputs);
        for (self.interface.outputs) |o| {
            self.allocator.free(o.comps);
            if (o.name.len > 0) self.allocator.free(o.name); // duped varying name (builtins are "")
        }
        self.allocator.free(self.interface.outputs);
        for (self.interface.uniforms) |u| self.allocator.free(u.name);
        self.allocator.free(self.interface.uniforms);
        for (self.interface.samplers) |s| self.allocator.free(s.name);
        self.allocator.free(self.interface.samplers);
        for (self.interface.uniform_blocks) |b| {
            self.allocator.free(b.name);
            self.allocator.free(b.members);
        }
        self.allocator.free(self.interface.uniform_blocks);
    }
};

/// The shader pipeline stage, used to resolve GLSL ES 1.00 `varying` (an `out` in a
/// vertex shader, an `in` in a fragment shader) and the implicit `gl_FragColor` output.
/// Mirrors `vulcan-spirv`'s Stage without depending on it (this module is IR-only).
pub const ShaderStage = enum { vertex, fragment, compute };

/// Compile a GLSL shader to a Vulcan IR function plus its interface. `in`/`attribute`
/// globals become the function parameters, the single `out`/`varying` (or `gl_FragColor`)
/// the returned value. The stage is unknown to the source for `varying` in GLSL ES 1.00,
/// so it defaults to `.fragment` here. Callers that know the stage use `compileShaderStage`.
pub fn compileShader(allocator: std.mem.Allocator, source: []const u8) Error!LoweredShader {
    return compileShaderStage(allocator, source, .fragment);
}

/// Stage-aware shader compile (see `compileShader`). The stage resolves GLSL ES 1.00
/// `varying`/`attribute` and `gl_FragColor`.
pub fn compileShaderStage(allocator: std.mem.Allocator, source: []const u8, stage: ShaderStage) Error!LoweredShader {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const pp = try preprocess.run(arena.allocator(), source);
    var p = try parser.Parser.init(arena.allocator(), pp);
    const ast = try p.parseModule();

    var main_fn: ?parser.Function = null;
    for (ast.functions) |f| if (std.mem.eql(u8, f.name, "main")) {
        main_fn = f;
    };
    const main = main_fn orelse return error.MissingMain;

    var func = Function.init(allocator);
    errdefer func.deinit();
    const entry = try func.appendBlock();

    var env: std.ArrayList(Var) = .empty;
    defer env.deinit(allocator);
    var inputs: std.ArrayList(ShaderVar) = .empty;
    errdefer inputs.deinit(allocator);

    var l = L{ .func = &func, .block = entry, .env = &env, .allocator = allocator, .stage = stage, .user_fns = ast.functions, .structs = ast.structs, .comp_arena = arena.allocator() };
    defer l.loops.deinit(allocator);
    defer l.samplers.deinit(allocator);
    defer l.sampler_cube.deinit(allocator);
    defer l.sampler_3d.deinit(allocator);
    defer l.sampler_2darray.deinit(allocator);
    defer l.sampler_shadow.deinit(allocator);
    defer l.sampler_cube_shadow.deinit(allocator);
    defer l.sampler_2darray_shadow.deinit(allocator);

    // `in`/`attribute` globals (and a fragment shader's `varying`s) -> parameters. A
    // vector input is one variable over N scalar params. GLSL ES 1.00 has no explicit
    // `layout(location=)`, so unlocated inputs are assigned locations in declaration
    // order (the gradient triangle's `attribute vec2 aPos; attribute vec3 aColor;` get
    // locations 0 and 1, matching the host vertex layout).
    var next_in_location: u32 = 0;
    for (ast.globals) |g| {
        if (effectiveQualifier(g.qualifier, stage) != .in_) continue;
        const loc = g.location orelse next_in_location;
        next_in_location = loc + 1;
        const n = vecLen(g.ty);
        // Dupe the input's name out of the parse arena so it outlives compilation (the GLES
        // layer resolves glGetAttribLocation against it). Freed in LoweredShader.deinit.
        const in_name = try allocator.dupe(u8, g.name);
        errdefer allocator.free(in_name);
        if (n == 0) {
            // A SCALAR input. The software gather ABI passes every VS input as an f32 in a
            // vector register, so a float scalar is an f32 param directly. An INTEGER scalar
            // attribute (glVertexAttribIPointer, `in int`/`in uint`) is ALSO delivered as an
            // f32 holding the raw integer VALUE (e.g. 200.0), then converted to int at entry -
            // the shader then sees a genuine int with no ABI change.
            if (g.ty == .int or g.ty == .uint) {
                const f32t = try func.types.intern(.{ .float = .f32 });
                const pv = try func.appendBlockParam(entry, f32t);
                const iv = try coerce(&l, .{ .value = pv, .ty = .float }, g.ty);
                try env.append(allocator, .{ .name = g.name, .val = .{ .scalar = iv } });
            } else {
                const pv = try func.appendBlockParam(entry, try irType(&func, g.ty));
                try env.append(allocator, .{ .name = g.name, .val = .{ .scalar = .{ .value = pv, .ty = g.ty } } });
            }
            try inputs.append(allocator, .{ .location = loc, .components = 1, .name = in_name });
        } else {
            // A VECTOR input, scalarized to N f32 params (the gather ABI is f32-only). For an
            // INTEGER vector (ivecN/uvecN, from glVertexAttribIPointer) each f32 param carries
            // the raw integer VALUE; convert it to the integer component type at entry so the
            // shader operates on genuine ints (float(aData.x), aData.x as an array index, ...).
            // A float vector (comp_ty == .float) keeps its f32 params verbatim - unchanged.
            const comp_ty = vecCompType(g.ty);
            const f32t = try func.types.intern(.{ .float = .f32 });
            var comps: [4]Value = undefined;
            for (0..n) |i| {
                const pv = try func.appendBlockParam(entry, f32t);
                comps[i] = if (comp_ty == .int or comp_ty == .uint)
                    (try coerce(&l, .{ .value = pv, .ty = .float }, comp_ty)).value
                else
                    pv;
            }
            try env.append(allocator, .{ .name = g.name, .val = .{ .vector = .{ .comps = comps, .len = n, .comp_ty = comp_ty } } });
            try inputs.append(allocator, .{ .location = loc, .components = n, .name = in_name });
        }
    }

    // Input builtins are synthesized as builtin-decorated input parameters when read.
    // gl_FragCoord (a vec4 fragment input), gl_VertexIndex (an int vertex input).
    if (bodyReferences(main.body, "gl_FragCoord")) {
        const f32t = try func.types.intern(.{ .float = .f32 });
        var comps: [4]Value = undefined;
        for (0..4) |i| comps[i] = try func.appendBlockParam(entry, f32t);
        try env.append(allocator, .{ .name = "gl_FragCoord", .val = .{ .vector = .{ .comps = comps, .len = 4 } } });
        try inputs.append(allocator, .{ .location = 0, .components = 4, .builtin = 15 }); // FragCoord
    }
    // gl_PointCoord (a vec2 fragment input, 0..1 across a point sprite; the rasterizer supplies it).
    if (bodyReferences(main.body, "gl_PointCoord")) {
        const f32t = try func.types.intern(.{ .float = .f32 });
        var comps: [4]Value = undefined;
        for (0..2) |i| comps[i] = try func.appendBlockParam(entry, f32t);
        try env.append(allocator, .{ .name = "gl_PointCoord", .val = .{ .vector = .{ .comps = comps, .len = 2 } } });
        try inputs.append(allocator, .{ .location = 0, .components = 2, .builtin = 16 }); // PointCoord
    }
    // gl_VertexIndex (Vulkan) / gl_VertexID (GLES ES 3.00) are the SAME DA-delivered vertex
    // id (BuiltIn 42): for a non-indexed, base-vertex-0 draw (the common case) they are equal,
    // so both names resolve to one synthesized i32 param. Enables the vertex-buffer-less
    // full-screen-triangle idiom (gl_Position derived from gl_VertexID) that every
    // post-process / compositor blit pass uses.
    if (bodyReferences(main.body, "gl_VertexIndex") or bodyReferences(main.body, "gl_VertexID")) {
        const pv = try func.appendBlockParam(entry, try irType(&func, .int));
        try env.append(allocator, .{ .name = "gl_VertexIndex", .val = .{ .scalar = .{ .value = pv, .ty = .int } } });
        try env.append(allocator, .{ .name = "gl_VertexID", .val = .{ .scalar = .{ .value = pv, .ty = .int } } });
        try inputs.append(allocator, .{ .location = 0, .components = 1, .builtin = 42 }); // VertexIndex
    }
    // gl_InstanceIndex (Vulkan) / gl_InstanceID (GLES ES 3.00): the DA-delivered instance id
    // (BuiltIn 43), equal for a base-instance-0 draw. A VS indexes per-instance data by it
    // (glDrawArraysInstanced). The backend sources it from ALD a[0x2f8] (nvidia) / a leading
    // i32 param (software).
    if (bodyReferences(main.body, "gl_InstanceIndex") or bodyReferences(main.body, "gl_InstanceID")) {
        const pv = try func.appendBlockParam(entry, try irType(&func, .int));
        try env.append(allocator, .{ .name = "gl_InstanceIndex", .val = .{ .scalar = .{ .value = pv, .ty = .int } } });
        try env.append(allocator, .{ .name = "gl_InstanceID", .val = .{ .scalar = .{ .value = pv, .ty = .int } } });
        try inputs.append(allocator, .{ .location = 0, .components = 1, .builtin = 43 }); // InstanceIndex
    }

    // `uniform` globals become a push-constant block of floats (a vector/matrix uniform is
    // its scalarized components), appended as parameters after the inputs. We ALSO record a
    // name->offset member table (`uniform_members`) in declaration order so the EGL/GLES
    // layer can resolve glGetUniformLocation + write glUniform* bytes at the matching block
    // offset. The packing here (tight, 4 bytes per scalar, declaration order) is exactly the
    // emitted PushConstant block's Offset = i*4 layout, so the table offsets line up with
    // the shader's loads.
    var uniform_count: u32 = 0;
    var uniform_members: std.ArrayList(UniformMember) = .empty;
    errdefer {
        for (uniform_members.items) |u| allocator.free(u.name);
        uniform_members.deinit(allocator);
    }
    // The sampler members (name -> binding), in declaration order. Names are duped out of
    // the parse arena so they outlive the compile (parallel to uniform_members).
    var sampler_members: std.ArrayList(SamplerMember) = .empty;
    errdefer {
        for (sampler_members.items) |s| allocator.free(s.name);
        sampler_members.deinit(allocator);
    }
    for (ast.globals) |g| {
        if (g.qualifier != .uniform) continue;
        // An opaque `sampler2D` is a separate descriptor, not a push-constant float. The
        // SURFACED descriptor binding is declaration-order + 2: bindings 0 and 1 are reserved
        // for the per-stage default uniform blocks (the push-constants the `uniform` floats
        // lower to - the VERTEX block at 0, the FRAGMENT block at 1), so a sampler never shares
        // the shared-constant-bank slot a uniform-block pointer uses (which on the nvidia backend
        // made a textured + uniform shader render black, or fault the GPU when the FS block landed
        // on the sampler's slot). This MUST match vulcan-spirv/emit.zig, which decorates the
        // SPIR-V variable with the same +2 binding. `texture(name,uv)` still resolves the variable
        // by DECLARATION INDEX via `L.samplerBinding` (the sampler_vars[] array index), so the
        // sample op is unchanged - only the Vulkan binding number the host uses to place the
        // descriptor shifts.
        if (g.ty == .sampler2d or g.ty == .sampler_cube or g.ty == .sampler3d or g.ty == .sampler2darray or g.ty == .sampler2dshadow or g.ty == .samplercubeshadow or g.ty == .sampler2darrayshadow) {
            const is_cube_shadow = g.ty == .samplercubeshadow;
            // A samplerCubeShadow is a Cube-dim image (flagged is_cube so the emitter picks Cube dim)
            // AND a shadow (handled by the scalar-returning cube-shadow path).
            const is_cube = g.ty == .sampler_cube or is_cube_shadow;
            const is_3d = g.ty == .sampler3d;
            // A sampler2DArrayShadow is a 2D-Arrayed image (flagged is_2darray so the emitter picks
            // 2D-Arrayed dim) AND a shadow (handled by the scalar-returning 2darray-shadow path).
            const is_2darray_shadow = g.ty == .sampler2darrayshadow;
            const is_2darray = g.ty == .sampler2darray or is_2darray_shadow;
            const is_shadow = g.ty == .sampler2dshadow;
            const binding: u32 = @as(u32, @intCast(l.samplers.items.len)) + 2;
            try l.samplers.append(allocator, g.name);
            try l.sampler_cube.append(allocator, is_cube);
            try l.sampler_3d.append(allocator, is_3d);
            try l.sampler_2darray.append(allocator, is_2darray);
            try l.sampler_shadow.append(allocator, is_shadow);
            try l.sampler_cube_shadow.append(allocator, is_cube_shadow);
            try l.sampler_2darray_shadow.append(allocator, is_2darray_shadow);
            const owned_s = try allocator.dupe(u8, g.name);
            errdefer allocator.free(owned_s);
            try sampler_members.append(allocator, .{ .name = owned_s, .binding = binding, .cube = is_cube, .tex3d = is_3d, .tex2darray = is_2darray });
            continue;
        }
        const f32t = try func.types.intern(.{ .float = .f32 });
        const md = matDim(g.ty);
        const n = vecLen(g.ty);
        // The name string lives in the parse arena (freed at function exit). Dupe it into
        // the caller's allocator so the member table outlives the compile.
        const owned_name = try allocator.dupe(u8, g.name);
        errdefer allocator.free(owned_name);
        // Floats one ELEMENT of this uniform occupies (the array stride): mat = dim*dim,
        // vec = len, scalar = 1.
        const elem_floats: u32 = if (md != 0) @intCast(@as(usize, md) * md) else if (n != 0) @intCast(n) else 1;
        const len: u32 = g.array_len orelse 1;
        const base_offset = uniform_count;
        // Build a Val per element (each reading its own push-constant params) so a
        // constant-indexed `name[i]` resolves to element i's components.
        var elems = try l.comp_arena.alloc(Val, len);
        for (0..len) |ei| {
            if (md != 0) {
                var comps: [16]Value = undefined;
                for (0..elem_floats) |i| comps[i] = try func.appendBlockParam(entry, f32t);
                elems[ei] = .{ .matrix = .{ .comps = comps, .dim = md } };
            } else if (n != 0) {
                var comps: [4]Value = undefined;
                for (0..n) |i| comps[i] = try func.appendBlockParam(entry, f32t);
                elems[ei] = .{ .vector = .{ .comps = comps, .len = n } };
            } else {
                const pv = try func.appendBlockParam(entry, f32t);
                elems[ei] = .{ .scalar = .{ .value = pv, .ty = .float } };
            }
            uniform_count += elem_floats;
        }
        if (g.array_len == null) {
            try env.append(allocator, .{ .name = g.name, .val = elems[0] });
        } else {
            try env.append(allocator, .{ .name = g.name, .val = .{ .array = .{ .elems = elems } } });
        }
        try uniform_members.append(allocator, .{ .name = owned_name, .offset_floats = base_offset, .float_count = elem_floats, .mat_dim = md, .array_len = len });
    }

    // The `out`/`varying` output globals: each a pre-declared assignable slot (scalar or
    // vector of zeros). A fragment shader's `varying`s are inputs (handled above), so only
    // a vertex shader reaches here for `varying`. Locations are assigned in declaration
    // order when not explicit (GLSL ES 1.00). A vertex shader may have several (one per
    // varying). A fragment shader has at most one located `out`.
    var out_decls: std.ArrayList(struct { name: []const u8, location: u32 }) = .empty;
    defer out_decls.deinit(allocator);
    var next_out_location: u32 = 0;
    for (ast.globals) |g| {
        if (effectiveQualifier(g.qualifier, stage) != .out_) continue;
        const loc = g.location orelse next_out_location;
        next_out_location = loc + 1;
        try out_decls.append(allocator, .{ .name = g.name, .location = loc });
        const n = vecLen(g.ty);
        const init_val: Val = if (n == 0)
            .{ .scalar = .{ .value = try zero(&l, g.ty), .ty = g.ty } }
        else blk: {
            const f32t = try func.types.intern(.{ .float = .f32 });
            var comps: [4]Value = undefined;
            for (0..n) |i| comps[i] = try func.appendInst(entry, f32t, .{ .fconst = 0 });
            break :blk .{ .vector = .{ .comps = comps, .len = n } };
        };
        try env.append(allocator, .{ .name = g.name, .val = init_val });
    }

    // gl_Position (a builtin vec4 vertex output) is pre-declared if the body writes it.
    const uses_gl_pos = assignsName(main.body, "gl_Position");
    if (uses_gl_pos) {
        const f32t = try func.types.intern(.{ .float = .f32 });
        var comps: [4]Value = undefined;
        for (0..4) |i| comps[i] = try func.appendInst(entry, f32t, .{ .fconst = 0 });
        try env.append(allocator, .{ .name = "gl_Position", .val = .{ .vector = .{ .comps = comps, .len = 4 } } });
    }

    // gl_PointSize (a builtin scalar float vertex output, default 1.0). Pre-declared if the
    // body writes it, so a point-primitive VS can size its points.
    const uses_gl_point_size = assignsName(main.body, "gl_PointSize");
    if (uses_gl_point_size) {
        const f32t = try func.types.intern(.{ .float = .f32 });
        const one = try func.appendInst(entry, f32t, .{ .fconst = 1.0 });
        try env.append(allocator, .{ .name = "gl_PointSize", .val = .{ .scalar = .{ .value = one, .ty = .float } } });
    }

    // gl_FragColor (GLSL ES 1.00): the implicit fragment-shader color output, a vec4 at
    // location 0. Pre-declared as a zero vec4 if the body writes it.
    const uses_gl_frag_color = assignsName(main.body, "gl_FragColor");
    if (uses_gl_frag_color) {
        const f32t = try func.types.intern(.{ .float = .f32 });
        var comps: [4]Value = undefined;
        for (0..4) |i| comps[i] = try func.appendInst(entry, f32t, .{ .fconst = 0 });
        try env.append(allocator, .{ .name = "gl_FragColor", .val = .{ .vector = .{ .comps = comps, .len = 4 } } });
    }

    // Unqualified module-scope variables that are a user struct and/or an array (e.g.
    // `LightSourceParameters lightSource[3];`): pre-declare a zero composite the body
    // then assigns into (mirrors how `out`/gl_FragColor slots are pre-declared).
    for (ast.globals) |g| {
        if (g.qualifier != .none) continue;
        if (g.struct_name == null and g.array_len == null) continue;
        const v = try defaultComposite(&l, g.ty, g.struct_name, g.array_len);
        try env.append(allocator, .{ .name = g.name, .val = v });
    }

    // `const` globals with a constant initializer (e.g. `const vec4 C = vec4(1.0);`).
    // The initializer is a constant expression. Lower it in the entry block and bind the
    // result by name so body references resolve to it (the value folds into the consumer).
    for (ast.globals) |g| {
        if (g.qualifier != .const_) continue;
        const init_expr = g.init orelse return error.ParseError; // a const must be initialized
        const v = try lowerExpr(&l, init_expr);
        try env.append(allocator, .{ .name = g.name, .val = v });
    }

    for (main.body) |stmt| _ = try lowerStmt(&l, .void, stmt);

    // Collect every output: gl_Position (BuiltIn) + each varying/out (located), or the
    // fragment color output (gl_FragColor / the single `out`).
    var outputs: std.ArrayList(ShaderOutput) = .empty;
    errdefer {
        for (outputs.items) |o| {
            allocator.free(o.comps);
            if (o.name.len > 0) allocator.free(o.name);
        }
        outputs.deinit(allocator);
    }
    if (uses_gl_pos) {
        const slot = l.lookup("gl_Position").?;
        try outputs.append(allocator, .{ .location = 0, .comps = try allocator.dupe(Value, slot.val.vector.comps[0..4]), .builtin = 0 }); // BuiltIn Position
    }
    if (uses_gl_point_size) {
        const slot = l.lookup("gl_PointSize").?;
        // A scalar builtin output (1 component). The SPIR-V emit makes it a float variable.
        try outputs.append(allocator, .{ .location = 0, .comps = try allocator.dupe(Value, &.{slot.val.scalar.value}), .builtin = 1 }); // BuiltIn PointSize
    }
    if (uses_gl_frag_color) {
        const slot = l.lookup("gl_FragColor").?;
        try outputs.append(allocator, .{ .location = 0, .comps = try allocator.dupe(Value, slot.val.vector.comps[0..4]) }); // a located color output
    }
    // Located varying / `out` outputs (a vertex shader's varyings, or a fragment shader's
    // explicit `out` color). gl_FragColor and an explicit `out` are mutually exclusive in
    // practice. Both are supported here.
    for (out_decls.items) |od| {
        const slot = l.lookup(od.name).?;
        const comps: []Value = switch (slot.val) {
            .scalar => |s| try allocator.dupe(Value, &.{s.value}),
            .vector => |vec| try allocator.dupe(Value, vec.comps[0..vec.len]),
            .matrix, .array, .structv => return error.Unsupported, // matrix/aggregate shader outputs are uncommon
        };
        // Surface the varying's source NAME. Dupe it: `od.name` borrows the AST, which is
        // arena-freed when this function returns, so the name must be owned by the LoweredShader
        // (freed in its deinit). Transform feedback maps glTransformFeedbackVaryings names to a VS
        // output location via this.
        try outputs.append(allocator, .{ .location = od.location, .comps = comps, .name = try allocator.dupe(u8, od.name) });
    }
    // Terminate the FINAL block (`l.block`), not `entry`: an inlined function call splits
    // main across multiple blocks (the inline body + a continuation exit block), so the
    // return flows through that continuation. Hardcoding `entry` here severed the inline's
    // jump-to-continuation (overwriting it with `ret`), leaving the continuation's phi
    // params - which carry a vector return value - undefined (the glmark2 light-phong FS,
    // `gl_FragColor += compute_color(...)`, produced a CompositeConstruct of undefined ids).
    func.setTerminator(l.block, .{ .ret = null }); // outputs flow through stored variables

    // Build the named-uniform-block records. Each block's members were appended to
    // `uniform_members` (they lower as default-block uniforms); locate them by name to
    // recover the block's contiguous byte range. The binding is `layout(binding=N)` when
    // given, else the block's declaration index (a distinct default so blocks that never
    // call glUniformBlockBinding do not collide on binding point 0).
    var block_records: std.ArrayList(UniformBlock) = .empty;
    errdefer {
        for (block_records.items) |b| allocator.free(b.name);
        block_records.deinit(allocator);
    }
    for (ast.uniform_blocks, 0..) |blk, bi| {
        // First find the block's tight base offset (its members are contiguous in declaration
        // order in the default block; the first member has the smallest tight offset).
        var min_off: u32 = std.math.maxInt(u32);
        for (blk.member_names) |mn| {
            for (uniform_members.items) |um| {
                if (!std.mem.eql(u8, um.name, mn)) continue;
                const off = um.offset_floats * 4;
                if (off < min_off) min_off = off;
            }
        }
        if (min_off == std.math.maxInt(u32)) min_off = 0; // empty block (no members resolved)

        // Walk the members in declaration order, laying out a std140 offset for each and
        // recording how to repack it into the shader's tight layout.
        var block_members: std.ArrayList(UniformBlockMember) = .empty;
        errdefer block_members.deinit(allocator);
        var std140_run: u32 = 0; // running std140 offset within the block
        for (blk.member_names) |mn| {
            const um = for (uniform_members.items) |cand| {
                if (std.mem.eql(u8, cand.name, mn)) break cand;
            } else continue;
            const al: u32 = @max(um.array_len, 1);
            const is_mat = um.mat_dim != 0;
            // Component count of one contiguous chunk (a matrix column, or a whole vector/scalar).
            const comps: u32 = if (is_mat) um.mat_dim else um.float_count;
            // std140 base alignment: matrices and arrays round to 16; else vecN alignment.
            const base_align: u32 = if (is_mat or al > 1) 16 else switch (comps) {
                1 => 4,
                2 => 8,
                else => 16,
            };
            const std140_off = std.mem.alignForward(u32, std140_run, base_align);
            const tight_off = um.offset_floats * 4 - min_off;
            const copy_bytes = comps * 4;
            var member: UniformBlockMember = undefined;
            if (is_mat) {
                // A matrix is an array of `mat_dim` columns, each a vecN 16-aligned; an array of
                // matrices is `al*mat_dim` such columns contiguous at a 16 stride (each column
                // already 16-sized in std140). Tight packs columns at `comps*4`.
                member = .{ .std140_offset = std140_off, .tight_offset = tight_off, .copy_bytes = copy_bytes, .unit_count = al * um.mat_dim, .std140_stride = 16, .tight_stride = copy_bytes };
                std140_run = std140_off + al * um.mat_dim * 16;
            } else if (al > 1) {
                // An array of scalars/vectors: each element occupies a 16-byte std140 stride.
                member = .{ .std140_offset = std140_off, .tight_offset = tight_off, .copy_bytes = copy_bytes, .unit_count = al, .std140_stride = 16, .tight_stride = copy_bytes };
                std140_run = std140_off + al * 16;
            } else {
                // A single scalar/vector: contiguous, no per-unit stride (vec3 = 12 bytes in both).
                member = .{ .std140_offset = std140_off, .tight_offset = tight_off, .copy_bytes = copy_bytes, .unit_count = 1, .std140_stride = copy_bytes, .tight_stride = copy_bytes };
                std140_run = std140_off + copy_bytes;
            }
            try block_members.append(allocator, member);
        }
        const std140_size = std.mem.alignForward(u32, std140_run, 16);
        const owned_name = try allocator.dupe(u8, blk.name);
        errdefer allocator.free(owned_name);
        try block_records.append(allocator, .{
            .name = owned_name,
            .binding = blk.binding orelse @as(u32, @intCast(bi)),
            .byte_offset = min_off,
            .byte_size = std140_size,
            .members = try block_members.toOwnedSlice(allocator),
        });
    }

    return .{
        .func = func,
        .interface = .{ .inputs = try inputs.toOwnedSlice(allocator), .outputs = try outputs.toOwnedSlice(allocator), .local_size = ast.local_size orelse .{ 1, 1, 1 }, .uniform_count = uniform_count, .sampler_count = @intCast(l.samplers.items.len), .uniforms = try uniform_members.toOwnedSlice(allocator), .samplers = try sampler_members.toOwnedSlice(allocator), .uniform_blocks = try block_records.toOwnedSlice(allocator) },
        .allocator = allocator,
    };
}

/// A scalar value with its GLSL type (float/int/uint/bool).
const Scalar = struct { value: Value, ty: Type };

/// A scalarized vector: its component values (length 2..4) and their GLSL component type
/// (`.float` for vecN, `.int` for ivecN, `.bool` for bvecN). Defaulting to `.float` keeps
/// every existing float-vector construction site correct without change.
const Vector = struct { comps: [4]Value, len: u8, comp_ty: Type = .float };

/// The GLSL vector type for a component type and length (the inverse of `vecCompType`).
fn vecTypeFor(comp_ty: Type, len: u8) Type {
    return switch (comp_ty) {
        .int => switch (len) {
            2 => .ivec2,
            3 => .ivec3,
            else => .ivec4,
        },
        .uint => switch (len) {
            2 => .uvec2,
            3 => .uvec3,
            else => .uvec4,
        },
        .bool => switch (len) {
            2 => .bvec2,
            3 => .bvec3,
            else => .bvec4,
        },
        else => switch (len) {
            2 => .vec2,
            3 => .vec3,
            else => .vec4,
        },
    };
}

/// The component (element) type of a GLSL vector type.
fn vecCompType(ty: Type) Type {
    return switch (ty) {
        .ivec2, .ivec3, .ivec4 => .int,
        .uvec2, .uvec3, .uvec4 => .uint,
        .bvec2, .bvec3, .bvec4 => .bool,
        else => .float,
    };
}

/// A scalarized float square matrix, column-major: comps[col*dim + row], dim in 2..4.
const Matrix = struct { comps: [16]Value, dim: u8 };

/// A fixed-length array, scalarized to its element Vals (length known at compile time).
/// The element slice is owned by the lowering's composite arena.
const Array = struct { elems: []Val };

/// A struct instance, scalarized to its field Vals in declaration order. `def` names the
/// struct type so member access can resolve a field name -> index. The field slice is
/// owned by the lowering's composite arena.
const StructVal = struct { def: []const u8, fields: []Val };

/// A lowered GLSL value: a scalar, a (float) vector, a (float) matrix, a fixed array, or a
/// user-struct instance. Arrays/structs are compile-time composites (constant-indexed),
/// matching the scalarized model: there is no runtime aggregate in the IR.
const Val = union(enum) {
    scalar: Scalar,
    vector: Vector,
    matrix: Matrix,
    array: Array,
    structv: StructVal,
};

/// Dimension of a square matrix type, or 0 for a non-matrix.
fn matDim(ty: Type) u8 {
    return switch (ty) {
        .mat2 => 2,
        .mat3 => 3,
        .mat4 => 4,
        else => 0,
    };
}

const Var = struct { name: []const u8, val: Val };

/// The enclosing loop, for `break` (jump to `exit`) and `continue` (jump to `cont`, the
/// increment/back-edge block). `len` is the live-variable count at the loop, used to
/// flatten the env onto the target block's parameters.
const LoopCtx = struct { exit: Block, cont: Block, len: usize };

/// The active inlined-call context. `exit` is the block control resumes at after the
/// inlined body. Every `return` in the body jumps there carrying the (coerced) return
/// value, and `exit`'s block params (a phi over all return paths) are the call's result.
/// `ret_ty` is the callee's declared return type (returns coerce to it).
const InlineCtx = struct { exit: Block, ret_ty: Type };

const L = struct {
    func: *Function,
    block: Block,
    env: *std.ArrayList(Var),
    allocator: std.mem.Allocator,
    /// The pipeline stage being lowered. A vertex shader has NO screen-space derivatives,
    /// so an implicit-LOD `texture2D(sampler, uv)` in a vertex shader (vertex texture fetch,
    /// e.g. terrain heightmap displacement) must lower to an EXPLICIT LOD-0 sample, not the
    /// derivative-based implicit one (which would fault / read garbage with no quad neighbours).
    stage: ShaderStage = .fragment,
    loops: std.ArrayList(LoopCtx) = .empty,
    /// Declared `uniform sampler2D` names, in declaration order. The index is the SPIR-V
    /// binding. `texture(name, uv)` resolves `name` to its binding here.
    samplers: std.ArrayList([]const u8) = .empty,
    /// Parallel to `samplers`: true when that sampler is a `samplerCube` (sampled by a vec3
    /// direction), false for a `sampler2D` (a vec2 uv). Picks the texture-lookup coord width.
    sampler_cube: std.ArrayList(bool) = .empty,
    /// Parallel to `samplers`: true when that sampler is a `sampler3D` (a vec3 coordinate, like
    /// cube but trilinear-3D at the host). Mutually exclusive with `sampler_cube`.
    sampler_3d: std.ArrayList(bool) = .empty,
    /// Parallel to `samplers`: true when that sampler is a `sampler2DArray` (a vec3 (u,v,layer),
    /// like 3D but the third coord is a raw layer index). Mutually exclusive with the others.
    sampler_2darray: std.ArrayList(bool) = .empty,
    /// Parallel to `samplers`: true when that sampler is a `sampler2DShadow` - a 2D depth image
    /// sampled with a vec3 (u, v, dref) that returns a SCALAR depth-compare result, not a vec4.
    /// Distinct from the vec3 cube/3D/array samplers (those return a vec4 texel).
    sampler_shadow: std.ArrayList(bool) = .empty,
    /// Parallel to `samplers`: true when that sampler is a `samplerCubeShadow` - a Cube depth image
    /// sampled with a vec4 (dir.xyz, dref) that returns a SCALAR depth-compare result. Also flagged
    /// in `sampler_cube` (so the emitter picks Cube dim); distinct from `sampler_shadow` (the 2D case).
    sampler_cube_shadow: std.ArrayList(bool) = .empty,
    /// Parallel to `samplers`: true when that sampler is a `sampler2DArrayShadow` - a 2D-array depth
    /// image sampled with a vec4 (uv, layer, dref) that returns a SCALAR depth-compare result. Also
    /// flagged in `sampler_2darray` (so the emitter picks 2D-Arrayed dim); distinct from the cube case.
    sampler_2darray_shadow: std.ArrayList(bool) = .empty,
    /// Module-level user functions (excluding `main`), for inlining at call sites.
    user_fns: []const parser.Function = &.{},
    /// User-declared struct definitions, for member-name -> field-index resolution and
    /// struct construction / default values.
    structs: []const parser.StructDef = &.{},
    /// Arena for composite Vals (array/struct element slices). Freed in bulk by the caller.
    comp_arena: std.mem.Allocator,
    /// When lowering an inlined function body, the active inline context: each `return`
    /// jumps to `exit` (carrying the value as edge args) instead of terminating the call
    /// site's block, so conditional/early returns merge into a phi at the exit. Null at top
    /// level. Nested inlines save/restore it.
    inline_ctx: ?InlineCtx = null,
    /// Recursion guard: how deep we are in nested function inlining.
    inline_depth: u32 = 0,

    /// The struct definition named `name`, or null.
    fn structDef(self: *L, name: []const u8) ?parser.StructDef {
        for (self.structs) |s| if (std.mem.eql(u8, s.name, name)) return s;
        return null;
    }

    /// Binding index of sampler `name`, or null if it is not a declared sampler.
    fn samplerBinding(self: *L, name: []const u8) ?u32 {
        for (self.samplers.items, 0..) |s, i| if (std.mem.eql(u8, s, name)) return @intCast(i);
        return null;
    }

    /// Whether sampler at declaration index `binding` is a cube sampler (a vec3 direction).
    fn samplerIsCube(self: *L, binding: u32) bool {
        return binding < self.sampler_cube.items.len and self.sampler_cube.items[binding];
    }

    /// Whether sampler at declaration index `binding` is a 3D sampler (a vec3 coordinate).
    fn samplerIs3d(self: *L, binding: u32) bool {
        return binding < self.sampler_3d.items.len and self.sampler_3d.items[binding];
    }

    /// Whether sampler at declaration index `binding` is a 2D-array sampler (a vec3 (u,v,layer)).
    fn samplerIs2dArray(self: *L, binding: u32) bool {
        return binding < self.sampler_2darray.items.len and self.sampler_2darray.items[binding];
    }

    /// Whether sampler `binding` takes a vec3 coordinate (cube OR 3D OR 2D-array). Shadow samplers
    /// also take a vec3 but are handled separately (they return a scalar), so they are NOT included.
    fn samplerIsVec3(self: *L, binding: u32) bool {
        return self.samplerIsCube(binding) or self.samplerIs3d(binding) or self.samplerIs2dArray(binding);
    }

    /// Whether sampler at declaration index `binding` is a `sampler2DShadow` (vec3 (u,v,dref) in,
    /// scalar depth-compare result out).
    fn samplerIsShadow(self: *L, binding: u32) bool {
        return binding < self.sampler_shadow.items.len and self.sampler_shadow.items[binding];
    }

    /// Whether sampler at declaration index `binding` is a `samplerCubeShadow` (vec4 (dir, dref) in,
    /// scalar depth-compare result out).
    fn samplerIsCubeShadow(self: *L, binding: u32) bool {
        return binding < self.sampler_cube_shadow.items.len and self.sampler_cube_shadow.items[binding];
    }

    /// Whether sampler at declaration index `binding` is a `sampler2DArrayShadow` (vec4 (uv, layer,
    /// dref) in, scalar depth-compare result out).
    fn samplerIs2dArrayShadow(self: *L, binding: u32) bool {
        return binding < self.sampler_2darray_shadow.items.len and self.sampler_2darray_shadow.items[binding];
    }

    fn lookup(self: *L, name: []const u8) ?*Var {
        var i = self.env.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.env.items[i].name, name)) return &self.env.items[i];
        }
        return null;
    }
};

/// Resolve a storage qualifier to its `in_`/`out_`/`uniform` role for the given stage.
/// GLSL ES 1.00 `attribute` is a vertex input. `varying` is the VS->FS link: an `out`
/// in the vertex shader and an `in` in the fragment shader.
fn effectiveQualifier(q: parser.Qualifier, stage: ShaderStage) parser.Qualifier {
    return switch (q) {
        .attribute => .in_,
        .varying => if (stage == .vertex) .out_ else .in_,
        else => q,
    };
}

/// Component count of a vector type, or 0 for a scalar. Integer and bool vectors are
/// scalarized like float vectors (their components are lowered as f32 in this subset).
fn vecLen(ty: Type) u8 {
    return switch (ty) {
        .vec2, .ivec2, .uvec2, .bvec2 => 2,
        .vec3, .ivec3, .uvec3, .bvec3 => 3,
        .vec4, .ivec4, .uvec4, .bvec4 => 4,
        else => 0,
    };
}

fn irType(func: *Function, ty: Type) Error!ir.types.Type {
    return switch (ty) {
        .float => func.types.intern(.{ .float = .f32 }),
        .int => func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }),
        .uint => func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } }),
        .bool => func.types.intern(.bool),
        .void, .vec2, .vec3, .vec4, .ivec2, .ivec3, .ivec4, .uvec2, .uvec3, .uvec4, .bvec2, .bvec3, .bvec4, .mat2, .mat3, .mat4, .sampler2d, .sampler_cube, .sampler3d, .sampler2darray, .sampler2dshadow, .samplercubeshadow, .sampler2darrayshadow => error.Unsupported,
    };
}

fn f32Type(l: *L) Error!ir.types.Type {
    return l.func.types.intern(.{ .float = .f32 });
}

fn lowerFunction(allocator: std.mem.Allocator, f: parser.Function, user_fns: []const parser.Function, structs: []const parser.StructDef) Error!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const entry = try func.appendBlock();

    var fn_arena = std.heap.ArenaAllocator.init(allocator);
    defer fn_arena.deinit();

    var env: std.ArrayList(Var) = .empty;
    defer env.deinit(allocator);
    for (f.params) |param| {
        const n = vecLen(param.ty);
        const md = matDim(param.ty);
        if (md != 0) {
            // A matrix parameter is scalarized to dim*dim float parameters (column-major).
            const f32t = try func.types.intern(.{ .float = .f32 });
            var comps: [16]Value = undefined;
            for (0..@as(usize, md) * md) |i| comps[i] = try func.appendBlockParam(entry, f32t);
            try env.append(allocator, .{ .name = param.name, .val = .{ .matrix = .{ .comps = comps, .dim = md } } });
        } else if (n == 0) {
            const pv = try func.appendBlockParam(entry, try irType(&func, param.ty));
            try env.append(allocator, .{ .name = param.name, .val = .{ .scalar = .{ .value = pv, .ty = param.ty } } });
        } else {
            // A vector parameter is scalarized to one parameter per component, typed by the
            // vector's component type (int for ivec, uint for uvec, else float).
            const comp_ty = vecCompType(param.ty);
            const ct = try irType(&func, comp_ty);
            var comps: [4]Value = undefined;
            for (0..n) |i| comps[i] = try func.appendBlockParam(entry, ct);
            try env.append(allocator, .{ .name = param.name, .val = .{ .vector = .{ .comps = comps, .len = n, .comp_ty = comp_ty } } });
        }
    }

    var l = L{ .func = &func, .block = entry, .env = &env, .allocator = allocator, .comp_arena = fn_arena.allocator(), .user_fns = user_fns, .structs = structs };
    defer l.loops.deinit(allocator);
    defer l.samplers.deinit(allocator);
    defer l.sampler_cube.deinit(allocator);
    defer l.sampler_3d.deinit(allocator);
    defer l.sampler_2darray.deinit(allocator);
    defer l.sampler_shadow.deinit(allocator);
    defer l.sampler_cube_shadow.deinit(allocator);
    defer l.sampler_2darray_shadow.deinit(allocator);
    var returned = false;
    for (f.body) |stmt| {
        if (try lowerStmt(&l, f.ret, stmt)) returned = true;
    }
    if (!returned) func.setTerminator(l.block, .{ .ret = null });
    return func;
}

/// Lower a statement. Returns true if it terminated the block.
/// Lower a statement, then tag every instruction it emitted (that is not already tagged by a
/// nested statement) with its source line, so a debug-info pass can build a `.debug_line`
/// address->line table. Innermost statement wins (nested ones tag first).
fn lowerStmt(l: *L, ret_ty: Type, stmt: parser.Stmt) Error!bool {
    const before = l.func.instCount();
    const returned = try lowerStmtInner(l, ret_ty, stmt);
    if (stmt.line != 0) {
        var i = before;
        while (i < l.func.instCount()) : (i += 1) {
            const inst: ir.function.Inst = @enumFromInt(@as(u32, @intCast(i)));
            if (!hasLineAttr(l.func, inst)) {
                try l.func.addAttr(.{ .inst = inst }, .{ .custom = .{ .namespace = "debug", .key = "line", .value = .{ .int = stmt.line } } });
            }
        }
    }
    return returned;
}

fn hasLineAttr(func: *const ir.function.Function, inst: ir.function.Inst) bool {
    var it = func.attributesOf(.{ .inst = inst });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "debug") and std.mem.eql(u8, c.key, "line")) return true,
        else => {},
    };
    return false;
}

fn lowerStmtInner(l: *L, ret_ty: Type, stmt: parser.Stmt) Error!bool {
    switch (stmt.kind) {
        .ret => |maybe| {
            // Inside an inlined function body: a `return` jumps to the inline's exit block,
            // carrying the (coerced) return value as edge args. The exit block's params are
            // the SSA-merged return value (a phi over every return path), so conditional /
            // early returns short-circuit correctly (the value is NOT a single overwritten
            // compile-time slot).
            if (l.inline_ctx) |*ctx| {
                if (maybe) |e| {
                    const v = try coerceVal(l, try lowerExpr(l, e), ctx.ret_ty);
                    var vals = [_]Val{v};
                    const args = try flattenVals(l, &vals);
                    defer l.allocator.free(args);
                    try l.func.setJump(l.block, ctx.exit, args);
                } else {
                    try l.func.setJump(l.block, ctx.exit, &.{});
                }
                return true; // this path returned. The inline continues at the exit block
            }
            if (maybe) |e| {
                const v = try lowerExpr(l, e);
                if (v != .scalar) return error.Unsupported; // vector return needs multi-value IR
                const c = try coerce(l, v.scalar, ret_ty);
                l.func.setTerminator(l.block, .{ .ret = c.value });
            } else l.func.setTerminator(l.block, .{ .ret = null });
            return true;
        },
        .decl => |d| {
            if (d.value) |e| {
                const v = try lowerExpr(l, e);
                // A struct/array local takes the composite as-is. A scalar/vector/matrix
                // coerces numerically.
                const stored = if (d.struct_name != null or d.array_len != null) v else try coerceVal(l, v, d.ty);
                try l.env.append(l.allocator, .{ .name = d.name, .val = stored });
            } else if (d.struct_name != null or d.array_len != null) {
                try l.env.append(l.allocator, .{ .name = d.name, .val = try defaultComposite(l, d.ty, d.struct_name, d.array_len) });
            } else {
                try l.env.append(l.allocator, .{ .name = d.name, .val = try zeroVal(l, d.ty) });
            }
            return false;
        },
        .assign => |a| {
            const v = try lowerExpr(l, a.value);
            const slot = l.lookup(a.name) orelse return error.UndefinedName;
            const target_ty = valType(slot.val);
            slot.val = try coerceVal(l, v, target_ty);
            return false;
        },
        .expr => |e| {
            _ = try lowerExpr(l, e);
            return false;
        },
        .break_ => {
            if (l.loops.items.len == 0) return error.Unsupported;
            const lc = l.loops.items[l.loops.items.len - 1];
            const vals = try l.allocator.alloc(Val, lc.len);
            defer l.allocator.free(vals);
            for (0..lc.len) |i| vals[i] = l.env.items[i].val;
            const args = try flattenVals(l, vals);
            defer l.allocator.free(args);
            try l.func.setJump(l.block, lc.exit, args);
            return true;
        },
        .continue_ => {
            if (l.loops.items.len == 0) return error.Unsupported;
            const lc = l.loops.items[l.loops.items.len - 1];
            const vals = try l.allocator.alloc(Val, lc.len);
            defer l.allocator.free(vals);
            for (0..lc.len) |i| vals[i] = l.env.items[i].val;
            const args = try flattenVals(l, vals);
            defer l.allocator.free(args);
            try l.func.setJump(l.block, lc.cont, args);
            return true;
        },
        .discard_ => {
            // Fragment kill: terminate the block, tagged so the SPIR-V emitter emits OpKill.
            l.func.setTerminator(l.block, .{ .ret = null });
            try l.func.addAttr(.{ .block = l.block }, .{ .custom = .{ .namespace = "cf", .key = "discard", .value = .{ .int = 0 } } });
            return true;
        },
        .swizzle_assign => |sa| {
            const slot = l.lookup(sa.name) orelse return error.UndefinedName;
            // `s.field = value` on a struct is a member write (the parser produces the same
            // node shape as a vector swizzle write; dispatch on the lowered value's kind).
            if (slot.val == .structv) {
                const idx = structFieldIndex(l, slot.val.structv.def, sa.field) orelse return error.UndefinedName;
                const v = try lowerExpr(l, sa.value);
                slot.val.structv.fields[idx] = try coerceVal(l, v, valType(slot.val.structv.fields[idx]));
                return false;
            }
            if (slot.val != .vector) return error.BadSwizzle;
            var vec = slot.val.vector;
            const v = try lowerExpr(l, sa.value);
            if (sa.field.len == 1) {
                if (v != .scalar) return error.TypeMismatch;
                const idx = swizzleIndex(sa.field[0]) orelse return error.BadSwizzle;
                if (idx >= vec.len) return error.BadSwizzle;
                vec.comps[idx] = (try coerce(l, v.scalar, vec.comp_ty)).value;
            } else {
                if (v != .vector or v.vector.len != sa.field.len) return error.TypeMismatch;
                for (sa.field, 0..) |ch, i| {
                    const idx = swizzleIndex(ch) orelse return error.BadSwizzle;
                    if (idx >= vec.len) return error.BadSwizzle;
                    vec.comps[idx] = v.vector.comps[i];
                }
            }
            slot.val = .{ .vector = vec };
            return false;
        },
        .store => |st| {
            try lowerStore(l, st.target, st.value);
            return false;
        },
        .if_ => |iff| return lowerIf(l, ret_ty, iff.cond, iff.then, iff.@"else"),
        .for_ => |f| return lowerFor(l, ret_ty, f.init, f.cond, f.incr, f.body),
    }
}

/// Lower `for (init, cond, incr) body` (a `while` is the same with empty init/incr) as a
/// header/body/exit loop. The header takes a block parameter (loop phi) for every live
/// variable: the preheader edge passes the initial value, the body's back-edge the
/// updated value.
fn lowerFor(l: *L, ret_ty: Type, init: []const parser.Stmt, cond_e: ?*parser.Expr, incr: []const parser.Stmt, body: []const parser.Stmt) Error!bool {
    // A constant-bound counting loop is fully unrolled so any array index by the loop
    // variable folds to a compile-time constant (the scalarized model has no runtime
    // aggregate). Falls through to the runtime-loop lowering when the loop is not a simple
    // constant-bound counter or its body breaks/continues.
    if (try tryUnrollFor(l, ret_ty, init, cond_e, incr, body)) |terminated| return terminated;

    for (init) |s| _ = try lowerStmt(l, ret_ty, s);
    const cond_expr = cond_e orelse return error.Unsupported; // an unconditional loop needs break

    const len = l.env.items.len;
    const snapshot = try l.allocator.alloc(Val, len);
    defer l.allocator.free(snapshot);
    for (0..len) |i| snapshot[i] = l.env.items[i].val;

    const preheader = l.block;
    const header = try l.func.appendBlock();
    const body_b = try l.func.appendBlock();
    const continue_b = try l.func.appendBlock(); // holds the increment + back-edge
    const exit_b = try l.func.appendBlock();

    // One header block parameter (loop phi) per live variable.
    const header_vals = try l.allocator.alloc(Val, len);
    defer l.allocator.free(header_vals);
    for (0..len) |i| header_vals[i] = try headerParam(l, header, snapshot[i]);

    // If the body has a `break`, the exit block needs block parameters (phis) merging the
    // normal exit (header's false edge) with each break edge.
    const has_brk = hasBreak(body);
    var exit_vals: []Val = &.{};
    if (has_brk) {
        exit_vals = try l.allocator.alloc(Val, len);
        for (0..len) |i| exit_vals[i] = try headerParam(l, exit_b, snapshot[i]);
    }
    defer if (has_brk) l.allocator.free(exit_vals);

    // The continue block always takes phis for the loop variables: the body's fall-through
    // edge (and each `continue`) carries values in, so the increment and back-edge read
    // params rather than values defined in an earlier block. This routes everything through
    // the edge-move path the backends handle reliably. A bare cross-block reference from
    // the continue block to the body's end miscompiles in the native regalloc.
    const cont_vals = try l.allocator.alloc(Val, len);
    defer l.allocator.free(cont_vals);
    for (0..len) |i| cont_vals[i] = try headerParam(l, continue_b, snapshot[i]);

    // preheader -> header carrying the initial values.
    const init_args = try flattenVals(l, snapshot);
    defer l.allocator.free(init_args);
    try l.func.setJump(preheader, header, init_args);

    // Evaluate the condition against the header phis.
    for (0..len) |i| l.env.items[i].val = header_vals[i];
    l.block = header;
    const cond = try lowerExpr(l, cond_expr);
    if (cond != .scalar or cond.scalar.ty != .bool) return error.Unsupported;
    // The false edge carries the header phis to the exit params when break is present.
    var exit_edge: []Value = &.{};
    if (has_brk) exit_edge = try flattenVals(l, header_vals);
    defer if (has_brk) l.allocator.free(exit_edge);
    try l.func.appendIf(header, cond.scalar.value, .{ .target = body_b, .args = &.{} }, .{ .target = exit_b, .args = exit_edge });
    // Record the merge (exit) and the dedicated continue block for OpLoopMerge and
    // structured block ordering.
    try l.func.addAttr(.{ .block = header }, .{ .custom = .{ .namespace = "cf", .key = "merge", .value = .{ .int = @intFromEnum(exit_b) } } });
    try l.func.addAttr(.{ .block = header }, .{ .custom = .{ .namespace = "cf", .key = "continue", .value = .{ .int = @intFromEnum(continue_b) } } });

    try l.loops.append(l.allocator, .{ .exit = exit_b, .cont = continue_b, .len = len });

    // Body. A nested if/loop leaves l.block at the body's end block, which flows to the
    // continue block. Without `continue` the body's values dominate it, so no phi needed.
    l.block = body_b;
    var terminated = false;
    for (body) |s| if (try lowerStmt(l, ret_ty, s)) {
        terminated = true;
    };
    if (!terminated) {
        // The fall-through edge carries the body's values to the continue phis.
        const bvals = try l.allocator.alloc(Val, len);
        defer l.allocator.free(bvals);
        for (0..len) |i| bvals[i] = l.env.items[i].val;
        const cont_edge = try flattenVals(l, bvals);
        defer l.allocator.free(cont_edge);
        try l.func.setJump(l.block, continue_b, cont_edge);
    }

    // Continue block: the increment (against the continue phis), then the back-edge.
    l.block = continue_b;
    for (0..len) |i| l.env.items[i].val = cont_vals[i];
    for (incr) |s| _ = try lowerStmt(l, ret_ty, s);
    const cont_end = l.block;
    const post = try l.allocator.alloc(Val, len);
    defer l.allocator.free(post);
    for (0..len) |i| post[i] = l.env.items[i].val;
    l.env.shrinkRetainingCapacity(len);
    const back_args = try flattenVals(l, post);
    defer l.allocator.free(back_args);
    try l.func.setJump(cont_end, header, back_args);

    _ = l.loops.pop();

    // After the loop, the variables hold the exit phis (with break) or the header phis.
    for (0..len) |i| l.env.items[i].val = if (has_brk) exit_vals[i] else header_vals[i];
    l.block = exit_b;
    return false;
}

/// Attempt to fully unroll a constant-bound counting `for` loop:
///   `for (int i = C0; i </<= C1; i++ | i += K) { body }`
/// Returns `true`/`false` (whether the unrolled body terminated the block) when the loop
/// matched and was unrolled, or `null` when it did not match (caller does the runtime loop).
/// A `break`/`continue` in the body disqualifies unrolling (runtime loop handles those).
fn tryUnrollFor(l: *L, ret_ty: Type, init: []const parser.Stmt, cond_e: ?*parser.Expr, incr: []const parser.Stmt, body: []const parser.Stmt) Error!?bool {
    const cond_expr = cond_e orelse return null;
    if (init.len != 1 or incr.len != 1) return null;
    if (hasBreak(body) or hasContinue(body)) return null;

    // init: `int i = C0;` (a typed decl with a constant initializer).
    const d = switch (init[0].kind) {
        .decl => |d| d,
        else => return null,
    };
    if (d.array_len != null or d.struct_name != null) return null;
    if (!isInt(d.ty)) return null;
    const var_name = d.name;
    const start = (try constInt(l, d.value orelse return null)) orelse return null;

    // cond: `i < C1` or `i <= C1` (loop variable on the left, constant on the right).
    const cb = switch (cond_expr.*) {
        .binary => |b| b,
        else => return null,
    };
    if (!(cb.op == .lt or cb.op == .le)) return null;
    if (cb.lhs.* != .ident or !std.mem.eql(u8, cb.lhs.ident, var_name)) return null;
    const limit = (try constInt(l, cb.rhs)) orelse return null;

    // incr: `i++`, `i += K`, or `i = i + K` (desugared to `assign i = i + K`).
    const step = stepOf(incr[0], var_name) orelse return null;
    if (step == 0) return null;

    // Bound the unroll count (guard against a runaway / malformed loop).
    const lv_idx = l.env.items.len;
    try l.env.append(l.allocator, .{ .name = var_name, .val = .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, d.ty), .{ .iconst = start }), .ty = d.ty } } });

    var i: i64 = start;
    var iters: u32 = 0;
    var terminated = false;
    while (if (cb.op == .lt) i < limit else i <= limit) : (i += step) {
        iters += 1;
        if (iters > 4096) return error.Unsupported;
        // Rebind the loop variable to this iteration's constant.
        l.env.items[lv_idx].val = .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, d.ty), .{ .iconst = i }), .ty = d.ty } };
        for (body) |s| if (try lowerStmt(l, ret_ty, s)) {
            terminated = true;
        };
        if (terminated) break;
        // Drop this iteration's body-local declarations, keep the loop var + outer mutations.
        l.env.shrinkRetainingCapacity(lv_idx + 1);
    }
    // Pop the loop variable. Outer-variable mutations remain in the env.
    if (!terminated) l.env.shrinkRetainingCapacity(lv_idx);
    return terminated;
}

/// The increment step of a counting loop's update statement, or null if it is not a simple
/// `i++` / `i += K` / `i = i + K` on `var_name`.
fn stepOf(s: parser.Stmt, var_name: []const u8) ?i64 {
    const a = switch (s.kind) {
        .assign => |a| a,
        .expr => return null,
        else => return null,
    };
    if (!std.mem.eql(u8, a.name, var_name)) return null;
    // The parser desugars `i++`/`i += K` to `i = i + <K>`. Match `i + K` / `K + i` / `i - K`.
    const b = switch (a.value.*) {
        .binary => |b| b,
        else => return null,
    };
    if (b.op == .add) {
        if (b.lhs.* == .ident and std.mem.eql(u8, b.lhs.ident, var_name)) {
            if (b.rhs.* == .int_lit) return b.rhs.int_lit;
        }
        if (b.rhs.* == .ident and std.mem.eql(u8, b.rhs.ident, var_name)) {
            if (b.lhs.* == .int_lit) return b.lhs.int_lit;
        }
    } else if (b.op == .sub) {
        if (b.lhs.* == .ident and std.mem.eql(u8, b.lhs.ident, var_name) and b.rhs.* == .int_lit) return -b.rhs.int_lit;
    }
    return null;
}

/// Whether a loop body contains a `continue` for this loop (recurses into `if`, not nested
/// loops).
fn hasContinue(body: []const parser.Stmt) bool {
    for (body) |stmt| switch (stmt.kind) {
        .continue_ => return true,
        .if_ => |iff| if (hasContinue(iff.then) or hasContinue(iff.@"else")) return true,
        else => {},
    };
    return false;
}

/// Whether a loop body contains a `break` for this loop. Recurses into `if` branches but
/// not into nested loops, whose `break` targets them.
fn hasBreak(body: []const parser.Stmt) bool {
    for (body) |stmt| switch (stmt.kind) {
        .break_ => return true,
        .if_ => |iff| if (hasBreak(iff.then) or hasBreak(iff.@"else")) return true,
        else => {},
    };
    return false;
}

/// Create header block parameter(s) for a value (one per scalar, N per vector) and return
/// the parameter-backed value.
fn headerParam(l: *L, header: Block, val: Val) Error!Val {
    switch (val) {
        .array, .structv => return error.Unsupported, // a composite live across a runtime loop is unsupported
        .scalar => |s| return .{ .scalar = .{ .value = try l.func.appendBlockParam(header, try irType(l.func, s.ty)), .ty = s.ty } },
        .vector => |vec| {
            // The phi param type must match the vector's component type: an integer vector
            // loop variable needs int params, not f32 (which would lose its type and read
            // garbage when the loop does not execute).
            const ct = try irType(l.func, vec.comp_ty);
            var comps: [4]Value = undefined;
            for (0..vec.len) |i| comps[i] = try l.func.appendBlockParam(header, ct);
            return .{ .vector = .{ .comps = comps, .len = vec.len, .comp_ty = vec.comp_ty } };
        },
        .matrix => |m| {
            const f32t = try f32Type(l);
            var comps: [16]Value = undefined;
            for (0..@as(usize, m.dim) * m.dim) |i| comps[i] = try l.func.appendBlockParam(header, f32t);
            return .{ .matrix = .{ .comps = comps, .dim = m.dim } };
        },
    }
}

/// Flatten a list of values to their scalar component values (a vector/matrix contributes
/// its lanes), matching the order `headerParam` creates block parameters.
fn flattenVals(l: *L, vals: []const Val) Error![]Value {
    var out: std.ArrayList(Value) = .empty;
    errdefer out.deinit(l.allocator);
    for (vals) |v| switch (v) {
        .scalar => |s| try out.append(l.allocator, s.value),
        .vector => |vec| for (0..vec.len) |i| try out.append(l.allocator, vec.comps[i]),
        .matrix => |m| for (0..@as(usize, m.dim) * m.dim) |i| try out.append(l.allocator, m.comps[i]),
        .array, .structv => return error.Unsupported, // a composite carried across a runtime loop is unsupported
    };
    return out.toOwnedSlice(l.allocator);
}

/// Lower `if (cond) then [else else]` as a diamond CFG: an `if` branches to then/else
/// blocks that re-converge at a continuation block. Variables modified differently in the
/// two branches become block parameters (phis) of the continuation, passed as edge args.
fn lowerIf(l: *L, ret_ty: Type, cond_e: *parser.Expr, then_body: []const parser.Stmt, else_body: []const parser.Stmt) Error!bool {
    const cond = try lowerExpr(l, cond_e);
    if (cond != .scalar or cond.scalar.ty != .bool) return error.Unsupported;
    const len = l.env.items.len;

    const snapshot = try l.allocator.alloc(Val, len);
    defer l.allocator.free(snapshot);
    for (0..len) |i| snapshot[i] = l.env.items[i].val;

    const then_b = try l.func.appendBlock();
    const else_b = try l.func.appendBlock();
    const cont = try l.func.appendBlock();
    const head = l.block;
    try l.func.appendIf(head, cond.scalar.value, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    // Record the structured merge so the SPIR-V emitter can emit OpSelectionMerge and
    // order blocks correctly even when branches contain nested control flow.
    try l.func.addAttr(.{ .block = head }, .{ .custom = .{ .namespace = "cf", .key = "merge", .value = .{ .int = @intFromEnum(cont) } } });

    // then branch (a nested if/loop leaves l.block at the branch's actual end block)
    l.block = then_b;
    var then_term = false;
    for (then_body) |s| if (try lowerStmt(l, ret_ty, s)) {
        then_term = true;
    };
    const then_end = l.block;
    const then_vals = try l.allocator.alloc(Val, len);
    defer l.allocator.free(then_vals);
    for (0..len) |i| then_vals[i] = l.env.items[i].val;

    // restore env, lower else branch
    l.env.shrinkRetainingCapacity(len);
    for (0..len) |i| l.env.items[i].val = snapshot[i];
    l.block = else_b;
    var else_term = false;
    for (else_body) |s| if (try lowerStmt(l, ret_ty, s)) {
        else_term = true;
    };
    const else_end = l.block;
    const else_vals = try l.allocator.alloc(Val, len);
    defer l.allocator.free(else_vals);
    for (0..len) |i| else_vals[i] = l.env.items[i].val;
    l.env.shrinkRetainingCapacity(len);

    // If one branch returns, only the other reaches the continuation (no phi needed: its
    // values dominate the continuation). If both return, the whole `if` terminates.
    if (then_term and else_term) return true;
    if (then_term or else_term) {
        const live = if (then_term) else_vals else then_vals;
        const live_end = if (then_term) else_end else then_end;
        for (0..len) |i| l.env.items[i].val = live[i];
        try l.func.setJump(live_end, cont, &.{});
        l.block = cont;
        return false;
    }

    // Merge: variables that differ between the branches become continuation phis.
    var then_args: std.ArrayList(Value) = .empty;
    defer then_args.deinit(l.allocator);
    var else_args: std.ArrayList(Value) = .empty;
    defer else_args.deinit(l.allocator);
    for (0..len) |i| l.env.items[i].val = try mergeVal(l, then_vals[i], else_vals[i], cont, &then_args, &else_args);
    try l.func.setJump(then_end, cont, then_args.items);
    try l.func.setJump(else_end, cont, else_args.items);
    l.block = cont;
    return false;
}

fn mergeVal(l: *L, then_v: Val, else_v: Val, cont: Block, then_args: *std.ArrayList(Value), else_args: *std.ArrayList(Value)) Error!Val {
    // Composites (array/struct) flowing through an `if`: merge element-by-element so a
    // member modified in one branch becomes a phi, an unchanged one passes through.
    if (then_v == .array and else_v == .array and then_v.array.elems.len == else_v.array.elems.len) {
        const elems = try l.comp_arena.alloc(Val, then_v.array.elems.len);
        for (then_v.array.elems, else_v.array.elems, 0..) |tv, ev, i| elems[i] = try mergeVal(l, tv, ev, cont, then_args, else_args);
        return .{ .array = .{ .elems = elems } };
    }
    if (then_v == .structv and else_v == .structv and then_v.structv.fields.len == else_v.structv.fields.len) {
        const fields = try l.comp_arena.alloc(Val, then_v.structv.fields.len);
        for (then_v.structv.fields, else_v.structv.fields, 0..) |tv, ev, i| fields[i] = try mergeVal(l, tv, ev, cont, then_args, else_args);
        return .{ .structv = .{ .def = then_v.structv.def, .fields = fields } };
    }
    if (then_v == .scalar and else_v == .scalar) {
        const ts = then_v.scalar;
        const es = else_v.scalar;
        if (ts.value == es.value and ts.ty == es.ty) return then_v;
        const param = try l.func.appendBlockParam(cont, try irType(l.func, ts.ty));
        try then_args.append(l.allocator, ts.value);
        try else_args.append(l.allocator, es.value);
        return .{ .scalar = .{ .value = param, .ty = ts.ty } };
    }
    if (then_v == .vector and else_v == .vector and then_v.vector.len == else_v.vector.len) {
        const tv = then_v.vector;
        const ev = else_v.vector;
        var same = true;
        for (0..tv.len) |i| if (tv.comps[i] != ev.comps[i]) {
            same = false;
        };
        if (same) return then_v;
        const ct = try irType(l.func, tv.comp_ty);
        var out: [4]Value = undefined;
        for (0..tv.len) |i| {
            out[i] = try l.func.appendBlockParam(cont, ct);
            try then_args.append(l.allocator, tv.comps[i]);
            try else_args.append(l.allocator, ev.comps[i]);
        }
        return .{ .vector = .{ .comps = out, .len = tv.len, .comp_ty = tv.comp_ty } };
    }
    // Matrices (e.g. an unchanged uniform mat4 live across the branch). A matrix is rarely
    // reassigned inside one branch. If its components are identical, no phi is needed. A
    // genuinely-divergent matrix becomes per-component phis (column-major).
    if (then_v == .matrix and else_v == .matrix and then_v.matrix.dim == else_v.matrix.dim) {
        const tm = then_v.matrix;
        const em = else_v.matrix;
        const fc: usize = @as(usize, tm.dim) * tm.dim;
        var same = true;
        for (0..fc) |i| if (tm.comps[i] != em.comps[i]) {
            same = false;
        };
        if (same) return then_v;
        const f32t = try f32Type(l);
        var out: [16]Value = undefined;
        for (0..fc) |i| {
            out[i] = try l.func.appendBlockParam(cont, f32t);
            try then_args.append(l.allocator, tm.comps[i]);
            try else_args.append(l.allocator, em.comps[i]);
        }
        return .{ .matrix = .{ .comps = out, .dim = tm.dim } };
    }
    return error.TypeMismatch;
}

/// GLSL type a Val represents (for assignment-target coercion).
fn valType(v: Val) Type {
    return switch (v) {
        .scalar => |s| s.ty,
        .vector => |vec| vecTypeFor(vec.comp_ty, vec.len),
        .matrix => |m| switch (m.dim) {
            2 => .mat2,
            3 => .mat3,
            else => .mat4,
        },
        // Composites have no scalar GLSL Type. `coerceVal` passes them through unchanged.
        .array, .structv => .void,
    };
}

fn lowerExpr(l: *L, e: *parser.Expr) Error!Val {
    switch (e.*) {
        .float_lit => |v| return .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, .float), .{ .fconst = v }), .ty = .float } },
        .int_lit => |v| return .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, .int), .{ .iconst = v }), .ty = .int } },
        .bool_lit => |v| return .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .iconst = if (v) 1 else 0 }), .ty = .bool } },
        .ident => |name| {
            const slot = l.lookup(name) orelse return error.UndefinedName;
            return slot.val;
        },
        .unary => |u| switch (u.op) {
            .neg => return negate(l, try lowerExpr(l, u.operand)),
            .not => {
                const a = try lowerExpr(l, u.operand);
                if (a != .scalar or a.scalar.ty != .bool) return error.Unsupported;
                const t = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .iconst = 1 });
                return .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .arith = .{ .op = .bit_xor, .lhs = a.scalar.value, .rhs = t } }), .ty = .bool } };
            },
            .bit_not => {
                // ~x is x XOR all-ones, integer only.
                const a = try lowerExpr(l, u.operand);
                if (a != .scalar or !isInt(a.scalar.ty)) return error.Unsupported;
                const it = try irType(l.func, a.scalar.ty);
                const ones = try l.func.appendInst(l.block, it, .{ .iconst = -1 });
                return .{ .scalar = .{ .value = try l.func.appendInst(l.block, it, .{ .arith = .{ .op = .bit_xor, .lhs = a.scalar.value, .rhs = ones } }), .ty = a.scalar.ty } };
            },
        },
        .binary => |b| return lowerBinary(l, b.op, b.lhs, b.rhs),
        .call => |c| return lowerCall(l, c.name, c.args),
        .swizzle => |s| return lowerSwizzleOrMember(l, s.value, s.field),
        .ternary => |t| return lowerTernary(l, t.cond, t.then, t.@"else"),
        .index => |ix| {
            const base = try lowerExpr(l, ix.value);
            const idx = try constIndex(l, ix.index);
            switch (base) {
                .array => |a| {
                    if (idx >= a.elems.len) return error.TypeMismatch;
                    return a.elems[idx];
                },
                // `vec[i]` accesses a component (constant index in the scalarized model).
                .vector => |vec| {
                    if (idx >= vec.len) return error.TypeMismatch;
                    return .{ .scalar = .{ .value = vec.comps[idx], .ty = vec.comp_ty } };
                },
                // `mat[i]` returns column i as a vector (column-major storage).
                .matrix => |m| {
                    if (idx >= m.dim) return error.TypeMismatch;
                    var comps: [4]Value = undefined;
                    for (0..m.dim) |row| comps[row] = m.comps[idx * m.dim + row];
                    return .{ .vector = .{ .comps = comps, .len = m.dim, .comp_ty = .float } };
                },
                else => return error.Unsupported,
            }
        },
        .struct_ctor => |sc| return lowerStructCtor(l, sc.name, sc.args),
    }
}

/// Evaluate a compile-time-constant integer index (a literal, a bound loop variable, or a
/// constant-foldable arithmetic of them). Loop variables are constants here because
/// constant-bound `for` loops are fully unrolled before any array index is reached.
fn constIndex(l: *L, e: *parser.Expr) Error!usize {
    const v = try constInt(l, e) orelse return error.Unsupported;
    if (v < 0) return error.TypeMismatch;
    return @intCast(v);
}

/// Best-effort constant integer fold of an expression. Returns null if not constant.
/// Resolves an identifier bound to a constant scalar (an unrolled loop variable's `iconst`).
fn constInt(l: *L, e: *parser.Expr) Error!?i64 {
    switch (e.*) {
        .int_lit => |v| return v,
        .float_lit => |v| return @intFromFloat(v),
        .unary => |u| if (u.op == .neg) {
            if (try constInt(l, u.operand)) |x| return -x;
            return null;
        } else return null,
        .binary => |b| {
            const a = (try constInt(l, b.lhs)) orelse return null;
            const c = (try constInt(l, b.rhs)) orelse return null;
            return switch (b.op) {
                .add => a + c,
                .sub => a - c,
                .mul => a * c,
                .div => if (c == 0) null else @divTrunc(a, c),
                .mod => if (c == 0) null else @rem(a, c),
                else => null,
            };
        },
        .ident => |name| {
            const slot = l.lookup(name) orelse return null;
            if (slot.val == .scalar) return constIntOf(l.func, slot.val.scalar.value);
            return null;
        },
        else => return null,
    }
}

/// Read back the integer constant a value was defined by (an `iconst`), or null if it is
/// not a constant. Used to fold an unrolled loop variable into a compile-time array index.
fn constIntOf(func: *Function, value: Value) ?i64 {
    const inst = func.definingInst(value) orelse return null;
    return switch (func.opcode(inst)) {
        .iconst => |v| v,
        .convert => |c| constIntOf(func, c.value), // an int<->int relabel preserves the value
        else => null,
    };
}

/// `cond ? then : else` lowers to the IR `select` (value-level, no branches). The
/// condition must be a bool. Scalars select directly, vectors select component-wise.
fn lowerTernary(l: *L, cond_e: *parser.Expr, then_e: *parser.Expr, else_e: *parser.Expr) Error!Val {
    const cond = try lowerExpr(l, cond_e);
    if (cond != .scalar or cond.scalar.ty != .bool) return error.Unsupported;
    const c = cond.scalar.value;
    const then = try lowerExpr(l, then_e);
    const els = try lowerExpr(l, else_e);

    if (then == .scalar and els == .scalar) {
        var a = then.scalar;
        var b = els.scalar;
        const common = try unify(l, &a, &b);
        return .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, common), .{ .select = .{ .cond = c, .then = a.value, .@"else" = b.value } }), .ty = common } };
    }
    if (then == .vector and els == .vector and then.vector.len == els.vector.len) {
        // Select per component in the vector's component type (an integer vector ternary
        // must produce int-typed selects, not f32).
        const comp_ty = then.vector.comp_ty;
        const ct = try irType(l.func, comp_ty);
        var out: Vector = .{ .comps = undefined, .len = then.vector.len, .comp_ty = comp_ty };
        for (0..then.vector.len) |i| out.comps[i] = try l.func.appendInst(l.block, ct, .{ .select = .{ .cond = c, .then = then.vector.comps[i], .@"else" = els.vector.comps[i] } });
        return .{ .vector = out };
    }
    return error.TypeMismatch;
}

fn negate(l: *L, v: Val) Error!Val {
    switch (v) {
        .scalar => |s| {
            const z = try zero(l, s.ty);
            return .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, s.ty), .{ .arith = .{ .op = .sub, .lhs = z, .rhs = s.value } }), .ty = s.ty } };
        },
        .vector => |vec| {
            const f32t = try f32Type(l);
            const z = try l.func.appendInst(l.block, f32t, .{ .fconst = 0 });
            var out: Vector = .{ .comps = undefined, .len = vec.len };
            for (0..vec.len) |i| out.comps[i] = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .sub, .lhs = z, .rhs = vec.comps[i] } });
            return .{ .vector = out };
        },
        .matrix, .array, .structv => return error.Unsupported,
    }
}

fn lowerBinary(l: *L, op: parser.BinOp, lhs: *parser.Expr, rhs: *parser.Expr) Error!Val {
    const a = try lowerExpr(l, lhs);
    const b = try lowerExpr(l, rhs);
    if (a == .scalar and b == .scalar) return lowerScalarBinary(l, op, a.scalar, b.scalar);

    // Matrix products (linear-algebra `*`, not component-wise).
    if (a == .matrix or b == .matrix) {
        if (op != .mul) return error.Unsupported;
        if (a == .matrix and b == .vector) {
            if (a.matrix.dim != b.vector.len) return error.TypeMismatch;
            return matVecMul(l, a.matrix, b.vector);
        }
        if (a == .matrix and b == .matrix) {
            if (a.matrix.dim != b.matrix.dim) return error.TypeMismatch;
            return matMatMul(l, a.matrix, b.matrix);
        }
        return error.Unsupported;
    }

    // Vector arithmetic is component-wise. A scalar operand broadcasts to each lane. The
    // component type comes from the vector operand (integer vectors do integer arithmetic).
    const arith_op = binArith(op) orelse return error.Unsupported; // no vector comparisons yet
    const comp_ty = if (a == .vector) a.vector.comp_ty else b.vector.comp_ty;
    const ct = try irType(l.func, comp_ty);
    const len = if (a == .vector) a.vector.len else b.vector.len;
    if (a == .vector and b == .vector and a.vector.len != b.vector.len) return error.TypeMismatch;
    var out: Vector = .{ .comps = undefined, .len = len, .comp_ty = comp_ty };
    for (0..len) |i| {
        const av = try laneValue(l, a, i, comp_ty);
        const bv = try laneValue(l, b, i, comp_ty);
        out.comps[i] = try l.func.appendInst(l.block, ct, .{ .arith = .{ .op = arith_op, .lhs = av, .rhs = bv } });
    }
    return .{ .vector = out };
}

/// Component `i` of a value: a vector's lane, or a scalar broadcast (coerced to the
/// vector's component type) to every lane.
fn laneValue(l: *L, v: Val, i: usize, comp_ty: Type) Error!Value {
    switch (v) {
        .vector => |vec| return vec.comps[i],
        .scalar => |s| return (try coerce(l, s, comp_ty)).value,
        .matrix, .array, .structv => return error.Unsupported,
    }
}

fn binArith(op: parser.BinOp) ?ir.function.BinOp {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
        .mod => .rem,
        .bit_and => .bit_and,
        .bit_or => .bit_or,
        .bit_xor => .bit_xor,
        .shl => .shl,
        .shr => .shr,
        else => null,
    };
}

fn lowerScalarBinary(l: *L, op: parser.BinOp, lhs: Scalar, rhs: Scalar) Error!Val {
    var a = lhs;
    var b = rhs;
    const common = try unify(l, &a, &b);

    if (binArith(op)) |ao| {
        if (op == .mod and common == .float) return error.Unsupported;
        return .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, common), .{ .arith = .{ .op = ao, .lhs = a.value, .rhs = b.value } }), .ty = common } };
    }
    const cmp: ?ir.function.CmpOp = switch (op) {
        .eq => .eq,
        .ne => .ne,
        .lt => .lt,
        .gt => .gt,
        .le => .le,
        .ge => .ge,
        else => null,
    };
    if (cmp) |co| {
        return .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = co, .lhs = a.value, .rhs = b.value } }), .ty = .bool } };
    }
    // Logical && / || on bools: bitwise on the 0/1 bool (no side effects to short-circuit).
    const logical: ?ir.function.BinOp = switch (op) {
        .logical_and => .bit_and,
        .logical_or => .bit_or,
        else => null,
    };
    if (logical) |lo| {
        if (common != .bool) return error.TypeMismatch;
        return .{ .scalar = .{ .value = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .arith = .{ .op = lo, .lhs = a.value, .rhs = b.value } }), .ty = .bool } };
    }
    return error.Unsupported;
}

fn lowerCall(l: *L, name: []const u8, args: []const *parser.Expr) Error!Val {
    // Vector constructors: vecN(...) gathers components (flattening vector args), or
    // splats a single scalar to all lanes.
    const ctor_len = if (std.mem.eql(u8, name, "vec2")) @as(u8, 2) else if (std.mem.eql(u8, name, "vec3")) @as(u8, 3) else if (std.mem.eql(u8, name, "vec4")) @as(u8, 4) else 0;
    if (ctor_len != 0) return constructVector(l, ctor_len, .float, args);
    const ivec_len = if (std.mem.eql(u8, name, "ivec2")) @as(u8, 2) else if (std.mem.eql(u8, name, "ivec3")) @as(u8, 3) else if (std.mem.eql(u8, name, "ivec4")) @as(u8, 4) else 0;
    if (ivec_len != 0) return constructVector(l, ivec_len, .int, args);
    const uvec_len = if (std.mem.eql(u8, name, "uvec2")) @as(u8, 2) else if (std.mem.eql(u8, name, "uvec3")) @as(u8, 3) else if (std.mem.eql(u8, name, "uvec4")) @as(u8, 4) else 0;
    if (uvec_len != 0) return constructVector(l, uvec_len, .uint, args);
    const bvec_len = if (std.mem.eql(u8, name, "bvec2")) @as(u8, 2) else if (std.mem.eql(u8, name, "bvec3")) @as(u8, 3) else if (std.mem.eql(u8, name, "bvec4")) @as(u8, 4) else 0;
    if (bvec_len != 0) return constructVector(l, bvec_len, .bool, args);

    // Vector relational functions: component-wise compare returning a bvec.
    if (relCmpOp(name)) |co| {
        if (args.len != 2) return error.Unsupported;
        return vectorCompare(l, co, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]));
    }
    // Boolean-vector reductions and component-wise not.
    if (std.mem.eql(u8, name, "any") or std.mem.eql(u8, name, "all")) {
        if (args.len != 1) return error.Unsupported;
        return reduceBvec(l, try lowerExpr(l, args[0]), std.mem.eql(u8, name, "all"));
    }
    if (std.mem.eql(u8, name, "not")) {
        if (args.len != 1) return error.Unsupported;
        return notBvec(l, try lowerExpr(l, args[0]));
    }
    if (std.mem.eql(u8, name, "bitCount")) return bitCountVal(l, try arg1(l, args));
    if (std.mem.eql(u8, name, "findLSB")) return bitScanVal(l, try arg1(l, args), false);
    if (std.mem.eql(u8, name, "findMSB")) return bitScanVal(l, try arg1(l, args), true);
    if (std.mem.eql(u8, name, "bitfieldReverse")) return bitfieldReverseVal(l, try arg1(l, args));
    if (std.mem.eql(u8, name, "bitfieldExtract")) return lowerBitfieldExtract(l, args);
    if (std.mem.eql(u8, name, "bitfieldInsert")) return lowerBitfieldInsert(l, args);
    if (std.mem.eql(u8, name, "packUnorm4x8")) return packNorm(l, try arg1(l, args), 4, 8, false);
    if (std.mem.eql(u8, name, "packSnorm4x8")) return packNorm(l, try arg1(l, args), 4, 8, true);
    if (std.mem.eql(u8, name, "packUnorm2x16")) return packNorm(l, try arg1(l, args), 2, 16, false);
    if (std.mem.eql(u8, name, "packSnorm2x16")) return packNorm(l, try arg1(l, args), 2, 16, true);
    if (std.mem.eql(u8, name, "unpackUnorm4x8")) return unpackNorm(l, try arg1(l, args), 4, 8, false);
    if (std.mem.eql(u8, name, "unpackSnorm4x8")) return unpackNorm(l, try arg1(l, args), 4, 8, true);
    if (std.mem.eql(u8, name, "unpackUnorm2x16")) return unpackNorm(l, try arg1(l, args), 2, 16, false);
    if (std.mem.eql(u8, name, "unpackSnorm2x16")) return unpackNorm(l, try arg1(l, args), 2, 16, true);
    // Bit-reinterpret builtins: reinterpret a value's bits as another 32-bit type.
    if (std.mem.eql(u8, name, "floatBitsToInt")) return reinterpretVal(l, try arg1(l, args), .int);
    if (std.mem.eql(u8, name, "floatBitsToUint")) return reinterpretVal(l, try arg1(l, args), .uint);
    if (std.mem.eql(u8, name, "intBitsToFloat") or std.mem.eql(u8, name, "uintBitsToFloat")) return reinterpretVal(l, try arg1(l, args), .float);
    const ctor_dim = if (std.mem.eql(u8, name, "mat2")) @as(u8, 2) else if (std.mem.eql(u8, name, "mat3")) @as(u8, 3) else if (std.mem.eql(u8, name, "mat4")) @as(u8, 4) else 0;
    if (ctor_dim != 0) return constructMatrix(l, ctor_dim, args);

    if (std.mem.eql(u8, name, "dot")) {
        if (args.len != 2) return error.Unsupported;
        const a = try lowerExpr(l, args[0]);
        const b = try lowerExpr(l, args[1]);
        return dotProduct(l, a, b);
    }
    if (std.mem.eql(u8, name, "min") or std.mem.eql(u8, name, "max")) {
        if (args.len != 2) return error.Unsupported;
        return minMaxVal(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]), std.mem.eql(u8, name, "max"));
    }
    if (std.mem.eql(u8, name, "clamp")) {
        if (args.len != 3) return error.Unsupported;
        const x = try lowerExpr(l, args[0]);
        const lo = try lowerExpr(l, args[1]);
        const hi = try lowerExpr(l, args[2]);
        return minMaxVal(l, try minMaxVal(l, x, lo, true), hi, false);
    }
    if (std.mem.eql(u8, name, "abs")) {
        if (args.len != 1) return error.Unsupported;
        return lowerAbs(l, try lowerExpr(l, args[0]));
    }
    if (std.mem.eql(u8, name, "mix")) {
        if (args.len != 3) return error.Unsupported;
        return lowerMix(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]), try lowerExpr(l, args[2]));
    }
    if (std.mem.eql(u8, name, "cross")) {
        if (args.len != 2) return error.Unsupported;
        return lowerCross(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]));
    }
    if (std.mem.eql(u8, name, "sqrt")) {
        if (args.len != 1) return error.Unsupported;
        return lowerSqrt(l, try lowerExpr(l, args[0]));
    }
    if (std.mem.eql(u8, name, "length")) {
        if (args.len != 1) return error.Unsupported;
        return lowerLength(l, try lowerExpr(l, args[0]));
    }
    if (std.mem.eql(u8, name, "normalize")) {
        if (args.len != 1) return error.Unsupported;
        return lowerNormalize(l, try lowerExpr(l, args[0]));
    }
    if (std.mem.eql(u8, name, "floor")) return mapFloat(l, try arg1(l, args), floorElem);
    if (std.mem.eql(u8, name, "ceil")) return mapFloat(l, try arg1(l, args), ceilElem);
    if (std.mem.eql(u8, name, "trunc")) return mapFloat(l, try arg1(l, args), truncElem);
    if (std.mem.eql(u8, name, "roundEven") or std.mem.eql(u8, name, "round")) return mapFloat(l, try arg1(l, args), roundElem);
    if (std.mem.eql(u8, name, "fract")) return mapFloat(l, try arg1(l, args), fractElem);
    if (std.mem.eql(u8, name, "sign")) return mapFloat(l, try arg1(l, args), signElem);
    if (std.mem.eql(u8, name, "radians")) return mapFloat(l, try arg1(l, args), radiansElem);
    if (std.mem.eql(u8, name, "degrees")) return mapFloat(l, try arg1(l, args), degreesElem);
    if (std.mem.eql(u8, name, "step")) {
        if (args.len != 2) return error.Unsupported;
        return mapFloat2(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]), stepElem); // step(edge, x)
    }
    if (std.mem.eql(u8, name, "mod")) {
        if (args.len != 2) return error.Unsupported;
        return mapFloat2(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]), modElem);
    }
    if (std.mem.eql(u8, name, "smoothstep")) {
        if (args.len != 3) return error.Unsupported;
        return lowerSmoothstep(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]), try lowerExpr(l, args[2]));
    }
    // Transcendentals: emitted as GLSL.std.450 extended instructions on the SPIR-V side.
    if (std.mem.eql(u8, name, "sin")) return mapFloat(l, try arg1(l, args), sinElem);
    if (std.mem.eql(u8, name, "cos")) return mapFloat(l, try arg1(l, args), cosElem);
    if (std.mem.eql(u8, name, "tan")) return mapFloat(l, try arg1(l, args), tanElem);
    if (std.mem.eql(u8, name, "asin")) return mapFloat(l, try arg1(l, args), asinElem);
    if (std.mem.eql(u8, name, "acos")) return mapFloat(l, try arg1(l, args), acosElem);
    if (std.mem.eql(u8, name, "atan")) return mapFloat(l, try arg1(l, args), atanElem);
    if (std.mem.eql(u8, name, "exp")) return mapFloat(l, try arg1(l, args), expElem);
    if (std.mem.eql(u8, name, "log")) return mapFloat(l, try arg1(l, args), logElem);
    if (std.mem.eql(u8, name, "exp2")) return mapFloat(l, try arg1(l, args), exp2Elem);
    if (std.mem.eql(u8, name, "log2")) return mapFloat(l, try arg1(l, args), log2Elem);
    if (std.mem.eql(u8, name, "inversesqrt")) return mapFloat(l, try arg1(l, args), invSqrtElem);
    // Fragment derivatives: core SPIR-V ops (valid only in a fragment shader).
    if (std.mem.eql(u8, name, "dFdx")) return mapFloat(l, try arg1(l, args), dfdxElem);
    if (std.mem.eql(u8, name, "dFdy")) return mapFloat(l, try arg1(l, args), dfdyElem);
    if (std.mem.eql(u8, name, "fwidth")) return mapFloat(l, try arg1(l, args), fwidthElem);
    // GLSL ES 1.00 spells the sampler lookup `texture2D`/`textureCube`. GLSL ES 3.00
    // uses the overloaded `texture`. All map to one sampled-image fetch here.
    if (std.mem.eql(u8, name, "texture") or std.mem.eql(u8, name, "texture2D") or std.mem.eql(u8, name, "textureCube") or std.mem.eql(u8, name, "texture3D")) {
        if (args.len != 2) return error.Unsupported;
        // The first argument names a declared sampler (not a value to lower).
        const sampler_name = switch (args[0].*) {
            .ident => |id| id,
            else => return error.Unsupported,
        };
        const binding = l.samplerBinding(sampler_name) orelse return error.UndefinedName;
        const coord = try lowerExpr(l, args[1]);
        if (coord != .vector) return error.TypeMismatch;
        // A sampler2DArrayShadow takes a vec4 (uv, layer, dref) and returns a SCALAR depth-compare
        // result. Checked FIRST (before cube-shadow / plain shadow / the vec3 array path, since this
        // sampler is also flagged 2darray).
        if (l.samplerIs2dArrayShadow(binding)) {
            if (coord.vector.len != 4) return error.TypeMismatch;
            return lowerTexture2dArrayShadow(l, binding, coord.vector);
        }
        // A samplerCubeShadow takes a vec4 (dir.xyz, dref) and returns a SCALAR depth-compare result.
        // Checked BEFORE the plain shadow and the vec3 cube path (this sampler is also flagged cube).
        if (l.samplerIsCubeShadow(binding)) {
            if (coord.vector.len != 4) return error.TypeMismatch;
            return lowerTextureCubeShadow(l, binding, coord.vector);
        }
        // A sampler2DShadow takes a vec3 (u, v, dref) and returns a SCALAR depth-compare result
        // (reference dref vs the stored depth), not a vec4. Checked before the cube/3D vec3 path.
        if (l.samplerIsShadow(binding)) {
            if (coord.vector.len != 3) return error.TypeMismatch;
            return lowerTextureShadow(l, binding, coord.vector);
        }
        // A cube or 3D sampler is addressed by a vec3 coordinate; a 2D sampler by a vec2 uv. The
        // coord width must match the sampler's declared kind. Cube + 3D share the vec3 sample
        // path (the host sampler dispatches cube-face-select vs trilinear-3D on the descriptor).
        if (l.samplerIsVec3(binding)) {
            if (coord.vector.len != 3) return error.TypeMismatch;
            return lowerTextureCube(l, binding, coord.vector);
        }
        if (coord.vector.len != 2) return error.TypeMismatch;
        // Vertex texture fetch: a vertex shader has no screen-space derivatives, so an
        // implicit-LOD `texture2D` here samples the base level. Lower it to an explicit
        // LOD-0 sample (the derivative-based implicit path is undefined in the vertex stage).
        if (l.stage == .vertex) return lowerTextureLod(l, binding, coord.vector, try fconst(l, 0));
        return lowerTexture(l, binding, coord.vector);
    }
    // Explicit-LOD sampling: GLSL ES 1.00 `texture2DLod(s, uv, lod)` / ES 3.00
    // `textureLod(s, uv, lod)`. Selects the mip level directly (used by prefiltered-env /
    // manual-LOD passes) instead of the fragment-derivative implicit LOD.
    if (std.mem.eql(u8, name, "textureLod") or std.mem.eql(u8, name, "texture2DLod") or std.mem.eql(u8, name, "textureCubeLod") or std.mem.eql(u8, name, "texture3DLod")) {
        if (args.len != 3) return error.Unsupported;
        const sampler_name = switch (args[0].*) {
            .ident => |id| id,
            else => return error.Unsupported,
        };
        const binding = l.samplerBinding(sampler_name) orelse return error.UndefinedName;
        const coord = try lowerExpr(l, args[1]);
        if (coord != .vector) return error.TypeMismatch;
        const lod_v = try lowerExpr(l, args[2]);
        const lod: Value = switch (lod_v) {
            .scalar => |s| s.value,
            else => return error.TypeMismatch,
        };
        // A cube or 3D sampler is addressed by a vec3 coordinate + the explicit LOD; a 2D sampler
        // by a vec2 uv + LOD. Cube + 3D share the vec3 explicit-LOD path.
        if (l.samplerIsVec3(binding)) {
            if (coord.vector.len != 3) return error.TypeMismatch;
            return lowerTextureCubeLod(l, binding, coord.vector, lod);
        }
        if (coord.vector.len != 2) return error.TypeMismatch;
        return lowerTextureLod(l, binding, coord.vector, lod);
    }
    // `textureGather(sampler2D, uv [, comp])` (GLSL ES 3.10): returns the 4 texels of the bilinear
    // footprint of one channel (default RED) as a vec4 in the order (i0j1, i1j1, i1j0, i0j0) - the
    // GL gather order. `comp` selects the channel and MUST be a constant 0..3.
    if (std.mem.eql(u8, name, "textureGather")) {
        if (args.len != 2 and args.len != 3) return error.Unsupported;
        const sampler_name = switch (args[0].*) {
            .ident => |id| id,
            else => return error.Unsupported,
        };
        const binding = l.samplerBinding(sampler_name) orelse return error.UndefinedName;
        const coord = try lowerExpr(l, args[1]);
        if (coord != .vector or coord.vector.len != 2) return error.TypeMismatch;
        const comp: i64 = if (args.len == 3) (try constInt(l, args[2])) orelse return error.Unsupported else 0;
        if (comp < 0 or comp > 3) return error.Unsupported;
        return lowerTextureGather(l, binding, coord.vector, @intCast(comp));
    }
    // `texelFetch(sampler, ivecN P [, int lod])` (GLSL ES 3.00): fetch the EXACT texel at INTEGER
    // coords P of mip level `lod` - NO filtering, NO normalization/wrapping (pixel-exact atlas reads,
    // data textures, look-up tables). sampler2D takes ivec2 -> `tex.fetch.<binding>`; sampler2DArray
    // and sampler3D take ivec3 (x, y, layer/z) -> `tex.fetch3.<binding>`. A samplerCube has no
    // texelFetch. The emitter turns each into an OpImageFetch.
    if (std.mem.eql(u8, name, "texelFetch")) {
        if (args.len != 2 and args.len != 3) return error.Unsupported;
        const sampler_name = switch (args[0].*) {
            .ident => |id| id,
            else => return error.Unsupported,
        };
        const binding = l.samplerBinding(sampler_name) orelse return error.UndefinedName;
        if (l.samplerIsCube(binding)) return error.Unsupported; // no texelFetch on a cube
        const is_vec3 = l.samplerIs3d(binding) or l.samplerIs2dArray(binding);
        const coord = try lowerExpr(l, args[1]);
        const want: u8 = if (is_vec3) 3 else 2;
        if (coord != .vector or coord.vector.len != want) return error.TypeMismatch;
        const i32t = try l.func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const lod: Value = if (args.len == 3) switch (try lowerExpr(l, args[2])) {
            .scalar => |s| s.value,
            else => return error.TypeMismatch,
        } else try l.func.appendInst(l.block, i32t, .{ .iconst = 0 });
        if (is_vec3) return lowerTexelFetch3(l, binding, coord.vector, lod);
        return lowerTexelFetch(l, binding, coord.vector, lod);
    }
    if (std.mem.eql(u8, name, "pow")) {
        if (args.len != 2) return error.Unsupported;
        return mapFloat2(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]), powElem);
    }
    if (std.mem.eql(u8, name, "distance")) {
        if (args.len != 2) return error.Unsupported;
        return lowerDistance(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]));
    }
    if (std.mem.eql(u8, name, "reflect")) {
        if (args.len != 2) return error.Unsupported;
        return lowerReflect(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]));
    }
    if (std.mem.eql(u8, name, "refract")) {
        if (args.len != 3) return error.Unsupported;
        return lowerRefract(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]), try lowerExpr(l, args[2]));
    }
    if (std.mem.eql(u8, name, "faceforward")) {
        if (args.len != 3) return error.Unsupported;
        return lowerFaceforward(l, try lowerExpr(l, args[0]), try lowerExpr(l, args[1]), try lowerExpr(l, args[2]));
    }

    // Scalar type-constructor conversions.
    const to: ?Type = if (std.mem.eql(u8, name, "float")) .float else if (std.mem.eql(u8, name, "int")) .int else if (std.mem.eql(u8, name, "uint")) .uint else null;
    if (to) |t| {
        if (args.len != 1) return error.Unsupported;
        const a = try lowerExpr(l, args[0]);
        if (a != .scalar) return error.Unsupported;
        return .{ .scalar = try coerce(l, a.scalar, t) };
    }

    // A user-defined function: inline it at the call site (no recursive calls).
    for (l.user_fns) |f| {
        if (std.mem.eql(u8, f.name, name)) return inlineCall(l, f, args);
    }
    return error.Unsupported;
}

/// Inline a user-defined GLSL function call. Lowers each argument, binds the formal
/// parameters to the argument Vals in a fresh scope on the shared env, lowers the body
/// (capturing `return <expr>` into a local slot instead of terminating the block), then
/// restores the scope. Returns the function's return Val (a void function returns a
/// dummy scalar 0). No recursion is supported (a depth guard rejects it).
fn inlineCall(l: *L, f: parser.Function, args: []const *parser.Expr) Error!Val {
    if (f.params.len != args.len) return error.Unsupported;
    if (l.inline_depth >= 16) return error.Unsupported; // recursion / runaway guard
    // Lower the arguments in the CURRENT scope first (before binding params).
    var arg_vals: [16]Val = undefined;
    if (args.len > arg_vals.len) return error.Unsupported;
    for (args, 0..) |a, i| arg_vals[i] = try coerceVal(l, try lowerExpr(l, a), f.params[i].ty);

    // Open a new scope: remember the env length, push the params bound to the arg Vals.
    const scope_base = l.env.items.len;
    for (f.params, 0..) |p, i| try l.env.append(l.allocator, .{ .name = p.name, .val = arg_vals[i] });

    // The inline's exit block: control resumes here after the body, and its params are the
    // SSA-merged return value (a phi over every `return` path). Each `return` jumps here
    // carrying its value, so an early return inside an `if` short-circuits correctly.
    const exit = try l.func.appendBlock();
    const result = try returnParams(l, exit, f.ret);

    const saved_ctx = l.inline_ctx;
    l.inline_ctx = .{ .exit = exit, .ret_ty = f.ret };
    l.inline_depth += 1;
    defer {
        l.inline_depth -= 1;
        l.inline_ctx = saved_ctx;
        l.env.shrinkRetainingCapacity(scope_base); // pop the inlined scope
    }
    var terminated = false;
    for (f.body) |stmt| if (try lowerStmt(l, f.ret, stmt)) {
        terminated = true;
        break; // this path returned (jumped to exit), no fall-through
    };
    // A non-returning fall-through (a void function, or one missing a final return): jump to
    // the exit carrying a default value so the exit's phis are well-formed.
    if (!terminated) {
        const dflt = try defaultReturn(l, f.ret);
        var vals = [_]Val{dflt};
        const dargs = try flattenVals(l, &vals);
        defer l.allocator.free(dargs);
        try l.func.setJump(l.block, exit, dargs);
    }
    l.block = exit;
    return result;
}

/// Create the inline-exit block's params for a return of type `ret` and return the Val they
/// form (a scalar, a float vector, or a float matrix). The arg order matches `flattenVals`.
fn returnParams(l: *L, exit: Block, ret: Type) Error!Val {
    const md = matDim(ret);
    if (md != 0) {
        const f32t = try f32Type(l);
        var comps: [16]Value = undefined;
        for (0..@as(usize, md) * md) |i| comps[i] = try l.func.appendBlockParam(exit, f32t);
        return .{ .matrix = .{ .comps = comps, .dim = md } };
    }
    const n = vecLen(ret);
    if (n != 0) {
        const comp_ty = vecCompType(ret);
        const ct = try irType(l.func, comp_ty);
        var comps: [4]Value = undefined;
        for (0..n) |i| comps[i] = try l.func.appendBlockParam(exit, ct);
        return .{ .vector = .{ .comps = comps, .len = n, .comp_ty = comp_ty } };
    }
    if (ret == .void) {
        // A void inline still gets one dummy scalar param so callers have a Val.
        return .{ .scalar = .{ .value = try l.func.appendBlockParam(exit, try irType(l.func, .float)), .ty = .float } };
    }
    return .{ .scalar = .{ .value = try l.func.appendBlockParam(exit, try irType(l.func, ret)), .ty = ret } };
}

/// A default return Val for a fall-through inline path (a zero of the return shape).
fn defaultReturn(l: *L, ret: Type) Error!Val {
    if (ret == .void) return .{ .scalar = .{ .value = try fconst(l, 0), .ty = .float } };
    return zeroVal(l, ret);
}

/// `distance(a, b)` = `length(a - b)`.
fn lowerDistance(l: *L, a: Val, b: Val) Error!Val {
    return lowerLength(l, try mapFloat2(l, a, b, fsub));
}

/// `reflect(I, N)` = `I - 2*dot(N, I)*N` (I and N are vectors).
fn lowerReflect(l: *L, incident: Val, normal: Val) Error!Val {
    if (incident != .vector or normal != .vector or incident.vector.len != normal.vector.len) return error.TypeMismatch;
    const d = (try dotProduct(l, normal, incident)).scalar.value;
    const two_d = try fmul(l, try fconst(l, 2), d);
    var out: [4]Value = undefined;
    for (0..incident.vector.len) |i| out[i] = try fsub(l, incident.vector.comps[i], try fmul(l, two_d, normal.vector.comps[i]));
    return .{ .vector = .{ .comps = out, .len = incident.vector.len } };
}

/// `refract(I, N, eta)` (I and N the same vector type, eta a scalar): the GLSL refraction
/// vector. With `d = dot(N, I)` and `k = 1 - eta*eta*(1 - d*d)`, the result is
/// `k < 0 ? vecN(0) : eta*I - (eta*d + sqrt(k))*N` (total internal reflection -> zero).
fn lowerRefract(l: *L, incident: Val, normal: Val, eta_v: Val) Error!Val {
    if (incident != .vector or normal != .vector or incident.vector.len != normal.vector.len) return error.TypeMismatch;
    // eta is a scalar float (broadcast). Coerce an int literal if needed.
    if (eta_v != .scalar) return error.TypeMismatch;
    const eta = (try coerce(l, eta_v.scalar, .float)).value;

    const d = (try dotProduct(l, normal, incident)).scalar.value; // dot(N, I)
    // k = 1 - eta*eta * (1 - d*d)
    const dd = try fmul(l, d, d);
    const one = try fconst(l, 1);
    const one_minus_dd = try fsub(l, one, dd);
    const eta2 = try fmul(l, eta, eta);
    const k = try fsub(l, one, try fmul(l, eta2, one_minus_dd));
    // sqrt(k) is only meaningful when k >= 0. The select discards it when k < 0.
    const sqrt_k = try sqrtScalar(l, k);
    // scale = eta*d + sqrt(k)
    const scale = try l.func.appendInst(l.block, try f32Type(l), .{ .arith = .{ .op = .add, .lhs = try fmul(l, eta, d), .rhs = sqrt_k } });
    // cond = k < 0  -> total internal reflection, return the zero vector.
    const cond = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .lt, .lhs = k, .rhs = try fconst(l, 0) } });
    const zero_v = try fconst(l, 0);
    var out: [4]Value = undefined;
    for (0..incident.vector.len) |i| {
        // eta*I[i] - scale*N[i]
        const refr = try fsub(l, try fmul(l, eta, incident.vector.comps[i]), try fmul(l, scale, normal.vector.comps[i]));
        out[i] = try l.func.appendInst(l.block, try f32Type(l), .{ .select = .{ .cond = cond, .then = zero_v, .@"else" = refr } });
    }
    return .{ .vector = .{ .comps = out, .len = incident.vector.len } };
}

/// `faceforward(N, I, Nref)` = `dot(Nref, I) < 0 ? N : -N`.
fn lowerFaceforward(l: *L, n: Val, incident: Val, nref: Val) Error!Val {
    if (n != .vector or incident != .vector or nref != .vector) return error.TypeMismatch;
    const d = (try dotProduct(l, nref, incident)).scalar.value;
    const cond = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .lt, .lhs = d, .rhs = try fconst(l, 0) } });
    var out: [4]Value = undefined;
    for (0..n.vector.len) |i| {
        const neg = try fsub(l, try fconst(l, 0), n.vector.comps[i]);
        out[i] = try l.func.appendInst(l.block, try f32Type(l), .{ .select = .{ .cond = cond, .then = n.vector.comps[i], .@"else" = neg } });
    }
    return .{ .vector = .{ .comps = out, .len = n.vector.len } };
}

fn dotProduct(l: *L, a: Val, b: Val) Error!Val {
    if (a != .vector or b != .vector or a.vector.len != b.vector.len) return error.TypeMismatch;
    const f32t = try f32Type(l);
    var acc = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .mul, .lhs = a.vector.comps[0], .rhs = b.vector.comps[0] } });
    for (1..a.vector.len) |i| {
        const prod = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .mul, .lhs = a.vector.comps[i], .rhs = b.vector.comps[i] } });
        acc = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = prod } });
    }
    return .{ .scalar = .{ .value = acc, .ty = .float } };
}

/// The longest operand length (1 for an all-scalar pair).
fn maxLen(a: Val, b: Val) u8 {
    const al: u8 = if (a == .vector) a.vector.len else 1;
    const bl: u8 = if (b == .vector) b.vector.len else 1;
    return @max(al, bl);
}

/// `min`/`max`: a single `<`/`>` plus a `select`, scalar or component-wise.
fn minMaxScalar(l: *L, x: Value, y: Value, ty: Type, want_max: bool) Error!Value {
    // Per the GLSL spec: max(x,y) returns y if x < y, otherwise x. min(x,y) returns y if y < x,
    // otherwise x. Both compare with `<` and select y on true. This NaN-propagates exactly like
    // real GL (e.g. max(NaN, 0.0) = NaN), which a `x > y ? x : y` form would not - it would
    // collapse NaN to the other operand and turn a degenerate normalize() into a clean zero.
    const bool_t = try irType(l.func, .bool);
    const cond = if (want_max)
        try l.func.appendInst(l.block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } })
    else
        try l.func.appendInst(l.block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = y, .rhs = x } });
    return l.func.appendInst(l.block, try irType(l.func, ty), .{ .select = .{ .cond = cond, .then = y, .@"else" = x } });
}

fn minMaxVal(l: *L, a: Val, b: Val, want_max: bool) Error!Val {
    if (a == .scalar and b == .scalar) {
        var x = a.scalar;
        var y = b.scalar;
        const common = try unify(l, &x, &y);
        return .{ .scalar = .{ .value = try minMaxScalar(l, x.value, y.value, common, want_max), .ty = common } };
    }
    const len = maxLen(a, b);
    if (a == .vector and b == .vector and a.vector.len != b.vector.len) return error.TypeMismatch;
    const comp_ty = if (a == .vector) a.vector.comp_ty else b.vector.comp_ty;
    var out: [4]Value = undefined;
    for (0..len) |i| out[i] = try minMaxScalar(l, try laneValue(l, a, i, comp_ty), try laneValue(l, b, i, comp_ty), comp_ty, want_max);
    return .{ .vector = .{ .comps = out, .len = len, .comp_ty = comp_ty } };
}

/// `abs(x)` = `x < 0 ? -x : x`.
fn absScalar(l: *L, x: Value, ty: Type) Error!Value {
    const z = try zero(l, ty);
    const cond = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = z } });
    const neg = try l.func.appendInst(l.block, try irType(l.func, ty), .{ .arith = .{ .op = .sub, .lhs = z, .rhs = x } });
    return l.func.appendInst(l.block, try irType(l.func, ty), .{ .select = .{ .cond = cond, .then = neg, .@"else" = x } });
}

fn lowerAbs(l: *L, v: Val) Error!Val {
    switch (v) {
        .scalar => |s| return .{ .scalar = .{ .value = try absScalar(l, s.value, s.ty), .ty = s.ty } },
        .vector => |vec| {
            var out: [4]Value = undefined;
            for (0..vec.len) |i| out[i] = try absScalar(l, vec.comps[i], vec.comp_ty);
            return .{ .vector = .{ .comps = out, .len = vec.len, .comp_ty = vec.comp_ty } };
        },
        .matrix, .array, .structv => return error.Unsupported,
    }
}

/// `mix(a, b, t)` = `a + (b - a) * t`, component-wise (a scalar `t` broadcasts). When `t`
/// is a bool / bvec it is instead the GLSL select form: pick `b` where the bool is true.
fn lowerMix(l: *L, a: Val, b: Val, t: Val) Error!Val {
    // Boolean select form: mix(x, y, bvec) / mix(x, y, bool).
    const bool_sel = (t == .vector and t.vector.comp_ty == .bool) or (t == .scalar and t.scalar.ty == .bool);
    if (bool_sel) return mixSelect(l, a, b, t);
    const f32t = try f32Type(l);
    const is_vec = (a == .vector or b == .vector);
    const len = maxLen(a, b);
    if (a == .vector and b == .vector and a.vector.len != b.vector.len) return error.TypeMismatch;
    var out: [4]Value = undefined;
    for (0..len) |i| {
        const ai = try laneValue(l, a, i, .float);
        const bi = try laneValue(l, b, i, .float);
        const ti = try laneValue(l, t, i, .float);
        const diff = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .sub, .lhs = bi, .rhs = ai } });
        const scaled = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .mul, .lhs = diff, .rhs = ti } });
        out[i] = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .add, .lhs = ai, .rhs = scaled } });
    }
    if (!is_vec) return .{ .scalar = .{ .value = out[0], .ty = .float } };
    return .{ .vector = .{ .comps = out, .len = len } };
}

/// `mix(a, b, cond)` select form: each lane is `cond ? b : a`. `cond` is a scalar bool
/// (broadcast) or a bvec. The result's component type follows `a`/`b`.
fn mixSelect(l: *L, a: Val, b: Val, cond: Val) Error!Val {
    const comp_ty = if (a == .vector) a.vector.comp_ty else if (b == .vector) b.vector.comp_ty else if (a == .scalar) a.scalar.ty else .float;
    const ct = try irType(l.func, comp_ty);
    const is_vec = (a == .vector or b == .vector or cond == .vector);
    if (!is_vec) {
        const ai = try laneValue(l, a, 0, comp_ty);
        const bi = try laneValue(l, b, 0, comp_ty);
        const c = cond.scalar.value;
        return .{ .scalar = .{ .value = try l.func.appendInst(l.block, ct, .{ .select = .{ .cond = c, .then = bi, .@"else" = ai } }), .ty = comp_ty } };
    }
    const len = if (cond == .vector) cond.vector.len else maxLen(a, b);
    var out: [4]Value = undefined;
    for (0..len) |i| {
        const ai = try laneValue(l, a, i, comp_ty);
        const bi = try laneValue(l, b, i, comp_ty);
        const c = if (cond == .vector) cond.vector.comps[i] else cond.scalar.value;
        out[i] = try l.func.appendInst(l.block, ct, .{ .select = .{ .cond = c, .then = bi, .@"else" = ai } });
    }
    return .{ .vector = .{ .comps = out, .len = len, .comp_ty = comp_ty } };
}

/// `cross(a, b)` for vec3: the standard component formulas.
fn lowerCross(l: *L, a: Val, b: Val) Error!Val {
    if (a != .vector or b != .vector or a.vector.len != 3 or b.vector.len != 3) return error.TypeMismatch;
    const ac = a.vector.comps;
    const bc = b.vector.comps;
    var out: [4]Value = undefined;
    out[0] = try mulSub(l, ac[1], bc[2], ac[2], bc[1]);
    out[1] = try mulSub(l, ac[2], bc[0], ac[0], bc[2]);
    out[2] = try mulSub(l, ac[0], bc[1], ac[1], bc[0]);
    return .{ .vector = .{ .comps = out, .len = 3 } };
}

fn sqrtScalar(l: *L, x: Value) Error!Value {
    return l.func.appendInst(l.block, try f32Type(l), .{ .unary = .{ .op = .sqrt, .value = x } });
}

/// The single argument of a builtin call (lowered).
fn arg1(l: *L, args: []const *parser.Expr) Error!Val {
    if (args.len != 1) return error.Unsupported;
    return lowerExpr(l, args[0]);
}

/// Apply a per-component float function to a scalar or vector.
fn mapFloat(l: *L, v: Val, comptime f: fn (*L, Value) Error!Value) Error!Val {
    switch (v) {
        .scalar => |s| return .{ .scalar = .{ .value = try f(l, (try coerce(l, s, .float)).value), .ty = .float } },
        .vector => |vec| {
            var out: [4]Value = undefined;
            for (0..vec.len) |i| out[i] = try f(l, vec.comps[i]);
            return .{ .vector = .{ .comps = out, .len = vec.len } };
        },
        .matrix, .array, .structv => return error.Unsupported,
    }
}

/// Apply a per-component float binary function (a scalar operand broadcasts).
fn mapFloat2(l: *L, a: Val, b: Val, comptime f: fn (*L, Value, Value) Error!Value) Error!Val {
    if (a == .scalar and b == .scalar) {
        return .{ .scalar = .{ .value = try f(l, (try coerce(l, a.scalar, .float)).value, (try coerce(l, b.scalar, .float)).value), .ty = .float } };
    }
    const len = maxLen(a, b);
    if (a == .vector and b == .vector and a.vector.len != b.vector.len) return error.TypeMismatch;
    var out: [4]Value = undefined;
    for (0..len) |i| out[i] = try f(l, try laneValue(l, a, i, .float), try laneValue(l, b, i, .float));
    return .{ .vector = .{ .comps = out, .len = len } };
}

fn fconst(l: *L, v: f64) Error!Value {
    return l.func.appendInst(l.block, try f32Type(l), .{ .fconst = v });
}
fn fmul(l: *L, a: Value, b: Value) Error!Value {
    return l.func.appendInst(l.block, try f32Type(l), .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
}
fn fsub(l: *L, a: Value, b: Value) Error!Value {
    return l.func.appendInst(l.block, try f32Type(l), .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } });
}

/// A unary GLSL.std.450 intrinsic, lowered to an IR call the SPIR-V emitter maps to
/// OpExtInst. The host backend cannot run these, so they are SPIR-V-only.
fn extCall1(l: *L, comptime name: []const u8, x: Value) Error!Value {
    return l.func.appendCall(l.block, try f32Type(l), "glsl." ++ name, &.{x});
}
fn sinElem(l: *L, x: Value) Error!Value {
    return extCall1(l, "sin", x);
}
fn cosElem(l: *L, x: Value) Error!Value {
    return extCall1(l, "cos", x);
}
fn tanElem(l: *L, x: Value) Error!Value {
    return extCall1(l, "tan", x);
}
fn asinElem(l: *L, x: Value) Error!Value {
    return extCall1(l, "asin", x);
}
fn acosElem(l: *L, x: Value) Error!Value {
    return extCall1(l, "acos", x);
}
fn atanElem(l: *L, x: Value) Error!Value {
    return extCall1(l, "atan", x);
}
fn expElem(l: *L, x: Value) Error!Value {
    return extCall1(l, "exp", x);
}
fn logElem(l: *L, x: Value) Error!Value {
    return extCall1(l, "log", x);
}
fn exp2Elem(l: *L, x: Value) Error!Value {
    return extCall1(l, "exp2", x);
}
fn log2Elem(l: *L, x: Value) Error!Value {
    return extCall1(l, "log2", x);
}
fn invSqrtElem(l: *L, x: Value) Error!Value {
    return extCall1(l, "inversesqrt", x);
}
fn powElem(l: *L, x: Value, y: Value) Error!Value {
    return l.func.appendCall(l.block, try f32Type(l), "glsl.pow", &.{ x, y });
}

/// `texture(samplerN, uv)`: one `tex.sample.<binding>` call carries the sample (the
/// emitter turns it into an OpImageSampleImplicitLod yielding a vec4), then four
/// `spirv.extract.<i>` calls pull the lanes (OpCompositeExtract). Scalarized to a vec4.
fn lowerTexture(l: *L, binding: u32, uv: Vector) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [24]u8 = undefined;
    const sample_name = std.fmt.bufPrint(&namebuf, "tex.sample.{d}", .{binding}) catch unreachable;
    const sample = try l.func.appendCall(l.block, f32t, sample_name, &.{ uv.comps[0], uv.comps[1] });
    var comps: [4]Value = undefined;
    inline for (0..4) |i| {
        comps[i] = try l.func.appendCall(l.block, f32t, "spirv.extract." ++ std.fmt.comptimePrint("{d}", .{i}), &.{sample});
    }
    return .{ .vector = .{ .comps = comps, .len = 4 } };
}

/// `texture(sampler2DShadowN, vec3(uv, dref))`: a hardware depth compare. Emits ONE
/// `tex.sample.shadow.<binding>` call carrying (u, v, dref); the emitter turns it into an
/// OpImageSampleDrefImplicitLod, which yields a SCALAR float (the compare result = the fraction of
/// the reference `dref` passing the stored depth). Returns a scalar, NOT a scalarized vec4.
fn lowerTextureShadow(l: *L, binding: u32, uvd: Vector) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [32]u8 = undefined;
    const sample_name = std.fmt.bufPrint(&namebuf, "tex.sample.shadow.{d}", .{binding}) catch unreachable;
    const sample = try l.func.appendCall(l.block, f32t, sample_name, &.{ uvd.comps[0], uvd.comps[1], uvd.comps[2] });
    return .{ .scalar = .{ .value = sample, .ty = .float } };
}

/// `texture(samplerCubeShadowN, vec4(dir, dref))`: a hardware cube depth compare. Emits ONE
/// `tex.sample.cube.shadow.<binding>` call carrying (x, y, z, dref); the emitter turns it into an
/// OpImageSampleDrefImplicitLod on a Cube image, which yields a SCALAR float (the compare result).
/// Returns a scalar, NOT a scalarized vec4. Mirrors lowerTextureShadow but with 3 coords + dref.
fn lowerTextureCubeShadow(l: *L, binding: u32, dird: Vector) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [40]u8 = undefined;
    const sample_name = std.fmt.bufPrint(&namebuf, "tex.sample.cube.shadow.{d}", .{binding}) catch unreachable;
    const sample = try l.func.appendCall(l.block, f32t, sample_name, &.{ dird.comps[0], dird.comps[1], dird.comps[2], dird.comps[3] });
    return .{ .scalar = .{ .value = sample, .ty = .float } };
}

/// `texture(sampler2DArrayShadowN, vec4(uv, layer, dref))`: a hardware 2D-array depth compare. Emits
/// ONE `tex.sample.2darray.shadow.<binding>` call carrying (u, v, layer, dref); the emitter turns it
/// into an OpImageSampleDrefImplicitLod on a 2D-Arrayed image, yielding a SCALAR float (the compare
/// result). Returns a scalar, NOT a scalarized vec4. Mirrors lowerTextureCubeShadow (vec3 coord + dref).
fn lowerTexture2dArrayShadow(l: *L, binding: u32, uvld: Vector) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [44]u8 = undefined;
    const sample_name = std.fmt.bufPrint(&namebuf, "tex.sample.2darray.shadow.{d}", .{binding}) catch unreachable;
    const sample = try l.func.appendCall(l.block, f32t, sample_name, &.{ uvld.comps[0], uvld.comps[1], uvld.comps[2], uvld.comps[3] });
    return .{ .scalar = .{ .value = sample, .ty = .float } };
}

/// `textureGather(sampler2DN, uv, comp)`: gathers channel `comp` of the 4 bilinear-footprint texels.
/// Emits `tex.gather.<comp>.<binding>` carrying (u, v); the emitter turns it into OpImageGather with
/// the component operand, and the host/HW returns the 4 texels as a vec4. Yields a scalarized vec4.
fn lowerTextureGather(l: *L, binding: u32, uv: Vector, comp: u32) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [32]u8 = undefined;
    const gather_name = std.fmt.bufPrint(&namebuf, "tex.gather.{d}.{d}", .{ comp, binding }) catch unreachable;
    const sample = try l.func.appendCall(l.block, f32t, gather_name, &.{ uv.comps[0], uv.comps[1] });
    var comps: [4]Value = undefined;
    inline for (0..4) |i| {
        comps[i] = try l.func.appendCall(l.block, f32t, "spirv.extract." ++ std.fmt.comptimePrint("{d}", .{i}), &.{sample});
    }
    return .{ .vector = .{ .comps = comps, .len = 4 } };
}

/// `texelFetch(sampler2DN, ivec2 P, int lod)`: an exact integer-coordinate texel fetch (no filter,
/// no normalization). Emits `tex.fetch.<binding>` carrying the two i32 coords + the i32 lod; the
/// emitter turns it into OpImageFetch and the host/HW returns the raw texel. Yields a scalarized vec4.
fn lowerTexelFetch(l: *L, binding: u32, coord: Vector, lod: Value) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [24]u8 = undefined;
    const name = std.fmt.bufPrint(&namebuf, "tex.fetch.{d}", .{binding}) catch unreachable;
    const fetch = try l.func.appendCall(l.block, f32t, name, &.{ coord.comps[0], coord.comps[1], lod });
    var comps: [4]Value = undefined;
    inline for (0..4) |i| {
        comps[i] = try l.func.appendCall(l.block, f32t, "spirv.extract." ++ std.fmt.comptimePrint("{d}", .{i}), &.{fetch});
    }
    return .{ .vector = .{ .comps = comps, .len = 4 } };
}

/// `texelFetch(sampler2DArray / sampler3D, ivec3 P, int lod)`: an exact integer-coordinate fetch on
/// a layered/volume texture. Emits `tex.fetch3.<binding>` carrying the three i32 coords + the i32
/// lod; the emitter turns it into an OpImageFetch on the array/3D image. Yields a scalarized vec4.
fn lowerTexelFetch3(l: *L, binding: u32, coord: Vector, lod: Value) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [24]u8 = undefined;
    const name = std.fmt.bufPrint(&namebuf, "tex.fetch3.{d}", .{binding}) catch unreachable;
    const fetch = try l.func.appendCall(l.block, f32t, name, &.{ coord.comps[0], coord.comps[1], coord.comps[2], lod });
    var comps: [4]Value = undefined;
    inline for (0..4) |i| {
        comps[i] = try l.func.appendCall(l.block, f32t, "spirv.extract." ++ std.fmt.comptimePrint("{d}", .{i}), &.{fetch});
    }
    return .{ .vector = .{ .comps = comps, .len = 4 } };
}

/// `textureLod(sampler2DN, uv, lod)`: an EXPLICIT-LOD 2D sample. Emits a `tex.sampleLod.
/// <binding>` call carrying (u, v, lod); the emitter turns it into OpImageSampleExplicitLod
/// with the Lod image operand, and the host sampler reads that level directly. Yields a vec4.
fn lowerTextureLod(l: *L, binding: u32, uv: Vector, lod: Value) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [32]u8 = undefined;
    const sample_name = std.fmt.bufPrint(&namebuf, "tex.sampleLod.{d}", .{binding}) catch unreachable;
    const sample = try l.func.appendCall(l.block, f32t, sample_name, &.{ uv.comps[0], uv.comps[1], lod });
    var comps: [4]Value = undefined;
    inline for (0..4) |i| {
        comps[i] = try l.func.appendCall(l.block, f32t, "spirv.extract." ++ std.fmt.comptimePrint("{d}", .{i}), &.{sample});
    }
    return .{ .vector = .{ .comps = comps, .len = 4 } };
}

/// `textureCube(samplerCubeN, dir)`: like `lowerTexture` but the sample carries a THREE-
/// component direction (`tex.sample.cube.<binding>`), which the emitter turns into an
/// OpImageSampleImplicitLod on a Cube-dim image. Yields a scalarized vec4.
fn lowerTextureCube(l: *L, binding: u32, dir: Vector) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [32]u8 = undefined;
    const sample_name = std.fmt.bufPrint(&namebuf, "tex.sample.cube.{d}", .{binding}) catch unreachable;
    const sample = try l.func.appendCall(l.block, f32t, sample_name, &.{ dir.comps[0], dir.comps[1], dir.comps[2] });
    var comps: [4]Value = undefined;
    inline for (0..4) |i| {
        comps[i] = try l.func.appendCall(l.block, f32t, "spirv.extract." ++ std.fmt.comptimePrint("{d}", .{i}), &.{sample});
    }
    return .{ .vector = .{ .comps = comps, .len = 4 } };
}

/// `textureCubeLod(samplerCubeN, dir, lod)`: a cube sample at an EXPLICIT LOD (prefiltered
/// environment maps). Emits `tex.sample.cube.lod.<binding>` carrying (x, y, z, lod), which the
/// emitter turns into OpImageSampleExplicitLod on a Cube-dim image. Yields a scalarized vec4.
fn lowerTextureCubeLod(l: *L, binding: u32, dir: Vector, lod: Value) Error!Val {
    const f32t = try f32Type(l);
    var namebuf: [40]u8 = undefined;
    const sample_name = std.fmt.bufPrint(&namebuf, "tex.sample.cube.lod.{d}", .{binding}) catch unreachable;
    const sample = try l.func.appendCall(l.block, f32t, sample_name, &.{ dir.comps[0], dir.comps[1], dir.comps[2], lod });
    var comps: [4]Value = undefined;
    inline for (0..4) |i| {
        comps[i] = try l.func.appendCall(l.block, f32t, "spirv.extract." ++ std.fmt.comptimePrint("{d}", .{i}), &.{sample});
    }
    return .{ .vector = .{ .comps = comps, .len = 4 } };
}

/// A fragment-derivative intrinsic, lowered to an IR call the SPIR-V emitter maps to a
/// core op (OpDPdx/OpDPdy/OpFwidth). Fragment-shader only, so SPIR-V emission only.
fn dfdxElem(l: *L, x: Value) Error!Value {
    return l.func.appendCall(l.block, try f32Type(l), "spirv.dpdx", &.{x});
}
fn dfdyElem(l: *L, x: Value) Error!Value {
    return l.func.appendCall(l.block, try f32Type(l), "spirv.dpdy", &.{x});
}
fn fwidthElem(l: *L, x: Value) Error!Value {
    return l.func.appendCall(l.block, try f32Type(l), "spirv.fwidth", &.{x});
}

fn floorElem(l: *L, x: Value) Error!Value {
    return l.func.appendInst(l.block, try f32Type(l), .{ .unary = .{ .op = .floor, .value = x } });
}
fn ceilElem(l: *L, x: Value) Error!Value {
    return l.func.appendInst(l.block, try f32Type(l), .{ .unary = .{ .op = .ceil, .value = x } });
}
fn truncElem(l: *L, x: Value) Error!Value {
    return l.func.appendInst(l.block, try f32Type(l), .{ .unary = .{ .op = .trunc, .value = x } });
}
fn roundElem(l: *L, x: Value) Error!Value {
    return l.func.appendInst(l.block, try f32Type(l), .{ .unary = .{ .op = .nearest, .value = x } });
}
fn fractElem(l: *L, x: Value) Error!Value {
    return fsub(l, x, try floorElem(l, x)); // x - floor(x)
}
fn radiansElem(l: *L, x: Value) Error!Value {
    return fmul(l, x, try fconst(l, 0.017453292519943295)); // pi/180
}
fn degreesElem(l: *L, x: Value) Error!Value {
    return fmul(l, x, try fconst(l, 57.29577951308232)); // 180/pi
}
fn signElem(l: *L, x: Value) Error!Value {
    const z = try fconst(l, 0);
    const gt = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = z } });
    const lt = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = z } });
    const neg = try l.func.appendInst(l.block, try f32Type(l), .{ .select = .{ .cond = lt, .then = try fconst(l, -1), .@"else" = z } });
    return l.func.appendInst(l.block, try f32Type(l), .{ .select = .{ .cond = gt, .then = try fconst(l, 1), .@"else" = neg } });
}
fn stepElem(l: *L, edge: Value, x: Value) Error!Value {
    const lt = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = edge } });
    return l.func.appendInst(l.block, try f32Type(l), .{ .select = .{ .cond = lt, .then = try fconst(l, 0), .@"else" = try fconst(l, 1) } });
}
fn modElem(l: *L, x: Value, y: Value) Error!Value {
    const q = try l.func.appendInst(l.block, try f32Type(l), .{ .arith = .{ .op = .div, .lhs = x, .rhs = y } });
    return fsub(l, x, try fmul(l, y, try floorElem(l, q))); // x - y*floor(x/y)
}

/// smoothstep(e0, e1, x) = t*t*(3 - 2t), t = clamp((x-e0)/(e1-e0), 0, 1).
fn lowerSmoothstep(l: *L, e0: Val, e1: Val, x: Val) Error!Val {
    const len = @max(maxLen(e0, e1), if (x == .vector) x.vector.len else 1);
    const is_vec = (e0 == .vector or e1 == .vector or x == .vector);
    var out: [4]Value = undefined;
    for (0..len) |i| {
        const a0 = try laneValue(l, e0, i, .float);
        const a1 = try laneValue(l, e1, i, .float);
        const xi = try laneValue(l, x, i, .float);
        const num = try fsub(l, xi, a0);
        const den = try fsub(l, a1, a0);
        const d = try l.func.appendInst(l.block, try f32Type(l), .{ .arith = .{ .op = .div, .lhs = num, .rhs = den } });
        const t = try minMaxScalar(l, try minMaxScalar(l, d, try fconst(l, 0), .float, true), try fconst(l, 1), .float, false);
        const two_t = try fmul(l, try fconst(l, 2), t);
        const three_minus = try fsub(l, try fconst(l, 3), two_t);
        out[i] = try fmul(l, try fmul(l, t, t), three_minus);
    }
    if (!is_vec) return .{ .scalar = .{ .value = out[0], .ty = .float } };
    return .{ .vector = .{ .comps = out, .len = len } };
}

fn lowerSqrt(l: *L, v: Val) Error!Val {
    switch (v) {
        .scalar => |s| return .{ .scalar = .{ .value = try sqrtScalar(l, (try coerce(l, s, .float)).value), .ty = .float } },
        .vector => |vec| {
            var out: [4]Value = undefined;
            for (0..vec.len) |i| out[i] = try sqrtScalar(l, vec.comps[i]);
            return .{ .vector = .{ .comps = out, .len = vec.len } };
        },
        .matrix, .array, .structv => return error.Unsupported,
    }
}

/// `length(v)` = `sqrt(dot(v, v))` (a scalar `|x|` for a scalar argument).
fn lowerLength(l: *L, v: Val) Error!Val {
    switch (v) {
        .scalar => |s| return .{ .scalar = .{ .value = try absScalar(l, (try coerce(l, s, .float)).value, .float), .ty = .float } },
        .vector => {
            const d = (try dotProduct(l, v, v)).scalar.value;
            return .{ .scalar = .{ .value = try sqrtScalar(l, d), .ty = .float } };
        },
        .matrix, .array, .structv => return error.Unsupported,
    }
}

/// `normalize(v)` = `v / length(v)`, component-wise.
fn lowerNormalize(l: *L, v: Val) Error!Val {
    if (v != .vector) return error.Unsupported;
    const f32t = try f32Type(l);
    const d = (try dotProduct(l, v, v)).scalar.value;
    const len = try sqrtScalar(l, d);
    var out: [4]Value = undefined;
    for (0..v.vector.len) |i| out[i] = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .div, .lhs = v.vector.comps[i], .rhs = len } });
    return .{ .vector = .{ .comps = out, .len = v.vector.len } };
}

/// `p*q - r*s` in f32.
fn mulSub(l: *L, p: Value, q: Value, r: Value, s: Value) Error!Value {
    const f32t = try f32Type(l);
    const pq = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .mul, .lhs = p, .rhs = q } });
    const rs = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .mul, .lhs = r, .rhs = s } });
    return l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .sub, .lhs = pq, .rhs = rs } });
}

/// `matN(...)`: a single scalar makes a diagonal matrix, otherwise the arguments (scalars
/// and column vectors) fill the matrix column-major.
fn constructMatrix(l: *L, dim: u8, args: []const *parser.Expr) Error!Val {
    const total = @as(usize, dim) * dim;
    var comps: [16]Value = undefined;
    var n: usize = 0;
    for (args) |arg| {
        const v = try lowerExpr(l, arg);
        switch (v) {
            .scalar => |s| {
                if (n >= total) return error.Unsupported;
                comps[n] = (try coerce(l, s, .float)).value;
                n += 1;
            },
            .vector => |vec| for (0..vec.len) |i| {
                if (n >= total) return error.Unsupported;
                comps[n] = vec.comps[i];
                n += 1;
            },
            .matrix, .array, .structv => return error.Unsupported,
        }
    }
    if (n == 1 and total > 1) {
        const s = comps[0];
        const z = try fconst(l, 0);
        for (0..dim) |col| for (0..dim) |row| {
            comps[col * dim + row] = if (col == row) s else z;
        };
        n = total;
    }
    if (n != total) return error.TypeMismatch;
    return .{ .matrix = .{ .comps = comps, .dim = dim } };
}

/// `m * v`: result[row] = sum over columns of m[col][row] * v[col] (column-major matrix).
fn matVecMul(l: *L, m: Matrix, v: Vector) Error!Val {
    const dim = m.dim;
    var out: [4]Value = undefined;
    for (0..dim) |row| {
        var acc = try fmul(l, m.comps[0 * dim + row], v.comps[0]);
        for (1..dim) |col| acc = try l.func.appendInst(l.block, try f32Type(l), .{ .arith = .{ .op = .add, .lhs = acc, .rhs = try fmul(l, m.comps[col * dim + row], v.comps[col]) } });
        out[row] = acc;
    }
    return .{ .vector = .{ .comps = out, .len = dim } };
}

/// `a * b`: result[col][row] = sum over k of a[k][row] * b[col][k] (column-major).
fn matMatMul(l: *L, a: Matrix, b: Matrix) Error!Val {
    const dim = a.dim;
    var out: [16]Value = undefined;
    for (0..dim) |col| for (0..dim) |row| {
        var acc = try fmul(l, a.comps[0 * dim + row], b.comps[col * dim + 0]);
        for (1..dim) |k| acc = try l.func.appendInst(l.block, try f32Type(l), .{ .arith = .{ .op = .add, .lhs = acc, .rhs = try fmul(l, a.comps[k * dim + row], b.comps[col * dim + k]) } });
        out[col * dim + row] = acc;
    };
    return .{ .matrix = .{ .comps = out, .dim = dim } };
}

fn uconst(l: *L, v: i64) Error!Value {
    return l.func.appendInst(l.block, try irType(l.func, .uint), .{ .iconst = v });
}
fn uArith(l: *L, op: ir.function.BinOp, a: Value, b: Value) Error!Value {
    return l.func.appendInst(l.block, try irType(l.func, .uint), .{ .arith = .{ .op = op, .lhs = a, .rhs = b } });
}
fn uShr(l: *L, a: Value, n: i64) Error!Value {
    return uArith(l, .shr, a, try uconst(l, n));
}
fn uShl(l: *L, a: Value, n: i64) Error!Value {
    return uArith(l, .shl, a, try uconst(l, n));
}

/// One stage of the bit-reversal SWAR: swap adjacent groups of `w` bits selected by `mask`.
fn revStage(l: *L, u: Value, mask: i64, w: i64) Error!Value {
    const m = try uconst(l, mask);
    const lo = try uShl(l, try uArith(l, .bit_and, u, m), w);
    const hi = try uArith(l, .bit_and, try uShr(l, u, w), m);
    return uArith(l, .bit_or, lo, hi);
}

/// Reverse the 32 bits of a scalar's value, returning it in component type `to`.
fn bitfieldReverseScalar(l: *L, x: Value, to: Type) Error!Value {
    var u = try l.func.appendInst(l.block, try irType(l.func, .uint), .{ .unary = .{ .op = .reinterpret, .value = x } });
    u = try revStage(l, u, 0x55555555, 1);
    u = try revStage(l, u, 0x33333333, 2);
    u = try revStage(l, u, 0x0F0F0F0F, 4);
    u = try revStage(l, u, 0x00FF00FF, 8);
    u = try uArith(l, .bit_or, try uShl(l, u, 16), try uShr(l, u, 16)); // swap 16-bit halves
    return l.func.appendInst(l.block, try irType(l.func, to), .{ .unary = .{ .op = .reinterpret, .value = u } });
}

/// Lower an argument expression to a scalar `int` Value (for offset/bits operands).
fn intArg(l: *L, e: *parser.Expr) Error!Value {
    const v = try lowerExpr(l, e);
    if (v != .scalar) return error.TypeMismatch;
    return (try coerce(l, v.scalar, .int)).value;
}

/// `bitfieldExtract(value, offset, bits)`: extract `bits` bits starting at `offset`,
/// zero-extended for unsigned or sign-extended for signed. Both directions use a
/// left-then-right double shift (the right shift is arithmetic for a signed value type),
/// with a `bits == 0` guard returning 0 (the shift-by-32 that case implies is undefined).
fn bitfieldExtractScalar(l: *L, value: Value, ty: Type, offset: Value, bits: Value) Error!Value {
    const t = try irType(l.func, ty);
    const c32 = try iconstI(l, 32);
    const shl_amt = try iArith(l, .sub, try iArith(l, .sub, c32, offset), bits); // 32 - offset - bits
    const shr_amt = try iArith(l, .sub, c32, bits); // 32 - bits
    const up = try l.func.appendInst(l.block, t, .{ .arith = .{ .op = .shl, .lhs = value, .rhs = shl_amt } });
    const res = try l.func.appendInst(l.block, t, .{ .arith = .{ .op = .shr, .lhs = up, .rhs = shr_amt } });
    const is_zero = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .eq, .lhs = bits, .rhs = try iconstI(l, 0) } });
    const zero_t = try l.func.appendInst(l.block, t, .{ .iconst = 0 });
    return l.func.appendInst(l.block, t, .{ .select = .{ .cond = is_zero, .then = zero_t, .@"else" = res } });
}

fn lowerBitfieldExtract(l: *L, args: []const *parser.Expr) Error!Val {
    if (args.len != 3) return error.Unsupported;
    const v = try lowerExpr(l, args[0]);
    const offset = try intArg(l, args[1]);
    const bits = try intArg(l, args[2]);
    switch (v) {
        .scalar => |s| return .{ .scalar = .{ .value = try bitfieldExtractScalar(l, s.value, s.ty, offset, bits), .ty = s.ty } },
        .vector => |vec| {
            var out: [4]Value = undefined;
            for (0..vec.len) |i| out[i] = try bitfieldExtractScalar(l, vec.comps[i], vec.comp_ty, offset, bits);
            return .{ .vector = .{ .comps = out, .len = vec.len, .comp_ty = vec.comp_ty } };
        },
        else => return error.Unsupported,
    }
}

/// `bitfieldInsert(base, insert, offset, bits)`: replace `bits` bits of `base` at `offset`
/// with the low bits of `insert`. Builds the field mask as `(0xFFFFFFFF >> (32-bits)) <<
/// offset`, with a `bits == 0` guard returning `base` unchanged.
fn bitfieldInsertScalar(l: *L, base: Value, insert: Value, ty: Type, offset: Value, bits: Value) Error!Value {
    const uintt = try irType(l.func, .uint);
    const shr_amt = try iArith(l, .sub, try iconstI(l, 32), bits); // 32 - bits
    const lowmask = try l.func.appendInst(l.block, uintt, .{ .arith = .{ .op = .shr, .lhs = try uconst(l, 0xFFFFFFFF), .rhs = shr_amt } });
    const mask = try l.func.appendInst(l.block, uintt, .{ .arith = .{ .op = .shl, .lhs = lowmask, .rhs = offset } });
    const base_u = try l.func.appendInst(l.block, uintt, .{ .unary = .{ .op = .reinterpret, .value = base } });
    const insert_u = try l.func.appendInst(l.block, uintt, .{ .unary = .{ .op = .reinterpret, .value = insert } });
    const cleared = try uArith(l, .bit_and, base_u, try uArith(l, .bit_xor, mask, try uconst(l, 0xFFFFFFFF)));
    const ins_shifted = try l.func.appendInst(l.block, uintt, .{ .arith = .{ .op = .shl, .lhs = insert_u, .rhs = offset } });
    const merged_u = try uArith(l, .bit_or, cleared, try uArith(l, .bit_and, ins_shifted, mask));
    const t = try irType(l.func, ty);
    const merged = try l.func.appendInst(l.block, t, .{ .unary = .{ .op = .reinterpret, .value = merged_u } });
    const is_zero = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .eq, .lhs = bits, .rhs = try iconstI(l, 0) } });
    return l.func.appendInst(l.block, t, .{ .select = .{ .cond = is_zero, .then = base, .@"else" = merged } });
}

fn lowerBitfieldInsert(l: *L, args: []const *parser.Expr) Error!Val {
    if (args.len != 4) return error.Unsupported;
    const base = try lowerExpr(l, args[0]);
    const insert = try lowerExpr(l, args[1]);
    const offset = try intArg(l, args[2]);
    const bits = try intArg(l, args[3]);
    if (base == .scalar and insert == .scalar) {
        return .{ .scalar = .{ .value = try bitfieldInsertScalar(l, base.scalar.value, insert.scalar.value, base.scalar.ty, offset, bits), .ty = base.scalar.ty } };
    }
    if (base == .vector and insert == .vector and base.vector.len == insert.vector.len) {
        var out: [4]Value = undefined;
        for (0..base.vector.len) |i| out[i] = try bitfieldInsertScalar(l, base.vector.comps[i], insert.vector.comps[i], base.vector.comp_ty, offset, bits);
        return .{ .vector = .{ .comps = out, .len = base.vector.len, .comp_ty = base.vector.comp_ty } };
    }
    return error.TypeMismatch;
}

/// `bitfieldReverse(x)`: bit-reverse a scalar or (component-wise) vector, preserving type.
fn bitfieldReverseVal(l: *L, v: Val) Error!Val {
    switch (v) {
        .scalar => |s| return .{ .scalar = .{ .value = try bitfieldReverseScalar(l, s.value, s.ty), .ty = s.ty } },
        .vector => |vec| {
            var out: [4]Value = undefined;
            for (0..vec.len) |i| out[i] = try bitfieldReverseScalar(l, vec.comps[i], vec.comp_ty);
            return .{ .vector = .{ .comps = out, .len = vec.len, .comp_ty = vec.comp_ty } };
        },
        else => return error.Unsupported,
    }
}

/// Population count of a 32-bit value via the classic SWAR bit-twiddle. Operates on the
/// bits as unsigned (so the shifts are logical) and returns an int count in [0, 32].
fn bitCountScalar(l: *L, x: Value) Error!Value {
    const uintt = try irType(l.func, .uint);
    var u = try l.func.appendInst(l.block, uintt, .{ .unary = .{ .op = .reinterpret, .value = x } });
    // u = u - ((u >> 1) & 0x55555555)
    u = try uArith(l, .sub, u, try uArith(l, .bit_and, try uShr(l, u, 1), try uconst(l, 0x55555555)));
    // u = (u & 0x33333333) + ((u >> 2) & 0x33333333)
    u = try uArith(l, .add, try uArith(l, .bit_and, u, try uconst(l, 0x33333333)), try uArith(l, .bit_and, try uShr(l, u, 2), try uconst(l, 0x33333333)));
    // u = (u + (u >> 4)) & 0x0F0F0F0F
    u = try uArith(l, .bit_and, try uArith(l, .add, u, try uShr(l, u, 4)), try uconst(l, 0x0F0F0F0F));
    // count = (u * 0x01010101) >> 24
    u = try uShr(l, try uArith(l, .mul, u, try uconst(l, 0x01010101)), 24);
    return l.func.appendInst(l.block, try irType(l.func, .int), .{ .unary = .{ .op = .reinterpret, .value = u } });
}

/// Clamp a float value to [lo, 1]: `min(max(x, lo), 1)`.
fn clampNorm(l: *L, x: Value, lo: f64) Error!Value {
    const m = try minMaxScalar(l, x, try fconst(l, lo), .float, true); // max(x, lo)
    return minMaxScalar(l, m, try fconst(l, 1), .float, false); // min(_, 1)
}

/// The fixed-point scale for a normalized pack: `2^bits - 1` (unorm) or `2^(bits-1) - 1`
/// (snorm), e.g. 255/65535 or 127/32767.
fn normScale(bits: u8, snorm: bool) f64 {
    const one: u64 = 1;
    return @floatFromInt(if (snorm) (one << @intCast(bits - 1)) - 1 else (one << @intCast(bits)) - 1);
}

/// Generic `pack{U,S}norm{4x8,2x16}`: clamp each of `comps` lanes to [lo,1], scale, round,
/// and pack into a uint field of `bits` bits (lane 0 in the low bits). Snorm lanes clamp to
/// [-1,1] and pack a signed two's-complement field.
fn packNorm(l: *L, v: Val, comps: u8, bits: u8, snorm: bool) Error!Val {
    if (v != .vector or v.vector.len != comps) return error.TypeMismatch;
    const f32t = try f32Type(l);
    const uintt = try irType(l.func, .uint);
    const scale = try fconst(l, normScale(bits, snorm));
    const lo: f64 = if (snorm) -1 else 0;
    const field_mask: i64 = if (bits == 8) 0xFF else 0xFFFF;
    var acc: ?Value = null;
    for (0..comps) |i| {
        const scaled = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .mul, .lhs = try clampNorm(l, v.vector.comps[i], lo), .rhs = scale } });
        const rounded = try l.func.appendInst(l.block, f32t, .{ .unary = .{ .op = .nearest, .value = scaled } });
        // Snorm converts to signed (so negatives are correct) then reinterprets to uint.
        const as_u = if (snorm)
            try l.func.appendInst(l.block, uintt, .{ .unary = .{ .op = .reinterpret, .value = try l.func.appendInst(l.block, try irType(l.func, .int), .{ .convert = .{ .value = rounded } }) } })
        else
            try l.func.appendInst(l.block, uintt, .{ .convert = .{ .value = rounded } });
        const masked = try uArith(l, .bit_and, as_u, try uconst(l, field_mask));
        const shifted = if (i == 0) masked else try uShl(l, masked, @intCast(bits * i));
        acc = if (acc) |a| try uArith(l, .bit_or, a, shifted) else shifted;
    }
    return .{ .scalar = .{ .value = acc.?, .ty = .uint } };
}

/// Generic `unpack{U,S}norm{4x8,2x16}`: extract each `bits`-wide field, convert to float,
/// and divide by the scale. Snorm sign-extends the field and clamps the result to [-1,1].
fn unpackNorm(l: *L, v: Val, comps: u8, bits: u8, snorm: bool) Error!Val {
    if (v != .scalar) return error.TypeMismatch;
    const f32t = try f32Type(l);
    const uintt = try irType(l.func, .uint);
    const intt = try irType(l.func, .int);
    const u = try l.func.appendInst(l.block, uintt, .{ .unary = .{ .op = .reinterpret, .value = v.scalar.value } });
    const scale = try fconst(l, normScale(bits, snorm));
    const field_mask: i64 = if (bits == 8) 0xFF else 0xFFFF;
    const sx: i64 = 32 - @as(i64, bits); // shift amount for sign extension
    var out: [4]Value = undefined;
    for (0..comps) |i| {
        const shifted = if (i == 0) u else try uShr(l, u, @intCast(bits * i));
        const field = try uArith(l, .bit_and, shifted, try uconst(l, field_mask));
        var f: Value = undefined;
        if (snorm) {
            const fi = try l.func.appendInst(l.block, intt, .{ .unary = .{ .op = .reinterpret, .value = field } });
            const up = try iArith(l, .shl, fi, try iconstI(l, sx));
            const sext = try iArith(l, .shr, up, try iconstI(l, sx)); // arithmetic: sign-extend
            const ff = try l.func.appendInst(l.block, f32t, .{ .convert = .{ .value = sext } });
            f = try clampNorm(l, try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .div, .lhs = ff, .rhs = scale } }), -1);
        } else {
            const ff = try l.func.appendInst(l.block, f32t, .{ .convert = .{ .value = field } });
            f = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .div, .lhs = ff, .rhs = scale } });
        }
        out[i] = f;
    }
    return .{ .vector = .{ .comps = out, .len = comps, .comp_ty = .float } };
}

fn iconstI(l: *L, v: i64) Error!Value {
    return l.func.appendInst(l.block, try irType(l.func, .int), .{ .iconst = v });
}
fn iArith(l: *L, op: ir.function.BinOp, a: Value, b: Value) Error!Value {
    return l.func.appendInst(l.block, try irType(l.func, .int), .{ .arith = .{ .op = op, .lhs = a, .rhs = b } });
}

/// `findLSB(x)`: index of the least-significant set bit, or -1 for 0. Isolate the lowest
/// set bit (`x & -x`), then its position is `bitCount(isolated - 1)`.
fn findLSBScalar(l: *L, x: Value) Error!Value {
    const intt = try irType(l.func, .int);
    const zero_i = try iconstI(l, 0);
    const isolate = try iArith(l, .bit_and, x, try iArith(l, .sub, zero_i, x)); // x & -x
    const idx = try bitCountScalar(l, try iArith(l, .sub, isolate, try iconstI(l, 1)));
    const is_zero = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = zero_i } });
    return l.func.appendInst(l.block, intt, .{ .select = .{ .cond = is_zero, .then = try iconstI(l, -1), .@"else" = idx } });
}

/// Index of the most-significant set bit of `x`'s bits (unsigned), or -1 for 0. Smear all
/// bits below the top set bit down, then the index is `bitCount(smeared) - 1`.
fn findMSBu(l: *L, x: Value) Error!Value {
    var u = try l.func.appendInst(l.block, try irType(l.func, .uint), .{ .unary = .{ .op = .reinterpret, .value = x } });
    inline for (.{ 1, 2, 4, 8, 16 }) |sh| u = try uArith(l, .bit_or, u, try uShr(l, u, sh));
    return iArith(l, .sub, try bitCountScalar(l, u), try iconstI(l, 1));
}

/// `findMSB(x)`: unsigned uses `findMSBu`; signed uses the most significant bit differing
/// from the sign bit, i.e. `findMSBu(x < 0 ? ~x : x)` (so 0 and -1 both give -1).
fn findMSBScalar(l: *L, x: Value, signed: bool) Error!Value {
    if (!signed) return findMSBu(l, x);
    const zero_i = try iconstI(l, 0);
    const cond = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = zero_i } });
    const notx = try iArith(l, .bit_xor, x, try iconstI(l, -1));
    const y = try l.func.appendInst(l.block, try irType(l.func, .int), .{ .select = .{ .cond = cond, .then = notx, .@"else" = x } });
    return findMSBu(l, y);
}

/// `findLSB`/`findMSB` over a scalar or (component-wise) vector, returning int/ivec. `msb`
/// selects the scan direction; signedness (for findMSB) comes from the value's type.
fn bitScanVal(l: *L, v: Val, msb: bool) Error!Val {
    switch (v) {
        .scalar => |s| {
            const r = if (msb) try findMSBScalar(l, s.value, s.ty == .int) else try findLSBScalar(l, s.value);
            return .{ .scalar = .{ .value = r, .ty = .int } };
        },
        .vector => |vec| {
            const signed = vec.comp_ty == .int;
            var out: [4]Value = undefined;
            for (0..vec.len) |i| out[i] = if (msb) try findMSBScalar(l, vec.comps[i], signed) else try findLSBScalar(l, vec.comps[i]);
            return .{ .vector = .{ .comps = out, .len = vec.len, .comp_ty = .int } };
        },
        else => return error.Unsupported,
    }
}

/// `bitCount(x)`: population count of a scalar or (component-wise) vector, returning int/ivec.
fn bitCountVal(l: *L, v: Val) Error!Val {
    switch (v) {
        .scalar => |s| return .{ .scalar = .{ .value = try bitCountScalar(l, s.value), .ty = .int } },
        .vector => |vec| {
            var out: [4]Value = undefined;
            for (0..vec.len) |i| out[i] = try bitCountScalar(l, vec.comps[i]);
            return .{ .vector = .{ .comps = out, .len = vec.len, .comp_ty = .int } };
        },
        else => return error.Unsupported,
    }
}

/// Reinterpret a scalar or vector's bits as component type `to` (same 32-bit width). Powers
/// floatBitsToInt / intBitsToFloat / floatBitsToUint / uintBitsToFloat.
fn reinterpretVal(l: *L, v: Val, to: Type) Error!Val {
    const t = try irType(l.func, to);
    switch (v) {
        .scalar => |s| return .{ .scalar = .{ .value = try l.func.appendInst(l.block, t, .{ .unary = .{ .op = .reinterpret, .value = s.value } }), .ty = to } },
        .vector => |vec| {
            var out: [4]Value = undefined;
            for (0..vec.len) |i| out[i] = try l.func.appendInst(l.block, t, .{ .unary = .{ .op = .reinterpret, .value = vec.comps[i] } });
            return .{ .vector = .{ .comps = out, .len = vec.len, .comp_ty = to } };
        },
        else => return error.Unsupported,
    }
}

/// Map a GLSL vector relational function name to its IR compare op, or null.
fn relCmpOp(name: []const u8) ?ir.function.CmpOp {
    if (std.mem.eql(u8, name, "lessThan")) return .lt;
    if (std.mem.eql(u8, name, "lessThanEqual")) return .le;
    if (std.mem.eql(u8, name, "greaterThan")) return .gt;
    if (std.mem.eql(u8, name, "greaterThanEqual")) return .ge;
    if (std.mem.eql(u8, name, "equal")) return .eq;
    if (std.mem.eql(u8, name, "notEqual")) return .ne;
    return null;
}

/// Component-wise compare of two vectors, producing a bvec (bool per lane). The `icmp`
/// opcode is generic: the backend picks integer or float compare from the operand type.
fn vectorCompare(l: *L, co: ir.function.CmpOp, a: Val, b: Val) Error!Val {
    if (a != .vector or b != .vector) return error.Unsupported;
    if (a.vector.len != b.vector.len) return error.TypeMismatch;
    const bool_t = try irType(l.func, .bool);
    var out: [4]Value = undefined;
    for (0..a.vector.len) |i| {
        out[i] = try l.func.appendInst(l.block, bool_t, .{ .icmp = .{ .op = co, .lhs = a.vector.comps[i], .rhs = b.vector.comps[i] } });
    }
    return .{ .vector = .{ .comps = out, .len = a.vector.len, .comp_ty = .bool } };
}

/// `all(bvec)`/`any(bvec)`: reduce the 0/1 bool lanes with bit-and / bit-or to one bool.
fn reduceBvec(l: *L, v: Val, is_all: bool) Error!Val {
    if (v != .vector or v.vector.comp_ty != .bool) return error.Unsupported;
    const bool_t = try irType(l.func, .bool);
    const op: ir.function.BinOp = if (is_all) .bit_and else .bit_or;
    var acc = v.vector.comps[0];
    for (1..v.vector.len) |i| {
        acc = try l.func.appendInst(l.block, bool_t, .{ .arith = .{ .op = op, .lhs = acc, .rhs = v.vector.comps[i] } });
    }
    return .{ .scalar = .{ .value = acc, .ty = .bool } };
}

/// `not(bvec)`: component-wise logical negation of the 0/1 bool lanes (xor with 1).
fn notBvec(l: *L, v: Val) Error!Val {
    if (v != .vector or v.vector.comp_ty != .bool) return error.Unsupported;
    const bool_t = try irType(l.func, .bool);
    const one = try l.func.appendInst(l.block, bool_t, .{ .iconst = 1 });
    var out: [4]Value = undefined;
    for (0..v.vector.len) |i| {
        out[i] = try l.func.appendInst(l.block, bool_t, .{ .arith = .{ .op = .bit_xor, .lhs = v.vector.comps[i], .rhs = one } });
    }
    return .{ .vector = .{ .comps = out, .len = v.vector.len, .comp_ty = .bool } };
}

fn constructVector(l: *L, len: u8, comp_ty: Type, args: []const *parser.Expr) Error!Val {
    var comps: [4]Value = undefined;
    var n: u8 = 0;
    for (args) |arg| {
        const v = try lowerExpr(l, arg);
        switch (v) {
            .scalar => |s| {
                if (n >= 4) return error.Unsupported;
                comps[n] = (try coerce(l, s, comp_ty)).value;
                n += 1;
            },
            .vector => |vec| {
                for (0..vec.len) |i| {
                    // GLSL truncation: vecN(vecM) with M > N drops the trailing components,
                    // e.g. vec3(someVec4) keeps xyz. Once we have `len` components, stop
                    // consuming this (oversized) vector rather than erroring. A source lane
                    // of a different component type is converted to the target type.
                    if (n >= len) break;
                    comps[n] = if (vec.comp_ty == comp_ty)
                        vec.comps[i]
                    else
                        (try coerce(l, .{ .value = vec.comps[i], .ty = vec.comp_ty }, comp_ty)).value;
                    n += 1;
                }
            },
            .matrix, .array, .structv => return error.Unsupported,
        }
    }
    if (n == 1 and len > 1) {
        // Splat: vec3(x) -> (x, x, x).
        for (1..len) |i| comps[i] = comps[0];
        n = len;
    }
    if (n != len) return error.TypeMismatch;
    return .{ .vector = .{ .comps = comps, .len = len, .comp_ty = comp_ty } };
}

/// `.field` on a value: a struct member access (resolves the named field) or a vector
/// swizzle. The struct path keys off the lowered value being a struct instance.
fn lowerSwizzleOrMember(l: *L, value: *parser.Expr, field: []const u8) Error!Val {
    const v = try lowerExpr(l, value);
    if (v == .structv) {
        const idx = structFieldIndex(l, v.structv.def, field) orelse return error.UndefinedName;
        return v.structv.fields[idx];
    }
    if (v != .vector) return error.BadSwizzle;
    if (field.len == 0 or field.len > 4) return error.BadSwizzle;
    var comps: [4]Value = undefined;
    for (field, 0..) |ch, i| {
        const idx = swizzleIndex(ch) orelse return error.BadSwizzle;
        if (idx >= v.vector.len) return error.BadSwizzle;
        comps[i] = v.vector.comps[idx];
    }
    if (field.len == 1) return .{ .scalar = .{ .value = comps[0], .ty = v.vector.comp_ty } };
    return .{ .vector = .{ .comps = comps, .len = @intCast(field.len), .comp_ty = v.vector.comp_ty } };
}

fn swizzleIndex(c: u8) ?u8 {
    return switch (c) {
        'x', 'r', 's' => 0,
        'y', 'g', 't' => 1,
        'z', 'b', 'p' => 2,
        'w', 'a', 'q' => 3,
        else => null,
    };
}

/// The index of field `name` in struct `def`, or null if there is no such field.
fn structFieldIndex(l: *L, def: []const u8, name: []const u8) ?usize {
    const sd = l.structDef(def) orelse return null;
    for (sd.fields, 0..) |f, i| if (std.mem.eql(u8, f.name, name)) return i;
    return null;
}

/// A zero-initialized composite Val for `struct_name`/`array_len` (else the scalar/vector/
/// matrix zero of `ty`). Recurses for nested structs and arrays-of-structs.
fn defaultComposite(l: *L, ty: Type, struct_name: ?[]const u8, array_len: ?u32) Error!Val {
    if (array_len) |len| {
        const elems = try l.comp_arena.alloc(Val, len);
        for (0..len) |i| elems[i] = try defaultComposite(l, ty, struct_name, null);
        return .{ .array = .{ .elems = elems } };
    }
    if (struct_name) |sn| {
        const sd = l.structDef(sn) orelse return error.UndefinedName;
        const fields = try l.comp_arena.alloc(Val, sd.fields.len);
        for (sd.fields, 0..) |f, i| fields[i] = try defaultComposite(l, f.ty, f.struct_name, f.array_len);
        return .{ .structv = .{ .def = sn, .fields = fields } };
    }
    return zeroVal(l, ty);
}

/// `StructName(args...)`: build a struct instance, binding each argument Val to the field
/// in declaration order (coerced to the field's type). Vector/struct fields take a matching
/// composite argument.
fn lowerStructCtor(l: *L, name: []const u8, args: []const *parser.Expr) Error!Val {
    const sd = l.structDef(name) orelse return error.UndefinedName;
    if (args.len != sd.fields.len) return error.TypeMismatch;
    const fields = try l.comp_arena.alloc(Val, sd.fields.len);
    for (sd.fields, 0..) |f, i| {
        const v = try lowerExpr(l, args[i]);
        // A struct/array field passes its composite through. A scalar/vector/matrix field
        // coerces numerically (matching the existing decl coercion).
        fields[i] = if (f.struct_name != null or f.array_len != null) v else try coerceVal(l, v, f.ty);
    }
    return .{ .structv = .{ .def = name, .fields = fields } };
}

/// Store `value` into a (possibly composite) lvalue expression: `a[i] = v`,
/// `a[i].field = v`, `s.field = v`, `a[i].field.xy = v`. Resolves the chain to the root
/// variable, rebuilds the modified composite, and rebinds the root in the env (SSA-style,
/// mirroring how a plain `assign` rebinds a name).
fn lowerStore(l: *L, target: *parser.Expr, value: *parser.Expr) Error!void {
    const v = try lowerExpr(l, value);
    const root_name = lvalueRoot(target) orelse return error.Unsupported;
    const slot = l.lookup(root_name) orelse return error.UndefinedName;
    slot.val = try rebuildLvalue(l, target, v);
}

/// The root identifier name of an lvalue chain (`a[i].f.xy` -> "a").
fn lvalueRoot(e: *parser.Expr) ?[]const u8 {
    return switch (e.*) {
        .ident => |n| n,
        .index => |ix| lvalueRoot(ix.value),
        .swizzle => |s| lvalueRoot(s.value),
        else => null,
    };
}

/// Compute the new value of the ROOT variable after assigning `new_val` to the location
/// named by lvalue `target`. Walks the chain: for `ident` the whole root is replaced. For
/// an `index`/`.field`, the parent container's current value (via `lowerExpr` on the parent
/// expression) is copied with the one element / member / swizzle updated, and the recursion
/// continues up the parent with the rebuilt container.
fn rebuildLvalue(l: *L, target: *parser.Expr, new_val: Val) Error!Val {
    switch (target.*) {
        .ident => {
            const cur = try lowerExpr(l, target); // for its declared type (coercion target)
            return coerceVal(l, new_val, valType(cur));
        },
        .index => |ix| {
            const container = try lowerExpr(l, ix.value);
            if (container != .array) return error.Unsupported;
            const idx = try constIndex(l, ix.index);
            if (idx >= container.array.elems.len) return error.TypeMismatch;
            const elems = try l.comp_arena.alloc(Val, container.array.elems.len);
            @memcpy(elems, container.array.elems);
            elems[idx] = new_val;
            return rebuildLvalue(l, ix.value, .{ .array = .{ .elems = elems } });
        },
        .swizzle => |s| {
            const container = try lowerExpr(l, s.value);
            if (container == .structv) {
                const fi = structFieldIndex(l, container.structv.def, s.field) orelse return error.UndefinedName;
                const fields = try l.comp_arena.alloc(Val, container.structv.fields.len);
                @memcpy(fields, container.structv.fields);
                fields[fi] = new_val;
                return rebuildLvalue(l, s.value, .{ .structv = .{ .def = container.structv.def, .fields = fields } });
            }
            if (container != .vector) return error.BadSwizzle;
            const updated = try applySwizzleWrite(l, container.vector, s.field, new_val);
            return rebuildLvalue(l, s.value, .{ .vector = updated });
        },
        else => return error.Unsupported,
    }
}

/// Apply a swizzle write to a vector value, returning the updated vector.
fn applySwizzleWrite(l: *L, vec_in: Vector, field: []const u8, new_val: Val) Error!Vector {
    var vec = vec_in;
    if (field.len == 1) {
        if (new_val != .scalar) return error.TypeMismatch;
        const idx = swizzleIndex(field[0]) orelse return error.BadSwizzle;
        if (idx >= vec.len) return error.BadSwizzle;
        vec.comps[idx] = (try coerce(l, new_val.scalar, vec.comp_ty)).value;
    } else {
        if (new_val != .vector or new_val.vector.len != field.len) return error.TypeMismatch;
        for (field, 0..) |ch, i| {
            const idx = swizzleIndex(ch) orelse return error.BadSwizzle;
            if (idx >= vec.len) return error.BadSwizzle;
            vec.comps[idx] = new_val.vector.comps[i];
        }
    }
    return vec;
}

/// Make two scalar operands share a type (float wins over int via int->float convert).
fn unify(l: *L, a: *Scalar, b: *Scalar) Error!Type {
    if (a.ty == b.ty) return a.ty;
    if (a.ty == .float and isInt(b.ty)) {
        b.* = try coerce(l, b.*, .float);
        return .float;
    }
    if (b.ty == .float and isInt(a.ty)) {
        a.* = try coerce(l, a.*, .float);
        return .float;
    }
    if (isInt(a.ty) and isInt(b.ty)) return .int;
    return error.TypeMismatch;
}

/// Coerce a whole Val to a target GLSL type. Scalars convert numerically, vectors must
/// already match.
fn coerceVal(l: *L, v: Val, to: Type) Error!Val {
    // A composite (array/struct) has no scalar target type. Pass it through unchanged
    // (the target was tagged `.void` by `valType`).
    if (v == .array or v == .structv) return v;
    if (matDim(to) != 0) {
        if (v != .matrix or v.matrix.dim != matDim(to)) return error.TypeMismatch;
        return v;
    }
    if (vecLen(to) != 0) {
        if (v != .vector or v.vector.len != vecLen(to)) return error.TypeMismatch;
        return v;
    }
    if (v != .scalar) return error.TypeMismatch;
    return .{ .scalar = try coerce(l, v.scalar, to) };
}

/// Convert a scalar to GLSL type `to`. int<->float go through the IR `convert`. bool
/// <-> int are explicit (`select 1/0` and `!= 0`), int<->uint just relabel.
fn coerce(l: *L, s: Scalar, to: Type) Error!Scalar {
    if (s.ty == to) return s;
    if (s.ty == .bool and isInt(to)) {
        const t = try irType(l.func, to);
        const one = try l.func.appendInst(l.block, t, .{ .iconst = 1 });
        const zero_v = try l.func.appendInst(l.block, t, .{ .iconst = 0 });
        return .{ .value = try l.func.appendInst(l.block, t, .{ .select = .{ .cond = s.value, .then = one, .@"else" = zero_v } }), .ty = to };
    }
    if (isInt(s.ty) and to == .bool) {
        const z = try l.func.appendInst(l.block, try irType(l.func, s.ty), .{ .iconst = 0 });
        return .{ .value = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = .ne, .lhs = s.value, .rhs = z } }), .ty = .bool };
    }
    const is_float_to = (to == .float);
    const is_float_from = (s.ty == .float);
    if (is_float_to != is_float_from) {
        return .{ .value = try l.func.appendInst(l.block, try irType(l.func, to), .{ .convert = .{ .value = s.value } }), .ty = to };
    }
    // int <-> uint: same bits, different IR signedness. A bare relabel would keep the
    // signed IR value, so the backend would pick signed shift/div/compare. Reinterpret so
    // the value carries the target signedness (the machine op is a no-op same-width move).
    return .{ .value = try l.func.appendInst(l.block, try irType(l.func, to), .{ .unary = .{ .op = .reinterpret, .value = s.value } }), .ty = to };
}

fn zero(l: *L, ty: Type) Error!Value {
    return switch (ty) {
        .float => l.func.appendInst(l.block, try irType(l.func, ty), .{ .fconst = 0 }),
        else => l.func.appendInst(l.block, try irType(l.func, ty), .{ .iconst = 0 }),
    };
}

/// A zero-initialized Val of the given type (a scalar, a zero vector, or a zero matrix),
/// for an uninitialized local declaration like `vec4 result;`.
fn zeroVal(l: *L, ty: Type) Error!Val {
    const md = matDim(ty);
    if (md != 0) {
        const f32t = try f32Type(l);
        var comps: [16]Value = undefined;
        for (0..@as(usize, md) * md) |i| comps[i] = try l.func.appendInst(l.block, f32t, .{ .fconst = 0 });
        return .{ .matrix = .{ .comps = comps, .dim = md } };
    }
    const n = vecLen(ty);
    if (n != 0) {
        const comp_ty = vecCompType(ty);
        var comps: [4]Value = undefined;
        for (0..n) |i| comps[i] = try zero(l, comp_ty);
        return .{ .vector = .{ .comps = comps, .len = n, .comp_ty = comp_ty } };
    }
    return .{ .scalar = .{ .value = try zero(l, ty), .ty = ty } };
}

fn isInt(ty: Type) bool {
    return ty == .int or ty == .uint;
}

/// Whether any top-level statement assigns to `name` (detects a `gl_Position` write).
fn assignsName(body: []const parser.Stmt, name: []const u8) bool {
    for (body) |stmt| if (stmt.kind == .assign and std.mem.eql(u8, stmt.kind.assign.name, name)) return true;
    return false;
}

/// Whether any expression in the body reads `name` (used to detect input builtins).
fn bodyReferences(body: []const parser.Stmt, name: []const u8) bool {
    for (body) |stmt| switch (stmt.kind) {
        .ret => |m| if (m) |e| {
            if (exprReferences(e, name)) return true;
        },
        .decl => |d| if (d.value) |e| {
            if (exprReferences(e, name)) return true;
        },
        .assign => |a| if (exprReferences(a.value, name)) return true,
        .swizzle_assign => |sa| if (exprReferences(sa.value, name)) return true,
        .store => |st| if (exprReferences(st.target, name) or exprReferences(st.value, name)) return true,
        .break_, .continue_, .discard_ => {},
        .expr => |x| if (exprReferences(x, name)) return true,
        .if_ => |iff| if (exprReferences(iff.cond, name) or bodyReferences(iff.then, name) or bodyReferences(iff.@"else", name)) return true,
        .for_ => |f| {
            if (f.cond) |c| if (exprReferences(c, name)) return true;
            if (bodyReferences(f.init, name) or bodyReferences(f.incr, name) or bodyReferences(f.body, name)) return true;
        },
    };
    return false;
}

fn exprReferences(e: *parser.Expr, name: []const u8) bool {
    return switch (e.*) {
        .ident => |n| std.mem.eql(u8, n, name),
        .unary => |u| exprReferences(u.operand, name),
        .binary => |b| exprReferences(b.lhs, name) or exprReferences(b.rhs, name),
        .swizzle => |s| exprReferences(s.value, name),
        .ternary => |t| exprReferences(t.cond, name) or exprReferences(t.then, name) or exprReferences(t.@"else", name),
        .call => |c| {
            for (c.args) |a| if (exprReferences(a, name)) return true;
            return false;
        },
        .index => |ix| exprReferences(ix.value, name) or exprReferences(ix.index, name),
        .struct_ctor => |sc| {
            for (sc.args) |a| if (exprReferences(a, name)) return true;
            return false;
        },
        .float_lit, .int_lit, .bool_lit => false,
    };
}

test "lowering tags instructions with their source line for debug info" {
    const a = std.testing.allocator;
    // The declaration is on line 2, the return on line 3.
    const src = "int f(int a) {\n  int b = a + 5;\n  return b * 2;\n}";
    var module = try compile(a, src);
    defer module.deinit(a);
    const func = module.find("f") orelse return error.MissingFunction;

    var saw_line2 = false;
    var saw_line3 = false;
    for (0..func.instCount()) |i| {
        var it = func.attributesOf(.{ .inst = @enumFromInt(@as(u32, @intCast(i))) });
        while (it.next()) |attr| switch (attr) {
            .custom => |c| if (std.mem.eql(u8, c.namespace, "debug") and std.mem.eql(u8, c.key, "line")) {
                if (c.value == .int and c.value.int == 2) saw_line2 = true;
                if (c.value == .int and c.value.int == 3) saw_line3 = true;
            },
            else => {},
        };
    }
    // The `a + 5` (line 2) and `b * 2` (line 3) instructions carry their source lines.
    try std.testing.expect(saw_line2);
    try std.testing.expect(saw_line3);
}
