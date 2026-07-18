//! Lower a SPIR-V module's first function to a Vulcan IR function. SPIR-V is SSA with
//! basic blocks and phi nodes: function parameters and each block's `OpPhi`s become block
//! parameters, branches carry phi values as edge arguments.
//!
//! Scalar slice: parameters, constants, arithmetic, comparison, select, numeric
//! conversion, control flow (branch / conditional branch / phi / return).
//!
//! Compute slice: a `void main()` reading/writing a storage buffer. Module-level
//! storage-buffer `OpVariable`s and the `gl_GlobalInvocationID` builtin become
//! synthesized entry parameters (dispatch supplies the buffer base address and the
//! per-thread invocation id). `OpAccessChain` lowers to pointer arithmetic
//! (`base + index*stride`), `OpLoad`/`OpStore` to IR load/store. Synthesized
//! entry-parameter order: invocation id (if the builtin is present), then each storage
//! buffer in declaration order, then explicit function parameters.
//!
//! Assumes a storage buffer is `{ T elems[] }` (single array member at offset 0).
//! Multi-member / nested-struct offsets are not modeled. Vectors, function calls, and
//! multiple functions are deferred (`error.Unsupported`).

const std = @import("std");
const ir = @import("vulcan-ir");
const binary = @import("binary.zig");
const op = @import("opcodes.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;

pub const Error = binary.Error || std.mem.Allocator.Error || error{ Unsupported, MalformedModule };

const TypeInfo = union(enum) {
    scalar: Type, // bool / int / float
    void,
    function: struct { ret: u32 },
    pointer: struct { pointee: u32 }, // a pointer to another type id
    array: struct { elem: u32 }, // a (runtime) array of another type id
    vector: VecType, // a SIMD vector of a scalar component type
    matrix: MatType, // a matrix: `cols` column vectors of type `col_vec`
    image, // an OpTypeImage (the opaque image resource of a sampler)
    sampled_image, // an OpTypeSampledImage (a combined image+sampler)
    @"struct", // members tracked via Module.members
    other, // not lowered

    fn asVector(self: TypeInfo) ?VecType {
        return switch (self) {
            .vector => |v| v,
            else => null,
        };
    }

    fn asMatrix(self: TypeInfo) ?MatType {
        return switch (self) {
            .matrix => |m| m,
            else => null,
        };
    }
};

const VecType = struct { elem: u32, len: u8 };

/// A matrix type: `cols` column vectors, each of type id `col_vec`. SPIR-V matrices are
/// column-major (`OpTypeMatrix %columnVectorType %columnCount`).
const MatType = struct { col_vec: u32, cols: u8 };

/// A scalarized vector value: its component scalar values. A vector SSA id maps to these
/// rather than a single Vulcan value, so no backend needs vector support.
const Vec = struct { comps: [4]Value = undefined, len: u8 = 0 };

/// A scalarized matrix value: its element scalar values in column-major order
/// (`comps[col*rows + row]`), so a matrix SSA id maps to scalars exactly like a vector,
/// and no backend needs matrix support. `OpMatrixTimesVector` etc. become FMul/FAdd over
/// these scalars.
const Mat = struct { comps: [16]Value = undefined, cols: u8 = 0, rows: u8 = 0 };

/// A pointer to a matrix laid out in a buffer (UBO / SSBO / push-constant). An
/// `OpAccessChain` ending at a matrix member records this so the following `OpLoad`
/// knows to fetch each element from `base + col*stride + row*4` (column-major) or
/// `base + row*stride + col*4` (row-major), where `base` is the IR pointer Value the
/// chain produced. `elem` is the (scalar float) element Vulcan type.
const MatPtr = struct { base: Value, cols: u8, rows: u8, stride: u32, row_major: bool, elem: Type };

/// A pointer to a vector laid out in a buffer (UBO / SSBO). An `OpAccessChain` ending at
/// a vector member - including a dynamic array element like `u.pos[gl_VertexIndex]`,
/// where the runtime index was folded into the pointer arithmetic - records this so the
/// following `OpLoad %vecN` fetches each component from `base + comp*elemSize` into a
/// scalarized `Vec`. `len` is the component count, `elem` the (scalar) component type.
const VecPtr = struct { base: Value, len: u8, elem: Type };

/// An `OpAccessChain` into a tracked Function-storage vector local addressing one component:
/// the local variable id and the constant component index. A following `OpLoad` yields that
/// component of the local's last-stored vector. An `OpStore` writes it back into the Vec.
const LocalComp = struct { local: u32, comp: u8 };

/// One struct member's type and byte offset (from its `OpMemberDecorate Offset`), plus
/// the matrix layout decorations a matrix member carries: `MatrixStride` (bytes between
/// columns, col-major / rows row-major) and whether it is `RowMajor` (default ColMajor).
/// `builtin` is the member's `OpMemberDecorate BuiltIn` value if any (0xFFFFFFFF = none),
/// so the gl_PerVertex interface block (a member decorated BuiltIn Position) is recognized.
const Member = struct { type_id: u32, offset: u32, matrix_stride: u32 = 0, row_major: bool = false, builtin: u32 = NO_BUILTIN };

/// Sentinel for "this member carries no BuiltIn decoration".
const NO_BUILTIN: u32 = std.math.maxInt(u32);

/// Pack a (struct type id, member index) into a hash-map key.
fn memberKey(struct_id: u32, member: u32) u64 {
    return (@as(u64, struct_id) << 32) | member;
}

const Const = struct { type_id: u32, bits: u64 };

/// One fragment-shader gradient-buffer slot: the attribute `slot` of the varying scalar
/// whose screen-space derivative this is, and which `axis` (dFdx / dFdy). Recorded per
/// dense buffer index so the backend (rasterizer) fills `grad_buf[index]` with that
/// varying's per-triangle d/dx or d/dy.
pub const GradDesc = struct { slot: u32, axis: enum { x, y } };

/// How a module-level `OpVariable` is used. `input`/`output` are the graphics attribute
/// interface: an Input variable feeds a vertex/fragment shader, an Output variable is its
/// product (clip-space position, fragment color). `sampler` is a combined-image-sampler
/// (an `OpVariable` in the UniformConstant storage class typed `OpTypeSampledImage`): it
/// reaches the shader as a descriptor pointer param, exactly like a buffer.
/// `vertex_index` / `instance_index` are vertex-shader BuiltIn INPUTs (gl_VertexIndex /
/// gl_InstanceIndex): each becomes a synthesized i32 entry param the dispatch supplies
/// per vertex, exactly like the compute `global_id`.
const VarKind = enum { buffer, sampler, global_id, vertex_index, instance_index, input, output, other };

/// The shader stage an `OpEntryPoint` selects. Lowering produces the attribute interface
/// for graphics stages, the kernel ABI for compute.
const Stage = enum { compute, vertex, fragment };

// NVIDIA attribute byte addresses the graphics backend expects, mirrored here so the
// target-agnostic frontend can tag input params and output stores with their slot.
// These match vulcan-target/nvidia/encode.zig.
const ATTR_POSITION: u32 = 0x70; // clip-space position output
const ATTR_POINT_SIZE: u32 = 0x6c; // gl_PointSize output (a scalar, below the position slot)
const ATTR_GENERIC0: u32 = 0x80; // first generic input attribute / varying

const BlockInfo = struct {
    block: Block,
    label: u32,
    phis: std.ArrayList(u32) = .empty,
};

/// Module state the lowering needs, indexed by SPIR-V id.
const Module = struct {
    types: []?TypeInfo,
    var_kind: []VarKind, // id -> how an OpVariable is used (.other if not a var)
    binding: []u32, // id -> the Binding decoration (a descriptor's Vulkan binding number)
    has_binding: []bool, // id -> whether a Binding decoration was present
    // Function-storage composite (vector) locals that mem2reg could not promote (because a
    // constant-index OpAccessChain takes a component's address). The lowering scalarizes them
    // directly: an OpStore records the scalarized vector here, a whole OpLoad yields it, and a
    // constant-index OpAccessChain + OpLoad yields one component. glslang emits these as
    // single-assignment, dominator-ordered store-then-read temporaries (the by-pointer
    // argument-passing pattern), so tracking the last stored value is correct.
    local_is_vec: []bool, // id -> whether it is a tracked Function-storage vector local
    local_vec: []Vec, // local var id -> the scalarized vector last stored to it
    local_comp_of: []?LocalComp, // access-chain result id -> (local var id, component index)
    buffers: std.ArrayList(u32) = .empty, // storage-buffer var ids, declaration order
    has_global_id: bool = false,
    value_of: []?Value, // id -> the Vulcan value it lowers to (scalars)
    vec_of: []Vec, // id -> a scalarized vector value's components (len 0 = not a vector)
    mat_of: []Mat, // id -> a scalarized matrix value's elements (cols 0 = not a matrix)
    mat_ptr: []?MatPtr, // access-chain result id -> matrix-in-memory layout (for OpLoad)
    vec_ptr: []?VecPtr, // access-chain result id -> vector-in-memory layout (for OpLoad)
    is_builtin_ptr: []bool, // id -> a pointer into the gl_GlobalInvocationID builtin
    // Combined-image-sampler descriptor pointer per value: an OpLoad of a sampler
    // variable, or an OpSampledImage, records here the descriptor pointer Value the
    // following OpImageSample* reads. Vec/scalar value_of stays unset for these.
    sampler_ptr_of: []?Value,
    // The SPIR-V image DIMENSION (Dim: 1D=0, 2D=1, 3D=2, Cube=3) of each type id (propagated
    // Image -> SampledImage -> Pointer) and of each loaded sampler VALUE. Lets lowerImageSample
    // distinguish a 3D sample (dim 2) from a cube sample (dim 3) - both take a vec3 coord but
    // need different host sampler tags (the GPU bakes a different TEX dim + TIC per kind; the
    // software host dispatches on the descriptor). Default 0 (unset) = the 2D path.
    type_image_dim: []u8,
    sampler_dim_of: []u8,
    // Parallel to type_image_dim / sampler_dim_of: the image's ARRAYED flag (OpTypeImage operand 4).
    // A `sampler2DArray` has Dim=2D (same as a plain sampler2D) but Arrayed=1, so the dim alone can
    // not tell them apart - the arrayed flag does. 1 = arrayed, 0 = not.
    type_image_arrayed: []u8,
    sampler_arrayed_of: []u8,
    // The synthesized host-sampler function-pointer entry param (lazily appended the
    // first time an image-sample op is lowered). The CPU backend calls it. A GPU
    // backend ignores it (it emits a TEX). Null until the first sample op.
    sampler_fn: ?Value = null,
    // The synthesized host CUBE-sampler function-pointer entry param, appended lazily the
    // first time a Cube-dim image sample (a vec3 direction) is lowered. The CPU backend
    // calls it (`void sampler_cube_fn(desc, x, y, z, lod, out)`); a GPU backend ignores it
    // and emits a cube TEX. Kept separate from `sampler_fn` so the 2D sample path (a vec2
    // coord, 5-arg call) is byte-identical. Null until the first cube sample.
    sampler_cube_fn: ?Value = null,
    // The synthesized host 3D-sampler function-pointer entry param (lazily appended the first
    // time a 3D-dim image sample - a vec3 into a volume - is lowered). Kept distinct from
    // sampler_cube_fn so the GPU backend emits the 3D TEX dim (not cube) for it; the CPU binds
    // the same host sampler for both (it dispatches on the descriptor).
    sampler_3d_fn: ?Value = null,
    // The synthesized host 2D-ARRAY-sampler function-pointer entry param (lazily appended the first
    // time a 2D-Arrayed image sample - a vec3 (u, v, layer) - is lowered). Distinct from
    // sampler_3d_fn so the GPU backend emits the 2D-array TEX dim + TIC (the third coord is a raw
    // layer index, not a normalized depth); the CPU binds the same host sampler (dispatches on desc).
    sampler_2darray_fn: ?Value = null,
    // The synthesized host SHADOW-sampler function-pointer entry param (lazily appended the first
    // time an OpImageSampleDref - GLSL sampler2DShadow - is lowered). The CPU backend calls it
    // (`f32 sampler_shadow_fn(desc, u, v, lod, dref)`) and the SCALAR result is the depth-compare
    // fraction (no out-pointer, unlike the vec4 samplers); a GPU backend emits a compare TEX.
    sampler_shadow_fn: ?Value = null,
    // The synthesized host CUBE-SHADOW-sampler function-pointer entry param (lazily appended the first
    // time an OpImageSampleDref on a Cube-dim image - GLSL samplerCubeShadow - is lowered). The CPU
    // backend calls it (`f32 sampler_cube_shadow_fn(desc, x, y, z, lod, dref)`) and the SCALAR result
    // is the depth-compare fraction; distinct from sampler_shadow_fn (the 2D case, a vec2 coord).
    sampler_cube_shadow_fn: ?Value = null,
    // The synthesized host 2D-ARRAY-SHADOW-sampler function-pointer entry param (lazily appended the
    // first time an OpImageSampleDref on a 2D-Arrayed image - GLSL sampler2DArrayShadow - is lowered).
    // The CPU backend calls it (`f32 sampler_2darray_shadow_fn(desc, u, v, layer, lod, dref)`) and the
    // SCALAR result is the depth-compare fraction; distinct from the cube (dim==3) and 2D shadow cases.
    sampler_2darray_shadow_fn: ?Value = null,
    // The synthesized host GATHER function-pointer entry param (lazily appended the first time an
    // OpImageGather - GLSL textureGather - is lowered). The CPU backend calls it (`void
    // sampler_gather_fn(desc, u, v, comp, out)`) writing the 4 footprint texels' `comp` channel; a
    // GPU backend ignores it and emits a TG4 (OpTld4). Null until the first gather.
    sampler_gather_fn: ?Value = null,
    // The synthesized host FETCH function-pointer entry param (lazily appended the first time an
    // OpImageFetch - GLSL texelFetch - is lowered). The CPU backend calls it (`void
    // sampler_fetch_fn(desc, x:i32, y:i32, lod:i32, out)`) returning the EXACT texel at integer
    // coords (no filter); a GPU backend ignores it and emits a TLD (OpTld, texelFetch). Null until first.
    sampler_fetch_fn: ?Value = null,
    // The host FETCH params for a 2D-ARRAY / 3D texelFetch (an ivec3 coord). Distinct tags so the GPU
    // backend emits the matching TLD dim (Array2D vs 3D); the CPU binds the same host fn (fetch the
    // exact texel at (x, y) of layer/slice z). ABI: `fn(desc, x:i32, y:i32, z:i32, lod:i32, out)`.
    sampler_fetch_array_fn: ?Value = null,
    sampler_fetch_3d_fn: ?Value = null,
    // The synthesized host-math function-pointer entry param (lazily appended the first time
    // a transcendental ext-inst - pow / exp / log / sin / cos - is lowered). The CPU backend
    // calls it (`f32 math_fn(op:i32, a:f32, b:f32)`). A GPU backend ignores it and emits a
    // MUFU. Null until the first such op.
    math_fn: ?Value = null,
    // Per-component graphics input access (glslang reads an Input vecN one scalar at a
    // time): an OpAccessChain into an Input variable with a constant component records,
    // here, the addressed component (a later OpLoad of this id yields that scalar param).
    input_comp_of: []?Value, // access-chain result id -> the Input vec component value
    // gl_PerVertex / gl_Position store: an OpAccessChain into the gl_PerVertex Output
    // struct addressing its BuiltIn Position member records the output variable + that it
    // is the position member, so the following OpStore routes to ATTR_POSITION.
    pos_chain_var: []u32, // access-chain result id -> gl_PerVertex Output var id (0 = no)
    // The clip-space position last stored to gl_Position, as a scalarized Vec. A shader
    // may read its own gl_Position back (e.g. vkcube's VS does `frag_pos =
    // gl_Position.xyz`), so an OpLoad of the gl_PerVertex Position member access chain
    // yields this. cols/len 0 until the position has been stored.
    position_vec: Vec = .{},
    // Screen-space derivatives (dFdx / dFdy / Fwidth) of fragment-shader input varyings.
    // A linearly-interpolated varying has a CONSTANT screen-space gradient per triangle
    // (the plane equation), so dFdx/dFdy of a varying load just return that gradient - the
    // rasterizer computes the per-triangle gradients and supplies them through a single
    // synthesized POINTER entry param (`grad_buf`, tagged `vulcan.gpu.grad_buf`), so the
    // FS's float-register interface (the varyings) stays unchanged regardless of how many
    // derivatives are taken. Each distinct (varying-scalar, axis) derivative gets a dense
    // index into that buffer. The load is `grad_buf + index*4`. `attr_slot_of` maps a
    // varying-component scalar Value to its attribute slot (so the rasterizer knows which
    // varying each gradient is of). `grad_index_x` / `grad_index_y` cache the assigned
    // buffer index for a varying-component Value. `grad_descs` records, per index, the
    // (slot, axis) so the backend fills the buffer. `grad_buf_param` is the lazily-appended
    // pointer param.
    attr_slot_of: std.AutoHashMapUnmanaged(Value, u32) = .empty,
    grad_index_x: std.AutoHashMapUnmanaged(Value, u32) = .empty,
    grad_index_y: std.AutoHashMapUnmanaged(Value, u32) = .empty,
    grad_descs: std.ArrayList(GradDesc) = .empty, // index -> (attr slot, axis)
    grad_buf_param: ?Value = null, // the synthesized grad_buf pointer param (lazy)
    grad_entry: Block = undefined, // the entry block, for appending the grad_buf param
    ptr_t: Type,
    i32_t: Type,
    global_id_value: ?Value = null,
    has_vertex_index: bool = false,
    vertex_index_value: ?Value = null,
    has_instance_index: bool = false,
    instance_index_value: ?Value = null,
    discard_fn: ?Value = null, // the synthesized discard (OpKill) function-pointer param (lazy)
    members: std.AutoHashMapUnmanaged(u64, Member) = .empty, // (struct,member) -> type+offset
    array_stride: []u32, // type id -> ArrayStride decoration (0 = derive from element)
    const_val: []i64, // id -> the integer value of an OpConstant (for struct indices)
    var_type: []u32, // variable id -> its (pointer) type id

    // Graphics interface (only used for vertex/fragment stages).
    stage: Stage = .compute,
    location: []u32, // input/output var id -> Location decoration (0 = none / position)
    has_location: []bool, // input/output var id -> whether Location was decorated
    is_position: []bool, // output var id -> whether it is the Position builtin
    is_point_size: []bool, // output var id -> whether it is the PointSize builtin
    is_frag_coord: []bool, // input var id -> whether it is the gl_FragCoord builtin
    is_point_coord: []bool, // input var id -> whether it is the gl_PointCoord builtin (a vec2)
    is_front_facing: []bool, // input var id -> whether it is the gl_FrontFacing builtin
    is_frag_depth: []bool, // output var id -> whether it is the gl_FragDepth builtin
    var_storage: []u32, // variable id -> its SPIR-V storage class

    fn pointee(self: *const Module, type_id: u32) ?u32 {
        if (type_id >= self.types.len) return null;
        return switch (self.types[type_id] orelse return null) {
            .pointer => |p| p.pointee,
            else => null,
        };
    }

    /// The type a variable points to (pointee of its pointer type).
    fn var_pointee(self: *const Module, var_id: u32) ?u32 {
        if (var_id >= self.var_type.len or self.var_type[var_id] == 0) return null;
        return self.pointee(self.var_type[var_id]);
    }
};

/// Whether variable `id` is a combined-image-sampler: its pointee type is an
/// `OpTypeSampledImage` (or a bare `OpTypeImage`, for a sampled separate image).
fn isSamplerVar(module: *const Module, id: u32) bool {
    const pointee = module.var_pointee(id) orelse return false;
    if (pointee >= module.types.len) return false;
    return switch (module.types[pointee] orelse return false) {
        .sampled_image, .image => true,
        else => false,
    };
}

/// Validate an id decoded from the untrusted word stream is in range `[0, bound)`, so it is
/// safe to use as an index into the per-id arrays (all sized `bound`). SPIR-V guarantees every
/// id is `< id_bound`, but a hostile/malformed module may carry an out-of-range id: as a write
/// index it would corrupt memory, as a read index it would fault. `lowerModule` is a public
/// entry point over raw `[]const u32` with no validation layer, so it must reject these.
fn checkId(id: u32, bound: usize) Error!u32 {
    if (id >= bound) return error.MalformedModule;
    return id;
}

/// Operand `i` of a decoded instruction, or `error.MalformedModule` if the instruction is too
/// short. `binary.Reader` only guarantees `word_count >= 1`, so a truncated instruction carries
/// fewer operands than its opcode requires; indexing past them would read out of bounds.
fn operandAt(operands: []const u32, i: usize) Error!u32 {
    if (i >= operands.len) return error.MalformedModule;
    return operands[i];
}

/// Operand `i` validated as an in-range id (length + id-bound checked): safe to use as an index
/// into the per-id arrays.
fn idOperandAt(operands: []const u32, i: usize, bound: usize) Error!u32 {
    return checkId(try operandAt(operands, i), bound);
}

/// The operand tail starting at index `i` (`operands[i..]`), or `error.MalformedModule` if the
/// instruction is too short to have a word there. A truncated instruction must not slice OOB.
fn operandsFrom(operands: []const u32, i: usize) Error![]const u32 {
    if (i > operands.len) return error.MalformedModule;
    return operands[i..];
}

/// The scalarized vector recorded for a value id (a copy), bounds-checked against the untrusted
/// id bound. A `len == 0` result means "not a (recorded) vector", as elsewhere.
fn vecOf(module: *const Module, id: u32) Error!Vec {
    if (id >= module.vec_of.len) return error.MalformedModule;
    return module.vec_of[id];
}

/// The integer value of an `OpConstant` id (used for struct-member / component indices),
/// bounds-checked against the untrusted id bound.
fn constValOf(module: *const Module, id: u32) Error!i64 {
    if (id >= module.const_val.len) return error.MalformedModule;
    return module.const_val[id];
}

/// A value id read from the untrusted stream, mapped to its lowered Vulcan Value. Rejects an
/// out-of-range id (OOB read of `value_of`) and an id that has no value yet (a forward/undefined
/// reference) - both are malformed for the single-function shapes this lowering accepts.
fn valueOf(module: *const Module, id: u32) Error!Value {
    if (id >= module.value_of.len) return error.MalformedModule;
    return module.value_of[id] orelse error.MalformedModule;
}

/// Lower the first function of the SPIR-V module in `words` to a fresh Vulcan IR
/// function. Caller owns and must `deinit` it.
pub fn lowerModule(allocator: std.mem.Allocator, words: []const u32) Error!Function {
    var r = try binary.Reader.init(words);
    const bound = r.header.id_bound;

    var types = try allocator.alloc(?TypeInfo, bound);
    defer allocator.free(types);
    @memset(types, null);
    // Image-Dim tracking (locals; published to `module` right after it is declared, since the
    // type-parsing pass below writes them). type_image_dim[type id] = SPIR-V Dim; sampler_dim_of
    // [value id] = the Dim of a loaded sampler value.
    const type_image_dim = try allocator.alloc(u8, bound);
    defer allocator.free(type_image_dim);
    @memset(type_image_dim, 0);
    const sampler_dim_of = try allocator.alloc(u8, bound);
    defer allocator.free(sampler_dim_of);
    @memset(sampler_dim_of, 0);
    const type_image_arrayed = try allocator.alloc(u8, bound);
    defer allocator.free(type_image_arrayed);
    @memset(type_image_arrayed, 0);
    const sampler_arrayed_of = try allocator.alloc(u8, bound);
    defer allocator.free(sampler_arrayed_of);
    @memset(sampler_arrayed_of, 0);
    var var_kind = try allocator.alloc(VarKind, bound);
    defer allocator.free(var_kind);
    @memset(var_kind, .other);
    var builtin_decor = try allocator.alloc(u32, bound); // id -> BuiltIn value (0 = none)
    defer allocator.free(builtin_decor);
    @memset(builtin_decor, 0);
    const array_stride = try allocator.alloc(u32, bound);
    defer allocator.free(array_stride);
    @memset(array_stride, 0);
    const const_val = try allocator.alloc(i64, bound);
    defer allocator.free(const_val);
    @memset(const_val, 0);
    const var_type = try allocator.alloc(u32, bound);
    defer allocator.free(var_type);
    @memset(var_type, 0);
    const location = try allocator.alloc(u32, bound);
    defer allocator.free(location);
    @memset(location, 0);
    const has_location = try allocator.alloc(bool, bound);
    defer allocator.free(has_location);
    @memset(has_location, false);
    // The Binding decoration per OpVariable id (a UBO/SSBO/sampler descriptor's
    // binding number). Used to give each descriptor entry param its Vulkan binding
    // so the backend places it at the matching constant-bank slot (a SHARED bank
    // across stages: a per-stage declaration-order slot would collide when, e.g., a
    // VS UBO at binding 0 and an FS sampler at binding 1 are both "first" in-stage).
    const binding = try allocator.alloc(u32, bound);
    defer allocator.free(binding);
    @memset(binding, 0);
    const has_binding = try allocator.alloc(bool, bound);
    defer allocator.free(has_binding);
    @memset(has_binding, false);
    const is_position = try allocator.alloc(bool, bound);
    defer allocator.free(is_position);
    @memset(is_position, false);
    const is_point_size = try allocator.alloc(bool, bound);
    defer allocator.free(is_point_size);
    @memset(is_point_size, false);
    const is_frag_coord = try allocator.alloc(bool, bound);
    defer allocator.free(is_frag_coord);
    @memset(is_frag_coord, false);
    const is_point_coord = try allocator.alloc(bool, bound);
    defer allocator.free(is_point_coord);
    @memset(is_point_coord, false);
    const is_front_facing = try allocator.alloc(bool, bound);
    defer allocator.free(is_front_facing);
    @memset(is_front_facing, false);
    const is_frag_depth = try allocator.alloc(bool, bound);
    defer allocator.free(is_frag_depth);
    @memset(is_frag_depth, false);
    const var_storage = try allocator.alloc(u32, bound);
    defer allocator.free(var_storage);
    @memset(var_storage, 0);

    var consts = std.AutoHashMapUnmanaged(u32, Const){};
    defer consts.deinit(allocator);
    // OpConstantComposite values: id -> the component constant ids (vectors only).
    var composite_consts = std.AutoHashMapUnmanaged(u32, []const u32){};
    defer {
        var it = composite_consts.iterator();
        while (it.next()) |e| allocator.free(e.value_ptr.*);
        composite_consts.deinit(allocator);
    }
    var insts: std.ArrayList(binary.Instruction) = .empty;
    defer insts.deinit(allocator);

    var func = Function.init(allocator);
    errdefer func.deinit();

    var module = Module{
        .types = types,
        .var_kind = var_kind,
        .type_image_dim = type_image_dim,
        .sampler_dim_of = sampler_dim_of,
        .type_image_arrayed = type_image_arrayed,
        .sampler_arrayed_of = sampler_arrayed_of,
        .binding = binding,
        .has_binding = has_binding,
        .value_of = undefined,
        .vec_of = undefined,
        .mat_of = undefined,
        .mat_ptr = undefined,
        .vec_ptr = undefined,
        .local_is_vec = undefined,
        .local_vec = undefined,
        .local_comp_of = undefined,
        .is_builtin_ptr = undefined,
        .sampler_ptr_of = undefined,
        .input_comp_of = undefined,
        .pos_chain_var = undefined,
        .ptr_t = try func.types.intern(.ptr),
        .i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }),
        .array_stride = array_stride,
        .const_val = const_val,
        .var_type = var_type,
        .location = location,
        .has_location = has_location,
        .is_position = is_position,
        .is_point_size = is_point_size,
        .is_frag_coord = is_frag_coord,
        .is_point_coord = is_point_coord,
        .is_front_facing = is_front_facing,
        .is_frag_depth = is_frag_depth,
        .var_storage = var_storage,
    };
    defer module.buffers.deinit(allocator);
    defer module.members.deinit(allocator);
    defer module.attr_slot_of.deinit(allocator);
    defer module.grad_index_x.deinit(allocator);
    defer module.grad_index_y.deinit(allocator);
    defer module.grad_descs.deinit(allocator);

    var func_ret_type: ?u32 = null;
    var in_function = false;
    var local_size_x: u32 = 1; // workgroup x dimension (OpExecutionMode LocalSize)
    var pending_vars: std.ArrayList([2]u32) = .empty; // (var id, storage class)
    defer pending_vars.deinit(allocator);

    while (try r.next()) |inst| {
        switch (inst.opcode) {
            op.TypeVoid => types[try idOperandAt(inst.operands, 0, bound)] = .void,
            op.TypeBool => types[try idOperandAt(inst.operands, 0, bound)] = .{ .scalar = try func.types.intern(.bool) },
            op.TypeInt => {
                const id = try idOperandAt(inst.operands, 0, bound);
                const width = try operandAt(inst.operands, 1);
                const signedness = try operandAt(inst.operands, 2);
                types[id] = .{
                    .scalar = try func.types.intern(.{
                        .int = .{
                            .signedness = if (signedness != 0) .signed else .unsigned,
                            // A width wider than the IR's u16 bit field is a malformed type, not a truncation.
                            .bits = std.math.cast(u16, width) orelse return error.MalformedModule,
                        },
                    }),
                };
            },
            op.TypeFloat => {
                const id = try idOperandAt(inst.operands, 0, bound);
                const width = try operandAt(inst.operands, 1);
                // Exhaustive on the widths SPIR-V can emit for this backend (16/32/64); any
                // other width is a malformed module, not a silent fallback to f32.
                const fk: ir.types.FloatKind = switch (width) {
                    16 => .f16,
                    32 => .f32,
                    64 => .f64,
                    else => return error.MalformedModule,
                };
                types[id] = .{ .scalar = try func.types.intern(.{ .float = fk }) };
            },
            op.TypeFunction => {
                // Referenced type ids (here the return type, below element/member/column
                // types) are validated so they can later index `types` without a bounds check.
                const id = try idOperandAt(inst.operands, 0, bound);
                types[id] = .{ .function = .{ .ret = try idOperandAt(inst.operands, 1, bound) } };
            },
            op.TypePointer => {
                const id = try idOperandAt(inst.operands, 0, bound);
                const pointee = try idOperandAt(inst.operands, 2, bound); // [result, storageClass, pointee]
                types[id] = .{ .pointer = .{ .pointee = pointee } };
                module.type_image_dim[id] = module.type_image_dim[pointee]; // carry an image dim through the pointer
                module.type_image_arrayed[id] = module.type_image_arrayed[pointee]; // and its arrayed flag
            },
            op.TypeArray, op.TypeRuntimeArray => {
                const id = try idOperandAt(inst.operands, 0, bound);
                types[id] = .{ .array = .{ .elem = try idOperandAt(inst.operands, 1, bound) } };
            },
            op.TypeVector => {
                const id = try idOperandAt(inst.operands, 0, bound);
                const elem = try idOperandAt(inst.operands, 1, bound);
                const len = try operandAt(inst.operands, 2);
                // A scalarized vector value is held in a fixed [4]Value (`Vec.comps`); a length
                // beyond that would overflow it as the components are materialized, so reject it.
                if (len == 0 or len > 4) return error.MalformedModule;
                types[id] = .{ .vector = .{ .elem = elem, .len = @intCast(len) } };
            },
            op.TypeMatrix => {
                const id = try idOperandAt(inst.operands, 0, bound);
                const col_vec = try idOperandAt(inst.operands, 1, bound);
                const cols = try operandAt(inst.operands, 2); // [result, columnVecType, columnCount]
                // Matrices scalarize into a fixed [16]Value (`Mat.comps`) = at most 4 columns of
                // at most 4 rows (the column vector's length is capped by TypeVector above).
                if (cols == 0 or cols > 4) return error.MalformedModule;
                types[id] = .{ .matrix = .{ .col_vec = col_vec, .cols = @intCast(cols) } };
            },
            op.TypeImage => {
                const id = try idOperandAt(inst.operands, 0, bound);
                types[id] = .image; // [result, sampledType, Dim, Depth, Arrayed, MS, Sampled, Format, ...]
                if (inst.operands.len >= 3) module.type_image_dim[id] = @intCast(inst.operands[2] & 0xff); // Dim
                if (inst.operands.len >= 5) module.type_image_arrayed[id] = @intCast(inst.operands[4] & 0xff); // Arrayed
            },
            op.TypeSampledImage => {
                const id = try idOperandAt(inst.operands, 0, bound);
                const image = try idOperandAt(inst.operands, 1, bound); // [result, imageType]
                types[id] = .sampled_image;
                module.type_image_dim[id] = module.type_image_dim[image];
                module.type_image_arrayed[id] = module.type_image_arrayed[image];
            },
            op.TypeStruct => {
                // [result, member0Type, member1Type, ...]. The member offsets may
                // already be present from earlier OpMemberDecorate, so preserve them.
                const id = try idOperandAt(inst.operands, 0, bound);
                types[id] = .@"struct";
                for (inst.operands[1..], 0..) |member_type_raw, i| {
                    const member_type = try checkId(member_type_raw, bound); // indexes `types` later
                    const key = memberKey(id, @intCast(i));
                    if (module.members.getPtr(key)) |m| {
                        m.type_id = member_type;
                    } else {
                        try module.members.put(allocator, key, .{ .type_id = member_type, .offset = 0 });
                    }
                }
            },
            op.EntryPoint => if (inst.operands.len >= 1) {
                // [ExecutionModel, entryPoint, name..., interface...]. The first entry
                // point's execution model selects the shader stage.
                module.stage = switch (inst.operands[0]) {
                    op.ExecutionModel.vertex => .vertex,
                    op.ExecutionModel.fragment => .fragment,
                    else => .compute,
                };
            },
            op.Decorate => if (inst.operands.len >= 3 and inst.operands[1] == op.Decoration.builtin) {
                const target = try checkId(inst.operands[0], bound);
                builtin_decor[target] = inst.operands[2];
                if (inst.operands[2] == op.BuiltIn.position) is_position[target] = true;
                if (inst.operands[2] == op.BuiltIn.point_size) is_point_size[target] = true;
                if (inst.operands[2] == op.BuiltIn.frag_coord) is_frag_coord[target] = true;
                if (inst.operands[2] == op.BuiltIn.point_coord) is_point_coord[target] = true;
                if (inst.operands[2] == op.BuiltIn.front_facing) is_front_facing[target] = true;
                if (inst.operands[2] == op.BuiltIn.frag_depth) is_frag_depth[target] = true;
            } else if (inst.operands.len >= 3 and inst.operands[1] == op.Decoration.array_stride) {
                array_stride[try checkId(inst.operands[0], bound)] = inst.operands[2];
            } else if (inst.operands.len >= 3 and inst.operands[1] == op.Decoration.location) {
                const target = try checkId(inst.operands[0], bound);
                location[target] = inst.operands[2];
                has_location[target] = true;
            } else if (inst.operands.len >= 3 and inst.operands[1] == op.Decoration.binding) {
                const target = try checkId(inst.operands[0], bound);
                binding[target] = inst.operands[2];
                has_binding[target] = true;
            },
            op.MemberDecorate => if (inst.operands.len >= 4 and inst.operands[2] == op.Decoration.offset) {
                // [struct, member, Offset, value] (annotations come before the type).
                const key = memberKey(inst.operands[0], inst.operands[1]);
                if (module.members.getPtr(key)) |m| {
                    m.offset = inst.operands[3];
                } else {
                    try module.members.put(allocator, key, .{ .type_id = 0, .offset = inst.operands[3] });
                }
            } else if (inst.operands.len >= 4 and inst.operands[2] == op.Decoration.matrix_stride) {
                // [struct, member, MatrixStride, bytes]: bytes between matrix columns.
                const key = memberKey(inst.operands[0], inst.operands[1]);
                if (module.members.getPtr(key)) |m| {
                    m.matrix_stride = inst.operands[3];
                } else {
                    try module.members.put(allocator, key, .{ .type_id = 0, .offset = 0, .matrix_stride = inst.operands[3] });
                }
            } else if (inst.operands.len >= 3 and inst.operands[2] == op.Decoration.row_major) {
                const key = memberKey(inst.operands[0], inst.operands[1]);
                if (module.members.getPtr(key)) |m| {
                    m.row_major = true;
                } else {
                    try module.members.put(allocator, key, .{ .type_id = 0, .offset = 0, .row_major = true });
                }
            } else if (inst.operands.len >= 4 and inst.operands[2] == op.Decoration.builtin) {
                // [struct, member, BuiltIn, value]: the gl_PerVertex interface block
                // decorates its members BuiltIn Position/PointSize/Clip/CullDistance.
                const key = memberKey(inst.operands[0], inst.operands[1]);
                if (module.members.getPtr(key)) |m| {
                    m.builtin = inst.operands[3];
                } else {
                    try module.members.put(allocator, key, .{ .type_id = 0, .offset = 0, .builtin = inst.operands[3] });
                }
            },
            op.ExecutionMode => if (inst.operands.len >= 3 and inst.operands[1] == op.ExecutionModeKind.local_size) {
                local_size_x = inst.operands[2]; // [entryPoint, LocalSize, x, y, z]
            },
            op.Constant => {
                const type_id = try operandAt(inst.operands, 0);
                const result = try idOperandAt(inst.operands, 1, bound); // indexes const_val
                const bits = constBits(inst.operands[2..]);
                try consts.put(allocator, result, .{ .type_id = type_id, .bits = bits });
                const_val[result] = @bitCast(bits); // for struct member indices
            },
            // The const result ids index value_of / vec_of when the constants are materialized.
            op.ConstantTrue => try consts.put(allocator, try idOperandAt(inst.operands, 1, bound), .{ .type_id = try operandAt(inst.operands, 0), .bits = 1 }),
            op.ConstantFalse => try consts.put(allocator, try idOperandAt(inst.operands, 1, bound), .{ .type_id = try operandAt(inst.operands, 0), .bits = 0 }),
            op.ConstantComposite => {
                // [type, result, comp0, comp1, ...]: a vector constant whose components
                // are themselves OpConstants. Record the component ids, materialized
                // component-wise in lowerFunction.
                const result = try idOperandAt(inst.operands, 1, bound);
                const comps = try allocator.dupe(u32, inst.operands[2..]);
                try composite_consts.put(allocator, result, comps);
            },
            op.Variable => {
                const type_id = try operandAt(inst.operands, 0); // [type, result, storageClass]
                const result = try idOperandAt(inst.operands, 1, bound); // indexes var_type / var_storage
                const storage = try operandAt(inst.operands, 2);
                try pending_vars.append(allocator, .{ result, storage });
                var_type[result] = type_id; // the variable's pointer type
                var_storage[result] = storage;
            },
            op.Function => {
                if (in_function) return error.Unsupported; // only the first function
                in_function = true;
                func_ret_type = try operandAt(inst.operands, 0);
            },
            op.FunctionEnd => break,
            else => if (in_function) try insts.append(allocator, inst),
        }
    }
    if (!in_function) return error.MalformedModule;

    // Classify variables now that all decorations and the stage are seen. A graphics
    // stage routes Input/Output variables to the attribute interface. A compute stage
    // routes storage buffers + the invocation id.
    for (pending_vars.items) |v| {
        const id = v[0];
        const class = v[1];
        if (module.stage == .vertex and class == op.StorageClass.input and builtin_decor[id] == op.BuiltIn.vertex_index) {
            // gl_VertexIndex: a vertex-shader BuiltIn input. Like the compute invocation
            // id, it becomes a synthesized i32 entry param the dispatch supplies (the
            // per-vertex index), NOT a vertex-attribute fetch.
            var_kind[id] = .vertex_index;
            module.has_vertex_index = true;
        } else if (module.stage == .vertex and class == op.StorageClass.input and builtin_decor[id] == op.BuiltIn.instance_index) {
            // gl_InstanceIndex: a vertex-shader BuiltIn input -> a synthesized i32 entry
            // param (the per-instance index, 0 for a single-instance draw).
            var_kind[id] = .instance_index;
            module.has_instance_index = true;
        } else if (module.stage != .compute and class == op.StorageClass.input) {
            var_kind[id] = .input;
        } else if (module.stage != .compute and class == op.StorageClass.output) {
            var_kind[id] = .output;
        } else if (class == op.StorageClass.uniform_constant and isSamplerVar(&module, id)) {
            // A combined-image-sampler (UniformConstant OpTypeSampledImage/OpTypeImage):
            // a descriptor that reaches the shader as a pointer param, like a buffer. It
            // joins module.buffers in declaration order so it gets a pointer entry param.
            var_kind[id] = .sampler;
            try module.buffers.append(allocator, id);
        } else if (class == op.StorageClass.storage_buffer or class == op.StorageClass.uniform or class == op.StorageClass.push_constant) {
            // Storage buffers, uniform blocks (UBOs), and push-constant blocks all reach
            // the shader the same way: a base pointer the shader reads at std-layout member
            // offsets via OpAccessChain + OpLoad. A push-constant block is just a Block-
            // decorated struct in the PushConstant storage class whose bytes come from
            // vkCmdPushConstants instead of a descriptor-bound buffer. The lowering is
            // identical (a pointer entry param), so the host supplies the pushed bytes there.
            var_kind[id] = .buffer;
            try module.buffers.append(allocator, id);
        } else if (builtin_decor[id] == op.BuiltIn.global_invocation_id) {
            var_kind[id] = .global_id;
            module.has_global_id = true;
        }
    }

    try lowerFunction(allocator, &func, &module, &consts, &composite_consts, insts.items, func_ret_type.?);

    // Tag the function with its shader stage so the backend selects the right lowering
    // (compute kernel ABI vs the graphics attribute interface).
    if (module.stage != .compute) {
        try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "stage", .value = .{ .string = @tagName(module.stage) } } });
    }

    // Record the workgroup x dimension so a GPU backend can fold the block offset into
    // the global invocation id (gid.x = blockIdx.x * local_size_x + threadIdx.x).
    if (module.has_global_id) {
        try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "local_size_x", .value = .{ .int = local_size_x } } });
    }

    // Emit the fragment-shader gradient-buffer layout: one func attr per grad_buf index
    // (in index order), encoding (axis | slot << 1) so the backend recovers, per buffer
    // slot, which varying scalar (the attr slot) and which axis (dFdx/dFdy) to fill. The
    // attr append order is the buffer index order. The backend reads them back in order.
    for (module.grad_descs.items) |gd| {
        const packed_val: i64 = @as(i64, gd.slot) << 1 | @intFromBool(gd.axis == .y);
        try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_slot", .value = .{ .int = packed_val } } });
    }
    return func;
}

fn constBits(operands: []const u32) u64 {
    if (operands.len >= 2) return @as(u64, operands[0]) | (@as(u64, operands[1]) << 32);
    if (operands.len == 1) return operands[0];
    return 0;
}

fn lowerFunction(allocator: std.mem.Allocator, func: *Function, module: *Module, consts: *const std.AutoHashMapUnmanaged(u32, Const), composite_consts: *const std.AutoHashMapUnmanaged(u32, []const u32), insts: []const binary.Instruction, ret_type_id: u32) Error!void {
    const bound = module.types.len;
    const value_of = try allocator.alloc(?Value, bound);
    defer allocator.free(value_of);
    @memset(value_of, null);
    module.value_of = value_of;
    const vec_of = try allocator.alloc(Vec, bound);
    defer allocator.free(vec_of);
    @memset(vec_of, .{});
    module.vec_of = vec_of;
    const mat_of = try allocator.alloc(Mat, bound);
    defer allocator.free(mat_of);
    @memset(mat_of, .{});
    module.mat_of = mat_of;
    const mat_ptr = try allocator.alloc(?MatPtr, bound);
    defer allocator.free(mat_ptr);
    @memset(mat_ptr, null);
    module.mat_ptr = mat_ptr;
    const vec_ptr = try allocator.alloc(?VecPtr, bound);
    defer allocator.free(vec_ptr);
    @memset(vec_ptr, null);
    module.vec_ptr = vec_ptr;
    const local_is_vec = try allocator.alloc(bool, bound);
    defer allocator.free(local_is_vec);
    @memset(local_is_vec, false);
    module.local_is_vec = local_is_vec;
    const local_vec = try allocator.alloc(Vec, bound);
    defer allocator.free(local_vec);
    @memset(local_vec, .{});
    module.local_vec = local_vec;
    const local_comp_of = try allocator.alloc(?LocalComp, bound);
    defer allocator.free(local_comp_of);
    @memset(local_comp_of, null);
    module.local_comp_of = local_comp_of;
    // Classify surviving Function-storage vector OpVariables (mem2reg leaves these when a
    // component's address is taken) so the body's store/load/access-chain scalarize them.
    {
        var id: u32 = 0;
        while (id < module.var_storage.len) : (id += 1) {
            if (module.var_storage[id] == op.StorageClass.function and module.var_kind[id] == .other) {
                if (module.var_pointee(id)) |pt| {
                    if (vectorInfo(module.types, pt) != null) local_is_vec[id] = true;
                }
            }
        }
    }
    const is_builtin_ptr = try allocator.alloc(bool, bound);
    defer allocator.free(is_builtin_ptr);
    @memset(is_builtin_ptr, false);
    module.is_builtin_ptr = is_builtin_ptr;
    const sampler_ptr_of = try allocator.alloc(?Value, bound);
    defer allocator.free(sampler_ptr_of);
    @memset(sampler_ptr_of, null);
    module.sampler_ptr_of = sampler_ptr_of;
    module.sampler_fn = null;
    module.sampler_cube_fn = null;
    module.sampler_3d_fn = null;
    module.sampler_2darray_fn = null;
    module.sampler_shadow_fn = null;
    module.sampler_cube_shadow_fn = null;
    module.sampler_2darray_shadow_fn = null;
    module.sampler_gather_fn = null;
    module.sampler_fetch_fn = null;
    module.sampler_fetch_array_fn = null;
    module.sampler_fetch_3d_fn = null;
    module.math_fn = null;
    const input_comp_of = try allocator.alloc(?Value, bound);
    defer allocator.free(input_comp_of);
    @memset(input_comp_of, null);
    module.input_comp_of = input_comp_of;
    const pos_chain_var = try allocator.alloc(u32, bound);
    defer allocator.free(pos_chain_var);
    @memset(pos_chain_var, 0);
    module.pos_chain_var = pos_chain_var;

    // Discover blocks (each OpLabel) and their phis.
    var blocks: std.ArrayList(BlockInfo) = .empty;
    defer {
        for (blocks.items) |*b| b.phis.deinit(allocator);
        blocks.deinit(allocator);
    }
    var cur: ?usize = null;
    for (insts) |inst| {
        switch (inst.opcode) {
            op.Label => {
                const b = try func.appendBlock();
                try blocks.append(allocator, .{ .block = b, .label = try operandAt(inst.operands, 0) });
                cur = blocks.items.len - 1;
            },
            // The phi result id indexes value_of / vec_of below, so it must be in range.
            op.Phi => try blocks.items[cur orelse return error.MalformedModule].phis.append(allocator, try idOperandAt(inst.operands, 1, bound)),
            else => {},
        }
    }
    if (blocks.items.len == 0) return error.MalformedModule;
    // Every result-producing body instruction writes its result into the per-id arrays
    // (value_of / vec_of / sampler_ptr_of / ...) at operand 1. Those arrays are sized to the
    // untrusted id bound, so validate each result id up front (it exists and is in range): a
    // later write with a missing or out-of-range id would corrupt memory. Reads are
    // bounds-checked at their use sites.
    for (insts) |inst| {
        if (definesResult(inst.opcode)) _ = try idOperandAt(inst.operands, 1, bound);
    }
    const entry = blocks.items[0].block;

    // Synthesized entry parameters: invocation id (if used), then each storage buffer
    // (a base pointer), then the explicit function parameters.
    if (module.has_global_id) {
        const gid = try func.appendBlockParam(entry, module.i32_t);
        module.global_id_value = gid;
        // Tag the invocation-id parameter so a GPU backend sources it from the hardware
        // thread id (S2R) rather than a uniform kernel argument. CPU backends ignore the
        // tag and treat it as an ordinary register parameter.
        try func.addAttr(.{ .value = gid }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = op.BuiltIn.global_invocation_id } } });
    }
    // Vertex-shader BuiltIn inputs (gl_VertexIndex / gl_InstanceIndex) become synthesized
    // i32 entry params BEFORE the f32 attribute inputs and the buffer pointers, tagged so
    // a GPU backend sources them from the hardware (vertex/instance id registers) and the
    // CPU/host backend recognizes them as the leading integer params the draw supplies.
    if (module.has_vertex_index) {
        const vi = try func.appendBlockParam(entry, module.i32_t);
        module.vertex_index_value = vi;
        try func.addAttr(.{ .value = vi }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = op.BuiltIn.vertex_index } } });
    }
    if (module.has_instance_index) {
        const ii = try func.appendBlockParam(entry, module.i32_t);
        module.instance_index_value = ii;
        try func.addAttr(.{ .value = ii }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = op.BuiltIn.instance_index } } });
    }
    // Graphics: each Input variable becomes a scalarized block parameter, one per vector
    // component, tagged with its attribute slot (ATTR_GENERIC0 + the Location's 16-byte
    // stride + the component's 4-byte offset). An OpLoad of the variable yields the
    // recorded Vec. The backend emits one ALD/IPA per scalar.
    module.grad_entry = entry;
    if (module.stage != .compute) try synthInputAttribs(allocator, func, module, entry);
    for (module.buffers.items) |buf_id| {
        const p = try func.appendBlockParam(entry, module.ptr_t);
        value_of[buf_id] = p;
        // Tag a combined-image-sampler descriptor param distinctly from a plain buffer
        // (UBO / SSBO / push-constant) param. Both are pointer params, but the host
        // backend must supply a TEXTURE descriptor for a sampler and the bound buffer's
        // base pointer for a buffer. The tag lets it tell them apart (a GPU backend
        // ignores it - it already routes by the texture op vs the load). Without this a
        // FS that reads a UBO/push-constant would be fed the texture descriptor.
        if (module.var_kind[buf_id] == .sampler) {
            try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_desc", .value = .{ .int = 1 } } });
        }
        // Tag the descriptor's Vulkan binding number so a GPU backend can place it at the
        // matching slot in the SHARED constant bank. The constant bank is shared across
        // the VS + FS, so a per-stage declaration-order slot would collide (a VS UBO at
        // binding 0 and an FS sampler at binding 1 are each "first" in their stage). The
        // CPU backend ignores it (it binds by declaration order, one buffer-list per call).
        if (module.has_binding[buf_id]) {
            try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "binding", .value = .{ .int = @intCast(module.binding[buf_id]) } } });
        } else if (module.var_storage[buf_id] == op.StorageClass.push_constant) {
            // The default uniform block (the GLSL front end's loose `uniform`s) lowers to a
            // Push-Constant block, which carries no Binding decoration - but it occupies a
            // RESERVED per-stage block slot (samplers are numbered from 2). The two stages' blocks
            // hold DIFFERENT uniforms but each lays out from offset 0, so they get DISTINCT slots:
            // the VERTEX block at binding 0, the FRAGMENT block at binding 1. Without an explicit
            // tag the GPU backend falls back to a declaration-order slot, so a sampler declared
            // BEFORE the block (glmark2 bump-normals: `sampler2D NormalMap;` then the uniforms)
            // would push the block onto the sampler's slot - reading the bindless handle as its
            // base pointer (Xid 31 MMU fault) - and the FS block at the VS's slot 0 clobbers the
            // VS's matrix (the merged-UBO collision). Pin each stage's block to its own slot.
            const block_binding: i64 = if (module.stage == .vertex) 0 else 1;
            try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "binding", .value = .{ .int = block_binding } } });
        }
    }
    for (insts) |inst| {
        if (inst.opcode == op.FunctionParameter) {
            const ty = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
            value_of[try idOperandAt(inst.operands, 1, bound)] = try func.appendBlockParam(entry, ty);
        }
    }

    // If the fragment shader discards (OpKill), synthesize a discard function-pointer
    // entry param (tagged `discard_fn`): a CPU backend calls it to signal the fragment
    // is killed; a GPU/TGSI backend ignores the param and emits a KILL for the call.
    for (insts) |inst| {
        if (inst.opcode == op.Kill) {
            const dp = try func.appendBlockParam(entry, module.ptr_t);
            try func.addAttr(.{ .value = dp }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "discard_fn", .value = .flag } });
            module.discard_fn = dp;
            break;
        }
    }

    // Non-entry block parameters = that block's phis (in order). A vector phi is scalarized
    // into one block parameter per component (recorded as a Vec), exactly like every other
    // vector value. Its edge arguments are the incoming vectors' components, in the same order.
    for (blocks.items) |b| {
        for (b.phis.items) |phi_id| {
            const result_type = phiResultType(insts, phi_id) orelse return error.Unsupported;
            if (vectorInfo(module.types, result_type)) |vi| {
                const elem = scalarType(module.types, vi.elem) orelse return error.Unsupported;
                var out: Vec = .{ .len = vi.len };
                var c: u8 = 0;
                while (c < vi.len) : (c += 1) out.comps[c] = try func.appendBlockParam(b.block, elem);
                module.vec_of[phi_id] = out;
            } else {
                const ty = scalarType(module.types, result_type) orelse return error.Unsupported;
                value_of[phi_id] = try func.appendBlockParam(b.block, ty);
            }
        }
    }

    // Materialize constants into the entry block.
    var it = consts.iterator();
    while (it.next()) |e| {
        const ty = scalarType(module.types, e.value_ptr.type_id) orelse return error.Unsupported;
        const bits = e.value_ptr.bits;
        value_of[e.key_ptr.*] = switch (func.types.type_kind(ty)) {
            // `fconst` holds an f64 regardless of the value's own type (the widest common
            // carrier); widen the decoded bits up from whatever width the literal word(s) held.
            .float => |f| try func.appendInst(entry, ty, .{
                .fconst = switch (f) {
                    .f64 => @bitCast(bits),
                    .f32 => @as(f64, @as(f32, @bitCast(@as(u32, @truncate(bits))))),
                    // An f16 constant is packed into the low 16 bits of a single literal word.
                    .f16 => @as(f64, @as(f16, @bitCast(@as(u16, @truncate(bits))))),
                },
            }),
            else => try func.appendInst(entry, ty, .{ .iconst = @bitCast(bits) }),
        };
    }

    // Materialize OpConstantComposite vectors: a scalarized Vec of the already-
    // materialized component scalar values (e.g. a constant fragment color).
    var ci = composite_consts.iterator();
    while (ci.next()) |e| {
        var out: Vec = .{ .len = 0 };
        for (e.value_ptr.*) |comp_id| {
            if (comp_id < module.vec_of.len and module.vec_of[comp_id].len > 0) {
                for (module.vec_of[comp_id].comps[0..module.vec_of[comp_id].len]) |c| {
                    if (out.len >= out.comps.len) return error.MalformedModule;
                    out.comps[out.len] = c;
                    out.len += 1;
                }
            } else {
                if (out.len >= out.comps.len) return error.MalformedModule;
                out.comps[out.len] = try valueOf(module, comp_id);
                out.len += 1;
            }
        }
        module.vec_of[e.key_ptr.*] = out;
    }

    // Pass 1: lower each block's body (everything but the terminator), so all result ids
    // have Values before terminators read phi arguments.
    var bi: usize = std.math.maxInt(usize);
    for (insts) |inst| {
        if (inst.opcode == op.Label) {
            bi = blockIndex(blocks.items, inst.operands[0]);
            continue;
        }
        if (bi == std.math.maxInt(usize)) continue;
        try lowerBodyInst(allocator, func, module, blocks.items[bi].block, inst);
    }

    // Pass 2: wire each block's terminator, computing phi edge arguments.
    bi = std.math.maxInt(usize);
    for (insts) |inst| {
        if (inst.opcode == op.Label) {
            bi = blockIndex(blocks.items, inst.operands[0]);
            continue;
        }
        if (bi == std.math.maxInt(usize)) continue;
        try lowerTerminator(allocator, func, module, value_of, blocks.items, bi, insts, inst);
    }
    _ = ret_type_id;
}

/// Synthesize entry-block parameters for a graphics shader's Input variables. Each Input
/// variable (declaration order) becomes one block parameter per vector component, tagged
/// with its attribute byte slot so the backend fetches it (ALD in a vertex shader, IPA in
/// a fragment shader). The variable's scalar/Vec value is recorded so an OpLoad resolves
/// to these params.
fn synthInputAttribs(allocator: std.mem.Allocator, func: *Function, module: *Module, entry: Block) Error!void {
    // Variables by ascending id (SPIR-V declaration order).
    var id: u32 = 0;
    while (id < module.var_kind.len) : (id += 1) {
        if (module.var_kind[id] != .input) continue;
        const pointee = module.var_pointee(id) orelse return error.Unsupported;
        // gl_FragCoord (a vec4 window-space position) / gl_FrontFacing (a bool): builtin
        // inputs, NOT Location varyings. Each component param is tagged `builtin` +
        // `bicomp` so a backend sources it from the fragment position / face rather than
        // an interpolated varying (a location-0 attr slot would alias a real varying).
        if (module.is_frag_coord[id] or module.is_point_coord[id] or module.is_front_facing[id]) {
            const bi: i64 = if (module.is_frag_coord[id]) op.BuiltIn.frag_coord else if (module.is_point_coord[id]) op.BuiltIn.point_coord else op.BuiltIn.front_facing;
            if (vectorInfo(module.types, pointee)) |vi| {
                const elem = scalarType(module.types, vi.elem) orelse return error.Unsupported;
                var out: Vec = .{ .len = vi.len };
                var c: u8 = 0;
                while (c < vi.len) : (c += 1) {
                    const p = try func.appendBlockParam(entry, elem);
                    try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = bi } } });
                    try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "bicomp", .value = .{ .int = c } } });
                    out.comps[c] = p;
                }
                module.vec_of[id] = out;
            } else if (module.is_front_facing[id]) {
                // gl_FrontFacing is a bool, but every fragment input is delivered as an f32:
                // the software graphics ABI passes inputs in FP registers, and the nvidia
                // raster delivers the facing flag as an attribute value. A bool PARAM would
                // be classed into a GPR and break the calling convention (the arg registers
                // would shift and the shader would write color to a garbage pointer). So take
                // an f32 param and convert it to the bool the shader uses with `!= 0` (nonzero
                // => front). On nvidia the compare becomes the ISETP that sets the facing
                // predicate; in software it tests the 1.0/0.0 the rasterizer supplies.
                const f32_t = try func.types.intern(.{ .float = .f32 });
                const bool_t = try func.types.intern(.bool);
                const p = try func.appendBlockParam(entry, f32_t);
                try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = bi } } });
                try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "bicomp", .value = .{ .int = 0 } } });
                const zero = try func.appendInst(entry, f32_t, .{ .fconst = 0 });
                module.value_of[id] = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .ne, .lhs = p, .rhs = zero } });
            } else {
                const elem = scalarType(module.types, pointee) orelse return error.Unsupported;
                const p = try func.appendBlockParam(entry, elem);
                try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "builtin", .value = .{ .int = bi } } });
                try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "bicomp", .value = .{ .int = 0 } } });
                module.value_of[id] = p;
            }
            continue;
        }
        const loc = if (module.has_location[id]) module.location[id] else 0;
        const slot_base = ATTR_GENERIC0 + loc * 0x10;
        if (vectorInfo(module.types, pointee)) |vi| {
            const elem = scalarType(module.types, vi.elem) orelse return error.Unsupported;
            var out: Vec = .{ .len = vi.len };
            var c: u8 = 0;
            while (c < vi.len) : (c += 1) {
                const p = try func.appendBlockParam(entry, elem);
                try tagAttr(func, p, "attr", slot_base + c * 4);
                // Record the param's attribute slot so a screen-space derivative
                // (dFdx/dFdy/Fwidth) of this varying component can be lowered to a
                // gradient param tagged with the same slot (fragment stage only).
                try module.attr_slot_of.put(allocator, p, slot_base + c * 4);
                out.comps[c] = p;
            }
            module.vec_of[id] = out;
        } else {
            const elem = scalarType(module.types, pointee) orelse return error.Unsupported;
            const p = try func.appendBlockParam(entry, elem);
            try tagAttr(func, p, "attr", slot_base);
            try module.attr_slot_of.put(allocator, p, slot_base);
            module.value_of[id] = p;
        }
    }
}

/// Lower an `OpStore` to a graphics Output variable: scalarize the stored value and tag
/// each component store with its output attribute slot. A vertex shader's Position
/// builtin maps to ATTR_POSITION (clip-space output). A vertex shader's Location output
/// is a generic varying. A fragment shader's Location output is the render-target color
/// (tagged `color_out` = component index, which the backend places in ROP color registers
/// R0..R3).
fn storeOutputAttrib(func: *Function, module: *Module, block: Block, var_id: u32, value_id: u32) Error!void {
    const comps = scalarComps(module, value_id) catch return error.MalformedModule;
    const n = comps.len;

    // gl_FragDepth: a scalar fragment DEPTH output (not a render-target color). Tag it
    // `frag_depth` so the backend writes the fragment's depth instead of a color channel.
    if (module.stage == .fragment and module.is_frag_depth[var_id]) {
        const ptr = try func.appendInst(block, module.i32_t, .{ .iconst = 0 });
        try tagAttr(func, ptr, "frag_depth", 0);
        try func.appendStore(block, comps[0], ptr);
        return;
    }

    // A fragment shader's color output Location selects the render target (MRT): the
    // color_out tag encodes target*4 + component, so target 0 stays 0..3 (unchanged for
    // single-RT shaders) and target 1 is 4..7, etc.
    const frag_target: u32 = if (module.stage == .fragment and module.has_location[var_id]) module.location[var_id] else 0;

    var c: u8 = 0;
    while (c < n) : (c += 1) {
        const key: []const u8 = if (module.stage == .fragment) "color_out" else "out_attr";
        const slot: u32 = if (module.stage == .fragment)
            frag_target * 4 + c // fragment render-target index * 4 + component
        else if (module.is_position[var_id])
            ATTR_POSITION + c * 4
        else if (module.is_point_size[var_id])
            ATTR_POINT_SIZE // gl_PointSize: a scalar, routed to its own attribute (not a varying)
        else blk: {
            const loc = if (module.has_location[var_id]) module.location[var_id] else 0;
            break :blk ATTR_GENERIC0 + loc * 0x10 + c * 4;
        };
        // A placeholder pointer value carries the output-slot tag. The backend dispatches
        // on it (AST for an attribute, a register move for color).
        const ptr = try func.appendInst(block, module.i32_t, .{ .iconst = @intCast(slot) });
        try tagAttr(func, ptr, key, slot);
        try func.appendStore(block, comps[c], ptr);
    }
}

/// The scalar component Values of an SSA id: a single scalar, or a vector's components.
/// Returns a slice into a shared fixed buffer (valid until the next call), so copy if it
/// must outlive the use.
fn scalarComps(module: *Module, id: u32) error{MalformedModule}![]const Value {
    const S = struct {
        var buf: [4]Value = undefined;
    };
    if (id < module.vec_of.len and module.vec_of[id].len > 0) {
        const v = module.vec_of[id];
        @memcpy(S.buf[0..v.len], v.comps[0..v.len]);
        return S.buf[0..v.len];
    }
    if (id >= module.value_of.len) return error.MalformedModule; // untrusted id, keep the read in bounds
    S.buf[0] = module.value_of[id] orelse return error.MalformedModule;
    return S.buf[0..1];
}

/// Attach a `vulcan.gpu` integer attribute named `key` (a graphics attribute slot) to
/// value `v`, mirroring how the backend reads it back.
fn tagAttr(func: *Function, v: Value, key: []const u8, slot: u32) Error!void {
    try func.addAttr(.{ .value = v }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = key, .value = .{ .int = @intCast(slot) } } });
}

/// Whether a body instruction defines a result id (always operand 1), as opposed to the
/// terminators / stores / no-result ops that define none. Every result-producing arm in
/// `lowerBodyInst` / the vector & ext helpers writes the per-id arrays at operand 1, so
/// validating that id up front (existence + id bound) makes every later write in-bounds. Unknown
/// opcodes count as result-defining (they are rejected by `lowerBodyInst` regardless).
fn definesResult(opcode: u16) bool {
    return switch (opcode) {
        op.Label,
        op.Store,
        op.Branch,
        op.BranchConditional,
        op.Switch,
        op.Return,
        op.ReturnValue,
        op.Unreachable,
        op.Kill,
        op.SelectionMerge,
        op.LoopMerge,
        op.Name,
        op.MemberName,
        op.Decorate,
        op.MemberDecorate,
        op.Nop,
        => false,
        else => true,
    };
}

/// Lower a non-terminator body instruction, recording its result Value.
fn lowerBodyInst(allocator: std.mem.Allocator, func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const value_of = module.value_of;
    const bound = value_of.len; // == the id bound; every per-id array is this length
    // Vectors are scalarized: composite/shuffle/vector-arith operate per component.
    if (try lowerVectorInst(allocator, func, module, block, inst)) return;

    if (binOpOf(inst.opcode)) |bop| {
        const ty = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
        const lhs = try valueOf(module, try operandAt(inst.operands, 2));
        const rhs = try valueOf(module, try operandAt(inst.operands, 3));
        value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .arith = .{ .op = bop, .lhs = lhs, .rhs = rhs } });
        return;
    }
    if (cmpOpOf(inst.opcode)) |cop| {
        const ty = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
        const lhs = try valueOf(module, try operandAt(inst.operands, 2));
        const rhs = try valueOf(module, try operandAt(inst.operands, 3));
        value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .icmp = .{ .op = cop, .lhs = lhs, .rhs = rhs } });
        return;
    }
    switch (inst.opcode) {
        op.Select => {
            const ty = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
            const cond = try valueOf(module, try operandAt(inst.operands, 2));
            const a = try valueOf(module, try operandAt(inst.operands, 3));
            const b = try valueOf(module, try operandAt(inst.operands, 4));
            value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .select = .{ .cond = cond, .then = a, .@"else" = b } });
        },
        op.ConvertFToU, op.ConvertFToS, op.ConvertSToF, op.ConvertUToF, op.UConvert, op.SConvert, op.FConvert => {
            const ty = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
            const v = try valueOf(module, try operandAt(inst.operands, 2));
            value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .convert = .{ .value = v } });
        },
        op.Bitcast => {
            // Reinterpret the operand's bits as the result type (e.g. int <-> uint). Lowers
            // to the IR `reinterpret` unary op.
            const ty = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
            const v = try valueOf(module, try operandAt(inst.operands, 2));
            value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .unary = .{ .op = .reinterpret, .value = v } });
        },
        op.SNegate, op.FNegate => {
            // Unary negate: 0 - x (integer `sub` for SNegate, float `sub` for FNegate,
            // dispatched by the result type in codegen).
            const ty = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
            const v = try valueOf(module, try operandAt(inst.operands, 2));
            const zero = if (inst.opcode == op.FNegate)
                try func.appendInst(block, ty, .{ .fconst = 0 })
            else
                try func.appendInst(block, ty, .{ .iconst = 0 });
            value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = v } });
        },
        op.Not => {
            // Bitwise complement: x ^ -1 (all ones).
            const ty = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
            const v = try valueOf(module, try operandAt(inst.operands, 2));
            value_of[inst.operands[1]] = try func.appendArithImm(block, ty, .bit_xor, v, -1);
        },
        op.ExtInst => try lowerExtInst(allocator, func, module, block, inst),
        op.DPdx, op.DPdy, op.Fwidth => {
            // A scalar screen-space derivative of a varying scalar.
            const axis: Axis = if (inst.opcode == op.DPdy) .y else .x;
            value_of[inst.operands[1]] = try derivativeOf(allocator, func, module, block, try valueOf(module, try operandAt(inst.operands, 2)), axis, inst.opcode == op.Fwidth);
        },
        op.Undef => {
            // A safe concrete value for an undefined: zero of the result type.
            const ty = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
            value_of[inst.operands[1]] = if (func.types.type_kind(ty) == .float)
                try func.appendInst(block, ty, .{ .fconst = 0 })
            else
                try func.appendInst(block, ty, .{ .iconst = 0 });
        },
        op.Nop => {},
        op.AccessChain => try lowerAccessChain(func, module, block, inst),
        op.Load => {
            const result = inst.operands[1];
            const ptr_id = try idOperandAt(inst.operands, 2, bound); // indexes many per-id arrays below
            if (ptr_id < module.local_is_vec.len and module.local_is_vec[ptr_id]) {
                // A whole-vector load of a tracked Function-storage vector local: yield the
                // scalarized vector last stored to it.
                module.vec_of[result] = module.local_vec[ptr_id];
            } else if (ptr_id < module.local_comp_of.len and module.local_comp_of[ptr_id] != null) {
                // A scalar load of a constant-index access chain into a tracked vector local:
                // yield that component of the local's last-stored vector.
                const lc = module.local_comp_of[ptr_id].?;
                const lv = module.local_vec[lc.local];
                if (lc.comp >= lv.len) return error.MalformedModule;
                value_of[result] = lv.comps[lc.comp];
            } else if (ptr_id < module.var_kind.len and module.var_kind[ptr_id] == .sampler) {
                // A load of a combined-image-sampler variable yields its descriptor
                // pointer (the synthesized pointer entry param). The following
                // OpSampledImage / OpImageSample* reads it.
                module.sampler_ptr_of[result] = value_of[ptr_id] orelse return error.MalformedModule;
                // Carry the sampler's image Dim (from the loaded variable's POINTER TYPE) so
                // lowerImageSample can pick the 3D vs cube host tag.
                if (ptr_id < module.var_type.len) {
                    const pt = module.var_type[ptr_id];
                    if (pt < module.type_image_dim.len) module.sampler_dim_of[result] = module.type_image_dim[pt];
                    if (pt < module.type_image_arrayed.len) module.sampler_arrayed_of[result] = module.type_image_arrayed[pt];
                }
            } else if (module.is_builtin_ptr[ptr_id]) {
                // A load of gl_GlobalInvocationID.x is the synthesized invocation id.
                value_of[result] = module.global_id_value orelse return error.MalformedModule;
            } else if (ptr_id < module.var_kind.len and module.var_kind[ptr_id] == .vertex_index) {
                // A load of gl_VertexIndex (directly or through a scalar access chain) is
                // the synthesized per-vertex index param.
                value_of[result] = module.vertex_index_value orelse return error.MalformedModule;
            } else if (ptr_id < module.var_kind.len and module.var_kind[ptr_id] == .instance_index) {
                value_of[result] = module.instance_index_value orelse return error.MalformedModule;
            } else if (ptr_id < module.input_comp_of.len and module.input_comp_of[ptr_id] != null) {
                // Pattern A: a scalar load of a per-component input access chain yields
                // the addressed input component param directly.
                value_of[result] = module.input_comp_of[ptr_id].?;
            } else if (ptr_id < module.var_kind.len and module.var_kind[ptr_id] == .input) {
                // A graphics Input variable: the load yields the synthesized
                // attribute params (a scalar, or a Vec for a vector input).
                if (module.vec_of[ptr_id].len > 0) {
                    module.vec_of[result] = module.vec_of[ptr_id];
                } else {
                    value_of[result] = value_of[ptr_id] orelse return error.MalformedModule;
                }
            } else if (ptr_id < module.mat_ptr.len and module.mat_ptr[ptr_id] != null) {
                // A matrix in a buffer: fetch each element (column-major) from
                // `base + col*stride + row*elemSize`, scalarizing it into a Mat.
                try loadMatrix(func, module, block, result, module.mat_ptr[ptr_id].?);
            } else if (ptr_id < module.vec_ptr.len and module.vec_ptr[ptr_id] != null) {
                // A vector in a buffer (e.g. a UBO `vec4` array element pulled by
                // gl_VertexIndex): fetch each component from `base + comp*elemSize` into
                // a scalarized Vec.
                try loadVector(func, module, block, result, module.vec_ptr[ptr_id].?);
            } else if (ptr_id < module.pos_chain_var.len and module.pos_chain_var[ptr_id] != 0 and module.pos_chain_var[ptr_id] != std.math.maxInt(u32)) {
                // A read-back of the shader's own gl_Position (the gl_PerVertex Position
                // member): yield the previously-stored clip-space position Vec. vkcube's
                // VS does this (`frag_pos = gl_Position.xyz`). The position must already
                // have been stored (glslang writes gl_Position before reading it back).
                if (module.position_vec.len == 0) return error.MalformedModule;
                module.vec_of[result] = module.position_vec;
            } else {
                const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
                const ptr = value_of[ptr_id] orelse return error.MalformedModule;
                value_of[result] = try func.appendInst(block, ty, .{ .load = .{ .ptr = ptr } });
            }
        },
        op.Store => {
            const ptr_id = try idOperandAt(inst.operands, 0, bound); // indexes per-id arrays below
            const val_id = try operandAt(inst.operands, 1);
            // A whole-vector store to a tracked Function-storage vector local: record the
            // scalarized vector as the local's current value (subsequent loads read it).
            if (ptr_id < module.local_is_vec.len and module.local_is_vec[ptr_id]) {
                const comps = scalarComps(module, val_id) catch return error.MalformedModule;
                var v: Vec = .{ .len = @intCast(comps.len) };
                for (comps, 0..) |cv, ci| v.comps[ci] = cv;
                module.local_vec[ptr_id] = v;
                return;
            }
            // A component store through a constant-index access chain into a tracked vector
            // local: update that one component of the local's current vector.
            if (ptr_id < module.local_comp_of.len and module.local_comp_of[ptr_id] != null) {
                const lc = module.local_comp_of[ptr_id].?;
                const sv = try valueOf(module, val_id);
                if (lc.comp >= module.local_vec[lc.local].len) {
                    if (lc.comp >= module.local_vec[lc.local].comps.len) return error.MalformedModule;
                    module.local_vec[lc.local].len = lc.comp + 1;
                }
                module.local_vec[lc.local].comps[lc.comp] = sv;
                return;
            }
            // Pattern B: a store through the gl_PerVertex Position member access chain is
            // the clip-space position output (routed to ATTR_POSITION). A store through a
            // non-position member (PointSize etc.) is dropped - we model position only.
            if (ptr_id < module.pos_chain_var.len and module.pos_chain_var[ptr_id] != 0) {
                const gpv = module.pos_chain_var[ptr_id];
                if (gpv == std.math.maxInt(u32)) return; // a PointSize/Clip member: drop
                module.is_position[gpv] = true; // route storeOutputAttrib to ATTR_POSITION
                // Remember the stored clip-space position so the shader can read it back
                // (e.g. `frag_pos = gl_Position.xyz`). Scalarized into a Vec.
                const comps = scalarComps(module, val_id) catch return error.MalformedModule;
                var pv: Vec = .{ .len = @intCast(comps.len) };
                for (comps, 0..) |cv, ci| pv.comps[ci] = cv;
                module.position_vec = pv;
                try storeOutputAttrib(func, module, block, gpv, val_id);
                return;
            }
            // A graphics Output variable: scalarize the stored vector and tag each
            // component store with its output attribute slot.
            if (ptr_id < module.var_kind.len and module.var_kind[ptr_id] == .output) {
                try storeOutputAttrib(func, module, block, ptr_id, val_id);
                return;
            }
            const ptr = try valueOf(module, ptr_id);
            const val = try valueOf(module, val_id);
            try func.appendStore(block, val, ptr);
        },
        op.SampledImage => {
            // [type, result, image, sampler]. A combined-image-sampler is already a
            // sampled image, so forward the image operand's descriptor pointer.
            const image_id = try operandAt(inst.operands, 2);
            module.sampler_ptr_of[inst.operands[1]] = (if (image_id < module.sampler_ptr_of.len) module.sampler_ptr_of[image_id] else null) orelse return error.MalformedModule;
        },
        op.ImageSampleImplicitLod, op.ImageSampleExplicitLod => try lowerImageSample(func, module, block, inst),
        op.ImageSampleDrefImplicitLod, op.ImageSampleDrefExplicitLod => try lowerImageSampleShadow(func, module, block, inst),
        op.ImageGather => try lowerImageGather(func, module, block, inst),
        op.Image => {
            // [type, result, sampledImage]: extract the raw image (for OpImageFetch). Carry the
            // descriptor pointer + dim + arrayed through unchanged - the following fetch reads them.
            const si = try operandAt(inst.operands, 2);
            module.sampler_ptr_of[inst.operands[1]] = (if (si < module.sampler_ptr_of.len) module.sampler_ptr_of[si] else null) orelse return error.MalformedModule;
            if (si < module.sampler_dim_of.len) module.sampler_dim_of[inst.operands[1]] = module.sampler_dim_of[si];
            if (si < module.sampler_arrayed_of.len) module.sampler_arrayed_of[inst.operands[1]] = module.sampler_arrayed_of[si];
        },
        op.ImageFetch => try lowerImageFetch(func, module, block, inst),
        op.Variable => {
            // A tracked Function-storage vector local is scalarized via local_vec (its
            // store/load/access-chain are handled below). Its declaration needs no IR.
            if (inst.operands[1] < module.local_is_vec.len and module.local_is_vec[inst.operands[1]]) return;
            return error.Unsupported; // other function-local variables (alloca) not yet modeled
        },
        op.Phi, op.FunctionParameter => {},
        op.SelectionMerge, op.LoopMerge, op.Name, op.MemberName, op.Decorate, op.MemberDecorate => {},
        op.Branch, op.BranchConditional, op.Switch, op.Return, op.ReturnValue, op.Unreachable, op.Kill => {}, // terminators (Pass 2)
        else => return error.Unsupported,
    }
}

fn vectorInfo(types: []const ?TypeInfo, type_id: u32) ?VecType {
    if (type_id >= types.len) return null;
    return (types[type_id] orelse return null).asVector();
}

/// Lower vector instructions by scalarizing: a vector value is a list of scalar component
/// values, a vector operation becomes one scalar operation per component. Returns true if
/// `inst` was a vector op (handled here), false if it is a scalar op for the caller. No
/// backend needs vector support.
fn lowerVectorInst(allocator: std.mem.Allocator, func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!bool {
    const v_of = module.vec_of;
    const bound = v_of.len; // == the id bound; every per-id array is this length
    switch (inst.opcode) {
        op.CompositeConstruct => {
            const vi = vectorInfo(module.types, try operandAt(inst.operands, 0)) orelse return false; // struct construct: not here
            var out: Vec = .{ .len = 0 };
            for ((try operandsFrom(inst.operands, 2))) |cid_raw| {
                const cid = try checkId(cid_raw, bound);
                // A component may itself be a (sub)vector, so flatten it.
                if (v_of[cid].len > 0) {
                    for (v_of[cid].comps[0..v_of[cid].len]) |c| {
                        if (out.len >= out.comps.len) return error.MalformedModule;
                        out.comps[out.len] = c;
                        out.len += 1;
                    }
                } else {
                    if (out.len >= out.comps.len) return error.MalformedModule;
                    out.comps[out.len] = module.value_of[cid] orelse return error.MalformedModule;
                    out.len += 1;
                }
            }
            if (out.len != vi.len) return error.MalformedModule;
            v_of[inst.operands[1]] = out;
            return true;
        },
        op.CompositeExtract => {
            // [type, result, composite, index...]. The indices are literals. A single
            // index into a vector yields the component scalar.
            const cv = v_of[try idOperandAt(inst.operands, 2, bound)];
            if (cv.len == 0) return false; // extracting from a struct/array: not here
            const idx = try operandAt(inst.operands, 3);
            if (idx >= cv.len) return error.MalformedModule;
            module.value_of[inst.operands[1]] = cv.comps[idx];
            return true;
        },
        op.VectorShuffle => {
            // [type, result, vec1, vec2, indices...]. An index < len(vec1) selects
            // from vec1, else from vec2 (offset by len(vec1)).
            const v1 = v_of[try idOperandAt(inst.operands, 2, bound)];
            const v2 = v_of[try idOperandAt(inst.operands, 3, bound)];
            var out: Vec = .{ .len = 0 };
            for ((try operandsFrom(inst.operands, 4))) |ix| {
                if (out.len >= out.comps.len) return error.MalformedModule;
                // 0xFFFFFFFF is SPIR-V's "undefined" component; any index must land in v1|v2.
                if (ix >= v1.len + v2.len) return error.MalformedModule;
                out.comps[out.len] = if (ix < v1.len) v1.comps[ix] else v2.comps[ix - v1.len];
                out.len += 1;
            }
            v_of[inst.operands[1]] = out;
            return true;
        },
        op.MatrixTimesVector => {
            try lowerMatrixTimesVector(func, module, block, inst);
            return true;
        },
        op.VectorTimesMatrix => {
            try lowerVectorTimesMatrix(func, module, block, inst);
            return true;
        },
        op.MatrixTimesScalar => {
            try lowerMatrixTimesScalar(func, module, block, inst);
            return true;
        },
        op.MatrixTimesMatrix => {
            try lowerMatrixTimesMatrix(func, module, block, inst);
            return true;
        },
        op.VectorTimesScalar => {
            const vi = vectorInfo(module.types, try operandAt(inst.operands, 0)) orelse return false;
            const elem = scalarType(module.types, vi.elem) orelse return error.Unsupported;
            const vec = v_of[try idOperandAt(inst.operands, 2, bound)];
            const s = try valueOf(module, try operandAt(inst.operands, 3));
            var out: Vec = .{ .len = vi.len };
            for (0..vi.len) |i| out.comps[i] = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = vec.comps[i], .rhs = s } });
            v_of[inst.operands[1]] = out;
            return true;
        },
        op.DPdx, op.DPdy, op.Fwidth => {
            // A screen-space derivative of a varying. For a vector operand, scalarize:
            // each component's derivative is the gradient of that varying component.
            const v = v_of[try idOperandAt(inst.operands, 2, bound)];
            if (v.len == 0) return false; // scalar derivative: handled in lowerBodyInst
            const axis: Axis = if (inst.opcode == op.DPdy) .y else .x;
            var out: Vec = .{ .len = v.len };
            var i: u8 = 0;
            while (i < v.len) : (i += 1) {
                out.comps[i] = try derivativeOf(allocator, func, module, block, v.comps[i], axis, inst.opcode == op.Fwidth);
            }
            v_of[inst.operands[1]] = out;
            return true;
        },
        op.Select => {
            // A vector select: component-wise `cond ? then : else`. The result is a vector;
            // a scalar-result select falls through to the scalar path in lowerBodyInst.
            // The condition is either a per-component bool VECTOR or (SPIR-V >= 1.4) a single
            // scalar bool broadcast across the components (e.g. `gl_FrontFacing ? a : b`).
            const vi = vectorInfo(module.types, try operandAt(inst.operands, 0)) orelse return false;
            const elem = scalarType(module.types, vi.elem) orelse return error.Unsupported;
            const cond_id = try idOperandAt(inst.operands, 2, bound);
            const cond_vec = v_of[cond_id];
            const a = v_of[try idOperandAt(inst.operands, 3, bound)];
            const b = v_of[try idOperandAt(inst.operands, 4, bound)];
            var out: Vec = .{ .len = vi.len };
            for (0..vi.len) |i| {
                const cond = if (cond_vec.len > 0)
                    cond_vec.comps[i]
                else
                    module.value_of[cond_id] orelse return error.MalformedModule;
                out.comps[i] = try func.appendInst(block, elem, .{ .select = .{ .cond = cond, .then = a.comps[i], .@"else" = b.comps[i] } });
            }
            v_of[inst.operands[1]] = out;
            return true;
        },
        op.Dot => {
            // A scalar result: sum of component products.
            const elem = scalarType(module.types, try operandAt(inst.operands, 0)) orelse return error.Unsupported;
            const v1 = v_of[try idOperandAt(inst.operands, 2, bound)];
            const v2 = v_of[try idOperandAt(inst.operands, 3, bound)];
            if (v1.len == 0) return false;
            var acc = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = v1.comps[0], .rhs = v2.comps[0] } });
            for (1..v1.len) |i| {
                const p = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = v1.comps[i], .rhs = v2.comps[i] } });
                acc = try func.appendInst(block, elem, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
            }
            module.value_of[inst.operands[1]] = acc;
            return true;
        },
        else => {
            // Component-wise arithmetic on a vector result (OpFAdd/OpIMul/...).
            if (binOpOf(inst.opcode)) |bop| {
                if (vectorInfo(module.types, try operandAt(inst.operands, 0))) |vi| {
                    const elem = scalarType(module.types, vi.elem) orelse return error.Unsupported;
                    const a = v_of[try idOperandAt(inst.operands, 2, bound)];
                    const b = v_of[try idOperandAt(inst.operands, 3, bound)];
                    var out: Vec = .{ .len = vi.len };
                    for (0..vi.len) |i| out.comps[i] = try func.appendInst(block, elem, .{ .arith = .{ .op = bop, .lhs = a.comps[i], .rhs = b.comps[i] } });
                    v_of[inst.operands[1]] = out;
                    return true;
                }
            }
            return false;
        },
    }
}

/// Which screen-space axis a derivative is taken along.
const Axis = enum { x, y };

/// The dFdx / dFdy gradient of a fragment-shader varying scalar. A linearly-interpolated
/// varying has a CONSTANT screen-space gradient per triangle, so the derivative is just
/// that gradient - the rasterizer computes it per triangle and supplies it through the
/// synthesized `grad_buf` pointer param. `scalar` must be a varying-component value
/// (recorded in `attr_slot_of`). Each distinct (scalar, axis) gets a dense `grad_buf`
/// index and the derivative LOADS `grad_buf[index]`. `fwidth` returns |dFdx| + |dFdy| (the
/// standard Fwidth definition) instead of a single-axis gradient.
fn derivativeOf(allocator: std.mem.Allocator, func: *Function, module: *Module, block: Block, scalar: Value, axis: Axis, fwidth: bool) Error!Value {
    if (fwidth) {
        const dx = try loadGradient(allocator, func, module, block, scalar, .x);
        const dy = try loadGradient(allocator, func, module, block, scalar, .y);
        const adx = try fabs(func, block, dx);
        const ady = try fabs(func, block, dy);
        const f32_t = try func.types.intern(.{ .float = .f32 });
        return func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = adx, .rhs = ady } });
    }
    return loadGradient(allocator, func, module, block, scalar, axis);
}

/// |x| as a select (x < 0 ? -x : x), float.
fn fabs(func: *Function, block: Block, x: Value) Error!Value {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);
    const zero = try func.appendInst(block, f32_t, .{ .fconst = 0 });
    const lt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = zero } });
    const neg = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = x } });
    return func.appendInst(block, f32_t, .{ .select = .{ .cond = lt, .then = neg, .@"else" = x } });
}

/// Load the dFdx / dFdy gradient of a varying-component scalar from the synthesized
/// `grad_buf` pointer param. The first time any derivative is taken, `grad_buf` is lazily
/// appended (a pointer entry param tagged `vulcan.gpu.grad_buf`, it lands in the GPR file
/// like the other buffer pointers - so the FS's float-register varying interface is
/// untouched no matter how many derivatives are taken). The first time a given (scalar,
/// axis) is requested, a dense buffer index is assigned and its (attr slot, axis) recorded
/// in `grad_descs` so the rasterizer fills `grad_buf[index]`. The result is a load of
/// `grad_buf + index*4`. The scalar MUST be a varying component (present in
/// `attr_slot_of`). Deriving an arbitrary non-affine value is unsupported (it would need
/// 2x2-pixel-quad shading) and errors.
fn loadGradient(allocator: std.mem.Allocator, func: *Function, module: *Module, block: Block, scalar: Value, axis: Axis) Error!Value {
    const slot = module.attr_slot_of.get(scalar) orelse return error.Unsupported;
    // Lazily create the grad_buf pointer param (once, the first derivative anywhere).
    if (module.grad_buf_param == null) {
        const gp = try func.appendBlockParam(module.grad_entry, module.ptr_t);
        try func.addAttr(.{ .value = gp }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_buf", .value = .{ .int = 0 } } });
        module.grad_buf_param = gp;
    }
    const cache = switch (axis) {
        .x => &module.grad_index_x,
        .y => &module.grad_index_y,
    };
    const index: u32 = if (cache.get(scalar)) |i| i else blk: {
        const i: u32 = @intCast(module.grad_descs.items.len);
        try module.grad_descs.append(allocator, .{ .slot = slot, .axis = switch (axis) {
            .x => .x,
            .y => .y,
        } });
        try cache.put(allocator, scalar, i);
        break :blk i;
    };
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const base = module.grad_buf_param.?;
    const ptr = if (index == 0)
        base
    else
        try func.appendInst(block, module.ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = try func.appendInst(block, module.i32_t, .{ .iconst = @intCast(index * 4) }) } });
    return func.appendInst(block, f32_t, .{ .load = .{ .ptr = ptr } });
}

/// Lower the float-math / vector GLSL ext-insts (sqrt, inversesqrt, length, cross,
/// normalize) to native arithmetic over scalarized components. Returns true if `which`
/// was one of these (and the result was recorded), false to fall through to the scalar
/// select/compare set.
fn lowerMathExtInst(func: *Function, module: *Module, block: Block, which: u32, result: u32, args: []const u32) Error!bool {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    switch (which) {
        // Transcendentals without a portable closed form (pow / exp / log / sin / cos and
        // their base-2 variants) are dispatched to a host math function pointer, exactly like
        // the image sampler: a `f32 math_fn(op:i32, a:f32, b:f32)` entry param the CPU backend
        // binds (a GPU backend emits a MUFU instead). `a` is the primary operand, `b` is the
        // exponent for pow (0 otherwise).
        op.Glsl.pow => {
            const a = try valueOf(module, try operandAt(args, 0));
            const b = try valueOf(module, try operandAt(args, 1));
            module.value_of[result] = try callMathFn(func, module, block, MATH_POW, a, b);
            return true;
        },
        op.Glsl.exp => {
            const x = try valueOf(module, try operandAt(args, 0));
            module.value_of[result] = try callMathFn(func, module, block, MATH_EXP, x, null);
            return true;
        },
        op.Glsl.log => {
            const x = try valueOf(module, try operandAt(args, 0));
            module.value_of[result] = try callMathFn(func, module, block, MATH_LOG, x, null);
            return true;
        },
        op.Glsl.exp2 => {
            const x = try valueOf(module, try operandAt(args, 0));
            module.value_of[result] = try callMathFn(func, module, block, MATH_EXP2, x, null);
            return true;
        },
        op.Glsl.log2 => {
            const x = try valueOf(module, try operandAt(args, 0));
            module.value_of[result] = try callMathFn(func, module, block, MATH_LOG2, x, null);
            return true;
        },
        op.Glsl.sin => {
            const x = try valueOf(module, try operandAt(args, 0));
            module.value_of[result] = try callMathFn(func, module, block, MATH_SIN, x, null);
            return true;
        },
        op.Glsl.cos => {
            const x = try valueOf(module, try operandAt(args, 0));
            module.value_of[result] = try callMathFn(func, module, block, MATH_COS, x, null);
            return true;
        },
        op.Glsl.sqrt => {
            const x = try valueOf(module, try operandAt(args, 0));
            module.value_of[result] = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .sqrt, .value = x } });
            return true;
        },
        op.Glsl.inverse_sqrt => {
            // 1.0 / sqrt(x).
            const x = try valueOf(module, try operandAt(args, 0));
            const s = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .sqrt, .value = x } });
            const one = try func.appendInst(block, f32_t, .{ .fconst = 1.0 });
            module.value_of[result] = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .div, .lhs = one, .rhs = s } });
            return true;
        },
        op.Glsl.length => {
            // length(v) = sqrt(dot(v, v)).
            const v = try vecOf(module, try operandAt(args, 0));
            if (v.len == 0) return false;
            const d2 = try dotSelf(func, block, v);
            module.value_of[result] = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .sqrt, .value = d2 } });
            return true;
        },
        op.Glsl.normalize => {
            // normalize(v) = v * inversesqrt(dot(v, v)).
            const v = try vecOf(module, try operandAt(args, 0));
            if (v.len == 0) return false;
            const d2 = try dotSelf(func, block, v);
            const s = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .sqrt, .value = d2 } });
            const one = try func.appendInst(block, f32_t, .{ .fconst = 1.0 });
            const inv = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .div, .lhs = one, .rhs = s } });
            var out: Vec = .{ .len = v.len };
            var i: u8 = 0;
            while (i < v.len) : (i += 1) {
                out.comps[i] = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = v.comps[i], .rhs = inv } });
            }
            module.vec_of[result] = out;
            return true;
        },
        op.Glsl.cross => {
            // cross(a, b) = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x).
            const a = try vecOf(module, try operandAt(args, 0));
            const b = try vecOf(module, try operandAt(args, 1));
            if (a.len != 3 or b.len != 3) return false;
            const c0 = try crossTerm(func, block, a.comps[1], b.comps[2], a.comps[2], b.comps[1]);
            const c1 = try crossTerm(func, block, a.comps[2], b.comps[0], a.comps[0], b.comps[2]);
            const c2 = try crossTerm(func, block, a.comps[0], b.comps[1], a.comps[1], b.comps[0]);
            module.vec_of[result] = .{ .len = 3, .comps = .{ c0, c1, c2, undefined } };
            return true;
        },
        else => return false,
    }
}

// Host-math function selector codes (passed as the first argument to `math_fn`). The CPU
// backend dispatches on these. A GPU backend maps each to its transcendental unit.
const MATH_POW: i64 = 0;
const MATH_EXP: i64 = 1;
const MATH_LOG: i64 = 2;
const MATH_EXP2: i64 = 3;
const MATH_LOG2: i64 = 4;
const MATH_SIN: i64 = 5;
const MATH_COS: i64 = 6;

/// Call the synthesized host-math function pointer for a transcendental: `f32 math_fn(op,
/// a, b)`. The `math_fn` pointer entry param is appended lazily (once, tagged
/// `vulcan.gpu.math_fn`), mirroring the image-sampler `sampler_fn` convention. `b` is the
/// second operand for `pow` (the exponent). For unary functions it is 0.
fn callMathFn(func: *Function, module: *Module, block: Block, math_op: i64, a: Value, b: ?Value) Error!Value {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const math_fn = if (module.math_fn) |m| m else blk: {
        const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
        try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "math_fn", .value = .flag } });
        module.math_fn = p;
        break :blk p;
    };
    const op_val = try func.appendInst(block, module.i32_t, .{ .iconst = math_op });
    const b_val = b orelse try func.appendInst(block, f32_t, .{ .fconst = 0 });
    return func.appendInst(block, f32_t, .{ .call_indirect = .{
        .target = math_fn,
        .args = try func.internValues(&.{ op_val, a, b_val }),
    } });
}

/// dot(v, v) = sum of v[i]*v[i] (float).
fn dotSelf(func: *Function, block: Block, v: Vec) Error!Value {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    var acc = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = v.comps[0], .rhs = v.comps[0] } });
    var i: u8 = 1;
    while (i < v.len) : (i += 1) {
        const p = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = v.comps[i], .rhs = v.comps[i] } });
        acc = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
    }
    return acc;
}

/// One cross-product component: p0*p1 - q0*q1 (float).
fn crossTerm(func: *Function, block: Block, p0: Value, p1: Value, q0: Value, q1: Value) Error!Value {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const a = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = p0, .rhs = p1 } });
    const b = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = q0, .rhs = q1 } });
    return func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } });
}

/// Lower a GLSL.std.450 extended instruction. Scalar set (min/max/abs/clamp) lowers to
/// select + compare. The float-math set (sqrt/inversesqrt and the vector ops cross/
/// normalize/length) lowers to native arithmetic (fsqrt + fmul/fadd/fdiv over scalarized
/// components). The comparison's signed/unsigned/float behavior comes from the operand
/// type in codegen, so one `icmp` shape serves the F/S/U variants.
fn lowerExtInst(allocator: std.mem.Allocator, func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    _ = allocator;
    if (inst.operands.len < 5) return error.Unsupported;
    const result = inst.operands[1];
    const which = inst.operands[3]; // operands: [type, result, set, instruction, args..]
    const args = inst.operands[4..];

    // Vector / float-math ext-insts (cross / normalize / length / sqrt / inversesqrt):
    // these operate on scalarized vectors or need an fsqrt, so they are lowered here
    // before the scalar select/compare set.
    if (try lowerMathExtInst(func, module, block, which, result, args)) return;

    const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
    const bool_t = try func.types.intern(.bool);

    const arg = struct {
        fn v(m: *Module, a: []const u32, i: usize) Error!Value {
            return valueOf(m, try operandAt(a, i));
        }
    }.v;

    value_of: {
        const out = switch (which) {
            op.Glsl.f_abs, op.Glsl.s_abs => blk: {
                const x = try arg(module, args, 0);
                const zero = if (func.types.type_kind(ty) == .float)
                    try func.appendInst(block, ty, .{ .fconst = 0 })
                else
                    try func.appendInst(block, ty, .{ .iconst = 0 });
                const lt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = zero } });
                const neg = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = x } });
                break :blk try func.appendInst(block, ty, .{ .select = .{ .cond = lt, .then = neg, .@"else" = x } });
            },
            op.Glsl.f_min, op.Glsl.u_min, op.Glsl.s_min => blk: {
                // min(a,b) = (b < a) ? b : a (the GLSL/SPIR-V definition, NaN-propagates).
                const a = try arg(module, args, 0);
                const b = try arg(module, args, 1);
                const lt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = b, .rhs = a } });
                break :blk try func.appendInst(block, ty, .{ .select = .{ .cond = lt, .then = b, .@"else" = a } });
            },
            op.Glsl.f_max, op.Glsl.u_max, op.Glsl.s_max => blk: {
                // max(a,b) = (a < b) ? b : a (the GLSL/SPIR-V definition, NaN-propagates, so
                // max(NaN, x) = NaN, matching real GL rather than collapsing to x).
                const a = try arg(module, args, 0);
                const b = try arg(module, args, 1);
                const lt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
                break :blk try func.appendInst(block, ty, .{ .select = .{ .cond = lt, .then = b, .@"else" = a } });
            },
            op.Glsl.floor => blk: {
                const x = try arg(module, args, 0);
                break :blk try func.appendInst(block, ty, .{ .unary = .{ .op = .floor, .value = x } });
            },
            op.Glsl.ceil => blk: {
                const x = try arg(module, args, 0);
                break :blk try func.appendInst(block, ty, .{ .unary = .{ .op = .ceil, .value = x } });
            },
            op.Glsl.trunc => blk: {
                const x = try arg(module, args, 0);
                break :blk try func.appendInst(block, ty, .{ .unary = .{ .op = .trunc, .value = x } });
            },
            op.Glsl.round_even => blk: {
                const x = try arg(module, args, 0);
                break :blk try func.appendInst(block, ty, .{ .unary = .{ .op = .nearest, .value = x } });
            },
            op.Glsl.sqrt => blk: {
                // sqrt has a hardware intrinsic (the IR sqrt op -> fsqrt), so it imports.
                const x = try arg(module, args, 0);
                break :blk try func.appendInst(block, ty, .{ .unary = .{ .op = .sqrt, .value = x } });
            },
            op.Glsl.inverse_sqrt => blk: {
                // inversesqrt(x) = 1 / sqrt(x).
                const x = try arg(module, args, 0);
                const s = try func.appendInst(block, ty, .{ .unary = .{ .op = .sqrt, .value = x } });
                const one = try func.appendInst(block, ty, .{ .fconst = 1 });
                break :blk try func.appendInst(block, ty, .{ .arith = .{ .op = .div, .lhs = one, .rhs = s } });
            },
            op.Glsl.fract => blk: {
                // fract(x) = x - floor(x).
                const x = try arg(module, args, 0);
                const fl = try func.appendInst(block, ty, .{ .unary = .{ .op = .floor, .value = x } });
                break :blk try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = fl } });
            },
            op.Glsl.f_clamp, op.Glsl.u_clamp, op.Glsl.s_clamp => blk: {
                // clamp(x, lo, hi) = min(max(x, lo), hi), each min/max in the spec `<`-and-
                // select-second form (consistent NaN behavior with the standalone min/max).
                const x = try arg(module, args, 0);
                const lo = try arg(module, args, 1);
                const hi = try arg(module, args, 2);
                // max(x, lo) = (x < lo) ? lo : x
                const xlt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = lo } });
                const m = try func.appendInst(block, ty, .{ .select = .{ .cond = xlt, .then = lo, .@"else" = x } });
                // min(m, hi) = (hi < m) ? hi : m
                const hlt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = hi, .rhs = m } });
                break :blk try func.appendInst(block, ty, .{ .select = .{ .cond = hlt, .then = hi, .@"else" = m } });
            },
            else => return error.Unsupported, // sqrt/sin/etc. need hardware intrinsics
        };
        module.value_of[result] = out;
        break :value_of;
    }
}

/// Lower `OpAccessChain`: produce a pointer Value by walking the index chain (a
/// getelementpointer). A chain into the `gl_GlobalInvocationID` builtin is marked (a later
/// load yields the invocation id). Otherwise the byte offset is accumulated: a struct
/// member index adds its decorated `Offset` (compile-time constant), an array index adds
/// `index * stride`. The result is `base + offset`.
fn lowerAccessChain(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const result = inst.operands[1];
    const base = try operandAt(inst.operands, 2);
    const indices = try operandsFrom(inst.operands, 3);

    if (base < module.var_kind.len and module.var_kind[base] == .global_id) {
        module.is_builtin_ptr[result] = true;
        return;
    }
    // A constant-index chain into a tracked Function-storage vector local: record (local,
    // component) so the following OpLoad/OpStore reads/updates that component of the local's
    // scalarized vector.
    if (base < module.local_is_vec.len and module.local_is_vec[base] and indices.len == 1) {
        const comp: u32 = @intCast(try constValOf(module, indices[0]));
        module.local_comp_of[result] = .{ .local = base, .comp = @intCast(comp) };
        return;
    }
    // A chain into gl_VertexIndex / gl_InstanceIndex (a scalar BuiltIn input): the result
    // pointer carries the same index "kind" as the variable, so the following OpLoad reads
    // the synthesized i32 index param. (These are scalar ints, so glslang usually loads the
    // variable directly - this handles a chain too, defensively.)
    if (base < module.var_kind.len and (module.var_kind[base] == .vertex_index or module.var_kind[base] == .instance_index)) {
        if (result < module.var_kind.len) module.var_kind[result] = module.var_kind[base];
        return;
    }

    // Pattern A: per-component access into a graphics Input vector. glslang reads an
    // input vecN one scalar at a time (`OpAccessChain %in %const_k` + `OpLoad`). The
    // Input variable is already scalarized into per-component params (synthInputAttribs),
    // so resolve the chain directly to the addressed component value. The following
    // OpLoad yields it. The index is a constant component.
    if (base < module.var_kind.len and module.var_kind[base] == .input and indices.len == 1) {
        const comp: usize = @intCast(try constValOf(module, indices[0]));
        if (module.vec_of[base].len > 0) {
            if (comp >= module.vec_of[base].len) return error.MalformedModule;
            module.input_comp_of[result] = module.vec_of[base].comps[comp];
        } else {
            // A scalar input (component 0 of a 1-wide value).
            module.input_comp_of[result] = try valueOf(module, base);
        }
        return;
    }

    // Pattern B: access into the gl_PerVertex Output interface block. glslang writes
    // gl_Position via `OpAccessChain %gl_PerVertex %member_of_Position` + `OpStore`. The
    // block is an Output struct with a member decorated BuiltIn Position. Addressing that
    // member records the position output so the OpStore routes to ATTR_POSITION. (A store
    // to a PointSize/ClipDistance member is dropped at the store - we model position only.)
    if (base < module.var_kind.len and module.var_kind[base] == .output and indices.len == 1) {
        if (module.var_pointee(base)) |struct_ty| {
            if (module.types[struct_ty]) |ti| if (ti == .@"struct") {
                const member: u32 = @intCast(try constValOf(module, indices[0]));
                if (module.members.get(memberKey(struct_ty, member))) |m| {
                    if (m.builtin == op.BuiltIn.position) {
                        module.pos_chain_var[result] = base;
                    } else {
                        // A non-position gl_PerVertex member (PointSize etc.): mark it so
                        // the OpStore is dropped rather than treated as a buffer store.
                        module.pos_chain_var[result] = std.math.maxInt(u32);
                    }
                    return;
                }
            };
        }
    }

    const base_ptr = try valueOf(module, base);
    var cur_type = module.var_pointee(base) orelse return error.Unsupported; // the pointee of the variable
    var const_off: i64 = 0; // accumulated compile-time byte offset
    var offset_val: ?Value = null; // accumulated runtime byte offset (i32)
    // If the chain steps into a matrix member, remember its layout so a matrix `OpLoad`
    // of the resulting pointer can fetch each element from the buffer.
    var mat_member: ?Member = null;

    for (indices) |idx_id| {
        switch (module.types[cur_type] orelse return error.Unsupported) {
            .@"struct" => {
                // A struct member index must be a constant. Add its byte offset.
                const member: u32 = @intCast(try constValOf(module, idx_id));
                const m = module.members.get(memberKey(cur_type, member)) orelse return error.Unsupported;
                const_off += m.offset;
                if (module.types[m.type_id]) |ti| if (ti == .matrix) {
                    mat_member = m;
                };
                cur_type = m.type_id;
            },
            .array => |arr| {
                const stride: u32 = if (module.array_stride[cur_type] != 0) module.array_stride[cur_type] else @intCast(vulcanScalarSize(func, scalarType(module.types, arr.elem) orelse return error.Unsupported));
                const idx = try valueOf(module, idx_id);
                const stride_c = try func.appendInst(block, module.i32_t, .{ .iconst = stride });
                const term = try func.appendInst(block, module.i32_t, .{ .arith = .{ .op = .mul, .lhs = idx, .rhs = stride_c } });
                offset_val = if (offset_val) |o| try func.appendInst(block, module.i32_t, .{ .arith = .{ .op = .add, .lhs = o, .rhs = term } }) else term;
                cur_type = arr.elem;
            },
            else => return error.Unsupported, // indexing a scalar
        }
    }

    // Combine the constant and runtime offsets, then add to the base pointer.
    if (const_off != 0) {
        const c = try func.appendInst(block, module.i32_t, .{ .iconst = const_off });
        offset_val = if (offset_val) |o| try func.appendInst(block, module.i32_t, .{ .arith = .{ .op = .add, .lhs = o, .rhs = c } }) else c;
    }
    const result_ptr = if (offset_val) |o|
        try func.appendInst(block, module.ptr_t, .{ .arith = .{ .op = .add, .lhs = base_ptr, .rhs = o } })
    else
        base_ptr;
    module.value_of[result] = result_ptr;

    // The chain addresses a whole matrix member: record its in-memory layout so the
    // following `OpLoad %matrix` reads each element from the buffer (a matrix is not a
    // scalar, so it cannot be loaded as one Value).
    if (mat_member) |m| if (module.types[cur_type]) |ti| if (ti.asMatrix()) |mt| {
        const col_vi = vectorInfo(module.types, mt.col_vec) orelse return error.Unsupported;
        const elem_ty = scalarType(module.types, col_vi.elem) orelse return error.Unsupported;
        // std140/430 column-major: stride between columns. Default = 16 if undecorated.
        const stride = if (m.matrix_stride != 0) m.matrix_stride else @as(u32, col_vi.len) * 4;
        module.mat_ptr[result] = .{
            .base = result_ptr,
            .cols = mt.cols,
            .rows = col_vi.len,
            .stride = stride,
            .row_major = m.row_major,
            .elem = elem_ty,
        };
        return;
    };

    // The chain addresses a whole vector member (e.g. `u.pos[gl_VertexIndex]`, a vec4
    // element of a UBO array): record its in-memory layout so the following `OpLoad
    // %vecN` fetches each component from the buffer into a scalarized Vec (a vector is
    // not a scalar, so it cannot be loaded as one Value).
    if (module.types[cur_type]) |ti| if (ti.asVector()) |vt| {
        const elem_ty = scalarType(module.types, vt.elem) orelse return error.Unsupported;
        module.vec_ptr[result] = .{ .base = result_ptr, .len = vt.len, .elem = elem_ty };
    };
}

/// Load a matrix stored in a buffer into a scalarized `Mat`. Each element (col j, row i)
/// lives at byte offset `j*stride + i*elemSize` (column-major) or `i*stride + j*elemSize`
/// (row-major) from the matrix base pointer. The Mat always holds elements in
/// column-major order (`comps[j*rows + i]`), matching how `OpMatrixTimesVector` reads it.
fn loadMatrix(func: *Function, module: *Module, block: Block, result: u32, mp: MatPtr) Error!void {
    const elem_size: u32 = @intCast(vulcanScalarSize(func, mp.elem));
    var out: Mat = .{ .cols = mp.cols, .rows = mp.rows };
    var j: u8 = 0;
    while (j < mp.cols) : (j += 1) {
        var i: u8 = 0;
        while (i < mp.rows) : (i += 1) {
            const byte_off: u32 = if (mp.row_major)
                @as(u32, i) * mp.stride + @as(u32, j) * elem_size
            else
                @as(u32, j) * mp.stride + @as(u32, i) * elem_size;
            const eptr = if (byte_off != 0) blk: {
                const off_c = try func.appendInst(block, module.i32_t, .{ .iconst = @intCast(byte_off) });
                break :blk try func.appendInst(block, module.ptr_t, .{ .arith = .{ .op = .add, .lhs = mp.base, .rhs = off_c } });
            } else mp.base;
            out.comps[@as(usize, j) * mp.rows + i] = try func.appendInst(block, mp.elem, .{ .load = .{ .ptr = eptr } });
        }
    }
    module.mat_of[result] = out;
}

/// Load a vector stored in a buffer into a scalarized `Vec`. Each component i lives at
/// byte offset `i*elemSize` from the vector base pointer (tightly packed, std140/430
/// vec component layout). The Vec holds the components in order, so a following
/// OpCompositeExtract / store reads them exactly like any other vector value.
fn loadVector(func: *Function, module: *Module, block: Block, result: u32, vp: VecPtr) Error!void {
    const elem_size: u32 = @intCast(vulcanScalarSize(func, vp.elem));
    var out: Vec = .{ .len = vp.len };
    var i: u8 = 0;
    while (i < vp.len) : (i += 1) {
        const byte_off: u32 = @as(u32, i) * elem_size;
        const eptr = if (byte_off != 0) blk: {
            const off_c = try func.appendInst(block, module.i32_t, .{ .iconst = @intCast(byte_off) });
            break :blk try func.appendInst(block, module.ptr_t, .{ .arith = .{ .op = .add, .lhs = vp.base, .rhs = off_c } });
        } else vp.base;
        out.comps[i] = try func.appendInst(block, vp.elem, .{ .load = .{ .ptr = eptr } });
    }
    module.vec_of[result] = out;
}

/// The fixed host-sampler ABI a CPU/native backend calls for an image sample:
///   `fn(desc_ptr: ptr, u: f32, v: f32, out: ptr) void`
/// `desc_ptr` is the bound combined-image-sampler descriptor (a target-defined struct:
/// pixels + width/height/pitch/format + filter + address modes), `out` receives the 4
/// sampled RGBA floats. AAPCS places `desc_ptr`/`out` in GPRs and `u`/`v` in FPRs, so
/// the gpr/fpr files do not collide. A GPU backend ignores `sampler_fn` and the call,
/// emitting a TEX from the descriptor + coord instead.
///
/// Lower `OpImageSampleImplicitLod` / `OpImageSampleExplicitLod`:
///   [type, result, sampledImage, coord, (operands...)]
/// The coord is a vec2 (scalarized). The result vec4 is built by calling the host
/// sampler through the synthesized `sampler_fn` pointer param into a stack slot, then
/// reloading the 4 floats. An explicit-LOD sample (textureLod) threads its LOD scalar as
/// the 4th call arg; implicit sampling passes 0.0 (base level). Gradient operands are not
/// yet honored (treated as implicit / lod 0).
fn lowerImageSample(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const bound = module.vec_of.len; // == the id bound
    const result = inst.operands[1];
    const sampled = try operandAt(inst.operands, 2);
    const coord = try idOperandAt(inst.operands, 3, bound); // indexes vec_of / value_of below

    const desc_ptr = (if (sampled < module.sampler_ptr_of.len) module.sampler_ptr_of[sampled] else null) orelse return error.MalformedModule;

    // The coordinate is a scalarized vec2 (u, v) for a 2D sample, or a vec3 for a cube (direction)
    // or 3D (volume) sample. Distinguish 3D from cube by the tracked SPIR-V image Dim (3D=2,
    // Cube=3); both take a vec3 but need different host tags (the GPU bakes a different TEX
    // dim + TIC). Dim 0 = untracked pre-built SPIR-V -> fall back to the coord length (cube).
    const cv = module.vec_of[coord];
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const dim: u8 = if (sampled < module.sampler_dim_of.len) module.sampler_dim_of[sampled] else 0;
    const arrayed: bool = sampled < module.sampler_arrayed_of.len and module.sampler_arrayed_of[sampled] != 0;
    // A 2D-Arrayed image has Dim=2D (=1) AND Arrayed=1 - the vec3 coord is (u, v, layer). Detect it
    // FIRST so it is not misread as a plain 2D sampler (same Dim). 3D=Dim 2, Cube=Dim 3.
    const is_2darray = dim == 1 and arrayed;
    const is_3d = dim == 2;
    const is_cube = dim == 3 or (dim == 0 and cv.len >= 3);
    const is_vec3 = is_2darray or is_3d or is_cube;
    const u: Value = if (cv.len >= 1) cv.comps[0] else (module.value_of[coord] orelse return error.MalformedModule);
    const v: Value = if (cv.len >= 2) cv.comps[1] else try func.appendInst(block, f32_t, .{ .fconst = 0 });

    // The level-of-detail passed to the host sampler. OpImageSampleExplicitLod with the Lod image
    // operand (bit 0x2) carries an explicit LOD scalar (textureLod); everything else is implicit
    // sampling (texture()). For an implicit 2D sample the CPU sampler needs the rasterizer's
    // derivative LOD (automatic mipmapping), so pass the IMPLICIT-LOD SENTINEL (a huge negative no
    // real LOD uses) - the software sampler reads `desc.implicit_lod` instead. A vec3 implicit sample
    // keeps 0.0 (the nvidia cube path uses this as an EXPLICIT LOD in its TEX.LL, and the raster
    // computes no implicit LOD for cube/3D). The GPU 2D backends ignore the 5th arg (Auto-LOD TEX).
    const implicit_lod_sentinel: f64 = -1.0e30;
    const lod: Value = blk: {
        if (inst.opcode == op.ImageSampleExplicitLod and inst.operands.len >= 6) {
            const image_operands = inst.operands[4];
            if (image_operands & 0x2 != 0) { // ImageOperands.Lod
                if (module.value_of[try checkId(inst.operands[5], bound)]) |lv| break :blk lv;
            }
        }
        break :blk try func.appendInst(block, f32_t, .{ .fconst = if (is_vec3) 0 else implicit_lod_sentinel });
    };

    // The host-sampler function pointer entry param (appended once per KIND, lazily). It is a
    // pointer param the dispatch supplies, tagged so a GPU backend picks the TEX dim. 2D, cube,
    // and 3D use SEPARATE tags (sampler_fn / sampler_cube_fn / sampler_3d_fn); the 2D call stays
    // byte-identical. The CPU binds the same host sampler for cube + 3D (it dispatches on the desc).
    const sampler_fn = if (is_2darray) blk: {
        if (module.sampler_2darray_fn) |s| break :blk s;
        const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
        try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_2darray_fn", .value = .flag } });
        module.sampler_2darray_fn = p;
        break :blk p;
    } else if (is_3d) blk: {
        if (module.sampler_3d_fn) |s| break :blk s;
        const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
        try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_3d_fn", .value = .flag } });
        module.sampler_3d_fn = p;
        break :blk p;
    } else if (is_cube) blk: {
        if (module.sampler_cube_fn) |s| break :blk s;
        const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
        try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_cube_fn", .value = .flag } });
        module.sampler_cube_fn = p;
        break :blk p;
    } else if (module.sampler_fn) |s| s else blk: {
        const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
        try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_fn", .value = .flag } });
        module.sampler_fn = p;
        break :blk p;
    };

    // A 16-byte stack slot (vec4 f32) for the sampler to write the RGBA result into.
    const out_ptr = try func.appendInst(block, module.ptr_t, .{ .alloca = .{ .elem = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 128 } }) } });

    // call_indirect sampler_fn(desc_ptr, u, v, [w,] lod, out_ptr) -> void. A cube sample
    // threads the 3rd direction component (w) as an extra arg before the lod.
    const call_args = if (is_vec3) blk: {
        const w: Value = cv.comps[2];
        break :blk try func.internValues(&.{ desc_ptr, u, v, w, lod, out_ptr });
    } else try func.internValues(&.{ desc_ptr, u, v, lod, out_ptr });
    _ = try func.appendStmtRaw(block, .{ .call_indirect = .{
        .target = sampler_fn,
        .args = call_args,
    } });

    // Reload the 4 sampled floats from the stack slot into the result Vec.
    var outv: Vec = .{ .len = 4 };
    var c: u8 = 0;
    while (c < 4) : (c += 1) {
        const eptr = if (c == 0) out_ptr else blk: {
            const off = try func.appendInst(block, module.i32_t, .{ .iconst = @as(i64, c) * 4 });
            break :blk try func.appendInst(block, module.ptr_t, .{ .arith = .{ .op = .add, .lhs = out_ptr, .rhs = off } });
        };
        outv.comps[c] = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = eptr } });
    }
    module.vec_of[result] = outv;
}

/// Lower `OpImageSampleDrefImplicitLod` / `OpImageSampleDrefExplicitLod`:
///   [type, result, sampledImage, coord, Dref, (operands...)]
/// sampler2DShadow depth compare: `coord` is a vec2 (u, v) and `Dref` is the compare reference. Calls
/// the host `f32 sampler_shadow_fn(desc, u, v, lod, dref)`; the SCALAR result IS the compare fraction
/// (no out-pointer, unlike the vec4 samplers). Implicit sampling passes the implicit-LOD sentinel so
/// the CPU sampler uses the rasterizer derivative LOD (shadow maps normally sit at the base level).
fn lowerImageSampleShadow(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const bound = module.vec_of.len; // == the id bound
    const result = inst.operands[1];
    const sampled = try operandAt(inst.operands, 2);
    const coord = try idOperandAt(inst.operands, 3, bound); // indexes vec_of / value_of below
    const dref_id = try idOperandAt(inst.operands, 4, bound);

    const desc_ptr = (if (sampled < module.sampler_ptr_of.len) module.sampler_ptr_of[sampled] else null) orelse return error.MalformedModule;
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const cv = module.vec_of[coord];
    const dref: Value = module.value_of[dref_id] orelse return error.MalformedModule;
    const implicit_lod_sentinel: f64 = -1.0e30;

    // A Cube-dim image (Dim=3, like lowerImageSample) is a samplerCubeShadow: the coord is a vec3
    // DIRECTION (x, y, z) and the call is `f32 sampler_cube_shadow_fn(desc, x, y, z, lod, dref)`. The
    // 2D case (sampler2DShadow) stays a vec2 (u, v) -> sampler_shadow_fn. Dim 0 = untracked pre-built
    // SPIR-V -> fall back to the 2D path.
    const dim: u8 = if (sampled < module.sampler_dim_of.len) module.sampler_dim_of[sampled] else 0;
    const arrayed: bool = sampled < module.sampler_arrayed_of.len and module.sampler_arrayed_of[sampled] != 0;
    const is_cube = dim == 3;
    // A 2D-Arrayed image (Dim=2D, Arrayed) is a sampler2DArrayShadow: the coord is a vec3 (u, v, layer)
    // and the call is `f32 sampler_2darray_shadow_fn(desc, u, v, layer, lod, dref)`. Detected BEFORE the
    // plain-2D shadow branch (which reads only u, v). Uses an explicit 0.0 LOD like the cube branch.
    const is_2darray = dim == 1 and arrayed;

    if (is_2darray) {
        const u: Value = if (cv.len >= 1) cv.comps[0] else (module.value_of[coord] orelse return error.MalformedModule);
        const v: Value = if (cv.len >= 2) cv.comps[1] else try func.appendInst(block, f32_t, .{ .fconst = 0 });
        const layer: Value = if (cv.len >= 3) cv.comps[2] else try func.appendInst(block, f32_t, .{ .fconst = 0 });
        const lod: Value = try func.appendInst(block, f32_t, .{ .fconst = 0 });
        const array_shadow_fn = if (module.sampler_2darray_shadow_fn) |s| s else blk: {
            const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
            try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_2darray_shadow_fn", .value = .flag } });
            module.sampler_2darray_shadow_fn = p;
            break :blk p;
        };
        const r = try func.appendInst(block, f32_t, .{ .call_indirect = .{
            .target = array_shadow_fn,
            .args = try func.internValues(&.{ desc_ptr, u, v, layer, lod, dref }),
        } });
        module.value_of[result] = r;
        return;
    }

    if (is_cube) {
        const x: Value = if (cv.len >= 1) cv.comps[0] else (module.value_of[coord] orelse return error.MalformedModule);
        const y: Value = if (cv.len >= 2) cv.comps[1] else try func.appendInst(block, f32_t, .{ .fconst = 0 });
        const z: Value = if (cv.len >= 3) cv.comps[2] else try func.appendInst(block, f32_t, .{ .fconst = 0 });
        // A cube sample uses an explicit 0.0 LOD (like the vec3 non-shadow cube path).
        const lod: Value = try func.appendInst(block, f32_t, .{ .fconst = 0 });
        const cube_shadow_fn = if (module.sampler_cube_shadow_fn) |s| s else blk: {
            const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
            try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_cube_shadow_fn", .value = .flag } });
            module.sampler_cube_shadow_fn = p;
            break :blk p;
        };
        const r = try func.appendInst(block, f32_t, .{ .call_indirect = .{
            .target = cube_shadow_fn,
            .args = try func.internValues(&.{ desc_ptr, x, y, z, lod, dref }),
        } });
        module.value_of[result] = r;
        return;
    }

    const u: Value = if (cv.len >= 1) cv.comps[0] else (module.value_of[coord] orelse return error.MalformedModule);
    const v: Value = if (cv.len >= 2) cv.comps[1] else try func.appendInst(block, f32_t, .{ .fconst = 0 });
    const lod: Value = try func.appendInst(block, f32_t, .{ .fconst = implicit_lod_sentinel });

    const shadow_fn = if (module.sampler_shadow_fn) |s| s else blk: {
        const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
        try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_shadow_fn", .value = .flag } });
        module.sampler_shadow_fn = p;
        break :blk p;
    };

    // f32 sampler_shadow_fn(desc, u, v, lod, dref) -> the depth-compare fraction (a scalar result).
    const r = try func.appendInst(block, f32_t, .{ .call_indirect = .{
        .target = shadow_fn,
        .args = try func.internValues(&.{ desc_ptr, u, v, lod, dref }),
    } });
    module.value_of[result] = r;
}

/// Lower `OpImageFetch resultType result image coord [Lod lodValue]` (GLSL texelFetch): call the
/// synthesized host `sampler_fetch_fn(desc, x:i32, y:i32, lod:i32, out)`, which writes the EXACT
/// texel at integer coords (no filtering) into a vec4 stack slot. A GPU backend ignores the call and
/// emits a TLD (OpTld). The coord is an ivec2 (its scalarized i32 components); the LOD is the Lod
/// image operand (0 when absent). Mirrors lowerImageGather but with integer coords.
fn lowerImageFetch(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const bound = module.vec_of.len; // == the id bound
    const result = inst.operands[1];
    const image = try operandAt(inst.operands, 2);
    const coord = try idOperandAt(inst.operands, 3, bound); // indexes vec_of / value_of below
    const desc_ptr = (if (image < module.sampler_ptr_of.len) module.sampler_ptr_of[image] else null) orelse return error.MalformedModule;
    const cv = module.vec_of[coord];
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const x: Value = if (cv.len >= 1) cv.comps[0] else (module.value_of[coord] orelse return error.MalformedModule);
    const y: Value = if (cv.len >= 2) cv.comps[1] else try func.appendInst(block, i32_t, .{ .iconst = 0 });
    // The Lod image operand: operands = [type, result, image, coord, imageOperands, lodValue].
    const lod: Value = blk: {
        if (inst.operands.len >= 6 and (inst.operands[4] & 0x2) != 0) {
            if (module.value_of[try checkId(inst.operands[5], bound)]) |lv| break :blk lv;
        }
        break :blk try func.appendInst(block, i32_t, .{ .iconst = 0 });
    };

    // A 3-component coord is a 2D-ARRAY or 3D texelFetch (x, y, layer/z). Pick the host tag by the
    // image's Arrayed flag (array vs 3D) so the GPU emits the matching TLD dim; a 2-component coord is
    // the plain 2D fetch. The CPU binds the same host fn for array/3D (fetch the exact texel of slice z).
    const out_ptr = try func.appendInst(block, module.ptr_t, .{ .alloca = .{ .elem = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 128 } }) } });
    if (cv.len >= 3) {
        const z: Value = cv.comps[2];
        const arrayed = image < module.sampler_arrayed_of.len and module.sampler_arrayed_of[image] != 0;
        const key = if (arrayed) "sampler_fetch_array_fn" else "sampler_fetch_3d_fn";
        const existing = if (arrayed) module.sampler_fetch_array_fn else module.sampler_fetch_3d_fn;
        const fetch3_fn = if (existing) |s| s else blk: {
            const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
            try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = key, .value = .flag } });
            if (arrayed) {
                module.sampler_fetch_array_fn = p;
            } else {
                module.sampler_fetch_3d_fn = p;
            }
            break :blk p;
        };
        _ = try func.appendStmtRaw(block, .{ .call_indirect = .{
            .target = fetch3_fn,
            .args = try func.internValues(&.{ desc_ptr, x, y, z, lod, out_ptr }),
        } });
        var outv3: Vec = .{ .len = 4 };
        var c3: u8 = 0;
        while (c3 < 4) : (c3 += 1) {
            const eptr = if (c3 == 0) out_ptr else blk: {
                const off = try func.appendInst(block, module.i32_t, .{ .iconst = @as(i64, c3) * 4 });
                break :blk try func.appendInst(block, module.ptr_t, .{ .arith = .{ .op = .add, .lhs = out_ptr, .rhs = off } });
            };
            outv3.comps[c3] = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = eptr } });
        }
        module.vec_of[result] = outv3;
        return;
    }

    const fetch_fn = if (module.sampler_fetch_fn) |s| s else blk: {
        const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
        try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_fetch_fn", .value = .flag } });
        module.sampler_fetch_fn = p;
        break :blk p;
    };

    _ = try func.appendStmtRaw(block, .{ .call_indirect = .{
        .target = fetch_fn,
        .args = try func.internValues(&.{ desc_ptr, x, y, lod, out_ptr }),
    } });
    var outv: Vec = .{ .len = 4 };
    var c: u8 = 0;
    while (c < 4) : (c += 1) {
        const eptr = if (c == 0) out_ptr else blk: {
            const off = try func.appendInst(block, module.i32_t, .{ .iconst = @as(i64, c) * 4 });
            break :blk try func.appendInst(block, module.ptr_t, .{ .arith = .{ .op = .add, .lhs = out_ptr, .rhs = off } });
        };
        outv.comps[c] = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = eptr } });
    }
    module.vec_of[result] = outv;
}

/// Lower `OpImageGather resultType result sampledImage coord component` (GLSL textureGather): call
/// the synthesized host `sampler_gather_fn(desc, u, v, comp, out)`, which writes the `comp` channel
/// of the 4 bilinear-footprint texels into a vec4 stack slot (GL gather order). The component is a
/// constant 0..3 (tracked in const_val), passed to the host as an f32. A GPU backend ignores the
/// call and emits a TG4. Structurally mirrors lowerImageSample (2D coord).
fn lowerImageGather(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const bound = module.vec_of.len; // == the id bound
    const result = inst.operands[1];
    const sampled = try operandAt(inst.operands, 2);
    const coord = try idOperandAt(inst.operands, 3, bound); // indexes vec_of / value_of below
    const comp_id = try operandAt(inst.operands, 4);
    const desc_ptr = (if (sampled < module.sampler_ptr_of.len) module.sampler_ptr_of[sampled] else null) orelse return error.MalformedModule;
    const cv = module.vec_of[coord];
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const u: Value = if (cv.len >= 1) cv.comps[0] else (module.value_of[coord] orelse return error.MalformedModule);
    const v: Value = if (cv.len >= 2) cv.comps[1] else try func.appendInst(block, f32_t, .{ .fconst = 0 });
    const comp_int: i64 = if (comp_id < module.const_val.len) module.const_val[comp_id] else 0;
    const comp: Value = try func.appendInst(block, f32_t, .{ .fconst = @floatFromInt(comp_int) });

    const gather_fn = if (module.sampler_gather_fn) |s| s else blk: {
        const p = try func.appendBlockParam(@enumFromInt(0), module.ptr_t);
        try func.addAttr(.{ .value = p }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_gather_fn", .value = .flag } });
        module.sampler_gather_fn = p;
        break :blk p;
    };

    const out_ptr = try func.appendInst(block, module.ptr_t, .{ .alloca = .{ .elem = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 128 } }) } });
    _ = try func.appendStmtRaw(block, .{ .call_indirect = .{
        .target = gather_fn,
        .args = try func.internValues(&.{ desc_ptr, u, v, comp, out_ptr }),
    } });
    var outv: Vec = .{ .len = 4 };
    var c: u8 = 0;
    while (c < 4) : (c += 1) {
        const eptr = if (c == 0) out_ptr else blk: {
            const off = try func.appendInst(block, module.i32_t, .{ .iconst = @as(i64, c) * 4 });
            break :blk try func.appendInst(block, module.ptr_t, .{ .arith = .{ .op = .add, .lhs = out_ptr, .rhs = off } });
        };
        outv.comps[c] = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = eptr } });
    }
    module.vec_of[result] = outv;
}

/// Lower `OpMatrixTimesVector`: result row i = sum over columns j of M[col j][row i] * v[j]
/// (the matrix is column-major). Produces a scalarized Vec via FMul/FAdd, so no backend
/// needs matrix support. The matrix operand is a scalarized `Mat`, the vector a `Vec`.
fn lowerMatrixTimesVector(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const bound = module.mat_of.len; // == the id bound
    const mat = module.mat_of[try idOperandAt(inst.operands, 2, bound)];
    const vec = module.vec_of[try idOperandAt(inst.operands, 3, bound)];
    if (mat.cols == 0 or vec.len == 0) return error.MalformedModule;
    if (mat.cols != vec.len) return error.MalformedModule;
    const elem = scalarType(module.types, inst.operands[0]) orelse vectorElemType(module, inst.operands[0]) orelse return error.Unsupported;
    var out: Vec = .{ .len = mat.rows };
    var i: u8 = 0;
    while (i < mat.rows) : (i += 1) {
        // acc = M[0][i]*v[0] + M[1][i]*v[1] + ...  (column-major: M[col j][row i] at j*rows+i).
        var acc = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = mat.comps[0 * mat.rows + i], .rhs = vec.comps[0] } });
        var j: u8 = 1;
        while (j < mat.cols) : (j += 1) {
            const p = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = mat.comps[@as(usize, j) * mat.rows + i], .rhs = vec.comps[j] } });
            acc = try func.appendInst(block, elem, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
        }
        out.comps[i] = acc;
    }
    module.vec_of[inst.operands[1]] = out;
}

/// Lower `OpVectorTimesMatrix`: result column j = dot(v, M[col j]) = sum_i v[i] * M[col j][row i].
/// The result length is the matrix's column count.
fn lowerVectorTimesMatrix(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const bound = module.vec_of.len; // == the id bound
    const vec = module.vec_of[try idOperandAt(inst.operands, 2, bound)];
    const mat = module.mat_of[try idOperandAt(inst.operands, 3, bound)];
    if (mat.cols == 0 or vec.len == 0) return error.MalformedModule;
    if (mat.rows != vec.len) return error.MalformedModule;
    const elem = scalarType(module.types, inst.operands[0]) orelse vectorElemType(module, inst.operands[0]) orelse return error.Unsupported;
    var out: Vec = .{ .len = mat.cols };
    var j: u8 = 0;
    while (j < mat.cols) : (j += 1) {
        var acc = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = vec.comps[0], .rhs = mat.comps[@as(usize, j) * mat.rows + 0] } });
        var i: u8 = 1;
        while (i < mat.rows) : (i += 1) {
            const p = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = vec.comps[i], .rhs = mat.comps[@as(usize, j) * mat.rows + i] } });
            acc = try func.appendInst(block, elem, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
        }
        out.comps[j] = acc;
    }
    module.vec_of[inst.operands[1]] = out;
}

/// Lower `OpMatrixTimesScalar`: scale every element of the matrix by the scalar, producing
/// a new scalarized `Mat`.
fn lowerMatrixTimesScalar(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const bound = module.mat_of.len; // == the id bound
    const mat = module.mat_of[try idOperandAt(inst.operands, 2, bound)];
    const s = try valueOf(module, try operandAt(inst.operands, 3));
    if (mat.cols == 0) return error.MalformedModule;
    const elem = matrixElemType(module, inst.operands[0]) orelse return error.Unsupported;
    var out: Mat = .{ .cols = mat.cols, .rows = mat.rows };
    const n: usize = @as(usize, mat.cols) * mat.rows;
    for (0..n) |k| out.comps[k] = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = mat.comps[k], .rhs = s } });
    module.mat_of[inst.operands[1]] = out;
}

/// Lower `OpMatrixTimesMatrix`: C = A * B, column-major. Column j of C is A times column j
/// of B, i.e. C[col j][row i] = sum_k A[col k][row i] * B[col j][row k].
fn lowerMatrixTimesMatrix(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const bound = module.mat_of.len; // == the id bound
    const a = module.mat_of[try idOperandAt(inst.operands, 2, bound)];
    const b = module.mat_of[try idOperandAt(inst.operands, 3, bound)];
    if (a.cols == 0 or b.cols == 0) return error.MalformedModule;
    if (a.cols != b.rows) return error.MalformedModule;
    const elem = matrixElemType(module, inst.operands[0]) orelse return error.Unsupported;
    var out: Mat = .{ .cols = b.cols, .rows = a.rows };
    var j: u8 = 0;
    while (j < b.cols) : (j += 1) {
        var i: u8 = 0;
        while (i < a.rows) : (i += 1) {
            var acc = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = a.comps[0 * a.rows + i], .rhs = b.comps[@as(usize, j) * b.rows + 0] } });
            var k: u8 = 1;
            while (k < a.cols) : (k += 1) {
                const p = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = a.comps[@as(usize, k) * a.rows + i], .rhs = b.comps[@as(usize, j) * b.rows + k] } });
                acc = try func.appendInst(block, elem, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
            }
            out.comps[@as(usize, j) * out.rows + i] = acc;
        }
    }
    module.mat_of[inst.operands[1]] = out;
}

/// The scalar element type of a matrix result id (the element of its column vector).
fn matrixElemType(module: *const Module, type_id: u32) ?Type {
    const mt = (if (type_id < module.types.len) module.types[type_id] else null) orelse return null;
    const m = mt.asMatrix() orelse return null;
    const cv = vectorInfo(module.types, m.col_vec) orelse return null;
    return scalarType(module.types, cv.elem);
}

/// The scalar element type of a vector result id (the component type).
fn vectorElemType(module: *const Module, type_id: u32) ?Type {
    const vi = vectorInfo(module.types, type_id) orelse return null;
    return scalarType(module.types, vi.elem);
}

fn lowerTerminator(allocator: std.mem.Allocator, func: *Function, module: *const Module, value_of: []const ?Value, blocks: []const BlockInfo, bi: usize, insts: []const binary.Instruction, inst: binary.Instruction) Error!void {
    switch (inst.opcode) {
        op.Return, op.Unreachable => func.setTerminator(blocks[bi].block, .{ .ret = null }),
        op.Kill => {
            // discard: call the synthesized discard_fn (a CPU backend signals the kill;
            // a GPU/TGSI backend emits a KILL for the call), then end the block.
            const df = module.discard_fn orelse return error.MalformedModule;
            const void_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 0 } });
            _ = try func.appendCallIndirect(blocks[bi].block, void_t, df, &.{});
            func.setTerminator(blocks[bi].block, .{ .ret = null });
        },
        op.ReturnValue => func.setTerminator(blocks[bi].block, .{ .ret = value_of[try idOperandAt(inst.operands, 0, value_of.len)] orelse return error.MalformedModule }),
        op.Branch => {
            const target = blockIndex(blocks, try operandAt(inst.operands, 0));
            const args = try phiArgs(allocator, module, value_of, blocks, target, blocks[bi].label, insts);
            defer allocator.free(args);
            try func.setJump(blocks[bi].block, blocks[target].block, args);
        },
        op.BranchConditional => {
            const cond = value_of[try idOperandAt(inst.operands, 0, value_of.len)] orelse return error.MalformedModule;
            const t = blockIndex(blocks, try operandAt(inst.operands, 1));
            const f = blockIndex(blocks, try operandAt(inst.operands, 2));
            const t_args = try phiArgs(allocator, module, value_of, blocks, t, blocks[bi].label, insts);
            defer allocator.free(t_args);
            const f_args = try phiArgs(allocator, module, value_of, blocks, f, blocks[bi].label, insts);
            defer allocator.free(f_args);
            try func.appendIf(blocks[bi].block, cond, .{ .target = blocks[t].block, .args = t_args }, .{ .target = blocks[f].block, .args = f_args });
        },
        op.Switch => try lowerSwitch(allocator, func, module, value_of, blocks, bi, insts, inst),
        else => {},
    }
}

/// Lower `OpSwitch %selector %default <lit0> %case0 <lit1> %case1 ...` to a chain of
/// equality tests. The IR has no native switch terminator, but `if` is a non-terminating
/// conditional the backend already handles, so each case becomes one `if (selector == litK)
/// then caseK else next`, the chain ending at `default`. The comparison/branch for case 0
/// goes in the switch block itself. Each subsequent comparison lives in a synthesized chain
/// block. The case/default edges carry their phi arguments computed against the ORIGINAL
/// switch block's label (SPIR-V records the switch block as the phi predecessor), so a phi at
/// a case/merge resolves correctly. A `default`-or-case literal may be 32- or 64-bit wide
/// (SPIR-V emits the literal as one or two words sized to the selector's width).
fn lowerSwitch(allocator: std.mem.Allocator, func: *Function, module: *const Module, value_of: []const ?Value, blocks: []const BlockInfo, bi: usize, insts: []const binary.Instruction, inst: binary.Instruction) Error!void {
    const selector = value_of[try idOperandAt(inst.operands, 0, value_of.len)] orelse return error.MalformedModule;
    const sel_ty = func.valueType(selector);
    const bool_t = try func.types.intern(.bool);
    const default_label = try operandAt(inst.operands, 1);

    // The selector literal width (in 32-bit words) follows the selector's integer width.
    const lit_words: usize = switch (func.types.type_kind(sel_ty)) {
        .int => |i| if (i.bits > 32) 2 else 1,
        else => 1,
    };
    const stride = lit_words + 1; // each (literal, label) pair

    // Count the cases.
    const rest = try operandsFrom(inst.operands, 2);
    if (rest.len % stride != 0) return error.MalformedModule;
    const ncases = rest.len / stride;

    const switch_label = blocks[bi].label;
    var cur_block = blocks[bi].block;

    var k: usize = 0;
    while (k < ncases) : (k += 1) {
        const lit_lo = rest[k * stride];
        const lit_val: i64 = if (lit_words == 2)
            @bitCast(@as(u64, lit_lo) | (@as(u64, rest[k * stride + 1]) << 32))
        else switch (func.types.type_kind(sel_ty)) {
            // Sign-extend a 32-bit signed case literal so `== ` matches the selector value.
            .int => |i| if (i.signedness == .signed) @as(i64, @as(i32, @bitCast(lit_lo))) else @as(i64, lit_lo),
            else => @as(i64, lit_lo),
        };
        const case_label = rest[k * stride + lit_words];
        const ci = blockIndex(blocks, case_label);

        // cmp = (selector == litK), placed in the current chain block.
        const litc = try func.appendInst(cur_block, sel_ty, .{ .iconst = lit_val });
        const cmp = try func.appendInst(cur_block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = selector, .rhs = litc } });

        const case_args = try phiArgs(allocator, module, value_of, blocks, ci, switch_label, insts);
        defer allocator.free(case_args);

        if (k + 1 < ncases) {
            // Not the last case: else continues to a fresh chain block.
            const next_block = try func.appendBlock();
            try func.appendIf(cur_block, cmp, .{ .target = blocks[ci].block, .args = case_args }, .{ .target = next_block, .args = &.{} });
            cur_block = next_block;
        } else {
            // Last case: else falls through to default.
            const di = blockIndex(blocks, default_label);
            const def_args = try phiArgs(allocator, module, value_of, blocks, di, switch_label, insts);
            defer allocator.free(def_args);
            try func.appendIf(cur_block, cmp, .{ .target = blocks[ci].block, .args = case_args }, .{ .target = blocks[di].block, .args = def_args });
        }
    }

    // A switch with no cases (only default) is just an unconditional jump to default.
    if (ncases == 0) {
        const di = blockIndex(blocks, default_label);
        const def_args = try phiArgs(allocator, module, value_of, blocks, di, switch_label, insts);
        defer allocator.free(def_args);
        try func.setJump(cur_block, blocks[di].block, def_args);
    }
}

fn phiArgs(allocator: std.mem.Allocator, module: *const Module, value_of: []const ?Value, blocks: []const BlockInfo, target: usize, pred_label: u32, insts: []const binary.Instruction) Error![]Value {
    const phis = blocks[target].phis.items;
    // A vector phi contributes one edge argument per component (matching its scalarized block
    // params), so the argument count is the sum of each phi's width.
    var args: std.ArrayList(Value) = .empty;
    errdefer args.deinit(allocator);
    for (phis) |phi_id| {
        const operands = phiOperandsOf(insts, phi_id) orelse return error.MalformedModule;
        // The incoming value id for this predecessor edge.
        var inc: ?u32 = null;
        var j: usize = 0;
        while (j + 1 < operands.len) : (j += 2) {
            if (operands[j + 1] == pred_label) {
                inc = operands[j];
                break;
            }
        }
        const inc_id = inc orelse return error.MalformedModule;
        // A vector phi: append each component of the incoming vector value.
        if (phi_id < module.vec_of.len and module.vec_of[phi_id].len > 0) {
            const want = module.vec_of[phi_id].len;
            if (inc_id >= module.vec_of.len) return error.MalformedModule; // untrusted incoming id
            const iv = module.vec_of[inc_id];
            if (iv.len != want) return error.MalformedModule;
            var c: u8 = 0;
            while (c < want) : (c += 1) try args.append(allocator, iv.comps[c]);
        } else {
            if (inc_id >= value_of.len) return error.MalformedModule; // untrusted incoming id
            try args.append(allocator, value_of[inc_id] orelse return error.MalformedModule);
        }
    }
    return args.toOwnedSlice(allocator);
}

fn phiOperandsOf(insts: []const binary.Instruction, phi_id: u32) ?[]const u32 {
    for (insts) |inst| {
        if (inst.opcode == op.Phi and inst.operands[1] == phi_id) return inst.operands[2..];
    }
    return null;
}

fn blockIndex(blocks: []const BlockInfo, label: u32) usize {
    for (blocks, 0..) |b, i| if (b.label == label) return i;
    return std.math.maxInt(usize);
}

fn scalarType(types: []const ?TypeInfo, type_id: u32) ?Type {
    if (type_id >= types.len) return null;
    return switch (types[type_id] orelse return null) {
        .scalar => |t| t,
        else => null,
    };
}

/// Byte size of an interned scalar Vulcan type (the array element stride).
fn vulcanScalarSize(func: *const Function, ty: Type) usize {
    return switch (func.types.type_kind(ty)) {
        .bool => 1,
        .int => |i| (@as(usize, i.bits) + 7) / 8,
        .float => |f| if (f == .f32) 4 else 8,
        .ptr => 8,
        else => 4,
    };
}

/// The result-type id of the `OpPhi` defining `result_id` (the phi's declared type).
fn phiResultType(insts: []const binary.Instruction, result_id: u32) ?u32 {
    for (insts) |inst| {
        if (inst.opcode == op.Phi and inst.operands[1] == result_id) return inst.operands[0];
    }
    return null;
}

fn binOpOf(opcode: u16) ?ir.function.BinOp {
    return switch (opcode) {
        op.IAdd, op.FAdd => .add,
        op.ISub, op.FSub => .sub,
        op.IMul, op.FMul => .mul,
        op.SDiv, op.UDiv, op.FDiv => .div,
        op.UMod, op.SRem, op.FRem, op.SMod => .rem,
        op.ShiftLeftLogical => .shl,
        op.ShiftRightLogical, op.ShiftRightArithmetic => .shr,
        op.BitwiseAnd, op.LogicalAnd => .bit_and,
        op.BitwiseOr, op.LogicalOr => .bit_or,
        op.BitwiseXor, op.LogicalNotEqual => .bit_xor,
        else => null,
    };
}

fn cmpOpOf(opcode: u16) ?ir.function.CmpOp {
    return switch (opcode) {
        op.IEqual, op.FOrdEqual => .eq,
        op.INotEqual, op.FOrdNotEqual => .ne,
        op.SLessThan, op.ULessThan, op.FOrdLessThan => .lt,
        op.SLessThanEqual, op.ULessThanEqual, op.FOrdLessThanEqual => .le,
        op.SGreaterThan, op.UGreaterThan, op.FOrdGreaterThan => .gt,
        op.SGreaterThanEqual, op.UGreaterThanEqual, op.FOrdGreaterThanEqual => .ge,
        else => null,
    };
}

const testing = std.testing;

test "lowers a straight-line scalar function (x*y + x)" {
    const allocator = testing.allocator;
    var b = try binary.Builder.init(allocator, 9);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 4 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 5 });
    try b.emit(allocator, op.Label, &.{6});
    try b.emit(allocator, op.IMul, &.{ 1, 7, 4, 5 });
    try b.emit(allocator, op.IAdd, &.{ 1, 8, 7, 4 });
    try b.emit(allocator, op.ReturnValue, &.{8});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();
    try testing.expectFmt(
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 * v1
        \\    let v3 = v2 + v0
        \\    ret v3
        \\}
    , "{f}", .{func});
}

test "lowers control flow with a phi (max of two values)" {
    const allocator = testing.allocator;
    var bld = try binary.Builder.init(allocator, 13);
    defer bld.deinit(allocator);
    try bld.emit(allocator, op.TypeBool, &.{1});
    try bld.emit(allocator, op.TypeInt, &.{ 2, 32, 1 });
    try bld.emit(allocator, op.TypeFunction, &.{ 3, 2, 2, 2 });
    try bld.emit(allocator, op.Function, &.{ 2, 4, 0, 3 });
    try bld.emit(allocator, op.FunctionParameter, &.{ 2, 5 });
    try bld.emit(allocator, op.FunctionParameter, &.{ 2, 6 });
    try bld.emit(allocator, op.Label, &.{7});
    try bld.emit(allocator, op.SGreaterThan, &.{ 1, 8, 5, 6 });
    try bld.emit(allocator, op.BranchConditional, &.{ 8, 9, 10 });
    try bld.emit(allocator, op.Label, &.{9});
    try bld.emit(allocator, op.Branch, &.{11});
    try bld.emit(allocator, op.Label, &.{10});
    try bld.emit(allocator, op.Branch, &.{11});
    try bld.emit(allocator, op.Label, &.{11});
    try bld.emit(allocator, op.Phi, &.{ 2, 12, 5, 9, 6, 10 });
    try bld.emit(allocator, op.ReturnValue, &.{12});
    try bld.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, bld.words.items);
    defer func.deinit();
    try testing.expectFmt(
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 > v1
        \\    if v2 { block1() } else { block2() }
        \\    ret void
        \\
        \\  block1():
        \\    block3(v0)
        \\
        \\  block2():
        \\    block3(v1)
        \\
        \\  block3(v3: i32):
        \\    ret v3
        \\}
    , "{f}", .{func});
}

test "lowers OpSwitch to an equality-test chain (case + default)" {
    const allocator = testing.allocator;
    // int f(int x) { switch (x) { case 0: return 100; case 1: return 200; default: return x; } }
    // ids: int=1 fnty=2 f=3 x=4 entry=5 c0=6 c1=7 c2=8 def=9 merge=10 phi=11
    //      const100=12 const200=13.
    var b = try binary.Builder.init(allocator, 14);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeInt, &.{ 1, 32, 1 });
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1, 1 });
    try b.emit(allocator, op.Constant, &.{ 1, 12, 100 });
    try b.emit(allocator, op.Constant, &.{ 1, 13, 200 });
    try b.emit(allocator, op.Function, &.{ 1, 3, 0, 2 });
    try b.emit(allocator, op.FunctionParameter, &.{ 1, 4 });
    try b.emit(allocator, op.Label, &.{5});
    try b.emit(allocator, op.SelectionMerge, &.{ 10, 0 });
    try b.emit(allocator, op.Switch, &.{ 4, 9, 0, 6, 1, 7 }); // default=9, case 0->6, case 1->7
    try b.emit(allocator, op.Label, &.{6}); // case 0
    try b.emit(allocator, op.Branch, &.{10});
    try b.emit(allocator, op.Label, &.{7}); // case 1
    try b.emit(allocator, op.Branch, &.{10});
    try b.emit(allocator, op.Label, &.{9}); // default
    try b.emit(allocator, op.Branch, &.{10});
    try b.emit(allocator, op.Label, &.{10}); // merge
    try b.emit(allocator, op.Phi, &.{ 1, 11, 12, 6, 13, 7, 4, 9 }); // 100 from c0, 200 from c1, x from default
    try b.emit(allocator, op.ReturnValue, &.{11});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    // The switch becomes a chain of `if (x == lit) caseK else next`. The merge block is a
    // phi-parametered block reached from each case. Verify it lowers (no error) and has the
    // expected block count: entry + 2 cases + default + merge + 1 synthesized chain block.
    try testing.expect(func.blockCount() >= 5);
    var buf: [2048]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{f}", .{func});
    try testing.expect(std.mem.indexOf(u8, text, "if ") != null); // the equality-test chain
}

test "lowers a compute shader: buffer store through a thread-indexed pointer" {
    const allocator = testing.allocator;
    // void main() { data[gl_GlobalInvocationID.x] = data[gid] * 2 }
    // ids: void=1 int=2 uint=3 v3uint=4 pInV3=5 pInU=6 arr=7 struct=8 pSbStruct=9
    //      pSbInt=10 voidfn=11 c0=12 c2=13 gid=14 buf=15 main=16 entry=17
    //      xptr=18 i=19 ep=20 v=21 v2=22.
    var b = try binary.Builder.init(allocator, 23);
    defer b.deinit(allocator);
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.global_invocation_id });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 1 });
    try b.emit(allocator, op.TypeInt, &.{ 3, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypeRuntimeArray, &.{ 7, 2 });
    try b.emit(allocator, op.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 11, 1 });
    try b.emit(allocator, op.Constant, &.{ 3, 12, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 13, 2 });
    try b.emit(allocator, op.Variable, &.{ 5, 14, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 9, 15, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 16, 0, 11 });
    try b.emit(allocator, op.Label, &.{17});
    try b.emit(allocator, op.AccessChain, &.{ 6, 18, 14, 12 }); // &gid.x
    try b.emit(allocator, op.Load, &.{ 3, 19, 18 }); // i = gid.x
    try b.emit(allocator, op.AccessChain, &.{ 10, 20, 15, 12, 19 }); // &data[i]
    try b.emit(allocator, op.Load, &.{ 2, 21, 20 }); // v = data[i]
    try b.emit(allocator, op.IMul, &.{ 2, 22, 21, 13 }); // v*2
    try b.emit(allocator, op.Store, &.{ 20, 22 }); // data[i] = v*2
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    // Structural checks (constant ordering is not deterministic, so no exact text):
    // two entry params (invocation id i32 + buffer ptr), a load, a store, a void return,
    // and pointer arithmetic producing the element address.
    var buf: [2048]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{f}", .{func});
    try testing.expect(std.mem.indexOf(u8, text, "block0(v0: i32, v1: ptr):") != null);
    try testing.expect(std.mem.indexOf(u8, text, "load i32") != null);
    try testing.expect(std.mem.indexOf(u8, text, "store ") != null);
    try testing.expect(std.mem.indexOf(u8, text, "ret void") != null);
    try testing.expect(std.mem.indexOf(u8, text, "v1 +") != null); // buf + offset (pointer add)
}

test "lowers a vertex shader: input attribute -> position passthrough" {
    const allocator = testing.allocator;
    // A pass-through VS: load the location-0 vec4 input, store it to gl_Position.
    // ids: void=1 f32=2 v4f=3 pIn=4 pOut=5 voidfn=6 inVar=7 outVar=8 main=9
    //      entry=10 loaded=11.
    var b = try binary.Builder.init(allocator, 12);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.vertex, 9, 0, 7, 8 });
    try b.emit(allocator, op.Decorate, &.{ 7, op.Decoration.location, 0 });
    try b.emit(allocator, op.Decorate, &.{ 8, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(allocator, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.output, 3 });
    try b.emit(allocator, op.TypeFunction, &.{ 6, 1 });
    try b.emit(allocator, op.Variable, &.{ 4, 7, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 5, 8, op.StorageClass.output });
    try b.emit(allocator, op.Function, &.{ 1, 9, 0, 6 });
    try b.emit(allocator, op.Label, &.{10});
    try b.emit(allocator, op.Load, &.{ 3, 11, 7 });
    try b.emit(allocator, op.Store, &.{ 8, 11 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    // 4 entry params (the vec4 input, scalarized), the first tagged attr=0x80.
    const params = func.blockParams(@enumFromInt(0));
    try testing.expectEqual(@as(usize, 4), params.len);
    try testing.expect(hasAttr(&func, params[0], "attr", 0x80));
    try testing.expect(hasAttr(&func, params[3], "attr", 0x8c)); // 0x80 + 3*4
    // The stage attribute is set on the function.
    try testing.expect(hasFuncStage(&func, "vertex"));
    // 4 stores, one per position component, tagged out_attr at ATTR_POSITION+k*4.
    var n_out_attr: usize = 0;
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        if (hasAttr(&func, @enumFromInt(i), "out_attr", 0x70)) n_out_attr += 1;
    }
    try testing.expectEqual(@as(usize, 1), n_out_attr); // exactly one at 0x70 (X)
}

test "lowers a fragment shader: constant color -> color output" {
    const allocator = testing.allocator;
    // A constant-red PS: store vec4(1,0,0,1) to the location-0 color output.
    // ids: void=1 f32=2 v4f=3 pOut=4 voidfn=5 c1=6 c0=7 red=8 outVar=9 main=10
    //      entry=11.
    var b = try binary.Builder.init(allocator, 12);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 9 });
    try b.emit(allocator, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 4, op.StorageClass.output, 3 });
    try b.emit(allocator, op.TypeFunction, &.{ 5, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 6, @bitCast(@as(f32, 1.0)) });
    try b.emit(allocator, op.Constant, &.{ 2, 7, 0 });
    try b.emit(allocator, op.ConstantComposite, &.{ 3, 8, 6, 7, 7, 6 }); // (1,0,0,1)
    try b.emit(allocator, op.Variable, &.{ 4, 9, op.StorageClass.output });
    try b.emit(allocator, op.Function, &.{ 1, 10, 0, 5 });
    try b.emit(allocator, op.Label, &.{11});
    try b.emit(allocator, op.Store, &.{ 9, 8 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    try testing.expect(hasFuncStage(&func, "fragment"));
    // 4 color-output stores, tagged color_out = component index 0..3.
    var seen = [_]bool{ false, false, false, false };
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        inline for (0..4) |k| if (hasAttr(&func, @enumFromInt(i), "color_out", k)) {
            seen[k] = true;
        };
    }
    try testing.expect(seen[0] and seen[1] and seen[2] and seen[3]);
}

test "lowers a push-constant block read: PushConstant storage -> a pointer entry param" {
    const allocator = testing.allocator;
    // A fragment shader reading a push constant:
    //   layout(push_constant) uniform PC { vec4 color; } pc;
    //   layout(location=0) out vec4 o; void main(){ o = pc.color; }
    // A PushConstant-storage OpVariable of a Block-decorated struct, read with
    // OpAccessChain (into member 0) + OpLoad %vec4, stored to the color output. This
    // must lower exactly like a UBO: the block becomes a base-pointer entry param the
    // shader loads the four color components from (member offset 0), NOT error.Unsupported.
    // ids: void=1 f32=2 v4f=3 int=4 i0=5 PC=6 pPCstruct=7 pcVar=8 pPCv4=9 pOutV4=10
    //      outVar=11 voidfn=12 main=13 entry=14 acColor=15 color=16.
    var b = try binary.Builder.init(allocator, 17);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.fragment, 13, 0, 11 });
    try b.emit(allocator, op.Decorate, &.{ 6, op.Decoration.block });
    try b.emit(allocator, op.MemberDecorate, &.{ 6, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(allocator, op.TypeInt, &.{ 4, 32, 1 });
    try b.emit(allocator, op.Constant, &.{ 4, 5, 0 }); // int 0 (member index)
    try b.emit(allocator, op.TypeStruct, &.{ 6, 3 }); // PC { vec4 color; }
    try b.emit(allocator, op.TypePointer, &.{ 7, op.StorageClass.push_constant, 6 });
    try b.emit(allocator, op.Variable, &.{ 7, 8, op.StorageClass.push_constant });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.push_constant, 3 }); // ptr to vec4 member
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.output, 3 });
    try b.emit(allocator, op.Variable, &.{ 10, 11, op.StorageClass.output });
    try b.emit(allocator, op.TypeFunction, &.{ 12, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 13, 0, 12 });
    try b.emit(allocator, op.Label, &.{14});
    try b.emit(allocator, op.AccessChain, &.{ 9, 15, 8, 5 }); // &pc.color
    try b.emit(allocator, op.Load, &.{ 3, 16, 15 }); // vec4 color
    try b.emit(allocator, op.Store, &.{ 11, 16 }); // o = pc.color
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    try testing.expect(hasFuncStage(&func, "fragment"));
    // The push-constant block is a base-pointer entry param (the last block param, after
    // any inputs - there are none here, so the sole entry param is the PC pointer). The
    // FS reads its four color components from that buffer pointer (4 loads) and stores
    // each to the color output (color_out 0..3).
    const params = func.blockParams(@enumFromInt(0));
    try testing.expectEqual(@as(usize, 1), params.len); // just the PC pointer param
    var buf: [4096]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{f}", .{func});
    try testing.expect(std.mem.indexOf(u8, text, "load f32") != null); // loads from the pointer
    var seen = [_]bool{ false, false, false, false };
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        inline for (0..4) |k| if (hasAttr(&func, @enumFromInt(i), "color_out", k)) {
            seen[k] = true;
        };
    }
    try testing.expect(seen[0] and seen[1] and seen[2] and seen[3]);
}

test "lowers a UBO mat4 * vec4 (the classic MVP) natively into scalar FMA" {
    const allocator = testing.allocator;
    // A vertex shader: layout(binding=0) uniform U { mat4 mvp; } u;
    //                  layout(location=0) in vec4 p; -> gl_Position = u.mvp * p;
    // The matrix is read with OpAccessChain (into the UBO member) + OpLoad %mat4 +
    // OpMatrixTimesVector - the native matrix path (no Prism rewrite).
    // ids: void=1 f32=2 v4f=3 uint=4 int=5 i0=6 mat4=7 U=8 pUniU=9 u=10 pUniMat=11
    //      pInV4=12 inP=13 pOutV4=14 outPos=15 voidfn=16 main=17 entry=18
    //      acMat=19 mat=20 p=21 prod=22 acOut=23.
    var b = try binary.Builder.init(allocator, 24);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.vertex, 17, 0, 13, 15 });
    try b.emit(allocator, op.Decorate, &.{ 8, op.Decoration.block });
    try b.emit(allocator, op.MemberDecorate, &.{ 8, 0, op.Decoration.col_major });
    try b.emit(allocator, op.MemberDecorate, &.{ 8, 0, op.Decoration.matrix_stride, 16 });
    try b.emit(allocator, op.MemberDecorate, &.{ 8, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.Decorate, &.{ 13, op.Decoration.location, 0 });
    try b.emit(allocator, op.Decorate, &.{ 15, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(allocator, op.TypeInt, &.{ 5, 32, 1 });
    try b.emit(allocator, op.Constant, &.{ 5, 6, 0 }); // int 0
    try b.emit(allocator, op.TypeMatrix, &.{ 7, 3, 4 }); // mat4 = 4 columns of v4float
    try b.emit(allocator, op.TypeStruct, &.{ 8, 7 }); // U { mat4 mvp; }
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.uniform, 8 });
    try b.emit(allocator, op.Variable, &.{ 9, 10, op.StorageClass.uniform });
    try b.emit(allocator, op.TypePointer, &.{ 11, op.StorageClass.uniform, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 12, op.StorageClass.input, 3 });
    try b.emit(allocator, op.Variable, &.{ 12, 13, op.StorageClass.input });
    try b.emit(allocator, op.TypePointer, &.{ 14, op.StorageClass.output, 3 });
    try b.emit(allocator, op.Variable, &.{ 14, 15, op.StorageClass.output });
    try b.emit(allocator, op.TypeFunction, &.{ 16, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 17, 0, 16 });
    try b.emit(allocator, op.Label, &.{18});
    try b.emit(allocator, op.AccessChain, &.{ 11, 19, 10, 6 }); // &u.mvp
    try b.emit(allocator, op.Load, &.{ 7, 20, 19 }); // mat4 mvp
    try b.emit(allocator, op.Load, &.{ 3, 21, 13 }); // vec4 p
    try b.emit(allocator, op.MatrixTimesVector, &.{ 3, 22, 20, 21 }); // mvp * p
    try b.emit(allocator, op.Store, &.{ 15, 22 }); // gl_Position = ...
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    // It must lower (no error.Unsupported from the matrix path) and produce the FMA
    // chain: each of the 4 result rows is M[0][i]*p0 + M[1][i]*p1 + M[2][i]*p2 + M[3][i]*p3.
    // 4 rows * (4 muls + 3 adds) = 28 arithmetic ops for the multiply alone, plus the
    // 16 matrix-element loads from the UBO buffer pointer.
    var buf: [8192]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{f}", .{func});
    // The matrix elements are read from the buffer param (loads), not a matrix value.
    try testing.expect(std.mem.indexOf(u8, text, "load f32") != null);
    // The result is stored to the position output (4 tagged stores).
    var n_pos: usize = 0;
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        if (hasAttr(&func, @enumFromInt(i), "out_attr", ATTR_POSITION)) n_pos += 1;
    }
    try testing.expectEqual(@as(usize, 1), n_pos); // exactly one store at ATTR_POSITION (X)
    // 16 element loads (4x4) for the matrix.
    try testing.expectEqual(@as(usize, 16), std.mem.count(u8, text, "load f32"));
}

test "lowers per-component vertex input access (glslang scalar-at-a-time reads)" {
    const allocator = testing.allocator;
    // A VS reading a vec4 input ONE component at a time, the real glslang shape:
    //   OpAccessChain %in %k + OpLoad %float, then CompositeConstruct back to a vec4,
    //   stored to gl_Position. Exercises pattern A (input component access) natively.
    // ids: void=1 f32=2 v4f=3 int=4 i0..i3=5..8 pInF=9 pInV4=10 pOutV4=11
    //      inVar=12 outVar=13 voidfn=14 main=15 entry=16 ac0..ac3=17..20
    //      x..w=21..24 res=25.
    var b = try binary.Builder.init(allocator, 26);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.vertex, 15, 0, 12, 13 });
    try b.emit(allocator, op.Decorate, &.{ 12, op.Decoration.location, 0 });
    try b.emit(allocator, op.Decorate, &.{ 13, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(allocator, op.TypeInt, &.{ 4, 32, 1 });
    try b.emit(allocator, op.Constant, &.{ 4, 5, 0 });
    try b.emit(allocator, op.Constant, &.{ 4, 6, 1 });
    try b.emit(allocator, op.Constant, &.{ 4, 7, 2 });
    try b.emit(allocator, op.Constant, &.{ 4, 8, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.input, 2 }); // ptr to float
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.input, 3 }); // ptr to v4f
    try b.emit(allocator, op.TypePointer, &.{ 11, op.StorageClass.output, 3 });
    try b.emit(allocator, op.Variable, &.{ 10, 12, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 11, 13, op.StorageClass.output });
    try b.emit(allocator, op.TypeFunction, &.{ 14, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 15, 0, 14 });
    try b.emit(allocator, op.Label, &.{16});
    try b.emit(allocator, op.AccessChain, &.{ 9, 17, 12, 5 }); // &in.x
    try b.emit(allocator, op.Load, &.{ 2, 21, 17 }); // in.x
    try b.emit(allocator, op.AccessChain, &.{ 9, 18, 12, 6 }); // &in.y
    try b.emit(allocator, op.Load, &.{ 2, 22, 18 }); // in.y
    try b.emit(allocator, op.AccessChain, &.{ 9, 19, 12, 7 }); // &in.z
    try b.emit(allocator, op.Load, &.{ 2, 23, 19 }); // in.z
    try b.emit(allocator, op.AccessChain, &.{ 9, 20, 12, 8 }); // &in.w
    try b.emit(allocator, op.Load, &.{ 2, 24, 20 }); // in.w
    try b.emit(allocator, op.CompositeConstruct, &.{ 3, 25, 21, 22, 23, 24 });
    try b.emit(allocator, op.Store, &.{ 13, 25 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    // 4 scalarized input params, the first tagged attr=0x80. The per-component loads
    // resolve to these params (no error.Unsupported from the access chains).
    const params = func.blockParams(@enumFromInt(0));
    try testing.expectEqual(@as(usize, 4), params.len);
    try testing.expect(hasAttr(&func, params[0], "attr", 0x80));
    try testing.expect(hasAttr(&func, params[3], "attr", 0x8c));
    try testing.expect(hasFuncStage(&func, "vertex"));
    // The reconstructed vec4 is stored to the position output: 4 stores at ATTR_POSITION.
    var n_pos: usize = 0;
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        if (hasAttr(&func, @enumFromInt(i), "out_attr", ATTR_POSITION)) n_pos += 1;
    }
    try testing.expectEqual(@as(usize, 1), n_pos);
}

test "lowers gl_PerVertex / gl_Position output (glslang interface block)" {
    const allocator = testing.allocator;
    // A passthrough VS writing position via the gl_PerVertex interface block, the real
    // glslang shape: an Output OpTypeStruct decorated Block whose member 0 is BuiltIn
    // Position. The store is OpAccessChain(%gl_PerVertex %int_0) + OpStore. Exercises
    // pattern B natively (no synthesized Position variable in Prism).
    // ids: void=1 f32=2 v4f=3 int=4 i0=5 perVtxStruct=6 pOutStruct=7 pInV4=8 pOutV4=9
    //      gl_PerVertex=10 inVar=11 voidfn=12 main=13 entry=14 loaded=15 posPtr=16.
    var b = try binary.Builder.init(allocator, 17);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.vertex, 13, 0, 11, 10 });
    try b.emit(allocator, op.Decorate, &.{ 6, op.Decoration.block }); // gl_PerVertex is a Block
    try b.emit(allocator, op.MemberDecorate, &.{ 6, 0, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(allocator, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(allocator, op.TypeInt, &.{ 4, 32, 1 });
    try b.emit(allocator, op.Constant, &.{ 4, 5, 0 }); // int 0 = position member index
    try b.emit(allocator, op.TypeStruct, &.{ 6, 3 }); // gl_PerVertex { vec4 gl_Position; }
    try b.emit(allocator, op.TypePointer, &.{ 7, op.StorageClass.output, 6 });
    try b.emit(allocator, op.TypePointer, &.{ 8, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.output, 3 }); // ptr to member
    try b.emit(allocator, op.Variable, &.{ 7, 10, op.StorageClass.output }); // gl_PerVertex
    try b.emit(allocator, op.Variable, &.{ 8, 11, op.StorageClass.input });
    try b.emit(allocator, op.TypeFunction, &.{ 12, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 13, 0, 12 });
    try b.emit(allocator, op.Label, &.{14});
    try b.emit(allocator, op.Load, &.{ 3, 15, 11 }); // load the vec4 input
    try b.emit(allocator, op.AccessChain, &.{ 9, 16, 10, 5 }); // &gl_PerVertex.gl_Position
    try b.emit(allocator, op.Store, &.{ 16, 15 }); // gl_Position = input
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    try testing.expect(hasFuncStage(&func, "vertex"));
    // The store through the gl_PerVertex Position member lands at ATTR_POSITION: exactly
    // one store tagged out_attr=0x70 (the X component, Y/Z/W follow at +4 each).
    var n_pos: usize = 0;
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        if (hasAttr(&func, @enumFromInt(i), "out_attr", ATTR_POSITION)) n_pos += 1;
    }
    try testing.expectEqual(@as(usize, 1), n_pos);
}

test "lowers a texturing fragment shader: OpImageSampleImplicitLod -> a host sampler call" {
    const allocator = testing.allocator;
    // FS: layout(binding=0) uniform sampler2D tex; layout(location=0) in vec2 uv;
    //     layout(location=0) out vec4 o; void main(){ o = texture(tex, uv); }
    // ids: void=1 fnty=2 f32=3 v2=4 v4=5 img=6 sImg=7 pUC=8 tex=9 pInV2=10 uv=11
    //      pOutV4=12 o=13 main=14 entry=15 si=16 c=17 res=18.
    var b = try binary.Builder.init(allocator, 19);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.fragment, 14, 0, 11, 13 });
    try b.emit(allocator, op.Decorate, &.{ 9, op.Decoration.binding, 0 });
    try b.emit(allocator, op.Decorate, &.{ 9, op.Decoration.descriptor_set, 0 });
    try b.emit(allocator, op.Decorate, &.{ 11, op.Decoration.location, 0 });
    try b.emit(allocator, op.Decorate, &.{ 13, op.Decoration.location, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1 });
    try b.emit(allocator, op.TypeFloat, &.{ 3, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 4, 3, 2 }); // v2float
    try b.emit(allocator, op.TypeVector, &.{ 5, 3, 4 }); // v4float
    try b.emit(allocator, op.TypeImage, &.{ 6, 3, op.Dim.dim_2d, 0, 0, 0, 1, 0 });
    try b.emit(allocator, op.TypeSampledImage, &.{ 7, 6 });
    try b.emit(allocator, op.TypePointer, &.{ 8, op.StorageClass.uniform_constant, 7 });
    try b.emit(allocator, op.Variable, &.{ 8, 9, op.StorageClass.uniform_constant }); // tex
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.input, 4 });
    try b.emit(allocator, op.Variable, &.{ 10, 11, op.StorageClass.input }); // uv
    try b.emit(allocator, op.TypePointer, &.{ 12, op.StorageClass.output, 5 });
    try b.emit(allocator, op.Variable, &.{ 12, 13, op.StorageClass.output }); // o
    try b.emit(allocator, op.Function, &.{ 1, 14, 0, 2 });
    try b.emit(allocator, op.Label, &.{15});
    try b.emit(allocator, op.Load, &.{ 7, 16, 9 }); // sampledImage = load tex
    try b.emit(allocator, op.Load, &.{ 4, 17, 11 }); // c = load uv
    try b.emit(allocator, op.ImageSampleImplicitLod, &.{ 5, 18, 16, 17 }); // texture(tex, uv)
    try b.emit(allocator, op.Store, &.{ 13, 18 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    try testing.expect(hasFuncStage(&func, "fragment"));
    // Entry params: uv (2 scalars, fpr) + the sampler descriptor pointer + the
    // appended sampler_fn pointer. The sampler_fn param carries the sampler_fn tag.
    const params = func.blockParams(@enumFromInt(0));
    var saw_sampler_fn = false;
    for (params) |p| {
        var it = func.attributesOf(.{ .value = p });
        while (it.next()) |attr| switch (attr) {
            .custom => |cu| if (std.mem.eql(u8, cu.namespace, "vulcan.gpu") and std.mem.eql(u8, cu.key, "sampler_fn")) {
                saw_sampler_fn = true;
            },
            else => {},
        };
    }
    try testing.expect(saw_sampler_fn);
    // The shader lowers to an indirect call (the host sampler) + an alloca for the
    // result slot, and 4 color-output stores of the reloaded RGBA.
    var buf: [4096]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{f}", .{func});
    try testing.expect(std.mem.indexOf(u8, text, "call_indirect") != null);
    try testing.expect(std.mem.indexOf(u8, text, "alloca") != null);
    var seen = [_]bool{ false, false, false, false };
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        inline for (0..4) |k| if (hasAttr(&func, @enumFromInt(i), "color_out", k)) {
            seen[k] = true;
        };
    }
    try testing.expect(seen[0] and seen[1] and seen[2] and seen[3]);
}

fn hasAttr(func: *const Function, v: Value, key: []const u8, slot: u32) bool {
    var it = func.attributesOf(.{ .value = v });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, key)) {
            return switch (c.value) {
                .int => |n| @as(u32, @intCast(n)) == slot,
                else => false,
            };
        },
        else => {},
    };
    return false;
}

fn hasFuncStage(func: *const Function, stage: []const u8) bool {
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "stage")) {
            return switch (c.value) {
                .string => |s| std.mem.eql(u8, s, stage),
                else => false,
            };
        },
        else => {},
    };
    return false;
}

/// Whether any value in the function carries a `vulcan.gpu.builtin = which` tag (the
/// synthesized BuiltIn entry param, e.g. gl_VertexIndex).
fn hasValueBuiltin(func: *const Function, which: u32) bool {
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        if (hasAttr(func, @enumFromInt(i), "builtin", which)) return true;
    }
    return false;
}

test "lowers gl_VertexIndex pulling a vec4 from a UBO array (vkcube vertex-pulling)" {
    const allocator = testing.allocator;
    // The vkcube-shaped vertex shader (vertex-pulling, ZERO vertex attributes):
    //   layout(binding=0) uniform U { vec4 pos[3]; } u;
    //   void main(){ gl_Position = u.pos[gl_VertexIndex]; }
    // gl_VertexIndex is a BuiltIn Input int. The position is PULLED from a UBO vec4
    // array indexed by that runtime value (a dynamic-index OpAccessChain).
    // ids: void=1 f32=2 v4f=3 int=4 i0=5 uint=6 u3=7 arr=8 U=9 pUniU=10 u=11
    //      pUniV4=12 pInInt=13 vidx=14 pOutV4=15 outPos=16 voidfn=17 main=18
    //      entry=19 vi=20 ac=21 pulled=22.
    var b = try binary.Builder.init(allocator, 23);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.vertex, 18, 0, 14, 16 });
    // The UBO is a Block struct. Member 0 (the vec4 array) is at offset 0, ArrayStride 16.
    try b.emit(allocator, op.Decorate, &.{ 9, op.Decoration.block });
    try b.emit(allocator, op.MemberDecorate, &.{ 9, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.Decorate, &.{ 8, op.Decoration.array_stride, 16 });
    try b.emit(allocator, op.Decorate, &.{ 11, op.Decoration.binding, 0 });
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.vertex_index });
    try b.emit(allocator, op.Decorate, &.{ 16, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(allocator, op.TypeInt, &.{ 4, 32, 1 }); // signed int
    try b.emit(allocator, op.Constant, &.{ 4, 5, 0 }); // int 0 (member index)
    try b.emit(allocator, op.TypeInt, &.{ 6, 32, 0 }); // uint (array length type)
    try b.emit(allocator, op.Constant, &.{ 6, 7, 3 }); // array length 3
    try b.emit(allocator, op.TypeArray, &.{ 8, 3, 7 }); // vec4[3]
    try b.emit(allocator, op.TypeStruct, &.{ 9, 8 }); // U { vec4 pos[3]; }
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.uniform, 9 });
    try b.emit(allocator, op.Variable, &.{ 10, 11, op.StorageClass.uniform });
    try b.emit(allocator, op.TypePointer, &.{ 12, op.StorageClass.uniform, 3 }); // ptr to vec4 element
    try b.emit(allocator, op.TypePointer, &.{ 13, op.StorageClass.input, 4 }); // ptr to int
    try b.emit(allocator, op.Variable, &.{ 13, 14, op.StorageClass.input }); // gl_VertexIndex
    try b.emit(allocator, op.TypePointer, &.{ 15, op.StorageClass.output, 3 });
    try b.emit(allocator, op.Variable, &.{ 15, 16, op.StorageClass.output }); // gl_Position
    try b.emit(allocator, op.TypeFunction, &.{ 17, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 18, 0, 17 });
    try b.emit(allocator, op.Label, &.{19});
    try b.emit(allocator, op.Load, &.{ 4, 20, 14 }); // int vi = gl_VertexIndex
    try b.emit(allocator, op.AccessChain, &.{ 12, 21, 11, 5, 20 }); // &u.pos[vi]
    try b.emit(allocator, op.Load, &.{ 3, 22, 21 }); // vec4 pulled
    try b.emit(allocator, op.Store, &.{ 16, 22 }); // gl_Position = pulled
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();

    try testing.expect(hasFuncStage(&func, "vertex"));
    // gl_VertexIndex synthesized a tagged i32 entry param.
    try testing.expect(hasValueBuiltin(&func, op.BuiltIn.vertex_index));

    var buf: [8192]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{f}", .{func});
    // The vec4 is fetched component-wise from the buffer (4 f32 loads), and the runtime
    // index*stride arithmetic produced a multiply (dynamic OpAccessChain).
    try testing.expectEqual(@as(usize, 4), std.mem.count(u8, text, "load f32"));
    try testing.expect(std.mem.indexOf(u8, text, "* ") != null or std.mem.indexOf(u8, text, "mul") != null);
    // The pulled vec4 is stored to the position output (4 tagged stores, one at X).
    var n_pos: usize = 0;
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        if (hasAttr(&func, @enumFromInt(i), "out_attr", ATTR_POSITION)) n_pos += 1;
    }
    try testing.expectEqual(@as(usize, 1), n_pos);
}

test "lowers a vertex shader that reads back its own gl_Position (vkcube frag_pos)" {
    const allocator = testing.allocator;
    // The vkcube shape: write gl_Position, then read it back to feed a varying:
    //   gl_Position = u.pos[gl_VertexIndex];  frag_pos = gl_Position.xyz;
    // The read-back is `OpAccessChain %gl_PerVertex %int_0` + `OpLoad %v4float`, which
    // must yield the just-stored clip-space position (not fault as an unmodeled load).
    // ids: void=1 f32=2 v4f=3 v3f=4 int=5 i0=6 uint=7 u3=8 arr=9 U=10 pUniU=11 u=12
    //      pUniV4=13 pInInt=14 vidx=15 gpvStruct=16 pOutGpv=17 gpv=18 pOutV3=19 fragpos=20
    //      pOutV4=21 voidfn=22 main=23 entry=24 vi=25 ac=26 pulled=27 acPos=28
    //      acPos2=29 loaded=30 shuf=31.
    var b = try binary.Builder.init(allocator, 32);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.vertex, 23, 0, 15, 18, 20 });
    try b.emit(allocator, op.Decorate, &.{ 10, op.Decoration.block });
    try b.emit(allocator, op.MemberDecorate, &.{ 10, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.Decorate, &.{ 9, op.Decoration.array_stride, 16 });
    try b.emit(allocator, op.Decorate, &.{ 12, op.Decoration.binding, 0 });
    try b.emit(allocator, op.Decorate, &.{ 15, op.Decoration.builtin, op.BuiltIn.vertex_index });
    // gl_PerVertex { vec4 gl_Position; } as an Output Block, member 0 = Position.
    try b.emit(allocator, op.Decorate, &.{ 16, op.Decoration.block });
    try b.emit(allocator, op.MemberDecorate, &.{ 16, 0, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(allocator, op.Decorate, &.{ 20, op.Decoration.location, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 4 }); // v4float
    try b.emit(allocator, op.TypeVector, &.{ 4, 2, 3 }); // v3float
    try b.emit(allocator, op.TypeInt, &.{ 5, 32, 1 });
    try b.emit(allocator, op.Constant, &.{ 5, 6, 0 }); // int 0
    try b.emit(allocator, op.TypeInt, &.{ 7, 32, 0 });
    try b.emit(allocator, op.Constant, &.{ 7, 8, 3 });
    try b.emit(allocator, op.TypeArray, &.{ 9, 3, 8 }); // vec4[3]
    try b.emit(allocator, op.TypeStruct, &.{ 10, 9 }); // U { vec4 pos[3]; }
    try b.emit(allocator, op.TypePointer, &.{ 11, op.StorageClass.uniform, 10 });
    try b.emit(allocator, op.Variable, &.{ 11, 12, op.StorageClass.uniform });
    try b.emit(allocator, op.TypePointer, &.{ 13, op.StorageClass.uniform, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 14, op.StorageClass.input, 5 });
    try b.emit(allocator, op.Variable, &.{ 14, 15, op.StorageClass.input }); // gl_VertexIndex
    try b.emit(allocator, op.TypeStruct, &.{ 16, 3 }); // gl_PerVertex { vec4 gl_Position; }
    try b.emit(allocator, op.TypePointer, &.{ 17, op.StorageClass.output, 16 });
    try b.emit(allocator, op.Variable, &.{ 17, 18, op.StorageClass.output }); // gl_PerVertex
    try b.emit(allocator, op.TypePointer, &.{ 19, op.StorageClass.output, 4 });
    try b.emit(allocator, op.Variable, &.{ 19, 20, op.StorageClass.output }); // frag_pos (v3)
    try b.emit(allocator, op.TypePointer, &.{ 21, op.StorageClass.output, 3 }); // ptr to Position member (v4)
    try b.emit(allocator, op.TypeFunction, &.{ 22, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 23, 0, 22 });
    try b.emit(allocator, op.Label, &.{24});
    try b.emit(allocator, op.Load, &.{ 5, 25, 15 }); // vi
    try b.emit(allocator, op.AccessChain, &.{ 13, 26, 12, 6, 25 }); // &u.pos[vi]
    try b.emit(allocator, op.Load, &.{ 3, 27, 26 }); // vec4 pulled
    try b.emit(allocator, op.AccessChain, &.{ 21, 28, 18, 6 }); // &gl_Position
    try b.emit(allocator, op.Store, &.{ 28, 27 }); // gl_Position = pulled
    try b.emit(allocator, op.AccessChain, &.{ 21, 29, 18, 6 }); // &gl_Position (read back)
    try b.emit(allocator, op.Load, &.{ 3, 30, 29 }); // vec4 gp = gl_Position
    try b.emit(allocator, op.VectorShuffle, &.{ 4, 31, 30, 30, 0, 1, 2 }); // gp.xyz
    try b.emit(allocator, op.Store, &.{ 20, 31 }); // frag_pos = gp.xyz
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();
    try testing.expect(hasFuncStage(&func, "vertex"));
    // The position read-back must lower (no error.MalformedModule) and the varying
    // frag_pos at Location 1... here location 0 generic. One store at ATTR_GENERIC0.
    var buf: [8192]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{f}", .{func});
    try testing.expect(std.mem.indexOf(u8, text, "load f32") != null); // pulled vec4 loads
}

test "lowers gl_InstanceIndex as a synthesized vertex entry param" {
    const allocator = testing.allocator;
    // A minimal VS reading gl_InstanceIndex (BuiltIn Input int) and converting it to a
    // float stored as gl_Position.x (just enough to prove the builtin lowers to a param).
    // ids: void=1 f32=2 v4f=3 int=4 c0f=5 pInInt=6 ii=7 pOutV4=8 outPos=9
    //      voidfn=10 main=11 entry=12 iv=13 fv=14 pos=15.
    var b = try binary.Builder.init(allocator, 16);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.vertex, 11, 0, 7, 9 });
    try b.emit(allocator, op.Decorate, &.{ 7, op.Decoration.builtin, op.BuiltIn.instance_index });
    try b.emit(allocator, op.Decorate, &.{ 9, op.Decoration.builtin, op.BuiltIn.position });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFloat, &.{ 2, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 4 });
    try b.emit(allocator, op.TypeInt, &.{ 4, 32, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 5, 0 }); // 0.0
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 4 });
    try b.emit(allocator, op.Variable, &.{ 6, 7, op.StorageClass.input }); // gl_InstanceIndex
    try b.emit(allocator, op.TypePointer, &.{ 8, op.StorageClass.output, 3 });
    try b.emit(allocator, op.Variable, &.{ 8, 9, op.StorageClass.output });
    try b.emit(allocator, op.TypeFunction, &.{ 10, 1 });
    try b.emit(allocator, op.Function, &.{ 1, 11, 0, 10 });
    try b.emit(allocator, op.Label, &.{12});
    try b.emit(allocator, op.Load, &.{ 4, 13, 7 }); // int iv = gl_InstanceIndex
    try b.emit(allocator, op.ConvertSToF, &.{ 2, 14, 13 }); // float fv = float(iv)
    try b.emit(allocator, op.CompositeConstruct, &.{ 3, 15, 14, 5, 5, 5 }); // (fv,0,0,0)
    try b.emit(allocator, op.Store, &.{ 9, 15 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();
    try testing.expect(hasFuncStage(&func, "vertex"));
    try testing.expect(hasValueBuiltin(&func, op.BuiltIn.instance_index));
}

/// Whether the function carries a `vulcan.gpu.grad_slot` func attr encoding (slot, axis):
/// the packed value is (slot << 1 | axis), axis 0 = dFdx, 1 = dFdy.
fn hasGradSlot(func: *const Function, slot: u32, is_y: bool) bool {
    const want: i64 = @as(i64, slot) << 1 | @intFromBool(is_y);
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "grad_slot")) {
            switch (c.value) {
                .int => |n| if (n == want) return true,
                else => {},
            }
        },
        else => {},
    };
    return false;
}

/// Whether any value carries the `vulcan.gpu.grad_buf` tag (the synthesized gradient-
/// buffer pointer param).
fn hasGradBuf(func: *const Function) bool {
    var i: u32 = 0;
    while (i < func.valueCount()) : (i += 1) {
        var it = func.attributesOf(.{ .value = @as(Value, @enumFromInt(i)) });
        while (it.next()) |attr| switch (attr) {
            .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "grad_buf")) return true,
            else => {},
        };
    }
    return false;
}

test "lowers fragment screen-space derivatives of a varying (dFdx/dFdy -> gradient params)" {
    const allocator = testing.allocator;
    // FS: layout(location=0) in vec3 frag_pos; layout(location=0) out vec4 o;
    //     void main(){ o = vec4(dFdx(frag_pos) + dFdy(frag_pos), 1.0); }
    // dFdx/dFdy of a varying lower to synthesized gradient params tagged with the
    // varying's attribute slots (location 0 -> 0x80, 0x84, 0x88).
    // ids: void=1 fn=2 f32=3 v3=4 v4=5 pIn=6 pOut=7 in=8 out=9 main=10 entry=11
    //      one=12 v=13 dx=14 dy=15 sum=16 res=17.
    var b = try binary.Builder.init(allocator, 18);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.fragment, 10, 0, 8, 9 });
    try b.emit(allocator, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(allocator, op.Decorate, &.{ 9, op.Decoration.location, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1 });
    try b.emit(allocator, op.TypeFloat, &.{ 3, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(allocator, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 7, op.StorageClass.output, 5 });
    try b.emit(allocator, op.Constant, &.{ 3, 12, @bitCast(@as(f32, 1.0)) });
    try b.emit(allocator, op.Variable, &.{ 6, 8, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 7, 9, op.StorageClass.output });
    try b.emit(allocator, op.Function, &.{ 1, 10, 0, 2 });
    try b.emit(allocator, op.Label, &.{11});
    try b.emit(allocator, op.Load, &.{ 4, 13, 8 }); // frag_pos
    try b.emit(allocator, op.DPdx, &.{ 4, 14, 13 }); // dFdx(frag_pos)
    try b.emit(allocator, op.DPdy, &.{ 4, 15, 13 }); // dFdy(frag_pos)
    try b.emit(allocator, op.FAdd, &.{ 4, 16, 14, 15 }); // sum = dFdx + dFdy (vec3)
    try b.emit(allocator, op.CompositeConstruct, &.{ 5, 17, 16, 12 }); // vec4(sum, 1.0)
    try b.emit(allocator, op.Store, &.{ 9, 17 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();
    try testing.expect(hasFuncStage(&func, "fragment"));
    // One synthesized grad_buf pointer param feeds all the gradients.
    try testing.expect(hasGradBuf(&func));
    // Gradient buffer slots for varying location 0, components 0/1/2 = attr slots
    // 0x80/0x84/0x88, both dFdx and dFdy.
    try testing.expect(hasGradSlot(&func, 0x80, false)); // dFdx of comp 0
    try testing.expect(hasGradSlot(&func, 0x84, false)); // dFdx of comp 1
    try testing.expect(hasGradSlot(&func, 0x88, false)); // dFdx of comp 2
    try testing.expect(hasGradSlot(&func, 0x80, true)); // dFdy of comp 0
    try testing.expect(hasGradSlot(&func, 0x88, true)); // dFdy of comp 2
}

test "lowers GLSL.std.450 cross + normalize + length + inversesqrt to native arithmetic" {
    const allocator = testing.allocator;
    // FS: in vec3 a(loc0), vec3 bb(loc1); out vec4 o;
    //     void main(){ o = vec4(normalize(cross(a, bb)), length(a)); }
    // Exercises cross (vec3), normalize (vec3 via inversesqrt+dot), and length (sqrt+dot).
    // ids: void=1 fn=2 f32=3 v3=4 v4=5 pIn=6 pOut=7 a=8 bb=9 o=10 main=11 entry=12
    //      av=13 bv=14 cr=15 nm=16 ln=17 res=18.
    const SET = 100; // OpExtInstImport id (ignored by the lowering)
    var b = try binary.Builder.init(allocator, 19);
    defer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.fragment, 11, 0, 8, 9, 10 });
    try b.emit(allocator, op.Decorate, &.{ 8, op.Decoration.location, 0 });
    try b.emit(allocator, op.Decorate, &.{ 9, op.Decoration.location, 1 });
    try b.emit(allocator, op.Decorate, &.{ 10, op.Decoration.location, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeFunction, &.{ 2, 1 });
    try b.emit(allocator, op.TypeFloat, &.{ 3, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(allocator, op.TypeVector, &.{ 5, 3, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 7, op.StorageClass.output, 5 });
    try b.emit(allocator, op.Variable, &.{ 6, 8, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 6, 9, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 7, 10, op.StorageClass.output });
    try b.emit(allocator, op.Function, &.{ 1, 11, 0, 2 });
    try b.emit(allocator, op.Label, &.{12});
    try b.emit(allocator, op.Load, &.{ 4, 13, 8 }); // a
    try b.emit(allocator, op.Load, &.{ 4, 14, 9 }); // bb
    try b.emit(allocator, op.ExtInst, &.{ 4, 15, SET, op.Glsl.cross, 13, 14 }); // cross(a,bb)
    try b.emit(allocator, op.ExtInst, &.{ 4, 16, SET, op.Glsl.normalize, 15 }); // normalize(...)
    try b.emit(allocator, op.ExtInst, &.{ 3, 17, SET, op.Glsl.length, 13 }); // length(a)
    try b.emit(allocator, op.CompositeConstruct, &.{ 5, 18, 16, 17 }); // vec4(vec3, scalar)
    try b.emit(allocator, op.Store, &.{ 10, 18 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});

    // Lowers without error -> cross/normalize/length are recognized + emitted natively.
    var func = try lowerModule(allocator, b.words.items);
    defer func.deinit();
    try testing.expect(hasFuncStage(&func, "fragment"));
}

// Hardening regression tests: `lowerModule` consumes an untrusted `[]const u32` with no
// validation layer, so a malformed word stream must be REJECTED (error.MalformedModule),
// never indexed out of bounds or panicked on.
test "rejects a result id at/beyond the id bound (would OOB the per-id arrays)" {
    const allocator = testing.allocator;
    var b = try binary.Builder.init(allocator, 4); // valid ids are [0, 4)
    defer b.deinit(allocator);
    // OpTypeInt's result id 4 == id_bound: `types[4]` (sized 4) would be an out-of-bounds write.
    try b.emit(allocator, op.TypeInt, &.{ 4, 32, 1 });
    try testing.expectError(error.MalformedModule, lowerModule(allocator, b.words.items));
}

test "rejects a truncated instruction (fewer operands than the opcode needs)" {
    const allocator = testing.allocator;
    var b = try binary.Builder.init(allocator, 4);
    defer b.deinit(allocator);
    // OpTypeInt needs [result, width, signedness]; this one carries only the result id, so
    // reading the width/signedness operands would slice past the instruction.
    try b.emit(allocator, op.TypeInt, &.{1});
    try testing.expectError(error.MalformedModule, lowerModule(allocator, b.words.items));
}

test "rejects an oversized vector length (would overflow the scalarized Vec)" {
    const allocator = testing.allocator;
    var b = try binary.Builder.init(allocator, 8);
    defer b.deinit(allocator);
    try b.emit(allocator, op.TypeFloat, &.{ 1, 32 });
    // A vec9 cannot be held in Vec.comps ([4]Value): reject rather than overflow it later.
    try b.emit(allocator, op.TypeVector, &.{ 2, 1, 9 });
    try testing.expectError(error.MalformedModule, lowerModule(allocator, b.words.items));
}
