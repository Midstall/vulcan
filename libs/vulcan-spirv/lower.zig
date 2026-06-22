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
    @"struct", // members tracked via Module.members
    other, // matrix / etc., not lowered

    fn asVector(self: TypeInfo) ?VecType {
        return switch (self) {
            .vector => |v| v,
            else => null,
        };
    }
};

const VecType = struct { elem: u32, len: u8 };

/// A scalarized vector value: its component scalar values. A vector SSA id maps to these
/// rather than a single Vulcan value, so no backend needs vector support.
const Vec = struct { comps: [4]Value = undefined, len: u8 = 0 };

/// One struct member's type and byte offset (from its `OpMemberDecorate Offset`).
const Member = struct { type_id: u32, offset: u32 };

/// Pack a (struct type id, member index) into a hash-map key.
fn memberKey(struct_id: u32, member: u32) u64 {
    return (@as(u64, struct_id) << 32) | member;
}

const Const = struct { type_id: u32, bits: u64 };

/// How a module-level `OpVariable` is used. `input`/`output` are the graphics attribute
/// interface: an Input variable feeds a vertex/fragment shader, an Output variable is its
/// product (clip-space position, fragment color).
const VarKind = enum { buffer, global_id, input, output, other };

/// The shader stage an `OpEntryPoint` selects. Lowering produces the attribute interface
/// for graphics stages, the kernel ABI for compute.
const Stage = enum { compute, vertex, fragment };

// NVIDIA attribute byte addresses the graphics backend expects, mirrored here so the
// target-agnostic frontend can tag input params and output stores with their slot.
// These match vulcan-target/nvidia/encode.zig.
const ATTR_POSITION: u32 = 0x70; // clip-space position output
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
    buffers: std.ArrayList(u32) = .empty, // storage-buffer var ids, declaration order
    has_global_id: bool = false,
    value_of: []?Value, // id -> the Vulcan value it lowers to (scalars)
    vec_of: []Vec, // id -> a scalarized vector value's components (len 0 = not a vector)
    is_builtin_ptr: []bool, // id -> a pointer into the gl_GlobalInvocationID builtin
    ptr_t: Type,
    i32_t: Type,
    global_id_value: ?Value = null,
    members: std.AutoHashMapUnmanaged(u64, Member) = .empty, // (struct,member) -> type+offset
    array_stride: []u32, // type id -> ArrayStride decoration (0 = derive from element)
    const_val: []i64, // id -> the integer value of an OpConstant (for struct indices)
    var_type: []u32, // variable id -> its (pointer) type id

    // Graphics interface (only used for vertex/fragment stages).
    stage: Stage = .compute,
    location: []u32, // input/output var id -> Location decoration (0 = none / position)
    has_location: []bool, // input/output var id -> whether Location was decorated
    is_position: []bool, // output var id -> whether it is the Position builtin
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

/// Lower the first function of the SPIR-V module in `words` to a fresh Vulcan IR
/// function. Caller owns and must `deinit` it.
pub fn lowerModule(allocator: std.mem.Allocator, words: []const u32) Error!Function {
    var r = try binary.Reader.init(words);
    const bound = r.header.id_bound;

    var types = try allocator.alloc(?TypeInfo, bound);
    defer allocator.free(types);
    @memset(types, null);
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
    const is_position = try allocator.alloc(bool, bound);
    defer allocator.free(is_position);
    @memset(is_position, false);
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
        .value_of = undefined,
        .vec_of = undefined,
        .is_builtin_ptr = undefined,
        .ptr_t = try func.types.intern(.ptr),
        .i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }),
        .array_stride = array_stride,
        .const_val = const_val,
        .var_type = var_type,
        .location = location,
        .has_location = has_location,
        .is_position = is_position,
        .var_storage = var_storage,
    };
    defer module.buffers.deinit(allocator);
    defer module.members.deinit(allocator);

    var func_ret_type: ?u32 = null;
    var in_function = false;
    var local_size_x: u32 = 1; // workgroup x dimension (OpExecutionMode LocalSize)
    var pending_vars: std.ArrayList([2]u32) = .empty; // (var id, storage class)
    defer pending_vars.deinit(allocator);

    while (try r.next()) |inst| {
        switch (inst.opcode) {
            op.TypeVoid => types[inst.operands[0]] = .void,
            op.TypeBool => types[inst.operands[0]] = .{ .scalar = try func.types.intern(.bool) },
            op.TypeInt => types[inst.operands[0]] = .{ .scalar = try func.types.intern(.{ .int = .{
                .signedness = if (inst.operands[2] != 0) .signed else .unsigned,
                .bits = @intCast(inst.operands[1]),
            } }) },
            op.TypeFloat => types[inst.operands[0]] = .{ .scalar = try func.types.intern(.{ .float = if (inst.operands[1] == 64) .f64 else .f32 }) },
            op.TypeFunction => types[inst.operands[0]] = .{ .function = .{ .ret = inst.operands[1] } },
            op.TypePointer => types[inst.operands[0]] = .{ .pointer = .{ .pointee = inst.operands[2] } }, // [result, storageClass, pointee]
            op.TypeArray, op.TypeRuntimeArray => types[inst.operands[0]] = .{ .array = .{ .elem = inst.operands[1] } },
            op.TypeVector => types[inst.operands[0]] = .{ .vector = .{ .elem = inst.operands[1], .len = @intCast(inst.operands[2]) } },
            op.TypeStruct => {
                // [result, member0Type, member1Type, ...]. The member offsets may
                // already be present from earlier OpMemberDecorate, so preserve them.
                types[inst.operands[0]] = .@"struct";
                for (inst.operands[1..], 0..) |member_type, i| {
                    const key = memberKey(inst.operands[0], @intCast(i));
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
                builtin_decor[inst.operands[0]] = inst.operands[2];
                if (inst.operands[2] == op.BuiltIn.position) is_position[inst.operands[0]] = true;
            } else if (inst.operands.len >= 3 and inst.operands[1] == op.Decoration.array_stride) {
                array_stride[inst.operands[0]] = inst.operands[2];
            } else if (inst.operands.len >= 3 and inst.operands[1] == op.Decoration.location) {
                location[inst.operands[0]] = inst.operands[2];
                has_location[inst.operands[0]] = true;
            },
            op.MemberDecorate => if (inst.operands.len >= 4 and inst.operands[2] == op.Decoration.offset) {
                // [struct, member, Offset, value] (annotations come before the type).
                const key = memberKey(inst.operands[0], inst.operands[1]);
                if (module.members.getPtr(key)) |m| {
                    m.offset = inst.operands[3];
                } else {
                    try module.members.put(allocator, key, .{ .type_id = 0, .offset = inst.operands[3] });
                }
            },
            op.ExecutionMode => if (inst.operands.len >= 3 and inst.operands[1] == op.ExecutionModeKind.local_size) {
                local_size_x = inst.operands[2]; // [entryPoint, LocalSize, x, y, z]
            },
            op.Constant => {
                const bits = constBits(inst.operands[2..]);
                try consts.put(allocator, inst.operands[1], .{ .type_id = inst.operands[0], .bits = bits });
                const_val[inst.operands[1]] = @bitCast(bits); // for struct member indices
            },
            op.ConstantTrue => try consts.put(allocator, inst.operands[1], .{ .type_id = inst.operands[0], .bits = 1 }),
            op.ConstantFalse => try consts.put(allocator, inst.operands[1], .{ .type_id = inst.operands[0], .bits = 0 }),
            op.ConstantComposite => {
                // [type, result, comp0, comp1, ...]: a vector constant whose components
                // are themselves OpConstants. Record the component ids, materialized
                // component-wise in lowerFunction.
                const comps = try allocator.dupe(u32, inst.operands[2..]);
                try composite_consts.put(allocator, inst.operands[1], comps);
            },
            op.Variable => {
                try pending_vars.append(allocator, .{ inst.operands[1], inst.operands[2] }); // [type, result, storageClass]
                var_type[inst.operands[1]] = inst.operands[0]; // the variable's pointer type
                var_storage[inst.operands[1]] = inst.operands[2];
            },
            op.Function => {
                if (in_function) return error.Unsupported; // only the first function
                in_function = true;
                func_ret_type = inst.operands[0];
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
        if (module.stage != .compute and class == op.StorageClass.input) {
            var_kind[id] = .input;
        } else if (module.stage != .compute and class == op.StorageClass.output) {
            var_kind[id] = .output;
        } else if (class == op.StorageClass.storage_buffer or class == op.StorageClass.uniform) {
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
    const is_builtin_ptr = try allocator.alloc(bool, bound);
    defer allocator.free(is_builtin_ptr);
    @memset(is_builtin_ptr, false);
    module.is_builtin_ptr = is_builtin_ptr;

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
                try blocks.append(allocator, .{ .block = b, .label = inst.operands[0] });
                cur = blocks.items.len - 1;
            },
            op.Phi => try blocks.items[cur orelse return error.MalformedModule].phis.append(allocator, inst.operands[1]),
            else => {},
        }
    }
    if (blocks.items.len == 0) return error.MalformedModule;
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
    // Graphics: each Input variable becomes a scalarized block parameter, one per vector
    // component, tagged with its attribute slot (ATTR_GENERIC0 + the Location's 16-byte
    // stride + the component's 4-byte offset). An OpLoad of the variable yields the
    // recorded Vec. The backend emits one ALD/IPA per scalar.
    if (module.stage != .compute) try synthInputAttribs(func, module, entry);
    for (module.buffers.items) |buf_id| value_of[buf_id] = try func.appendBlockParam(entry, module.ptr_t);
    for (insts) |inst| {
        if (inst.opcode == op.FunctionParameter) {
            const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
            value_of[inst.operands[1]] = try func.appendBlockParam(entry, ty);
        }
    }

    // Non-entry block parameters = that block's phis (in order).
    for (blocks.items) |b| {
        for (b.phis.items) |phi_id| {
            const ty = scalarTypeOfResult(module.types, insts, phi_id) orelse return error.Unsupported;
            value_of[phi_id] = try func.appendBlockParam(b.block, ty);
        }
    }

    // Materialize constants into the entry block.
    var it = consts.iterator();
    while (it.next()) |e| {
        const ty = scalarType(module.types, e.value_ptr.type_id) orelse return error.Unsupported;
        const bits = e.value_ptr.bits;
        value_of[e.key_ptr.*] = switch (func.types.type_kind(ty)) {
            .float => |f| try func.appendInst(entry, ty, .{ .fconst = if (f == .f32)
                @as(f64, @as(f32, @bitCast(@as(u32, @truncate(bits)))))
            else
                @bitCast(bits) }),
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
                    out.comps[out.len] = c;
                    out.len += 1;
                }
            } else {
                out.comps[out.len] = value_of[comp_id] orelse return error.MalformedModule;
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
        try lowerTerminator(allocator, func, value_of, blocks.items, bi, insts, inst);
    }
    _ = ret_type_id;
}

/// Synthesize entry-block parameters for a graphics shader's Input variables. Each Input
/// variable (declaration order) becomes one block parameter per vector component, tagged
/// with its attribute byte slot so the backend fetches it (ALD in a vertex shader, IPA in
/// a fragment shader). The variable's scalar/Vec value is recorded so an OpLoad resolves
/// to these params.
fn synthInputAttribs(func: *Function, module: *Module, entry: Block) Error!void {
    // Variables by ascending id (SPIR-V declaration order).
    var id: u32 = 0;
    while (id < module.var_kind.len) : (id += 1) {
        if (module.var_kind[id] != .input) continue;
        const pointee = module.var_pointee(id) orelse return error.Unsupported;
        const loc = if (module.has_location[id]) module.location[id] else 0;
        const slot_base = ATTR_GENERIC0 + loc * 0x10;
        if (vectorInfo(module.types, pointee)) |vi| {
            const elem = scalarType(module.types, vi.elem) orelse return error.Unsupported;
            var out: Vec = .{ .len = vi.len };
            var c: u8 = 0;
            while (c < vi.len) : (c += 1) {
                const p = try func.appendBlockParam(entry, elem);
                try tagAttr(func, p, "attr", slot_base + c * 4);
                out.comps[c] = p;
            }
            module.vec_of[id] = out;
        } else {
            const elem = scalarType(module.types, pointee) orelse return error.Unsupported;
            const p = try func.appendBlockParam(entry, elem);
            try tagAttr(func, p, "attr", slot_base);
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

    var c: u8 = 0;
    while (c < n) : (c += 1) {
        const key: []const u8 = if (module.stage == .fragment) "color_out" else "out_attr";
        const slot: u32 = if (module.stage == .fragment)
            c // fragment color component index -> R0..R3
        else if (module.is_position[var_id])
            ATTR_POSITION + c * 4
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
    S.buf[0] = module.value_of[id] orelse return error.MalformedModule;
    return S.buf[0..1];
}

/// Attach a `vulcan.gpu` integer attribute named `key` (a graphics attribute slot) to
/// value `v`, mirroring how the backend reads it back.
fn tagAttr(func: *Function, v: Value, key: []const u8, slot: u32) Error!void {
    try func.addAttr(.{ .value = v }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = key, .value = .{ .int = @intCast(slot) } } });
}

/// Lower a non-terminator body instruction, recording its result Value.
fn lowerBodyInst(allocator: std.mem.Allocator, func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    const value_of = module.value_of;
    // Vectors are scalarized: composite/shuffle/vector-arith operate per component.
    if (try lowerVectorInst(func, module, block, inst)) return;

    if (binOpOf(inst.opcode)) |bop| {
        const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
        const lhs = value_of[inst.operands[2]] orelse return error.MalformedModule;
        const rhs = value_of[inst.operands[3]] orelse return error.MalformedModule;
        value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .arith = .{ .op = bop, .lhs = lhs, .rhs = rhs } });
        return;
    }
    if (cmpOpOf(inst.opcode)) |cop| {
        const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
        const lhs = value_of[inst.operands[2]] orelse return error.MalformedModule;
        const rhs = value_of[inst.operands[3]] orelse return error.MalformedModule;
        value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .icmp = .{ .op = cop, .lhs = lhs, .rhs = rhs } });
        return;
    }
    switch (inst.opcode) {
        op.Select => {
            const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
            const cond = value_of[inst.operands[2]] orelse return error.MalformedModule;
            const a = value_of[inst.operands[3]] orelse return error.MalformedModule;
            const b = value_of[inst.operands[4]] orelse return error.MalformedModule;
            value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .select = .{ .cond = cond, .then = a, .@"else" = b } });
        },
        op.ConvertFToU, op.ConvertFToS, op.ConvertSToF, op.ConvertUToF, op.UConvert, op.SConvert, op.FConvert => {
            const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
            const v = value_of[inst.operands[2]] orelse return error.MalformedModule;
            value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .convert = .{ .value = v } });
        },
        op.SNegate, op.FNegate => {
            // Unary negate: 0 - x (integer `sub` for SNegate, float `sub` for FNegate,
            // dispatched by the result type in codegen).
            const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
            const v = value_of[inst.operands[2]] orelse return error.MalformedModule;
            const zero = if (inst.opcode == op.FNegate)
                try func.appendInst(block, ty, .{ .fconst = 0 })
            else
                try func.appendInst(block, ty, .{ .iconst = 0 });
            value_of[inst.operands[1]] = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = v } });
        },
        op.Not => {
            // Bitwise complement: x ^ -1 (all ones).
            const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
            const v = value_of[inst.operands[2]] orelse return error.MalformedModule;
            value_of[inst.operands[1]] = try func.appendArithImm(block, ty, .bit_xor, v, -1);
        },
        op.ExtInst => try lowerExtInst(func, module, block, inst),
        op.Undef => {
            // A safe concrete value for an undefined: zero of the result type.
            const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
            value_of[inst.operands[1]] = if (func.types.type_kind(ty) == .float)
                try func.appendInst(block, ty, .{ .fconst = 0 })
            else
                try func.appendInst(block, ty, .{ .iconst = 0 });
        },
        op.Nop => {},
        op.AccessChain => try lowerAccessChain(func, module, block, inst),
        op.Load => {
            const result = inst.operands[1];
            const ptr_id = inst.operands[2];
            if (module.is_builtin_ptr[ptr_id]) {
                // A load of gl_GlobalInvocationID.x is the synthesized invocation id.
                value_of[result] = module.global_id_value orelse return error.MalformedModule;
            } else if (ptr_id < module.var_kind.len and module.var_kind[ptr_id] == .input) {
                // A graphics Input variable: the load yields the synthesized
                // attribute params (a scalar, or a Vec for a vector input).
                if (module.vec_of[ptr_id].len > 0) {
                    module.vec_of[result] = module.vec_of[ptr_id];
                } else {
                    value_of[result] = value_of[ptr_id] orelse return error.MalformedModule;
                }
            } else {
                const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
                const ptr = value_of[ptr_id] orelse return error.MalformedModule;
                value_of[result] = try func.appendInst(block, ty, .{ .load = .{ .ptr = ptr } });
            }
        },
        op.Store => {
            const ptr_id = inst.operands[0];
            // A graphics Output variable: scalarize the stored vector and tag each
            // component store with its output attribute slot.
            if (ptr_id < module.var_kind.len and module.var_kind[ptr_id] == .output) {
                try storeOutputAttrib(func, module, block, ptr_id, inst.operands[1]);
                return;
            }
            const ptr = value_of[ptr_id] orelse return error.MalformedModule;
            const val = value_of[inst.operands[1]] orelse return error.MalformedModule;
            try func.appendStore(block, val, ptr);
        },
        op.Variable => return error.Unsupported, // function-local variables (alloca) not yet modeled
        op.Phi, op.FunctionParameter => {},
        op.SelectionMerge, op.LoopMerge, op.Name, op.MemberName, op.Decorate, op.MemberDecorate => {},
        op.Branch, op.BranchConditional, op.Return, op.ReturnValue, op.Unreachable => {},
        else => return error.Unsupported,
    }
    _ = allocator;
}

fn vectorInfo(types: []const ?TypeInfo, type_id: u32) ?VecType {
    if (type_id >= types.len) return null;
    return (types[type_id] orelse return null).asVector();
}

/// Lower vector instructions by scalarizing: a vector value is a list of scalar component
/// values, a vector operation becomes one scalar operation per component. Returns true if
/// `inst` was a vector op (handled here), false if it is a scalar op for the caller. No
/// backend needs vector support.
fn lowerVectorInst(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!bool {
    const v_of = module.vec_of;
    switch (inst.opcode) {
        op.CompositeConstruct => {
            const vi = vectorInfo(module.types, inst.operands[0]) orelse return false; // struct construct: not here
            var out: Vec = .{ .len = 0 };
            for (inst.operands[2..]) |cid| {
                // A component may itself be a (sub)vector, so flatten it.
                if (v_of[cid].len > 0) {
                    for (v_of[cid].comps[0..v_of[cid].len]) |c| {
                        out.comps[out.len] = c;
                        out.len += 1;
                    }
                } else {
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
            const cv = v_of[inst.operands[2]];
            if (cv.len == 0) return false; // extracting from a struct/array: not here
            const idx = inst.operands[3];
            if (idx >= cv.len) return error.MalformedModule;
            module.value_of[inst.operands[1]] = cv.comps[idx];
            return true;
        },
        op.VectorShuffle => {
            // [type, result, vec1, vec2, indices...]. An index < len(vec1) selects
            // from vec1, else from vec2 (offset by len(vec1)).
            const v1 = v_of[inst.operands[2]];
            const v2 = v_of[inst.operands[3]];
            var out: Vec = .{ .len = 0 };
            for (inst.operands[4..]) |ix| {
                out.comps[out.len] = if (ix < v1.len) v1.comps[ix] else v2.comps[ix - v1.len];
                out.len += 1;
            }
            v_of[inst.operands[1]] = out;
            return true;
        },
        op.VectorTimesScalar => {
            const vi = vectorInfo(module.types, inst.operands[0]) orelse return false;
            const elem = scalarType(module.types, vi.elem) orelse return error.Unsupported;
            const vec = v_of[inst.operands[2]];
            const s = module.value_of[inst.operands[3]] orelse return error.MalformedModule;
            var out: Vec = .{ .len = vi.len };
            for (0..vi.len) |i| out.comps[i] = try func.appendInst(block, elem, .{ .arith = .{ .op = .mul, .lhs = vec.comps[i], .rhs = s } });
            v_of[inst.operands[1]] = out;
            return true;
        },
        op.Dot => {
            // A scalar result: sum of component products.
            const elem = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
            const v1 = v_of[inst.operands[2]];
            const v2 = v_of[inst.operands[3]];
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
                if (vectorInfo(module.types, inst.operands[0])) |vi| {
                    const elem = scalarType(module.types, vi.elem) orelse return error.Unsupported;
                    const a = v_of[inst.operands[2]];
                    const b = v_of[inst.operands[3]];
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

/// Lower a GLSL.std.450 extended instruction (min/max/abs/clamp) to select + compare. The
/// comparison's signed/unsigned/float behavior comes from the operand type in codegen, so
/// one `icmp` shape serves the F/S/U variants.
fn lowerExtInst(func: *Function, module: *Module, block: Block, inst: binary.Instruction) Error!void {
    if (inst.operands.len < 5) return error.Unsupported;
    const ty = scalarType(module.types, inst.operands[0]) orelse return error.Unsupported;
    const result = inst.operands[1];
    const which = inst.operands[3]; // operands: [type, result, set, instruction, args..]
    const args = inst.operands[4..];
    const bool_t = try func.types.intern(.bool);

    const arg = struct {
        fn v(m: *Module, a: []const u32, i: usize) Error!Value {
            return m.value_of[a[i]] orelse error.MalformedModule;
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
                const a = try arg(module, args, 0);
                const b = try arg(module, args, 1);
                const lt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
                break :blk try func.appendInst(block, ty, .{ .select = .{ .cond = lt, .then = a, .@"else" = b } });
            },
            op.Glsl.f_max, op.Glsl.u_max, op.Glsl.s_max => blk: {
                const a = try arg(module, args, 0);
                const b = try arg(module, args, 1);
                const gt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
                break :blk try func.appendInst(block, ty, .{ .select = .{ .cond = gt, .then = a, .@"else" = b } });
            },
            op.Glsl.f_clamp, op.Glsl.u_clamp, op.Glsl.s_clamp => blk: {
                // clamp(x, lo, hi) = min(max(x, lo), hi)
                const x = try arg(module, args, 0);
                const lo = try arg(module, args, 1);
                const hi = try arg(module, args, 2);
                const gt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = lo } });
                const m = try func.appendInst(block, ty, .{ .select = .{ .cond = gt, .then = x, .@"else" = lo } });
                const lt = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = m, .rhs = hi } });
                break :blk try func.appendInst(block, ty, .{ .select = .{ .cond = lt, .then = m, .@"else" = hi } });
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
    const base = inst.operands[2];
    const indices = inst.operands[3..];

    if (base < module.var_kind.len and module.var_kind[base] == .global_id) {
        module.is_builtin_ptr[result] = true;
        return;
    }

    const base_ptr = module.value_of[base] orelse return error.MalformedModule;
    var cur_type = module.var_pointee(base) orelse return error.Unsupported; // the pointee of the variable
    var const_off: i64 = 0; // accumulated compile-time byte offset
    var offset_val: ?Value = null; // accumulated runtime byte offset (i32)

    for (indices) |idx_id| {
        switch (module.types[cur_type] orelse return error.Unsupported) {
            .@"struct" => {
                // A struct member index must be a constant. Add its byte offset.
                const member: u32 = @intCast(module.const_val[idx_id]);
                const m = module.members.get(memberKey(cur_type, member)) orelse return error.Unsupported;
                const_off += m.offset;
                cur_type = m.type_id;
            },
            .array => |arr| {
                const stride: u32 = if (module.array_stride[cur_type] != 0) module.array_stride[cur_type] else @intCast(vulcanScalarSize(func, scalarType(module.types, arr.elem) orelse return error.Unsupported));
                const idx = module.value_of[idx_id] orelse return error.MalformedModule;
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
    module.value_of[result] = if (offset_val) |o|
        try func.appendInst(block, module.ptr_t, .{ .arith = .{ .op = .add, .lhs = base_ptr, .rhs = o } })
    else
        base_ptr;
}

fn lowerTerminator(allocator: std.mem.Allocator, func: *Function, value_of: []const ?Value, blocks: []const BlockInfo, bi: usize, insts: []const binary.Instruction, inst: binary.Instruction) Error!void {
    switch (inst.opcode) {
        op.Return, op.Unreachable => func.setTerminator(blocks[bi].block, .{ .ret = null }),
        op.ReturnValue => func.setTerminator(blocks[bi].block, .{ .ret = value_of[inst.operands[0]] orelse return error.MalformedModule }),
        op.Branch => {
            const target = blockIndex(blocks, inst.operands[0]);
            const args = try phiArgs(allocator, value_of, blocks, target, blocks[bi].label, insts);
            defer allocator.free(args);
            try func.setJump(blocks[bi].block, blocks[target].block, args);
        },
        op.BranchConditional => {
            const cond = value_of[inst.operands[0]] orelse return error.MalformedModule;
            const t = blockIndex(blocks, inst.operands[1]);
            const f = blockIndex(blocks, inst.operands[2]);
            const t_args = try phiArgs(allocator, value_of, blocks, t, blocks[bi].label, insts);
            defer allocator.free(t_args);
            const f_args = try phiArgs(allocator, value_of, blocks, f, blocks[bi].label, insts);
            defer allocator.free(f_args);
            try func.appendIf(blocks[bi].block, cond, .{ .target = blocks[t].block, .args = t_args }, .{ .target = blocks[f].block, .args = f_args });
        },
        else => {},
    }
}

fn phiArgs(allocator: std.mem.Allocator, value_of: []const ?Value, blocks: []const BlockInfo, target: usize, pred_label: u32, insts: []const binary.Instruction) Error![]Value {
    const phis = blocks[target].phis.items;
    const args = try allocator.alloc(Value, phis.len);
    errdefer allocator.free(args);
    for (phis, 0..) |phi_id, i| {
        const operands = phiOperandsOf(insts, phi_id) orelse return error.MalformedModule;
        var j: usize = 0;
        var found: ?Value = null;
        while (j + 1 < operands.len) : (j += 2) {
            if (operands[j + 1] == pred_label) {
                found = value_of[operands[j]] orelse return error.MalformedModule;
                break;
            }
        }
        args[i] = found orelse return error.MalformedModule;
    }
    return args;
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

fn scalarTypeOfResult(types: []const ?TypeInfo, insts: []const binary.Instruction, result_id: u32) ?Type {
    for (insts) |inst| {
        if (inst.opcode == op.Phi and inst.operands[1] == result_id) return scalarType(types, inst.operands[0]);
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
