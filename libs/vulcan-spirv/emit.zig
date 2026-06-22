//! Emit SPIR-V from a Vulcan IR function: the reverse of `lower.zig`. Handles scalar
//! arithmetic, int<->float conversions, comparisons, select, vectors, and structured
//! control flow.
//!
//! Constants are hoisted to the declaration section (SPIR-V requires `OpConstant`
//! there). A comparison with an integer IR result emits a bool `OpSLessThan`-style op
//! then an `OpSelect` to 0/1, mirroring how GLSL's `a < b` yields an int.

const std = @import("std");
const ir = @import("vulcan-ir");
const binary = @import("binary.zig");
const op = @import("opcodes.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;

pub const Error = error{ UnsupportedConstruct, MultiBlockUnsupported } || std.mem.Allocator.Error;

/// Emit `func` as a standalone SPIR-V module (words), exporting it under `name` via a
/// LinkageAttributes Export decoration (so the module validates as a linkable library
/// without an entry point). Caller owns the result.
pub fn emitModule(allocator: std.mem.Allocator, func: *const Function, name: []const u8) Error![]u32 {
    var e = try Emitter.init(allocator, func);
    defer e.deinit();
    try e.run(name);
    return e.finish();
}

const TypeId = struct { ty: Type, id: u32 };
const PtrId = struct { pointee: u32, storage: u32, id: u32 };
const VecId = struct { elem: u32, len: u8, id: u32 };
const IntConst = struct { ty: Type, val: i64, id: u32 };

/// An input interface variable resolved to SPIR-V ids: the variable, its component
/// element type, the component count, and the base parameter index it feeds.
const InputRec = struct { var_id: u32, elem_ty: u32, comps: u8, base: usize };

/// Shader-specific block context: load inputs into the entry block, store the output
/// before each `return`.
const ShaderBlockCtx = struct {
    input_recs: []const InputRec,
    out_var: u32,
    out_value_ty: u32,
    output: ?OutputVar,
    // Push-constant uniforms (flattened to floats): block variable, pointer-to-float
    // member type, float type, the parameter index they begin at, and the count.
    pc_var: u32 = 0,
    pc_float_ptr: u32 = 0,
    f32_ty: u32 = 0,
    uniform_base: usize = 0,
    uniform_count: u32 = 0,
};

/// Which pipeline stage a shader entry point drives.
pub const Stage = enum { fragment, vertex, compute };

/// An `in` interface variable: its pipeline location (or a SPIR-V BuiltIn number, e.g. 15
/// = FragCoord, when `builtin` is set) and component count (1 scalar, 2..4 vector). A
/// vector input consumes that many parameters (a scalarized vector is N scalar params).
pub const InterfaceVar = struct { location: u32, components: u8 = 1, builtin: ?u32 = null };

/// An `out` interface variable: its location (or a SPIR-V BuiltIn number, e.g. 0 =
/// Position for gl_Position) and the scalarized component values forming it (1 scalar, N
/// vector). Constructed into a vector and stored.
pub const OutputVar = struct { location: u32, components: []const Value, builtin: ?u32 = null };

/// Describes a shader's entry-point interface, supplied alongside the IR function.
pub const ShaderInfo = struct {
    stage: Stage,
    inputs: []const InterfaceVar = &.{},
    output: ?OutputVar = null,
    local_size: [3]u32 = .{ 1, 1, 1 },
    /// Number of scalarized (float) uniform components, placed in a push-constant block.
    /// These follow the inputs in the parameter list.
    uniform_count: u32 = 0,
    /// Number of `sampler2D` descriptors (UniformConstant sampled-image variables), bound
    /// at descriptor set 0, bindings 0..sampler_count.
    sampler_count: u32 = 0,
};

fn executionModel(stage: Stage) u32 {
    return switch (stage) {
        .vertex => op.ExecutionModel.vertex,
        .fragment => op.ExecutionModel.fragment,
        .compute => op.ExecutionModel.gl_compute,
    };
}

/// Emit `func` as a SPIR-V shader entry point described by `info`. Caller owns it.
pub fn emitShader(allocator: std.mem.Allocator, func: *const Function, info: ShaderInfo) Error![]u32 {
    var e = try Emitter.init(allocator, func);
    defer e.deinit();
    try e.runShader(info);
    return e.finish();
}

const Emitter = struct {
    allocator: std.mem.Allocator,
    func: *const Function,
    next_id: u32 = 1,
    caps: std.ArrayList(u32) = .empty, // OpCapability
    ext_imports: std.ArrayList(u32) = .empty, // OpExtInstImport (GLSL.std.450)
    preamble: std.ArrayList(u32) = .empty, // memory model + entry point + execution mode
    debug: std.ArrayList(u32) = .empty, // OpName
    annotations: std.ArrayList(u32) = .empty, // OpDecorate
    decls: std.ArrayList(u32) = .empty, // types + constants
    body: std.ArrayList(u32) = .empty, // the function
    type_cache: std.ArrayList(TypeId) = .empty,
    ptr_cache: std.ArrayList(PtrId) = .empty,
    vec_cache: std.ArrayList(VecId) = .empty,
    int_consts: std.ArrayList(IntConst) = .empty,
    void_id: u32 = 0,
    bool_id: u32 = 0,
    true_id: u32 = 0,
    false_id: u32 = 0,
    uint_id: u32 = 0,
    glsl_ext_id: u32 = 0,
    /// OpVariable id of each `sampler2D` (index = binding) plus the shared sampled-image
    /// type id, for `tex.sample.<n>` calls. Empty when the shader has no samplers.
    sampler_vars: []u32 = &.{},
    sampled_image_ty: u32 = 0,
    value_ids: []u32,

    fn init(allocator: std.mem.Allocator, func: *const Function) Error!Emitter {
        const ids = try allocator.alloc(u32, func.valueCount());
        @memset(ids, 0);
        return .{ .allocator = allocator, .func = func, .value_ids = ids };
    }

    fn deinit(self: *Emitter) void {
        self.caps.deinit(self.allocator);
        self.ext_imports.deinit(self.allocator);
        self.preamble.deinit(self.allocator);
        self.debug.deinit(self.allocator);
        self.annotations.deinit(self.allocator);
        self.decls.deinit(self.allocator);
        self.body.deinit(self.allocator);
        self.type_cache.deinit(self.allocator);
        self.ptr_cache.deinit(self.allocator);
        self.vec_cache.deinit(self.allocator);
        self.int_consts.deinit(self.allocator);
        if (self.sampler_vars.len > 0) self.allocator.free(self.sampler_vars);
        self.allocator.free(self.value_ids);
    }

    fn fresh(self: *Emitter) u32 {
        const id = self.next_id;
        self.next_id += 1;
        return id;
    }
    fn emit(self: *Emitter, list: *std.ArrayList(u32), opcode: u16, operands: []const u32) Error!void {
        const word_count: u32 = @intCast(1 + operands.len);
        try list.append(self.allocator, (word_count << 16) | opcode);
        try list.appendSlice(self.allocator, operands);
    }
    /// Append `s` as a SPIR-V literal string: little-endian bytes packed 4 per word,
    /// null-terminated (trailing zero word when the length is a multiple of 4).
    fn appendString(self: *Emitter, list: *std.ArrayList(u32), s: []const u8) Error!void {
        var i: usize = 0;
        while (i < s.len) : (i += 4) {
            var word: u32 = 0;
            inline for (0..4) |j| {
                if (i + j < s.len) word |= @as(u32, s[i + j]) << @intCast(j * 8);
            }
            try list.append(self.allocator, word);
        }
        if (s.len % 4 == 0) try list.append(self.allocator, 0);
    }

    fn setVal(self: *Emitter, v: Value, id: u32) void {
        self.value_ids[@intFromEnum(v)] = id;
    }
    /// SPIR-V id of a value, allocating one on first reference. Makes forward references
    /// work: a loop-header `OpPhi` naming a back-edge value defined later gets the same id
    /// the definition will use.
    fn idFor(self: *Emitter, v: Value) u32 {
        const existing = self.value_ids[@intFromEnum(v)];
        if (existing != 0) return existing;
        const id = self.fresh();
        self.value_ids[@intFromEnum(v)] = id;
        return id;
    }

    fn typeId(self: *Emitter, ty: Type) Error!u32 {
        const kind = self.func.types.type_kind(ty);
        if (kind == .bool) return self.boolTypeId(); // one shared OpTypeBool
        for (self.type_cache.items) |t| if (t.ty == ty) return t.id;
        const id = self.fresh();
        switch (kind) {
            .int => |i| try self.emit(&self.decls, op.TypeInt, &.{ id, i.bits, if (i.signedness == .signed) 1 else 0 }),
            .float => |f| try self.emit(&self.decls, op.TypeFloat, &.{ id, if (f == .f64) 64 else 32 }),
            else => return error.UnsupportedConstruct,
        }
        try self.type_cache.append(self.allocator, .{ .ty = ty, .id = id });
        return id;
    }
    fn voidTypeId(self: *Emitter) Error!u32 {
        if (self.void_id == 0) {
            self.void_id = self.fresh();
            try self.emit(&self.decls, op.TypeVoid, &.{self.void_id});
        }
        return self.void_id;
    }
    fn boolTypeId(self: *Emitter) Error!u32 {
        if (self.bool_id == 0) {
            self.bool_id = self.fresh();
            try self.emit(&self.decls, op.TypeBool, &.{self.bool_id});
        }
        return self.bool_id;
    }

    /// Id of the imported `GLSL.std.450` extended instruction set (sqrt, floor, ...),
    /// emitted into the ext-import section on first use.
    fn glslExtId(self: *Emitter) Error!u32 {
        if (self.glsl_ext_id == 0) {
            self.glsl_ext_id = self.fresh();
            var ops: std.ArrayList(u32) = .empty;
            defer ops.deinit(self.allocator);
            try ops.append(self.allocator, self.glsl_ext_id);
            try self.appendString(&ops, "GLSL.std.450");
            try self.emit(&self.ext_imports, op.ExtInstImport, ops.items);
        }
        return self.glsl_ext_id;
    }

    /// A 32-bit unsigned int constant `val` (for OpAccessChain indices), via a cached uint
    /// type. Emitted fresh each call. Indices are few, so not pooled.
    fn uintTypeId(self: *Emitter) Error!u32 {
        if (self.uint_id == 0) {
            self.uint_id = self.fresh();
            try self.emit(&self.decls, op.TypeInt, &.{ self.uint_id, 32, 0 });
        }
        return self.uint_id;
    }
    fn indexConstId(self: *Emitter, val: u32) Error!u32 {
        const id = self.fresh();
        try self.emit(&self.decls, op.Constant, &.{ try self.uintTypeId(), id, val });
        return id;
    }

    /// A bool constant (`OpConstantTrue`/`OpConstantFalse`), hoisted to the decls.
    fn boolConstId(self: *Emitter, val: bool) Error!u32 {
        const slot = if (val) &self.true_id else &self.false_id;
        if (slot.* == 0) {
            slot.* = self.fresh();
            try self.emit(&self.decls, if (val) op.ConstantTrue else op.ConstantFalse, &.{ try self.boolTypeId(), slot.* });
        }
        return slot.*;
    }

    /// An `OpConstant` of integer type `ty` and value `val`, hoisted to the decls.
    fn intConstId(self: *Emitter, ty: Type, val: i64) Error!u32 {
        for (self.int_consts.items) |c| if (c.ty == ty and c.val == val) return c.id;
        const tid = try self.typeId(ty);
        const id = self.fresh();
        try self.emit(&self.decls, op.Constant, &.{ tid, id, @as(u32, @truncate(@as(u64, @bitCast(val)))) });
        try self.int_consts.append(self.allocator, .{ .ty = ty, .val = val, .id = id });
        return id;
    }

    /// The non-terminating `if` instruction of a block, if any.
    fn ifOf(self: *Emitter, block: Block) ?ir.function.If {
        for (self.func.blockInsts(block)) |inst| {
            if (self.func.opcode(inst) == .@"if") return self.func.opcode(inst).@"if";
        }
        return null;
    }

    /// Argument values on `pred`'s control-flow edge to `target`, or null if `pred` does
    /// not branch to `target`.
    fn edgeArgsTo(self: *Emitter, pred: Block, target: Block) ?[]const Value {
        if (self.ifOf(pred)) |iff| {
            if (iff.then.target == target) return self.func.blockArgs(iff.then);
            if (iff.@"else".target == target) return self.func.blockArgs(iff.@"else");
            return null;
        }
        if (self.func.terminator(pred)) |t| switch (t) {
            .jump => |j| if (j.target == target) return self.func.blockArgs(j),
            .ret => {},
        };
        return null;
    }

    fn run(self: *Emitter, name: []const u8) Error!void {
        try self.emit(&self.caps, op.Capability, &.{1}); // Shader
        try self.emit(&self.caps, op.Capability, &.{5}); // Linkage (exported library function)
        try self.emit(&self.preamble, op.MemoryModel, &.{ 0, 1 }); // Logical, GLSL450

        const n = self.func.blockCount();
        const entry: Block = @enumFromInt(0);
        const params = self.func.blockParams(entry);

        // Return type and value: from whichever block returns (one in an if/else diamond).
        var ret_val: ?Value = null;
        for (0..n) |i| {
            if (self.func.terminator(@enumFromInt(i))) |t| switch (t) {
                .ret => |maybe| ret_val = maybe,
                .jump => {},
            };
        }
        const ret_ty_id = if (ret_val) |v| try self.typeId(self.func.valueType(v)) else try self.voidTypeId();

        // Function type: OpTypeFunction %ret %param...
        var fn_operands: std.ArrayList(u32) = .empty;
        defer fn_operands.deinit(self.allocator);
        const fn_ty_id = self.fresh();
        try fn_operands.append(self.allocator, fn_ty_id);
        try fn_operands.append(self.allocator, ret_ty_id);
        for (params) |p| try fn_operands.append(self.allocator, try self.typeId(self.func.valueType(p)));
        try self.emit(&self.decls, op.TypeFunction, fn_operands.items);

        // Name the function and decorate it as an exported linkable symbol.
        const fn_id = self.fresh();
        var name_ops: std.ArrayList(u32) = .empty;
        defer name_ops.deinit(self.allocator);
        try name_ops.append(self.allocator, fn_id);
        try self.appendString(&name_ops, name);
        try self.emit(&self.debug, op.Name, name_ops.items);

        var link_ops: std.ArrayList(u32) = .empty;
        defer link_ops.deinit(self.allocator);
        try link_ops.append(self.allocator, fn_id);
        try link_ops.append(self.allocator, 41); // LinkageAttributes decoration
        try self.appendString(&link_ops, name);
        try link_ops.append(self.allocator, 0); // Export linkage type
        try self.emit(&self.annotations, op.Decorate, link_ops.items);

        for (0..n) |i| try self.hoistConstants(@enumFromInt(i));

        // Pre-assign a label id per block (branches forward-reference them).
        const labels = try self.allocator.alloc(u32, n);
        defer self.allocator.free(labels);
        for (0..n) |i| labels[i] = self.fresh();

        try self.emit(&self.body, op.Function, &.{ ret_ty_id, fn_id, 0, fn_ty_id });
        for (params) |p| {
            const pid = self.fresh();
            try self.emit(&self.body, op.FunctionParameter, &.{ try self.typeId(self.func.valueType(p)), pid });
            self.setVal(p, pid);
        }

        const visited = try self.allocator.alloc(bool, n);
        defer self.allocator.free(visited);
        @memset(visited, false);
        var stops: std.ArrayList(u32) = .empty;
        defer stops.deinit(self.allocator);
        try self.emitBlock(entry, &stops, visited, labels, null);
        try self.emit(&self.body, op.FunctionEnd, &.{});
    }

    /// Load each shader input into the entry block, mapping the function parameters.
    /// A vector input is loaded then split into its component parameters.
    fn emitInputLoads(self: *Emitter, sc: ShaderBlockCtx, params: []const Value) Error!void {
        for (sc.input_recs) |rec| {
            const value_ty = if (rec.comps == 1) rec.elem_ty else try self.vecTypeId(rec.elem_ty, rec.comps);
            const loaded = self.fresh();
            try self.emit(&self.body, op.Load, &.{ value_ty, loaded, rec.var_id });
            if (rec.comps == 1) {
                self.setVal(params[rec.base], loaded);
            } else {
                for (0..rec.comps) |c| {
                    const ext = self.fresh();
                    try self.emit(&self.body, op.CompositeExtract, &.{ rec.elem_ty, ext, loaded, @intCast(c) });
                    self.setVal(params[rec.base + c], ext);
                }
            }
        }
    }

    /// Load each push-constant uniform float into its parameter (AccessChain + Load).
    fn emitUniformLoads(self: *Emitter, sc: ShaderBlockCtx, params: []const Value) Error!void {
        var j: u32 = 0;
        while (j < sc.uniform_count) : (j += 1) {
            const idx = try self.indexConstId(j);
            const ptr = self.fresh();
            try self.emit(&self.body, op.AccessChain, &.{ sc.pc_float_ptr, ptr, sc.pc_var, idx });
            const val = self.fresh();
            try self.emit(&self.body, op.Load, &.{ sc.f32_ty, val, ptr });
            self.setVal(params[sc.uniform_base + j], val);
        }
    }

    /// Construct (for a vector) and store the shader output before a `return`.
    fn emitOutputStore(self: *Emitter, sc: ShaderBlockCtx) Error!void {
        const o = sc.output orelse return;
        if (o.components.len == 1) {
            try self.emit(&self.body, op.Store, &.{ sc.out_var, self.idFor(o.components[0]) });
        } else {
            var cc: std.ArrayList(u32) = .empty;
            defer cc.deinit(self.allocator);
            const val_id = self.fresh();
            try cc.append(self.allocator, sc.out_value_ty);
            try cc.append(self.allocator, val_id);
            for (o.components) |comp| try cc.append(self.allocator, self.idFor(comp));
            try self.emit(&self.body, op.CompositeConstruct, cc.items);
            try self.emit(&self.body, op.Store, &.{ sc.out_var, val_id });
        }
    }

    /// Read a `cf.<key>` integer block attribute (the merge/continue block index the GLSL
    /// lowering records on a structured-control-flow head).
    fn cfAttr(self: *Emitter, block: Block, key: []const u8) ?u32 {
        var it = self.func.attributesOf(.{ .block = block });
        while (it.next()) |attr| switch (attr) {
            .custom => |c| {
                if (std.mem.eql(u8, c.namespace, "cf") and std.mem.eql(u8, c.key, key)) {
                    switch (c.value) {
                        .int => |v| return @intCast(v),
                        else => {},
                    }
                }
            },
            else => {},
        };
        return null;
    }

    /// Structured merge block of an `if`/loop head: the recorded `cf.merge`, else (for a
    /// simple diamond) the block both branches jump to.
    fn mergeOf(self: *Emitter, block: Block) ?u32 {
        if (self.cfAttr(block, "merge")) |m| return m;
        const iff = self.ifOf(block) orelse return null;
        const tt = self.func.terminator(iff.then.target) orelse return null;
        const et = self.func.terminator(iff.@"else".target) orelse return null;
        if (tt != .jump or et != .jump or tt.jump.target != et.jump.target) return null;
        return @intFromEnum(tt.jump.target);
    }

    fn continueOf(self: *Emitter, block: Block) ?u32 {
        return self.cfAttr(block, "continue");
    }

    /// Emit blocks in structured order: a block, then its branch subtrees, then its merge
    /// last, so each construct's merge follows the whole construct as SPIR-V requires.
    /// `stops` holds the merges of enclosing constructs, which this level must not emit.
    fn emitBlock(self: *Emitter, block: Block, stops: *std.ArrayList(u32), visited: []bool, labels: []const u32, shader: ?ShaderBlockCtx) Error!void {
        const idx = @intFromEnum(block);
        for (stops.items) |s| if (s == idx) return;
        if (visited[idx]) return;
        visited[idx] = true;
        try self.emitFunctionBlock(block, labels, shader);

        if (self.ifOf(block)) |iff| {
            const merge = self.mergeOf(block) orelse return error.MultiBlockUnsupported;
            try stops.append(self.allocator, merge);
            try self.emitBlock(iff.then.target, stops, visited, labels, shader);
            if (self.continueOf(block) == null) try self.emitBlock(iff.@"else".target, stops, visited, labels, shader);
            _ = stops.pop();
            try self.emitBlock(@enumFromInt(merge), stops, visited, labels, shader);
        } else if (self.func.terminator(block)) |t| switch (t) {
            .jump => |j| try self.emitBlock(j.target, stops, visited, labels, shader),
            .ret => {},
        };
    }

    /// Emit one block of a (possibly multi-block) function: its label, its phis (non-entry
    /// block parameters) or shader input loads (entry block), its instructions, and its
    /// control-flow exit (with the shader output store before any `return`).
    fn emitFunctionBlock(self: *Emitter, block: Block, labels: []const u32, shader: ?ShaderBlockCtx) Error!void {
        const n = self.func.blockCount();
        const idx = @intFromEnum(block);
        try self.emit(&self.body, op.Label, &.{labels[idx]});

        if (idx == 0) {
            if (shader) |sc| {
                try self.emitInputLoads(sc, self.func.blockParams(block));
                if (sc.uniform_count > 0) try self.emitUniformLoads(sc, self.func.blockParams(block));
            }
        } else {
            // Block parameters of a non-entry block become OpPhi.
            const bparams = self.func.blockParams(block);
            for (bparams, 0..) |param, pi| {
                var phi: std.ArrayList(u32) = .empty;
                defer phi.deinit(self.allocator);
                const phi_ty = try self.typeId(self.func.valueType(param));
                try phi.append(self.allocator, phi_ty);
                try phi.append(self.allocator, self.idFor(param));
                for (0..n) |j| {
                    if (self.edgeArgsTo(@enumFromInt(j), block)) |args| {
                        try phi.append(self.allocator, self.idFor(args[pi]));
                        try phi.append(self.allocator, labels[j]);
                    }
                }
                try self.emit(&self.body, op.Phi, phi.items);
            }
        }

        // Instructions (the structured `if` is emitted as the block exit, not here).
        for (self.func.blockInsts(block)) |inst| {
            if (self.func.opcode(inst) != .@"if") try self.emitInst(inst);
        }

        // Control-flow exit.
        if (self.ifOf(block)) |iff| {
            const merge = self.mergeOf(block) orelse return error.MultiBlockUnsupported;
            if (self.continueOf(block)) |cont| {
                try self.emit(&self.body, op.LoopMerge, &.{ labels[merge], labels[cont], 0 });
            } else {
                try self.emit(&self.body, op.SelectionMerge, &.{ labels[merge], 0 });
            }
            try self.emit(&self.body, op.BranchConditional, &.{ self.idFor(iff.cond), labels[@intFromEnum(iff.then.target)], labels[@intFromEnum(iff.@"else".target)] });
        } else if (self.func.terminator(block)) |t| switch (t) {
            .jump => |j| try self.emit(&self.body, op.Branch, &.{labels[@intFromEnum(j.target)]}),
            .ret => |maybe| {
                // `discard` (fragment kill) is a `ret` block tagged `cf.discard`: emit
                // OpKill instead of writing the output and returning.
                if (self.cfAttr(block, "discard") != null) {
                    try self.emit(&self.body, op.Kill, &.{});
                } else {
                    if (shader) |sc| try self.emitOutputStore(sc);
                    if (maybe) |v| {
                        try self.emit(&self.body, op.ReturnValue, &.{self.idFor(v)});
                    } else try self.emit(&self.body, op.Return, &.{});
                }
            },
        } else {
            if (shader) |sc| try self.emitOutputStore(sc);
            try self.emit(&self.body, op.Return, &.{});
        }
    }

    /// Hoist every constant in `entry` to a module-level `OpConstant` (SPIR-V requires
    /// constants in the declaration section), mapping each IR value to the constant id.
    fn hoistConstants(self: *Emitter, entry: Block) Error!void {
        for (self.func.blockInsts(entry)) |inst| {
            const result = self.func.instResult(inst) orelse continue;
            switch (self.func.opcode(inst)) {
                .iconst => |v| {
                    const ty = self.func.valueType(result);
                    const id = if (self.func.types.type_kind(ty) == .bool) try self.boolConstId(v != 0) else try self.intConstId(ty, v);
                    self.setVal(result, id);
                },
                .fconst => |v| {
                    const ty = self.func.valueType(result);
                    const tid = try self.typeId(ty);
                    const id = self.fresh();
                    if (self.func.types.type_kind(ty).float == .f64) {
                        const bits: u64 = @bitCast(v);
                        try self.emit(&self.decls, op.Constant, &.{ tid, id, @truncate(bits), @truncate(bits >> 32) });
                    } else {
                        const bits: u32 = @bitCast(@as(f32, @floatCast(v)));
                        try self.emit(&self.decls, op.Constant, &.{ tid, id, bits });
                    }
                    self.setVal(result, id);
                },
                else => {},
            }
        }
    }

    /// An `OpTypeVector elem len` id, cached.
    fn vecTypeId(self: *Emitter, elem: u32, len: u8) Error!u32 {
        for (self.vec_cache.items) |v| if (v.elem == elem and v.len == len) return v.id;
        const id = self.fresh();
        try self.emit(&self.decls, op.TypeVector, &.{ id, elem, len });
        try self.vec_cache.append(self.allocator, .{ .elem = elem, .len = len, .id = id });
        return id;
    }

    /// An `OpTypePointer storage pointee` id, cached.
    fn ptrTypeId(self: *Emitter, pointee: u32, storage: u32) Error!u32 {
        for (self.ptr_cache.items) |p| if (p.pointee == pointee and p.storage == storage) return p.id;
        const id = self.fresh();
        try self.emit(&self.decls, op.TypePointer, &.{ id, storage, pointee });
        try self.ptr_cache.append(self.allocator, .{ .pointee = pointee, .storage = storage, .id = id });
        return id;
    }

    /// Emit a shader entry point. Inputs map to the function's parameters (one `Input`
    /// variable each, loaded at entry). The output value, if any, maps to a single
    /// `Output` variable stored before `OpReturn`. `main` itself is `void()`.
    fn runShader(self: *Emitter, info: ShaderInfo) Error!void {
        const entry: Block = @enumFromInt(0);
        const params = self.func.blockParams(entry);

        var total_in: usize = 0;
        for (info.inputs) |iv| total_in += iv.components;
        if (total_in + info.uniform_count != params.len) return error.UnsupportedConstruct;

        try self.emit(&self.caps, op.Capability, &.{1}); // Shader
        try self.emit(&self.preamble, op.MemoryModel, &.{ 0, 1 }); // Logical, GLSL450

        const void_id = try self.voidTypeId();
        const fnty = self.fresh();
        try self.emit(&self.decls, op.TypeFunction, &.{ fnty, void_id });
        const main_id = self.fresh();

        var interface: std.ArrayList(u32) = .empty;
        defer interface.deinit(self.allocator);

        // Input variables: a scalar input is one scalar variable, a vector input is one
        // OpTypeVector variable whose components map to consecutive parameters.
        var input_recs: std.ArrayList(InputRec) = .empty;
        defer input_recs.deinit(self.allocator);
        var base: usize = 0;
        for (info.inputs) |iv| {
            const elem_ty = try self.typeId(self.func.valueType(params[base]));
            const value_ty = if (iv.components == 1) elem_ty else try self.vecTypeId(elem_ty, iv.components);
            const ptr_ty = try self.ptrTypeId(value_ty, op.StorageClass.input);
            const var_id = self.fresh();
            try self.emit(&self.decls, op.Variable, &.{ ptr_ty, var_id, op.StorageClass.input });
            if (iv.builtin) |b| {
                try self.emit(&self.annotations, op.Decorate, &.{ var_id, op.Decoration.builtin, b });
            } else {
                try self.emit(&self.annotations, op.Decorate, &.{ var_id, op.Decoration.location, iv.location });
            }
            try interface.append(self.allocator, var_id);
            try input_recs.append(self.allocator, .{ .var_id = var_id, .elem_ty = elem_ty, .comps = iv.components, .base = base });
            base += iv.components;
        }

        // Sampler variables: one UniformConstant OpTypeSampledImage per `sampler2D`, bound
        // at descriptor set 0 / bindings 0.. Not part of the SPIR-V 1.0 entry interface.
        if (info.sampler_count > 0) {
            var f32_ty: u32 = 0;
            for (params) |p| {
                const t = self.func.valueType(p);
                if (self.func.types.type_kind(t) == .float) {
                    f32_ty = try self.typeId(t);
                    break;
                }
            }
            if (f32_ty == 0) return error.UnsupportedConstruct; // sampler shaders need float coords
            const img_ty = self.fresh();
            // OpTypeImage float 2D depth=0 arrayed=0 ms=0 sampled=1 format=Unknown.
            try self.emit(&self.decls, op.TypeImage, &.{ img_ty, f32_ty, op.Dim.dim_2d, 0, 0, 0, 1, 0 });
            const si_ty = self.fresh();
            try self.emit(&self.decls, op.TypeSampledImage, &.{ si_ty, img_ty });
            self.sampled_image_ty = si_ty;
            const ptr_ty = try self.ptrTypeId(si_ty, op.StorageClass.uniform_constant);
            const vars = try self.allocator.alloc(u32, info.sampler_count);
            for (0..info.sampler_count) |i| {
                const v = self.fresh();
                try self.emit(&self.decls, op.Variable, &.{ ptr_ty, v, op.StorageClass.uniform_constant });
                try self.emit(&self.annotations, op.Decorate, &.{ v, op.Decoration.descriptor_set, 0 });
                try self.emit(&self.annotations, op.Decorate, &.{ v, op.Decoration.binding, @intCast(i) });
                vars[i] = v;
            }
            self.sampler_vars = vars;
        }

        // Output variable: a scalar value or a constructed vector.
        var out_var: u32 = 0;
        var out_value_ty: u32 = 0;
        if (info.output) |o| {
            const elem_ty = try self.typeId(self.func.valueType(o.components[0]));
            out_value_ty = if (o.components.len == 1) elem_ty else try self.vecTypeId(elem_ty, @intCast(o.components.len));
            const ptr_ty = try self.ptrTypeId(out_value_ty, op.StorageClass.output);
            out_var = self.fresh();
            try self.emit(&self.decls, op.Variable, &.{ ptr_ty, out_var, op.StorageClass.output });
            if (o.builtin) |b| {
                try self.emit(&self.annotations, op.Decorate, &.{ out_var, op.Decoration.builtin, b });
            } else {
                try self.emit(&self.annotations, op.Decorate, &.{ out_var, op.Decoration.location, o.location });
            }
            try interface.append(self.allocator, out_var);
        }

        // OpEntryPoint <model> %main "main" <interface...>
        var ep: std.ArrayList(u32) = .empty;
        defer ep.deinit(self.allocator);
        try ep.append(self.allocator, executionModel(info.stage));
        try ep.append(self.allocator, main_id);
        try self.appendString(&ep, "main");
        try ep.appendSlice(self.allocator, interface.items);
        try self.emit(&self.preamble, op.EntryPoint, ep.items);
        switch (info.stage) {
            .fragment => try self.emit(&self.preamble, op.ExecutionMode, &.{ main_id, op.ExecutionModeKind.origin_upper_left }),
            .compute => try self.emit(&self.preamble, op.ExecutionMode, &.{ main_id, op.ExecutionModeKind.local_size, info.local_size[0], info.local_size[1], info.local_size[2] }),
            .vertex => {},
        }

        const n = self.func.blockCount();
        for (0..n) |i| try self.hoistConstants(@enumFromInt(i));

        // Pre-assign a label id per block (branches forward-reference them).
        const labels = try self.allocator.alloc(u32, n);
        defer self.allocator.free(labels);
        for (0..n) |i| labels[i] = self.fresh();

        // void main(): load inputs in the entry block, run the body, store the output
        // before returning.
        try self.emit(&self.body, op.Function, &.{ void_id, main_id, 0, fnty });
        // Push-constant block for the (float-flattened) uniforms.
        var sc = ShaderBlockCtx{ .input_recs = input_recs.items, .out_var = out_var, .out_value_ty = out_value_ty, .output = info.output };
        if (info.uniform_count > 0) {
            const f32_ty = try self.typeId(self.func.valueType(params[total_in]));
            var members: std.ArrayList(u32) = .empty;
            defer members.deinit(self.allocator);
            const struct_ty = self.fresh();
            try members.append(self.allocator, struct_ty);
            for (0..info.uniform_count) |_| try members.append(self.allocator, f32_ty);
            try self.emit(&self.decls, op.TypeStruct, members.items);
            try self.emit(&self.annotations, op.Decorate, &.{ struct_ty, op.Decoration.block });
            for (0..info.uniform_count) |i| try self.emit(&self.annotations, op.MemberDecorate, &.{ struct_ty, @intCast(i), op.Decoration.offset, @intCast(i * 4) });
            const struct_ptr = try self.ptrTypeId(struct_ty, op.StorageClass.push_constant);
            const pc_var = self.fresh();
            try self.emit(&self.decls, op.Variable, &.{ struct_ptr, pc_var, op.StorageClass.push_constant });
            sc.pc_var = pc_var;
            sc.pc_float_ptr = try self.ptrTypeId(f32_ty, op.StorageClass.push_constant);
            sc.f32_ty = f32_ty;
            sc.uniform_base = total_in;
            sc.uniform_count = info.uniform_count;
        }
        const visited = try self.allocator.alloc(bool, n);
        defer self.allocator.free(visited);
        @memset(visited, false);
        var stops: std.ArrayList(u32) = .empty;
        defer stops.deinit(self.allocator);
        try self.emitBlock(entry, &stops, visited, labels, sc);
        try self.emit(&self.body, op.FunctionEnd, &.{});
    }

    fn emitInst(self: *Emitter, inst: ir.function.Inst) Error!void {
        const result = self.func.instResult(inst) orelse return;
        const res_ty = self.func.valueType(result);
        switch (self.func.opcode(inst)) {
            .iconst, .fconst => {}, // already hoisted
            .arith => |a| {
                const opcode = try arithOpcode(self.func.types.type_kind(res_ty), a.op);
                const ty = try self.typeId(res_ty);
                const lhs = self.idFor(a.lhs);
                const rhs = self.idFor(a.rhs);
                try self.emit(&self.body, opcode, &.{ ty, self.idFor(result), lhs, rhs });
            },
            .convert => |c| {
                const from = self.func.types.type_kind(self.func.valueType(c.value));
                const to = self.func.types.type_kind(res_ty);
                const opcode = try convertOpcode(from, to);
                const ty = try self.typeId(res_ty);
                const val = self.idFor(c.value);
                try self.emit(&self.body, opcode, &.{ ty, self.idFor(result), val });
            },
            .icmp => |c| {
                const operand_kind = self.func.types.type_kind(self.func.valueType(c.lhs));
                const opcode = cmpOpcode(operand_kind, c.op);
                const bool_ty = try self.boolTypeId();
                const lhs = self.idFor(c.lhs);
                const rhs = self.idFor(c.rhs);
                if (self.func.types.type_kind(res_ty) == .bool) {
                    try self.emit(&self.body, opcode, &.{ bool_ty, self.idFor(result), lhs, rhs });
                } else {
                    // Int result: compare to a bool then select 1/0.
                    const cmp_id = self.fresh();
                    try self.emit(&self.body, opcode, &.{ bool_ty, cmp_id, lhs, rhs });
                    const one = try self.intConstId(res_ty, 1);
                    const zero = try self.intConstId(res_ty, 0);
                    try self.emit(&self.body, op.Select, &.{ try self.typeId(res_ty), self.idFor(result), cmp_id, one, zero });
                }
            },
            .select => |s| {
                const ty = try self.typeId(res_ty);
                const cond = self.idFor(s.cond);
                const then = self.idFor(s.then);
                const els = self.idFor(s.@"else");
                try self.emit(&self.body, op.Select, &.{ ty, self.idFor(result), cond, then, els });
            },
            .unary => |u| {
                const ty = try self.typeId(res_ty);
                const val = self.idFor(u.value);
                if (u.op == .reinterpret) {
                    try self.emit(&self.body, op.Bitcast, &.{ ty, self.idFor(result), val });
                } else {
                    const glsl_op: u32 = switch (u.op) {
                        .sqrt => op.Glsl.sqrt,
                        .floor => op.Glsl.floor,
                        .ceil => op.Glsl.ceil,
                        .trunc => op.Glsl.trunc,
                        .nearest => op.Glsl.round_even,
                        .reinterpret => unreachable,
                    };
                    const ext = try self.glslExtId();
                    try self.emit(&self.body, op.ExtInst, &.{ ty, self.idFor(result), ext, glsl_op, val });
                }
            },
            .call => |c| {
                const name = self.func.symbolName(c.symbol);
                const args = self.func.valueList(c.args);
                // A `tex.sample.<binding>` call samples a sampled-image: construct the uv
                // vector, load the sampler, OpImageSampleImplicitLod -> a vec4 result.
                if (std.mem.startsWith(u8, name, "tex.sample.")) {
                    const binding = std.fmt.parseInt(usize, name["tex.sample.".len..], 10) catch return error.UnsupportedConstruct;
                    if (binding >= self.sampler_vars.len) return error.UnsupportedConstruct;
                    const f32_ty = try self.typeId(self.func.valueType(args[0]));
                    const v2 = try self.vecTypeId(f32_ty, 2);
                    const v4 = try self.vecTypeId(f32_ty, 4);
                    const uv = self.fresh();
                    try self.emit(&self.body, op.CompositeConstruct, &.{ v2, uv, self.idFor(args[0]), self.idFor(args[1]) });
                    const loaded = self.fresh();
                    try self.emit(&self.body, op.Load, &.{ self.sampled_image_ty, loaded, self.sampler_vars[binding] });
                    try self.emit(&self.body, op.ImageSampleImplicitLod, &.{ v4, self.idFor(result), loaded, uv });
                    return;
                }
                // A `spirv.extract.<i>` call pulls lane i of a composite (the sample's vec4).
                if (std.mem.startsWith(u8, name, "spirv.extract.")) {
                    const lane = std.fmt.parseInt(u32, name["spirv.extract.".len..], 10) catch return error.UnsupportedConstruct;
                    try self.emit(&self.body, op.CompositeExtract, &.{ try self.typeId(res_ty), self.idFor(result), self.idFor(args[0]), lane });
                    return;
                }
                // A `spirv.<name>` call is a core SPIR-V op (fragment derivatives
                // OpDPdx/OpDPdy/OpFwidth). Operands are `resultType, result, args...`.
                if (coreOpOf(name)) |opcode| {
                    var ops: std.ArrayList(u32) = .empty;
                    defer ops.deinit(self.allocator);
                    try ops.append(self.allocator, try self.typeId(res_ty));
                    try ops.append(self.allocator, self.idFor(result));
                    for (args) |a| try ops.append(self.allocator, self.idFor(a));
                    try self.emit(&self.body, opcode, ops.items);
                    return;
                }
                // Otherwise a `glsl.<name>` call is a GLSL.std.450 extended instruction.
                const glsl_op = glslExtOf(name) orelse return error.UnsupportedConstruct;
                var ops: std.ArrayList(u32) = .empty;
                defer ops.deinit(self.allocator);
                try ops.append(self.allocator, try self.typeId(res_ty));
                try ops.append(self.allocator, self.idFor(result));
                try ops.append(self.allocator, try self.glslExtId());
                try ops.append(self.allocator, glsl_op);
                for (args) |a| try ops.append(self.allocator, self.idFor(a));
                try self.emit(&self.body, op.ExtInst, ops.items);
            },
            else => return error.UnsupportedConstruct,
        }
    }

    fn finish(self: *Emitter) Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(self.allocator);
        try out.appendSlice(self.allocator, &.{ binary.magic, 0x00010000, 0, self.next_id, 0 });
        try out.appendSlice(self.allocator, self.caps.items);
        try out.appendSlice(self.allocator, self.ext_imports.items);
        try out.appendSlice(self.allocator, self.preamble.items);
        try out.appendSlice(self.allocator, self.debug.items);
        try out.appendSlice(self.allocator, self.annotations.items);
        try out.appendSlice(self.allocator, self.decls.items);
        try out.appendSlice(self.allocator, self.body.items);
        return out.toOwnedSlice(self.allocator);
    }
};

const Terminator = ir.function.Terminator;
const TypeKind = ir.types.TypeKind;
const BinOp = ir.function.BinOp;
const CmpOp = ir.function.CmpOp;

fn arithOpcode(kind: TypeKind, o: BinOp) Error!u16 {
    if (kind == .bool) return switch (o) {
        .bit_and => op.LogicalAnd,
        .bit_or => op.LogicalOr,
        .bit_xor => op.LogicalNotEqual,
        else => error.UnsupportedConstruct,
    };
    if (kind == .float) return switch (o) {
        .add => op.FAdd,
        .sub => op.FSub,
        .mul => op.FMul,
        .div => op.FDiv,
        .rem => op.FRem,
        else => error.UnsupportedConstruct,
    };
    const signed = kind == .int and kind.int.signedness == .signed;
    return switch (o) {
        .add => op.IAdd,
        .sub => op.ISub,
        .mul => op.IMul,
        .div => if (signed) op.SDiv else op.UDiv,
        .rem => if (signed) op.SRem else op.UMod,
        .bit_and => op.BitwiseAnd,
        .bit_or => op.BitwiseOr,
        .bit_xor => op.BitwiseXor,
        .shl => op.ShiftLeftLogical,
        .shr => if (signed) op.ShiftRightArithmetic else op.ShiftRightLogical,
    };
}

fn cmpOpcode(kind: TypeKind, o: CmpOp) u16 {
    if (kind == .float) return switch (o) {
        .eq => op.FOrdEqual,
        .ne => op.FOrdNotEqual,
        .lt => op.FOrdLessThan,
        .le => op.FOrdLessThanEqual,
        .gt => op.FOrdGreaterThan,
        .ge => op.FOrdGreaterThanEqual,
    };
    const signed = kind == .int and kind.int.signedness == .signed;
    return switch (o) {
        .eq => op.IEqual,
        .ne => op.INotEqual,
        .lt => if (signed) op.SLessThan else op.ULessThan,
        .le => if (signed) op.SLessThanEqual else op.ULessThanEqual,
        .gt => if (signed) op.SGreaterThan else op.UGreaterThan,
        .ge => if (signed) op.SGreaterThanEqual else op.UGreaterThanEqual,
    };
}

/// Map a `glsl.<name>` intrinsic call to its GLSL.std.450 extended-instruction number.
/// Returns null if `name` is not a `glsl.` intrinsic.
fn glslExtOf(name: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, name, "glsl.")) return null;
    const n = name[5..];
    const table = .{
        .{ "sin", op.Glsl.sin },   .{ "cos", op.Glsl.cos },   .{ "tan", op.Glsl.tan },
        .{ "asin", op.Glsl.asin }, .{ "acos", op.Glsl.acos }, .{ "atan", op.Glsl.atan },
        .{ "pow", op.Glsl.pow },   .{ "exp", op.Glsl.exp },   .{ "log", op.Glsl.log },
        .{ "exp2", op.Glsl.exp2 }, .{ "log2", op.Glsl.log2 }, .{ "inversesqrt", op.Glsl.inverse_sqrt },
    };
    inline for (table) |e| if (std.mem.eql(u8, n, e[0])) return e[1];
    return null;
}

/// Map a `spirv.<name>` intrinsic call to a core SPIR-V opcode (the fragment derivatives).
fn coreOpOf(name: []const u8) ?u16 {
    if (!std.mem.startsWith(u8, name, "spirv.")) return null;
    const n = name[6..];
    if (std.mem.eql(u8, n, "dpdx")) return op.DPdx;
    if (std.mem.eql(u8, n, "dpdy")) return op.DPdy;
    if (std.mem.eql(u8, n, "fwidth")) return op.Fwidth;
    return null;
}

fn convertOpcode(from: TypeKind, to: TypeKind) Error!u16 {
    if (from == .float and to != .float) return if (to.int.signedness == .signed) op.ConvertFToS else op.ConvertFToU;
    if (from != .float and to == .float) return if (from.int.signedness == .signed) op.ConvertSToF else op.ConvertUToF;
    return error.UnsupportedConstruct;
}

const testing = std.testing;

test "emits a scalar function and reads it back (round-trip through the frontend)" {
    const allocator = testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // f(x, y) = x * y + x
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(i32k);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const xy = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = xy, .rhs = x } });
    func.setTerminator(b, .{ .ret = r });

    const words = try emitModule(allocator, &func, "f");
    defer allocator.free(words);

    // Header is valid SPIR-V (Reader.init validates magic/version).
    _ = try binary.Reader.init(words);
    try testing.expectEqual(binary.magic, words[0]);

    // The frontend reads it back to an equivalent IR function.
    var back = try @import("lower.zig").lowerModule(allocator, words);
    defer back.deinit();
    try testing.expectFmt(
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 * v1
        \\    let v3 = v2 + v0
        \\    ret v3
        \\}
    , "{f}", .{back});
}

test "emits a fragment shader entry point (in -> out)" {
    const allocator = testing.allocator;
    // out = in * 2.0 + 1.0
    var func = Function.init(allocator);
    defer func.deinit();
    const f32t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, f32t);
    const two = try func.appendInst(b, f32t, .{ .fconst = 2.0 });
    const xm = try func.appendInst(b, f32t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = two } });
    const one = try func.appendInst(b, f32t, .{ .fconst = 1.0 });
    const y = try func.appendInst(b, f32t, .{ .arith = .{ .op = .add, .lhs = xm, .rhs = one } });
    func.setTerminator(b, .{ .ret = y });

    const words = try emitShader(allocator, &func, .{
        .stage = .fragment,
        .inputs = &.{.{ .location = 0, .components = 1 }},
        .output = .{ .location = 0, .components = &.{y} },
    });
    defer allocator.free(words);

    // Walk the module and confirm the entry-point structure.
    var reader = try binary.Reader.init(words);
    var saw_entry = false;
    var input_vars: u32 = 0;
    var output_vars: u32 = 0;
    var saw_origin = false;
    while (try reader.next()) |inst| {
        switch (inst.opcode) {
            op.EntryPoint => {
                saw_entry = true;
                try testing.expectEqual(@as(u32, op.ExecutionModel.fragment), inst.operands[0]);
            },
            op.ExecutionMode => if (inst.operands[1] == op.ExecutionModeKind.origin_upper_left) {
                saw_origin = true;
            },
            op.Variable => switch (inst.operands[2]) {
                op.StorageClass.input => input_vars += 1,
                op.StorageClass.output => output_vars += 1,
                else => {},
            },
            else => {},
        }
    }
    try testing.expect(saw_entry);
    try testing.expect(saw_origin);
    try testing.expectEqual(@as(u32, 1), input_vars);
    try testing.expectEqual(@as(u32, 1), output_vars);
}
