//! Lower a parsed GLSL module to Vulcan IR. Handles scalar functions (float/int/uint/
//! bool) and float vectors (vec2/vec3/vec4), scalarized to per-component scalar values as
//! the SPIR-V frontend does (no backend vector support needed): a `vecN` value is N scalar
//! IR values, vector arithmetic is component-wise, swizzles select components, `dot` is a
//! sum of products. Bodies flow in SSA form (assignment rebinds a name), so no allocas.

const std = @import("std");
const ir = @import("vulcan-ir");
const parser = @import("parser.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = parser.Type;

pub const Error = parser.Error || error{ Unsupported, TypeMismatch, UndefinedName, MissingMain, TooManyOutputs, BadSwizzle } || std.mem.Allocator.Error;

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
    var p = try parser.Parser.init(arena.allocator(), source);
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
        const name = try allocator.dupe(u8, fn_ast.name);
        errdefer allocator.free(name);
        var func = try lowerFunction(allocator, fn_ast);
        errdefer func.deinit();
        try list.append(allocator, .{ .name = name, .func = func });
    }
    return .{ .functions = try list.toOwnedSlice(allocator) };
}

/// An input interface variable: its location (or a SPIR-V BuiltIn number when `builtin`
/// is set, e.g. gl_FragCoord) and component count (1 scalar, 2..4 vector).
pub const ShaderVar = struct { location: u32, components: u8, builtin: ?u32 = null };

/// The output interface variable: its location (or a builtin, e.g. gl_Position) and the
/// scalarized component IR values. `builtin` is the SPIR-V BuiltIn number (0 = Position)
/// when the output is a builtin rather than a located varying.
pub const ShaderOutput = struct { location: u32, comps: []Value, builtin: ?u32 = null };

pub const ShaderInterface = struct {
    inputs: []ShaderVar,
    output: ?ShaderOutput,
    local_size: [3]u32,
    uniform_count: u32 = 0,
    sampler_count: u32 = 0,
};

pub const LoweredShader = struct {
    func: Function,
    interface: ShaderInterface,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LoweredShader) void {
        self.func.deinit();
        self.allocator.free(self.interface.inputs);
        if (self.interface.output) |o| self.allocator.free(o.comps);
    }
};

/// Compile a GLSL shader to a Vulcan IR function plus its interface. `in` globals become
/// the function parameters, the single `out` global the returned value.
pub fn compileShader(allocator: std.mem.Allocator, source: []const u8) Error!LoweredShader {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var p = try parser.Parser.init(arena.allocator(), source);
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

    var l = L{ .func = &func, .block = entry, .env = &env, .allocator = allocator };
    defer l.loops.deinit(allocator);
    defer l.samplers.deinit(allocator);

    // `in` globals -> parameters. A vector input is one variable over N scalar params.
    for (ast.globals) |g| {
        if (g.qualifier != .in_) continue;
        const n = vecLen(g.ty);
        if (n == 0) {
            const pv = try func.appendBlockParam(entry, try irType(&func, g.ty));
            try env.append(allocator, .{ .name = g.name, .val = .{ .scalar = .{ .value = pv, .ty = g.ty } } });
            try inputs.append(allocator, .{ .location = g.location orelse 0, .components = 1 });
        } else {
            const f32t = try func.types.intern(.{ .float = .f32 });
            var comps: [4]Value = undefined;
            for (0..n) |i| comps[i] = try func.appendBlockParam(entry, f32t);
            try env.append(allocator, .{ .name = g.name, .val = .{ .vector = .{ .comps = comps, .len = n } } });
            try inputs.append(allocator, .{ .location = g.location orelse 0, .components = n });
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
    if (bodyReferences(main.body, "gl_VertexIndex")) {
        const pv = try func.appendBlockParam(entry, try irType(&func, .int));
        try env.append(allocator, .{ .name = "gl_VertexIndex", .val = .{ .scalar = .{ .value = pv, .ty = .int } } });
        try inputs.append(allocator, .{ .location = 0, .components = 1, .builtin = 42 }); // VertexIndex
    }

    // `uniform` globals become a push-constant block of floats (a vector/matrix uniform is
    // its scalarized components), appended as parameters after the inputs.
    var uniform_count: u32 = 0;
    for (ast.globals) |g| {
        if (g.qualifier != .uniform) continue;
        // An opaque `sampler2D` is a separate descriptor, not a push-constant float. Its
        // declaration order is its binding (see `L.samplerBinding`).
        if (g.ty == .sampler2d) {
            try l.samplers.append(allocator, g.name);
            continue;
        }
        const f32t = try func.types.intern(.{ .float = .f32 });
        const md = matDim(g.ty);
        const n = vecLen(g.ty);
        if (md != 0) {
            var comps: [16]Value = undefined;
            for (0..@as(usize, md) * md) |i| comps[i] = try func.appendBlockParam(entry, f32t);
            try env.append(allocator, .{ .name = g.name, .val = .{ .matrix = .{ .comps = comps, .dim = md } } });
            uniform_count += @intCast(@as(usize, md) * md);
        } else if (n != 0) {
            var comps: [4]Value = undefined;
            for (0..n) |i| comps[i] = try func.appendBlockParam(entry, f32t);
            try env.append(allocator, .{ .name = g.name, .val = .{ .vector = .{ .comps = comps, .len = n } } });
            uniform_count += n;
        } else {
            const pv = try func.appendBlockParam(entry, f32t);
            try env.append(allocator, .{ .name = g.name, .val = .{ .scalar = .{ .value = pv, .ty = .float } } });
            uniform_count += 1;
        }
    }

    // The single `out` global: a pre-declared assignable slot (scalar or vector of zeros).
    var out_name: ?[]const u8 = null;
    var out_location: u32 = 0;
    for (ast.globals) |g| {
        if (g.qualifier != .out_) continue;
        if (out_name != null) return error.TooManyOutputs;
        out_name = g.name;
        out_location = g.location orelse 0;
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

    for (main.body) |stmt| _ = try lowerStmt(&l, .void, stmt);

    var output: ?ShaderOutput = null;
    if (uses_gl_pos) {
        const slot = l.lookup("gl_Position").?;
        output = .{ .location = 0, .comps = try allocator.dupe(Value, slot.val.vector.comps[0..4]), .builtin = 0 }; // BuiltIn Position
    } else if (out_name) |name| {
        const slot = l.lookup(name).?;
        const comps: []Value = switch (slot.val) {
            .scalar => |s| try allocator.dupe(Value, &.{s.value}),
            .vector => |vec| try allocator.dupe(Value, vec.comps[0..vec.len]),
            .matrix => return error.Unsupported, // matrix shader outputs are uncommon
        };
        output = .{ .location = out_location, .comps = comps };
    }
    func.setTerminator(entry, .{ .ret = null }); // outputs flow through stored variables

    return .{
        .func = func,
        .interface = .{ .inputs = try inputs.toOwnedSlice(allocator), .output = output, .local_size = ast.local_size orelse .{ 1, 1, 1 }, .uniform_count = uniform_count, .sampler_count = @intCast(l.samplers.items.len) },
        .allocator = allocator,
    };
}

/// A scalar value with its GLSL type (float/int/uint/bool).
const Scalar = struct { value: Value, ty: Type };

/// A scalarized float vector: its component values (length 2..4).
const Vector = struct { comps: [4]Value, len: u8 };

/// A scalarized float square matrix, column-major: comps[col*dim + row], dim in 2..4.
const Matrix = struct { comps: [16]Value, dim: u8 };

/// A lowered GLSL value: a scalar, a (float) vector, or a (float) matrix.
const Val = union(enum) {
    scalar: Scalar,
    vector: Vector,
    matrix: Matrix,
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

const L = struct {
    func: *Function,
    block: Block,
    env: *std.ArrayList(Var),
    allocator: std.mem.Allocator,
    loops: std.ArrayList(LoopCtx) = .empty,
    /// Declared `uniform sampler2D` names, in declaration order. The index is the SPIR-V
    /// binding. `texture(name, uv)` resolves `name` to its binding here.
    samplers: std.ArrayList([]const u8) = .empty,

    /// Binding index of sampler `name`, or null if it is not a declared sampler.
    fn samplerBinding(self: *L, name: []const u8) ?u32 {
        for (self.samplers.items, 0..) |s, i| if (std.mem.eql(u8, s, name)) return @intCast(i);
        return null;
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

/// Component count of a vector type, or 0 for a scalar.
fn vecLen(ty: Type) u8 {
    return switch (ty) {
        .vec2 => 2,
        .vec3 => 3,
        .vec4 => 4,
        else => 0,
    };
}

fn irType(func: *Function, ty: Type) Error!ir.types.Type {
    return switch (ty) {
        .float => func.types.intern(.{ .float = .f32 }),
        .int => func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }),
        .uint => func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } }),
        .bool => func.types.intern(.bool),
        .void, .vec2, .vec3, .vec4, .mat2, .mat3, .mat4, .sampler2d => error.Unsupported,
    };
}

fn f32Type(l: *L) Error!ir.types.Type {
    return l.func.types.intern(.{ .float = .f32 });
}

fn lowerFunction(allocator: std.mem.Allocator, f: parser.Function) Error!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const entry = try func.appendBlock();

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
            // A vector parameter is scalarized to one float parameter per component.
            const f32t = try func.types.intern(.{ .float = .f32 });
            var comps: [4]Value = undefined;
            for (0..n) |i| comps[i] = try func.appendBlockParam(entry, f32t);
            try env.append(allocator, .{ .name = param.name, .val = .{ .vector = .{ .comps = comps, .len = n } } });
        }
    }

    var l = L{ .func = &func, .block = entry, .env = &env, .allocator = allocator };
    defer l.loops.deinit(allocator);
    defer l.samplers.deinit(allocator);
    var returned = false;
    for (f.body) |stmt| {
        if (try lowerStmt(&l, f.ret, stmt)) returned = true;
    }
    if (!returned) func.setTerminator(l.block, .{ .ret = null });
    return func;
}

/// Lower a statement. Returns true if it terminated the block.
fn lowerStmt(l: *L, ret_ty: Type, stmt: parser.Stmt) Error!bool {
    switch (stmt) {
        .ret => |maybe| {
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
                const stored = try coerceVal(l, try lowerExpr(l, e), d.ty);
                try l.env.append(l.allocator, .{ .name = d.name, .val = stored });
            } else {
                try l.env.append(l.allocator, .{ .name = d.name, .val = .{ .scalar = .{ .value = try zero(l, d.ty), .ty = d.ty } } });
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
            if (slot.val != .vector) return error.BadSwizzle;
            var vec = slot.val.vector;
            const v = try lowerExpr(l, sa.value);
            if (sa.field.len == 1) {
                if (v != .scalar) return error.TypeMismatch;
                const idx = swizzleIndex(sa.field[0]) orelse return error.BadSwizzle;
                if (idx >= vec.len) return error.BadSwizzle;
                vec.comps[idx] = (try coerce(l, v.scalar, .float)).value;
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
        .if_ => |iff| return lowerIf(l, ret_ty, iff.cond, iff.then, iff.@"else"),
        .for_ => |f| return lowerFor(l, ret_ty, f.init, f.cond, f.incr, f.body),
    }
}

/// Lower `for (init, cond, incr) body` (a `while` is the same with empty init/incr) as a
/// header/body/exit loop. The header takes a block parameter (loop phi) for every live
/// variable: the preheader edge passes the initial value, the body's back-edge the
/// updated value.
fn lowerFor(l: *L, ret_ty: Type, init: []const parser.Stmt, cond_e: ?*parser.Expr, incr: []const parser.Stmt, body: []const parser.Stmt) Error!bool {
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

/// Whether a loop body contains a `break` for this loop. Recurses into `if` branches but
/// not into nested loops, whose `break` targets them.
fn hasBreak(body: []const parser.Stmt) bool {
    for (body) |stmt| switch (stmt) {
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
        .scalar => |s| return .{ .scalar = .{ .value = try l.func.appendBlockParam(header, try irType(l.func, s.ty)), .ty = s.ty } },
        .vector => |vec| {
            const f32t = try f32Type(l);
            var comps: [4]Value = undefined;
            for (0..vec.len) |i| comps[i] = try l.func.appendBlockParam(header, f32t);
            return .{ .vector = .{ .comps = comps, .len = vec.len } };
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
        const f32t = try f32Type(l);
        var out: [4]Value = undefined;
        for (0..tv.len) |i| {
            out[i] = try l.func.appendBlockParam(cont, f32t);
            try then_args.append(l.allocator, tv.comps[i]);
            try else_args.append(l.allocator, ev.comps[i]);
        }
        return .{ .vector = .{ .comps = out, .len = tv.len } };
    }
    return error.TypeMismatch;
}

/// GLSL type a Val represents (for assignment-target coercion).
fn valType(v: Val) Type {
    return switch (v) {
        .scalar => |s| s.ty,
        .vector => |vec| switch (vec.len) {
            2 => .vec2,
            3 => .vec3,
            else => .vec4,
        },
        .matrix => |m| switch (m.dim) {
            2 => .mat2,
            3 => .mat3,
            else => .mat4,
        },
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
        },
        .binary => |b| return lowerBinary(l, b.op, b.lhs, b.rhs),
        .call => |c| return lowerCall(l, c.name, c.args),
        .swizzle => |s| return lowerSwizzle(l, s.value, s.field),
        .ternary => |t| return lowerTernary(l, t.cond, t.then, t.@"else"),
    }
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
        const f32t = try f32Type(l);
        var out: Vector = .{ .comps = undefined, .len = then.vector.len };
        for (0..then.vector.len) |i| out.comps[i] = try l.func.appendInst(l.block, f32t, .{ .select = .{ .cond = c, .then = then.vector.comps[i], .@"else" = els.vector.comps[i] } });
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
        .matrix => return error.Unsupported,
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

    // Vector arithmetic is component-wise. A scalar operand broadcasts to each lane.
    const arith_op = binArith(op) orelse return error.Unsupported; // no vector comparisons yet
    const f32t = try f32Type(l);
    const len = if (a == .vector) a.vector.len else b.vector.len;
    if (a == .vector and b == .vector and a.vector.len != b.vector.len) return error.TypeMismatch;
    var out: Vector = .{ .comps = undefined, .len = len };
    for (0..len) |i| {
        const av = try laneValue(l, a, i);
        const bv = try laneValue(l, b, i);
        out.comps[i] = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = arith_op, .lhs = av, .rhs = bv } });
    }
    return .{ .vector = out };
}

/// Component `i` of a value: a vector's lane, or a scalar broadcast (coerced to float) to
/// every lane.
fn laneValue(l: *L, v: Val, i: usize) Error!Value {
    switch (v) {
        .vector => |vec| return vec.comps[i],
        .scalar => |s| return (try coerce(l, s, .float)).value,
        .matrix => return error.Unsupported,
    }
}

fn binArith(op: parser.BinOp) ?ir.function.BinOp {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
        .mod => .rem,
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
    if (ctor_len != 0) return constructVector(l, ctor_len, args);
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
    if (std.mem.eql(u8, name, "texture")) {
        if (args.len != 2) return error.Unsupported;
        // The first argument names a declared sampler (not a value to lower).
        const sampler_name = switch (args[0].*) {
            .ident => |id| id,
            else => return error.Unsupported,
        };
        const binding = l.samplerBinding(sampler_name) orelse return error.UndefinedName;
        const uv = try lowerExpr(l, args[1]);
        if (uv != .vector or uv.vector.len != 2) return error.TypeMismatch;
        return lowerTexture(l, binding, uv.vector);
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
    return error.Unsupported;
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
    const cmp: ir.function.CmpOp = if (want_max) .gt else .lt;
    const cond = try l.func.appendInst(l.block, try irType(l.func, .bool), .{ .icmp = .{ .op = cmp, .lhs = x, .rhs = y } });
    return l.func.appendInst(l.block, try irType(l.func, ty), .{ .select = .{ .cond = cond, .then = x, .@"else" = y } });
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
    var out: [4]Value = undefined;
    for (0..len) |i| out[i] = try minMaxScalar(l, try laneValue(l, a, i), try laneValue(l, b, i), .float, want_max);
    return .{ .vector = .{ .comps = out, .len = len } };
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
            for (0..vec.len) |i| out[i] = try absScalar(l, vec.comps[i], .float);
            return .{ .vector = .{ .comps = out, .len = vec.len } };
        },
        .matrix => return error.Unsupported,
    }
}

/// `mix(a, b, t)` = `a + (b - a) * t`, component-wise (a scalar `t` broadcasts).
fn lowerMix(l: *L, a: Val, b: Val, t: Val) Error!Val {
    const f32t = try f32Type(l);
    const is_vec = (a == .vector or b == .vector);
    const len = maxLen(a, b);
    if (a == .vector and b == .vector and a.vector.len != b.vector.len) return error.TypeMismatch;
    var out: [4]Value = undefined;
    for (0..len) |i| {
        const ai = try laneValue(l, a, i);
        const bi = try laneValue(l, b, i);
        const ti = try laneValue(l, t, i);
        const diff = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .sub, .lhs = bi, .rhs = ai } });
        const scaled = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .mul, .lhs = diff, .rhs = ti } });
        out[i] = try l.func.appendInst(l.block, f32t, .{ .arith = .{ .op = .add, .lhs = ai, .rhs = scaled } });
    }
    if (!is_vec) return .{ .scalar = .{ .value = out[0], .ty = .float } };
    return .{ .vector = .{ .comps = out, .len = len } };
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
        .matrix => return error.Unsupported,
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
    for (0..len) |i| out[i] = try f(l, try laneValue(l, a, i), try laneValue(l, b, i));
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
        const a0 = try laneValue(l, e0, i);
        const a1 = try laneValue(l, e1, i);
        const xi = try laneValue(l, x, i);
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
        .matrix => return error.Unsupported,
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
        .matrix => return error.Unsupported,
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
            .matrix => return error.Unsupported,
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

fn constructVector(l: *L, len: u8, args: []const *parser.Expr) Error!Val {
    var comps: [4]Value = undefined;
    var n: u8 = 0;
    for (args) |arg| {
        const v = try lowerExpr(l, arg);
        switch (v) {
            .scalar => |s| {
                if (n >= 4) return error.Unsupported;
                comps[n] = (try coerce(l, s, .float)).value;
                n += 1;
            },
            .vector => |vec| {
                for (0..vec.len) |i| {
                    if (n >= 4) return error.Unsupported;
                    comps[n] = vec.comps[i];
                    n += 1;
                }
            },
            .matrix => return error.Unsupported,
        }
    }
    if (n == 1 and len > 1) {
        // Splat: vec3(x) -> (x, x, x).
        for (1..len) |i| comps[i] = comps[0];
        n = len;
    }
    if (n != len) return error.TypeMismatch;
    return .{ .vector = .{ .comps = comps, .len = len } };
}

fn lowerSwizzle(l: *L, value: *parser.Expr, field: []const u8) Error!Val {
    const v = try lowerExpr(l, value);
    if (v != .vector) return error.BadSwizzle;
    if (field.len == 0 or field.len > 4) return error.BadSwizzle;
    var comps: [4]Value = undefined;
    for (field, 0..) |ch, i| {
        const idx = swizzleIndex(ch) orelse return error.BadSwizzle;
        if (idx >= v.vector.len) return error.BadSwizzle;
        comps[i] = v.vector.comps[idx];
    }
    if (field.len == 1) return .{ .scalar = .{ .value = comps[0], .ty = .float } };
    return .{ .vector = .{ .comps = comps, .len = @intCast(field.len) } };
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
    return .{ .value = s.value, .ty = to };
}

fn zero(l: *L, ty: Type) Error!Value {
    return switch (ty) {
        .float => l.func.appendInst(l.block, try irType(l.func, ty), .{ .fconst = 0 }),
        else => l.func.appendInst(l.block, try irType(l.func, ty), .{ .iconst = 0 }),
    };
}

fn isInt(ty: Type) bool {
    return ty == .int or ty == .uint;
}

/// Whether any top-level statement assigns to `name` (detects a `gl_Position` write).
fn assignsName(body: []const parser.Stmt, name: []const u8) bool {
    for (body) |stmt| if (stmt == .assign and std.mem.eql(u8, stmt.assign.name, name)) return true;
    return false;
}

/// Whether any expression in the body reads `name` (used to detect input builtins).
fn bodyReferences(body: []const parser.Stmt, name: []const u8) bool {
    for (body) |stmt| switch (stmt) {
        .ret => |m| if (m) |e| {
            if (exprReferences(e, name)) return true;
        },
        .decl => |d| if (d.value) |e| {
            if (exprReferences(e, name)) return true;
        },
        .assign => |a| if (exprReferences(a.value, name)) return true,
        .swizzle_assign => |sa| if (exprReferences(sa.value, name)) return true,
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
        .float_lit, .int_lit, .bool_lit => false,
    };
}
