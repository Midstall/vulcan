//! Emit portable C99 from a Vulcan IR function: a source-level backend. Each SSA
//! value becomes a C variable declared once at the top of the function, blocks become
//! labels, block parameters are assigned across control-flow edges (parallel copies via
//! temporaries), and the non-terminating high-profile `if` lowers to a C `if/else` with
//! `goto` edges. The output compiles with any C99 compiler, so it doubles as a
//! portability path and a differential oracle against the native backends.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Type = ir.types.Type;
const TypeTable = ir.types.TypeTable;

pub const Error = error{Unsupported} || std.mem.Allocator.Error;

/// Emit `func` as a single C function named `name`. Caller owns the returned source.
pub fn emitFunction(allocator: std.mem.Allocator, func: *const Function, name: []const u8) Error![]u8 {
    var e = Emitter{ .allocator = allocator, .func = func };
    defer e.deinit();
    try e.run(name);
    return e.out.toOwnedSlice(allocator);
}

/// One named function in a module.
pub const NamedFunc = struct { name: []const u8, func: *const Function };

/// Emit a whole module: a forward prototype for every function, then every body, so the
/// functions may call one another in any order. Caller owns the returned source.
pub fn emitModule(allocator: std.mem.Allocator, funcs: []const NamedFunc) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    // Each function namespaces its aggregate typedefs with a `{name}_` prefix, so the
    // typedefs of two functions never clash even for identical layouts.
    // 1) Every function's aggregate typedefs, once, at the very top.
    for (funcs) |nf| {
        const prefix = try std.fmt.allocPrint(allocator, "{s}_", .{nf.name});
        defer allocator.free(prefix);
        var e = Emitter{ .allocator = allocator, .func = nf.func, .prefix = prefix };
        defer e.deinit();
        try e.collectAggs();
        try e.emitTypedefs();
        try out.appendSlice(allocator, e.out.items);
    }
    // 2) Forward prototypes, so functions may call one another in any order.
    for (funcs) |nf| {
        const prefix = try std.fmt.allocPrint(allocator, "{s}_", .{nf.name});
        defer allocator.free(prefix);
        var e = Emitter{ .allocator = allocator, .func = nf.func, .prefix = prefix };
        defer e.deinit();
        try e.assignNames();
        try e.collectAggs();
        try e.signature(nf.name);
        try e.w(";\n");
        try out.appendSlice(allocator, e.out.items);
    }
    try out.append(allocator, '\n');
    // 3) Bodies. Typedefs already emitted above, so bodies only reference them.
    for (funcs) |nf| {
        const prefix = try std.fmt.allocPrint(allocator, "{s}_", .{nf.name});
        defer allocator.free(prefix);
        var e = Emitter{ .allocator = allocator, .func = nf.func, .prefix = prefix, .emit_typedefs = false };
        defer e.deinit();
        try e.run(nf.name);
        try out.appendSlice(allocator, e.out.items);
    }
    return out.toOwnedSlice(allocator);
}

const Emitter = struct {
    allocator: std.mem.Allocator,
    func: *const Function,
    out: std.ArrayList(u8) = .empty,
    /// value index (`@intFromEnum`) -> textual name number, matching the IR printer's
    /// deterministic block-by-block walk so the C reads like the IR.
    names: []u32 = &.{},
    /// block index -> whether any edge targets it (so it needs a `goto` label).
    is_target: []bool = &.{},
    /// Aggregate (struct/vector) types used, in dependency order. Each gets a C typedef
    /// named `{prefix}agg{index}`, so `emitType` resolves an aggregate to a stable name.
    aggs: std.ArrayList(Type) = .empty,
    /// Prefix for typedef names, so a module can namespace each function's aggregates.
    prefix: []const u8 = "",
    /// Whether `run` emits the aggregate typedefs itself. A module emits them once up top
    /// and sets this false for the bodies.
    emit_typedefs: bool = true,

    fn deinit(self: *Emitter) void {
        self.out.deinit(self.allocator);
        self.allocator.free(self.names);
        self.allocator.free(self.is_target);
        self.aggs.deinit(self.allocator);
    }

    fn w(self: *Emitter, bytes: []const u8) Error!void {
        try self.out.appendSlice(self.allocator, bytes);
    }

    fn print(self: *Emitter, comptime fmt: []const u8, args: anytype) Error!void {
        try self.out.print(self.allocator, fmt, args);
    }

    /// The C variable name of a value, e.g. `v3`.
    fn name(self: *Emitter, v: Value) u32 {
        return self.names[@intFromEnum(v)];
    }

    /// Emit the C signature `<ret> sym(<params>)` (no trailing brace or semicolon).
    /// Requires `assignNames` to have run.
    fn signature(self: *Emitter, sym: []const u8) Error!void {
        const func = self.func;
        const entry_params = func.blockParams(@as(Block, @enumFromInt(0)));
        // The return type is the type of the first `ret` value, void otherwise.
        try self.emitType(self.returnType());
        try self.print(" {s}(", .{sym});
        if (entry_params.len == 0) {
            try self.w("void");
        } else {
            for (entry_params, 0..) |p, i| {
                if (i != 0) try self.w(", ");
                try self.emitType(func.valueType(p));
                try self.print(" v{d}", .{self.name(p)});
            }
        }
        try self.w(")");
    }

    fn run(self: *Emitter, sym: []const u8) Error!void {
        try self.assignNames();
        try self.findTargets();
        try self.collectAggs();
        if (self.emit_typedefs) try self.emitTypedefs();
        try self.emitGlobalExterns();

        const func = self.func;
        try self.signature(sym);
        try self.w(" {\n");

        // Declare every value that is not an entry-block parameter (those are the C
        // function parameters) at the top, so `goto` never jumps over a declaration.
        try self.emitDeclarations();

        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(@as(u32, @intCast(bi)));
            // A label only for blocks some edge jumps to; an empty statement follows so it
            // stays legal even for an empty block.
            if (self.is_target[bi]) try self.print("block{d}:;\n", .{bi});
            var branched = false;
            for (func.blockInsts(block)) |inst| {
                if (func.opcode(inst) == .@"if") branched = true;
                try self.emitInst(inst);
            }
            if (!branched) try self.emitTerminator(block);
        }

        try self.w("}\n");
    }

    /// Build the value-name map with the same walk the IR printer uses.
    fn assignNames(self: *Emitter) Error!void {
        const func = self.func;
        self.names = try self.allocator.alloc(u32, func.valueCount());
        var n: u32 = 0;
        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(@as(u32, @intCast(bi)));
            for (func.blockParams(block)) |p| {
                self.names[@intFromEnum(p)] = n;
                n += 1;
            }
            for (func.blockInsts(block)) |inst| {
                if (func.instResult(inst)) |res| {
                    self.names[@intFromEnum(res)] = n;
                    n += 1;
                }
            }
        }
    }

    /// Mark blocks that are the target of some `if` edge or `jump`, so only they get a label.
    fn findTargets(self: *Emitter) Error!void {
        const func = self.func;
        self.is_target = try self.allocator.alloc(bool, func.blockCount());
        @memset(self.is_target, false);
        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(@as(u32, @intCast(bi)));
            for (func.blockInsts(block)) |inst| {
                if (func.opcode(inst) == .@"if") {
                    const cf = func.opcode(inst).@"if";
                    self.is_target[@intFromEnum(cf.then.target)] = true;
                    self.is_target[@intFromEnum(cf.@"else".target)] = true;
                }
            }
            if (func.terminator(block)) |term| switch (term) {
                .jump => |j| self.is_target[@intFromEnum(j.target)] = true,
                .ret => {},
            };
        }
    }

    /// Register every aggregate type used by a value, in dependency order (a struct's
    /// aggregate fields are registered before the struct itself), so the typedefs compile.
    /// Array element types reach the IR only through an `alloca`, so scan those too.
    fn collectAggs(self: *Emitter) Error!void {
        const func = self.func;
        for (0..func.valueCount()) |vi| {
            try self.registerAgg(func.valueType(@enumFromInt(vi)));
        }
        for (0..func.instCount()) |ii| {
            const inst: Inst = @enumFromInt(@as(u32, @intCast(ii)));
            if (func.opcode(inst) == .alloca) try self.registerAgg(func.opcode(inst).alloca.elem);
        }
    }

    fn registerAgg(self: *Emitter, ty: Type) Error!void {
        switch (self.func.types.type_kind(ty)) {
            .@"struct", .vector, .array, .slice => {},
            else => return,
        }
        for (self.aggs.items) |existing| if (existing == ty) return;
        // Register aggregate members first so their typedefs precede this one. A slice is a
        // flat `{ptr, len}` pair, so its element type needs no typedef of its own.
        switch (self.func.types.type_kind(ty)) {
            .@"struct" => |fields| for (fields) |f| try self.registerAgg(f),
            .vector => |v| try self.registerAgg(v.elem),
            .array => |a| try self.registerAgg(a.elem),
            else => {},
        }
        try self.aggs.append(self.allocator, ty);
    }

    /// Declare each global referenced by a `global_addr` as `extern char <sym>[];`, so
    /// taking its address is legal. Redundant identical externs across a module are fine.
    fn emitGlobalExterns(self: *Emitter) Error!void {
        const func = self.func;
        var any = false;
        var seen: std.ArrayList([]const u8) = .empty;
        defer seen.deinit(self.allocator);
        for (0..func.instCount()) |ii| {
            const inst: Inst = @enumFromInt(@as(u32, @intCast(ii)));
            if (func.opcode(inst) != .global_addr) continue;
            const nm = func.symbolName(func.opcode(inst).global_addr.symbol);
            var dup = false;
            for (seen.items) |s| if (std.mem.eql(u8, s, nm)) {
                dup = true;
            };
            if (dup) continue;
            try seen.append(self.allocator, nm);
            try self.print("extern char {s}[];\n", .{nm});
            any = true;
        }
        if (any) try self.w("\n");
    }

    /// The typedef name of an aggregate type (must be registered).
    fn aggName(self: *Emitter, out: *std.ArrayList(u8), ty: Type) Error!void {
        for (self.aggs.items, 0..) |existing, i| {
            if (existing == ty) {
                try out.print(self.allocator, "{s}agg{d}", .{ self.prefix, i });
                return;
            }
        }
        return error.Unsupported;
    }

    /// Emit a C `typedef struct { ... }` for each registered aggregate. A struct keeps its
    /// field types; a vector becomes N numbered fields of its element type, so `extract` and
    /// `struct_new` treat both uniformly.
    fn emitTypedefs(self: *Emitter) Error!void {
        if (self.aggs.items.len == 0) return;
        for (self.aggs.items, 0..) |ty, i| {
            try self.w("typedef struct { ");
            switch (self.func.types.type_kind(ty)) {
                .@"struct" => |fields| for (fields, 0..) |f, fi| {
                    try self.emitType(f);
                    try self.print(" f{d}; ", .{fi});
                },
                .vector => |v| for (0..v.len) |fi| {
                    try self.emitType(v.elem);
                    try self.print(" f{d}; ", .{fi});
                },
                // A fixed array wraps a C array member `e[N]`, so the struct stays
                // assignable and `&slot` gives the array's base address for an alloca.
                .array => |a| {
                    try self.emitType(a.elem);
                    try self.print(" e[{d}]; ", .{a.len});
                },
                // A slice is a fat pointer: base address plus a runtime length.
                .slice => try self.w("void* f0; int64_t f1; "),
                else => unreachable,
            }
            try self.print("}} {s}agg{d};\n", .{ self.prefix, i });
        }
        try self.w("\n");
    }

    /// The function's return type: the type of any `ret` value, else void (a null type).
    fn returnType(self: *Emitter) ?Type {
        for (0..self.func.blockCount()) |bi| {
            const block: Block = @enumFromInt(@as(u32, @intCast(bi)));
            if (self.func.terminator(block)) |term| switch (term) {
                .ret => |v| if (v) |vv| return self.func.valueType(vv),
                .jump => {},
            };
        }
        return null;
    }

    /// Declare all instruction results and non-entry block parameters.
    fn emitDeclarations(self: *Emitter) Error!void {
        const func = self.func;
        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(@as(u32, @intCast(bi)));
            if (bi != 0) {
                for (func.blockParams(block)) |p| try self.emitDecl(p);
            }
            for (func.blockInsts(block)) |inst| {
                if (func.instResult(inst)) |res| try self.emitDecl(res);
                // An alloca also needs backing storage for the slot it points at.
                if (func.opcode(inst) == .alloca) {
                    const elem = func.opcode(inst).alloca.elem;
                    const res = func.instResult(inst).?;
                    try self.w("    ");
                    try self.emitType(elem);
                    try self.print(" slot{d};\n", .{self.name(res)});
                }
            }
        }
    }

    fn emitDecl(self: *Emitter, v: Value) Error!void {
        try self.w("    ");
        try self.emitType(self.func.valueType(v));
        try self.print(" v{d};\n", .{self.name(v)});
    }

    fn emitInst(self: *Emitter, inst: Inst) Error!void {
        const func = self.func;
        const op = func.opcode(inst);
        const res = func.instResult(inst);
        switch (op) {
            .iconst => |val| try self.print("    v{d} = {d};\n", .{ self.name(res.?), val }),
            .fconst => |val| try self.emitFconst(res.?, val),
            .arith => |a| if (self.vectorLen(func.valueType(res.?))) |n| {
                // An element-wise vector op: one C statement per lane.
                for (0..n) |k| try self.print("    v{d}.f{d} = v{d}.f{d} {s} v{d}.f{d};\n", .{
                    self.name(res.?), k, self.name(a.lhs), k, a.op.symbol(), self.name(a.rhs), k,
                });
            } else try self.print("    v{d} = v{d} {s} v{d};\n", .{
                self.name(res.?), self.name(a.lhs), a.op.symbol(), self.name(a.rhs),
            }),
            .arith_imm => |a| if (self.vectorLen(func.valueType(res.?))) |n| {
                for (0..n) |k| try self.print("    v{d}.f{d} = v{d}.f{d} {s} {d};\n", .{
                    self.name(res.?), k, self.name(a.lhs), k, a.op.symbol(), a.imm,
                });
            } else try self.print("    v{d} = v{d} {s} {d};\n", .{
                self.name(res.?), self.name(a.lhs), a.op.symbol(), a.imm,
            }),
            .icmp => |c| try self.print("    v{d} = (v{d} {s} v{d});\n", .{
                self.name(res.?), self.name(c.lhs), c.op.symbol(), self.name(c.rhs),
            }),
            .select => |s| try self.print("    v{d} = v{d} ? v{d} : v{d};\n", .{
                self.name(res.?), self.name(s.cond), self.name(s.then), self.name(s.@"else"),
            }),
            .@"if" => |cf| {
                try self.print("    if (v{d}) {{\n", .{self.name(cf.cond)});
                try self.emitEdge(cf.then.target, func.blockArgs(cf.then), "        ", true);
                try self.w("    } else {\n");
                try self.emitEdge(cf.@"else".target, func.blockArgs(cf.@"else"), "        ", true);
                try self.w("    }\n");
            },
            .convert => |cv| {
                // A C cast performs the int<->float / width numeric conversion.
                try self.print("    v{d} = (", .{self.name(res.?)});
                try self.emitType(func.valueType(res.?));
                try self.print(")v{d};\n", .{self.name(cv.value)});
            },
            .unary => |u| try self.emitUnary(res.?, u),
            .alloca => try self.print("    v{d} = (void*)&slot{d};\n", .{ self.name(res.?), self.name(res.?) }),
            .load => |ld| {
                try self.print("    v{d} = *(", .{self.name(res.?)});
                try self.emitType(func.valueType(res.?));
                try self.print("*)v{d};\n", .{self.name(ld.ptr)});
            },
            .store => |st| {
                try self.w("    *(");
                try self.emitType(func.valueType(st.value));
                try self.print("*)v{d} = v{d};\n", .{ self.name(st.ptr), self.name(st.value) });
            },
            .struct_new => |sn| {
                // Build the aggregate with a positional compound literal.
                try self.print("    v{d} = (", .{self.name(res.?)});
                try self.emitType(func.valueType(res.?));
                try self.w("){ ");
                for (func.valueList(sn.fields), 0..) |field, i| {
                    if (i != 0) try self.w(", ");
                    try self.print("v{d}", .{self.name(field)});
                }
                try self.w(" };\n");
            },
            .extract => |ex| if (func.types.type_kind(func.valueType(ex.aggregate)) == .array) {
                try self.print("    v{d} = v{d}.e[{d}];\n", .{ self.name(res.?), self.name(ex.aggregate), ex.index });
            } else try self.print("    v{d} = v{d}.f{d};\n", .{
                self.name(res.?), self.name(ex.aggregate), ex.index,
            }),
            .call => |c| {
                try self.w("    ");
                if (res) |r| try self.print("v{d} = ", .{self.name(r)});
                try self.print("{s}(", .{func.symbolName(c.symbol)});
                for (func.valueList(c.args), 0..) |arg, i| {
                    if (i != 0) try self.w(", ");
                    try self.print("v{d}", .{self.name(arg)});
                }
                try self.w(");\n");
            },
            .global_addr => |ga| {
                // The global is declared `extern char <sym>[];` above, so it decays to its
                // base address.
                try self.print("    v{d} = (void*)({s});\n", .{ self.name(res.?), func.symbolName(ga.symbol) });
            },
            .call_indirect => |ci| {
                try self.w("    ");
                if (res) |r| try self.print("v{d} = ", .{self.name(r)});
                // Cast the target pointer to a function pointer of the right signature.
                try self.w("((");
                try self.emitType(if (res) |r| func.valueType(r) else null);
                try self.w(" (*)(");
                const args = func.valueList(ci.args);
                if (args.len == 0) {
                    try self.w("void");
                } else for (args, 0..) |arg, i| {
                    if (i != 0) try self.w(", ");
                    try self.emitType(func.valueType(arg));
                }
                try self.print("))v{d})(", .{self.name(ci.target)});
                for (args, 0..) |arg, i| {
                    if (i != 0) try self.w(", ");
                    try self.print("v{d}", .{self.name(arg)});
                }
                try self.w(");\n");
            },
        }
    }

    /// The lane count if `ty` is a vector, else null.
    fn vectorLen(self: *Emitter, ty: Type) ?u32 {
        return switch (self.func.types.type_kind(ty)) {
            .vector => |v| v.len,
            else => null,
        };
    }

    fn emitUnary(self: *Emitter, res: Value, u: ir.function.Unary) Error!void {
        if (u.op == .reinterpret) {
            // Same-width bit reinterpret (int<->float): copy the bytes, no conversion.
            try self.print("    memcpy(&v{d}, &v{d}, sizeof(v{d}));\n", .{ self.name(res), self.name(u.value), self.name(res) });
            return;
        }
        const is_f32 = switch (self.func.types.type_kind(self.func.valueType(res))) {
            .float => |f| f == .f32,
            else => false,
        };
        // libm has a `f`-suffixed single-precision variant of each of these.
        const fname: []const u8 = switch (u.op) {
            .sqrt => if (is_f32) "sqrtf" else "sqrt",
            .floor => if (is_f32) "floorf" else "floor",
            .ceil => if (is_f32) "ceilf" else "ceil",
            .trunc => if (is_f32) "truncf" else "trunc",
            .nearest => if (is_f32) "rintf" else "rint", // round to nearest, ties to even
            .reinterpret => unreachable,
        };
        try self.print("    v{d} = {s}(v{d});\n", .{ self.name(res), fname, self.name(u.value) });
    }

    fn emitFconst(self: *Emitter, res: Value, val: f64) Error!void {
        // Scientific notation always carries a decimal point/exponent, so it is a valid C
        // floating constant. An f32 result takes the `f` suffix to match its type.
        const is_f32 = switch (self.func.types.type_kind(self.func.valueType(res))) {
            .float => |f| f == .f32,
            else => false,
        };
        if (is_f32) {
            try self.print("    v{d} = {e}f;\n", .{ self.name(res), @as(f32, @floatCast(val)) });
        } else {
            try self.print("    v{d} = {e};\n", .{ self.name(res), val });
        }
    }

    fn emitTerminator(self: *Emitter, block: Block) Error!void {
        const term = self.func.terminator(block) orelse {
            try self.w("    return;\n");
            return;
        };
        switch (term) {
            .ret => |v| if (v) |vv| {
                try self.print("    return v{d};\n", .{self.name(vv)});
            } else {
                try self.w("    return;\n");
            },
            .jump => |j| try self.emitEdge(j.target, self.func.blockArgs(j), "    ", false),
        }
    }

    /// Emit a control-flow edge: assign the target block's parameters from the passed
    /// arguments (through temporaries, so a swap across the edge is correct) and `goto`.
    /// Each emitted line is prefixed with `indent`. When `scoped` the caller already
    /// opened a fresh C block, so the temporaries need no extra brace of their own.
    fn emitEdge(self: *Emitter, target: Block, args: []const Value, indent: []const u8, scoped: bool) Error!void {
        const params = self.func.blockParams(target);
        if (params.len == 0) {
            try self.print("{s}goto block{d};\n", .{ indent, @intFromEnum(target) });
            return;
        }
        // Temporaries need their own scope so they never collide with another
        // predecessor's temporaries for the same target block. An `if`/`else` branch is
        // already such a scope, so only a bare terminator adds a brace.
        const inner = if (scoped) indent else try std.mem.concat(self.allocator, u8, &.{ indent, "    " });
        defer if (!scoped) self.allocator.free(inner);
        if (!scoped) try self.print("{s}{{\n", .{indent});
        for (params, args) |p, arg| {
            try self.print("{s}", .{inner});
            try self.emitType(self.func.valueType(p));
            try self.print(" t{d} = v{d};\n", .{ self.name(p), self.name(arg) });
        }
        for (params) |p| {
            try self.print("{s}v{d} = t{d};\n", .{ inner, self.name(p), self.name(p) });
        }
        try self.print("{s}goto block{d};\n", .{ inner, @intFromEnum(target) });
        if (!scoped) try self.print("{s}}}\n", .{indent});
    }

    /// Print the C spelling of an IR type. A null type is `void`.
    fn emitType(self: *Emitter, ty: ?Type) Error!void {
        const t = ty orelse return self.w("void");
        switch (self.func.types.type_kind(t)) {
            .bool => try self.w("bool"),
            .int => |i| {
                const width: u16 = if (i.bits <= 8) 8 else if (i.bits <= 16) 16 else if (i.bits <= 32) 32 else 64;
                try self.print("{s}int{d}_t", .{ if (i.signedness == .unsigned) "u" else "", width });
            },
            .float => |f| try self.w(if (f == .f32) "float" else "double"),
            .ptr => try self.w("void*"),
            .@"struct", .vector, .array, .slice => try self.aggName(&self.out, t),
        }
    }
};

test "emits a function returning a constant" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const v = try func.appendInst(entry, i32_t, .{ .iconst = 42 });
    func.setTerminator(entry, .{ .ret = v });

    const src = try emitFunction(std.testing.allocator, &func, "answer");
    defer std.testing.allocator.free(src);

    try std.testing.expectEqualStrings(
        \\int32_t answer(void) {
        \\    int32_t v0;
        \\    v0 = 42;
        \\    return v0;
        \\}
        \\
    , src);
}

test "emits params and arithmetic" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    const src = try emitFunction(std.testing.allocator, &func, "add");
    defer std.testing.allocator.free(src);

    try std.testing.expectEqualStrings(
        \\int32_t add(int32_t v0, int32_t v1) {
        \\    int32_t v2;
        \\    v2 = v0 + v1;
        \\    return v2;
        \\}
        \\
    , src);
}

test "emits a struct typedef, construction, and extract" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const st = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });
    const entry = try func.appendBlock();
    const a = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    const b = try func.appendInst(entry, i32_t, .{ .iconst = 9 });
    const s = try func.appendStructNew(entry, st, &.{ a, b });
    const x = try func.appendInst(entry, i32_t, .{ .extract = .{ .aggregate = s, .index = 1 } });
    func.setTerminator(entry, .{ .ret = x });

    const src = try emitFunction(std.testing.allocator, &func, "pick");
    defer std.testing.allocator.free(src);

    try std.testing.expectEqualStrings(
        \\typedef struct { int32_t f0; int32_t f1; } agg0;
        \\
        \\int32_t pick(void) {
        \\    int32_t v0;
        \\    int32_t v1;
        \\    agg0 v2;
        \\    int32_t v3;
        \\    v0 = 7;
        \\    v1 = 9;
        \\    v2 = (agg0){ v0, v1 };
        \\    v3 = v2.f1;
        \\    return v3;
        \\}
        \\
    , src);
}

test "emits an if/else diamond with a merge parameter" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const merge = try func.appendBlock();
    const r = try func.appendBlockParam(merge, i32_t);
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, cond, .{ .target = merge, .args = &.{a} }, .{ .target = merge, .args = &.{b} });
    func.setTerminator(merge, .{ .ret = r });

    const src = try emitFunction(std.testing.allocator, &func, "maxi");
    defer std.testing.allocator.free(src);

    try std.testing.expectEqualStrings(
        \\int32_t maxi(int32_t v0, int32_t v1) {
        \\    bool v2;
        \\    int32_t v3;
        \\    v2 = (v0 > v1);
        \\    if (v2) {
        \\        int32_t t3 = v0;
        \\        v3 = t3;
        \\        goto block1;
        \\    } else {
        \\        int32_t t3 = v1;
        \\        v3 = t3;
        \\        goto block1;
        \\    }
        \\block1:;
        \\    return v3;
        \\}
        \\
    , src);
}
