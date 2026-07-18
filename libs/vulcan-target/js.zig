//! Emit JavaScript from a Vulcan IR function: a second source-level backend alongside
//! `c.zig`. JavaScript has one number type and no `goto`, so the mapping differs from C:
//!
//!   - Integers are BigInt, canonicalized to their type after each op with
//!     `BigInt.asIntN(bits, x)` / `asUintN`, so wrapping matches any bit width including 64.
//!   - Floats are Number; an f32 result is `Math.fround`-normalized so single precision is
//!     exact, and an f16 result is `Math.f16round`-normalized the same way (scalar f16 only;
//!     f16 nested in a composite is rejected, see `functionUsesCompositeF16`).
//!   - Booleans are JS booleans; pointers are BigInt byte offsets into a shared memory.
//!   - Aggregates (struct/vector/array/slice) are JS arrays `[f0, f1, ...]`.
//!   - Control flow is a `while (true) switch (__blk)` state machine (blocks are cases,
//!     jumps assign `__blk` and `break`), since JS lacks `goto`.
//!   - Memory (alloca/load/store) uses a DataView over one ArrayBuffer, set up by the
//!     runtime preamble (`runtime_preamble`), which also holds the reinterpret and
//!     round-to-even helpers. A single function depends on that preamble being present, the
//!     way a C function depends on its headers.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Type = ir.types.Type;

pub const Error = error{Unsupported} || std.mem.Allocator.Error;

/// The shared runtime every emitted function assumes: a linear memory + DataView, a bump
/// allocator for `alloca`, typed load/store helpers, bit-reinterpret helpers, and a
/// round-half-to-even `__rint`. `emitModule` includes it once; a lone `emitFunction` output
/// needs it prepended by the caller.
pub const runtime_preamble =
    \\const __mem = new ArrayBuffer(1 << 20);
    \\const __dv = new DataView(__mem);
    \\let __sp = 0;
    \\function __alloca(n){ const p = __sp; __sp += n; return BigInt(p); }
    \\function __ld_i8(p){ return BigInt(__dv.getInt8(Number(p))); }
    \\function __ld_u8(p){ return BigInt(__dv.getUint8(Number(p))); }
    \\function __ld_i16(p){ return BigInt(__dv.getInt16(Number(p), true)); }
    \\function __ld_u16(p){ return BigInt(__dv.getUint16(Number(p), true)); }
    \\function __ld_i32(p){ return BigInt(__dv.getInt32(Number(p), true)); }
    \\function __ld_u32(p){ return BigInt(__dv.getUint32(Number(p), true)); }
    \\function __ld_i64(p){ return __dv.getBigInt64(Number(p), true); }
    \\function __ld_u64(p){ return __dv.getBigUint64(Number(p), true); }
    \\function __ld_f16(p){ return __dv.getFloat16(Number(p), true); }
    \\function __ld_f32(p){ return __dv.getFloat32(Number(p), true); }
    \\function __ld_f64(p){ return __dv.getFloat64(Number(p), true); }
    \\function __st_i8(p,v){ __dv.setInt8(Number(p), Number(v)); }
    \\function __st_u8(p,v){ __dv.setUint8(Number(p), Number(v)); }
    \\function __st_i16(p,v){ __dv.setInt16(Number(p), Number(v), true); }
    \\function __st_u16(p,v){ __dv.setUint16(Number(p), Number(v), true); }
    \\function __st_i32(p,v){ __dv.setInt32(Number(p), Number(v), true); }
    \\function __st_u32(p,v){ __dv.setUint32(Number(p), Number(v), true); }
    \\function __st_i64(p,v){ __dv.setBigInt64(Number(p), BigInt.asIntN(64, v), true); }
    \\function __st_u64(p,v){ __dv.setBigUint64(Number(p), BigInt.asUintN(64, v), true); }
    \\function __st_f16(p,v){ __dv.setFloat16(Number(p), v, true); }
    \\function __st_f32(p,v){ __dv.setFloat32(Number(p), v, true); }
    \\function __st_f64(p,v){ __dv.setFloat64(Number(p), v, true); }
    \\const __cvt = new DataView(new ArrayBuffer(8));
    \\function __bi_i32_f32(x){ __cvt.setInt32(0, Number(x), true); return __cvt.getFloat32(0, true); }
    \\function __bi_f32_i32(x){ __cvt.setFloat32(0, x, true); return BigInt(__cvt.getInt32(0, true)); }
    \\function __bi_i64_f64(x){ __cvt.setBigInt64(0, BigInt.asIntN(64, x), true); return __cvt.getFloat64(0, true); }
    \\function __bi_f64_i64(x){ __cvt.setFloat64(0, x, true); return __cvt.getBigInt64(0, true); }
    \\function __rint(x){ const f = Math.floor(x); const d = x - f; if (d < 0.5) return f; if (d > 0.5) return f + 1; return (f % 2 === 0) ? f : f + 1; }
    \\
;

/// Emit `func` as a single JS function named `name`. Caller owns the returned source. The
/// output assumes `runtime_preamble` is in scope.
pub fn emitFunction(allocator: std.mem.Allocator, func: *const Function, name: []const u8) Error![]u8 {
    // JS has `Math.f16round` (mirrors `Math.fround` for f32), so scalar f16 lowers the same
    // way f32 does: every fround site below gets an f16round twin. f16 nested in a
    // vector/aggregate has no tested emission path here (the other scalar-f16 backends draw
    // the same line), so that composite case still rejects cleanly. Covers both this direct
    // entry and emitModule, which delegates here per function.
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;

    var e = Emitter{ .allocator = allocator, .func = func };
    defer e.deinit();
    try e.run(name);
    return e.out.toOwnedSlice(allocator);
}

/// One named function in a module.
pub const NamedFunc = struct { name: []const u8, func: *const Function };

/// Emit a whole module: the runtime preamble followed by every function. JS hoists function
/// declarations, so they may call one another in any order with no forward prototypes.
pub fn emitModule(allocator: std.mem.Allocator, funcs: []const NamedFunc) Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, runtime_preamble);
    try out.append(allocator, '\n');
    for (funcs) |nf| {
        const body = try emitFunction(allocator, nf.func, nf.name);
        defer allocator.free(body);
        try out.appendSlice(allocator, body);
    }
    return out.toOwnedSlice(allocator);
}

const Emitter = struct {
    allocator: std.mem.Allocator,
    func: *const Function,
    out: std.ArrayList(u8) = .empty,
    names: []u32 = &.{},
    uses_alloca: bool = false,

    fn deinit(self: *Emitter) void {
        self.out.deinit(self.allocator);
        self.allocator.free(self.names);
    }

    fn w(self: *Emitter, bytes: []const u8) Error!void {
        try self.out.appendSlice(self.allocator, bytes);
    }

    fn print(self: *Emitter, comptime fmt: []const u8, args: anytype) Error!void {
        try self.out.print(self.allocator, fmt, args);
    }

    fn name(self: *Emitter, v: Value) u32 {
        return self.names[@intFromEnum(v)];
    }

    fn kind(self: *Emitter, ty: Type) ir.types.TypeKind {
        return self.func.types.type_kind(ty);
    }

    fn run(self: *Emitter, sym: []const u8) Error!void {
        try self.assignNames();
        const func = self.func;
        for (0..func.instCount()) |ii| {
            if (func.opcode(@enumFromInt(ii)) == .alloca) self.uses_alloca = true;
        }

        // Signature: entry-block parameters are the JS parameters.
        const entry_params = func.blockParams(@as(Block, @enumFromInt(0)));
        try self.print("function {s}(", .{sym});
        for (entry_params, 0..) |p, i| {
            if (i != 0) try self.w(", ");
            try self.print("v{d}", .{self.name(p)});
        }
        try self.w(") {\n");

        try self.emitDeclarations();
        if (self.uses_alloca) try self.w("  const __base = __sp;\n");

        if (self.isStraightLine()) {
            // One block with no branching: emit it directly, no state machine.
            const entry: Block = @enumFromInt(0);
            for (func.blockInsts(entry)) |inst| try self.emitInst(inst, "  ");
            try self.emitTerminator(entry, "  ");
        } else {
            try self.w("  let __blk = 0;\n  while (true) {\n    switch (__blk) {\n");
            for (0..func.blockCount()) |bi| {
                const block: Block = @enumFromInt(@as(u32, @intCast(bi)));
                try self.print("      case {d}: {{\n", .{bi});
                var branched = false;
                for (func.blockInsts(block)) |inst| {
                    if (func.opcode(inst) == .@"if") branched = true;
                    try self.emitInst(inst, "        ");
                }
                if (!branched) try self.emitTerminator(block, "        ");
                try self.w("      }\n");
            }
            try self.w("    }\n  }\n");
        }
        try self.w("}\n");
    }

    /// A single block ending in a plain return, with no `if`: the simple straight-line
    /// shape that needs no `switch` state machine.
    fn isStraightLine(self: *Emitter) bool {
        const func = self.func;
        if (func.blockCount() != 1) return false;
        const entry: Block = @enumFromInt(0);
        for (func.blockInsts(entry)) |inst| {
            if (func.opcode(inst) == .@"if") return false;
        }
        return switch (func.terminator(entry) orelse return true) {
            .ret => true,
            .jump => false,
        };
    }

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

    /// Declare every value that is not an entry-block parameter with `let`, at function
    /// scope, so a value defined in one block is visible from another.
    fn emitDeclarations(self: *Emitter) Error!void {
        const func = self.func;
        var first = true;
        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(@as(u32, @intCast(bi)));
            if (bi != 0) for (func.blockParams(block)) |p| try self.declare(p, &first);
            for (func.blockInsts(block)) |inst| {
                if (func.instResult(inst)) |res| try self.declare(res, &first);
            }
        }
        if (!first) try self.w(";\n");
    }

    fn declare(self: *Emitter, v: Value, first: *bool) Error!void {
        if (first.*) {
            try self.w("  let ");
            first.* = false;
        } else {
            try self.w(", ");
        }
        try self.print("v{d}", .{self.name(v)});
    }

    fn emitInst(self: *Emitter, inst: Inst, indent: []const u8) Error!void {
        const func = self.func;
        const op = func.opcode(inst);
        const res = func.instResult(inst);
        switch (op) {
            .iconst => |val| try self.emitIconst(res.?, val, indent),
            .fconst => |val| {
                try self.print("{s}v{d} = ", .{ indent, self.name(res.?) });
                // A literal prints already rounded to its own type, so the JS constant is
                // the exact value the IR asked for, not a second re-rounding at runtime.
                switch (self.floatKind(func.valueType(res.?))) {
                    .f16 => try self.print("Math.f16round({e})", .{@as(f64, @as(f16, @floatCast(val)))}),
                    .f32 => try self.print("Math.fround({e})", .{@as(f32, @floatCast(val))}),
                    .f64 => try self.print("{e}", .{val}),
                }
                try self.w(";\n");
            },
            .arith => |a| try self.emitArith(res.?, a.op, a.lhs, .{ .value = a.rhs }, indent),
            .arith_imm => |a| try self.emitArith(res.?, a.op, a.lhs, .{ .imm = a.imm }, indent),
            .icmp => |c| {
                const sym = switch (c.op) {
                    .eq => "===",
                    .ne => "!==",
                    else => c.op.symbol(),
                };
                try self.print("{s}v{d} = (v{d} {s} v{d});\n", .{ indent, self.name(res.?), self.name(c.lhs), sym, self.name(c.rhs) });
            },
            .select => |s| try self.print("{s}v{d} = v{d} ? v{d} : v{d};\n", .{
                indent, self.name(res.?), self.name(s.cond), self.name(s.then), self.name(s.@"else"),
            }),
            .@"if" => |cf| {
                try self.print("{s}if (v{d}) {{\n", .{ indent, self.name(cf.cond) });
                try self.emitEdge(cf.then.target, func.blockArgs(cf.then), indent);
                try self.print("{s}}} else {{\n", .{indent});
                try self.emitEdge(cf.@"else".target, func.blockArgs(cf.@"else"), indent);
                try self.print("{s}}}\n", .{indent});
            },
            .convert => |cv| try self.emitConvert(res.?, cv.value, indent),
            .unary => |u| try self.emitUnary(res.?, u, indent),
            .alloca => |al| try self.print("{s}v{d} = __alloca({d});\n", .{ indent, self.name(res.?), self.typeSize(al.elem) }),
            .load => |ld| try self.emitLoad(res.?, ld.ptr, indent),
            .store => |st| try self.emitStore(st.value, st.ptr, indent),
            .prefetch => {}, // a hint, JS has no prefetch, dropped
            // dot is aarch64+dotprod-only in practice; the JS backend has no lowering for it.
            .dot => return error.Unsupported,
            // matmul is et-soc-only (a later task); the JS backend has no lowering for it.
            .matmul => return error.Unsupported,
            .struct_new => |sn| {
                try self.print("{s}v{d} = [", .{ indent, self.name(res.?) });
                for (func.valueList(sn.fields), 0..) |field, i| {
                    if (i != 0) try self.w(", ");
                    try self.print("v{d}", .{self.name(field)});
                }
                try self.w("];\n");
            },
            .extract => |ex| try self.print("{s}v{d} = v{d}[{d}];\n", .{ indent, self.name(res.?), self.name(ex.aggregate), ex.index }),
            .call => |c| {
                try self.print("{s}", .{indent});
                if (res) |r| try self.print("v{d} = ", .{self.name(r)});
                try self.print("{s}(", .{func.symbolName(c.symbol)});
                try self.emitArgs(func.valueList(c.args));
                try self.w(");\n");
            },
            .call_indirect => |ci| {
                try self.print("{s}", .{indent});
                if (res) |r| try self.print("v{d} = ", .{self.name(r)});
                try self.print("v{d}(", .{self.name(ci.target)});
                try self.emitArgs(func.valueList(ci.args));
                try self.w(");\n");
            },
            // A global resolves to a same-named binding in scope (a function for
            // call_indirect, or a byte offset for a data global).
            .global_addr => |ga| try self.print("{s}v{d} = {s};\n", .{ indent, self.name(res.?), func.symbolName(ga.symbol) }),
        }
    }

    fn emitArgs(self: *Emitter, args: []const Value) Error!void {
        for (args, 0..) |arg, i| {
            if (i != 0) try self.w(", ");
            try self.print("v{d}", .{self.name(arg)});
        }
    }

    const Rhs = union(enum) { value: Value, imm: i64 };

    fn emitIconst(self: *Emitter, res: Value, val: i64, indent: []const u8) Error!void {
        const ty = self.func.valueType(res);
        if (self.kind(ty) == .bool) {
            try self.print("{s}v{d} = {s};\n", .{ indent, self.name(res), if (val != 0) "true" else "false" });
            return;
        }
        try self.print("{s}v{d} = ", .{ indent, self.name(res) });
        try self.wrapPre(ty);
        try self.print("{d}n", .{val});
        try self.wrapPost(ty);
        try self.w(";\n");
    }

    fn emitArith(self: *Emitter, res: Value, bop: ir.function.BinOp, lhs: Value, rhs: Rhs, indent: []const u8) Error!void {
        const ty = self.func.valueType(res);
        // Booleans use logical operators so the result stays a boolean.
        if (self.kind(ty) == .bool) {
            const sym = switch (bop) {
                .bit_and => "&&",
                .bit_or => "||",
                .bit_xor => "!==",
                else => return error.Unsupported,
            };
            try self.print("{s}v{d} = (v{d} {s} v{d});\n", .{ indent, self.name(res), self.name(lhs), sym, self.name(rhs.value) });
            return;
        }
        if (self.kind(ty) == .vector) {
            const v = self.kind(ty).vector;
            try self.print("{s}v{d} = [", .{ indent, self.name(res) });
            for (0..v.len) |lane| {
                if (lane != 0) try self.w(", ");
                try self.wrapPre(v.elem);
                try self.print("v{d}[{d}] {s} ", .{ self.name(lhs), lane, bop.symbol() });
                try self.rhsLane(rhs, lane);
                try self.wrapPost(v.elem);
            }
            try self.w("];\n");
            return;
        }
        try self.print("{s}v{d} = ", .{ indent, self.name(res) });
        try self.wrapPre(ty);
        try self.print("v{d} {s} ", .{ self.name(lhs), bop.symbol() });
        try self.rhsScalar(rhs);
        try self.wrapPost(ty);
        try self.w(";\n");
    }

    fn rhsScalar(self: *Emitter, rhs: Rhs) Error!void {
        switch (rhs) {
            .value => |v| try self.print("v{d}", .{self.name(v)}),
            .imm => |imm| try self.print("{d}n", .{imm}),
        }
    }

    fn rhsLane(self: *Emitter, rhs: Rhs, lane: usize) Error!void {
        switch (rhs) {
            .value => |v| try self.print("v{d}[{d}]", .{ self.name(v), lane }),
            .imm => |imm| try self.print("{d}n", .{imm}),
        }
    }

    fn emitConvert(self: *Emitter, res: Value, src: Value, indent: []const u8) Error!void {
        const rty = self.func.valueType(res);
        const sty = self.func.valueType(src);
        try self.print("{s}v{d} = ", .{ indent, self.name(res) });
        switch (self.kind(rty)) {
            .float => switch (self.floatKind(rty)) {
                .f16 => try self.print("Math.f16round(Number(v{d}))", .{self.name(src)}),
                .f32 => try self.print("Math.fround(Number(v{d}))", .{self.name(src)}),
                .f64 => try self.print("Number(v{d})", .{self.name(src)}),
            },
            .int => {
                try self.wrapPre(rty);
                switch (self.kind(sty)) {
                    .float => try self.print("BigInt(Math.trunc(v{d}))", .{self.name(src)}),
                    .bool => try self.print("BigInt(v{d} ? 1 : 0)", .{self.name(src)}),
                    else => try self.print("v{d}", .{self.name(src)}),
                }
                try self.wrapPost(rty);
            },
            .bool => switch (self.kind(sty)) {
                .float => try self.print("(v{d} !== 0)", .{self.name(src)}),
                else => try self.print("(v{d} !== 0n)", .{self.name(src)}),
            },
            else => return error.Unsupported,
        }
        try self.w(";\n");
    }

    fn emitUnary(self: *Emitter, res: Value, u: ir.function.Unary, indent: []const u8) Error!void {
        const rty = self.func.valueType(res);
        try self.print("{s}v{d} = ", .{ indent, self.name(res) });
        if (u.op == .reinterpret) {
            const helper = try self.reinterpretHelper(rty, self.func.valueType(u.value));
            try self.print("{s}(v{d})", .{ helper, self.name(u.value) });
            try self.w(";\n");
            return;
        }
        const fname: []const u8 = switch (u.op) {
            .sqrt => "Math.sqrt",
            .floor => "Math.floor",
            .ceil => "Math.ceil",
            .trunc => "Math.trunc",
            .nearest => "__rint",
            .reinterpret => unreachable,
        };
        switch (self.floatKind(rty)) {
            .f16 => try self.print("Math.f16round({s}(v{d}))", .{ fname, self.name(u.value) }),
            .f32 => try self.print("Math.fround({s}(v{d}))", .{ fname, self.name(u.value) }),
            .f64 => try self.print("{s}(v{d})", .{ fname, self.name(u.value) }),
        }
        try self.w(";\n");
    }

    fn reinterpretHelper(self: *Emitter, rty: Type, sty: Type) Error![]const u8 {
        const r = self.kind(rty);
        const s = self.kind(sty);
        if (r == .float and r.float == .f32 and s == .int) return "__bi_i32_f32";
        if (r == .int and s == .float and s.float == .f32) return "__bi_f32_i32";
        if (r == .float and r.float == .f64 and s == .int) return "__bi_i64_f64";
        if (r == .int and s == .float and s.float == .f64) return "__bi_f64_i64";
        return error.Unsupported;
    }

    fn emitLoad(self: *Emitter, res: Value, ptr: Value, indent: []const u8) Error!void {
        const ty = self.func.valueType(res);
        if (self.kind(ty) == .bool) {
            try self.print("{s}v{d} = (__dv.getUint8(Number(v{d})) !== 0);\n", .{ indent, self.name(res), self.name(ptr) });
            return;
        }
        try self.print("{s}v{d} = __ld_{s}(v{d});\n", .{ indent, self.name(res), self.memSuffix(ty), self.name(ptr) });
    }

    fn emitStore(self: *Emitter, value: Value, ptr: Value, indent: []const u8) Error!void {
        const ty = self.func.valueType(value);
        if (self.kind(ty) == .bool) {
            try self.print("{s}__dv.setUint8(Number(v{d}), v{d} ? 1 : 0);\n", .{ indent, self.name(ptr), self.name(value) });
            return;
        }
        try self.print("{s}__st_{s}(v{d}, v{d});\n", .{ indent, self.memSuffix(ty), self.name(ptr), self.name(value) });
    }

    fn emitTerminator(self: *Emitter, block: Block, indent: []const u8) Error!void {
        const restore: []const u8 = if (self.uses_alloca) "__sp = __base; " else "";
        const term = self.func.terminator(block) orelse {
            try self.print("{s}{s}return;\n", .{ indent, restore });
            return;
        };
        switch (term) {
            .ret => |v| if (v) |vv| {
                try self.print("{s}{s}return v{d};\n", .{ indent, restore, self.name(vv) });
            } else {
                try self.print("{s}{s}return;\n", .{ indent, restore });
            },
            .jump => |j| try self.emitEdge(j.target, self.func.blockArgs(j), indent),
        }
    }

    /// Emit a control-flow edge: assign the target block's parameters from the arguments
    /// (through temporaries, so a swap across the edge is correct), set `__blk`, and break
    /// back to the dispatch loop.
    fn emitEdge(self: *Emitter, target: Block, args: []const Value, indent: []const u8) Error!void {
        const params = self.func.blockParams(target);
        if (params.len == 0) {
            try self.print("{s}  __blk = {d}; break;\n", .{ indent, @intFromEnum(target) });
            return;
        }
        try self.print("{s}  {{\n", .{indent});
        for (params, args) |p, arg| {
            try self.print("{s}    let t{d} = v{d};\n", .{ indent, self.name(p), self.name(arg) });
        }
        for (params) |p| {
            try self.print("{s}    v{d} = t{d};\n", .{ indent, self.name(p), self.name(p) });
        }
        try self.print("{s}    __blk = {d}; break;\n", .{ indent, @intFromEnum(target) });
        try self.print("{s}  }}\n", .{indent});
    }

    /// Write the prefix that canonicalizes an expression of type `ty`: `BigInt.asIntN(...)`
    /// for integers/pointers, `Math.fround(` for f32, `Math.f16round(` for f16 (so a plain JS
    /// Number `+`/`*`/etc, computed at double precision, gets re-rounded to the op's actual
    /// width once the expression closes), nothing for f64.
    fn wrapPre(self: *Emitter, ty: Type) Error!void {
        switch (self.kind(ty)) {
            .int => |i| try self.print("BigInt.as{s}N({d}, ", .{ if (i.signedness == .unsigned) "Uint" else "Int", i.bits }),
            .ptr => try self.w("BigInt.asIntN(64, "),
            .float => |f| switch (f) {
                .f16 => try self.w("Math.f16round("),
                .f32 => try self.w("Math.fround("),
                .f64 => {},
            },
            else => {},
        }
    }

    fn wrapPost(self: *Emitter, ty: Type) Error!void {
        switch (self.kind(ty)) {
            .int, .ptr => try self.w(")"),
            .float => |f| switch (f) {
                .f16, .f32 => try self.w(")"),
                .f64 => {},
            },
            else => {},
        }
    }

    /// The IR float kind of a (known-float) type, so every f16/f32/f64 site switches
    /// exhaustively instead of collapsing f16 into the f64 branch by omission.
    fn floatKind(self: *Emitter, ty: Type) ir.types.FloatKind {
        return self.kind(ty).float;
    }

    /// The load/store helper suffix for a scalar type (bool is handled separately).
    fn memSuffix(self: *Emitter, ty: Type) []const u8 {
        return switch (self.kind(ty)) {
            .int => |i| if (i.bits <= 8)
                (if (i.signedness == .unsigned) "u8" else "i8")
            else if (i.bits <= 16)
                (if (i.signedness == .unsigned) "u16" else "i16")
            else if (i.bits <= 32)
                (if (i.signedness == .unsigned) "u32" else "i32")
            else
                (if (i.signedness == .unsigned) "u64" else "i64"),
            .float => |f| switch (f) {
                .f16 => "f16",
                .f32 => "f32",
                .f64 => "f64",
            },
            .ptr => "i64",
            else => "i32",
        };
    }

    /// A packed byte size for a type, enough to stride an alloca'd array by its element.
    fn typeSize(self: *Emitter, ty: Type) u64 {
        return switch (self.kind(ty)) {
            .bool => 1,
            .int => |i| (i.bits + 7) / 8,
            .float => |f| switch (f) {
                .f16 => 2,
                .f32 => 4,
                .f64 => 8,
            },
            .ptr => 8,
            .vector => |v| v.len * self.typeSize(v.elem),
            .array => |a| a.len * self.typeSize(a.elem),
            .slice => 16,
            .@"struct" => |fields| blk: {
                var total: u64 = 0;
                for (fields) |f| total += self.typeSize(f);
                break :blk total;
            },
        };
    }
};

test "emits a straight-line function returning a constant" {
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
        \\function add(v0, v1) {
        \\  let v2;
        \\  v2 = BigInt.asIntN(32, v0 + v1);
        \\  return v2;
        \\}
        \\
    , src);
}

test "an f16 add emits Math.f16round, mirroring the f32 Math.fround path" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f16_t);
    const b = try func.appendBlockParam(entry, f16_t);
    const sum = try func.appendInst(entry, f16_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    const src = try emitFunction(std.testing.allocator, &func, "add");
    defer std.testing.allocator.free(src);

    // Same shape as the f32 add would emit, but `Math.f16round` in place of `Math.fround`:
    // the JS engine does the `+` at double precision, then this rounds it back to the
    // nearest half so the result matches real f16 arithmetic.
    try std.testing.expectEqualStrings(
        \\function add(v0, v1) {
        \\  let v2;
        \\  v2 = Math.f16round(v0 + v1);
        \\  return v2;
        \\}
        \\
    , src);

    // A module also accepts a scalar-f16 function now; only a composite f16 rejects (below).
    const named = [_]NamedFunc{.{ .name = "add", .func = &func }};
    const mod_src = try emitModule(std.testing.allocator, &named);
    defer std.testing.allocator.free(mod_src);
    try std.testing.expect(std.mem.indexOf(u8, mod_src, "Math.f16round(v0 + v1)") != null);
}

test "composite f16 (a vector of half) is still rejected cleanly" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const v2 = try func.types.intern(.{ .vector = .{ .len = 2, .elem = f16_t } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, v2);
    func.setTerminator(entry, .{ .ret = a });

    try std.testing.expectError(error.Unsupported, emitFunction(std.testing.allocator, &func, "vh"));

    const named = [_]NamedFunc{.{ .name = "vh", .func = &func }};
    try std.testing.expectError(error.Unsupported, emitModule(std.testing.allocator, &named));
}

test "emits a state machine for an if/else diamond" {
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
        \\function maxi(v0, v1) {
        \\  let v2, v3;
        \\  let __blk = 0;
        \\  while (true) {
        \\    switch (__blk) {
        \\      case 0: {
        \\        v2 = (v0 > v1);
        \\        if (v2) {
        \\          {
        \\            let t3 = v0;
        \\            v3 = t3;
        \\            __blk = 1; break;
        \\          }
        \\        } else {
        \\          {
        \\            let t3 = v1;
        \\            v3 = t3;
        \\            __blk = 1; break;
        \\          }
        \\        }
        \\      }
        \\      case 1: {
        \\        return v3;
        \\      }
        \\    }
        \\  }
        \\}
        \\
    , src);
}
