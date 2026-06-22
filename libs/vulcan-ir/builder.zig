//! Constructs IR in Zig. Wraps a `Function` and tracks a current insertion
//! block, so callers append instructions without threading the block through
//! every call. Result types are derived from operands where possible. Frontends
//! build IR through this rather than the text parser.

const std = @import("std");
const function = @import("function.zig");
const types = @import("types.zig");

const Function = function.Function;
const Block = function.Block;
const Value = function.Value;
const EdgeDesc = function.EdgeDesc;
const CmpOp = function.CmpOp;
const BinOp = function.BinOp;
const Type = types.Type;
const Error = std.mem.Allocator.Error;

/// Constructs IR into a function, tracking a current insertion block.
pub const Builder = struct {
    func: *Function,
    block: Block,

    pub fn init(func: *Function, block: Block) Builder {
        return .{ .func = func, .block = block };
    }

    /// Redirect subsequent instructions to a different block.
    pub fn switchTo(self: *Builder, block: Block) void {
        self.block = block;
    }

    /// An integer constant of type `ty`.
    pub fn iconst(self: *Builder, ty: Type, value: i64) Error!Value {
        return self.func.appendInst(self.block, ty, .{ .iconst = value });
    }

    /// A binary arithmetic/bitwise op. The result type is taken from `lhs`.
    pub fn arith(self: *Builder, op: BinOp, lhs: Value, rhs: Value) Error!Value {
        const ty = self.func.valueType(lhs);
        return self.func.appendInst(self.block, ty, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    }

    /// Integer addition (shorthand for `arith(.add, ...)`).
    pub fn iadd(self: *Builder, lhs: Value, rhs: Value) Error!Value {
        return self.arith(.add, lhs, rhs);
    }

    /// Integer comparison. Produces a `bool`, signedness comes from the operands.
    pub fn icmp(self: *Builder, op: CmpOp, lhs: Value, rhs: Value) Error!Value {
        const bool_t = try self.func.types.intern(.bool);
        return self.func.appendInst(self.block, bool_t, .{ .icmp = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    }

    /// Value-producing conditional. The result type is taken from `then_value`.
    pub fn select(self: *Builder, cond: Value, then_value: Value, else_value: Value) Error!Value {
        const ty = self.func.valueType(then_value);
        return self.func.appendInst(self.block, ty, .{ .select = .{ .cond = cond, .then = then_value, .@"else" = else_value } });
    }

    /// Construct an aggregate from `fields`. The struct type is inferred.
    pub fn structNew(self: *Builder, fields: []const Value) Error!Value {
        var field_types: std.ArrayList(Type) = .empty;
        defer field_types.deinit(self.func.allocator);
        for (fields) |f| try field_types.append(self.func.allocator, self.func.valueType(f));
        const st = try self.func.types.intern(.{ .@"struct" = field_types.items });
        return self.func.appendStructNew(self.block, st, fields);
    }

    /// Extract field `index` from aggregate `aggregate`.
    pub fn extract(self: *Builder, aggregate: Value, index: u32) Error!Value {
        const field_ty = switch (self.func.types.type_kind(self.func.valueType(aggregate))) {
            .@"struct" => |fields| fields[index],
            else => unreachable, // caller must pass an aggregate
        };
        return self.func.appendInst(self.block, field_ty, .{ .extract = .{ .aggregate = aggregate, .index = index } });
    }

    /// Load a value of type `ty` from `ptr`.
    pub fn load(self: *Builder, ty: Type, ptr: Value) Error!Value {
        return self.func.appendInst(self.block, ty, .{ .load = .{ .ptr = ptr } });
    }

    /// Store `value` to `ptr`.
    pub fn store(self: *Builder, value: Value, ptr: Value) Error!void {
        return self.func.appendStore(self.block, value, ptr);
    }

    /// Return from the function, optionally with a value.
    pub fn ret(self: *Builder, value: ?Value) void {
        self.func.setTerminator(self.block, .{ .ret = value });
    }

    /// Jump unconditionally to `target`, passing `args` to its parameters.
    pub fn jump(self: *Builder, target: Block, args: []const Value) Error!void {
        return self.func.setJump(self.block, target, args);
    }

    /// Append a non-terminating conditional on `cond`, choosing one of two edges.
    /// Control continues to the block's terminator afterward.
    pub fn @"if"(self: *Builder, cond: Value, then_edge: EdgeDesc, else_edge: EdgeDesc) Error!void {
        return self.func.appendIf(self.block, cond, then_edge, else_edge);
    }
};

test "builder appends instructions to the current block" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();

    var b = Builder.init(&func, entry);
    const x = try b.iconst(i32_t, 10);
    const y = try b.iconst(i32_t, 20);
    const sum = try b.iadd(x, y);

    try std.testing.expectEqual(i32_t, func.valueType(sum));

    const op = func.opcode(func.definingInst(sum).?);
    try std.testing.expectEqual(x, op.arith.lhs);
    try std.testing.expectEqual(y, op.arith.rhs);
}

test "builder builds arithmetic with any operator" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);

    var bld = Builder.init(&func, entry);
    const d = try bld.arith(.mul, a, b);

    try std.testing.expectEqual(i32_t, func.valueType(d));
    try std.testing.expectEqual(function.BinOp.mul, func.opcode(func.definingInst(d).?).arith.op);
}

test "builder builds a comparison producing bool" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);

    var bld = Builder.init(&func, entry);
    const c = try bld.icmp(.gt, a, b);

    try std.testing.expectEqual(bool_t, func.valueType(c));
    try std.testing.expectEqual(function.CmpOp.gt, func.opcode(func.definingInst(c).?).icmp.op);
}

test "builder builds struct construction and extraction" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);

    var bld = Builder.init(&func, entry);
    const s = try bld.structNew(&.{ a, b });
    const f0 = try bld.extract(s, 0);

    try std.testing.expectEqual(i32_t, func.valueType(f0));
}

test "builder builds loads and stores" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);

    var bld = Builder.init(&func, entry);
    const v = try bld.load(i32_t, p);
    try bld.store(v, p);

    try std.testing.expectEqual(i32_t, func.valueType(v));
    const insts = func.blockInsts(entry);
    try std.testing.expectEqual(p, func.opcode(insts[insts.len - 1]).store.ptr);
}

test "builder builds a select" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const cond = try func.appendBlockParam(entry, bool_t);
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);

    var bld = Builder.init(&func, entry);
    const c = try bld.select(cond, a, b);

    try std.testing.expectEqual(i32_t, func.valueType(c));
}

test "builder builds a branching function" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const cond = try func.appendBlockParam(entry, bool_t);
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();

    var b_ = Builder.init(&func, entry);
    try b_.@"if"(cond, .{ .target = then_b }, .{ .target = else_b });

    b_.switchTo(then_b);
    b_.ret(a);

    b_.switchTo(else_b);
    b_.ret(b);

    const if_inst = func.blockInsts(entry)[func.blockInsts(entry).len - 1];
    try std.testing.expectEqual(cond, func.opcode(if_inst).@"if".cond);
    try std.testing.expectEqual(function.Terminator{ .ret = a }, func.terminator(then_b).?);
    try std.testing.expectEqual(function.Terminator{ .ret = b }, func.terminator(else_b).?);
}
