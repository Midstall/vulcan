//! Algebraic simplification rewrites arithmetic identities that constant folding misses because an
//! operand is not a compile-time constant, like `x + 0 -> x`, `x * 1 -> x`, `x - x -> 0`,
//! `x * 0 -> 0`, `x & x -> x`, `x ^ x -> 0`, `x << 0 -> x`, and `x / 1 -> x`.
//!
//! These are integer only. Floating point `x + 0.0` is not `x` because it flushes `-0.0` to `+0.0`,
//! and `x * 0.0` is not `0.0` once NaN and infinities are in play, so applying them to floats would
//! be unsound.
//!
//! An identity that yields a value, like `x + 0 -> x`, replaces every use of the result with the
//! surviving operand and lets DCE remove the now-dead instruction. An identity that yields a
//! constant, like `x * 0 -> 0`, rewrites the instruction to an `iconst` in place, the same way
//! constant folding does.
//!
//! `select` also folds. A constant condition picks one arm (`select(true, a, b) -> a`) and identical
//! arms collapse (`select(c, x, x) -> x`). Both hold for any type since they only choose a value, and
//! commonly fire once constant folding has resolved an `icmp` condition.
//!
//! A self-comparison folds to a constant bool: `x == x -> 1`, `x < x -> 0`, and so on. icmp is
//! integer, so there is no NaN caveat, and the resulting constant can go on to feed branch folding.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const BinOp = ir.function.BinOp;

pub const pass_def = pass.Pass{ .name = "simplify", .run = run };

fn isInt(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int, .bool => true,
        else => false,
    };
}

const Simplified = union(enum) { none, value: Value, constant: i64 };

/// The identity for `lhs op rhs`, given each operand's constant value if known (`lc`/`rc`) and
/// whether `lhs` and `rhs` are the same SSA value (`same`). For `arith_imm`, `rhs` is null and the
/// immediate is passed as `rc`. Only identities that hold for every value of the variable operand.
fn simplify(op: BinOp, lhs: Value, rhs: ?Value, lc: ?i64, rc: ?i64, same: bool) Simplified {
    const other: Simplified = if (rhs) |r| .{ .value = r } else .none; // the rhs value (arith only)
    switch (op) {
        .add => {
            if (rc == 0) return .{ .value = lhs }; // x + 0
            if (lc == 0) return other; // 0 + x
        },
        .sub => {
            if (rc == 0) return .{ .value = lhs }; // x - 0
            if (same) return .{ .constant = 0 }; // x - x
        },
        .mul => {
            if (lc == 0 or rc == 0) return .{ .constant = 0 }; // x * 0 / 0 * x
            if (rc == 1) return .{ .value = lhs }; // x * 1
            if (lc == 1) return other; // 1 * x
        },
        .bit_and => {
            if (lc == 0 or rc == 0) return .{ .constant = 0 }; // x & 0
            if (same) return .{ .value = lhs }; // x & x
        },
        .bit_or => {
            if (rc == 0) return .{ .value = lhs }; // x | 0
            if (lc == 0) return other; // 0 | x
            if (same) return .{ .value = lhs }; // x | x
        },
        .bit_xor => {
            if (rc == 0) return .{ .value = lhs }; // x ^ 0
            if (lc == 0) return other; // 0 ^ x
            if (same) return .{ .constant = 0 }; // x ^ x
        },
        .shl, .shr => {
            if (rc == 0) return .{ .value = lhs }; // x << 0 / x >> 0
            if (lc == 0) return .{ .constant = 0 }; // 0 << y / 0 >> y
        },
        .div => if (rc == 1) return .{ .value = lhs }, // x / 1  (x / x is UB when x == 0, so skipped)
        .rem => if (rc == 1) return .{ .constant = 0 }, // x % 1
    }
    return .none;
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;

    // Map each value defined by an `iconst` to its constant (to spot 0/1 operands).
    var consts = try allocator.alloc(?i64, func.valueCount());
    defer allocator.free(consts);
    @memset(consts, null);
    for (0..func.instCount()) |i| {
        const inst: ir.function.Inst = @enumFromInt(i);
        if (func.opcode(inst) == .iconst) {
            if (func.instResult(inst)) |r| consts[@intFromEnum(r)] = func.opcode(inst).iconst;
        }
    }

    var changed = false;
    for (0..func.instCount()) |i| {
        const inst: ir.function.Inst = @enumFromInt(i);
        const result = func.instResult(inst) orelse continue;
        if (consts[@intFromEnum(result)] != null) continue; // already a constant

        // `select` folds for any type since it just picks an existing value. A constant condition
        // resolves to one arm, and identical arms collapse to that value.
        if (func.opcode(inst) == .select) {
            const sel = func.opcode(inst).select;
            const repl: ?Value = if (sel.then == sel.@"else")
                sel.then // select(c, x, x) -> x
            else if (consts[@intFromEnum(sel.cond)]) |cv|
                (if (cv != 0) sel.then else sel.@"else") // select(const, a, b) -> a / b
            else
                null;
            if (repl) |v| {
                func.replaceAllUses(result, v);
                changed = true;
            }
            continue;
        }

        if (!isInt(func, result)) continue; // the identities below are integer-only
        const s: Simplified = switch (func.opcode(inst)) {
            .arith => |a| simplify(a.op, a.lhs, a.rhs, consts[@intFromEnum(a.lhs)], consts[@intFromEnum(a.rhs)], a.lhs == a.rhs),
            .arith_imm => |a| simplify(a.op, a.lhs, null, consts[@intFromEnum(a.lhs)], a.imm, false),
            // A comparison of a value with itself is constant (icmp is integer, so no NaN caveat).
            .icmp => |c| if (c.lhs == c.rhs) Simplified{ .constant = switch (c.op) {
                .eq, .le, .ge => @as(i64, 1),
                .ne, .lt, .gt => @as(i64, 0),
            } } else .none,
            else => .none,
        };
        switch (s) {
            .none => {},
            .value => |v| {
                func.replaceAllUses(result, v); // the defining instruction is now dead (DCE removes it)
                changed = true;
            },
            .constant => |c| {
                func.opcodeMut(inst).* = .{ .iconst = c };
                consts[@intFromEnum(result)] = c;
                changed = true;
            },
        }
    }
    return changed;
}

const testing = std.testing;

/// Build a one-block function with an i32 param, returning the block, the param value, and its type.
fn oneParam(func: *Function) !struct { b: ir.function.Block, x: Value, t: ir.types.Type } {
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    return .{ .b = b, .x = x, .t = t };
}

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

test "x + 0 simplifies to x (the return now yields x directly)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const s = try oneParam(&func);
    const zero = try func.appendInst(s.b, s.t, .{ .iconst = 0 });
    const y = try func.appendInst(s.b, s.t, .{ .arith = .{ .op = .add, .lhs = s.x, .rhs = zero } });
    func.setTerminator(s.b, .{ .ret = y });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(s.x, func.terminator(s.b).?.ret.?); // ret x, not ret (x+0)
}

test "x * 0 simplifies to the constant 0" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const s = try oneParam(&func);
    const zero = try func.appendInst(s.b, s.t, .{ .iconst = 0 });
    const y = try func.appendInst(s.b, s.t, .{ .arith = .{ .op = .mul, .lhs = s.x, .rhs = zero } });
    func.setTerminator(s.b, .{ .ret = y });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(@as(i64, 0), func.opcode(func.definingInst(y).?).iconst);
}

test "x - x simplifies to the constant 0" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const s = try oneParam(&func);
    const y = try func.appendInst(s.b, s.t, .{ .arith = .{ .op = .sub, .lhs = s.x, .rhs = s.x } });
    func.setTerminator(s.b, .{ .ret = y });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(@as(i64, 0), func.opcode(func.definingInst(y).?).iconst);
}

test "arith_imm: x * 1 simplifies to x, x & x to x" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const s = try oneParam(&func);
    const one_mul = try func.appendArithImm(s.b, s.t, .mul, s.x, 1); // x * 1 -> x
    const anded = try func.appendInst(s.b, s.t, .{ .arith = .{ .op = .bit_and, .lhs = one_mul, .rhs = one_mul } }); // (x)&(x) -> x
    func.setTerminator(s.b, .{ .ret = anded });

    try testing.expect(try runOnce(allocator, &func));
    // x*1 -> x turns `anded` into x & x, and x & x -> x, so the return is x.
    try testing.expectEqual(s.x, func.terminator(s.b).?.ret.?);
}

test "no change when there is nothing to simplify" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const s = try oneParam(&func);
    const two = try func.appendInst(s.b, s.t, .{ .iconst = 2 });
    const y = try func.appendInst(s.b, s.t, .{ .arith = .{ .op = .mul, .lhs = s.x, .rhs = two } }); // x * 2: kept
    func.setTerminator(s.b, .{ .ret = y });
    try testing.expect(!try runOnce(allocator, &func));
}

test "float x + 0.0 is left alone (unsound to simplify)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const zero = try func.appendInst(b, t, .{ .fconst = 0.0 });
    const y = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = zero } });
    func.setTerminator(b, .{ .ret = y });
    try testing.expect(!try runOnce(allocator, &func)); // floats are not simplified
}

test "select(true, a, b) folds to a and select(false, a, b) folds to b" {
    const allocator = testing.allocator;
    inline for (.{ .{ 1, true }, .{ 0, false } }) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try oneParam(&func);
        const y = try func.appendBlockParam(s.b, s.t);
        const bool_t = try func.types.intern(.bool);
        const cond = try func.appendInst(s.b, bool_t, .{ .iconst = case[0] });
        const sel = try func.appendInst(s.b, s.t, .{ .select = .{ .cond = cond, .then = s.x, .@"else" = y } });
        func.setTerminator(s.b, .{ .ret = sel });
        try testing.expect(try runOnce(allocator, &func));
        const expected = if (case[1]) s.x else y;
        try testing.expectEqual(expected, func.terminator(s.b).?.ret.?);
    }
}

test "select(c, x, x) folds to x for a non-constant condition" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const s = try oneParam(&func);
    const bool_t = try func.types.intern(.bool);
    const cond = try func.appendBlockParam(s.b, bool_t); // runtime condition
    const sel = try func.appendInst(s.b, s.t, .{ .select = .{ .cond = cond, .then = s.x, .@"else" = s.x } });
    func.setTerminator(s.b, .{ .ret = sel });
    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(s.x, func.terminator(s.b).?.ret.?);
}

test "self-comparison folds to a constant bool" {
    const allocator = testing.allocator;
    const bool_t_kind = @as(ir.types.TypeKind, .bool);
    const Case = struct { op: ir.function.CmpOp, expect: i64 };
    inline for (.{
        Case{ .op = .eq, .expect = 1 }, Case{ .op = .ne, .expect = 0 },
        Case{ .op = .lt, .expect = 0 }, Case{ .op = .le, .expect = 1 },
        Case{ .op = .gt, .expect = 0 }, Case{ .op = .ge, .expect = 1 },
    }) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try oneParam(&func);
        const bool_t = try func.types.intern(bool_t_kind);
        const cmp = try func.appendInst(s.b, bool_t, .{ .icmp = .{ .op = case.op, .lhs = s.x, .rhs = s.x } });
        func.setTerminator(s.b, .{ .ret = cmp });
        try testing.expect(try runOnce(allocator, &func));
        try testing.expectEqual(case.expect, func.opcode(func.definingInst(cmp).?).iconst);
    }
}
