//! Strength reduction replaces an integer multiply, divide, or remainder by a constant power of two
//! with a cheap shift or mask.
//!   x * 2^k  -> x << k      any signedness, the low bits are identical
//!   x /u 2^k -> x >> k      unsigned only, a logical shift. signed division rounds differently
//!   x %u 2^k -> x & (2^k-1) unsigned only
//! This pays off most where division is slow or absent. On the NVIDIA GPU path it sidesteps the
//! roughly 256-instruction `lowerdiv` expansion entirely. It handles both `arith_imm` and an `arith`
//! with a constant operand, where DCE later cleans the now-dead constant. Signed div/rem by a power
//! of two is not a plain shift, since it rounds toward zero rather than toward negative infinity, so
//! it is left alone.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const BinOp = ir.function.BinOp;
const Opcode = ir.function.Opcode;

pub const pass_def = pass.Pass{ .name = "strength", .run = run };

const IntTy = struct { bits: u16, signedness: std.builtin.Signedness };

fn intType(func: *const Function, v: Value) ?IntTy {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| .{ .bits = i.bits, .signedness = i.signedness },
        else => null, // bool / float / vector: not reduced here
    };
}

/// If `c` is 2^k with 1 <= k < bits, return k, otherwise null. Bounding k below the width keeps the
/// shift inside the type so it matches the wrapped multiply or divide it replaces.
fn log2Pow2(c: i64, bits: u16) ?u6 {
    if (c <= 1) return null;
    const u: u64 = @bitCast(c);
    if (u & (u - 1) != 0) return null; // not a single power of two
    const k = @ctz(u);
    if (k == 0 or k >= bits) return null;
    return @intCast(k);
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;

    // Constant values (to spot a power-of-two operand of an `arith`).
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
        // Extract (op, x, c) where `c` is the constant operand, or skip.
        var op: BinOp = undefined;
        var x: Value = undefined;
        var c: i64 = undefined;
        switch (func.opcode(inst)) {
            .arith_imm => |a| {
                op = a.op;
                x = a.lhs;
                c = a.imm;
            },
            .arith => |a| {
                op = a.op;
                if (a.op == .mul) { // commutative: either operand may be the constant
                    if (consts[@intFromEnum(a.rhs)]) |cc| {
                        x = a.lhs;
                        c = cc;
                    } else if (consts[@intFromEnum(a.lhs)]) |cc| {
                        x = a.rhs;
                        c = cc;
                    } else continue;
                } else if (a.op == .div or a.op == .rem) { // only the divisor (rhs)
                    x = a.lhs;
                    c = consts[@intFromEnum(a.rhs)] orelse continue;
                } else continue;
            },
            else => continue,
        }
        if (op != .mul and op != .div and op != .rem) continue;
        const ty = intType(func, x) orelse continue;
        const k = log2Pow2(c, ty.bits) orelse continue;

        const new: ?Opcode = switch (op) {
            .mul => .{ .arith_imm = .{ .op = .shl, .lhs = x, .imm = @as(i64, k) } },
            .div => if (ty.signedness == .unsigned) Opcode{ .arith_imm = .{ .op = .shr, .lhs = x, .imm = @as(i64, k) } } else null,
            .rem => if (ty.signedness == .unsigned) Opcode{ .arith_imm = .{ .op = .bit_and, .lhs = x, .imm = @bitCast((@as(u64, 1) << k) - 1) } } else null,
            else => null,
        };
        if (new) |n| {
            func.opcodeMut(inst).* = n;
            changed = true;
        }
    }
    return changed;
}

const testing = std.testing;

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

fn intTy(func: *Function, bits: u16, signedness: std.builtin.Signedness) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = signedness, .bits = bits } });
}

test "x * 8 becomes x << 3" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendArithImm(b, t, .mul, x, 8);
    func.setTerminator(b, .{ .ret = y });

    try testing.expect(try runOnce(allocator, &func));
    const a = func.opcode(func.definingInst(y).?).arith_imm;
    try testing.expectEqual(BinOp.shl, a.op);
    try testing.expectEqual(@as(i64, 3), a.imm);
}

test "unsigned x / 4 becomes x >> 2, x % 4 becomes x & 3" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .unsigned);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const q = try func.appendArithImm(b, t, .div, x, 4);
    const r = try func.appendArithImm(b, t, .rem, x, 4);
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = q, .rhs = r } });
    func.setTerminator(b, .{ .ret = s });

    try testing.expect(try runOnce(allocator, &func));
    const qi = func.opcode(func.definingInst(q).?).arith_imm;
    try testing.expectEqual(BinOp.shr, qi.op);
    try testing.expectEqual(@as(i64, 2), qi.imm);
    const ri = func.opcode(func.definingInst(r).?).arith_imm;
    try testing.expectEqual(BinOp.bit_and, ri.op);
    try testing.expectEqual(@as(i64, 3), ri.imm);
}

test "signed division by a power of two is left alone (rounding differs)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const q = try func.appendArithImm(b, t, .div, x, 4);
    func.setTerminator(b, .{ .ret = q });
    try testing.expect(!try runOnce(allocator, &func));
    try testing.expectEqual(BinOp.div, func.opcode(func.definingInst(q).?).arith_imm.op);
}

test "non-power-of-two multiply is left alone" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendArithImm(b, t, .mul, x, 3);
    func.setTerminator(b, .{ .ret = y });
    try testing.expect(!try runOnce(allocator, &func));
}

test "arith form: x * (iconst 16) reduces via the constant operand" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 32, .signed);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const c16 = try func.appendInst(b, t, .{ .iconst = 16 });
    const y = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = c16 } });
    func.setTerminator(b, .{ .ret = y });

    try testing.expect(try runOnce(allocator, &func));
    const a = func.opcode(func.definingInst(y).?).arith_imm; // rewritten to arith_imm shl
    try testing.expectEqual(BinOp.shl, a.op);
    try testing.expectEqual(@as(i64, 4), a.imm);
    try testing.expectEqual(x, a.lhs);
}
