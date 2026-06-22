//! Constant folding: evaluate `arith`/`arith_imm`/`icmp` instructions whose
//! operands are compile-time constants, replacing them in place. Width-aware:
//! results wrap to the operand's bit width and signedness.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const BinOp = ir.function.BinOp;
const CmpOp = ir.function.CmpOp;

pub const pass_def = pass.Pass{ .name = "constfold", .run = run };

/// Wrap `v` to `bits` bits with `signedness`, returning the canonical `i64` an
/// `iconst` of that type holds.
fn wrap(v: i64, bits: u16, signedness: std.builtin.Signedness) i64 {
    if (bits >= 64) return v;
    const mask: u64 = (@as(u64, 1) << @intCast(bits)) - 1;
    const low: u64 = @as(u64, @bitCast(v)) & mask;
    return switch (signedness) {
        .unsigned => @bitCast(low),
        .signed => blk: {
            const sign_bit: u64 = @as(u64, 1) << @intCast(bits - 1);
            break :blk @bitCast(if (low & sign_bit != 0) low | ~mask else low);
        },
    };
}

const IntTy = struct { bits: u16, signedness: std.builtin.Signedness };

fn intType(func: *const Function, v: Value) ?IntTy {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| .{ .bits = i.bits, .signedness = i.signedness },
        .bool => .{ .bits = 1, .signedness = .unsigned },
        else => null,
    };
}

/// Evaluate `lhs op rhs` at width `ty`, or null if it cannot be folded safely.
fn evalBin(op: BinOp, lhs: i64, rhs: i64, ty: IntTy) ?i64 {
    const a: u64 = @bitCast(lhs);
    const b: u64 = @bitCast(rhs);
    const raw: i64 = switch (op) {
        .add => @bitCast(a +% b),
        .sub => @bitCast(a -% b),
        .mul => @bitCast(a *% b),
        .bit_and => @bitCast(a & b),
        .bit_or => @bitCast(a | b),
        .bit_xor => @bitCast(a ^ b),
        .shl => @bitCast(a << @intCast(@as(u64, @bitCast(rhs)) & 63)),
        .shr => switch (ty.signedness) {
            .unsigned => @bitCast((wrapU(a, ty.bits)) >> @intCast(@as(u64, @bitCast(rhs)) & 63)),
            .signed => wrap(lhs, ty.bits, .signed) >> @intCast(@as(u64, @bitCast(rhs)) & 63),
        },
        .div, .rem => blk: {
            if (rhs == 0) return null; // division by zero: leave it to runtime
            const l = wrap(lhs, ty.bits, ty.signedness);
            const r = wrap(rhs, ty.bits, ty.signedness);
            break :blk switch (ty.signedness) {
                .signed => sblk: {
                    if (r == -1 and l == std.math.minInt(i64)) return null; // overflow
                    break :sblk if (op == .div) @divTrunc(l, r) else @rem(l, r);
                },
                .unsigned => ublk: {
                    const lu: u64 = wrapU(@bitCast(l), ty.bits);
                    const ru: u64 = wrapU(@bitCast(r), ty.bits);
                    break :ublk @bitCast(if (op == .div) lu / ru else lu % ru);
                },
            };
        },
    };
    return wrap(raw, ty.bits, ty.signedness);
}

fn wrapU(v: u64, bits: u16) u64 {
    if (bits >= 64) return v;
    return v & ((@as(u64, 1) << @intCast(bits)) - 1);
}

fn evalCmp(op: CmpOp, lhs: i64, rhs: i64, ty: IntTy) i64 {
    const l = wrap(lhs, ty.bits, ty.signedness);
    const r = wrap(rhs, ty.bits, ty.signedness);
    const result = switch (ty.signedness) {
        .signed => switch (op) {
            .eq => l == r,
            .ne => l != r,
            .lt => l < r,
            .le => l <= r,
            .gt => l > r,
            .ge => l >= r,
        },
        .unsigned => blk: {
            const lu = wrapU(@bitCast(l), ty.bits);
            const ru = wrapU(@bitCast(r), ty.bits);
            break :blk switch (op) {
                .eq => lu == ru,
                .ne => lu != ru,
                .lt => lu < ru,
                .le => lu <= ru,
                .gt => lu > ru,
                .ge => lu >= ru,
            };
        },
    };
    return @intFromBool(result);
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;

    // Map every value defined by an `iconst` to its constant.
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
    // A folded value can feed another fold, so iterate to a local fixpoint.
    var again = true;
    while (again) {
        again = false;
        for (0..func.instCount()) |i| {
            const inst: ir.function.Inst = @enumFromInt(i);
            const result = func.instResult(inst) orelse continue;
            if (consts[@intFromEnum(result)] != null) continue; // already a constant
            const folded: ?i64 = switch (func.opcode(inst)) {
                .arith => |a| blk: {
                    const lc = consts[@intFromEnum(a.lhs)] orelse break :blk null;
                    const rc = consts[@intFromEnum(a.rhs)] orelse break :blk null;
                    const ty = intType(func, result) orelse break :blk null;
                    break :blk evalBin(a.op, lc, rc, ty);
                },
                .arith_imm => |a| blk: {
                    const lc = consts[@intFromEnum(a.lhs)] orelse break :blk null;
                    const ty = intType(func, result) orelse break :blk null;
                    break :blk evalBin(a.op, lc, a.imm, ty);
                },
                .icmp => |c| blk: {
                    const lc = consts[@intFromEnum(c.lhs)] orelse break :blk null;
                    const rc = consts[@intFromEnum(c.rhs)] orelse break :blk null;
                    const ty = intType(func, c.lhs) orelse break :blk null;
                    break :blk evalCmp(c.op, lc, rc, ty);
                },
                else => null,
            };
            if (folded) |value| {
                func.opcodeMut(inst).* = .{ .iconst = value };
                consts[@intFromEnum(result)] = value;
                changed = true;
                again = true;
            }
        }
    }
    return changed;
}

test "folds nested integer arithmetic to a constant" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const c2 = try func.appendInst(b, i32_t, .{ .iconst = 2 });
    const c3 = try func.appendInst(b, i32_t, .{ .iconst = 3 });
    const sum = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = c2, .rhs = c3 } });
    const prod = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .mul, .lhs = sum, .rhs = c2 } });
    func.setTerminator(b, .{ .ret = prod });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));

    // (2 + 3) * 2 == 10, and prod is now an iconst.
    try std.testing.expectEqual(@as(i64, 10), func.opcode(func.definingInst(prod).?).iconst);
}

test "folds an unsigned comparison to a bool constant" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const c200 = try func.appendInst(b, u8_t, .{ .iconst = 200 });
    const c100 = try func.appendInst(b, u8_t, .{ .iconst = 100 });
    const cmp = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .gt, .lhs = c200, .rhs = c100 } });
    func.setTerminator(b, .{ .ret = cmp });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(i64, 1), func.opcode(func.definingInst(cmp).?).iconst); // 200 > 100 unsigned
}
