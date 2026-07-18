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

    // Second phase: replace 64-bit division/remainder by a non-power-of-two constant with a
    // magic-number multiply (Granlund-Montgomery). This needs a multi-instruction sequence, so it
    // rebuilds the instruction list of any block that holds an eligible div/rem. Width is limited to
    // 64 because the `mulh` high-multiply this emits is defined only on 64-bit operands; 32-bit and
    // below want a widening multiply, which waits on int->int widening `convert` in the backends.
    if (try magicPhase(allocator, func, consts)) changed = true;
    return changed;
}

/// Rewrite each eligible 64-bit `div`/`rem` by a constant into a magic-number multiply. Returns
/// whether anything changed. `consts` maps each value to its `iconst`, if any.
fn magicPhase(allocator: std.mem.Allocator, func: *Function, consts: []const ?i64) pass.Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @enumFromInt(bi);
        var has = false;
        for (func.blockInsts(block)) |inst| {
            if (magicDivisor(func, consts, inst) != null) {
                has = true;
                break;
            }
        }
        if (!has) continue;
        changed = true;

        var out: std.ArrayList(ir.function.Inst) = .empty;
        defer out.deinit(allocator);
        const original = try allocator.dupe(ir.function.Inst, func.blockInsts(block));
        defer allocator.free(original);
        for (original) |inst| {
            const d = magicDivisor(func, consts, inst) orelse {
                try out.append(allocator, inst);
                continue;
            };
            const a = func.opcode(inst).arith;
            const ty = intType(func, a.lhs).?;
            const result = func.instResult(inst).?;
            const value = try emitMagic(func, &out, allocator, a.lhs, d, ty, a.op == .rem);
            func.replaceAllUses(result, value);
        }
        try func.setBlockInsts(block, out.items);
    }
    return changed;
}

/// The constant divisor of `inst` if it is a 64-bit `div`/`rem` by a non-power-of-two constant of
/// magnitude at least 3 (the case the magic lowering handles), else null.
fn magicDivisor(func: *const Function, consts: []const ?i64, inst: ir.function.Inst) ?i64 {
    const a = switch (func.opcode(inst)) {
        .arith => |ar| ar,
        else => return null,
    };
    if (a.op != .div and a.op != .rem) return null;
    const ty = intType(func, a.lhs) orelse return null;
    if (ty.bits != 64) return null;
    const d = consts[@intFromEnum(a.rhs)] orelse return null;
    const ad: u64 = @abs(d);
    if (ad < 3) return null; // 0/1 are handled elsewhere, 2 is a power of two
    if (ad & (ad - 1) == 0) return null; // a power of two: the shift path already handled it
    return d;
}

/// Emit the magic-number sequence for `x op d` (quotient, or remainder when `is_rem`) at type `ty`,
/// appending instructions to `out`, and return the value holding the result.
fn emitMagic(func: *Function, out: *std.ArrayList(ir.function.Inst), allocator: std.mem.Allocator, x: Value, d: i64, ty: IntTy, is_rem: bool) pass.Error!Value {
    const t = func.valueType(x);
    const bld = Builder{ .f = func, .o = out, .a = allocator, .ty = t };
    const q = if (ty.signedness == .signed)
        try emitSignedQuotient(bld, x, d)
    else
        try emitUnsignedQuotient(bld, x, @bitCast(d));
    if (!is_rem) return q;
    // remainder = x - q*d
    const dc = try bld.konst(d);
    const qd = try bld.op(.mul, q, dc);
    return bld.op(.sub, x, qd);
}

/// A small emit helper binding the function, output list, allocator, and result type.
const Builder = struct {
    f: *Function,
    o: *std.ArrayList(ir.function.Inst),
    a: std.mem.Allocator,
    ty: ir.types.Type,
    fn konst(self: Builder, c: i64) pass.Error!Value {
        const v = try self.f.createInst(self.ty, .{ .iconst = c });
        try self.o.append(self.a, self.f.definingInst(v).?);
        return v;
    }
    fn op(self: Builder, o: BinOp, lhs: Value, rhs: Value) pass.Error!Value {
        const v = try self.f.createInst(self.ty, .{ .arith = .{ .op = o, .lhs = lhs, .rhs = rhs } });
        try self.o.append(self.a, self.f.definingInst(v).?);
        return v;
    }
    fn opImm(self: Builder, o: BinOp, lhs: Value, imm: i64) pass.Error!Value {
        const v = try self.f.createInst(self.ty, .{ .arith_imm = .{ .op = o, .lhs = lhs, .imm = imm } });
        try self.o.append(self.a, self.f.definingInst(v).?);
        return v;
    }
};

fn emitSignedQuotient(b: Builder, x: Value, d: i64) pass.Error!Value {
    const mag = signedMagic(d);
    const mc = try b.konst(mag.m);
    var q = try b.op(.mulh, x, mc);
    if (d > 0 and mag.m < 0) q = try b.op(.add, q, x);
    if (d < 0 and mag.m > 0) q = try b.op(.sub, q, x);
    if (mag.s > 0) q = try b.opImm(.shr, q, mag.s); // arithmetic shift (signed type)
    // Add 1 when q is negative: an arithmetic shift by 63 gives 0 or -1, subtracting it adds the 1.
    const sgn = try b.opImm(.shr, q, 63);
    return b.op(.sub, q, sgn);
}

fn emitUnsignedQuotient(b: Builder, x: Value, d: u64) pass.Error!Value {
    const mag = unsignedMagic(d);
    const mc = try b.konst(@bitCast(mag.m));
    const hi = try b.op(.mulh, x, mc); // unsigned high multiply (mulhu)
    if (!mag.add) {
        return if (mag.s > 0) b.opImm(.shr, hi, mag.s) else hi; // logical shift (unsigned type)
    }
    // The multiplier needed an extra bit: q = ((x - hi) >> 1 + hi) >> (s-1).
    const diff = try b.op(.sub, x, hi);
    const half = try b.opImm(.shr, diff, 1);
    const sum = try b.op(.add, half, hi);
    return b.opImm(.shr, sum, @as(i64, mag.s) - 1);
}

/// Granlund-Montgomery signed magic for a 64-bit divisor (Hacker's Delight ch.10, 64-bit form).
/// Returns the multiplier `m` and post-shift `s` so `x / d == addBack(mulhs(x, m)) >> s` with a
/// final "add the sign bit" correction. Computed in 128-bit to sidestep the intermediate overflows.
fn signedMagic(d: i64) struct { m: i64, s: u6 } {
    const w: u8 = 64;
    const ad: u128 = @abs(d);
    const two_w1: u128 = @as(u128, 1) << (w - 1); // 2^63
    const t: u128 = two_w1 + (@as(u64, @bitCast(d)) >> 63);
    const anc: u128 = t - 1 - t % ad;
    var p: u8 = w - 1;
    var q1: u128 = two_w1 / anc;
    var r1: u128 = two_w1 - q1 * anc;
    var q2: u128 = two_w1 / ad;
    var r2: u128 = two_w1 - q2 * ad;
    // The algorithm converges with p at most 2*w for any |d| >= 2 (guaranteed by magicDivisor); the
    // bound is a hard cap so a malformed input can never spin, per IronStyle's "bound everything".
    while (p < 2 * w) {
        p += 1;
        q1 *= 2;
        r1 *= 2;
        if (r1 >= anc) {
            q1 += 1;
            r1 -= anc;
        }
        q2 *= 2;
        r2 *= 2;
        if (r2 >= ad) {
            q2 += 1;
            r2 -= ad;
        }
        const delta = ad - r2;
        if (!(q1 < delta or (q1 == delta and r1 == 0))) break;
    }
    var m: i64 = @bitCast(@as(u64, @truncate(q2 + 1)));
    if (d < 0) m = -%m;
    return .{ .m = m, .s = @intCast(p - w) };
}

/// Granlund-Montgomery unsigned magic for a 64-bit divisor (Hacker's Delight ch.10, 64-bit form).
/// Returns `m`, post-shift `s`, and `add` (whether the multiplier overflowed 64 bits and needs the
/// add-back variant). Computed in 128-bit so the doublings never overflow.
fn unsignedMagic(d: u64) struct { m: u64, s: u6, add: bool } {
    const w: u8 = 64;
    const two_w1: u128 = @as(u128, 1) << (w - 1); // 2^63
    const two_w: u128 = @as(u128, 1) << w; // 2^64
    const dd: u128 = d;
    const nc: u128 = (two_w - 1) - (two_w % dd);
    var p: u8 = w - 1;
    var add = false;
    var q1: u128 = two_w1 / nc;
    var r1: u128 = two_w1 - q1 * nc;
    var q2: u128 = (two_w1 - 1) / dd;
    var r2: u128 = (two_w1 - 1) - q2 * dd;
    while (true) {
        p += 1;
        if (r1 >= nc - r1) {
            q1 = 2 * q1 + 1;
            r1 = 2 * r1 - nc;
        } else {
            q1 = 2 * q1;
            r1 = 2 * r1;
        }
        if (r2 + 1 >= dd - r2) {
            if (q2 >= two_w1 - 1) add = true;
            q2 = 2 * q2 + 1;
            r2 = 2 * r2 + 1 - dd;
        } else {
            if (q2 >= two_w1) add = true;
            q2 = 2 * q2;
            r2 = 2 * r2 + 1;
        }
        const delta = dd - 1 - r2;
        if (!(p < 2 * w and (q1 < delta or (q1 == delta and r1 == 0)))) break;
    }
    return .{ .m = @truncate(q2 + 1), .s = @intCast(p - w), .add = add };
}

const testing = std.testing;
const constfold = @import("constfold.zig");
const dce = @import("dce.zig");

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

/// Apply the magic-number lowering to `x op d` at 64-bit `signedness` with both operands constant,
/// then constant-fold the emitted sequence to a single value. Returns that folded result, which must
/// equal the real division for the lowering to be correct. `x`/`d` carry the raw 64-bit bit pattern.
fn foldMagic(allocator: std.mem.Allocator, x: i64, d: i64, signedness: std.builtin.Signedness, is_rem: bool) !i64 {
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, 64, signedness);
    const b = try func.appendBlock();
    const xc = try func.appendInst(b, t, .{ .iconst = x });
    const dc = try func.appendInst(b, t, .{ .iconst = d });
    const r = try func.appendInst(b, t, .{ .arith = .{ .op = if (is_rem) .rem else .div, .lhs = xc, .rhs = dc } });
    func.setTerminator(b, .{ .ret = r });

    const pipeline = [_]pass.Pass{ pass_def, constfold.pass_def, dce.pass_def };
    _ = try pass.runToFixpoint(allocator, &func, &pipeline, 16);

    const ret = func.terminator(b).?.ret.?;
    const inst = func.definingInst(ret) orelse return error.NotFolded;
    return switch (func.opcode(inst)) {
        .iconst => |c| c,
        else => error.NotFolded, // the whole magic chain must fold to a constant
    };
}

fn expectDivU(allocator: std.mem.Allocator, x: u64, d: u64) !void {
    const got: u64 = @bitCast(try foldMagic(allocator, @bitCast(x), @bitCast(d), .unsigned, false));
    try testing.expectEqual(x / d, got);
    const gotr: u64 = @bitCast(try foldMagic(allocator, @bitCast(x), @bitCast(d), .unsigned, true));
    try testing.expectEqual(x % d, gotr);
}

fn expectDivS(allocator: std.mem.Allocator, x: i64, d: i64) !void {
    const got = try foldMagic(allocator, x, d, .signed, false);
    try testing.expectEqual(@divTrunc(x, d), got);
    const gotr = try foldMagic(allocator, x, d, .signed, true);
    try testing.expectEqual(@rem(x, d), gotr);
}

test "unsigned 64-bit magic division matches real division across divisors and dividends" {
    const allocator = testing.allocator;
    const divisors = [_]u64{ 3, 5, 6, 7, 9, 10, 11, 100, 1000, 7919, 0xFFFFFFFF };
    const dividends = [_]u64{ 0, 1, 2, 9, 10, 99, 100, 101, 1_000_000_000_000, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000 };
    for (divisors) |d| {
        for (dividends) |x| try expectDivU(allocator, x, d);
    }
}

test "signed 64-bit magic division matches real division across divisors and dividends" {
    const allocator = testing.allocator;
    const divisors = [_]i64{ 3, 5, 7, 10, 100, 1000, -3, -7, -100 };
    const dividends = [_]i64{ 0, 1, -1, 9, -9, 100, -100, 101, -101, 1_000_000_000_000, -1_000_000_000_000, std.math.maxInt(i64), std.math.minInt(i64) };
    for (divisors) |d| {
        for (dividends) |x| try expectDivS(allocator, x, d);
    }
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
