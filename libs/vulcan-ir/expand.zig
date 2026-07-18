//! Expands the `mulh` BinOp (high half of a full-width product) into plain multiplies, shifts, and
//! masks, for backends that have no high-multiply instruction (wasm/spirv/x86/x86_64/nvidia). The
//! two native scalar backends (aarch64 smulh/umulh, riscv64 mulh/mulhu) lower `mulh` directly and
//! never call this. Producing `mulh` is the magic-number divide lowering's job (`strength.zig`); a
//! backend without native support runs this once before instruction selection so its isel never
//! meets a `mulh`.
//!
//! The high half is computed from half-width limbs. For a W-bit value the limbs are H = W/2 bits:
//! splitting a = ahi*2^H + alo and b likewise, the full product's high W bits are
//!   hihi + (lohi >> H) + (hilo >> H) + (((lolo >> H) + (lohi & m) + (hilo & m)) >> H)
//! where lolo = alo*blo, lohi = alo*bhi, hilo = ahi*blo, hihi = ahi*bhi, m = 2^H - 1. Every shift is
//! masked back to H bits, so an arithmetic right shift is fine even on a signed type (the sign fill
//! lands above bit H and is masked away), which is why no unsigned reinterpret is needed. For a
//! signed `mulh` the unsigned high half is corrected by `- (a<0 ? b : 0) - (b<0 ? a : 0)`.

const std = @import("std");
const function = @import("function.zig");
const types = @import("types.zig");

const Function = function.Function;
const Value = function.Value;
const Block = function.Block;
const Inst = function.Inst;
const BinOp = function.BinOp;

/// Rewrite every `arith` with op `.mulh` in `func` into an equivalent limb sequence. Returns
/// whether anything was rewritten.
pub fn expandMulh(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        // Does this block hold a mulh? Rebuild its instruction list only if so.
        var has = false;
        for (func.blockInsts(block)) |inst| {
            if (isMulh(func, inst)) {
                has = true;
                break;
            }
        }
        if (!has) continue;
        changed = true;

        var out: std.ArrayList(Inst) = .empty;
        defer out.deinit(allocator);
        // Snapshot the original list: appending new insts to the function's pool must not perturb
        // the sequence we are iterating.
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        for (original) |inst| {
            if (!isMulh(func, inst)) {
                try out.append(allocator, inst);
                continue;
            }
            const a = func.opcode(inst).arith;
            const result = func.instResult(inst).?;
            const high = try emitLimbs(func, &out, allocator, a.lhs, a.rhs, func.valueType(result));
            func.replaceAllUses(result, high);
        }
        try func.setBlockInsts(block, out.items);
    }
    return changed;
}

fn isMulh(func: *const Function, inst: Inst) bool {
    return switch (func.opcode(inst)) {
        .arith => |a| a.op == .mulh,
        else => false,
    };
}

/// Emit the limb sequence for `high half of (lhs * rhs)` at type `ty`, appending each instruction
/// to `out`, and return the value holding the high half.
fn emitLimbs(func: *Function, out: *std.ArrayList(Inst), allocator: std.mem.Allocator, lhs: Value, rhs: Value, ty: types.Type) std.mem.Allocator.Error!Value {
    const info = switch (func.types.type_kind(ty)) {
        .int => |i| i,
        else => unreachable, // mulh is integer-only (verify/strength guarantee it)
    };
    const w: u16 = info.bits;
    const h: i64 = @intCast(w / 2);
    const mask_h: i64 = (@as(i64, 1) << @intCast(w / 2)) - 1;

    const b = struct {
        f: *Function,
        o: *std.ArrayList(Inst),
        a: std.mem.Allocator,
        ty: types.Type,
        fn konst(self: @This(), c: i64) std.mem.Allocator.Error!Value {
            const v = try self.f.createInst(self.ty, .{ .iconst = c });
            try self.o.append(self.a, self.f.definingInst(v).?);
            return v;
        }
        fn op(self: @This(), o: BinOp, x: Value, y: Value) std.mem.Allocator.Error!Value {
            const v = try self.f.createInst(self.ty, .{ .arith = .{ .op = o, .lhs = x, .rhs = y } });
            try self.o.append(self.a, self.f.definingInst(v).?);
            return v;
        }
    }{ .f = func, .o = out, .a = allocator, .ty = ty };

    const m = try b.konst(mask_h);
    const hs = try b.konst(h);

    const alo = try b.op(.bit_and, lhs, m);
    const ahi_s = try b.op(.shr, lhs, hs);
    const ahi = try b.op(.bit_and, ahi_s, m);
    const blo = try b.op(.bit_and, rhs, m);
    const bhi_s = try b.op(.shr, rhs, hs);
    const bhi = try b.op(.bit_and, bhi_s, m);

    const lolo = try b.op(.mul, alo, blo);
    const lohi = try b.op(.mul, alo, bhi);
    const hilo = try b.op(.mul, ahi, blo);
    const hihi = try b.op(.mul, ahi, bhi);

    const lolo_hi_s = try b.op(.shr, lolo, hs);
    const lolo_hi = try b.op(.bit_and, lolo_hi_s, m);
    const lohi_lo = try b.op(.bit_and, lohi, m);
    const hilo_lo = try b.op(.bit_and, hilo, m);
    const cross0 = try b.op(.add, lolo_hi, lohi_lo);
    const cross = try b.op(.add, cross0, hilo_lo);

    const lohi_hi_s = try b.op(.shr, lohi, hs);
    const lohi_hi = try b.op(.bit_and, lohi_hi_s, m);
    const hilo_hi_s = try b.op(.shr, hilo, hs);
    const hilo_hi = try b.op(.bit_and, hilo_hi_s, m);
    const cross_hi_s = try b.op(.shr, cross, hs);
    const cross_hi = try b.op(.bit_and, cross_hi_s, m);

    const s0 = try b.op(.add, hihi, lohi_hi);
    const s1 = try b.op(.add, s0, hilo_hi);
    const unsigned_high = try b.op(.add, s1, cross_hi);
    if (info.signedness == .unsigned) return unsigned_high;

    // Signed correction: subtract b where a is negative, and a where b is negative. The sign mask is
    // an arithmetic shift of the ORIGINAL signed operand by W-1 (all ones when negative, else zero).
    const wm1 = try b.konst(@as(i64, @intCast(w - 1)));
    const amask = try b.op(.shr, lhs, wm1);
    const bmask = try b.op(.shr, rhs, wm1);
    const ca = try b.op(.bit_and, amask, rhs);
    const cb = try b.op(.bit_and, bmask, lhs);
    const c0 = try b.op(.sub, unsigned_high, ca);
    return b.op(.sub, c0, cb);
}

const testing = std.testing;

fn intTy(func: *Function, bits: u16, signedness: std.builtin.Signedness) !types.Type {
    return func.types.intern(.{ .int = .{ .signedness = signedness, .bits = bits } });
}

/// The i128 oracle: the true high `bits` of the full-width product, matching `mulh` semantics.
fn oracleHigh(a: i64, b: i64, bits: u16, signedness: std.builtin.Signedness) i64 {
    const shift: u7 = @intCast(bits);
    return switch (signedness) {
        .signed => @truncate(@as(i128, a) * @as(i128, b) >> shift),
        .unsigned => blk: {
            const au: u128 = @as(u64, @bitCast(a)) & maskBits(bits);
            const bu: u128 = @as(u64, @bitCast(b)) & maskBits(bits);
            break :blk @bitCast(@as(u64, @truncate((au * bu) >> shift)));
        },
    };
}

fn maskBits(bits: u16) u64 {
    return if (bits >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(bits)) - 1;
}

/// Sign- or zero-extend the low `bits` of `v` to a canonical i64, matching how a W-bit register
/// value reads back. Used by the test evaluator to model each op at its declared width.
fn wrapTo(v: i64, bits: u16, signedness: std.builtin.Signedness) i64 {
    if (bits >= 64) return v;
    const low: u64 = @as(u64, @bitCast(v)) & maskBits(bits);
    return switch (signedness) {
        .unsigned => @bitCast(low),
        .signed => blk: {
            const sign = @as(u64, 1) << @intCast(bits - 1);
            break :blk @bitCast(if (low & sign != 0) low | ~maskBits(bits) else low);
        },
    };
}

/// Evaluate a value whose whole dataflow is constants (iconst leaves, arith nodes), modelling each
/// op at its result type's width and signedness. Only for the tests below (inputs are constants).
fn evalConst(func: *const Function, v: Value) i64 {
    const inst = func.definingInst(v).?;
    const info = switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i,
        else => unreachable,
    };
    return switch (func.opcode(inst)) {
        .iconst => |c| wrapTo(c, info.bits, info.signedness),
        .arith => |a| blk: {
            const l = evalConst(func, a.lhs);
            const r = evalConst(func, a.rhs);
            const raw: i64 = switch (a.op) {
                .add => l +% r,
                .sub => l -% r,
                .mul => l *% r,
                .bit_and => l & r,
                .shr => switch (info.signedness) {
                    .signed => l >> @intCast(@as(u64, @bitCast(r)) & 63),
                    .unsigned => @bitCast((@as(u64, @bitCast(l)) & maskBits(info.bits)) >> @intCast(@as(u64, @bitCast(r)) & 63)),
                },
                else => unreachable, // the expansion only emits the ops above
            };
            break :blk wrapTo(raw, info.bits, info.signedness);
        },
        else => unreachable,
    };
}

fn expectMulhExpands(bits: u16, signedness: std.builtin.Signedness, a: i64, b: i64) !void {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, bits, signedness);
    const e = try func.appendBlock();
    const av = try func.appendInst(e, t, .{ .iconst = a });
    const bv = try func.appendInst(e, t, .{ .iconst = b });
    const r = try func.appendInst(e, t, .{ .arith = .{ .op = .mulh, .lhs = av, .rhs = bv } });
    func.setTerminator(e, .{ .ret = r });

    try testing.expect(try expandMulh(allocator, &func));
    for (func.blockInsts(e)) |inst| try testing.expect(!isMulh(&func, inst)); // no mulh survives
    const got = evalConst(&func, func.terminator(e).?.ret.?);
    try testing.expectEqual(oracleHigh(a, b, bits, signedness), got);
}

test "expandMulh matches the i128 oracle for signed 64-bit" {
    try expectMulhExpands(64, .signed, 0x123456789, 0x9876543);
    try expectMulhExpands(64, .signed, -0x123456789, 0x9876543);
    try expectMulhExpands(64, .signed, -3, -7);
    try expectMulhExpands(64, .signed, std.math.maxInt(i64), std.math.maxInt(i64));
    try expectMulhExpands(64, .signed, std.math.minInt(i64), 2);
}

test "expandMulh matches the i128 oracle for unsigned 64-bit" {
    try expectMulhExpands(64, .unsigned, @bitCast(@as(u64, 0xFFFFFFFF00000000)), @bitCast(@as(u64, 0x2)));
    try expectMulhExpands(64, .unsigned, @bitCast(~@as(u64, 0)), @bitCast(~@as(u64, 0)));
    try expectMulhExpands(64, .unsigned, 0x123456789, 0x9876543);
}

test "expandMulh matches the i128 oracle for 32-bit widths" {
    try expectMulhExpands(32, .signed, 100000, 100000);
    try expectMulhExpands(32, .signed, -100000, 100000);
    try expectMulhExpands(32, .unsigned, @bitCast(@as(u64, 0xFFFF0000)), 0x30000);
}
