//! Known-bits analysis and the always-safe rewrites it unlocks. For each value it computes which
//! bits are known to be 0 and which known to be 1, propagated forward through the bit-precise ops
//! (constants, and/or/xor, and constant-distance shifts). It then removes masks the analysis proves
//! redundant: `x & c` collapses to `x` when every bit `c` clears is already known 0 in `x`, and a
//! comparison folds to a constant when the operands' known bits make it decidable. This is the
//! substrate under GCC's VRP / LLVM's ValueTracking: build once, and ordinary rewrites fire in many
//! more places. Carrying ops (add/sub/mul) and width-changing converts are treated as fully unknown
//! for now, which is conservative (never wrong, only less precise); extension elimination waits on
//! int->int widening `convert` landing in the backends.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const cfg_mod = @import("cfg.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const BinOp = ir.function.BinOp;

pub const pass_def = pass.Pass{ .name = "knownbits", .run = run };

/// Per-value known bits: a set bit in `zeros` means that bit is known 0, in `ones` known 1. A bit
/// set in neither is unknown. A well-formed entry never has the same bit set in both.
const Bits = struct { zeros: u64 = 0, ones: u64 = 0 };

/// The mask of meaningful bits for an `n`-bit value (all ones for 64).
fn widthMask(bits: u16) u64 {
    return if (bits >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(bits)) - 1;
}

fn intWidth(func: *const Function, v: Value) ?u16 {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.bits,
        else => null,
    };
}

fn isUnsigned(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.signedness == .unsigned,
        else => false,
    };
}

/// Compute known bits for every value, single forward pass in reverse-postorder. Block parameters
/// stay unknown (loop-carried precision is not needed for the redundant-mask rewrites), so a single
/// pass suffices: in RPO every instruction operand is either a constant, a param, or an
/// already-visited instruction result.
fn analyze(allocator: std.mem.Allocator, func: *const Function) pass.Error![]Bits {
    const bits = try allocator.alloc(Bits, func.valueCount());
    errdefer allocator.free(bits);
    @memset(bits, .{});

    var cfg = try cfg_mod.build(allocator, func);
    defer cfg.deinit(allocator);
    const rpo = try cfg.reversePostorder(allocator);
    defer allocator.free(rpo);

    for (rpo) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            const result = func.instResult(inst) orelse continue;
            const w = intWidth(func, result) orelse continue;
            bits[@intFromEnum(result)] = transfer(func, bits, inst, w, isUnsigned(func, result));
        }
    }
    return bits;
}

/// The known bits of `inst`'s result, from its operands' known bits. Masked to the result width.
/// `unsigned` is the result type's signedness, which decides whether `shr` fills the vacated high
/// bits with zero (logical) or the sign bit (arithmetic).
fn transfer(func: *const Function, bits: []const Bits, inst: Inst, w: u16, unsigned: bool) Bits {
    const mask = widthMask(w);
    const r: Bits = switch (func.opcode(inst)) {
        .iconst => |c| .{ .ones = @as(u64, @bitCast(c)) & mask, .zeros = ~@as(u64, @bitCast(c)) & mask },
        .arith => |a| binary(a.op, bits[@intFromEnum(a.lhs)], bits[@intFromEnum(a.rhs)], null, unsigned),
        .arith_imm => |a| binary(a.op, bits[@intFromEnum(a.lhs)], constBits(a.imm), a.imm, unsigned),
        .select => |s| meet(bits[@intFromEnum(s.then)], bits[@intFromEnum(s.@"else")]),
        else => .{},
    };
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}

fn constBits(c: i64) Bits {
    const u: u64 = @bitCast(c);
    return .{ .ones = u, .zeros = ~u };
}

/// The conservative meet (a bit is known only if both agree): used for `select` and could serve a
/// block-param join. Keeps only bits known-and-equal on both sides.
fn meet(a: Bits, b: Bits) Bits {
    return .{ .zeros = a.zeros & b.zeros, .ones = a.ones & b.ones };
}

/// Known bits of `lhs op rhs`. `imm` is the constant immediate for the shift amount when the rhs is
/// an `arith_imm` operand, else null. Only the bit-precise ops are modelled; carrying ops are left
/// fully unknown (conservative).
fn binary(op: BinOp, a: Bits, b: Bits, imm: ?i64, unsigned: bool) Bits {
    return switch (op) {
        .bit_and => .{ .ones = a.ones & b.ones, .zeros = a.zeros | b.zeros },
        .bit_or => .{ .ones = a.ones | b.ones, .zeros = a.zeros & b.zeros },
        .bit_xor => blk: {
            const known = (a.zeros | a.ones) & (b.zeros | b.ones); // both operands known at this bit
            const val = a.ones ^ b.ones;
            break :blk .{ .ones = val & known, .zeros = ~val & known };
        },
        .shl => if (imm) |k| shiftLeft(a, k) else .{},
        // Only an unsigned (logical) shift zero-fills the vacated high bits. A signed shift is
        // arithmetic (sign-fill), so its high bits are not known 0; treat it as unknown.
        .shr => if (imm != null and unsigned) shiftRightLogical(a, imm.?) else .{},
        // Carrying ops: the carry chain makes the result bits hard to know, so treat as fully
        // unknown (conservative). Listed explicitly so a new BinOp forces a decision here.
        .add, .sub, .mul, .mulh, .div, .rem => .{},
    };
}

fn shiftLeft(a: Bits, k: i64) Bits {
    if (k < 0 or k >= 64) return .{};
    const s: u6 = @intCast(k);
    const low_zero = (@as(u64, 1) << s) - 1; // the vacated low bits are known 0
    return .{ .ones = a.ones << s, .zeros = (a.zeros << s) | low_zero };
}

fn shiftRightLogical(a: Bits, k: i64) Bits {
    if (k < 0 or k >= 64) return .{};
    const s: u6 = @intCast(k);
    const high_zero = ~(~@as(u64, 0) >> s); // the vacated high bits are known 0 (logical shift)
    return .{ .ones = a.ones >> s, .zeros = (a.zeros >> s) | high_zero };
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    if (func.blockCount() == 0) return false;
    const bits = try analyze(allocator, func);
    defer allocator.free(bits);

    var changed = false;
    for (0..func.instCount()) |i| {
        const inst: Inst = @enumFromInt(i);
        const result = func.instResult(inst) orelse continue;
        // Redundant mask: `x & c` is `x` when every bit that `c` clears is already known 0 in x.
        switch (func.opcode(inst)) {
            .arith_imm => |a| if (a.op == .bit_and) {
                if (redundantMask(func, bits, a.lhs, a.imm)) {
                    func.replaceAllUses(result, a.lhs);
                    changed = true;
                }
            },
            // A known-bit conflict proves the operands unequal, so eq/ne folds to a constant bool.
            // Rewriting the icmp in place (its result stays a bool) keeps every use valid and hands
            // branchfold a constant condition.
            .icmp => |cmp| if (cmp.op == .eq or cmp.op == .ne) {
                const ba = bits[@intFromEnum(cmp.lhs)];
                const bb = bits[@intFromEnum(cmp.rhs)];
                const conflict = (ba.ones & bb.zeros) | (ba.zeros & bb.ones);
                if (conflict != 0) {
                    func.opcodeMut(inst).* = .{ .iconst = if (cmp.op == .ne) 1 else 0 };
                    changed = true;
                }
            },
            // Redundant sign/zero-extension: a widen of a narrow of `s` recovers `s` when `s` already
            // fits the narrower width (its extended bits are known), so the round-trip is dropped.
            .convert => if (redundantExtend(func, bits, inst)) |s| {
                func.replaceAllUses(result, s);
                changed = true;
            },
            else => {},
        }
    }
    return changed;
}

/// If `inst` is a widening convert `widen(narrow(s))` that recovers `s` unchanged, return `s`. The
/// round-trip is the identity when the intermediate narrow lost nothing: the bits `s` holds above the
/// narrow width already match what the widen re-fills (zero for an unsigned intermediate, the sign for
/// a signed one), which known-bits can prove. Requires `s` to have the same type as the result so
/// forwarding it does not change any downstream signedness interpretation.
fn redundantExtend(func: *const Function, bits: []const Bits, inst: Inst) ?Value {
    const cv = func.opcode(inst).convert;
    const result = func.instResult(inst).?;
    const dst_w = intWidth(func, result) orelse return null;
    const mid = cv.value;
    const mid_w = intWidth(func, mid) orelse return null;
    if (dst_w <= mid_w) return null; // must be a widening convert

    const mid_inst = func.definingInst(mid) orelse return null;
    const inner = switch (func.opcode(mid_inst)) {
        .convert => |c| c,
        else => return null,
    };
    const s = inner.value;
    const s_w = intWidth(func, s) orelse return null;
    if (s_w != dst_w) return null; // the narrow's source must be exactly the width we widen back to
    if (isUnsigned(func, s) != isUnsigned(func, result)) return null; // same type as the result

    const high = widthMask(dst_w) & ~widthMask(mid_w); // the bits above the narrow width
    const sz = bits[@intFromEnum(s)].zeros;
    if (isUnsigned(func, mid)) {
        if (sz & high == high) return s; // zero-extend recovers s iff those bits are known 0
    } else {
        const sign: u64 = @as(u64, 1) << @intCast(mid_w - 1);
        if (sz & high == high and sz & sign != 0) return s; // sign-extend of a known-nonnegative s
    }
    return null;
}

/// True when `x & c == x`: the bits `c` clears (within x's width) are all known 0 in x already.
fn redundantMask(func: *const Function, bits: []const Bits, x: Value, c: i64) bool {
    const w = intWidth(func, x) orelse return false;
    const mask = widthMask(w);
    const cleared = ~@as(u64, @bitCast(c)) & mask; // bits the AND would clear
    return cleared & ~bits[@intFromEnum(x)].zeros == 0; // all already known 0 in x
}

const testing = std.testing;

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

fn uintTy(func: *Function, bits: u16) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = bits } });
}

test "a mask that clears only already-zero bits is removed" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try uintTy(&func, 64);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const lo = try func.appendArithImm(b, t, .bit_and, x, 0xFF); // lo: high bits known 0
    const redundant = try func.appendArithImm(b, t, .bit_and, lo, 0xFFFF); // clears only known-0 bits
    func.setTerminator(b, .{ .ret = redundant });

    try testing.expect(try runOnce(allocator, &func));
    // The redundant `& 0xFFFF` now forwards `lo` directly.
    try testing.expectEqual(lo, func.terminator(b).?.ret.?);
}

test "a zero-extend of a truncation of an already-narrow value is eliminated" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u64t = try uintTy(&func, 64);
    const u32t = try uintTy(&func, 32);
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, u64t);
    const s = try func.appendArithImm(b, u64t, .bit_and, p, 0xFFFFFFFF); // high 32 known 0
    const n = try func.appendInst(b, u32t, .{ .convert = .{ .value = s } }); // narrow to u32
    const w = try func.appendInst(b, u64t, .{ .convert = .{ .value = n } }); // widen back to u64
    func.setTerminator(b, .{ .ret = w });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(s, func.terminator(b).?.ret.?); // round-trip recovered s
}

test "a sign-extend round-trip of a known-nonnegative value is eliminated" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i64t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, i64t);
    const s = try func.appendArithImm(b, i64t, .bit_and, p, 0x7FFFFFFF); // nonnegative, fits i32
    const n = try func.appendInst(b, i32t, .{ .convert = .{ .value = s } });
    const w = try func.appendInst(b, i64t, .{ .convert = .{ .value = n } });
    func.setTerminator(b, .{ .ret = w });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(s, func.terminator(b).?.ret.?);
}

test "a widen of a truncation is kept when the high bits are not known" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u64t = try uintTy(&func, 64);
    const u32t = try uintTy(&func, 32);
    const b = try func.appendBlock();
    const s = try func.appendBlockParam(b, u64t); // unknown high bits
    const n = try func.appendInst(b, u32t, .{ .convert = .{ .value = s } });
    const w = try func.appendInst(b, u64t, .{ .convert = .{ .value = n } });
    func.setTerminator(b, .{ .ret = w });

    try testing.expect(!try runOnce(allocator, &func)); // truncation may lose bits, keep the round-trip
}

test "a mask that clears a possibly-set bit is kept" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try uintTy(&func, 64);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const masked = try func.appendArithImm(b, t, .bit_and, x, 0xFF); // clears real bits of unknown x
    func.setTerminator(b, .{ .ret = masked });

    try testing.expect(!try runOnce(allocator, &func)); // not redundant, left alone
}

test "a mask clearing the high bits after a logical shift is removed, after an arithmetic shift is kept" {
    const allocator = testing.allocator;
    const clear_top_8: i64 = @bitCast(@as(u64, 0x00FFFFFFFFFFFFFF));
    // Unsigned: `x >>u 8` has its top 8 bits known 0, so `& 0x00FF..FF` is redundant.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try uintTy(&func, 64);
        const b = try func.appendBlock();
        const x = try func.appendBlockParam(b, t);
        const y = try func.appendArithImm(b, t, .shr, x, 8);
        const z = try func.appendArithImm(b, t, .bit_and, y, clear_top_8);
        func.setTerminator(b, .{ .ret = z });
        try testing.expect(try runOnce(allocator, &func));
        try testing.expectEqual(y, func.terminator(b).?.ret.?);
    }
    // Signed: `x >>s 8` sign-fills the top 8 bits, so they are not known 0 and the mask must stay.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
        const b = try func.appendBlock();
        const x = try func.appendBlockParam(b, t);
        const y = try func.appendArithImm(b, t, .shr, x, 8);
        const z = try func.appendArithImm(b, t, .bit_and, y, clear_top_8);
        func.setTerminator(b, .{ .ret = z });
        try testing.expect(!try runOnce(allocator, &func)); // mask is not redundant
    }
}

test "eq against zero folds to false when a bit is known one" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try uintTy(&func, 64);
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const set = try func.appendArithImm(b, t, .bit_or, x, 1); // bit 0 known 1
    const zero = try func.appendInst(b, t, .{ .iconst = 0 });
    const eq = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .eq, .lhs = set, .rhs = zero } });
    func.setTerminator(b, .{ .ret = eq });

    try testing.expect(try runOnce(allocator, &func));
    // (x | 1) == 0 is always false; the icmp is now a constant 0.
    try testing.expectEqual(@as(i64, 0), func.opcode(func.definingInst(eq).?).iconst);
}
