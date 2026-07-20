//! Known-bits analysis and the always-safe rewrites it unlocks. For each value it computes which
//! bits are known to be 0 and which known to be 1, propagated forward through the bit-precise ops
//! (constants, and/or/xor, and constant-distance shifts). It then removes masks the analysis proves
//! redundant: `x & c` collapses to `x` when every bit `c` clears is already known 0 in `x`, and a
//! comparison folds to a constant when the operands' known bits make it decidable. This is the
//! substrate under GCC's VRP / LLVM's ValueTracking: build once, and ordinary rewrites fire in many
//! more places. Add/sub carry precisely through the carry chain (LLVM's computeForAddCarry). `mul`
//! is a sound shift-and-add accumulation of the known partial products plus a leading-zero bound
//! (see `mulKnown`), and `mulh` reuses it at double width. Unsigned `div`/`rem` are precise for a
//! known power-of-two divisor (exact shift/mask) and otherwise a sound magnitude bound from the
//! divisor's known range (see `divKnown`/`remKnown`). Signed `div`/`rem` stay fully unknown, since
//! rounding toward zero with a sign makes the bound not worth chasing. An int->int `convert` is
//! precise too (see `convertKnown`): widening reuses `extendBits` (sign- or zero-extend per the
//! source's signedness), narrowing keeps the source's low bits, and an int<->float convert is fully
//! unknown. That forward known-bits result on a convert is new precision the redundant-mask and
//! icmp-fold rewrites can now key off of downstream, on top of `redundantExtend`, which separately
//! eliminates a whole widen-of-a-narrow round-trip by reading the inner source's own known bits.

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
        .arith => |a| binary(a.op, bits[@intFromEnum(a.lhs)], bits[@intFromEnum(a.rhs)], null, unsigned, mask, w),
        .arith_imm => |a| binary(a.op, bits[@intFromEnum(a.lhs)], constBits(a.imm), a.imm, unsigned, mask, w),
        .select => |s| meet(bits[@intFromEnum(s.then)], bits[@intFromEnum(s.@"else")]),
        .convert => |cv| convertKnown(func, bits, cv.value, w),
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
/// an `arith_imm` operand, else null. `mask` is the width mask of the result, needed by the carrying
/// ops (add/sub/mul). `w` is the result width in bits (`mask == widthMask(w)`), needed by `mul`'s
/// leading-zero bound. Only the bit-precise ops, add/sub, and mul are modelled. The remaining
/// carrying ops are left fully unknown (conservative).
fn binary(op: BinOp, a: Bits, b: Bits, imm: ?i64, unsigned: bool, mask: u64, w: u16) Bits {
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
        // a + b with a known-0 carry-in.
        .add => addCarry(a, b, true, false, mask),
        // a - b == a + ~b + 1, a known-1 carry-in. Negating b's known bits (zeros <-> ones) is exact:
        // bitwise-not of a value with known bit k is the complement of that same bit.
        .sub => blk: {
            const nb = Bits{ .zeros = b.ones, .ones = b.zeros };
            break :blk addCarry(a, nb, false, true, mask);
        },
        // Shift-and-add accumulation of the known partial products, plus a leading-zero bound.
        .mul => mulKnown(a, b, w, mask),
        // The high half of the double-width product: extend both operands to 2w and reuse mulKnown
        // there, then shift the top half down.
        .mulh => mulhKnown(a, b, w, unsigned),
        // Unsigned div/rem: precise for a known power-of-two divisor, else a sound magnitude bound
        // (see divKnown/remKnown). Signed div/rem round toward zero with a sign, which known-bits
        // does not chase, so they stay fully unknown.
        .div => if (unsigned) divKnown(a, b, w, mask) else .{},
        .rem => if (unsigned) remKnown(a, b, w, mask) else .{},
    };
}

/// Known bits of the low `w` bits of `a * b`: `a * b == sum over set bits i of b of (a << i)`.
/// Accumulate the known bits of that sum with `addCarry`. A bit of `b` known 1 means the partial
/// product `a << i` is definitely added, known 0 means it definitely is not, and unknown means it is
/// added in some concrete instantiations and not in others, so the sound claim is the meet (only
/// bits both possibilities agree on) of "added" and "not added". Finished off with a leading-zero
/// bound: if `a`'s known-min value needs at most `w - lz_a` bits and `b`'s at most `w - lz_b`, the
/// product needs at most `(w - lz_a) + (w - lz_b)` bits, so any width-`w` bits above that are known
/// 0 (this can catch high zero bits the accumulation alone misses, since accumulation only tracks
/// what carries could touch, not the coarser magnitude bound).
fn mulKnown(a: Bits, b: Bits, w: u16, mask: u64) Bits {
    var acc = Bits{ .zeros = mask, .ones = 0 }; // the empty sum, known 0 within width
    var i: u16 = 0;
    while (i < w) : (i += 1) {
        const bit = @as(u64, 1) << @intCast(i);
        const shifted = shiftLeft(a, i); // known bits of (a << i), width-masked by shiftLeft itself
        if (b.ones & bit != 0) {
            acc = addCarry(acc, shifted, true, false, mask);
        } else if (b.zeros & bit != 0) {
            // Not added: acc unchanged.
        } else {
            acc = meet(addCarry(acc, shifted, true, false, mask), acc);
        }
    }
    const lz = @as(u32, countMinLeadingZeros(a, w)) + countMinLeadingZeros(b, w);
    if (lz > w) {
        const top_zero_bits = lz - w; // number of known-0 high bits of the product
        // top_zero_bits is at most w (lz is at most 2w), but a u6 shift amount only reaches 63, so
        // guard the w == 64, top_zero_bits == 64 corner (the whole product is known 0) explicitly.
        const high_zero = if (top_zero_bits >= 64) mask else ~(mask >> @intCast(top_zero_bits)) & mask;
        acc.zeros |= high_zero;
    }
    return .{ .zeros = acc.zeros & mask, .ones = acc.ones & mask };
}

/// Known bits of `src` (a `src_w`-bit value) sign- or zero-extended to `dst_w` bits (`dst_w >=
/// src_w`). Zero-extend (`signed == false`): the high bits `[src_w, dst_w)` are known 0. Sign-extend
/// (`signed == true`): the high bits replicate the source sign bit (`src_w - 1`), known only when
/// that bit itself is known. Task 6 (int->int widening convert) reuses this directly.
fn extendBits(src: Bits, src_w: u16, dst_w: u16, signed: bool) Bits {
    std.debug.assert(dst_w >= src_w); // an extend never narrows
    const src_mask = widthMask(src_w);
    var out = Bits{ .zeros = src.zeros & src_mask, .ones = src.ones & src_mask };
    const high = widthMask(dst_w) & ~src_mask; // the bits [src_w, dst_w)
    if (!signed) {
        out.zeros |= high; // zero-extend: high bits known 0
    } else {
        const sign_bit = @as(u64, 1) << @intCast(src_w - 1);
        if (src.zeros & sign_bit != 0) {
            out.zeros |= high; // sign known 0: high bits known 0
        } else if (src.ones & sign_bit != 0) {
            out.ones |= high; // sign known 1: high bits known 1
        }
        // else the sign bit is unknown, so the high bits stay unknown too.
    }
    return out;
}

/// Known bits of an int->int convert of `src` into a `dst_w`-bit result. Widening (`dst_w > src_w`)
/// sign- or zero-extends per the SOURCE signedness, via `extendBits`. Narrowing or same-width keeps
/// only the low `dst_w` bits of the source (a truncation drops the rest, but the bits that remain are
/// exactly the source's known low bits). A non-int source (an int<->float convert) is fully unknown.
fn convertKnown(func: *const Function, bits: []const Bits, src: Value, dst_w: u16) Bits {
    const src_w = intWidth(func, src) orelse return .{}; // float source: unknown
    const src_bits = bits[@intFromEnum(src)];
    if (dst_w > src_w) return extendBits(src_bits, src_w, dst_w, !isUnsigned(func, src)); // sext if signed
    const dmask = widthMask(dst_w);
    return .{ .zeros = src_bits.zeros & dmask, .ones = src_bits.ones & dmask }; // narrow/same: low bits
}

/// Known bits of the high `w` bits of the double-width `a * b` product, signed or unsigned per
/// `unsigned`. Extends both operands to `2w` bits (zero-extend when unsigned, sign-extend when
/// signed, matching how a real mulh widens its operands before multiplying), takes the full `2w`-bit
/// product's known bits via `mulKnown`, and shifts the high half down to bit 0.
fn mulhKnown(a: Bits, b: Bits, w: u16, unsigned: bool) Bits {
    if (@as(u32, w) * 2 > 64) return .{}; // the double width would exceed u64: conservative
    const dw: u16 = w * 2;
    const dmask = widthMask(dw);
    const ax = extendBits(a, w, dw, !unsigned);
    const bx = extendBits(b, w, dw, !unsigned);
    const prod = mulKnown(ax, bx, dw, dmask); // known bits of the full 2w-bit product
    const mask = widthMask(w);
    return .{ .zeros = (prod.zeros >> @intCast(w)) & mask, .ones = (prod.ones >> @intCast(w)) & mask };
}

/// The count of high bits (within width `w`) of the concrete value `v` that are 0, starting from
/// bit `w - 1` and stopping at the first set bit. Unlike `countMinLeadingZeros`, this takes a plain
/// value rather than a `Bits` pattern: `divKnown`/`remKnown` use it on a computed magnitude bound
/// (`q_max`, `bound`), not on an operand's known bits directly.
fn leadingZerosWithin(v: u64, w: u16) u16 {
    var count: u16 = 0;
    while (count < w) : (count += 1) {
        const s: u6 = @intCast(w - 1 - count);
        if ((v >> s) & 1 != 0) break;
    }
    return count;
}

/// Known bits of the unsigned quotient `a / b`, sound over the defined domain `b != 0` (`div` is UB
/// at a zero divisor, so soundness need not hold there). Exact when `b` is fully known and a power of
/// two (`udiv` by `2^k` is a logical shift right by `k`). Otherwise a magnitude bound: every real
/// divisor consistent with `b`'s known bits and nonzero is at least `divisor_min = max(1,
/// minValue(b))`, so the quotient is at most `maxValue(a) / divisor_min`, which bounds how many low
/// bits the result needs and so how many high bits (within `w`) are known 0.
fn divKnown(a: Bits, b: Bits, w: u16, mask: u64) Bits {
    if ((b.zeros | b.ones) & mask == mask) {
        const bv = b.ones & mask;
        if (bv != 0 and bv & (bv - 1) == 0) {
            const k: u6 = @intCast(@ctz(bv));
            const shifted = shiftRightLogical(a, k);
            // shiftRightLogical zero-fills relative to a full 64-bit register, not width `w`, so at
            // w < 64 it under-claims: the top `k` bits within width `w` are also always vacated by a
            // w-bit logical shift, so OR them in explicitly to get the exact result the brief expects.
            const top_zero = ~(mask >> k) & mask;
            return .{ .zeros = (shifted.zeros | top_zero) & mask, .ones = shifted.ones & mask };
        }
    }
    const divisor_min = @max(@as(u64, 1), minValue(b, mask));
    const q_max = maxValue(a, mask) / divisor_min;
    const lead = leadingZerosWithin(q_max, w);
    if (lead == 0) return .{}; // no known-0 high bits from this bound
    // Guard the w == 64, lead == 64 corner (q_max == 0, the whole result is known 0): a u6 shift
    // amount cannot reach 64, so a plain `mask >> lead` would be undefined behavior there.
    const high_zero = if (lead >= 64) mask else ~(mask >> @intCast(lead)) & mask;
    return .{ .zeros = high_zero, .ones = 0 };
}

/// Known bits of the unsigned remainder `a % b`, sound over the defined domain `b != 0`. Exact when
/// `b` is fully known and a power of two (`urem` by `2^k` is `a & (2^k - 1)`: the low `k` bits are
/// `a`'s own low `k` bits, and the rest are known 0). Otherwise a magnitude bound: `a % b < b <=
/// maxValue(b)` for every nonzero `b` consistent with the pattern, so the remainder is at most
/// `maxValue(b) - 1`, bounding its known-0 high bits the same way `divKnown`'s quotient bound does.
fn remKnown(a: Bits, b: Bits, w: u16, mask: u64) Bits {
    if ((b.zeros | b.ones) & mask == mask) {
        const bv = b.ones & mask;
        if (bv != 0 and bv & (bv - 1) == 0) {
            const k: u6 = @intCast(@ctz(bv));
            const low_mask = (@as(u64, 1) << k) - 1;
            return .{
                .zeros = (~low_mask & mask) | (a.zeros & low_mask),
                .ones = a.ones & low_mask,
            };
        }
    }
    const b_max = maxValue(b, mask);
    if (b_max == 0) return .{ .zeros = mask, .ones = 0 }; // b forced to 0: only reachable via the
    // excluded b == 0 domain, but stay well-formed rather than underflow the bound below.
    const bound = b_max - 1;
    const lead = leadingZerosWithin(bound, w);
    if (lead == 0) return .{};
    const high_zero = if (lead >= 64) mask else ~(mask >> @intCast(lead)) & mask;
    return .{ .zeros = high_zero, .ones = 0 };
}

/// Known bits of `a + b + carry_in`, where the carry-in is known-0 (`carry_zero`) or known-1
/// (`carry_one`, and both false is invalid, both true never happens for our two call sites). Per
/// LLVM's `KnownBits::computeForAddCarry`. `mask` is the width mask of the result.
fn addCarry(a: Bits, b: Bits, carry_zero: bool, carry_one: bool, mask: u64) Bits {
    const possible_sum_zero = maxValue(a, mask) +% maxValue(b, mask) +% @as(u64, if (carry_zero) 0 else 1);
    const possible_sum_one = minValue(a, mask) +% minValue(b, mask) +% @as(u64, if (carry_one) 1 else 0);
    const carry_known_zero = ~(possible_sum_zero ^ a.zeros ^ b.zeros);
    const carry_known_one = possible_sum_one ^ a.ones ^ b.ones;
    const a_known = a.zeros | a.ones;
    const b_known = b.zeros | b.ones;
    const known = a_known & b_known & (carry_known_zero | carry_known_one);
    return .{ .zeros = ~possible_sum_zero & known & mask, .ones = possible_sum_one & known & mask };
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

/// The minimum concrete value consistent with `a` (its unknown bits taken as 0), within `mask`.
fn minValue(a: Bits, mask: u64) u64 {
    return a.ones & mask;
}

/// The maximum concrete value consistent with `a` (its unknown bits taken as 1), within `mask`.
fn maxValue(a: Bits, mask: u64) u64 {
    return ~a.zeros & mask;
}

/// The count of low bits (within width `w`) known 0, starting from bit 0 and stopping at the first
/// bit that is not known 0 (known 1 or unknown).
fn countMinTrailingZeros(a: Bits, w: u16) u16 {
    var count: u16 = 0;
    while (count < w) : (count += 1) {
        const s: u6 = @intCast(count);
        if ((a.zeros >> s) & 1 == 0) break;
    }
    return count;
}

/// The count of high bits (within width `w`) known 0, starting from bit `w - 1` and stopping at the
/// first bit that is not known 0.
fn countMinLeadingZeros(a: Bits, w: u16) u16 {
    var count: u16 = 0;
    while (count < w) : (count += 1) {
        const s: u6 = @intCast(w - 1 - count);
        if ((a.zeros >> s) & 1 == 0) break;
    }
    return count;
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

test "minValue and maxValue bound the range of values consistent with a pattern" {
    const mask = widthMask(8);
    // Low 4 bits known 0, bit 4 known 1, top 3 bits unknown.
    const a = Bits{ .zeros = 0b0000_1111, .ones = 0b0001_0000 };
    try testing.expectEqual(@as(u64, 0b0001_0000), minValue(a, mask));
    try testing.expectEqual(@as(u64, 0b1111_0000), maxValue(a, mask));
}

test "countMinTrailingZeros counts known-0 low bits and stops at the first bit that is not known 0" {
    const some_known = Bits{ .zeros = 0b0000_0111, .ones = 0 }; // bits 0..2 known 0, bit 3 unknown
    try testing.expectEqual(@as(u16, 3), countMinTrailingZeros(some_known, 8));
    const all_known_zero = Bits{ .zeros = 0xFF, .ones = 0 };
    try testing.expectEqual(@as(u16, 8), countMinTrailingZeros(all_known_zero, 8));
    const bit0_known_one = Bits{ .zeros = 0, .ones = 1 };
    try testing.expectEqual(@as(u16, 0), countMinTrailingZeros(bit0_known_one, 8));
    const nothing_known = Bits{};
    try testing.expectEqual(@as(u16, 0), countMinTrailingZeros(nothing_known, 8));
}

test "countMinLeadingZeros counts known-0 high bits and stops at the first bit that is not known 0" {
    const some_known = Bits{ .zeros = 0b1110_0000, .ones = 0 }; // bits 5..7 known 0
    try testing.expectEqual(@as(u16, 3), countMinLeadingZeros(some_known, 8));
    const all_known_zero = Bits{ .zeros = 0xFF, .ones = 0 };
    try testing.expectEqual(@as(u16, 8), countMinLeadingZeros(all_known_zero, 8));
    const top_bit_known_one = Bits{ .zeros = 0, .ones = 0b1000_0000 };
    try testing.expectEqual(@as(u16, 0), countMinLeadingZeros(top_bit_known_one, 8));
    const nothing_known = Bits{};
    try testing.expectEqual(@as(u16, 0), countMinLeadingZeros(nothing_known, 8));
}

// TEST-ONLY exhaustive brute-force soundness harness. A "pattern" assigns each of `w` bits one of
// known-0, known-1, unknown, i.e. a well-formed `Bits` (zeros & ones == 0). This enumerates every
// 3^w pattern as a base-3 counter, and for each pattern every concrete value whose known bits match
// it (the unknown bits range over all combinations). Later tasks gate every new known-bits transfer
// on passing this harness before it ships. This task proves the harness itself is correct by running
// it against the transfers above it, which are already known correct.

/// The `idx`-th of the 3^w patterns over `w` bits: digit 0 means that bit is known 0, digit 1 means
/// known 1, digit 2 means unknown. `idx` must be less than `patternCount(w)`.
fn patternFromIndex(idx: u32, w: u16) Bits {
    var zeros: u64 = 0;
    var ones: u64 = 0;
    var rem = idx;
    var i: u16 = 0;
    while (i < w) : (i += 1) {
        const digit = rem % 3;
        rem /= 3;
        const s: u6 = @intCast(i);
        switch (digit) {
            0 => zeros |= @as(u64, 1) << s,
            1 => ones |= @as(u64, 1) << s,
            2 => {},
            else => unreachable, // digit is rem % 3, always in 0..2
        }
    }
    return .{ .zeros = zeros, .ones = ones };
}

/// True when concrete value `v` (within `mask`) matches every known bit of pattern `p`.
fn consistentWith(p: Bits, v: u64, mask: u64) bool {
    return (v & p.zeros) == 0 and (~v & p.ones & mask) == 0;
}

/// 3^w, the number of patterns over `w` bits. Every caller keeps `w` small (at most 5).
fn patternCount(w: u16) u32 {
    var n: u32 = 1;
    var i: u16 = 0;
    while (i < w) : (i += 1) n *= 3;
    return n;
}

/// Assert `transfer` is a sound known-bits abstraction of the concrete binary op `realOp`: for
/// every pair of patterns, and every pair of concrete values consistent with them, no bit `transfer`
/// claims known 0 is ever really 1, and no bit it claims known 1 is ever really 0. Keep `w` small (at
/// most 5): the sweep is 3^w patterns squared times 2^w values squared.
fn assertSoundBinary(
    comptime realOp: fn (u64, u64, u16) u64,
    comptime xfer: fn (Bits, Bits, u16) Bits,
    w: u16,
) void {
    std.debug.assert(w >= 1 and w <= 5); // keep the exhaustive sweep fast
    const mask = widthMask(w);
    const patterns = patternCount(w);

    var pa_idx: u32 = 0;
    while (pa_idx < patterns) : (pa_idx += 1) {
        const pa = patternFromIndex(pa_idx, w);
        var pb_idx: u32 = 0;
        while (pb_idx < patterns) : (pb_idx += 1) {
            const pb = patternFromIndex(pb_idx, w);
            const claimed = xfer(pa, pb, w);
            std.debug.assert(claimed.zeros & claimed.ones == 0); // well-formed: never both known

            var va: u64 = 0;
            while (va <= mask) : (va += 1) {
                if (!consistentWith(pa, va, mask)) continue;
                var vb: u64 = 0;
                while (vb <= mask) : (vb += 1) {
                    if (!consistentWith(pb, vb, mask)) continue;
                    const real = realOp(va, vb, w) & mask;
                    std.debug.assert(claimed.zeros & real == 0); // no claimed-0 bit is really set
                    std.debug.assert(claimed.ones & ~real & mask == 0); // every claimed-1 bit is really set
                }
            }
        }
    }
}

/// Like `assertSoundBinary`, but skips any concrete pair `(va, vb)` for which `skip(va, vb)` is true
/// before evaluating `realOp`, and does not check soundness for it. Some ops (div, rem) are UB at
/// certain inputs (a zero divisor), so soundness only needs to hold over the defined domain. `skip`
/// carves out the undefined part so the harness stays honest without calling `realOp` there.
fn assertSoundBinaryFiltered(
    comptime realOp: fn (u64, u64, u16) u64,
    comptime xfer: fn (Bits, Bits, u16) Bits,
    w: u16,
    comptime skip: fn (u64, u64) bool,
) void {
    std.debug.assert(w >= 1 and w <= 5); // keep the exhaustive sweep fast
    const mask = widthMask(w);
    const patterns = patternCount(w);

    var pa_idx: u32 = 0;
    while (pa_idx < patterns) : (pa_idx += 1) {
        const pa = patternFromIndex(pa_idx, w);
        var pb_idx: u32 = 0;
        while (pb_idx < patterns) : (pb_idx += 1) {
            const pb = patternFromIndex(pb_idx, w);
            const claimed = xfer(pa, pb, w);
            std.debug.assert(claimed.zeros & claimed.ones == 0); // well-formed: never both known

            var va: u64 = 0;
            while (va <= mask) : (va += 1) {
                if (!consistentWith(pa, va, mask)) continue;
                var vb: u64 = 0;
                while (vb <= mask) : (vb += 1) {
                    if (!consistentWith(pb, vb, mask)) continue;
                    if (skip(va, vb)) continue; // outside the defined domain, no claim to check
                    const real = realOp(va, vb, w) & mask;
                    std.debug.assert(claimed.zeros & real == 0); // no claimed-0 bit is really set
                    std.debug.assert(claimed.ones & ~real & mask == 0); // every claimed-1 bit is really set
                }
            }
        }
    }
}

/// Analogous to `assertSoundBinary` for a unary (or width-changing) op: one operand pattern at
/// `src_w`, with `realOp`/`transfer` producing a result read at `dst_w`.
fn assertSoundUnary(
    comptime realOp: fn (u64, u16) u64,
    comptime xfer: fn (Bits, u16) Bits,
    src_w: u16,
    dst_w: u16,
) void {
    std.debug.assert(src_w >= 1 and src_w <= 5);
    std.debug.assert(dst_w >= 1 and dst_w <= 64);
    const src_mask = widthMask(src_w);
    const dst_mask = widthMask(dst_w);
    const patterns = patternCount(src_w);

    var pa_idx: u32 = 0;
    while (pa_idx < patterns) : (pa_idx += 1) {
        const pa = patternFromIndex(pa_idx, src_w);
        const claimed = xfer(pa, dst_w);
        std.debug.assert(claimed.zeros & claimed.ones == 0);

        var va: u64 = 0;
        while (va <= src_mask) : (va += 1) {
            if (!consistentWith(pa, va, src_mask)) continue;
            const real = realOp(va, dst_w) & dst_mask;
            std.debug.assert(claimed.zeros & real == 0);
            std.debug.assert(claimed.ones & ~real & dst_mask == 0);
        }
    }
}

fn realBitAnd(va: u64, vb: u64, w: u16) u64 {
    return (va & vb) & widthMask(w);
}
fn transferBitAnd(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.bit_and, a, b, null, false, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}
fn realBitOr(va: u64, vb: u64, w: u16) u64 {
    return (va | vb) & widthMask(w);
}
fn transferBitOr(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.bit_or, a, b, null, false, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}
fn realBitXor(va: u64, vb: u64, w: u16) u64 {
    return (va ^ vb) & widthMask(w);
}
fn transferBitXor(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.bit_xor, a, b, null, false, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}

test "knownbits harness self-test: bit_and/bit_or/bit_xor transfers are sound" {
    var w: u16 = 1;
    while (w <= 4) : (w += 1) {
        assertSoundBinary(realBitAnd, transferBitAnd, w);
        assertSoundBinary(realBitOr, transferBitOr, w);
        assertSoundBinary(realBitXor, transferBitXor, w);
    }
}

/// Fixed-shift-amount fixtures for shl, parameterized by the comptime shift amount `k` so each
/// instantiation is a plain `fn (u64, u16) u64` / `fn (Bits, u16) Bits` pair `assertSoundUnary` can
/// call.
fn ShiftLeftFixture(comptime k: u16) type {
    return struct {
        fn real(v: u64, w: u16) u64 {
            const s: u6 = @intCast(k);
            return (v << s) & widthMask(w);
        }
        fn transfer(a: Bits, w: u16) Bits {
            const mask = widthMask(w);
            const r = binary(.shl, a, constBits(k), k, false, mask, w);
            return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
        }
    };
}

/// Fixed-shift-amount fixtures for a logical (unsigned) shr.
fn ShiftRightLogicalFixture(comptime k: u16) type {
    return struct {
        fn real(v: u64, w: u16) u64 {
            const s: u6 = @intCast(k);
            return (v >> s) & widthMask(w);
        }
        fn transfer(a: Bits, w: u16) Bits {
            const mask = widthMask(w);
            const r = binary(.shr, a, constBits(k), k, true, mask, w);
            return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
        }
    };
}

test "knownbits harness self-test: shl/shr(logical) transfers are sound" {
    comptime var w: u16 = 1;
    inline while (w <= 4) : (w += 1) {
        comptime var k: u16 = 0;
        inline while (k < w) : (k += 1) {
            const Shl = ShiftLeftFixture(k);
            assertSoundUnary(Shl.real, Shl.transfer, w, w);
            const Shr = ShiftRightLogicalFixture(k);
            assertSoundUnary(Shr.real, Shr.transfer, w, w);
        }
    }
}

fn realAdd(va: u64, vb: u64, w: u16) u64 {
    return (va +% vb) & widthMask(w);
}
fn transferAdd(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.add, a, b, null, false, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}
fn realSub(va: u64, vb: u64, w: u16) u64 {
    return (va -% vb) & widthMask(w);
}
fn transferSub(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.sub, a, b, null, false, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}

test "knownbits: add transfer is sound" {
    var w: u16 = 1;
    while (w <= 5) : (w += 1) {
        assertSoundBinary(realAdd, transferAdd, w);
    }
}

test "knownbits: sub transfer is sound" {
    var w: u16 = 1;
    while (w <= 5) : (w += 1) {
        assertSoundBinary(realSub, transferSub, w);
    }
}

test "knownbits: add recovers exact bits when both operands are fully known" {
    const mask = widthMask(8);
    const a = Bits{ .zeros = ~@as(u64, 0b0010_1101) & mask, .ones = 0b0010_1101 }; // 45
    const b = Bits{ .zeros = ~@as(u64, 0b0000_0111) & mask, .ones = 0b0000_0111 }; // 7
    const r = binary(.add, a, b, null, false, mask, 8);
    // 45 + 7 == 52 (0b0011_0100), and every bit of a fully-known sum is fully known.
    try testing.expectEqual(@as(u64, 0b0011_0100), r.ones & mask);
    try testing.expectEqual(mask, r.zeros | r.ones);
}

test "knownbits: add recovers known-zero high bits when both operands' high bits are known zero" {
    const mask = widthMask(8);
    // Both operands confined to the low nibble (high 4 bits known 0): the sum cannot carry past bit
    // 4, so the top 3 bits of the 8-bit result are known 0 (bit 4 may or may not be set by carry).
    const a = Bits{ .zeros = 0b1111_0000, .ones = 0 };
    const b = Bits{ .zeros = 0b1111_0000, .ones = 0 };
    const r = binary(.add, a, b, null, false, mask, 8);
    try testing.expectEqual(@as(u64, 0b1110_0000), r.zeros & 0b1110_0000);
}

test "knownbits: sub recovers exact bits when both operands are fully known" {
    const mask = widthMask(8);
    const a = Bits{ .zeros = ~@as(u64, 0b0010_1101) & mask, .ones = 0b0010_1101 }; // 45
    const b = Bits{ .zeros = ~@as(u64, 0b0000_0111) & mask, .ones = 0b0000_0111 }; // 7
    const r = binary(.sub, a, b, null, false, mask, 8);
    // 45 - 7 == 38 (0b0010_0110), and every bit of a fully-known difference is fully known.
    try testing.expectEqual(@as(u64, 0b0010_0110), r.ones & mask);
    try testing.expectEqual(mask, r.zeros | r.ones);
}

test "knownbits: sub of equal known-low-bits operands recovers a known-zero low bit" {
    const mask = widthMask(8);
    // Both operands share known bit 0 == 1 (odd), so the difference's low bit is known 0.
    const a = Bits{ .zeros = 0, .ones = 0b0000_0001 };
    const b = Bits{ .zeros = 0, .ones = 0b0000_0001 };
    const r = binary(.sub, a, b, null, false, mask, 8);
    try testing.expectEqual(@as(u64, 1), r.zeros & 1);
}

fn realMul(va: u64, vb: u64, w: u16) u64 {
    return (va *% vb) & widthMask(w);
}
fn transferMul(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.mul, a, b, null, false, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}

test "knownbits: mul transfer is sound" {
    // O(w) addCarrys per pattern pair, 3^w patterns squared, 2^w concrete values squared: w<=4 keeps
    // this fast (w=5 also passes but is noticeably slower, so it is left out of the default sweep).
    var w: u16 = 1;
    while (w <= 4) : (w += 1) {
        assertSoundBinary(realMul, transferMul, w);
    }
}

test "knownbits: mul recovers the sum of trailing zeros as low known-zero bits" {
    const mask = widthMask(8);
    // a's low 2 bits known 0 (a multiple of 4), b's low bit known 0 (a multiple of 2): the product is
    // a multiple of 8, so its low 3 bits are known 0.
    const a = Bits{ .zeros = 0b0000_0011, .ones = 0 };
    const b = Bits{ .zeros = 0b0000_0001, .ones = 0 };
    const r = binary(.mul, a, b, null, false, mask, 8);
    try testing.expectEqual(@as(u64, 0b0000_0111), r.zeros & 0b0000_0111);
}

test "knownbits: mul recovers the exact product when both operands are fully known" {
    const mask = widthMask(8);
    const a = Bits{ .zeros = ~@as(u64, 13) & mask, .ones = 13 };
    const b = Bits{ .zeros = ~@as(u64, 9) & mask, .ones = 9 };
    const r = binary(.mul, a, b, null, false, mask, 8);
    // 13 * 9 == 117 (0b0111_0101), and every bit of a fully-known product is fully known.
    try testing.expectEqual(@as(u64, 117 & mask), r.ones & mask);
    try testing.expectEqual(mask, r.zeros | r.ones);
}

test "knownbits: mul recovers known-zero high bits from the leading-zero bound" {
    const mask = widthMask(8);
    // a confined to the low nibble (high 4 bits known 0, so a <= 15) and b confined to 2 bits (high 6
    // bits known 0, so b <= 3): the product is at most 45, which fits 6 bits, so the top 2 bits of the
    // 8-bit result are known 0.
    const a = Bits{ .zeros = 0b1111_0000, .ones = 0 };
    const b = Bits{ .zeros = 0b1111_1100, .ones = 0 };
    const r = binary(.mul, a, b, null, false, mask, 8);
    try testing.expectEqual(@as(u64, 0b1100_0000), r.zeros & 0b1100_0000);
}

test "knownbits: mul by a known constant recovers bits shifted up from the multiplier" {
    const mask = widthMask(8);
    // b is the fully-known constant 4 (0b0100): a * 4 is a << 2, so a's known bits reappear shifted
    // left by 2, and the low 2 bits of the product are known 0.
    const a = Bits{ .zeros = ~@as(u64, 0b0000_1010) & mask, .ones = 0b0000_1010 }; // a is exactly 10
    const b = Bits{ .zeros = ~@as(u64, 4) & mask, .ones = 4 };
    const r = binary(.mul, a, b, null, false, mask, 8);
    // 10 * 4 == 40 (0b0010_1000), and since a is fully known and b is fully known, the product is
    // fully known too.
    try testing.expectEqual(@as(u64, 40), r.ones & mask);
    try testing.expectEqual(mask, r.zeros | r.ones);
}

test "extendBits zero-extends with the high bits known 0" {
    // src is 4 bits, known-0 low nibble bit 0, bits 1..3 unknown. Zero-extended to 8 bits, the top 4
    // bits are known 0 regardless of what was known in src.
    const src = Bits{ .zeros = 0b0001, .ones = 0 };
    const r = extendBits(src, 4, 8, false);
    try testing.expectEqual(@as(u64, 0b1111_0001), r.zeros);
    try testing.expectEqual(@as(u64, 0), r.ones);
}

test "extendBits sign-extends a known-negative value with the high bits known 1" {
    // src is 4 bits, fully known -1 (0b1111): sign-extended to 8 bits it stays -1, so every bit of the
    // wider value is known 1.
    const src = Bits{ .zeros = 0, .ones = 0b1111 };
    const r = extendBits(src, 4, 8, true);
    try testing.expectEqual(@as(u64, 0), r.zeros);
    try testing.expectEqual(@as(u64, 0xFF), r.ones);
}

test "extendBits sign-extends a known-nonnegative value with the high bits known 0" {
    // src is 4 bits, sign bit (bit 3) known 0, low bits unknown: sign-extension fills the high bits
    // with that known-0 sign, so bits 4..7 of the 8-bit result are known 0 too.
    const src = Bits{ .zeros = 0b1000, .ones = 0 };
    const r = extendBits(src, 4, 8, true);
    try testing.expectEqual(@as(u64, 0b1111_1000), r.zeros);
    try testing.expectEqual(@as(u64, 0), r.ones);
}

test "extendBits leaves the high bits unknown when the sign bit is unknown" {
    // src's sign bit (bit 3) is unknown, so a sign-extend cannot know which way the high bits go.
    const src = Bits{};
    const r = extendBits(src, 4, 8, true);
    try testing.expectEqual(@as(u64, 0), r.zeros & 0b1111_0000);
    try testing.expectEqual(@as(u64, 0), r.ones & 0b1111_0000);
}

/// Fixture pairing `extendBits` with its concrete reference op, for a fixed `src_w` and `signed`, so
/// each instantiation is a plain `fn (u64, u16) u64` / `fn (Bits, u16) Bits` pair `assertSoundUnary`
/// can drive across every `dst_w`.
fn ExtendFixture(comptime src_w: u16, comptime signed: bool) type {
    return struct {
        fn real(v: u64, dst_w: u16) u64 {
            const src_mask = widthMask(src_w);
            const val = v & src_mask;
            if (!signed) return val; // zero-extend: nothing above src_w to fill
            const sign_bit = @as(u64, 1) << @intCast(src_w - 1);
            if (val & sign_bit == 0) return val; // nonnegative: nothing to fill
            const dst_mask = widthMask(dst_w);
            return val | (~src_mask & dst_mask); // negative: fill [src_w, dst_w) with 1s
        }
        fn transfer(a: Bits, dst_w: u16) Bits {
            return extendBits(a, src_w, dst_w, signed);
        }
    };
}

test "knownbits harness self-test: extendBits transfer is sound (zero-extend and sign-extend)" {
    comptime var src_w: u16 = 1;
    inline while (src_w <= 4) : (src_w += 1) {
        comptime var dst_w: u16 = src_w;
        inline while (dst_w <= 8) : (dst_w += 1) {
            const Zext = ExtendFixture(src_w, false);
            assertSoundUnary(Zext.real, Zext.transfer, src_w, dst_w);
            const Sext = ExtendFixture(src_w, true);
            assertSoundUnary(Sext.real, Sext.transfer, src_w, dst_w);
        }
    }
}

fn realTrunc(v: u64, dst_w: u16) u64 {
    return v & widthMask(dst_w);
}
fn transferTrunc(a: Bits, dst_w: u16) Bits {
    const dmask = widthMask(dst_w);
    return .{ .zeros = a.zeros & dmask, .ones = a.ones & dmask };
}

test "knownbits: convert trunc keeps low bits and is sound" {
    // Narrowing (or same-width) convert of a src_w-bit value down to dst_w bits: convertKnown's
    // narrow path is exactly "mask the source's known bits to the low dst_w bits", which is what
    // transferTrunc does here.
    var src_w: u16 = 2;
    while (src_w <= 5) : (src_w += 1) {
        var dst_w: u16 = 1;
        while (dst_w < src_w) : (dst_w += 1) {
            assertSoundUnary(realTrunc, transferTrunc, src_w, dst_w);
        }
    }
}

test "knownbits: a zext result has known-0 high bits (analyze)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u8t = try uintTy(&func, 8);
    const u32t = try uintTy(&func, 32);
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, u8t); // fully unknown 8-bit value
    const z = try func.appendInst(b, u32t, .{ .convert = .{ .value = p } }); // zext to u32
    func.setTerminator(b, .{ .ret = z });

    const bits = try analyze(allocator, &func);
    defer allocator.free(bits);
    // Zero-extension zero-fills the high bits regardless of the source's own known bits.
    const high_24: u64 = widthMask(32) & ~widthMask(8);
    try testing.expectEqual(high_24, bits[@intFromEnum(z)].zeros & high_24);
}

fn realMulhU(va: u64, vb: u64, w: u16) u64 {
    const mask = widthMask(w);
    const a = va & mask;
    const b = vb & mask;
    // 2w <= 10 for every w this harness drives, so the product fits a plain u64 multiply exactly.
    return ((a * b) >> @intCast(w)) & mask;
}
fn transferMulhU(a: Bits, b: Bits, w: u16) Bits {
    return mulhKnown(a, b, w, true);
}

/// Sign-extends the low `w` bits of `v` to a full `i64`, for building the signed mulh reference op.
fn sextW(v: u64, w: u16) i64 {
    const mask = widthMask(w);
    const val = v & mask;
    const sign_bit = @as(u64, 1) << @intCast(w - 1);
    if (val & sign_bit == 0) return @bitCast(val);
    return @bitCast(val | ~mask); // negative: fill every bit above w with 1s
}

fn realMulhS(va: u64, vb: u64, w: u16) u64 {
    const mask = widthMask(w);
    const a = sextW(va, w);
    const b = sextW(vb, w);
    // Exact in i64: |a|, |b| < 2^(w-1) with w <= 4 here, so the product is tiny next to i64 range.
    const prod = a * b;
    return (@as(u64, @bitCast(prod)) >> @intCast(w)) & mask;
}
fn transferMulhS(a: Bits, b: Bits, w: u16) Bits {
    return mulhKnown(a, b, w, false);
}

test "knownbits: mulh transfer is sound (unsigned)" {
    // 2w must fit u64 (trivially true here) and the harness itself wants w <= 5 for the exhaustive
    // sweep; mulh doubles the width internally so keep w small (1..4, giving 2w in 2..8).
    var w: u16 = 1;
    while (w <= 4) : (w += 1) {
        assertSoundBinary(realMulhU, transferMulhU, w);
    }
}

test "knownbits: mulh transfer is sound (signed)" {
    var w: u16 = 1;
    while (w <= 4) : (w += 1) {
        assertSoundBinary(realMulhS, transferMulhS, w);
    }
}

test "knownbits: mulh recovers a known-zero high half from small operands" {
    // Both operands confined to the low 2 bits of a 4-bit unsigned value (at most 3 each): the full
    // product is at most 9, which fits in the low 4 bits, so the high 4-bit half is entirely known 0.
    const mask4 = widthMask(4);
    const a = Bits{ .zeros = 0b1100, .ones = 0 }; // bits 2,3 known 0: a is at most 3
    const b = Bits{ .zeros = 0b1100, .ones = 0 }; // bits 2,3 known 0: b is at most 3
    const r = mulhKnown(a, b, 4, true);
    try testing.expectEqual(mask4, r.zeros);
}

test "knownbits: mulh recovers the exact high half when both operands are fully known" {
    // Fully-known unsigned 4-bit operands 13 and 9: 13 * 9 == 117 == 0b0111_0101, whose high 4 bits
    // (0b0111 == 7) are the mulh result.
    const mask4 = widthMask(4);
    const a = Bits{ .zeros = ~@as(u64, 13) & mask4, .ones = 13 };
    const b = Bits{ .zeros = ~@as(u64, 9) & mask4, .ones = 9 };
    const r = mulhKnown(a, b, 4, true);
    try testing.expectEqual(@as(u64, 7), r.ones & mask4);
    try testing.expectEqual(mask4, r.zeros | r.ones);
}

test "knownbits: mulh(signed) of two known-negative operands recovers the exact high half" {
    // Fully-known signed 4-bit operands -1 and -1 (0b1111 each): (-1) * (-1) == 1, whose 8-bit
    // representation is 0b0000_0001, so the high 4-bit half is exactly 0.
    const mask4 = widthMask(4);
    const neg_one = Bits{ .zeros = 0, .ones = mask4 };
    const r = mulhKnown(neg_one, neg_one, 4, false);
    try testing.expectEqual(mask4, r.zeros);
}

/// True for any pair the filtered div/rem harness must skip: `div`/`rem` are UB at a zero divisor,
/// so soundness is only claimed over `vb != 0`.
fn skipVbZero(_: u64, vb: u64) bool {
    return vb == 0;
}

fn realUDiv(va: u64, vb: u64, w: u16) u64 {
    return (va / vb) & widthMask(w);
}
fn transferUDiv(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.div, a, b, null, true, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}
fn realURem(va: u64, vb: u64, w: u16) u64 {
    return (va % vb) & widthMask(w);
}
fn transferURem(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.rem, a, b, null, true, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}

test "knownbits: udiv transfer is sound (vb != 0)" {
    var w: u16 = 1;
    while (w <= 4) : (w += 1) {
        assertSoundBinaryFiltered(realUDiv, transferUDiv, w, skipVbZero);
    }
}

test "knownbits: urem transfer is sound (vb != 0)" {
    var w: u16 = 1;
    while (w <= 4) : (w += 1) {
        assertSoundBinaryFiltered(realURem, transferURem, w, skipVbZero);
    }
}

fn realSDiv(va: u64, vb: u64, w: u16) u64 {
    const a = sextW(va, w);
    const b = sextW(vb, w);
    return @as(u64, @bitCast(@divTrunc(a, b))) & widthMask(w);
}
fn transferSDiv(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.div, a, b, null, false, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}
fn realSRem(va: u64, vb: u64, w: u16) u64 {
    const a = sextW(va, w);
    const b = sextW(vb, w);
    return @as(u64, @bitCast(@rem(a, b))) & widthMask(w);
}
fn transferSRem(a: Bits, b: Bits, w: u16) Bits {
    const mask = widthMask(w);
    const r = binary(.rem, a, b, null, false, mask, w);
    return .{ .zeros = r.zeros & mask, .ones = r.ones & mask };
}

test "knownbits: sdiv/srem stay unknown and are trivially sound" {
    // Signed div/rem round toward zero with a sign, which the transfer does not chase: it always
    // returns fully-unknown, which is trivially sound over any domain (including the filtered one).
    var w: u16 = 1;
    while (w <= 4) : (w += 1) {
        assertSoundBinaryFiltered(realSDiv, transferSDiv, w, skipVbZero);
        assertSoundBinaryFiltered(realSRem, transferSRem, w, skipVbZero);
    }
    // Confirm the transfer genuinely claims nothing, even when both operands are fully known (where
    // an unsigned div/rem of the same bit patterns would be exact): a real gap, not a coincidence.
    const mask = widthMask(8);
    const a = Bits{ .zeros = ~@as(u64, 6) & mask, .ones = 6 };
    const b = Bits{ .zeros = ~@as(u64, 2) & mask, .ones = 2 };
    const rd = binary(.div, a, b, null, false, mask, 8);
    try testing.expectEqual(Bits{}, rd);
    const rr = binary(.rem, a, b, null, false, mask, 8);
    try testing.expectEqual(Bits{}, rr);
}

test "knownbits: udiv by a known power of two is an exact logical shift" {
    const mask = widthMask(8);
    // b fully known 4 (2^2), a fully known 13: 13 / 4 == 3, and since both operands are fully known
    // the quotient is fully known too, recovered via the shift-right-logical fast path.
    const a = Bits{ .zeros = ~@as(u64, 13) & mask, .ones = 13 };
    const b = Bits{ .zeros = ~@as(u64, 4) & mask, .ones = 4 };
    const r = binary(.div, a, b, null, true, mask, 8);
    try testing.expectEqual(@as(u64, 3), r.ones & mask);
    try testing.expectEqual(mask, r.zeros | r.ones);
}

test "knownbits: urem by a known power of two keeps a's low bits and zeros the rest" {
    const mask = widthMask(8);
    // b fully known 4 (2^2): the remainder is a's low 2 bits, high bits known 0. a fully known 13
    // (0b0000_1101): 13 % 4 == 1, and since both operands are fully known the remainder is too.
    const a = Bits{ .zeros = ~@as(u64, 13) & mask, .ones = 13 };
    const b = Bits{ .zeros = ~@as(u64, 4) & mask, .ones = 4 };
    const r = binary(.rem, a, b, null, true, mask, 8);
    try testing.expectEqual(@as(u64, 1), r.ones & mask);
    try testing.expectEqual(mask, r.zeros | r.ones);
}

test "knownbits: udiv recovers known-zero high bits of the quotient from a bounded divisor" {
    const mask = widthMask(8);
    // a is fully unknown (dividend up to 255). b's bits 3..7 known 0 and bit 2 known 1 (b in
    // 4..7, so divisor_min == 4): q_max == 255 / 4 == 63 == 0b0011_1111, whose top 2 bits are known
    // 0, so the quotient's top 2 bits (6, 7) are known 0 too.
    const a = Bits{};
    const b = Bits{ .zeros = 0b1110_0000, .ones = 0b0000_0100 };
    const r = binary(.div, a, b, null, true, mask, 8);
    try testing.expectEqual(@as(u64, 0b1100_0000), r.zeros & 0b1100_0000);
}

test "knownbits: urem recovers known-zero high bits of the remainder from a bounded divisor" {
    const mask = widthMask(8);
    // a is fully unknown. b's bits 5..7 known 0 (so maxValue(b) == 31, bound == 30 == 0b0001_1110):
    // the remainder is always < b <= 31, needing at most 5 bits, so its top 3 bits (5, 6, 7) are
    // known 0.
    const a = Bits{};
    const b = Bits{ .zeros = 0b1110_0000, .ones = 0 };
    const r = binary(.rem, a, b, null, true, mask, 8);
    try testing.expectEqual(@as(u64, 0b1110_0000), r.zeros & 0b1110_0000);
}

// Task 7: the three downstream consumers (redundant mask, icmp eq/ne fold, redundant extend) now
// fire through the precise add/mul/convert transfers above, proven end to end with IR-level tests
// that build a Function, run the pass, and assert the specific rewrite fired.

test "knownbits: redundant mask removed via a convert's known-0 high bits" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u8t = try uintTy(&func, 8);
    const u32t = try uintTy(&func, 32);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, u8t);
    // z: a widening int->int convert, high 24 bits known 0 per convertKnown/extendBits.
    const z = try func.appendInst(b, u32t, .{ .convert = .{ .value = x } });
    // m clears only bits 8..31, which are already known 0 in z, so it is redundant.
    const m = try func.appendArithImm(b, u32t, .bit_and, z, 0xFF);
    func.setTerminator(b, .{ .ret = m });

    try testing.expect(try runOnce(allocator, &func));
    // The redundant `& 0xFF` now forwards `z` directly.
    try testing.expectEqual(z, func.terminator(b).?.ret.?);
}

test "knownbits: icmp eq folds to false via a convert's known-bit conflict" {
    const allocator = testing.allocator;
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const u8t = try uintTy(&func, 8);
        const u32t = try uintTy(&func, 32);
        const bool_t = try func.types.intern(.bool);
        const b = try func.appendBlock();
        const x = try func.appendBlockParam(b, u8t);
        const z = try func.appendInst(b, u32t, .{ .convert = .{ .value = x } }); // bit 8 known 0
        const c = try func.appendInst(b, u32t, .{ .iconst = 0x100 }); // bit 8 known 1: conflict
        const eq = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .eq, .lhs = z, .rhs = c } });
        func.setTerminator(b, .{ .ret = eq });

        try testing.expect(try runOnce(allocator, &func));
        try testing.expectEqual(@as(i64, 0), func.opcode(func.definingInst(eq).?).iconst);
    }
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const u8t = try uintTy(&func, 8);
        const u32t = try uintTy(&func, 32);
        const bool_t = try func.types.intern(.bool);
        const b = try func.appendBlock();
        const x = try func.appendBlockParam(b, u8t);
        const z = try func.appendInst(b, u32t, .{ .convert = .{ .value = x } });
        const c = try func.appendInst(b, u32t, .{ .iconst = 0x100 });
        const ne = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .ne, .lhs = z, .rhs = c } });
        func.setTerminator(b, .{ .ret = ne });

        try testing.expect(try runOnce(allocator, &func));
        try testing.expectEqual(@as(i64, 1), func.opcode(func.definingInst(ne).?).iconst);
    }
}

test "knownbits: redundant mask removed via an add's known-0 low bits" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try uintTy(&func, 32);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const a = try func.appendArithImm(b, t, .shl, x, 8); // a: low 8 bits known 0
    const c = try func.appendArithImm(b, t, .shl, y, 8); // c: low 8 bits known 0
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = c } });
    // s's low 8 bits are known 0 too (no carry can reach below bit 8), so clearing exactly those
    // bits is redundant.
    const clear_low_8: i64 = @bitCast(@as(u64, 0xFFFFFF00));
    const m = try func.appendArithImm(b, t, .bit_and, s, clear_low_8);
    func.setTerminator(b, .{ .ret = m });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(s, func.terminator(b).?.ret.?);
}

test "knownbits: redundant sign/zero extend still fires after the convert transfer became precise (no regression)" {
    // Covered directly by the existing round-trip tests above: "a zero-extend of a truncation of an
    // already-narrow value is eliminated" and "a sign-extend round-trip of a known-nonnegative value
    // is eliminated" both call redundantExtend, which reads the inner source's own known bits (not
    // the convert transfer added by Task 6). Re-running the same shape here confirms neither the add
    // nor the convert transfer precision regressed that path.
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
    try testing.expectEqual(s, func.terminator(b).?.ret.?); // round-trip recovered s, no regression
}
