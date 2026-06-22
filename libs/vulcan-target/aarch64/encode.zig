//! AArch64 (A64) instruction encoders. The host is aarch64, so these are validated
//! by native execution (see tests/native.zig), not just by encoding tables.

const std = @import("std");

/// A general register x0..x30. Index 31 is the zero register (wzr/xzr).
pub const Reg = enum(u5) {
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    zr,
};

fn n(reg: Reg) u32 {
    return @intFromEnum(reg);
}

/// `ret` (return to the address in x30, the link register).
pub fn ret() u32 {
    return 0xD65F0000 | (30 << 5);
}

/// `svc #imm16` (supervisor call, a Linux syscall, the call number in x8).
pub fn svc(imm: u16) u32 {
    return 0xD4000001 | (@as(u32, imm) << 5);
}

/// `add wd, wn, wm` (32-bit register add).
pub fn add(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x0B000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `sub wd, wn, wm` (32-bit register subtract).
pub fn sub(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4B000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `mul wd, wn, wm` (32-bit), i.e. `madd wd, wn, wm, wzr`.
pub fn mul(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x1B000000 | (n(rm) << 16) | (31 << 10) | (n(rn) << 5) | n(rd);
}

/// `and wd, wn, wm` (32-bit register bitwise and).
pub fn andr(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x0A000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `orr wd, wn, wm` (32-bit register bitwise or).
pub fn orr(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x2A000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `eor wd, wn, wm` (32-bit register bitwise exclusive-or).
pub fn eor(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4A000000 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `mov xd, xm` (64-bit register move), i.e. `orr xd, xzr, xm`. Always copies the
/// full register. 32-bit values carry zero in their high half (w-form ops
/// zero-extend), so this is also correct for them, and it preserves 64-bit values
/// (a 32-bit move would truncate i64/pointers passing through param setup, block-edge
/// moves, and returns).
pub fn mov(rd: Reg, rm: Reg) u32 {
    return sf64 | orr(rd, .zr, rm);
}

/// `movz wd, #imm16, lsl #(16*shift)` (move a 16-bit immediate, zeroing the rest).
pub fn movz(rd: Reg, imm: u16, shift: u2) u32 {
    return 0x52800000 | (@as(u32, shift) << 21) | (@as(u32, imm) << 5) | n(rd);
}

/// `movk wd, #imm16, lsl #(16*shift)` (move a 16-bit immediate, keeping the rest).
pub fn movk(rd: Reg, imm: u16, shift: u2) u32 {
    return 0x72800000 | (@as(u32, shift) << 21) | (@as(u32, imm) << 5) | n(rd);
}

/// `movz xd, #imm16, lsl #(16*shift)` (64-bit: zero the other 48 bits).
pub fn movz64(rd: Reg, imm: u16, shift: u2) u32 {
    return 0xD2800000 | (@as(u32, shift) << 21) | (@as(u32, imm) << 5) | n(rd);
}

/// `movk xd, #imm16, lsl #(16*shift)` (64-bit: keep the other bits).
pub fn movk64(rd: Reg, imm: u16, shift: u2) u32 {
    return 0xF2800000 | (@as(u32, shift) << 21) | (@as(u32, imm) << 5) | n(rd);
}

/// `add wd, wn, #imm12` (32-bit add of an unsigned 12-bit immediate).
pub fn addImm(rd: Reg, rn: Reg, imm: u12) u32 {
    return 0x11000000 | (@as(u32, imm) << 10) | (n(rn) << 5) | n(rd);
}

/// `sub wd, wn, #imm12` (32-bit subtract of an unsigned 12-bit immediate).
pub fn subImm(rd: Reg, rn: Reg, imm: u12) u32 {
    return 0x51000000 | (@as(u32, imm) << 10) | (n(rn) << 5) | n(rd);
}

/// AArch64 condition codes (the subset codegen emits).
pub const Cond = enum(u4) {
    eq = 0,
    ne = 1,
    hs = 2, // unsigned >=
    lo = 3, // unsigned <
    mi = 4, // negative (used for float <)
    hi = 8, // unsigned >
    ls = 9, // unsigned <=
    ge = 10, // signed >=
    lt = 11, // signed <
    gt = 12, // signed >
    le = 13, // signed <=
};

/// `cmp wn, wm` (32-bit compare, i.e. `subs wzr, wn, wm`): sets the flags.
pub fn cmp(rn: Reg, rm: Reg) u32 {
    return 0x6B00001F | (n(rm) << 16) | (n(rn) << 5);
}

/// `cset wd, cond` (32-bit): wd = 1 if `cond` holds, else 0. Encoded as
/// `csinc wd, wzr, wzr, invert(cond)`.
pub fn cset(rd: Reg, cond: Cond) u32 {
    const inv: u32 = @as(u4, @intFromEnum(cond)) ^ 1; // invert toggles the low bit
    return 0x1A800400 | (31 << 16) | (inv << 12) | (31 << 5) | n(rd);
}

/// `cbnz wt, label` (32-bit): branch if `wt` is non-zero. `off` is the signed
/// byte displacement from this instruction (a multiple of 4).
pub fn cbnz(rt: Reg, off: i21) u32 {
    const imm19: u32 = @as(u32, @bitCast(@as(i32, off) >> 2)) & 0x7FFFF;
    return 0x35000000 | (imm19 << 5) | n(rt);
}

/// `b label` (unconditional branch). `off` is the signed byte displacement.
pub fn b(off: i28) u32 {
    const imm26: u32 = @as(u32, @bitCast(@as(i32, off) >> 2)) & 0x3FFFFFF;
    return 0x14000000 | imm26;
}

/// `bl label` (branch with link: call, sets x30). Signed byte displacement.
pub fn bl(off: i28) u32 {
    const imm26: u32 = @as(u32, @bitCast(@as(i32, off) >> 2)) & 0x3FFFFFF;
    return 0x94000000 | imm26;
}

/// `blr xn` (indirect branch with link: call through a register).
pub fn blr(rn: Reg) u32 {
    return 0xD63F0000 | (n(rn) << 5);
}

/// `stp xt1, xt2, [sp, #imm]!` (64-bit, pre-index): save a register pair and move
/// sp by `imm` bytes (a multiple of 8). Use `.zr` for the sp base.
pub fn stpPre(rt1: Reg, rt2: Reg, rn: Reg, imm: i10) u32 {
    const imm7: u32 = @as(u32, @bitCast(@as(i32, imm) >> 3)) & 0x7F;
    return 0xA9800000 | (imm7 << 15) | (n(rt2) << 10) | (n(rn) << 5) | n(rt1);
}

/// `ldp xt1, xt2, [sp], #imm` (64-bit, post-index): restore a pair, then move sp.
pub fn ldpPost(rt1: Reg, rt2: Reg, rn: Reg, imm: i10) u32 {
    const imm7: u32 = @as(u32, @bitCast(@as(i32, imm) >> 3)) & 0x7F;
    return 0xA8C00000 | (imm7 << 15) | (n(rt2) << 10) | (n(rn) << 5) | n(rt1);
}

/// `str xt, [xn, #off]` (64-bit unsigned offset, `off` a multiple of 8).
pub fn strOff(rt: Reg, rn: Reg, off: u15) u32 {
    return 0xF9000000 | ((@as(u32, off) >> 3) << 10) | (n(rn) << 5) | n(rt);
}

/// `ldr xt, [xn, #off]` (64-bit unsigned offset).
pub fn ldrOff(rt: Reg, rn: Reg, off: u15) u32 {
    return 0xF9400000 | ((@as(u32, off) >> 3) << 10) | (n(rn) << 5) | n(rt);
}

/// `add xd, xn, #imm12` (64-bit add immediate). With `xn`/`xd` = `.zr` the base
/// is the stack pointer (Rn/Rd 31 means SP in the add/sub-immediate form).
pub fn addImm64(rd: Reg, rn: Reg, imm: u12) u32 {
    return 0x91000000 | (@as(u32, imm) << 10) | (n(rn) << 5) | n(rd);
}

/// `sub xd, xn, #imm12` (64-bit subtract immediate). Used to open a stack frame
/// (`sub sp, sp, #frame`).
pub fn subImm64(rd: Reg, rn: Reg, imm: u12) u32 {
    return 0xD1000000 | (@as(u32, imm) << 10) | (n(rn) << 5) | n(rd);
}

/// `str wt, [xn, #off]` (32-bit store, `off` a multiple of 4).
pub fn strW(rt: Reg, rn: Reg, off: u14) u32 {
    return 0xB9000000 | ((@as(u32, off) >> 2) << 10) | (n(rn) << 5) | n(rt);
}

/// `ldr wt, [xn, #off]` (32-bit load).
pub fn ldrW(rt: Reg, rn: Reg, off: u14) u32 {
    return 0xB9400000 | ((@as(u32, off) >> 2) << 10) | (n(rn) << 5) | n(rt);
}

/// `strb wt, [xn]` (store the low byte).
pub fn strb(rt: Reg, rn: Reg) u32 {
    return 0x39000000 | (n(rn) << 5) | n(rt);
}

/// `ldrsb wt, [xn]` (load a byte, sign-extended into the 32-bit register).
pub fn ldrsb(rt: Reg, rn: Reg) u32 {
    return 0x39C00000 | (n(rn) << 5) | n(rt);
}

/// `ldrb wt, [xn]` (load a byte, zero-extended).
pub fn ldrb(rt: Reg, rn: Reg) u32 {
    return 0x39400000 | (n(rn) << 5) | n(rt);
}

/// `strh wt, [xn]` (store the low halfword).
pub fn strh(rt: Reg, rn: Reg) u32 {
    return 0x79000000 | (n(rn) << 5) | n(rt);
}

/// `ldrh wt, [xn]` (load a halfword, zero-extended).
pub fn ldrh(rt: Reg, rn: Reg) u32 {
    return 0x79400000 | (n(rn) << 5) | n(rt);
}

/// `ldrsh wt, [xn]` (load a halfword, sign-extended into the 32-bit register).
pub fn ldrsh(rt: Reg, rn: Reg) u32 {
    return 0x79C00000 | (n(rn) << 5) | n(rt);
}

/// `sdiv wd, wn, wm` (32-bit signed divide).
pub fn sdiv(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x1AC00C00 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `udiv wd, wn, wm` (32-bit unsigned divide).
pub fn udiv(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x1AC00800 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `msub wd, wn, wm, wa` (32-bit): wd = wa - wn*wm. Used to form a remainder.
pub fn msub(rd: Reg, rn: Reg, rm: Reg, ra: Reg) u32 {
    return 0x1B008000 | (n(rm) << 16) | (n(ra) << 10) | (n(rn) << 5) | n(rd);
}

/// `lsl wd, wn, wm` (32-bit logical shift left by a register).
pub fn lslv(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x1AC02000 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `lsr wd, wn, wm` (32-bit logical shift right by a register).
pub fn lsrv(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x1AC02400 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `asr wd, wn, wm` (32-bit arithmetic shift right by a register).
pub fn asrv(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x1AC02800 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

// 64-bit (x-register) data processing. The `sf` bit (bit 31) widens the operation
// to 64 bits. These are needed for pointer/address arithmetic, which is 64-bit.
const sf64: u32 = 1 << 31;

/// `add xd, xn, xm` (64-bit register add).
pub fn add64(rd: Reg, rn: Reg, rm: Reg) u32 {
    return sf64 | add(rd, rn, rm);
}

/// `sub xd, xn, xm` (64-bit register subtract).
pub fn sub64(rd: Reg, rn: Reg, rm: Reg) u32 {
    return sf64 | sub(rd, rn, rm);
}

/// `mul xd, xn, xm` (64-bit), i.e. `madd xd, xn, xm, xzr`.
pub fn mul64(rd: Reg, rn: Reg, rm: Reg) u32 {
    return sf64 | mul(rd, rn, rm);
}

/// `lsl xd, xn, xm` (64-bit logical shift left by a register).
pub fn lslv64(rd: Reg, rn: Reg, rm: Reg) u32 {
    return sf64 | lslv(rd, rn, rm);
}

/// `lsr xd, xn, xm` (64-bit logical shift right by a register).
pub fn lsrv64(rd: Reg, rn: Reg, rm: Reg) u32 {
    return sf64 | lsrv(rd, rn, rm);
}

/// `asr xd, xn, xm` (64-bit arithmetic shift right by a register).
pub fn asrv64(rd: Reg, rn: Reg, rm: Reg) u32 {
    return sf64 | asrv(rd, rn, rm);
}

/// `csel wd, wn, wm, cond` (32-bit): wd = cond ? wn : wm.
pub fn csel(rd: Reg, rn: Reg, rm: Reg, cond: Cond) u32 {
    return 0x1A800000 | (n(rm) << 16) | (@as(u32, @intFromEnum(cond)) << 12) | (n(rn) << 5) | n(rd);
}

// FP register operands are also indices 0..31, naming v0..v31 (s/d views). `dbl`
// selects the double-precision (d) form, otherwise the single-precision (s) form.

fn fpType(dbl: bool) u32 {
    return if (dbl) @as(u32, 1) << 22 else 0; // the ftype field (S=00, D=01)
}

/// `fadd`/`fsub`/`fmul`/`fdiv` (scalar single/double).
pub fn fadd(rd: Reg, rn: Reg, rm: Reg, dbl: bool) u32 {
    return 0x1E202800 | fpType(dbl) | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}
pub fn fsub(rd: Reg, rn: Reg, rm: Reg, dbl: bool) u32 {
    return 0x1E203800 | fpType(dbl) | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}
pub fn fmul(rd: Reg, rn: Reg, rm: Reg, dbl: bool) u32 {
    return 0x1E200800 | fpType(dbl) | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}
pub fn fdiv(rd: Reg, rn: Reg, rm: Reg, dbl: bool) u32 {
    return 0x1E201800 | fpType(dbl) | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

// NEON: 128-bit vectors (the v0..v31 registers, Q view).

/// `ldr qt, [xn, #off]` / `str qt, [xn, #off]`: a 128-bit SIMD&FP load/store. `off` is a
/// byte offset, scaled by 16 (so it must be 16-aligned), 0 for the common `[xn]` case.
pub fn ldrQ(rt: Reg, rn: Reg, off: u16) u32 {
    return 0x3DC00000 | ((@as(u32, off) >> 4) << 10) | (n(rn) << 5) | n(rt);
}
pub fn strQ(rt: Reg, rn: Reg, off: u16) u32 {
    return 0x3D800000 | ((@as(u32, off) >> 4) << 10) | (n(rn) << 5) | n(rt);
}

/// NEON `fadd`/`fsub`/`fmul`/`fdiv` Vd.4S, Vn.4S, Vm.4S: lane-wise over 4 single-precision
/// floats packed in a 128-bit register.
pub fn faddVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4E20D400 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}
pub fn fsubVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4EA0D400 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}
pub fn fmulVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x6E20DC00 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}
pub fn fdivVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x6E20FC00 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `dup sd, vn.s[index]`: copy single-precision lane `index` of a vector to a scalar
/// register (the FP register file's S view). The standard way to extract a NEON lane.
pub fn dupLane(rd: Reg, rn: Reg, index: u2) u32 {
    const imm5: u32 = (@as(u32, index) << 3) | 0b100; // S element: index in bits[4:3]
    return 0x5E000400 | (imm5 << 16) | (n(rn) << 5) | n(rd);
}

/// `ins vd.s[lane], vn.s[0]`: insert single-precision lane 0 of `vn` into lane `lane` of
/// `vd`. The standard way to build a NEON vector from scalar registers a lane at a time.
pub fn insLane(rd: Reg, lane: u2, rn: Reg) u32 {
    const imm5: u32 = (@as(u32, lane) << 3) | 0b100; // destination S element
    return 0x6E000400 | (imm5 << 16) | (n(rn) << 5) | n(rd); // imm4 = 0 -> source lane 0
}

/// `mov vd.16b, vn.16b` (an alias for `orr vd.16b, vn.16b, vn.16b`): copy a whole 128-bit
/// vector register. Use this for vector register moves. `fmovReg` only copies 64 bits.
pub fn movVec(rd: Reg, rn: Reg) u32 {
    return 0x4EA01C00 | (n(rn) << 16) | (n(rn) << 5) | n(rd);
}

/// `fcmp sn, sm` / `fcmp dn, dm`: set the flags from a floating-point compare.
pub fn fcmp(rn: Reg, rm: Reg, dbl: bool) u32 {
    return 0x1E202000 | fpType(dbl) | (n(rm) << 16) | (n(rn) << 5);
}

/// `fcsel dd, dn, dm, cond` (and single form): dd = cond ? dn : dm.
pub fn fcsel(rd: Reg, rn: Reg, rm: Reg, cond: Cond, dbl: bool) u32 {
    return 0x1E200C00 | fpType(dbl) | (n(rm) << 16) | (@as(u32, @intFromEnum(cond)) << 12) | (n(rn) << 5) | n(rd);
}

/// `fmov dd, dn` (a 64-bit FP register move, copies the low 64 bits, which covers
/// both the single and double views).
pub fn fmovReg(rd: Reg, rn: Reg) u32 {
    return 0x1E604000 | (n(rn) << 5) | n(rd);
}

/// `fmov sd, wn` / `fmov dd, xn`: move general-register bits into an FP register.
pub fn fmovFromGpr(rd: Reg, rn: Reg, dbl: bool) u32 {
    return if (dbl) 0x9E670000 | (n(rn) << 5) | n(rd) else 0x1E270000 | (n(rn) << 5) | n(rd);
}

/// `fmov wd/xd, sn/dn`: move FP-register bits into a general register (reinterpret).
pub fn fmovToGpr(rd: Reg, rn: Reg, dbl: bool) u32 {
    return if (dbl) 0x9E660000 | (n(rn) << 5) | n(rd) else 0x1E260000 | (n(rn) << 5) | n(rd);
}

/// `fsqrt sd/dd, sn/dn`.
pub fn fsqrt(rd: Reg, rn: Reg, dbl: bool) u32 {
    return (if (dbl) @as(u32, 0x1E61C000) else 0x1E21C000) | (n(rn) << 5) | n(rd);
}

/// `frint<mode> sd/dd, sn/dn`: round to integral float. `base` selects the mode
/// (frintn nearest-even, frintp +inf, frintm -inf, frintz toward-zero).
fn frint(base: u32, rd: Reg, rn: Reg, dbl: bool) u32 {
    return (if (dbl) base | 0x00400000 else base) | (n(rn) << 5) | n(rd);
}
pub fn frintn(rd: Reg, rn: Reg, dbl: bool) u32 {
    return frint(0x1E244000, rd, rn, dbl);
}
pub fn frintp(rd: Reg, rn: Reg, dbl: bool) u32 {
    return frint(0x1E24C000, rd, rn, dbl);
}
pub fn frintm(rd: Reg, rn: Reg, dbl: bool) u32 {
    return frint(0x1E254000, rd, rn, dbl);
}
pub fn frintz(rd: Reg, rn: Reg, dbl: bool) u32 {
    return frint(0x1E25C000, rd, rn, dbl);
}

/// `scvtf`/`ucvtf` sd/dd, wn: convert a 32-bit integer to floating point.
pub fn cvtIntToFloat(rd: Reg, rn: Reg, dbl: bool, signed: bool) u32 {
    const base: u32 = if (signed) 0x1E220000 else 0x1E230000;
    return base | fpType(dbl) | (n(rn) << 5) | n(rd);
}

/// `fcvtzs`/`fcvtzu` wd, sn/dn: convert floating point to a 32-bit integer
/// (round toward zero).
pub fn cvtFloatToInt(rd: Reg, rn: Reg, dbl_src: bool, signed: bool) u32 {
    const base: u32 = if (signed) 0x1E380000 else 0x1E390000;
    return base | fpType(dbl_src) | (n(rn) << 5) | n(rd);
}

/// `fcvt`: single<->double precision conversion. `to_double` widens s->d, else d->s.
pub fn fcvt(rd: Reg, rn: Reg, to_double: bool) u32 {
    return if (to_double) 0x1E22C000 | (n(rn) << 5) | n(rd) else 0x1E624000 | (n(rn) << 5) | n(rd);
}

/// `str st/dt, [xn, #off]` (FP store).
pub fn strFp(rt: Reg, rn: Reg, off: u15, dbl: bool) u32 {
    return if (dbl)
        0xFD000000 | ((@as(u32, off) >> 3) << 10) | (n(rn) << 5) | n(rt)
    else
        0xBD000000 | ((@as(u32, off) >> 2) << 10) | (n(rn) << 5) | n(rt);
}

/// `ldr st/dt, [xn, #off]` (FP load).
pub fn ldrFp(rt: Reg, rn: Reg, off: u15, dbl: bool) u32 {
    return if (dbl)
        0xFD400000 | ((@as(u32, off) >> 3) << 10) | (n(rn) << 5) | n(rt)
    else
        0xBD400000 | ((@as(u32, off) >> 2) << 10) | (n(rn) << 5) | n(rt);
}

test "known A64 encodings" {
    // Confirmed by native JIT execution: `add w0, w0, #42` and `ret`.
    try std.testing.expectEqual(@as(u32, 0x1100A800), addImm(.x0, .x0, 42));
    try std.testing.expectEqual(@as(u32, 0xD65F03C0), ret());
    // `movz w0, #42` (used as a JIT smoke-test instruction).
    try std.testing.expectEqual(@as(u32, 0x52800540), movz(.x0, 42, 0));
}

test "64-bit ALU sets the sf bit (bit 31) over the 32-bit form" {
    // The 64-bit forms are exactly the 32-bit ones with sf (bit 31) set.
    try std.testing.expectEqual(add(.x0, .x1, .x2) | (@as(u32, 1) << 31), add64(.x0, .x1, .x2));
    try std.testing.expectEqual(@as(u32, 0x8B020020), add64(.x0, .x1, .x2)); // add x0, x1, x2
    try std.testing.expectEqual(@as(u32, 0xCB020020), sub64(.x0, .x1, .x2)); // sub x0, x1, x2
    try std.testing.expectEqual(@as(u32, 0x9AC22020), lslv64(.x0, .x1, .x2)); // lsl x0, x1, x2
}
