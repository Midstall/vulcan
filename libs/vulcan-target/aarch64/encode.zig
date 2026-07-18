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

/// `nop` (no-operation, used to pad a loop header up to the fetch-alignment boundary).
pub fn nop() u32 {
    return 0xD503201F;
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

/// `smulh xd, xn, xm` (high 64 bits of the signed 64x64 product). Ra (bits 14:10) is 31.
pub fn smulh(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x9B407C00 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// `umulh xd, xn, xm` (high 64 bits of the unsigned 64x64 product). Ra (bits 14:10) is 31.
pub fn umulh(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x9BC07C00 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
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

/// `b.cond label` (conditional branch): branch if `cond` holds. `off` is the signed
/// byte displacement from this instruction (a multiple of 4), the same convention as
/// `cbnz`/`b` (patched later). Bit layout: [31:24]=0x54, [23:5]=imm19 (the offset in
/// words), [4]=0, [3:0]=cond.
pub fn bcc(cond: Cond, off: i21) u32 {
    const imm19: u32 = @as(u32, @bitCast(@as(i32, off) >> 2)) & 0x7FFFF;
    return 0x54000000 | (imm19 << 5) | @as(u32, @intFromEnum(cond));
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

/// `prfm pldl1keep, [xn]` (prefetch hint, no architectural effect on results,
/// only a microarchitectural hint to bring `[xn]` into L1). prfop = PLDL1KEEP
/// (0b00000), imm12 = 0 (no offset): bits [31:22]=0b1111100110, [21:10]=imm12,
/// [9:5]=Rn, [4:0]=Rt(=prfop).
pub fn prfm(rn: Reg) u32 {
    return 0xF9800000 | (n(rn) << 5);
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

/// `add xd, xn, #imm12, LSL #12` (the shifted add-immediate form, sh=1). The
/// imm12 is scaled by 4096, so this reaches stack frames wider than 12 bits when
/// paired with the unshifted form. Rn/Rd 31 still means SP here (add/sub-immediate),
/// which the shifted-register form would instead read as XZR.
pub fn addImm64Shift(rd: Reg, rn: Reg, imm: u12) u32 {
    return 0x91400000 | (@as(u32, imm) << 10) | (n(rn) << 5) | n(rd);
}

/// `sub xd, xn, #imm12, LSL #12` (the shifted sub-immediate form, sh=1).
pub fn subImm64Shift(rd: Reg, rn: Reg, imm: u12) u32 {
    return 0xD1400000 | (@as(u32, imm) << 10) | (n(rn) << 5) | n(rd);
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

// FP register operands are also indices 0..31, naming v0..v31 (s/d/h views). The scalar
// FP encodings carry a 2-bit `ftype` field at bits [23:22] selecting the operand precision.

/// The scalar-FP `ftype` selector (bits [23:22]). `single`/`double` are the base-ISA S/D
/// forms; `half` is the FEAT_FP16 H form for NATIVE 16-bit arithmetic. Callers that only ever
/// deal with f32/f64 (the pre-f16 world) pass `.single`/`.double`, whose bits are byte-identical
/// to the old `fpType(false)`/`fpType(true)`, so those encodings do not change. Only the native
/// f16 path (gated on the model's FEAT_FP16) passes `.half`.
pub const FKind = enum { single, double, half };

fn ftype(kind: FKind) u32 {
    return switch (kind) {
        .single => 0, // ftype = 00
        .double => @as(u32, 1) << 22, // ftype = 01
        .half => @as(u32, 3) << 22, // ftype = 11 (FEAT_FP16). H-form words confirmed with `as` on-host.
    };
}

/// `fadd`/`fsub`/`fmul`/`fdiv` (scalar single/double/half).
pub fn fadd(rd: Reg, rn: Reg, rm: Reg, kind: FKind) u32 {
    return 0x1E202800 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}
pub fn fsub(rd: Reg, rn: Reg, rm: Reg, kind: FKind) u32 {
    return 0x1E203800 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}
pub fn fmul(rd: Reg, rn: Reg, rm: Reg, kind: FKind) u32 {
    return 0x1E200800 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}
pub fn fdiv(rd: Reg, rn: Reg, rm: Reg, kind: FKind) u32 {
    return 0x1E201800 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

// Floating-point data-processing (3-source), base 0x1F000000: Rd = op(Rn*Rm, Ra), fused
// into a single instruction so the product is never separately rounded (one rounding
// instead of two). Layout: ftype (23:22) selects single/double as above, o1 (21) and o0
// (15) pick the variant, Rm (20:16), Ra (14:10, the accumulator), Rn (9:5), Rd (4:0).
// Confirmed against the running aarch64 host: assembled with `as`, disassembled with
// `objdump`, and executed to check the arithmetic (fmadd = Ra+Rn*Rm, fmsub = Ra-Rn*Rm,
// fnmsub = Rn*Rm-Ra), not just the bit layout copied from the ARM ARM.

// The fused 3-source ops are only ever emitted for f32/f64 (f16 fusion is disabled in isel, so
// the intermediate multiply still rounds to half per-op), so they keep a plain `dbl` selector.

/// `fmadd sd/dd, sn, sm, sa`: Rd = Ra + Rn*Rm (o1=0, o0=0).
pub fn fmadd(rd: Reg, rn: Reg, rm: Reg, ra: Reg, dbl: bool) u32 {
    return 0x1F000000 | ftype(if (dbl) .double else .single) | (n(rm) << 16) | (n(ra) << 10) | (n(rn) << 5) | n(rd);
}

/// `fmsub sd/dd, sn, sm, sa`: Rd = Ra - Rn*Rm (o1=0, o0=1).
pub fn fmsub(rd: Reg, rn: Reg, rm: Reg, ra: Reg, dbl: bool) u32 {
    return 0x1F008000 | ftype(if (dbl) .double else .single) | (n(rm) << 16) | (n(ra) << 10) | (n(rn) << 5) | n(rd);
}

/// `fnmsub sd/dd, sn, sm, sa`: Rd = Rn*Rm - Ra (o1=1, o0=1).
pub fn fnmsub(rd: Reg, rn: Reg, rm: Reg, ra: Reg, dbl: bool) u32 {
    return 0x1F208000 | ftype(if (dbl) .double else .single) | (n(rm) << 16) | (n(ra) << 10) | (n(rn) << 5) | n(rd);
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

/// NEON `fmla Vd.4S, Vn.4S, Vm.4S`: Vd = Vd + Vn*Vm (ACCUMULATES into Vd, like sdot/udot,
/// so the caller must place the running value in `rd` before this executes). Confirmed
/// against this aarch64 host: assembled with `as`, disassembled with `objdump` for the bit
/// layout, and executed (lanes [2,3,4,5]*[10,10,10,10]+[1,1,1,1] -> [21,31,41,51]) to check
/// the arithmetic, not just copied from the ARM ARM.
pub fn fmlaVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4E20CC00 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// NEON `fmls Vd.4S, Vn.4S, Vm.4S`: Vd = Vd - Vn*Vm (ACCUMULATES into Vd). Bit 23 set over
/// `fmlaVec`. Confirmed the same way (lanes [100]*4 - [2,3,4,5]*[10,10,10,10] -> [80,70,60,50]).
pub fn fmlsVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4EA0CC00 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// NEON `fneg Vd.4S, Vn.4S`: lane-wise floating-point negate over 4 single floats.
pub fn fnegVec(rd: Reg, rn: Reg) u32 {
    return 0x6EA0F800 | (n(rn) << 5) | n(rd);
}

/// NEON `fsqrt Vd.4S, Vn.4S`: lane-wise floating-point square root over 4 single floats.
pub fn fsqrtVec(rd: Reg, rn: Reg) u32 {
    return 0x6EA1F800 | (n(rn) << 5) | n(rd);
}

/// NEON `fmin Vd.4S, Vn.4S, Vm.4S`: lane-wise minimum (IEEE min) over 4 single floats.
pub fn fminVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4EA0F400 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// NEON `fmax Vd.4S, Vn.4S, Vm.4S`: lane-wise maximum (IEEE max) over 4 single floats.
pub fn fmaxVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4E20F400 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

// NEON lane-wise floating-point compares produce a per-lane MASK (all-ones 0xFFFFFFFF
// when the relation holds, all-zeros otherwise) - exactly what `bsl` consumes for a
// branch-free masked blend. `fcmeq`/`fcmgt`/`fcmge` are the >, >=, == primitives.
// <, <=, != are formed by swapping operands or by `mvn`-ing an `fcmeq`.

/// NEON `fcmeq Vd.4S, Vn.4S, Vm.4S`: per-lane equal mask.
pub fn fcmeqVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4E20E400 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// NEON `fcmgt Vd.4S, Vn.4S, Vm.4S`: per-lane greater-than mask (Vn > Vm).
pub fn fcmgtVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x6EA0E400 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// NEON `fcmge Vd.4S, Vn.4S, Vm.4S`: per-lane greater-or-equal mask (Vn >= Vm).
pub fn fcmgeVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x6E20E400 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// NEON `mvn Vd.16B, Vn.16B` (alias of `not`): bitwise complement of a whole 128-bit
/// register. Turns an `fcmeq` mask into a `!=` mask, or inverts any lane mask.
pub fn mvnVec(rd: Reg, rn: Reg) u32 {
    return 0x6E205800 | (n(rn) << 5) | n(rd);
}

/// NEON `bsl Vd.16B, Vn.16B, Vm.16B`: per-bit select - Vd = (Vn & Vd) | (Vm & ~Vd).
/// With Vd holding a lane mask, this picks Vn's lane where the mask is set, Vm's where
/// clear: the branch-free masked blend for a vectorized OpSelect / if-then-else.
pub fn bslVec(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x6E601C00 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// NEON `dup Vd.4S, Wn`: splat a 32-bit general register into all 4 lanes of a vector.
/// Materializes a scalar (an interpolated varying, a constant) as a 4-lane vector.
pub fn dupFromGpr(rd: Reg, rn: Reg) u32 {
    return 0x4E040C00 | (n(rn) << 5) | n(rd);
}

/// NEON `dup Vd.4S, Vn.s[index]`: splat single-precision lane `index` across all 4 lanes.
pub fn dupVecLane(rd: Reg, rn: Reg, index: u2) u32 {
    const imm5: u32 = (@as(u32, index) << 3) | 0b100; // S element: index in bits[4:3]
    return 0x4E000400 | (imm5 << 16) | (n(rn) << 5) | n(rd);
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

/// NEON `sdot Vd.4S, Vn.16B, Vm.16B` (Altra: `features.aarch64.dotprod`): the 4-way
/// signed INT8 dot-product-accumulate. For each 32-bit lane d in 0..3, Vd[d] +=
/// sum over k in 0..3 of sext(Vn.b[4d+k]) * sext(Vm.b[4d+k]). ACCUMULATES into Vd,
/// so the caller must place the running sum in `rd` before this executes.
pub fn sdot(rd: Reg, rn: Reg, rm: Reg) u32 {
    return 0x4E809400 | (n(rm) << 16) | (n(rn) << 5) | n(rd);
}

/// NEON `udot Vd.4S, Vn.16B, Vm.16B`: the unsigned form of `sdot` (zero-extends each
/// int8 lane instead of sign-extending). Bit 29 (U) is the only difference from `sdot`.
pub fn udot(rd: Reg, rn: Reg, rm: Reg) u32 {
    return sdot(rd, rn, rm) | (1 << 29);
}

/// `fcmp sn, sm` / `fcmp dn, dm` / `fcmp hn, hm`: set the flags from a floating-point compare.
pub fn fcmp(rn: Reg, rm: Reg, kind: FKind) u32 {
    return 0x1E202000 | ftype(kind) | (n(rm) << 16) | (n(rn) << 5);
}

/// `fcsel dd, dn, dm, cond` (and single/half forms): dd = cond ? dn : dm.
pub fn fcsel(rd: Reg, rn: Reg, rm: Reg, cond: Cond, kind: FKind) u32 {
    return 0x1E200C00 | ftype(kind) | (n(rm) << 16) | (@as(u32, @intFromEnum(cond)) << 12) | (n(rn) << 5) | n(rd);
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

/// `fmov hd, wn` (FEAT_FP16): move the low 16 bits of a general register into the H view,
/// materializing a NATIVE half from its raw IEEE bit pattern. ftype=11, sf=0, opcode=111.
/// Confirmed: fmov h0,w1 -> 0x1EE70020 (`as`/`objdump` on this aarch64 host).
pub fn fmovHfromGpr(rd: Reg, rn: Reg) u32 {
    return 0x1EE70000 | (n(rn) << 5) | n(rd);
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

/// `scvtf`/`ucvtf` sd/dd/hd, wn: convert a 32-bit integer to floating point. `kind` selects the
/// destination precision (`.half` is the native FEAT_FP16 `scvtf hd, wn`, confirmed on-host).
pub fn cvtIntToFloat(rd: Reg, rn: Reg, kind: FKind, signed: bool) u32 {
    const base: u32 = if (signed) 0x1E220000 else 0x1E230000;
    return base | ftype(kind) | (n(rn) << 5) | n(rd);
}

/// `fcvtzs`/`fcvtzu` wd, sn/dn/hn: convert floating point to a 32-bit integer (round toward
/// zero). `kind` selects the SOURCE precision (`.half` is the native FEAT_FP16 `fcvtzs wd, hn`).
pub fn cvtFloatToInt(rd: Reg, rn: Reg, kind: FKind, signed: bool) u32 {
    const base: u32 = if (signed) 0x1E380000 else 0x1E390000;
    return base | ftype(kind) | (n(rn) << 5) | n(rd);
}

/// `fcvt`: single<->double precision conversion. `to_double` widens s->d, else d->s.
pub fn fcvt(rd: Reg, rn: Reg, to_double: bool) u32 {
    return if (to_double) 0x1E22C000 | (n(rn) << 5) | n(rd) else 0x1E624000 | (n(rn) << 5) | n(rd);
}

// Half-precision FCVT (float<->float). These are BASE-ISA (Armv8.0-A): only half-precision
// *arithmetic* needs FEAT_FP16, the precision conversions do not. The f16 emulation holds a
// half value as its f32 widening in an S register, so `fcvtSfromH` recovers that widening at
// an f16 memory boundary and `fcvtHfromS`/`fcvtHfromD` perform the round-to-nearest-even to
// half that every f16 result / narrowing convert must do. FCVT layout: ftype (23:22) = source
// precision (00 single, 01 double, 11 half), opc (16:15) = destination precision (00 single,
// 01 double, 11 half). Golden words confirmed with `as`/`objdump` on this aarch64 host.

/// `fcvt sd, hn` (widen an IEEE half in the H view to single, exact). ftype=11 (half source),
/// opc=00 (single dest). Confirmed: fcvt s0,h0 -> 0x1EE24000.
pub fn fcvtSfromH(rd: Reg, rn: Reg) u32 {
    return 0x1EE24000 | (n(rn) << 5) | n(rd);
}

/// `fcvt hd, sn` (narrow a single to an IEEE half, round-to-nearest-even). ftype=00 (single
/// source), opc=11 (half dest). Confirmed: fcvt h0,s0 -> 0x1E23C000.
pub fn fcvtHfromS(rd: Reg, rn: Reg) u32 {
    return 0x1E23C000 | (n(rn) << 5) | n(rd);
}

/// `fcvt hd, dn` (narrow a double directly to an IEEE half, single round-to-nearest-even).
/// Used for f64->f16 so the result rounds ONCE (matching Zig's `@as(f16, d)`), rather than
/// double-rounding through single. ftype=01 (double source), opc=11 (half dest). Confirmed:
/// fcvt h0,d0 -> 0x1E63C000.
pub fn fcvtHfromD(rd: Reg, rn: Reg) u32 {
    return 0x1E63C000 | (n(rn) << 5) | n(rd);
}

/// `fcvt dd, hn` (widen a native IEEE half in the H view directly to double, exact - a half is
/// exactly representable in double). Used only by the native f16 path for f16->f64; the emulation
/// path widens the S-held f32 form with the base `fcvt d,s` instead. ftype=11 (half source),
/// opc=01 (double dest). Confirmed: fcvt d0,h1 -> 0x1EE2C020 (`as`/`objdump` on-host).
pub fn fcvtDfromH(rd: Reg, rn: Reg) u32 {
    return 0x1EE2C000 | (n(rn) << 5) | n(rd);
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

/// `ldr ht, [xn, #off]` (16-bit SIMD&FP load, size=01; `off` a multiple of 2). Reads a
/// 16-bit IEEE-half memory object into the H view. This is the SIMD&FP form (target is an
/// h-register), distinct from the GPR `ldrh` above (target is a w-register). Confirmed:
/// ldr h0,[x0] -> 0x7D400000, ldr h1,[x2,#8] -> 0x7D401041.
pub fn ldrHfp(rt: Reg, rn: Reg, off: u13) u32 {
    // The immediate is scaled by the 2-byte access size, so a byte offset must be halfword
    // aligned; an odd offset would silently lose its low bit here. Callers only pass 0 today.
    std.debug.assert(off % 2 == 0);
    return 0x7D400000 | ((@as(u32, off) >> 1) << 10) | (n(rn) << 5) | n(rt);
}

/// `str ht, [xn, #off]` (16-bit SIMD&FP store; `off` a multiple of 2). Writes the H view of
/// an already-half value to a 16-bit IEEE-half memory object. Confirmed: str h0,[x0] ->
/// 0x7D000000.
pub fn strHfp(rt: Reg, rn: Reg, off: u13) u32 {
    std.debug.assert(off % 2 == 0); // halfword-aligned; see ldrHfp
    return 0x7D000000 | ((@as(u32, off) >> 1) << 10) | (n(rn) << 5) | n(rt);
}

test "known A64 encodings" {
    // Confirmed by native JIT execution: `add w0, w0, #42` and `ret`.
    try std.testing.expectEqual(@as(u32, 0x1100A800), addImm(.x0, .x0, 42));
    try std.testing.expectEqual(@as(u32, 0xD65F03C0), ret());
    // `movz w0, #42` (used as a JIT smoke-test instruction).
    try std.testing.expectEqual(@as(u32, 0x52800540), movz(.x0, 42, 0));
}

test "smulh / umulh encoding (high half of a 64x64 product)" {
    try std.testing.expectEqual(@as(u32, 0x9b4a7d28), smulh(.x8, .x9, .x10));
    try std.testing.expectEqual(@as(u32, 0x9bca7d28), umulh(.x8, .x9, .x10));
}

test "nop encoding" {
    try std.testing.expectEqual(@as(u32, 0xD503201F), nop());
}

test "prfm encoding" {
    // prfm pldl1keep, [x0]  ->  0xF9800000 | (0 << 5) = 0xF9800000
    try std.testing.expectEqual(@as(u32, 0xF9800000), prfm(.x0));
    // prfm pldl1keep, [x5]  ->  0xF9800000 | (5 << 5) = 0xF98000A0
    try std.testing.expectEqual(@as(u32, 0xF98000A0), prfm(.x5));
}

test "B.cond conditional-branch encoding" {
    // Hand-computed from the bit layout: 0x54000000 | (imm19<<5) | cond, imm19 = off/4.
    // b.lt .+0  -> imm19=0, cond=lt(11) -> 0x5400000B
    try std.testing.expectEqual(@as(u32, 0x5400000B), bcc(.lt, 0));
    // b.eq .+8  -> imm19=2, cond=eq(0)  -> 0x54000000 | (2<<5) = 0x54000040
    try std.testing.expectEqual(@as(u32, 0x54000040), bcc(.eq, 8));
    // b.gt .+12 -> imm19=3, cond=gt(12) -> 0x54000060 | 0x0C = 0x5400006C
    try std.testing.expectEqual(@as(u32, 0x5400006C), bcc(.gt, 12));
    // b.ge .-4  -> imm19=-1 (0x7FFFF), cond=ge(10) -> 0x54FFFFE0 | 0x0A = 0x54FFFFEA
    try std.testing.expectEqual(@as(u32, 0x54FFFFEA), bcc(.ge, -4));
    // bit[4] is always 0 (distinguishes B.cond from the consistent-conditional forms).
    try std.testing.expectEqual(@as(u32, 0), bcc(.hi, 20) & 0x10);
}

test "NEON vector op encodings" {
    // Cross-checked against the ARM A64 reference encodings (.4S form, v0/v1/v2).
    try std.testing.expectEqual(@as(u32, 0x6EA0F800), fnegVec(.x0, .x0)); // fneg v0.4s, v0.4s
    try std.testing.expectEqual(@as(u32, 0x6EA1F800), fsqrtVec(.x0, .x0)); // fsqrt v0.4s, v0.4s
    try std.testing.expectEqual(@as(u32, 0x4EA2F400), fminVec(.x0, .x0, .x2)); // fmin v0.4s,v0.4s,v2.4s
    try std.testing.expectEqual(@as(u32, 0x4E22F420), fmaxVec(.x0, .x1, .x2)); // fmax v0.4s,v1.4s,v2.4s
    try std.testing.expectEqual(@as(u32, 0x4E22E420), fcmeqVec(.x0, .x1, .x2)); // fcmeq v0.4s,v1.4s,v2.4s
    try std.testing.expectEqual(@as(u32, 0x6EA2E420), fcmgtVec(.x0, .x1, .x2)); // fcmgt v0.4s,v1.4s,v2.4s
    try std.testing.expectEqual(@as(u32, 0x6E22E420), fcmgeVec(.x0, .x1, .x2)); // fcmge v0.4s,v1.4s,v2.4s
    try std.testing.expectEqual(@as(u32, 0x6E621C20), bslVec(.x0, .x1, .x2)); // bsl v0.16b,v1.16b,v2.16b
    try std.testing.expectEqual(@as(u32, 0x4E040C20), dupFromGpr(.x0, .x1)); // dup v0.4s, w1
    try std.testing.expectEqual(@as(u32, 0x4E140420), dupVecLane(.x0, .x1, 2)); // dup v0.4s, v1.s[2]
    try std.testing.expectEqual(@as(u32, 0x6E205820), mvnVec(.x0, .x1)); // mvn v0.16b, v1.16b
}

test "FMLA/FMLS vector encodings (NEON accumulate-into-Vd)" {
    // Golden words obtained by assembling `fmla/fmls v{0,1}.4s, v{1,2}.4s, v{2,3}.4s` with
    // `as` and disassembling with `objdump` on this aarch64 host.
    try std.testing.expectEqual(@as(u32, 0x4E22CC20), fmlaVec(.x0, .x1, .x2)); // fmla v0.4s,v1.4s,v2.4s
    try std.testing.expectEqual(@as(u32, 0x4E23CC41), fmlaVec(.x1, .x2, .x3)); // fmla v1.4s,v2.4s,v3.4s
    try std.testing.expectEqual(@as(u32, 0x4EA2CC20), fmlsVec(.x0, .x1, .x2)); // fmls v0.4s,v1.4s,v2.4s
    try std.testing.expectEqual(@as(u32, 0x4EA3CC41), fmlsVec(.x1, .x2, .x3)); // fmls v1.4s,v2.4s,v3.4s
}

test "SDOT/UDOT encodings (Altra INT8 dot-product)" {
    // Cross-checked against the ARM A64 reference encoding (Vd.4S, Vn.16B, Vm.16B):
    // sdot v0.4s, v0.4s, v0.4s -> 0x4E809400, the fixed base with every register field zero.
    try std.testing.expectEqual(@as(u32, 0x4E809400), sdot(.x0, .x0, .x0));
    // udot is sdot with bit 29 (U) set.
    try std.testing.expectEqual(@as(u32, 0x6E809400), udot(.x0, .x0, .x0));
    // Distinct rd/rn/rm pins the field positions: rd bits[4:0], rn bits[9:5], rm bits[20:16].
    // sdot v1.4s, v2.4s, v3.4s -> 0x4E809400 | (3<<16) | (2<<5) | 1 = 0x4E839441
    try std.testing.expectEqual(@as(u32, 0x4E839441), sdot(.x1, .x2, .x3));
    // udot v1.4s, v2.4s, v3.4s -> the same fields, U set: 0x6E839441
    try std.testing.expectEqual(@as(u32, 0x6E839441), udot(.x1, .x2, .x3));
}

test "fused multiply-add/sub encodings (fmadd/fmsub/fnmsub)" {
    // Golden words obtained by assembling `fmadd/fmsub/fnmsub {s,d}0, {s,d}1, {s,d}2, {s,d}3`
    // with `as` and disassembling with `objdump` on this aarch64 host, then confirmed to
    // compute the right value by executing each variant (fmadd = Ra+Rn*Rm, fmsub = Ra-Rn*Rm,
    // fnmsub = Rn*Rm-Ra) - not just copied from the ARM ARM bit layout.
    try std.testing.expectEqual(@as(u32, 0x1f020c20), fmadd(.x0, .x1, .x2, .x3, false)); // fmadd s0,s1,s2,s3
    try std.testing.expectEqual(@as(u32, 0x1f420c20), fmadd(.x0, .x1, .x2, .x3, true)); // fmadd d0,d1,d2,d3
    try std.testing.expectEqual(@as(u32, 0x1f028c20), fmsub(.x0, .x1, .x2, .x3, false)); // fmsub s0,s1,s2,s3
    try std.testing.expectEqual(@as(u32, 0x1f428c20), fmsub(.x0, .x1, .x2, .x3, true)); // fmsub d0,d1,d2,d3
    try std.testing.expectEqual(@as(u32, 0x1f228c20), fnmsub(.x0, .x1, .x2, .x3, false)); // fnmsub s0,s1,s2,s3
    try std.testing.expectEqual(@as(u32, 0x1f628c20), fnmsub(.x0, .x1, .x2, .x3, true)); // fnmsub d0,d1,d2,d3
}

test "half-precision FCVT encodings (base-ISA widen/narrow for the f16 emulation)" {
    // Golden words obtained by assembling each `fcvt` with `as` and disassembling with
    // `objdump` on this aarch64 host, then confirmed to widen/narrow correctly by executing
    // the f16 differentials in tests/native.zig, not just copied from the ARM ARM bit layout.
    try std.testing.expectEqual(@as(u32, 0x1EE24000), fcvtSfromH(.x0, .x0)); // fcvt s0, h0
    try std.testing.expectEqual(@as(u32, 0x1EE240E5), fcvtSfromH(.x5, .x7)); // fcvt s5, h7
    try std.testing.expectEqual(@as(u32, 0x1E23C000), fcvtHfromS(.x0, .x0)); // fcvt h0, s0
    try std.testing.expectEqual(@as(u32, 0x1E23C0E5), fcvtHfromS(.x5, .x7)); // fcvt h5, s7
    try std.testing.expectEqual(@as(u32, 0x1E63C000), fcvtHfromD(.x0, .x0)); // fcvt h0, d0
    try std.testing.expectEqual(@as(u32, 0x1E63C0E5), fcvtHfromD(.x5, .x7)); // fcvt h5, d7
}

test "native half-precision (FEAT_FP16) H-form encodings" {
    // Golden words assembled with `as -march=armv8.2-a+fp16` and disassembled with `objdump`
    // on this aarch64 host (which has FEAT_FP16), not copied from the ARM ARM bit layout.
    try std.testing.expectEqual(@as(u32, 0x1EE22820), fadd(.x0, .x1, .x2, .half)); // fadd h0, h1, h2
    try std.testing.expectEqual(@as(u32, 0x1EE23820), fsub(.x0, .x1, .x2, .half)); // fsub h0, h1, h2
    try std.testing.expectEqual(@as(u32, 0x1EE20820), fmul(.x0, .x1, .x2, .half)); // fmul h0, h1, h2
    try std.testing.expectEqual(@as(u32, 0x1EE21820), fdiv(.x0, .x1, .x2, .half)); // fdiv h0, h1, h2
    try std.testing.expectEqual(@as(u32, 0x1EE22020), fcmp(.x1, .x2, .half)); // fcmp h1, h2
    try std.testing.expectEqual(@as(u32, 0x1EE21C20), fcsel(.x0, .x1, .x2, .ne, .half)); // fcsel h0, h1, h2, ne
    try std.testing.expectEqual(@as(u32, 0x1EE20020), cvtIntToFloat(.x0, .x1, .half, true)); // scvtf h0, w1
    try std.testing.expectEqual(@as(u32, 0x1EE30020), cvtIntToFloat(.x0, .x1, .half, false)); // ucvtf h0, w1
    try std.testing.expectEqual(@as(u32, 0x1EF80020), cvtFloatToInt(.x0, .x1, .half, true)); // fcvtzs w0, h1
    try std.testing.expectEqual(@as(u32, 0x1EF90020), cvtFloatToInt(.x0, .x1, .half, false)); // fcvtzu w0, h1
    try std.testing.expectEqual(@as(u32, 0x1EE70020), fmovHfromGpr(.x0, .x1)); // fmov h0, w1
    try std.testing.expectEqual(@as(u32, 0x1EE2C020), fcvtDfromH(.x0, .x1)); // fcvt d0, h1
    // The `.single`/`.double` selectors are byte-identical to the old `fpType(false)`/`(true)`,
    // so the base-ISA f32/f64 forms do not change (the guardrail for the emulation path).
    try std.testing.expectEqual(@as(u32, 0x1E222820), fadd(.x0, .x1, .x2, .single)); // fadd s0, s1, s2
    try std.testing.expectEqual(@as(u32, 0x1E622820), fadd(.x0, .x1, .x2, .double)); // fadd d0, d1, d2
}

test "half-precision FP load/store encodings (SIMD&FP h-form, not the GPR ldrh/strh)" {
    // Golden words from `as`/`objdump` on this aarch64 host.
    try std.testing.expectEqual(@as(u32, 0x7D400000), ldrHfp(.x0, .x0, 0)); // ldr h0, [x0]
    try std.testing.expectEqual(@as(u32, 0x7D400065), ldrHfp(.x5, .x3, 0)); // ldr h5, [x3]
    try std.testing.expectEqual(@as(u32, 0x7D000000), strHfp(.x0, .x0, 0)); // str h0, [x0]
    try std.testing.expectEqual(@as(u32, 0x7D000065), strHfp(.x5, .x3, 0)); // str h5, [x3]
    // The 12-bit immediate offset is scaled by 2 (a halfword): #8 -> field 4.
    try std.testing.expectEqual(@as(u32, 0x7D401041), ldrHfp(.x1, .x2, 8)); // ldr h1, [x2, #8]
    try std.testing.expectEqual(@as(u32, 0x7D002082), strHfp(.x2, .x4, 16)); // str h2, [x4, #16]
    // The h-form is distinct from the GPR halfword ops (different opcode, FP target).
    try std.testing.expect(ldrHfp(.x0, .x0, 0) != ldrh(.x0, .x0));
    try std.testing.expect(strHfp(.x0, .x0, 0) != strh(.x0, .x0));
}

test "64-bit ALU sets the sf bit (bit 31) over the 32-bit form" {
    // The 64-bit forms are exactly the 32-bit ones with sf (bit 31) set.
    try std.testing.expectEqual(add(.x0, .x1, .x2) | (@as(u32, 1) << 31), add64(.x0, .x1, .x2));
    try std.testing.expectEqual(@as(u32, 0x8B020020), add64(.x0, .x1, .x2)); // add x0, x1, x2
    try std.testing.expectEqual(@as(u32, 0xCB020020), sub64(.x0, .x1, .x2)); // sub x0, x1, x2
    try std.testing.expectEqual(@as(u32, 0x9AC22020), lslv64(.x0, .x1, .x2)); // lsl x0, x1, x2
}

test "shifted add/sub immediate (LSL #12) for large stack frames" {
    // The sh=1 form scales imm12 by 4096, so a frame wider than 12 bits opens in
    // two instructions. Rn/Rd 31 stays SP here (add/sub-immediate form).
    // sub sp, sp, #1, LSL #12  (== sub sp, sp, #4096)
    try std.testing.expectEqual(@as(u32, 0xD14007FF), subImm64Shift(.zr, .zr, 1));
    // add sp, sp, #1, LSL #12
    try std.testing.expectEqual(@as(u32, 0x914007FF), addImm64Shift(.zr, .zr, 1));
    // The shifted form is the unshifted opcode with sh (bit 22) set.
    try std.testing.expectEqual(subImm64(.x0, .x1, 5) | (@as(u32, 1) << 22), subImm64Shift(.x0, .x1, 5));
    try std.testing.expectEqual(addImm64(.x0, .x1, 5) | (@as(u32, 1) << 22), addImm64Shift(.x0, .x1, 5));
    // A 5000-byte frame splits as hi=1 (×4096) + lo=904, reconstructing exactly.
    const frame: usize = 5000;
    try std.testing.expectEqual(@as(usize, 4096 * (frame >> 12) + (frame & 0xFFF)), frame);
}
