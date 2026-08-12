//! x86-64 instruction encoding. Each encoder returns an `Inst` (a byte buffer and a length).
//! An x86 instruction is at most 15 bytes long. Tests validate the encoders in two ways:
//! execution under qemu-x86_64 (tests/qemu.zig), and the encoding tests here, which check
//! the output against known machine code.
//!
//! Encoding model for a 64-bit operand size: a REX prefix `0x48 | R | X | B` (W=1 selects
//! 64-bit, R and B extend the ModRM reg and rm fields to r8..r15), then the opcode, then a
//! register-direct ModRM byte `0xC0 | (reg<<3) | rm`.

const std = @import("std");

/// A 64-bit general register. The low 3 bits give the hardware encoding. r8..r15
/// have bit 3 set, so they need a REX prefix bit.
pub const Reg = enum(u4) {
    rax = 0,
    rcx = 1,
    rdx = 2,
    rbx = 3,
    rsp = 4,
    rbp = 5,
    rsi = 6,
    rdi = 7,
    r8 = 8,
    r9 = 9,
    r10 = 10,
    r11 = 11,
    r12 = 12,
    r13 = 13,
    r14 = 14,
    r15 = 15,
};

fn n(r: Reg) u8 {
    return @intFromEnum(r);
}

/// An SSE/AVX vector register (xmm0..xmm15). It uses the same 0..15 index space as `Reg`,
/// but it is a separate physical register file. The SSE opcodes below address it.
pub const Xmm = enum(u4) {
    xmm0 = 0,
    xmm1,
    xmm2,
    xmm3,
    xmm4,
    xmm5,
    xmm6,
    xmm7,
    xmm8,
    xmm9,
    xmm10,
    xmm11,
    xmm12,
    xmm13,
    xmm14,
    xmm15,
};

fn xn(x: Xmm) u8 {
    return @intFromEnum(x);
}

/// A two-xmm SSE op: `prefix` (F3/F2/66), `0F`, `op`, then a register-direct ModRM byte
/// (reg = dst, rm = src). This inserts a REX byte only when an operand is xmm8..15.
fn sseRR(prefix: u8, op: u8, dst: Xmm, src: Xmm) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ prefix, rex, 0x0F, op, mod });
    }
    return Inst.of(&.{ prefix, 0x0F, op, mod });
}

/// `movss dst, src` (xmm-to-xmm): copy the low 32-bit float lane (F3 0F 10 /r).
pub fn movssRR(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF3, 0x10, dst, src);
}
/// Scalar single-precision `addss`/`subss`/`mulss`/`divss dst, src` (F3 0F 58/5C/59/5E /r).
pub fn addss(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF3, 0x58, dst, src);
}
pub fn subss(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF3, 0x5C, dst, src);
}
pub fn mulss(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF3, 0x59, dst, src);
}
pub fn divss(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF3, 0x5E, dst, src);
}

/// An SSE op between an xmm `reg` field and a general-register `rm` field, for example movd.
fn sseXmmGpr(prefix: u8, op: u8, x: Xmm, g: Reg) Inst {
    const r = xn(x);
    const b = n(g);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ prefix, rex, 0x0F, op, mod });
    }
    return Inst.of(&.{ prefix, 0x0F, op, mod });
}
/// `movd dst, src`: copy 32 bits between a general register and an xmm register's low lane
/// (66 0F 6E to load into xmm, 66 0F 7E to read it out).
pub fn movdToXmm(dst: Xmm, src: Reg) Inst {
    return sseXmmGpr(0x66, 0x6E, dst, src);
}
pub fn movdFromXmm(dst: Reg, src: Xmm) Inst {
    return sseXmmGpr(0x66, 0x7E, src, dst);
}
/// `cvtsi2ss dst, src` (F3 0F 2A /r): convert a 32-bit signed integer in a general register
/// to a scalar single in `dst`.
pub fn cvtsi2ss(dst: Xmm, src: Reg) Inst {
    return sseXmmGpr(0xF3, 0x2A, dst, src);
}
/// `cvttss2si dst, src` (F3 0F 2C /r): convert a scalar single to a 32-bit signed integer in a
/// general register, with truncation. ModRM reg is the general dst, rm is the xmm src.
pub fn cvttss2si(dst: Reg, src: Xmm) Inst {
    const r = n(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ 0xF3, rex, 0x0F, 0x2C, mod });
    }
    return Inst.of(&.{ 0xF3, 0x0F, 0x2C, mod });
}

/// A two-xmm packed op with no mandatory prefix (`[REX] 0F op modrm`). This covers the
/// packed-single SSE forms addps/subps/mulps/divps and the packed move movups.
fn ssePacked(op: u8, dst: Xmm, src: Xmm) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ rex, 0x0F, op, mod });
    }
    return Inst.of(&.{ 0x0F, op, mod });
}
/// Packed single-precision `addps`/`subps`/`mulps`/`divps dst, src` (0F 58/5C/59/5E /r).
pub fn addps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x58, dst, src);
}
pub fn subps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x5C, dst, src);
}
pub fn mulps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x59, dst, src);
}
pub fn divps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x5E, dst, src);
}
/// `movups dst, src` (xmm-to-xmm): copy a whole 128-bit register (0F 10 /r).
pub fn movupsRR(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x10, dst, src);
}
/// Packed bitwise `andps`/`andnps`/`orps dst, src` (0F 54/55/56 /r). andnps computes
/// `dst = (NOT dst) AND src`. All three are SSE1, so every x86-64 CPU has them.
pub fn andps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x54, dst, src);
}
pub fn andnps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x55, dst, src);
}
pub fn orps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x56, dst, src);
}
/// Packed compare `cmpps dst, src, imm8` (0F C2 /r ib). For each lane, `dst = predicate(dst,
/// src) ? all-ones : all-zero`. imm selects the predicate: 0=EQ, 1=LT, 2=LE, 4=NEQ, 5=NLT,
/// 6=NLE. Predicates 0, 1, 2, and 4 are ordered compares.
pub fn cmpps(dst: Xmm, src: Xmm, imm: u8) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ rex, 0x0F, 0xC2, mod, imm });
    }
    return Inst.of(&.{ 0x0F, 0xC2, mod, imm });
}

// AVX (256-bit YMM) uses the VEX prefix. An AVX instruction is three-operand (dst = src1 op
// src2) and 256 bits wide (VEX.L = 1). The encoders here always use the 3-byte VEX (C4 ...)
// form, so any of ymm0..15 can appear in any operand. `vvvv` holds src1 in 1's-complement
// form. For an op with no src1, `vvvv` is unused: it is encoded as register 0, so its
// inverted field reads 1111.

/// A 3-byte VEX prefix, register-direct: ModRM.reg = `reg`, ModRM.rm = `rm`, VEX.vvvv =
/// `vvvv` (register numbers, `vvvv` = 0 means unused). `pp` selects the implied prefix (00 =
/// none, 01 = 66, 10 = F3, 11 = F2). `mmmmm` selects the opcode map (00001 = 0F, 00011 =
/// 0F3A).
fn vex3(reg: u8, rm: u8, vvvv: u8, opcode: u8, pp: u2, mmmmm: u5, imm: ?u8) Inst {
    const not_r: u8 = (~(reg >> 3)) & 1;
    const not_b: u8 = (~(rm >> 3)) & 1;
    const byte2: u8 = (not_r << 7) | (1 << 6) | (not_b << 5) | mmmmm; // X unused (no index) -> 1
    const not_vvvv: u8 = (~vvvv) & 0xF;
    const byte3: u8 = (not_vvvv << 3) | (1 << 2) | pp; // W=0, L=1 (256-bit)
    const mod: u8 = 0xC0 | ((reg & 7) << 3) | (rm & 7);
    if (imm) |i| return Inst.of(&.{ 0xC4, byte2, byte3, opcode, mod, i });
    return Inst.of(&.{ 0xC4, byte2, byte3, opcode, mod });
}

/// A 256-bit packed-single AVX op, `dst = src1 op src2`: VEX.NDS.256.0F.WIG `op` /r.
fn vexPacked256(op: u8, dst: Xmm, src1: Xmm, src2: Xmm) Inst {
    return vex3(xn(dst), xn(src2), xn(src1), op, 0b00, 0b00001, null);
}
pub fn vaddps(dst: Xmm, src1: Xmm, src2: Xmm) Inst {
    return vexPacked256(0x58, dst, src1, src2);
}
pub fn vsubps(dst: Xmm, src1: Xmm, src2: Xmm) Inst {
    return vexPacked256(0x5C, dst, src1, src2);
}
pub fn vmulps(dst: Xmm, src1: Xmm, src2: Xmm) Inst {
    return vexPacked256(0x59, dst, src1, src2);
}
pub fn vdivps(dst: Xmm, src1: Xmm, src2: Xmm) Inst {
    return vexPacked256(0x5E, dst, src1, src2);
}
/// `vmovups ymm, ymm` (VEX.256.0F.WIG 10 /r): copy a whole 256-bit register. vvvv is unused.
pub fn vmovupsRR(dst: Xmm, src: Xmm) Inst {
    return vex3(xn(dst), xn(src), 0, 0x10, 0b00, 0b00001, null);
}
/// `vinsertf128 ymm1, ymm2, xmm3, imm8` (VEX.256.66.0F3A.W0 18 /r ib): put `src2`'s 128 bits
/// into the half of `dst` chosen by `imm` (0 = low, 1 = high). The other half comes from `src1`.
pub fn vinsertf128(dst: Xmm, src1: Xmm, src2: Xmm, imm: u8) Inst {
    return vex3(xn(dst), xn(src2), xn(src1), 0x18, 0b01, 0b00011, imm);
}
/// `vextractf128 xmm1, ymm2, imm8` (VEX.256.66.0F3A.W0 19 /r ib): extract the 128-bit half
/// of `src` chosen by `imm` into `dst`. This is a store-form op, so ModRM puts src in reg
/// and dst in rm.
pub fn vextractf128(dst: Xmm, src: Xmm, imm: u8) Inst {
    return vex3(xn(src), xn(dst), 0, 0x19, 0b01, 0b00011, imm);
}

/// A 256-bit `vmovups` to or from `[rsp+disp]` (VEX.256.0F.WIG `op`, ModRM mod=10 reg=x
/// rm=SIB, SIB base=rsp, disp32): spill or reload a whole ymm register. vvvv and the SIB
/// index are unused.
fn vexStack(op: u8, x: Xmm, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    const r = xn(x);
    const not_r: u8 = (~(r >> 3)) & 1;
    const byte2: u8 = (not_r << 7) | (1 << 6) | (1 << 5) | 0b00001; // X,B unused -> 1, map 0F
    const byte3: u8 = (0xF << 3) | (1 << 2); // vvvv unused (1111), L=1 (256-bit), pp=00
    const mod: u8 = 0x84 | ((r & 7) << 3); // mod=10 (disp32), reg=x, rm=100 (SIB)
    return Inst.of(&.{ 0xC4, byte2, byte3, op, mod, 0x24, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}
pub fn vmovupsLoad(dst: Xmm, disp: i32) Inst {
    return vexStack(0x10, dst, disp);
}
pub fn vmovupsStore(disp: i32, src: Xmm) Inst {
    return vexStack(0x11, src, disp);
}
/// `ucomiss a, b` (0F 2E /r): unordered compare of two scalar singles. It sets ZF, PF, and CF
/// the same way an integer `cmp a, b` does. PF marks an unordered or NaN result.
pub fn ucomiss(a: Xmm, b: Xmm) Inst {
    return ssePacked(0x2E, a, b);
}

// F16C converts between half and single precision. Both ops are VEX-only: there is no legacy
// SSE encoding. Both operate on 128-bit registers, so they need the 3-byte VEX with L=0
// rather than the L=1 form the AVX helpers above use. `vex128` is that L=0 variant of `vex3`.

/// A 3-byte VEX prefix, register-direct, at 128-bit width (VEX.L=0, VEX.W=0). Otherwise
/// identical to `vex3`: ModRM.reg = `reg`, ModRM.rm = `rm`, VEX.vvvv = `vvvv` (0 means
/// unused, encoded as 1111). `pp` selects the implied prefix (01 = 66). `mmmmm` selects the
/// opcode map (00010 = 0F38, 00011 = 0F3A).
fn vex128(reg: u8, rm: u8, vvvv: u8, opcode: u8, pp: u2, mmmmm: u5, imm: ?u8) Inst {
    const not_r: u8 = (~(reg >> 3)) & 1;
    const not_b: u8 = (~(rm >> 3)) & 1;
    const byte2: u8 = (not_r << 7) | (1 << 6) | (not_b << 5) | mmmmm; // X unused (no index) -> 1
    const not_vvvv: u8 = (~vvvv) & 0xF;
    const byte3: u8 = (not_vvvv << 3) | pp; // W=0, L=0 (128-bit)
    const mod: u8 = 0xC0 | ((reg & 7) << 3) | (rm & 7);
    if (imm) |i| return Inst.of(&.{ 0xC4, byte2, byte3, opcode, mod, i });
    return Inst.of(&.{ 0xC4, byte2, byte3, opcode, mod });
}

/// `vcvtph2ps dst, src` (VEX.128.66.0F38.W0 13 /r): widen the four packed IEEE halves in the
/// low 64 bits of `src` to four packed f32 values in `dst`. Lane 0 holds the scalar half. The
/// other lanes widen with no ill effect. This has no NDS operand, so vvvv is unused (1111).
pub fn vcvtph2ps(dst: Xmm, src: Xmm) Inst {
    return vex128(xn(dst), xn(src), 0, 0x13, 0b01, 0b00010, null);
}

/// `vcvtps2ph dst, src, imm8` (VEX.128.66.0F3A.W0 1D /r ib): narrow the four packed f32
/// values in `src` to four packed IEEE halves in the low 64 bits of `dst`, and zero
/// dst[127:64]. This is a store-form op, so ModRM.reg is the source and ModRM.rm is the
/// destination. When imm8 bit 2 is 0, bits 1:0 select the rounding mode directly (0 = round
/// to nearest even). MXCSR is not used.
pub fn vcvtps2ph(dst: Xmm, src: Xmm, imm: u8) Inst {
    return vex128(xn(src), xn(dst), 0, 0x1D, 0b01, 0b00011, imm);
}

// Scalar double-precision (f64) SSE2 ops: the same encoders with an F2 prefix, plus the
// 64-bit conversions, ucomisd (66 prefix), and the f32/f64 conversions in both directions
// (xx 0F 5A).
/// Scalar double-precision `addsd`/`subsd`/`mulsd`/`divsd dst, src` (F2 0F 58/5C/59/5E /r).
pub fn addsd(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF2, 0x58, dst, src);
}
pub fn subsd(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF2, 0x5C, dst, src);
}
pub fn mulsd(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF2, 0x59, dst, src);
}
pub fn divsd(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF2, 0x5E, dst, src);
}
/// `ucomisd a, b` (66 0F 2E /r): the f64 unordered compare. It sets flags the same way
/// ucomiss does.
pub fn ucomisd(a: Xmm, b: Xmm) Inst {
    return sseRR(0x66, 0x2E, a, b);
}
/// `cvtss2sd dst, src` (F3 0F 5A) widens an f32 to an f64. `cvtsd2ss` (F2 0F 5A) narrows it back.
pub fn cvtss2sd(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF3, 0x5A, dst, src);
}
pub fn cvtsd2ss(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF2, 0x5A, dst, src);
}
/// Scalar single-precision `sqrtss dst, src` (F3 0F 51 /r).
pub fn sqrtss(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF3, 0x51, dst, src);
}
/// Scalar double-precision `sqrtsd dst, src` (F2 0F 51 /r).
pub fn sqrtsd(dst: Xmm, src: Xmm) Inst {
    return sseRR(0xF2, 0x51, dst, src);
}
/// Packed single-precision `sqrtps dst, src` (0F 51 /r): a per-lane square root. This is SSE1.
pub fn sqrtps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x51, dst, src);
}
/// `roundps dst, src, imm8` (66 0F 3A 08 /r ib): per-lane IEEE rounding for f32 (SSE4.1).
pub fn roundps(dst: Xmm, src: Xmm, imm: u8) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ 0x66, rex, 0x0F, 0x3A, 0x08, mod, imm });
    }
    return Inst.of(&.{ 0x66, 0x0F, 0x3A, 0x08, mod, imm });
}
/// `roundss dst, src, imm8` (66 0F 3A 0A /r ib): IEEE rounding for f32 (SSE4.1).
pub fn roundss(dst: Xmm, src: Xmm, imm: u8) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ 0x66, rex, 0x0F, 0x3A, 0x0A, mod, imm });
    }
    return Inst.of(&.{ 0x66, 0x0F, 0x3A, 0x0A, mod, imm });
}
/// `roundsd dst, src, imm8` (66 0F 3A 0B /r ib): IEEE rounding for f64 (SSE4.1).
pub fn roundsd(dst: Xmm, src: Xmm, imm: u8) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ 0x66, rex, 0x0F, 0x3A, 0x0B, mod, imm });
    }
    return Inst.of(&.{ 0x66, 0x0F, 0x3A, 0x0B, mod, imm });
}
/// `cvtsi2sd dst, src` (F2 0F 2A /r): convert a 32-bit signed int in a general register to
/// a scalar double.
pub fn cvtsi2sd(dst: Xmm, src: Reg) Inst {
    return sseXmmGpr(0xF2, 0x2A, dst, src);
}
/// `cvttsd2si dst, src` (F2 0F 2C /r): truncate a scalar double to a 32-bit signed int.
pub fn cvttsd2si(dst: Reg, src: Xmm) Inst {
    const r = n(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ 0xF2, rex, 0x0F, 0x2C, mod });
    }
    return Inst.of(&.{ 0xF2, 0x0F, 0x2C, mod });
}
/// `movq dst, src`: copy all 64 bits between a general register and an xmm register's low
/// lane (66 REX.W 0F 6E to load, 66 REX.W 0F 7E to read out). This materializes an f64
/// constant.
fn movqXmmGpr(op: u8, x: Xmm, g: Reg) Inst {
    const r = xn(x);
    const b = n(g);
    const rex: u8 = 0x48 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8); // REX.W
    return Inst.of(&.{ 0x66, rex, 0x0F, op, 0xC0 | ((r & 7) << 3) | (b & 7) });
}
pub fn movqToXmm(dst: Xmm, src: Reg) Inst {
    return movqXmmGpr(0x6E, dst, src);
}
pub fn movqFromXmm(dst: Reg, src: Xmm) Inst {
    return movqXmmGpr(0x7E, src, dst);
}
/// `mov dst, imm64` (REX.W B8+r io): load a full 64-bit immediate (the f64 bit pattern).
pub fn movImm64(dst: Reg, imm: u64) Inst {
    return Inst.of(&.{
        0x48 | @as(u8, @intFromBool(n(dst) >= 8)),
        0xB8 | (n(dst) & 7),
        @truncate(imm),
        @truncate(imm >> 8),
        @truncate(imm >> 16),
        @truncate(imm >> 24),
        @truncate(imm >> 32),
        @truncate(imm >> 40),
        @truncate(imm >> 48),
        @truncate(imm >> 56),
    });
}

/// `insertps dst, src, imm8` (66 0F 3A 21 /r ib): insert a lane of `src` into `dst`. With
/// imm = lane<<4, this copies src's lane 0 into dst's lane `lane` (src-lane and zero-mask
/// both 0).
pub fn insertps(dst: Xmm, src: Xmm, imm: u8) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ 0x66, rex, 0x0F, 0x3A, 0x21, mod, imm });
    }
    return Inst.of(&.{ 0x66, 0x0F, 0x3A, 0x21, mod, imm });
}
/// `pshufd dst, src, imm8` (66 0F 70 /r ib): shuffle 32-bit lanes. With imm = lane, this puts
/// src's lane `lane` into dst's lane 0. This extracts a lane to a scalar position.
pub fn pshufd(dst: Xmm, src: Xmm, imm: u8) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ 0x66, rex, 0x0F, 0x70, mod, imm });
    }
    return Inst.of(&.{ 0x66, 0x0F, 0x70, mod, imm });
}

/// An encoded instruction: up to 15 bytes.
pub const Inst = struct {
    bytes: [15]u8 = undefined,
    len: u8 = 0,

    fn of(b: []const u8) Inst {
        var i: Inst = .{ .len = @intCast(b.len) };
        @memcpy(i.bytes[0..b.len], b);
        return i;
    }
    pub fn slice(self: *const Inst) []const u8 {
        return self.bytes[0..self.len];
    }
};

/// The REX prefix for a 64-bit op with ModRM `reg` and `rm` register operands.
fn rexW(reg: Reg, rm: Reg) u8 {
    return 0x48 | (@as(u8, @intFromBool(n(reg) >= 8)) << 2) | @intFromBool(n(rm) >= 8);
}

/// The REX prefix for a two-register op at the requested operand size, or null when none is
/// needed. `w` = true selects a 64-bit operand size (REX.W). At a 32-bit operand size, a
/// REX.B or REX.R bit is still needed to name r8..r15, so this emits a REX with W=0 then.
/// When neither applies (32-bit, and both operands are rax..rdi), it emits no REX byte at
/// all. A bare 0x40 is legal but non-canonical, and omitting it keeps the 32-bit encodings
/// small.
fn rexRR(reg: Reg, rm: Reg, w: bool) ?u8 {
    const ext: u8 = (@as(u8, @intFromBool(n(reg) >= 8)) << 2) | @intFromBool(n(rm) >= 8);
    if (w) return 0x48 | ext;
    if (ext != 0) return 0x40 | ext;
    return null;
}

/// A register-direct ModRM byte (mod = 11).
fn modrm(reg: Reg, rm: Reg) u8 {
    return 0xC0 | ((n(reg) & 7) << 3) | (n(rm) & 7);
}

/// A two-register ALU op: `opcode` with `reg` as the source and `rm` as the destination. This
/// is the `/r` form, for example ADD/SUB/MOV r/m, r. `w` picks the operand size: true selects
/// 64-bit (REX.W), false selects 32-bit. A 32-bit op auto-zeroes the upper 32 bits of its
/// destination register, so an i32, i16, or i8 value stays clean, and every later
/// width-sensitive op reads the correct value.
fn aluRR(opcode: u8, src: Reg, dst: Reg, w: bool) Inst {
    if (rexRR(src, dst, w)) |rex| return Inst.of(&.{ rex, opcode, modrm(src, dst) });
    return Inst.of(&.{ opcode, modrm(src, dst) });
}

/// `mov dst, imm` loads an immediate. `w` = true selects `mov r64, imm32` (REX.W C7 /0), which
/// sign-extends the imm32 to 64 bits. `w` = false selects `mov r32, imm32` (B8+rd id), which
/// zero-extends into the full 64-bit register, so a result of 32 bits or fewer stays clean in
/// its upper half. This emits no REX for a 32-bit mov to rax..rdi.
pub fn movImm(dst: Reg, imm: i32, w: bool) Inst {
    const u: u32 = @bitCast(imm);
    if (w) return Inst.of(&.{
        0x48 | @as(u8, @intFromBool(n(dst) >= 8)),
        0xC7,
        0xC0 | (n(dst) & 7), // /0
        @truncate(u),
        @truncate(u >> 8),
        @truncate(u >> 16),
        @truncate(u >> 24),
    });
    // A 32-bit `mov r32, imm32` zero-extends. r8..r15 need REX.B (W=0). rax..rdi need no REX.
    if (n(dst) >= 8) return Inst.of(&.{
        0x41,
        0xB8 | (n(dst) & 7),
        @truncate(u),
        @truncate(u >> 8),
        @truncate(u >> 16),
        @truncate(u >> 24),
    });
    return Inst.of(&.{
        0xB8 | (n(dst) & 7),
        @truncate(u),
        @truncate(u >> 8),
        @truncate(u >> 16),
        @truncate(u >> 24),
    });
}

/// `mov al, imm8` (B0 ib): set AL to `imm`. The System V variadic-call convention requires AL
/// to hold the number of vector (xmm) registers used to pass arguments, so a variadic callee
/// such as `printf` knows how many xmm registers to spill into its register save area. This
/// is a 2-byte form that touches only AL, the low byte of RAX. RAX is caller-saved and holds
/// no argument here.
pub fn movAlImm(imm: u8) Inst {
    return Inst.of(&.{ 0xB0, imm });
}

/// `mov dst, src` (89 /r: store the reg operand into the r/m operand). This is always 64-bit.
/// A plain register copy does not depend on width: a clean i32 stays clean, and the next
/// width-sensitive op that reads a dirty upper half cleans it. So this always uses REX.W, to
/// keep every move byte-identical.
pub fn movReg(dst: Reg, src: Reg) Inst {
    return aluRR(0x89, src, dst, true);
}

/// `add dst, src` (01 /r).
pub fn add(dst: Reg, src: Reg, w: bool) Inst {
    return aluRR(0x01, src, dst, w);
}

/// `sub dst, src` (29 /r).
pub fn sub(dst: Reg, src: Reg, w: bool) Inst {
    return aluRR(0x29, src, dst, w);
}

/// `and dst, src` (21 /r).
pub fn andr(dst: Reg, src: Reg, w: bool) Inst {
    return aluRR(0x21, src, dst, w);
}

/// `or dst, src` (09 /r).
pub fn orr(dst: Reg, src: Reg, w: bool) Inst {
    return aluRR(0x09, src, dst, w);
}

/// `xor dst, src` (31 /r).
pub fn xorr(dst: Reg, src: Reg, w: bool) Inst {
    return aluRR(0x31, src, dst, w);
}

/// `imul dst, src` (0F AF /r: dst = dst * src). A two-operand signed multiply. `w` picks the
/// operand size. A 32-bit imul zero-extends its 32-bit product into the destination.
pub fn imul(dst: Reg, src: Reg, w: bool) Inst {
    if (rexRR(dst, src, w)) |rex| return Inst.of(&.{ rex, 0x0F, 0xAF, modrm(dst, src) });
    return Inst.of(&.{ 0x0F, 0xAF, modrm(dst, src) });
}

/// The optional REX prefix for a single-register op at operand size `w`, given whether that
/// register is r8..r15 (`ext`). `w` = true selects 64-bit (REX.W). `w` = false selects 32-bit:
/// REX.B only if `ext` is true, otherwise no REX at all, so the encoding stays small.
fn rexOne(ext: bool, w: bool) ?u8 {
    if (w) return 0x48 | @as(u8, @intFromBool(ext));
    if (ext) return 0x41; // REX.B, W=0
    return null;
}

/// An ALU op against a 32-bit immediate (81 /digit id): ADD=/0, OR=/1, AND=/4, SUB=/5,
/// XOR=/6. `w` picks the operand size: REX.W for 64-bit, no REX or REX.B for 32-bit.
pub fn aluImm(digit: u3, dst: Reg, imm: i32, w: bool) Inst {
    const u: u32 = @bitCast(imm);
    const mrm: u8 = 0xC0 | (@as(u8, digit) << 3) | (n(dst) & 7);
    if (rexOne(n(dst) >= 8, w)) |rex| return Inst.of(&.{ rex, 0x81, mrm, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
    return Inst.of(&.{ 0x81, mrm, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// `imul dst, src, imm32` (69 /r id): dst = src * imm. This is the three-operand form. `w`
/// picks the operand size.
pub fn imulImm(dst: Reg, src: Reg, imm: i32, w: bool) Inst {
    const u: u32 = @bitCast(imm);
    const mrm = modrm(dst, src);
    if (rexRR(dst, src, w)) |rex| return Inst.of(&.{ rex, 0x69, mrm, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
    return Inst.of(&.{ 0x69, mrm, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// A shift of `dst` by an immediate count (C1 /digit ib): SHL=/4, SHR=/5, SAR=/7. `w` picks
/// the operand size. A 32-bit shr/sar shifts within 32 bits, filling from bit 31. An i32
/// needs this, since a 64-bit shift would cross the bit 31/32 boundary.
pub fn shiftImm(digit: u3, dst: Reg, imm8: u8, w: bool) Inst {
    const mrm: u8 = 0xC0 | (@as(u8, digit) << 3) | (n(dst) & 7);
    if (rexOne(n(dst) >= 8, w)) |rex| return Inst.of(&.{ rex, 0xC1, mrm, imm8 });
    return Inst.of(&.{ 0xC1, mrm, imm8 });
}

/// `cqo` (REX.W 99): sign-extend RAX into RDX:RAX, the 64-bit dividend for `idiv`.
pub fn cqo() Inst {
    return Inst.of(&.{ 0x48, 0x99 });
}

/// `cdq` (99, no REX.W): sign-extend EAX into EDX:EAX, the 32-bit dividend for a 32-bit `idiv`.
pub fn cdq() Inst {
    return Inst.of(&.{0x99});
}

/// `idiv src` (F7 /7): a signed divide of RDX:RAX or EDX:EAX by `src`. The quotient goes to
/// (R/E)AX, the remainder to (R/E)DX. `src` must not be RAX or RDX. `w` picks the operand size.
pub fn idiv(src: Reg, w: bool) Inst {
    const mrm: u8 = 0xC0 | (7 << 3) | (n(src) & 7);
    if (rexOne(n(src) >= 8, w)) |rex| return Inst.of(&.{ rex, 0xF7, mrm });
    return Inst.of(&.{ 0xF7, mrm });
}

/// `div src` (F7 /6): an unsigned divide of RDX:RAX or EDX:EAX by `src`. Clear the high half
/// first. `w` picks the operand size. A 32-bit div reads only EAX, the low 32 bits, as the
/// dividend, so it correctly ignores a dirty upper half of RAX.
pub fn divu(src: Reg, w: bool) Inst {
    const mrm: u8 = 0xC0 | (6 << 3) | (n(src) & 7);
    if (rexOne(n(src) >= 8, w)) |rex| return Inst.of(&.{ rex, 0xF7, mrm });
    return Inst.of(&.{ 0xF7, mrm });
}

/// A shift of `dst` by the count held in CL (D3 /digit). `shl` = /4. `shr` = /5, a logical
/// shift. `sar` = /7, an arithmetic shift. `w` picks the operand size.
fn shiftCl(digit: u8, dst: Reg, w: bool) Inst {
    const mrm: u8 = 0xC0 | (digit << 3) | (n(dst) & 7);
    if (rexOne(n(dst) >= 8, w)) |rex| return Inst.of(&.{ rex, 0xD3, mrm });
    return Inst.of(&.{ 0xD3, mrm });
}
pub fn shlCl(dst: Reg, w: bool) Inst {
    return shiftCl(4, dst, w);
}
pub fn shrCl(dst: Reg, w: bool) Inst {
    return shiftCl(5, dst, w);
}
pub fn sarCl(dst: Reg, w: bool) Inst {
    return shiftCl(7, dst, w);
}

/// `cmp a, b` (39 /r): compute `a - b` and set flags, without changing the operands. `w`
/// picks the operand size, so an i32 compare sets flags from the low 32 bits only.
pub fn cmp(a: Reg, b: Reg, w: bool) Inst {
    return aluRR(0x39, b, a, w);
}

/// `test a, b` (85 /r): compute `a & b` and set flags. `test r, r` tests whether `r` is zero.
/// `w` picks the operand size.
pub fn testReg(a: Reg, b: Reg, w: bool) Inst {
    return aluRR(0x85, b, a, w);
}

/// `test al, al` (84 C0): set flags from the low byte of RAX only. The System V variadic
/// prologue decides whether to save xmm registers based on AL, the count of vector registers
/// the caller used. So it must test exactly AL, not EAX. The `mov al, imm8` caller convention
/// leaves EAX's upper bytes dirty.
pub fn testAlAl() Inst {
    return Inst.of(&.{ 0x84, 0xC0 });
}

/// A condition code: the low nibble of the Jcc/SETcc opcode.
pub const Cond = enum(u8) {
    e = 0x4, // equal / zero
    ne = 0x5, // not equal
    l = 0xC, // signed <
    ge = 0xD, // signed >=
    le = 0xE, // signed <=
    g = 0xF, // signed >
    b = 0x2, // unsigned <
    ae = 0x3, // unsigned >=
    be = 0x6, // unsigned <=
    a = 0x7, // unsigned >
    p = 0xA, // parity: an unordered or NaN float compare
    np = 0xB, // not parity: an ordered float compare
};

/// The logical inverse of a condition. In the x86 condition-code encoding, bit 0 toggles the
/// sense, so the inverse is the paired code: e/ne, l/ge, le/g, b/ae, be/a, p/np. This lets an
/// `if` fall through the else edge, by branching to the else label on the inverted predicate
/// (jnz becomes jz). The switch covers every `Cond`, so a newly added condition forces a
/// matching inverse here.
pub fn invertCond(cond: Cond) Cond {
    return switch (cond) {
        .e => .ne,
        .ne => .e,
        .l => .ge,
        .ge => .l,
        .le => .g,
        .g => .le,
        .b => .ae,
        .ae => .b,
        .be => .a,
        .a => .be,
        .p => .np,
        .np => .p,
    };
}

/// `cmovcc dst, src` (0F 40+cc /r): move `src` into `dst` if the condition holds.
pub fn cmovcc(dst: Reg, src: Reg, cond: Cond) Inst {
    return Inst.of(&.{ rexW(dst, src), 0x0F, 0x40 | @intFromEnum(cond), modrm(dst, src) });
}

/// `setcc dst8` (0F 90+cc): set the low byte of `dst` to 0 or 1 from the flags. This always
/// emits a REX prefix, so the encoding addresses the low byte (spl/sil/... for regs 4..7,
/// r8b.. for 8..15), and never the legacy ah/ch/dh/bh.
pub fn setcc(dst: Reg, cond: Cond) Inst {
    return Inst.of(&.{ 0x40 | @as(u8, @intFromBool(n(dst) >= 8)), 0x0F, 0x90 | @intFromEnum(cond), 0xC0 | (n(dst) & 7) });
}

/// `movzx dst, dst8` (REX.W 0F B6 /r): zero-extend `src`'s low byte into `dst`.
pub fn movzxByte(dst: Reg, src: Reg) Inst {
    return Inst.of(&.{ rexW(dst, src), 0x0F, 0xB6, modrm(dst, src) });
}

/// `movsx dst, src8` (REX.W 0F BE /r): sign-extend `src`'s low byte into the full 64-bit `dst`.
pub fn movsxByte(dst: Reg, src: Reg) Inst {
    return Inst.of(&.{ rexW(dst, src), 0x0F, 0xBE, modrm(dst, src) });
}

/// `movzx dst, src16` (REX.W 0F B7 /r): zero-extend `src`'s low word into the full 64-bit `dst`.
pub fn movzxWord(dst: Reg, src: Reg) Inst {
    return Inst.of(&.{ rexW(dst, src), 0x0F, 0xB7, modrm(dst, src) });
}

/// `movsx dst, src16` (REX.W 0F BF /r): sign-extend `src`'s low word into the full 64-bit `dst`.
pub fn movsxWord(dst: Reg, src: Reg) Inst {
    return Inst.of(&.{ rexW(dst, src), 0x0F, 0xBF, modrm(dst, src) });
}

/// `movsxd dst, src32` (REX.W 63 /r): sign-extend `src`'s low 32 bits into the full 64-bit
/// `dst`. This is the register-to-register counterpart of `movsxdFromMem`. It is needed
/// because a 32-bit ALU result auto-zero-extends into its 64-bit register (an architectural
/// rule), which is wrong for a signed widening convert. A negative i32 must sign-extend, not
/// zero-extend, into an i64.
pub fn movsxdReg(dst: Reg, src: Reg) Inst {
    return Inst.of(&.{ rexW(dst, src), 0x63, modrm(dst, src) });
}

/// `mov dst32, src32` (89 /r, no REX.W): a 32-bit register copy. Unlike `movReg`, which
/// always uses REX.W, this only touches the low 32 bits. The architectural 32-bit-write side
/// effect zero-extends the upper 32 bits of `dst`, regardless of what `src`'s upper half
/// held. That is the correct unsigned 32-to-64 widening convert, independent of whether
/// `src` was already clean.
pub fn movReg32(dst: Reg, src: Reg) Inst {
    return aluRR(0x89, src, dst, false);
}

/// `jcc rel32` (0F 80+cc cd): branch when the condition holds. `rel` is measured from the
/// end of the instruction.
pub fn jcc(cond: Cond, rel: i32) Inst {
    const u: u32 = @bitCast(rel);
    return Inst.of(&.{ 0x0F, 0x80 | @intFromEnum(cond), @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// `jmp rel32` (E9 cd): unconditional relative jump.
pub fn jmp(rel: i32) Inst {
    const u: u32 = @bitCast(rel);
    return Inst.of(&.{ 0xE9, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// `mov dst, [rsp + disp32]` (REX.W 8B /r): load a stack slot. The RSP base needs a
/// SIB byte, 0x24.
pub fn movFromStack(dst: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    return Inst.of(&.{ rexW(dst, .rax), 0x8B, 0x84 | ((n(dst) & 7) << 3), 0x24, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// `mov [rsp + disp32], src` (REX.W 89 /r): store to a stack slot.
pub fn movToStack(disp: i32, src: Reg) Inst {
    const u: u32 = @bitCast(disp);
    return Inst.of(&.{ rexW(src, .rax), 0x89, 0x84 | ((n(src) & 7) << 3), 0x24, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// An SSE load or store to `[rsp + disp32]`. The RSP base needs the 0x24 SIB byte. `prefix`
/// null selects packed (movups), 0xF3 selects scalar single (movss). REX.R is added for
/// xmm8..15.
fn sseStack(prefix: ?u8, op: u8, x: Xmm, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    const r = xn(x);
    var buf: [11]u8 = undefined;
    var i: usize = 0;
    if (prefix) |p| {
        buf[i] = p;
        i += 1;
    }
    if (r >= 8) {
        buf[i] = 0x44; // REX.R
        i += 1;
    }
    buf[i] = 0x0F;
    buf[i + 1] = op;
    buf[i + 2] = 0x84 | ((r & 7) << 3); // mod=10 (disp32), reg=x, rm=100 (SIB)
    buf[i + 3] = 0x24; // SIB: base=rsp, no index
    buf[i + 4] = @truncate(u);
    buf[i + 5] = @truncate(u >> 8);
    buf[i + 6] = @truncate(u >> 16);
    buf[i + 7] = @truncate(u >> 24);
    return Inst.of(buf[0 .. i + 8]);
}
/// `movss xmm, [rsp+disp]` / `movss [rsp+disp], xmm` (F3 0F 10 / 11): reload/spill a scalar.
pub fn movssLoad(dst: Xmm, disp: i32) Inst {
    return sseStack(0xF3, 0x10, dst, disp);
}
pub fn movssStore(disp: i32, src: Xmm) Inst {
    return sseStack(0xF3, 0x11, src, disp);
}
/// `movsd [rsp+disp], xmm` (F2 0F 11): spill a scalar double (exactly 8 bytes, unlike the
/// 16-byte `movups`). This stores a `double` struct-return eightbyte without overwriting the
/// eight bytes past it.
pub fn movsdStore(disp: i32, src: Xmm) Inst {
    return sseStack(0xF2, 0x11, src, disp);
}
/// `movups xmm, [rsp+disp]` / `movups [rsp+disp], xmm` (0F 10 / 11): reload/spill a vector.
pub fn movupsLoad(dst: Xmm, disp: i32) Inst {
    return sseStack(null, 0x10, dst, disp);
}
pub fn movupsStore(disp: i32, src: Xmm) Inst {
    return sseStack(null, 0x11, src, disp);
}
/// `movaps [rsp+disp], xmm` (0F 29 /r): store an aligned 16-byte SSE register. The System V
/// variadic prologue uses this for the register-save-area xmm block. The save area sits at a
/// 16-byte-aligned frame offset, so the aligned form is safe. REX.R is added for xmm8..15.
pub fn movapsStore(disp: i32, src: Xmm) Inst {
    return sseStack(null, 0x29, src, disp);
}

/// A general `mov` to or from `[base + disp32]` with an arbitrary base register (mod=10,
/// disp32). The data register is the ModRM `reg` field, the base is the `rm` field. A base
/// of rsp(4) or r12 needs the 0x24 SIB byte. rm=rbp(5) is fine at mod=10, since disp32 is
/// explicit. `op` is 0x8B for a load (mov r, [base+disp]) or 0x89 for a store (mov
/// [base+disp], r).
fn movMem(op: u8, data: Reg, base: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    const r = n(data);
    const b = n(base);
    const rex: u8 = 0x48 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
    const modrm_byte: u8 = 0x80 | ((r & 7) << 3) | (b & 7); // mod=10, reg=data, rm=base
    if ((b & 7) == 4) { // rsp/r12 base needs a SIB byte (base=rm, no index)
        return Inst.of(&.{ rex, op, modrm_byte, 0x24, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
    }
    return Inst.of(&.{ rex, op, modrm_byte, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}
/// `mov dst, [base + disp32]` (REX.W 8B /r): load 64 bits from `[base+disp]`.
pub fn movFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    return movMem(0x8B, dst, base, disp);
}
/// `mov [base + disp32], src` (REX.W 89 /r): store 64 bits to `[base+disp]`.
pub fn movToMem(base: Reg, disp: i32, src: Reg) Inst {
    return movMem(0x89, src, base, disp);
}
/// A memory mov without REX.W, a 32-bit operand: reads or writes 4 bytes. A 32-bit load
/// zero-extends into the full 64-bit register.
fn movMem32(op: u8, data: Reg, base: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    const r = n(data);
    const b = n(base);
    const modrm_byte: u8 = 0x80 | ((r & 7) << 3) | (b & 7); // mod=10, reg=data, rm=base
    const ext = r >= 8 or b >= 8;
    const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
    const sib = (b & 7) == 4; // rsp/r12 base needs a SIB byte
    var buf: [8]u8 = undefined;
    var i: usize = 0;
    if (ext) {
        buf[i] = rex;
        i += 1;
    }
    buf[i] = op;
    i += 1;
    buf[i] = modrm_byte;
    i += 1;
    if (sib) {
        buf[i] = 0x24;
        i += 1;
    }
    buf[i] = @truncate(u);
    buf[i + 1] = @truncate(u >> 8);
    buf[i + 2] = @truncate(u >> 16);
    buf[i + 3] = @truncate(u >> 24);
    return Inst.of(buf[0 .. i + 4]);
}
/// 32-bit integer load `mov r32, [base+disp]` (zero-extends to the 64-bit register).
pub fn movFromMem32(dst: Reg, base: Reg, disp: i32) Inst {
    return movMem32(0x8B, dst, base, disp);
}
/// 32-bit integer store `mov [base+disp], r32` (writes exactly 4 bytes).
pub fn movToMem32(base: Reg, disp: i32, src: Reg) Inst {
    return movMem32(0x89, src, base, disp);
}
/// Sign-extending 32-bit load `movsxd r64, [base+disp]` (63 /r, REX.W): a signed i32 into i64.
pub fn movsxdFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    return movMem(0x63, dst, base, disp);
}

/// `add dst, qword [base+disp32]` (REX.W 03 /r): add the 64-bit value at `[base+disp]` into
/// `dst`. The `va_arg` expansion uses this to compute `p = reg_save_area + offset` in one op.
/// The offset, already in `dst`, and the save-area base, in memory, need no third register.
pub fn addRegMem(dst: Reg, base: Reg, disp: i32) Inst {
    return movMem(0x03, dst, base, disp);
}

/// An ALU op with an 8-bit sign-extended immediate against a memory operand `[base+disp32]`
/// (83 /digit ib): ADD=/0, SUB=/5, and so on. `w` picks the operand size, REX.W for a 64-bit
/// memory location, else 32-bit. The `va_arg` expansion uses this to bump a `va_list` offset
/// field in place (`add dword [list], 8`, `add qword [list+8], 8`) with no register at all. A
/// base of rsp/r12 needs the 0x24 SIB byte, and r8..15 needs REX.B.
pub fn aluMemImm8(digit: u3, base: Reg, disp: i32, imm8: i8, w: bool) Inst {
    const u: u32 = @bitCast(disp);
    const b = n(base);
    const modrm_byte: u8 = 0x80 | (@as(u8, digit) << 3) | (b & 7); // mod=10, reg=digit, rm=base
    const need_rex = w or b >= 8;
    const rex: u8 = 0x40 | (@as(u8, @intFromBool(w)) << 3) | @intFromBool(b >= 8);
    const sib = (b & 7) == 4; // rsp/r12 base needs a SIB byte
    var buf: [9]u8 = undefined;
    var i: usize = 0;
    if (need_rex) {
        buf[i] = rex;
        i += 1;
    }
    buf[i] = 0x83;
    i += 1;
    buf[i] = modrm_byte;
    i += 1;
    if (sib) {
        buf[i] = 0x24;
        i += 1;
    }
    buf[i] = @truncate(u);
    buf[i + 1] = @truncate(u >> 8);
    buf[i + 2] = @truncate(u >> 16);
    buf[i + 3] = @truncate(u >> 24);
    i += 4;
    buf[i] = @bitCast(imm8);
    i += 1;
    return Inst.of(buf[0..i]);
}

/// `movzx dst, word ptr [base+disp32]` (0F B7 /r): load 16 bits, zero-extended into the
/// 32-bit register (and so into the full 64-bit register). This brings an IEEE half out of
/// memory before it is widened to f32. A base of rsp/r12 (rm low bits = 100) needs the 0x24
/// SIB byte. An r8..15 base or dst needs a REX bit.
pub fn movzxWordFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    const r = n(dst);
    const b = n(base);
    const modrm_byte: u8 = 0x80 | ((r & 7) << 3) | (b & 7); // mod=10 (disp32), reg=dst, rm=base
    const ext = r >= 8 or b >= 8;
    const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
    const sib = (b & 7) == 4;
    var buf: [9]u8 = undefined;
    var i: usize = 0;
    if (ext) {
        buf[i] = rex;
        i += 1;
    }
    buf[i] = 0x0F;
    buf[i + 1] = 0xB7;
    buf[i + 2] = modrm_byte;
    i += 3;
    if (sib) {
        buf[i] = 0x24;
        i += 1;
    }
    buf[i] = @truncate(u);
    buf[i + 1] = @truncate(u >> 8);
    buf[i + 2] = @truncate(u >> 16);
    buf[i + 3] = @truncate(u >> 24);
    return Inst.of(buf[0 .. i + 4]);
}

/// `mov word ptr [base+disp32], src` (66 89 /r): store the low 16 bits of `src`. The 66
/// operand-size prefix, before any REX, makes this a 16-bit store, so it writes exactly 2
/// bytes, an IEEE half. A base of rsp/r12 needs the 0x24 SIB byte. r8..15 needs a REX bit.
pub fn movToMem16(base: Reg, disp: i32, src: Reg) Inst {
    const u: u32 = @bitCast(disp);
    const r = n(src);
    const b = n(base);
    const modrm_byte: u8 = 0x80 | ((r & 7) << 3) | (b & 7); // mod=10 (disp32), reg=src, rm=base
    const ext = r >= 8 or b >= 8;
    const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
    const sib = (b & 7) == 4;
    var buf: [10]u8 = undefined;
    var i: usize = 0;
    buf[i] = 0x66; // 16-bit operand size, must precede REX
    i += 1;
    if (ext) {
        buf[i] = rex;
        i += 1;
    }
    buf[i] = 0x89;
    buf[i + 1] = modrm_byte;
    i += 2;
    if (sib) {
        buf[i] = 0x24;
        i += 1;
    }
    buf[i] = @truncate(u);
    buf[i + 1] = @truncate(u >> 8);
    buf[i + 2] = @truncate(u >> 16);
    buf[i + 3] = @truncate(u >> 24);
    return Inst.of(buf[0 .. i + 4]);
}

/// `mov byte ptr [base+disp32], src` (88 /r): store the low 8 bits of `src`, exactly 1 byte.
/// This emits a REX prefix whenever the source names one of spl/bpl/sil/dil (register index
/// 4..7), so the encoding addresses the low byte rather than the legacy ah/ch/dh/bh, and also
/// for any r8..15 source or base. A base of rsp/r12 needs the 0x24 SIB byte.
pub fn movToMem8(base: Reg, disp: i32, src: Reg) Inst {
    const u: u32 = @bitCast(disp);
    const r = n(src);
    const b = n(base);
    const modrm_byte: u8 = 0x80 | ((r & 7) << 3) | (b & 7); // mod=10 (disp32), reg=src, rm=base
    // A source register index of 4 or more needs a REX prefix to select spl/bpl/sil/dil/
    // r8b..r15b as an 8-bit operand. Without REX, indices 4..7 mean ah/ch/dh/bh.
    const need_rex = r >= 4 or b >= 8;
    const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
    const sib = (b & 7) == 4;
    var buf: [8]u8 = undefined;
    var i: usize = 0;
    if (need_rex) {
        buf[i] = rex;
        i += 1;
    }
    buf[i] = 0x88;
    buf[i + 1] = modrm_byte;
    i += 2;
    if (sib) {
        buf[i] = 0x24;
        i += 1;
    }
    buf[i] = @truncate(u);
    buf[i + 1] = @truncate(u >> 8);
    buf[i + 2] = @truncate(u >> 16);
    buf[i + 3] = @truncate(u >> 24);
    return Inst.of(buf[0 .. i + 4]);
}

/// A sign- or zero-extending narrow load, `movsx`/`movzx dst32, byte/word ptr [base+disp32]`,
/// using a two-byte opcode `0F op`. The destination is a 32-bit register (no REX.W), so the
/// load extends into 32 bits, and the 32-bit write auto-zeroes the upper 32 bits, keeping an
/// i8 or i16 result clean. A base of rsp/r12 needs the 0x24 SIB byte. An r8..15 dst or base
/// needs a REX bit.
fn extMem(op: u8, dst: Reg, base: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    const r = n(dst);
    const b = n(base);
    const modrm_byte: u8 = 0x80 | ((r & 7) << 3) | (b & 7); // mod=10 (disp32), reg=dst, rm=base
    const ext = r >= 8 or b >= 8;
    const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
    const sib = (b & 7) == 4;
    var buf: [9]u8 = undefined;
    var i: usize = 0;
    if (ext) {
        buf[i] = rex;
        i += 1;
    }
    buf[i] = 0x0F;
    buf[i + 1] = op;
    buf[i + 2] = modrm_byte;
    i += 3;
    if (sib) {
        buf[i] = 0x24;
        i += 1;
    }
    buf[i] = @truncate(u);
    buf[i + 1] = @truncate(u >> 8);
    buf[i + 2] = @truncate(u >> 16);
    buf[i + 3] = @truncate(u >> 24);
    return Inst.of(buf[0 .. i + 4]);
}
/// `movsx r32, byte ptr [base+disp]` (0F BE): load 1 byte, sign-extended (a signed i8 load).
pub fn movsxByteFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    return extMem(0xBE, dst, base, disp);
}
/// `movzx r32, byte ptr [base+disp]` (0F B6): load 1 byte, zero-extended (an unsigned i8 load).
pub fn movzxByteFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    return extMem(0xB6, dst, base, disp);
}
/// `movsx r32, word ptr [base+disp]` (0F BF): load 2 bytes, sign-extended (a signed i16 load).
pub fn movsxWordFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    return extMem(0xBF, dst, base, disp);
}

/// `lea dst, [rsp + disp32]` (REX.W 8D /r): materialize a stack address, for example an
/// alloca slot. The rsp base needs the 0x24 SIB byte, mod=10 for the disp32 form.
pub fn leaFromStack(dst: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    return Inst.of(&.{ rexW(dst, .rax), 0x8D, 0x84 | ((n(dst) & 7) << 3), 0x24, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// `lea dst, [rip + disp32]` (REX.W 8D /r): materialize a PC-relative symbol address, a
/// `global_addr`. Mod=00, rm=101 is the RIP-relative special case (no SIB, no base
/// register), so the `rm` side of the REX contributes nothing. The trailing `.rax`
/// argument to `rexW` mirrors `leaFromStack`'s trick: it picks a low (REX.B=0) register
/// for a field that is not a real general-register operand.
pub fn leaRipRel(dst: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    const modrm_byte: u8 = 0x05 | ((n(dst) & 7) << 3);
    return Inst.of(&.{ rexW(dst, .rax), 0x8D, modrm_byte, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// `mov dst, qword [rip + disp32]` (REX.W 8B /r): load a 64-bit value from a PC-relative
/// address into `dst`, a GOT-indirect `global_addr` (`via_got`). Byte-identical to
/// `leaRipRel`, except the opcode is `0x8B` (mov r64, r/m64), not `0x8D` (lea). `lea`
/// computes the address `rip+disp32`. This instead reads the 8 bytes there, the symbol's
/// address that the loader wrote into the GOT slot. It uses the same mod=00, rm=101
/// RIP-relative ModRM form (no SIB, no base register), so the `rm` side of the REX
/// contributes nothing, and the trailing `.rax` argument to `rexW` picks a low (REX.B=0)
/// register for a field that is not a real general-register operand.
pub fn movRipRel(dst: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    const modrm_byte: u8 = 0x05 | ((n(dst) & 7) << 3);
    return Inst.of(&.{ rexW(dst, .rax), 0x8B, modrm_byte, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// An SSE load or store to `[base + disp32]` with an arbitrary base register. `prefix` null
/// selects packed (movups), 0xF3 selects scalar single (movss). REX is added for an xmm8..15
/// data register or an r8..15 base. A base of rsp/r12 (rm low bits = 100) needs the 0x24 SIB
/// byte.
fn sseMem(prefix: ?u8, op: u8, x: Xmm, base: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    const r = xn(x);
    const b = n(base);
    var buf: [12]u8 = undefined;
    var i: usize = 0;
    if (prefix) |p| {
        buf[i] = p;
        i += 1;
    }
    if (r >= 8 or b >= 8) {
        buf[i] = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8); // REX.R / REX.B
        i += 1;
    }
    buf[i] = 0x0F;
    buf[i + 1] = op;
    buf[i + 2] = 0x80 | ((r & 7) << 3) | (b & 7); // mod=10 (disp32), reg=x, rm=base
    i += 3;
    if ((b & 7) == 4) { // rsp/r12 base needs a SIB byte
        buf[i] = 0x24;
        i += 1;
    }
    buf[i] = @truncate(u);
    buf[i + 1] = @truncate(u >> 8);
    buf[i + 2] = @truncate(u >> 16);
    buf[i + 3] = @truncate(u >> 24);
    return Inst.of(buf[0 .. i + 4]);
}
/// `movss xmm, [base+disp]` / `movss [base+disp], xmm` (F3 0F 10 / 11): load/store a scalar.
pub fn movssLoadMem(dst: Xmm, base: Reg, disp: i32) Inst {
    return sseMem(0xF3, 0x10, dst, base, disp);
}
pub fn movssStoreMem(base: Reg, disp: i32, src: Xmm) Inst {
    return sseMem(0xF3, 0x11, src, base, disp);
}
/// `movsd xmm, [base+disp]` / `movsd [base+disp], xmm` (F2 0F 10 / 11): load/store an f64.
pub fn movsdLoadMem(dst: Xmm, base: Reg, disp: i32) Inst {
    return sseMem(0xF2, 0x10, dst, base, disp);
}
pub fn movsdStoreMem(base: Reg, disp: i32, src: Xmm) Inst {
    return sseMem(0xF2, 0x11, src, base, disp);
}
/// `movups xmm, [base+disp]` / `movups [base+disp], xmm` (0F 10 / 11): load/store a vector.
pub fn movupsLoadMem(dst: Xmm, base: Reg, disp: i32) Inst {
    return sseMem(null, 0x10, dst, base, disp);
}
pub fn movupsStoreMem(base: Reg, disp: i32, src: Xmm) Inst {
    return sseMem(null, 0x11, src, base, disp);
}

/// `call rel32` (E8 cd): a relative call. `rel` is measured from the end of the instruction.
/// The 4-byte displacement is the last 4 bytes, and is the relocation target.
pub fn callRel(rel: i32) Inst {
    const u: u32 = @bitCast(rel);
    return Inst.of(&.{ 0xE8, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// `call r64` (FF /2): an indirect call through a register. This needs no REX.W, since call
/// defaults to a 64-bit operand size in long mode. r8..r15 need the REX.B extension bit.
pub fn callReg(reg: Reg) Inst {
    const rn = n(reg);
    const mrm: u8 = 0xD0 | (rn & 7); // ModRM mod=11, reg field=/2, rm=reg
    if (rn >= 8) return Inst.of(&.{ 0x41, 0xFF, mrm });
    return Inst.of(&.{ 0xFF, mrm });
}

/// `push r64` (0x50+rd): push a 64-bit register onto the stack (rsp -= 8). This needs no
/// REX.W, since push defaults to a 64-bit operand size in long mode. r8..r15 need the REX.B
/// extension bit, so `push r12` is `0x41 0x54`.
pub fn pushReg(r: Reg) Inst {
    const rn = n(r);
    if (rn >= 8) return Inst.of(&.{ 0x41, 0x50 | (rn & 7) });
    return Inst.of(&.{0x50 | rn});
}

/// `pop r64` (0x58+rd): pop the top of the stack into a 64-bit register (rsp += 8). This
/// needs no REX.W. r8..r15 need the REX.B extension bit, so `pop r12` is `0x41 0x5C`.
pub fn popReg(r: Reg) Inst {
    const rn = n(r);
    if (rn >= 8) return Inst.of(&.{ 0x41, 0x58 | (rn & 7) });
    return Inst.of(&.{0x58 | rn});
}

/// `ret` (near return).
pub fn ret() Inst {
    return Inst.of(&.{0xC3});
}

/// `syscall` (the SYSCALL instruction).
pub fn syscall() Inst {
    return Inst.of(&.{ 0x0F, 0x05 });
}

test "known x86-64 encodings" {
    // 64-bit (REX.W) forms.
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00 }, movImm(.rax, 42, true).slice()); // mov rax, 42
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x89, 0xD8 }, movReg(.rax, .rbx).slice()); // mov rax, rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x01, 0xD8 }, add(.rax, .rbx, true).slice()); // add rax, rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x29, 0xD8 }, sub(.rax, .rbx, true).slice()); // sub rax, rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x0F, 0xAF, 0xC3 }, imul(.rax, .rbx, true).slice()); // imul rax, rbx
    // 32-bit forms: no REX for rax..rdi, and `mov r32, imm32` uses the zero-extending B8+rd.
    try std.testing.expectEqualSlices(u8, &.{ 0xB8, 0x2A, 0x00, 0x00, 0x00 }, movImm(.rax, 42, false).slice()); // mov eax, 42
    try std.testing.expectEqualSlices(u8, &.{ 0x01, 0xD8 }, add(.rax, .rbx, false).slice()); // add eax, ebx
    try std.testing.expectEqualSlices(u8, &.{ 0x29, 0xD8 }, sub(.rax, .rbx, false).slice()); // sub eax, ebx
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0xAF, 0xC3 }, imul(.rax, .rbx, false).slice()); // imul eax, ebx
    // A 32-bit op naming r8..r15 still needs a REX (W=0) for the extension bit.
    try std.testing.expectEqualSlices(u8, &.{ 0x41, 0xB8, 0x2A, 0x00, 0x00, 0x00 }, movImm(.r8, 42, false).slice()); // mov r8d, 42
    try std.testing.expectEqualSlices(u8, &.{ 0x44, 0x01, 0xC0 }, add(.rax, .r8, false).slice()); // add eax, r8d
    try std.testing.expectEqualSlices(u8, &.{0xC3}, ret().slice());
    // push/pop r64: 0x50+rd / 0x58+rd, REX.B for r8..r15.
    try std.testing.expectEqualSlices(u8, &.{0x53}, pushReg(.rbx).slice()); // push rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x41, 0x54 }, pushReg(.r12).slice()); // push r12
    try std.testing.expectEqualSlices(u8, &.{0x5B}, popReg(.rbx).slice()); // pop rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x41, 0x5C }, popReg(.r12).slice()); // pop r12
    try std.testing.expectEqualSlices(u8, &.{0x50}, pushReg(.rax).slice()); // push rax
    try std.testing.expectEqualSlices(u8, &.{ 0x41, 0x5F }, popReg(.r15).slice()); // pop r15
    // r8 needs REX.B. mov r8, rax: REX.WB=0x49, 89 /r reg=rax(0) rm=r8(0) -> C0.
    try std.testing.expectEqualSlices(u8, &.{ 0x49, 0x89, 0xC0 }, movReg(.r8, .rax).slice());
}

test "known SSE encodings" {
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x58, 0xC1 }, addss(.xmm0, .xmm1).slice()); // addss xmm0, xmm1
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x10, 0xD3 }, movssRR(.xmm2, .xmm3).slice()); // movss xmm2, xmm3
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x59, 0xC2 }, mulss(.xmm0, .xmm2).slice()); // mulss xmm0, xmm2
    // xmm8 needs REX.R: addss xmm8, xmm0 -> F3 44 0F 58 C0.
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x44, 0x0F, 0x58, 0xC0 }, addss(.xmm8, .xmm0).slice());
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x6E, 0xC0 }, movdToXmm(.xmm0, .rax).slice()); // movd xmm0, eax
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x7E, 0xC0 }, movdFromXmm(.rax, .xmm0).slice()); // movd eax, xmm0
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0x58, 0xC1 }, addps(.xmm0, .xmm1).slice()); // addps xmm0, xmm1
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0x10, 0xC1 }, movupsRR(.xmm0, .xmm1).slice()); // movups xmm0, xmm1
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x3A, 0x21, 0xC1, 0x10 }, insertps(.xmm0, .xmm1, 0x10).slice()); // insertps xmm0, xmm1, 1
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x70, 0xC1, 0x02 }, pshufd(.xmm0, .xmm1, 0x02).slice()); // pshufd xmm0, xmm1, 2
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0x2E, 0xC1 }, ucomiss(.xmm0, .xmm1).slice()); // ucomiss xmm0, xmm1
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x2A, 0xC0 }, cvtsi2ss(.xmm0, .rax).slice()); // cvtsi2ss xmm0, eax
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x2C, 0xC0 }, cvttss2si(.rax, .xmm0).slice()); // cvttss2si eax, xmm0
    try std.testing.expectEqualSlices(u8, &.{ 0xF2, 0x0F, 0x58, 0xC1 }, addsd(.xmm0, .xmm1).slice()); // addsd xmm0, xmm1
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x2E, 0xC1 }, ucomisd(.xmm0, .xmm1).slice()); // ucomisd xmm0, xmm1
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x5A, 0xC1 }, cvtss2sd(.xmm0, .xmm1).slice()); // cvtss2sd xmm0, xmm1
    try std.testing.expectEqualSlices(u8, &.{ 0xF2, 0x0F, 0x2A, 0xC0 }, cvtsi2sd(.xmm0, .rax).slice()); // cvtsi2sd xmm0, eax
    try std.testing.expectEqualSlices(u8, &.{ 0xF2, 0x0F, 0x2C, 0xC0 }, cvttsd2si(.rax, .xmm0).slice()); // cvttsd2si eax, xmm0
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x48, 0x0F, 0x6E, 0xC0 }, movqToXmm(.xmm0, .rax).slice()); // movq xmm0, rax
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11 }, movImm64(.rax, 0x1122334455667788).slice()); // movabs rax, imm64
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x10, 0x84, 0x24, 0x10, 0x00, 0x00, 0x00 }, movssLoad(.xmm0, 0x10).slice()); // movss xmm0, [rsp+16]
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0x11, 0x8C, 0x24, 0x20, 0x00, 0x00, 0x00 }, movupsStore(0x20, .xmm1).slice()); // movups [rsp+32], xmm1
    // sqrtss xmm0, xmm1 -> F3 0F 51 C1
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x51, 0xC1 }, sqrtss(.xmm0, .xmm1).slice());
    // sqrtsd xmm0, xmm1 -> F2 0F 51 C1
    try std.testing.expectEqualSlices(u8, &.{ 0xF2, 0x0F, 0x51, 0xC1 }, sqrtsd(.xmm0, .xmm1).slice());
    // roundss xmm0, xmm1, 0x00 (nearest) -> 66 0F 3A 0A C1 00 (SSE4.1)
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x3A, 0x0A, 0xC1, 0x00 }, roundss(.xmm0, .xmm1, 0x00).slice());
    // roundss xmm0, xmm1, 0x01 (floor) -> 66 0F 3A 0A C1 01
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x3A, 0x0A, 0xC1, 0x01 }, roundss(.xmm0, .xmm1, 0x01).slice());
    // roundss xmm0, xmm1, 0x02 (ceil) -> 66 0F 3A 0A C1 02
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x3A, 0x0A, 0xC1, 0x02 }, roundss(.xmm0, .xmm1, 0x02).slice());
    // roundss xmm0, xmm1, 0x03 (trunc) -> 66 0F 3A 0A C1 03
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x3A, 0x0A, 0xC1, 0x03 }, roundss(.xmm0, .xmm1, 0x03).slice());
    // roundsd xmm0, xmm1, 0x01 (floor) -> 66 0F 3A 0B C1 01
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x0F, 0x3A, 0x0B, 0xC1, 0x01 }, roundsd(.xmm0, .xmm1, 0x01).slice());
}

test "known F16C encodings" {
    // vcvtph2ps xmm0, xmm1 -> VEX.128.66.0F38.W0 13 /r -> C4 E2 79 13 C1.
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0xE2, 0x79, 0x13, 0xC1 }, vcvtph2ps(.xmm0, .xmm1).slice());
    // vcvtps2ph xmm0, xmm1, 0 -> VEX.128.66.0F3A.W0 1D /r ib (reg=src, rm=dst) -> C4 E3 79 1D C8 00.
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0xE3, 0x79, 0x1D, 0xC8, 0x00 }, vcvtps2ph(.xmm0, .xmm1, 0).slice());
    // The reserved scratch registers actually used by isel: vcvtph2ps xmm13, xmm14 -> C4 42 79 13 EE.
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0x42, 0x79, 0x13, 0xEE }, vcvtph2ps(.xmm13, .xmm14).slice());
    // vcvtps2ph xmm15, xmm13, 0 (store-form: reg=src=xmm13, rm=dst=xmm15) -> C4 43 79 1D EF 00.
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0x43, 0x79, 0x1D, 0xEF, 0x00 }, vcvtps2ph(.xmm15, .xmm13, 0).slice());
}

test "known AVX (VEX 256-bit) encodings" {
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0xE1, 0x74, 0x58, 0xC2 }, vaddps(.xmm0, .xmm1, .xmm2).slice()); // vaddps ymm0, ymm1, ymm2
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0xE1, 0x74, 0x59, 0xC2 }, vmulps(.xmm0, .xmm1, .xmm2).slice()); // vmulps ymm0, ymm1, ymm2
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0xE1, 0x7C, 0x10, 0xC1 }, vmovupsRR(.xmm0, .xmm1).slice()); // vmovups ymm0, ymm1
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0xE3, 0x75, 0x18, 0xC2, 0x01 }, vinsertf128(.xmm0, .xmm1, .xmm2, 1).slice()); // vinsertf128 ymm0, ymm1, xmm2, 1
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0xE3, 0x7D, 0x19, 0xC8, 0x01 }, vextractf128(.xmm0, .xmm1, 1).slice()); // vextractf128 xmm0, ymm1, 1
    // High registers exercise VEX.R (dst ymm8) and VEX.B (src2 ymm9).
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0x41, 0x74, 0x58, 0xC1 }, vaddps(.xmm8, .xmm1, .xmm9).slice()); // vaddps ymm8, ymm1, ymm9
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0xE1, 0x7C, 0x11, 0x84, 0x24, 0x20, 0x00, 0x00, 0x00 }, vmovupsStore(0x20, .xmm0).slice()); // vmovups [rsp+32], ymm0
    try std.testing.expectEqualSlices(u8, &.{ 0xC4, 0xE1, 0x7C, 0x10, 0x84, 0x24, 0x10, 0x00, 0x00, 0x00 }, vmovupsLoad(.xmm0, 0x10).slice()); // vmovups ymm0, [rsp+16]
}

test "memory-operand encodings" {
    // mov rax, [rcx+16]: REX.W 8B, mod=10 reg=rax(0) rm=rcx(1) -> 0x81, disp32.
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x8B, 0x81, 0x10, 0x00, 0x00, 0x00 }, movFromMem(.rax, .rcx, 16).slice());
    // mov [rdx+8], rsi: REX.W 89, mod=10 reg=rsi(6) rm=rdx(2) -> 0xB2, disp32.
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x89, 0xB2, 0x08, 0x00, 0x00, 0x00 }, movToMem(.rdx, 8, .rsi).slice());
    // an rsp base needs the 0x24 SIB byte: mov rax, [rsp+16].
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x8B, 0x84, 0x24, 0x10, 0x00, 0x00, 0x00 }, movFromMem(.rax, .rsp, 16).slice());
    // an rbp base at mod=10 keeps the plain disp32 form (no SIB): mov rax, [rbp+0].
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x8B, 0x85, 0x00, 0x00, 0x00, 0x00 }, movFromMem(.rax, .rbp, 0).slice());
    // r8 base needs REX.B: mov rax, [r8+0] -> REX.WB=0x49.
    try std.testing.expectEqualSlices(u8, &.{ 0x49, 0x8B, 0x80, 0x00, 0x00, 0x00, 0x00 }, movFromMem(.rax, .r8, 0).slice());
    // lea rax, [rsp+32]: REX.W 8D, SIB 0x24.
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x8D, 0x84, 0x24, 0x20, 0x00, 0x00, 0x00 }, leaFromStack(.rax, 32).slice());
    // movss xmm0, [rcx+16] (F3 0F 10), movups [rdx+0], xmm1 (0F 11).
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x10, 0x81, 0x10, 0x00, 0x00, 0x00 }, movssLoadMem(.xmm0, .rcx, 16).slice());
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0x11, 0x8A, 0x00, 0x00, 0x00, 0x00 }, movupsStoreMem(.rdx, 0, .xmm1).slice());
    // movss [rsp+8], xmm2 needs the SIB byte: F3 0F 11, reg=xmm2(2) rm=rsp(4).
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x11, 0x94, 0x24, 0x08, 0x00, 0x00, 0x00 }, movssStoreMem(.rsp, 8, .xmm2).slice());
    // movzx eax, word ptr [rcx+16]: 0F B7, mod=10 reg=rax(0) rm=rcx(1) -> 0x81, disp32.
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0xB7, 0x81, 0x10, 0x00, 0x00, 0x00 }, movzxWordFromMem(.rax, .rcx, 16).slice());
    // movzx r10d, word ptr [r11+0]: both extended -> REX.RB=0x45, SIB not needed (r11 low bits=3).
    try std.testing.expectEqualSlices(u8, &.{ 0x45, 0x0F, 0xB7, 0x93, 0x00, 0x00, 0x00, 0x00 }, movzxWordFromMem(.r10, .r11, 0).slice());
    // mov word ptr [rdx+0], ax: 66 89, mod=10 reg=rax(0) rm=rdx(2) -> 0x82, disp32.
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x89, 0x82, 0x00, 0x00, 0x00, 0x00 }, movToMem16(.rdx, 0, .rax).slice());
    // mov word ptr [rsp+8], ax needs the SIB byte: 66 89, reg=rax(0) rm=rsp(4) -> 0x84.
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x89, 0x84, 0x24, 0x08, 0x00, 0x00, 0x00 }, movToMem16(.rsp, 8, .rax).slice());
    // mov byte ptr [rdx+0], al (88 /r): no REX for a low source (rax), reg=rax(0) rm=rdx(2) -> 0x82.
    try std.testing.expectEqualSlices(u8, &.{ 0x88, 0x82, 0x00, 0x00, 0x00, 0x00 }, movToMem8(.rdx, 0, .rax).slice());
    // mov byte ptr [rdx+0], sil: rsi (index 6) needs a REX to address sil, else it means dh.
    try std.testing.expectEqualSlices(u8, &.{ 0x40, 0x88, 0xB2, 0x00, 0x00, 0x00, 0x00 }, movToMem8(.rdx, 0, .rsi).slice());
    // movsx eax, byte ptr [rcx+16] (0F BE): sign-extending i8 load, reg=rax(0) rm=rcx(1) -> 0x81.
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0xBE, 0x81, 0x10, 0x00, 0x00, 0x00 }, movsxByteFromMem(.rax, .rcx, 16).slice());
    // movzx eax, byte ptr [rcx+0] (0F B6): zero-extending i8 load.
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0xB6, 0x81, 0x00, 0x00, 0x00, 0x00 }, movzxByteFromMem(.rax, .rcx, 0).slice());
    // movsx eax, word ptr [rcx+0] (0F BF): sign-extending i16 load.
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0xBF, 0x81, 0x00, 0x00, 0x00, 0x00 }, movsxWordFromMem(.rax, .rcx, 0).slice());
}

test "control-flow encodings" {
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x39, 0xD8 }, cmp(.rax, .rbx, true).slice()); // cmp rax, rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x85, 0xC0 }, testReg(.rax, .rax, true).slice()); // test rax, rax
    try std.testing.expectEqualSlices(u8, &.{ 0x39, 0xD8 }, cmp(.rax, .rbx, false).slice()); // cmp eax, ebx (32-bit)
    try std.testing.expectEqualSlices(u8, &.{ 0x85, 0xC0 }, testReg(.rax, .rax, false).slice()); // test eax, eax (32-bit)
    try std.testing.expectEqualSlices(u8, &.{ 0x40, 0x0F, 0x9F, 0xC0 }, setcc(.rax, .g).slice()); // setg al
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x0F, 0xB6, 0xC0 }, movzxByte(.rax, .rax).slice()); // movzx rax, al
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0x85, 0x00, 0x00, 0x00, 0x00 }, jcc(.ne, 0).slice()); // jne
    try std.testing.expectEqualSlices(u8, &.{ 0xE9, 0x05, 0x00, 0x00, 0x00 }, jmp(5).slice()); // jmp +5
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x0F, 0x45, 0xC3 }, cmovcc(.rax, .rbx, .ne).slice()); // cmovne rax, rbx
}

test "invertCond pairs every condition with its logical inverse" {
    try std.testing.expectEqual(Cond.e, invertCond(.ne));
    try std.testing.expectEqual(Cond.ne, invertCond(.e));
    try std.testing.expectEqual(Cond.ge, invertCond(.l));
    try std.testing.expectEqual(Cond.l, invertCond(.ge));
    try std.testing.expectEqual(Cond.g, invertCond(.le));
    try std.testing.expectEqual(Cond.le, invertCond(.g));
    try std.testing.expectEqual(Cond.ae, invertCond(.b));
    try std.testing.expectEqual(Cond.b, invertCond(.ae));
    try std.testing.expectEqual(Cond.a, invertCond(.be));
    try std.testing.expectEqual(Cond.be, invertCond(.a));
    try std.testing.expectEqual(Cond.np, invertCond(.p));
    try std.testing.expectEqual(Cond.p, invertCond(.np));
    // The inverse is an involution and the low bit toggles the sense.
    inline for (std.meta.tags(Cond)) |c| {
        try std.testing.expectEqual(c, invertCond(invertCond(c)));
        try std.testing.expectEqual(@intFromEnum(c) ^ 1, @intFromEnum(invertCond(c)));
    }
}

test "division and shift encodings" {
    // 64-bit forms.
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x99 }, cqo().slice()); // cqo
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xF7, 0xF9 }, idiv(.rcx, true).slice()); // idiv rcx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xF7, 0xF1 }, divu(.rcx, true).slice()); // div rcx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xD3, 0xE0 }, shlCl(.rax, true).slice()); // shl rax, cl
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xD3, 0xF8 }, sarCl(.rax, true).slice()); // sar rax, cl
    // 32-bit forms: cdq (no REX.W) sign-extends eax->edx, and idiv/div/shifts drop REX.W.
    try std.testing.expectEqualSlices(u8, &.{0x99}, cdq().slice()); // cdq
    try std.testing.expectEqualSlices(u8, &.{ 0xF7, 0xF9 }, idiv(.rcx, false).slice()); // idiv ecx
    try std.testing.expectEqualSlices(u8, &.{ 0xF7, 0xF1 }, divu(.rcx, false).slice()); // div ecx
    try std.testing.expectEqualSlices(u8, &.{ 0xD3, 0xE0 }, shlCl(.rax, false).slice()); // shl eax, cl
    try std.testing.expectEqualSlices(u8, &.{ 0xD3, 0xF8 }, sarCl(.rax, false).slice()); // sar eax, cl
    try std.testing.expectEqualSlices(u8, &.{ 0xD3, 0xE8 }, shrCl(.rax, false).slice()); // shr eax, cl
}
