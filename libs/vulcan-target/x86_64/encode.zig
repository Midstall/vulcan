//! x86-64 instruction encoding. Each encoder returns an `Inst` (byte buffer + length,
//! x86 instructions are at most 15 bytes). Validated by execution under qemu-x86_64
//! (tests/qemu.zig) plus the encoding tests here against known machine code.
//!
//! Encoding model (64-bit operand size): a REX prefix `0x48 | R | X | B` (W=1 for 64-bit,
//! R/B extend the ModRM reg/rm fields to r8..r15), the opcode, then a register-direct
//! ModRM byte `0xC0 | (reg<<3) | rm`.

const std = @import("std");

/// A 64-bit general register. The low 3 bits are the hardware encoding. r8..r15
/// have bit 3 set and need a REX prefix bit.
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

/// An SSE/AVX vector register (xmm0..xmm15). The same 0..15 index space as `Reg` but a
/// separate physical register file, addressed by the SSE opcodes below.
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

/// A two-xmm SSE op: `prefix` (F3/F2/66), `0F`, `op`, register-direct ModRM (reg = dst,
/// rm = src). A REX byte is inserted only when an operand is xmm8..15.
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

/// A cross-file SSE op between an xmm `reg` field and a general `rm` field (e.g. movd).
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
/// `cvttss2si dst, src` (F3 0F 2C /r): convert (with truncation) a scalar single to a 32-bit
/// signed integer in a general register. ModRM reg = the general dst, rm = the xmm src.
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

/// A two-xmm PACKED op with no mandatory prefix (`[REX] 0F op modrm`): the packed-single
/// SSE forms addps/subps/mulps/divps and the packed move movups.
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
/// `dst = (NOT dst) AND src`. All SSE1, so present on the x86-64 baseline.
pub fn andps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x54, dst, src);
}
pub fn andnps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x55, dst, src);
}
pub fn orps(dst: Xmm, src: Xmm) Inst {
    return ssePacked(0x56, dst, src);
}
/// Packed compare `cmpps dst, src, imm8` (0F C2 /r ib): per lane `dst = predicate(dst, src) ?
/// all-ones : all-zero`. imm predicates: 0=EQ 1=LT 2=LE 4=NEQ 5=NLT 6=NLE (ordered 0/1/2/4).
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

// AVX (256-bit YMM) via the VEX prefix. AVX instructions are three-operand (dst = src1 op
// src2) and 256 bits wide (VEX.L = 1). Always uses the 3-byte VEX (C4 ...) form so any of
// ymm0..15 can appear in any operand. `vvvv` holds src1 in 1's-complement. For ops without
// a src1 it is unused, encoded as register 0 (its inverted field then reads 1111).

/// 3-byte VEX, register-direct: ModRM.reg = `reg`, ModRM.rm = `rm`, VEX.vvvv = `vvvv`
/// (register numbers, `vvvv` = 0 means unused). `pp` selects the implied prefix (00 none,
/// 01 = 66, 10 = F3, 11 = F2), `mmmmm` the opcode map (00001 = 0F, 00011 = 0F3A).
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

/// A 256-bit packed-single AVX op (`dst = src1 op src2`): VEX.NDS.256.0F.WIG `op` /r.
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
/// `vmovups ymm, ymm` (VEX.256.0F.WIG 10 /r): copy a whole 256-bit register (vvvv unused).
pub fn vmovupsRR(dst: Xmm, src: Xmm) Inst {
    return vex3(xn(dst), xn(src), 0, 0x10, 0b00, 0b00001, null);
}
/// `vinsertf128 ymm1, ymm2, xmm3, imm8` (VEX.256.66.0F3A.W0 18 /r ib): put `src2`'s 128 bits
/// into the half of `dst` chosen by `imm` (0 = low, 1 = high), the other half from `src1`.
pub fn vinsertf128(dst: Xmm, src1: Xmm, src2: Xmm, imm: u8) Inst {
    return vex3(xn(dst), xn(src2), xn(src1), 0x18, 0b01, 0b00011, imm);
}
/// `vextractf128 xmm1, ymm2, imm8` (VEX.256.66.0F3A.W0 19 /r ib): extract the 128-bit half
/// of `src` selected by `imm` into `dst`. The store-form ModRM puts src in reg, dst in rm.
pub fn vextractf128(dst: Xmm, src: Xmm, imm: u8) Inst {
    return vex3(xn(src), xn(dst), 0, 0x19, 0b01, 0b00011, imm);
}

/// A 256-bit `vmovups` to/from `[rsp+disp]` (VEX.256.0F.WIG `op`, ModRM mod=10 reg=x rm=SIB,
/// SIB base=rsp, disp32): spill/reload a whole ymm. vvvv and the SIB index are unused.
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
/// `ucomiss a, b` (0F 2E /r): unordered compare two scalar singles, setting ZF/PF/CF like
/// an integer `cmp a, b` (PF marks an unordered/NaN result).
pub fn ucomiss(a: Xmm, b: Xmm) Inst {
    return ssePacked(0x2E, a, b);
}

// F16C (half<->single conversion). Both ops are VEX-only (there is NO legacy SSE encoding)
// and operate on 128-bit registers, so they need the 3-byte VEX with L=0 rather than the
// L=1 form the AVX helpers above use. `vex128` is that L=0 variant of `vex3`.

/// A 3-byte VEX (register-direct) at 128-bit width (VEX.L=0, VEX.W=0). Otherwise identical
/// to `vex3`: ModRM.reg = `reg`, ModRM.rm = `rm`, VEX.vvvv = `vvvv` (0 means unused, encoded
/// as 1111). `pp` selects the implied prefix (01 = 66), `mmmmm` the opcode map (00010 = 0F38,
/// 00011 = 0F3A).
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
/// low 64 bits of `src` to four packed f32 in `dst`. Lane 0 is the scalar half we hold; the
/// other lanes widen harmlessly. Has no NDS operand, so vvvv is unused (1111).
pub fn vcvtph2ps(dst: Xmm, src: Xmm) Inst {
    return vex128(xn(dst), xn(src), 0, 0x13, 0b01, 0b00010, null);
}

/// `vcvtps2ph dst, src, imm8` (VEX.128.66.0F3A.W0 1D /r ib): narrow the four packed f32 in
/// `src` to four packed IEEE halves in the low 64 bits of `dst` (zeroing dst[127:64]). This is
/// a STORE-form op, so ModRM.reg is the SOURCE and ModRM.rm the DESTINATION. imm8 with bit2=0
/// selects the rounding mode from bits1:0 (0 = round-to-nearest-even), NOT MXCSR.
pub fn vcvtps2ph(dst: Xmm, src: Xmm, imm: u8) Inst {
    return vex128(xn(src), xn(dst), 0, 0x1D, 0b01, 0b00011, imm);
}

// Double-precision (scalar f64) SSE2: the same encoders with an F2 prefix, plus the 64-bit
// conversions, ucomisd (66 prefix), and the f32<->f64 conversions (xx 0F 5A).
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
/// `ucomisd a, b` (66 0F 2E /r): the f64 unordered compare (flags like ucomiss).
pub fn ucomisd(a: Xmm, b: Xmm) Inst {
    return sseRR(0x66, 0x2E, a, b);
}
/// `cvtss2sd dst, src` (F3 0F 5A) widens f32 to f64. `cvtsd2ss` (F2 0F 5A) narrows.
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
/// Packed single-precision `sqrtps dst, src` (0F 51 /r): per-lane square root (SSE1).
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
/// `roundss dst, src, imm8` (F3 0F 3A /11 ib): IEEE rounding for f32.
pub fn roundss(dst: Xmm, src: Xmm, imm: u8) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ 0xF3, rex, 0x0F, 0x3A, 0x11, mod, imm });
    }
    return Inst.of(&.{ 0xF3, 0x0F, 0x3A, 0x11, mod, imm });
}
/// `roundsd dst, src, imm8` (F2 0F 3A /11 ib): IEEE rounding for f64.
pub fn roundsd(dst: Xmm, src: Xmm, imm: u8) Inst {
    const r = xn(dst);
    const b = xn(src);
    const mod: u8 = 0xC0 | ((r & 7) << 3) | (b & 7);
    if (r >= 8 or b >= 8) {
        const rex: u8 = 0x40 | (@as(u8, @intFromBool(r >= 8)) << 2) | @intFromBool(b >= 8);
        return Inst.of(&.{ 0xF2, rex, 0x0F, 0x3A, 0x11, mod, imm });
    }
    return Inst.of(&.{ 0xF2, 0x0F, 0x3A, 0x11, mod, imm });
}
/// `cvtsi2sd dst, src` (F2 0F 2A /r): 32-bit signed int in a gpr to a scalar double.
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
/// `movq dst, src`: copy all 64 bits between a general register and an xmm low lane (66 REX.W
/// 0F 6E to load, 66 REX.W 0F 7E to read out). Used to materialize an f64 constant.
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
/// imm = lane<<4 this copies src's lane 0 into dst's lane `lane` (src-lane and zero-mask 0).
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
/// `pshufd dst, src, imm8` (66 0F 70 /r ib): shuffle 32-bit lanes. With imm = lane this puts
/// src's lane `lane` into dst's lane 0 (used to extract a lane to a scalar position).
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

/// REX prefix for a 64-bit op with ModRM `reg` and `rm` register operands.
fn rexW(reg: Reg, rm: Reg) u8 {
    return 0x48 | (@as(u8, @intFromBool(n(reg) >= 8)) << 2) | @intFromBool(n(rm) >= 8);
}

/// Register-direct ModRM byte (mod = 11).
fn modrm(reg: Reg, rm: Reg) u8 {
    return 0xC0 | ((n(reg) & 7) << 3) | (n(rm) & 7);
}

/// A 64-bit two-register ALU op: `opcode` with `reg` as the source and `rm` as the
/// destination (the `/r` form, e.g. ADD/SUB/MOV r/m64, r64).
fn aluRR(opcode: u8, src: Reg, dst: Reg) Inst {
    return Inst.of(&.{ rexW(src, dst), opcode, modrm(src, dst) });
}

/// `mov dst, imm32` (C7 /0): load a sign-extended 32-bit immediate.
pub fn movImm(dst: Reg, imm: i32) Inst {
    const u: u32 = @bitCast(imm);
    return Inst.of(&.{
        0x48 | @as(u8, @intFromBool(n(dst) >= 8)),
        0xC7,
        0xC0 | (n(dst) & 7), // /0
        @truncate(u),
        @truncate(u >> 8),
        @truncate(u >> 16),
        @truncate(u >> 24),
    });
}

/// `mov dst, src` (89 /r: store the reg operand into the r/m operand).
pub fn movReg(dst: Reg, src: Reg) Inst {
    return aluRR(0x89, src, dst);
}

/// `add dst, src` (01 /r).
pub fn add(dst: Reg, src: Reg) Inst {
    return aluRR(0x01, src, dst);
}

/// `sub dst, src` (29 /r).
pub fn sub(dst: Reg, src: Reg) Inst {
    return aluRR(0x29, src, dst);
}

/// `and dst, src` (21 /r).
pub fn andr(dst: Reg, src: Reg) Inst {
    return aluRR(0x21, src, dst);
}

/// `or dst, src` (09 /r).
pub fn orr(dst: Reg, src: Reg) Inst {
    return aluRR(0x09, src, dst);
}

/// `xor dst, src` (31 /r).
pub fn xorr(dst: Reg, src: Reg) Inst {
    return aluRR(0x31, src, dst);
}

/// `imul dst, src` (0F AF /r: dst = dst * src). Two-operand signed multiply.
pub fn imul(dst: Reg, src: Reg) Inst {
    return Inst.of(&.{ rexW(dst, src), 0x0F, 0xAF, modrm(dst, src) });
}

/// An ALU op against a 32-bit immediate (REX.W 81 /digit id): ADD=/0, OR=/1,
/// AND=/4, SUB=/5, XOR=/6.
pub fn aluImm(digit: u3, dst: Reg, imm: i32) Inst {
    const u: u32 = @bitCast(imm);
    return Inst.of(&.{
        0x48 | @as(u8, @intFromBool(n(dst) >= 8)),
        0x81,
        0xC0 | (@as(u8, digit) << 3) | (n(dst) & 7),
        @truncate(u),
        @truncate(u >> 8),
        @truncate(u >> 16),
        @truncate(u >> 24),
    });
}

/// `imul dst, src, imm32` (REX.W 69 /r id): dst = src * imm (three-operand).
pub fn imulImm(dst: Reg, src: Reg, imm: i32) Inst {
    const u: u32 = @bitCast(imm);
    return Inst.of(&.{
        rexW(dst, src),     0x69,
        modrm(dst, src),    @truncate(u),
        @truncate(u >> 8),  @truncate(u >> 16),
        @truncate(u >> 24),
    });
}

/// A shift of `dst` by an immediate count (REX.W C1 /digit ib): SHL=/4, SHR=/5,
/// SAR=/7.
pub fn shiftImm(digit: u3, dst: Reg, imm8: u8) Inst {
    return Inst.of(&.{ 0x48 | @as(u8, @intFromBool(n(dst) >= 8)), 0xC1, 0xC0 | (@as(u8, digit) << 3) | (n(dst) & 7), imm8 });
}

/// `cqo` (REX.W 99): sign-extend RAX into RDX:RAX (the dividend for `idiv`).
pub fn cqo() Inst {
    return Inst.of(&.{ 0x48, 0x99 });
}

/// `idiv src` (REX.W F7 /7): signed divide RDX:RAX by `src`, quotient -> RAX,
/// remainder -> RDX. `src` must not be RAX or RDX.
pub fn idiv(src: Reg) Inst {
    return Inst.of(&.{ 0x48 | @as(u8, @intFromBool(n(src) >= 8)), 0xF7, 0xC0 | (7 << 3) | (n(src) & 7) });
}

/// `div src` (REX.W F7 /6): unsigned divide RDX:RAX by `src` (clear RDX first).
pub fn divu(src: Reg) Inst {
    return Inst.of(&.{ 0x48 | @as(u8, @intFromBool(n(src) >= 8)), 0xF7, 0xC0 | (6 << 3) | (n(src) & 7) });
}

/// A shift of `dst` by the count in CL (REX.W D3 /digit). `shl` = /4, `shr` = /5
/// (logical), `sar` = /7 (arithmetic).
fn shiftCl(digit: u8, dst: Reg) Inst {
    return Inst.of(&.{ 0x48 | @as(u8, @intFromBool(n(dst) >= 8)), 0xD3, 0xC0 | (digit << 3) | (n(dst) & 7) });
}
pub fn shlCl(dst: Reg) Inst {
    return shiftCl(4, dst);
}
pub fn shrCl(dst: Reg) Inst {
    return shiftCl(5, dst);
}
pub fn sarCl(dst: Reg) Inst {
    return shiftCl(7, dst);
}

/// `cmp a, b` (39 /r): compute `a - b` and set flags (operands unchanged).
pub fn cmp(a: Reg, b: Reg) Inst {
    return aluRR(0x39, b, a);
}

/// `test a, b` (85 /r): compute `a & b` and set flags. `test r, r` tests for zero.
pub fn testReg(a: Reg, b: Reg) Inst {
    return aluRR(0x85, b, a);
}

/// A condition code (the low nibble of the Jcc/SETcc opcode).
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
    p = 0xA, // parity (an unordered/NaN float compare)
    np = 0xB, // not parity (an ordered float compare)
};

/// `cmovcc dst, src` (0F 40+cc /r): move `src` into `dst` if the condition holds.
pub fn cmovcc(dst: Reg, src: Reg, cond: Cond) Inst {
    return Inst.of(&.{ rexW(dst, src), 0x0F, 0x40 | @intFromEnum(cond), modrm(dst, src) });
}

/// `setcc dst8` (0F 90+cc): set the low byte of `dst` to 0/1 from the flags. A REX
/// prefix is always emitted so the low byte (spl/sil/... for regs 4..7, r8b.. for
/// 8..15) is addressed, never the legacy ah/ch/dh/bh.
pub fn setcc(dst: Reg, cond: Cond) Inst {
    return Inst.of(&.{ 0x40 | @as(u8, @intFromBool(n(dst) >= 8)), 0x0F, 0x90 | @intFromEnum(cond), 0xC0 | (n(dst) & 7) });
}

/// `movzx dst, dst8` (REX.W 0F B6 /r): zero-extend `src`'s low byte into `dst`.
pub fn movzxByte(dst: Reg, src: Reg) Inst {
    return Inst.of(&.{ rexW(dst, src), 0x0F, 0xB6, modrm(dst, src) });
}

/// `jcc rel32` (0F 80+cc cd): branch when the condition holds. `rel` is from the
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

/// `mov dst, [rsp + disp32]` (REX.W 8B /r): load a stack slot (the RSP base needs a
/// SIB byte, 0x24).
pub fn movFromStack(dst: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    return Inst.of(&.{ rexW(dst, .rax), 0x8B, 0x84 | ((n(dst) & 7) << 3), 0x24, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// `mov [rsp + disp32], src` (REX.W 89 /r): store to a stack slot.
pub fn movToStack(disp: i32, src: Reg) Inst {
    const u: u32 = @bitCast(disp);
    return Inst.of(&.{ rexW(src, .rax), 0x89, 0x84 | ((n(src) & 7) << 3), 0x24, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// An SSE load/store to `[rsp + disp32]` (the RSP base needs the 0x24 SIB byte). `prefix`
/// null = packed (movups), 0xF3 = scalar single (movss). REX.R is added for xmm8..15.
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
/// `movups xmm, [rsp+disp]` / `movups [rsp+disp], xmm` (0F 10 / 11): reload/spill a vector.
pub fn movupsLoad(dst: Xmm, disp: i32) Inst {
    return sseStack(null, 0x10, dst, disp);
}
pub fn movupsStore(disp: i32, src: Xmm) Inst {
    return sseStack(null, 0x11, src, disp);
}

/// A general `mov` to/from `[base + disp32]` with an ARBITRARY base register (mod=10,
/// disp32). The data register is the ModRM `reg` field, the base the `rm` field. A base
/// of rsp(4)/r12 needs the 0x24 SIB byte, and rm=rbp(5) is fine at mod=10 (disp32 explicit).
/// `op` is 0x8B for a load (mov r, [base+disp]) or 0x89 for a store (mov [base+disp], r).
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
/// A memory mov without REX.W (32-bit operand): reads/writes 4 bytes. A 32-bit load
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

/// `movzx dst, word ptr [base+disp32]` (0F B7 /r): load 16 bits, zero-extended into the 32-bit
/// (and thus the full 64-bit) register. Used to bring an IEEE half out of memory before it is
/// widened to f32. A base of rsp/r12 (rm low bits = 100) needs the 0x24 SIB byte; r8..15 base
/// or dst needs a REX bit.
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
/// operand-size prefix (before any REX) makes it a 16-bit store, so exactly 2 bytes are
/// written (an IEEE half). A base of rsp/r12 needs the 0x24 SIB byte; r8..15 needs a REX bit.
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

/// `lea dst, [rsp + disp32]` (REX.W 8D /r): materialize a stack address (e.g. an alloca slot).
/// The rsp base needs the 0x24 SIB byte, mod=10 for the disp32 form.
pub fn leaFromStack(dst: Reg, disp: i32) Inst {
    const u: u32 = @bitCast(disp);
    return Inst.of(&.{ rexW(dst, .rax), 0x8D, 0x84 | ((n(dst) & 7) << 3), 0x24, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// An SSE load/store to `[base + disp32]` with an arbitrary base register. `prefix` null =
/// packed (movups), 0xF3 = scalar single (movss). REX is added for an xmm8..15 data register
/// or an r8..15 base. A base of rsp/r12 (rm low bits = 100) needs the 0x24 SIB byte.
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

/// `call rel32` (E8 cd): a relative call. `rel` is from the end of the instruction.
/// The 4-byte displacement is the last 4 bytes (the relocation target).
pub fn callRel(rel: i32) Inst {
    const u: u32 = @bitCast(rel);
    return Inst.of(&.{ 0xE8, @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) });
}

/// `call r64` (FF /2): an indirect call through a register. No REX.W (call defaults to
/// 64-bit operand size in long mode); r8..r15 need the REX.B extension bit.
pub fn callReg(reg: Reg) Inst {
    const rn = n(reg);
    const mrm: u8 = 0xD0 | (rn & 7); // ModRM mod=11, reg field=/2, rm=reg
    if (rn >= 8) return Inst.of(&.{ 0x41, 0xFF, mrm });
    return Inst.of(&.{ 0xFF, mrm });
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
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00 }, movImm(.rax, 42).slice()); // mov rax, 42
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x89, 0xD8 }, movReg(.rax, .rbx).slice()); // mov rax, rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x01, 0xD8 }, add(.rax, .rbx).slice()); // add rax, rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x29, 0xD8 }, sub(.rax, .rbx).slice()); // sub rax, rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x0F, 0xAF, 0xC3 }, imul(.rax, .rbx).slice()); // imul rax, rbx
    try std.testing.expectEqualSlices(u8, &.{0xC3}, ret().slice());
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
    // roundss xmm0, xmm1, 0x00 (nearest) -> F3 0F 3A 11 C1 00
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x3A, 0x11, 0xC1, 0x00 }, roundss(.xmm0, .xmm1, 0x00).slice());
    // roundss xmm0, xmm1, 0x01 (floor) -> F3 0F 3A 11 C1 01
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x3A, 0x11, 0xC1, 0x01 }, roundss(.xmm0, .xmm1, 0x01).slice());
    // roundss xmm0, xmm1, 0x02 (ceil) -> F3 0F 3A 11 C1 02
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x3A, 0x11, 0xC1, 0x02 }, roundss(.xmm0, .xmm1, 0x02).slice());
    // roundss xmm0, xmm1, 0x03 (trunc) -> F3 0F 3A 11 C1 03
    try std.testing.expectEqualSlices(u8, &.{ 0xF3, 0x0F, 0x3A, 0x11, 0xC1, 0x03 }, roundss(.xmm0, .xmm1, 0x03).slice());
    // roundsd xmm0, xmm1, 0x01 (floor) -> F2 0F 3A 11 C1 01
    try std.testing.expectEqualSlices(u8, &.{ 0xF2, 0x0F, 0x3A, 0x11, 0xC1, 0x01 }, roundsd(.xmm0, .xmm1, 0x01).slice());
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
}

test "control-flow encodings" {
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x39, 0xD8 }, cmp(.rax, .rbx).slice()); // cmp rax, rbx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x85, 0xC0 }, testReg(.rax, .rax).slice()); // test rax, rax
    try std.testing.expectEqualSlices(u8, &.{ 0x40, 0x0F, 0x9F, 0xC0 }, setcc(.rax, .g).slice()); // setg al
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x0F, 0xB6, 0xC0 }, movzxByte(.rax, .rax).slice()); // movzx rax, al
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0x85, 0x00, 0x00, 0x00, 0x00 }, jcc(.ne, 0).slice()); // jne
    try std.testing.expectEqualSlices(u8, &.{ 0xE9, 0x05, 0x00, 0x00, 0x00 }, jmp(5).slice()); // jmp +5
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x0F, 0x45, 0xC3 }, cmovcc(.rax, .rbx, .ne).slice()); // cmovne rax, rbx
}

test "division and shift encodings" {
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0x99 }, cqo().slice()); // cqo
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xF7, 0xF9 }, idiv(.rcx).slice()); // idiv rcx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xF7, 0xF1 }, divu(.rcx).slice()); // div rcx
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xD3, 0xE0 }, shlCl(.rax).slice()); // shl rax, cl
    try std.testing.expectEqualSlices(u8, &.{ 0x48, 0xD3, 0xF8 }, sarCl(.rax).slice()); // sar rax, cl
}
