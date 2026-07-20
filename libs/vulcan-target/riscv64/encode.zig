//! RISC-V instruction encoding to 32-bit machine words. Base integer
//! (RV32I/RV64I) plus M/F/D/Zbb and RVV, verified against known encodings.

const std = @import("std");

/// A RISC-V integer register, x0 through x31.
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
    x31,
};

fn num(reg: Reg) u32 {
    return @intFromEnum(reg);
}

/// A RISC-V floating-point register, f0 through f31.
pub const FReg = enum(u5) {
    f0,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
    f13,
    f14,
    f15,
    f16,
    f17,
    f18,
    f19,
    f20,
    f21,
    f22,
    f23,
    f24,
    f25,
    f26,
    f27,
    f28,
    f29,
    f30,
    f31,
};

fn fnum(reg: FReg) u32 {
    return @intFromEnum(reg);
}

/// A RISC-V Vector (RVV) register v0..v31.
pub const VReg = enum(u5) {
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
    v12,
    v13,
    v14,
    v15,
    v16,
    v17,
    v18,
    v19,
    v20,
    v21,
    v22,
    v23,
    v24,
    v25,
    v26,
    v27,
    v28,
    v29,
    v30,
    v31,
};

fn vnum(reg: VReg) u32 {
    return @intFromEnum(reg);
}

// RVV (V extension). The vector unit is pinned to a 4-lane f32 group: vsetivli
// with AVL=4 and vtype 0xD0 (SEW=32/e32, LMUL=1/m1, tail- and mask-agnostic).
// All ops are unmasked (vm=1). Lowers the fixed <4 x f32> from the auto-vectorizer.

/// `vsetivli rd, avl, vtype` (OP-V, funct3=111): configure VL/SEW/LMUL from immediates.
pub fn vsetivli(rd: Reg, avl: u5, vtype: u10) u32 {
    return 0b1010111 | (num(rd) << 7) | (0b111 << 12) | (@as(u32, avl) << 15) | (@as(u32, vtype) << 20) | (@as(u32, 0b11) << 30);
}
/// `vle32.v vd, (rs1)`: unit-stride 32-bit-element vector load (width=110, vm=1).
pub fn vle32(vd: VReg, rs1: Reg) u32 {
    return 0b0000111 | (vnum(vd) << 7) | (0b110 << 12) | (num(rs1) << 15) | (@as(u32, 1) << 25);
}
/// `vse32.v vs3, (rs1)`: unit-stride 32-bit-element vector store.
pub fn vse32(vs3: VReg, rs1: Reg) u32 {
    return 0b0100111 | (vnum(vs3) << 7) | (0b110 << 12) | (num(rs1) << 15) | (@as(u32, 1) << 25);
}
/// An OPFVV op (vector-vector float, funct3=001): `vd = vs2 <op> vs1`, unmasked.
fn opfvv(funct6: u6, vd: VReg, vs2: VReg, vs1: VReg) u32 {
    return 0b1010111 | (vnum(vd) << 7) | (0b001 << 12) | (vnum(vs1) << 15) | (vnum(vs2) << 20) | (@as(u32, 1) << 25) | (@as(u32, funct6) << 26);
}
pub fn vfadd_vv(vd: VReg, vs2: VReg, vs1: VReg) u32 {
    return opfvv(0b000000, vd, vs2, vs1);
}
pub fn vfsub_vv(vd: VReg, vs2: VReg, vs1: VReg) u32 {
    return opfvv(0b000010, vd, vs2, vs1);
}
pub fn vfmul_vv(vd: VReg, vs2: VReg, vs1: VReg) u32 {
    return opfvv(0b100100, vd, vs2, vs1);
}
pub fn vfdiv_vv(vd: VReg, vs2: VReg, vs1: VReg) u32 {
    return opfvv(0b100000, vd, vs2, vs1);
}
// Fused multiply-add/sub family (OPFVV): unlike the plain arithmetic ops above, these
// accumulate INTO vd (vd is a third source operand as well as the destination), so the
// assembler operand order is conventionally written `vd, vs1, vs2` (the two multiplicands)
// rather than the `vd, vs2, vs1` used for vfadd/vfsub/etc above. vs1/vs2 still land in the
// same bit fields as `opfvv` expects; only the argument order in these wrappers changes to
// match the assembler mnemonic. Field layout and funct6 values cross-checked byte-for-byte
// against `zig cc -target riscv64-linux-musl` (LLVM's assembler) output for
// `vfmacc.vv v3,v1,v2` / `vfmsac.vv v3,v1,v2` / `vfnmsac.vv v3,v1,v2`.
/// `vfmacc.vv vd, vs1, vs2`: vd = vs1*vs2 + vd.
pub fn vfmacc_vv(vd: VReg, vs1: VReg, vs2: VReg) u32 {
    return opfvv(0b101100, vd, vs2, vs1);
}
/// `vfmsac.vv vd, vs1, vs2`: vd = vs1*vs2 - vd.
pub fn vfmsac_vv(vd: VReg, vs1: VReg, vs2: VReg) u32 {
    return opfvv(0b101110, vd, vs2, vs1);
}
/// `vfnmsac.vv vd, vs1, vs2`: vd = -(vs1*vs2) + vd = vd - vs1*vs2.
pub fn vfnmsac_vv(vd: VReg, vs1: VReg, vs2: VReg) u32 {
    return opfvv(0b101111, vd, vs2, vs1);
}
/// `vfmv.f.s rd, vs2`: move lane 0 to a float register (the unpack primitive).
pub fn vfmv_f_s(rd: FReg, vs2: VReg) u32 {
    return 0b1010111 | (fnum(rd) << 7) | (0b001 << 12) | (vnum(vs2) << 20) | (@as(u32, 1) << 25) | (@as(u32, 0b010000) << 26);
}
/// `vfmv.s.f vd, rs1`: set lane 0 from a float register (the pack seed, funct3=101 OPFVF).
pub fn vfmv_s_f(vd: VReg, rs1: FReg) u32 {
    return 0b1010111 | (vnum(vd) << 7) | (0b101 << 12) | (fnum(rs1) << 15) | (@as(u32, 1) << 25) | (@as(u32, 0b010000) << 26);
}
/// `vfslide1up.vf vd, vs2, rs1`: shift lanes up by one, inserting `rs1` at lane 0 (pack step).
pub fn vfslide1up_vf(vd: VReg, vs2: VReg, rs1: FReg) u32 {
    return 0b1010111 | (vnum(vd) << 7) | (0b101 << 12) | (fnum(rs1) << 15) | (vnum(vs2) << 20) | (@as(u32, 1) << 25) | (@as(u32, 0b001110) << 26);
}
/// `vslidedown.vi vd, vs2, uimm`: shift lanes down by `uimm`, bringing lane uimm to lane 0.
pub fn vslidedown_vi(vd: VReg, vs2: VReg, uimm: u5) u32 {
    return 0b1010111 | (vnum(vd) << 7) | (0b011 << 12) | (@as(u32, uimm) << 15) | (vnum(vs2) << 20) | (@as(u32, 1) << 25) | (@as(u32, 0b001111) << 26);
}
/// `vmv.v.v vd, vs1`: copy a whole vector register (OPIVV funct6=010111, vs2=0, unmasked).
pub fn vmv_v_v(vd: VReg, vs1: VReg) u32 {
    return 0b1010111 | (vnum(vd) << 7) | (0b000 << 12) | (vnum(vs1) << 15) | (@as(u32, 1) << 25) | (@as(u32, 0b010111) << 26);
}

/// Encode an OP-FP (floating-point R-type) instruction. `rm` is the rounding
/// mode (0b111 = dynamic).
fn fpRType(funct7: u7, rm: u3, rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return @as(u32, 0b1010011) |
        (fnum(rd) << 7) |
        (@as(u32, rm) << 12) |
        (fnum(rs1) << 15) |
        (fnum(rs2) << 20) |
        (@as(u32, funct7) << 25);
}

/// `fadd.s rd, rs1, rs2` (single-precision add, dynamic rounding)
pub fn fadd_s(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0000000, 0b111, rd, rs1, rs2);
}

/// `fsub.s rd, rs1, rs2`
pub fn fsub_s(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0000100, 0b111, rd, rs1, rs2);
}

/// `fmul.s rd, rs1, rs2`
pub fn fmul_s(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0001000, 0b111, rd, rs1, rs2);
}

/// `fdiv.s rd, rs1, rs2`
pub fn fdiv_s(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0001100, 0b111, rd, rs1, rs2);
}

/// `fadd.d rd, rs1, rs2` (double-precision)
pub fn fadd_d(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0000001, 0b111, rd, rs1, rs2);
}

/// `fsub.d rd, rs1, rs2`
pub fn fsub_d(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0000101, 0b111, rd, rs1, rs2);
}

/// `fmul.d rd, rs1, rs2`
pub fn fmul_d(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0001001, 0b111, rd, rs1, rs2);
}

/// `fdiv.d rd, rs1, rs2`
pub fn fdiv_d(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0001101, 0b111, rd, rs1, rs2);
}

/// Encode a floating-point compare (rd is an integer register).
fn fpCmp(funct7: u7, funct3: u3, rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return @as(u32, 0b1010011) |
        (num(rd) << 7) |
        (@as(u32, funct3) << 12) |
        (fnum(rs1) << 15) |
        (fnum(rs2) << 20) |
        (@as(u32, funct7) << 25);
}

/// `feq.s rd, rs1, rs2` (set if equal)
pub fn feq_s(rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return fpCmp(0b1010000, 0b010, rd, rs1, rs2);
}

/// `flt.s rd, rs1, rs2` (set if less-than)
pub fn flt_s(rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return fpCmp(0b1010000, 0b001, rd, rs1, rs2);
}

/// `fle.s rd, rs1, rs2` (set if less-or-equal)
pub fn fle_s(rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return fpCmp(0b1010000, 0b000, rd, rs1, rs2);
}

/// `feq.d rd, rs1, rs2`
pub fn feq_d(rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return fpCmp(0b1010001, 0b010, rd, rs1, rs2);
}

/// `flt.d rd, rs1, rs2`
pub fn flt_d(rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return fpCmp(0b1010001, 0b001, rd, rs1, rs2);
}

/// `fle.d rd, rs1, rs2`
pub fn fle_d(rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return fpCmp(0b1010001, 0b000, rd, rs1, rs2);
}

/// `flw rd, imm(rs1)` (load single-precision float, `rs1` is an integer base).
pub fn flw(rd: FReg, rs1: Reg, imm: i12) u32 {
    return @as(u32, 0b0000111) | (fnum(rd) << 7) | (0b010 << 12) | (num(rs1) << 15) | (@as(u32, @as(u12, @bitCast(imm))) << 20);
}

/// `fld rd, imm(rs1)` (load double-precision float)
pub fn fld(rd: FReg, rs1: Reg, imm: i12) u32 {
    return @as(u32, 0b0000111) | (fnum(rd) << 7) | (0b011 << 12) | (num(rs1) << 15) | (@as(u32, @as(u12, @bitCast(imm))) << 20);
}

/// `fsw rs2, imm(rs1)` (store single-precision float, `rs1` is an integer base).
pub fn fsw(rs2: FReg, rs1: Reg, imm: i12) u32 {
    const bits: u12 = @bitCast(imm);
    return @as(u32, 0b0100111) | ((@as(u32, bits) & 0x1f) << 7) | (0b010 << 12) | (num(rs1) << 15) | (fnum(rs2) << 20) | (((@as(u32, bits) >> 5) & 0x7f) << 25);
}

/// `fsd rs2, imm(rs1)` (store double-precision float)
pub fn fsd(rs2: FReg, rs1: Reg, imm: i12) u32 {
    const bits: u12 = @bitCast(imm);
    return @as(u32, 0b0100111) | ((@as(u32, bits) & 0x1f) << 7) | (0b011 << 12) | (num(rs1) << 15) | (fnum(rs2) << 20) | (((@as(u32, bits) >> 5) & 0x7f) << 25);
}

/// `fmv.w.x rd, rs1` (move 32 integer bits into a float register, no convert).
pub fn fmv_w_x(rd: FReg, rs1: Reg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (num(rs1) << 15) | (@as(u32, 0b1111000) << 25);
}

/// `fmv.x.w rd, rs1` (move a float register's 32 bits into an integer register, no convert).
pub fn fmv_x_w(rd: Reg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (num(rd) << 7) | (fnum(rs1) << 15) | (@as(u32, 0b1110000) << 25);
}

/// `ecall` (environment call, a Linux syscall under qemu user mode).
pub fn ecall() u32 {
    return 0x00000073;
}

/// `fmv.d.x rd, rs1` (move 64 integer bits into a float register, RV64).
pub fn fmv_d_x(rd: FReg, rs1: Reg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (num(rs1) << 15) | (@as(u32, 0b1111001) << 25);
}

// Scalar fused multiply-add family (R4-type, one rounding instead of two - legal because
// Vulcan permits fp-contraction). Layout: rs3[31:27] fmt[26:25] rs2[24:20] rs1[19:15]
// rm[14:12] rd[11:7] opcode[6:0]. fmt: 00 = single (.s), 01 = double (.d). rm = 111 (dynamic).
// One opcode per variant (unlike the et-soc VPU's `vpuPsFma`, which packs all four into one
// opcode via a `sel` field): FMADD = 0x43, FMSUB = 0x47, FNMSUB = 0x4B, FNMADD = 0x4F (deferred,
// unused by the fusion in isel.zig). Confirmed against `disasm.zig`'s independently-written `fma`
// decoder (already verified against llvm-objdump: see "decodes RV64 word ops, FMA, and base
// pseudos") - the golden word there round-trips through these encoders in the tests below.
fn fma4Type(opcode: u7, fmt: u2, rd: FReg, rs1: FReg, rs2: FReg, rs3: FReg) u32 {
    return @as(u32, opcode) |
        (fnum(rd) << 7) |
        (@as(u32, 0b111) << 12) |
        (fnum(rs1) << 15) |
        (fnum(rs2) << 20) |
        (@as(u32, fmt) << 25) |
        (fnum(rs3) << 27);
}

/// `fmadd.s rd, rs1, rs2, rs3`: rd = rs1*rs2 + rs3 (single-precision, dynamic rounding).
pub fn fmadd_s(rd: FReg, rs1: FReg, rs2: FReg, rs3: FReg) u32 {
    return fma4Type(0b1000011, 0b00, rd, rs1, rs2, rs3);
}

/// `fmadd.d rd, rs1, rs2, rs3`: rd = rs1*rs2 + rs3 (double-precision).
pub fn fmadd_d(rd: FReg, rs1: FReg, rs2: FReg, rs3: FReg) u32 {
    return fma4Type(0b1000011, 0b01, rd, rs1, rs2, rs3);
}

/// `fmsub.s rd, rs1, rs2, rs3`: rd = rs1*rs2 - rs3 (single-precision).
pub fn fmsub_s(rd: FReg, rs1: FReg, rs2: FReg, rs3: FReg) u32 {
    return fma4Type(0b1000111, 0b00, rd, rs1, rs2, rs3);
}

/// `fmsub.d rd, rs1, rs2, rs3`: rd = rs1*rs2 - rs3 (double-precision).
pub fn fmsub_d(rd: FReg, rs1: FReg, rs2: FReg, rs3: FReg) u32 {
    return fma4Type(0b1000111, 0b01, rd, rs1, rs2, rs3);
}

/// `fnmsub.s rd, rs1, rs2, rs3`: rd = -(rs1*rs2) + rs3 = rs3 - rs1*rs2 (single-precision).
pub fn fnmsub_s(rd: FReg, rs1: FReg, rs2: FReg, rs3: FReg) u32 {
    return fma4Type(0b1001011, 0b00, rd, rs1, rs2, rs3);
}

/// `fnmsub.d rd, rs1, rs2, rs3`: rd = -(rs1*rs2) + rs3 = rs3 - rs1*rs2 (double-precision).
pub fn fnmsub_d(rd: FReg, rs1: FReg, rs2: FReg, rs3: FReg) u32 {
    return fma4Type(0b1001011, 0b01, rd, rs1, rs2, rs3);
}

/// `fcvt.s.w rd, rs1` (signed 32-bit integer -> single float, dynamic rounding).
pub fn fcvt_s_w(rd: FReg, rs1: Reg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b111) << 12) | (num(rs1) << 15) | (@as(u32, 0b1101000) << 25);
}

/// `fcvt.w.s rd, rs1` (single float -> signed 32-bit integer, round-toward-zero).
pub fn fcvt_w_s(rd: Reg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (num(rd) << 7) | (@as(u32, 0b001) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b1100000) << 25);
}

/// `fcvt.d.w rd, rs1` (signed 32-bit integer -> double float, exact, rm ignored).
pub fn fcvt_d_w(rd: FReg, rs1: Reg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b111) << 12) | (num(rs1) << 15) | (@as(u32, 0b1101001) << 25);
}

/// `fcvt.w.d rd, rs1` (double float -> signed 32-bit integer, round-toward-zero).
pub fn fcvt_w_d(rd: Reg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (num(rd) << 7) | (@as(u32, 0b001) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b1100001) << 25);
}

/// `fcvt.d.s rd, rs1` (single float -> double float, exact, rm ignored). Used by the f16 -> f64
/// convert: an f16 is held as its exact f32 widening, so widening that f32 to f64 is lossless.
pub fn fcvt_d_s(rd: FReg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b000) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b0100001) << 25);
}

/// `fcvt.s.d rd, rs1` (double float -> single float, dynamic rounding). Used by the f64 -> f16
/// convert: reduce to f32 first (one round), then the software routine rounds f32 -> f16.
pub fn fcvt_s_d(rd: FReg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b111) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b00001) << 20) | (@as(u32, 0b0100000) << 25);
}

/// `fmv.s rd, rs` (single-precision register move, via `fsgnj.s rd, rs, rs`).
pub fn fmv_s(rd: FReg, rs: FReg) u32 {
    return fpRType(0b0010000, 0b000, rd, rs, rs);
}

/// `fmv.d rd, rs` (double-precision register move).
pub fn fmv_d(rd: FReg, rs: FReg) u32 {
    return fpRType(0b0010001, 0b000, rd, rs, rs);
}

// The Zfh extension gives real IEEE-754 binary16 (f16) instructions, so a Zfh-capable model can
// hold an f16 natively in a float register (NaN-boxed into the low 16 bits) instead of emulating
// it as its f32 widening. Every OP-FP form reuses `fpRType`/`fpCmp` with the half fmt: the format
// field sits in funct7 bits [26:25] (00 = single, 01 = double, 10 = half), so each half funct7 is
// its single-precision sibling with bit 25 set. The loads/stores are new width-1 (`funct3 = 001`)
// LOAD-FP / STORE-FP forms. All encoders below were validated byte-for-byte against
// `riscv64-unknown-elf-as -march=rv64gc_zfh` (golden words in the unit test at the bottom of this
// file). Only reached when a caller threads `ModelCaps.zfh = true`; the default emulation path
// (see isel.zig) never emits any of these, so it stays byte-identical.

/// `flh rd, imm(rs1)` (load half-precision float, `rs1` is an integer base). LOAD-FP `funct3 = 001`
/// (width 1 = half), sibling of `flw` (010) and `fld` (011).
pub fn flh(rd: FReg, rs1: Reg, imm: i12) u32 {
    return @as(u32, 0b0000111) | (fnum(rd) << 7) | (0b001 << 12) | (num(rs1) << 15) | (@as(u32, @as(u12, @bitCast(imm))) << 20);
}

/// `fsh rs2, imm(rs1)` (store half-precision float, `rs1` is an integer base). STORE-FP width 1.
pub fn fsh(rs2: FReg, rs1: Reg, imm: i12) u32 {
    const bits: u12 = @bitCast(imm);
    return @as(u32, 0b0100111) | ((@as(u32, bits) & 0x1f) << 7) | (0b001 << 12) | (num(rs1) << 15) | (fnum(rs2) << 20) | (((@as(u32, bits) >> 5) & 0x7f) << 25);
}

/// `fadd.h rd, rs1, rs2` (half-precision add, single-rounded, dynamic rounding).
pub fn fadd_h(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0000010, 0b111, rd, rs1, rs2);
}

/// `fsub.h rd, rs1, rs2`
pub fn fsub_h(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0000110, 0b111, rd, rs1, rs2);
}

/// `fmul.h rd, rs1, rs2`
pub fn fmul_h(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0001010, 0b111, rd, rs1, rs2);
}

/// `fdiv.h rd, rs1, rs2`
pub fn fdiv_h(rd: FReg, rs1: FReg, rs2: FReg) u32 {
    return fpRType(0b0001110, 0b111, rd, rs1, rs2);
}

/// `fsqrt.h rd, rs1` (half-precision square root, dynamic rounding). rs2 selector = 0.
pub fn fsqrt_h(rd: FReg, rs1: FReg) u32 {
    return fpRType(0b0101110, 0b111, rd, rs1, @enumFromInt(0));
}

/// `feq.h rd, rs1, rs2` (set if equal, integer result). Half sibling of `feq.s`.
pub fn feq_h(rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return fpCmp(0b1010010, 0b010, rd, rs1, rs2);
}

/// `flt.h rd, rs1, rs2` (set if less-than)
pub fn flt_h(rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return fpCmp(0b1010010, 0b001, rd, rs1, rs2);
}

/// `fle.h rd, rs1, rs2` (set if less-or-equal)
pub fn fle_h(rd: Reg, rs1: FReg, rs2: FReg) u32 {
    return fpCmp(0b1010010, 0b000, rd, rs1, rs2);
}

/// `fcvt.s.h rd, rs1` (half -> single, always exact so the rounding mode is ignored; the assembler
/// emits rm = 000, matched here). rs2 selector = 00010 (source is half).
pub fn fcvt_s_h(rd: FReg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b000) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b00010) << 20) | (@as(u32, 0b0100000) << 25);
}

/// `fcvt.h.s rd, rs1` (single -> half, rounds; dynamic rounding). fmt = half, rs2 selector = 0 (S).
pub fn fcvt_h_s(rd: FReg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b111) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b00000) << 20) | (@as(u32, 0b0100010) << 25);
}

/// `fcvt.d.h rd, rs1` (half -> double, always exact, rm ignored). rs2 selector = 00010 (half).
pub fn fcvt_d_h(rd: FReg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b000) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b00010) << 20) | (@as(u32, 0b0100001) << 25);
}

/// `fcvt.h.d rd, rs1` (double -> half, rounds; dynamic rounding). fmt = half, rs2 selector = 00001 (D).
pub fn fcvt_h_d(rd: FReg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b111) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b00001) << 20) | (@as(u32, 0b0100010) << 25);
}

/// `fcvt.w.h rd, rs1` (half -> signed 32-bit integer, round-toward-zero). fmt = half, rs2 = 0 (W).
pub fn fcvt_w_h(rd: Reg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (num(rd) << 7) | (@as(u32, 0b001) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b00000) << 20) | (@as(u32, 0b1100010) << 25);
}

/// `fcvt.wu.h rd, rs1` (half -> unsigned 32-bit integer, round-toward-zero). rs2 selector = 00001 (WU).
pub fn fcvt_wu_h(rd: Reg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (num(rd) << 7) | (@as(u32, 0b001) << 12) | (fnum(rs1) << 15) | (@as(u32, 0b00001) << 20) | (@as(u32, 0b1100010) << 25);
}

/// `fcvt.h.w rd, rs1` (signed 32-bit integer -> half, rounds; dynamic rounding). rs2 = 0 (W source).
pub fn fcvt_h_w(rd: FReg, rs1: Reg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b111) << 12) | (num(rs1) << 15) | (@as(u32, 0b00000) << 20) | (@as(u32, 0b1101010) << 25);
}

/// `fcvt.h.wu rd, rs1` (unsigned 32-bit integer -> half, rounds; dynamic rounding). rs2 = 00001 (WU).
pub fn fcvt_h_wu(rd: FReg, rs1: Reg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (@as(u32, 0b111) << 12) | (num(rs1) << 15) | (@as(u32, 0b00001) << 20) | (@as(u32, 0b1101010) << 25);
}

/// `fmv.h rd, rs` (half-precision register move, via `fsgnj.h rd, rs, rs`).
pub fn fmv_h(rd: FReg, rs: FReg) u32 {
    return fpRType(0b0010010, 0b000, rd, rs, rs);
}

/// `fmv.x.h rd, rs1` (move a half float register's low 16 bits, sign-extended, into an integer
/// register, no convert). Half sibling of `fmv.x.w` (funct7 1110000 -> 1110010).
pub fn fmv_x_h(rd: Reg, rs1: FReg) u32 {
    return @as(u32, 0b1010011) | (num(rd) << 7) | (fnum(rs1) << 15) | (@as(u32, 0b1110010) << 25);
}

/// `fmv.h.x rd, rs1` (move an integer register's low 16 bits into a half float register, NaN-boxing
/// it into the register's low half, no convert). Half sibling of `fmv.w.x` (funct7 1111000 -> 1111010).
pub fn fmv_h_x(rd: FReg, rs1: Reg) u32 {
    return @as(u32, 0b1010011) | (fnum(rd) << 7) | (num(rs1) << 15) | (@as(u32, 0b1111010) << 25);
}

// et-soc VPU (CORE-ET Erbium) packed-single, 8-lane f32. Custom opcodes, decoded
// directly by the Minion frontend from ordinary 32-bit instruction words (see
// core-et/rtl/inc/instructions.vh). NOT RVV: fixed 8 lanes, no vtype, predicated by
// an M0..M7 mask register bank instead of a vector length. No emulator or silicon
// runs these here, so every encoder is validated in the test block below against the
// RTL `casex` match pattern (care mask + match bits), not by execution.

/// PS arithmetic: R-type, opcode 0x7B (1111011). `funct3` is the rounding mode for
/// add/sub/mul/div (dynamic = 0b111). `FADD_PS`/`FSUB_PS`/`FMUL_PS`/`FDIV_PS`,
/// instructions.vh:211-214.
fn vpuPsRType(funct7: u7, funct3: u3, fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return @as(u32, 0b1111011) |
        (fnum(fd) << 7) |
        (@as(u32, funct3) << 12) |
        (fnum(fs1) << 15) |
        (fnum(fs2) << 20) |
        (@as(u32, funct7) << 25);
}

/// `fadd.ps fd, fs1, fs2` (8-lane f32 add, dynamic rounding). instructions.vh:211.
pub fn fadd_ps(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPsRType(0b0000000, 0b111, fd, fs1, fs2);
}

/// `fsub.ps fd, fs1, fs2`. instructions.vh:212.
pub fn fsub_ps(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPsRType(0b0000100, 0b111, fd, fs1, fs2);
}

/// `fmul.ps fd, fs1, fs2`. instructions.vh:213.
pub fn fmul_ps(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPsRType(0b0001000, 0b111, fd, fs1, fs2);
}

/// `fdiv.ps fd, fs1, fs2`. instructions.vh:214.
pub fn fdiv_ps(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPsRType(0b0001100, 0b111, fd, fs1, fs2);
}

/// PS fused multiply-add: R4-type, opcode 0x5B (1011011). `sel` (bits [26:25]) picks
/// the variant: 00=madd, 01=msub, 10=nmsub, 11=nmadd (instructions.vh:220-223); `fs3`
/// (the accumulate operand) sits at bits [31:27]. Unlike scalar RV32F, which spends
/// four opcodes on this, the VPU packs all four variants into one opcode via `sel`.
fn vpuPsFma(sel: u2, funct3: u3, fd: FReg, fs1: FReg, fs2: FReg, fs3: FReg) u32 {
    return @as(u32, 0b1011011) |
        (fnum(fd) << 7) |
        (@as(u32, funct3) << 12) |
        (fnum(fs1) << 15) |
        (fnum(fs2) << 20) |
        (@as(u32, sel) << 25) |
        (fnum(fs3) << 27);
}

/// `fmadd.ps fd, fs1, fs2, fs3` (`fd = fs1*fs2 + fs3`, per lane, dynamic rounding).
/// `FMADD_PS`, instructions.vh:220 (sel = 00).
pub fn fmadd_ps(fd: FReg, fs1: FReg, fs2: FReg, fs3: FReg) u32 {
    return vpuPsFma(0b00, 0b111, fd, fs1, fs2, fs3);
}

/// `flw.ps fd, imm(rs1)`: 256-bit unit load of all 8 lanes from `imm(rs1)` into `fd`.
/// `FLW_PS`, opcode 0x0B (0001011), funct3=010, I-type, instructions.vh:194.
pub fn flw_ps(fd: FReg, rs1: Reg, imm: i12) u32 {
    return @as(u32, 0b0001011) |
        (fnum(fd) << 7) |
        (@as(u32, 0b010) << 12) |
        (num(rs1) << 15) |
        (@as(u32, @as(u12, @bitCast(imm))) << 20);
}

/// `fsw.ps fs, imm(rs1)`: 256-bit store of all 8 lanes of `fs` to `imm(rs1)`. `FSW_PS`,
/// opcode 0x0B, funct3=110, S-type (the stored register sits in the standard rs2 slot,
/// like the scalar `fsw` above), instructions.vh:195.
pub fn fsw_ps(fs: FReg, rs1: Reg, imm: i12) u32 {
    const bits: u12 = @bitCast(imm);
    return @as(u32, 0b0001011) |
        ((@as(u32, bits) & 0x1f) << 7) |
        (@as(u32, 0b110) << 12) |
        (num(rs1) << 15) |
        (fnum(fs) << 20) |
        (((@as(u32, bits) >> 5) & 0x7f) << 25);
}

/// `fbcx.ps fd, rs1`: broadcast the low 32 bits of integer register `rs1` into all 8
/// lanes of `fd` (mask-controlled). `FBCX_PS`, opcode 0x0B, funct3=011,
/// instructions.vh:197.
pub fn fbcx_ps(fd: FReg, rs1: Reg) u32 {
    return @as(u32, 0b0001011) |
        (fnum(fd) << 7) |
        (@as(u32, 0b011) << 12) |
        (num(rs1) << 15);
}

/// `fmvs.x.ps rd, fs1, index`: move lane `index` (0..7) of packed-single register `fs1`
/// into integer register `rd`, sign-filling the upper 32 bits. `FMVS_X_PS`, opcode
/// 0x7B, instructions.vh:236 (`111000000XXXXXXXX010XXXXX1111011`).
///
/// RTL-DERIVED FIELD PLACEMENT (the feasibility report flagged this field as
/// spec-approximate; verified here against `vpu_decoder.v:122` and `vpu_ctrl.v`).
/// `FMVS_X_PS`'s decode row sets `ren1=Y, ren2=N, ren3=N` with no swap, so
/// `vpu_ctrl.v` reads the source through `id_ra1 = id_core_inst_int[VPU_INST_REN1_RA_SEL]`,
/// i.e. `fs1` is the standard rs1 slot, bits [19:15] (NOT the rs2 slot, which the
/// feasibility report's draft encoder used). The lane `index` lives in the low 3 bits
/// of the rs2 slot, bits [22:20]: the casex's fixed `111000000` prefix (bits [31:23])
/// is funct7 [31:25] = 0b1110000 plus the top two rs2 bits [24:23] hardwired to 0. A
/// `u3` index keeps those top two bits at 0 by construction.
pub fn fmvs_x_ps(rd: Reg, fs1: FReg, index: u3) u32 {
    return @as(u32, 0b1111011) |
        (num(rd) << 7) |
        (@as(u32, 0b010) << 12) |
        (fnum(fs1) << 15) |
        (@as(u32, index) << 20) |
        (@as(u32, 0b1110000) << 25);
}

/// `mov.m.x md, xs, imm8`: write VPU mask register `md` (0..7, one of the eight lane
/// masks) to `(xs[7:0] | imm8)`. `MOV_M_X`, opcode 0x7B, instructions.vh:312
/// (`0101011XXXXXXXXXXXXX00XXX1111011`). To set M0 = 0xFF for full unmasked 8-lane
/// ops (the one-time preamble analogous to RVV's `vsetivli`), call
/// `mov_m_x(0, .x0, 0xFF)`.
///
/// RTL-DERIVED FIELD PLACEMENT (the feasibility report flagged the md/imm8 split as
/// spec-approximate; verified here against `vpu_mask.v` and `intpipe_decode.v`).
/// `vpu_mask.v`'s F2 stage computes `f2_regmask_wdata = f2_in1[7:0] | f2_imm[7:0]` for
/// this op: `f2_in1` is the bypassed value of `xs`, read via the standard rs1 slot
/// bits [19:15] (`MOV_M_X`'s decode row sets `fromint=Y`, and
/// `intpipe_decode.v:417` reads it through `A1_RS1`); `f2_imm` is
/// `{inst[24:20], inst[14:12]}` (the `maskop` case of `vpu_ctrl.v`'s imm mux), an
/// 8-bit immediate split across the rs2 slot (high 5 bits) and the rm/funct3 slot
/// (low 3 bits). The destination mask register `md` sits in the low 3 bits of the
/// standard rd slot, bits [9:7]; the casex's `...XXX00XXX...` hardwires bits [11:10]
/// to 0, which a `u3` for `md` satisfies by construction.
pub fn mov_m_x(md: u3, xs: Reg, imm8: u8) u32 {
    return @as(u32, 0b1111011) |
        (@as(u32, md) << 7) |
        ((@as(u32, imm8) & 0x7) << 12) |
        (num(xs) << 15) |
        ((@as(u32, imm8) >> 3) << 20) |
        (@as(u32, 0b0101011) << 25);
}

// et-soc pi (packed-integer), 8-lane int32 SIMD - the integer sibling of the PS
// (packed-single f32) unit above. Same opcode families (0x7B for reg-reg/unary ops,
// 0x5F for immediate broadcast), but funct3 here is a REAL operand selector, not a
// rounding mode as it is for PS: funct7[31:25] picks the op family, funct3[14:12]
// picks the variant within it. All MATCH/MASK constants below are authoritative,
// extracted from binutils esperanto-opc.h (not RTL-derived like the PS lane-extract
// and mask-write ops above). No emulator or silicon runs these here, so correctness
// is checked only against esperanto-opc.h's match/mask pairs in the tests below.

/// pi reg-reg R-type: opcode 0x7B (1111011). Same field layout as `vpuPsRType`
/// above, but `funct3` is a real operand selector here (e.g. signed vs. unsigned,
/// or which shift/compare), never a rounding mode.
fn vpuPiRType(funct7: u7, funct3: u3, fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return @as(u32, 0b1111011) |
        (fnum(fd) << 7) |
        (@as(u32, funct3) << 12) |
        (fnum(fs1) << 15) |
        (fnum(fs2) << 20) |
        (@as(u32, funct7) << 25);
}

/// `fadd.pi fd, fs1, fs2` (8-lane int32 add). MATCH 0x0600007b.
pub fn fadd_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0000011, 0b000, fd, fs1, fs2);
}

/// `fsub.pi fd, fs1, fs2`. MATCH 0x0e00007b.
pub fn fsub_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0000111, 0b000, fd, fs1, fs2);
}

/// `fmul.pi fd, fs1, fs2` (low 32 bits of the per-lane product). MATCH 0x1600007b.
pub fn fmul_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0001011, 0b000, fd, fs1, fs2);
}

/// `fmulh.pi fd, fs1, fs2` (high 32 bits of the signed per-lane product). MATCH
/// 0x1600107b.
pub fn fmulh_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0001011, 0b001, fd, fs1, fs2);
}

/// `fmulhu.pi fd, fs1, fs2` (high 32 bits of the unsigned per-lane product). MATCH
/// 0x1600207b.
pub fn fmulhu_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0001011, 0b010, fd, fs1, fs2);
}

/// `fmin.pi fd, fs1, fs2` (signed per-lane minimum). MATCH 0x2e00007b.
pub fn fmin_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0010111, 0b000, fd, fs1, fs2);
}

/// `fminu.pi fd, fs1, fs2` (unsigned per-lane minimum). MATCH 0x2e00207b.
pub fn fminu_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0010111, 0b010, fd, fs1, fs2);
}

/// `fmax.pi fd, fs1, fs2` (signed per-lane maximum). MATCH 0x2e00107b.
pub fn fmax_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0010111, 0b001, fd, fs1, fs2);
}

/// `fmaxu.pi fd, fs1, fs2` (unsigned per-lane maximum). MATCH 0x2e00307b.
pub fn fmaxu_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0010111, 0b011, fd, fs1, fs2);
}

/// `fsll.pi fd, fs1, fs2` (per-lane shift left logical, shift count carried in
/// `fs2`, one packed-int vector register per Vulcan's 8-lane int32 layout). MATCH
/// 0x0600107b.
pub fn fsll_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0000011, 0b001, fd, fs1, fs2);
}

/// `fsrl.pi fd, fs1, fs2` (per-lane shift right logical, shift count in `fs2`).
/// MATCH 0x0600507b.
pub fn fsrl_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0000011, 0b101, fd, fs1, fs2);
}

/// `fsra.pi fd, fs1, fs2` (per-lane shift right arithmetic, shift count in `fs2`).
/// MATCH 0x0e00507b.
pub fn fsra_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0000111, 0b101, fd, fs1, fs2);
}

/// `fand.pi fd, fs1, fs2` (per-lane bitwise and). MATCH 0x0600707b.
pub fn fand_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0000011, 0b111, fd, fs1, fs2);
}

/// `for.pi fd, fs1, fs2` (per-lane bitwise or). MATCH 0x0600607b.
pub fn for_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0000011, 0b110, fd, fs1, fs2);
}

/// `fxor.pi fd, fs1, fs2` (per-lane bitwise xor). MATCH 0x0600407b.
pub fn fxor_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b0000011, 0b100, fd, fs1, fs2);
}

/// `feq.pi fd, fs1, fs2` (per-lane compare; result is -1/0 written into `fd`'s
/// lanes, NOT a GPR - unlike the scalar `feq.s` above, which writes an integer
/// register). MATCH 0xa600207b.
pub fn feq_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b1010011, 0b010, fd, fs1, fs2);
}

/// `flt.pi fd, fs1, fs2` (signed per-lane less-than, result -1/0 per lane). MATCH
/// 0xa600107b.
pub fn flt_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b1010011, 0b001, fd, fs1, fs2);
}

/// `fltu.pi fd, fs1, fs2` (unsigned per-lane less-than, result -1/0 per lane).
/// MATCH 0xa600307b.
pub fn fltu_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b1010011, 0b011, fd, fs1, fs2);
}

/// `fle.pi fd, fs1, fs2` (signed per-lane less-or-equal, result -1/0 per lane).
/// MATCH 0xa600007b.
pub fn fle_pi(fd: FReg, fs1: FReg, fs2: FReg) u32 {
    return vpuPiRType(0b1010011, 0b000, fd, fs1, fs2);
}

/// pi saturate: unary R-type, opcode 0x7B. Unlike the reg-reg ops above, the field
/// at bits [24:20] (a register operand's `rs2` slot in `vpuPiRType`) is NOT a
/// register here - it's a fixed sub-selector distinguishing the signed and
/// unsigned clamp variants. MASK 0xfff0707f (funct7 + the whole sub-selector field
/// + funct3 + opcode all care; `rd`/`rs1`, the real operands, don't).
fn vpuPiSat(funct7: u7, funct3: u3, sel: u5, fd: FReg, fs1: FReg) u32 {
    return @as(u32, 0b1111011) |
        (fnum(fd) << 7) |
        (@as(u32, funct3) << 12) |
        (fnum(fs1) << 15) |
        (@as(u32, sel) << 20) |
        (@as(u32, funct7) << 25);
}

/// `fsat8.pi fd, fs1` (clamp each int32 lane into the signed int8 range). MATCH
/// 0x0600307b (sub-selector = 0).
pub fn fsat8_pi(fd: FReg, fs1: FReg) u32 {
    return vpuPiSat(0b0000011, 0b011, 0, fd, fs1);
}

/// `fsatu8.pi fd, fs1` (clamp each int32 lane into the unsigned uint8 range).
/// MATCH 0x0610307b - sub-selector = 1 is the *only* bit distinguishing this from
/// `fsat8_pi` (bit 20 of the word).
pub fn fsatu8_pi(fd: FReg, fs1: FReg) u32 {
    return vpuPiSat(0b0000011, 0b011, 1, fd, fs1);
}

/// pi convert: unary R-type, opcode 0x7B, `rs2` field fixed at 0 (mirrors the
/// scalar `fcvt.s.w`/`fcvt.w.s` convention of a fixed selector in that slot).
/// `funct3` is fixed at 0: the authoritative MATCH words below pin it there, even
/// though MASK 0xfff0007f treats it as don't-care for op classification (unlike
/// the scalar fcvt encoders above, which spend funct3 on a real rounding mode).
fn vpuPiCvt(funct7: u7, fd: FReg, fs1: FReg) u32 {
    return @as(u32, 0b1111011) |
        (fnum(fd) << 7) |
        (fnum(fs1) << 15) |
        (@as(u32, funct7) << 25);
}

/// `fcvt.ps.pw fd, fs1` (8 int32 lanes -> 8 f32 lanes). MATCH 0xd000007b. Same
/// funct7 (0b1101000) as the scalar `fcvt_s_w` above.
pub fn fcvt_ps_pw(fd: FReg, fs1: FReg) u32 {
    return vpuPiCvt(0b1101000, fd, fs1);
}

/// `fcvt.pw.ps fd, fs1` (8 f32 lanes -> 8 int32 lanes, round-toward-zero
/// truncation). MATCH 0xc000007b. Same funct7 (0b1100000) as the scalar
/// `fcvt_w_s` above.
pub fn fcvt_pw_ps(fd: FReg, fs1: FReg) u32 {
    return vpuPiCvt(0b1100000, fd, fs1);
}

/// `fbci.pi fd, imm`: broadcast an immediate into all 8 int32 lanes of `fd`.
/// Opcode 0x5F - only the opcode is authoritative here (MATCH base 0x0000005f,
/// MASK 0x0000007f per esperanto-opc.h; no funct3/rd/imm field placement is
/// confirmed against it).
///
/// UNVERIFIED FIELD LAYOUT: this encoder makes the conservative, `lui`-shaped
/// guess that the immediate is a 20-bit U-type-style hi field at bits [31:12]
/// (`imm << 12`), by analogy with `lui`/`auipc` elsewhere in this file. This has
/// NOT been cross-checked against esperanto-opc.h's immediate layout the way every
/// other encoder in this section has been, so treat it as a placeholder pending
/// real confirmation. If exact-immediate correctness matters now, prefer the
/// confirmed register-broadcast path instead: `fbcx_ps(fd, rs1)` above (GPR into
/// all 8 lanes) already works for pi vectors too, since a broadcast is type-
/// agnostic bit motion, not an arithmetic op.
pub fn fbci_pi(fd: FReg, imm: u20) u32 {
    return @as(u32, 0b1011111) | (fnum(fd) << 7) | (@as(u32, imm) << 12);
}

/// Encode an R-type instruction: `funct7 | rs2 | rs1 | funct3 | rd | opcode`.
fn rType(opcode: u7, funct3: u3, funct7: u7, rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return @as(u32, opcode) |
        (num(rd) << 7) |
        (@as(u32, funct3) << 12) |
        (num(rs1) << 15) |
        (num(rs2) << 20) |
        (@as(u32, funct7) << 25);
}

/// Encode an I-type instruction: `imm[11:0] | rs1 | funct3 | rd | opcode`.
fn iType(opcode: u7, funct3: u3, rd: Reg, rs1: Reg, imm: i12) u32 {
    return @as(u32, opcode) |
        (num(rd) << 7) |
        (@as(u32, funct3) << 12) |
        (num(rs1) << 15) |
        (@as(u32, @as(u12, @bitCast(imm))) << 20);
}

/// `add rd, rs1, rs2`
pub fn add(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b000, 0b0000000, rd, rs1, rs2);
}

/// `sub rd, rs1, rs2`
pub fn sub(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b000, 0b0100000, rd, rs1, rs2);
}

/// `mul rd, rs1, rs2` (M extension)
pub fn mul(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b000, 0b0000001, rd, rs1, rs2);
}

/// `div rd, rs1, rs2` (signed, M extension)
pub fn div(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b100, 0b0000001, rd, rs1, rs2);
}

/// `divu rd, rs1, rs2` (unsigned, M extension)
pub fn divu(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b101, 0b0000001, rd, rs1, rs2);
}

/// `rem rd, rs1, rs2` (signed remainder, M extension)
pub fn rem(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b110, 0b0000001, rd, rs1, rs2);
}

/// `remu rd, rs1, rs2` (unsigned remainder, M extension)
pub fn remu(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b111, 0b0000001, rd, rs1, rs2);
}

/// `mulh rd, rs1, rs2` (high 64 bits of the signed 64x64 product, M extension)
pub fn mulh(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b001, 0b0000001, rd, rs1, rs2);
}

/// `mulhu rd, rs1, rs2` (high 64 bits of the unsigned 64x64 product, M extension)
pub fn mulhu(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b011, 0b0000001, rd, rs1, rs2);
}

/// `mulhsu rd, rs1, rs2` (high 64 of a signed rs1 by unsigned rs2 product, M extension)
pub fn mulhsu(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b010, 0b0000001, rd, rs1, rs2);
}

/// `and rd, rs1, rs2`
pub fn and_(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b111, 0b0000000, rd, rs1, rs2);
}

/// `or rd, rs1, rs2`
pub fn or_(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b110, 0b0000000, rd, rs1, rs2);
}

/// `xor rd, rs1, rs2`
pub fn xor_(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b100, 0b0000000, rd, rs1, rs2);
}

/// `sll rd, rs1, rs2` (shift left logical)
pub fn sll(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b001, 0b0000000, rd, rs1, rs2);
}

/// `srl rd, rs1, rs2` (shift right logical)
pub fn srl(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b101, 0b0000000, rd, rs1, rs2);
}

/// `sra rd, rs1, rs2` (shift right arithmetic)
pub fn sra(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b101, 0b0100000, rd, rs1, rs2);
}

/// `sh1add rd, rs1, rs2` (Zba): rd = rs2 + (rs1 << 1).
pub fn sh1add(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b010, 0b0010000, rd, rs1, rs2);
}

/// `sh2add rd, rs1, rs2` (Zba): rd = rs2 + (rs1 << 2).
pub fn sh2add(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b100, 0b0010000, rd, rs1, rs2);
}

/// `sh3add rd, rs1, rs2` (Zba): rd = rs2 + (rs1 << 3).
pub fn sh3add(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b110, 0b0010000, rd, rs1, rs2);
}

/// `slt rd, rs1, rs2` (set if rs1 < rs2, signed)
pub fn slt(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b010, 0b0000000, rd, rs1, rs2);
}

/// `sltu rd, rs1, rs2` (set if rs1 < rs2, unsigned)
pub fn sltu(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return rType(0b0110011, 0b011, 0b0000000, rd, rs1, rs2);
}

/// `xori rd, rs1, imm`
pub fn xori(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0010011, 0b100, rd, rs1, imm);
}

/// `sltiu rd, rs1, imm` (set if rs1 < imm, unsigned)
pub fn sltiu(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0010011, 0b011, rd, rs1, imm);
}

/// `addi rd, rs1, imm`
pub fn addi(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0010011, 0b000, rd, rs1, imm);
}

/// `nop` (`addi x0, x0, 0`, a no-operation), used to pad a loop header up to the
/// fetch-alignment boundary.
pub fn nop() u32 {
    return 0x00000013;
}

/// `andi rd, rs1, imm`
pub fn andi(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0010011, 0b111, rd, rs1, imm);
}

/// `ori rd, rs1, imm`
pub fn ori(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0010011, 0b110, rd, rs1, imm);
}

/// `slli rd, rs1, shamt` (logical left shift by a 6-bit immediate, RV64)
pub fn slli(rd: Reg, rs1: Reg, shamt: u6) u32 {
    return iType(0b0010011, 0b001, rd, rs1, @intCast(shamt));
}

/// `srli rd, rs1, shamt` (logical right shift by a 6-bit immediate, RV64)
pub fn srli(rd: Reg, rs1: Reg, shamt: u6) u32 {
    return iType(0b0010011, 0b101, rd, rs1, @intCast(shamt));
}

/// `srai rd, rs1, shamt` (arithmetic right shift by a 6-bit immediate, RV64)
pub fn srai(rd: Reg, rs1: Reg, shamt: u6) u32 {
    return iType(0b0010011, 0b101, rd, rs1, @as(i12, 0x400) | @as(i12, shamt));
}

/// `jalr rd, rs1, imm` (jump and link register, `ret` is `jalr x0, ra, 0`).
pub fn jalr(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b1100111, 0b000, rd, rs1, imm);
}

/// `csrrs rd, csr, rs1` (atomic read CSR into rd, then set the bits in rs1).
pub fn csrrs(rd: Reg, csr: u12, rs1: Reg) u32 {
    return @as(u32, 0b1110011) | (num(rd) << 7) | (@as(u32, 0b010) << 12) | (num(rs1) << 15) | (@as(u32, csr) << 20);
}

/// `csrrw rd, csr, rs1` (atomic read CSR into rd, then write rs1 into it). SYSTEM
/// opcode 0x73, funct3 001, csr in bits [31:20], rs1 in bits [19:15], rd in bits
/// [11:7]. `csrw csr, rs1` is the assembler pseudo for `csrrw x0, csr, rs1` (the
/// read is discarded): see `csrw` below.
pub fn csrrw(rd: Reg, csr: u12, rs1: Reg) u32 {
    return @as(u32, 0b1110011) | (num(rd) << 7) | (@as(u32, 0b001) << 12) | (num(rs1) << 15) | (@as(u32, csr) << 20);
}

/// `csrw csr, rs1` (write rs1 into csr, discarding the old value): `csrrw x0, csr, rs1`.
pub fn csrw(csr: u12, rs1: Reg) u32 {
    return csrrw(.x0, csr, rs1);
}

/// `rev8 rd, rs1` (reverse byte order of a 64-bit register, Zbb extension).
pub fn rev8(rd: Reg, rs1: Reg) u32 {
    return iType(0b0010011, 0b101, rd, rs1, 0x6b8);
}

/// `lui rd, imm` (load upper immediate: `imm` fills bits 31:12, lower 12 zero).
pub fn lui(rd: Reg, imm: u20) u32 {
    return @as(u32, 0b0110111) | (num(rd) << 7) | (@as(u32, imm) << 12);
}

/// `auipc rd, imm` (add upper immediate to PC: `rd = pc + (imm << 12)`,
/// sign-extended). The PC-relative companion to `lui`.
pub fn auipc(rd: Reg, imm: u20) u32 {
    return @as(u32, 0b0010111) | (num(rd) << 7) | (@as(u32, imm) << 12);
}

/// Encode a B-type branch. `imm` is a byte offset (even). Its bits are scattered
/// across the instruction word.
fn bType(opcode: u7, funct3: u3, rs1: Reg, rs2: Reg, imm: i13) u32 {
    const u: u32 = @as(u13, @bitCast(imm));
    return @as(u32, opcode) |
        (((u >> 11) & 1) << 7) | // imm[11]
        (((u >> 1) & 0xf) << 8) | // imm[4:1]
        (@as(u32, funct3) << 12) |
        (num(rs1) << 15) |
        (num(rs2) << 20) |
        (((u >> 5) & 0x3f) << 25) | // imm[10:5]
        (((u >> 12) & 1) << 31); // imm[12]
}

/// `beq rs1, rs2, offset` (branch if equal)
pub fn beq(rs1: Reg, rs2: Reg, imm: i13) u32 {
    return bType(0b1100011, 0b000, rs1, rs2, imm);
}

/// `bne rs1, rs2, offset` (branch if not equal)
pub fn bne(rs1: Reg, rs2: Reg, imm: i13) u32 {
    return bType(0b1100011, 0b001, rs1, rs2, imm);
}

/// `blt rs1, rs2, offset` (branch if signed less-than)
pub fn blt(rs1: Reg, rs2: Reg, imm: i13) u32 {
    return bType(0b1100011, 0b100, rs1, rs2, imm);
}

/// `bge rs1, rs2, offset` (branch if signed greater-or-equal)
pub fn bge(rs1: Reg, rs2: Reg, imm: i13) u32 {
    return bType(0b1100011, 0b101, rs1, rs2, imm);
}

/// `bltu rs1, rs2, offset` (branch if unsigned less-than)
pub fn bltu(rs1: Reg, rs2: Reg, imm: i13) u32 {
    return bType(0b1100011, 0b110, rs1, rs2, imm);
}

/// `bgeu rs1, rs2, offset` (branch if unsigned greater-or-equal)
pub fn bgeu(rs1: Reg, rs2: Reg, imm: i13) u32 {
    return bType(0b1100011, 0b111, rs1, rs2, imm);
}

/// `jal rd, offset` (jump and link). The 21-bit immediate is scattered too.
pub fn jal(rd: Reg, imm: i21) u32 {
    const u: u32 = @as(u21, @bitCast(imm));
    return @as(u32, 0b1101111) |
        (num(rd) << 7) |
        (((u >> 12) & 0xff) << 12) | // imm[19:12]
        (((u >> 11) & 1) << 20) | // imm[11]
        (((u >> 1) & 0x3ff) << 21) | // imm[10:1]
        (((u >> 20) & 1) << 31); // imm[20]
}

/// Encode an S-type instruction (stores): the immediate is split across two
/// fields, `imm[11:5]` and `imm[4:0]`.
fn sType(opcode: u7, funct3: u3, rs1: Reg, rs2: Reg, imm: i12) u32 {
    const bits: u12 = @bitCast(imm);
    const imm_4_0: u32 = bits & 0x1f;
    const imm_11_5: u32 = (@as(u32, bits) >> 5) & 0x7f;
    return @as(u32, opcode) |
        (imm_4_0 << 7) |
        (@as(u32, funct3) << 12) |
        (num(rs1) << 15) |
        (num(rs2) << 20) |
        (imm_11_5 << 25);
}

/// `lb rd, imm(rs1)` (load 8-bit, sign-extended)
pub fn lb(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0000011, 0b000, rd, rs1, imm);
}

/// `lh rd, imm(rs1)` (load 16-bit, sign-extended)
pub fn lh(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0000011, 0b001, rd, rs1, imm);
}

/// `lbu rd, imm(rs1)` (load 8-bit, zero-extended)
pub fn lbu(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0000011, 0b100, rd, rs1, imm);
}

/// `lhu rd, imm(rs1)` (load 16-bit, zero-extended)
pub fn lhu(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0000011, 0b101, rd, rs1, imm);
}

/// `lw rd, imm(rs1)` (load 32-bit, sign-extended)
pub fn lw(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0000011, 0b010, rd, rs1, imm);
}

/// `lwu rd, imm(rs1)` (load 32-bit, zero-extended, RV64)
pub fn lwu(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0000011, 0b110, rd, rs1, imm);
}

/// `ld rd, imm(rs1)` (load 64-bit)
pub fn ld(rd: Reg, rs1: Reg, imm: i12) u32 {
    return iType(0b0000011, 0b011, rd, rs1, imm);
}

/// `sb rs2, imm(rs1)` (store 8-bit)
pub fn sb(rs2: Reg, rs1: Reg, imm: i12) u32 {
    return sType(0b0100011, 0b000, rs1, rs2, imm);
}

/// `sh rs2, imm(rs1)` (store 16-bit)
pub fn sh(rs2: Reg, rs1: Reg, imm: i12) u32 {
    return sType(0b0100011, 0b001, rs1, rs2, imm);
}

/// `sw rs2, imm(rs1)` (store 32-bit)
pub fn sw(rs2: Reg, rs1: Reg, imm: i12) u32 {
    return sType(0b0100011, 0b010, rs1, rs2, imm);
}

/// `sd rs2, imm(rs1)` (store 64-bit)
pub fn sd(rs2: Reg, rs1: Reg, imm: i12) u32 {
    return sType(0b0100011, 0b011, rs1, rs2, imm);
}

/// `prefetch.r rs1, offset` (Zicbop: prefetch for read). OP-IMM major opcode with `ori`'s funct3
/// (0b110) and rd hardwired to x0, so on hardware without Zicbop this decodes as a plain
/// `ori x0, rs1, imm`: a no-op, since x0 discards the result. Zicbop repurposes the 12-bit
/// I-immediate: imm[11:5] (bits [31:25], the same position as an R-type funct7) is the byte
/// offset from rs1, and imm[4:0] (bits [24:20], the same position as an R-type rs2) selects the
/// variant: 00000 = prefetch.i, 00001 = prefetch.r, 00011 = prefetch.w. Because the variant
/// selector occupies imm[4:0], the offset itself can only ever supply imm[11:5]: it must be a
/// multiple of 32 (one cache block), and the caller is asserted to have already rounded it.
pub fn prefetch_r(rs1: Reg, offset: i12) u32 {
    const off_u: u12 = @bitCast(offset);
    std.debug.assert(off_u & 0x1f == 0); // low 5 bits are the variant field, not part of the offset
    const variant_r: u32 = 0b00001;
    return @as(u32, 0b0010011) |
        (@as(u32, 0) << 7) | // rd = x0, fixed by the ISA (Zicbop prefetch instructions never write back)
        (@as(u32, 0b110) << 12) | // funct3 = ori's, the ORI-shaped no-op fallback
        (num(rs1) << 15) |
        (variant_r << 20) |
        ((@as(u32, off_u) >> 5) << 25); // offset[11:5]
}

// et-soc tensor/matmul unit (CORE-ET Erbium): a pure CSR-write protocol, NOT a set
// of tensor instructions. Every op is `csrw <csr>, rs1` writing a packed 64-bit
// descriptor (built by the packers below); some ops also take a second packed value
// through x31 (stride, and for tensor_load a stride|id pair). Bit layouts below are
// copied field-for-field from the golden software header
// `et-platform/et-common-libs/include/etsoc/isa/tensors.h` (the `tensor_fma`,
// `tensor_load`, and `tensor_store` inline helpers), and the tensor_fma layout is
// additionally cross-checked bit-by-bit against a hand-written matmul kernel
// (`/tmp/etsoc-build/matmul/mm.s`) that ran correctly on sw-sysemu: its descriptor
// `0x0008800000002001` (fp32, A 2x2, B 2x4, K=2, scp_loc_a=0, scp_loc_b=2,
// first_pass=1) decodes to exactly bits {51, 47, 13, 0} set, matching this layout's
// a_rows-1=1@51, a_cols-1=1@47, scp_b=2@[19:12] (bit13), first_pass=1@0 - see the
// test below. Unlike the packers, whose callers pass human-meaningful counts (rows,
// not rows-1), tensors.h's own C wrappers take the pre-adjusted field values
// directly; the assembly comments in mm.s spell out the distinction ("arows_f=1
// (->2)"). tensor_store's layout has no executed proof in mm.s (the kernel reads C
// back with `fsw.ps` instead of `tensor_store`), so it is encoded verbatim from
// tensors.h only - flagged for Task 3's sw-sysemu differential test to confirm.

/// TensorFMA element type (tensors.h `opcode` field, bits [3:1] of the descriptor).
/// fp32 stays fp32; fp16 inputs accumulate to fp32; int8 inputs accumulate to int32.
pub const TensorType = enum(u3) {
    fp32 = 0,
    fp16 = 1,
    int8 = 3,
};

/// tensor_fma CSR number.
pub const CSR_TENSOR_FMA: u12 = 0x801;
/// tensor_wait CSR number (stall until the tensor op named by the written event id completes).
pub const CSR_TENSOR_WAIT: u12 = 0x830;
/// tensor_load CSR number (x31 carries stride|id).
pub const CSR_TENSOR_LOAD: u12 = 0x83f;
/// tensor_store CSR number (x31 carries the memory row stride).
pub const CSR_TENSOR_STORE: u12 = 0x87f;
/// tensor_quant CSR number: runs the requantize epilogue (int32 -> scale ->
/// saturate -> pack to int8) in place on the vector register file.
pub const CSR_TENSOR_QUANT: u12 = 0x806;
/// mcache_control CSR number: the L1 scratchpad enable state machine. Must be
/// sequenced `csrw 0x7e0, 1` then `csrw 0x7e0, 3` - writing 3 directly is a silent
/// no-op, and any subsequent tensor op then raises tensor_error bit 4 (SCP disabled).
pub const CSR_MCACHE_CONTROL: u12 = 0x7e0;

/// tensor_wait event id: tensor_load with id=0 has completed.
pub const TENSOR_WAIT_LOAD_0: u64 = 0;
/// tensor_wait event id: tensor_load with id=1 has completed.
pub const TENSOR_WAIT_LOAD_1: u64 = 1;
/// tensor_wait event id: all previously issued tensor_fma instructions are complete.
pub const TENSOR_WAIT_FMA: u64 = 7;
/// tensor_wait event id: all previously issued tensor_store instructions are complete.
pub const TENSOR_WAIT_STORE: u64 = 8;
/// tensor_wait event id: all previously issued tensor_quant instructions are complete.
pub const TENSOR_WAIT_QUANT: u64 = 10;

/// Pack the tensor_fma descriptor (written via `csrw CSR_TENSOR_FMA, rs1`): `C =
/// A*B` (optionally accumulating into the existing C), where A is `a_rows x a_cols`
/// and B is `a_cols x b_cols`, both read from L1 scratchpad lines, and C is written
/// to the vector register file (f0.. two f-regs per row). Layout (tensors.h
/// `tensor_fma`, all fields relative to bit 0): 56:55 = b_cols/4 - 1, 54:51 =
/// a_rows - 1, 50:47 = a_cols - 1 (the K dimension), 46:43 = aoffset (A's starting
/// column), bit23 = tenc_loc (0 = C lives in the vector regfile - the only mode this
/// packer exposes; hardcode tenb_loc=0 too, matching the L1-scratchpad-only path
/// used here), bit22 = tenb_unsigned (B), bit21 = tena_unsigned (A) - the hardware
/// reads bit22 as the B operand's sign flag and bit21 as A's (sw-sysemu ub/ua), bit20 = tenb_loc
/// (hardcoded 0 = SCP), 19:12 = scp_loc_b, 11:4 = scp_loc_a, 3:1 = type, bit0 =
/// first_pass (1 = fresh C, 0 = accumulate into existing C).
pub fn packTensorFma(
    type_: TensorType,
    a_rows: u5,
    a_cols: u5,
    b_cols: u5,
    aoffset: u4,
    scp_a: u8,
    scp_b: u8,
    tenc_in_mem: bool,
    tena_unsigned: bool,
    tenb_unsigned: bool,
    first_pass: bool,
) u64 {
    const b_cols_field: u2 = @intCast(b_cols / 4 - 1);
    const a_rows_field: u4 = @intCast(a_rows - 1);
    const a_cols_field: u4 = @intCast(a_cols - 1);
    return (@as(u64, b_cols_field) << 55) |
        (@as(u64, a_rows_field) << 51) |
        (@as(u64, a_cols_field) << 47) |
        (@as(u64, aoffset) << 43) |
        (@as(u64, @intFromBool(tenc_in_mem)) << 23) |
        (@as(u64, @intFromBool(tenb_unsigned)) << 22) | // hardware bit22 = B's sign flag (ub)
        (@as(u64, @intFromBool(tena_unsigned)) << 21) | // hardware bit21 = A's sign flag (ua)
        (@as(u64, scp_b) << 12) |
        (@as(u64, scp_a) << 4) |
        (@as(u64, @intFromEnum(type_)) << 1) |
        @as(u64, @intFromBool(first_pass));
}

/// Pack the tensor_load descriptor (written via `csrw CSR_TENSOR_LOAD, rs1`, with
/// x31 set beforehand by `tensorLoadX31`): load `num_lines` 64-byte lines from
/// `addr` (bypassing the L1 cache) into L1 scratchpad starting at line
/// `dst_scp_line`. Layout (tensors.h `tensor_load`, transformation/offset fields
/// not exposed by this packer and left 0, matching the plain-load path used here):
/// bit63 = use_tmask, bit62 = use_coop, 61:59 = transformation (0), 58:53 =
/// dst_start (the SCP line), bit52 = use_tenb, then `addr` masked to
/// `addr & 0xFFFFFFFFFFC0` (64-byte aligned, and also clears the top 16 bits so it
/// can never collide with the descriptor's own high fields), with `num_lines - 1`
/// OR'd into bits [3:0].
pub fn packTensorLoad(dst_scp_line: u6, num_lines: u5, addr: u64, use_coop: bool, use_tmask: bool, use_tenb: bool) u64 {
    const num_lines_field: u4 = @intCast(num_lines - 1);
    const addr_masked = addr & 0xFFFFFFFFFFC0;
    return (@as(u64, @intFromBool(use_tmask)) << 63) |
        (@as(u64, @intFromBool(use_coop)) << 62) |
        (@as(u64, dst_scp_line) << 53) |
        (@as(u64, @intFromBool(use_tenb)) << 52) |
        addr_masked |
        @as(u64, num_lines_field);
}

/// Pack the x31 value that must be set immediately before `csrw CSR_TENSOR_LOAD,
/// rs1`: the memory stride between consecutive loaded lines, plus a 1-bit id
/// (matched by the corresponding `tensor_wait` event, `TENSOR_WAIT_LOAD_0/1`).
/// tensors.h masks `stride` with the narrower `0xFFFFFFFFFFC0` (also clearing the
/// top 16 bits); this clears only the low 6 (64-byte alignment), which is
/// equivalent for any realistic (non-canonical-high-bit) stride value.
pub fn tensorLoadX31(stride: u64, id: u1) u64 {
    return (stride & ~@as(u64, 0x3f)) | id;
}

/// Pack the tensor_store descriptor (written via `csrw CSR_TENSOR_STORE, rs1`,
/// with x31 set beforehand by `tensorStoreX31`): read `rows` rows of TenC (the FMA
/// accumulator) from the vector register file, starting at register `start_reg`
/// and stepping `reg_stride` registers per row, and write them to `addr`. Per
/// tensors.h `tensor_store` (NOT `tensor_store_scp`, which instead moves L1
/// scratchpad contents rather than the vector regfile): the hardware field stores
/// `rows - 1` and writes that-plus-one rows, hence this packer's `rows` param is
/// the actual row count. Layout: 63:62 = reg_stride, 61:57 = start_reg, 56:55 =
/// cols, then `addr` masked to `addr & 0xFFFFFFFFFFF0` (tensors.h uses this
/// literal 48-bit-wide, 16-byte-aligned mask for the CSR-embedded addr - narrower
/// than tensor_load's 64-byte mask, copied verbatim since it is the golden
/// source), 54:51 = rows - 1, 50:49 = coop_store, 3:0 = reserved (0).
///
/// UNCONFIRMED: no proof kernel exercises tensor_store (mm.s reads TenC back with
/// plain `fsw.ps` instead), so this layout is taken from tensors.h only and is not
/// cross-checked against an executed descriptor. Flagged for Task 3's sw-sysemu
/// differential test.
pub fn packTensorStore(start_reg: u5, reg_stride: u2, cols: u2, rows: u5, addr: u64, coop_store: u2) u64 {
    const rows_field: u4 = @intCast(rows - 1);
    const addr_masked = addr & 0xFFFFFFFFFFF0;
    return (@as(u64, reg_stride) << 62) |
        (@as(u64, start_reg) << 57) |
        (@as(u64, cols) << 55) |
        addr_masked |
        (@as(u64, rows_field) << 51) |
        (@as(u64, coop_store) << 49);
}

/// Pack the x31 value that must be set immediately before `csrw CSR_TENSOR_STORE,
/// rs1`: the memory stride between consecutive stored rows. tensors.h masks with
/// the literal (and, relative to tensor_load/tensor_store's own addr mask,
/// inconsistent - likely a copy/paste artifact in the golden header) 44-bit-wide
/// `0xFFFFFFFFFF0`, copied verbatim here rather than "cleaned up", since matching
/// the golden source bit-for-bit is safer pending Task 3's sysemu confirmation.
pub fn tensorStoreX31(stride: u64) u64 {
    return stride & 0xFFFFFFFFFF0;
}

/// tensor_quant transform funct code (tensors.h `tensor_quant`, 4 bits per chain
/// slot). `last` (0) terminates the chain: any slot at or after the first `last`
/// is ignored by the hardware, so trailing slots should be filled with `last`.
pub const QuantTransform = enum(u4) {
    last = 0,
    i32_to_f32 = 1,
    f32_to_i32 = 2,
    i32_relu = 3,
    i32_add_row = 4,
    i32_add_col = 5,
    fp32_mul_row = 6,
    fp32_mul_col = 7,
    satint8 = 8,
    satuint8 = 9,
    pack_128b = 10,
};

/// Pack the tensor_quant descriptor (written via `csrw CSR_TENSOR_QUANT, rs1`):
/// run the `transforms` chain, in order (`transforms[0]` first), on TenC starting
/// at register `start_reg`, requantizing/packing it in place on the vector
/// register file. Layout (tensors.h `tensor_quant`, all fields relative to bit 0):
/// 61:57 = start_reg, 56:55 = col_field (acols = (col_field+1)*4, i.e. 4/8/12/16
/// output cols), 54:51 = row_field (arows = row_field+1, i.e. 1..16 rows), 50:45 =
/// scp_loc (the L1 scratchpad line of the first scale/bias vector), 44:40 unused
/// (left 0), 39:0 = the ten transform slots, 4 bits each, trans0 in bits 3:0 up to
/// trans9 in bits 39:36.
pub fn packTensorQuant(start_reg: u5, col_field: u2, row_field: u4, scp_loc: u6, transforms: [10]QuantTransform) u64 {
    var transforms_field: u64 = 0;
    for (transforms, 0..) |transform, i| {
        transforms_field |= @as(u64, @intFromEnum(transform)) << @intCast(i * 4);
    }
    return (@as(u64, start_reg) << 57) |
        (@as(u64, col_field) << 55) |
        (@as(u64, row_field) << 51) |
        (@as(u64, scp_loc) << 45) |
        transforms_field;
}

test "nop encoding" {
    // addi x0, x0, 0: opcode 0b0010011, all other fields zero.
    try std.testing.expectEqual(@as(u32, 0x00000013), nop());
}

test "encodes immediate arithmetic and shifts" {
    try std.testing.expectEqual(@as(u32, 0x00517093), andi(.x1, .x2, 5)); // andi x1,x2,5
    try std.testing.expectEqual(@as(u32, 0x00516093), ori(.x1, .x2, 5)); // ori x1,x2,5
    try std.testing.expectEqual(@as(u32, 0x00511093), slli(.x1, .x2, 5)); // slli x1,x2,5
    try std.testing.expectEqual(@as(u32, 0x00515093), srli(.x1, .x2, 5)); // srli x1,x2,5
    try std.testing.expectEqual(@as(u32, 0x40515093), srai(.x1, .x2, 5)); // srai x1,x2,5
}

test "encodes sub-word loads and stores" {
    // funct3 selects width: lb=000, lh=001, lbu=100, lhu=101. sb=000, sh=001.
    try std.testing.expectEqual(@as(u32, 0x00010083), lb(.x1, .x2, 0)); // lb x1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00011083), lh(.x1, .x2, 0)); // lh x1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00014083), lbu(.x1, .x2, 0)); // lbu x1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00015083), lhu(.x1, .x2, 0)); // lhu x1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00110023), sb(.x1, .x2, 0)); // sb x1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00111023), sh(.x1, .x2, 0)); // sh x1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00016083), lwu(.x1, .x2, 0)); // lwu x1, 0(x2)
}

test "encodes R-type and I-type instructions" {
    // add x1, x2, x3  ->  0x003100b3
    try std.testing.expectEqual(@as(u32, 0x003100b3), add(.x1, .x2, .x3));
    // addi x1, x2, 5  ->  0x00510093
    try std.testing.expectEqual(@as(u32, 0x00510093), addi(.x1, .x2, 5));
}

test "encodes RVV vector ops" {
    try std.testing.expectEqual(@as(u32, 0xcd027057), vsetivli(.x0, 4, 0xD0)); // vsetivli x0, 4, e32, m1, ta, ma
    try std.testing.expectEqual(@as(u32, 0x02056087), vle32(.v1, .x10)); // vle32.v v1, (x10)
    try std.testing.expectEqual(@as(u32, 0x021111d7), vfadd_vv(.v3, .v1, .v2)); // vfadd.vv v3, v1, v2
}

test "encodes RVV vector FMA (vfmacc/vfmsac/vfnmsac, cross-checked against `zig cc`'s assembler)" {
    try std.testing.expectEqual(@as(u32, 0xb22091d7), vfmacc_vv(.v3, .v1, .v2)); // vfmacc.vv v3, v1, v2
    try std.testing.expectEqual(@as(u32, 0xba2091d7), vfmsac_vv(.v3, .v1, .v2)); // vfmsac.vv v3, v1, v2
    try std.testing.expectEqual(@as(u32, 0xbe2091d7), vfnmsac_vv(.v3, .v1, .v2)); // vfnmsac.vv v3, v1, v2
}

test "encodes single-precision float arithmetic" {
    // OP-FP, dynamic rounding (rm=111). fadd.s f1, f2, f3, etc.
    try std.testing.expectEqual(@as(u32, 0x003170d3), fadd_s(.f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x083170d3), fsub_s(.f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x103170d3), fmul_s(.f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x183170d3), fdiv_s(.f1, .f2, .f3));
}

test "encodes f16 helper float<->float converts (fcvt.d.s, fcvt.s.d)" {
    // Golden words are the standard RV64GC encodings (fcvt.d.s f0,f0 = 0x42000053,
    // fcvt.s.d f0,f0 = 0x40107053), used by the f16<->f64 convert paths.
    try std.testing.expectEqual(@as(u32, 0x42000053), fcvt_d_s(.f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x420080d3), fcvt_d_s(.f1, .f1));
    try std.testing.expectEqual(@as(u32, 0x40107053), fcvt_s_d(.f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x4010f0d3), fcvt_s_d(.f1, .f1));
}

test "encodes Zfh half-precision ops (golden words from riscv64-unknown-elf-as -march=rv64gc_zfh)" {
    // ft0 = f0, ft1 = f1, ft2 = f2, a0 = x10. Each golden word is the exact `as` output for the
    // matching mnemonic, so this pins the native Zfh path (only reached under `ModelCaps.zfh`).
    try std.testing.expectEqual(@as(u32, 0x00051007), flh(.f0, .x10, 0)); // flh ft0, 0(a0)
    try std.testing.expectEqual(@as(u32, 0x00051027), fsh(.f0, .x10, 0)); // fsh ft0, 0(a0)
    try std.testing.expectEqual(@as(u32, 0x0420f053), fadd_h(.f0, .f1, .f2)); // fadd.h ft0, ft1, ft2
    try std.testing.expectEqual(@as(u32, 0x0c20f053), fsub_h(.f0, .f1, .f2)); // fsub.h
    try std.testing.expectEqual(@as(u32, 0x1420f053), fmul_h(.f0, .f1, .f2)); // fmul.h
    try std.testing.expectEqual(@as(u32, 0x1c20f053), fdiv_h(.f0, .f1, .f2)); // fdiv.h
    try std.testing.expectEqual(@as(u32, 0x5c00f053), fsqrt_h(.f0, .f1)); // fsqrt.h ft0, ft1
    try std.testing.expectEqual(@as(u32, 0xa420a553), feq_h(.x10, .f1, .f2)); // feq.h a0, ft1, ft2
    try std.testing.expectEqual(@as(u32, 0xa4209553), flt_h(.x10, .f1, .f2)); // flt.h
    try std.testing.expectEqual(@as(u32, 0xa4208553), fle_h(.x10, .f1, .f2)); // fle.h
    try std.testing.expectEqual(@as(u32, 0x40208053), fcvt_s_h(.f0, .f1)); // fcvt.s.h ft0, ft1
    try std.testing.expectEqual(@as(u32, 0x4400f053), fcvt_h_s(.f0, .f1)); // fcvt.h.s
    try std.testing.expectEqual(@as(u32, 0x42208053), fcvt_d_h(.f0, .f1)); // fcvt.d.h
    try std.testing.expectEqual(@as(u32, 0x4410f053), fcvt_h_d(.f0, .f1)); // fcvt.h.d
    try std.testing.expectEqual(@as(u32, 0xc4009553), fcvt_w_h(.x10, .f1)); // fcvt.w.h a0, ft1, rtz
    try std.testing.expectEqual(@as(u32, 0xc4109553), fcvt_wu_h(.x10, .f1)); // fcvt.wu.h a0, ft1, rtz
    try std.testing.expectEqual(@as(u32, 0xd4057053), fcvt_h_w(.f0, .x10)); // fcvt.h.w ft0, a0
    try std.testing.expectEqual(@as(u32, 0xd4157053), fcvt_h_wu(.f0, .x10)); // fcvt.h.wu ft0, a0
    try std.testing.expectEqual(@as(u32, 0x24108053), fmv_h(.f0, .f1)); // fmv.h ft0, ft1 (fsgnj.h)
    try std.testing.expectEqual(@as(u32, 0xe4008553), fmv_x_h(.x10, .f1)); // fmv.x.h a0, ft1
    try std.testing.expectEqual(@as(u32, 0xf4050053), fmv_h_x(.f0, .x10)); // fmv.h.x ft0, a0
}

test "encodes float comparisons" {
    try std.testing.expectEqual(@as(u32, 0xa03120d3), feq_s(.x1, .f2, .f3)); // feq.s x1, f2, f3
    try std.testing.expectEqual(@as(u32, 0xa03110d3), flt_s(.x1, .f2, .f3)); // flt.s
    try std.testing.expectEqual(@as(u32, 0xa03100d3), fle_s(.x1, .f2, .f3)); // fle.s
    try std.testing.expectEqual(@as(u32, 0xa23120d3), feq_d(.x1, .f2, .f3)); // feq.d
}

test "encodes float loads and stores" {
    try std.testing.expectEqual(@as(u32, 0x00012087), flw(.f1, .x2, 0)); // flw f1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00013087), fld(.f1, .x2, 0)); // fld f1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00112027), fsw(.f1, .x2, 0)); // fsw f1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00113027), fsd(.f1, .x2, 0)); // fsd f1, 0(x2)
}

test "encodes double-precision float arithmetic" {
    try std.testing.expectEqual(@as(u32, 0x023170d3), fadd_d(.f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x0a3170d3), fsub_d(.f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x123170d3), fmul_d(.f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x1a3170d3), fdiv_d(.f1, .f2, .f3));
}

test "encodes fused multiply-add/sub (fmadd/fmsub/fnmsub, .s/.d)" {
    // rd=f0, rs1=f1, rs2=f2, rs3=f3: pins opcode + fmt + all four field positions at once
    // (rd/rs1/rs2/rs3 are all distinct registers, so a swapped field shows up immediately).
    try std.testing.expectEqual(@as(u32, 0x1820f043), fmadd_s(.f0, .f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x1a20f043), fmadd_d(.f0, .f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x1820f047), fmsub_s(.f0, .f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x1a20f047), fmsub_d(.f0, .f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x1820f04b), fnmsub_s(.f0, .f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x1a20f04b), fnmsub_d(.f0, .f1, .f2, .f3));

    // Cross-check against the golden word in "decodes RV64 word ops, FMA, and base pseudos"
    // (0x7ab577c3, verified there against llvm-objdump: fmadd.d f15, f10, f11, f15), so the
    // encoder and the independently-written disasm decoder agree on every field.
    try std.testing.expectEqual(@as(u32, 0x7ab577c3), fmadd_d(.f15, .f10, .f11, .f15));
}

test "encodes int<->float conversions" {
    try std.testing.expectEqual(@as(u32, 0xd00170d3), fcvt_s_w(.f1, .x2)); // fcvt.s.w f1, x2
    try std.testing.expectEqual(@as(u32, 0xc00110d3), fcvt_w_s(.x1, .f2)); // fcvt.w.s x1, f2
    try std.testing.expectEqual(@as(u32, 0xd20170d3), fcvt_d_w(.f1, .x2)); // fcvt.d.w f1, x2
    try std.testing.expectEqual(@as(u32, 0xc20110d3), fcvt_w_d(.x1, .f2)); // fcvt.w.d x1, f2
}

test "encodes divu, rem, remu" {
    try std.testing.expectEqual(@as(u32, 0x023150b3), divu(.x1, .x2, .x3)); // divu x1,x2,x3
    try std.testing.expectEqual(@as(u32, 0x023160b3), rem(.x1, .x2, .x3)); // rem x1,x2,x3
    try std.testing.expectEqual(@as(u32, 0x023170b3), remu(.x1, .x2, .x3)); // remu x1,x2,x3
}

test "encodes shifts" {
    try std.testing.expectEqual(@as(u32, 0x003110b3), sll(.x1, .x2, .x3)); // sll x1,x2,x3
    try std.testing.expectEqual(@as(u32, 0x003150b3), srl(.x1, .x2, .x3)); // srl x1,x2,x3
    try std.testing.expectEqual(@as(u32, 0x403150b3), sra(.x1, .x2, .x3)); // sra x1,x2,x3
}

test "encodes set-less-than and immediate forms" {
    try std.testing.expectEqual(@as(u32, 0x003120b3), slt(.x1, .x2, .x3)); // slt x1,x2,x3
    try std.testing.expectEqual(@as(u32, 0x003130b3), sltu(.x1, .x2, .x3)); // sltu x1,x2,x3
    try std.testing.expectEqual(@as(u32, 0x00114093), xori(.x1, .x2, 1)); // xori x1,x2,1
    try std.testing.expectEqual(@as(u32, 0x00113093), sltiu(.x1, .x2, 1)); // sltiu x1,x2,1
}

test "encodes J-type jal" {
    try std.testing.expectEqual(@as(u32, 0x0000006f), jal(.x0, 0)); // jal x0, 0
    try std.testing.expectEqual(@as(u32, 0x0080006f), jal(.x0, 8)); // jal x0, +8
}

test "encodes B-type branches" {
    try std.testing.expectEqual(@as(u32, 0x00000063), beq(.x0, .x0, 0)); // beq x0, x0, 0
    try std.testing.expectEqual(@as(u32, 0x00208463), beq(.x1, .x2, 8)); // beq x1, x2, +8
    try std.testing.expectEqual(@as(u32, 0x00209463), bne(.x1, .x2, 8)); // bne x1, x2, +8
    // Unsigned branches differ from beq only in funct3 (bits 14:12): bltu=110 -> |0x6000,
    // bgeu=111 -> |0x7000. Hand-derived from the validated beq(.x1,.x2,8)=0x00208463.
    try std.testing.expectEqual(@as(u32, 0x0020e463), bltu(.x1, .x2, 8)); // bltu x1, x2, +8
    try std.testing.expectEqual(@as(u32, 0x0020f463), bgeu(.x1, .x2, 8)); // bgeu x1, x2, +8
    // blt=100 -> |0x4000, bge=101 -> |0x5000, sanity for the signed forms too.
    try std.testing.expectEqual(@as(u32, 0x0020c463), blt(.x1, .x2, 8)); // blt x1, x2, +8
    try std.testing.expectEqual(@as(u32, 0x0020d463), bge(.x1, .x2, 8)); // bge x1, x2, +8
}

test "encodes rev8" {
    // rev8 x1, x2 (byte-reverse, Zbb)
    try std.testing.expectEqual(@as(u32, 0x6b815093), rev8(.x1, .x2));
}

test "encodes Zba sh-add family" {
    // Base encodings (rd/rs1/rs2 all x0): funct7 0b0010000, funct3 010/100/110, opcode 0x33.
    try std.testing.expectEqual(@as(u32, 0x20002033), sh1add(.x0, .x0, .x0));
    try std.testing.expectEqual(@as(u32, 0x20004033), sh2add(.x0, .x0, .x0));
    try std.testing.expectEqual(@as(u32, 0x20006033), sh3add(.x0, .x0, .x0));
    // With registers: sh2add x1, x2, x3 = base | rs2<<20 | rs1<<15 | rd<<7.
    try std.testing.expectEqual(@as(u32, 0x20004033 | (3 << 20) | (2 << 15) | (1 << 7)), sh2add(.x1, .x2, .x3));
}

test "encodes csrrs" {
    // csrrs x0, mstatus(0x300), x5
    try std.testing.expectEqual(@as(u32, 0x3002a073), csrrs(.x0, 0x300, .x5));
}

test "encodes lui" {
    // lui x1, 0x12345  ->  0x123450b7
    try std.testing.expectEqual(@as(u32, 0x123450b7), lui(.x1, 0x12345));
}

test "encodes loads and stores" {
    try std.testing.expectEqual(@as(u32, 0x00012083), lw(.x1, .x2, 0)); // lw x1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00013083), ld(.x1, .x2, 0)); // ld x1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00112023), sw(.x1, .x2, 0)); // sw x1, 0(x2)
    try std.testing.expectEqual(@as(u32, 0x00113023), sd(.x1, .x2, 0)); // sd x1, 0(x2)
}

test "encodes Zicbop prefetch.r, and it decodes ORI-shaped" {
    // prefetch.r x10, 0: opcode 0b0010011 (0x13), rd=x0, funct3=0b110 (<<12 = 0x6000), rs1=x10
    // (<<15 = 0x50000), variant field (imm[4:0]) = 00001 (<<20 = 0x100000), offset[11:5] = 0.
    // 0x13 + 0x6000 + 0x50000 + 0x100000 = 0x156013.
    try std.testing.expectEqual(@as(u32, 0x00156013), prefetch_r(.x10, 0));
    // A nonzero, 32-aligned offset (64 = 2 cache lines) only ever touches imm[11:5]: offset[11:5]
    // = 64 >> 5 = 2, so the top 7 bits of the immediate become 2 (<<25 = 0x04000000) on top of the
    // same base word.
    try std.testing.expectEqual(@as(u32, 0x04156013), prefetch_r(.x10, 64));

    // Structurally, every prefetch.r decodes as an ORI-shaped instruction (opcode 0x13, funct3
    // 0b110, rd 0) regardless of rs1/offset: this is exactly what makes it a harmless
    // `ori x0, rs1, imm` no-op on hardware without Zicbop.
    const w = prefetch_r(.x14, 96);
    try std.testing.expectEqual(@as(u32, 0x13), w & 0x7f); // opcode
    try std.testing.expectEqual(@as(u32, 0b110), (w >> 12) & 0x7); // funct3
    try std.testing.expectEqual(@as(u32, 0), (w >> 7) & 0x1f); // rd
    try std.testing.expectEqual(@as(u32, 0b00001), (w >> 20) & 0x1f); // rs2/variant field: prefetch.r
}

test "encodes jalr (function return)" {
    // ret == jalr x0, ra, 0  ->  0x00008067
    try std.testing.expectEqual(@as(u32, 0x00008067), jalr(.x0, .x1, 0));
}

test "encodes the arithmetic and bitwise R-type instructions" {
    try std.testing.expectEqual(@as(u32, 0x403100b3), sub(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x023100b3), mul(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x023140b3), div(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x023110b3), mulh(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x023120b3), mulhsu(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x023130b3), mulhu(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x003170b3), and_(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x003160b3), or_(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x003140b3), xor_(.x1, .x2, .x3));
}

// et-soc VPU (CORE-ET Erbium) packed-single tests. No emulator decodes these custom
// opcodes, so correctness is checked against the RTL `casex` match patterns in
// core-et/rtl/inc/instructions.vh (care mask = the pattern's fixed bits, match = their
// required values; the `X` bits are operand fields and are masked out), plus a couple
// of full-word encodings pinned by hand so operand field *placement* (not just the
// opcode) is validated too.

test "encodes et-soc VPU packed-single arithmetic (RTL-mask validated)" {
    // FADD_PS/FSUB_PS/FMUL_PS/FDIV_PS: instructions.vh:211-214, opcode 0x7B (1111011),
    // funct7 [31:25] discriminates the op. Care = funct7[31:25] | opcode[6:0].
    const care: u32 = 0xFE00007F;
    try std.testing.expectEqual(@as(u32, 0x0000007B), fadd_ps(.f3, .f9, .f21) & care);
    try std.testing.expectEqual(@as(u32, 0x0800007B), fsub_ps(.f3, .f9, .f21) & care);
    try std.testing.expectEqual(@as(u32, 0x1000007B), fmul_ps(.f3, .f9, .f21) & care);
    try std.testing.expectEqual(@as(u32, 0x1800007B), fdiv_ps(.f3, .f9, .f21) & care);

    // Exact full-word pin (operand field placement): fadd.ps f16, f17, f18, dynamic rm.
    // funct7=0000000 rs2=f18(10010) rs1=f17(10001) rm=111 rd=f16(10000) opcode=1111011.
    try std.testing.expectEqual(@as(u32, 0x0128F87B), fadd_ps(.f16, .f17, .f18));
}

test "encodes et-soc VPU packed-single FMA (RTL-mask validated)" {
    // FMADD_PS: instructions.vh:220, opcode 0x5B (1011011), sel = bits [26:25] = 00,
    // fs3 (the accumulate operand) = bits [31:27]. Care = sel[26:25] | opcode[6:0].
    const word = fmadd_ps(.f1, .f2, .f3, .f4);
    try std.testing.expectEqual(@as(u32, 0x0000005B), word & 0x0600007F);
    try std.testing.expectEqual(@as(u32, 0b00), (word >> 25) & 0b11); // sel bits [26:25]
    try std.testing.expectEqual(@as(u32, 4), (word >> 27) & 0x1F); // fs3=f4 at [31:27]

    // Exact full-word pin: fmadd.ps f1, f2, f3, f4, dynamic rm.
    try std.testing.expectEqual(@as(u32, 0x203170DB), word);
}

test "encodes et-soc VPU packed-single load/store/broadcast (RTL-mask validated)" {
    // FLW_PS/FSW_PS/FBCX_PS: instructions.vh:194,195,197, all opcode 0x0B (0001011),
    // discriminated by funct3 [14:12]. Care = funct3[14:12] | opcode[6:0].
    const care: u32 = 0x707F;
    try std.testing.expectEqual(@as(u32, 0x0000200B), flw_ps(.f7, .x14, 100) & care);
    try std.testing.expectEqual(@as(u32, 0x0000600B), fsw_ps(.f7, .x14, 100) & care);
    try std.testing.expectEqual(@as(u32, 0x0000300B), fbcx_ps(.f7, .x14) & care);

    // FSW_PS split immediate: imm[4:0] -> bits [11:7], imm[11:5] -> bits [31:25].
    // imm = 0x2A = 0b0000_0010_1010 -> imm[4:0] = 0b01010 (0xA), imm[11:5] = 0b0000001 (0x1).
    const sw_ps_word = fsw_ps(.f5, .x10, 0x2A);
    try std.testing.expectEqual(@as(u32, 0xA), (sw_ps_word >> 7) & 0x1F);
    try std.testing.expectEqual(@as(u32, 0x1), (sw_ps_word >> 25) & 0x7F);

    // Exact full-word pin: fsw.ps f5, 0x2A(x10).
    try std.testing.expectEqual(@as(u32, 0x0255650B), sw_ps_word);
}

test "encodes et-soc VPU lane extract and mask write (RTL-derived field positions)" {
    // FMVS_X_PS: instructions.vh:236. RTL-derived and verified against vpu_decoder.v:122
    // (ren1=Y reads fs1 through the standard rs1 slot [19:15]) and the casex's fixed
    // bits [24:23]=00 (the lane index lives in the remaining rs2-slot bits [22:20]).
    // Care = funct7[31:25] | rs2-top-2[24:23] | funct3[14:12] | opcode[6:0].
    const fmvs_care: u32 = 0xFF80707F;
    try std.testing.expectEqual(@as(u32, 0xE000207B), fmvs_x_ps(.x5, .f11, 0) & fmvs_care);
    // fs1 sits at the rs1 slot [19:15]; index sits at [22:20], not the rs1 slot (this
    // is the field swap the feasibility report's draft got backwards).
    const lane = fmvs_x_ps(.x5, .f11, 5);
    try std.testing.expectEqual(@as(u32, 11), (lane >> 15) & 0x1F); // fs1 = f11
    try std.testing.expectEqual(@as(u32, 5), (lane >> 20) & 0x7); // index = 5

    // MOV_M_X: instructions.vh:312. RTL-derived and verified against vpu_mask.v (F2
    // stage ORs xs's low 8 bits with imm8) and intpipe_decode.v:417 (A1_RS1 reads xs
    // through the standard rs1 slot [19:15]). Care = funct7[31:25] | rd-top-2[11:10] |
    // opcode[6:0].
    const mov_care: u32 = 0xFE000C7F;
    try std.testing.expectEqual(@as(u32, 0x5600007B), mov_m_x(0, .x0, 0) & mov_care);

    // M0 = 0xFF preamble (the full-8-lane-unmasked setup): md=0, xs=x0 contributes 0,
    // imm8=0xFF supplies the mask bits directly. Reconstruct imm8 from its RTL split
    // ({inst[24:20], inst[14:12]}) and confirm it round-trips to 0xFF.
    const m0_setup = mov_m_x(0, .x0, 0xFF);
    const imm8_lo: u32 = (m0_setup >> 12) & 0x7;
    const imm8_hi: u32 = (m0_setup >> 20) & 0x1F;
    try std.testing.expectEqual(@as(u32, 0xFF), imm8_lo | (imm8_hi << 3));
}

// et-soc pi (packed-integer) tests. MATCH constants are extracted from binutils
// esperanto-opc.h (authoritative, not RTL-derived): every all-f0 form below must equal
// its MATCH exactly, and a distinct-register form pins rd/rs1/rs2 field placement.

test "encodes et-soc pi reg-reg arithmetic (esperanto-opc.h MATCH-verified)" {
    try std.testing.expectEqual(@as(u32, 0x0600007b), fadd_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x0e00007b), fsub_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x1600007b), fmul_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x1600107b), fmulh_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x1600207b), fmulhu_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x2e00007b), fmin_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x2e00207b), fminu_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x2e00107b), fmax_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x2e00307b), fmaxu_pi(.f0, .f0, .f0));

    // Distinct-register form: fadd.pi f3, f1, f2 pins rd=f3[11:7], rs1=f1[19:15],
    // rs2=f2[24:20] all at once (a swapped field shows up immediately).
    const w = fadd_pi(.f3, .f1, .f2);
    try std.testing.expectEqual(@as(u32, 3), (w >> 7) & 0x1f); // rd = f3
    try std.testing.expectEqual(@as(u32, 1), (w >> 15) & 0x1f); // rs1 = f1
    try std.testing.expectEqual(@as(u32, 2), (w >> 20) & 0x1f); // rs2 = f2
}

test "encodes et-soc pi shifts (esperanto-opc.h MATCH-verified)" {
    try std.testing.expectEqual(@as(u32, 0x0600107b), fsll_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x0600507b), fsrl_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x0e00507b), fsra_pi(.f0, .f0, .f0));

    const w = fsrl_pi(.f3, .f1, .f2);
    try std.testing.expectEqual(@as(u32, 3), (w >> 7) & 0x1f); // rd = f3
    try std.testing.expectEqual(@as(u32, 1), (w >> 15) & 0x1f); // rs1 = f1
    try std.testing.expectEqual(@as(u32, 2), (w >> 20) & 0x1f); // rs2 = f2 (shift count)
}

test "encodes et-soc pi bitwise ops (esperanto-opc.h MATCH-verified)" {
    try std.testing.expectEqual(@as(u32, 0x0600707b), fand_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x0600607b), for_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x0600407b), fxor_pi(.f0, .f0, .f0));

    const w = fand_pi(.f3, .f1, .f2);
    try std.testing.expectEqual(@as(u32, 3), (w >> 7) & 0x1f); // rd = f3
    try std.testing.expectEqual(@as(u32, 1), (w >> 15) & 0x1f); // rs1 = f1
    try std.testing.expectEqual(@as(u32, 2), (w >> 20) & 0x1f); // rs2 = f2
}

test "encodes et-soc pi compares (esperanto-opc.h MATCH-verified)" {
    try std.testing.expectEqual(@as(u32, 0xa600207b), feq_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0xa600107b), flt_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0xa600307b), fltu_pi(.f0, .f0, .f0));
    try std.testing.expectEqual(@as(u32, 0xa600007b), fle_pi(.f0, .f0, .f0));

    const w = feq_pi(.f3, .f1, .f2);
    try std.testing.expectEqual(@as(u32, 3), (w >> 7) & 0x1f); // rd = f3
    try std.testing.expectEqual(@as(u32, 1), (w >> 15) & 0x1f); // rs1 = f1
    try std.testing.expectEqual(@as(u32, 2), (w >> 20) & 0x1f); // rs2 = f2
}

test "encodes et-soc pi saturate (esperanto-opc.h MATCH-verified, sub-selector bit)" {
    try std.testing.expectEqual(@as(u32, 0x0600307b), fsat8_pi(.f0, .f0));
    try std.testing.expectEqual(@as(u32, 0x0610307b), fsatu8_pi(.f0, .f0));

    // fsatu8_pi differs from fsat8_pi by exactly the sub-selector bit at [20].
    try std.testing.expectEqual(@as(u32, 0), (fsat8_pi(.f0, .f0) >> 20) & 1);
    try std.testing.expectEqual(@as(u32, 1), (fsatu8_pi(.f0, .f0) >> 20) & 1);

    const w = fsat8_pi(.f3, .f1);
    try std.testing.expectEqual(@as(u32, 3), (w >> 7) & 0x1f); // rd = f3
    try std.testing.expectEqual(@as(u32, 1), (w >> 15) & 0x1f); // rs1 = f1
}

test "encodes et-soc pi convert (esperanto-opc.h MATCH-verified)" {
    try std.testing.expectEqual(@as(u32, 0xd000007b), fcvt_ps_pw(.f0, .f0));
    try std.testing.expectEqual(@as(u32, 0xc000007b), fcvt_pw_ps(.f0, .f0));

    const w = fcvt_ps_pw(.f3, .f1);
    try std.testing.expectEqual(@as(u32, 3), (w >> 7) & 0x1f); // rd = f3
    try std.testing.expectEqual(@as(u32, 1), (w >> 15) & 0x1f); // rs1 = f1
}

test "encodes et-soc pi broadcast immediate (opcode-only MATCH; imm layout is a guess)" {
    // Only the opcode (bits [6:0]) is authoritative per esperanto-opc.h; confirm the
    // all-zero form hits it and that rd/imm land where the U-type-style guess puts them.
    try std.testing.expectEqual(@as(u32, 0x0000005f), fbci_pi(.f0, 0));
    try std.testing.expectEqual(@as(u32, 0x5f), fbci_pi(.f0, 0) & 0x7f);

    const w = fbci_pi(.f3, 0xABCDE);
    try std.testing.expectEqual(@as(u32, 0x5f), w & 0x7f); // opcode
    try std.testing.expectEqual(@as(u32, 3), (w >> 7) & 0x1f); // rd = f3
    try std.testing.expectEqual(@as(u32, 0xABCDE), w >> 12); // imm[31:12]
}

// et-soc tensor CSR-write protocol tests.

test "encodes csrrw" {
    // csrrw x0, 0x801(tensor_fma), x5: opcode 0x73, rd=x0, funct3=001 (0x1000),
    // rs1=x5 (5<<15=0x28000), csr=0x801<<20=0x80100000. Sum = 0x80129073.
    try std.testing.expectEqual(@as(u32, 0x80129073), csrrw(.x0, 0x801, .x5));
    // csrw is the x0-destination pseudo for csrrw and must produce the same word.
    try std.testing.expectEqual(csrrw(.x0, 0x801, .x5), csrw(0x801, .x5));
    // A nonzero rd is still encoded (csrrw's read side): csrrw x3, 0x300(mstatus), x1.
    // rd=x3(3<<7=0x180), funct3=0x1000, rs1=x1(1<<15=0x8000), csr=0x300<<20=0x30000000.
    try std.testing.expectEqual(@as(u32, 0x300091F3), csrrw(.x3, 0x300, .x1));
}

test "packTensorFma matches the proven matmul kernel's descriptor" {
    // /tmp/etsoc-build/matmul/mm.s: fp32, A(2x2) @ B(2x4), K=2, A/B in SCP (A at
    // line 0, B at line 2), C fresh (first_pass=1) -> proven correct on sw-sysemu.
    try std.testing.expectEqual(
        @as(u64, 0x0008800000002001),
        packTensorFma(.fp32, 2, 2, 4, 0, 0, 2, false, false, false, true),
    );
}

test "packTensorFma field placement" {
    // Toggle one field at a time off the proof-value baseline and check only the
    // expected bits move.
    const base = packTensorFma(.fp32, 2, 2, 4, 0, 0, 2, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 1), base & 1); // first_pass at bit0
    try std.testing.expectEqual(@as(u64, 0), (base >> 1) & 0x7); // type = fp32 = 0

    const int8_first_pass_off = packTensorFma(.int8, 2, 2, 4, 0, 0, 2, false, false, false, false);
    try std.testing.expectEqual(@as(u64, 0), int8_first_pass_off & 1); // first_pass cleared
    try std.testing.expectEqual(@as(u64, 3), (int8_first_pass_off >> 1) & 0x7); // int8 = 3

    // scp_a occupies bits [11:4]; scp_b occupies bits [19:12] (confirmed above by
    // the proof value's scp_b=2 landing at bit 13).
    const with_scp_a = packTensorFma(.fp32, 2, 2, 4, 0, 0xAB, 0, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 0xAB), (with_scp_a >> 4) & 0xFF);
    const with_scp_b = packTensorFma(.fp32, 2, 2, 4, 0, 0, 0xCD, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 0xCD), (with_scp_b >> 12) & 0xFF);

    // a_rows-1 at [54:51], a_cols(K)-1 at [50:47], b_cols/4-1 at [56:55], aoffset at [46:43].
    const dims = packTensorFma(.fp32, 16, 16, 16, 9, 0, 0, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 15), (dims >> 51) & 0xF); // a_rows-1 = 15
    try std.testing.expectEqual(@as(u64, 15), (dims >> 47) & 0xF); // a_cols-1 = 15
    try std.testing.expectEqual(@as(u64, 3), (dims >> 55) & 0x3); // 16/4 - 1 = 3
    try std.testing.expectEqual(@as(u64, 9), (dims >> 43) & 0xF); // aoffset

    // tenc_in_mem/tena_unsigned/tenb_unsigned each own a single bit: 23, 22, 21.
    const flags = packTensorFma(.fp32, 2, 2, 4, 0, 0, 0, true, true, true, true);
    try std.testing.expectEqual(@as(u64, 1), (flags >> 23) & 1);
    try std.testing.expectEqual(@as(u64, 1), (flags >> 22) & 1);
    try std.testing.expectEqual(@as(u64, 1), (flags >> 21) & 1);
}

test "assert TensorType enum values" {
    try std.testing.expectEqual(@as(u3, 0), @intFromEnum(TensorType.fp32));
    try std.testing.expectEqual(@as(u3, 3), @intFromEnum(TensorType.int8));
    try std.testing.expectEqual(@as(u3, 1), @intFromEnum(TensorType.fp16));
}

test "packTensorLoad field placement" {
    // dst_scp_line at [58:53] (6 bits), use_tenb at bit52, use_coop at bit62,
    // use_tmask at bit63, (num_lines-1) at [3:0], addr masked to 64-byte alignment.
    const d = packTensorLoad(5, 4, 0x1000, true, true, true);
    try std.testing.expectEqual(@as(u64, 5), (d >> 53) & 0x3F); // dst_scp_line
    try std.testing.expectEqual(@as(u64, 1), (d >> 52) & 1); // use_tenb
    try std.testing.expectEqual(@as(u64, 1), (d >> 62) & 1); // use_coop
    try std.testing.expectEqual(@as(u64, 1), (d >> 63) & 1); // use_tmask
    try std.testing.expectEqual(@as(u64, 3), d & 0xF); // num_lines - 1 = 3
    try std.testing.expectEqual(@as(u64, 0x1000), d & 0xFFFFFFFFFFC0); // addr, 64B aligned

    // Cross-check against mm.s: `la t0, matdata; ori t0, t0, 3; csrw 0x83f, t0` with
    // no use_tmask/use_coop/use_tenb/dst_start set - only num_lines-1=3 lands below
    // the (64-byte-aligned) address bits, matching a plain `addr | 3`.
    const proof_addr: u64 = 0x2000; // stand-in 64-byte-aligned address
    const proof = packTensorLoad(0, 4, proof_addr, false, false, false);
    try std.testing.expectEqual(proof_addr | 3, proof);
}

test "tensorLoadX31 field placement" {
    // mm.s: `li x31, 0x40` (stride=0x40, id=0).
    try std.testing.expectEqual(@as(u64, 0x40), tensorLoadX31(0x40, 0));
    try std.testing.expectEqual(@as(u64, 0x41), tensorLoadX31(0x40, 1));
    // Low 6 bits of stride are discarded (64-byte-line granularity), id lives there instead.
    try std.testing.expectEqual(@as(u64, 0x80), tensorLoadX31(0xBF, 0));
}

test "packTensorStore field placement" {
    // reg_stride at [63:62], start_reg at [61:57], cols at [56:55], rows-1 at
    // [54:51], coop_store at [50:49], addr masked into the low bits (16B aligned).
    const s = packTensorStore(3, 2, 1, 2, 0x4000, 1);
    try std.testing.expectEqual(@as(u64, 2), (s >> 62) & 0x3); // reg_stride
    try std.testing.expectEqual(@as(u64, 3), (s >> 57) & 0x1F); // start_reg
    try std.testing.expectEqual(@as(u64, 1), (s >> 55) & 0x3); // cols
    try std.testing.expectEqual(@as(u64, 1), (s >> 51) & 0xF); // rows - 1 = 1 (rows=2)
    try std.testing.expectEqual(@as(u64, 1), (s >> 49) & 0x3); // coop_store
    try std.testing.expectEqual(@as(u64, 0x4000), s & 0xFFFFFFFFFFF0); // addr
}

test "tensorStoreX31 masks to the golden 44-bit stride width" {
    try std.testing.expectEqual(@as(u64, 0x40), tensorStoreX31(0x40));
    // Bits above the 44-bit mask (copied verbatim from tensors.h) are cleared too.
    try std.testing.expectEqual(@as(u64, 0), tensorStoreX31(1 << 44));
}

test "tensor CSR and wait-event constants" {
    try std.testing.expectEqual(@as(u12, 0x801), CSR_TENSOR_FMA);
    try std.testing.expectEqual(@as(u12, 0x830), CSR_TENSOR_WAIT);
    try std.testing.expectEqual(@as(u12, 0x83f), CSR_TENSOR_LOAD);
    try std.testing.expectEqual(@as(u12, 0x87f), CSR_TENSOR_STORE);
    try std.testing.expectEqual(@as(u12, 0x806), CSR_TENSOR_QUANT);
    try std.testing.expectEqual(@as(u12, 0x7e0), CSR_MCACHE_CONTROL);
    try std.testing.expectEqual(@as(u64, 0), TENSOR_WAIT_LOAD_0);
    try std.testing.expectEqual(@as(u64, 1), TENSOR_WAIT_LOAD_1);
    try std.testing.expectEqual(@as(u64, 7), TENSOR_WAIT_FMA);
    try std.testing.expectEqual(@as(u64, 8), TENSOR_WAIT_STORE);
    try std.testing.expectEqual(@as(u64, 10), TENSOR_WAIT_QUANT);
}

test "assert QuantTransform enum values" {
    try std.testing.expectEqual(@as(u4, 1), @intFromEnum(QuantTransform.i32_to_f32));
    try std.testing.expectEqual(@as(u4, 6), @intFromEnum(QuantTransform.fp32_mul_row));
    try std.testing.expectEqual(@as(u4, 8), @intFromEnum(QuantTransform.satint8));
    try std.testing.expectEqual(@as(u4, 10), @intFromEnum(QuantTransform.pack_128b));
}

test "packTensorQuant matches a real decoded proof-kernel descriptor" {
    // row_field=1 -> bit51, scp_loc=3 -> 3<<45, transforms {i32_to_f32, fp32_mul_row,
    // f32_to_i32, satint8, pack_128b} -> 0xA8261.
    try std.testing.expectEqual(
        @as(u64, 0x00086000000A8261),
        packTensorQuant(0, 0, 1, 3, .{
            .i32_to_f32,
            .fp32_mul_row,
            .f32_to_i32,
            .satint8,
            .pack_128b,
            .last,
            .last,
            .last,
            .last,
            .last,
        }),
    );
}

test "packTensorQuant field placement" {
    const all_last = [_]QuantTransform{.last} ** 10;
    const start = packTensorQuant(5, 0, 0, 0, all_last);
    try std.testing.expectEqual(@as(u64, 5), start >> 57); // start_reg at bit 57

    const col = packTensorQuant(0, 3, 0, 0, all_last);
    try std.testing.expectEqual(@as(u64, 3), (col >> 55) & 0x3); // col_field at bit 55

    const row = packTensorQuant(0, 0, 0xF, 0, all_last);
    try std.testing.expectEqual(@as(u64, 0xF), (row >> 51) & 0xF); // row_field at bit 51

    const scp = packTensorQuant(0, 0, 0, 0x2A, all_last);
    try std.testing.expectEqual(@as(u64, 0x2A), (scp >> 45) & 0x3F); // scp_loc at bit 45

    var last_slot = all_last;
    last_slot[9] = .pack_128b;
    const slot9 = packTensorQuant(0, 0, 0, 0, last_slot);
    try std.testing.expectEqual(@as(u64, 0xA), (slot9 >> 36) & 0xF); // transforms[9] at bit 36
}
