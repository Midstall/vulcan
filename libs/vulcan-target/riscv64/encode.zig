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

/// `fmv.s rd, rs` (single-precision register move, via `fsgnj.s rd, rs, rs`).
pub fn fmv_s(rd: FReg, rs: FReg) u32 {
    return fpRType(0b0010000, 0b000, rd, rs, rs);
}

/// `fmv.d rd, rs` (double-precision register move).
pub fn fmv_d(rd: FReg, rs: FReg) u32 {
    return fpRType(0b0010001, 0b000, rd, rs, rs);
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

test "encodes single-precision float arithmetic" {
    // OP-FP, dynamic rounding (rm=111). fadd.s f1, f2, f3, etc.
    try std.testing.expectEqual(@as(u32, 0x003170d3), fadd_s(.f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x083170d3), fsub_s(.f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x103170d3), fmul_s(.f1, .f2, .f3));
    try std.testing.expectEqual(@as(u32, 0x183170d3), fdiv_s(.f1, .f2, .f3));
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
}

test "encodes rev8" {
    // rev8 x1, x2 (byte-reverse, Zbb)
    try std.testing.expectEqual(@as(u32, 0x6b815093), rev8(.x1, .x2));
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

test "encodes jalr (function return)" {
    // ret == jalr x0, ra, 0  ->  0x00008067
    try std.testing.expectEqual(@as(u32, 0x00008067), jalr(.x0, .x1, 0));
}

test "encodes the arithmetic and bitwise R-type instructions" {
    try std.testing.expectEqual(@as(u32, 0x403100b3), sub(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x023100b3), mul(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x023140b3), div(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x003170b3), and_(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x003160b3), or_(.x1, .x2, .x3));
    try std.testing.expectEqual(@as(u32, 0x003140b3), xor_(.x1, .x2, .x3));
}
