//! RVC (compressed-instruction) emission for RV64GC targets. When the C extension is enabled,
//! `compress` rewrites a resolved 32-bit instruction stream to a smaller mixed 16/32-bit stream,
//! replacing eligible instructions with their 2-byte compressed forms. This is the reverse of
//! `disasm.decode16`.
//!
//! Branch relaxation is kept simple and correct: branches (`jal` / B-type) stay 32-bit, so once
//! every other instruction's size is decided the new layout is fixed in a single pass and each
//! branch's byte displacement is recomputed against it (no iterative convergence needed). `jalr`
//! that compresses to `c.jr`/`c.jalr` has no displacement, so it compresses freely.
//!
//! Correctness is validated by a disassembly round-trip: every compressed halfword decodes (via
//! the llvm-validated `decode16`) to the same instruction the original word did.

const std = @import("std");
const encode = @import("encode.zig");
const disasm = @import("disasm.zig");

// 32-bit instruction field accessors.
fn opc(w: u32) u32 {
    return w & 0x7f;
}
fn rd(w: u32) u32 {
    return (w >> 7) & 0x1f;
}
fn f3(w: u32) u32 {
    return (w >> 12) & 7;
}
fn rs1(w: u32) u32 {
    return (w >> 15) & 0x1f;
}
fn rs2(w: u32) u32 {
    return (w >> 20) & 0x1f;
}
fn f7(w: u32) u32 {
    return (w >> 25) & 0x7f;
}
fn iimm(w: u32) i32 {
    return @as(i32, @bitCast(w)) >> 20;
}
fn simm(w: u32) i32 {
    const raw: u32 = (f7(w) << 5) | rd(w);
    return @as(i32, @bitCast(raw << 20)) >> 20;
}

/// A compressed register field is 3 bits selecting x8..x15.
fn crp(r: u32) ?u3 {
    return if (r >= 8 and r <= 15) @intCast(r - 8) else null;
}
fn imm6(v: i32) u16 { // the 6-bit immediate common to c.addi/c.li: imm[5] at 12, imm[4:0] at 6:2
    const u: u16 = @intCast(@as(u32, @bitCast(v)) & 0x3f);
    return ((u >> 5) << 12) | ((u & 0x1f) << 2);
}
/// The base of a CI-format compressed instruction in quadrant 1: funct3 at [15:13], op=01.
fn ciBase(funct3: u16) u16 {
    return (funct3 << 13) | 0b01;
}

/// The RVC halfword for a compressible 32-bit instruction, or null if it cannot compress.
pub fn tryCompress(w: u32) ?u16 {
    switch (opc(w)) {
        0b0010011 => switch (f3(w)) { // OP-IMM
            0 => { // addi -> c.li / c.mv / c.addi / c.addi16sp
                const d = rd(w);
                const s = rs1(w);
                const imm = iimm(w);
                if (d == 0) return null;
                if (s == 0 and imm >= -32 and imm <= 31) // c.li rd, imm
                    return ciBase(0b010) | imm6(imm) | (@as(u16, @intCast(d)) << 7);
                if (imm == 0 and s != 0) // c.mv rd, rs (addi rd, rs, 0)
                    return 0b100_0_00000_00000_10 | (@as(u16, @intCast(d)) << 7) | (@as(u16, @intCast(s)) << 2);
                if (d == 2 and s == 2 and imm != 0 and imm >= -512 and imm <= 511 and @rem(imm, 16) == 0) { // c.addi16sp
                    const u: u16 = @intCast(@as(u32, @bitCast(imm)) & 0x3ff);
                    return 0b011_0_00010_00000_01 |
                        (((u >> 9) & 1) << 12) | (((u >> 4) & 1) << 6) | (((u >> 6) & 1) << 5) |
                        (((u >> 7) & 3) << 3) | (((u >> 5) & 1) << 2);
                }
                if (d == s and imm != 0 and imm >= -32 and imm <= 31) // c.addi rd, rd, imm
                    return ciBase(0b000) | imm6(imm) | (@as(u16, @intCast(d)) << 7);
                if (s == 2 and imm > 0 and imm <= 1020 and @rem(imm, 4) == 0) { // c.addi4spn rd', sp, nzuimm
                    const dp = crp(d) orelse return null;
                    const u: u16 = @intCast(imm);
                    return (@as(u16, dp) << 2) |
                        (((u >> 4) & 3) << 11) | (((u >> 6) & 0xf) << 7) | (((u >> 2) & 1) << 6) | (((u >> 3) & 1) << 5);
                }
                return null;
            },
            1 => { // slli -> c.slli
                const d = rd(w);
                const shamt = (w >> 20) & 0x3f;
                if (d == 0 or d != rs1(w) or shamt == 0) return null;
                return 0b000_0000000000_10 | (@as(u16, @intCast(shamt >> 5)) << 12) |
                    (@as(u16, @intCast(d)) << 7) | (@as(u16, @intCast(shamt & 0x1f)) << 2);
            },
            5 => { // srli / srai -> c.srli / c.srai (MISC-ALU, compressed regs)
                const d = rd(w);
                if (d != rs1(w)) return null;
                const dp = crp(d) orelse return null;
                const shamt = (w >> 20) & 0x3f;
                if (shamt == 0) return null;
                const sub: u16 = if (f7(w) == 0b0100000) 0b01 else 0b00; // srai vs srli
                return 0b100_0_00_000_00000_01 | (sub << 10) | (@as(u16, dp) << 7) |
                    (@as(u16, @intCast(shamt >> 5)) << 12) | (@as(u16, @intCast(shamt & 0x1f)) << 2);
            },
            7 => { // andi -> c.andi (MISC-ALU sub=10, compressed regs)
                const d = rd(w);
                const imm = iimm(w);
                if (d != rs1(w) or imm < -32 or imm > 31) return null;
                const dp = crp(d) orelse return null;
                return 0b100_0_10_000_00000_01 | (@as(u16, dp) << 7) | imm6(imm);
            },
            else => return null,
        },
        0b0011011 => { // addiw -> c.addiw
            if (f3(w) != 0) return null;
            const d = rd(w);
            const imm = iimm(w);
            if (d == 0 or d != rs1(w) or imm < -32 or imm > 31) return null;
            return ciBase(0b001) | imm6(imm) | (@as(u16, @intCast(d)) << 7);
        },
        0b0110111 => { // lui -> c.lui (rd not x0/x2, nonzero 6-bit signed field)
            const d = rd(w);
            if (d == 0 or d == 2) return null;
            const imm20: i32 = @as(i32, @bitCast(w & 0xffff_f000)) >> 12; // sign-extended 20-bit field
            if (imm20 == 0 or imm20 < -32 or imm20 > 31) return null; // must fit the 6-bit c.lui field
            const field: u16 = @intCast(@as(u32, @bitCast(imm20)) & 0x3f);
            return (0b011 << 13) | 0b01 | (@as(u16, @intCast(d)) << 7) |
                (((field >> 5) & 1) << 12) | ((field & 0x1f) << 2);
        },
        0b0110011 => { // OP: add -> c.add / c.mv, and sub/xor/or/and -> MISC-ALU (compressed regs)
            const d = rd(w);
            const s1 = rs1(w);
            const s2 = rs2(w);
            if (f3(w) == 0 and f7(w) == 0) { // add
                if (s1 == 0 and s2 != 0 and d != 0) // c.mv rd, rs2
                    return 0b100_0_00000_00000_10 | (@as(u16, @intCast(d)) << 7) | (@as(u16, @intCast(s2)) << 2);
                if (d == s1 and s2 != 0) // c.add rd, rs2
                    return 0b100_1_00000_00000_10 | (@as(u16, @intCast(d)) << 7) | (@as(u16, @intCast(s2)) << 2);
                return null;
            }
            // sub/xor/or/and rd', rd', rs2' with all three in x8..x15.
            if (d != s1) return null;
            const dp = crp(d) orelse return null;
            const sp = crp(s2) orelse return null;
            const sub2: u16 = switch (f3(w)) {
                0 => if (f7(w) == 0b0100000) 0b00 else return null, // sub
                4 => 0b01, // xor
                6 => 0b10, // or
                7 => 0b11, // and
                else => return null,
            };
            return 0b100_0_11_000_00_000_01 | (@as(u16, dp) << 7) | (sub2 << 5) | (@as(u16, sp) << 2);
        },
        0b1100111 => { // jalr -> c.jr / c.jalr (offset 0)
            if (iimm(w) != 0) return null;
            const d = rd(w);
            const s = rs1(w);
            if (s == 0) return null;
            if (d == 0) return 0b100_0_00000_00000_10 | (@as(u16, @intCast(s)) << 7); // c.jr rs
            if (d == 1) return 0b100_1_00000_00000_10 | (@as(u16, @intCast(s)) << 7); // c.jalr rs
            return null;
        },
        0b0000011 => return switch (f3(w)) { // loads
            2 => loadSp(w, false) orelse loadCrp(w, false), // lw
            3 => loadSp(w, true) orelse loadCrp(w, true), // ld
            else => null,
        },
        0b0100011 => return switch (f3(w)) { // stores
            2 => storeSp(w, false) orelse storeCrp(w, false), // sw
            3 => storeSp(w, true) orelse storeCrp(w, true), // sd
            else => null,
        },
        0b0000111 => return switch (f3(w)) { // load-fp: fld -> c.fldsp / c.fld (double only, c.flw is RV32)
            3 => fldSp(w) orelse fldCrp(w),
            else => null,
        },
        0b0100111 => return switch (f3(w)) { // store-fp: fsd -> c.fsdsp / c.fsd
            3 => fsdSp(w) orelse fsdCrp(w),
            else => null,
        },
        else => return null,
    }
}

/// Compress a double-precision fp load or store into the c.fld/c.fsd family. These shrink float
/// functions directly, since isel saves and restores callee-saved fp registers with sp-relative
/// fsd/fld in every float prologue and epilogue and lowers fp memory to fld/fsd.
fn fldSp(w: u32) ?u16 { // c.fldsp: any fp rd, sp base, uimm[5|4:3|8:6] scale 8
    if (rs1(w) != 2) return null;
    const off = iimm(w);
    if (off < 0 or off > 511 or @rem(off, 8) != 0) return null;
    const u: u16 = @intCast(off);
    return (0b001 << 13) | 0b10 | (@as(u16, @intCast(rd(w))) << 7) |
        (((u >> 5) & 1) << 12) | (((u >> 3) & 3) << 5) | (((u >> 6) & 7) << 2);
}
fn fldCrp(w: u32) ?u16 { // c.fld: fp rd' + int base both x8..15, uimm[5:3|7:6] scale 8
    const dp = crp(rd(w)) orelse return null;
    const bp = crp(rs1(w)) orelse return null;
    const off = iimm(w);
    if (off < 0 or off > 255 or @rem(off, 8) != 0) return null;
    const u: u16 = @intCast(off);
    return (0b001 << 13) | (@as(u16, dp) << 2) | (@as(u16, bp) << 7) |
        (((u >> 3) & 7) << 10) | (((u >> 6) & 3) << 5);
}
fn fsdSp(w: u32) ?u16 { // c.fsdsp: any fp rs2, sp base, uimm[5:3|8:6] scale 8
    if (rs1(w) != 2) return null;
    const off = simm(w);
    if (off < 0 or off > 511 or @rem(off, 8) != 0) return null;
    const u: u16 = @intCast(off);
    return (0b101 << 13) | 0b10 | (@as(u16, @intCast(rs2(w))) << 2) |
        (((u >> 3) & 7) << 10) | (((u >> 6) & 7) << 7);
}
fn fsdCrp(w: u32) ?u16 { // c.fsd: fp rs2' + int base both x8..15, uimm[5:3|7:6] scale 8
    const sp2 = crp(rs2(w)) orelse return null;
    const bp = crp(rs1(w)) orelse return null;
    const off = simm(w);
    if (off < 0 or off > 255 or @rem(off, 8) != 0) return null;
    const u: u16 = @intCast(off);
    return (0b101 << 13) | (@as(u16, sp2) << 2) | (@as(u16, bp) << 7) |
        (((u >> 3) & 7) << 10) | (((u >> 6) & 3) << 5);
}

fn loadSp(w: u32, dbl: bool) ?u16 {
    if (rs1(w) != 2) return null; // sp base
    const d = rd(w);
    if (d == 0) return null;
    const off = iimm(w);
    if (off < 0) return null;
    const u: u16 = @intCast(off);
    if (dbl) { // c.ldsp: uimm[5|4:3|8:6], scale 8
        if (off > 511 or @rem(off, 8) != 0) return null;
        return 0b011_0_00000_00000_10 | (((u >> 5) & 1) << 12) | (@as(u16, @intCast(d)) << 7) |
            (((u >> 3) & 3) << 5) | (((u >> 6) & 7) << 2);
    } else { // c.lwsp: uimm[5|4:2|7:6], scale 4
        if (off > 255 or @rem(off, 4) != 0) return null;
        return 0b010_0_00000_00000_10 | (((u >> 5) & 1) << 12) | (@as(u16, @intCast(d)) << 7) |
            (((u >> 2) & 7) << 4) | (((u >> 6) & 3) << 2);
    }
}

fn storeSp(w: u32, dbl: bool) ?u16 {
    if (rs1(w) != 2) return null;
    const s2 = rs2(w);
    const off = simm(w);
    if (off < 0) return null;
    const u: u16 = @intCast(off);
    if (dbl) { // c.sdsp: uimm[5:3|8:6]
        if (off > 511 or @rem(off, 8) != 0) return null;
        return 0b111_000000_00000_10 | (((u >> 3) & 7) << 10) | (((u >> 6) & 7) << 7) | (@as(u16, @intCast(s2)) << 2);
    } else { // c.swsp: uimm[5:2|7:6]
        if (off > 255 or @rem(off, 4) != 0) return null;
        return 0b110_000000_00000_10 | (((u >> 2) & 0xf) << 9) | (((u >> 6) & 3) << 7) | (@as(u16, @intCast(s2)) << 2);
    }
}

fn loadCrp(w: u32, dbl: bool) ?u16 {
    const dp = crp(rd(w)) orelse return null;
    const bp = crp(rs1(w)) orelse return null;
    const off = iimm(w);
    if (off < 0) return null;
    const u: u16 = @intCast(off);
    if (dbl) { // c.ld: uimm[5:3|7:6], scale 8
        if (off > 255 or @rem(off, 8) != 0) return null;
        return 0b011_000_000_00_000_00 | (((u >> 3) & 7) << 10) | (@as(u16, bp) << 7) | (((u >> 6) & 3) << 5) | (@as(u16, dp) << 2);
    } else { // c.lw: uimm[5:3|2|6], scale 4
        if (off > 127 or @rem(off, 4) != 0) return null;
        return 0b010_000_000_00_000_00 | (((u >> 3) & 7) << 10) | (@as(u16, bp) << 7) | (((u >> 2) & 1) << 6) | (((u >> 6) & 1) << 5) | (@as(u16, dp) << 2);
    }
}

fn storeCrp(w: u32, dbl: bool) ?u16 {
    const sp2 = crp(rs2(w)) orelse return null;
    const bp = crp(rs1(w)) orelse return null;
    const off = simm(w);
    if (off < 0) return null;
    const u: u16 = @intCast(off);
    if (dbl) { // c.sd
        if (off > 255 or @rem(off, 8) != 0) return null;
        return 0b111_000_000_00_000_00 | (((u >> 3) & 7) << 10) | (@as(u16, bp) << 7) | (((u >> 6) & 3) << 5) | (@as(u16, sp2) << 2);
    } else { // c.sw
        if (off > 127 or @rem(off, 4) != 0) return null;
        return 0b110_000_000_00_000_00 | (((u >> 3) & 7) << 10) | (@as(u16, bp) << 7) | (((u >> 2) & 1) << 6) | (((u >> 6) & 1) << 5) | (@as(u16, sp2) << 2);
    }
}

/// The compressed control-transfer form a 32-bit word can take (or `none`). Control transfers are
/// variable-size: a `jal x0` or x0-form `beq`/`bne` shrinks to c.j/c.beqz/c.bnez when its final
/// displacement fits the small compressed range, otherwise it stays 32-bit. Whether it fits depends
/// on the layout, which depends on which transfers shrank, so `compress` resolves it with an
/// iterative relaxation fixpoint over `compressBranch(w, disp)`, the size decision for one transfer.
const CForm = enum { none, cj, beqz, bnez };

/// Classify a control transfer by form alone (registers, not displacement). `reg` is the CB
/// register operand (x8..15 mapped to 0..7). Unused for c.j.
fn classifyBranch(w: u32) struct { form: CForm, reg: u3 } {
    if (isJal(w)) return .{ .form = if (rd(w) == 0) .cj else .none, .reg = 0 }; // c.jal is RV32-only
    // B-type: c.beqz/c.bnez compare one x8..15 register against x0 (beq/bne are symmetric).
    const funct = f3(w);
    if (funct != 0 and funct != 1) return .{ .form = .none, .reg = 0 };
    const other: u32 = if (rs2(w) == 0 and rs1(w) != 0) rs1(w) else if (rs1(w) == 0 and rs2(w) != 0) rs2(w) else return .{ .form = .none, .reg = 0 };
    const cr = crp(other) orelse return .{ .form = .none, .reg = 0 };
    return .{ .form = if (funct == 0) .beqz else .bnez, .reg = cr };
}

/// Whether `disp` (bytes, always even here) fits a given compressed form's signed range.
fn branchFits(form: CForm, disp: i32) bool {
    return switch (form) {
        .none => false,
        .cj => disp >= -2048 and disp <= 2046, // CJ imm[11:1]
        .beqz, .bnez => disp >= -256 and disp <= 254, // CB imm[8:1]
    };
}

/// The c.j halfword (jal x0) for byte displacement `disp`. Inverse of disasm.cjimm.
fn encCJ(disp: i32) u16 {
    const u: u32 = @bitCast(disp);
    var h: u16 = (0b101 << 13) | 0b01;
    h |= @intCast(((u >> 11) & 1) << 12);
    h |= @intCast(((u >> 4) & 1) << 11);
    h |= @intCast(((u >> 8) & 3) << 9);
    h |= @intCast(((u >> 10) & 1) << 8);
    h |= @intCast(((u >> 6) & 1) << 7);
    h |= @intCast(((u >> 7) & 1) << 6);
    h |= @intCast(((u >> 1) & 7) << 3);
    h |= @intCast(((u >> 5) & 1) << 2);
    return h;
}

/// The c.beqz/c.bnez halfword for register `reg` (x8..15 as 0..7) and byte displacement `disp`.
/// Inverse of disasm.cbimm.
fn encCB(form: CForm, reg: u3, disp: i32) u16 {
    const u: u32 = @bitCast(disp);
    var h: u16 = (@as(u16, if (form == .beqz) 0b110 else 0b111) << 13) | 0b01;
    h |= @as(u16, reg) << 7;
    h |= @intCast(((u >> 8) & 1) << 12);
    h |= @intCast(((u >> 3) & 3) << 10);
    h |= @intCast(((u >> 6) & 3) << 5);
    h |= @intCast(((u >> 1) & 3) << 3);
    h |= @intCast(((u >> 5) & 1) << 2);
    return h;
}

/// The RVC halfword for a control transfer whose final byte displacement is `disp`, or null if it
/// is not compressible (wrong form/register or out of the compressed range).
fn compressBranch(w: u32, disp: i32) ?u16 {
    const c = classifyBranch(w);
    if (!branchFits(c.form, disp)) return null;
    return switch (c.form) {
        .none => null,
        .cj => encCJ(disp),
        .beqz, .bnez => encCB(c.form, c.reg, disp),
    };
}

fn isJal(w: u32) bool {
    return opc(w) == 0b1101111;
}
fn isBranch(w: u32) bool {
    return opc(w) == 0b1100011;
}
fn jimm(w: u32) i32 {
    const u = (((w >> 31) & 1) << 20) | (((w >> 12) & 0xff) << 12) | (((w >> 20) & 1) << 11) | (((w >> 21) & 0x3ff) << 1);
    return @as(i32, @bitCast(u << 11)) >> 11;
}
fn bimm(w: u32) i32 {
    const u = (((w >> 31) & 1) << 12) | (((w >> 7) & 1) << 11) | (((w >> 25) & 0x3f) << 5) | (((w >> 8) & 0xf) << 1);
    return @as(i32, @bitCast(u << 19)) >> 19;
}

/// Replace the J-type immediate of `w` with `disp` (bytes), keeping its rd field.
fn setJalImm(w: u32, disp: i32) u32 {
    const base = w & 0x00000fff; // opcode + rd + funct-region low? keep [11:0] (opcode+rd)
    const u: u32 = @bitCast(disp);
    const imm = (((u >> 20) & 1) << 31) | (((u >> 1) & 0x3ff) << 21) | (((u >> 11) & 1) << 20) | (((u >> 12) & 0xff) << 12);
    return base | imm;
}

/// Replace the B-type immediate of `w` with `disp` (bytes), keeping rs1/rs2/funct3.
fn setBranchImm(w: u32, disp: i32) u32 {
    const base = w & 0x01ff_f07f; // keep rs2[24:20] rs1[19:15] f3[14:12] opcode[6:0]
    const u: u32 = @bitCast(disp);
    const imm = (((u >> 12) & 1) << 31) | (((u >> 5) & 0x3f) << 25) | (((u >> 1) & 0xf) << 8) | (((u >> 11) & 1) << 7);
    return base | imm;
}

/// Map an original byte offset to the compressed layout. Offsets that fall on an instruction start
/// use that instruction's new offset. Offsets past the code (trailing data) shift by the amount the
/// code as a whole shrank, since appended data keeps its distance from the code's end.
fn mapTarget(offs: []const usize, n: usize, target_old: usize) usize {
    const old_end = n * 4;
    if (target_old < old_end and target_old % 4 == 0) return offs[target_old / 4];
    return offs[n] + (target_old - old_end);
}

/// A resolved PC-relative `auipc` + `addi`/load/store/`jalr` pair (word indices `hi` and `lo`) whose
/// combined target is byte offset `target` in the original 32-bit layout. Compression must recompute
/// the pair's hi20/lo12 for the shrunk layout and must not compress the `lo` instruction, since a
/// 2-byte form has no room for the 12-bit low immediate. `target` may point past the code into
/// trailing data appended after it.
pub const PcrelPair = struct { hi: usize, lo: usize, target: usize };

/// Compress a resolved 32-bit RV64 instruction stream to a mixed 16/32-bit RV64GC stream. See
/// `compressPairs`. This is the common case with no PC-relative pairs to preserve.
pub fn compress(allocator: std.mem.Allocator, code: []const u32) std.mem.Allocator.Error![]u8 {
    return compressCore(allocator, code, &.{}, &.{}, null);
}

/// Like `compress`, but preserves PC-relative `auipc` pairs: each `lo` site is kept 32-bit and every
/// pair's hi20/lo12 is recomputed against the shrunk layout (its target byte offset remapped, whether
/// it lands in code or in trailing data). Non-branch instructions compress on form. Control transfers
/// shrink to c.j/c.beqz/c.bnez when their displacement fits, decided by an iterative relaxation
/// fixpoint that starts all-small, expands any that overflow, and repeats until stable. All branch
/// displacements are recomputed for the final layout. Caller owns the result.
pub fn compressPairs(allocator: std.mem.Allocator, code: []const u32, pairs: []const PcrelPair) std.mem.Allocator.Error![]u8 {
    return compressCore(allocator, code, pairs, &.{}, null);
}

/// The linker primitive: compress `code`, keeping every word index in `pinned` verbatim and 32-bit (a
/// later linker patches those sites itself: `jal` call targets and `auipc`/pcrel-lo pairs), while
/// still compressing everything else and recomputing purely-internal branch displacements. Writes the
/// new byte offset of each old word index into `out_offsets` (which must have length `code.len + 1`),
/// so the caller can remap its relocation offsets and symbol values onto the shrunk layout. Caller
/// owns the returned bytes.
pub fn compressPinned(allocator: std.mem.Allocator, code: []const u32, pinned: []const usize, out_offsets: []usize) std.mem.Allocator.Error![]u8 {
    std.debug.assert(out_offsets.len == code.len + 1);
    return compressCore(allocator, code, &.{}, pinned, out_offsets);
}

fn compressCore(allocator: std.mem.Allocator, code: []const u32, pairs: []const PcrelPair, pinned: []const usize, out_offsets: ?[]usize) std.mem.Allocator.Error![]u8 {
    const n = code.len;

    // The `lo` site of each pcrel pair must stay 32-bit so a later linker (or the recompute below)
    // can patch its 12-bit low immediate. `pin` sites are stronger: emitted verbatim (no compression
    // and no displacement recompute) because an external linker owns their immediate.
    const lo_pinned = try allocator.alloc(bool, n);
    defer allocator.free(lo_pinned);
    @memset(lo_pinned, false);
    for (pairs) |p| lo_pinned[p.lo] = true;
    const pin = try allocator.alloc(bool, n);
    defer allocator.free(pin);
    @memset(pin, false);
    for (pinned) |idx| pin[idx] = true;
    // Per-instruction plan. `half` is the fixed non-branch compression. For a control transfer,
    // `form`/`reg`/`target` describe its possible compressed shape, `mappable` says its target
    // index is in range (else leave it untouched), and `small` is the current relaxation guess.
    const half = try allocator.alloc(?u16, n);
    defer allocator.free(half);
    const form = try allocator.alloc(CForm, n);
    defer allocator.free(form);
    const creg_ = try allocator.alloc(u3, n);
    defer allocator.free(creg_);
    const target = try allocator.alloc(usize, n);
    defer allocator.free(target);
    const mappable = try allocator.alloc(bool, n);
    defer allocator.free(mappable);
    const small = try allocator.alloc(bool, n);
    defer allocator.free(small);

    for (code, 0..) |w, i| {
        form[i] = .none;
        mappable[i] = false;
        creg_[i] = 0;
        target[i] = i;
        if (pin[i]) {
            half[i] = null; // emitted verbatim, the external linker owns this site's immediate
            small[i] = false;
        } else if (isJal(w) or isBranch(w)) {
            half[i] = null;
            const disp0 = if (isJal(w)) jimm(w) else bimm(w);
            const ti: i64 = @as(i64, @intCast(i)) + @divTrunc(disp0, 4);
            if (ti >= 0 and ti <= @as(i64, @intCast(n))) {
                mappable[i] = true;
                target[i] = @intCast(ti);
                const c = classifyBranch(w);
                form[i] = c.form;
                creg_[i] = c.reg;
            }
            small[i] = form[i] != .none; // optimistic: assume it compresses
        } else if (lo_pinned[i]) {
            half[i] = null; // pcrel_lo12 site: keep 32-bit so its low immediate stays patchable
            small[i] = false;
        } else {
            half[i] = tryCompress(w);
            small[i] = half[i] != null;
        }
    }

    const offs = try allocator.alloc(usize, n + 1);
    defer allocator.free(offs);
    const sizeOf = struct {
        fn f(hlf: []const ?u16, sml: []const bool, i: usize) usize {
            return if (hlf[i] != null or sml[i]) 2 else 4;
        }
    }.f;

    // Relaxation: recompute offsets, expand any compressed transfer whose displacement no longer
    // fits, and repeat. Sizes only grow, so this converges.
    while (true) {
        var off: usize = 0;
        for (0..n) |i| {
            offs[i] = off;
            off += sizeOf(half, small, i);
        }
        offs[n] = off;
        var changed = false;
        for (0..n) |i| {
            if (form[i] != .none and small[i]) {
                const disp: i32 = @intCast(@as(i64, @intCast(offs[target[i]])) - @as(i64, @intCast(offs[i])));
                if (!branchFits(form[i], disp)) {
                    small[i] = false;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }

    // Export the settled layout (new byte offset per old word index) for a linker to remap onto.
    if (out_offsets) |dst| @memcpy(dst, offs);

    // Recompute each PC-relative pair against the settled layout. `hi`/`lo` are both pinned 32-bit,
    // so their words are emitted verbatim below unless overridden here.
    const patched = try allocator.alloc(?u32, n);
    defer allocator.free(patched);
    @memset(patched, null);
    for (pairs) |p| {
        const new_hi = offs[p.hi];
        const new_target = mapTarget(offs, n, p.target);
        const pcrel: i64 = @as(i64, @intCast(new_target)) - @as(i64, @intCast(new_hi));
        const u: u64 = @bitCast(pcrel);
        const hi20: u20 = @truncate((u +% 0x800) >> 12);
        const lo12: u12 = @truncate(u);
        patched[p.hi] = (code[p.hi] & 0x0000_0fff) | (@as(u32, hi20) << 12); // auipc: keep rd, set imm20
        patched[p.lo] = (code[p.lo] & 0x000f_ffff) | (@as(u32, lo12) << 20); // I-type: keep rd/rs1/f3/op, set imm
    }

    // Emit against the settled layout.
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (code, 0..) |w, i| {
        if (half[i]) |h| {
            try out.append(allocator, @intCast(h & 0xff));
            try out.append(allocator, @intCast(h >> 8));
            continue;
        }
        if (patched[i]) |pw| { // a recomputed pcrel hi/lo site (always 32-bit)
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, pw, .little);
            try out.appendSlice(allocator, &buf);
            continue;
        }
        if ((isJal(w) or isBranch(w)) and mappable[i]) {
            const disp: i32 = @intCast(@as(i64, @intCast(offs[target[i]])) - @as(i64, @intCast(offs[i])));
            if (small[i]) {
                const h = switch (form[i]) {
                    .cj => encCJ(disp),
                    .beqz, .bnez => encCB(form[i], creg_[i], disp),
                    .none => unreachable,
                };
                try out.append(allocator, @intCast(h & 0xff));
                try out.append(allocator, @intCast(h >> 8));
                continue;
            }
            const word = if (isJal(w)) setJalImm(w, disp) else setBranchImm(w, disp);
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, word, .little);
            try out.appendSlice(allocator, &buf);
            continue;
        }
        // A non-branch that did not compress, or an unmappable transfer: emit unchanged.
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, w, .little);
        try out.appendSlice(allocator, &buf);
    }
    return out.toOwnedSlice(allocator);
}

test "each compressed halfword decodes to the same instruction as the original word" {
    const a = std.testing.allocator;
    // Instructions Vulcan emits that RVC can compress. Each must (1) compress and (2) the
    // compressed form must disassemble identically to the original 32-bit form.
    const words = [_]u32{
        encode.addi(.x10, .x5, 0), // mv x10, x5
        encode.addi(.x12, .x0, 0), // li x12, 0
        encode.addi(.x8, .x8, 7), // c.addi
        0x0015051b, // addiw x10, x10, 1 -> c.addiw (no encoder helper for addiw)
        encode.add(.x11, .x11, .x10), // c.add
        encode.slli(.x11, .x11, 2), // c.slli
        encode.jalr(.x0, .x1, 0), // ret
        encode.jalr(.x1, .x5, 0), // c.jalr
        encode.ld(.x8, .x9, 16), // c.ld
        encode.sd(.x8, .x9, 8), // c.sd
        encode.ld(.x10, .x2, 24), // c.ldsp
        encode.sd(.x1, .x2, 8), // c.sdsp
        encode.lw(.x8, .x9, 12), // c.lw
        encode.sub(.x8, .x8, .x9), // c.sub
        encode.and_(.x8, .x8, .x9), // c.and
        encode.addi(.x2, .x2, -32), // c.addi16sp
        encode.addi(.x8, .x2, 16), // c.addi4spn (rd' = sp + nzuimm)
        encode.andi(.x9, .x9, 7), // c.andi
        encode.srli(.x8, .x8, 3), // c.srli
        encode.srai(.x9, .x9, 2), // c.srai
        encode.fsd(.f8, .x2, 16), // c.fsdsp (fp save to a stack slot)
        encode.fld(.f9, .x2, 24), // c.fldsp (fp restore)
        encode.fld(.f8, .x9, 8), // c.fld (fp reg + base both x8..15)
        encode.fsd(.f10, .x11, 16), // c.fsd
    };
    for (words) |w| {
        const h = tryCompress(w) orelse {
            std.debug.print("did not compress: 0x{x:0>8}\n", .{w});
            return error.NotCompressed;
        };
        const full = try disasm.one(a, w);
        defer a.free(full);
        const comp = try disasm.oneCompressed(a, h);
        defer a.free(comp);
        try std.testing.expectEqualStrings(full, comp);
    }
}

test "control transfers compress to c.j / c.beqz / c.bnez and decode identically" {
    const a = std.testing.allocator;
    // (word, its own byte displacement). The original 32-bit form and the compressed form must
    // disassemble to the same text (target included). x8..x15 for the CB register operand.
    const Case = struct { w: u32, disp: i32 };
    const cases = [_]Case{
        .{ .w = encode.jal(.x0, 20), .disp = 20 }, // j .+20 -> c.j
        .{ .w = encode.jal(.x0, -40), .disp = -40 }, // j .-40 -> c.j
        .{ .w = encode.beq(.x8, .x0, 16), .disp = 16 }, // beqz x8 -> c.beqz
        .{ .w = encode.bne(.x10, .x0, -8), .disp = -8 }, // bnez x10 -> c.bnez
        .{ .w = encode.beq(.x0, .x15, 32), .disp = 32 }, // beqz x15 (operands swapped)
    };
    for (cases) |c| {
        const h = compressBranch(c.w, c.disp) orelse {
            std.debug.print("branch did not compress: 0x{x:0>8}\n", .{c.w});
            return error.NotCompressed;
        };
        const full = try disasm.one(a, c.w);
        defer a.free(full);
        const comp = try disasm.oneCompressed(a, h);
        defer a.free(comp);
        try std.testing.expectEqualStrings(full, comp);
    }
}

test "a control transfer that cannot be a CB/CJ stays 32-bit" {
    // Out of range (c.beqz is +/-254, c.j is +/-2046), wrong register (x1 not in x8..15), and a
    // two-register branch all fail to compress.
    try std.testing.expectEqual(@as(?u16, null), compressBranch(encode.beq(.x8, .x0, 300), 300));
    try std.testing.expectEqual(@as(?u16, null), compressBranch(encode.jal(.x0, 4096), 4096));
    try std.testing.expectEqual(@as(?u16, null), compressBranch(encode.beq(.x1, .x0, 8), 8));
    try std.testing.expectEqual(@as(?u16, null), compressBranch(encode.beq(.x8, .x9, 8), 8));
    try std.testing.expectEqual(@as(?u16, null), compressBranch(encode.blt(.x8, .x0, 8), 8)); // blt has no CB form
}

test "compress emits an in-range c.j and recomputes its displacement" {
    const a = std.testing.allocator;
    const code = [_]u32{
        encode.jal(.x0, 12), // 0: j to index 3 (the ret)
        encode.addi(.x10, .x0, 0), // 1: c.li
        encode.addi(.x11, .x0, 0), // 2: c.li
        encode.jalr(.x0, .x1, 0), // 3: ret
    };
    const bytes = try compress(a, &code);
    defer a.free(bytes);
    // Everything compresses now, including the jump: 4 * 2 = 8 bytes. The c.j lands on the ret at
    // byte 6 (2 + 2 + 2), so its displacement is +6.
    try std.testing.expectEqual(@as(usize, 8), bytes.len);
    const text = try disasm.formatBytes(a, bytes, 0);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "j .+6") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "00000006: 8082      ret") != null);
}

test "lui compresses to c.lui with the correct halfword bits" {
    // c.lui carries a nonzero 6-bit signed field in bits [12] and [6:2], with rd in bits [11:7].
    // Assert exact halfwords (the two decoders format lui differently, hex vs unsigned decimal, so a
    // text compare is not meaningful here). base = funct3=011 | op=01 = 0x6001.
    // rd=10, field=5    -> 0x6001 | 10<<7 | 5<<2                 = 0x6515
    try std.testing.expectEqual(@as(?u16, 0x6515), tryCompress(encode.lui(.x10, 5)));
    // rd=11, imm=-1     -> field 0x3f: 0x6001 | 11<<7 | 1<<12 | 0x1f<<2 = 0x75FD
    try std.testing.expectEqual(@as(?u16, 0x75FD), tryCompress(encode.lui(.x11, 0xFFFFF)));
    // rd=12, imm=-32    -> field 0x20: 0x6001 | 12<<7 | 1<<12         = 0x7601
    try std.testing.expectEqual(@as(?u16, 0x7601), tryCompress(encode.lui(.x12, 0xFFFE0)));
    // And each round-trips through the llvm-validated decoder to a `lui x{rd}, ...` (same value).
    const a = std.testing.allocator;
    const comp = try disasm.oneCompressed(a, tryCompress(encode.lui(.x10, 5)).?);
    defer a.free(comp);
    try std.testing.expectEqualStrings("lui x10, 5", comp);
    // Out of the 6-bit range and the reserved rd (x2, which is c.addi16sp) do not compress.
    try std.testing.expectEqual(@as(?u16, null), tryCompress(encode.lui(.x10, 40)));
    try std.testing.expectEqual(@as(?u16, null), tryCompress(encode.lui(.x2, 5)));
}

test "a float save / restore sequence compresses (c.fsdsp / c.fldsp)" {
    const a = std.testing.allocator;
    // What a float prologue/epilogue looks like: push ra, save two callee-saved fp regs, ... , then
    // reload and return. Every piece is compressible, so 6 * 2 = 12 bytes down from 24.
    const code = [_]u32{
        encode.addi(.x2, .x2, -32), // c.addi16sp
        encode.fsd(.f8, .x2, 0), // c.fsdsp
        encode.fsd(.f9, .x2, 8), // c.fsdsp
        encode.fld(.f8, .x2, 0), // c.fldsp
        encode.fld(.f9, .x2, 8), // c.fldsp
        encode.jalr(.x0, .x1, 0), // ret
    };
    const bytes = try compress(a, &code);
    defer a.free(bytes);
    try std.testing.expectEqual(@as(usize, 12), bytes.len);
    const text = try disasm.formatBytes(a, bytes, 0);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, ".short") == null); // all really compressed
    try std.testing.expect(std.mem.indexOf(u8, text, "fsd f8, 0(x2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fld f9, 8(x2)") != null);
}

test "single-precision fp load/store does not compress (no RV64 c-form)" {
    // c.flw/c.fsw are RV32-only. On RV64 those encodings are c.ld/c.sd, so 32-bit flw/fsw stay wide.
    try std.testing.expectEqual(@as(?u16, null), tryCompress(encode.flw(.f8, .x2, 8)));
    try std.testing.expectEqual(@as(?u16, null), tryCompress(encode.fsw(.f8, .x2, 8)));
    // And a misaligned (non-multiple-of-8) fld offset cannot use the scaled compressed immediate.
    try std.testing.expectEqual(@as(?u16, null), tryCompress(encode.fld(.f8, .x2, 4)));
}

test "compressPairs keeps a pcrel pair 32-bit and recomputes its target through the shrink" {
    const a = std.testing.allocator;
    // auipc a0, hi ; addi a0, a0, lo  point at data appended after the code. A compressible filler
    // between the pair and the data shifts the target, so the pair must be recomputed.
    const data_word: usize = 4; // data at word 4 = byte 16 in the original layout
    const code = [_]u32{
        encode.auipc(.x10, 0), // 0: hi (target patched below to point at byte 16)
        encode.addi(.x10, .x10, 16), // 1: lo (pcrel to +16)
        encode.addi(.x8, .x8, 1), // 2: compressible filler -> c.addi (2 bytes)
        encode.jalr(.x0, .x1, 0), // 3: ret (compresses)
    };
    const pairs = [_]PcrelPair{.{ .hi = 0, .lo = 1, .target = data_word * 4 }};
    const bytes = try compressPairs(a, &code, &pairs);
    defer a.free(bytes);
    // Layout after: auipc(4) + addi(4) + c.addi(2) + ret(2) = 12 bytes, so data now sits at byte 12.
    try std.testing.expectEqual(@as(usize, 12), bytes.len);
    // The auipc is still 4 bytes at offset 0, the addi still 4 bytes at offset 4 (both pinned), and
    // together they must now form a pcrel of +12 (was +16). pcrel 12: hi20=0, lo12=12.
    const auipc_w = std.mem.readInt(u32, bytes[0..4], .little);
    const addi_w = std.mem.readInt(u32, bytes[4..8], .little);
    try std.testing.expectEqual(encode.auipc(.x10, 0), auipc_w); // hi20 still 0 (target within 2KB)
    try std.testing.expectEqual(encode.addi(.x10, .x10, 12), addi_w); // lo12 recomputed 16 -> 12
    // The filler really compressed (proving the shift happened), and the ret is compressed too.
    const text = try disasm.formatBytes(a, bytes, 0);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, ".short") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "0000000a: 8082      ret") != null);
}

test "compressPinned keeps pinned sites verbatim and reports the shrunk offset map" {
    const a = std.testing.allocator;
    // A call `jal` (imm patched later by a linker, 0 here) and a `c.addi`-able add that are BOTH
    // pinned: they must stay verbatim 32-bit, while the surrounding li and ret still compress.
    const code = [_]u32{
        encode.addi(.x10, .x0, 5), // 0: c.li -> 2 bytes
        encode.jal(.x1, 0), // 1: PINNED call site (linker patches its target)
        encode.addi(.x11, .x11, 3), // 2: PINNED (would be c.addi, but kept 32-bit)
        encode.jalr(.x0, .x1, 0), // 3: ret -> 2 bytes
    };
    var offs: [5]usize = undefined;
    const bytes = try compressPinned(a, &code, &.{ 1, 2 }, &offs);
    defer a.free(bytes);
    // Layout: c.li(2) + jal(4) + addi(4) + ret(2) = 12 bytes, with the offset map to match.
    try std.testing.expectEqual(@as(usize, 12), bytes.len);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 6, 10, 12 }, &offs);
    // Pinned sites verbatim at their remapped offsets.
    try std.testing.expectEqual(encode.jal(.x1, 0), std.mem.readInt(u32, bytes[2..6], .little));
    try std.testing.expectEqual(encode.addi(.x11, .x11, 3), std.mem.readInt(u32, bytes[6..10], .little));
    // Non-pinned neighbours still compressed.
    const text = try disasm.formatBytes(a, bytes, 0);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "li x10, 5") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "0000000a: 8082      ret") != null);
}

test "non-compressible instructions return null" {
    // A wide immediate, a non-sp/large offset, and a branch are not compressible.
    try std.testing.expectEqual(@as(?u16, null), tryCompress(encode.addi(.x10, .x5, 100)));
    try std.testing.expectEqual(@as(?u16, null), tryCompress(encode.lui(.x5, 0x12345)));
    try std.testing.expectEqual(@as(?u16, null), tryCompress(encode.beq(.x1, .x2, 16)));
}

test "compression shrinks a realistic prologue / body / epilogue" {
    const a = std.testing.allocator;
    // A frame setup, a small body with one branch, and teardown. All 9 compress (the branch is a
    // beqz on x10, which is in x8..15 and in range, so it becomes c.beqz). 9*2 = 18 bytes, down
    // from 9*4 = 36 (a 50% reduction).
    const code = [_]u32{
        encode.addi(.x2, .x2, -16), // c.addi16sp
        encode.sd(.x1, .x2, 8), // c.sdsp
        encode.addi(.x10, .x0, 20), // c.li (in the ±31 range)
        encode.add(.x10, .x10, .x11), // c.add
        encode.beq(.x10, .x0, 8), // c.beqz (x10 in x8..15, in range)
        encode.addi(.x10, .x10, 1), // c.addi
        encode.ld(.x1, .x2, 8), // c.ldsp
        encode.addi(.x2, .x2, 16), // c.addi16sp
        encode.jalr(.x0, .x1, 0), // ret
    };
    const bytes = try compress(a, &code);
    defer a.free(bytes);
    try std.testing.expectEqual(@as(usize, 18), bytes.len);
    // The compressed stream still disassembles cleanly (no `.short` fallbacks) and keeps the ret.
    const text = try disasm.formatBytes(a, bytes, 0);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, ".short") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "ret") != null);
}

test "compress recomputes branch displacements for the smaller layout" {
    const a = std.testing.allocator;
    // A forward branch over two compressible movs, then a compressible target.
    const code = [_]u32{
        encode.beq(.x1, .x2, 12), // 0: branch to index 3 (byte 12)
        encode.addi(.x10, .x0, 0), // 1: li -> compresses to 2 bytes
        encode.addi(.x11, .x0, 0), // 2: li -> 2 bytes
        encode.jalr(.x0, .x1, 0), // 3: ret (target) -> 2 bytes
    };
    const bytes = try compress(a, &code);
    defer a.free(bytes);
    // Layout: branch(4) + li(2) + li(2) = 8, so the branch's new displacement is 8 bytes.
    // Disassemble and confirm the branch is still 32-bit and now jumps +8, landing on the ret.
    const text = try disasm.formatBytes(a, bytes, 0);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "beq x1, x2, .+8") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "00000008: 8082      ret") != null);
}
