//! NVIDIA SASS instruction encoder. Encoding is shared across Volta..Blackwell
//! (only per-instruction latencies differ). Each instruction is 128 bits = four
//! little-endian dwords.
//!
//! Bit layout (predicate@12, dst@16, scheduling control@105..) and the MOV / STG /
//! LDG / S2R / BRA / EXIT opcodes are verified live on Blackwell sm_120 by prism's
//! assembler. Remaining ALU opcodes and field offsets come from Mesa's NAK encoder
//! (src/nouveau/compiler/nak/sm70_encode.rs): the 9-bit base opcode in bits 0..8
//! with a source form (1 = register) in bits 9..11, source operands srcA@24,
//! srcB@32, srcC@64. End-to-end hardware confirmation is deferred to prism's GPU
//! dispatch path.

const std = @import("std");

/// The zero general-purpose register (reads 0, writes discarded).
pub const RZ: u8 = 255;
/// The always-true predicate register.
pub const PT: u8 = 7;

/// A 128-bit instruction: four dwords, little-endian.
pub const Inst = [4]u32;

/// Per-instruction scheduling plus the guard predicate. The default `stall` is
/// conservative (covers a back-to-back fixed-latency register dependency).
/// Variable-latency ops (global loads) use `wr_barrier` plus a consumer `wait_mask`.
pub const Control = struct {
    stall: u4 = 15,
    wr_barrier: u3 = 7, // scoreboard to set on completion (7 = none)
    rd_barrier: u3 = 7,
    wait_mask: u6 = 0, // scoreboards to wait on before issue
    pred: u8 = PT, // guard predicate (PT = unconditional)
    pred_neg: bool = false, // guard on !pred
};

/// Set `width` bits at bit offset `lo` within the instruction.
fn setBits(inst: *Inst, lo: usize, width: usize, val: u64) void {
    var i: usize = 0;
    while (i < width) : (i += 1) {
        const bit = lo + i;
        const off: u5 = @intCast(bit % 32);
        const b: u32 = @intCast((val >> @intCast(i)) & 1);
        inst[bit / 32] = (inst[bit / 32] & ~(@as(u32, 1) << off)) | (b << off);
    }
}

/// Start an instruction with the guard predicate and scheduling control filled in.
fn base(c: Control) Inst {
    var w: Inst = .{ 0, 0, 0, 0 };
    setBits(&w, 12, 3, c.pred);
    if (c.pred_neg) setBits(&w, 15, 1, 1);
    setBits(&w, 105, 4, c.stall);
    setBits(&w, 110, 3, c.wr_barrier);
    setBits(&w, 113, 3, c.rd_barrier);
    setBits(&w, 116, 6, c.wait_mask);
    return w;
}

/// Comparison for ISETP / FSETP. Values match NAK's `IntCmpOp` encoding.
pub const Cmp = enum(u3) { lt = 1, eq = 2, le = 3, gt = 4, ne = 5, ge = 6 };

/// A fixed-latency ALU op with a register second source: the 9-bit base opcode in
/// bits 0..8 plus the source form in bits 9..11 (1 = register, as NAK's
/// `encode_alu` does). `dst` at 16, srcA at 24, srcB at 32, srcC at 64.
fn alu(op: u9, dst: u8, a: u8, b: u8, c_in: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 9, op);
    setBits(&w, 9, 3, 1); // form: register srcB
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, a);
    setBits(&w, 32, 8, b);
    setBits(&w, 64, 8, c_in);
    return w;
}

/// `MOV dst, imm32`: load a 32-bit immediate (ALU MOV 0x002, form 4). Verified.
pub fn movImm(dst: u8, imm: u32, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 9, 0x002);
    setBits(&w, 9, 3, 4); // form: 32-bit immediate
    setBits(&w, 16, 8, dst);
    setBits(&w, 32, 32, imm);
    setBits(&w, 72, 4, 0xf); // all quad lanes
    return w;
}

/// `MOV dst, src`: copy a 32-bit GPR (ALU MOV 0x002, form 1). Verified.
pub fn movReg(dst: u8, src: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 9, 0x002);
    setBits(&w, 9, 3, 1); // form: register source
    setBits(&w, 16, 8, dst);
    setBits(&w, 32, 8, src);
    setBits(&w, 72, 4, 0xf);
    return w;
}

// Integer ALU ops use the regular (non-uniform) base opcodes: the low form, bit
// 0x080 clear. The uniform-datapath variants (UIADD3 0x090, UIMAD 0x0a4, ...)
// write uniform registers. Using them with a GPR destination is an illegal
// encoding that faults on Blackwell ("Illegal Instruction Encoding"). IADD3
// (0x010) and IMAD (0x024) are verified live on a Blackwell GB20x via prism's
// SPIR-V compute path. LOP3/ISETP/SEL/SHF follow the same uniform->regular rule
// (the regular float ALU ops 0x020-0x023 confirm it).

/// `IADD3 dst, a, b, RZ`: 32-bit integer add (dst = a + b). Regular IADD3 0x010.
/// (Add an immediate by materializing it with `movImm` first.)
pub fn iadd3(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = alu(0x010, dst, a, b, RZ, c);
    setBits(&w, 81, 3, PT); // carry-out predicate = none
    setBits(&w, 84, 3, PT);
    return w;
}

/// `IADD3 dst, a, -b`: integer subtract (dst = a - b), via the srcB negate
/// modifier (bit 63, per NAK's `set_alu_reg(32..40, 62, 63, ..)`).
pub fn isub(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = iadd3(dst, a, b, c);
    setBits(&w, 63, 1, 1); // negate srcB
    return w;
}

/// `IADD3 dst, a, b, RZ` writing a carry-out to predicate `cout` (the low half of
/// a 64-bit add). NAK puts the carry-out predicate at the first result-predicate
/// field (81..83).
pub fn iadd3CarryOut(dst: u8, a: u8, b: u8, cout: u8, c: Control) Inst {
    var w = alu(0x010, dst, a, b, RZ, c);
    setBits(&w, 81, 3, cout); // carry-out predicate
    setBits(&w, 84, 3, PT);
    return w;
}

/// `IADD3.X dst, a, b, RZ` with a carry-in from predicate `cin` (the high half of
/// a 64-bit add). A real predicate in the carry-in source (87..89) selects the
/// extended `.X` form.
pub fn iadd3CarryIn(dst: u8, a: u8, b: u8, cin: u8, c: Control) Inst {
    var w = alu(0x010, dst, a, b, RZ, c);
    setBits(&w, 81, 3, PT);
    setBits(&w, 84, 3, PT);
    setBits(&w, 87, 3, cin); // carry-in predicate
    return w;
}

/// `IMAD dst, a, b, c_in`: 32-bit multiply-add (dst = a*b + c_in). Regular IMAD
/// 0x024. A plain multiply is `imad(dst, a, b, RZ, ...)`. The result-predicate
/// field (81..83) must be PT. Leaving it 0 (P0) is rejected on Blackwell.
pub fn imad(dst: u8, a: u8, b: u8, c_in: u8, c: Control) Inst {
    var w = alu(0x024, dst, a, b, c_in, c);
    setBits(&w, 81, 3, PT); // result predicate = none
    return w;
}

/// `LOP3.LUT dst, a, b, RZ, lut`: bitwise op via a 3-input lookup table. AND/OR/
/// XOR use luts 0xC0/0xFC/0x3C. Regular LOP3 0x012.
pub fn lop3(dst: u8, a: u8, b: u8, lut: u8, c: Control) Inst {
    var w = alu(0x012, dst, a, b, RZ, c);
    setBits(&w, 72, 8, lut);
    setBits(&w, 81, 3, PT); // predicate dst = none
    return w;
}

pub const LUT_AND: u8 = 0xC0;
pub const LUT_OR: u8 = 0xFC;
pub const LUT_XOR: u8 = 0x3C;

/// `PLOP3.LUT dst_pred, p_a, p_b, PT, lut, 0x0`: a predicate-logic op combining two
/// source predicates into a destination predicate (the warp form, NAK OpPLop3 opcode
/// 0x81c). This is how a boolean-valued `&&`/`||`/`^^` (SPIR-V LogicalAnd/Or/NotEqual,
/// which the shared lowering emits as a bool-typed `.binary` bit_and/bit_or/bit_xor)
/// lands in a predicate register - the analogue of LOP3 for integers. The 8-bit LUT
/// (LUT_AND/OR/XOR, with src0=0xF0, src1=0xCC at the top of the 3-input truth table) is
/// split across bits 64..67 (low 3) and 72..77 (high 5) exactly as NAK does. The third
/// predicate source is PT (true, identity for these 2-input ops) and the second
/// predicate dest is PT (none). Predicate sources: p_b@77..80(+not@80),
/// p_a@87..90(+not@90), PT@68..71. Dest@81..84, dst1=PT@84..87.
pub fn plop3(dst_pred: u8, p_a: u8, p_b: u8, lut: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x81c); // PLOP3 warp form (full 12-bit opcode)
    setBits(&w, 68, 3, PT); // src2 predicate = PT (true)
    setBits(&w, 71, 1, 0); // src2 not
    setBits(&w, 77, 3, p_b); // src1 predicate
    setBits(&w, 80, 1, 0); // src1 not
    setBits(&w, 87, 3, p_a); // src0 predicate
    setBits(&w, 90, 1, 0); // src0 not
    setBits(&w, 16, 8, 0); // ops[1].lut (second dest) = 0
    setBits(&w, 64, 3, lut & 0x7); // ops[0].lut low 3
    setBits(&w, 72, 5, lut >> 3); // ops[0].lut high 5
    setBits(&w, 81, 3, dst_pred); // dest predicate
    setBits(&w, 84, 3, PT); // second dest predicate = none
    return w;
}

/// `ISETP.cmp.AND dst_pred, a, b, PT`: set a predicate from an integer compare
/// (dst_pred = (a cmp b)). NAK base 0x08c. Comparison op at 76..78, integer type
/// (signed) at 73, combine-op (AND=0) at 74..75, result predicate at 81..83.
/// Regular ISETP 0x00c.
pub fn isetp(dst_pred: u8, a: u8, b: u8, cmp: Cmp, signed: bool, c: Control) Inst {
    var w = alu(0x00c, RZ, a, b, RZ, c);
    setBits(&w, 73, 1, if (signed) 1 else 0); // I32 vs U32
    setBits(&w, 74, 2, 0); // set-op = AND
    setBits(&w, 76, 3, @intFromEnum(cmp)); // comparison
    setBits(&w, 68, 3, PT); // low compare predicate (unused for 32-bit)
    setBits(&w, 81, 3, dst_pred); // result predicate
    setBits(&w, 84, 3, PT); // second result predicate = none
    setBits(&w, 87, 3, PT); // accumulate predicate = PT
    return w;
}

/// `FSETP dst_pred, a, b, cmp`: floating-point set-predicate (the ORDERED float
/// comparison). NAK OpFSetP opcode 0x00b (via encode_alu): set-op (AND) at 74..76,
/// the FLOAT compare op at 76..80 (4 bits, the ordered codes match `Cmp`'s lt=1..ge=6),
/// ftz at 80, result predicate at 81..84, dst1=None(PT) at 84..87, accum=PT at 87..90.
/// Distinct from `isetp`: the comparison reads the operands as IEEE floats, not as the
/// integer bit-patterns - REQUIRED for `min`/`max`/`clamp` of float values (an integer
/// compare of float bits mis-orders negatives, so `max(0.0, x)` wrongly returns a
/// negative x. The software backend already uses a float compare for float operands).
pub fn fsetp(dst_pred: u8, a: u8, b: u8, cmp: Cmp, c: Control) Inst {
    var w = alu(0x00b, RZ, a, b, RZ, c);
    setBits(&w, 74, 2, 0); // set-op = AND
    setBits(&w, 76, 4, @intFromEnum(cmp)); // FLOAT comparison (ordered: lt=1..ge=6)
    setBits(&w, 80, 1, 0); // ftz off
    setBits(&w, 81, 3, dst_pred); // result predicate
    setBits(&w, 84, 3, PT); // second result predicate = none
    setBits(&w, 87, 3, PT); // accumulate predicate = PT
    return w;
}

/// `SEL dst, a, b, pred`: dst = pred ? a : b. Regular SEL 0x007, predicate at 87..89.
pub fn sel(dst: u8, a: u8, b: u8, pred: u8, c: Control) Inst {
    var w = alu(0x007, dst, a, b, RZ, c);
    setBits(&w, 87, 3, pred);
    return w;
}

/// `SHF.L/R dst, value, shift, RZ`: shift left/right. Regular SHF 0x019, integer
/// type (S32 for an arithmetic right shift) at 73..74, wrap at 75, right at 76.
/// A right shift puts the value in the high (srcC) operand, as the funnel form
/// requires.
pub fn shf(dst: u8, value: u8, shift: u8, right: bool, arithmetic: bool, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 9, 0x019);
    setBits(&w, 9, 3, 1); // register form
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, if (right) RZ else value);
    setBits(&w, 32, 8, shift);
    setBits(&w, 64, 8, if (right) value else RZ);
    setBits(&w, 73, 2, if (arithmetic) 1 else 0); // S32 vs U32
    setBits(&w, 75, 1, 1); // wrap the shift count
    if (right) setBits(&w, 76, 1, 1);
    return w;
}

/// The Volta floating-point ALU ops (FADD/FMUL/FFMA) carry THREE predicate source/
/// result fields that must all be PT (the always-true predicate). Unlike the integer
/// IMAD - which tolerates the 0 (P0) default in the upper two - the FP ops are
/// rejected as an "illegal instruction encoding" (Xid 13, SM warp exception) on
/// Blackwell sm_120 unless 81..83, 84..86, AND 87..89 are all PT. Proven live: a UBO
/// vec4 multiply faulted with only 81..83 set, and drew cleanly once all three were PT.
fn setFpPreds(w: *Inst) void {
    setBits(w, 81, 3, PT);
    setBits(w, 84, 3, PT);
    setBits(w, 87, 3, PT);
}

/// `FADD dst, a, b`: 32-bit float add. NAK base 0x021. Needs the FP predicate fields
/// (see setFpPreds).
pub fn fadd(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = alu(0x021, dst, a, b, RZ, c);
    setFpPreds(&w);
    return w;
}

/// `FADD dst, a, -b`: float subtract (dst = a - b), via the srcB negate modifier.
pub fn fsub(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = fadd(dst, a, b, c);
    setBits(&w, 63, 1, 1); // negate srcB
    return w;
}

/// `FMUL dst, a, b`: 32-bit float multiply. NAK base 0x020. Needs the FP predicate
/// fields (see setFpPreds) AND the PDIV field at bits 84..86 set to 4.
///
/// NAK's `OpFMul::encode` (sm70_encode.rs) does `set_field(84..87, 0x4)` after the
/// generic ALU encode - the "PDIV" field. FADD/FFMA do NOT set it. Leaving bits
/// 84..86 at the PT default (7) that `setFpPreds` writes corrupts the multiply: a
/// bare `FMUL 0.5, 0.5` reads back saturated (~1.0) instead of 0.25 on Blackwell
/// sm_120 (proven by a frame oracle - a fragment that outputs `0.5*0.5` saturates
/// while `0.5+0.0` via FADD reads 0.5 correctly). So override bits 84..86 to 4.
/// This is the field that made every "FMUL on the GPU saturates / cube is white /
/// derivative 22x too large" symptom: any shader doing an FP multiply mis-computed.
pub fn fmul(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = alu(0x020, dst, a, b, RZ, c);
    setFpPreds(&w);
    setBits(&w, 84, 3, 4); // PDIV field = 4 (NAK OpFMul, NOT the PT default)
    return w;
}

/// `FFMA dst, a, b, c_in`: fused multiply-add (dst = a*b + c_in). NAK base 0x023.
/// Needs the FP predicate fields (see setFpPreds).
pub fn ffma(dst: u8, a: u8, b: u8, c_in: u8, c: Control) Inst {
    var w = alu(0x023, dst, a, b, c_in, c);
    setFpPreds(&w);
    return w;
}

// 32-bit int <-> float. NAK encodes the operand sizes as log2(bytes): 4 bytes -> 2.

/// `I2F.F32 dst, src`: convert a 32-bit integer to f32. NAK base 0x106, src
/// signedness@74, dst-size-log2@75, src-size-log2@84.
pub fn i2f(dst: u8, src: u8, src_signed: bool, c: Control) Inst {
    var w = alu(0x106, dst, RZ, src, RZ, c);
    if (src_signed) setBits(&w, 74, 1, 1);
    setBits(&w, 75, 2, 2); // dst = 4 bytes (f32)
    setBits(&w, 84, 2, 2); // src = 4 bytes (i32)
    return w;
}

/// `F2I.S32 dst, src`: convert an f32 to a 32-bit integer, truncating toward zero
/// (C / SPIR-V FToS semantics). NAK base 0x105, dst signedness@72, dst-size-log2@75,
/// round-mode@78 (Zero = 3), src-size-log2@84.
pub fn f2i(dst: u8, src: u8, dst_signed: bool, c: Control) Inst {
    return f2iRound(dst, src, dst_signed, .zero, c);
}

/// The F2I rounding mode (bits 78..80): nearest-even, floor (toward -inf), ceil (toward
/// +inf), or zero (truncate). floor/ceil + an i2f back implement GLSL floor()/ceil() on the
/// integer-representable range (the rounding ops have no direct F32->F32 instruction here).
pub const F2IRound = enum(u2) { nearest = 0, floor = 1, ceil = 2, zero = 3 };

/// Like `f2i` but with an explicit rounding mode (floor/ceil/nearest/zero).
pub fn f2iRound(dst: u8, src: u8, dst_signed: bool, mode: F2IRound, c: Control) Inst {
    var w = alu(0x105, dst, RZ, src, RZ, c);
    if (dst_signed) setBits(&w, 72, 1, 1);
    setBits(&w, 75, 2, 2); // dst = 4 bytes (i32)
    setBits(&w, 78, 2, @intFromEnum(mode));
    setBits(&w, 84, 2, 2); // src = 4 bytes (f32)
    return w;
}

/// The multifunction-unit (MUFU) operation selector (NAK SM70 MuFuOp, bits 74..80).
pub const MuFuOp = enum(u6) { cos = 0, sin = 1, exp2 = 2, log2 = 3, rcp = 4, rsq = 5, sqrt = 8 };

/// `MUFU.op dst, src`: a transcendental on the special-function unit (reciprocal,
/// reciprocal-sqrt, sqrt, sin/cos/exp2/log2). NAK base 0x108, the operand in srcB
/// (bit 32, register form), the op selector at 74..80, F32 type (bit 72 = 0). Used to
/// lower `inversesqrt` (RSQ), `sqrt` (SQRT) and a float reciprocal (RCP, for FP
/// divide a/b = a * RCP(b)). Variable latency on the SFU pipe? No - MUFU is fixed
/// latency (a few cycles). The default stall covers a back-to-back dependency.
pub fn mufu(dst: u8, src: u8, mfop: MuFuOp, c: Control) Inst {
    var w = alu(0x108, dst, RZ, src, RZ, c);
    setBits(&w, 72, 1, 0); // op_type = F32
    setBits(&w, 74, 6, @intFromEnum(mfop)); // MUFU op selector
    return w;
}

/// `LDC dst, c[bank][offset]`: load a 32-bit value from a constant bank (the
/// kernel-parameter ABI loads inputs this way). NAK opcode 0xb82: dst@16, dynamic
/// offset register@24 (RZ = static), 16-bit immediate offset@38, bank index@54,
/// mem type B32@73. Fixed latency on the constant cache.
pub fn ldc(dst: u8, bank: u5, offset: u16, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0xb82);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, RZ); // no dynamic offset
    setBits(&w, 38, 16, offset);
    setBits(&w, 54, 5, bank);
    setBits(&w, 73, 3, 4); // B32
    return w;
}

/// `LDG.E dst, [addr:addr+1]`: load a 32-bit value from the 64-bit global address
/// in the register pair (addr, addr+1). Volta LDG 0x981, mirrors prism's STG.
pub fn ldgU32(dst: u8, addr: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x981);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, addr);
    setBits(&w, 64, 8, RZ); // URZ uniform base
    setBits(&w, 72, 1, 1); // 64-bit uniform
    setBits(&w, 73, 3, 4); // type B32
    setBits(&w, 77, 4, 0xa); // STRONG / SYS
    setBits(&w, 84, 3, 1); // eviction NORMAL
    setBits(&w, 90, 1, 1); // 64-bit GPR address
    setBits(&w, 91, 1, 1); // UGPR mode
    return w;
}

/// `STG.E.STRONG.SYS [addr:addr+1], data`: store a 32-bit GPR to the 64-bit
/// global address in (addr, addr+1). Verified bit-for-bit on hardware (prism).
pub fn stgU32(addr: u8, data: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x986);
    setBits(&w, 24, 8, addr);
    setBits(&w, 90, 1, 1); // 64-bit GPR address
    setBits(&w, 64, 8, RZ); // URZ uniform base
    setBits(&w, 72, 1, 1); // 64-bit uniform
    setBits(&w, 32, 8, data);
    setBits(&w, 73, 3, 4); // type B32
    setBits(&w, 77, 4, 0xa); // STRONG / SYS
    setBits(&w, 84, 3, 1); // eviction NORMAL
    setBits(&w, 91, 1, 1); // UGPR mode (required or the SM traps)
    return w;
}

/// `S2R dst, sysval`: read a special register (thread/block id, etc.). Variable
/// latency: set a `wr_barrier` and drain it before use. Verified (prism).
pub fn s2r(dst: u8, sysval: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x919);
    setBits(&w, 16, 8, dst);
    setBits(&w, 72, 8, sysval);
    return w;
}

/// `BRA target`: relative branch. The branch's taken CONDITION is `Control.pred`
/// (+ `Control.pred_neg` = branch on !pred). PT (the default) = an unconditional
/// branch. `delta` is NAK's relative offset in 32-bit WORD units (4 words per
/// 128-bit instruction): `target_ip - cur_ip - 4` = `(dst_instr - cur_instr - 1)*4`.
/// Volta BRA 0x947.
///
/// NAK (sm70_encode.rs OpBra) puts the taken condition at the PREDICATE-SOURCE
/// field `set_pred_src(87..90, 90, cond)` - bits 87..89 hold the predicate register
/// and bit 90 is the negate - NOT the instruction guard at 12..14 (a guard-false
/// BRA would mis-execute on Blackwell). The instruction guard stays PT so the BRA
/// always issues. Whether it is TAKEN is decided by the 87..89 predicate. On
/// sm>=100 the relative offset is SPLIT across two fields: the low 8 bits at
/// `16..24` and the high 48 bits at `34..82` (`set_rel_offset2(16..24, 34..82)`).
pub fn bra(delta: i32, c: Control) Inst {
    // The instruction guard (12..14) must be PT: the BRA always issues, the
    // 87..89 condition decides whether it is taken. Force the guard to PT
    // regardless of what was passed in `Control.pred`.
    var guard = c;
    guard.pred = PT;
    guard.pred_neg = false;
    var w = base(guard);
    setBits(&w, 0, 12, 0x947);
    setBits(&w, 32, 1, 0); // !.U (this is a regular, non-uniform BRA)
    // Taken-condition predicate at 87..89 (+ negate at 90), like NAK's set_pred_src.
    setBits(&w, 87, 3, c.pred);
    if (c.pred_neg) setBits(&w, 90, 1, 1);
    // The relative offset, SPLIT: low 8 bits at 16..24, high 48 bits at 34..82.
    // The combined field is 56-bit SIGNED. Sign-extend `delta` into the high half
    // so a backward branch (negative delta) sets the high bits correctly.
    const off: u64 = @as(u64, @bitCast(@as(i64, delta))) & ((@as(u64, 1) << 56) - 1);
    setBits(&w, 16, 8, off & 0xff);
    setBits(&w, 34, 48, off >> 8);
    return w;
}

/// `EXIT`: terminate the warp. Verified (prism).
pub fn exit(c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x94d);
    setBits(&w, 87, 3, 7);
    return w;
}

/// KIL: discard the current fragment (OpKill). Opcode 0x95b with the pred-source at
/// 87..90 = PT (unconditional at this point; a conditional `if (cond) discard` is
/// gated by the surrounding structured control flow). From NAK's SM70Op for OpKill
/// (sm70_encode.rs: set_opcode(0x95b) + set_pred_src(87..90, 90, True)).
pub fn kil(c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x95b);
    setBits(&w, 87, 3, 7);
    return w;
}

// Convergence barriers (Volta+ structured control flow). On Volta and later the
// warp does not implicitly reconverge at the end of a divergent `if`: a divergent
// branch SPLITS the warp into independently-scheduled sub-warps, and any quad-
// dependent op afterwards (a TEX texture fetch or a derivative SHFL, which both
// rely on all four lanes of a 2x2 pixel quad being active in lock-step) reads
// garbage from the lanes that took the other path. NAK (Mesa) wraps every
// divergent control-flow region in a hardware convergence barrier: BSSY sets a
// reconvergence point at a barrier register before the branch, and BSYNC at the
// join forces the sub-warps to rendezvous there, restoring quad uniformity before
// the next quad op. These three encoders mirror NAK's sm70_encode.rs byte-for-byte
// (the encoding is shared Volta..Blackwell). The convergence-barrier registers
// (Bar regs B0..B15) are a register file SEPARATE from the GPRs.

/// `BCLEAR Bbar`: clear/initialize a convergence-barrier register. NAK opcode
/// 0x355 with the .CLEAR bit (84). The GPR dst field (16..24) is RZ (unused), the
/// barrier reg index goes in bits 24..28.
pub fn bclear(bar: u4, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x355);
    setBits(&w, 16, 8, RZ); // set_dst(None) -> RZ
    setBits(&w, 24, 4, bar); // set_bar_dst(24..28)
    setBits(&w, 84, 1, 1); // .CLEAR
    return w;
}

/// `BSSY Bbar, target`: set up a convergence barrier at `target` (the join/
/// reconvergence point). Emitted just before a divergent branch. NAK opcode 0x945:
/// barrier-dst@16..20, a 30-bit relative offset to the target@34..64, an
/// unconditional predicate (PT)@87..90 (+ not-bit 90 = 0). `delta` is the byte
/// offset from the next instruction (the same convention as `bra`).
pub fn bssy(bar: u4, delta: i32, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x945);
    setBits(&w, 16, 4, bar); // set_bar_dst(16..20)
    setBits(&w, 34, 30, @as(u30, @truncate(@as(u32, @bitCast(delta))))); // set_rel_offset(34..64)
    setBits(&w, 87, 3, PT); // set_pred_src(87..90, 90, True)
    return w;
}

/// `BSYNC Bbar`: reconverge the warp at the barrier set up by a prior BSSY.
/// Emitted at the join point. NAK opcode 0x941: barrier-src@16..20, unconditional
/// predicate (PT)@87..90.
pub fn bsync(bar: u4, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x941);
    setBits(&w, 16, 4, bar); // set_bar_src(16..20)
    setBits(&w, 87, 3, PT); // set_pred_src(87..90, 90, True)
    return w;
}

// Graphics I/O: the vertex/fragment shader attribute interface. These encodings
// are verified live on Blackwell by prism's assembler (a passthrough vertex
// shader runs).

/// `ALD dst..dst+comps-1, a[addr]`: load `comps` vertex input-attribute words
/// into consecutive GPRs (a vertex shader reading a fetched attribute). Variable
/// latency: set a write barrier and drain it before the consumer.
pub fn ald(dst: u8, addr: u16, comps: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x321);
    setBits(&w, 16, 8, dst);
    setBits(&w, 32, 8, RZ); // vertex (RZ: not per-vertex addressed)
    setBits(&w, 24, 8, RZ); // dynamic offset (RZ: static)
    setBits(&w, 40, 10, addr);
    setBits(&w, 74, 2, comps - 1);
    return w;
}

/// `AST o[addr], data..data+comps-1`: store `comps` GPRs to a shader output
/// attribute (e.g. the clip-space position at ATTR_POSITION).
pub fn ast(addr: u16, data: u8, comps: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x322);
    setBits(&w, 32, 8, data);
    setBits(&w, 64, 8, RZ); // vertex
    setBits(&w, 24, 8, RZ); // dynamic offset
    setBits(&w, 40, 10, addr);
    setBits(&w, 74, 2, comps - 1);
    return w;
}

/// `IPA dst, a[addr]`: interpolate one component of a fragment input attribute.
/// On SM70+ a single IPA does the full perspective-correct interpolation
/// implicitly. `addr` is the attribute byte address (4-aligned). The encoder
/// stores addr>>2. Variable latency like ALD.
pub fn ipa(dst: u8, addr: u16, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x326);
    setBits(&w, 16, 8, dst);
    setBits(&w, 64, 8, addr >> 2); // attribute addr / 4
    setBits(&w, 76, 2, 0); // loc = Default
    setBits(&w, 78, 2, 0); // freq = Pass (implicit perspective)
    setBits(&w, 32, 8, RZ); // offset reg = RZ (required for Default loc)
    setBits(&w, 81, 3, PT); // pred_dst = none
    return w;
}

/// `IPA.CONSTANT dst, a[addr]`: read a FLAT fragment attribute (no interpolation).
/// Same opcode as `ipa` but with the interp frequency field (bits 78..80) set to
/// Constant(1): the value is taken as-is from the attribute (NAK emits this for
/// gl_FrontFacing and other flat sysval reads). The interp MODE (Constant vs
/// ScreenLinear vs Perspective) is not in the instruction, it lives in the SPH imap.
pub fn ipaConstant(dst: u8, addr: u16, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x326);
    setBits(&w, 16, 8, dst);
    setBits(&w, 64, 8, addr >> 2); // attribute addr / 4
    setBits(&w, 76, 2, 0); // loc = Default
    setBits(&w, 78, 2, 1); // freq = Constant (flat, no interpolation)
    setBits(&w, 32, 8, RZ); // offset reg = RZ
    setBits(&w, 81, 3, PT); // pred_dst = none
    return w;
}

// Screen-space derivatives (dFdx/dFdy). The GPU shades fragments in 2x2 pixel
// QUADS whose four lanes are co-resident in the warp. A derivative is the
// difference between a fragment and its quad neighbour, read directly from the
// neighbour's register via a warp SHUFFLE within the quad segment. NAK lowers
// nir_op_fddx/fddy to SHFL.BFLY (XOR the lane index with 1 for the horizontal
// neighbour, 2 for the vertical) then FSWZADD, which combines the shuffled
// neighbour and self into the per-quad gradient with the correct per-lane sign.

/// The SHFL quad-shuffle `c` operand NAK passes for a quad derivative: the low 5
/// bits are the segment width minus one in the low byte (0x03 = a 4-lane quad
/// segment) and the upper byte (0x1c << 8) is the clamp/bound. Matches NAK's
/// `0x3 | (0x1c << 8)` for fddx/fddy.
pub const SHFL_QUAD_C: u16 = 0x1c03;

/// `SHFL.BFLY dst, src, lane=imm, c=imm`: butterfly warp shuffle - read the
/// register `src` from the lane whose index is (this_lane XOR `lane_xor`). With
/// `lane_xor` = 1 the source is the horizontal quad neighbour, 2 the vertical.
/// `c` packs the quad segment (SHFL_QUAD_C). Both lane and c are immediates, so
/// NAK's all-immediate SHFL form (opcode 0xf89): imm_c at 40..53, imm_lane at
/// 53..58, the BFLY op (3) at 58..60, src at 24, dst at 16. Fixed latency, but it
/// reads `src` from a neighbour lane, so the scheduler waits on `src`'s producer
/// (the IPA that interpolated the varying) exactly like any srcA read.
pub fn shflBflyQuad(dst: u8, src: u8, lane_xor: u5, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0xf89); // SHFL, lane imm + c imm form
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, src);
    setBits(&w, 40, 13, SHFL_QUAD_C); // imm_c (segment/clamp)
    setBits(&w, 53, 5, lane_xor); // imm_lane (XOR mask)
    setBits(&w, 58, 2, 3); // op = BFLY
    setBits(&w, 81, 3, PT); // in_bounds pred dst = none (PT)
    return w;
}

/// `FSWZADD dst, src0, src1`: the quad swizzle-add that finishes a derivative.
/// `src0` is the SHFL'd neighbour, `src1` is self. The four 2-bit lane ops in
/// `lane_ops` tell each quad lane whether to compute `src1 - src0` (SubRight),
/// `src0 - src1` (SubLeft), or add. NAK uses [SubLeft,SubRight,SubLeft,SubRight]
/// for dFdx and [SubLeft,SubLeft,SubRight,SubRight] for dFdy, giving every lane in
/// the quad the same (coarse) gradient with the correct sign. NAK SM70 opcode
/// 0x822: src0 at 24, src1 at 64, the packed sub-op at 32..40, the non-divergent
/// derivative mode at bit 77 (required on sm>=100 / Blackwell), round-to-nearest
/// at 78..80, ftz off at 80.
pub const SwzOp = enum(u2) { add = 0, sub_left = 1, sub_right = 2, move_left = 3 };
pub fn fswzadd(dst: u8, src0: u8, src1: u8, lane_ops: [4]SwzOp, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x822);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, src0);
    setBits(&w, 64, 8, src1);
    // Pack the four lane ops: op[i] occupies bits ((len-1-i)*2) of the sub-op byte,
    // exactly NAK's `subop |= swz_op << ((ops.len()-i-1)*2)`.
    var subop: u8 = 0;
    inline for (lane_ops, 0..) |o, i| subop |= @as(u8, @intFromEnum(o)) << ((3 - i) * 2);
    setBits(&w, 32, 8, subop);
    setBits(&w, 77, 1, 1); // deriv_mode = NonDivergent (sm>=100, fswzadd.ndv)
    setBits(&w, 78, 2, 0); // round mode = nearest-even
    setBits(&w, 80, 1, 0); // ftz off
    return w;
}

/// `TEX dst..dst+3, [coord:coord+1], handle`: a bindless 2D texture sample. On
/// Blackwell (sm_100/sm_120) the bindless TEX opcode is 0xd61 with the bindless
/// marker at bit 91 (NAK SM70+ encoder, the e.sm >= 100 path). The result RGBA goes
/// to dst..dst+3 (channel mask 0xf). `coord` is the first of a consecutive register
/// PAIR holding (u, v) as f32 in normalized [0,1] coords. The hardware reads the
/// pair from the bit-24 source. `handle` is a 32-bit register holding the bindless
/// texture handle = (TIC index & 0xfffff) | (TSC index << 20). The GPU indexes the
/// bound TEX_HEADER_POOL / TEX_SAMPLER_POOL with it. LOD is forced to level 0
/// (TexLodMode::Zero), so no screen-space derivatives are needed - correct for a
/// single-mip sampled image (the dFdx/dFdy auto-LOD path is a later milestone).
/// Variable latency: the result lands an unknown number of cycles after issue, so
/// the scheduler sets a write barrier and waits before any consumer reads dst.
/// MUFU (multifunction/special-function unit) opcode as it appears in the encoded
/// instruction's low 12 bits: `alu()` ORs the NAK base 0x108 with the register-srcB
/// form bit (1 << 9 = 0x200), so the scheduler sees 0x308. Variable-latency
/// (decoupled) on sm120, so the scheduler scoreboards it (schedule.isVariableLatency).
pub const MUFU_OPCODE: u32 = 0x308;

pub const TEX_OPCODE: u32 = 0xd61;
/// The uniform zero register (URZ) on sm>=100: NAK's `zero_reg(UGPR) = ugpr_max() =
/// 255` for Blackwell. The bindless TEX's uniform handle/offset operand fields MUST
/// reference URZ, not uniform register 0 (which is garbage) - leaving them 0 faults the
/// SM (Xid 13, "Graphics SM Global Exception").
pub const URZ: u8 = 255;
/// The NAK-style texture DIMENSION field (bits 61-63): selects how many coordinate
/// registers the HW reads from `coord` and the texture target. `_2D` (=1) is verified by the
/// working 2D texture path; `_3D` (=2) is verified by a per-slice 3D readback. `cube` (=3) is NOT a
/// HW target here - it is an INTERNAL MARKER the isel uses to trigger the cube lowering: the native
/// Blackwell cube TEX modes do not select the face from the direction (dim=3 ignores it, dim=4/5/6
/// return the border, dim=7/ARRAY_CUBE reads a stale array layer non-deterministically), so a cube
/// sample is lowered to the major-axis (direction -> face + face u,v) math + a `_2D` sample of a
/// 6-face-wide atlas (u' = (face+u)/6). The emitted TEX therefore uses `dim_2d`, never 3.
pub const TexDim = struct {
    pub const dim_2d: u8 = 1;
    pub const dim_3d: u8 = 2;
    pub const cube: u8 = 3;
    /// A 2D ARRAY (NAK set_tex_dim Array2D = 5): the HW reads a 3-register coord (layer, u, v) with
    /// the LAYER FIRST (a raw index, not normalized). Used by `sampler2DArray`; needs a TWO_D_ARRAY
    /// TIC. Verified on the RTX 5070.
    pub const array_2d: u8 = 5;
};

/// A bindless 2D texture sample (see `tex`): dim = _2D, a coord PAIR.
pub fn tex2d(dst: u8, coord: u8, handle: u8, c: Control) Inst {
    return tex(dst, coord, handle, TexDim.dim_2d, c);
}

/// `TEX dst..dst+3, [coord..], handle` with an explicit DIMENSION. `dim` selects the texture
/// target + the coordinate register count (2D = coord pair, 3D/cube = coord triple). All other
/// fields match the verified 2D encoding.
pub fn tex(dst: u8, coord: u8, handle: u8, dim: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, TEX_OPCODE); // TEX (bindless, sm>=100)
    setBits(&w, 91, 1, 1); // bindless marker (e.set_bit(91, true))
    // On Volta+ (incl. Blackwell) the 4-channel TEX RESULT IS SPLIT ACROSS TWO
    // destination register groups (NAK from_nir: dst_comps > 2 -> dsts[0]=dst[0..2],
    // dsts[1]=dst[2..]). dst[0] (bits 16:23) holds components 0,1 (R,G) at dst..dst+1,
    // dst[1] (bits 64:71) holds components 2,3 (B,A) at dst+2..dst+3. For a consecutive
    // RGBA block at `dst` that means dst[0]=dst (R,G) and dst[1]=dst+2 (B,A). Putting RZ
    // in dst[1] (the natural "no second dest") DISCARDS B and A - the channels then read
    // back as uninitialized garbage, which is the position-dependent B/A corruption.
    setBits(&w, 16, 8, dst); // dst[0] = R,G at dst, dst+1
    setBits(&w, 64, 8, dst + 2); // dst[1] = B,A at dst+2, dst+3
    setBits(&w, 81, 3, PT); // fault predicate dst = none (PT)
    setBits(&w, 24, 8, coord); // src[0] = coord register pair (u in coord, v in coord+1)
    setBits(&w, 32, 8, handle); // src[1] = the bindless handle register
    // sm>=100 ONLY: the TEX also has two UNIFORM-register operands (the uniform handle
    // at bit 40 and the uniform offset at bit 48), each 8 bits. NAK sets both to the
    // uniform zero register (URZ = 255). Leaving them 0 makes the TEX read uniform
    // register 0 as the handle/offset = garbage -> the SM faults (Xid 13). This is the
    // load-bearing Blackwell-specific field the from-scratch encoder must replicate.
    setBits(&w, 40, 8, URZ); // uniform handle = URZ
    setBits(&w, 48, 8, URZ); // uniform offset = URZ
    setBits(&w, 60, 1, 0); // .scalar = false
    setBits(&w, 61, 3, dim); // dim = _2D(1) / _3D(2) / cube(3) - selects coord count + target
    setBits(&w, 72, 4, 0xf); // channel_mask = RGBA
    setBits(&w, 76, 2, 0); // deriv_mode = Auto (sm>=100 set_tex_deriv_mode 76..78)
    setBits(&w, 84, 3, 1); // mem eviction priority = Normal (84..87)
    // lod_mode = Auto (0): set_tex_lod_mode2(59..60, 87..90). bit59=0, bits87..89=0. The HW
    // computes the LOD from the 2x2 fragment quad's screen-space texture-coordinate derivatives
    // (implicit LOD), so a mipmapped texture (TIC MAX_MIP_LEVEL>0 + TSC MIP_POINT/LINEAR) minifies
    // through its chain. A single-mip texture (MAX_MIP_LEVEL=0) clamps to level 0 regardless, so
    // Auto is correct there too (the computed LOD just resolves to the only level).
    setBits(&w, 59, 1, 0);
    setBits(&w, 87, 3, 0);
    return w;
}

/// `TEX.LL` - a texture sample with an EXPLICIT LOD (GLSL textureLod / textureCubeLod). This is the
/// SAME opcode as `tex` (OpTex 0xd61 - NAK emits OpTex for textureLod/txl; OpTld/0xd67 is texelFetch,
/// integer coords) but with lod_mode = Lod: (1) lod_mode via NAK set_tex_lod_mode2(59..60, 87..90) =
/// value 3, split low-bits-first -> bit 59 = 1, bits 87..89 = 1; (2) CRUCIAL on Blackwell (sm>=120):
/// NAK forces deriv_mode = DerivXY (=3, bits 76..77) whenever lod_mode != Zero - leaving it Auto (the
/// `tex` default) makes the explicit LOD flaky/ignored. The HW takes the LOD from the coord register
/// just past the `dim` spatial coords (a 2D sample reads coord, coord+1 = u,v and coord+2 = f32 LOD).
pub fn texLod(dst: u8, coord: u8, handle: u8, dim: u8, c: Control) Inst {
    var w = tex(dst, coord, handle, dim, c);
    setBits(&w, 59, 1, 1); // lod_mode2 low bit: Lod(3) & 1
    setBits(&w, 87, 3, 1); // lod_mode2 high bits: Lod(3) >> 1
    setBits(&w, 76, 2, 3); // deriv_mode = DerivXY (sm>=120 requires it for lod_mode != Zero)
    return w;
}

/// `TEX.SCR.Z` - a DEPTH-COMPARE texture sample (GLSL sampler2DShadow / SPIR-V OpImageSampleDref). Same
/// OpTex opcode as `tex` but with z_cmpr set (sm70 encoder bit 78, NAK sm70_encode.rs): the HW compares
/// the shader-supplied reference (dref) against the fetched depth using the TSC's DEPTH_COMPARE_FUNC and
/// returns a SINGLE scalar pass fraction (0/1, or PCF-blended) rather than a filtered RGBA vec4. The dref
/// is a src1 register RIGHT AFTER the handle (src1 = [handle, dref]; NAK nak_nir_lower_tex.c packs z_cmpr
/// last in src1, so with no lod/offset it is handle_reg + 1 - the caller must place it there). Because the
/// result is scalar, channel_mask = R only and there is no second destination (dst[1] = RZ), so only `dst`
/// is written (the scheduler spans the write barrier by the channel-mask popcount). LOD is Auto (implicit
/// fragment-quad derivatives), which is correct for a base-level shadow map.
pub fn texShadow(dst: u8, coord: u8, handle: u8, dim: u8, c: Control) Inst {
    var w = tex(dst, coord, handle, dim, c);
    setBits(&w, 78, 1, 1); // z_cmpr: depth-compare (dref at src1[1] = handle+1), scalar result
    setBits(&w, 64, 8, RZ); // no second destination (1-component scalar result)
    setBits(&w, 72, 4, 0x1); // channel_mask = R only (the compare pass fraction)
    return w;
}

/// The bindless texture-GATHER opcode on Blackwell (sm>=100): NAK OpTld4 (`e.sm >= 100 ->
/// set_opcode(0xd64) + set_bit(91)`). Distinct from TEX (0xd61): TLD4 fetches ONE component of the
/// 4 bilinear-footprint texels rather than a filtered sample.
pub const TLD4_OPCODE: u32 = 0xd64;

/// `TLD4 dst..dst+3, [coord:coord+1], handle` - a bindless texture GATHER (GLSL textureGather).
/// Returns the single component `comp` (0..3) of the 4 texels of the bilinear footprint at (u,v) as
/// the RGBA result, in the GL gather order (lower-left, lower-right, upper-right, upper-left). The
/// coord / handle / dst-split layout matches `tex` (dst[0]=R,G at 16..24; dst[1]=B,A at 64..72;
/// coord pair at 24..32; bindless handle at 32..40; URZ uniform operands; bindless marker bit 91).
/// The component select occupies bits 87..88 (NAK `set_field(87..89, comp)`) - the same bits `tex`
/// uses for lod_mode2 high, repurposed since a gather has no LOD mode. offset_mode = None (76..78=0);
/// the LOD is implicit (the 2x2-quad derivatives), which is correct for a fragment-stage gather.
pub fn tld4(dst: u8, coord: u8, handle: u8, dim: u8, comp: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, TLD4_OPCODE); // TLD4 (bindless, sm>=100)
    setBits(&w, 91, 1, 1); // bindless marker
    setBits(&w, 16, 8, dst); // dst[0] = R,G at dst, dst+1
    setBits(&w, 64, 8, dst + 2); // dst[1] = B,A at dst+2, dst+3
    setBits(&w, 81, 3, PT); // fault predicate dst = none (PT)
    setBits(&w, 24, 8, coord); // src[0] = coord pair (u in coord, v in coord+1)
    setBits(&w, 32, 8, handle); // src[1] = bindless handle register
    setBits(&w, 40, 8, URZ); // uniform handle = URZ (sm>=100)
    setBits(&w, 48, 8, URZ); // uniform offset = URZ (sm>=100)
    setBits(&w, 60, 1, 0); // .scalar = false
    setBits(&w, 61, 3, dim); // dim (_2D)
    setBits(&w, 72, 4, 0xf); // channel_mask = RGBA (the 4 gathered texels)
    setBits(&w, 76, 2, 0); // offset_mode = None
    setBits(&w, 84, 3, 1); // mem eviction priority = Normal (84..87)
    setBits(&w, 87, 2, comp); // gather component (0..3)
    return w;
}

/// The bindless texel-FETCH opcode on Blackwell (sm>=100): NAK OpTld (`e.sm >= 100 -> set_opcode(
/// 0xd67) + set_bit(91)`). texelFetch: an exact texel at INTEGER coords + an explicit LOD, no filter.
pub const TLD_OPCODE: u32 = 0xd67;

/// `TLD dst..dst+3, [coord:coord+1], [handle:lod]` - a bindless texel FETCH (GLSL texelFetch). The
/// coord registers hold INTEGER texel coords (x in coord, y in coord+1), NOT normalized floats. Like
/// TEX.LL it uses Lod mode (an explicit LOD): the HW reads the LOD from src1[1] = handle_reg + 1, so
/// `handle` is the first of a consecutive (handle, lod) pair. Same dst-split + URZ operands as `tex`.
/// offset_mode = None at bits 56..58 (sm>=100). No filtering / no normalization (integer fetch).
pub fn tld(dst: u8, coord: u8, handle: u8, dim: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, TLD_OPCODE); // TLD (bindless, sm>=100)
    setBits(&w, 91, 1, 1); // bindless marker
    setBits(&w, 16, 8, dst); // dst[0] = R,G at dst, dst+1
    setBits(&w, 64, 8, dst + 2); // dst[1] = B,A at dst+2, dst+3
    setBits(&w, 81, 3, PT); // fault predicate dst = none (PT)
    setBits(&w, 24, 8, coord); // src[0] = INTEGER coord pair (x in coord, y in coord+1)
    setBits(&w, 32, 8, handle); // src[1] = handle; the LOD is at handle+1 (Lod mode)
    setBits(&w, 40, 8, URZ); // uniform handle = URZ (sm>=100)
    setBits(&w, 48, 8, URZ); // uniform offset = URZ (sm>=100)
    setBits(&w, 56, 2, 0); // offset_mode = None (sm>=100: bits 56..58)
    setBits(&w, 60, 1, 0); // .scalar = false
    setBits(&w, 61, 3, dim); // dim (_2D)
    setBits(&w, 72, 4, 0xf); // channel_mask = RGBA
    setBits(&w, 84, 3, 1); // mem eviction priority = Normal (84..87)
    // lod_mode = Lod (explicit): set_tex_lod_mode2(59..60, 87..90) = 3 -> bit 59 = 1, bits 87..89 = 1.
    setBits(&w, 59, 1, 1);
    setBits(&w, 87, 3, 1);
    return w;
}

/// Shader attribute addresses: the clip-space position output, and the first
/// generic varying / vertex input. System-value index for the vertex id.
pub const ATTR_POSITION: u16 = 0x70;
pub const ATTR_GENERIC0: u16 = 0x80;
/// The vertex-id special register (Volta SV_VERTEXID). For a non-indexed draw with
/// SET_VERTEX_ID_BASE = 0 this is exactly Vulkan's gl_VertexIndex. A vertex shader
/// that pulls its vertices from a UBO array sources gl_VertexIndex from here (S2R).
pub const SR_VERTEX_ID: u8 = 0x2f;
/// The instance-id special register (Volta SV_INSTANCEID) = gl_InstanceIndex with
/// SET_GLOBAL_BASE_INSTANCE_INDEX = 0.
pub const SR_INSTANCE_ID: u8 = 0x2e;

/// The Data-Assembler-delivered vertex-id / instance-id ATTRIBUTE addresses
/// (NAK_ATTR_VERTEX_ID / NAK_ATTR_INSTANCE_ID). On Volta+ a vertex shader reads
/// gl_VertexIndex / gl_InstanceIndex from the attribute interface (ALD), NOT from
/// a special register: the fixed-function DA writes the per-vertex id into the
/// attribute RAM at these addresses (this is what NAK emits for SystemValue
/// VertexId/InstanceId). The SPH must declare it consumes the sysval (imap_sys).
/// SET_DA_OUTPUT vertex_id_uses_array_start + SET_VERTEX_ID_BASE = 0 (set in the
/// draw-state init) make the delivered value Vulkan's gl_VertexIndex for a
/// non-indexed draw.
pub const ATTR_VERTEX_ID: u16 = 0x2fc;
pub const ATTR_INSTANCE_ID: u16 = 0x2f8;

/// gl_FrontFacing's attribute address (NAK_ATTR_FRONT_FACE). A fragment shader reads
/// it as a FLAT (constant-frequency) attribute: the raster delivers a nonzero value
/// for a front-facing primitive, zero for back. The SPH must declare it via the
/// sysval imap (imap_system_values_c bit for a[0x3fc]).
pub const ATTR_FRONT_FACE: u16 = 0x3fc;

/// gl_PointCoord's attribute address (NAK_ATTR_POINT_SPRITE_S/T). A fragment shader
/// reads the point-sprite s coord at a[0x2e0] and t at a[0x2e4], IPA'd with normal
/// (perspective-free SCREEN_LINEAR) frequency across the sprite quad. The SPH must
/// declare these two inputs via the imap (IMAP_POINT_SPRITE_S/T, MW bits 344/345) and
/// the draw-state must enable SET_POINT_SPRITE + SET_POINT_SPRITE_SELECT (done once).
pub const ATTR_POINT_SPRITE: u16 = 0x2e0;

/// The constant bank a graphics shader reads its bound UBO base addresses + bindless texture
/// handles from. It is the HW ROOT TABLE 1 (root table T is exposed as constant bank
/// `graphics_root_table_first_cb + T`), NOT a bound external cbuf. On Blackwell the old
/// LOAD_CONSTANT_BUFFER -> LDC c[0] path is NOT COHERENT at high TPC occupancy (a tall render
/// target lights up more TPCs; the LDC returns 0 -> the uniform LDG faults @ 0x0). The HW root
/// table (written via SET_ROOT_TABLE_SELECTOR + LOAD_ROOT_TABLE) IS coherent - this is nvk's
/// Blackwell path. See [[prism-glmark2-perf-cliff]]. The dispatch side (prism's draw path) writes
/// each bound UBO's 64-bit VA / each sampler's handle into ROOT TABLE 1 at `graphics_ubo_cb_base +
/// slot*8`; the shader prologue loads the pair back with two LDCs from `graphics_const_bank`.
pub const graphics_root_table: u3 = 1;
pub const graphics_root_table_first_cb: u5 = 24; // NVK_HW_ROOT_TABLE_FIRST_CB: root table T = c[24+T]
pub const graphics_const_bank: u5 = graphics_root_table_first_cb + graphics_root_table; // c[25]

/// Root-table-1 BYTE offset where the UBO base addresses / texture handles live (SET_ROOT_TABLE_
/// SELECTOR's offset field is 8-bit, so this + slot*8 for 8 slots must stay < 256). Was 0x140 in the
/// old bound-cb0 scheme; rebased into the 256-byte root table 1.
pub const graphics_ubo_cb_base: u16 = 0x40;

/// Root-table-1 byte offset of the CUBE half-texel (0.5 / face_width, as f32) the dispatch side
/// writes when a cubemap is bound. The cube lowering reads it (LDC c[graphics_const_bank]
/// [cube_halftexel_cb]) to clamp the within-face u to [half_texel, 1 - half_texel] so a LINEAR tap
/// near a face edge stays inside the face's atlas column. Below graphics_ubo_cb_base's per-slot area.
pub const cube_halftexel_cb: u16 = 0x00;

/// Special-register indices for `s2r`.
pub const SR_LANEID: u8 = 0x00; // the warp lane index (0..31). The FIRST special reg on Volta+
pub const SR_TID_X: u8 = 0x21; // threadIdx.x
pub const SR_CTAID_X: u8 = 0x25; // blockIdx.x

test "MOV imm matches the hardware-verified encoding" {
    const w = movImm(2, 0xcafe, .{});
    try std.testing.expectEqual(@as(u32, 0x802), w[0] & 0xfff); // ALU MOV (0x002) form 4
    try std.testing.expectEqual(@as(u32, 0xcafe), w[1]); // immediate in dword 1
    try std.testing.expectEqual(@as(u32, 2), (w[0] >> 16) & 0xff); // dst R2
}

test "STG matches the hardware-verified bits (prism, live on Blackwell)" {
    const w = stgU32(0, 2, .{});
    try std.testing.expectEqual(@as(u32, 0x00007986), w[0]);
    try std.testing.expectEqual(@as(u32, 0x00000002), w[1]);
    try std.testing.expectEqual(@as(u32, 0x0c1149ff), w[2]);
}

test "EXIT matches the hardware-verified opcode" {
    const w = exit(.{ .stall = 1 });
    try std.testing.expectEqual(@as(u32, 0x94d), w[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, PT), (w[0] >> 12) & 0x7); // unconditional
    try std.testing.expectEqual(@as(u32, 1), (w[3] >> 9) & 0xf); // stall = 1 at bit 105
}

test "the guard predicate and control bits land in the right fields" {
    const w = iadd3(3, 1, 2, .{ .pred = 0, .pred_neg = true, .stall = 2, .wait_mask = 0b10 });
    try std.testing.expectEqual(@as(u32, 0x210), w[0] & 0xfff); // IADD3 base 0x010 + reg form
    try std.testing.expectEqual(@as(u32, 0), (w[0] >> 12) & 0x7); // guard predicate P0
    try std.testing.expectEqual(@as(u32, 1), (w[0] >> 15) & 0x1); // negated
    try std.testing.expectEqual(@as(u32, 3), (w[0] >> 16) & 0xff); // dst R3
    try std.testing.expectEqual(@as(u32, 1), (w[0] >> 24) & 0xff); // a R1
    try std.testing.expectEqual(@as(u32, 2), w[1] & 0xff); // b R2 at bit 32
    try std.testing.expectEqual(@as(u32, 2), (w[3] >> 9) & 0xf); // stall = 2
    try std.testing.expectEqual(@as(u32, 0b10), (w[3] >> 20) & 0x3f); // wait_mask at bit 116
}

test "ISETP places the comparison and result predicate (NAK layout)" {
    const w = isetp(0, 1, 2, .lt, true, .{});
    try std.testing.expectEqual(@as(u32, 0x20c), w[0] & 0xfff); // base 0x00c + reg form
    try std.testing.expectEqual(@as(u32, @intFromEnum(Cmp.lt)), (w[2] >> 12) & 0x7); // cmp at bit 76
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> 9) & 0x1); // signed (I32) at bit 73
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> 17) & 0x7); // dst predicate P0 at bit 81
}

test "PLOP3 places the warp-form opcode, the LUT split, and the predicate sources/dest (NAK layout)" {
    // `p2 = p0 AND p1` (dst P2, srcs P0/P1, third src PT). NAK OpPLop3 warp form 0x81c:
    // LUT split bits 64..67 (low 3) + 72..77 (high 5), src0@87..90, src1@77..80, src2(PT)@68..71,
    // dst0@81..84, dst1(PT)@84..87.
    const w = plop3(2, 0, 1, LUT_AND, .{});
    try std.testing.expectEqual(@as(u32, 0x81c), w[0] & 0xfff); // warp-form opcode
    try std.testing.expectEqual(@as(u32, LUT_AND & 0x7), (w[2] >> 0) & 0x7); // lut low 3 at bit 64
    try std.testing.expectEqual(@as(u32, LUT_AND >> 3), (w[2] >> 8) & 0x1f); // lut high 5 at bit 72
    try std.testing.expectEqual(@as(u32, 2), (w[2] >> 17) & 0x7); // dst0 = P2 at bit 81
    try std.testing.expectEqual(@as(u32, PT), (w[2] >> 20) & 0x7); // dst1 = PT at bit 84
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> 13) & 0x7); // src1 (p_b) = P1 at bit 77
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> 23) & 0x7); // src0 (p_a) = P0 at bit 87
    try std.testing.expectEqual(@as(u32, PT), (w[2] >> 4) & 0x7); // src2 = PT at bit 68
}

test "subtract sets the srcB negate modifier (bit 63)" {
    const add = iadd3(3, 1, 2, .{});
    const sub = isub(3, 1, 2, .{});
    try std.testing.expectEqual(@as(u32, 0), (add[1] >> 31) & 0x1); // add: srcB not negated
    try std.testing.expectEqual(@as(u32, 1), (sub[1] >> 31) & 0x1); // sub: bit 63 set (word1 bit 31)
    try std.testing.expectEqual(@as(u32, 0x210), sub[0] & 0xfff); // still IADD3
    try std.testing.expectEqual(@as(u32, 1), (fsub(3, 1, 2, .{})[1] >> 31) & 0x1); // FADD negate too
}

test "int<->float conversions carry size and signedness fields" {
    const to_f = i2f(2, 1, true, .{}); // I2F.F32 R2, R1 (signed i32 -> f32)
    try std.testing.expectEqual(@as(u32, 0x306), to_f[0] & 0xfff); // base 0x106 | reg form
    try std.testing.expectEqual(@as(u32, 1), (to_f[2] >> 10) & 0x1); // src signed at bit 74
    try std.testing.expectEqual(@as(u32, 2), (to_f[2] >> 11) & 0x3); // dst size log2 at bit 75
    try std.testing.expectEqual(@as(u32, 2), (to_f[2] >> 20) & 0x3); // src size log2 at bit 84

    const to_i = f2i(2, 1, true, .{}); // F2I.S32 R2, R1 (f32 -> signed i32)
    try std.testing.expectEqual(@as(u32, 0x305), to_i[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 1), (to_i[2] >> 8) & 0x1); // dst signed at bit 72
    try std.testing.expectEqual(@as(u32, 3), (to_i[2] >> 14) & 0x3); // round-toward-zero at bit 78
}

test "LDG mirrors STG with the load opcode" {
    const w = ldgU32(4, 0, .{ .wr_barrier = 0 });
    try std.testing.expectEqual(@as(u32, 0x981), w[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 16) & 0xff); // dst R4
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> 26) & 0x1); // 64-bit addr (bit 90)
}

test "FSETP encodes the float ordered compare (vs ISETP integer)" {
    // FSETP P0, R6, R7, .gt: a FLOAT (ordered) comparison. NAK OpFSetP opcode 0x00b,
    // the float compare op at 76..80, result predicate at 81..84.
    const w = fsetp(0, 6, 7, .gt, .{});
    try std.testing.expectEqual(@as(u32, 0x20b), w[0] & 0xfff); // base 0x00b | reg form (1<<9)
    try std.testing.expectEqual(@as(u32, 6), (w[0] >> 24) & 0xff); // srcA R6
    try std.testing.expectEqual(@as(u32, 7), (w[1] >> 0) & 0xff); // srcB R7 at bit 32
    try std.testing.expectEqual(@as(u32, @intFromEnum(Cmp.gt)), (w[2] >> (76 - 64)) & 0xf); // float cmp at 76..80
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (81 - 64)) & 0x7); // P0 at 81..84
    // The opcode must differ from ISETP (0x00c) so a float compare is not an integer one.
    try std.testing.expect((w[0] & 0xfff) != (isetp(0, 6, 7, .gt, false, .{})[0] & 0xfff));
}

test "FMUL sets the PDIV field (bits 84..86 = 4), FADD/FFMA do not" {
    // NAK OpFMul::encode does `set_field(84..87, 0x4)` (the PDIV field) AFTER the
    // generic ALU encode. FADD/FFMA never touch it. Leaving it at the PT default (7)
    // that setFpPreds writes corrupts the multiply on Blackwell sm_120 (a bare
    // `FMUL 0.5,0.5` saturates to ~1.0 instead of 0.25, proven by a frame oracle).
    const m = fmul(8, 4, 6, .{});
    try std.testing.expectEqual(@as(u32, 0x220), m[0] & 0xfff); // FMUL base 0x020 | reg form
    try std.testing.expectEqual(@as(u32, 4), (m[2] >> (84 - 64)) & 0x7); // PDIV field = 4
    // FADD/FFMA leave bits 84..86 at the setFpPreds PT default (7), NOT 4.
    const a = fadd(8, 4, 6, .{});
    try std.testing.expectEqual(@as(u32, 7), (a[2] >> (84 - 64)) & 0x7);
    const f = ffma(8, 4, 6, 5, .{});
    try std.testing.expectEqual(@as(u32, 7), (f[2] >> (84 - 64)) & 0x7);
}

test "SHFL.BFLY quad shuffle encodes the NAK fddx/fddy form" {
    // SHFL.BFLY R5, R4, lane=1, c=SHFL_QUAD_C: the horizontal quad neighbour (dFdx).
    const w = shflBflyQuad(5, 4, 1, .{});
    try std.testing.expectEqual(@as(u32, 0xf89), w[0] & 0xfff); // both-immediate SHFL form
    try std.testing.expectEqual(@as(u32, 5), (w[0] >> 16) & 0xff); // dst R5
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // src R4 at bit 24
    try std.testing.expectEqual(@as(u32, SHFL_QUAD_C), (w[1] >> 8) & 0x1fff); // imm_c at bit 40
    try std.testing.expectEqual(@as(u32, 1), (w[1] >> 21) & 0x1f); // imm_lane (XOR 1) at bit 53
    try std.testing.expectEqual(@as(u32, 3), (w[1] >> 26) & 0x3); // op = BFLY (3) at bit 58
    // dFdy uses XOR 2 (the vertical neighbour).
    try std.testing.expectEqual(@as(u32, 2), (shflBflyQuad(5, 4, 2, .{})[1] >> 21) & 0x1f);
}

test "FSWZADD packs the quad lane ops and the Blackwell non-divergent bit" {
    // dFdx ops [SubLeft, SubRight, SubLeft, SubRight] = (1,2,1,2) packed high->low.
    const w = fswzadd(6, 5, 4, .{ .sub_left, .sub_right, .sub_left, .sub_right }, .{});
    try std.testing.expectEqual(@as(u32, 0x822), w[0] & 0xfff); // FSWZADD
    try std.testing.expectEqual(@as(u32, 6), (w[0] >> 16) & 0xff); // dst R6
    try std.testing.expectEqual(@as(u32, 5), (w[0] >> 24) & 0xff); // src0 (shuffled) R5 at 24
    try std.testing.expectEqual(@as(u32, 4), (w[2] >> 0) & 0xff); // src1 (self) R4 at bit 64
    // subop byte at bit 32: op0<<6 | op1<<4 | op2<<2 | op3 = 1<<6|2<<4|1<<2|2 = 0x66.
    try std.testing.expectEqual(@as(u32, 0x66), (w[1] >> 0) & 0xff);
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> 13) & 0x1); // deriv_mode NDV at bit 77
    // dFdy ops [SubLeft, SubLeft, SubRight, SubRight] = 1<<6|1<<4|2<<2|2 = 0x5a.
    try std.testing.expectEqual(@as(u32, 0x5a), (fswzadd(6, 5, 4, .{ .sub_left, .sub_left, .sub_right, .sub_right }, .{})[1]) & 0xff);
}

test "graphics attribute load/store/interpolate (prism-verified layout)" {
    // ALD R0..R3, a[ATTR_GENERIC0]: a 4-component vertex attribute load.
    const a = ald(0, ATTR_GENERIC0, 4, .{ .wr_barrier = 0 });
    try std.testing.expectEqual(@as(u32, 0x321), a[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 0), (a[0] >> 16) & 0xff); // dst R0
    try std.testing.expectEqual(@as(u32, ATTR_GENERIC0), (a[1] >> 8) & 0x3ff); // attr addr at bit 40
    try std.testing.expectEqual(@as(u32, 3), (a[2] >> 10) & 0x3); // comps-1 at bit 74

    // AST o[ATTR_POSITION], R0..R3: write the clip-space position.
    const s = ast(ATTR_POSITION, 0, 4, .{});
    try std.testing.expectEqual(@as(u32, 0x322), s[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, ATTR_POSITION), (s[1] >> 8) & 0x3ff);
    try std.testing.expectEqual(@as(u32, 0), s[1] & 0xff); // data R0 at bit 32

    // IPA R5, a[ATTR_GENERIC0]: interpolate a fragment varying (addr>>2 at bit 64).
    const i = ipa(5, ATTR_GENERIC0, .{ .wr_barrier = 0 });
    try std.testing.expectEqual(@as(u32, 0x326), i[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 5), (i[0] >> 16) & 0xff); // dst R5
    try std.testing.expectEqual(@as(u32, ATTR_GENERIC0 >> 2), i[2] & 0xff);
}
