//! NVIDIA SASS instruction encoder. Encoding is shared across Volta through
//! Blackwell. Only per-instruction latencies differ. Each instruction is 128
//! bits, that is, four little-endian dwords.
//!
//! The bit layout (predicate at bit 12, dst at bit 16, scheduling control
//! from bit 105) and the MOV, STG, LDG, S2R, BRA, and EXIT opcodes are
//! verified live on Blackwell sm_120 by prism's assembler. The remaining ALU
//! opcodes and field offsets come from Mesa's NAK encoder
//! (src/nouveau/compiler/nak/sm70_encode.rs): the 9-bit base opcode is in
//! bits 0..8, with a source form (1 = register) in bits 9..11, and source
//! operands srcA at bit 24, srcB at bit 32, srcC at bit 64. End-to-end
//! hardware confirmation is deferred to prism's GPU dispatch path.

const std = @import("std");

/// The zero general-purpose register (reads 0, writes discarded).
pub const RZ: u8 = 255;
/// The always-true predicate register.
pub const PT: u8 = 7;

/// A 128-bit instruction: four dwords, little-endian.
pub const Inst = [4]u32;

/// Per-instruction scheduling plus the guard predicate. The default `stall`
/// is conservative and covers a back-to-back fixed-latency register
/// dependency. Variable-latency ops (global loads) use `wr_barrier` plus a
/// consumer `wait_mask`.
pub const Control = struct {
    stall: u4 = 15,
    wr_barrier: u3 = 7, // scoreboard to set on completion (7 = none)
    rd_barrier: u3 = 7,
    wait_mask: u6 = 0, // scoreboards to wait on before issue
    /// The operand-reuse bits, 122..125, one per ALU source operand. Bit 122
    /// marks the operand at bits 24..31, bit 123 the SECOND source (bits
    /// 32..39 in form 1, bits 64..71 in form 2), and bit 124 the THIRD source
    /// (bits 64..71 in forms 1 and 4). `markReuse` in schedule.zig sets them.
    reuse_mask: u4 = 0,
    pred: u8 = PT, // guard predicate (PT = unconditional)
    pred_neg: bool = false, // guard on !pred
};

/// The two's-complement low `width` bits of a signed value, for a signed
/// immediate field such as the address displacement of LDG, STG, LDS and STS.
fn signedBits(val: i32, width: usize) u64 {
    const mask: u64 = (@as(u64, 1) << @intCast(width)) - 1;
    return @as(u64, @bitCast(@as(i64, val))) & mask;
}

/// Whether `offset` fits the 24-BIT SIGNED address displacement that LDG, STG,
/// LDS and STS carry at bits 40..63. The instruction selector asks before it
/// folds a constant byte offset into a memory access, because a displacement
/// that does not fit would wrap and address the wrong memory.
pub fn fitsAddrOffset(offset: i64) bool {
    return offset >= -(1 << 23) and offset < (1 << 23);
}

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
    setBits(&w, 122, 4, c.reuse_mask);
    return w;
}

/// Comparison for ISETP / FSETP. Values match NAK's `IntCmpOp` encoding.
pub const Cmp = enum(u3) { lt = 1, eq = 2, le = 3, gt = 4, ne = 5, ge = 6 };

/// A fixed-latency ALU op with a register second source: the 9-bit base
/// opcode in bits 0..8, plus the source form in bits 9..11 (1 = register,
/// as NAK's `encode_alu` does). `dst` is at bit 16, srcA at 24, srcB at 32,
/// srcC at 64.
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

// Integer ALU ops use the regular, non-uniform, base opcodes: the low form,
// with bit 0x080 clear. The uniform-datapath variants (UIADD3 0x090, UIMAD
// 0x0a4, and so on) write uniform registers. Using them with a GPR
// destination is an illegal encoding that faults on Blackwell with "Illegal
// Instruction Encoding". IADD3 (0x010) and IMAD (0x024) are verified live
// on a Blackwell GB20x through prism's SPIR-V compute path. LOP3, ISETP,
// SEL, and SHF follow the same uniform-to-regular rule. The regular float
// ALU ops 0x020-0x023 confirm it.

/// Write one IADD3 carry-in operand: the predicate index at `lo`, three bits
/// wide, and the negate flag at `not_bit`. NAK's `set_pred_src` does this.
fn setCarryIn(w: *Inst, lo: usize, not_bit: usize, pred: u8, negate: bool) void {
    setBits(w, lo, 3, pred);
    setBits(w, not_bit, 1, @intFromBool(negate));
}

/// `IADD3 dst, a, b, RZ`: 32-bit integer add (dst = a + b). Regular IADD3
/// 0x010. To add an immediate, materialize it with `movImm` first.
///
/// Both carry-in sources are set to the constant false, which NAK writes as
/// `!PT`: predicate PT in the field with the negate bit set. NAK's
/// `OpIAdd3::encode` does the same with `set_pred_src(87..90, 90, false)` and
/// `set_pred_src(77..80, 80, false)`. Leaving the fields zero names P0 with no
/// negate, which is a live predicate this backend gives to booleans.
pub fn iadd3(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = alu(0x010, dst, a, b, RZ, c);
    setBits(&w, 81, 3, PT); // carry-out predicate = none
    setBits(&w, 84, 3, PT);
    setCarryIn(&w, 87, 90, PT, true); // carry-in 0 = !PT
    setCarryIn(&w, 77, 80, PT, true); // carry-in 1 = !PT
    return w;
}

/// `IADD3 dst, a, -b`: integer subtract (dst = a - b), via the srcB negate
/// modifier (bit 63, per NAK's `set_alu_reg(32..40, 62, 63, ..)`).
pub fn isub(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = iadd3(dst, a, b, c);
    setBits(&w, 63, 1, 1); // negate srcB
    return w;
}

/// `IADD3 dst, a, b, RZ` writing a carry-out to predicate `cout` (the low
/// half of a 64-bit add). NAK puts the carry-out predicate at the first
/// result-predicate field (81..83).
pub fn iadd3CarryOut(dst: u8, a: u8, b: u8, cout: u8, c: Control) Inst {
    var w = iadd3(dst, a, b, c);
    setBits(&w, 81, 3, cout); // carry-out predicate
    return w;
}

/// `IADD3.X dst, a, b, RZ, cin, !PT` with a carry-in from predicate `cin` (the
/// high half of a 64-bit add).
///
/// BIT 74 IS THE `.X` FLAG, and it is the only thing that makes the hardware
/// read the carry-in at all. NAK's `OpIAdd3X::encode` sets it with
/// `e.set_bit(74, true)` after encoding the same 0x010 opcode `OpIAdd3` uses.
/// Without bit 74 the instruction is a plain IADD3 that ignores bits 77..80
/// and 87..90, so a 64-bit add silently drops every carry out of the low half.
/// `nvdisasm -b SM120` confirms both readings: it prints `IADD3` for the same
/// word with bit 74 clear and `IADD3.X ..., P6, !PT` with it set.
///
/// The second carry-in is the constant false, as NAK leaves it for a plain
/// two-operand extended add.
pub fn iadd3CarryIn(dst: u8, a: u8, b: u8, cin: u8, c: Control) Inst {
    var w = iadd3(dst, a, b, c);
    setBits(&w, 74, 1, 1); // .X: read the carry-in
    setCarryIn(&w, 87, 90, cin, false); // carry-in 0 = cin
    return w;
}

/// `IMAD dst, a, b, c_in`: 32-bit multiply-add (dst = a*b + c_in). Regular
/// IMAD 0x024. A plain multiply is `imad(dst, a, b, RZ, ...)`. The
/// result-predicate field (81..83) must be PT. Leaving it 0 (P0) is rejected
/// on Blackwell.
pub fn imad(dst: u8, a: u8, b: u8, c_in: u8, c: Control) Inst {
    var w = alu(0x024, dst, a, b, c_in, c);
    setBits(&w, 81, 3, PT); // result predicate = none
    return w;
}

/// `LOP3.LUT dst, a, b, RZ, lut`: bitwise op through a 3-input lookup table.
/// AND, OR, and XOR use luts 0xC0, 0xFC, and 0x3C. Regular LOP3 0x012.
pub fn lop3(dst: u8, a: u8, b: u8, lut: u8, c: Control) Inst {
    var w = alu(0x012, dst, a, b, RZ, c);
    setBits(&w, 72, 8, lut);
    setBits(&w, 81, 3, PT); // predicate dst = none
    return w;
}

pub const LUT_AND: u8 = 0xC0;
pub const LUT_OR: u8 = 0xFC;
pub const LUT_XOR: u8 = 0x3C;

/// `PLOP3.LUT dst_pred, p_a, p_b, PT, lut, 0x0`: a predicate-logic op
/// combining two source predicates into a destination predicate (the warp
/// form, NAK OpPLop3 opcode 0x81c). This is how a boolean-valued
/// `&&`/`||`/`^^` (SPIR-V LogicalAnd/Or/NotEqual, which the shared lowering
/// emits as a bool-typed `.binary` bit_and/bit_or/bit_xor) lands in a
/// predicate register, the analogue of LOP3 for integers. The 8-bit LUT
/// (LUT_AND/OR/XOR, with src0=0xF0, src1=0xCC at the top of the 3-input
/// truth table) is split across bits 64..67 (low 3) and 72..77 (high 5),
/// exactly as NAK does. The third predicate source is PT (true, identity
/// for these 2-input ops), and the second predicate dest is PT (none).
/// Predicate sources: p_b at 77..80 (plus not at 80), p_a at 87..90 (plus
/// not at 90), PT at 68..71. Dest at 81..84, dst1=PT at 84..87.
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

/// `ISETP.cmp.AND dst_pred, a, b, PT`: set a predicate from an integer
/// compare (dst_pred = (a cmp b)). NAK base 0x08c. Comparison op at 76..78,
/// integer type (signed) at 73, combine-op (AND=0) at 74..75, result
/// predicate at 81..83. Regular ISETP 0x00c.
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

/// `FSETP dst_pred, a, b, cmp`: floating-point set-predicate (the ordered
/// float comparison). NAK OpFSetP opcode 0x00b (through encode_alu): set-op
/// (AND) at 74..76, the float compare op at 76..80 (4 bits, the ordered
/// codes match `Cmp`'s lt=1..ge=6), ftz at 80, result predicate at 81..84,
/// dst1=None(PT) at 84..87, accum=PT at 87..90. Distinct from `isetp`: the
/// comparison reads the operands as IEEE floats, not as the integer bit
/// patterns. This is required for `min`, `max`, and `clamp` of float
/// values, since an integer compare of float bits mis-orders negatives, so
/// `max(0.0, x)` would wrongly return a negative x. The software backend
/// already uses a float compare for float operands.
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

/// Set the three shift-form fields SHF shares between its register and its
/// immediate operand form.
///
/// THIS HARDWARE HAS ONLY A FUNNEL SHIFT: it shifts the 64-bit pair
/// (high:low) and gives back one 32-bit half. A left shift puts the value in
/// the LOW operand, RZ in the high one, and takes the LOW half. A right shift
/// is the mirror: the value goes in the HIGH operand, RZ in the low one, and
/// the answer comes out of the HIGH half. `dst_high` at bit 80 is what selects
/// that half, and a right shift without it returns the low half of the funnel,
/// which is `value << (32 - count)`. A live GPU run of `0x1234 >> 3` returned
/// 0x80000000, which is exactly that.
///
/// Bits 73..74 name the DATA TYPE, and NAK's table is I64 = 0, U64 = 1,
/// I32 = 2, U32 = 3 (`OpShf::encode`). These are 32-bit shifts, so the field is
/// 2 or 3 and never 0 or 1. A 64-bit type there gives the same answer for a
/// left shift, because the high operand is RZ, and the wrong answer for a
/// right shift.
fn setShiftForm(w: *Inst, right: bool, arithmetic: bool) void {
    setBits(w, 73, 2, if (arithmetic) 2 else 3); // I32 vs U32
    setBits(w, 75, 1, 1); // wrap the shift count
    setBits(w, 76, 1, @intFromBool(right));
    setBits(w, 80, 1, @intFromBool(right)); // take the HIGH half for a right shift
}

/// `SHF.L/R dst, value, shift, RZ`: shift left or right. Regular SHF 0x019.
/// A right shift puts the value in the high (srcC) operand and takes the high
/// half of the funnel. See `setShiftForm`.
pub fn shf(dst: u8, value: u8, shift: u8, right: bool, arithmetic: bool, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 9, 0x019);
    setBits(&w, 9, 3, 1); // register form
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, if (right) RZ else value);
    setBits(&w, 32, 8, shift);
    setBits(&w, 64, 8, if (right) value else RZ);
    setShiftForm(&w, right, arithmetic);
    return w;
}

// BITS 81..90 OF A FLOAT ALU OP ARE NOT PREDICATE FIELDS, AND THEY MUST BE ZERO.
//
// This backend used to write PT (7) into 81..83, 84..86 and 87..89 of FADD, FMUL and
// FFMA, on the theory that they were the same predicate source and result fields the
// integer ops carry. THAT WAS WRONG, and on FADD it silently changed the arithmetic:
// `nvdisasm` read the result as `FHADD.BF16 Rd, Ra.H1, Rb`, a HALF-PRECISION add that
// truncates source A to bfloat16. A live GPU run of `x + x` with
// x = 0x3EFF_FFFC returned 0x3F7F_7FFE where the correct answer is 0x3F7F_FFFC, which is
// exactly what truncating A to bf16 predicts. `fsub` is built on `fadd`, so float
// subtract carried the same defect, and a naive matmul returned 4486 for 12276.
//
// Nothing caught it because every earlier float test used operands whose low 16 mantissa
// bits are zero (0.5, 1.0, 2.0), and for those the truncation changes nothing.
//
// GROUND TRUTH, read out of a ptxas cubin for sm_120 with
// `nvdisasm -b SM120 -c -hex` and the raw section bytes:
//
//     FADD R5, R6, R7      06057221 00000007 00000000 140fe200
//     FADD R7, R6, -R7     06077221 80000007 00000000 000fc400
//     FMUL R9, R6, R7      06097220 00000007 00400000 041fe200
//     FFMA R5, R4, R5, R7  04057223 00000005 00000007 001fca00
//
// Word 2 holds bits 64..95. FADD and FFMA leave EVERY bit of 72..95 clear. FMUL sets one
// field, bits 84..86 = 4, which is NAK's PDIV; the rest are clear. NAK's `OpFAdd::encode`,
// `OpFMul::encode` and `OpFFma::encode` agree: none of them touches 81..90.
//
// The old comment claimed a UBO vec4 multiply faulted unless all three were PT. The field
// that actually fixed that multiply is FMUL's PDIV at 84..86, which `fmul` still writes.

/// `FADD dst, a, b`: 32-bit float add. NAK base 0x021. Bits 81..90 stay CLEAR: see the
/// note above.
pub fn fadd(dst: u8, a: u8, b: u8, c: Control) Inst {
    return alu(0x021, dst, a, b, RZ, c);
}

/// `FADD dst, a, -b`: float subtract (dst = a - b), via the srcB negate modifier.
pub fn fsub(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = fadd(dst, a, b, c);
    setBits(&w, 63, 1, 1); // negate srcB
    return w;
}

/// `FMUL dst, a, b`: 32-bit float multiply. NAK base 0x020, plus the PDIV field at bits
/// 84..86, which must be 4.
///
/// NAK's `OpFMul::encode` (sm70_encode.rs) does `set_field(84..87, 0x4)` after the generic
/// ALU encode, and ptxas emits the same. FADD and FFMA do not set it. Left at 7, the
/// multiply is corrupted: a bare `FMUL 0.5, 0.5` reads back saturated (about 1.0) instead
/// of 0.25 on Blackwell sm_120. Proven by a frame oracle: a fragment that outputs
/// `0.5*0.5` saturated, while `0.5+0.0` through FADD read 0.5 correctly. This is the field
/// behind every "FMUL on the GPU saturates", "cube is white", and "derivative 22x too
/// large" symptom: any shader doing an FP multiply mis-computed.
pub fn fmul(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = alu(0x020, dst, a, b, RZ, c);
    setBits(&w, 84, 3, 4); // PDIV field = 4 (NAK OpFMul)
    return w;
}

/// `FFMA dst, a, b, c_in`: fused multiply-add (dst = a*b + c_in). NAK base 0x023. Bits
/// 81..90 stay CLEAR, as they do for FADD.
pub fn ffma(dst: u8, a: u8, b: u8, c_in: u8, c: Control) Inst {
    return alu(0x023, dst, a, b, c_in, c);
}

// 32-bit int-to-float and float-to-int. NAK encodes the operand sizes as
// log2(bytes): 4 bytes maps to 2.

/// I2F opcode as it appears in the encoded instruction's low 12 bits: `alu()`
/// ORs the NAK base 0x106 with the register-srcB form bit (1 << 9 = 0x200),
/// so the scheduler sees 0x306. Variable-latency (decoupled) on sm120, so the
/// scheduler scoreboards it (schedule.isVariableLatency).
pub const I2F_OPCODE: u32 = 0x306;
/// F2I opcode in the same low 12 bits: NAK base 0x105 with the register-srcB
/// form bit. Decoupled on sm120, exactly as I2F is.
pub const F2I_OPCODE: u32 = 0x305;

/// `I2F.F32 dst, src`: convert a 32-bit integer to f32. NAK base 0x106, src
/// signedness at bit 74, dst-size-log2 at 75, src-size-log2 at 84.
pub fn i2f(dst: u8, src: u8, src_signed: bool, c: Control) Inst {
    var w = alu(0x106, dst, RZ, src, RZ, c);
    if (src_signed) setBits(&w, 74, 1, 1);
    setBits(&w, 75, 2, 2); // dst = 4 bytes (f32)
    setBits(&w, 84, 2, 2); // src = 4 bytes (i32)
    return w;
}

/// `F2I.S32 dst, src`: convert an f32 to a 32-bit integer, truncating
/// toward zero (C / SPIR-V FToS semantics). NAK base 0x105, dst signedness
/// at bit 72, dst-size-log2 at 75, round-mode at 78 (Zero = 3), src-size-log2 at 84.
pub fn f2i(dst: u8, src: u8, dst_signed: bool, c: Control) Inst {
    return f2iRound(dst, src, dst_signed, .zero, c);
}

/// The F2I rounding mode (bits 78..80): nearest-even, floor (toward -inf),
/// ceil (toward +inf), or zero (truncate). floor or ceil, plus an i2f back,
/// implement GLSL floor() and ceil() on the integer-representable range,
/// since the rounding ops have no direct F32-to-F32 instruction here.
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

/// `MUFU.op dst, src`: a transcendental on the special-function unit
/// (reciprocal, reciprocal-sqrt, sqrt, sin, cos, exp2, log2). NAK base
/// 0x108, the operand in srcB (bit 32, register form), the op selector at
/// 74..80, F32 type (bit 72 = 0). Used to lower `inversesqrt` (RSQ), `sqrt`
/// (SQRT), and a float reciprocal (RCP, for FP divide a/b = a * RCP(b)).
/// MUFU has fixed latency, a few cycles, not variable latency on the SFU
/// pipe. The default stall covers a back-to-back dependency.
pub fn mufu(dst: u8, src: u8, mfop: MuFuOp, c: Control) Inst {
    var w = alu(0x108, dst, RZ, src, RZ, c);
    setBits(&w, 72, 1, 0); // op_type = F32
    setBits(&w, 74, 6, @intFromEnum(mfop)); // MUFU op selector
    return w;
}

// ---------------------------------------------------------------------------
// Immediate-operand ALU forms.
//
// The three ALU operand slots are srcA at bits 24..31 (a register only), srcB
// at 32..63, and srcC at 64..71 (a register only). The 3-bit FORM field at
// bits 9..11 tells the hardware what the 32..63 slot holds. NAK's `encode_alu`
// (Mesa `sm70_encode.rs`) picks it:
//
//   form 1: srcB is a REGISTER at 32..39, srcC is a register at 64..71.
//   form 4: srcB is a 32-BIT IMMEDIATE at 32..63, srcC is a register at 64..71.
//
// So an immediate operand costs no MOV and no register. Before these forms the
// instruction selector materialized every constant with `movImm` into a scratch
// register and then read that register, which is two instructions where the
// hardware needs one.
//
// The IMMEDIATE OCCUPIES THE WHOLE 32..63 RANGE, so bits 62 and 63 are value
// bits and not the srcB abs/negate modifiers the register form puts there.
// `isub` and `fsub` negate their register operand with bit 63; the immediate
// forms below must negate the VALUE instead. See `iaddImm` and `fsubImm`.
//
// The scheduler reads the same form field: `schedule.readsSrc` treats bits
// 32..39 as a register source only when the form is 1, so an immediate in that
// slot is not mistaken for a GPR number. That is what makes these forms safe to
// emit without a scheduler change.

/// A fixed-latency ALU op whose second source is a 32-bit immediate: the 9-bit
/// base opcode in bits 0..8, form 4 in bits 9..11, `dst` at 16, srcA at 24, the
/// immediate at 32..63, and srcC at 64.
fn aluImm(op: u9, dst: u8, a: u8, imm: u32, c_in: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 9, op);
    setBits(&w, 9, 3, 4); // form: 32-bit immediate srcB
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, a);
    setBits(&w, 32, 32, imm);
    setBits(&w, 64, 8, c_in);
    return w;
}

/// `IADD3 dst, a, imm, RZ`: add a 32-bit immediate. The carry fields are the
/// constant false, exactly as `iadd3` writes them, because a zero there names
/// P0 and P0 is a predicate this backend gives to booleans.
///
/// TO SUBTRACT AN IMMEDIATE, negate the value and add it. The register form's
/// negate modifier is bit 63, which this form uses as an immediate value bit.
pub fn iadd3Imm(dst: u8, a: u8, imm: u32, c: Control) Inst {
    var w = aluImm(0x010, dst, a, imm, RZ, c);
    setBits(&w, 81, 3, PT); // carry-out predicate = none
    setBits(&w, 84, 3, PT);
    setCarryIn(&w, 87, 90, PT, true); // carry-in 0 = !PT
    setCarryIn(&w, 77, 80, PT, true); // carry-in 1 = !PT
    return w;
}

/// `IADD3 dst, a, imm, RZ` writing a carry-out to predicate `cout`: the low half
/// of a 64-bit add against a constant, such as a global pointer plus a constant
/// byte offset. The high half is the same `iadd3CarryIn` the register form uses.
pub fn iadd3CarryOutImm(dst: u8, a: u8, imm: u32, cout: u8, c: Control) Inst {
    var w = iadd3Imm(dst, a, imm, c);
    setBits(&w, 81, 3, cout); // carry-out predicate
    return w;
}

/// `IADD3.X dst, a, imm, RZ, cin, !PT`: the HIGH half of a 64-bit add against a
/// constant. `imm` is the high word of the constant, which is 0 for a positive
/// value and 0xFFFFFFFF for a negative one, so the low half's constant is sign
/// extended and not zero extended.
///
/// BIT 74 IS THE `.X` FLAG. See `iadd3CarryIn` for why it and nothing else makes
/// the hardware read the carry.
pub fn iadd3CarryInImm(dst: u8, a: u8, imm: u32, cin: u8, c: Control) Inst {
    var w = iadd3Imm(dst, a, imm, c);
    setBits(&w, 74, 1, 1); // .X: read the carry-in
    setCarryIn(&w, 87, 90, cin, false); // carry-in 0 = cin
    return w;
}

/// `IMAD dst, a, imm, c_in`: multiply by a 32-bit immediate and add `c_in`.
/// A plain multiply passes `RZ` for `c_in`. The result-predicate field must be
/// PT: leaving it 0 (P0) is rejected on Blackwell.
pub fn imadImm(dst: u8, a: u8, imm: u32, c_in: u8, c: Control) Inst {
    var w = aluImm(0x024, dst, a, imm, c_in, c);
    setBits(&w, 81, 3, PT); // result predicate = none
    return w;
}

/// `LOP3.LUT dst, a, imm, RZ, lut`: a bitwise op against a 32-bit immediate.
pub fn lop3Imm(dst: u8, a: u8, imm: u32, lut: u8, c: Control) Inst {
    var w = aluImm(0x012, dst, a, imm, RZ, c);
    setBits(&w, 72, 8, lut);
    setBits(&w, 81, 3, PT); // predicate dst = none
    return w;
}

/// `SHF.L/R dst, value, imm, RZ`: shift by a constant count. The field layout
/// matches `shf`: a right shift puts the value in the srcC slot, because the
/// hardware has only the funnel shift.
pub fn shfImm(dst: u8, value: u8, shift: u32, right: bool, arithmetic: bool, c: Control) Inst {
    var w = aluImm(0x019, dst, if (right) RZ else value, shift, if (right) value else RZ, c);
    setShiftForm(&w, right, arithmetic);
    return w;
}

/// A fixed-latency ALU op whose THIRD source is a 32-bit immediate: form 2. The
/// immediate still occupies bits 32..63, and the operand that form 4 would put
/// there moves to the srcC slot at 64..71.
///
/// FADD is the one op here that needs this. NAK's `OpFAdd::encode` passes an
/// immediate second operand as `encode_alu(0x021, dst, srcs[0], Src::ZERO,
/// srcs[1])`, which is src1 = a register and src2 = the immediate, and
/// `encode_alu` answers form 2 for that pair. Encoded as form 4 instead, the
/// hardware reads the operand from somewhere else: a live GPU run of
/// `x + 0.25` returned 0.
fn aluImm2(op: u9, dst: u8, a: u8, imm: u32, b: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 9, op);
    setBits(&w, 9, 3, 2); // form: 32-bit immediate srcC, register srcB at 64
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, a);
    setBits(&w, 32, 32, imm);
    setBits(&w, 64, 8, b);
    return w;
}

/// `FADD dst, a, imm`: add a 32-bit float immediate, given as its IEEE-754
/// binary32 bit pattern.
///
/// TWO THINGS DIFFER FROM THE REGISTER FORM, and each of them returned 0 from a
/// live GPU run of `2.0 + 0.25` when it was wrong.
///
/// It is FORM 2, not form 4. See `aluImm2`.
///
/// It leaves bits 81..90 CLEAR, exactly as the register form does. Writing PT
/// into them returned 0 here, and turned the register form into a bfloat16 add:
/// see the note above `fadd`.
pub fn faddImm(dst: u8, a: u8, imm: u32, c: Control) Inst {
    return aluImm2(0x021, dst, a, imm, RZ, c);
}

/// `FADD dst, a, -imm`: subtract a float immediate. The register form negates
/// srcB with bit 63, which this form uses as the sign bit of the immediate, so
/// the SIGN BIT OF THE VALUE is flipped instead. That gives the same result for
/// every finite value and for an infinity, and it keeps a NaN a NaN.
pub fn fsubImm(dst: u8, a: u8, imm: u32, c: Control) Inst {
    return faddImm(dst, a, imm ^ 0x8000_0000, c);
}

/// `FMUL dst, a, imm`: multiply by a 32-bit float immediate, given as its
/// IEEE-754 binary32 bit pattern. Bits 84..86 hold NAK's PDIV field, which must
/// be 4: see `fmul`. Every other bit of 81..90 stays clear.
pub fn fmulImm(dst: u8, a: u8, imm: u32, c: Control) Inst {
    var w = aluImm(0x020, dst, a, imm, RZ, c);
    setBits(&w, 84, 3, 4); // PDIV field = 4 (NAK OpFMul)
    return w;
}

/// `FFMA dst, a, imm, c_in`: fused multiply-add whose MULTIPLIER is a 32-bit
/// float immediate, given as its IEEE-754 binary32 bit pattern. This is the
/// shape `x * k + y` takes, which a scaled accumulation writes.
///
/// IT IS FORM 4, NOT FORM 2. The form field states which of the two later
/// sources is the immediate, and NAK's `encode_alu` picks it from that pair:
/// an immediate at src1 with a register at src2 is form 4, and an immediate at
/// src2 is form 2. `faddImm` is form 2 because FADD passes its immediate as
/// src2 with `Src::ZERO` at src1. FFMA keeps a real addend at src2, so the
/// immediate is src1 and the form is 4, exactly as `imadImm` encodes the same
/// operand pattern.
///
/// `nvdisasm -b SM120 -c` reads this word back as `FFMA R6, R4, 0.25, R7`, so
/// NVIDIA's own decoder agrees on the form and on all three operands.
pub fn ffmaImm(dst: u8, a: u8, imm: u32, c_in: u8, c: Control) Inst {
    return aluImm(0x023, dst, a, imm, c_in, c);
}

/// `FFMA dst, a, b, imm`: fused multiply-add whose ADDEND is a 32-bit float
/// immediate, given as its IEEE-754 binary32 bit pattern, and whose MULTIPLIER
/// is the register `b`. This is the shape `x * y + k` takes.
///
/// IT IS FORM 2, NOT FORM 4. The form field states which of the two later
/// sources is the immediate, and NAK's `encode_alu` picks it from that pair:
/// an immediate at src2 is form 2 and an immediate at src1 is form 4. Here the
/// addend is the immediate, so it sits at bits 32..63 as src2 and the
/// multiplier register moves to the srcC slot at bits 64..71, exactly as
/// `aluImm2` lays it out.
///
/// THIS IS THE FORM PTXAS EMITS for the contracted `acc = acc * k1 + k2`
/// chain: it hoists the multiplier k1 into a register ahead of the loop and
/// keeps the addend k2 in the instruction. Read out of an sm_120 cubin
/// (`nvdisasm -b SM120 -c`): `FFMA R0, R0, R9.reuse, 0.05`, whose 12-bit
/// opcode is 0x423. The immediate-FFMA test below pins the same encoding from
/// this side, and `nvdisasm` reads this exact word back as
/// `FFMA R6, R4, R9, 0.05`.
pub fn ffmaAddendImm(dst: u8, a: u8, b: u8, imm: u32, c: Control) Inst {
    return aluImm2(0x023, dst, a, imm, b, c);
}

/// `IMAD.WIDE dst:dst+1, a, b, c:c+1`: a 32 by 32 multiply added to a 64-bit
/// value, giving a 64-bit result. NAK opcode 0x025, the same encoding as `imad`
/// with the wide opcode. `signed` selects IMAD.WIDE over IMAD.WIDE.U32, and it
/// sign-extends the 32-bit product into the 64-bit sum.
///
/// THIS IS ONE INSTRUCTION WHERE A GLOBAL ARRAY INDEX OTHERWISE TAKES FOUR.
/// `base + index * scale` with a 64-bit base needs a shift, a carry-out add and
/// a carry-in add, plus a MOV for the scale. IMAD.WIDE scales, sign-extends and
/// adds the 64-bit base together, and its carry cannot be dropped because the
/// hardware never splits it.
///
/// `dst` and a register `c` must each be an even register, because both name a
/// 64-bit pair. The caller owns both registers of each pair.
pub fn imadWide(dst: u8, a: u8, b: u8, c_in: u8, signed: bool, c: Control) Inst {
    var w = alu(0x025, dst, a, b, c_in, c);
    setBits(&w, 73, 1, @intFromBool(signed));
    setBits(&w, 81, 3, PT); // result predicate = none
    return w;
}

/// `IMAD.WIDE dst:dst+1, a, imm, c:c+1`: the immediate-scale form of
/// `imadWide`, which is what an array index uses: the scale is the element size
/// and it is a compile-time constant.
pub fn imadWideImm(dst: u8, a: u8, imm: u32, c_in: u8, signed: bool, c: Control) Inst {
    var w = aluImm(0x025, dst, a, imm, c_in, c);
    setBits(&w, 73, 1, @intFromBool(signed));
    setBits(&w, 81, 3, PT); // result predicate = none
    return w;
}

/// `LEA dst, a, b, shift`: `dst = (a << shift) + b`, all 32 bits. ONE
/// instruction where a shift and an add take two, which is what an array index
/// into a 32-bit address space costs.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpLea`. Opcode 0x011 through
/// the generic ALU encode, so `a` is srcA at 24..31 and `b` is srcB at 32..39.
/// The shift count is a 5-bit field at 75..80. Bit 80 selects the HIGH half of
/// a 64-bit shift-and-add and stays clear here. Bit 74 is the `.X` carry-in
/// form and stays clear. Bits 81..84 hold the OVERFLOW predicate destination,
/// which must be PT and not the zero default, because P0 is a register the
/// boolean allocator hands out. Bit 72 negates the shifted value and stays
/// clear.
///
/// `ir.rs` `impl Foldable for OpLea` states the arithmetic: shift `a` left,
/// then add `b`. The shift is a 32-bit shift, so a count above 31 is refused
/// by the caller and never encoded.
pub fn lea(dst: u8, a: u8, b: u8, shift: u5, c: Control) Inst {
    var w = alu(0x011, dst, a, b, RZ, c);
    setBits(&w, 72, 1, 0); // do not negate the shifted value
    setBits(&w, 74, 1, 0); // not the .X carry-in form
    setBits(&w, 75, 5, shift);
    setBits(&w, 80, 1, 0); // the LOW half, not LEA.HI
    setBits(&w, 81, 3, PT); // overflow predicate = none
    return w;
}

/// The LEA opcode as it appears in the encoded instruction's low 12 bits: `alu`
/// ORs the NAK base 0x011 with the register-srcB form bit (1 << 9).
pub const LEA_OPCODE: u32 = 0x211;

/// The IMAD.WIDE opcode as it appears in the encoded instruction's low 12 bits,
/// in both operand forms. `alu` ORs the NAK base 0x025 with the register-srcB
/// form bit (1 << 9), and `aluImm` with the immediate form bit (4 << 9).
///
/// The scheduler needs both: IMAD.WIDE writes a REGISTER PAIR and reads one at
/// its srcC, and no other ALU op does either, so `schedule.dstSpan` and
/// `schedule.srcSpan` name these opcodes to give it a span of 2.
pub const IMAD_WIDE_OPCODE: u32 = 0x225;
/// The immediate-scale form of `IMAD_WIDE_OPCODE`.
pub const IMAD_WIDE_IMM_OPCODE: u32 = 0x825;

/// `LDC dst, c[bank][offset]`: load a 32-bit value from a constant bank
/// (the kernel-parameter ABI loads inputs this way). NAK opcode 0xb82: dst
/// at bit 16, dynamic offset register at 24 (RZ = static), 16-bit immediate
/// offset at 38, bank index at 54, mem type B32 at 73. Fixed latency on the
/// constant cache.
pub fn ldc(dst: u8, bank: u5, offset: u16, c: Control) Inst {
    return ldcSized(dst, bank, offset, .b32, c);
}

/// `LDC.64 dst, c[bank][offset]`: read EIGHT bytes of a constant bank into the
/// register pair (dst, dst+1). A 64-bit pointer parameter is one of these, and
/// so is any pair of adjacent 32-bit scalars.
///
/// One instruction where two LDCs were emitted before, and the parameter
/// prologue of every kernel with a pointer paid that. `dst` must be EVEN, and
/// `offset` must be 8-ALIGNED, because both name a 64-bit quantity.
pub fn ldcWide(dst: u8, bank: u5, offset: u16, c: Control) Inst {
    std.debug.assert(dst % 2 == 0);
    std.debug.assert(offset % 8 == 0);
    return ldcSized(dst, bank, offset, .b64, c);
}

/// `LDC dst, c[bank][offset]` at an explicit access width. The width field is
/// the same `MemType` at bits 73..75 that LDG, STG, LDS and STS use, and it
/// decides how many consecutive registers the load fills.
pub fn ldcSized(dst: u8, bank: u5, offset: u16, ty: MemType, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0xb82);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, RZ); // no dynamic offset
    setBits(&w, 38, 16, offset);
    setBits(&w, 54, 5, bank);
    setBits(&w, 73, 3, @intFromEnum(ty));
    return w;
}

/// The width of a memory access, and how a narrow load fills its destination
/// register. The values are NAK's `MemType`: Mesa `sm70_encode.rs`,
/// `SM70Encoder::set_mem_type`, which writes this 3-bit field at bits 73..76 of
/// LDG, STG, LDS, STS and LDC.
///
/// An 8-bit or 16-bit load extends its value into the whole 32-bit destination
/// register: `u8` and `u16` add zeros, `i8` and `i16` copy the sign bit. An
/// 8-bit or 16-bit store writes only its own bytes, so it keeps the neighbouring
/// bytes that a 32-bit store of a byte-wide value destroys.
///
/// `b64` and `b128` move a BLOCK of consecutive registers, `regCount` of them,
/// that starts at the named register. The caller must own the whole block,
/// because the hardware writes or reads every register in it.
///
/// An ATOMIC uses a DIFFERENT and wider width field. See `AtomType`.
pub const MemType = enum(u3) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    b32 = 4,
    b64 = 5,
    b128 = 6,

    /// How many consecutive 32-bit registers the access moves.
    pub fn regCount(self: MemType) u8 {
        return switch (self) {
            .u8, .i8, .u16, .i16, .b32 => 1,
            .b64 => 2,
            .b128 => 4,
        };
    }

    /// How many bytes of memory the access touches.
    pub fn byteSize(self: MemType) u8 {
        return switch (self) {
            .u8, .i8 => 1,
            .u16, .i16 => 2,
            .b32 => 4,
            .b64 => 8,
            .b128 => 16,
        };
    }
};

/// The memory-order field of a global access, bits 77..80. The values are
/// NAK's `set_mem_order` table for sm >= 80.
///
/// `weak` is the DEFAULT, and it is what both ptxas and NAK emit for an
/// ordinary load or store. A weak access still becomes visible to the host
/// when the grid completes, because the launch's end-of-grid release flushes
/// L2. The strong orders exist for a kernel that must publish a value to
/// another CTA or to the host WHILE it still runs, and they cost bandwidth on
/// every access that asks for them.
pub const MemOrder = enum(u4) {
    weak = 0x0,
    constant = 0x4,
    strong_cta = 0x5,
    strong_gpu = 0x7,
    strong_sys = 0xa,
};

/// `LDG.E dst, [addr:addr+1]`: load `ty` from the 64-bit global address in the
/// register pair (addr, addr+1). Volta LDG 0x981.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpLd`, the `MemSpace::Global`
/// arm. The 24-BIT SIGNED BYTE DISPLACEMENT at bits 40..63 is added to that address by
/// the hardware. `set_reg_addr(24..32, addr, 90)` for the address pair;
/// `set_ureg_addr(32, uniform, 72)` for the uniform base, URZ here, which also
/// sets bit 72; `set_rev_pred_src(64..67, 67, pred)` for the guard;
/// `set_pred_dst(81..84, None)` for the fault predicate; and `set_mem_access`,
/// which writes `set_mem_type(73..76)`, the memory order at 77..81 and the
/// eviction priority at 84..87. A `b64` load writes (dst, dst+1) and a `b128`
/// load writes (dst .. dst+3), so the caller must own the whole block.
///
/// THE UNIFORM BASE OF A LOAD SITS AT BIT 32, not at bit 64 where a store keeps
/// it. A store needs bits 32..40 for its data register and moves the uniform
/// base out of the way; a load has no data register and leaves bits 64..67 for
/// the guard predicate instead. Writing the uniform base at 64 therefore left
/// UR0 in the address and `!P0` in the guard, and `nvdisasm -b SM120` printed
/// exactly that: `LDG.E.LTC256B.STRONG.SYS P0, R4, [R2.64+UR0], !P0`.
///
/// The guard predicate is left zero, which is the encoding of an unconditional
/// access: the field is REVERSED (NAK writes `7 - index`), so PT encodes as 0.
/// The fault-predicate destination is PT and not the zero default, because P0
/// is a register the boolean allocator hands out.
pub fn ldgOrdered(dst: u8, addr: u8, offset: i32, ty: MemType, order: MemOrder, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x981);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, addr);
    setBits(&w, 40, 24, signedBits(offset, 24)); // signed byte displacement
    setBits(&w, 32, 8, URZ); // uniform base, at 32 for a LOAD
    setBits(&w, 64, 4, 0); // guard = PT, reversed, no negate
    setBits(&w, 72, 1, 1); // 64-bit uniform
    setBits(&w, 73, 3, @intFromEnum(ty));
    setBits(&w, 77, 4, @intFromEnum(order));
    setBits(&w, 81, 3, PT); // fault predicate = none
    setBits(&w, 84, 3, 1); // eviction NORMAL
    setBits(&w, 90, 1, 1); // 64-bit GPR address
    setBits(&w, 91, 1, 1); // UGPR mode
    return w;
}

/// `LDG.E dst, [addr:addr+1]` at the default weak memory order. This is the
/// form the instruction selector emits; a kernel that needs a stronger order
/// asks for it through `ldgOrdered`.
pub fn ldg(dst: u8, addr: u8, ty: MemType, c: Control) Inst {
    return ldgAt(dst, addr, 0, ty, c);
}

/// `LDG.E dst, [addr:addr+1 + offset]`: the weak-order load with a CONSTANT
/// BYTE DISPLACEMENT in the instruction. A constant array index or a struct
/// field offset goes here, so it costs no address arithmetic and no register.
/// `fitsAddrOffset` is the range the field holds.
pub fn ldgAt(dst: u8, addr: u8, offset: i32, ty: MemType, c: Control) Inst {
    return ldgOrdered(dst, addr, offset, ty, .weak, c);
}

/// `LDG.E.32 dst, [addr:addr+1]`: the single-word shorthand for `ldg`.
pub fn ldgU32(dst: u8, addr: u8, c: Control) Inst {
    return ldg(dst, addr, .b32, c);
}

/// `STG.E [addr:addr+1], data`: store `ty` from the GPR block that
/// starts at `data` to the 64-bit global address in (addr, addr+1). The 32-bit
/// form is verified bit-for-bit on hardware (prism).
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpSt`, the `MemSpace::Global`
/// arm. `set_reg_addr(24..32, addr, 90)`; `set_ureg_addr(64, uniform, 72)` for
/// the uniform base, which for a STORE sits at bit 64; `set_reg_src(32..40)`
/// for the data; and `set_mem_access` as for `ldg`. A `b64` store reads
/// (data, data+1) and a `b128` store reads (data .. data+3).
pub fn stgOrdered(addr: u8, data: u8, offset: i32, ty: MemType, order: MemOrder, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x986);
    setBits(&w, 24, 8, addr);
    setBits(&w, 40, 24, signedBits(offset, 24)); // signed byte displacement
    setBits(&w, 90, 1, 1); // 64-bit GPR address
    setBits(&w, 64, 8, URZ); // uniform base, at 64 for a STORE
    setBits(&w, 72, 1, 1); // 64-bit uniform
    setBits(&w, 32, 8, data);
    setBits(&w, 73, 3, @intFromEnum(ty));
    setBits(&w, 77, 4, @intFromEnum(order));
    setBits(&w, 84, 3, 1); // eviction NORMAL
    setBits(&w, 91, 1, 1); // UGPR mode (required or the SM traps)
    return w;
}

/// `STG.E [addr:addr+1], data` at the default weak memory order. This is the
/// form the instruction selector emits; a kernel that needs a stronger order
/// asks for it through `stgOrdered`.
pub fn stg(addr: u8, data: u8, ty: MemType, c: Control) Inst {
    return stgAt(addr, data, 0, ty, c);
}

/// `STG.E [addr:addr+1 + offset], data`: the weak-order store with a CONSTANT
/// BYTE DISPLACEMENT in the instruction. See `ldgAt`.
pub fn stgAt(addr: u8, data: u8, offset: i32, ty: MemType, c: Control) Inst {
    return stgOrdered(addr, data, offset, ty, .weak, c);
}

/// `STG.E.32 [addr:addr+1], data`: the single-word shorthand for `stg`.
pub fn stgU32(addr: u8, data: u8, c: Control) Inst {
    return stg(addr, data, .b32, c);
}

// Workgroup shared memory. A shared address is NOT a 64-bit global address: it
// is a 32-bit byte offset into the CTA's shared-memory window, so LDS and STS
// read it out of ONE register, not out of an aligned pair. These three
// encodings are transcribed from Mesa NAK `sm70_encode.rs`, which is the
// authoritative bit-level reference (see the `nak-sass-encoding-reference`
// note). None of them is hardware-verified here, because this repository
// cannot execute SASS.

/// `LDS dst, [addr]`: load `ty` from workgroup shared memory at the 32-bit
/// window offset in `addr`.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpLd`, the `MemSpace::Shared`
/// arm plus the common tail of that `encode`. Opcode 0x984 (bits 0..12);
/// `set_dst` at 16..24; `set_reg_src(24..32)` for the address; `set_ureg_src(32)`
/// for the uniform base, which is URZ here because there is none (8 bits wide on
/// sm>=100, the Blackwell target of this backend); `set_field(40..64)` for the
/// 24-bit immediate offset, which stays 0 because the isel materializes every
/// offset into the address register; `set_mem_type(73..76)` for the width;
/// `set_field(78..80)` = 0 (`OffsetStride::X1.encode_sm75()`);
/// `set_upred_src(87..90, 90)` = UPT, NAK's `true_reg` index 7, so the access is
/// unconditional; and `set_bit(91, true)`, NAK's "always enable UGPR mode".
///
/// The shared arm sets NO memory order and NO eviction priority: NAK asserts
/// that a shared access is `Strong(CTA)`/`Normal` and then leaves bits 77 and
/// 81..87 at zero, unlike the global LDG/STG path above. Variable latency: the
/// scoreboard scheduler assigns the write barrier and the consumer waits.
pub fn lds(dst: u8, addr: u8, ty: MemType, c: Control) Inst {
    return ldsAt(dst, addr, 0, ty, c);
}

/// `LDS dst, [addr + offset]`: the shared load with a CONSTANT BYTE
/// DISPLACEMENT in the instruction, at bits 40..63. A fixed slot of a shared
/// tile goes here, so it costs no address arithmetic and no register.
/// `fitsAddrOffset` is the range the field holds.
pub fn ldsAt(dst: u8, addr: u8, offset: i32, ty: MemType, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x984);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, addr); // 32-bit shared-window offset, ONE register
    setBits(&w, 32, 8, URZ); // no uniform base
    setBits(&w, 40, 24, signedBits(offset, 24)); // signed byte displacement
    setBits(&w, 73, 3, @intFromEnum(ty));
    setBits(&w, 78, 2, 0); // offset stride X1
    setBits(&w, 87, 3, PT); // UPT: unconditional
    setBits(&w, 91, 1, 1); // UGPR mode
    return w;
}

/// `LDS.32 dst, [addr]`: the single-word shorthand for `lds`.
pub fn ldsU32(dst: u8, addr: u8, c: Control) Inst {
    return lds(dst, addr, .b32, c);
}

/// `STS [addr], data`: store `ty` from the GPR block that starts at `data` to
/// workgroup shared memory at the 32-bit window offset in `addr`.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpSt`, the `MemSpace::Shared`
/// arm plus the common tail of that `encode`. Opcode 0x988 (bits 0..12);
/// `set_reg_src(24..32)` for the address; `set_reg_src(32..40)` for the data;
/// `set_field(40..64)` for the 24-bit SIGNED byte displacement, which `stsAt` fills;
/// `set_ureg_src(64)` for the uniform base, URZ here (8 bits on sm>=100);
/// `set_mem_type(73..76)` for the width; `set_field(78..80)` = 0
/// (`OffsetStride::X1`); and `set_bit(91, has_ugpr)`, true for this target.
///
/// The uniform base sits at bit 64 for a store and at bit 32 for a load. That
/// is not a transcription slip: NAK uses the two different starts, because a
/// store needs bits 32..40 for its data register.
pub fn sts(addr: u8, data: u8, ty: MemType, c: Control) Inst {
    return stsAt(addr, data, 0, ty, c);
}

/// `STS [addr + offset], data`: the shared store with a CONSTANT BYTE
/// DISPLACEMENT in the instruction. See `ldsAt`.
pub fn stsAt(addr: u8, data: u8, offset: i32, ty: MemType, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x988);
    setBits(&w, 24, 8, addr); // 32-bit shared-window offset, ONE register
    setBits(&w, 32, 8, data);
    setBits(&w, 40, 24, signedBits(offset, 24)); // signed byte displacement
    setBits(&w, 64, 8, URZ); // no uniform base
    setBits(&w, 73, 3, @intFromEnum(ty));
    setBits(&w, 78, 2, 0); // offset stride X1
    setBits(&w, 91, 1, 1); // UGPR mode
    return w;
}

/// `STS.32 [addr], data`: the single-word shorthand for `sts`.
pub fn stsU32(addr: u8, data: u8, c: Control) Inst {
    return sts(addr, data, .b32, c);
}

// Atomic read-modify-write. An atomic reads a memory location, combines it with
// a data operand and writes the result back. No other thread can see the
// location between that read and that write.
//
// The encodings below are transcribed from Mesa NAK `sm70_encode.rs`,
// `impl SM70Op for OpAtom`, which is the authoritative bit-level reference (see
// the `nak-sass-encoding-reference` note). None of them is hardware-verified
// here, because this repository cannot execute SASS.
//
// Nothing emits them yet: the IR has no atomic operation. Like `barSync` they
// are encoder-only until it gets one.
//
// A GLOBAL atomic takes a 64-bit address in an aligned register PAIR
// (addr, addr+1), the same as LDG and STG. A SHARED atomic takes a 32-bit
// offset into the CTA window in ONE register, the same as LDS and STS.

/// The read-modify-write operation an atomic applies. Values from NAK
/// `sm70_encode.rs`, `SM70Encoder::set_atom_op`.
///
/// Compare-and-swap is deliberately absent. NAK gives it its own opcode and a
/// second data operand, so it has its own encoders here: `atomgCas` and
/// `atomsCas`.
pub const AtomOp = enum(u4) {
    add = 0,
    min = 1,
    max = 2,
    /// Increment, but wrap to 0 when the value reaches the data operand.
    inc = 3,
    /// Decrement, but wrap to the data operand when the value reaches 0.
    dec = 4,
    bit_and = 5,
    bit_or = 6,
    bit_xor = 7,
    /// Exchange: write the data operand and give back the old value.
    exch = 8,
};

/// The type an atomic operates on. It sets the access width, and it tells
/// `min` and `max` whether to compare signed or unsigned. Values from NAK
/// `sm70_encode.rs`, `SM70Encoder::set_atom_type`, the `sm >= 90` arm, which
/// writes a 4-bit field at bits 73..77.
///
/// That field is WIDER than the 3-bit `MemType` field of a load or a store, and
/// it holds different values. Do not mix the two.
///
/// The float types are deliberately absent. On sm >= 90 a float atomic is a
/// DIFFERENT opcode (0x9a3 with a destination, 0x9a6 without) with a different
/// operation selector (`set_atom_op_sm90_float`), so a float type written into
/// this field selects an integer atomic of the same width and quietly computes
/// integer arithmetic on a float bit pattern.
pub const AtomType = enum(u4) {
    u32 = 0,
    i32 = 1,
    u64 = 2,
    i64 = 3,

    /// True for the 64-bit types. A shared atomic accepts these only for
    /// `exch` and for compare-and-swap.
    pub fn isWide(self: AtomType) bool {
        return switch (self) {
            .u32, .i32 => false,
            .u64, .i64 => true,
        };
    }
};

/// `ATOMG`, the global atomic that gives back the old value. sm >= 100 form.
pub const ATOMG_OPCODE: u32 = 0x9a8;
/// `RED`, the global atomic reduction, which gives back nothing.
pub const RED_OPCODE: u32 = 0x98e;
/// `ATOMS`, the shared-memory atomic that gives back the old value.
pub const ATOMS_OPCODE: u32 = 0x98c;
/// `ATOMG.CAS`, the global compare-and-swap.
pub const ATOMG_CAS_OPCODE: u32 = 0x3a9;
/// `ATOMS.CAS`, the shared compare-and-swap.
pub const ATOMS_CAS_OPCODE: u32 = 0x38d;

/// `ATOMG.E.<op> dst, [addr:addr+1], data`: apply `op` to global memory at the
/// 64-bit address in (addr, addr+1) and put the OLD value in `dst`.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpAtom`, the
/// `MemSpace::Global` arm with a destination and an integer type, on the
/// `has_ugpr` and `sm >= 100` path. Opcode 0x9a8; `set_atom_op(87..91)` for the
/// operation, 4 bits wide because a destination form must fit `exch`;
/// `set_reg_addr(24..32, addr, 63)` for the address, whose size bit is at 63 on
/// sm >= 100 and NOT at 90 the way LDG and STG place it; `set_reg_src(32..40)`
/// for the data; `set_field(40..63)` for the 23-bit immediate address offset,
/// which stays 0 because the isel materializes every offset into the address
/// register; `set_ureg_addr(64, ..., 72)` for the uniform base, URZ here, which
/// also sets bit 72; `set_atom_type(73..77)` for the type; `set_mem_order`
/// (77..81) = 0xa, `Strong(System)`; `set_pred_dst(81..84)` = PT for the
/// fault predicate NAK leaves as `Dst::None`; `set_eviction_priority(84..87)`
/// = 1, `Normal`; and `set_bit(91, has_ugpr)`.
///
/// The bit-63 address size is why this form is sm >= 100 only. On Volta through
/// Hopper the same opcode puts that bit at 70. This backend targets Blackwell,
/// which is also why `URZ` is 255 rather than 63.
///
/// Variable latency: the scoreboard scheduler assigns the write barrier for
/// `dst` and the wait on each consumer.
pub fn atomg(dst: u8, addr: u8, data: u8, op: AtomOp, ty: AtomType, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, ATOMG_OPCODE);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, addr); // 64-bit address in (addr, addr+1)
    setBits(&w, 32, 8, data);
    setBits(&w, 40, 23, 0); // immediate address offset
    setBits(&w, 63, 1, 1); // the address GPR is a 64-bit pair
    setBits(&w, 64, 8, URZ); // no uniform base
    setBits(&w, 72, 1, 1); // the uniform base is 64 bits
    setBits(&w, 73, 4, @intFromEnum(ty));
    setBits(&w, 77, 4, 0xa); // STRONG / SYS
    setBits(&w, 81, 3, PT); // no fault predicate
    setBits(&w, 84, 3, 1); // eviction NORMAL
    setBits(&w, 87, 4, @intFromEnum(op));
    setBits(&w, 91, 1, 1); // UGPR mode
    return w;
}

/// `RED.E.<op> [addr:addr+1], data`: apply `op` to global memory at the 64-bit
/// address in (addr, addr+1) and give back NOTHING. This is the reduction form,
/// which every atomic whose result nobody reads should use, because it writes no
/// register and so needs no scoreboard.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpAtom`, the
/// `MemSpace::Global` arm with `self.dst.is_none()` and an integer type. Opcode
/// 0x98e; `set_atom_op(87..90)` for the operation, only 3 bits wide here;
/// `set_reg_src(32..40)` for the data; `set_field(40..64)` for the 24-bit
/// immediate address offset, 0 here; `set_reg_addr(24..32, addr, 90)` for the
/// address, whose size bit stays at 90 in this form; `set_ureg_addr(64, ..., 72)`
/// for the uniform base, URZ here, which also sets bit 72; `set_bit(91, true)`;
/// `set_mem_order(77..81)` = 0xa; `set_eviction_priority(84..87)` = 1; and the
/// common tail's `set_dst(&Dst::None)`, which writes RZ into bits 16..24.
///
/// The operation field is 3 bits, so `exch` (8) does not fit. NAK's `set_field`
/// refuses a value that overflows its range; the assert below refuses it here,
/// because a truncated 8 would silently become `add`. An exchange has to give
/// back the old value anyway, so `atomg` is the correct encoder for it.
pub fn redg(addr: u8, data: u8, op: AtomOp, ty: AtomType, c: Control) Inst {
    std.debug.assert(op != .exch); // does not fit the 3-bit reduction operation field
    var w = base(c);
    setBits(&w, 0, 12, RED_OPCODE);
    setBits(&w, 16, 8, RZ); // NAK's set_dst(Dst::None): the destination field reads RZ
    setBits(&w, 24, 8, addr); // 64-bit address in (addr, addr+1)
    setBits(&w, 32, 8, data);
    setBits(&w, 40, 24, 0); // immediate address offset
    setBits(&w, 64, 8, URZ); // no uniform base
    setBits(&w, 72, 1, 1); // the uniform base is 64 bits
    setBits(&w, 73, 4, @intFromEnum(ty));
    setBits(&w, 77, 4, 0xa); // STRONG / SYS
    setBits(&w, 84, 3, 1); // eviction NORMAL
    setBits(&w, 87, 3, @intFromEnum(op));
    setBits(&w, 90, 1, 1); // the address GPR is a 64-bit pair
    setBits(&w, 91, 1, 1); // UGPR mode
    return w;
}

/// `ATOMS.<op> dst, [addr], data`: apply `op` to workgroup shared memory at the
/// 32-bit window offset in `addr` and put the OLD value in `dst`.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpAtom`, the
/// `MemSpace::Shared` arm without compare-and-swap, on the `has_ugpr` path.
/// Opcode 0x98c; `set_ureg_src(64)` for the uniform base, URZ here (8 bits on
/// sm >= 100); `set_bit(91, true)`; `set_reg_src(32..40)` for the data;
/// `set_atom_op(87..91)` for the operation; `set_reg_src(24..32)` for the
/// address; `set_field(40..64)` for the 24-bit immediate offset, 0 here;
/// `set_field(78..80)` = 0 (`OffsetStride::X1`); `set_dst` at 16..24; and
/// `set_atom_type(73..77)`.
///
/// Like LDS and STS this sets NO memory order and NO eviction priority: NAK
/// asserts a shared access is `Strong(CTA)`/`Normal` and leaves bits 77 and
/// 81..87 at zero.
///
/// A 64-bit shared atomic only exists for `exch` and for compare-and-swap. NAK
/// asserts that ("64-bit Shared atomics only support CmpExch or Exch"), naming
/// `AtomType::U64`; the assert below covers `i64` too, because the limit is the
/// access width, not the signedness. Refusing is safe: a wrong width here
/// corrupts memory that no test in this repository can observe.
///
/// Variable latency, like `atomg`.
pub fn atoms(dst: u8, addr: u8, data: u8, op: AtomOp, ty: AtomType, c: Control) Inst {
    std.debug.assert(!ty.isWide() or op == .exch);
    var w = base(c);
    setBits(&w, 0, 12, ATOMS_OPCODE);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, addr); // 32-bit shared-window offset, ONE register
    setBits(&w, 32, 8, data);
    setBits(&w, 40, 24, 0); // immediate offset
    setBits(&w, 64, 8, URZ); // no uniform base
    setBits(&w, 73, 4, @intFromEnum(ty));
    setBits(&w, 78, 2, 0); // offset stride X1
    setBits(&w, 87, 4, @intFromEnum(op));
    setBits(&w, 91, 1, 1); // UGPR mode
    return w;
}

/// `ATOMG.E.CAS dst, [addr:addr+1], cmp, data`: compare-and-swap in global
/// memory. Read the location, write `data` into it if it equals `cmp`, and put
/// the OLD value in `dst` either way. The caller compares `dst` with `cmp` to
/// learn whether the swap happened.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpAtom`, the
/// `MemSpace::Global` arm with `AtomOp::CmpExch` and a destination. Opcode
/// 0x3a9; `set_reg_addr(24..32, addr, 72)` for the address, whose size bit is at
/// 72 in THIS form; `set_reg_src(32..40)` for the compare operand;
/// `set_field(40..64)` for the 24-bit immediate offset, 0 here;
/// `set_reg_src(64..72)` for the swap data; `set_pred_dst(81..84)` = PT;
/// `set_dst` at 16..24; `set_atom_type(73..77)`; `set_mem_order(77..81)` = 0xa;
/// and `set_eviction_priority(84..87)` = 1.
///
/// This form has NO uniform base and does NOT set bit 91: NAK asserts the
/// uniform address is zero and puts the swap data in bits 64..72 instead, so
/// there is no room for one. That is also why the address size bit moves back to
/// 72 here, where `atomg` puts it at 63.
pub fn atomgCas(dst: u8, addr: u8, cmp: u8, data: u8, ty: AtomType, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, ATOMG_CAS_OPCODE);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, addr); // 64-bit address in (addr, addr+1)
    setBits(&w, 32, 8, cmp);
    setBits(&w, 40, 24, 0); // immediate address offset
    setBits(&w, 64, 8, data);
    setBits(&w, 72, 1, 1); // the address GPR is a 64-bit pair
    setBits(&w, 73, 4, @intFromEnum(ty));
    setBits(&w, 77, 4, 0xa); // STRONG / SYS
    setBits(&w, 81, 3, PT); // no fault predicate
    setBits(&w, 84, 3, 1); // eviction NORMAL
    return w;
}

/// `ATOMS.CAS dst, [addr], cmp, data`: compare-and-swap in workgroup shared
/// memory at the 32-bit window offset in `addr`. The result convention matches
/// `atomgCas`.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpAtom`, the
/// `MemSpace::Shared` arm with `AtomOp::CmpExch`. Opcode 0x38d;
/// `set_reg_src(32..40)` for the compare operand; `set_reg_src(64..72)` for the
/// swap data; `set_reg_src(24..32)` for the address; `set_field(40..64)` for the
/// 24-bit immediate offset, 0 here; `set_field(78..80)` = 0
/// (`OffsetStride::X1`); `set_dst` at 16..24; and `set_atom_type(73..77)`.
///
/// This form does NOT set bit 91 and writes no uniform base: NAK asserts the
/// uniform address is zero and puts the swap data in bits 64..72. It sets no
/// memory order and no eviction priority, for the same reason `atoms` does not.
pub fn atomsCas(dst: u8, addr: u8, cmp: u8, data: u8, ty: AtomType, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, ATOMS_CAS_OPCODE);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, addr); // 32-bit shared-window offset, ONE register
    setBits(&w, 32, 8, cmp);
    setBits(&w, 40, 24, 0); // immediate offset
    setBits(&w, 64, 8, data);
    setBits(&w, 73, 4, @intFromEnum(ty));
    setBits(&w, 78, 2, 0); // offset stride X1
    return w;
}

/// `BAR.SYNC`: the workgroup barrier. Every thread of the CTA waits here, so a
/// shared-memory tile one warp stages is visible to the others after it.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpBar`, which sets the opcode
/// 0xb1d and NOTHING else. `nak/ir.rs` prints `OpBar` as `bar.sync`, and
/// `from_nir.rs` emits it for a `nir_intrinsic_barrier` whose execution scope is
/// the workgroup. The guard predicate in bits 12..15 comes from `base`, which is
/// where NAK's driver loop puts it too (`encode_sm70_shader` calls `set_pred`
/// after each op's own `encode`).
///
/// Two things a caller must get right, neither of them in this encoding:
///
///   - The launch descriptor has to declare the barrier. NAK sets
///     `info.num_control_barriers = 1` next to this op, and `qmd.rs` writes it
///     into the QMD `BARRIER_COUNT` field. A dispatch that leaves BARRIER_COUNT
///     at 0 and runs a kernel with a BAR.SYNC is undefined.
///   - Every thread must reach the barrier. A divergent branch AROUND a
///     BAR.SYNC corrupts a staged shared tile on Blackwell (measured on the
///     GB10). Guard the work with a predicate, or make sure the divergent
///     region is wrapped in the BSSY/BSYNC pair the isel emits, so the warp is
///     reconverged before the barrier.
pub fn barSync(c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0xb1d);
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

/// `BRA target`: relative branch. The branch's taken condition is
/// `Control.pred` (plus `Control.pred_neg`, meaning branch on !pred). PT,
/// the default, means an unconditional branch. `delta` is NAK's relative
/// offset in 32-bit word units (4 words per 128-bit instruction):
/// `target_ip - cur_ip - 4` = `(dst_instr - cur_instr - 1)*4`. Volta BRA
/// 0x947.
///
/// NAK (sm70_encode.rs OpBra) puts the taken condition at the
/// predicate-source field `set_pred_src(87..90, 90, cond)`: bits 87..89
/// hold the predicate register and bit 90 is the negate. This is not the
/// instruction guard at 12..14. A guard-false BRA would mis-execute on
/// Blackwell. The instruction guard stays PT so the BRA always issues.
/// Whether it is taken is decided by the 87..89 predicate. On sm>=100 the
/// relative offset is split across two fields: the low 8 bits at `16..24`
/// and the high 48 bits at `34..82` (`set_rel_offset2(16..24, 34..82)`).
pub fn bra(delta: i32, c: Control) Inst {
    // The instruction guard (12..14) must be PT: the BRA always issues, and
    // the 87..89 condition decides whether it is taken. Force the guard to
    // PT regardless of what was passed in `Control.pred`.
    var guard = c;
    guard.pred = PT;
    guard.pred_neg = false;
    var w = base(guard);
    setBits(&w, 0, 12, 0x947);
    setBits(&w, 32, 1, 0); // !.U (this is a regular, non-uniform BRA)
    // Taken-condition predicate at 87..89, plus negate at 90, like NAK's set_pred_src.
    setBits(&w, 87, 3, c.pred);
    if (c.pred_neg) setBits(&w, 90, 1, 1);
    // The relative offset, split: low 8 bits at 16..24, high 48 bits at
    // 34..82. The combined field is 56-bit signed. Sign-extend `delta` into
    // the high half so a backward branch (negative delta) sets the high
    // bits correctly.
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

/// KIL: discard the current fragment (OpKill). Opcode 0x95b with the
/// pred-source at 87..90 = PT. This is unconditional at this point. A
/// conditional `if (cond) discard` is gated by the surrounding structured
/// control flow. From NAK's SM70Op for OpKill (sm70_encode.rs:
/// set_opcode(0x95b) plus set_pred_src(87..90, 90, True)).
pub fn kil(c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x95b);
    setBits(&w, 87, 3, 7);
    return w;
}

// Convergence barriers (Volta-and-later structured control flow). On Volta
// and later, the warp does not implicitly reconverge at the end of a
// divergent `if`: a divergent branch splits the warp into
// independently-scheduled sub-warps, and any quad-dependent op afterward (a
// TEX texture fetch or a derivative SHFL, both of which rely on all four
// lanes of a 2x2 pixel quad being active in lock-step) reads garbage from
// the lanes that took the other path. NAK (Mesa) wraps every divergent
// control-flow region in a hardware convergence barrier: BSSY sets a
// reconvergence point at a barrier register before the branch, and BSYNC at
// the join forces the sub-warps to rendezvous there, restoring quad
// uniformity before the next quad op. These three encoders mirror NAK's
// sm70_encode.rs byte-for-byte, since the encoding is shared from Volta
// through Blackwell. The convergence-barrier registers (Bar regs B0..B15)
// are a register file separate from the GPRs.

/// `BCLEAR Bbar`: clear or initialize a convergence-barrier register. NAK
/// opcode 0x355 with the .CLEAR bit (84). The GPR dst field (16..24) is RZ,
/// unused. The barrier register index goes in bits 24..28.
pub fn bclear(bar: u4, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x355);
    setBits(&w, 16, 8, RZ); // set_dst(None) -> RZ
    setBits(&w, 24, 4, bar); // set_bar_dst(24..28)
    setBits(&w, 84, 1, 1); // .CLEAR
    return w;
}

/// `BSSY Bbar, target`: set up a convergence barrier at `target` (the join
/// or reconvergence point). Emitted just before a divergent branch. NAK
/// opcode 0x945: barrier-dst at 16..20, a 30-bit relative offset to the
/// target at 34..64, an unconditional predicate (PT) at 87..90 (plus
/// not-bit 90 = 0). `delta` is the byte offset from the next instruction,
/// the same convention as `bra`.
pub fn bssy(bar: u4, delta: i32, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x945);
    setBits(&w, 16, 4, bar); // set_bar_dst(16..20)
    setBits(&w, 34, 30, @as(u30, @truncate(@as(u32, @bitCast(delta))))); // set_rel_offset(34..64)
    setBits(&w, 87, 3, PT); // set_pred_src(87..90, 90, True)
    return w;
}

/// `BSYNC Bbar`: reconverge the warp at the barrier set up by a prior BSSY.
/// Emitted at the join point. NAK opcode 0x941: barrier-src at 16..20,
/// unconditional predicate (PT) at 87..90.
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

/// `ALD dst..dst+comps-1, a[addr]`: load `comps` vertex input-attribute
/// words into consecutive GPRs (a vertex shader reading a fetched
/// attribute). Variable latency: set a write barrier and drain it before
/// the consumer.
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
/// attribute, for example the clip-space position at ATTR_POSITION.
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

/// `IPA dst, a[addr]`: interpolate one component of a fragment input
/// attribute. On SM70 and later, a single IPA does the full
/// perspective-correct interpolation implicitly. `addr` is the attribute
/// byte address (4-aligned). The encoder stores addr>>2. Variable latency
/// like ALD.
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

/// `IPA.CONSTANT dst, a[addr]`: read a flat fragment attribute, with no
/// interpolation. Same opcode as `ipa` but with the interp frequency field
/// (bits 78..80) set to Constant(1): the value is taken as-is from the
/// attribute. NAK emits this for gl_FrontFacing and other flat sysval
/// reads. The interp mode (Constant, ScreenLinear, or Perspective) is not
/// in the instruction. It lives in the SPH imap.
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

// Screen-space derivatives (dFdx/dFdy). The GPU shades fragments in 2x2
// pixel quads whose four lanes are co-resident in the warp. A derivative is
// the difference between a fragment and its quad neighbour, read directly
// from the neighbour's register through a warp shuffle within the quad
// segment. NAK lowers nir_op_fddx/fddy to SHFL.BFLY (XOR the lane index
// with 1 for the horizontal neighbour, 2 for the vertical), then FSWZADD,
// which combines the shuffled neighbour and self into the per-quad gradient
// with the correct per-lane sign.

/// The SHFL quad-shuffle `c` operand NAK passes for a quad derivative: the
/// low 5 bits are the segment width minus one in the low byte (0x03 = a
/// 4-lane quad segment), and the upper byte (0x1c << 8) is the clamp or
/// bound. Matches NAK's `0x3 | (0x1c << 8)` for fddx/fddy.
pub const SHFL_QUAD_C: u16 = 0x1c03;

/// `SHFL.BFLY dst, src, lane=imm, c=imm`: butterfly warp shuffle. Reads the
/// register `src` from the lane whose index is (this_lane XOR `lane_xor`).
/// With `lane_xor` = 1 the source is the horizontal quad neighbour, and
/// with 2 it is the vertical neighbour. `c` packs the quad segment
/// (SHFL_QUAD_C). Both lane and c are immediates, so this uses NAK's
/// all-immediate SHFL form (opcode 0xf89): imm_c at 40..53, imm_lane at
/// 53..58, the BFLY op (3) at 58..60, src at 24, dst at 16. Fixed latency,
/// but it reads `src` from a neighbour lane, so the scheduler waits on
/// `src`'s producer (the IPA that interpolated the varying) exactly like
/// any srcA read.
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

/// `FSWZADD dst, src0, src1`: the quad swizzle-add that finishes a
/// derivative. `src0` is the SHFL'd neighbour, and `src1` is self. The four
/// 2-bit lane ops in `lane_ops` tell each quad lane whether to compute
/// `src1 - src0` (SubRight), `src0 - src1` (SubLeft), or add. NAK uses
/// [SubLeft,SubRight,SubLeft,SubRight] for dFdx and
/// [SubLeft,SubLeft,SubRight,SubRight] for dFdy, giving every lane in the
/// quad the same coarse gradient with the correct sign. NAK SM70 opcode
/// 0x822: src0 at 24, src1 at 64, the packed sub-op at 32..40, the
/// non-divergent derivative mode at bit 77 (required on sm>=100/Blackwell),
/// round-to-nearest at 78..80, ftz off at 80.
pub const SwzOp = enum(u2) { add = 0, sub_left = 1, sub_right = 2, move_left = 3 };
pub fn fswzadd(dst: u8, src0: u8, src1: u8, lane_ops: [4]SwzOp, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x822);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, src0);
    setBits(&w, 64, 8, src1);
    // Pack the four lane ops: op[i] occupies bits ((len-1-i)*2) of the
    // sub-op byte, exactly NAK's `subop |= swz_op << ((ops.len()-i-1)*2)`.
    var subop: u8 = 0;
    inline for (lane_ops, 0..) |o, i| subop |= @as(u8, @intFromEnum(o)) << ((3 - i) * 2);
    setBits(&w, 32, 8, subop);
    setBits(&w, 77, 1, 1); // deriv_mode = NonDivergent (sm>=100, fswzadd.ndv)
    setBits(&w, 78, 2, 0); // round mode = nearest-even
    setBits(&w, 80, 1, 0); // ftz off
    return w;
}

/// `TEX dst..dst+3, [coord:coord+1], handle`: a bindless 2D texture sample.
/// On Blackwell (sm_100/sm_120) the bindless TEX opcode is 0xd61 with the
/// bindless marker at bit 91 (NAK SM70+ encoder, the e.sm >= 100 path). The
/// result RGBA goes to dst..dst+3 (channel mask 0xf). `coord` is the first
/// of a consecutive register pair holding (u, v) as f32 in normalized
/// [0,1] coordinates. The hardware reads the pair from the bit-24 source.
/// `handle` is a 32-bit register holding the bindless texture handle =
/// (TIC index & 0xfffff) | (TSC index << 20). The GPU indexes the bound
/// TEX_HEADER_POOL/TEX_SAMPLER_POOL with it. LOD is forced to level 0
/// (TexLodMode::Zero), so no screen-space derivatives are needed. This is
/// correct for a single-mip sampled image. The dFdx/dFdy auto-LOD path is a
/// later addition. Variable latency: the result lands an unknown number of
/// cycles after issue, so the scheduler sets a write barrier and waits
/// before any consumer reads dst.
/// MUFU (multifunction/special-function unit) opcode as it appears in the
/// encoded instruction's low 12 bits: `alu()` ORs the NAK base 0x108 with
/// the register-srcB form bit (1 << 9 = 0x200), so the scheduler sees
/// 0x308. Variable-latency (decoupled) on sm120, so the scheduler
/// scoreboards it (schedule.isVariableLatency).
pub const MUFU_OPCODE: u32 = 0x308;

pub const TEX_OPCODE: u32 = 0xd61;
/// The uniform zero register (URZ) on sm>=100: NAK's `zero_reg(UGPR) =
/// ugpr_max() = 255` for Blackwell. The bindless TEX's uniform
/// handle/offset operand fields must reference URZ, not uniform register 0,
/// which is garbage. Leaving them 0 faults the SM (Xid 13, "Graphics SM
/// Global Exception").
pub const URZ: u8 = 255;
/// The NAK-style texture dimension field (bits 61-63): selects how many
/// coordinate registers the hardware reads from `coord`, and the texture
/// target. `_2D` (=1) is verified by the working 2D texture path. `_3D`
/// (=2) is verified by a per-slice 3D readback. `cube` (=3) is not a
/// hardware target here. It is an internal marker the isel uses to trigger
/// the cube lowering. The native Blackwell cube TEX modes do not select the
/// face from the direction: dim=3 ignores it, dim=4/5/6 return the border,
/// and dim=7/ARRAY_CUBE reads a stale array layer non-deterministically. So
/// a cube sample is lowered to the major-axis (direction to face plus face
/// u, v) math plus a `_2D` sample of a 6-face-wide atlas
/// (u' = (face+u)/6). The emitted TEX therefore uses `dim_2d`, never 3.
pub const TexDim = struct {
    pub const dim_2d: u8 = 1;
    pub const dim_3d: u8 = 2;
    pub const cube: u8 = 3;
    /// A 2D array (NAK set_tex_dim Array2D = 5): the hardware reads a
    /// 3-register coordinate (layer, u, v) with the layer first, a raw
    /// index, not normalized. Used by `sampler2DArray`. Needs a
    /// TWO_D_ARRAY TIC. Verified on the RTX 5070.
    pub const array_2d: u8 = 5;
};

/// A bindless 2D texture sample (see `tex`): dim = _2D, a coordinate pair.
pub fn tex2d(dst: u8, coord: u8, handle: u8, c: Control) Inst {
    return tex(dst, coord, handle, TexDim.dim_2d, c);
}

/// `TEX dst..dst+3, [coord..], handle` with an explicit dimension. `dim`
/// selects the texture target and the coordinate register count (2D = a
/// coordinate pair, 3D/cube = a coordinate triple). All other fields match
/// the verified 2D encoding.
pub fn tex(dst: u8, coord: u8, handle: u8, dim: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, TEX_OPCODE); // TEX (bindless, sm>=100)
    setBits(&w, 91, 1, 1); // bindless marker (e.set_bit(91, true))
    // On Volta and later, including Blackwell, the 4-channel TEX result is
    // split across two destination register groups (NAK from_nir:
    // dst_comps > 2 -> dsts[0]=dst[0..2], dsts[1]=dst[2..]). dst[0] (bits
    // 16:23) holds components 0,1 (R,G) at dst..dst+1. dst[1] (bits 64:71)
    // holds components 2,3 (B,A) at dst+2..dst+3. For a consecutive RGBA
    // block at `dst`, that means dst[0]=dst (R,G) and dst[1]=dst+2 (B,A).
    // Putting RZ in dst[1], the natural "no second dest", discards B and A:
    // the channels then read back as uninitialized garbage, which is the
    // position-dependent B/A corruption.
    setBits(&w, 16, 8, dst); // dst[0] = R,G at dst, dst+1
    setBits(&w, 64, 8, dst + 2); // dst[1] = B,A at dst+2, dst+3
    setBits(&w, 81, 3, PT); // fault predicate dst = none (PT)
    setBits(&w, 24, 8, coord); // src[0] = coord register pair (u in coord, v in coord+1)
    setBits(&w, 32, 8, handle); // src[1] = the bindless handle register
    // sm>=100 only: the TEX also has two uniform-register operands (the
    // uniform handle at bit 40 and the uniform offset at bit 48), each 8
    // bits. NAK sets both to the uniform zero register (URZ = 255).
    // Leaving them 0 makes the TEX read uniform register 0 as the
    // handle/offset, which is garbage, and the SM faults (Xid 13). This is
    // the load-bearing Blackwell-specific field the from-scratch encoder
    // must replicate.
    setBits(&w, 40, 8, URZ); // uniform handle = URZ
    setBits(&w, 48, 8, URZ); // uniform offset = URZ
    setBits(&w, 60, 1, 0); // .scalar = false
    setBits(&w, 61, 3, dim); // dim = _2D(1), _3D(2), or cube(3); selects coordinate count + target
    setBits(&w, 72, 4, 0xf); // channel_mask = RGBA
    setBits(&w, 76, 2, 0); // deriv_mode = Auto (sm>=100 set_tex_deriv_mode 76..78)
    setBits(&w, 84, 3, 1); // mem eviction priority = Normal (84..87)
    // lod_mode = Auto (0): set_tex_lod_mode2(59..60, 87..90). bit59=0,
    // bits87..89=0. The hardware computes the LOD from the 2x2 fragment
    // quad's screen-space texture-coordinate derivatives (implicit LOD), so
    // a mipmapped texture (TIC MAX_MIP_LEVEL>0 plus TSC MIP_POINT/LINEAR)
    // minifies through its chain. A single-mip texture (MAX_MIP_LEVEL=0)
    // clamps to level 0 regardless, so Auto is correct there too, since the
    // computed LOD just resolves to the only level.
    setBits(&w, 59, 1, 0);
    setBits(&w, 87, 3, 0);
    return w;
}

/// `TEX.LL`: a texture sample with an explicit LOD (GLSL textureLod /
/// textureCubeLod). This is the same opcode as `tex` (OpTex 0xd61. NAK
/// emits OpTex for textureLod/txl, while OpTld/0xd67 is texelFetch with
/// integer coordinates), but with lod_mode = Lod. First, lod_mode through
/// NAK set_tex_lod_mode2(59..60, 87..90) = value 3, split low-bits-first,
/// so bit 59 = 1 and bits 87..89 = 1. Second, and crucial on Blackwell
/// (sm>=120): NAK forces deriv_mode = DerivXY (=3, bits 76..77) whenever
/// lod_mode != Zero. Leaving it Auto, the `tex` default, makes the
/// explicit LOD flaky or ignored. The hardware takes the LOD from the
/// register just past `handle`, NOT from the coordinate block: NAK reads
/// src1 as [handle, lod]. The isel passes handle = coord + 2 for a 2D
/// sample, so the registers are coord, coord+1 = u,v, coord+2 = the
/// bindless handle, and coord+3 = the f32 LOD. `schedule.srcSpan` models
/// that src1 as a 2-register run for this reason.
pub fn texLod(dst: u8, coord: u8, handle: u8, dim: u8, c: Control) Inst {
    var w = tex(dst, coord, handle, dim, c);
    setBits(&w, 59, 1, 1); // lod_mode2 low bit: Lod(3) & 1
    setBits(&w, 87, 3, 1); // lod_mode2 high bits: Lod(3) >> 1
    setBits(&w, 76, 2, 3); // deriv_mode = DerivXY (sm>=120 requires it for lod_mode != Zero)
    return w;
}

/// `TEX.SCR.Z`: a depth-compare texture sample (GLSL sampler2DShadow /
/// SPIR-V OpImageSampleDref). Same OpTex opcode as `tex` but with z_cmpr
/// set (sm70 encoder bit 78, NAK sm70_encode.rs): the hardware compares the
/// shader-supplied reference (dref) against the fetched depth using the
/// TSC's DEPTH_COMPARE_FUNC and returns a single scalar pass fraction (0/1,
/// or PCF-blended) rather than a filtered RGBA vec4. The dref is a src1
/// register right after the handle (src1 = [handle, dref]). NAK
/// nak_nir_lower_tex.c packs z_cmpr last in src1, so with no lod or offset
/// it is handle_reg + 1. The caller must place it there. Because the
/// result is scalar, channel_mask = R only and there is no second
/// destination (dst[1] = RZ), so only `dst` is written. The scheduler spans
/// the write barrier by the channel-mask popcount. LOD is Auto (implicit
/// fragment-quad derivatives), which is correct for a base-level shadow map.
pub fn texShadow(dst: u8, coord: u8, handle: u8, dim: u8, c: Control) Inst {
    var w = tex(dst, coord, handle, dim, c);
    setBits(&w, 78, 1, 1); // z_cmpr: depth-compare (dref at src1[1] = handle+1), scalar result
    setBits(&w, 64, 8, RZ); // no second destination (1-component scalar result)
    setBits(&w, 72, 4, 0x1); // channel_mask = R only (the compare pass fraction)
    return w;
}

/// The bindless texture-gather opcode on Blackwell (sm>=100): NAK OpTld4
/// (`e.sm >= 100 -> set_opcode(0xd64) + set_bit(91)`). Distinct from TEX
/// (0xd61): TLD4 fetches one component of the 4 bilinear-footprint texels
/// rather than a filtered sample.
pub const TLD4_OPCODE: u32 = 0xd64;

/// `TLD4 dst..dst+3, [coord:coord+1], handle`: a bindless texture gather
/// (GLSL textureGather). Returns the single component `comp` (0..3) of the
/// 4 texels of the bilinear footprint at (u,v) as the RGBA result, in the
/// GL gather order (lower-left, lower-right, upper-right, upper-left). The
/// coordinate/handle/dst-split layout matches `tex`: dst[0]=R,G at 16..24,
/// dst[1]=B,A at 64..72, coordinate pair at 24..32, bindless handle at
/// 32..40, URZ uniform operands, bindless marker bit 91. The component
/// select occupies bits 87..88 (NAK `set_field(87..89, comp)`), the same
/// bits `tex` uses for lod_mode2 high, repurposed since a gather has no LOD
/// mode. offset_mode = None (76..78=0). The LOD is implicit (the 2x2-quad
/// derivatives), which is correct for a fragment-stage gather.
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

/// The bindless texel-fetch opcode on Blackwell (sm>=100): NAK OpTld
/// (`e.sm >= 100 -> set_opcode(0xd67) + set_bit(91)`). texelFetch: an exact
/// texel at integer coordinates plus an explicit LOD, with no filter.
pub const TLD_OPCODE: u32 = 0xd67;

/// `TLD dst..dst+3, [coord:coord+1], [handle:lod]`: a bindless texel fetch
/// (GLSL texelFetch). The coordinate registers hold integer texel
/// coordinates (x in coord, y in coord+1), not normalized floats. Like
/// TEX.LL, it uses Lod mode (an explicit LOD): the hardware reads the LOD
/// from src1[1] = handle_reg + 1, so `handle` is the first of a consecutive
/// (handle, lod) pair. Same dst-split and URZ operands as `tex`.
/// offset_mode = None at bits 56..58 (sm>=100). No filtering or
/// normalization (integer fetch).
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

// Tensor-core instructions. HMMA and IMMA multiply two small matrices and add a
// third. LDSM loads the matrix fragments out of shared memory, and MOVM
// transposes one 8x8 fragment inside the registers of a warp.
//
// These four are WARP-COLLECTIVE. All 32 lanes of the warp run one instruction
// together, and each lane holds a slice of each matrix in its OWN registers. A
// lane holds a RUN of registers per matrix, not one register, so the operand of
// each encoder below names the FIRST register of that run. The run length comes
// from the tile shape and the element type. See the `nvidia-hmma-design-study`
// note for the per-lane element map and for the run lengths.
//
// The encodings come from Mesa NAK `sm70_encode.rs`, which is the authoritative
// bit-level reference (see the `nak-sass-encoding-reference` note). None of them
// is hardware-verified here, because this repository cannot execute SASS.
//
// Nothing emits them yet. The IR `matmul` reads POINTERS to row-major memory and
// a tensor core reads REGISTERS in the per-lane fragment layout, so a lowering
// needs a load-and-shuffle stage that does not exist. These are encoder-only
// until it does, in the same way `barSync` and the atomics were.

/// HMMA, the half-precision matrix multiply-accumulate.
pub const HMMA_OPCODE: u32 = 0x23c;
/// IMMA, the integer matrix multiply-accumulate.
pub const IMMA_OPCODE: u32 = 0x237;
/// LDSM, the load of matrix fragments out of shared memory.
pub const LDSM_OPCODE: u32 = 0x83b;
/// MOVM, the in-register transpose of one 8x8 fragment of 16-bit elements.
pub const MOVM_OPCODE: u32 = 0x23a;

/// The tile an HMMA computes, named as `m` by `n` by `k`. Values from NAK
/// `sm70_encode.rs`, `impl SM70Op for OpHmma`: the selector is split across bit
/// 75 (the low bit) and bit 78 (the high bit) by `set_field2(75..76, 78..79)`.
///
/// `m16n8k4` needs sm >= 80, which this Blackwell backend always satisfies. It is
/// the tf32 shape, and the source-type field below has no confirmed tf32 value, so
/// no caller can reach it usefully yet. It stays listed because the selector value
/// is confirmed and a missing name would invite a guess later.
pub const HmmaSize = enum(u2) { m16n8k8 = 0, m16n8k16 = 1, m16n8k4 = 2 };

/// The element type of the HMMA accumulator and result. NAK writes bit 76 for
/// this and asserts the type is one of these two.
pub const HmmaDstType = enum(u1) { f16 = 0, f32 = 1 };

/// `HMMA dst, a, b, c`: the fp16 matrix multiply-accumulate `dst = a * b + c`,
/// run by all 32 lanes of the warp together.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpHmma`. Opcode 0x23c (bits
/// 0..12); `set_dst` at 16..24; `set_reg_src(24..32)` for the A fragment,
/// `set_reg_src(32..40)` for B and `set_reg_src(64..72)` for C, each the FIRST
/// register of that lane's run; `set_field2(75..76, 78..79)` for the tile;
/// `set_bit(76, dst_type == F32)`; and `set_field(82..84)` for the source type,
/// where F16 is 0.
///
/// The source type has only ONE confirmed value. NAK leaves BF16 (1) and TF32 (2)
/// commented out in that same match and hits `unreachable!` for them, so this
/// encoder writes 0 and takes no source-type parameter. A guess there would send
/// the tensor core a different element type and give silently wrong numbers.
///
/// HMMA sets NO bit 74, unlike IMMA below. That difference is in the NAK source,
/// not a transcription slip.
///
/// On sm >= 90 NAK also writes `set_rev_upred_src(87..90, 90, &true.into())`.
/// That path encodes the always-true uniform predicate, whose register index is 7,
/// REVERSED as `7 - 7 = 0`, and clears the negate bit at 90. Every bit it writes
/// is a zero, so it leaves the instruction unchanged and this encoder omits it.
/// The test pins bits 87..91 at zero so a later edit cannot break that silently.
pub fn hmma(dst: u8, a: u8, b: u8, c_in: u8, size: HmmaSize, dst_type: HmmaDstType, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, HMMA_OPCODE);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, a);
    setBits(&w, 32, 8, b);
    setBits(&w, 64, 8, c_in);
    const tile = @intFromEnum(size);
    setBits(&w, 75, 1, tile & 1); // tile selector, low bit
    setBits(&w, 78, 1, tile >> 1); // tile selector, high bit
    setBits(&w, 76, 1, @intFromEnum(dst_type));
    setBits(&w, 82, 2, 0); // source type F16, the only confirmed value
    return w;
}

/// The tile an IMMA computes. Values from NAK `sm70_encode.rs`, `impl SM70Op for
/// OpImma`: the 3-bit selector is split across bit 75 (the low bit) and bits
/// 85..87 (the two high bits) by `set_field2(75..76, 85..87)`. The gaps at 1, 3
/// and 7 are gaps in the NAK table, not values to fill in.
///
/// Every shape except `m8n8k16` needs sm >= 80, which this Blackwell backend
/// always satisfies.
pub const ImmaSize = enum(u3) { m8n8k16 = 0, m8n8k32 = 2, m16n8k16 = 4, m16n8k32 = 5, m16n8k64 = 6 };

/// One IMMA input operand: its signedness and its element width. NAK writes the
/// signedness at bit 76 for A and bit 78 for B, and the "this operand is 4-bit" flag
/// at bit 83 for A and bit 84 for B.
pub const ImmaOperand = struct {
    /// Whether the elements are signed. False means unsigned.
    signed: bool,
    /// Whether the elements are 4 bits wide. False means 8 bits wide.
    four_bit: bool = false,
};

/// Whether `size` accepts an operand of `four_bit` width. NAK asserts this table
/// in `impl SM70Op for OpImma` before it writes bits 83 and 84: the k16 shapes take
/// 8-bit operands only, `m8n8k32` and `m16n8k64` take 4-bit operands only, and
/// `m16n8k32` takes either.
fn immaWidthFits(size: ImmaSize, four_bit: bool) bool {
    return switch (size) {
        .m8n8k16, .m16n8k16 => !four_bit,
        .m8n8k32, .m16n8k64 => four_bit,
        .m16n8k32 => true,
    };
}

/// `IMMA dst, a, b, c`: the integer matrix multiply-accumulate `dst = a * b + c`,
/// run by all 32 lanes of the warp together. The accumulator and the result are
/// int32 whatever the input width is.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpImma`. Opcode 0x237 (bits
/// 0..12); `set_dst` at 16..24; `set_reg_src(24..32)` for the A fragment,
/// `set_reg_src(32..40)` for B and `set_reg_src(64..72)` for C, each the FIRST
/// register of that lane's run; `set_bit(74, true)`, which NAK names SRC1.COL;
/// `set_field2(75..76, 85..87)` for the tile; `set_bit(76)` and `set_bit(78)` for
/// the signedness of A and of B; `set_bit(82)` for saturation; and `set_bit(83)`
/// and `set_bit(84)` for a 4-bit A and a 4-bit B.
///
/// `saturate` clamps the int32 result to the input range instead of letting it
/// wrap.
///
/// On sm >= 90 NAK also writes `set_rev_upred_src(87..90, 90, &true.into())`,
/// which writes only zeros. See `hmma` for why this encoder omits it.
pub fn imma(
    dst: u8,
    a: u8,
    b: u8,
    c_in: u8,
    size: ImmaSize,
    a_op: ImmaOperand,
    b_op: ImmaOperand,
    saturate: bool,
    c: Control,
) Inst {
    // The hardware has no encoding for a width the tile does not take. NAK asserts
    // the same table before it writes bits 83 and 84.
    std.debug.assert(immaWidthFits(size, a_op.four_bit));
    std.debug.assert(immaWidthFits(size, b_op.four_bit));
    var w = base(c);
    setBits(&w, 0, 12, IMMA_OPCODE);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, a);
    setBits(&w, 32, 8, b);
    setBits(&w, 64, 8, c_in);
    setBits(&w, 74, 1, 1); // SRC1.COL
    const tile = @intFromEnum(size);
    setBits(&w, 75, 1, tile & 1); // tile selector, low bit
    setBits(&w, 85, 2, tile >> 1); // tile selector, two high bits
    setBits(&w, 76, 1, @intFromBool(a_op.signed));
    setBits(&w, 78, 1, @intFromBool(b_op.signed));
    setBits(&w, 82, 1, @intFromBool(saturate));
    setBits(&w, 83, 1, @intFromBool(a_op.four_bit));
    setBits(&w, 84, 1, @intFromBool(b_op.four_bit));
    return w;
}

/// How many 8x8 fragments one LDSM loads. Values from NAK `sm70_encode.rs`,
/// `impl SM70Op for OpLdsm`, the `set_field(72..74)` match: a count of 1 encodes
/// as 0, 2 as 1 and 4 as 2. NAK panics on any other count.
pub const LdsmCount = enum(u2) {
    x1 = 0,
    x2 = 1,
    x4 = 2,

    /// The number of fragments this count names. Each fragment gives one lane 32
    /// bits, so this is also the number of destination registers the load fills.
    pub fn matrices(self: LdsmCount) u32 {
        return switch (self) {
            .x1 => 1,
            .x2 => 2,
            .x4 => 4,
        };
    }
};

/// `LDSM dst, [addr]`: load `count` fragments of 8 by 8 16-bit elements out of
/// workgroup shared memory into the registers of the warp, in the per-lane layout
/// a following HMMA or IMMA reads.
///
/// Every lane gives its OWN address, so the 32 addresses of the warp select the
/// rows, and the hardware then spreads each row across the lanes. That is the
/// point of the instruction: a plain LDS gives each lane the elements at its own
/// address, and the fragment layout needs elements that live in another lane.
///
/// `transpose` selects NAK's `LdsmSize::MT8N8` instead of `M8N8`, which reads each
/// 8 by 8 fragment transposed. NAK leaves `M8N8Nx2` (2) and `M8N8Nx4` (3) commented
/// out, so those two values stay out of this encoder.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpLdsm`. Opcode 0x83b (bits
/// 0..12); `set_dst` at 16..24, the FIRST of `count` destination registers;
/// `set_reg_src(24..32)` for the address, ONE register holding a 32-bit offset into
/// the shared window, the same form LDS and STS take; `set_ureg_src(32)` for the
/// uniform base, URZ here because there is none (8 bits wide on sm >= 100);
/// `set_field(40..64)` for the 24-bit immediate offset, 0 here because a lowering
/// would materialize every offset into the address register; `set_field(72..74)`
/// for the fragment count; and `set_field(78..80)` for the transpose.
///
/// LDSM writes bit 91 as `!uniform_addr.is_zero()`, which is FALSE here. LDS and
/// STS write that same bit as TRUE, because NAK always enables UGPR mode for them.
/// The two are different rules in the NAK source, not a transcription slip.
pub fn ldsm(dst: u8, addr: u8, count: LdsmCount, transpose: bool, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, LDSM_OPCODE);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, addr); // 32-bit shared-window offset, ONE register
    setBits(&w, 32, 8, URZ); // no uniform base
    setBits(&w, 40, 24, 0); // immediate offset
    setBits(&w, 72, 2, @intFromEnum(count));
    setBits(&w, 78, 2, @intFromBool(transpose)); // M8N8 = 0, MT8N8 = 1
    setBits(&w, 91, 1, 0); // no uniform address
    return w;
}

/// `MOVM dst, src`: transpose one 8 by 8 fragment of 16-bit elements across the
/// registers of the warp. A fragment loaded in row order needs this before it can
/// feed the column operand of an HMMA.
///
/// Source: NAK `sm70_encode.rs`, `impl SM70Op for OpMovm`. Opcode 0x23a (bits
/// 0..12); `set_dst` at 16..24; `set_reg_src(24..32)` for the source; and
/// `set_field(78..80, 0)`, the MT88 mode. NAK marks the other two modes of that
/// field as a TODO and gives them no names, so this encoder writes only MT88.
pub fn movm(dst: u8, src: u8, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, MOVM_OPCODE);
    setBits(&w, 16, 8, dst);
    setBits(&w, 24, 8, src);
    setBits(&w, 78, 2, 0); // MT88
    return w;
}

/// Shader attribute addresses: the clip-space position output, and the
/// first generic varying or vertex input. System-value index for the
/// vertex ID.
pub const ATTR_POSITION: u16 = 0x70;
pub const ATTR_GENERIC0: u16 = 0x80;
/// The vertex-ID special register (Volta SV_VERTEXID). For a non-indexed
/// draw with SET_VERTEX_ID_BASE = 0, this is exactly Vulkan's
/// gl_VertexIndex. A vertex shader that pulls its vertices from a UBO array
/// sources gl_VertexIndex from here (S2R).
pub const SR_VERTEX_ID: u8 = 0x2f;
/// The instance-ID special register (Volta SV_INSTANCEID) = gl_InstanceIndex
/// with SET_GLOBAL_BASE_INSTANCE_INDEX = 0.
pub const SR_INSTANCE_ID: u8 = 0x2e;

/// The Data-Assembler-delivered vertex-ID/instance-ID attribute addresses
/// (NAK_ATTR_VERTEX_ID/NAK_ATTR_INSTANCE_ID). On Volta and later, a vertex
/// shader reads gl_VertexIndex/gl_InstanceIndex from the attribute
/// interface (ALD), not from a special register. The fixed-function DA
/// writes the per-vertex ID into the attribute RAM at these addresses. This
/// is what NAK emits for SystemValue VertexId/InstanceId. The SPH must
/// declare it consumes the sysval (imap_sys). SET_DA_OUTPUT
/// vertex_id_uses_array_start plus SET_VERTEX_ID_BASE = 0, set in the
/// draw-state init, make the delivered value Vulkan's gl_VertexIndex for a
/// non-indexed draw.
pub const ATTR_VERTEX_ID: u16 = 0x2fc;
pub const ATTR_INSTANCE_ID: u16 = 0x2f8;

/// gl_FrontFacing's attribute address (NAK_ATTR_FRONT_FACE). A fragment
/// shader reads it as a flat (constant-frequency) attribute: the raster
/// delivers a nonzero value for a front-facing primitive, zero for back.
/// The SPH must declare it through the sysval imap (imap_system_values_c
/// bit for a[0x3fc]).
pub const ATTR_FRONT_FACE: u16 = 0x3fc;

/// gl_PointCoord's attribute address (NAK_ATTR_POINT_SPRITE_S/T). A
/// fragment shader reads the point-sprite s coordinate at a[0x2e0] and t at
/// a[0x2e4], IPA'd with normal (perspective-free SCREEN_LINEAR) frequency
/// across the sprite quad. The SPH must declare these two inputs through
/// the imap (IMAP_POINT_SPRITE_S/T, MW bits 344/345), and the draw state
/// must enable SET_POINT_SPRITE plus SET_POINT_SPRITE_SELECT, done once.
pub const ATTR_POINT_SPRITE: u16 = 0x2e0;

/// The constant bank a graphics shader reads its bound UBO base addresses
/// and bindless texture handles from. It is the hardware root table 1
/// (root table T is exposed as constant bank
/// `graphics_root_table_first_cb + T`), not a bound external cbuf. On
/// Blackwell, the old LOAD_CONSTANT_BUFFER-to-LDC-c[0] path is not coherent
/// at high TPC occupancy: a tall render target lights up more TPCs, the
/// LDC returns 0, and the uniform LDG faults at 0x0. The hardware root
/// table, written through SET_ROOT_TABLE_SELECTOR plus LOAD_ROOT_TABLE, is
/// coherent. This is nvk's Blackwell path. See
/// [[prism-glmark2-perf-cliff]]. The dispatch side (prism's draw path)
/// writes each bound UBO's 64-bit VA and each sampler's handle into root
/// table 1 at `graphics_ubo_cb_base + slot*8`. The shader prologue loads
/// the pair back with two LDCs from `graphics_const_bank`.
pub const graphics_root_table: u3 = 1;
pub const graphics_root_table_first_cb: u5 = 24; // NVK_HW_ROOT_TABLE_FIRST_CB: root table T = c[24+T]
pub const graphics_const_bank: u5 = graphics_root_table_first_cb + graphics_root_table; // c[25]

/// Root-table-1 byte offset where the UBO base addresses and texture
/// handles live (SET_ROOT_TABLE_SELECTOR's offset field is 8-bit, so this
/// plus slot*8 for 8 slots must stay under 256). Was 0x140 in the old
/// bound-cb0 scheme, rebased into the 256-byte root table 1.
pub const graphics_ubo_cb_base: u16 = 0x40;

/// Root-table-1 byte offset of the cube half-texel (0.5 / face_width, as
/// f32) the dispatch side writes when a cubemap is bound. The cube
/// lowering reads it (LDC
/// c[graphics_const_bank][cube_halftexel_cb]) to clamp the within-face u to
/// [half_texel, 1 - half_texel], so a linear tap near a face edge stays
/// inside the face's atlas column. Below graphics_ubo_cb_base's per-slot area.
pub const cube_halftexel_cb: u16 = 0x00;

/// Special-register indices for `s2r`. Every index here comes from Mesa NAK's
/// `enum nak_sv` in `src/nouveau/compiler/nak_private.h`, which is the
/// authoritative list of the Volta-and-later special registers. Do not derive a
/// new index by counting from a known one, because the list has holes: 0x24 lies
/// between the thread-id group and the block-id group and is not a grid axis.
pub const SR_LANEID: u8 = 0x00; // NAK_SV_LANE_ID: the warp lane index (0..31); the first special register on Volta and later
pub const SR_TID_X: u8 = 0x21; // NAK_SV_TID_X: threadIdx.x
pub const SR_TID_Y: u8 = 0x22; // NAK_SV_TID_Y: threadIdx.y
pub const SR_TID_Z: u8 = 0x23; // NAK_SV_TID_Z: threadIdx.z
pub const SR_CTAID_X: u8 = 0x25; // NAK_SV_CTAID_X: blockIdx.x
pub const SR_CTAID_Y: u8 = 0x26; // NAK_SV_CTAID_Y: blockIdx.y
pub const SR_CTAID_Z: u8 = 0x27; // NAK_SV_CTAID_Z: blockIdx.z

/// The thread-id and the block-id special registers by axis, x first. A caller
/// that holds an axis index reads the register out of these tables instead of
/// doing arithmetic on `SR_TID_X` or `SR_CTAID_X`.
pub const sr_tid = [3]u8{ SR_TID_X, SR_TID_Y, SR_TID_Z };
pub const sr_ctaid = [3]u8{ SR_CTAID_X, SR_CTAID_Y, SR_CTAID_Z };

test "the grid special-register indices match NAK's nak_sv list" {
    // A wrong index here still assembles and still runs. The kernel then reads a different
    // axis, or a register that holds something else, and computes wrong answers with no
    // fault. So the numbers are pinned against the reference list.
    try std.testing.expectEqual([3]u8{ 0x21, 0x22, 0x23 }, sr_tid);
    try std.testing.expectEqual([3]u8{ 0x25, 0x26, 0x27 }, sr_ctaid);
    // The hole at 0x24. This records that the two groups are separate lists.
    try std.testing.expectEqual(@as(u8, 2), SR_CTAID_X - SR_TID_Z);
}

test "MOV imm matches the hardware-verified encoding" {
    const w = movImm(2, 0xcafe, .{});
    try std.testing.expectEqual(@as(u32, 0x802), w[0] & 0xfff); // ALU MOV (0x002) form 4
    try std.testing.expectEqual(@as(u32, 0xcafe), w[1]); // immediate in dword 1
    try std.testing.expectEqual(@as(u32, 2), (w[0] >> 16) & 0xff); // dst R2
}

test "STG matches the hardware-verified bits (prism, live on Blackwell)" {
    // The word prism proved live carried the memory order STRONG/SYS in bits
    // 77..80, which is 0xa and adds 0x14000 to dword 2. The default order is
    // now weak, the same order ptxas and NAK give an ordinary store, so the
    // one field differs from the recorded word and every other bit matches.
    const w = stgU32(0, 2, .{});
    try std.testing.expectEqual(@as(u32, 0x00007986), w[0]);
    try std.testing.expectEqual(@as(u32, 0x00000002), w[1]);
    try std.testing.expectEqual(@as(u32, 0x0c1009ff), w[2]);
    try std.testing.expectEqual(@as(u32, 0x0c1149ff), stgOrdered(0, 2, 0, .b32, .strong_sys, .{})[2]);
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
    // `p2 = p0 AND p1` (dst P2, srcs P0/P1, third src PT). NAK OpPLop3 warp
    // form 0x81c: LUT split bits 64..67 (low 3) plus 72..77 (high 5), src0
    // at 87..90, src1 at 77..80, src2(PT) at 68..71, dst0 at 81..84, dst1(PT)
    // at 84..87.
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

test "the 64-bit add chain sets the .X bit and names the carry predicate (NAK OpIAdd3X)" {
    // BIT 74 IS THE `.X` FLAG. NAK's OpIAdd3X encodes the same 0x010 opcode as
    // a plain IADD3 and then calls `e.set_bit(74, true)`. Without it the
    // hardware ignores bits 87..90 and the high half of every 64-bit add drops
    // its carry. `nvdisasm -b SM120` prints `IADD3 R11, PT, PT, R7, RZ, RZ` for
    // the word with bit 74 clear and
    // `IADD3.X R11, PT, PT, R7, RZ, RZ, P6, !PT` for the word with it set.
    const lo = iadd3CarryOut(10, 6, 9, 6, .{});
    const hi = iadd3CarryIn(11, 7, RZ, 6, .{});

    try std.testing.expectEqual(@as(u32, 0x210), lo[0] & 0xfff); // IADD3, register form
    try std.testing.expectEqual(@as(u32, 0x210), hi[0] & 0xfff); // same opcode for .X
    try std.testing.expectEqual(@as(u32, 0), (lo[2] >> (74 - 64)) & 0x1); // the low half is not .X
    try std.testing.expectEqual(@as(u32, 1), (hi[2] >> (74 - 64)) & 0x1); // the high half IS .X

    // The carry travels through P6: written by the low add at 81..83, read by
    // the high add at 87..89 with the negate bit clear.
    try std.testing.expectEqual(@as(u32, 6), (lo[2] >> (81 - 64)) & 0x7);
    try std.testing.expectEqual(@as(u32, 6), (hi[2] >> (87 - 64)) & 0x7);
    try std.testing.expectEqual(@as(u32, 0), (hi[2] >> (90 - 64)) & 0x1);

    // Both carry-in operands of a plain add, and the second of an extended
    // add, are the constant false: PT with the negate bit set.
    const plain = iadd3(3, 1, 2, .{});
    try std.testing.expectEqual(@as(u32, PT), (plain[2] >> (87 - 64)) & 0x7);
    try std.testing.expectEqual(@as(u32, 1), (plain[2] >> (90 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, PT), (plain[2] >> (77 - 64)) & 0x7);
    try std.testing.expectEqual(@as(u32, 1), (plain[2] >> (80 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, PT), (hi[2] >> (77 - 64)) & 0x7);
    try std.testing.expectEqual(@as(u32, 1), (hi[2] >> (80 - 64)) & 0x1);

    // A plain add carries neither the .X bit nor a carry-out predicate.
    try std.testing.expectEqual(@as(u32, 0), (plain[2] >> (74 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, PT), (plain[2] >> (81 - 64)) & 0x7);
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
    // FSETP P0, R6, R7, .gt: a float (ordered) comparison. NAK OpFSetP
    // opcode 0x00b, the float compare op at 76..80, result predicate at 81..84.
    const w = fsetp(0, 6, 7, .gt, .{});
    try std.testing.expectEqual(@as(u32, 0x20b), w[0] & 0xfff); // base 0x00b | reg form (1<<9)
    try std.testing.expectEqual(@as(u32, 6), (w[0] >> 24) & 0xff); // srcA R6
    try std.testing.expectEqual(@as(u32, 7), (w[1] >> 0) & 0xff); // srcB R7 at bit 32
    try std.testing.expectEqual(@as(u32, @intFromEnum(Cmp.gt)), (w[2] >> (76 - 64)) & 0xf); // float cmp at 76..80
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (81 - 64)) & 0x7); // P0 at 81..84
    // The opcode must differ from ISETP (0x00c) so a float compare is not an integer one.
    try std.testing.expect((w[0] & 0xfff) != (isetp(0, 6, 7, .gt, false, .{})[0] & 0xfff));
}

test "FMUL sets the PDIV field, and every other bit of 81..90 stays clear" {
    // The bit-for-bit shape of a ptxas sm_120 cubin. Word 2 holds bits 64..95:
    //
    //     FADD R5, R6, R7      06057221 00000007 00000000 140fe200
    //     FMUL R9, R6, R7      06097220 00000007 00400000 041fe200
    //     FFMA R5, R4, R5, R7  04057223 00000005 00000007 001fca00
    //
    // FMUL sets ONE field above bit 71: PDIV at 84..86 = 4. FADD and FFMA set nothing
    // there, and a PT written into 81..90 makes FADD a bfloat16 add. See the note above
    // `fadd`. Bits 64..71 are the srcC REGISTER slot, which these encoders fill with RZ
    // where ptxas leaves 0; a two-source float op ignores that slot, and FMUL with RZ in it
    // is proven on hardware, so the check starts above it.
    const m = fmul(8, 4, 6, .{});
    try std.testing.expectEqual(@as(u32, 0x220), m[0] & 0xfff); // FMUL base 0x020 | reg form
    try std.testing.expectEqual(@as(u32, 4), (m[2] >> (84 - 64)) & 0x7); // PDIV field = 4
    try std.testing.expectEqual(@as(u32, 0x4000), m[2] >> 8); // PDIV, and NOTHING else

    const a = fadd(8, 4, 6, .{});
    try std.testing.expectEqual(@as(u32, 0), a[2] >> 8);
    const s = fsub(8, 4, 6, .{});
    try std.testing.expectEqual(@as(u32, 0), s[2] >> 8);
    const f = ffma(8, 4, 6, 5, .{});
    try std.testing.expectEqual(@as(u32, 5), f[2] & 0xff); // srcC R5 at bits 64..71
    try std.testing.expectEqual(@as(u32, 0), f[2] >> 8); // and nothing above it
}

test "the immediate FFMA is form 4: the multiplier at src1, the addend still a register" {
    // `FFMA R6, R4, 0.25, R7`. The form field is the whole difference between this and a
    // wrong encoding, and getting it wrong is silent: `faddImm` is FORM 2 because FADD
    // passes its immediate as src2, and a live GPU run of `x + 0.25` returned 0 while it
    // was encoded as form 4. FFMA keeps a real addend at src2, so its immediate is src1
    // and the form is 4, the same pair `imadImm` encodes.
    //
    // `nvdisasm -b SM120 -c` reads this exact word back as `FFMA R6, R4, 0.25, R7`.
    const quarter: u32 = @bitCast(@as(f32, 0.25));
    const w = ffmaImm(6, 4, quarter, 7, .{});
    try std.testing.expectEqual(@as(u32, 0x823), w[0] & 0xfff); // 0x023 with form 4 at bits 9..11
    try std.testing.expectEqual(@as(u32, 6), (w[0] >> 16) & 0xff); // dst R6
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // srcA R4
    try std.testing.expectEqual(quarter, w[1]); // the multiplier at bits 32..63
    try std.testing.expectEqual(@as(u32, 7), w[2] & 0xff); // the addend R7 at bits 64..71
    // Bits 81..90 stay clear, exactly as the register FFMA and FADD leave them. Writing PT
    // into them turned the register FADD into a bfloat16 add. See the note above `fadd`.
    try std.testing.expectEqual(@as(u32, 0), w[2] >> 8);
}

test "the addend-immediate FFMA is form 2: the multiplier at bits 64, the addend at 32" {
    // `FFMA R6, R4, R9, 0.05`. The form is the whole difference between this and `ffmaImm`,
    // and the two forms put their operands in OPPOSITE halves of the same two fields: form 2
    // keeps the multiplier REGISTER at bits 64..71 and moves the addend IMMEDIATE into
    // bits 32..63. ptxas emits exactly this opcode (0x423) for the contracted
    // `acc = acc * 0.9 + 0.05` loop shape on sm_120.
    //
    // `nvdisasm -b SM120 -c` reads this exact word back as `FFMA R6, R4, R9, 0.05`.
    const five_hundredths: u32 = @bitCast(@as(f32, 0.05));
    const w = ffmaAddendImm(6, 4, 9, five_hundredths, .{});
    try std.testing.expectEqual(@as(u32, 0x423), w[0] & 0xfff); // 0x023 with form 2 at bits 9..11
    try std.testing.expectEqual(@as(u32, 6), (w[0] >> 16) & 0xff); // dst R6
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // srcA R4
    try std.testing.expectEqual(five_hundredths, w[1]); // the addend at bits 32..63
    try std.testing.expectEqual(@as(u32, 9), w[2] & 0xff); // the multiplier R9 at bits 64..71
    // Bits 81..90 stay clear, exactly as every other float ALU op leaves them.
    try std.testing.expectEqual(@as(u32, 0), w[2] >> 8);
    // The two FFMA immediate forms differ in the form field and in which operand each
    // later field holds, and nothing else: the same three logical operands, swapped.
    const form4 = ffmaImm(6, 4, @bitCast(@as(f32, 0.9)), 9, .{});
    try std.testing.expectEqual(@as(u32, 0x823), form4[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 9), form4[2] & 0xff); // the addend register at 64
}

test "the operand-reuse bits land at 122..125, one per ALU source field" {
    // ptxas emits these bits on sm_120 (the guard in NAK's `set_instr_deps` that
    // gates them on sm < 120 is stale), and the hardware runs such code. The mapping
    // is verified against `nvdisasm -b SM120` one bit at a time: bit 122 marks the
    // operand at bits 24..31, bit 123 the second source (bits 32..39 in form 1, and
    // bits 64..71 in form 2, which is where the form puts the second source), and bit
    // 124 the third source (bits 64..71 in forms 1 and 4). See `markReuse`.
    const w = ffma(6, 4, 7, 8, .{ .reuse_mask = 0b0111 });
    try std.testing.expectEqual(@as(u32, 0b0111), (w[3] >> 26) & 0xf);
    try std.testing.expectEqual(@as(u32, 0), ffma(6, 4, 7, 8, .{})[3] >> 26);
    // The bits sit beside the wait mask, above bit 121, and touch nothing else.
    const a = ffma(6, 4, 7, 8, .{ .stall = 15, .wait_mask = 0x3f });
    const b = ffma(6, 4, 7, 8, .{ .stall = 15, .wait_mask = 0x3f, .reuse_mask = 0b1001 });
    try std.testing.expectEqual(@as(u32, 0b1001) << 26, a[3] ^ b[3]);
}

test "the immediate IMAD keeps its addend, which is what an integer multiply-add needs" {
    // A plain integer multiply is `imadImm(dst, a, imm, RZ, ...)`, so the addend field is
    // the ONLY difference between it and a contracted `base + index * stride`. This pins
    // that the field carries a real register and that the result predicate stays PT, which
    // Blackwell rejects when it is left at 0 (P0).
    const w = imadImm(6, 4, 3, 7, .{});
    try std.testing.expectEqual(@as(u32, 0x824), w[0] & 0xfff); // 0x024 with form 4
    try std.testing.expectEqual(@as(u32, 3), w[1]); // the multiplier at bits 32..63
    try std.testing.expectEqual(@as(u32, 7), w[2] & 0xff); // the addend R7, not RZ
    try std.testing.expectEqual(@as(u32, PT), (w[2] >> (81 - 64)) & 0x7); // result predicate
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

test "LDS reads shared memory through ONE address register (NAK OpLd, MemSpace::Shared)" {
    const w = ldsU32(6, 4, .{});
    // Every dword, so a stray bit anywhere in the 128-bit word fails this.
    try std.testing.expectEqual([4]u32{ 0x04067984, 0x000000ff, 0x0b800800, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, 0x984), w[0] & 0xfff); // LDS, not LDG (0x981)
    try std.testing.expectEqual(@as(u32, 6), (w[0] >> 16) & 0xff); // dst R6
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // address R4
    try std.testing.expectEqual(@as(u32, URZ), w[1] & 0xff); // uniform base URZ at bit 32
    try std.testing.expectEqual(@as(u32, 0), (w[1] >> 8) & 0xffffff); // immediate offset 0 at 40..64
    try std.testing.expectEqual(@as(u32, 4), (w[2] >> (73 - 64)) & 0x7); // mem type B32
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (78 - 64)) & 0x3); // offset stride X1
    try std.testing.expectEqual(@as(u32, PT), (w[2] >> (87 - 64)) & 0x7); // UPT, unconditional
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (91 - 64)) & 0x1); // UGPR mode
    // A shared access carries NO 64-bit-address bit and NO memory order, unlike LDG.
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (90 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (81 - 64)) & 0x3f); // 81..87 clear
}

test "STS writes shared memory through ONE address register (NAK OpSt, MemSpace::Shared)" {
    const w = stsU32(4, 6, .{});
    try std.testing.expectEqual([4]u32{ 0x04007988, 0x00000006, 0x080008ff, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, 0x988), w[0] & 0xfff); // STS, not STG (0x986)
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // address R4
    try std.testing.expectEqual(@as(u32, 6), w[1] & 0xff); // data R6 at bit 32
    try std.testing.expectEqual(@as(u32, 0), (w[1] >> 8) & 0xffffff); // immediate offset 0 at 40..64
    try std.testing.expectEqual(@as(u32, URZ), w[2] & 0xff); // uniform base URZ at bit 64
    try std.testing.expectEqual(@as(u32, 4), (w[2] >> (73 - 64)) & 0x7); // mem type B32
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (78 - 64)) & 0x3); // offset stride X1
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (91 - 64)) & 0x1); // UGPR mode
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (90 - 64)) & 0x1); // no 64-bit address bit
}

test "BAR.SYNC is the bare opcode plus the guard predicate (NAK OpBar)" {
    const w = barSync(.{});
    try std.testing.expectEqual([4]u32{ 0x00007b1d, 0, 0, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, 0xb1d), w[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, PT), (w[0] >> 12) & 0x7); // unconditional
    try std.testing.expectEqual(@as(u32, 0), (w[0] >> 15) & 0x1); // not negated
    // NAK's OpBar writes the opcode and nothing else: no register, no field.
    try std.testing.expectEqual(@as(u32, 0), (w[0] >> 16));
    try std.testing.expectEqual(@as(u32, 0), w[1]);
    try std.testing.expectEqual(@as(u32, 0), w[2]);
}

test "a barrier under a guard predicate keeps the predicate field" {
    // The hardware quirk note: a divergent BRANCH around a BAR.SYNC corrupts a
    // staged shared tile, so a guarded barrier region has to be PREDICATED. The
    // predicate has to survive into the encoding for that to work.
    const w = barSync(.{ .pred = 2, .pred_neg = true });
    try std.testing.expectEqual(@as(u32, 0xb1d), w[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 2), (w[0] >> 12) & 0x7);
    try std.testing.expectEqual(@as(u32, 1), (w[0] >> 15) & 0x1);
}

test "the memory type field carries every access width (NAK set_mem_type)" {
    // NAK sm70_encode.rs, SM70Encoder::set_mem_type: U8 = 0, I8 = 1, U16 = 2,
    // I16 = 3, B32 = 4, B64 = 5, B128 = 6, in the 3-bit field at bits 73..76.
    // A wrong value here silently moves the wrong number of bytes, so pin each.
    const cases = [_]struct { ty: MemType, code: u32, regs: u8, bytes: u8 }{
        .{ .ty = .u8, .code = 0, .regs = 1, .bytes = 1 },
        .{ .ty = .i8, .code = 1, .regs = 1, .bytes = 1 },
        .{ .ty = .u16, .code = 2, .regs = 1, .bytes = 2 },
        .{ .ty = .i16, .code = 3, .regs = 1, .bytes = 2 },
        .{ .ty = .b32, .code = 4, .regs = 1, .bytes = 4 },
        .{ .ty = .b64, .code = 5, .regs = 2, .bytes = 8 },
        .{ .ty = .b128, .code = 6, .regs = 4, .bytes = 16 },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c.code, @as(u32, @intFromEnum(c.ty)));
        try std.testing.expectEqual(c.regs, c.ty.regCount());
        try std.testing.expectEqual(c.bytes, c.ty.byteSize());
        // The same 3-bit field at 73..76 in all four memory encoders.
        try std.testing.expectEqual(c.code, (ldg(6, 4, c.ty, .{})[2] >> (73 - 64)) & 0x7);
        try std.testing.expectEqual(c.code, (stg(4, 6, c.ty, .{})[2] >> (73 - 64)) & 0x7);
        try std.testing.expectEqual(c.code, (lds(6, 4, c.ty, .{})[2] >> (73 - 64)) & 0x7);
        try std.testing.expectEqual(c.code, (sts(4, 6, c.ty, .{})[2] >> (73 - 64)) & 0x7);
    }
}

test "a 64-bit LDG changes ONLY the memory type field" {
    // The width must not disturb the address pair, the memory order, the
    // eviction priority or the UGPR bits, all of which a global access needs.
    const w32 = ldgU32(6, 4, .{});
    const w64 = ldg(6, 4, .b64, .{});
    try std.testing.expectEqual([4]u32{ 0x04067981, 0x000000ff, 0x0c1e0b00, 0x000fde00 }, w64);
    try std.testing.expectEqual(@as(u32, 5), (w64[2] >> (73 - 64)) & 0x7); // B64, not B32
    try std.testing.expectEqual(@as(u32, 4), (w32[2] >> (73 - 64)) & 0x7);
    // Every other bit of every dword is identical.
    const mask: u32 = ~(@as(u32, 0x7) << (73 - 64));
    try std.testing.expectEqual(w32[0], w64[0]);
    try std.testing.expectEqual(w32[1], w64[1]);
    try std.testing.expectEqual(w32[2] & mask, w64[2] & mask);
    try std.testing.expectEqual(w32[3], w64[3]);
    // The address is still a 64-bit register pair at the default weak order.
    try std.testing.expectEqual(@as(u32, 4), (w64[0] >> 24) & 0xff); // address R4:R5
    try std.testing.expectEqual(@as(u32, 1), (w64[2] >> (90 - 64)) & 0x1); // 64-bit GPR address
    try std.testing.expectEqual(@as(u32, 0), (w64[2] >> (77 - 64)) & 0xf); // weak, the default
}

test "LDG keeps its uniform base at bit 32, guards on PT and writes no fault predicate" {
    // The three fields a LOAD places differently from a store. NAK's OpLd
    // global arm: set_ureg_addr(32, ..) for the uniform base, NOT 64 where a
    // store keeps it; set_rev_pred_src(64..67, 67, ..) for the guard, whose
    // index field is REVERSED, so an unconditional access encodes as plain
    // zero; and set_pred_dst(81..84, None) = PT for the fault predicate.
    //
    // Writing the uniform base at bit 64, as this encoder once did, put UR0
    // into the address, `!P0` into the guard and P0 into the fault
    // destination. `nvdisasm -b SM120` rendered that word as
    // `LDG.E.LTC256B.STRONG.SYS P0, R4, [R2.64+UR0], !P0` and renders the
    // word below as `LDG.E R4, [R2.64+URZ]`. P0 is a register the boolean
    // allocator hands out, so both the read and the write collided with it.
    const w = ldgU32(4, 2, .{});
    try std.testing.expectEqual(@as(u32, URZ), w[1] & 0xff); // uniform base URZ at bit 32
    try std.testing.expectEqual(@as(u32, 0), w[2] & 0x7); // guard PT, reversed to 0
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (67 - 64)) & 0x1); // guard not negated
    try std.testing.expectEqual(@as(u32, PT), (w[2] >> (81 - 64)) & 0x7); // no fault predicate
    // A store puts the same uniform base at bit 64 and its data at bit 32.
    const st = stgU32(2, 4, .{});
    try std.testing.expectEqual(@as(u32, 4), st[1] & 0xff); // data R4 at bit 32
    try std.testing.expectEqual(@as(u32, URZ), st[2] & 0xff); // uniform base URZ at bit 64
}

test "every memory order selector matches NAK set_mem_order for sm >= 80" {
    // NAK sm70_encode.rs, set_mem_order, the sm >= 80 arm: Weak 0x0,
    // Constant 0x4, Strong(CTA) 0x5, Strong(GPU) 0x7, Strong(System) 0xa.
    const cases = [_]struct { order: MemOrder, code: u32 }{
        .{ .order = .weak, .code = 0x0 },
        .{ .order = .constant, .code = 0x4 },
        .{ .order = .strong_cta, .code = 0x5 },
        .{ .order = .strong_gpu, .code = 0x7 },
        .{ .order = .strong_sys, .code = 0xa },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c.code, (ldgOrdered(6, 4, 0, .b32, c.order, .{})[2] >> (77 - 64)) & 0xf);
        try std.testing.expectEqual(c.code, (stgOrdered(4, 6, 0, .b32, c.order, .{})[2] >> (77 - 64)) & 0xf);
    }
    // The selector the instruction selector gets is the weak one.
    try std.testing.expectEqual(@as(u32, 0), (ldgU32(6, 4, .{})[2] >> (77 - 64)) & 0xf);
    try std.testing.expectEqual(@as(u32, 0), (stgU32(4, 6, .{})[2] >> (77 - 64)) & 0xf);
}

test "a 128-bit STG and a byte STG keep the global store frame" {
    const w128 = stg(4, 6, .b128, .{});
    try std.testing.expectEqual([4]u32{ 0x04007986, 0x00000006, 0x0c100dff, 0x000fde00 }, w128);
    try std.testing.expectEqual(@as(u32, 0x986), w128[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 6), w128[1] & 0xff); // data block base R6 at bit 32
    try std.testing.expectEqual(@as(u32, 6), (w128[2] >> (73 - 64)) & 0x7); // B128

    const w8 = stg(4, 6, .u8, .{});
    try std.testing.expectEqual(@as(u32, 0), (w8[2] >> (73 - 64)) & 0x7); // U8
    try std.testing.expectEqual(@as(u32, 1), (w8[2] >> (91 - 64)) & 0x1); // UGPR mode still set
}

test "a 64-bit LDS keeps the ONE-register shared address form" {
    const w = lds(6, 4, .b64, .{});
    try std.testing.expectEqual([4]u32{ 0x04067984, 0x000000ff, 0x0b800a00, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, 0x984), w[0] & 0xfff); // LDS
    try std.testing.expectEqual(@as(u32, 5), (w[2] >> (73 - 64)) & 0x7); // B64
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // address R4, no pair
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (90 - 64)) & 0x1); // no 64-bit address bit
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (81 - 64)) & 0x3f); // 81..87 still clear
}

test "a 128-bit STS keeps the ONE-register shared address form" {
    const w = sts(4, 6, .b128, .{});
    try std.testing.expectEqual([4]u32{ 0x04007988, 0x00000006, 0x08000cff, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, 0x988), w[0] & 0xfff); // STS
    try std.testing.expectEqual(@as(u32, 6), (w[2] >> (73 - 64)) & 0x7); // B128
    try std.testing.expectEqual(@as(u32, URZ), w[2] & 0xff); // uniform base URZ at bit 64
}

test "ATOMG places the operation, the type and the sm100 address size bit (NAK OpAtom global)" {
    // NAK sm70_encode.rs, impl SM70Op for OpAtom, MemSpace::Global with a
    // destination, has_ugpr and sm >= 100: opcode 0x9a8, atom op at 87..91,
    // set_reg_addr(24..32, addr, 63), data at 32..40, ureg addr at 64 with its
    // size bit at 72, atom type at 73..77, mem order at 77..81, fault pred at
    // 81..84, eviction at 84..87, bit 91 for UGPR mode.
    const w = atomg(6, 4, 8, .add, .u32, .{});
    try std.testing.expectEqual([4]u32{ 0x040679a8, 0x80000008, 0x081f41ff, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, ATOMG_OPCODE), w[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 6), (w[0] >> 16) & 0xff); // old value into R6
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // address R4:R5
    try std.testing.expectEqual(@as(u32, 8), w[1] & 0xff); // data R8 at bit 32
    try std.testing.expectEqual(@as(u32, 0), (w[1] >> 8) & 0x7fffff); // 23-bit offset 0 at 40..63
    try std.testing.expectEqual(@as(u32, 1), (w[1] >> 31) & 0x1); // address size bit at 63, NOT 90
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (90 - 64)) & 0x1); // bit 90 stays clear
    try std.testing.expectEqual(@as(u32, URZ), w[2] & 0xff); // uniform base URZ at bit 64
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (72 - 64)) & 0x1); // 64-bit uniform base
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (73 - 64)) & 0xf); // atom type U32, 4 bits
    try std.testing.expectEqual(@as(u32, 0xa), (w[2] >> (77 - 64)) & 0xf); // STRONG / SYS
    try std.testing.expectEqual(@as(u32, PT), (w[2] >> (81 - 64)) & 0x7); // no fault predicate
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (84 - 64)) & 0x7); // eviction NORMAL
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (87 - 64)) & 0xf); // atom op ADD
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (91 - 64)) & 0x1); // UGPR mode
}

test "every ATOMG operation and type selector matches NAK" {
    // NAK set_atom_op: Add 0, Min 1, Max 2, Inc 3, Dec 4, And 5, Or 6, Xor 7,
    // Exch 8. NAK set_atom_type (sm >= 90): U32 0, I32 1, U64 2, I64 3. A wrong
    // selector here does the wrong arithmetic to live memory on real silicon.
    const ops = [_]struct { op: AtomOp, code: u32 }{
        .{ .op = .add, .code = 0 },
        .{ .op = .min, .code = 1 },
        .{ .op = .max, .code = 2 },
        .{ .op = .inc, .code = 3 },
        .{ .op = .dec, .code = 4 },
        .{ .op = .bit_and, .code = 5 },
        .{ .op = .bit_or, .code = 6 },
        .{ .op = .bit_xor, .code = 7 },
        .{ .op = .exch, .code = 8 },
    };
    for (ops) |o| {
        try std.testing.expectEqual(o.code, (atomg(6, 4, 8, o.op, .u32, .{})[2] >> (87 - 64)) & 0xf);
        try std.testing.expectEqual(o.code, (atoms(6, 4, 8, o.op, .u32, .{})[2] >> (87 - 64)) & 0xf);
    }
    const types = [_]struct { ty: AtomType, code: u32 }{
        .{ .ty = .u32, .code = 0 },
        .{ .ty = .i32, .code = 1 },
        .{ .ty = .u64, .code = 2 },
        .{ .ty = .i64, .code = 3 },
    };
    for (types) |t| {
        try std.testing.expectEqual(t.code, (atomg(6, 4, 8, .add, t.ty, .{})[2] >> (73 - 64)) & 0xf);
        try std.testing.expectEqual(t.code, (redg(4, 8, .add, t.ty, .{})[2] >> (73 - 64)) & 0xf);
        try std.testing.expectEqual(t.code, (atomgCas(6, 4, 8, 10, t.ty, .{})[2] >> (73 - 64)) & 0xf);
    }
}

test "RED writes RZ as its destination and keeps the address size bit at 90 (NAK OpAtom, dst none)" {
    // NAK's common tail calls set_dst(&Dst::None), which writes zero_reg(GPR) =
    // RZ into bits 16..24. The reduction form also keeps the address size bit at
    // 90, where the destination form moves it to 63.
    const w = redg(4, 8, .add, .u32, .{});
    try std.testing.expectEqual([4]u32{ 0x04ff798e, 0x00000008, 0x0c1141ff, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, RED_OPCODE), w[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, RZ), (w[0] >> 16) & 0xff); // no destination register
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // address R4:R5
    try std.testing.expectEqual(@as(u32, 8), w[1] & 0xff); // data R8 at bit 32
    try std.testing.expectEqual(@as(u32, 0), (w[1] >> 8) & 0xffffff); // 24-bit offset 0 at 40..64
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (90 - 64)) & 0x1); // address size bit at 90
    try std.testing.expectEqual(@as(u32, 0), (w[1] >> 31) & 0x1); // bit 63 stays clear
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (87 - 64)) & 0x7); // atom op ADD, 3 bits
    try std.testing.expectEqual(@as(u32, 0xa), (w[2] >> (77 - 64)) & 0xf); // STRONG / SYS
    // The 3-bit reduction operation field cannot reach the values above 7.
    try std.testing.expectEqual(@as(u32, 7), (redg(4, 8, .bit_xor, .u32, .{})[2] >> (87 - 64)) & 0x7);
}

test "ATOMS reads ONE shared address register and sets no memory order (NAK OpAtom, MemSpace::Shared)" {
    const w = atoms(6, 4, 8, .add, .u32, .{});
    try std.testing.expectEqual([4]u32{ 0x0406798c, 0x00000008, 0x080000ff, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, ATOMS_OPCODE), w[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 6), (w[0] >> 16) & 0xff); // old value into R6
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // shared offset R4, no pair
    try std.testing.expectEqual(@as(u32, 8), w[1] & 0xff); // data R8 at bit 32
    try std.testing.expectEqual(@as(u32, URZ), w[2] & 0xff); // uniform base URZ at bit 64
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (73 - 64)) & 0xf); // atom type U32
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (78 - 64)) & 0x3); // offset stride X1
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (87 - 64)) & 0xf); // atom op ADD
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (91 - 64)) & 0x1); // UGPR mode
    // A shared atomic carries no memory order, no fault predicate and no
    // eviction priority: NAK asserts Strong(CTA)/Normal and writes nothing.
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (77 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (81 - 64)) & 0x3f);
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (90 - 64)) & 0x1); // no 64-bit address bit
}

test "compare-and-swap takes a second data operand at bit 64 (NAK OpAtom CmpExch)" {
    // The global form is opcode 0x3a9 with its address size bit back at 72, and
    // the shared form is 0x38d. Both put the compare operand at 32..40 and the
    // swap data at 64..72, and neither sets the UGPR bit 91, because the swap
    // data occupies the field a uniform base would use.
    const g = atomgCas(6, 4, 8, 10, .u32, .{});
    try std.testing.expectEqual([4]u32{ 0x040673a9, 0x00000008, 0x001f410a, 0x000fde00 }, g);
    try std.testing.expectEqual(@as(u32, ATOMG_CAS_OPCODE), g[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 6), (g[0] >> 16) & 0xff); // old value into R6
    try std.testing.expectEqual(@as(u32, 4), (g[0] >> 24) & 0xff); // address R4:R5
    try std.testing.expectEqual(@as(u32, 8), g[1] & 0xff); // compare operand R8 at bit 32
    try std.testing.expectEqual(@as(u32, 10), g[2] & 0xff); // swap data R10 at bit 64
    try std.testing.expectEqual(@as(u32, 1), (g[2] >> (72 - 64)) & 0x1); // address size bit at 72
    try std.testing.expectEqual(@as(u32, 0), (g[2] >> (73 - 64)) & 0xf); // atom type U32
    try std.testing.expectEqual(@as(u32, 0xa), (g[2] >> (77 - 64)) & 0xf); // STRONG / SYS
    try std.testing.expectEqual(@as(u32, PT), (g[2] >> (81 - 64)) & 0x7); // no fault predicate
    try std.testing.expectEqual(@as(u32, 1), (g[2] >> (84 - 64)) & 0x7); // eviction NORMAL
    try std.testing.expectEqual(@as(u32, 0), (g[2] >> (91 - 64)) & 0x1); // no UGPR bit

    const s = atomsCas(6, 4, 8, 10, .u32, .{});
    try std.testing.expectEqual([4]u32{ 0x0406738d, 0x00000008, 0x0000000a, 0x000fde00 }, s);
    try std.testing.expectEqual(@as(u32, ATOMS_CAS_OPCODE), s[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 6), (s[0] >> 16) & 0xff); // old value into R6
    try std.testing.expectEqual(@as(u32, 4), (s[0] >> 24) & 0xff); // shared offset R4, no pair
    try std.testing.expectEqual(@as(u32, 8), s[1] & 0xff); // compare operand R8 at bit 32
    try std.testing.expectEqual(@as(u32, 10), s[2] & 0xff); // swap data R10 at bit 64
    try std.testing.expectEqual(@as(u32, 0), (s[2] >> (73 - 64)) & 0xf); // atom type U32
    try std.testing.expectEqual(@as(u32, 0), (s[2] >> (78 - 64)) & 0x3); // offset stride X1
    try std.testing.expectEqual(@as(u32, 0), (s[2] >> (91 - 64)) & 0x1); // no UGPR bit
    // A 64-bit compare-and-swap only changes the type field.
    try std.testing.expectEqual(@as(u32, 2), (atomsCas(6, 4, 8, 10, .u64, .{})[2] >> (73 - 64)) & 0xf);
}

test "an atomic under a guard predicate keeps the predicate field" {
    const w = atomg(6, 4, 8, .add, .u32, .{ .pred = 3, .pred_neg = true });
    try std.testing.expectEqual(@as(u32, ATOMG_OPCODE), w[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 3), (w[0] >> 12) & 0x7);
    try std.testing.expectEqual(@as(u32, 1), (w[0] >> 15) & 0x1);
}

test "HMMA places the three fragment operands and the fp32 result flag (NAK OpHmma)" {
    // A = R0.., B = R4.., C = R8.., D = R8.. (the accumulator in place), 16x8x16, fp32 result.
    const w = hmma(8, 0, 4, 8, .m16n8k16, .f32, .{});
    // Every dword, so a stray bit anywhere in the 128-bit word fails this.
    try std.testing.expectEqual([4]u32{ 0x0008723c, 0x00000004, 0x00001808, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, HMMA_OPCODE), w[0] & 0xfff); // 0x23c, not IMMA 0x237
    try std.testing.expectEqual(@as(u32, 8), (w[0] >> 16) & 0xff); // D fragment starts at R8
    try std.testing.expectEqual(@as(u32, 0), (w[0] >> 24) & 0xff); // A fragment starts at R0
    try std.testing.expectEqual(@as(u32, 4), w[1] & 0xff); // B fragment at bit 32
    try std.testing.expectEqual(@as(u32, 8), w[2] & 0xff); // C fragment at bit 64
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (75 - 64)) & 0x1); // tile low bit
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (78 - 64)) & 0x1); // tile high bit
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (76 - 64)) & 0x1); // fp32 result
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (82 - 64)) & 0x3); // source type F16
    // HMMA does NOT set bit 74. IMMA does. The difference is in the NAK source.
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (74 - 64)) & 0x1);
    // The sm>=90 uniform-predicate path writes only zeros, so bits 87..91 stay clear.
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (87 - 64)) & 0xf);
}

test "the HMMA tile selector splits across bit 75 and bit 78 (NAK set_field2)" {
    // NAK writes the selector low bit first into 75..76, then the rest into 78..79. A
    // selector written whole into either field would send the tensor core a different
    // tile shape, read the wrong registers of every lane, and give wrong numbers with no
    // fault. So pin both halves of all three values.
    const cases = [_]struct { size: HmmaSize, lo: u32, hi: u32 }{
        .{ .size = .m16n8k8, .lo = 0, .hi = 0 },
        .{ .size = .m16n8k16, .lo = 1, .hi = 0 },
        .{ .size = .m16n8k4, .lo = 0, .hi = 1 },
    };
    for (cases) |c| {
        const w = hmma(8, 0, 4, 8, c.size, .f32, .{});
        try std.testing.expectEqual(c.lo, (w[2] >> (75 - 64)) & 0x1);
        try std.testing.expectEqual(c.hi, (w[2] >> (78 - 64)) & 0x1);
    }
}

test "an fp16 HMMA result changes ONLY bit 76" {
    const f32_form = hmma(8, 0, 4, 8, .m16n8k8, .f32, .{});
    const f16_form = hmma(8, 0, 4, 8, .m16n8k8, .f16, .{});
    try std.testing.expectEqual(@as(u32, 1), (f32_form[2] >> (76 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, 0), (f16_form[2] >> (76 - 64)) & 0x1);
    // Nothing else moves: the two words differ by exactly the one bit.
    try std.testing.expectEqual(f32_form[0], f16_form[0]);
    try std.testing.expectEqual(f32_form[1], f16_form[1]);
    try std.testing.expectEqual(f32_form[3], f16_form[3]);
    try std.testing.expectEqual(@as(u32, 1) << (76 - 64), f32_form[2] ^ f16_form[2]);
}

test "IMMA places the fragment operands, the SRC1.COL bit and the signedness (NAK OpImma)" {
    const w = imma(8, 0, 4, 8, .m16n8k32, .{ .signed = true }, .{ .signed = true }, false, .{});
    try std.testing.expectEqual([4]u32{ 0x00087237, 0x00000004, 0x00405c08, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, IMMA_OPCODE), w[0] & 0xfff); // 0x237, not HMMA 0x23c
    try std.testing.expectEqual(@as(u32, 8), (w[0] >> 16) & 0xff); // D fragment starts at R8
    try std.testing.expectEqual(@as(u32, 0), (w[0] >> 24) & 0xff); // A fragment starts at R0
    try std.testing.expectEqual(@as(u32, 4), w[1] & 0xff); // B fragment at bit 32
    try std.testing.expectEqual(@as(u32, 8), w[2] & 0xff); // C fragment at bit 64
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (74 - 64)) & 0x1); // SRC1.COL, always set
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (75 - 64)) & 0x1); // tile low bit
    try std.testing.expectEqual(@as(u32, 2), (w[2] >> (85 - 64)) & 0x3); // tile high bits
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (76 - 64)) & 0x1); // A signed
    try std.testing.expectEqual(@as(u32, 1), (w[2] >> (78 - 64)) & 0x1); // B signed
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (82 - 64)) & 0x1); // no saturation
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (83 - 64)) & 0x1); // A is 8-bit
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (84 - 64)) & 0x1); // B is 8-bit
    // The sm>=90 uniform-predicate path writes only zeros, so bits 87..91 stay clear.
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (87 - 64)) & 0xf);
}

test "the IMMA tile selector splits across bit 75 and bits 85..87 (NAK set_field2)" {
    // The same split rule as HMMA, but the high half is TWO bits and sits at 85, not 78.
    // The table has gaps at 1, 3 and 7, so each named value is pinned on both halves.
    const cases = [_]struct { size: ImmaSize, four: bool, lo: u32, hi: u32 }{
        .{ .size = .m8n8k16, .four = false, .lo = 0, .hi = 0 },
        .{ .size = .m8n8k32, .four = true, .lo = 0, .hi = 1 },
        .{ .size = .m16n8k16, .four = false, .lo = 0, .hi = 2 },
        .{ .size = .m16n8k32, .four = false, .lo = 1, .hi = 2 },
        .{ .size = .m16n8k64, .four = true, .lo = 0, .hi = 3 },
    };
    for (cases) |c| {
        const op: ImmaOperand = .{ .signed = false, .four_bit = c.four };
        const w = imma(8, 0, 4, 8, c.size, op, op, false, .{});
        try std.testing.expectEqual(c.lo, (w[2] >> (75 - 64)) & 0x1);
        try std.testing.expectEqual(c.hi, (w[2] >> (85 - 64)) & 0x3);
    }
}

test "each IMMA operand flag lands in its own bit (NAK OpImma bits 76, 78, 82, 83, 84)" {
    // A and B carry SEPARATE signedness and width bits, which is what mixed-signedness
    // int8 needs. A swap between the A bit and the B bit computes a different product and
    // never faults, so each is set alone and read back alone.
    const plain: ImmaOperand = .{ .signed = false };
    const signed: ImmaOperand = .{ .signed = true };
    const a_only = imma(8, 0, 4, 8, .m16n8k16, signed, plain, false, .{});
    try std.testing.expectEqual(@as(u32, 1), (a_only[2] >> (76 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, 0), (a_only[2] >> (78 - 64)) & 0x1);
    const b_only = imma(8, 0, 4, 8, .m16n8k16, plain, signed, false, .{});
    try std.testing.expectEqual(@as(u32, 0), (b_only[2] >> (76 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, 1), (b_only[2] >> (78 - 64)) & 0x1);
    // Saturation is bit 82 and nothing else moves with it.
    const sat = imma(8, 0, 4, 8, .m16n8k16, plain, plain, true, .{});
    const unsat = imma(8, 0, 4, 8, .m16n8k16, plain, plain, false, .{});
    try std.testing.expectEqual(@as(u32, 1) << (82 - 64), sat[2] ^ unsat[2]);
    // The 4-bit flags are bits 83 and 84. m16n8k32 is the one tile that takes either width.
    const a4 = imma(8, 0, 4, 8, .m16n8k32, .{ .signed = false, .four_bit = true }, plain, false, .{});
    try std.testing.expectEqual(@as(u32, 1), (a4[2] >> (83 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, 0), (a4[2] >> (84 - 64)) & 0x1);
    const b4 = imma(8, 0, 4, 8, .m16n8k32, plain, .{ .signed = false, .four_bit = true }, false, .{});
    try std.testing.expectEqual(@as(u32, 0), (b4[2] >> (83 - 64)) & 0x1);
    try std.testing.expectEqual(@as(u32, 1), (b4[2] >> (84 - 64)) & 0x1);
}

test "LDSM reads ONE shared address register and names the fragment count (NAK OpLdsm)" {
    const w = ldsm(8, 4, .x4, false, .{});
    try std.testing.expectEqual([4]u32{ 0x0408783b, 0x000000ff, 0x00000200, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, LDSM_OPCODE), w[0] & 0xfff); // 0x83b, not LDS 0x984
    try std.testing.expectEqual(@as(u32, 8), (w[0] >> 16) & 0xff); // first destination R8
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // address R4, ONE register
    try std.testing.expectEqual(@as(u32, URZ), w[1] & 0xff); // uniform base URZ at bit 32
    try std.testing.expectEqual(@as(u32, 0), (w[1] >> 8) & 0xffffff); // immediate offset 0 at 40..64
    try std.testing.expectEqual(@as(u32, 2), (w[2] >> (72 - 64)) & 0x3); // 4 fragments
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (78 - 64)) & 0x3); // M8N8, not transposed
    // LDSM writes bit 91 as "there is a uniform address", which is FALSE. LDS and STS
    // write that same bit as TRUE. The two rules differ in the NAK source.
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (91 - 64)) & 0x1);
}

test "the LDSM fragment count and the transpose each carry their own field" {
    // A count of 1, 2 and 4 encodes as 0, 1 and 2. A wrong code loads a different number
    // of registers per lane and leaves the rest of the fragment holding old values.
    const cases = [_]struct { count: LdsmCount, code: u32, matrices: u32 }{
        .{ .count = .x1, .code = 0, .matrices = 1 },
        .{ .count = .x2, .code = 1, .matrices = 2 },
        .{ .count = .x4, .code = 2, .matrices = 4 },
    };
    for (cases) |c| {
        const w = ldsm(8, 4, c.count, false, .{});
        try std.testing.expectEqual(c.code, (w[2] >> (72 - 64)) & 0x3);
        try std.testing.expectEqual(c.matrices, c.count.matrices());
    }
    // The transpose is bits 78..80 = 1 and changes nothing else.
    const plain = ldsm(8, 4, .x2, false, .{});
    const trans = ldsm(8, 4, .x2, true, .{});
    try std.testing.expectEqual(@as(u32, 1), (trans[2] >> (78 - 64)) & 0x3);
    try std.testing.expectEqual(@as(u32, 1) << (78 - 64), plain[2] ^ trans[2]);
}

test "MOVM is the opcode, one source and the MT88 mode (NAK OpMovm)" {
    const w = movm(8, 4, .{});
    try std.testing.expectEqual([4]u32{ 0x0408723a, 0x00000000, 0x00000000, 0x000fde00 }, w);
    try std.testing.expectEqual(@as(u32, MOVM_OPCODE), w[0] & 0xfff); // 0x23a, one below HMMA
    try std.testing.expectEqual(@as(u32, 8), (w[0] >> 16) & 0xff); // dst R8
    try std.testing.expectEqual(@as(u32, 4), (w[0] >> 24) & 0xff); // src R4
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (78 - 64)) & 0x3); // MT88, the only named mode
    // NAK's OpMovm writes nothing else. Bits 32..71 stay clear.
    try std.testing.expectEqual(@as(u32, 0), w[1]);
    try std.testing.expectEqual(@as(u32, 0), w[2] & 0xff);
}

test "a tensor op under a guard predicate keeps the predicate field" {
    // A tensor op is warp-collective, so a lowering cannot branch around it for part of
    // the warp. A guarded region has to be PREDICATED instead, the same rule the hardware
    // quirk note gives for BAR.SYNC. The predicate has to survive into the encoding.
    const h = hmma(8, 0, 4, 8, .m16n8k16, .f32, .{ .pred = 2, .pred_neg = true });
    try std.testing.expectEqual(@as(u32, HMMA_OPCODE), h[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 2), (h[0] >> 12) & 0x7);
    try std.testing.expectEqual(@as(u32, 1), (h[0] >> 15) & 0x1);
    const l = ldsm(8, 4, .x1, false, .{ .pred = 5, .pred_neg = false });
    try std.testing.expectEqual(@as(u32, LDSM_OPCODE), l[0] & 0xfff);
    try std.testing.expectEqual(@as(u32, 5), (l[0] >> 12) & 0x7);
    try std.testing.expectEqual(@as(u32, 0), (l[0] >> 15) & 0x1);
}

test "LEA encodes (a << shift) + b in one instruction" {
    // `nvdisasm -b SM120 -c` reads this word back as `LEA R9, R7, R6, 0x2`, which is
    // `(R7 << 2) + R6`. The shift is a 5-bit field, and 0x1f decodes too.
    //
    // NOTHING IN THE INSTRUCTION SELECTOR EMITS LEA YET. It is here, and checked against
    // NAK field by field plus the disassembler, so the selection side is the only work
    // left. `atomg` and the tensor ops were added the same way.
    const w = lea(9, 7, 6, 2, .{});
    try std.testing.expectEqual(LEA_OPCODE, w[0] & 0xfff); // base 0x011 | register form
    try std.testing.expectEqual(@as(u32, 9), (w[0] >> 16) & 0xff); // dst R9
    try std.testing.expectEqual(@as(u32, 7), (w[0] >> 24) & 0xff); // srcA R7, the shifted value
    try std.testing.expectEqual(@as(u32, 6), w[1] & 0xff); // srcB R6, the addend, at bit 32
    try std.testing.expectEqual(@as(u32, 2), (w[2] >> (75 - 64)) & 0x1f); // shift count at 75..80
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (80 - 64)) & 0x1); // the LOW half, not LEA.HI
    try std.testing.expectEqual(@as(u32, 0), (w[2] >> (74 - 64)) & 0x1); // not the .X carry-in form
    // The overflow predicate destination must be PT. A zero there names P0, which the
    // boolean allocator hands out. See the zero-field note at the top of this file.
    try std.testing.expectEqual(@as(u32, PT), (w[2] >> (81 - 64)) & 0x7);
    // The widest shift the 5-bit field holds.
    try std.testing.expectEqual(@as(u32, 31), (lea(9, 7, 6, 31, .{})[2] >> (75 - 64)) & 0x1f);
}
