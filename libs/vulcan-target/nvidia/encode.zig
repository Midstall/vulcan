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

/// `FADD dst, a, b`: 32-bit float add. NAK base 0x021.
pub fn fadd(dst: u8, a: u8, b: u8, c: Control) Inst {
    return alu(0x021, dst, a, b, RZ, c);
}

/// `FADD dst, a, -b`: float subtract (dst = a - b), via the srcB negate modifier.
pub fn fsub(dst: u8, a: u8, b: u8, c: Control) Inst {
    var w = fadd(dst, a, b, c);
    setBits(&w, 63, 1, 1); // negate srcB
    return w;
}

/// `FMUL dst, a, b`: 32-bit float multiply. NAK base 0x020.
pub fn fmul(dst: u8, a: u8, b: u8, c: Control) Inst {
    return alu(0x020, dst, a, b, RZ, c);
}

/// `FFMA dst, a, b, c_in`: fused multiply-add (dst = a*b + c_in). NAK base 0x023.
pub fn ffma(dst: u8, a: u8, b: u8, c_in: u8, c: Control) Inst {
    return alu(0x023, dst, a, b, c_in, c);
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
    var w = alu(0x105, dst, RZ, src, RZ, c);
    if (dst_signed) setBits(&w, 72, 1, 1);
    setBits(&w, 75, 2, 2); // dst = 4 bytes (i32)
    setBits(&w, 78, 2, 3); // round toward zero
    setBits(&w, 84, 2, 2); // src = 4 bytes (f32)
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

/// `BRA target`: relative branch (guard with `Control.pred` for a conditional
/// branch). `delta` is the byte offset from the next instruction. Volta BRA 0x947.
pub fn bra(delta: i32, c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x947);
    setBits(&w, 34, 32, @as(u32, @bitCast(delta)));
    return w;
}

/// `EXIT`: terminate the warp. Verified (prism).
pub fn exit(c: Control) Inst {
    var w = base(c);
    setBits(&w, 0, 12, 0x94d);
    setBits(&w, 87, 3, 7);
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

/// Shader attribute addresses: the clip-space position output, and the first
/// generic varying / vertex input. System-value index for the vertex id.
pub const ATTR_POSITION: u16 = 0x70;
pub const ATTR_GENERIC0: u16 = 0x80;
pub const SR_VERTEX_ID: u8 = 0x2f;

/// Special-register indices for `s2r`.
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
