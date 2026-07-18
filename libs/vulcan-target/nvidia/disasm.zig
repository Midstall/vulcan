//! NVIDIA SASS disassembler for sm_120 (Blackwell). This is the strict INVERSE of
//! `encode.zig`: it decodes exactly the instructions the encoder produces, dumps a
//! readable per-instruction listing with destination/source register numbers, the
//! guard predicate, and the scheduling control bits (stall, write/read barrier, wait
//! mask). It is the oracle for the register-allocation / liveness hunt - so a human
//! (or a diff) can SEE which physical register each value landed in and which
//! scoreboards gate it.
//!
//! It does NOT aim to cover the full SASS ISA - only the subset `encode.zig` emits.
//! An unknown 12-bit opcode is printed as `??.<hex>` with the raw dwords so nothing
//! is silently mis-decoded. Field offsets mirror encode.zig bit-for-bit (the same
//! `setBits(lo,width)` positions). The decode reads them back with `getBits`.
//!
//! Usage: `disasm.format(allocator, code)` returns an owned string. `disasm.decode`
//! returns a slice of structured `Inst` for programmatic diffing (the two-of-three
//! register-collision diff the bring-up needs).

const std = @import("std");
const encode = @import("encode.zig");

const RZ = encode.RZ;
const PT = encode.PT;

/// Read `width` bits at bit offset `lo` from a 128-bit instruction (four LE dwords).
/// Mirrors encode.setBits exactly, handling fields that span a 32-bit word boundary.
fn getBits(w: encode.Inst, lo: usize, width: usize) u64 {
    var v: u64 = 0;
    var i: usize = 0;
    while (i < width) : (i += 1) {
        const bit = lo + i;
        const off: u5 = @intCast(bit % 32);
        const b: u64 = (w[bit / 32] >> off) & 1;
        v |= b << @intCast(i);
    }
    return v;
}

/// The scheduling control + guard a decoded instruction carries (the inverse of
/// encode.base): the stall, the scoreboard a variable-latency op sets, the read
/// barrier, and the wait mask its consumers carry.
pub const Sched = struct {
    pred: u8,
    pred_neg: bool,
    stall: u4,
    wr_barrier: u3,
    rd_barrier: u3,
    wait_mask: u6,
};

fn decodeSched(w: encode.Inst) Sched {
    return .{
        .pred = @intCast(getBits(w, 12, 3)),
        .pred_neg = getBits(w, 15, 1) != 0,
        .stall = @intCast(getBits(w, 105, 4)),
        .wr_barrier = @intCast(getBits(w, 110, 3)),
        .rd_barrier = @intCast(getBits(w, 113, 3)),
        .wait_mask = @intCast(getBits(w, 116, 6)),
    };
}

/// A decoded instruction: the mnemonic, its destination register (if it writes a
/// GPR or a predicate), the source registers it reads, and the schedule/guard. The
/// caller diffs `dst` / `srcs` / `sched.wait_mask` across compiled variants.
pub const Inst = struct {
    /// The 12-bit opcode (low bits), for opcode-level diffing.
    opcode: u12,
    mnemonic: []const u8,
    /// Destination GPR (255 = none/RZ/not a GPR-writer).
    dst: u8 = RZ,
    /// Destination predicate register (7 = none).
    dst_pred: u8 = PT,
    /// Source GPRs this instruction reads (RZ entries are placeholders to keep the
    /// positions stable for diffing). Up to four (srcA@24, srcB@32, srcC@64, + a
    /// derived address-pair high half).
    srcs: [4]u8 = .{ RZ, RZ, RZ, RZ },
    /// An immediate / offset operand the instruction carries (MOV.imm value, LDC
    /// CB offset, ALD/AST attribute byte address). `has_imm` gates printing.
    imm: u32 = 0,
    has_imm: bool = false,
    /// Constant bank index (LDC only).
    bank: u5 = 0,
    sched: Sched,
    /// The four raw dwords (for an unknown opcode or a byte-exact cross-check).
    raw: encode.Inst,
};

/// Decode one 128-bit instruction. Recognises exactly the opcodes encode.zig emits.
pub fn decodeOne(w: encode.Inst) Inst {
    const op: u12 = @intCast(getBits(w, 0, 12));
    const sched = decodeSched(w);
    var d = Inst{ .opcode = op, .mnemonic = "??", .sched = sched, .raw = w };

    const dst: u8 = @intCast(getBits(w, 16, 8));
    const srcA: u8 = @intCast(getBits(w, 24, 8));
    const srcB: u8 = @intCast(getBits(w, 32, 8));
    const srcC: u8 = @intCast(getBits(w, 64, 8));
    const form = getBits(w, 9, 3);

    // ALU-family ops carry the source FORM in bits 9..11, so their full low-12-bit
    // value is `base | form << 9` (e.g. MOV.imm = 0x002 | 4<<9 = 0x802, IADD3.reg =
    // 0x010 | 1<<9 = 0x210, MUFU = 0x108 | 1<<9 = 0x308). The non-ALU ops below set
    // all 12 bits directly. Dispatch on the 9-bit base for the ALU family, and on the
    // full value for the special ops (whose 9-bit bases do not collide with any ALU
    // base the encoder emits).
    const base9: u9 = @intCast(op & 0x1ff);
    switch (op) {
        0xb82 => { // LDC
            d.mnemonic = "LDC";
            d.dst = dst;
            // dynamic offset reg at 24 (RZ = static), not a real data dependency.
            d.imm = @intCast(getBits(w, 38, 16)); // 16-bit static CB offset
            d.bank = @intCast(getBits(w, 54, 5));
            d.has_imm = true;
        },
        0x981 => { // LDG.E (64-bit addr pair at bit 24)
            d.mnemonic = "LDG";
            d.dst = dst;
            d.srcs = .{ srcA, RZ, RZ, srcA +% 1 }; // addr lo + addr hi
        },
        0x986 => { // STG.E
            d.mnemonic = "STG";
            d.srcs = .{ srcA, srcB, RZ, srcA +% 1 }; // addr lo, data, addr hi
        },
        0x919 => { // S2R
            d.mnemonic = "S2R";
            d.dst = dst;
        },
        0x947 => { // BRA
            d.mnemonic = "BRA";
        },
        0x94d => { // EXIT
            d.mnemonic = "EXIT";
        },
        0x355 => { // BCLEAR (convergence barrier init)
            d.mnemonic = "BCLEAR";
            d.dst = @intCast(getBits(w, 24, 4)); // barrier reg @24..28
        },
        0x945 => { // BSSY (set convergence barrier at a reconvergence target)
            d.mnemonic = "BSSY";
            d.dst = @intCast(getBits(w, 16, 4)); // barrier-dst reg @16..20
        },
        0x941 => { // BSYNC (reconverge at a barrier)
            d.mnemonic = "BSYNC";
            d.srcs = .{ @intCast(getBits(w, 16, 4)), RZ, RZ, RZ }; // barrier-src reg @16..20
        },
        0x321 => { // ALD (vertex attribute load)
            d.mnemonic = "ALD";
            d.dst = dst;
            d.imm = @intCast(getBits(w, 40, 10)); // attribute byte address
            d.has_imm = true;
        },
        0x322 => { // AST (output attribute store)
            d.mnemonic = "AST";
            d.srcs = .{ srcB, RZ, RZ, RZ }; // data at bit 32
            d.imm = @intCast(getBits(w, 40, 10)); // output attribute byte address
            d.has_imm = true;
        },
        0x326 => { // IPA (fragment varying interpolate)
            d.mnemonic = "IPA";
            d.dst = dst;
            d.imm = @intCast(@as(u32, @intCast(getBits(w, 64, 8))) << 2); // attr byte addr
            d.has_imm = true;
        },
        0xf89 => { // SHFL.BFLY (all-immediate quad form)
            d.mnemonic = "SHFL.BFLY";
            d.dst = dst;
            d.srcs = .{ srcA, RZ, RZ, RZ }; // src at bit 24
        },
        0x822 => { // FSWZADD
            d.mnemonic = "FSWZADD";
            d.dst = dst;
            d.srcs = .{ srcA, srcC, RZ, RZ }; // src0@24 (shuffled), src1@64 (self)
        },
        encode.TEX_OPCODE => { // TEX (bindless 2D)
            d.mnemonic = "TEX";
            d.dst = dst; // dst[0] = R,G and dst+2 (bit 64) = B,A
            d.srcs = .{ srcA, srcB, RZ, RZ }; // coord pair @24, handle @32
        },
        else => switch (base9) {
            // ALU MOV: form 4 = immediate, form 1 = register copy.
            0x002 => {
                d.dst = dst;
                if (form == 4) {
                    d.mnemonic = "MOV.imm";
                    d.imm = @intCast(getBits(w, 32, 32)); // 32-bit immediate value
                    d.has_imm = true;
                } else {
                    d.mnemonic = "MOV";
                    d.srcs[0] = srcB; // movReg sources at bit 32
                }
            },
            0x010 => { // IADD3 (and ISUB / carry variants)
                d.mnemonic = if (getBits(w, 63, 1) != 0) "IADD3.sub" else "IADD3";
                d.dst = dst;
                d.srcs = .{ srcA, srcB, RZ, RZ };
            },
            0x012 => { // LOP3.LUT
                d.mnemonic = "LOP3";
                d.dst = dst;
                d.srcs = .{ srcA, srcB, RZ, RZ };
            },
            0x00c => { // ISETP -> predicate
                d.mnemonic = "ISETP";
                d.dst_pred = @intCast(getBits(w, 81, 3));
                d.srcs = .{ srcA, srcB, RZ, RZ };
            },
            0x00b => { // FSETP (float set-predicate) -> predicate
                d.mnemonic = "FSETP";
                d.dst_pred = @intCast(getBits(w, 81, 3));
                d.srcs = .{ srcA, srcB, RZ, RZ };
            },
            0x007 => { // SEL
                d.mnemonic = "SEL";
                d.dst = dst;
                d.srcs = .{ srcA, srcB, RZ, RZ };
            },
            0x019 => { // SHF
                d.mnemonic = if (getBits(w, 76, 1) != 0) "SHF.R" else "SHF.L";
                d.dst = dst;
                d.srcs = .{ srcA, srcB, srcC, RZ };
            },
            0x021 => {
                d.mnemonic = if (getBits(w, 63, 1) != 0) "FADD.sub" else "FADD";
                d.dst = dst;
                d.srcs = .{ srcA, srcB, RZ, RZ };
            },
            0x020 => {
                d.mnemonic = "FMUL";
                d.dst = dst;
                d.srcs = .{ srcA, srcB, RZ, RZ };
            },
            0x023 => {
                d.mnemonic = "FFMA";
                d.dst = dst;
                d.srcs = .{ srcA, srcB, srcC, RZ };
            },
            0x024 => {
                d.mnemonic = "IMAD";
                d.dst = dst;
                d.srcs = .{ srcA, srcB, srcC, RZ };
            },
            0x106 => {
                d.mnemonic = "I2F";
                d.dst = dst;
                d.srcs = .{ srcB, RZ, RZ, RZ }; // i2f sources at bit 32
            },
            0x105 => {
                d.mnemonic = "F2I";
                d.dst = dst;
                d.srcs = .{ srcB, RZ, RZ, RZ };
            },
            0x108 => { // MUFU (encode emits the register form -> full op 0x308)
                d.mnemonic = "MUFU";
                d.dst = dst;
                d.srcs = .{ srcB, RZ, RZ, RZ };
            },
            else => {
                d.mnemonic = "??";
            },
        },
    }
    return d;
}

/// Decode an entire compiled code stream (`[]u32`, four dwords per instruction).
/// The caller owns the returned slice.
pub fn decode(allocator: std.mem.Allocator, code: []const u32) ![]Inst {
    std.debug.assert(code.len % 4 == 0);
    const n = code.len / 4;
    const out = try allocator.alloc(Inst, n);
    for (0..n) |i| {
        const w: encode.Inst = .{ code[i * 4], code[i * 4 + 1], code[i * 4 + 2], code[i * 4 + 3] };
        out[i] = decodeOne(w);
    }
    return out;
}

fn regName(buf: []u8, r: u8) []const u8 {
    if (r == RZ) return "RZ";
    // r is a u8, so the longest output is "R255" (4 bytes); callers pass a buffer
    // that fits. A failure here is a buffer-sizing bug in our code, not input.
    return std.fmt.bufPrint(buf, "R{d}", .{r}) catch unreachable;
}

/// Format a compiled code stream as a readable per-instruction listing. The caller
/// owns the returned string. Each line:
///   `NNN: [@PRED] MNEMONIC dst <- srcs   {stall, wb=N, wait=0xMM}`
pub fn format(allocator: std.mem.Allocator, code: []const u32) ![]u8 {
    const insts = try decode(allocator, code);
    defer allocator.free(insts);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var ba: [8]u8 = undefined;
    var bb: [8]u8 = undefined;

    for (insts, 0..) |it, i| {
        try out.print(allocator, "{d:>4}: ", .{i});
        // Guard predicate.
        if (it.sched.pred != PT or it.sched.pred_neg) {
            try out.print(allocator, "@{s}P{d} ", .{ if (it.sched.pred_neg) "!" else "", it.sched.pred });
        }
        try out.print(allocator, "{s}", .{it.mnemonic});
        var d_print_skip_dst = false;
        if (it.opcode == 0x947) { // BRA: print the taken condition + the word-unit delta
            // The condition lives at bits 87..89 (+ negate 90), the offset is SPLIT
            // (low 8 @16..24, high 48 @34..82), a 56-bit signed value.
            const cond: u8 = @intCast(getBits(it.raw, 87, 3));
            const cneg = getBits(it.raw, 90, 1) != 0;
            const lo: u64 = getBits(it.raw, 16, 8);
            const hi: u64 = getBits(it.raw, 34, 48);
            const raw56: u64 = lo | (hi << 8);
            const delta: i64 = (@as(i64, @bitCast(raw56 << 8))) >> 8; // sign-extend 56-bit
            if (cond != PT) try out.print(allocator, " @{s}P{d}", .{ if (cneg) "!" else "", cond });
            try out.print(allocator, " {d}", .{delta});
        }
        if (it.opcode == 0x945) { // BSSY: print B<n> + the byte delta to the reconvergence target
            const raw30: u32 = @intCast(getBits(it.raw, 34, 30));
            const delta: i32 = (@as(i32, @bitCast(raw30 << 2))) >> 2; // sign-extend the 30-bit field
            try out.print(allocator, " B{d}, {d}", .{ it.dst, delta });
            d_print_skip_dst = true;
        }
        if (it.opcode == 0x355) { // BCLEAR: print B<n>
            try out.print(allocator, " B{d}", .{it.dst});
            d_print_skip_dst = true;
        }
        if (it.opcode == 0x941) { // BSYNC: print B<n>
            try out.print(allocator, " B{d}", .{it.srcs[0]});
            d_print_skip_dst = true;
        }
        // Destination.
        if (d_print_skip_dst) {
            // barrier op: dst/srcs already printed as B<n>
        } else if (it.dst_pred != PT) {
            try out.print(allocator, " P{d}", .{it.dst_pred});
        } else if (it.dst != RZ) {
            try out.print(allocator, " {s}", .{regName(&ba, it.dst)});
            // TEX writes a split RGBA block: dst..dst+1 here, dst[1] = bit64.
            if (it.opcode == encode.TEX_OPCODE) {
                const hi: u8 = @intCast(getBits(it.raw, 64, 8));
                try out.print(allocator, "..{s}|{s}..", .{ regName(&ba, it.dst + 1), regName(&bb, hi) });
            }
        }
        // Sources.
        var first = true;
        if (!d_print_skip_dst) {
            for (it.srcs) |s| {
                if (s == RZ) continue;
                try out.print(allocator, "{s}{s}", .{ if (first) " <- " else ", ", regName(&ba, s) });
                first = false;
            }
        }
        // Immediate / memory offset operand (MOV.imm value, LDC c[bank][off],
        // ALD/AST attribute byte address).
        if (it.has_imm) {
            if (it.opcode == 0xb82) {
                try out.print(allocator, ", c[{d}][0x{x}]", .{ it.bank, it.imm });
            } else if (it.opcode == 0x321 or it.opcode == 0x322) {
                try out.print(allocator, " @0x{x}", .{it.imm});
            } else {
                try out.print(allocator, ", 0x{x}", .{it.imm});
            }
        }
        // Scheduling control.
        try out.print(allocator, "   {{stall={d}", .{it.sched.stall});
        if (it.sched.wr_barrier != 7) try out.print(allocator, ", wb={d}", .{it.sched.wr_barrier});
        if (it.sched.rd_barrier != 7) try out.print(allocator, ", rb={d}", .{it.sched.rd_barrier});
        if (it.sched.wait_mask != 0) try out.print(allocator, ", wait=0x{x}", .{it.sched.wait_mask});
        try out.print(allocator, "}}", .{});
        if (it.opcode != 0x947 and it.opcode != 0x94d and it.mnemonic[0] == '?') {
            try out.print(allocator, "  [raw {x:0>8} {x:0>8} {x:0>8} {x:0>8}]", .{ it.raw[0], it.raw[1], it.raw[2], it.raw[3] });
        }
        try out.print(allocator, "\n", .{});
    }
    return out.toOwnedSlice(allocator);
}

// Tests: the disassembler is the strict inverse of the encoder.

test "decodes the hardware-verified MOV/STG/EXIT round-trip" {
    const mov = decodeOne(encode.movImm(2, 0xcafe, .{}));
    try std.testing.expectEqual(@as(u12, 0x802), mov.opcode); // MOV imm = base 0x002 | form 4<<9
    try std.testing.expectEqualStrings("MOV.imm", mov.mnemonic);
    try std.testing.expectEqual(@as(u8, 2), mov.dst);

    const movr = decodeOne(encode.movReg(5, 7, .{}));
    try std.testing.expectEqualStrings("MOV", movr.mnemonic);
    try std.testing.expectEqual(@as(u8, 5), movr.dst);
    try std.testing.expectEqual(@as(u8, 7), movr.srcs[0]);

    const stg = decodeOne(encode.stgU32(0, 2, .{}));
    try std.testing.expectEqualStrings("STG", stg.mnemonic);
    try std.testing.expectEqual(@as(u8, 0), stg.srcs[0]); // addr lo
    try std.testing.expectEqual(@as(u8, 2), stg.srcs[1]); // data
    try std.testing.expectEqual(@as(u8, 1), stg.srcs[3]); // addr hi (lo+1)

    const ex = decodeOne(encode.exit(.{ .stall = 1 }));
    try std.testing.expectEqualStrings("EXIT", ex.mnemonic);
    try std.testing.expectEqual(@as(u4, 1), ex.sched.stall);
}

test "decodes the scheduling control (barriers + wait mask + guard)" {
    const w = encode.iadd3(3, 1, 2, .{ .pred = 0, .pred_neg = true, .stall = 2, .wait_mask = 0b10, .wr_barrier = 4 });
    const d = decodeOne(w);
    try std.testing.expectEqualStrings("IADD3", d.mnemonic);
    try std.testing.expectEqual(@as(u8, 3), d.dst);
    try std.testing.expectEqual(@as(u8, 1), d.srcs[0]);
    try std.testing.expectEqual(@as(u8, 2), d.srcs[1]);
    try std.testing.expectEqual(@as(u8, 0), d.sched.pred);
    try std.testing.expect(d.sched.pred_neg);
    try std.testing.expectEqual(@as(u4, 2), d.sched.stall);
    try std.testing.expectEqual(@as(u3, 4), d.sched.wr_barrier);
    try std.testing.expectEqual(@as(u6, 0b10), d.sched.wait_mask);
}

test "decodes the derivative + texture ops the wall hunt needs" {
    const shfl = decodeOne(encode.shflBflyQuad(5, 4, 1, .{}));
    try std.testing.expectEqualStrings("SHFL.BFLY", shfl.mnemonic);
    try std.testing.expectEqual(@as(u8, 5), shfl.dst);
    try std.testing.expectEqual(@as(u8, 4), shfl.srcs[0]);

    const fswz = decodeOne(encode.fswzadd(6, 5, 4, .{ .sub_left, .sub_right, .sub_left, .sub_right }, .{}));
    try std.testing.expectEqualStrings("FSWZADD", fswz.mnemonic);
    try std.testing.expectEqual(@as(u8, 6), fswz.dst);
    try std.testing.expectEqual(@as(u8, 5), fswz.srcs[0]); // shuffled
    try std.testing.expectEqual(@as(u8, 4), fswz.srcs[1]); // self

    const tex = decodeOne(encode.tex2d(20, 24, 26, .{}));
    try std.testing.expectEqualStrings("TEX", tex.mnemonic);
    try std.testing.expectEqual(@as(u8, 20), tex.dst); // R,G
    try std.testing.expectEqual(@as(u8, 24), tex.srcs[0]); // coord pair
    try std.testing.expectEqual(@as(u8, 26), tex.srcs[1]); // handle

    const ipa = decodeOne(encode.ipa(7, encode.ATTR_GENERIC0, .{}));
    try std.testing.expectEqualStrings("IPA", ipa.mnemonic);
    try std.testing.expectEqual(@as(u8, 7), ipa.dst);

    const mufu = decodeOne(encode.mufu(8, 9, .rcp, .{}));
    try std.testing.expectEqualStrings("MUFU", mufu.mnemonic);
    try std.testing.expectEqual(@as(u8, 8), mufu.dst);
    try std.testing.expectEqual(@as(u8, 9), mufu.srcs[0]);
}

test "format produces a readable listing with registers and barriers" {
    const insts = [_]encode.Inst{
        encode.s2r(4, encode.SR_TID_X, .{ .wr_barrier = 0 }),
        encode.imad(5, 4, 6, RZ, .{ .wait_mask = 1 }),
        encode.exit(.{ .stall = 1 }),
    };
    // Flatten to the [code]u32 stream the disassembler consumes (4 dwords/inst).
    var code: [insts.len * 4]u32 = undefined;
    for (insts, 0..) |w, i| {
        code[i * 4 + 0] = w[0];
        code[i * 4 + 1] = w[1];
        code[i * 4 + 2] = w[2];
        code[i * 4 + 3] = w[3];
    }
    const txt = try format(std.testing.allocator, &code);
    defer std.testing.allocator.free(txt);
    // The listing must name the S2R's destination register and its write barrier,
    // and the IMAD's wait mask - the exact fields the register/scoreboard diff reads.
    try std.testing.expect(std.mem.indexOf(u8, txt, "S2R R4") != null);
    try std.testing.expect(std.mem.indexOf(u8, txt, "wb=0") != null);
    try std.testing.expect(std.mem.indexOf(u8, txt, "IMAD R5") != null);
    try std.testing.expect(std.mem.indexOf(u8, txt, "wait=0x1") != null);
    try std.testing.expect(std.mem.indexOf(u8, txt, "EXIT") != null);
}
