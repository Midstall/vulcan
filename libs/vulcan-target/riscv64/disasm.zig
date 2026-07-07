//! RISC-V (RV64GCV subset) disassembler: the inverse of `encode.zig`, decoding exactly the
//! instructions the encoder emits. RISC-V has a regular fixed 32-bit format, so decode keys
//! on the opcode (bits[6:0]) plus funct3/funct7. Anything outside the emitted subset prints
//! as `.word 0x<hex>`.
//!
//! `one` renders a single instruction word; `format` renders a code buffer with addresses.
//! Validated by round-tripping every encoder function.

const std = @import("std");
const encode = @import("encode.zig");

pub fn one(allocator: std.mem.Allocator, word: u32) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try decode(allocator, &buf, word, null);
    return buf.toOwnedSlice(allocator);
}

/// Render a single 16-bit compressed (RVC) instruction. Caller owns the result.
pub fn oneCompressed(allocator: std.mem.Allocator, half: u16) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    _ = try decode16(allocator, &buf, half, null);
    return buf.toOwnedSlice(allocator);
}

pub fn format(allocator: std.mem.Allocator, code: []const u32) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    for (code, 0..) |word, i| {
        try buf.print(allocator, "{x:0>4}: {x:0>8}  ", .{ i * 4, word });
        try decode(allocator, &buf, word, null);
        try buf.append(allocator, '\n');
    }
    return buf.toOwnedSlice(allocator);
}

/// Render a byte stream that may mix 16-bit compressed (RVC) and 32-bit instructions, stepping
/// by the length each encoding declares (bits[1:0] != 11 -> 16-bit). `base` is the address of the
/// first byte. Caller owns the result. Use this for real RV64GC images; `format` is for a pure
/// 32-bit `[]u32` (Vulcan's own output).
pub fn formatBytes(allocator: std.mem.Allocator, code: []const u8, base: u64) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var pos: usize = 0;
    while (pos + 2 <= code.len) {
        const h = std.mem.readInt(u16, code[pos..][0..2], .little);
        if (h & 3 != 3) { // 16-bit compressed
            try buf.print(allocator, "{x:0>8}: {x:0>4}      ", .{ base + pos, h });
            if (!try decode16(allocator, &buf, h, null)) try buf.print(allocator, ".short 0x{x:0>4}", .{h});
            try buf.append(allocator, '\n');
            pos += 2;
        } else if (pos + 4 <= code.len) { // 32-bit
            const w = std.mem.readInt(u32, code[pos..][0..4], .little);
            try buf.print(allocator, "{x:0>8}: {x:0>8}  ", .{ base + pos, w });
            try decode(allocator, &buf, w, null);
            try buf.append(allocator, '\n');
            pos += 4;
        } else break;
    }
    return buf.toOwnedSlice(allocator);
}

/// A defined symbol by absolute address (function or data), for resolving branch/jump targets.
pub const AddrSym = struct { name: []const u8, addr: u64, size: u64, is_func: bool };

/// The context that makes a listing absolute: the current instruction's address and the symbols.
const Context = struct { pc: u64, syms: []const AddrSym };

/// Render an RV64GC ELF `.text` as an objdump-style listing: absolute addresses, function labels,
/// variable-length (RVC-aware) stepping, and branch/jump targets resolved to `0x<addr> <sym+off>`.
/// `base` is the section load address; `syms` are all defined symbols. Caller owns the result.
/// A decoded source-line row keyed by absolute address (from a `.debug_line` program), for
/// objdump `-S`-style annotation of a real ELF.
pub const AddrLine = struct { addr: u64, line: u32 };

pub fn formatElf(allocator: std.mem.Allocator, code: []const u8, base: u64, syms: []const AddrSym, rvc: bool) std.mem.Allocator.Error![]u8 {
    return formatElfImpl(allocator, code, base, syms, rvc, &.{});
}

/// Like `formatElf`, but interleaves `; line N` markers from `lines` (sorted by address) before the
/// instruction at each address, so a listing of a `-g` object reads with its source lines.
pub fn formatElfWithLines(allocator: std.mem.Allocator, code: []const u8, base: u64, syms: []const AddrSym, rvc: bool, lines: []const AddrLine) std.mem.Allocator.Error![]u8 {
    return formatElfImpl(allocator, code, base, syms, rvc, lines);
}

fn formatElfImpl(allocator: std.mem.Allocator, code: []const u8, base: u64, syms: []const AddrSym, rvc: bool, lines: []const AddrLine) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var pos: usize = 0;
    var li: usize = 0;
    while (pos + 2 <= code.len) {
        const addr = base + pos;
        for (syms) |s| if (s.is_func and s.addr == addr) {
            if (pos != 0) try buf.append(allocator, '\n');
            try buf.print(allocator, "{x:0>16} <{s}>:\n", .{ addr, s.name });
        };
        while (li < lines.len and lines[li].addr < addr) : (li += 1) {}
        while (li < lines.len and lines[li].addr == addr) : (li += 1) {
            try buf.print(allocator, "; line {d}\n", .{lines[li].line});
        }
        const h = std.mem.readInt(u16, code[pos..][0..2], .little);
        const ctx: Context = .{ .pc = addr, .syms = syms };
        if (rvc and h & 3 != 3) {
            try buf.print(allocator, "{x:0>8}: {x:0>4}      ", .{ addr, h });
            if (!try decode16(allocator, &buf, h, ctx)) try buf.print(allocator, ".short 0x{x:0>4}", .{h});
            try buf.append(allocator, '\n');
            pos += 2;
        } else if (pos + 4 <= code.len) {
            const w = std.mem.readInt(u32, code[pos..][0..4], .little);
            try buf.print(allocator, "{x:0>8}: {x:0>8}  ", .{ addr, w });
            try decode(allocator, &buf, w, ctx);
            try buf.append(allocator, '\n');
            pos += 4;
        } else break;
    }
    return buf.toOwnedSlice(allocator);
}

/// Render a PC-relative target `rel_bytes` from the current PC: absolute `0x<addr> <sym+off>` with
/// a context, else the self-contained `.±N` form.
fn renderTarget(a: std.mem.Allocator, b: *std.ArrayList(u8), ctx: ?Context, rel_bytes: i64) !void {
    if (ctx) |c| {
        const abs: u64 = @intCast(@as(i64, @intCast(c.pc)) + rel_bytes);
        try b.print(a, "0x{x}", .{abs});
        try annotateSym(a, b, c.syms, abs);
    } else {
        try b.print(a, ".{s}{d}", .{ sgn(@intCast(rel_bytes)), mag(@intCast(rel_bytes)) });
    }
}

fn annotateSym(a: std.mem.Allocator, b: *std.ArrayList(u8), syms: []const AddrSym, addr: u64) !void {
    var best: ?AddrSym = null;
    for (syms) |s| {
        if (s.addr <= addr and (best == null or s.addr > best.?.addr)) best = s;
    }
    if (best) |s| {
        if (addr == s.addr) {
            try b.print(a, " <{s}>", .{s.name});
        } else {
            try b.print(a, " <{s}+0x{x}>", .{ s.name, addr - s.addr });
        }
    }
}

/// A source-line-table row: the byte offset where a source line's code begins.
pub const SourceLine = struct { offset: u32, line: u32 };

/// Render a listing with source-line markers interleaved (objdump `-S` style). `lines` must be
/// sorted by offset. Caller owns the result.
pub fn formatWithLines(allocator: std.mem.Allocator, code: []const u32, lines: []const SourceLine) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var li: usize = 0;
    for (code, 0..) |word, i| {
        const byte_off: u32 = @intCast(i * 4);
        while (li < lines.len and lines[li].offset == byte_off) : (li += 1) {
            try buf.print(allocator, "; line {d}\n", .{lines[li].line});
        }
        try buf.print(allocator, "{x:0>4}: {x:0>8}  ", .{ byte_off, word });
        try decode(allocator, &buf, word, null);
        try buf.append(allocator, '\n');
    }
    return buf.toOwnedSlice(allocator);
}

/// A function's location in a linked image: its name and its word offset.
pub const Sym = struct { name: []const u8, word: usize };

/// Render a whole linked module: each function gets a `name:` label at its offset, and every
/// `jal` (a resolved call/tail-call) that lands on a function is annotated with its name, so
/// a linked image (from link.compileModule) reads as a symbolized listing. Caller owns it.
pub fn formatModule(allocator: std.mem.Allocator, code: []const u32, syms: []const Sym) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    for (code, 0..) |word, i| {
        for (syms) |s| if (s.word == i) {
            if (i != 0) try buf.append(allocator, '\n');
            try buf.print(allocator, "{s}:\n", .{s.name});
        };
        try buf.print(allocator, "{x:0>4}: {x:0>8}  ", .{ i * 4, word });
        try decode(allocator, &buf, word, null);
        if (word & 0x7F == 0b1101111) { // jal: annotate a target that is a function symbol
            const target_byte = @as(i64, @intCast(i * 4)) + jimm(word);
            if (target_byte >= 0 and @rem(target_byte, 4) == 0) {
                const tw: usize = @intCast(@divExact(target_byte, 4));
                for (syms) |s| if (s.word == tw) {
                    try buf.print(allocator, "  <{s}>", .{s.name});
                    break;
                };
            }
        }
        try buf.append(allocator, '\n');
    }
    return buf.toOwnedSlice(allocator);
}

fn rd(w: u32) u32 {
    return (w >> 7) & 0x1F;
}
fn f3(w: u32) u32 {
    return (w >> 12) & 7;
}
fn rs1(w: u32) u32 {
    return (w >> 15) & 0x1F;
}
fn rs2(w: u32) u32 {
    return (w >> 20) & 0x1F;
}
fn f7(w: u32) u32 {
    return (w >> 25) & 0x7F;
}
fn f6(w: u32) u32 {
    return (w >> 26) & 0x3F;
}

/// Sign-extended 12-bit I-type immediate.
fn iimm(w: u32) i32 {
    return @as(i32, @bitCast(w)) >> 20;
}
/// Sign-extended 12-bit S-type (store) immediate.
fn simm(w: u32) i32 {
    const raw: u32 = (f7(w) << 5) | rd(w);
    return @as(i32, @bitCast(raw << 20)) >> 20;
}
/// Sign-extended 13-bit B-type (branch) byte offset.
fn bimm(w: u32) i32 {
    const u = (((w >> 31) & 1) << 12) | (((w >> 7) & 1) << 11) |
        (((w >> 25) & 0x3F) << 5) | (((w >> 8) & 0xF) << 1);
    return @as(i32, @bitCast(u << 19)) >> 19;
}
/// Sign-extended 21-bit J-type (jal) byte offset.
fn jimm(w: u32) i32 {
    const u = (((w >> 31) & 1) << 20) | (((w >> 12) & 0xFF) << 12) |
        (((w >> 20) & 1) << 11) | (((w >> 21) & 0x3FF) << 1);
    return @as(i32, @bitCast(u << 11)) >> 11;
}

fn decode(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32, ctx: ?Context) !void {
    switch (w & 0x7F) {
        0b0110011 => try opReg(a, b, w),
        0b0111011 => try opReg32(a, b, w), // OP-32: addw/subw/sllw/... (RV64 word ops)
        0b0010011 => try opImm(a, b, w),
        0b0011011 => try opImm32(a, b, w), // OP-IMM-32: addiw/slliw/srliw/sraiw
        0b0000011 => try loads(a, b, w),
        0b0100011 => try stores(a, b, w),
        0b1100011 => try branches(a, b, w, ctx),
        0b1101111 => { // jal, with j (rd=0) / jal (rd=ra) pseudos
            if (rd(w) == 0) {
                try b.appendSlice(a, "j ");
            } else if (rd(w) == 1) {
                try b.appendSlice(a, "jal ");
            } else {
                try b.print(a, "jal x{d}, ", .{rd(w)});
            }
            try renderTarget(a, b, ctx, jimm(w));
        },
        0b1100111 => if (iimm(w) == 0 and rd(w) == 0 and rs1(w) == 1)
            try b.appendSlice(a, "ret") // jalr x0, 0(ra)
        else if (iimm(w) == 0 and rd(w) == 0)
            try b.print(a, "jr x{d}", .{rs1(w)}) // jalr x0, 0(rs)
        else if (iimm(w) == 0 and rd(w) == 1)
            try b.print(a, "jalr x{d}", .{rs1(w)}) // jalr ra, 0(rs)
        else
            try b.print(a, "jalr x{d}, {d}(x{d})", .{ rd(w), iimm(w), rs1(w) }),
        0b0110111 => try b.print(a, "lui x{d}, 0x{x}", .{ rd(w), (w >> 12) & 0xFFFFF }),
        0b0010111 => try b.print(a, "auipc x{d}, 0x{x}", .{ rd(w), (w >> 12) & 0xFFFFF }),
        0b1110011 => try system(a, b, w),
        0b1010011 => try opFp(a, b, w),
        0b1000011 => try fma(a, b, w, "fmadd"),
        0b1000111 => try fma(a, b, w, "fmsub"),
        0b1001011 => try fma(a, b, w, "fnmsub"),
        0b1001111 => try fma(a, b, w, "fnmadd"),
        0b0000111 => try loadFp(a, b, w),
        0b0100111 => try storeFp(a, b, w),
        0b1010111 => try opV(a, b, w),
        else => try b.print(a, ".word 0x{x:0>8}", .{w}),
    }
}

fn opReg(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    const m = (f7(w) == 0b0000001); // M extension (mul/div/rem)
    if (!m and rs1(w) == 0) { // neg / snez pseudos (op against x0)
        if (f3(w) == 0b000 and f7(w) == 0b0100000) return b.print(a, "neg x{d}, x{d}", .{ rd(w), rs2(w) });
        if (f3(w) == 0b011) return b.print(a, "snez x{d}, x{d}", .{ rd(w), rs2(w) });
    }
    const mnem: []const u8 = switch (f3(w)) {
        0b000 => if (f7(w) == 0b0100000) "sub" else if (m) "mul" else "add",
        0b001 => if (m) "mulh" else "sll",
        0b010 => if (m) "mulhsu" else "slt",
        0b011 => if (m) "mulhu" else "sltu",
        0b100 => if (m) "div" else "xor",
        0b101 => if (f7(w) == 0b0100000) "sra" else if (m) "divu" else "srl",
        0b110 => if (m) "rem" else "or",
        0b111 => if (m) "remu" else "and",
        else => unreachable,
    };
    try b.print(a, "{s} x{d}, x{d}, x{d}", .{ mnem, rd(w), rs1(w), rs2(w) });
}

/// OP-32: the RV64 `*w` register ops that operate on the low 32 bits and sign-extend.
fn opReg32(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    const m = (f7(w) == 0b0000001);
    if (f3(w) == 0b000 and f7(w) == 0b0100000 and rs1(w) == 0) // negw pseudo
        return b.print(a, "negw x{d}, x{d}", .{ rd(w), rs2(w) });
    const mnem: []const u8 = switch (f3(w)) {
        0b000 => if (f7(w) == 0b0100000) "subw" else if (m) "mulw" else "addw",
        0b001 => "sllw",
        0b100 => "divw",
        0b101 => if (f7(w) == 0b0100000) "sraw" else if (m) "divuw" else "srlw",
        0b110 => "remw",
        0b111 => "remuw",
        else => "unknown",
    };
    try b.print(a, "{s} x{d}, x{d}, x{d}", .{ mnem, rd(w), rs1(w), rs2(w) });
}

/// OP-IMM-32: `addiw` and the `*iw` shifts (shamt is 5 bits for RV64 word shifts).
fn opImm32(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    switch (f3(w)) {
        0b000 => if (iimm(w) == 0) // sext.w rd, rs1
            try b.print(a, "sext.w x{d}, x{d}", .{ rd(w), rs1(w) })
        else
            try b.print(a, "addiw x{d}, x{d}, {d}", .{ rd(w), rs1(w), iimm(w) }),
        0b001 => try b.print(a, "slliw x{d}, x{d}, {d}", .{ rd(w), rs1(w), rs2(w) }),
        0b101 => if (f7(w) == 0b0100000)
            try b.print(a, "sraiw x{d}, x{d}, {d}", .{ rd(w), rs1(w), rs2(w) })
        else
            try b.print(a, "srliw x{d}, x{d}, {d}", .{ rd(w), rs1(w), rs2(w) }),
        else => try b.print(a, ".word 0x{x:0>8}", .{w}),
    }
}

fn opImm(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    switch (f3(w)) {
        0b000 => if (rs1(w) == 0) // li rd, imm
            try b.print(a, "li x{d}, {d}", .{ rd(w), iimm(w) })
        else if (iimm(w) == 0) // mv rd, rs1
            try b.print(a, "mv x{d}, x{d}", .{ rd(w), rs1(w) })
        else
            try b.print(a, "addi x{d}, x{d}, {d}", .{ rd(w), rs1(w), iimm(w) }),
        0b100 => if (iimm(w) == -1) // not rd, rs1
            try b.print(a, "not x{d}, x{d}", .{ rd(w), rs1(w) })
        else
            try b.print(a, "xori x{d}, x{d}, {d}", .{ rd(w), rs1(w), iimm(w) }),
        0b110 => try b.print(a, "ori x{d}, x{d}, {d}", .{ rd(w), rs1(w), iimm(w) }),
        0b111 => try b.print(a, "andi x{d}, x{d}, {d}", .{ rd(w), rs1(w), iimm(w) }),
        0b011 => if (iimm(w) == 1) // seqz rd, rs1
            try b.print(a, "seqz x{d}, x{d}", .{ rd(w), rs1(w) })
        else
            try b.print(a, "sltiu x{d}, x{d}, {d}", .{ rd(w), rs1(w), iimm(w) }),
        // RV64 shift-immediates use a 6-bit shamt (bits[25:20]).
        0b001 => try b.print(a, "slli x{d}, x{d}, {d}", .{ rd(w), rs1(w), (w >> 20) & 0x3F }),
        0b101 => if (iimm(w) == 0x6b8) // rev8 (Zbb) shares the shift-right slot
            try b.print(a, "rev8 x{d}, x{d}", .{ rd(w), rs1(w) })
        else if ((w >> 30) & 1 != 0)
            try b.print(a, "srai x{d}, x{d}, {d}", .{ rd(w), rs1(w), (w >> 20) & 0x3F })
        else
            try b.print(a, "srli x{d}, x{d}, {d}", .{ rd(w), rs1(w), (w >> 20) & 0x3F }),
        else => unreachable,
    }
}

fn loads(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    const mnem: []const u8 = switch (f3(w)) {
        0b000 => "lb",
        0b001 => "lh",
        0b010 => "lw",
        0b011 => "ld",
        0b100 => "lbu",
        0b101 => "lhu",
        0b110 => "lwu",
        0b111 => "??",
        else => unreachable,
    };
    try b.print(a, "{s} x{d}, {d}(x{d})", .{ mnem, rd(w), iimm(w), rs1(w) });
}

fn stores(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    const mnem: []const u8 = switch (f3(w)) {
        0b000 => "sb",
        0b001 => "sh",
        0b010 => "sw",
        0b011 => "sd",
        else => "??",
    };
    try b.print(a, "{s} x{d}, {d}(x{d})", .{ mnem, rs2(w), simm(w), rs1(w) });
}

fn branches(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32, ctx: ?Context) !void {
    // Prefer the zero-compare pseudos objdump prints (beqz/bnez/blez/bgez/bltz/bgtz).
    const one_zero = rs1(w) == 0 or rs2(w) == 0;
    if (one_zero) {
        const reg = if (rs1(w) == 0) rs2(w) else rs1(w);
        const pseudo: ?[]const u8 = switch (f3(w)) {
            0b000 => "beqz",
            0b001 => "bnez",
            0b100 => if (rs1(w) == 0) "bgtz" else "bltz", // blt x0,rs / blt rs,x0
            0b101 => if (rs1(w) == 0) "blez" else "bgez", // bge x0,rs / bge rs,x0
            else => null,
        };
        if (pseudo) |p| {
            try b.print(a, "{s} x{d}, ", .{ p, reg });
            return renderTarget(a, b, ctx, bimm(w));
        }
    }
    const mnem: []const u8 = switch (f3(w)) {
        0b000 => "beq",
        0b001 => "bne",
        0b100 => "blt",
        0b101 => "bge",
        0b110 => "bltu",
        0b111 => "bgeu",
        else => "??",
    };
    try b.print(a, "{s} x{d}, x{d}, ", .{ mnem, rs1(w), rs2(w) });
    try renderTarget(a, b, ctx, bimm(w));
}

fn system(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    if (w == 0x00000073) return b.appendSlice(a, "ecall");
    if (f3(w) == 0b010) return b.print(a, "csrrs x{d}, 0x{x}, x{d}", .{ rd(w), (w >> 20) & 0xFFF, rs1(w) });
    try b.print(a, ".word 0x{x:0>8}", .{w});
}

fn loadFp(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    switch (f3(w)) {
        0b010 => try b.print(a, "flw f{d}, {d}(x{d})", .{ rd(w), iimm(w), rs1(w) }),
        0b011 => try b.print(a, "fld f{d}, {d}(x{d})", .{ rd(w), iimm(w), rs1(w) }),
        0b110 => try b.print(a, "vle32.v v{d}, (x{d})", .{ rd(w), rs1(w) }),
        else => try b.print(a, ".word 0x{x:0>8}", .{w}),
    }
}

fn storeFp(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    switch (f3(w)) {
        0b010 => try b.print(a, "fsw f{d}, {d}(x{d})", .{ rs2(w), simm(w), rs1(w) }),
        0b011 => try b.print(a, "fsd f{d}, {d}(x{d})", .{ rs2(w), simm(w), rs1(w) }),
        0b110 => try b.print(a, "vse32.v v{d}, (x{d})", .{ rd(w), rs1(w) }),
        else => try b.print(a, ".word 0x{x:0>8}", .{w}),
    }
}

fn opFp(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    switch (f7(w)) {
        0b0000000 => try b.print(a, "fadd.s f{d}, f{d}, f{d}", .{ rd(w), rs1(w), rs2(w) }),
        0b0000100 => try b.print(a, "fsub.s f{d}, f{d}, f{d}", .{ rd(w), rs1(w), rs2(w) }),
        0b0001000 => try b.print(a, "fmul.s f{d}, f{d}, f{d}", .{ rd(w), rs1(w), rs2(w) }),
        0b0001100 => try b.print(a, "fdiv.s f{d}, f{d}, f{d}", .{ rd(w), rs1(w), rs2(w) }),
        0b0000001 => try b.print(a, "fadd.d f{d}, f{d}, f{d}", .{ rd(w), rs1(w), rs2(w) }),
        0b0000101 => try b.print(a, "fsub.d f{d}, f{d}, f{d}", .{ rd(w), rs1(w), rs2(w) }),
        0b0001001 => try b.print(a, "fmul.d f{d}, f{d}, f{d}", .{ rd(w), rs1(w), rs2(w) }),
        0b0001101 => try b.print(a, "fdiv.d f{d}, f{d}, f{d}", .{ rd(w), rs1(w), rs2(w) }),
        0b0010000 => try b.print(a, "fmv.s f{d}, f{d}", .{ rd(w), rs1(w) }), // fsgnj.s rs,rs
        0b0010001 => try b.print(a, "fmv.d f{d}, f{d}", .{ rd(w), rs1(w) }),
        0b1010000 => try fcmp(a, b, w, ".s"),
        0b1010001 => try fcmp(a, b, w, ".d"),
        0b1100000 => try b.print(a, "fcvt.w.s x{d}, f{d}", .{ rd(w), rs1(w) }),
        0b1100001 => try b.print(a, "fcvt.w.d x{d}, f{d}", .{ rd(w), rs1(w) }),
        0b1101000 => try b.print(a, "fcvt.s.w f{d}, x{d}", .{ rd(w), rs1(w) }),
        0b1101001 => try b.print(a, "fcvt.d.w f{d}, x{d}", .{ rd(w), rs1(w) }),
        0b1110000 => try b.print(a, "fmv.x.w x{d}, f{d}", .{ rd(w), rs1(w) }),
        0b1111000 => try b.print(a, "fmv.w.x f{d}, x{d}", .{ rd(w), rs1(w) }),
        0b1111001 => try b.print(a, "fmv.d.x f{d}, x{d}", .{ rd(w), rs1(w) }),
        else => try b.print(a, ".word 0x{x:0>8}", .{w}),
    }
}

/// Fused multiply-add family (R4-type): `fmadd/fmsub/fnmsub/fnmadd .s/.d rd, rs1, rs2, rs3`.
fn fma(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32, mnem: []const u8) !void {
    const suffix: []const u8 = if ((w >> 25) & 3 == 1) ".d" else ".s";
    const rs3 = (w >> 27) & 0x1F;
    try b.print(a, "{s}{s} f{d}, f{d}, f{d}, f{d}", .{ mnem, suffix, rd(w), rs1(w), rs2(w), rs3 });
}

fn fcmp(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32, suffix: []const u8) !void {
    const mnem: []const u8 = switch (f3(w)) {
        0b010 => "feq",
        0b001 => "flt",
        0b000 => "fle",
        else => "??",
    };
    try b.print(a, "{s}{s} x{d}, f{d}, f{d}", .{ mnem, suffix, rd(w), rs1(w), rs2(w) });
}

fn opV(a: std.mem.Allocator, b: *std.ArrayList(u8), w: u32) !void {
    switch (f3(w)) {
        0b111 => try b.print(a, "vsetivli x{d}, {d}, 0x{x}", .{ rd(w), rs1(w), (w >> 20) & 0x3FF }),
        0b001 => switch (f6(w)) { // OPFVV
            0b000000 => try b.print(a, "vfadd.vv v{d}, v{d}, v{d}", .{ rd(w), rs2(w), rs1(w) }),
            0b000010 => try b.print(a, "vfsub.vv v{d}, v{d}, v{d}", .{ rd(w), rs2(w), rs1(w) }),
            0b100100 => try b.print(a, "vfmul.vv v{d}, v{d}, v{d}", .{ rd(w), rs2(w), rs1(w) }),
            0b100000 => try b.print(a, "vfdiv.vv v{d}, v{d}, v{d}", .{ rd(w), rs2(w), rs1(w) }),
            0b010000 => try b.print(a, "vfmv.f.s f{d}, v{d}", .{ rd(w), rs2(w) }),
            else => try b.print(a, ".word 0x{x:0>8}", .{w}),
        },
        0b101 => switch (f6(w)) { // OPFVF
            0b010000 => try b.print(a, "vfmv.s.f v{d}, f{d}", .{ rd(w), rs1(w) }),
            0b001110 => try b.print(a, "vfslide1up.vf v{d}, v{d}, f{d}", .{ rd(w), rs2(w), rs1(w) }),
            else => try b.print(a, ".word 0x{x:0>8}", .{w}),
        },
        0b011 => try b.print(a, "vslidedown.vi v{d}, v{d}, {d}", .{ rd(w), rs2(w), rs1(w) }), // uimm in rs1 slot
        0b000 => try b.print(a, "vmv.v.v v{d}, v{d}", .{ rd(w), rs1(w) }),
        else => try b.print(a, ".word 0x{x:0>8}", .{w}),
    }
}

fn sgn(v: i32) []const u8 {
    return if (v < 0) "-" else "+";
}
fn mag(v: i32) u32 {
    return @intCast(if (v < 0) -v else v);
}

/// A single bit `n` of `h`.
fn cb(h: u16, n: u4) u32 {
    return (h >> n) & 1;
}
/// A compressed 3-bit register field (`rd'`/`rs1'`/`rs2'`) maps 0..7 to x8..x15.
fn creg(h: u16, lo: u4) u32 {
    return 8 + ((h >> lo) & 7);
}

/// Decode a 16-bit compressed instruction, expanding it to its base mnemonic (and the pseudo a
/// disassembler prints: `ret`/`mv`/`li`/`j`/`nop`/`beqz`/`bnez`). Returns false for an unknown /
/// reserved encoding so the caller can fall back.
fn decode16(a: std.mem.Allocator, b: *std.ArrayList(u8), h: u16, ctx: ?Context) !bool {
    const quadrant = h & 3;
    const f = (h >> 13) & 7;
    switch (quadrant) {
        0 => switch (f) {
            0b000 => { // c.addi4spn -> addi rd', x2, nzuimm
                const nz = (((h >> 11) & 3) << 4) | (((h >> 7) & 0xF) << 6) | (cb(h, 6) << 2) | (cb(h, 5) << 3);
                if (nz == 0) return false; // reserved
                try b.print(a, "addi x{d}, x2, {d}", .{ creg(h, 2), nz });
            },
            0b010 => { // c.lw rd', uimm(rs1')
                const uimm = (((h >> 10) & 7) << 3) | (cb(h, 6) << 2) | (cb(h, 5) << 6);
                try b.print(a, "lw x{d}, {d}(x{d})", .{ creg(h, 2), uimm, creg(h, 7) });
            },
            0b011 => { // c.ld rd', uimm(rs1')
                const uimm = (((h >> 10) & 7) << 3) | (((h >> 5) & 3) << 6);
                try b.print(a, "ld x{d}, {d}(x{d})", .{ creg(h, 2), uimm, creg(h, 7) });
            },
            0b110 => { // c.sw rs2', uimm(rs1')
                const uimm = (((h >> 10) & 7) << 3) | (cb(h, 6) << 2) | (cb(h, 5) << 6);
                try b.print(a, "sw x{d}, {d}(x{d})", .{ creg(h, 2), uimm, creg(h, 7) });
            },
            0b111 => { // c.sd rs2', uimm(rs1')
                const uimm = (((h >> 10) & 7) << 3) | (((h >> 5) & 3) << 6);
                try b.print(a, "sd x{d}, {d}(x{d})", .{ creg(h, 2), uimm, creg(h, 7) });
            },
            0b001 => { // c.fld
                const uimm = (((h >> 10) & 7) << 3) | (((h >> 5) & 3) << 6);
                try b.print(a, "fld f{d}, {d}(x{d})", .{ creg(h, 2), uimm, creg(h, 7) });
            },
            0b101 => { // c.fsd
                const uimm = (((h >> 10) & 7) << 3) | (((h >> 5) & 3) << 6);
                try b.print(a, "fsd f{d}, {d}(x{d})", .{ creg(h, 2), uimm, creg(h, 7) });
            },
            else => return false,
        },
        1 => switch (f) {
            0b000 => { // c.addi rd, rd, nzimm (rd==0 -> nop)
                const rdn = (h >> 7) & 0x1F;
                const imm = cimm6(h);
                if (rdn == 0) {
                    try b.appendSlice(a, "nop");
                    return true;
                }
                try b.print(a, "addi x{d}, x{d}, {d}", .{ rdn, rdn, imm });
            },
            0b001 => { // c.addiw rd, rd, imm
                const rdn = (h >> 7) & 0x1F;
                try b.print(a, "addiw x{d}, x{d}, {d}", .{ rdn, rdn, cimm6(h) });
            },
            0b010 => { // c.li rd, imm -> li
                try b.print(a, "li x{d}, {d}", .{ (h >> 7) & 0x1F, cimm6(h) });
            },
            0b011 => { // c.addi16sp (rd==2) or c.lui
                const rdn = (h >> 7) & 0x1F;
                if (rdn == 2) {
                    const raw = (cb(h, 12) << 9) | (cb(h, 6) << 4) | (cb(h, 5) << 6) | (((h >> 3) & 3) << 7) | (cb(h, 2) << 5);
                    try b.print(a, "addi x2, x2, {d}", .{signext(raw, 10)});
                } else {
                    const raw = (cb(h, 12) << 17) | (((h >> 2) & 0x1F) << 12);
                    try b.print(a, "lui x{d}, {d}", .{ rdn, @as(u32, @bitCast(signext(raw, 18))) >> 12 });
                }
            },
            0b100 => { // MISC-ALU
                const sub = (h >> 10) & 3;
                const rdp = creg(h, 7);
                switch (sub) {
                    0b00 => try b.print(a, "srli x{d}, x{d}, {d}", .{ rdp, rdp, cshamt(h) }),
                    0b01 => try b.print(a, "srai x{d}, x{d}, {d}", .{ rdp, rdp, cshamt(h) }),
                    0b10 => try b.print(a, "andi x{d}, x{d}, {d}", .{ rdp, rdp, cimm6(h) }),
                    else => { // register-register
                        const rs2p = creg(h, 2);
                        const mnem: []const u8 = if (cb(h, 12) == 0) switch ((h >> 5) & 3) {
                            0 => "sub",
                            1 => "xor",
                            2 => "or",
                            else => "and",
                        } else switch ((h >> 5) & 3) {
                            0 => "subw",
                            1 => "addw",
                            else => return false,
                        };
                        try b.print(a, "{s} x{d}, x{d}, x{d}", .{ mnem, rdp, rdp, rs2p });
                    },
                }
            },
            0b101 => { // c.j
                try b.appendSlice(a, "j ");
                try renderTarget(a, b, ctx, cjimm(h));
            },
            0b110 => { // c.beqz
                try b.print(a, "beqz x{d}, ", .{creg(h, 7)});
                try renderTarget(a, b, ctx, cbimm(h));
            },
            0b111 => { // c.bnez
                try b.print(a, "bnez x{d}, ", .{creg(h, 7)});
                try renderTarget(a, b, ctx, cbimm(h));
            },
            else => return false,
        },
        2 => switch (f) {
            0b000 => { // c.slli
                const rdn = (h >> 7) & 0x1F;
                try b.print(a, "slli x{d}, x{d}, {d}", .{ rdn, rdn, cshamt(h) });
            },
            0b010 => { // c.lwsp
                const uimm = (cb(h, 12) << 5) | (((h >> 4) & 7) << 2) | (((h >> 2) & 3) << 6);
                try b.print(a, "lw x{d}, {d}(x2)", .{ (h >> 7) & 0x1F, uimm });
            },
            0b011 => { // c.ldsp
                const uimm = (cb(h, 12) << 5) | (((h >> 5) & 3) << 3) | (((h >> 2) & 7) << 6);
                try b.print(a, "ld x{d}, {d}(x2)", .{ (h >> 7) & 0x1F, uimm });
            },
            0b100 => { // c.jr / c.mv / c.jalr / c.add / c.ebreak
                const rdn = (h >> 7) & 0x1F;
                const rs2n = (h >> 2) & 0x1F;
                if (cb(h, 12) == 0) {
                    if (rs2n == 0) {
                        if (rdn == 1) { // jr ra -> ret
                            try b.appendSlice(a, "ret");
                            return true;
                        }
                        try b.print(a, "jr x{d}", .{rdn});
                    } else {
                        try b.print(a, "mv x{d}, x{d}", .{ rdn, rs2n });
                    }
                } else {
                    if (rdn == 0 and rs2n == 0) {
                        try b.appendSlice(a, "ebreak");
                        return true;
                    }
                    if (rs2n == 0) {
                        try b.print(a, "jalr x{d}", .{rdn});
                    } else {
                        try b.print(a, "add x{d}, x{d}, x{d}", .{ rdn, rdn, rs2n });
                    }
                }
            },
            0b110 => { // c.swsp
                const uimm = (((h >> 9) & 0xF) << 2) | (((h >> 7) & 3) << 6);
                try b.print(a, "sw x{d}, {d}(x2)", .{ (h >> 2) & 0x1F, uimm });
            },
            0b111 => { // c.sdsp
                const uimm = (((h >> 10) & 7) << 3) | (((h >> 7) & 7) << 6);
                try b.print(a, "sd x{d}, {d}(x2)", .{ (h >> 2) & 0x1F, uimm });
            },
            0b001 => { // c.fldsp
                const uimm = (cb(h, 12) << 5) | (((h >> 5) & 3) << 3) | (((h >> 2) & 7) << 6);
                try b.print(a, "fld f{d}, {d}(x2)", .{ (h >> 7) & 0x1F, uimm });
            },
            0b101 => { // c.fsdsp
                const uimm = (((h >> 10) & 7) << 3) | (((h >> 7) & 7) << 6);
                try b.print(a, "fsd f{d}, {d}(x2)", .{ (h >> 2) & 0x1F, uimm });
            },
            else => return false,
        },
        else => return false,
    }
    return true;
}

/// The sign-extended 6-bit immediate common to c.addi/c.li/c.andi (imm[5]=h[12], imm[4:0]=h[6:2]).
fn cimm6(h: u16) i32 {
    return signext((cb(h, 12) << 5) | ((h >> 2) & 0x1F), 6);
}
/// The 6-bit shift amount of c.slli/c.srli/c.srai (RV64: up to 63).
fn cshamt(h: u16) u32 {
    return (cb(h, 12) << 5) | ((h >> 2) & 0x1F);
}
/// The CJ (c.j) sign-extended byte offset.
fn cjimm(h: u16) i32 {
    const u = (cb(h, 12) << 11) | (cb(h, 11) << 4) | (((h >> 9) & 3) << 8) | (cb(h, 8) << 10) |
        (cb(h, 7) << 6) | (cb(h, 6) << 7) | (((h >> 3) & 7) << 1) | (cb(h, 2) << 5);
    return signext(u, 12);
}
/// The CB (c.beqz/c.bnez) sign-extended byte offset.
fn cbimm(h: u16) i32 {
    const u = (cb(h, 12) << 8) | (((h >> 10) & 3) << 3) | (((h >> 5) & 3) << 6) |
        (((h >> 3) & 3) << 1) | (cb(h, 2) << 5);
    return signext(u, 9);
}
/// Sign-extend the low `bits` of `v`.
fn signext(v: u32, bits: u5) i32 {
    const shift: u5 = @intCast(32 - @as(u32, bits));
    return @as(i32, @bitCast(v << shift)) >> shift;
}

fn expectOne(word: u32, expected: []const u8) !void {
    const s = try one(std.testing.allocator, word);
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(expected, s);
}

test "round-trips integer register and immediate ops" {
    try expectOne(encode.add(.x1, .x2, .x3), "add x1, x2, x3");
    try expectOne(encode.sub(.x1, .x2, .x3), "sub x1, x2, x3");
    try expectOne(encode.mul(.x1, .x2, .x3), "mul x1, x2, x3");
    try expectOne(encode.divu(.x1, .x2, .x3), "divu x1, x2, x3");
    try expectOne(encode.remu(.x1, .x2, .x3), "remu x1, x2, x3");
    try expectOne(encode.and_(.x1, .x2, .x3), "and x1, x2, x3");
    try expectOne(encode.sra(.x1, .x2, .x3), "sra x1, x2, x3");
    try expectOne(encode.sltu(.x1, .x2, .x3), "sltu x1, x2, x3");
    try expectOne(encode.addi(.x1, .x2, -5), "addi x1, x2, -5");
    try expectOne(encode.andi(.x1, .x2, 5), "andi x1, x2, 5");
    try expectOne(encode.slli(.x1, .x2, 5), "slli x1, x2, 5");
    try expectOne(encode.srai(.x1, .x2, 5), "srai x1, x2, 5");
    try expectOne(encode.rev8(.x1, .x2), "rev8 x1, x2");
    try expectOne(encode.lui(.x5, 0x12345), "lui x5, 0x12345");
    try expectOne(encode.auipc(.x5, 0x10), "auipc x5, 0x10");
}

test "round-trips loads, stores, branches, jumps, system" {
    try expectOne(encode.lw(.x1, .x2, 8), "lw x1, 8(x2)");
    try expectOne(encode.ld(.x1, .x2, -8), "ld x1, -8(x2)");
    try expectOne(encode.lbu(.x1, .x2, 0), "lbu x1, 0(x2)");
    try expectOne(encode.sw(.x1, .x2, 4), "sw x1, 4(x2)");
    try expectOne(encode.sd(.x3, .x2, -16), "sd x3, -16(x2)");
    try expectOne(encode.beq(.x1, .x2, 16), "beq x1, x2, .+16");
    try expectOne(encode.bne(.x1, .x2, -8), "bne x1, x2, .-8");
    try expectOne(encode.blt(.x1, .x2, 12), "blt x1, x2, .+12");
    try expectOne(encode.jal(.x1, 2048), "jal .+2048"); // jal ra -> `jal` (ra implied)
    try expectOne(encode.jalr(.x0, .x1, 0), "ret"); // jalr x0, 0(ra) -> ret
    try expectOne(encode.ecall(), "ecall");
    try expectOne(encode.csrrs(.x5, 0xC00, .x0), "csrrs x5, 0xc00, x0");
}

test "round-trips scalar floating point" {
    try expectOne(encode.fadd_s(.f0, .f1, .f2), "fadd.s f0, f1, f2");
    try expectOne(encode.fdiv_d(.f0, .f1, .f2), "fdiv.d f0, f1, f2");
    try expectOne(encode.flt_s(.x1, .f2, .f3), "flt.s x1, f2, f3");
    try expectOne(encode.fle_d(.x1, .f2, .f3), "fle.d x1, f2, f3");
    try expectOne(encode.flw(.f1, .x2, 4), "flw f1, 4(x2)");
    try expectOne(encode.fsd(.f1, .x2, 8), "fsd f1, 8(x2)");
    try expectOne(encode.fmv_w_x(.f0, .x1), "fmv.w.x f0, x1");
    try expectOne(encode.fmv_x_w(.x0, .f1), "fmv.x.w x0, f1");
    try expectOne(encode.fcvt_s_w(.f0, .x1), "fcvt.s.w f0, x1");
    try expectOne(encode.fcvt_w_s(.x0, .f1), "fcvt.w.s x0, f1");
    try expectOne(encode.fmv_s(.f0, .f1), "fmv.s f0, f1");
}

fn expectC(h: u16, expected: []const u8) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    try std.testing.expect(try decode16(std.testing.allocator, &buf, h, null));
    try std.testing.expectEqualStrings(expected, buf.items);
}

test "decodes RVC compressed instructions (verified against llvm-objdump)" {
    try expectC(0x4601, "li x12, 0"); // c.li
    try expectC(0x95aa, "add x11, x11, x10"); // c.add
    try expectC(0x4114, "lw x13, 0(x10)"); // c.lw
    try expectC(0x0511, "addi x10, x10, 4"); // c.addi
    try expectC(0x8082, "ret"); // c.jr ra
    try expectC(0x8e29, "xor x12, x12, x10"); // c.xor (MISC-ALU)
    try expectC(0x050a, "slli x10, x10, 2"); // c.slli
    try expectC(0x842a, "mv x8, x10"); // c.mv
    try expectC(0x0001, "nop"); // c.nop
    try expectC(0xca09, "beqz x12, .+18"); // c.beqz
    try expectC(0xfa75, "bnez x12, .-12"); // c.bnez
}

test "decodes RV64 word ops, FMA, and base pseudos (verified against llvm-objdump)" {
    try expectOne(0x41f6551b, "sraiw x10, x12, 31");
    try expectOne(0x40a6053b, "subw x10, x12, x10");
    try expectOne(0x0005c603, "lbu x12, 0(x11)");
    try expectOne(0x7ab577c3, "fmadd.d f15, f10, f11, f15");
    try expectOne(0x40b0053b, "negw x10, x11"); // subw rd, x0, rs -> negw
    try expectOne(0x40b00533, "neg x10, x11"); // sub rd, x0, rs -> neg
    try expectOne(0x0005851b, "sext.w x10, x11"); // addiw rd, rs, 0
    try expectOne(0x02061693, "slli x13, x12, 32"); // RV64 6-bit shamt
}

test "round-trips RVV vector ops" {
    try expectOne(encode.vsetivli(.x0, 4, 0xD0), "vsetivli x0, 4, 0xd0");
    try expectOne(encode.vle32(.v1, .x2), "vle32.v v1, (x2)");
    try expectOne(encode.vse32(.v1, .x2), "vse32.v v1, (x2)");
    try expectOne(encode.vfadd_vv(.v1, .v2, .v3), "vfadd.vv v1, v2, v3");
    try expectOne(encode.vfmul_vv(.v1, .v2, .v3), "vfmul.vv v1, v2, v3");
    try expectOne(encode.vfmv_f_s(.f1, .v2), "vfmv.f.s f1, v2");
    try expectOne(encode.vfmv_s_f(.v1, .f2), "vfmv.s.f v1, f2");
    try expectOne(encode.vfslide1up_vf(.v1, .v2, .f3), "vfslide1up.vf v1, v2, f3");
    try expectOne(encode.vslidedown_vi(.v1, .v2, 3), "vslidedown.vi v1, v2, 3");
    try expectOne(encode.vmv_v_v(.v1, .v2), "vmv.v.v v1, v2");
}
