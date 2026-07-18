//! AArch64 (A64) disassembler: the inverse of `encode.zig`. It decodes exactly the
//! instructions the encoder emits into a readable listing, so codegen debugging does not
//! mean hand-decoding hex. Field positions mirror encode.zig bit-for-bit. Anything outside
//! the emitted subset prints as `.word 0x<hex>` so nothing is silently mis-decoded.
//!
//! `one` renders a single 32-bit instruction word; `format` renders a whole code buffer
//! with byte addresses. Both are validated by round-tripping every encoder function.

const std = @import("std");
const encode = @import("encode.zig");

/// Render one instruction word as `mnemonic operands`. Caller owns the result.
pub fn one(allocator: std.mem.Allocator, word: u32) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    try decode(allocator, &buf, word, null);
    return buf.toOwnedSlice(allocator);
}

/// Render a whole instruction stream (one word per line, `addr: hex  text`). Caller owns it.
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

/// A source-line-table row: the byte offset where a source line's code begins.
pub const SourceLine = struct { offset: u32, line: u32 };

/// Render a listing with source-line markers interleaved (objdump `-S` style): a `; line N`
/// header precedes the instructions belonging to that source line. `lines` must be sorted by
/// offset (as isel produces). Caller owns the result.
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
/// resolved `bl` is annotated with the callee's name (`<helper>`), so a linked image reads
/// like a symbolized listing. Caller owns the result.
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
        // Annotate any PC-relative branch (call or local) with the function it lands in, as
        // `<sym>` or `<sym+0xN>`, the way objdump does. Makes real control flow readable.
        if (branchTargetWord(word, i)) |target| {
            if (containingSym(syms, target)) |s| {
                const off = target - @as(i64, @intCast(s.word));
                if (off == 0) {
                    try buf.print(allocator, "  <{s}>", .{s.name});
                } else {
                    try buf.print(allocator, "  <{s}+0x{x}>", .{ s.name, @as(u64, @intCast(off)) * 4 });
                }
            }
        }
        try buf.append(allocator, '\n');
    }
    return buf.toOwnedSlice(allocator);
}

/// A defined symbol by absolute address (a function or data object), for resolving branch and
/// `adrp` targets when disassembling an ELF image.
pub const AddrSym = struct { name: []const u8, addr: u64, size: u64, is_func: bool };

/// The context that makes a listing absolute: the current instruction's address and the image's
/// symbols. When present, PC-relative targets render as `0x<addr> <sym+off>` (objdump style).
const Context = struct { pc: u64, syms: []const AddrSym };

/// Render an ELF code section as an objdump-style listing: absolute addresses, function labels,
/// and branch/`adrp` targets resolved to `0x<addr> <sym+off>`. `base` is the section's load
/// address; `syms` are all defined symbols (functions and data). Caller owns the result.
/// A decoded source-line row keyed by absolute address (from a `.debug_line` program), for
/// objdump `-S`-style annotation of a real ELF.
pub const AddrLine = struct { addr: u64, line: u32 };

pub fn formatElf(allocator: std.mem.Allocator, code: []const u32, base: u64, syms: []const AddrSym) std.mem.Allocator.Error![]u8 {
    return formatElfImpl(allocator, code, base, syms, &.{});
}

/// Like `formatElf`, but interleaves `; line N` markers from `lines` (sorted by address) before the
/// instruction at each address, so a listing of a `-g` object reads with its source lines.
pub fn formatElfWithLines(allocator: std.mem.Allocator, code: []const u32, base: u64, syms: []const AddrSym, lines: []const AddrLine) std.mem.Allocator.Error![]u8 {
    return formatElfImpl(allocator, code, base, syms, lines);
}

fn formatElfImpl(allocator: std.mem.Allocator, code: []const u32, base: u64, syms: []const AddrSym, lines: []const AddrLine) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var li: usize = 0;
    for (code, 0..) |word, i| {
        const addr = base + @as(u64, i) * 4;
        for (syms) |s| if (s.is_func and s.addr == addr) {
            if (i != 0) try buf.append(allocator, '\n');
            try buf.print(allocator, "{x:0>16} <{s}>:\n", .{ addr, s.name });
        };
        while (li < lines.len and lines[li].addr < addr) : (li += 1) {} // skip any before this insn
        while (li < lines.len and lines[li].addr == addr) : (li += 1) {
            try buf.print(allocator, "; line {d}\n", .{lines[li].line});
        }
        try buf.print(allocator, "{x:0>8}: {x:0>8}  ", .{ addr, word });
        try decode(allocator, &buf, word, .{ .pc = addr, .syms = syms });
        try buf.append(allocator, '\n');
    }
    return buf.toOwnedSlice(allocator);
}

/// Render a PC-relative code target `rel_bytes` from the current PC: absolute `0x<addr> <sym+off>`
/// when a context is present, else the self-contained `.±N` form.
fn renderTarget(a: std.mem.Allocator, buf: *std.ArrayList(u8), ctx: ?Context, rel_bytes: i64) !void {
    if (ctx) |c| {
        // Wrapping add (like the adrp path): a disassembler must never panic on the
        // bytes it is given, including a branch whose displacement runs below zero.
        const abs: u64 = @bitCast(@as(i64, @bitCast(c.pc)) +% rel_bytes);
        try buf.print(a, "0x{x}", .{abs});
        try annotateSym(a, buf, c.syms, abs);
    } else {
        try buf.print(a, ".{s}{d}", .{ signI(rel_bytes), absI(rel_bytes) });
    }
}

/// Append ` <sym+off>` for the symbol that contains `addr`, if any.
fn annotateSym(a: std.mem.Allocator, buf: *std.ArrayList(u8), syms: []const AddrSym, addr: u64) !void {
    if (symbolAt(syms, addr)) |m| {
        if (m.off == 0) {
            try buf.print(a, " <{s}>", .{m.name});
        } else {
            try buf.print(a, " <{s}+0x{x}>", .{ m.name, m.off });
        }
    }
}

const Match = struct { name: []const u8, off: u64 };

/// The symbol containing `addr` (the greatest-addressed symbol at or before it), or null.
fn symbolAt(syms: []const AddrSym, addr: u64) ?Match {
    var best: ?AddrSym = null;
    for (syms) |s| {
        if (s.addr <= addr and (best == null or s.addr > best.?.addr)) best = s;
    }
    if (best) |s| return .{ .name = s.name, .off = addr - s.addr };
    return null;
}

fn signI(v: i64) []const u8 {
    return if (v < 0) "-" else "+";
}
fn absI(v: i64) u64 {
    return @intCast(if (v < 0) -v else v);
}

/// The target word index of a PC-relative branch (b/bl/b.cond/cbz/cbnz/tbz/tbnz) at `index`,
/// or null if `w` is not such a branch.
fn branchTargetWord(w: u32, index: usize) ?i64 {
    const base: i64 = @intCast(index);
    if (w & 0x7C000000 == 0x14000000) { // b / bl: imm26
        return base + (@as(i32, @bitCast(w << 6)) >> 6);
    }
    if (w & 0xFF000010 == 0x54000000 or w & 0x7E000000 == 0x34000000) { // b.cond / cbz / cbnz: imm19
        return base + (@as(i32, @bitCast(w << 8)) >> 13);
    }
    if (w & 0x7E000000 == 0x36000000) { // tbz / tbnz: imm14
        return base + (@as(i32, @bitCast(w << 13)) >> 18);
    }
    return null;
}

/// The symbol whose range contains word `target` (the greatest symbol at or before it), or null.
fn containingSym(syms: []const Sym, target: i64) ?Sym {
    if (target < 0) return null;
    var best: ?Sym = null;
    for (syms) |s| {
        if (s.word <= target and (best == null or s.word > best.?.word)) best = s;
    }
    return best;
}

fn rd(w: u32) u5 {
    return @intCast(w & 0x1F);
}
fn rn(w: u32) u5 {
    return @intCast((w >> 5) & 0x1F);
}
fn rm(w: u32) u5 {
    return @intCast((w >> 16) & 0x1F);
}
fn ra(w: u32) u5 {
    return @intCast((w >> 10) & 0x1F);
}
fn sf(w: u32) bool {
    return (w >> 31) & 1 != 0;
}

/// A 32/64-bit general register (`w`/`x`), rendering 31 as the zero register.
fn gp(buf: *std.ArrayList(u8), a: std.mem.Allocator, x64: bool, r: u5) !void {
    const c: u8 = if (x64) 'x' else 'w';
    if (r == 31) {
        try buf.print(a, "{c}zr", .{c});
    } else {
        try buf.print(a, "{c}{d}", .{ c, r });
    }
}

/// A general register in an address/stack context, rendering 31 as `sp`.
fn sp(buf: *std.ArrayList(u8), a: std.mem.Allocator, x64: bool, r: u5) !void {
    if (r == 31) {
        try buf.appendSlice(a, "sp");
    } else {
        try gp(buf, a, x64, r);
    }
}

fn condName(v: u4) []const u8 {
    // binutils spelling (cs/cc rather than the hs/lo synonyms) so listings match objdump.
    return switch (v) {
        0 => "eq",
        1 => "ne",
        2 => "cs",
        3 => "cc",
        4 => "mi",
        5 => "pl",
        6 => "vs",
        7 => "vc",
        8 => "hi",
        9 => "ls",
        10 => "ge",
        11 => "lt",
        12 => "gt",
        13 => "le",
        14 => "al",
        15 => "nv",
    };
}

/// Masks for the three-register data-processing form: clear rd, rn, rm; keep the opcode
/// and the op2/shift field in bits[10:15]. The `sf` bit is dropped so one entry covers both
/// the w and x forms.
const rrr_mask: u32 = 0x7FE0FC00;

const RRR = struct { match: u32, mnem: []const u8 };

/// The 32-bit-base opcodes of the shared three-register form (sf dropped).
const rrr_table = [_]RRR{
    .{ .match = 0x0B000000, .mnem = "add" },
    .{ .match = 0x4B000000, .mnem = "sub" },
    .{ .match = 0x0A000000, .mnem = "and" },
    .{ .match = 0x2A000000, .mnem = "orr" },
    .{ .match = 0x4A000000, .mnem = "eor" },
    .{ .match = 0x1AC00C00, .mnem = "sdiv" },
    .{ .match = 0x1AC00800, .mnem = "udiv" },
    .{ .match = 0x1AC02000, .mnem = "lsl" },
    .{ .match = 0x1AC02400, .mnem = "lsr" },
    .{ .match = 0x1AC02800, .mnem = "asr" },
};

fn decode(a: std.mem.Allocator, buf: *std.ArrayList(u8), w: u32, ctx: ?Context) !void {
    // Exact, operandless / special encodings first.
    if (w == encode.ret()) return buf.appendSlice(a, "ret");
    if (w & 0xFFE0001F == 0xD4000001) return buf.print(a, "svc #{d}", .{(w >> 5) & 0xFFFF});
    if (w & 0xFFFFFC1F == 0xD63F0000) {
        try buf.appendSlice(a, "blr ");
        return sp(buf, a, true, rn(w));
    }

    // `mov` (register) is `orr Xd, xzr, Xm` with rn == 31; render it as a move.
    if (w & 0x7FE0FFE0 == 0x2A0003E0) {
        try buf.appendSlice(a, "mov ");
        try gp(buf, a, sf(w), rd(w));
        try buf.appendSlice(a, ", ");
        return gp(buf, a, sf(w), rm(w));
    }

    // neg Rd, Rm = sub Rd, RZR, Rm (rn == 31), the objdump alias.
    if (w & 0x7FE0FC00 == 0x4B000000 and rn(w) == 31) {
        try buf.appendSlice(a, "neg ");
        try gp(buf, a, sf(w), rd(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, sf(w), rm(w));
        return;
    }

    // The shared three-register form.
    for (rrr_table) |e| {
        if (w & rrr_mask == e.match) {
            try buf.print(a, "{s} ", .{e.mnem});
            try gp(buf, a, sf(w), rd(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rn(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rm(w));
            return;
        }
    }

    // madd/msub (mul is madd with ra == 31).
    if (w & 0x7FE08000 == 0x1B000000 or w & 0x7FE08000 == 0x1B008000) {
        const is_sub = (w >> 15) & 1 != 0;
        if (!is_sub and ra(w) == 31) {
            try buf.appendSlice(a, "mul ");
            try gp(buf, a, sf(w), rd(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rn(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rm(w));
            return;
        }
        if (is_sub and ra(w) == 31) { // mneg Rd, Rn, Rm = msub Rd, Rn, Rm, RZR
            try buf.appendSlice(a, "mneg ");
            try gp(buf, a, sf(w), rd(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rn(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rm(w));
            return;
        }
        try buf.print(a, "{s} ", .{if (is_sub) "msub" else "madd"});
        try gp(buf, a, sf(w), rd(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, sf(w), rn(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, sf(w), rm(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, sf(w), ra(w));
        return;
    }

    // add/sub immediate (bit30 = op is sub, bit29 = S sets flags, bit22 = shift-by-12). When S
    // is set and rd == 31 the alias is cmp / cmn. Rn/Rd 31 otherwise mean sp.
    if (w & 0x1F800000 == 0x11000000) {
        const is_sub = (w >> 30) & 1 != 0;
        const setflags = (w >> 29) & 1 != 0;
        const shift12 = (w >> 22) & 1 != 0;
        const imm: u32 = (w >> 10) & 0xFFF;
        if (setflags and rd(w) == 31) {
            try buf.print(a, "{s} ", .{if (is_sub) "cmp" else "cmn"});
            try sp(buf, a, sf(w), rn(w));
            try buf.print(a, ", #{d}", .{imm});
            if (shift12) try buf.appendSlice(a, ", lsl #12");
            return;
        }
        // mov Rd, Rn (to/from SP) = add Rd, Rn, #0 with no shift.
        if (!setflags and !is_sub and imm == 0 and !shift12 and (rd(w) == 31 or rn(w) == 31)) {
            try buf.appendSlice(a, "mov ");
            try sp(buf, a, sf(w), rd(w));
            try buf.appendSlice(a, ", ");
            try sp(buf, a, sf(w), rn(w));
            return;
        }
        const mnem = if (setflags) (if (is_sub) "subs" else "adds") else (if (is_sub) "sub" else "add");
        try buf.print(a, "{s} ", .{mnem});
        try sp(buf, a, sf(w), rd(w));
        try buf.appendSlice(a, ", ");
        try sp(buf, a, sf(w), rn(w));
        try buf.print(a, ", #{d}", .{imm});
        if (shift12) try buf.appendSlice(a, ", lsl #12");
        return;
    }

    // movn (move wide negated): rendered as `mov Rd, #<~imm>`, matching objdump's alias.
    if (w & 0x7F800000 == 0x12800000) {
        const imm: u64 = (w >> 5) & 0xFFFF;
        const shift: u6 = @intCast(((w >> 21) & 3) * 16);
        const val = ~(imm << shift);
        try buf.appendSlice(a, "mov ");
        try gp(buf, a, sf(w), rd(w));
        try buf.print(a, ", #0x{x}", .{if (sf(w)) val else val & 0xFFFFFFFF});
        return;
    }

    // movz / movk (32/64-bit, with an optional lsl of the 16-bit immediate). objdump aliases
    // movz to `mov Rd, #<imm<<shift>>`, so render the assembled value for movz.
    if (w & 0x7F800000 == 0x52800000 or w & 0x7F800000 == 0x72800000) {
        const keep = (w >> 29) & 1 != 0; // opc 11 -> movk, 10 -> movz
        const imm: u64 = (w >> 5) & 0xFFFF;
        const shift: u6 = @intCast(((w >> 21) & 3) * 16);
        if (!keep) { // movz -> mov Rd, #value
            const val = imm << shift;
            try buf.appendSlice(a, "mov ");
            try gp(buf, a, sf(w), rd(w));
            try buf.print(a, ", #0x{x}", .{if (sf(w)) val else val & 0xFFFFFFFF});
            return;
        }
        try buf.appendSlice(a, "movk ");
        try gp(buf, a, sf(w), rd(w));
        try buf.print(a, ", #{d}", .{imm});
        if (shift != 0) try buf.print(a, ", lsl #{d}", .{shift});
        return;
    }

    // adds/subs (shifted register, shift 0), with cmn/cmp aliases when rd == 31.
    if (w & 0x7FE0FC00 == 0x2B000000 or w & 0x7FE0FC00 == 0x6B000000) {
        const is_sub = (w >> 30) & 1 != 0;
        if (rd(w) == 31) {
            try buf.print(a, "{s} ", .{if (is_sub) "cmp" else "cmn"});
            try gp(buf, a, sf(w), rn(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rm(w));
            return;
        }
        if (is_sub and rn(w) == 31) { // negs Rd, Rm = subs Rd, RZR, Rm
            try buf.appendSlice(a, "negs ");
            try gp(buf, a, sf(w), rd(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rm(w));
            return;
        }
        try buf.print(a, "{s} ", .{if (is_sub) "subs" else "adds"});
        try gp(buf, a, sf(w), rd(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, sf(w), rn(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, sf(w), rm(w));
        return;
    }

    // cset (csinc Rd, zr, zr, invert(cond)). The mask excludes the cond field [12:15].
    if (w & 0x7FFF0FE0 == 0x1A9F07E0) {
        try buf.appendSlice(a, "cset ");
        try gp(buf, a, sf(w), rd(w));
        try buf.print(a, ", {s}", .{condName(@intCast(((w >> 12) & 0xF) ^ 1))});
        return;
    }

    // csel Rd, Rn, Rm, cond.
    if (w & 0x7FE00C00 == 0x1A800000) {
        try buf.appendSlice(a, "csel ");
        try gp(buf, a, sf(w), rd(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, sf(w), rn(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, sf(w), rm(w));
        try buf.print(a, ", {s}", .{condName(@intCast((w >> 12) & 0xF))});
        return;
    }

    // cbz / cbnz Rt, label (bit24 selects nz).
    if (w & 0x7E000000 == 0x34000000) {
        const nz = (w >> 24) & 1 != 0;
        const imm19: i32 = @as(i32, @bitCast(w << 8)) >> 13; // sign-extend bits[5:23]
        try buf.print(a, "{s} ", .{if (nz) "cbnz" else "cbz"});
        try gp(buf, a, sf(w), rd(w));
        try buf.appendSlice(a, ", ");
        try renderTarget(a, buf, ctx, @as(i64, imm19) * 4);
        return;
    }

    // tbz / tbnz Rt, #bit, label (bit24 selects nz). Test-bit index is b5:b40.
    if (w & 0x7E000000 == 0x36000000) {
        const nz = (w >> 24) & 1 != 0;
        const bit: u32 = (((w >> 31) & 1) << 5) | ((w >> 19) & 0x1F);
        const imm14: i32 = (@as(i32, @bitCast(w << 13)) >> 18) * 4; // bits[18:5], scaled
        try buf.print(a, "{s} ", .{if (nz) "tbnz" else "tbz"});
        try gp(buf, a, (w >> 31) & 1 != 0, rd(w));
        try buf.print(a, ", #{d}, ", .{bit});
        try renderTarget(a, buf, ctx, imm14);
        return;
    }

    // b.cond label (bit4 == 0 distinguishes it from the consistent-conditional-branch forms).
    if (w & 0xFF000010 == 0x54000000) {
        const imm19: i32 = (@as(i32, @bitCast(w << 8)) >> 13) * 4;
        try buf.print(a, "b.{s} ", .{condName(@intCast(w & 0xF))});
        try renderTarget(a, buf, ctx, imm19);
        return;
    }

    // b / bl label (imm26 << 2).
    if (w & 0xFC000000 == 0x14000000 or w & 0xFC000000 == 0x94000000) {
        const link = (w >> 31) & 1 != 0;
        const imm26: i32 = (@as(i32, @bitCast(w << 6)) >> 6) << 2;
        try buf.print(a, "{s} ", .{if (link) "bl" else "b"});
        try renderTarget(a, buf, ctx, imm26);
        return;
    }

    // stp/ldp pair, pre/post-index (64-bit).
    if (w & 0xFFC00000 == 0xA9800000 or w & 0xFFC00000 == 0xA8C00000) {
        const load = (w >> 22) & 1 != 0;
        const imm7: i32 = (@as(i32, @bitCast(w << 10)) >> 25) * 8; // bits[15:21], scaled by 8
        try buf.print(a, "{s} ", .{if (load) "ldp" else "stp"});
        try gp(buf, a, true, rd(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, true, ra(w)); // rt2 is bits[10:15]
        try buf.appendSlice(a, ", [");
        try sp(buf, a, true, rn(w));
        if (load) { // post-index: `[base], #imm`
            try buf.print(a, "], #{d}", .{imm7});
        } else { // pre-index: `[base, #imm]!`
            try buf.print(a, ", #{d}]!", .{imm7});
        }
        return;
    }

    // Logical immediate: and/orr/eor/ands, with mov (orr from zr) and tst (ands to zr) aliases.
    if (w & 0x1F800000 == 0x12000000) {
        const opc = (w >> 29) & 3;
        const nn: u1 = @intCast((w >> 22) & 1);
        const immr: u6 = @intCast((w >> 16) & 0x3F);
        const imms: u6 = @intCast((w >> 10) & 0x3F);
        if (decodeBitmask(sf(w), nn, imms, immr)) |val| {
            if (opc == 1 and rn(w) == 31) { // mov Rd, #imm
                try buf.appendSlice(a, "mov ");
                try gp(buf, a, sf(w), rd(w));
                try buf.print(a, ", #0x{x}", .{val});
                return;
            }
            if (opc == 3 and rd(w) == 31) { // tst Rn, #imm
                try buf.appendSlice(a, "tst ");
                try gp(buf, a, sf(w), rn(w));
                try buf.print(a, ", #0x{x}", .{val});
                return;
            }
            try buf.print(a, "{s} ", .{switch (opc) {
                0 => "and",
                1 => "orr",
                2 => "eor",
                else => "ands",
            }});
            try sp(buf, a, sf(w), rd(w)); // and/orr/eor immediate can target sp
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rn(w));
            try buf.print(a, ", #0x{x}", .{val});
            return;
        }
    }

    // extr Rd, Rn, Rm, #lsb (ror when rn == rm). bit23 == 1 separates it from the bitfield ops.
    if (w & 0x7F800000 == 0x13800000) {
        const lsb: u32 = (w >> 10) & 0x3F;
        if (rn(w) == rm(w)) {
            try buf.appendSlice(a, "ror ");
            try gp(buf, a, sf(w), rd(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rn(w));
            try buf.print(a, ", #{d}", .{lsb});
        } else {
            try buf.appendSlice(a, "extr ");
            try gp(buf, a, sf(w), rd(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rn(w));
            try buf.appendSlice(a, ", ");
            try gp(buf, a, sf(w), rm(w));
            try buf.print(a, ", #{d}", .{lsb});
        }
        return;
    }

    // STUR/LDUR: unscaled signed 9-bit offset (integer). size 00=b 01=h 10=w 11=x; opc picks
    // store / load / sign-extending load.
    if (w & 0x3F200C00 == 0x38000000) {
        const size = w >> 30;
        const opc = (w >> 22) & 3;
        const imm9: i32 = @as(i32, @bitCast(w << 11)) >> 23; // bits[20:12], sign-extended
        const load = opc != 0;
        const signed = opc >= 2;
        const to_x = size == 3 or opc == 2; // x-sized access, or sign-extend to a 64-bit reg
        const suffix: []const u8 = switch (size) {
            0 => "b",
            1 => "h",
            2 => if (signed) "w" else "",
            else => "",
        };
        try buf.print(a, "{s}{s}{s} ", .{ if (load) "ldur" else "stur", if (signed) "s" else "", suffix });
        try gp(buf, a, to_x, rd(w));
        try buf.appendSlice(a, ", [");
        try sp(buf, a, true, rn(w));
        if (imm9 != 0) {
            try buf.print(a, ", #{d}]", .{imm9});
        } else {
            try buf.appendSlice(a, "]");
        }
        return;
    }

    // STUR/LDUR (FP/SIMD), unscaled signed 9-bit offset. Width is b/h/s/d by size, or q (128-bit)
    // when opc bit1 is set.
    if (w & 0x3F200C00 == 0x3C000000) {
        const size = w >> 30;
        const opc = (w >> 22) & 3;
        const imm9: i32 = @as(i32, @bitCast(w << 11)) >> 23;
        const load = opc & 1 != 0;
        const c: u8 = if (opc & 2 != 0) 'q' else switch (size) {
            0 => 'b',
            1 => 'h',
            2 => 's',
            else => 'd',
        };
        try buf.print(a, "{s} {c}{d}, [", .{ if (load) "ldur" else "stur", c, rd(w) });
        try sp(buf, a, true, rn(w));
        if (imm9 != 0) {
            try buf.print(a, ", #{d}]", .{imm9});
        } else {
            try buf.appendSlice(a, "]");
        }
        return;
    }

    // umulh / smulh (unsigned/signed 64x64 -> high 64). Ra field is 31.
    if (w & 0xFFE0FC00 == 0x9BC07C00 or w & 0xFFE0FC00 == 0x9B407C00) {
        const uns = (w >> 23) & 1 != 0;
        try buf.print(a, "{s} x{d}, x{d}, x{d}", .{ if (uns) "umulh" else "smulh", rd(w), rn(w), rm(w) });
        return;
    }

    // Bitfield move (SBFM/UBFM/BFM) with the usual aliases.
    if (w & 0x1F800000 == 0x13000000 and (w >> 29) & 3 != 3) {
        try bitfield(a, buf, w);
        return;
    }

    // stp/ldp signed offset (bit31 selects 32/64-bit, bit22 selects load).
    if (w & 0x7FC00000 == 0x29000000 or w & 0x7FC00000 == 0x29400000) {
        const x64 = (w >> 31) & 1 != 0;
        const load = (w >> 22) & 1 != 0;
        const scale: u5 = if (x64) 3 else 2;
        const imm7: i32 = (@as(i32, @bitCast(w << 10)) >> 25) << scale; // bits[21:15], scaled
        try buf.print(a, "{s} ", .{if (load) "ldp" else "stp"});
        try gp(buf, a, x64, rd(w));
        try buf.appendSlice(a, ", ");
        try gp(buf, a, x64, ra(w)); // rt2 is bits[14:10]
        try buf.appendSlice(a, ", [");
        try sp(buf, a, true, rn(w));
        if (imm7 != 0) {
            try buf.print(a, ", #{d}]", .{imm7});
        } else {
            try buf.appendSlice(a, "]");
        }
        return;
    }

    // adr / adrp Rd, imm (PC-relative address; adrp is page-scaled).
    if (w & 0x1F000000 == 0x10000000) {
        const page = (w >> 31) & 1 != 0;
        const immlo: i64 = (w >> 29) & 3;
        const immhi: i64 = @as(i32, @bitCast((w & 0x00FFFFE0) << 8)) >> 13; // bits[23:5], sign-extended
        var imm: i64 = (immhi << 2) | immlo;
        if (page) imm <<= 12;
        try buf.print(a, "{s} ", .{if (page) "adrp" else "adr"});
        try gp(buf, a, true, rd(w));
        if (ctx) |c| { // absolute target: adrp is page-based, adr is byte-based
            const pc_base: u64 = if (page) c.pc & ~@as(u64, 0xFFF) else c.pc;
            const abs: u64 = pc_base +% @as(u64, @bitCast(imm));
            try buf.print(a, ", 0x{x}", .{abs});
            try annotateSym(a, buf, c.syms, abs);
        } else if (imm < 0) {
            try buf.print(a, ", .-0x{x}", .{@as(u64, @intCast(-imm))});
        } else {
            try buf.print(a, ", .+0x{x}", .{@as(u64, @intCast(imm))});
        }
        return;
    }

    if (try memImm(a, buf, w)) return;
    if (try fpDecode(a, buf, w)) return;
    if (try vecDecode(a, buf, w)) return;

    // Unknown: emit the raw word so nothing is silently lost.
    try buf.print(a, ".word 0x{x:0>8}", .{w});
}

/// Decode a logical-immediate `(N:immr:imms)` to its 32/64-bit value, or null if the encoding is
/// reserved. This is ARM's `DecodeBitMasks` (immediate result only): a run of `s+1` ones,
/// rotated right by `r` within an `esize`-bit element, replicated across the register.
fn decodeBitmask(x64: bool, n: u1, imms: u6, immr: u6) ?u64 {
    const width: u32 = if (x64) 64 else 32;
    // len = position of the highest set bit of (N : NOT(imms)), a 7-bit quantity.
    const combined: u7 = (@as(u7, n) << 6) | (~imms & 0x3F);
    if (combined == 0) return null;
    var len: u32 = 6;
    while (len > 0 and (combined & (@as(u7, 1) << @intCast(len))) == 0) : (len -= 1) {}
    if (len == 0) return null; // element size 1 is not a valid logical immediate
    const esize: u32 = @as(u32, 1) << @intCast(len);
    if (esize > width) return null; // N == 1 is only valid for the 64-bit form
    const levels: u32 = esize - 1;
    const s: u32 = imms & levels;
    const r: u32 = immr & levels;
    if (s == levels) return null; // an all-ones element is reserved
    const elem: u64 = (@as(u64, 1) << @intCast(s + 1)) - 1;
    const rotated = rorInEsize(elem, r, esize);
    var result: u64 = 0;
    var pos: u32 = 0;
    while (pos < width) : (pos += esize) result |= rotated << @intCast(pos);
    return if (x64) result else result & 0xFFFFFFFF;
}

/// Rotate the low `esize` bits of `v` right by `r`.
fn rorInEsize(v: u64, r: u32, esize: u32) u64 {
    const mask: u64 = if (esize == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(esize)) - 1;
    const rr = r % esize;
    if (rr == 0) return v & mask;
    return ((v >> @intCast(rr)) | (v << @intCast(esize - rr))) & mask;
}

/// Scaled-unsigned-offset loads/stores (integer and FP). Returns whether it matched.
fn memImm(a: std.mem.Allocator, buf: *std.ArrayList(u8), w: u32) !bool {
    const M = struct { match: u32, mnem: []const u8, x64: bool, scale: u5, fp: bool, dbl: bool };
    const table = [_]M{
        .{ .match = 0xF9000000, .mnem = "str", .x64 = true, .scale = 3, .fp = false, .dbl = false },
        .{ .match = 0xF9400000, .mnem = "ldr", .x64 = true, .scale = 3, .fp = false, .dbl = false },
        .{ .match = 0xB9000000, .mnem = "str", .x64 = false, .scale = 2, .fp = false, .dbl = false },
        .{ .match = 0xB9400000, .mnem = "ldr", .x64 = false, .scale = 2, .fp = false, .dbl = false },
        .{ .match = 0x39000000, .mnem = "strb", .x64 = false, .scale = 0, .fp = false, .dbl = false },
        .{ .match = 0x39400000, .mnem = "ldrb", .x64 = false, .scale = 0, .fp = false, .dbl = false },
        .{ .match = 0x39C00000, .mnem = "ldrsb", .x64 = false, .scale = 0, .fp = false, .dbl = false },
        .{ .match = 0x79000000, .mnem = "strh", .x64 = false, .scale = 1, .fp = false, .dbl = false },
        .{ .match = 0x79400000, .mnem = "ldrh", .x64 = false, .scale = 1, .fp = false, .dbl = false },
        .{ .match = 0x79C00000, .mnem = "ldrsh", .x64 = false, .scale = 1, .fp = false, .dbl = false },
        .{ .match = 0xBD000000, .mnem = "str", .x64 = false, .scale = 2, .fp = true, .dbl = false },
        .{ .match = 0xBD400000, .mnem = "ldr", .x64 = false, .scale = 2, .fp = true, .dbl = false },
        .{ .match = 0xFD000000, .mnem = "str", .x64 = false, .scale = 3, .fp = true, .dbl = true },
        .{ .match = 0xFD400000, .mnem = "ldr", .x64 = false, .scale = 3, .fp = true, .dbl = true },
        .{ .match = 0x3D800000, .mnem = "str", .x64 = false, .scale = 4, .fp = true, .dbl = false }, // Q
        .{ .match = 0x3DC00000, .mnem = "ldr", .x64 = false, .scale = 4, .fp = true, .dbl = false }, // Q
        .{ .match = 0x7D000000, .mnem = "str", .x64 = false, .scale = 1, .fp = true, .dbl = false }, // H (f16)
        .{ .match = 0x7D400000, .mnem = "ldr", .x64 = false, .scale = 1, .fp = true, .dbl = false }, // H (f16)
    };
    for (table) |m| {
        if (w & 0xFFC00000 != m.match) continue;
        const off: u32 = ((w >> 10) & 0xFFF) << m.scale;
        try buf.print(a, "{s} ", .{m.mnem});
        if (m.fp) {
            // The FP view: Q (128-bit), D (64-bit), H (16-bit, the f16 boundary form), else S.
            const c: u8 = if (m.scale == 4) 'q' else if (m.dbl) 'd' else if (m.scale == 1) 'h' else 's';
            try buf.print(a, "{c}{d}", .{ c, rd(w) });
        } else {
            try gp(buf, a, m.x64, rd(w));
        }
        try buf.appendSlice(a, ", [");
        try sp(buf, a, true, rn(w));
        if (off != 0) {
            try buf.print(a, ", #{d}]", .{off});
        } else {
            try buf.appendSlice(a, "]");
        }
        return true;
    }
    // strb/ldrb/etc with a zero-offset `[xn]` form already covered (off field 0).
    return false;
}

/// Scalar floating-point instructions. Returns whether it matched.
fn fpDecode(a: std.mem.Allocator, buf: *std.ArrayList(u8), w: u32) !bool {
    const dbl = (w >> 22) & 1 != 0;
    const fc: u8 = if (dbl) 'd' else 's';
    // Three-register scalar FP arithmetic.
    const FP3 = struct { match: u32, mnem: []const u8 };
    const fp3 = [_]FP3{
        .{ .match = 0x1E202800, .mnem = "fadd" },
        .{ .match = 0x1E203800, .mnem = "fsub" },
        .{ .match = 0x1E200800, .mnem = "fmul" },
        .{ .match = 0x1E201800, .mnem = "fdiv" },
    };
    for (fp3) |e| {
        if (w & 0xFFA0FC00 == e.match) {
            try buf.print(a, "{s} {c}{d}, {c}{d}, {c}{d}", .{ e.mnem, fc, rd(w), fc, rn(w), fc, rm(w) });
            return true;
        }
    }
    // Floating-point data-processing (3-source): Rd = op(Rn*Rm, Ra). Mask excludes ftype
    // (bit 22, read via `dbl` above) and the four 5-bit register fields.
    const FP3A = struct { match: u32, mnem: []const u8 };
    const fp3a = [_]FP3A{
        .{ .match = 0x1F000000, .mnem = "fmadd" },
        .{ .match = 0x1F008000, .mnem = "fmsub" },
        .{ .match = 0x1F200000, .mnem = "fnmadd" },
        .{ .match = 0x1F208000, .mnem = "fnmsub" },
    };
    for (fp3a) |e| {
        if (w & 0xFF208000 == e.match) {
            try buf.print(a, "{s} {c}{d}, {c}{d}, {c}{d}, {c}{d}", .{ e.mnem, fc, rd(w), fc, rn(w), fc, rm(w), fc, ra(w) });
            return true;
        }
    }
    if (w & 0xFFA00C00 == 0x1E200C00) { // fcsel (mask excludes the cond field [12:15])
        try buf.print(a, "fcsel {c}{d}, {c}{d}, {c}{d}, {s}", .{ fc, rd(w), fc, rn(w), fc, rm(w), condName(@intCast((w >> 12) & 0xF)) });
        return true;
    }
    if (w & 0xFFA0FC1F == 0x1E202000) { // fcmp
        try buf.print(a, "fcmp {c}{d}, {c}{d}", .{ fc, rn(w), fc, rm(w) });
        return true;
    }
    if (w & 0xFFFFFC00 == 0x1E604000) { // fmov (reg, 64-bit view)
        try buf.print(a, "fmov d{d}, d{d}", .{ rd(w), rn(w) });
        return true;
    }
    // fmov between a general register and an FP register (both directions, s/d).
    const FMOV = struct { match: u32, to_gpr: bool, x64: bool };
    const fmov = [_]FMOV{
        .{ .match = 0x1E270000, .to_gpr = false, .x64 = false },
        .{ .match = 0x9E670000, .to_gpr = false, .x64 = true },
        .{ .match = 0x1E260000, .to_gpr = true, .x64 = false },
        .{ .match = 0x9E660000, .to_gpr = true, .x64 = true },
    };
    for (fmov) |e| {
        if (w & 0xFFFFFC00 == e.match) {
            const f: u8 = if (e.x64) 'd' else 's';
            try buf.appendSlice(a, "fmov ");
            if (e.to_gpr) {
                try gp(buf, a, e.x64, rd(w));
                try buf.print(a, ", {c}{d}", .{ f, rn(w) });
            } else {
                try buf.print(a, "{c}{d}, ", .{ f, rd(w) });
                try gp(buf, a, e.x64, rn(w));
            }
            return true;
        }
    }
    // Two-register scalar FP ops (sqrt, rounding, conversions).
    const FP2 = struct { match: u32, mnem: []const u8 };
    const fp2 = [_]FP2{
        .{ .match = 0x1E21C000, .mnem = "fsqrt" },
        .{ .match = 0x1E244000, .mnem = "frintn" },
        .{ .match = 0x1E24C000, .mnem = "frintp" },
        .{ .match = 0x1E254000, .mnem = "frintm" },
        .{ .match = 0x1E25C000, .mnem = "frintz" },
    };
    for (fp2) |e| {
        if (w & 0xFFBFFC00 == e.match) {
            try buf.print(a, "{s} {c}{d}, {c}{d}", .{ e.mnem, fc, rd(w), fc, rn(w) });
            return true;
        }
    }
    if (w & 0xFFBFFC00 == 0x1E220000) { // scvtf sd, wn
        try buf.print(a, "scvtf {c}{d}, ", .{ fc, rd(w) });
        try gp(buf, a, false, rn(w));
        return true;
    }
    if (w & 0xFFBFFC00 == 0x1E230000) { // ucvtf
        try buf.print(a, "ucvtf {c}{d}, ", .{ fc, rd(w) });
        try gp(buf, a, false, rn(w));
        return true;
    }
    if (w & 0xFFBFFC00 == 0x1E380000) { // fcvtzs wd, sn
        try buf.appendSlice(a, "fcvtzs ");
        try gp(buf, a, false, rd(w));
        try buf.print(a, ", {c}{d}", .{ fc, rn(w) });
        return true;
    }
    if (w & 0xFFBFFC00 == 0x1E390000) { // fcvtzu
        try buf.appendSlice(a, "fcvtzu ");
        try gp(buf, a, false, rd(w));
        try buf.print(a, ", {c}{d}", .{ fc, rn(w) });
        return true;
    }
    if (w & 0xFFFFFC00 == 0x1E22C000) { // fcvt d, s (s->d)
        try buf.print(a, "fcvt d{d}, s{d}", .{ rd(w), rn(w) });
        return true;
    }
    if (w & 0xFFFFFC00 == 0x1E624000) { // fcvt s, d (d->s)
        try buf.print(a, "fcvt s{d}, d{d}", .{ rd(w), rn(w) });
        return true;
    }
    if (w & 0xFFFFFC00 == 0x1EE24000) { // fcvt s, h (h->s, f16 widen)
        try buf.print(a, "fcvt s{d}, h{d}", .{ rd(w), rn(w) });
        return true;
    }
    if (w & 0xFFFFFC00 == 0x1E23C000) { // fcvt h, s (s->h, f16 narrow)
        try buf.print(a, "fcvt h{d}, s{d}", .{ rd(w), rn(w) });
        return true;
    }
    if (w & 0xFFFFFC00 == 0x1E63C000) { // fcvt h, d (d->h, f16 narrow, single round)
        try buf.print(a, "fcvt h{d}, d{d}", .{ rd(w), rn(w) });
        return true;
    }
    return false;
}

/// NEON (128-bit vector) instructions. Returns whether it matched.
fn vecDecode(a: std.mem.Allocator, buf: *std.ArrayList(u8), w: u32) !bool {
    // Three-register 4S / 16B forms.
    const V3 = struct { match: u32, mnem: []const u8, b16: bool };
    const v3 = [_]V3{
        .{ .match = 0x4E20D400, .mnem = "fadd", .b16 = false },
        .{ .match = 0x4EA0D400, .mnem = "fsub", .b16 = false },
        .{ .match = 0x6E20DC00, .mnem = "fmul", .b16 = false },
        .{ .match = 0x6E20FC00, .mnem = "fdiv", .b16 = false },
        .{ .match = 0x4EA0F400, .mnem = "fmin", .b16 = false },
        .{ .match = 0x4E20F400, .mnem = "fmax", .b16 = false },
        .{ .match = 0x4E20E400, .mnem = "fcmeq", .b16 = false },
        .{ .match = 0x6EA0E400, .mnem = "fcmgt", .b16 = false },
        .{ .match = 0x6E20E400, .mnem = "fcmge", .b16 = false },
        .{ .match = 0x6E601C00, .mnem = "bsl", .b16 = true },
        .{ .match = 0x4E20CC00, .mnem = "fmla", .b16 = false },
        .{ .match = 0x4EA0CC00, .mnem = "fmls", .b16 = false },
    };
    for (v3) |e| {
        if (w == e.match | (@as(u32, rm(w)) << 16) | (@as(u32, rn(w)) << 5) | rd(w)) {
            const s = if (e.b16) "16b" else "4s";
            try buf.print(a, "{s} v{d}.{s}, v{d}.{s}, v{d}.{s}", .{ e.mnem, rd(w), s, rn(w), s, rm(w), s });
            return true;
        }
    }
    // Two-register vector forms.
    if (w & 0xFFFFFC00 == 0x6EA0F800) return vec2(a, buf, w, "fneg", "4s");
    if (w & 0xFFFFFC00 == 0x6EA1F800) return vec2(a, buf, w, "fsqrt", "4s");
    if (w & 0xFFFFFC00 == 0x6E205800) return vec2(a, buf, w, "mvn", "16b");
    // mov vd.16b, vn.16b is orr with rm == rn.
    if (w & 0xFFE0FC00 == 0x4EA01C00 and rm(w) == rn(w)) {
        try buf.print(a, "mov v{d}.16b, v{d}.16b", .{ rd(w), rn(w) });
        return true;
    }
    if (w & 0xFFFFFC00 == 0x4E040C00) { // dup Vd.4S, Wn
        try buf.print(a, "dup v{d}.4s, ", .{rd(w)});
        try gp(buf, a, false, rn(w));
        return true;
    }
    // dup Vd.4S, Vn.s[i] / dup Sd, Vn.s[i] / ins Vd.s[i], Vn.s[0]: imm5 in bits[16:20].
    const imm5 = (w >> 16) & 0x1F;
    const lane = (imm5 >> 3) & 3;
    if (w & 0xFFE0FC00 == 0x4E000400) {
        try buf.print(a, "dup v{d}.4s, v{d}.s[{d}]", .{ rd(w), rn(w), lane });
        return true;
    }
    if (w & 0xFFE0FC00 == 0x5E000400) {
        try buf.print(a, "dup s{d}, v{d}.s[{d}]", .{ rd(w), rn(w), lane });
        return true;
    }
    if (w & 0xFFE08C00 == 0x6E000400) {
        try buf.print(a, "ins v{d}.s[{d}], v{d}.s[0]", .{ rd(w), lane, rn(w) });
        return true;
    }
    return false;
}

fn vec2(a: std.mem.Allocator, buf: *std.ArrayList(u8), w: u32, mnem: []const u8, suffix: []const u8) !bool {
    try buf.print(a, "{s} v{d}.{s}, v{d}.{s}", .{ mnem, rd(w), suffix, rn(w), suffix });
    return true;
}

/// Render a bitfield-move (SBFM/UBFM/BFM) using the alias a debugger expects (uxt*/sxt*/lsl/lsr/
/// asr/ubfx/sbfx/bfi/bfxil). `opc` selects the family; `immr`/`imms` drive the alias choice.
fn bitfield(a: std.mem.Allocator, buf: *std.ArrayList(u8), w: u32) !void {
    const opc = (w >> 29) & 3;
    const x64 = sf(w);
    const width: u32 = if (x64) 64 else 32;
    const immr: u32 = (w >> 16) & 0x3F;
    const imms: u32 = (w >> 10) & 0x3F;

    // Rd, Rn at the register width; a helper for the common `mnem Rd, Rn, #a, #b` shape.
    const two = struct {
        fn go(al: std.mem.Allocator, b: *std.ArrayList(u8), mnem: []const u8, xw: bool, d: u5, n: u5) !void {
            try b.print(al, "{s} ", .{mnem});
            try gp(b, al, xw, d);
            try b.appendSlice(al, ", ");
            try gp(b, al, xw, n);
        }
    }.go;

    switch (opc) {
        2 => { // UBFM
            if (imms + 1 == immr and imms != width - 1) { // lsl #(width-1-imms)
                try two(a, buf, "lsl", x64, rd(w), rn(w));
                return buf.print(a, ", #{d}", .{width - 1 - imms});
            }
            if (imms == width - 1) { // lsr #immr
                try two(a, buf, "lsr", x64, rd(w), rn(w));
                return buf.print(a, ", #{d}", .{immr});
            }
            if (immr == 0 and imms == 7 and !x64) return two(a, buf, "uxtb", false, rd(w), rn(w));
            if (immr == 0 and imms == 15 and !x64) return two(a, buf, "uxth", false, rd(w), rn(w));
            if (imms < immr) { // ubfiz Rd, Rn, #(width-immr), #(imms+1)
                try two(a, buf, "ubfiz", x64, rd(w), rn(w));
                return buf.print(a, ", #{d}, #{d}", .{ width - immr, imms + 1 });
            }
            try two(a, buf, "ubfx", x64, rd(w), rn(w)); // ubfx Rd, Rn, #immr, #(imms-immr+1)
            return buf.print(a, ", #{d}, #{d}", .{ immr, imms - immr + 1 });
        },
        0 => { // SBFM
            if (imms == width - 1) { // asr #immr
                try two(a, buf, "asr", x64, rd(w), rn(w));
                return buf.print(a, ", #{d}", .{immr});
            }
            if (immr == 0 and imms == 7) return two(a, buf, "sxtb", x64, rd(w), rn(w));
            if (immr == 0 and imms == 15) return two(a, buf, "sxth", x64, rd(w), rn(w));
            if (immr == 0 and imms == 31 and x64) { // sxtw Xd, Wn
                try buf.appendSlice(a, "sxtw ");
                try gp(buf, a, true, rd(w));
                try buf.appendSlice(a, ", ");
                return gp(buf, a, false, rn(w));
            }
            if (imms < immr) { // sbfiz
                try two(a, buf, "sbfiz", x64, rd(w), rn(w));
                return buf.print(a, ", #{d}, #{d}", .{ width - immr, imms + 1 });
            }
            try two(a, buf, "sbfx", x64, rd(w), rn(w));
            return buf.print(a, ", #{d}, #{d}", .{ immr, imms - immr + 1 });
        },
        else => { // BFM
            if (imms < immr) { // bfi Rd, Rn, #(width-immr), #(imms+1)
                try two(a, buf, "bfi", x64, rd(w), rn(w));
                return buf.print(a, ", #{d}, #{d}", .{ width - immr, imms + 1 });
            }
            try two(a, buf, "bfxil", x64, rd(w), rn(w)); // bfxil Rd, Rn, #immr, #(imms-immr+1)
            return buf.print(a, ", #{d}, #{d}", .{ immr, imms - immr + 1 });
        },
    }
}

const Reg = encode.Reg;

fn expectOne(word: u32, expected: []const u8) !void {
    const s = try one(std.testing.allocator, word);
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(expected, s);
}

test "formatWithLines interleaves source-line markers" {
    const a = std.testing.allocator;
    const code = [_]u32{ encode.add(.x0, .x1, .x2), encode.mul(.x0, .x0, .x1), encode.ret() };
    // line 2 covers the add; line 3 covers the mul (and the ret rides the last line).
    const lines = [_]SourceLine{ .{ .offset = 0, .line = 2 }, .{ .offset = 4, .line = 3 } };
    const text = try formatWithLines(a, &code, &lines);
    defer a.free(text);
    try std.testing.expectEqualStrings(
        \\; line 2
        \\0000: 0b020020  add w0, w1, w2
        \\; line 3
        \\0004: 1b017c00  mul w0, w0, w1
        \\0008: d65f03c0  ret
        \\
    , text);
}

test "round-trips integer data processing" {
    try expectOne(encode.ret(), "ret");
    try expectOne(encode.add(.x0, .x1, .x2), "add w0, w1, w2");
    try expectOne(encode.sub(.x3, .x4, .x5), "sub w3, w4, w5");
    try expectOne(encode.add64(.x0, .x1, .x2), "add x0, x1, x2");
    try expectOne(encode.mul(.x0, .x1, .x2), "mul w0, w1, w2");
    try expectOne(encode.msub(.x0, .x1, .x2, .x3), "msub w0, w1, w2, w3");
    try expectOne(encode.sdiv(.x7, .x8, .x9), "sdiv w7, w8, w9");
    try expectOne(encode.lslv(.x1, .x2, .x3), "lsl w1, w2, w3");
    try expectOne(encode.orr(.x1, .x2, .x3), "orr w1, w2, w3");
    try expectOne(encode.mov(.x5, .x6), "mov x5, x6");
    try expectOne(encode.addImm(.x0, .x1, 5), "add w0, w1, #5");
    try expectOne(encode.subImm64(.zr, .zr, 16), "sub sp, sp, #16");
    try expectOne(encode.addImm64Shift(.x0, .x1, 2), "add x0, x1, #2, lsl #12");
    try expectOne(encode.movz(.x0, 42, 0), "mov w0, #0x2a");
    try expectOne(encode.movk(.x0, 7, 1), "movk w0, #7, lsl #16");
    try expectOne(encode.movz64(.x1, 9, 2), "mov x1, #0x900000000");
}

test "round-trips flags, selects, and branches" {
    try expectOne(encode.cmp(.x1, .x2), "cmp w1, w2");
    try expectOne(encode.cset(.x0, .lt), "cset w0, lt");
    try expectOne(encode.csel(.x0, .x1, .x2, .gt), "csel w0, w1, w2, gt");
    try expectOne(encode.blr(.x8), "blr x8");
    try expectOne(encode.b(16), "b .+16");
    try expectOne(encode.bl(-8), "bl .-8");
    try expectOne(encode.cbnz(.x3, 12), "cbnz w3, .+12");
    try expectOne(encode.svc(0), "svc #0");
}

test "round-trips memory access" {
    try expectOne(encode.stpPre(.x29, .x30, .zr, -16), "stp x29, x30, [sp, #-16]!");
    try expectOne(encode.ldpPost(.x29, .x30, .zr, 16), "ldp x29, x30, [sp], #16");
    try expectOne(encode.strOff(.x0, .x1, 8), "str x0, [x1, #8]");
    try expectOne(encode.ldrOff(.x0, .x1, 0), "ldr x0, [x1]");
    try expectOne(encode.strW(.x0, .x1, 4), "str w0, [x1, #4]");
    try expectOne(encode.ldrb(.x2, .x3), "ldrb w2, [x3]");
    try expectOne(encode.ldrsh(.x2, .x3), "ldrsh w2, [x3]");
    try expectOne(encode.ldrQ(.x0, .x1, 0), "ldr q0, [x1]");
}

test "decodes common system-binary instructions (verified against objdump)" {
    // These encodings are not emitted by encode.zig; the expected text is what binutils
    // objdump prints for the same words, so we recognize real compiler output.
    try expectOne(0xeb090108, "subs x8, x8, x9");
    try expectOne(0x6b090108, "subs w8, w8, w9");
    try expectOne(0xab090108, "adds x8, x8, x9");
    try expectOne(0xeb08012a, "subs x10, x9, x8");
    try expectOne(0x12003c08, "and w8, w0, #0xffff");
    try expectOne(0x12000108, "and w8, w8, #0x1");
    try expectOne(0x12000908, "and w8, w8, #0x7");
    try expectOne(0x12000529, "and w9, w9, #0x3");
    try expectOne(0x36000088, "tbz w8, #0, .+16");
    try expectOne(0x340000e8, "cbz w8, .+28");
    try expectOne(0x540000c3, "b.cc .+24");
    try expectOne(0x54000063, "b.cc .+12");
    try expectOne(0xb1000508, "adds x8, x8, #1");
    try expectOne(0xf1000508, "subs x8, x8, #1");
    try expectOne(0x71000508, "subs w8, w8, #1");
    try expectOne(0x93ccf58c, "ror x12, x12, #61");
    try expectOne(0xf85f83a8, "ldur x8, [x29, #-8]");
    try expectOne(0xf81f03a8, "stur x8, [x29, #-16]");
    try expectOne(0xf81403a9, "stur x9, [x29, #-192]");
    try expectOne(0x785f83a8, "ldurh w8, [x29, #-8]");
    try expectOne(0xa9417bfd, "ldp x29, x30, [sp, #16]");
    try expectOne(0xa9027bfd, "stp x29, x30, [sp, #32]");
}

test "formatModule annotates local branch targets with <sym+offset>" {
    const a = std.testing.allocator;
    // fn f at word 0: a backward branch to itself (+0) and a forward branch into g.
    // fn g at word 3.
    const code = [_]u32{
        encode.b(8), // 0: b .+8  -> word 2 (f+0x8)
        encode.ret(), // 1
        encode.b(8), // 2: b .+8  -> word 4 (g+0x4)
        encode.ret(), // 3: g
        encode.ret(), // 4
    };
    const syms = [_]Sym{ .{ .name = "f", .word = 0 }, .{ .name = "g", .word = 3 } };
    const text = try formatModule(a, &code, &syms);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "b .+8  <f+0x8>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "b .+8  <g+0x4>") != null);
}

test "formatElf resolves absolute addresses, branch and adrp targets to symbols" {
    const a = std.testing.allocator;
    const base: u64 = 0x1000;
    const code = [_]u32{
        encode.bl(8), // 0x1000: bl 0x1008 <callee>
        0xB0000000, // 0x1004: adrp x0, 0x2000 <gv>  (page +1)
        encode.ret(), // 0x1008: callee
    };
    const syms = [_]AddrSym{
        .{ .name = "start", .addr = 0x1000, .size = 8, .is_func = true },
        .{ .name = "callee", .addr = 0x1008, .size = 4, .is_func = true },
        .{ .name = "gv", .addr = 0x2000, .size = 16, .is_func = false },
    };
    const text = try formatElf(a, &code, base, &syms);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "0000000000001000 <start>:") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "0000000000001008 <callee>:") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "bl 0x1008 <callee>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "adrp x0, 0x2000 <gv>") != null);
}

test "decodes bitfield, movn, mulh, and SIMD unscaled loads (verified against objdump)" {
    // Bitfield-move aliases.
    try expectOne(0x53003d08, "uxth w8, w8");
    try expectOne(0x53001d08, "uxtb w8, w8");
    try expectOne(0x93407c00, "sxtw x0, w0");
    try expectOne(0x13001d08, "sxtb w8, w8");
    try expectOne(0x13003d08, "sxth w8, w8");
    try expectOne(0x13037d08, "asr w8, w8, #3");
    try expectOne(0x53047d08, "lsr w8, w8, #4");
    try expectOne(0x531f7908, "lsl w8, w8, #1");
    try expectOne(0xd3401128, "ubfx x8, x9, #0, #5");
    try expectOne(0x33000828, "bfxil w8, w1, #0, #3");
    // movn (mov negated), mulh, and FP/SIMD unscaled loads/stores.
    try expectOne(0x92800aa1, "mov x1, #0xffffffffffffffaa");
    try expectOne(0x12800aa1, "mov w1, #0xffffffaa");
    try expectOne(0x9bca7d28, "umulh x8, x9, x10");
    try expectOne(0x9b4a7d28, "smulh x8, x9, x10");
    try expectOne(0x93407d28, "sxtw x8, w9"); // sbfm x8, x9, #0, #31 alias, cross-checks the encoder
    try expectOne(0x3cdf03a0, "ldur q0, [x29, #-16]");
    try expectOne(0x3c9e03a0, "stur q0, [x29, #-32]");
    try expectOne(0xfc5f03a0, "ldur d0, [x29, #-16]");
    try expectOne(0xbc5f03a0, "ldur s0, [x29, #-16]");
}

test "round-trips scalar and vector floating point" {
    try expectOne(encode.fadd(.x0, .x1, .x2, .single), "fadd s0, s1, s2");
    try expectOne(encode.fdiv(.x0, .x1, .x2, .double), "fdiv d0, d1, d2");
    try expectOne(encode.fcmp(.x1, .x2, .single), "fcmp s1, s2");
    try expectOne(encode.fcsel(.x0, .x1, .x2, .mi, .double), "fcsel d0, d1, d2, mi");
    try expectOne(encode.fsqrt(.x0, .x1, false), "fsqrt s0, s1");
    try expectOne(encode.fmovFromGpr(.x0, .x1, false), "fmov s0, w1");
    try expectOne(encode.fmovToGpr(.x0, .x1, true), "fmov x0, d1");
    try expectOne(encode.cvtIntToFloat(.x0, .x1, .single, true), "scvtf s0, w1");
    try expectOne(encode.cvtFloatToInt(.x0, .x1, .single, true), "fcvtzs w0, s1");
    try expectOne(encode.fcvt(.x0, .x1, true), "fcvt d0, s1");
    try expectOne(encode.faddVec(.x0, .x1, .x2), "fadd v0.4s, v1.4s, v2.4s");
    try expectOne(encode.fmulVec(.x0, .x1, .x2), "fmul v0.4s, v1.4s, v2.4s");
    try expectOne(encode.fmlaVec(.x0, .x1, .x2), "fmla v0.4s, v1.4s, v2.4s");
    try expectOne(encode.fmlsVec(.x0, .x1, .x2), "fmls v0.4s, v1.4s, v2.4s");
    try expectOne(encode.fnegVec(.x0, .x1), "fneg v0.4s, v1.4s");
    try expectOne(encode.mvnVec(.x0, .x1), "mvn v0.16b, v1.16b");
    try expectOne(encode.bslVec(.x0, .x1, .x2), "bsl v0.16b, v1.16b, v2.16b");
    try expectOne(encode.dupFromGpr(.x0, .x1), "dup v0.4s, w1");
    try expectOne(encode.dupLane(.x0, .x1, 2), "dup s0, v1.s[2]");
    try expectOne(encode.insLane(.x0, 1, .x1), "ins v0.s[1], v1.s[0]");
    try expectOne(encode.movVec(.x0, .x1), "mov v0.16b, v1.16b");
    try expectOne(encode.fmadd(.x0, .x1, .x2, .x3, false), "fmadd s0, s1, s2, s3");
    try expectOne(encode.fmadd(.x0, .x1, .x2, .x3, true), "fmadd d0, d1, d2, d3");
    try expectOne(encode.fmsub(.x0, .x1, .x2, .x3, false), "fmsub s0, s1, s2, s3");
    try expectOne(encode.fnmsub(.x0, .x1, .x2, .x3, true), "fnmsub d0, d1, d2, d3");
}
