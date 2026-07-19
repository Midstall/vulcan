//! x86 (32-bit / i386) disassembler for the subset `encode.zig` emits. Like the x86-64
//! decoder but simpler: no REX, no SSE/VEX, 32-bit register names, `mov r, imm32` short
//! form, `push`, `int 0x80`, and `cdq`. Parses ModRM/SIB/disp to render operands and to
//! compute each instruction's length so it can step a stream. Unknown bytes print as
//! `.byte 0x<hex>`. Validated by round-tripping every encoder function.

const std = @import("std");
const encode = @import("encode.zig");

pub fn format(allocator: std.mem.Allocator, code: []const u8) std.mem.Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var pos: usize = 0;
    while (pos < code.len) {
        try out.print(allocator, "{x:0>4}: ", .{pos});
        const len = try decode(allocator, &out, code, pos);
        try out.append(allocator, '\n');
        pos += len;
    }
    return out.toOwnedSlice(allocator);
}

pub fn disasmInst(allocator: std.mem.Allocator, out: *std.ArrayList(u8), code: []const u8, pos: usize) std.mem.Allocator.Error!usize {
    return decode(allocator, out, code, pos);
}

const reg = [_][]const u8{ "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi" };
const reg16 = [_][]const u8{ "ax", "cx", "dx", "bx", "sp", "bp", "si", "di" };
const reg8 = [_][]const u8{ "al", "cl", "dl", "bl", "ah", "ch", "dh", "bh" };

const D = struct {
    code: []const u8,
    i: usize,
    esc: u8 = 0,
    mand: u8 = 0, // 0x66 operand-size / 0xF2 / 0xF3 prefix

    fn byte(d: *D) u8 {
        const b = if (d.i < d.code.len) d.code[d.i] else 0;
        d.i += 1;
        return b;
    }
    fn peek(d: *D) u8 {
        return if (d.i < d.code.len) d.code[d.i] else 0;
    }
    fn imm32(d: *D) i32 {
        const u: u32 = @as(u32, d.byte()) | (@as(u32, d.byte()) << 8) | (@as(u32, d.byte()) << 16) | (@as(u32, d.byte()) << 24);
        return @bitCast(u);
    }
};

/// A register at the operand size (16-bit with a 0x66 prefix, else 32-bit).
fn rv(d: *const D, n: u8) []const u8 {
    return if (d.mand == 0x66) reg16[n] else reg[n];
}
/// The `<size> ptr ` memory hint at the operand size.
fn ptrHint(d: *const D) []const u8 {
    return if (d.mand == 0x66) "word ptr " else "dword ptr ";
}

const Operands = struct { reg: u8, mod: u8, rm: u8, mem: bool, disp: i32, index: u8 = 8, scale: u8 = 1 };

fn modrm(d: *D) Operands {
    const m = d.byte();
    const mod = m >> 6;
    const rg = (m >> 3) & 7;
    var rm = m & 7;
    if (mod == 3) return .{ .reg = rg, .mod = mod, .rm = rm, .mem = false, .disp = 0 };
    var index: u8 = 8;
    var scale: u8 = 1;
    if (rm == 4) { // SIB
        const sib = d.byte();
        rm = sib & 7; // base
        const idx = (sib >> 3) & 7;
        if (idx != 4) { // 100 = no index register
            index = idx;
            scale = @as(u8, 1) << @intCast(sib >> 6);
        }
    }
    var disp: i32 = 0;
    if (mod == 1) {
        disp = @as(i8, @bitCast(d.byte()));
    } else if (mod == 2 or (mod == 0 and (m & 7) == 5)) {
        disp = d.imm32();
    }
    return .{ .reg = rg, .mod = mod, .rm = rm, .mem = true, .disp = disp, .index = index, .scale = scale };
}

fn memHint(a: std.mem.Allocator, b: *std.ArrayList(u8), o: Operands, hint: []const u8) !void {
    try b.print(a, "{s}[{s}", .{ hint, reg[o.rm] });
    if (o.index != 8) {
        if (o.scale == 1) {
            try b.print(a, " + {s}", .{reg[o.index]});
        } else {
            try b.print(a, " + {d}*{s}", .{ o.scale, reg[o.index] });
        }
    }
    if (o.disp != 0) try b.print(a, " {s} {d}", .{ if (o.disp < 0) "-" else "+", @abs(o.disp) });
    try b.append(a, ']');
}

fn memOperand(a: std.mem.Allocator, b: *std.ArrayList(u8), o: Operands) !void {
    try memHint(a, b, o, "");
}

fn decode(a: std.mem.Allocator, b: *std.ArrayList(u8), code: []const u8, pos: usize) !usize {
    var d = D{ .code = code, .i = pos };
    // Legacy prefixes: 0x66/0xF2/0xF3 recorded; segment/addr-size/lock consumed to stay in sync.
    while (true) switch (d.peek()) {
        0x66, 0xF2, 0xF3 => d.mand = d.byte(),
        0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65, 0x67, 0xF0 => _ = d.byte(),
        else => break,
    };
    if (d.peek() == 0x0F) {
        _ = d.byte();
        d.esc = 0x0F;
    }
    const op = d.byte();
    if (d.esc == 0x0F) {
        switch (op) {
            0x1F => _ = modrm(&d), // multi-byte nop: consume the ModRM (text appended below)
            0xAF => {
                const o = modrm(&d);
                try b.print(a, "imul {s}, ", .{rv(&d, o.reg)});
                if (o.mem) try memHint(a, b, o, ptrHint(&d)) else try b.appendSlice(a, rv(&d, o.rm));
            },
            0xB6, 0xB7, 0xBE, 0xBF => { // movzx / movsx: src is 8-bit (B6/BE) or 16-bit
                const o = modrm(&d);
                const wide = op == 0xB7 or op == 0xBF;
                const mnem: []const u8 = if (op == 0xBE or op == 0xBF) "movsx" else "movzx";
                try b.print(a, "{s} {s}, ", .{ mnem, rv(&d, o.reg) });
                if (o.mem) try memHint(a, b, o, if (wide) "word ptr " else "byte ptr ") else try b.appendSlice(a, if (wide) reg16[o.rm] else reg8[o.rm]);
            },
            0x40...0x4F => { // cmovcc
                const o = modrm(&d);
                try b.print(a, "cmov{s} {s}, {s}", .{ condName(op & 0xF), rv(&d, o.reg), rv(&d, o.rm) });
            },
            0x90...0x9F => {
                const o = modrm(&d);
                try b.print(a, "set{s} {s}", .{ condName(op & 0xF), reg8[o.rm] });
            },
            0x80...0x8F => {
                const rel = d.imm32();
                try b.print(a, "j{s} .{s}{d}", .{ condName(op & 0xF), sgn(rel), @abs(rel) });
            },
            else => if (!try sse(a, b, &d, op)) try b.print(a, ".byte 0x0f{x:0>2}", .{op}),
        }
        if (op == 0x1F) try b.appendSlice(a, "nop");
        return d.i - pos;
    }
    switch (op) {
        0x01, 0x09, 0x21, 0x29, 0x31, 0x39, 0x85, 0x89 => { // ALU r/m, reg (store direction)
            const mnem: []const u8 = switch (op) {
                0x01 => "add",
                0x09 => "or",
                0x21 => "and",
                0x29 => "sub",
                0x31 => "xor",
                0x39 => "cmp",
                0x85 => "test",
                0x89 => "mov",
                else => unreachable,
            };
            const o = modrm(&d);
            if (o.mem) {
                try b.print(a, "{s} ", .{mnem});
                try memHint(a, b, o, ptrHint(&d));
                try b.print(a, ", {s}", .{rv(&d, o.reg)});
            } else {
                try b.print(a, "{s} {s}, {s}", .{ mnem, rv(&d, o.rm), rv(&d, o.reg) });
            }
        },
        0x03, 0x0B, 0x13, 0x1B, 0x23, 0x2B, 0x33, 0x3B => { // ALU reg, r/m (load direction)
            const o = modrm(&d);
            try b.print(a, "{s} {s}, ", .{ aluName(op >> 3), rv(&d, o.reg) });
            if (o.mem) try memHint(a, b, o, ptrHint(&d)) else try b.appendSlice(a, rv(&d, o.rm));
        },
        0x05, 0x0D, 0x15, 0x1D, 0x25, 0x2D, 0x35, 0x3D => // ALU eax, imm
        try b.print(a, "{s} {s}, {d}", .{ aluName(op >> 3), rv(&d, 0), d.imm32() }),
        0x8B, 0x8D => { // mov reg, [mem] (hinted) / lea reg, [mem] (no hint)
            const o = modrm(&d);
            try b.print(a, "{s} {s}, ", .{ if (op == 0x8B) "mov" else "lea", rv(&d, o.reg) });
            try memHint(a, b, o, if (op == 0x8B) ptrHint(&d) else "");
        },
        0x81, 0x83 => { // ALU r/m, imm (imm32 for 0x81, sign-extended imm8 for 0x83)
            const o = modrm(&d);
            const imm: i32 = if (op == 0x81) d.imm32() else @as(i8, @bitCast(d.byte()));
            try b.print(a, "{s} ", .{aluName(o.reg)});
            if (o.mem) try memHint(a, b, o, ptrHint(&d)) else try b.appendSlice(a, rv(&d, o.rm));
            try b.print(a, ", {d}", .{imm});
        },
        0xC7 => { // mov r/m, imm32
            const o = modrm(&d);
            try b.appendSlice(a, "mov ");
            if (o.mem) try memHint(a, b, o, ptrHint(&d)) else try b.appendSlice(a, rv(&d, o.rm));
            try b.print(a, ", {d}", .{d.imm32()});
        },
        0x69 => {
            const o = modrm(&d);
            const imm = d.imm32();
            try b.print(a, "imul {s}, {s}, {d}", .{ rv(&d, o.reg), rv(&d, o.rm), imm });
        },
        0xC1, 0xD1, 0xD3 => { // shift r/m by imm8 (C1) / by 1 (D1) / by CL (D3)
            const o = modrm(&d);
            const mnem: []const u8 = switch (o.reg) {
                0 => "rol",
                1 => "ror",
                4 => "shl",
                5 => "shr",
                6 => "shl",
                7 => "sar",
                else => "sh?",
            };
            try b.print(a, "{s} ", .{mnem});
            if (o.mem) try memHint(a, b, o, ptrHint(&d)) else try b.appendSlice(a, rv(&d, o.rm));
            if (op == 0xC1) {
                try b.print(a, ", {d}", .{d.byte()});
            } else if (op == 0xD1) {
                try b.appendSlice(a, ", 1");
            } else try b.appendSlice(a, ", cl");
        },
        0xF7 => { // group 3: test/not/neg/mul/imul/div/idiv r/m
            const o = modrm(&d);
            switch (o.reg) {
                0 => try b.print(a, "test {s}, {d}", .{ rv(&d, o.rm), d.imm32() }),
                2 => try b.print(a, "not {s}", .{rv(&d, o.rm)}),
                3 => try b.print(a, "neg {s}", .{rv(&d, o.rm)}),
                4 => try b.print(a, "mul {s}", .{rv(&d, o.rm)}),
                5 => try b.print(a, "imul {s}", .{rv(&d, o.rm)}),
                6 => try b.print(a, "div {s}", .{rv(&d, o.rm)}),
                else => try b.print(a, "idiv {s}", .{rv(&d, o.rm)}),
            }
        },
        0xFF => { // group 5: inc/dec/call/jmp/push r/m
            const o = modrm(&d);
            const mnem: []const u8 = switch (o.reg) {
                0 => "inc",
                1 => "dec",
                2 => "call",
                4 => "jmp",
                6 => "push",
                else => "grp5?",
            };
            try b.print(a, "{s} ", .{mnem});
            if (o.mem) try memHint(a, b, o, if (o.reg >= 2) "dword ptr " else ptrHint(&d)) else try b.appendSlice(a, rv(&d, o.rm));
        },
        // Byte ALU: r/m8,r8 (+0) and r8,r/m8 (+2), plus test/mov byte forms.
        0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x84, 0x88 => {
            const mnem = if (op == 0x84) "test" else if (op == 0x88) "mov" else aluName(op >> 3);
            const o = modrm(&d);
            try b.print(a, "{s} ", .{mnem});
            if (o.mem) try memHint(a, b, o, "byte ptr ") else try b.appendSlice(a, reg8[o.rm]);
            try b.print(a, ", {s}", .{reg8[o.reg]});
        },
        0x02, 0x0A, 0x12, 0x1A, 0x22, 0x2A, 0x32, 0x3A, 0x8A => {
            const mnem = if (op == 0x8A) "mov" else aluName(op >> 3);
            const o = modrm(&d);
            try b.print(a, "{s} {s}, ", .{ mnem, reg8[o.reg] });
            if (o.mem) try memHint(a, b, o, "byte ptr ") else try b.appendSlice(a, reg8[o.rm]);
        },
        0x99 => try b.appendSlice(a, if (d.mand == 0x66) "cwd" else "cdq"),
        0x98 => try b.appendSlice(a, if (d.mand == 0x66) "cbw" else "cwde"),
        0xB8...0xBF => try b.print(a, "mov {s}, {d}", .{ rv(&d, op & 7), d.imm32() }),
        0x40...0x47 => try b.print(a, "inc {s}", .{rv(&d, op & 7)}), // single-byte inc (32-bit mode)
        0x48...0x4F => try b.print(a, "dec {s}", .{rv(&d, op & 7)}), // single-byte dec
        0x50...0x57 => try b.print(a, "push {s}", .{reg[op & 7]}),
        0x58...0x5F => try b.print(a, "pop {s}", .{reg[op & 7]}),
        0x68 => try b.print(a, "push {d}", .{d.imm32()}),
        0x6A => try b.print(a, "push {d}", .{@as(i32, @as(i8, @bitCast(d.byte())))}),
        0x70...0x7F => { // jcc rel8
            const rel: i32 = @as(i8, @bitCast(d.byte()));
            try b.print(a, "j{s} .{s}{d}", .{ condName(op & 0xF), sgn(rel), @abs(rel) });
        },
        0xEB => {
            const rel: i32 = @as(i8, @bitCast(d.byte()));
            try b.print(a, "jmp .{s}{d}", .{ sgn(rel), @abs(rel) });
        },
        0xA8 => try b.print(a, "test al, {d}", .{d.byte()}),
        0xA9 => try b.print(a, "test {s}, {d}", .{ rv(&d, 0), d.imm32() }),
        0xC6 => { // mov r/m8, imm8
            const o = modrm(&d);
            try b.appendSlice(a, "mov ");
            if (o.mem) try memHint(a, b, o, "byte ptr ") else try b.appendSlice(a, reg8[o.rm]);
            try b.print(a, ", {d}", .{d.byte()});
        },
        0xE8 => {
            const rel = d.imm32();
            try b.print(a, "call .{s}{d}", .{ sgn(rel), @abs(rel) });
        },
        0xE9 => {
            const rel = d.imm32();
            try b.print(a, "jmp .{s}{d}", .{ sgn(rel), @abs(rel) });
        },
        0xC3 => try b.appendSlice(a, "ret"),
        0xCD => try b.print(a, "int 0x{x}", .{d.byte()}),
        0x90 => try b.appendSlice(a, "nop"), // also the inter-function alignment padding
        0xD8...0xDF => try x87(a, b, &d, op),
        else => try b.print(a, ".byte 0x{x:0>2}", .{op}),
    }
    return d.i - pos;
}

fn xmm(a: std.mem.Allocator, b: *std.ArrayList(u8), n: u8) !void {
    try b.print(a, "xmm{d}", .{n});
}

/// Decode an x87 FPU instruction (0xD8..0xDF). Consumes the ModRM (so the stream stays synced)
/// and renders the common memory loads/stores and the `fXXX st(i)` register forms.
fn x87(a: std.mem.Allocator, b: *std.ArrayList(u8), d: *D, op: u8) !void {
    const o = modrm(d);
    if (o.mem) {
        // Memory forms: mnemonic and operand size depend on the opcode and the /digit.
        const size: []const u8 = switch (op) {
            0xDD, 0xDC => "qword ptr ",
            0xDF => if (o.reg >= 5) "qword ptr " else "word ptr ",
            0xDB => "dword ptr ",
            else => "dword ptr ", // 0xD8/0xD9
        };
        const mnem: []const u8 = switch (op) {
            0xD9, 0xDD => switch (o.reg) {
                0 => "fld",
                2 => "fst",
                3 => "fstp",
                else => "fldenv",
            },
            0xDB, 0xDF => switch (o.reg) {
                0 => "fild",
                3 => "fistp",
                5 => "fild",
                7 => "fistp",
                else => "fild",
            },
            0xD8, 0xDC => switch (o.reg) {
                0 => "fadd",
                1 => "fmul",
                4 => "fsub",
                6 => "fdiv",
                else => "fadd",
            },
            else => "fld",
        };
        try b.print(a, "{s} {s}", .{ mnem, size });
        return memHint(a, b, o, "");
    }
    // Register forms `fXXX st(i)` (o.rm selects the stack slot). Kept coarse.
    const mnem: []const u8 = switch (op) {
        0xD8, 0xDC => "fadd",
        0xD9 => "fld",
        0xDD => "fst",
        else => "fxch",
    };
    try b.print(a, "{s} st({d})", .{ mnem, o.rm });
}

/// The SSE mnemonic for a (mandatory-prefix, opcode) pair, or null if not a known SSE op.
fn sseName(mand: u8, op: u8) ?[]const u8 {
    return switch (mand) {
        0xF3 => switch (op) {
            0x10, 0x11 => "movss",
            0x58 => "addss",
            0x59 => "mulss",
            0x5C => "subss",
            0x5E => "divss",
            0x51 => "sqrtss",
            0x5A => "cvtss2sd",
            0x6F, 0x7F => "movdqu",
            else => null,
        },
        0xF2 => switch (op) {
            0x10, 0x11 => "movsd",
            0x58 => "addsd",
            0x59 => "mulsd",
            0x5C => "subsd",
            0x5E => "divsd",
            0x51 => "sqrtsd",
            0x5A => "cvtsd2ss",
            else => null,
        },
        0x66 => switch (op) {
            0x2E => "ucomisd",
            0x28, 0x29 => "movapd",
            0x6F, 0x7F => "movdqa",
            0xEF => "pxor",
            0xDB => "pand",
            0xEB => "por",
            0xD4 => "paddq",
            0xFE => "paddd",
            0xFA => "psubd",
            0xF4 => "pmuludq",
            0x62 => "punpckldq",
            0x6A => "punpckhdq",
            0xD6 => "movq",
            else => null,
        },
        else => switch (op) {
            0x10, 0x11 => "movups",
            0x58 => "addps",
            0x59 => "mulps",
            0x5C => "subps",
            0x5E => "divps",
            0x2E => "ucomiss",
            0x28, 0x29 => "movaps",
            else => null,
        },
    };
}

fn sseHint(mand: u8, op: u8) []const u8 {
    const scalar = switch (op) {
        0x10, 0x11, 0x51, 0x58, 0x59, 0x5A, 0x5C, 0x5E, 0x2A, 0x2C, 0x2E => true,
        else => false,
    };
    if (scalar and mand == 0xF3) return "dword ptr ";
    if (scalar and mand == 0xF2) return "qword ptr ";
    return "xmmword ptr ";
}

/// Decode an SSE/SSE2 0F opcode. Returns whether it matched.
fn sse(a: std.mem.Allocator, b: *std.ArrayList(u8), d: *D, op: u8) !bool {
    if (op == 0x6E or op == 0x7E) { // movd xmm<->r32
        const o = modrm(d);
        if (op == 0x6E) {
            try b.appendSlice(a, "movd ");
            try xmm(a, b, o.reg);
            try b.print(a, ", {s}", .{reg[o.rm]});
        } else {
            try b.print(a, "movd {s}, ", .{reg[o.rm]});
            try xmm(a, b, o.reg);
        }
        return true;
    }
    if (op == 0x2A or op == 0x2C) { // cvtsi2ss/sd, cvttss2si/cvttsd2si
        const o = modrm(d);
        if (op == 0x2A) {
            try b.print(a, "{s} ", .{if (d.mand == 0xF2) "cvtsi2sd" else "cvtsi2ss"});
            try xmm(a, b, o.reg);
            try b.print(a, ", {s}", .{reg[o.rm]});
        } else {
            try b.print(a, "{s} {s}, ", .{ if (d.mand == 0xF2) "cvttsd2si" else "cvttss2si", reg[o.reg] });
            try xmm(a, b, o.rm);
        }
        return true;
    }
    if (op == 0x70) { // pshufd/pshuflw xmm, xmm, imm8
        const o = modrm(d);
        try b.appendSlice(a, "pshufd ");
        try xmm(a, b, o.reg);
        try b.appendSlice(a, ", ");
        if (o.mem) try memHint(a, b, o, "xmmword ptr ") else try xmm(a, b, o.rm);
        try b.print(a, ", {d}", .{d.byte()});
        return true;
    }
    const mnem = sseName(d.mand, op) orelse return false;
    const o = modrm(d);
    const store = op == 0x11 or op == 0x29 or op == 0x7F or op == 0xD6;
    try b.print(a, "{s} ", .{mnem});
    if (o.mem) {
        if (store) {
            try memHint(a, b, o, sseHint(d.mand, op));
            try b.appendSlice(a, ", ");
            try xmm(a, b, o.reg);
        } else {
            try xmm(a, b, o.reg);
            try b.appendSlice(a, ", ");
            try memHint(a, b, o, sseHint(d.mand, op));
        }
    } else {
        try xmm(a, b, o.reg);
        try b.appendSlice(a, ", ");
        try xmm(a, b, o.rm);
    }
    return true;
}

/// The ALU mnemonic for an opcode group index (bits[5:3], or a /digit).
fn aluName(g: u8) []const u8 {
    return switch (g & 7) {
        0 => "add",
        1 => "or",
        2 => "adc",
        3 => "sbb",
        4 => "and",
        5 => "sub",
        6 => "xor",
        else => "cmp",
    };
}

/// A function's byte offset in a linked image.
pub const Sym = struct { name: []const u8, offset: usize };

/// Render a whole linked module: label each function at its byte offset and annotate every
/// resolved `call rel32` that lands on a function with its name. Caller owns the result.
/// A decoded source-line row keyed by byte offset (address == offset in a relocatable `.text`).
pub const AddrLine = struct { addr: u64, line: u32 };

pub fn formatModule(allocator: std.mem.Allocator, code: []const u8, syms: []const Sym) std.mem.Allocator.Error![]u8 {
    return formatModuleImpl(allocator, code, syms, &.{});
}

/// Like `formatModule`, but interleaves `; line N` markers from `lines` (sorted by address).
pub fn formatModuleWithLines(allocator: std.mem.Allocator, code: []const u8, syms: []const Sym, lines: []const AddrLine) std.mem.Allocator.Error![]u8 {
    return formatModuleImpl(allocator, code, syms, lines);
}

fn formatModuleImpl(allocator: std.mem.Allocator, code: []const u8, syms: []const Sym, lines: []const AddrLine) std.mem.Allocator.Error![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var pos: usize = 0;
    var li: usize = 0;
    while (pos < code.len) {
        for (syms) |s| if (s.offset == pos) {
            if (pos != 0) try buf.append(allocator, '\n');
            try buf.print(allocator, "{s}:\n", .{s.name});
        };
        while (li < lines.len and lines[li].addr < pos) : (li += 1) {}
        while (li < lines.len and lines[li].addr == pos) : (li += 1) {
            try buf.print(allocator, "; line {d}\n", .{lines[li].line});
        }
        try buf.print(allocator, "{x:0>4}: ", .{pos});
        const len = try decode(allocator, &buf, code, pos);
        // `decode` reports len 5 even for a truncated call (its cursor reads past the
        // end as 0), so guard the 4-byte read against the real buffer length.
        if (code[pos] == 0xE8 and len == 5 and code.len - pos >= 5) { // call rel32 -> target = end + rel32
            const rel: i32 = @bitCast(std.mem.readInt(u32, code[pos + 1 ..][0..4], .little));
            const target = @as(i64, @intCast(pos)) + @as(i64, @intCast(len)) + rel;
            if (target >= 0) for (syms) |s| {
                if (s.offset == @as(usize, @intCast(target))) {
                    try buf.print(allocator, "  <{s}>", .{s.name});
                    break;
                }
            };
        }
        try buf.append(allocator, '\n');
        pos += len;
    }
    return buf.toOwnedSlice(allocator);
}

fn condName(cc: u8) []const u8 {
    return switch (cc) {
        0x4 => "e",
        0x5 => "ne",
        0xC => "l",
        0xD => "ge",
        0xE => "le",
        0xF => "g",
        0x2 => "b",
        0x3 => "ae",
        0x6 => "be",
        0x7 => "a",
        else => "?",
    };
}

fn sgn(v: i32) []const u8 {
    return if (v < 0) "-" else "+";
}

fn expectOne(inst: encode.Inst, expected: []const u8) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    _ = try decode(std.testing.allocator, &out, inst.slice(), 0);
    try std.testing.expectEqualStrings(expected, out.items);
}

test "round-trips i386 integer ops" {
    try expectOne(encode.movImm(.eax, 42), "mov eax, 42");
    try expectOne(encode.movReg(.eax, .ebx), "mov eax, ebx");
    try expectOne(encode.add(.eax, .ebx), "add eax, ebx");
    try expectOne(encode.sub(.ecx, .edx), "sub ecx, edx");
    try expectOne(encode.xorr(.eax, .eax), "xor eax, eax");
    try expectOne(encode.imul(.eax, .ebx), "imul eax, ebx");
    try expectOne(encode.cmp(.esi, .edi), "cmp esi, edi");
    try expectOne(encode.aluImm(0, .eax, 5), "add eax, 5");
    try expectOne(encode.aluImm(5, .ebx, 7), "sub ebx, 7");
    try expectOne(encode.imulImm(.eax, .ebx, 3), "imul eax, ebx, 3");
    try expectOne(encode.shiftImm(4, .eax, 2), "shl eax, 2");
    try expectOne(encode.shrCl(.ebx), "shr ebx, cl");
    try expectOne(encode.idiv(.ecx), "idiv ecx");
    try expectOne(encode.cdq(), "cdq");
    try expectOne(encode.ret(), "ret");
    try expectOne(encode.int80(), "int 0x80");
}

test "round-trips i386 flags, stack, branches" {
    try expectOne(encode.setcc(.eax, .l), "setl al");
    try expectOne(encode.movzxByte(.eax, .ebx), "movzx eax, bl");
    try expectOne(encode.jcc(.e, 16), "je .+16");
    try expectOne(encode.jmp(-8), "jmp .-8");
    try expectOne(encode.callRel(32), "call .+32");
    try expectOne(encode.pushReg(.ebp), "push ebp");
    try expectOne(encode.pushImm(100), "push 100");
    try expectOne(encode.movFromStack(.eax, 4), "mov eax, dword ptr [esp + 4]");
    try expectOne(encode.stackLoad(.eax, 16), "mov eax, dword ptr [esp + 16]");
    try expectOne(encode.stackStore(16, .ebx), "mov dword ptr [esp + 16], ebx");
}

test "round-trips i386 reg+disp32 memory encoders" {
    try expectOne(encode.movFromMem32(.eax, .ecx, 4), "mov eax, dword ptr [ecx + 4]");
    try expectOne(encode.movToMem32(.ecx, 8, .edx), "mov dword ptr [ecx + 8], edx");
    try expectOne(encode.movFromMem32(.eax, .esp, 16), "mov eax, dword ptr [esp + 16]");
    try expectOne(encode.movToMem16(.ebx, 2, .esi), "mov word ptr [ebx + 2], si");
    try expectOne(encode.movToMem8(.ebx, 1, .eax), "mov byte ptr [ebx + 1], al");
    try expectOne(encode.movzxByteFromMem(.eax, .ecx, 0), "movzx eax, byte ptr [ecx]");
    try expectOne(encode.movsxByteFromMem(.edx, .ebx, -4), "movsx edx, byte ptr [ebx - 4]");
    try expectOne(encode.movzxWordFromMem(.eax, .ecx, 0), "movzx eax, word ptr [ecx]");
    try expectOne(encode.movsxWordFromMem(.edx, .ebx, -4), "movsx edx, word ptr [ebx - 4]");
}
