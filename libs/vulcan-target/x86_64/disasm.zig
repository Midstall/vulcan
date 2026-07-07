//! x86-64 disassembler for the subset `encode.zig` emits. Unlike the fixed-width RISC
//! backends, x86 is variable length, so the decoder parses the real instruction structure
//! (legacy/mandatory prefix, REX or 3-byte VEX, 0F/0F3A opcode escape, ModRM+SIB+disp,
//! immediate) to both render the instruction AND compute its length, which is how it steps
//! through a byte stream. Anything outside the emitted subset prints as `.byte 0x<hex>`.
//!
//! `format` renders a whole code buffer; `disasmInst` decodes one instruction and reports
//! its length. Validated by round-tripping every encoder function.

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

/// Decode one instruction; append its text and return its byte length.
pub fn disasmInst(allocator: std.mem.Allocator, out: *std.ArrayList(u8), code: []const u8, pos: usize) std.mem.Allocator.Error!usize {
    return decode(allocator, out, code, pos);
}

/// A function's byte offset in a linked image.
pub const Sym = struct { name: []const u8, offset: usize };

/// Render a whole linked module: each function gets a `name:` label at its byte offset, and
/// every resolved `call rel32` that lands on a function is annotated with its name, so a
/// linked image (from link.compileModule) reads as a symbolized listing. Caller owns it.
/// A decoded source-line row keyed by byte offset (from a `.debug_line` program), for objdump
/// `-S`-style annotation. In a relocatable object `.text` sits at address 0, so address == offset.
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
        // A `call rel32` (0xE8): its target is `end + rel32`; annotate a hit symbol.
        if (code[pos] == 0xE8 and len == 5) {
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

const gpr = [_][]const u8{
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
};
const gpr32 = [_][]const u8{
    "eax", "ecx", "edx",  "ebx",  "esp",  "ebp",  "esi",  "edi",
    "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
};
const gpr16 = [_][]const u8{
    "ax",  "cx",  "dx",   "bx",   "sp",   "bp",   "si",   "di",
    "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w",
};
const gpr8 = [_][]const u8{
    "al",  "cl",  "dl",   "bl",   "spl",  "bpl",  "sil",  "dil",
    "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
};

/// A general register at the instruction's operand size: 64-bit with REX.W, 16-bit with a 0x66
/// prefix, else 32-bit (the 64-bit-mode default). Addresses always use the 64-bit names.
fn rv(d: *const D, n: u8) []const u8 {
    if (d.rexW) return gpr[n];
    if (d.mand == 0x66) return gpr16[n];
    return gpr32[n];
}

/// A general register that is 64-bit with REX.W else 32-bit, ignoring a 0x66 that is a mandatory
/// SSE prefix (movd/movq, cvtsi2ss, cvttss2si) rather than an operand-size override.
fn rvq(d: *const D, n: u8) []const u8 {
    return if (d.rexW) gpr[n] else gpr32[n];
}

fn xmm(a: std.mem.Allocator, b: *std.ArrayList(u8), n: u8) !void {
    try b.print(a, "xmm{d}", .{n});
}

/// Decoder cursor + parsed prefix/REX/VEX state.
const D = struct {
    code: []const u8,
    i: usize,
    mand: u8 = 0, // mandatory/legacy prefix (0x66/0xF2/0xF3), 0 = none
    rexW: bool = false,
    R: u8 = 0, // ModRM.reg extension bit (REX.R / VEX)
    X: u8 = 0, // SIB.index extension bit (REX.X / VEX)
    B: u8 = 0, // ModRM.rm / base extension bit
    vex: bool = false,
    vvvv: u8 = 0, // VEX source-1 register
    esc: u16 = 0, // opcode map: 0 (1-byte), 0x0F, 0x38, 0x3A

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
    fn imm64(d: *D) u64 {
        var u: u64 = 0;
        inline for (0..8) |k| u |= @as(u64, d.byte()) << (k * 8);
        return u;
    }
};

fn decode(a: std.mem.Allocator, b: *std.ArrayList(u8), code: []const u8, pos: usize) !usize {
    var d = D{ .code = code, .i = pos };

    // Legacy prefixes: 0x66/0xF2/0xF3 are mandatory/operand-size (recorded); segment overrides,
    // address-size, and lock are consumed so the stream stays in sync.
    while (true) switch (d.peek()) {
        0x66, 0xF2, 0xF3 => d.mand = d.byte(),
        0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65, 0x67, 0xF0 => _ = d.byte(),
        else => break,
    };

    if (d.peek() == 0xC4) { // 3-byte VEX
        _ = d.byte();
        const byte2 = d.byte();
        const byte3 = d.byte();
        d.vex = true;
        d.R = ((byte2 >> 7) & 1) ^ 1;
        d.X = ((byte2 >> 6) & 1) ^ 1;
        d.B = ((byte2 >> 5) & 1) ^ 1;
        d.esc = switch (byte2 & 0x1F) {
            1 => 0x0F,
            2 => 0x38,
            3 => 0x3A,
            else => 0x0F,
        };
        d.rexW = (byte3 >> 7) & 1 != 0;
        d.vvvv = (~(byte3 >> 3)) & 0xF;
        d.mand = switch (byte3 & 3) {
            1 => 0x66,
            2 => 0xF3,
            3 => 0xF2,
            else => 0,
        };
    } else {
        if (d.peek() & 0xF0 == 0x40) { // REX
            const rex = d.byte();
            d.rexW = rex & 8 != 0;
            d.R = (rex >> 2) & 1;
            d.X = (rex >> 1) & 1;
            d.B = rex & 1;
        }
        if (d.peek() == 0x0F) {
            _ = d.byte();
            d.esc = 0x0F;
            if (d.peek() == 0x38 or d.peek() == 0x3A) d.esc = d.byte();
        }
    }

    const op = d.byte();
    try dispatch(a, b, &d, op);
    return d.i - pos;
}

/// ModRM operands: returns reg (with the R extension) and, for a register-direct rm, the rm
/// register; for a memory rm it records base+disp and advances past SIB/disp.
const Operands = struct {
    reg: u8,
    mod: u8,
    rm: u8, // register number (direct) or base register (memory)
    mem: bool,
    disp: i32,
    index: u8 = 16, // SIB index register, 16 = none
    scale: u8 = 1, // SIB scale (1/2/4/8)
};

fn modrm(d: *D) Operands {
    const m = d.byte();
    const mod = m >> 6;
    const reg = (d.R << 3) | ((m >> 3) & 7);
    var rm = (d.B << 3) | (m & 7);
    if (mod == 3) return .{ .reg = reg, .mod = mod, .rm = rm, .mem = false, .disp = 0 };
    var index: u8 = 16;
    var scale: u8 = 1;
    // Memory operand. A base of rsp/r12 (low bits 100) carries a SIB byte.
    if ((m & 7) == 4) {
        const sib = d.byte();
        rm = (d.B << 3) | (sib & 7); // base
        const idx = (d.X << 3) | ((sib >> 3) & 7);
        if (idx != 4) { // index 100 with no REX.X means "no index register"
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
    return .{ .reg = reg, .mod = mod, .rm = rm, .mem = true, .disp = disp, .index = index, .scale = scale };
}

/// The `<size> ptr ` hint llvm prints before a memory operand, at the operand size.
fn ptrHint(d: *const D) []const u8 {
    if (d.rexW) return "qword ptr ";
    if (d.mand == 0x66) return "word ptr ";
    return "dword ptr ";
}

/// Render a memory operand `<hint>[base + index*scale + disp]`. Address registers are 64-bit.
fn memHint(a: std.mem.Allocator, b: *std.ArrayList(u8), o: Operands, hint: []const u8) !void {
    try b.print(a, "{s}[{s}", .{ hint, gpr[o.rm] });
    if (o.index != 16) {
        if (o.scale == 1) {
            try b.print(a, " + {s}", .{gpr[o.index]});
        } else {
            try b.print(a, " + {d}*{s}", .{ o.scale, gpr[o.index] });
        }
    }
    if (o.disp != 0) try b.print(a, " {s} {d}", .{ if (o.disp < 0) "-" else "+", @abs(o.disp) });
    try b.append(a, ']');
}

/// A memory operand with no size hint (the default for callers that pass a register too).
fn memOperand(a: std.mem.Allocator, b: *std.ArrayList(u8), o: Operands) !void {
    try memHint(a, b, o, "");
}

fn dispatch(a: std.mem.Allocator, b: *std.ArrayList(u8), d: *D, op: u8) !void {
    if (d.vex) return vexOp(a, b, d, op);
    if (d.esc == 0x0F) return twoByte(a, b, d, op);
    if (d.esc == 0x3A) return threeByteA(a, b, d, op);

    // One-byte opcodes.
    switch (op) {
        0x01, 0x09, 0x21, 0x29, 0x31, 0x39, 0x85, 0x89 => {
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
            const o = modrm(d);
            if (o.mem) { // store form: mov [mem], reg
                try b.print(a, "{s} ", .{mnem});
                try memHint(a, b, o, ptrHint(d));
                try b.print(a, ", {s}", .{rv(d, o.reg)});
            } else {
                try b.print(a, "{s} {s}, {s}", .{ mnem, rv(d, o.rm), rv(d, o.reg) });
            }
        },
        0x8B, 0x8D => { // mov reg, [mem] (size-hinted) / lea reg, [mem] (address, no hint)
            const o = modrm(d);
            try b.print(a, "{s} {s}, ", .{ if (op == 0x8B) "mov" else "lea", rv(d, o.reg) });
            try memHint(a, b, o, if (op == 0x8B) ptrHint(d) else "");
        },
        0xC7 => { // mov r/m, imm32 (/0)
            const o = modrm(d);
            const imm = d.imm32();
            if (o.mem) {
                try b.appendSlice(a, "mov ");
                try memHint(a, b, o, ptrHint(d));
                try b.print(a, ", {d}", .{imm});
            } else try b.print(a, "mov {s}, {d}", .{ rv(d, o.rm), imm });
        },
        0x81 => { // ALU r/m, imm32 (/digit)
            const o = modrm(d);
            const imm = d.imm32();
            const mnem: []const u8 = switch ((o.reg & 7)) {
                0 => "add",
                1 => "or",
                4 => "and",
                5 => "sub",
                6 => "xor",
                else => "alu?",
            };
            try b.print(a, "{s} {s}, {d}", .{ mnem, rv(d, o.rm), imm });
        },
        0x69 => { // imul reg, r/m, imm32
            const o = modrm(d);
            const imm = d.imm32();
            try b.print(a, "imul {s}, {s}, {d}", .{ rv(d, o.reg), rv(d, o.rm), imm });
        },
        0xC1, 0xD3 => { // shift r/m by imm8 / by CL
            const o = modrm(d);
            const mnem: []const u8 = switch ((o.reg & 7)) {
                4 => "shl",
                5 => "shr",
                7 => "sar",
                else => "sh?",
            };
            if (op == 0xC1) {
                try b.print(a, "{s} {s}, {d}", .{ mnem, rv(d, o.rm), d.byte() });
            } else try b.print(a, "{s} {s}, cl", .{ mnem, rv(d, o.rm) });
        },
        0xF7 => { // group 3: test/not/neg/mul/imul/div/idiv r/m
            const o = modrm(d);
            switch (o.reg & 7) {
                0 => try b.print(a, "test {s}, {d}", .{ rv(d, o.rm), d.imm32() }),
                2 => try b.print(a, "not {s}", .{rv(d, o.rm)}),
                3 => try b.print(a, "neg {s}", .{rv(d, o.rm)}),
                4 => try b.print(a, "mul {s}", .{rv(d, o.rm)}),
                5 => try b.print(a, "imul {s}", .{rv(d, o.rm)}),
                6 => try b.print(a, "div {s}", .{rv(d, o.rm)}),
                else => try b.print(a, "idiv {s}", .{rv(d, o.rm)}),
            }
        },
        0x98 => try b.appendSlice(a, if (d.rexW) "cdqe" else "cwde"),
        0x99 => try b.appendSlice(a, if (d.rexW) "cqo" else "cdq"),
        0xB8...0xBF => { // mov reg, imm (imm64 with REX.W, else imm32)
            const r = (d.B << 3) | (op & 7);
            if (d.rexW) {
                try b.print(a, "mov {s}, {d}", .{ rv(d, r), d.imm64() });
            } else {
                try b.print(a, "mov {s}, {d}", .{ rv(d, r), @as(u32, @bitCast(d.imm32())) });
            }
        },
        0xE8 => {
            const rel = d.imm32();
            try b.print(a, "call .{s}{d}", .{ relSign(rel), @abs(rel) });
        },
        0xE9 => {
            const rel = d.imm32();
            try b.print(a, "jmp .{s}{d}", .{ relSign(rel), @abs(rel) });
        },
        0xC3 => try b.appendSlice(a, "ret"),
        0x90 => try b.appendSlice(a, "nop"), // also the inter-function alignment padding

        // ALU reg <- r/m (the load direction, opcode +3 within each ALU group).
        0x03, 0x0B, 0x13, 0x1B, 0x23, 0x2B, 0x33, 0x3B => {
            const o = modrm(d);
            try b.print(a, "{s} {s}, ", .{ aluName(op >> 3), rv(d, o.reg) });
            if (o.mem) try memHint(a, b, o, ptrHint(d)) else try b.appendSlice(a, rv(d, o.rm));
        },
        // ALU al/eax/rax, imm (opcode +5 within each group).
        0x05, 0x0D, 0x15, 0x1D, 0x25, 0x2D, 0x35, 0x3D => {
            try b.print(a, "{s} {s}, {d}", .{ aluName(op >> 3), rv(d, 0), d.imm32() });
        },
        0x83 => { // ALU r/m, imm8 (sign-extended), /digit
            const o = modrm(d);
            const imm: i32 = @as(i8, @bitCast(d.byte()));
            try b.print(a, "{s} ", .{aluName(o.reg & 7)});
            if (o.mem) try memHint(a, b, o, ptrHint(d)) else try b.appendSlice(a, rv(d, o.rm));
            try b.print(a, ", {d}", .{imm});
        },
        0x70...0x7F => { // jcc rel8 (short conditional)
            const rel: i32 = @as(i8, @bitCast(d.byte()));
            try b.print(a, "j{s} .{s}{d}", .{ condName(op & 0xF), relSign(rel), @abs(rel) });
        },
        0xEB => { // jmp rel8 (short)
            const rel: i32 = @as(i8, @bitCast(d.byte()));
            try b.print(a, "jmp .{s}{d}", .{ relSign(rel), @abs(rel) });
        },
        0x50...0x57 => try b.print(a, "push {s}", .{gpr[(d.B << 3) | (op & 7)]}), // push r64
        0x58...0x5F => try b.print(a, "pop {s}", .{gpr[(d.B << 3) | (op & 7)]}), // pop r64
        0x63 => { // movsxd r64, r/m32
            const o = modrm(d);
            try b.print(a, "movsxd {s}, ", .{gpr[o.reg]});
            if (o.mem) try memHint(a, b, o, "dword ptr ") else try b.appendSlice(a, gpr32[o.rm]);
        },
        0xFF => { // group 5: inc/dec/call/jmp/push r/m
            const o = modrm(d);
            const digit = o.reg & 7;
            const mnem: []const u8 = switch (digit) {
                0 => "inc",
                1 => "dec",
                2 => "call",
                4 => "jmp",
                6 => "push",
                else => "grp5?",
            };
            try b.print(a, "{s} ", .{mnem});
            // call/jmp/push through a pointer are 64-bit; inc/dec take the operand size.
            const hint: []const u8 = if (digit >= 2) "qword ptr " else ptrHint(d);
            if (o.mem) try memHint(a, b, o, hint) else try b.appendSlice(a, gpr[o.rm]);
        },
        0xA8 => try b.print(a, "test al, {d}", .{d.byte()}),
        0xA9 => try b.print(a, "test {s}, {d}", .{ rv(d, 0), d.imm32() }),

        // Byte ALU: r/m8,r8 (opcode +0) and r8,r/m8 (opcode +2), plus test/mov byte forms.
        0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x84, 0x88 => { // r/m8, r8
            const mnem = if (op == 0x84) "test" else if (op == 0x88) "mov" else aluName(op >> 3);
            const o = modrm(d);
            try b.print(a, "{s} ", .{mnem});
            if (o.mem) try memHint(a, b, o, "byte ptr ") else try b.appendSlice(a, gpr8[o.rm]);
            try b.print(a, ", {s}", .{gpr8[o.reg]});
        },
        0x02, 0x0A, 0x12, 0x1A, 0x22, 0x2A, 0x32, 0x3A, 0x8A => { // r8, r/m8
            const mnem = if (op == 0x8A) "mov" else aluName(op >> 3);
            const o = modrm(d);
            try b.print(a, "{s} {s}, ", .{ mnem, gpr8[o.reg] });
            if (o.mem) try memHint(a, b, o, "byte ptr ") else try b.appendSlice(a, gpr8[o.rm]);
        },
        0xC6 => { // mov r/m8, imm8
            const o = modrm(d);
            try b.appendSlice(a, "mov ");
            if (o.mem) try memHint(a, b, o, "byte ptr ") else try b.appendSlice(a, gpr8[o.rm]);
            try b.print(a, ", {d}", .{d.byte()});
        },
        else => try b.print(a, ".byte 0x{x:0>2}", .{op}),
    }
}

fn twoByte(a: std.mem.Allocator, b: *std.ArrayList(u8), d: *D, op: u8) !void {
    // SSE scalar/packed and integer 0F ops, keyed on the mandatory prefix.
    switch (op) {
        0x05 => return b.appendSlice(a, "syscall"),
        0x1F => { // multi-byte nop (0F 1F /r): consume the ModRM so the stream stays synced
            _ = modrm(d);
            return b.appendSlice(a, "nop");
        },
        0x1E => { // endbr64 (F3 0F 1E FA) / other reserved-nop hints
            _ = d.byte();
            return b.appendSlice(a, "endbr64");
        },
        0xAF => { // imul reg, r/m
            const o = modrm(d);
            try b.print(a, "imul {s}, ", .{rv(d, o.reg)});
            if (o.mem) return memHint(a, b, o, ptrHint(d));
            return b.appendSlice(a, rv(d, o.rm));
        },
        0x73 => { // group: psrlq/psrldq/psllq/pslldq xmm, imm8 (/digit selects)
            const o = modrm(d);
            const imm = d.byte();
            const mnem: []const u8 = switch (o.reg & 7) {
                2 => "psrlq",
                3 => "psrldq",
                6 => "psllq",
                else => "pslldq",
            };
            return b.print(a, "{s} xmm{d}, {d}", .{ mnem, o.rm, imm });
        },
        0xB6, 0xB7, 0xBE, 0xBF => { // movzx / movsx: dst is 32/64-bit, src is 8-bit (B6/BE) or 16-bit
            const o = modrm(d);
            const wide = op == 0xB7 or op == 0xBF; // 16-bit source
            const mnem: []const u8 = if (op == 0xBE or op == 0xBF) "movsx" else "movzx";
            try b.print(a, "{s} {s}, ", .{ mnem, rv(d, o.reg) });
            if (o.mem) return memHint(a, b, o, if (wide) "word ptr " else "byte ptr ");
            return b.appendSlice(a, if (wide) gpr16[o.rm] else gpr8[o.rm]);
        },
        0x40...0x4F => { // cmovcc
            const o = modrm(d);
            return b.print(a, "cmov{s} {s}, {s}", .{ condName(op & 0xF), rv(d, o.reg), rv(d, o.rm) });
        },
        0x90...0x9F => { // setcc r/m8 (a byte register)
            const o = modrm(d);
            return b.print(a, "set{s} {s}", .{ condName(op & 0xF), gpr8[o.rm] });
        },
        0x80...0x8F => { // jcc rel32
            const rel = d.imm32();
            return b.print(a, "j{s} .{s}{d}", .{ condName(op & 0xF), relSign(rel), @abs(rel) });
        },
        else => {},
    }

    // SSE data-movement / arithmetic. movd/movq cross the register files.
    if (op == 0x6E or op == 0x7E) { // movd/movq xmm<->gpr
        const o = modrm(d);
        const mn: []const u8 = if (d.rexW) "movq" else "movd";
        if (op == 0x6E) { // to xmm: reg=xmm, rm=gpr
            try b.print(a, "{s} ", .{mn});
            try xmm(a, b, o.reg);
            return b.print(a, ", {s}", .{rvq(d, o.rm)});
        } else { // from xmm: reg=xmm, rm=gpr, dst is the gpr
            try b.print(a, "{s} {s}, ", .{ mn, rvq(d, o.rm) });
            return xmm(a, b, o.reg);
        }
    }
    if (op == 0x2A) { // cvtsi2ss/sd xmm, gpr
        const o = modrm(d);
        try b.print(a, "{s} ", .{if (d.mand == 0xF2) "cvtsi2sd" else "cvtsi2ss"});
        try xmm(a, b, o.reg);
        return b.print(a, ", {s}", .{rvq(d, o.rm)});
    }
    if (op == 0x2C) { // cvttss2si/cvttsd2si gpr, xmm
        const o = modrm(d);
        try b.print(a, "{s} {s}, ", .{ if (d.mand == 0xF2) "cvttsd2si" else "cvttss2si", rvq(d, o.reg) });
        return xmm(a, b, o.rm);
    }
    if (op == 0x70) { // pshufd xmm, xmm, imm8
        const o = modrm(d);
        try b.appendSlice(a, "pshufd ");
        try xmm(a, b, o.reg);
        try b.appendSlice(a, ", ");
        try rmXmmOrMem(a, b, o);
        return b.print(a, ", {d}", .{d.byte()});
    }

    // The remaining two-xmm ops (arith, moves, converts, compares).
    const mnem = sseName(d.mand, op) orelse return b.print(a, ".byte 0x0f{x:0>2}", .{op});
    const o = modrm(d);
    try b.print(a, "{s} ", .{mnem});
    // Store-direction opcodes put the memory/register destination first.
    const store = op == 0x11 or op == 0x29 or op == 0x7F or op == 0xD6;
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
}

fn threeByteA(a: std.mem.Allocator, b: *std.ArrayList(u8), d: *D, op: u8) !void {
    // 66 0F 3A 21 insertps; F3/F2 0F 3A 11 roundss/roundsd (as this backend emits them).
    const o = modrm(d);
    const mnem: []const u8 = if (op == 0x21) "insertps" else if (d.mand == 0xF2) "roundsd" else "roundss";
    try b.print(a, "{s} ", .{mnem});
    try xmm(a, b, o.reg);
    try b.appendSlice(a, ", ");
    try xmm(a, b, o.rm);
    try b.print(a, ", {d}", .{d.byte()});
}

fn vexOp(a: std.mem.Allocator, b: *std.ArrayList(u8), d: *D, op: u8) !void {
    if (d.esc == 0x3A) { // vinsertf128 (0x18) / vextractf128 (0x19)
        const o = modrm(d);
        const imm = d.byte();
        if (op == 0x18) {
            try b.appendSlice(a, "vinsertf128 ");
            try ymm(a, b, o.reg);
            try b.appendSlice(a, ", ");
            try ymm(a, b, d.vvvv);
            try b.appendSlice(a, ", ");
            try xmm(a, b, o.rm);
        } else {
            try b.appendSlice(a, "vextractf128 ");
            try xmm(a, b, o.rm);
            try b.appendSlice(a, ", ");
            try ymm(a, b, o.reg);
        }
        return b.print(a, ", {d}", .{imm});
    }
    const o = modrm(d);
    const mnem: []const u8 = switch (op) {
        0x58 => "vaddps",
        0x5C => "vsubps",
        0x59 => "vmulps",
        0x5E => "vdivps",
        0x10 => "vmovups",
        0x11 => "vmovups",
        else => "vex?",
    };
    if (o.mem) { // vmovups spill/reload
        try b.print(a, "{s} ", .{mnem});
        if (op == 0x11) {
            try memOperand(a, b, o);
            try b.appendSlice(a, ", ");
            try ymm(a, b, o.reg);
        } else {
            try ymm(a, b, o.reg);
            try b.appendSlice(a, ", ");
            try memOperand(a, b, o);
        }
        return;
    }
    try b.print(a, "{s} ", .{mnem});
    try ymm(a, b, o.reg);
    // Three-operand forms carry src1 in vvvv; the two-operand vmovups does not.
    if (op != 0x10 and op != 0x11) {
        try b.appendSlice(a, ", ");
        try ymm(a, b, d.vvvv);
    }
    try b.appendSlice(a, ", ");
    try ymm(a, b, o.rm);
}

fn ymm(a: std.mem.Allocator, b: *std.ArrayList(u8), n: u8) !void {
    try b.print(a, "ymm{d}", .{n});
}

fn rmXmmOrMem(a: std.mem.Allocator, b: *std.ArrayList(u8), o: Operands) !void {
    if (o.mem) return memOperand(a, b, o);
    return xmm(a, b, o.rm);
}

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
            0xDF => "pandn",
            0xD4 => "paddq",
            0xFB => "psubq",
            0xFE => "paddd",
            0xFA => "psubd",
            0xF4 => "pmuludq",
            0xD5 => "pmullw",
            0xFC => "paddb",
            0xFD => "paddw",
            0xEC => "paddsb",
            0xED => "paddsw",
            0xE2 => "psrad",
            0xD6 => "movq",
            else => null,
        },
        else => switch (op) { // no mandatory prefix: packed / other
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

/// The `<size> ptr ` hint for an SSE memory operand: a scalar single is dword, a scalar double is
/// qword, everything packed/integer (incl. movdqu/movdqa where F3/66 are mandatory) is xmmword.
fn sseHint(mand: u8, op: u8) []const u8 {
    const scalar = switch (op) {
        0x10, 0x11, 0x51, 0x58, 0x59, 0x5A, 0x5C, 0x5E, 0x2A, 0x2C, 0x2E => true,
        else => false,
    };
    if (scalar and mand == 0xF3) return "dword ptr ";
    if (scalar and mand == 0xF2) return "qword ptr ";
    return "xmmword ptr ";
}

/// The ALU mnemonic for an opcode group index (bits[5:3] of the opcode, or a /digit).
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
        0xA => "p",
        0xB => "np",
        else => "?",
    };
}

fn relSign(v: i32) []const u8 {
    return if (v < 0) "-" else "+";
}

fn expectOne(inst: encode.Inst, expected: []const u8) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    _ = try decode(std.testing.allocator, &out, inst.slice(), 0);
    try std.testing.expectEqualStrings(expected, out.items);
}

test "round-trips integer ops" {
    try expectOne(encode.movImm(.rax, 42), "mov rax, 42");
    try expectOne(encode.movReg(.rax, .rbx), "mov rax, rbx");
    try expectOne(encode.add(.rax, .rbx), "add rax, rbx");
    try expectOne(encode.sub(.rcx, .rdx), "sub rcx, rdx");
    try expectOne(encode.xorr(.rax, .rax), "xor rax, rax");
    try expectOne(encode.imul(.rax, .rbx), "imul rax, rbx");
    try expectOne(encode.cmp(.rsi, .rdi), "cmp rsi, rdi");
    try expectOne(encode.aluImm(0, .rax, 5), "add rax, 5");
    try expectOne(encode.aluImm(5, .rbx, 7), "sub rbx, 7");
    try expectOne(encode.imulImm(.rax, .rbx, 3), "imul rax, rbx, 3");
    try expectOne(encode.shiftImm(4, .rax, 2), "shl rax, 2");
    try expectOne(encode.shrCl(.rbx), "shr rbx, cl");
    try expectOne(encode.idiv(.rcx), "idiv rcx");
    try expectOne(encode.cqo(), "cqo");
    try expectOne(encode.movImm64(.r8, 0x1234), "mov r8, 4660");
    try expectOne(encode.ret(), "ret");
    try expectOne(encode.syscall(), "syscall");
}

test "round-trips flags, branches, memory" {
    try expectOne(encode.setcc(.rax, .l), "setl al");
    try expectOne(encode.cmovcc(.rax, .rbx, .ne), "cmovne rax, rbx");
    try expectOne(encode.movzxByte(.rax, .rbx), "movzx rax, bl");
    try expectOne(encode.jcc(.e, 16), "je .+16");
    try expectOne(encode.jmp(-8), "jmp .-8");
    try expectOne(encode.callRel(32), "call .+32");
    try expectOne(encode.movFromStack(.rax, 8), "mov rax, qword ptr [rsp + 8]");
    try expectOne(encode.movToStack(16, .rbx), "mov qword ptr [rsp + 16], rbx");
    try expectOne(encode.movFromMem(.rax, .rbx, 4), "mov rax, qword ptr [rbx + 4]");
    try expectOne(encode.leaFromStack(.rax, 24), "lea rax, [rsp + 24]");
}

test "round-trips SSE and AVX" {
    try expectOne(encode.addss(.xmm0, .xmm1), "addss xmm0, xmm1");
    try expectOne(encode.divsd(.xmm2, .xmm3), "divsd xmm2, xmm3");
    try expectOne(encode.movdToXmm(.xmm0, .rax), "movd xmm0, eax");
    try expectOne(encode.movdFromXmm(.rax, .xmm0), "movd eax, xmm0");
    try expectOne(encode.cvtsi2ss(.xmm0, .rax), "cvtsi2ss xmm0, eax");
    try expectOne(encode.cvttss2si(.rax, .xmm0), "cvttss2si eax, xmm0");
    try expectOne(encode.sqrtss(.xmm0, .xmm1), "sqrtss xmm0, xmm1");
    try expectOne(encode.addps(.xmm0, .xmm1), "addps xmm0, xmm1");
    try expectOne(encode.ucomiss(.xmm0, .xmm1), "ucomiss xmm0, xmm1");
    try expectOne(encode.pshufd(.xmm0, .xmm1, 2), "pshufd xmm0, xmm1, 2");
    try expectOne(encode.insertps(.xmm0, .xmm1, 0x10), "insertps xmm0, xmm1, 16");
    try expectOne(encode.movssStore(8, .xmm1), "movss dword ptr [rsp + 8], xmm1");
    try expectOne(encode.vaddps(.xmm0, .xmm1, .xmm2), "vaddps ymm0, ymm1, ymm2");
    try expectOne(encode.vmovupsRR(.xmm0, .xmm1), "vmovups ymm0, ymm1");
    try expectOne(encode.vextractf128(.xmm0, .xmm1, 1), "vextractf128 xmm0, ymm1, 1");
}

test "formatModuleWithLines interleaves source-line markers before instructions" {
    const a = std.testing.allocator;
    // nop (0x90) at offset 0, ret (0xC3) at offset 1; line 5 then line 6.
    const code = [_]u8{ 0x90, 0xC3 };
    const lines = [_]AddrLine{ .{ .addr = 0, .line = 5 }, .{ .addr = 1, .line = 6 } };
    const out = try formatModuleWithLines(a, &code, &.{}, &lines);
    defer a.free(out);
    const at5 = std.mem.indexOf(u8, out, "; line 5") orelse return error.NoLine5;
    const at6 = std.mem.indexOf(u8, out, "; line 6") orelse return error.NoLine6;
    try std.testing.expect(at5 < at6); // line 5 marks the nop, line 6 marks the ret
    // plain formatModule (no lines) omits the markers.
    const plain = try formatModule(a, &code, &.{});
    defer a.free(plain);
    try std.testing.expect(std.mem.indexOf(u8, plain, "; line") == null);
}
