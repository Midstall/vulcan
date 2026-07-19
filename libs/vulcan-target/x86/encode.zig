//! x86 (32-bit / i386) instruction encoding. Like x86-64 but with no REX prefix and the
//! 32-bit register file. The default operand size is 32-bit. Validated by execution under
//! qemu-i386 plus the encoding tests here.

const std = @import("std");

/// A 32-bit general register (3-bit hardware encoding).
pub const Reg = enum(u3) {
    eax = 0,
    ecx = 1,
    edx = 2,
    ebx = 3,
    esp = 4,
    ebp = 5,
    esi = 6,
    edi = 7,
};

fn n(r: Reg) u8 {
    return @intFromEnum(r);
}

/// An encoded instruction: up to 15 bytes.
pub const Inst = struct {
    bytes: [15]u8 = undefined,
    len: u8 = 0,

    fn of(b: []const u8) Inst {
        var i: Inst = .{ .len = @intCast(b.len) };
        @memcpy(i.bytes[0..b.len], b);
        return i;
    }
    pub fn slice(self: *const Inst) []const u8 {
        return self.bytes[0..self.len];
    }
};

/// Register-direct ModRM byte (mod = 11).
fn modrm(reg: Reg, rm: Reg) u8 {
    return 0xC0 | (n(reg) << 3) | n(rm);
}

/// A two-register ALU op (`/r` form): `reg` is the source, `rm` the destination.
fn aluRR(opcode: u8, src: Reg, dst: Reg) Inst {
    return Inst.of(&.{ opcode, modrm(src, dst) });
}

fn imm32(v: i32) [4]u8 {
    const u: u32 = @bitCast(v);
    return .{ @truncate(u), @truncate(u >> 8), @truncate(u >> 16), @truncate(u >> 24) };
}

/// `mov dst, imm32` (B8+r: the short load-immediate form).
pub fn movImm(dst: Reg, v: i32) Inst {
    const b = imm32(v);
    return Inst.of(&.{ 0xB8 | n(dst), b[0], b[1], b[2], b[3] });
}

/// `mov dst, src` (89 /r).
pub fn movReg(dst: Reg, src: Reg) Inst {
    return aluRR(0x89, src, dst);
}

/// `mov dst, [esp + disp8]` (8B /r with a SIB byte, since the ESP base requires SIB):
/// load a stack slot. Used to read cdecl stack arguments.
pub fn movFromStack(dst: Reg, disp8: u8) Inst {
    return Inst.of(&.{ 0x8B, 0x44 | (n(dst) << 3), 0x24, disp8 }); // mod=01 rm=100(SIB), SIB=esp base
}

/// `mov dst, [esp + disp32]` (8B /r, mod=10): load a spill slot.
pub fn stackLoad(dst: Reg, disp: i32) Inst {
    const b = imm32(disp);
    return Inst.of(&.{ 0x8B, 0x84 | (n(dst) << 3), 0x24, b[0], b[1], b[2], b[3] });
}

/// `mov [esp + disp32], src` (89 /r, mod=10): store to a spill slot.
pub fn stackStore(disp: i32, src: Reg) Inst {
    const b = imm32(disp);
    return Inst.of(&.{ 0x89, 0x84 | (n(src) << 3), 0x24, b[0], b[1], b[2], b[3] });
}

/// A `[base + disp32]` memory access (mod=10, disp32 always explicit so any base register,
/// including ebp, is safe). `data` is the ModRM `reg` field (dst for a load, src for a store).
/// A base of esp (rm low bits = 100) is not directly encodable and needs the 0x24 SIB byte.
/// `prefix` is an optional operand-size override (0x66), `opc` the one or two opcode bytes.
fn memOp(comptime prefix: []const u8, comptime opc: []const u8, data: Reg, base: Reg, disp: i32) Inst {
    const b = imm32(disp);
    const modrm_byte: u8 = 0x80 | ((n(data) & 7) << 3) | (n(base) & 7);
    const sib = (n(base) & 7) == 4; // esp base needs SIB 0x24
    var buf: [prefix.len + opc.len + 2 + 4]u8 = undefined;
    var i: usize = 0;
    inline for (prefix) |p| {
        buf[i] = p;
        i += 1;
    }
    inline for (opc) |o| {
        buf[i] = o;
        i += 1;
    }
    buf[i] = modrm_byte;
    i += 1;
    if (sib) {
        buf[i] = 0x24;
        i += 1;
    }
    buf[i] = b[0];
    buf[i + 1] = b[1];
    buf[i + 2] = b[2];
    buf[i + 3] = b[3];
    i += 4;
    return Inst.of(buf[0..i]);
}

/// `mov dst, [base + disp32]` (8B /r): 32-bit load.
pub fn movFromMem32(dst: Reg, base: Reg, disp: i32) Inst {
    return memOp(&.{}, &.{0x8B}, dst, base, disp);
}

/// `mov [base + disp32], src` (89 /r): 32-bit store.
pub fn movToMem32(base: Reg, disp: i32, src: Reg) Inst {
    return memOp(&.{}, &.{0x89}, src, base, disp);
}

/// `mov word ptr [base + disp32], src` (66 89 /r): 16-bit store (writes the low 16 bits).
pub fn movToMem16(base: Reg, disp: i32, src: Reg) Inst {
    return memOp(&.{0x66}, &.{0x89}, src, base, disp);
}

/// `mov byte ptr [base + disp32], src` (88 /r): 8-bit store (writes the low byte). The caller
/// is responsible for `src` being a byte-addressable register (eax/ecx/edx/ebx).
pub fn movToMem8(base: Reg, disp: i32, src: Reg) Inst {
    return memOp(&.{}, &.{0x88}, src, base, disp);
}

/// `movzx dst, byte ptr [base + disp32]` (0F B6 /r): load a byte, zero-extend into r32.
pub fn movzxByteFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    return memOp(&.{}, &.{ 0x0F, 0xB6 }, dst, base, disp);
}

/// `movsx dst, byte ptr [base + disp32]` (0F BE /r): load a byte, sign-extend into r32.
pub fn movsxByteFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    return memOp(&.{}, &.{ 0x0F, 0xBE }, dst, base, disp);
}

/// `movzx dst, word ptr [base + disp32]` (0F B7 /r): load 16 bits, zero-extend into r32.
pub fn movzxWordFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    return memOp(&.{}, &.{ 0x0F, 0xB7 }, dst, base, disp);
}

/// `movsx dst, word ptr [base + disp32]` (0F BF /r): load 16 bits, sign-extend into r32.
pub fn movsxWordFromMem(dst: Reg, base: Reg, disp: i32) Inst {
    return memOp(&.{}, &.{ 0x0F, 0xBF }, dst, base, disp);
}

/// `lea dst, [esp + disp32]` (8D /r): materialize a stack address (e.g. an alloca slot).
pub fn leaFromStack(dst: Reg, disp: i32) Inst {
    return memOp(&.{}, &.{0x8D}, dst, .esp, disp);
}

/// `add dst, src` (01 /r).
pub fn add(dst: Reg, src: Reg) Inst {
    return aluRR(0x01, src, dst);
}

/// `sub dst, src` (29 /r).
pub fn sub(dst: Reg, src: Reg) Inst {
    return aluRR(0x29, src, dst);
}

/// `and dst, src` (21 /r).
pub fn andr(dst: Reg, src: Reg) Inst {
    return aluRR(0x21, src, dst);
}

/// `or dst, src` (09 /r).
pub fn orr(dst: Reg, src: Reg) Inst {
    return aluRR(0x09, src, dst);
}

/// `xor dst, src` (31 /r).
pub fn xorr(dst: Reg, src: Reg) Inst {
    return aluRR(0x31, src, dst);
}

/// `imul dst, src` (0F AF /r: dst = dst * src).
pub fn imul(dst: Reg, src: Reg) Inst {
    return Inst.of(&.{ 0x0F, 0xAF, modrm(dst, src) });
}

/// `push imm32` (68 id).
pub fn pushImm(v: i32) Inst {
    const b = imm32(v);
    return Inst.of(&.{ 0x68, b[0], b[1], b[2], b[3] });
}

/// `push reg` (50+r).
pub fn pushReg(r: Reg) Inst {
    return Inst.of(&.{0x50 | n(r)});
}

/// `cmp a, b` (39 /r): compute `a - b`, set flags.
pub fn cmp(a: Reg, b: Reg) Inst {
    return aluRR(0x39, b, a);
}

/// `test a, b` (85 /r): compute `a & b`, set flags.
pub fn testReg(a: Reg, b: Reg) Inst {
    return aluRR(0x85, b, a);
}

/// A condition code (low nibble of Jcc/SETcc).
pub const Cond = enum(u8) {
    e = 0x4,
    ne = 0x5,
    l = 0xC,
    ge = 0xD,
    le = 0xE,
    g = 0xF,
    b = 0x2,
    ae = 0x3,
    be = 0x6,
    a = 0x7,
};

/// `setcc dst8` (0F 90+cc). In 32-bit there is no REX, so `dst` MUST be one of
/// EAX/ECX/EDX/EBX (registers 0..3, whose low byte is al/cl/dl/bl). The allocator
/// constrains boolean values to those registers.
pub fn setcc(dst: Reg, cond: Cond) Inst {
    return Inst.of(&.{ 0x0F, 0x90 | @intFromEnum(cond), 0xC0 | n(dst) });
}

/// `movzx dst, src8` (0F B6 /r): zero-extend `src`'s low byte. `src` must be 0..3.
pub fn movzxByte(dst: Reg, src: Reg) Inst {
    return Inst.of(&.{ 0x0F, 0xB6, modrm(dst, src) });
}

/// `jcc rel32` (0F 80+cc cd).
pub fn jcc(cond: Cond, rel: i32) Inst {
    const b = imm32(rel);
    return Inst.of(&.{ 0x0F, 0x80 | @intFromEnum(cond), b[0], b[1], b[2], b[3] });
}

/// `jmp rel32` (E9 cd).
pub fn jmp(rel: i32) Inst {
    const b = imm32(rel);
    return Inst.of(&.{ 0xE9, b[0], b[1], b[2], b[3] });
}

/// An ALU op against a 32-bit immediate (81 /digit id): ADD=/0, OR=/1, AND=/4,
/// SUB=/5, XOR=/6.
pub fn aluImm(digit: u3, dst: Reg, imm: i32) Inst {
    const b = imm32(imm);
    return Inst.of(&.{ 0x81, 0xC0 | (@as(u8, digit) << 3) | n(dst), b[0], b[1], b[2], b[3] });
}

/// `imul dst, src, imm32` (69 /r id): dst = src * imm.
pub fn imulImm(dst: Reg, src: Reg, imm: i32) Inst {
    const b = imm32(imm);
    return Inst.of(&.{ 0x69, modrm(dst, src), b[0], b[1], b[2], b[3] });
}

/// A shift of `dst` by an immediate count (C1 /digit ib): SHL=/4, SHR=/5, SAR=/7.
pub fn shiftImm(digit: u3, dst: Reg, imm8: u8) Inst {
    return Inst.of(&.{ 0xC1, 0xC0 | (@as(u8, digit) << 3) | n(dst), imm8 });
}

/// `cdq` (99): sign-extend EAX into EDX:EAX (the dividend for `idiv`).
pub fn cdq() Inst {
    return Inst.of(&.{0x99});
}

/// `idiv src` (F7 /7): signed EDX:EAX / src, quotient -> EAX, remainder -> EDX.
pub fn idiv(src: Reg) Inst {
    return Inst.of(&.{ 0xF7, 0xC0 | (7 << 3) | n(src) });
}

/// `div src` (F7 /6): unsigned EDX:EAX / src (clear EDX first).
pub fn divu(src: Reg) Inst {
    return Inst.of(&.{ 0xF7, 0xC0 | (6 << 3) | n(src) });
}

fn shiftCl(digit: u8, dst: Reg) Inst {
    return Inst.of(&.{ 0xD3, 0xC0 | (digit << 3) | n(dst) });
}
pub fn shlCl(dst: Reg) Inst {
    return shiftCl(4, dst);
}
pub fn shrCl(dst: Reg) Inst {
    return shiftCl(5, dst);
}
pub fn sarCl(dst: Reg) Inst {
    return shiftCl(7, dst);
}

/// `ret` (near return, cdecl is caller-cleaned).
pub fn ret() Inst {
    return Inst.of(&.{0xC3});
}

/// `call rel32` (E8 cd): a relative call. `rel` is from the end of this instruction.
pub fn callRel(rel: i32) Inst {
    const b = imm32(rel);
    return Inst.of(&.{ 0xE8, b[0], b[1], b[2], b[3] });
}

/// `int 0x80`: the i386 Linux system-call gate.
pub fn int80() Inst {
    return Inst.of(&.{ 0xCD, 0x80 });
}

test "known i386 encodings" {
    try std.testing.expectEqualSlices(u8, &.{ 0xB8, 0x2A, 0x00, 0x00, 0x00 }, movImm(.eax, 42).slice()); // mov eax, 42
    try std.testing.expectEqualSlices(u8, &.{ 0x89, 0xD8 }, movReg(.eax, .ebx).slice()); // mov eax, ebx
    try std.testing.expectEqualSlices(u8, &.{ 0x01, 0xD8 }, add(.eax, .ebx).slice()); // add eax, ebx
    try std.testing.expectEqualSlices(u8, &.{ 0x29, 0xD8 }, sub(.eax, .ebx).slice()); // sub eax, ebx
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0xAF, 0xC3 }, imul(.eax, .ebx).slice()); // imul eax, ebx
    try std.testing.expectEqualSlices(u8, &.{ 0x8B, 0x44, 0x24, 0x04 }, movFromStack(.eax, 4).slice()); // mov eax, [esp+4]
    try std.testing.expectEqualSlices(u8, &.{0xC3}, ret().slice());
    try std.testing.expectEqualSlices(u8, &.{ 0xCD, 0x80 }, int80().slice());
}

test "x86-32 reg+disp32 load and store encoders match known bytes" {
    // mov eax, [ecx+4]: modrm = 0x80 | (eax<<3) | ecx = 0x80 | 0 | 1 = 0x81, no SIB (ecx isn't esp)
    try std.testing.expectEqualSlices(u8, &.{ 0x8B, 0x81, 0x04, 0x00, 0x00, 0x00 }, movFromMem32(.eax, .ecx, 4).slice());
    // mov [ecx+8], edx: modrm = 0x80 | (edx<<3) | ecx = 0x80 | 0x10 | 1 = 0x91
    try std.testing.expectEqualSlices(u8, &.{ 0x89, 0x91, 0x08, 0x00, 0x00, 0x00 }, movToMem32(.ecx, 8, .edx).slice());
    // mov eax, [esp+16]: esp base (rm=100) needs the 0x24 SIB byte, modrm = 0x80 | 0 | 4 = 0x84
    try std.testing.expectEqualSlices(u8, &.{ 0x8B, 0x84, 0x24, 0x10, 0x00, 0x00, 0x00 }, movFromMem32(.eax, .esp, 16).slice());
    // mov word ptr [ebx+2], si: 0x66 operand-size prefix, modrm = 0x80 | (esi<<3) | ebx = 0x80 | 0x30 | 3 = 0xB3
    try std.testing.expectEqualSlices(u8, &.{ 0x66, 0x89, 0xB3, 0x02, 0x00, 0x00, 0x00 }, movToMem16(.ebx, 2, .esi).slice());
    // mov byte ptr [ebx+1], al: modrm = 0x80 | (eax<<3) | ebx = 0x80 | 0 | 3 = 0x83
    try std.testing.expectEqualSlices(u8, &.{ 0x88, 0x83, 0x01, 0x00, 0x00, 0x00 }, movToMem8(.ebx, 1, .eax).slice());
    // movzx eax, byte ptr [ecx]: modrm = 0x80 | (eax<<3) | ecx = 0x80 | 0 | 1 = 0x81
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0xB6, 0x81, 0x00, 0x00, 0x00, 0x00 }, movzxByteFromMem(.eax, .ecx, 0).slice());
    // movsx edx, word ptr [ebx-4]: modrm = 0x80 | (edx<<3) | ebx = 0x80 | 0x10 | 3 = 0x93, disp -4 as u32 LE
    try std.testing.expectEqualSlices(u8, &.{ 0x0F, 0xBF, 0x93, 0xFC, 0xFF, 0xFF, 0xFF }, movsxWordFromMem(.edx, .ebx, -4).slice());
    // lea eax, [esp+16]: esp base needs the SIB byte, modrm = 0x80 | (eax<<3) | esp = 0x84
    try std.testing.expectEqualSlices(u8, &.{ 0x8D, 0x84, 0x24, 0x10, 0x00, 0x00, 0x00 }, leaFromStack(.eax, 16).slice());
}
