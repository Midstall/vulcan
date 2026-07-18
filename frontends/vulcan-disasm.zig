//! vulcan-disasm: disassemble a binary to canonical text assembly.
//!
//! It auto-detects the input by its magic:
//!   * an ELF object/executable -> the `.text` machine code, decoded for its `e_machine`
//!     (aarch64 / riscv64 / x86-64 / x86)
//!   * a SPIR-V binary (`.spv`) -> a spirv-dis-style `%id = OpName ...` listing
//!
//! Usage:
//!   vulcan-disasm <file>        an ELF or SPIR-V binary
//!
//! The listing goes to stderr, one instruction per line.

const std = @import("std");
const target = @import("vulcan-target");
const spirv = @import("vulcan-spirv");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip(); // argv0
    const input = it.next() orelse return usage();

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, input, allocator, .limited(64 * 1024 * 1024));

    const listing = if (isElf(bytes))
        try disassembleElf(allocator, bytes)
    else if (isSpirv(bytes))
        try spirv.disassembleBytes(allocator, bytes)
    else {
        std.debug.print("vulcan-disasm: '{s}' is not an ELF or SPIR-V binary\n", .{input});
        return error.NotBinary;
    };

    // The disassembly is the program's output: write it to stdout, not stderr.
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.writeAll(listing);
    try w.interface.flush();
}

fn usage() error{Usage} {
    std.debug.print("usage: vulcan-disasm <file>   (an ELF or SPIR-V binary)\n", .{});
    return error.Usage;
}

fn isElf(b: []const u8) bool {
    return b.len >= 4 and std.mem.eql(u8, b[0..4], "\x7fELF");
}

fn isSpirv(b: []const u8) bool {
    if (b.len < 4) return false;
    const w = std.mem.readInt(u32, b[0..4], .little);
    return w == spirv.binary.magic or @byteSwap(w) == spirv.binary.magic;
}

/// Locate an ELF's `.text`, pick the decoder for its `e_machine`, and return a symbolized
/// listing (functions labeled and calls annotated from `.symtab`; falls back to a plain listing
/// for a stripped file).
fn disassembleElf(allocator: std.mem.Allocator, image: []const u8) ![]u8 {
    const elf = target.elf_read;
    const t = elf.findText(image) catch |err| {
        std.debug.print("vulcan-disasm: {s}\n", .{@errorName(err)});
        return error.BadElf;
    };
    // aarch64 and riscv64 get the richer objdump-style listing: absolute addresses and
    // branch/adrp targets resolved against the full symbol table (functions and data).
    if (t.machine == elf.EM_AARCH64) {
        const syms = try elf.symbols(allocator, image);
        const addr_syms = try allocator.alloc(target.aarch64.disasm.AddrSym, syms.len);
        for (syms, 0..) |s, i| addr_syms[i] = .{ .name = s.name, .addr = s.addr, .size = s.size, .is_func = s.is_func };
        // With embedded DWARF line info, interleave source-line markers (objdump -S style).
        const lines = try debugLines(allocator, image, target.aarch64.disasm.AddrLine);
        return target.aarch64.disasm.formatElfWithLines(allocator, try asWords(allocator, t.bytes), t.addr, addr_syms, lines);
    }
    if (t.machine == elf.EM_RISCV) {
        const syms = try elf.symbols(allocator, image);
        const addr_syms = try allocator.alloc(target.riscv64.disasm.AddrSym, syms.len);
        for (syms, 0..) |s, i| addr_syms[i] = .{ .name = s.name, .addr = s.addr, .size = s.size, .is_func = s.is_func };
        // Variable-length (RVC) stepping only when the target enables the C extension.
        const rvc = t.flags & elf.EF_RISCV_RVC != 0;
        const lines = try debugLines(allocator, image, target.riscv64.disasm.AddrLine);
        return target.riscv64.disasm.formatElfWithLines(allocator, t.bytes, t.addr, addr_syms, rvc, lines);
    }

    const funcs = try elf.functions(allocator, image);
    switch (t.machine) {
        elf.EM_AARCH64, elf.EM_RISCV => unreachable, // handled above
        elf.EM_X86_64 => return target.x86_64.disasm.formatModuleWithLines(allocator, t.bytes, try byteSyms(allocator, target.x86_64.disasm.Sym, funcs), try debugLines(allocator, image, target.x86_64.disasm.AddrLine)),
        elf.EM_386 => return target.x86.disasm.formatModuleWithLines(allocator, t.bytes, try byteSyms(allocator, target.x86.disasm.Sym, funcs), try debugLines(allocator, image, target.x86.disasm.AddrLine)),
        else => {
            std.debug.print("vulcan-disasm: unsupported ELF machine 0x{x}\n", .{t.machine});
            return error.Unsupported;
        },
    }
}

/// Decode the ELF's `.debug_line` (if present) into address-keyed source-line rows for `AL`
/// (a backend's `AddrLine`), dropping end-of-sequence rows and sorting by address. Empty if the
/// object has no line info.
fn debugLines(allocator: std.mem.Allocator, image: []const u8, comptime AL: type) ![]AL {
    const dl = (try target.elf_read.sectionByName(image, ".debug_line")) orelse return &.{};
    const rows = try target.dwarf.decodeLine(allocator, dl);
    var out: std.ArrayList(AL) = .empty;
    for (rows) |r| if (!r.end_sequence) try out.append(allocator, .{ .addr = r.address, .line = r.line });
    std.mem.sort(AL, out.items, {}, struct {
        fn lt(_: void, a: AL, b: AL) bool {
            return a.addr < b.addr;
        }
    }.lt);
    return out.toOwnedSlice(allocator);
}

/// Convert `.text` function symbols to a byte-addressed backend's `Sym` (x86).
fn byteSyms(allocator: std.mem.Allocator, comptime Sym: type, funcs: []const target.elf_read.FuncSym) ![]Sym {
    const out = try allocator.alloc(Sym, funcs.len);
    for (funcs, 0..) |f, i| out[i] = .{ .name = f.name, .offset = @intCast(f.offset) };
    return out;
}

/// Reinterpret a byte-aligned instruction section as little-endian 32-bit words (fixed-width
/// ISAs). The section length is a multiple of 4 for well-formed aarch64/riscv64 `.text`.
fn asWords(allocator: std.mem.Allocator, bytes: []const u8) ![]u32 {
    const words = try allocator.alloc(u32, bytes.len / 4);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
    return words;
}
