//! Minimal ELF reader: locate the `.text` section of an ELF object or executable and report
//! its machine so a disassembler can be chosen, plus its `.text` function symbols. Supports
//! both ELF32 and ELF64, little-endian (every target Vulcan emits). The complement of the ELF
//! writers in `object.zig` / `ld.zig`. Built on the `std.elf` header structs so there is no
//! hand-rolled offset arithmetic: the ELF32/ELF64 split is a single comptime parameter.

const std = @import("std");
const elf = std.elf;

pub const Error = error{ NotElf, Unsupported, Malformed, NoText };

/// ELF `e_machine` values for the architectures Vulcan disassembles.
pub const EM_386: u16 = @intFromEnum(elf.EM.@"386");
pub const EM_X86_64: u16 = @intFromEnum(elf.EM.X86_64);
pub const EM_AARCH64: u16 = @intFromEnum(elf.EM.AARCH64);
pub const EM_RISCV: u16 = @intFromEnum(elf.EM.RISCV);

/// `e_flags` bit indicating a RISC-V image uses the compressed (C) extension.
pub const EF_RISCV_RVC: u32 = 0x1;

/// The located code section: its machine and the raw `.text` bytes (borrowed from the input).
pub const Text = struct {
    machine: u16,
    is_64: bool,
    /// The virtual address `.text` loads at (`sh_addr`), or 0 for a relocatable object.
    addr: u64,
    /// The ELF header `e_flags`, which is arch-specific and carries `EF_RISCV_RVC` on RISC-V.
    flags: u32,
    bytes: []const u8,
};

/// A defined function symbol in `.text`: its name and its byte offset into that section.
pub const FuncSym = struct { name: []const u8, offset: u64 };

/// Any defined symbol, by absolute value (address): a function or a data object. Used to resolve
/// branch and `adrp` targets to names when disassembling.
pub const Symbol = struct { name: []const u8, addr: u64, size: u64, is_func: bool };

/// The `std.elf` header structs for a given width. `bytesToValue` decodes each straight from
/// the mapped image (little-endian, which the callers enforce and every Vulcan host uses).
fn Layout(comptime is_64: bool) type {
    return if (is_64) struct {
        const Ehdr = elf.Elf64_Ehdr;
        const Shdr = elf.Elf64_Shdr;
        const Sym = elf.Elf64_Sym;
    } else struct {
        const Ehdr = elf.Elf32_Ehdr;
        const Shdr = elf.Elf32_Shdr;
        const Sym = elf.Elf32_Sym;
    };
}

/// Decode a fixed-size POD header struct from `image` at byte offset `off`.
fn peek(comptime T: type, image: []const u8, off: u64) Error!T {
    const o: usize = @intCast(off);
    if (o + @sizeOf(T) > image.len) return error.Malformed;
    return std.mem.bytesToValue(T, image[o..][0..@sizeOf(T)]);
}

/// Validate the ELF magic and identify the class (`true` = ELF64). Little-endian only.
fn is64(image: []const u8) Error!bool {
    if (image.len < @sizeOf(elf.Elf32_Ehdr) or !std.mem.eql(u8, image[0..4], "\x7fELF")) return error.NotElf;
    if (image[elf.EI_DATA] != elf.ELFDATA2LSB) return error.Unsupported;
    return switch (image[elf.EI_CLASS]) {
        elf.ELFCLASS32 => false,
        elf.ELFCLASS64 => true,
        else => error.Malformed,
    };
}

/// The `[off, off+size)` slice of `image`, bounds-checked.
fn slice(image: []const u8, off: u64, size: u64) Error![]const u8 {
    const o: usize = @intCast(off);
    const s: usize = @intCast(size);
    if (o + s > image.len) return error.Malformed;
    return image[o .. o + s];
}

/// Read section header `idx`.
fn shdr(comptime L: type, image: []const u8, eh: L.Ehdr, idx: u16) Error!L.Shdr {
    return peek(L.Shdr, image, eh.e_shoff + @as(u64, idx) * eh.e_shentsize);
}

/// The section-name string table (section `e_shstrndx`).
fn sectionNames(comptime L: type, image: []const u8, eh: L.Ehdr) Error![]const u8 {
    if (eh.e_shoff == 0 or eh.e_shnum == 0 or eh.e_shstrndx >= eh.e_shnum) return error.Malformed;
    const sh = try shdr(L, image, eh, eh.e_shstrndx);
    return slice(image, sh.sh_offset, sh.sh_size);
}

fn nameAt(names: []const u8, off: u32) []const u8 {
    return if (off < names.len) std.mem.sliceTo(names[off..], 0) else "";
}

pub fn findText(image: []const u8) Error!Text {
    return if (try is64(image)) findTextGeneric(Layout(true), true, image) else findTextGeneric(Layout(false), false, image);
}

fn findTextGeneric(comptime L: type, comptime is_64: bool, image: []const u8) Error!Text {
    const eh = try peek(L.Ehdr, image, 0);
    const names = try sectionNames(L, image, eh);
    var i: u16 = 0;
    while (i < eh.e_shnum) : (i += 1) {
        const sh = try shdr(L, image, eh, i);
        if (!std.mem.eql(u8, nameAt(names, sh.sh_name), ".text")) continue;
        return .{
            .machine = @intFromEnum(eh.e_machine),
            .is_64 = is_64,
            .addr = sh.sh_addr,
            .flags = eh.e_flags,
            .bytes = try slice(image, sh.sh_offset, sh.sh_size),
        };
    }
    return error.NoText;
}

/// The raw bytes of the section named `name` (e.g. `.debug_line`), borrowed from `image`, or null if
/// absent. Lets a DWARF consumer pull a metadata section out of a real object.
pub fn sectionByName(image: []const u8, name: []const u8) Error!?[]const u8 {
    return if (try is64(image)) sectionByNameGeneric(Layout(true), image, name) else sectionByNameGeneric(Layout(false), image, name);
}

fn sectionByNameGeneric(comptime L: type, image: []const u8, name: []const u8) Error!?[]const u8 {
    const eh = try peek(L.Ehdr, image, 0);
    const names = try sectionNames(L, image, eh);
    var i: u16 = 0;
    while (i < eh.e_shnum) : (i += 1) {
        const sh = try shdr(L, image, eh, i);
        if (sh.sh_type == 8) continue; // SHT_NOBITS (.bss) has no file bytes
        if (std.mem.eql(u8, nameAt(names, sh.sh_name), name)) return try slice(image, sh.sh_offset, sh.sh_size);
    }
    return null;
}

pub fn functions(allocator: std.mem.Allocator, image: []const u8) (Error || std.mem.Allocator.Error)![]FuncSym {
    return if (try is64(image)) functionsGeneric(Layout(true), allocator, image) else functionsGeneric(Layout(false), allocator, image);
}

/// Extract the function symbols (`STT_FUNC`) defined in `.text` from the `.symtab`, sorted by
/// offset. Empty for a stripped file with no symbol table. Names borrow from `image`, and the slice
/// is caller-owned.
fn functionsGeneric(comptime L: type, allocator: std.mem.Allocator, image: []const u8) (Error || std.mem.Allocator.Error)![]FuncSym {
    const eh = try peek(L.Ehdr, image, 0);
    const names = try sectionNames(L, image, eh);

    // Locate `.text` (symbols must reference its index, and its load base rebases addresses to
    // section offsets) and the symbol table.
    var text_idx: ?u16 = null;
    var text_addr: u64 = 0;
    var symtab: ?L.Shdr = null;
    var i: u16 = 0;
    while (i < eh.e_shnum) : (i += 1) {
        const sh = try shdr(L, image, eh, i);
        if (std.mem.eql(u8, nameAt(names, sh.sh_name), ".text")) {
            text_idx = i;
            text_addr = sh.sh_addr;
        }
        if (sh.sh_type == elf.SHT_SYMTAB) symtab = sh;
    }
    const tx = text_idx orelse return error.NoText;
    const sym = symtab orelse return allocator.alloc(FuncSym, 0); // stripped

    if (sym.sh_entsize == 0 or sym.sh_link >= eh.e_shnum) return error.Malformed;
    const strs = try slice(image, (try shdr(L, image, eh, @intCast(sym.sh_link))).sh_offset, (try shdr(L, image, eh, @intCast(sym.sh_link))).sh_size);

    var list: std.ArrayList(FuncSym) = .empty;
    errdefer list.deinit(allocator);
    const count = sym.sh_size / sym.sh_entsize;
    var k: u64 = 0;
    while (k < count) : (k += 1) {
        const s = try peek(L.Sym, image, sym.sh_offset + k * sym.sh_entsize);
        if (s.st_type() != elf.STT_FUNC or s.st_shndx != tx) continue;
        const name = nameAt(strs, s.st_name);
        if (name.len == 0) continue;
        try list.append(allocator, .{ .name = name, .offset = s.st_value - text_addr });
    }
    const out = try list.toOwnedSlice(allocator);
    std.mem.sort(FuncSym, out, {}, struct {
        fn lt(_: void, a: FuncSym, b: FuncSym) bool {
            return a.offset < b.offset;
        }
    }.lt);
    return out;
}

pub fn symbols(allocator: std.mem.Allocator, image: []const u8) (Error || std.mem.Allocator.Error)![]Symbol {
    return if (try is64(image)) symbolsGeneric(Layout(true), allocator, image) else symbolsGeneric(Layout(false), allocator, image);
}

/// Collect every defined function or data symbol (by absolute address) from `.symtab`, sorted by
/// address. Undefined and non-code/data symbols are skipped. Empty for a stripped file.
fn symbolsGeneric(comptime L: type, allocator: std.mem.Allocator, image: []const u8) (Error || std.mem.Allocator.Error)![]Symbol {
    const eh = try peek(L.Ehdr, image, 0);
    if (eh.e_shoff == 0 or eh.e_shnum == 0) return error.Malformed;

    var symtab: ?L.Shdr = null;
    var i: u16 = 0;
    while (i < eh.e_shnum) : (i += 1) {
        const sh = try shdr(L, image, eh, i);
        if (sh.sh_type == elf.SHT_SYMTAB) symtab = sh;
    }
    const sym = symtab orelse return allocator.alloc(Symbol, 0);
    if (sym.sh_entsize == 0 or sym.sh_link >= eh.e_shnum) return error.Malformed;
    const str = try shdr(L, image, eh, @intCast(sym.sh_link));
    const strs = try slice(image, str.sh_offset, str.sh_size);

    var list: std.ArrayList(Symbol) = .empty;
    errdefer list.deinit(allocator);
    const count = sym.sh_size / sym.sh_entsize;
    var k: u64 = 0;
    while (k < count) : (k += 1) {
        const s = try peek(L.Sym, image, sym.sh_offset + k * sym.sh_entsize);
        const t = s.st_type();
        if (t != elf.STT_FUNC and t != elf.STT_OBJECT) continue;
        if (s.st_shndx == 0 or s.st_shndx >= 0xff00) continue; // undefined / reserved section
        const name = nameAt(strs, s.st_name);
        if (name.len == 0) continue;
        try list.append(allocator, .{ .name = name, .addr = s.st_value, .size = s.st_size, .is_func = t == elf.STT_FUNC });
    }
    const out = try list.toOwnedSlice(allocator);
    std.mem.sort(Symbol, out, {}, struct {
        fn lt(_: void, a: Symbol, b: Symbol) bool {
            return a.addr < b.addr;
        }
    }.lt);
    return out;
}

test "finds .text and machine in a hand-built ELF64" {
    const a = std.testing.allocator;
    const code = [_]u8{ 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };
    const image = try buildElf64(a, EM_AARCH64, 0x400000, &code);
    defer a.free(image);

    const t = try findText(image);
    try std.testing.expectEqual(EM_AARCH64, t.machine);
    try std.testing.expect(t.is_64);
    try std.testing.expectEqual(@as(u64, 0x400000), t.addr);
    try std.testing.expectEqualSlices(u8, &code, t.bytes);
}

test "rejects non-ELF input" {
    try std.testing.expectError(error.NotElf, findText("not an elf at all, just text padding to length!!"));
}

test "reads function symbols from a real cc-compiled .o" {
    const a = std.testing.allocator;
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "t.c", .data = "int helper(int x){return x+1;}\nint entry(void){return helper(41);}\n" });

    const cc = std.process.run(a, io, .{ .argv = &.{ "cc", "-c", "-O0", "-o", "t.o", "t.c" }, .cwd = .{ .dir = tmp.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest, // no C compiler here
        else => return err,
    };
    defer a.free(cc.stdout);
    defer a.free(cc.stderr);
    if (cc.term != .exited or cc.term.exited != 0) return error.SkipZigTest;

    const obj = try tmp.dir.readFileAlloc(io, "t.o", a, .limited(1 << 20));
    defer a.free(obj);

    const funcs = try functions(a, obj);
    defer a.free(funcs);
    // Both defined functions show up as symbols in .text.
    var saw_helper = false;
    var saw_entry = false;
    for (funcs) |f| {
        if (std.mem.eql(u8, f.name, "helper")) saw_helper = true;
        if (std.mem.eql(u8, f.name, "entry")) saw_entry = true;
    }
    try std.testing.expect(saw_helper);
    try std.testing.expect(saw_entry);
}

/// Build a minimal ELF64 with a single `.text` section (test helper): header, `.text` data,
/// `.shstrtab`, then a 3-entry section table (null, `.text`, `.shstrtab`).
fn buildElf64(allocator: std.mem.Allocator, machine: u16, text_addr: u64, code: []const u8) ![]u8 {
    var shstr: std.ArrayList(u8) = .empty;
    defer shstr.deinit(allocator);
    try shstr.append(allocator, 0);
    const text_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".text\x00");
    const shstr_name: u32 = @intCast(shstr.items.len);
    try shstr.appendSlice(allocator, ".shstrtab\x00");

    const text_off: u64 = 64;
    const shstr_off: u64 = text_off + code.len;
    const shoff: u64 = shstr_off + shstr.items.len;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, &.{ 0x7f, 'E', 'L', 'F', 2, 1, 1, 0 });
    try out.appendSlice(allocator, &(.{0} ** 8));
    try appendInt(allocator, &out, u16, 1); // e_type = ET_REL
    try appendInt(allocator, &out, u16, machine);
    try appendInt(allocator, &out, u32, 1);
    try appendInt(allocator, &out, u64, 0); // e_entry
    try appendInt(allocator, &out, u64, 0); // e_phoff
    try appendInt(allocator, &out, u64, shoff);
    try appendInt(allocator, &out, u32, 0);
    try appendInt(allocator, &out, u16, 64); // e_ehsize
    try appendInt(allocator, &out, u16, 0);
    try appendInt(allocator, &out, u16, 0);
    try appendInt(allocator, &out, u16, 64); // e_shentsize
    try appendInt(allocator, &out, u16, 3); // e_shnum
    try appendInt(allocator, &out, u16, 2); // e_shstrndx

    try out.appendSlice(allocator, code);
    try out.appendSlice(allocator, shstr.items);

    try appendShdr64(allocator, &out, 0, 0, 0, 0, 0); // null
    try appendShdr64(allocator, &out, text_name, 1, text_addr, text_off, code.len); // .text PROGBITS
    try appendShdr64(allocator, &out, shstr_name, 3, 0, shstr_off, shstr.items.len); // .shstrtab
    return out.toOwnedSlice(allocator);
}

fn appendInt(allocator: std.mem.Allocator, out: *std.ArrayList(u8), comptime T: type, v: T) !void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, v, .little);
    try out.appendSlice(allocator, &buf);
}

fn appendShdr64(allocator: std.mem.Allocator, out: *std.ArrayList(u8), name: u32, sh_type: u32, addr: u64, offset: u64, size: u64) !void {
    try appendInt(allocator, out, u32, name);
    try appendInt(allocator, out, u32, sh_type);
    try appendInt(allocator, out, u64, 0); // sh_flags
    try appendInt(allocator, out, u64, addr); // sh_addr
    try appendInt(allocator, out, u64, offset);
    try appendInt(allocator, out, u64, size);
    try appendInt(allocator, out, u32, 0);
    try appendInt(allocator, out, u32, 0);
    try appendInt(allocator, out, u64, 0);
    try appendInt(allocator, out, u64, 0);
}
