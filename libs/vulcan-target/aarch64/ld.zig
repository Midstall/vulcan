//! Vulcan's own AArch64 object linker: parse ELF64 AArch64 relocatable objects (as
//! emitted by object.zig), concatenate their `.text`, and resolve cross-object
//! `R_AARCH64_CALL26` relocations into one code image. Calls are PC-relative, so the
//! image is position-independent (maps at any address). Only `.text` and CALL26 are
//! handled, matching what the selector emits.

const std = @import("std");
const encode = @import("encode.zig");
const object = @import("object.zig");

pub const Error = std.mem.Allocator.Error || error{
    MalformedObject,
    UndefinedSymbol,
    DuplicateSymbol,
    RelocationOutOfRange,
    UnsupportedReloc,
};

const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_RELA: u32 = 4;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHN_UNDEF: u16 = 0;
const EM_AARCH64: u16 = 183;

fn alignUp(v: u64, a: u64) u64 {
    return (v + a - 1) & ~(a - 1);
}

fn rdInt(comptime T: type, buf: []const u8, off: u64) Error!T {
    const o: usize = @intCast(off);
    if (o + @sizeOf(T) > buf.len) return error.MalformedObject;
    return std.mem.readInt(T, buf[o..][0..@sizeOf(T)], .little);
}

fn strAt(strtab: []const u8, off: u32) Error![]const u8 {
    if (off >= strtab.len) return error.MalformedObject;
    const end = std.mem.indexOfScalarPos(u8, strtab, off, 0) orelse return error.MalformedObject;
    return strtab[off..end];
}

const ObjSymbol = struct { name: []const u8, value: u64, defined: bool, local: bool };

const ParsedObject = struct {
    text: []const u8,
    symbols: []ObjSymbol,
    relocs: []object.Reloc,

    fn deinit(self: *ParsedObject, allocator: std.mem.Allocator) void {
        allocator.free(self.symbols);
        allocator.free(self.relocs);
    }
};

/// Parse one ELF64 AArch64 relocatable object into its `.text`, symbols, and
/// relocations. Section roles are found by type/flags. Names borrow from `buf`.
fn parseObject(allocator: std.mem.Allocator, buf: []const u8) Error!ParsedObject {
    if (buf.len < 64) return error.MalformedObject;
    if (!std.mem.eql(u8, buf[0..4], "\x7fELF")) return error.MalformedObject;
    if (buf[4] != 2 or buf[5] != 1) return error.MalformedObject; // ELFCLASS64, LSB
    if (try rdInt(u16, buf, 18) != EM_AARCH64) return error.MalformedObject;

    const shoff = try rdInt(u64, buf, 40);
    const shentsize = try rdInt(u16, buf, 58);
    const shnum = try rdInt(u16, buf, 60);

    var text: []const u8 = &.{};
    var symtab_ndx: ?u16 = null;
    var rela_ndx: ?u16 = null;

    var i: u16 = 0;
    while (i < shnum) : (i += 1) {
        const hdr = shoff + @as(u64, i) * shentsize;
        const typ = try rdInt(u32, buf, hdr + 4);
        const flags = try rdInt(u64, buf, hdr + 8);
        const sh_off = try rdInt(u64, buf, hdr + 24);
        const sh_size = try rdInt(u64, buf, hdr + 32);
        if (typ == SHT_PROGBITS and (flags & SHF_EXECINSTR) != 0) {
            if (@as(usize, @intCast(sh_off + sh_size)) > buf.len) return error.MalformedObject;
            text = buf[@intCast(sh_off)..@intCast(sh_off + sh_size)];
        }
        if (typ == SHT_SYMTAB) symtab_ndx = i;
        if (typ == SHT_RELA) rela_ndx = i;
    }
    const si = symtab_ndx orelse return error.MalformedObject;

    const sym_hdr = shoff + @as(u64, si) * shentsize;
    const sym_off = try rdInt(u64, buf, sym_hdr + 24);
    const sym_size = try rdInt(u64, buf, sym_hdr + 32);
    const sym_link = try rdInt(u32, buf, sym_hdr + 40);
    const str_hdr = shoff + @as(u64, sym_link) * shentsize;
    const str_off = try rdInt(u64, buf, str_hdr + 24);
    const str_size = try rdInt(u64, buf, str_hdr + 32);
    if (@as(usize, @intCast(str_off + str_size)) > buf.len) return error.MalformedObject;
    const strtab = buf[@intCast(str_off)..@intCast(str_off + str_size)];

    const sym_count: usize = @intCast(sym_size / 24);
    var symbols = try allocator.alloc(ObjSymbol, sym_count);
    errdefer allocator.free(symbols);
    var k: usize = 0;
    while (k < sym_count) : (k += 1) {
        const e = sym_off + @as(u64, k) * 24;
        const st_name = try rdInt(u32, buf, e + 0);
        const st_info = try rdInt(u8, buf, e + 4);
        const st_shndx = try rdInt(u16, buf, e + 6);
        const st_value = try rdInt(u64, buf, e + 8);
        symbols[k] = .{
            .name = if (st_name == 0) "" else try strAt(strtab, st_name),
            .value = st_value,
            .defined = st_shndx != SHN_UNDEF,
            .local = (st_info >> 4) == 0, // STB_LOCAL
        };
    }

    var relocs: []object.Reloc = &.{};
    if (rela_ndx) |ri| {
        const rela_hdr = shoff + @as(u64, ri) * shentsize;
        const rela_off = try rdInt(u64, buf, rela_hdr + 24);
        const rela_size = try rdInt(u64, buf, rela_hdr + 32);
        const rela_count: usize = @intCast(rela_size / 24);
        relocs = try allocator.alloc(object.Reloc, rela_count);
        errdefer allocator.free(relocs);
        var r: usize = 0;
        while (r < rela_count) : (r += 1) {
            const e = rela_off + @as(u64, r) * 24;
            const r_offset = try rdInt(u64, buf, e + 0);
            const r_info = try rdInt(u64, buf, e + 8);
            const r_addend = try rdInt(i64, buf, e + 16);
            const typ: u32 = @truncate(r_info & 0xffffffff);
            if (typ != @intFromEnum(object.RelocType.call26)) return error.UnsupportedReloc;
            relocs[r] = .{ .offset = r_offset, .symbol = @intCast(r_info >> 32), .type = .call26, .addend = r_addend };
        }
    }

    return .{ .text = text, .symbols = symbols, .relocs = relocs };
}

/// A resolved symbol in the linked image: its name and final absolute address.
pub const ResolvedSymbol = struct { name: []const u8, address: u64 };

/// A linked code image: the relocated `.text` (in input order) plus each defined
/// symbol's absolute address. The code is meant to load at `base`.
pub const Image = struct {
    code: []u8,
    symbols: []ResolvedSymbol,
    base: u64,

    pub fn deinit(self: *Image, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        for (self.symbols) |s| allocator.free(s.name);
        allocator.free(self.symbols);
    }

    pub fn addressOf(self: *const Image, name: []const u8) ?u64 {
        for (self.symbols) |s| if (std.mem.eql(u8, s.name, name)) return s.address;
        return null;
    }
};

/// Link a set of relocatable objects into one image, laid out in the order given
/// with each `.text` concatenated at `base`. Every `R_AARCH64_CALL26` (a `bl`) is
/// resolved. Undefined call targets are an error. The caller owns the image.
pub fn linkObjects(allocator: std.mem.Allocator, objs: []const []const u8, base: u64) Error!Image {
    var parsed = try allocator.alloc(ParsedObject, objs.len);
    var parsed_n: usize = 0;
    defer {
        var i: usize = 0;
        while (i < parsed_n) : (i += 1) parsed[i].deinit(allocator);
        allocator.free(parsed);
    }
    for (objs, 0..) |obj_bytes, oi| {
        parsed[oi] = try parseObject(allocator, obj_bytes);
        parsed_n = oi + 1;
    }

    // Pack each object's `.text` back to back (4-byte aligned).
    const text_at = try allocator.alloc(u64, objs.len);
    defer allocator.free(text_at);
    var total: u64 = 0;
    for (0..parsed_n) |oi| {
        text_at[oi] = alignUp(total, 4);
        total = text_at[oi] + parsed[oi].text.len;
    }

    // Resolve each defined, non-local symbol to its image offset.
    var symbols: std.ArrayList(ResolvedSymbol) = .empty;
    errdefer {
        for (symbols.items) |s| allocator.free(s.name);
        symbols.deinit(allocator);
    }
    for (0..parsed_n) |oi| {
        for (parsed[oi].symbols) |sym| {
            if (!sym.defined or sym.local or sym.name.len == 0) continue;
            if (findAddress(symbols.items, sym.name) != null) return error.DuplicateSymbol;
            const name = try allocator.dupe(u8, sym.name);
            errdefer allocator.free(name);
            try symbols.append(allocator, .{ .name = name, .address = base + text_at[oi] + sym.value });
        }
    }

    var code = try allocator.alloc(u8, @intCast(total));
    errdefer allocator.free(code);
    @memset(code, 0);
    for (0..parsed_n) |oi| {
        if (parsed[oi].text.len > 0) @memcpy(code[@intCast(text_at[oi])..][0..parsed[oi].text.len], parsed[oi].text);
    }

    // Apply each CALL26 relocation (base cancels: `bl` is PC-relative).
    for (0..parsed_n) |oi| {
        for (parsed[oi].relocs) |r| {
            if (r.symbol >= parsed[oi].symbols.len) return error.MalformedObject;
            const name = parsed[oi].symbols[r.symbol].name;
            const target = findAddress(symbols.items, name) orelse return error.UndefinedSymbol;
            const site = text_at[oi] + r.offset;
            const delta = (@as(i64, @intCast(target)) - @as(i64, @intCast(base))) - @as(i64, @intCast(site)) + r.addend;
            try applyCall26(code, site, delta);
        }
    }

    return .{ .code = code, .symbols = try symbols.toOwnedSlice(allocator), .base = base };
}

/// Patch a `bl`/`b` (R_AARCH64_CALL26) at image offset `site` with the byte
/// displacement `delta` (+/-128MiB, multiple of 4).
fn applyCall26(code: []u8, site: u64, delta: i64) Error!void {
    const s: usize = @intCast(site);
    if (s + 4 > code.len) return error.MalformedObject;
    if (delta < -(1 << 27) or delta >= (1 << 27) or (delta & 3) != 0) return error.RelocationOutOfRange;
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    const is_bl = (word & 0xFC000000) == 0x94000000; // BL vs B share the encoding but bit 31
    const patched = if (is_bl) encode.bl(@intCast(delta)) else encode.b(@intCast(delta));
    std.mem.writeInt(u32, code[s..][0..4], patched, .little);
}

fn findAddress(symbols: []const ResolvedSymbol, name: []const u8) ?u64 {
    for (symbols) |s| if (std.mem.eql(u8, s.name, name)) return s.address;
    return null;
}

const ehdr_size: usize = 64;
const phdr_size: usize = 56;
/// The code sits at a 64 KiB-aligned file offset (headers and padding fill the first
/// region) so the single PT_LOAD's p_offset and p_vaddr stay congruent modulo the
/// page size. 64 KiB suits AArch64's max page size and is valid on 4/16 KiB systems
/// too, keeping the executable portable. `base` must be 64 KiB-aligned.
const exec_code_offset: usize = 0x10000;

/// Wrap `code` in a minimal static ELF64 AArch64 executable (`ET_EXEC`): one read+write+
/// execute `PT_LOAD` segment mapping `code` at `base` (in-memory size `mem_size` >= code.len,
/// the tail zero-initialized), entering at `entry_addr`. The caller owns the bytes.
pub fn writeElfExec(allocator: std.mem.Allocator, code: []const u8, mem_size: u64, base: u64, entry_addr: u64) std.mem.Allocator.Error![]u8 {
    const buf = try allocator.alloc(u8, exec_code_offset + code.len);
    @memset(buf, 0);

    buf[0] = 0x7f;
    buf[1] = 'E';
    buf[2] = 'L';
    buf[3] = 'F';
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT

    const w = std.mem.writeInt;
    w(u16, buf[16..18], 2, .little); // e_type = ET_EXEC
    w(u16, buf[18..20], 183, .little); // e_machine = EM_AARCH64
    w(u32, buf[20..24], 1, .little); // e_version
    w(u64, buf[24..32], entry_addr, .little); // e_entry
    w(u64, buf[32..40], ehdr_size, .little); // e_phoff
    w(u16, buf[52..54], ehdr_size, .little); // e_ehsize
    w(u16, buf[54..56], phdr_size, .little); // e_phentsize
    w(u16, buf[56..58], 1, .little); // e_phnum

    // The single PT_LOAD covers the whole file from offset 0 (so the ELF and program headers
    // are mapped, which a real kernel's loader expects), at a load address one page below
    // `base`. The page of headers + padding then places the code itself at exactly `base`,
    // keeping symbol addresses (resolved at `base`) correct while satisfying the loader.
    const load = base - exec_code_offset;
    const p = buf[ehdr_size..];
    w(u32, p[0..4], 1, .little); // p_type = PT_LOAD
    w(u32, p[4..8], 7, .little); // p_flags = R|W|X
    w(u64, p[8..16], 0, .little); // p_offset
    w(u64, p[16..24], load, .little); // p_vaddr
    w(u64, p[24..32], load, .little); // p_paddr
    w(u64, p[32..40], exec_code_offset + code.len, .little); // p_filesz
    w(u64, p[40..48], exec_code_offset + @max(mem_size, code.len), .little); // p_memsz
    w(u64, p[48..56], 0x10000, .little); // p_align (64 KiB, the max AArch64 page size)

    @memcpy(buf[exec_code_offset..], code);
    return buf;
}

/// Emit a runnable `ET_EXEC` for a linked image, entering at symbol `entry_name`.
pub fn writeExecutable(allocator: std.mem.Allocator, image: *const Image, entry_name: []const u8) Error![]u8 {
    const entry_addr = image.addressOf(entry_name) orelse return error.UndefinedSymbol;
    return writeElfExec(allocator, image.code, image.code.len, image.base, entry_addr);
}
