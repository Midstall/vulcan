//! Object-file linker: parse ELF64 RISC-V relocatable objects (from object.zig),
//! concatenate their `.text`, resolve cross-object relocations, and produce a
//! final code image. lld is used only as a comparison oracle in the tests.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const object = @import("object.zig");
const link = @import("link.zig");
const compress = @import("compress.zig");
const harness = @import("tests/harness.zig");

const Function = ir.function.Function;

pub const Error = std.mem.Allocator.Error || error{
    MalformedObject,
    UndefinedSymbol,
    DuplicateSymbol,
    RelocationOutOfRange,
    UnsupportedReloc,
};

// Recognized ELF section/symbol type codes.
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHN_UNDEF: u16 = 0;
const EM_RISCV: u16 = 243;

/// Which allocatable section a symbol or chunk belongs to (`undef` = none).
const SecKind = enum { undef, text, rodata, data, bss };

fn alignUp(v: u64, a: u64) u64 {
    return (v + a - 1) & ~(a - 1);
}

fn rdInt(comptime T: type, buf: []const u8, off: u64) Error!T {
    const o: usize = @intCast(off);
    if (o + @sizeOf(T) > buf.len) return error.MalformedObject;
    return std.mem.readInt(T, buf[o..][0..@sizeOf(T)], .little);
}

/// A symbol parsed out of one object's symbol table. `section` says which
/// allocatable section `value` is an offset into.
const ObjSymbol = struct {
    name: []const u8,
    value: u64,
    defined: bool,
    local: bool,
    section: SecKind,
};

/// The pieces lifted out of a single relocatable object: each allocatable
/// section's bytes (`.bss` has only a size), its symbols, and its relocations.
const ParsedObject = struct {
    text: []const u8,
    rodata: []const u8 = &.{},
    data: []const u8 = &.{},
    bss_size: u64 = 0,
    symbols: []ObjSymbol,
    relocs: []object.Reloc,

    fn deinit(self: *ParsedObject, allocator: std.mem.Allocator) void {
        allocator.free(self.symbols);
        allocator.free(self.relocs);
    }
};

/// Classify a section header by its type and flags into an allocatable kind.
fn classify(typ: u32, flags: u64) SecKind {
    if (typ == SHT_NOBITS) return .bss;
    if (typ != SHT_PROGBITS or (flags & SHF_ALLOC) == 0) return .undef;
    if ((flags & SHF_EXECINSTR) != 0) return .text;
    if ((flags & SHF_WRITE) != 0) return .data;
    return .rodata;
}

/// Read a NUL-terminated string from a string table at `off`.
fn strAt(strtab: []const u8, off: u32) Error![]const u8 {
    if (off >= strtab.len) return error.MalformedObject;
    const end = std.mem.indexOfScalarPos(u8, strtab, off, 0) orelse return error.MalformedObject;
    return strtab[off..end];
}

/// Parse one ELF64 RISC-V relocatable object into its `.text`, symbols, and
/// relocations. Section roles are identified by type (not by name) for
/// robustness. Names borrow from `buf`.
fn parseObject(allocator: std.mem.Allocator, buf: []const u8) Error!ParsedObject {
    if (buf.len < 64) return error.MalformedObject;
    if (!std.mem.eql(u8, buf[0..4], "\x7fELF")) return error.MalformedObject;
    if (buf[4] != 2 or buf[5] != 1) return error.MalformedObject; // ELFCLASS64, LSB
    if (try rdInt(u16, buf, 18) != EM_RISCV) return error.MalformedObject;

    const shoff = try rdInt(u64, buf, 40);
    const shentsize = try rdInt(u16, buf, 58);
    const shnum = try rdInt(u16, buf, 60);

    // Map each section index to its allocatable kind, and pick up the allocatable
    // section bytes plus the symbol/relocation tables.
    var kinds = try allocator.alloc(SecKind, shnum);
    defer allocator.free(kinds);
    var text: []const u8 = &.{};
    var rodata: []const u8 = &.{};
    var data: []const u8 = &.{};
    var bss_size: u64 = 0;
    var symtab_ndx: ?u16 = null;
    var rela_ndx: ?u16 = null;

    var i: u16 = 0;
    while (i < shnum) : (i += 1) {
        const hdr = shoff + @as(u64, i) * shentsize;
        const typ = try rdInt(u32, buf, hdr + 4);
        const flags = try rdInt(u64, buf, hdr + 8);
        const sh_off = try rdInt(u64, buf, hdr + 24);
        const sh_size = try rdInt(u64, buf, hdr + 32);
        const kind = classify(typ, flags);
        kinds[i] = kind;
        switch (kind) {
            .text, .rodata, .data => {
                if (@as(usize, @intCast(sh_off + sh_size)) > buf.len) return error.MalformedObject;
                const bytes = buf[@intCast(sh_off)..@intCast(sh_off + sh_size)];
                switch (kind) {
                    .text => text = bytes,
                    .rodata => rodata = bytes,
                    .data => data = bytes,
                    else => unreachable,
                }
            },
            .bss => bss_size += sh_size,
            .undef => {},
        }
        if (typ == SHT_SYMTAB) symtab_ndx = i;
        if (typ == SHT_RELA) rela_ndx = i;
    }
    const si = symtab_ndx orelse return error.MalformedObject;

    // Symbol table and its string table (via sh_link).
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
        const section: SecKind = if (st_shndx != SHN_UNDEF and st_shndx < shnum) kinds[st_shndx] else .undef;
        symbols[k] = .{
            .name = if (st_name == 0) "" else try strAt(strtab, st_name),
            .value = st_value,
            .defined = st_shndx != SHN_UNDEF,
            .local = (st_info >> 4) == 0, // STB_LOCAL
            .section = section,
        };
    }

    // Relocations (if any). The RELA section applies to `.text` in these objects.
    var relocs: []object.Reloc = &.{};
    if (rela_ndx) |ri| {
        const rela_hdr = shoff + @as(u64, ri) * shentsize;
        const rela_off = try rdInt(u64, buf, rela_hdr + 24);
        const rela_size = try rdInt(u64, buf, rela_hdr + 32);
        const rela_count: usize = @intCast(rela_size / 24);
        relocs = try allocator.alloc(object.Reloc, rela_count);
        var r: usize = 0;
        while (r < rela_count) : (r += 1) {
            const e = rela_off + @as(u64, r) * 24;
            const r_offset = try rdInt(u64, buf, e + 0);
            const r_info = try rdInt(u64, buf, e + 8);
            const r_addend = try rdInt(i64, buf, e + 16);
            const typ: u32 = @truncate(r_info & 0xffffffff);
            const sym_index: u32 = @intCast(r_info >> 32);
            const rt: object.RelocType = switch (typ) {
                @intFromEnum(object.RelocType.jal) => .jal,
                @intFromEnum(object.RelocType.call) => .call,
                @intFromEnum(object.RelocType.pcrel_hi20) => .pcrel_hi20,
                @intFromEnum(object.RelocType.pcrel_lo12_i) => .pcrel_lo12_i,
                else => return error.UnsupportedReloc,
            };
            relocs[r] = .{
                .offset = r_offset,
                .symbol = sym_index,
                .type = rt,
                .addend = r_addend,
            };
        }
    }

    return .{ .text = text, .rodata = rodata, .data = data, .bss_size = bss_size, .symbols = symbols, .relocs = relocs };
}

/// A resolved symbol in the linked image: its name and final absolute address.
pub const ResolvedSymbol = struct { name: []const u8, address: u64 };

/// A linked code image: the relocated `.text` (in input order) plus the symbol
/// table giving each defined symbol's absolute address. The code is meant to
/// load at `base`.
pub const Image = struct {
    code: []u8,
    symbols: []ResolvedSymbol,
    base: u64,
    /// Total in-memory size including `.bss` (>= `code.len`).
    memsz: u64,

    pub fn deinit(self: *Image, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        for (self.symbols) |s| allocator.free(s.name);
        allocator.free(self.symbols);
    }

    pub fn addressOf(self: *const Image, name: []const u8) ?u64 {
        for (self.symbols) |s| {
            if (std.mem.eql(u8, s.name, name)) return s.address;
        }
        return null;
    }
};

/// Patch a single relocation into the image. These are all PC-relative, so only
/// image offsets matter. `site` and `target` are both offsets within `code`.
fn applyReloc(code: []u8, site: u64, target: i64, typ: object.RelocType) Error!void {
    const s: usize = @intCast(site);
    if (s + 4 > code.len) return error.MalformedObject;
    const delta = target - @as(i64, @intCast(site));
    switch (typ) {
        .jal => {
            const word = std.mem.readInt(u32, code[s..][0..4], .little);
            const rd: encode.Reg = @enumFromInt(@as(u5, @truncate(word >> 7)));
            if (delta < -(1 << 20) or delta >= (1 << 20) or (delta & 1) != 0) return error.RelocationOutOfRange;
            const patched = encode.jal(rd, @intCast(delta));
            std.mem.writeInt(u32, code[s..][0..4], patched, .little);
        },
        .pcrel_hi20 => {
            // The high 20 bits of the PC-relative delta go in an `auipc`/`lui`
            // U-immediate (bits 31:12). The +0x800 pre-rounds for the lo12 sign.
            const hi: u32 = @truncate(@as(u64, @bitCast(delta +% 0x800)) >> 12);
            const word = std.mem.readInt(u32, code[s..][0..4], .little);
            std.mem.writeInt(u32, code[s..][0..4], (word & 0x0000_0fff) | (hi << 12), .little);
        },
        .call => {
            // A standard far call: `auipc` at `site` plus `jalr` at `site + 4`,
            // patched together. Reaches a 32-bit (+/-2GiB) PC-relative target.
            if (s + 8 > code.len) return error.MalformedObject;
            if (delta < -(1 << 31) or delta >= (1 << 31)) return error.RelocationOutOfRange;
            const hi: u32 = @truncate(@as(u64, @bitCast(delta +% 0x800)) >> 12);
            const lo: u32 = @as(u12, @truncate(@as(u64, @bitCast(delta))));
            const auipc_w = std.mem.readInt(u32, code[s..][0..4], .little);
            std.mem.writeInt(u32, code[s..][0..4], (auipc_w & 0x0000_0fff) | (hi << 12), .little);
            const jalr_w = std.mem.readInt(u32, code[s + 4 ..][0..4], .little);
            std.mem.writeInt(u32, code[s + 4 ..][0..4], (jalr_w & 0x000f_ffff) | (lo << 20), .little);
        },
        else => return error.UnsupportedReloc,
    }
}

/// Patch a PCREL_LO12_I relocation: the low 12 bits of `pcrel` go in an I-type
/// immediate (bits 31:20). `pcrel` is the delta the paired `auipc` used.
fn patchLo12(code: []u8, site: u64, pcrel: i64) Error!void {
    const s: usize = @intCast(site);
    if (s + 4 > code.len) return error.MalformedObject;
    const lo: u32 = @as(u12, @truncate(@as(u64, @bitCast(pcrel))));
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    std.mem.writeInt(u32, code[s..][0..4], (word & 0x000f_ffff) | (lo << 20), .little);
}

/// Resolves an undefined (external) symbol name to its absolute runtime address,
/// or null if unknown. The JIT uses this to bind calls to host/runtime functions.
pub const Resolver = struct {
    context: *anyopaque,
    func: *const fn (context: *anyopaque, name: []const u8) ?u64,

    pub fn resolve(self: Resolver, name: []const u8) ?u64 {
        return self.func(self.context, name);
    }
};

/// Each external call gets a 3-instruction stub (`auipc`, `ld`, `jr`) that loads
/// the target's absolute address from a GOT slot and jumps, so a near `jal`/`call`
/// can reach any 64-bit address.
const stub_words = 3;
const stub_bytes = stub_words * 4;

/// An external symbol bound by the resolver: its name and absolute address.
const Extern = struct { name: []const u8, addr: u64 };

/// Link a set of relocatable objects into one image, laid out in the order given
/// with `.text` first at `base`, then `.rodata`, `.data`, and `.bss` (memory
/// only). Resolves every cross-object and intra-object relocation. The caller
/// owns the returned image.
pub fn linkObjects(allocator: std.mem.Allocator, objs: []const []const u8, base: u64) Error!Image {
    return linkObjectsResolved(allocator, objs, base, null);
}

/// Like `linkObjects`, but undefined symbols referenced by calls are bound to
/// absolute addresses via `resolver` (the JIT use case). For each such external,
/// the linker appends a GOT slot (its absolute address) and a stub that loads it
/// and jumps, then routes the call to the stub. Without a resolver, undefined
/// symbols are an error, as before.
pub fn linkObjectsResolved(allocator: std.mem.Allocator, objs: []const []const u8, base: u64, resolver: ?Resolver) Error!Image {
    return linkImpl(allocator, objs, base, resolver, false);
}

/// Like `linkObjectsResolved`, but RVC-compresses each object's `.text` before layout (the C-extension
/// output path). Every relocation site is pinned 32-bit so the linker can still patch it, and reloc
/// offsets + `.text` symbol values are remapped onto the shrunk layout; the normal resolution passes
/// then compute correct call/PC-relative values against the compressed addresses.
pub fn linkObjectsCompressed(allocator: std.mem.Allocator, objs: []const []const u8, base: u64, resolver: ?Resolver) Error!Image {
    return linkImpl(allocator, objs, base, resolver, true);
}

fn linkImpl(allocator: std.mem.Allocator, objs: []const []const u8, base: u64, resolver: ?Resolver, compress_text: bool) Error!Image {
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

    // Compress each object's `.text` before layout. Pin every reloc site (call/PC-relative) so the
    // linker's later patches land on intact 32-bit instructions, and remap this object's reloc
    // offsets and `.text` symbol values through the shrunk-layout offset map.
    var owned_texts = try allocator.alloc(?[]u8, parsed_n);
    for (owned_texts) |*t| t.* = null;
    defer {
        for (owned_texts) |t| if (t) |b| allocator.free(b);
        allocator.free(owned_texts);
    }
    if (compress_text) {
        for (0..parsed_n) |oi| {
            const p = &parsed[oi];
            const nwords = p.text.len / 4;
            if (nwords == 0) continue;
            const words = try allocator.alloc(u32, nwords);
            defer allocator.free(words);
            for (0..nwords) |k| words[k] = std.mem.readInt(u32, p.text[k * 4 ..][0..4], .little);
            const pins = try allocator.alloc(usize, p.relocs.len);
            defer allocator.free(pins);
            for (p.relocs, 0..) |r, ri| pins[ri] = @intCast(r.offset / 4);
            const offmap = try allocator.alloc(usize, nwords + 1);
            defer allocator.free(offmap);
            const cbytes = try compress.compressPinned(allocator, words, pins, offmap);
            for (p.relocs) |*r| r.offset = offmap[@intCast(r.offset / 4)];
            for (p.symbols) |*s| {
                if (s.section == .text and s.defined) s.value = offmap[@intCast(s.value / 4)];
            }
            owned_texts[oi] = cbytes;
            p.text = cbytes;
        }
    }

    // Pack each object's sections within its region, recording per-object offsets.
    var text_at = try allocator.alloc(u64, objs.len);
    defer allocator.free(text_at);
    var rodata_at = try allocator.alloc(u64, objs.len);
    defer allocator.free(rodata_at);
    var data_at = try allocator.alloc(u64, objs.len);
    defer allocator.free(data_at);
    var bss_at = try allocator.alloc(u64, objs.len);
    defer allocator.free(bss_at);
    var text_total: u64 = 0;
    var rodata_total: u64 = 0;
    var data_total: u64 = 0;
    var bss_total: u64 = 0;
    for (0..parsed_n) |oi| {
        text_at[oi] = alignUp(text_total, 4);
        text_total = text_at[oi] + parsed[oi].text.len;
        rodata_at[oi] = alignUp(rodata_total, 8);
        rodata_total = rodata_at[oi] + parsed[oi].rodata.len;
        data_at[oi] = alignUp(data_total, 8);
        data_total = data_at[oi] + parsed[oi].data.len;
        bss_at[oi] = alignUp(bss_total, 8);
        bss_total = bss_at[oi] + parsed[oi].bss_size;
    }

    const rodata_region = alignUp(text_total, 8);
    const data_region = alignUp(rodata_region + rodata_total, 8);
    var data_end: u64 = text_total;
    if (rodata_total > 0) data_end = rodata_region + rodata_total;
    if (data_total > 0) data_end = data_region + data_total;

    const regionOffset = struct {
        fn f(section: SecKind, oi: usize, ro: u64, da: u64, bs: u64, t_at: []const u64, ro_at: []const u64, da_at: []const u64, bs_at: []const u64) u64 {
            return switch (section) {
                .text => t_at[oi],
                .rodata => ro + ro_at[oi],
                .data => da + da_at[oi],
                .bss => bs + bs_at[oi],
                .undef => unreachable,
            };
        }
    }.f;

    // Resolve each defined, non-local symbol to its final address.
    var symbols: std.ArrayList(ResolvedSymbol) = .empty;
    errdefer {
        for (symbols.items) |s| allocator.free(s.name);
        symbols.deinit(allocator);
    }
    for (0..parsed_n) |oi| {
        for (parsed[oi].symbols) |sym| {
            if (!sym.defined or sym.local or sym.name.len == 0) continue;
            if (findSymbol(symbols.items, sym.name) != null) return error.DuplicateSymbol;
            const region = regionOffset(sym.section, oi, rodata_region, data_region, alignUp(data_end, 8), text_at, rodata_at, data_at, bss_at);
            const name = try allocator.dupe(u8, sym.name);
            errdefer allocator.free(name);
            try symbols.append(allocator, .{ .name = name, .address = base + region + sym.value });
        }
    }

    // Collect the external call targets the resolver can bind: undefined symbols
    // referenced by a `jal`/`call` relocation. Each needs a stub + GOT slot.
    var externs: std.ArrayList(Extern) = .empty;
    defer externs.deinit(allocator);
    if (resolver) |res| {
        for (0..parsed_n) |oi| {
            for (parsed[oi].relocs) |r| {
                if (r.type != .jal and r.type != .call) continue;
                if (r.symbol >= parsed[oi].symbols.len) return error.MalformedObject;
                const name = parsed[oi].symbols[r.symbol].name;
                if (findSymbol(symbols.items, name) != null) continue; // defined or already a stub
                if (externIndex(externs.items, name) != null) continue;
                const addr = res.resolve(name) orelse continue; // unresolved: surfaced later as UndefinedSymbol
                try externs.append(allocator, .{ .name = name, .addr = addr });
            }
        }
    }

    // Lay out the stub/GOT region after the data, then `.bss` after that.
    const ext_n = externs.items.len;
    const stub_region = alignUp(data_end, 4);
    const got_region = alignUp(stub_region + ext_n * stub_bytes, 8);
    const filesz = if (ext_n > 0) got_region + ext_n * 8 else data_end;
    const bss_region = alignUp(filesz, 8);
    const memsz = if (bss_total > 0) bss_region + bss_total else filesz;

    // The loadable file image: text, rodata, data laid into place (bss is zero).
    var code = try allocator.alloc(u8, @intCast(filesz));
    errdefer allocator.free(code);
    @memset(code, 0);
    for (0..parsed_n) |oi| {
        if (parsed[oi].text.len > 0) @memcpy(code[@intCast(text_at[oi])..][0..parsed[oi].text.len], parsed[oi].text);
        if (parsed[oi].rodata.len > 0) @memcpy(code[@intCast(rodata_region + rodata_at[oi])..][0..parsed[oi].rodata.len], parsed[oi].rodata);
        if (parsed[oi].data.len > 0) @memcpy(code[@intCast(data_region + data_at[oi])..][0..parsed[oi].data.len], parsed[oi].data);
    }

    // Emit each stub and its GOT slot, and register the stub as the symbol's
    // address so calls route to it.
    for (externs.items, 0..) |ext, i| {
        const stub_off = stub_region + i * stub_bytes;
        const got_off = got_region + i * 8;
        std.mem.writeInt(u64, code[@intCast(got_off)..][0..8], ext.addr, .little);
        // `auipc t0, %pcrel_hi(got)`, then `ld t0, %pcrel_lo(got)(t0)`, then `jr t0`
        const pcrel: i64 = @as(i64, @intCast(got_off)) - @as(i64, @intCast(stub_off));
        const hi: u20 = @truncate(@as(u64, @bitCast(pcrel +% 0x800)) >> 12);
        const lo: i12 = @bitCast(@as(u12, @truncate(@as(u64, @bitCast(pcrel)))));
        std.mem.writeInt(u32, code[@intCast(stub_off)..][0..4], encode.auipc(.x5, hi), .little);
        std.mem.writeInt(u32, code[@intCast(stub_off + 4)..][0..4], encode.ld(.x5, .x5, lo), .little);
        std.mem.writeInt(u32, code[@intCast(stub_off + 8)..][0..4], encode.jalr(.x0, .x5, 0), .little);
        const name = try allocator.dupe(u8, ext.name);
        errdefer allocator.free(name);
        try symbols.append(allocator, .{ .name = name, .address = base + stub_off });
    }

    // Pass 2a: resolve calls and PCREL_HI20 (sites live in `.text`). Record each
    // hi20 site's resolved target so the paired lo12 recomputes the same delta.
    var hi_target: std.AutoHashMapUnmanaged(u64, i64) = .empty;
    defer hi_target.deinit(allocator);
    for (0..parsed_n) |oi| {
        for (parsed[oi].relocs) |r| {
            if (r.type == .pcrel_lo12_i) continue;
            if (r.symbol >= parsed[oi].symbols.len) return error.MalformedObject;
            const target_name = parsed[oi].symbols[r.symbol].name;
            const target_addr = findSymbol(symbols.items, target_name) orelse return error.UndefinedSymbol;
            const target_off: i64 = @as(i64, @intCast(target_addr - base)) + r.addend;
            const site = text_at[oi] + r.offset;
            try applyReloc(code, site, target_off, r.type);
            if (r.type == .pcrel_hi20) try hi_target.put(allocator, site, target_off);
        }
    }

    // Pass 2b: resolve PCREL_LO12 against its paired `auipc`.
    for (0..parsed_n) |oi| {
        for (parsed[oi].relocs) |r| {
            if (r.type != .pcrel_lo12_i) continue;
            if (r.symbol >= parsed[oi].symbols.len) return error.MalformedObject;
            const auipc_site = text_at[oi] + parsed[oi].symbols[r.symbol].value;
            const target_off = hi_target.get(auipc_site) orelse return error.MalformedObject;
            const pcrel = target_off - @as(i64, @intCast(auipc_site));
            try patchLo12(code, text_at[oi] + r.offset, pcrel);
        }
    }

    return .{
        .code = code,
        .symbols = try symbols.toOwnedSlice(allocator),
        .base = base,
        .memsz = memsz,
    };
}

/// File layout for the emitted executables: ELF header, one program header, code.
const ehdr_size: usize = 64;
const phdr_size: usize = 56;
/// The code is placed at a page-aligned file offset (the headers and padding fill the first
/// page). This keeps the code's load address at `base` while satisfying the loader's mmap
/// requirement that p_offset and p_vaddr be congruent modulo the page size, so the ELF runs
/// under a real loader (e.g. qemu user mode), not only a flat firmware loader.
const exec_code_offset: usize = 0x1000;

/// Wrap `code` in a minimal static ELF64 RISC-V executable (`ET_EXEC`): a single
/// read+write+execute `PT_LOAD` segment mapping `code` at `base` with an
/// in-memory size of `mem_size` (>= `code.len`, the tail is zero-initialized
/// `.bss`), entering at `entry_addr`. The caller owns the returned bytes.
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
    w(u16, buf[18..20], 243, .little); // e_machine = EM_RISCV
    w(u32, buf[20..24], 1, .little); // e_version
    w(u64, buf[24..32], entry_addr, .little); // e_entry
    w(u64, buf[32..40], ehdr_size, .little); // e_phoff
    w(u16, buf[52..54], ehdr_size, .little); // e_ehsize
    w(u16, buf[54..56], phdr_size, .little); // e_phentsize
    w(u16, buf[56..58], 1, .little); // e_phnum

    const p = buf[ehdr_size..];
    w(u32, p[0..4], 1, .little); // p_type = PT_LOAD
    w(u32, p[4..8], 7, .little); // p_flags = R|W|X (covers code + writable/bss data)
    w(u64, p[8..16], exec_code_offset, .little); // p_offset
    w(u64, p[16..24], base, .little); // p_vaddr
    w(u64, p[24..32], base, .little); // p_paddr
    w(u64, p[32..40], code.len, .little); // p_filesz
    w(u64, p[40..48], @max(mem_size, code.len), .little); // p_memsz
    w(u64, p[48..56], 0x1000, .little); // p_align

    @memcpy(buf[exec_code_offset..], code);
    return buf;
}

/// Emit a runnable `ET_EXEC` for a linked image, entering at symbol `entry_name`.
pub fn writeExecutable(allocator: std.mem.Allocator, image: *const Image, entry_name: []const u8) Error![]u8 {
    const entry_addr = image.addressOf(entry_name) orelse return error.UndefinedSymbol;
    return writeElfExec(allocator, image.code, image.memsz, image.base, entry_addr);
}

fn findSymbol(symbols: []const ResolvedSymbol, name: []const u8) ?u64 {
    for (symbols) |s| {
        if (std.mem.eql(u8, s.name, name)) return s.address;
    }
    return null;
}

fn externIndex(externs: []const Extern, name: []const u8) ?usize {
    for (externs, 0..) |e, i| {
        if (std.mem.eql(u8, e.name, name)) return i;
    }
    return null;
}

test "our linker resolves a cross-object call to the reference bytes" {
    const allocator = std.testing.allocator;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee, compiled to its own object.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const x = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = x });
    }
    var callee_mod: link.Module = .{};
    defer callee_mod.deinit(allocator);
    try callee_mod.addFunction(allocator, "callee", &callee);
    const callee_obj = try object.writeModule(allocator, &callee_mod);
    defer allocator.free(callee_obj);

    // caller, compiled to its own object: "callee" is an undefined external.
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        caller.setTerminator(b, .{ .ret = r });
    }
    var caller_mod: link.Module = .{};
    defer caller_mod.deinit(allocator);
    try caller_mod.addFunction(allocator, "caller", &caller);
    const caller_obj = try object.writeModule(allocator, &caller_mod);
    defer allocator.free(caller_obj);

    // The in-memory IR linker's bytes are the reference (lld matches these too).
    var combined: link.Module = .{};
    defer combined.deinit(allocator);
    try combined.addFunction(allocator, "callee", &callee);
    try combined.addFunction(allocator, "caller", &caller);
    var reference = try link.compileModule(allocator, &combined);
    defer reference.deinit(allocator);

    // The object linker, given the two objects (callee first), must produce the
    // same relocated code.
    var image = try linkObjects(allocator, &.{ callee_obj, caller_obj }, 0x80000000);
    defer image.deinit(allocator);

    try std.testing.expectEqual(reference.code.len * 4, image.code.len);
    for (reference.code, 0..) |word, i| {
        const got = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(word, got);
    }
    try std.testing.expectEqual(@as(?u64, 0x80000000), image.addressOf("callee"));

    // The linker can wrap the image into a runnable ET_EXEC entering at "caller".
    const exe = try writeExecutable(allocator, &image, "caller");
    defer allocator.free(exe);
    try std.testing.expectEqualSlices(u8, "\x7fELF", exe[0..4]);
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, exe[16..18], .little)); // ET_EXEC
    try std.testing.expectEqual(@as(u16, 243), std.mem.readInt(u16, exe[18..20], .little)); // EM_RISCV
    // e_entry is caller's absolute address (callee is one word, so caller@base+4).
    try std.testing.expectEqual(@as(u64, 0x80000004), std.mem.readInt(u64, exe[24..32], .little));
    // The single PT_LOAD maps the code at the image base.
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, exe[56..58], .little)); // e_phnum
    try std.testing.expectEqual(@as(u64, 0x80000000), std.mem.readInt(u64, exe[64 + 16 ..][0..8], .little)); // p_vaddr
    // The loadable bytes are exactly the linked image (at the page-aligned code offset).
    try std.testing.expectEqualSlices(u8, image.code, exe[0x1000..]);

    // Unresolved entry name is rejected.
    try std.testing.expectError(error.UndefinedSymbol, writeExecutable(allocator, &image, "nope"));
}

test "code linked by our own linker runs correctly on River" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry(a, b) -> add(a, b). The stub calls whatever sits at code[0], so the
    // entry must be linked first. add() is a separate object.
    var add = Function.init(allocator);
    defer add.deinit();
    {
        const t = try add.types.intern(i32k);
        const b = try add.appendBlock();
        const x = try add.appendBlockParam(b, t);
        const y = try add.appendBlockParam(b, t);
        const s = try add.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
        add.setTerminator(b, .{ .ret = s });
    }
    var add_mod: link.Module = .{};
    defer add_mod.deinit(allocator);
    try add_mod.addFunction(allocator, "add", &add);
    const add_obj = try object.writeModule(allocator, &add_mod);
    defer allocator.free(add_obj);

    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const b = try entry.appendBlock();
        const x = try entry.appendBlockParam(b, t);
        const y = try entry.appendBlockParam(b, t);
        const r = try entry.appendCall(b, t, "add", &.{ x, y });
        entry.setTerminator(b, .{ .ret = r });
    }
    var entry_mod: link.Module = .{};
    defer entry_mod.deinit(allocator);
    try entry_mod.addFunction(allocator, "entry", &entry);
    const entry_obj = try object.writeModule(allocator, &entry_mod);
    defer allocator.free(entry_obj);

    // Link entry first so it lands at the image start. add() follows.
    var image = try linkObjects(allocator, &.{ entry_obj, add_obj }, harness.load_address);
    defer image.deinit(allocator);

    // Reinterpret the byte image as machine words and execute it on River.
    const words = try allocator.alloc(u32, image.code.len / 4);
    defer allocator.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);

    try std.testing.expectEqual(@as(i64, 42), try harness.runCode(io, allocator, words, &.{ 20, 22 }, harness.river));
}

test "global data loaded via PC-relative addressing runs on River" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&K), where K is a module-level i32 constant holding 42.
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.intern(.ptr);
        const b = try entry.appendBlock();
        const p = try entry.appendGlobalAddr(b, ptr_t, "K");
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = p } });
        entry.setTerminator(b, .{ .ret = v });
    }

    const k_bytes = [_]u8{ 42, 0, 0, 0 }; // i32 42, little-endian
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addData(allocator, "K", &k_bytes);

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);

    // The linker resolves the PCREL_HI20/LO12 pair. The result runs on River.
    var image = try linkObjects(allocator, &.{obj}, harness.load_address);
    defer image.deinit(allocator);

    const words = try allocator.alloc(u32, image.code.len / 4);
    defer allocator.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(i64, 42), try harness.runCode(io, allocator, words, &.{}, harness.river));
}

test "initialized writable .data global is read back on River" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&D), D a writable i32 initialized to 7 (lives in `.data`).
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.intern(.ptr);
        const b = try entry.appendBlock();
        const p = try entry.appendGlobalAddr(b, ptr_t, "D");
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = p } });
        entry.setTerminator(b, .{ .ret = v });
    }
    const d_bytes = [_]u8{ 7, 0, 0, 0 };
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addWritable(allocator, "D", &d_bytes);

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);
    var image = try linkObjects(allocator, &.{obj}, harness.load_address);
    defer image.deinit(allocator);

    const words = try allocator.alloc(u32, image.code.len / 4);
    defer allocator.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(i64, 7), try harness.runCode(io, allocator, words, &.{}, harness.river));
}

test "zero-initialized .bss global is writable and reads back on River" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() stores 99 to B then returns it. B is a zero-init i32 in `.bss`.
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.intern(.ptr);
        const b = try entry.appendBlock();
        const p = try entry.appendGlobalAddr(b, ptr_t, "B");
        const c = try entry.appendInst(b, t, .{ .iconst = 99 });
        try entry.appendStore(b, c, p);
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = p } });
        entry.setTerminator(b, .{ .ret = v });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addBss(allocator, "B", 4);

    const obj = try object.writeModule(allocator, &module);
    defer allocator.free(obj);
    var image = try linkObjects(allocator, &.{obj}, harness.load_address);
    defer image.deinit(allocator);

    // The image carries no .bss bytes, but memsz covers it.
    try std.testing.expect(image.memsz > image.code.len);

    const words = try allocator.alloc(u32, image.code.len / 4);
    defer allocator.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(i64, 99), try harness.runCode(io, allocator, words, &.{}, harness.river));
}

// The lld byte-equivalence check for the PCREL pair lives in object.zig
// (next to the lld test helper), confirming the pcrel math matches lld exactly.
