//! One shared ELF relocatable-object serializer for every Vulcan backend. A backend
//! hands over its already-built sections, symbols, and relocations in a neutral,
//! architecture-independent form. This file writes them once into a valid `ET_REL`
//! object, at ELF32 or ELF64 width per `Config.class`. It replaces the per-backend
//! copies of the ELF header, section-header, symbol-table, and string-table layout code,
//! so those bytes are laid out one time, not four.
//!
//! This file owns two rules the backends must not repeat. It owns the symbol sort:
//! ELF requires every local symbol to precede every global one, so `emit` places the
//! input symbols locals-before-globals (stable) and sets the `.symtab` `sh_info` to the
//! first global. It owns the matching reloc-symbol remap: each `OutReloc.symbol` is an
//! index into the caller's input `symbols` array, and `emit` rewrites it to the symbol's
//! post-sort index. So a backend passes relocs against input indices and never sorts.
//!
//! For `use_rela = false` (i386) the output holds `SHT_REL` entries with no addend field.
//! The caller has already baked each addend into the section bytes, so `emit` never reads
//! or edits a section's bytes. For `use_rela = true` the output holds `SHT_RELA` entries
//! that carry the addend.

const std = @import("std");

/// Allocation is the only failure mode. `emit` builds the whole object in memory.
pub const Error = std.mem.Allocator.Error;

/// The ELF width. `elf32` is i386 (x86). `elf64` is every other target.
pub const Class = enum { elf32, elf64 };

/// A symbol's binding. `emit` sorts locals ahead of globals. The numeric value is the
/// ELF `STB_*` code (`STB_LOCAL` = 0, `STB_GLOBAL` = 1).
pub const Binding = enum(u8) { local = 0, global = 1 };

/// A symbol's type. `func` is a code entry point (`STT_FUNC`). `object` is a data object
/// (`STT_OBJECT`). `notype` is unknown (`STT_NOTYPE`), as for an undefined import.
pub const SymType = enum { notype, func, object };

/// One output section. A `sh_type` of `SHT_PROGBITS` (1) puts `bytes` in the file. A
/// `sh_type` of `SHT_NOBITS` (8) occupies no file space and carries no `bytes`. Its
/// `size` still gives the in-memory extent. `relocs` are the relocations whose target is
/// this section.
pub const OutSection = struct {
    name: []const u8,
    /// `SHT_PROGBITS` (1) or `SHT_NOBITS` (8).
    sh_type: u32,
    /// `SHF_ALLOC` (2), `SHF_WRITE` (1), `SHF_EXECINSTR` (4), combined with a bit-or.
    flags: u64,
    /// The file bytes. Empty for a `SHT_NOBITS` section.
    bytes: []const u8 = &.{},
    /// The in-memory size. Equal to `bytes.len` for a `SHT_PROGBITS` section.
    size: u64,
    addralign: u64,
    relocs: []const OutReloc = &.{},
};

/// One relocation against a symbol, at `offset` bytes into its owning section. `symbol`
/// is an index into the caller's input `symbols` array. `emit` remaps it to the
/// post-sort symbol index. `addend` is used only when `Config.use_rela` is true.
pub const OutReloc = struct {
    offset: u64,
    symbol: u32,
    r_type: u32,
    addend: i64 = 0,
};

/// One symbol. A defined symbol lives in `section` (an index into the input sections
/// slice) at `value`. An undefined symbol (`defined = false`) has `st_shndx = SHN_UNDEF`,
/// and its `section` is ignored.
pub const OutSymbol = struct {
    name: []const u8,
    /// Index into the input sections slice. Ignored when `defined` is false.
    section: u32,
    value: u64,
    size: u64,
    binding: Binding,
    sym_type: SymType,
    defined: bool,
};

/// The output shape. `class` picks the ELF width. `machine` is the ELF `e_machine` code.
/// `use_rela` picks `SHT_RELA` (with addend) over `SHT_REL` (no addend).
pub const Config = struct {
    class: Class,
    machine: u16,
    use_rela: bool,
};

const ET_REL: u16 = 1;
const EV_CURRENT: u32 = 1;

const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_REL: u32 = 9;
const SHT_NOBITS: u32 = 8;

const SHF_INFO_LINK: u64 = 0x40;

const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;

const SHN_UNDEF: u16 = 0;

const ELFCLASS32: u8 = 1;
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;

fn alignUp(v: u64, a: u64) u64 {
    return (v + a - 1) & ~(a - 1);
}

fn putInt(buf: []u8, comptime T: type, value: T) void {
    std.mem.writeInt(T, buf[0..@sizeOf(T)], value, .little);
}

/// A growable string table. It starts with a leading NUL byte, then holds
/// NUL-terminated names. `add` returns each appended name's byte offset.
const StrTab = struct {
    bytes: std.ArrayList(u8) = .empty,

    fn init(allocator: std.mem.Allocator) Error!StrTab {
        var t: StrTab = .{};
        try t.bytes.append(allocator, 0);
        return t;
    }
    fn deinit(self: *StrTab, allocator: std.mem.Allocator) void {
        self.bytes.deinit(allocator);
    }
    fn add(self: *StrTab, allocator: std.mem.Allocator, name: []const u8) Error!u32 {
        const off: u32 = @intCast(self.bytes.items.len);
        try self.bytes.appendSlice(allocator, name);
        try self.bytes.append(allocator, 0);
        return off;
    }
};

/// One emitted section header, before its file offset is assigned. `has_bytes` is false
/// for a `SHT_NOBITS` section, so the layout pass gives it no file space.
const Shdr = struct {
    name: []const u8,
    typ: u32,
    flags: u64,
    addralign: u64,
    entsize: u64 = 0,
    link: u32 = 0,
    info: u32 = 0,
    has_bytes: bool,
    bytes: []const u8 = &.{},
    size: u64,
};

/// The per-class byte widths of every ELF structure this file writes.
const Widths = struct {
    ehsize: u64,
    shentsize: u64,
    symentsize: u64,
    relentsize: u64,
    relaentsize: u64,

    fn of(class: Class) Widths {
        return switch (class) {
            .elf64 => .{ .ehsize = 64, .shentsize = 64, .symentsize = 24, .relentsize = 16, .relaentsize = 24 },
            .elf32 => .{ .ehsize = 52, .shentsize = 40, .symentsize = 16, .relentsize = 8, .relaentsize = 12 },
        };
    }
};

/// Serialize the neutral object into ELF bytes. The section header order is: the reserved
/// null section, then each input section, then one reloc section per input section that
/// has relocations, then `.symtab`, `.strtab`, `.shstrtab`. The caller owns the result.
pub fn emit(
    allocator: std.mem.Allocator,
    sections: []const OutSection,
    symbols: []const OutSymbol,
    cfg: Config,
) Error![]u8 {
    const w = Widths.of(cfg.class);
    const reloc_entsize = if (cfg.use_rela) w.relaentsize else w.relentsize;
    const reloc_type: u32 = if (cfg.use_rela) SHT_RELA else SHT_REL;

    // Own the symbol sort. Place every local before every global, keeping the input order
    // within each binding group (a stable partition). `order[pos]` is the input index now
    // at symbol-table slot `pos + 1` (slot 0 is the null symbol). `new_index[input]` is
    // the reverse map, the post-sort slot of an input symbol, used by the reloc remap.
    const order = try allocator.alloc(u32, symbols.len);
    defer allocator.free(order);
    const new_index = try allocator.alloc(u32, symbols.len);
    defer allocator.free(new_index);
    var n: usize = 0;
    for (symbols, 0..) |s, i| if (s.binding == .local) {
        order[n] = @intCast(i);
        n += 1;
    };
    const local_count = n;
    for (symbols, 0..) |s, i| if (s.binding == .global) {
        order[n] = @intCast(i);
        n += 1;
    };
    for (order, 0..) |input_i, pos| new_index[input_i] = @intCast(pos + 1);
    // The null symbol at slot 0 is local, so the first global sits after it and the locals.
    const first_global: u32 = @intCast(1 + local_count);

    // The symbol-name string table, filled in sorted order.
    var strtab = try StrTab.init(allocator);
    defer strtab.deinit(allocator);
    const sym_name_off = try allocator.alloc(u32, symbols.len);
    defer allocator.free(sym_name_off);
    for (order, 0..) |input_i, pos| sym_name_off[pos] = try strtab.add(allocator, symbols[input_i].name);

    // The symbol table. Slot 0 is the null symbol.
    const sym_bytes = (1 + symbols.len) * w.symentsize;
    const symtab = try allocator.alloc(u8, sym_bytes);
    defer allocator.free(symtab);
    @memset(symtab, 0);
    for (order, 0..) |input_i, pos| {
        const s = symbols[input_i];
        const e = symtab[(pos + 1) * w.symentsize ..][0..w.symentsize];
        const binding: u8 = switch (s.binding) {
            .local => STB_LOCAL,
            .global => STB_GLOBAL,
        };
        const styp: u8 = switch (s.sym_type) {
            .notype => STT_NOTYPE,
            .func => STT_FUNC,
            .object => STT_OBJECT,
        };
        const info: u8 = (binding << 4) | styp;
        // A defined symbol names its section by header index (its input index plus one,
        // past the null section). An undefined symbol points at `SHN_UNDEF`.
        const shndx: u16 = if (s.defined) @intCast(s.section + 1) else SHN_UNDEF;
        writeSym(e, cfg.class, sym_name_off[pos], info, shndx, s.value, s.size);
    }

    // Build the section-header list, and, alongside it, the packed reloc bytes for every
    // section that has relocations. The header indices are known as the list grows: the
    // reserved null section is index 0, so the first appended header is index 1.
    var shstrtab = try StrTab.init(allocator);
    defer shstrtab.deinit(allocator);

    var headers: std.ArrayList(Shdr) = .empty;
    defer headers.deinit(allocator);

    // Reloc section names (".rela.<name>"/".rel.<name>") and the packed reloc buffers are
    // allocated here. Both must outlive the header list, so free them at the end.
    var reloc_names: std.ArrayList([]u8) = .empty;
    defer {
        for (reloc_names.items) |name| allocator.free(name);
        reloc_names.deinit(allocator);
    }
    var reloc_bufs: std.ArrayList([]u8) = .empty;
    defer {
        for (reloc_bufs.items) |b| allocator.free(b);
        reloc_bufs.deinit(allocator);
    }

    // The reserved null section is index 0. Each input section is index `j + 1`. Count the
    // sections that carry relocations, so the `.symtab` header index is known up front (a
    // reloc section's `sh_link` needs it).
    var reloc_section_count: usize = 0;
    for (sections) |sec| {
        if (sec.relocs.len > 0) reloc_section_count += 1;
    }
    const symtab_ndx: u32 = @intCast(1 + sections.len + reloc_section_count);

    // The input sections, in order.
    for (sections) |sec| {
        const is_nobits = sec.sh_type == SHT_NOBITS;
        try headers.append(allocator, .{
            .name = sec.name,
            .typ = sec.sh_type,
            .flags = sec.flags,
            .addralign = sec.addralign,
            .has_bytes = !is_nobits,
            .bytes = sec.bytes,
            .size = sec.size,
        });
    }

    // One reloc section per input section that has relocations. Its `sh_info` names the
    // target section by header index. Each `OutReloc.symbol` is remapped from its input
    // index to the post-sort symbol slot.
    const prefix = if (cfg.use_rela) ".rela" else ".rel";
    for (sections, 0..) |sec, j| {
        if (sec.relocs.len == 0) continue;
        const name = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, sec.name });
        try reloc_names.append(allocator, name);

        const rbuf = try allocator.alloc(u8, sec.relocs.len * reloc_entsize);
        try reloc_bufs.append(allocator, rbuf);
        for (sec.relocs, 0..) |r, i| {
            const e = rbuf[i * reloc_entsize ..][0..reloc_entsize];
            const sym = new_index[r.symbol];
            writeReloc(e, cfg.class, cfg.use_rela, r.offset, sym, r.r_type, r.addend);
        }

        try headers.append(allocator, .{
            .name = name,
            .typ = reloc_type,
            .flags = SHF_INFO_LINK,
            .addralign = 8,
            .entsize = reloc_entsize,
            .link = symtab_ndx,
            .info = @intCast(j + 1),
            .has_bytes = true,
            .bytes = rbuf,
            .size = rbuf.len,
        });
    }

    const strtab_ndx: u32 = symtab_ndx + 1;
    try headers.append(allocator, .{
        .name = ".symtab",
        .typ = SHT_SYMTAB,
        .flags = 0,
        .addralign = 8,
        .entsize = w.symentsize,
        .link = strtab_ndx,
        .info = first_global,
        .has_bytes = true,
        .bytes = symtab,
        .size = sym_bytes,
    });
    try headers.append(allocator, .{
        .name = ".strtab",
        .typ = SHT_STRTAB,
        .flags = 0,
        .addralign = 1,
        .has_bytes = true,
        .bytes = strtab.bytes.items,
        .size = strtab.bytes.items.len,
    });
    // `.shstrtab` is the last header. Its own bytes are built after the offset pass below,
    // because that pass adds every header's name to it.
    try headers.append(allocator, .{
        .name = ".shstrtab",
        .typ = SHT_STRTAB,
        .flags = 0,
        .addralign = 1,
        .has_bytes = true,
        .size = 0,
    });

    const shstrtab_ndx: u16 = @intCast(headers.items.len);
    const section_count: u16 = @intCast(headers.items.len + 1);

    // Assign each section's file offset, 8-byte aligned. A no-bytes section takes no file
    // space. `.shstrtab` (the last header) is laid out after its bytes exist.
    const offsets = try allocator.alloc(u64, headers.items.len);
    defer allocator.free(offsets);
    var off: u64 = w.ehsize;
    for (headers.items, 0..) |h, i| {
        off = alignUp(off, 8);
        offsets[i] = off;
        if (h.has_bytes) off += h.size;
    }
    const name_in_shstr = try allocator.alloc(u32, headers.items.len);
    defer allocator.free(name_in_shstr);
    for (headers.items, 0..) |h, i| name_in_shstr[i] = try shstrtab.add(allocator, h.name);
    const shstr_idx = headers.items.len - 1;
    offsets[shstr_idx] = alignUp(off, 8);
    off = offsets[shstr_idx] + shstrtab.bytes.items.len;

    off = alignUp(off, 8);
    const shoff = off;
    const total = shoff + @as(u64, section_count) * w.shentsize;

    var buf = try allocator.alloc(u8, @intCast(total));
    errdefer allocator.free(buf);
    @memset(buf, 0);

    writeEhdr(buf, cfg, shoff, w, section_count, shstrtab_ndx);

    // The reserved null section header is all zero.
    writeShdr(buf, cfg.class, shoff, 0, w.shentsize, .{
        .name = ".null",
        .typ = 0,
        .flags = 0,
        .addralign = 0,
        .has_bytes = false,
        .size = 0,
    }, 0, 0);
    for (headers.items, 0..) |h, i| {
        const ndx: u16 = @intCast(i + 1);
        // `.shstrtab` is the last header. Its bytes and size are only known now.
        var out = h;
        const content = if (i == shstr_idx) shstrtab.bytes.items else h.bytes;
        if (i == shstr_idx) out.size = shstrtab.bytes.items.len;
        if (h.has_bytes and content.len > 0) @memcpy(buf[@intCast(offsets[i])..][0..content.len], content);
        writeShdr(buf, cfg.class, shoff, ndx, w.shentsize, out, offsets[i], name_in_shstr[i]);
    }

    return buf;
}

/// Write one symbol-table entry at the class-specific width. The ELF32 and ELF64 symbol
/// structures order their fields differently, so each width has its own layout.
fn writeSym(dst: []u8, class: Class, name: u32, info: u8, shndx: u16, value: u64, size: u64) void {
    switch (class) {
        .elf64 => {
            putInt(dst[0..4], u32, name);
            dst[4] = info;
            dst[5] = 0; // st_other
            putInt(dst[6..8], u16, shndx);
            putInt(dst[8..16], u64, value);
            putInt(dst[16..24], u64, size);
        },
        .elf32 => {
            putInt(dst[0..4], u32, name);
            putInt(dst[4..8], u32, @truncate(value));
            putInt(dst[8..12], u32, @truncate(size));
            dst[12] = info;
            dst[13] = 0; // st_other
            putInt(dst[14..16], u16, shndx);
        },
    }
}

/// Write one relocation entry at the class-specific width. `use_rela` adds the addend
/// field. The `r_info` packing differs by width: `(sym << 32) | type` for ELF64, and
/// `(sym << 8) | type` for ELF32.
fn writeReloc(dst: []u8, class: Class, use_rela: bool, offset: u64, sym: u32, r_type: u32, addend: i64) void {
    switch (class) {
        .elf64 => {
            putInt(dst[0..8], u64, offset);
            putInt(dst[8..16], u64, (@as(u64, sym) << 32) | r_type);
            if (use_rela) putInt(dst[16..24], i64, addend);
        },
        .elf32 => {
            putInt(dst[0..4], u32, @truncate(offset));
            putInt(dst[4..8], u32, (sym << 8) | (r_type & 0xff));
            if (use_rela) putInt(dst[8..12], i32, @truncate(addend));
        },
    }
}

/// Write the ELF header for the chosen width into the start of `buf`.
fn writeEhdr(buf: []u8, cfg: Config, shoff: u64, w: Widths, section_count: u16, shstrtab_ndx: u16) void {
    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = switch (cfg.class) {
        .elf32 => ELFCLASS32,
        .elf64 => ELFCLASS64,
    };
    buf[5] = ELFDATA2LSB;
    buf[6] = 1; // EI_VERSION
    putInt(buf[16..18], u16, ET_REL);
    putInt(buf[18..20], u16, cfg.machine);
    putInt(buf[20..24], u32, EV_CURRENT);
    switch (cfg.class) {
        .elf64 => {
            putInt(buf[40..48], u64, shoff);
            putInt(buf[52..54], u16, @intCast(w.ehsize));
            putInt(buf[58..60], u16, @intCast(w.shentsize));
            putInt(buf[60..62], u16, section_count);
            putInt(buf[62..64], u16, shstrtab_ndx);
        },
        .elf32 => {
            putInt(buf[32..36], u32, @intCast(shoff));
            putInt(buf[40..42], u16, @intCast(w.ehsize));
            putInt(buf[46..48], u16, @intCast(w.shentsize));
            putInt(buf[48..50], u16, section_count);
            putInt(buf[50..52], u16, shstrtab_ndx);
        },
    }
}

/// Write one section header at index `idx` for the chosen width.
fn writeShdr(buf: []u8, class: Class, shoff: u64, idx: u16, shentsize: u64, h: Shdr, offset: u64, name: u32) void {
    const e = buf[@intCast(shoff + idx * shentsize)..][0..@intCast(shentsize)];
    switch (class) {
        .elf64 => {
            putInt(e[0..4], u32, name);
            putInt(e[4..8], u32, h.typ);
            putInt(e[8..16], u64, h.flags);
            putInt(e[16..24], u64, 0); // sh_addr
            putInt(e[24..32], u64, offset);
            putInt(e[32..40], u64, h.size);
            putInt(e[40..44], u32, h.link);
            putInt(e[44..48], u32, h.info);
            putInt(e[48..56], u64, h.addralign);
            putInt(e[56..64], u64, h.entsize);
        },
        .elf32 => {
            putInt(e[0..4], u32, name);
            putInt(e[4..8], u32, h.typ);
            putInt(e[8..12], u32, @truncate(h.flags));
            putInt(e[12..16], u32, 0); // sh_addr
            putInt(e[16..20], u32, @intCast(offset));
            putInt(e[20..24], u32, @intCast(h.size));
            putInt(e[24..28], u32, h.link);
            putInt(e[28..32], u32, h.info);
            putInt(e[32..36], u32, @intCast(h.addralign));
            putInt(e[36..40], u32, @intCast(h.entsize));
        },
    }
}

const testing = std.testing;

/// Build the shared neutral object the round-trip tests reuse: two exec sections
/// (`.text.a`, `.text.b`), a read-only `.rodata.k`, and a zero-fill `.bss.z`, plus a FUNC
/// symbol in each text section, an OBJECT symbol in `.rodata.k`, one undefined import, and
/// one LOCAL symbol placed after two GLOBAL symbols in the input array. `emit` must sort
/// that LOCAL symbol ahead of every GLOBAL one, so its post-sort index differs from its
/// input index. One reloc in `.text.a` targets symbol `b` (input index 1), a GLOBAL symbol
/// the sort pushes past the newly inserted LOCAL one, so its post-sort index also differs
/// from its input index. This exercises the reloc-symbol remap, not just the sort.
const text_a_bytes = [_]u8{ 0x90, 0x90, 0x90, 0x90 };
const text_b_bytes = [_]u8{ 0xcc, 0xcc, 0xcc, 0xcc };
const rodata_k_bytes = [_]u8{ 42, 0, 0, 0 };

fn buildSections(reloc: []const OutReloc) [4]OutSection {
    return .{
        .{ .name = ".text.a", .sh_type = SHT_PROGBITS, .flags = 0x2 | 0x4, .bytes = &text_a_bytes, .size = text_a_bytes.len, .addralign = 4, .relocs = reloc },
        .{ .name = ".text.b", .sh_type = SHT_PROGBITS, .flags = 0x2 | 0x4, .bytes = &text_b_bytes, .size = text_b_bytes.len, .addralign = 4 },
        .{ .name = ".rodata.k", .sh_type = SHT_PROGBITS, .flags = 0x2, .bytes = &rodata_k_bytes, .size = rodata_k_bytes.len, .addralign = 4 },
        .{ .name = ".bss.z", .sh_type = SHT_NOBITS, .flags = 0x2 | 0x1, .size = 8, .addralign = 8 },
    };
}

const round_trip_symbols = [_]OutSymbol{
    .{ .name = "a", .section = 0, .value = 0, .size = 4, .binding = .global, .sym_type = .func, .defined = true },
    .{ .name = "b", .section = 1, .value = 0, .size = 4, .binding = .global, .sym_type = .func, .defined = true },
    // A LOCAL symbol, placed after the two GLOBAL symbols above. `emit` must move it
    // ahead of every GLOBAL symbol in the output symbol table, so this input index (2)
    // is not its post-sort index.
    .{ .name = "lbl", .section = 0, .value = 2, .size = 1, .binding = .local, .sym_type = .notype, .defined = true },
    .{ .name = "k", .section = 2, .value = 0, .size = 4, .binding = .global, .sym_type = .object, .defined = true },
    .{ .name = "ext", .section = 0, .value = 0, .size = 0, .binding = .global, .sym_type = .notype, .defined = false },
};

// The parser this file round-trips against lives in the linker (`vulcan-link`).
// `vulcan-target` already depends on `vulcan-link` (see `build.zig`), so importing it
// here creates no dependency cycle. The public `elf.parseObject` dispatches on the ELF
// class byte to the same code paths the brief names `parseObject64`/`parseObject32`.
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

test "emit round-trips an ELF64 object through the linker parser" {
    const elf = @import("vulcan-link").elf;
    const allocator = testing.allocator;

    // `R_X86_64_PLT32` (numeric 4) is a call-style text reloc the parser accepts.
    const relocs = [_]OutReloc{.{ .offset = 1, .symbol = 1, .r_type = 4, .addend = -4 }};
    const sections = buildSections(&relocs);

    const bytes = try emit(allocator, &sections, &round_trip_symbols, .{ .class = .elf64, .machine = 62, .use_rela = true });
    defer allocator.free(bytes);

    var parsed = try elf.parseObject(allocator, bytes);
    defer parsed.deinit(allocator);

    try testing.expectEqual(elf.Arch.x86_64, parsed.arch);
    try checkParsed(&parsed);
}

test "emit round-trips an ELF32 object through the linker parser" {
    const elf = @import("vulcan-link").elf;
    const allocator = testing.allocator;

    // `R_386_PC32` (numeric 2). ELF32 uses `SHT_REL`, so the addend is ignored by `emit`
    // and the parser reads none.
    const relocs = [_]OutReloc{.{ .offset = 1, .symbol = 1, .r_type = 2, .addend = 0 }};
    const sections = buildSections(&relocs);

    const bytes = try emit(allocator, &sections, &round_trip_symbols, .{ .class = .elf32, .machine = 3, .use_rela = false });
    defer allocator.free(bytes);

    // The class byte and machine confirm the ELF32/i386 width was written.
    try testing.expectEqual(ELFCLASS32, bytes[4]);
    try testing.expectEqual(@as(u16, 3), std.mem.readInt(u16, bytes[18..20], .little));

    var parsed = try elf.parseObject(allocator, bytes);
    defer parsed.deinit(allocator);

    try testing.expectEqual(elf.Arch.x86, parsed.arch);
    try checkParsed(&parsed);
}

/// Assert the parsed object holds the four sections by name with the right flags and
/// sizes, that symbol `b` resolves to `.text.b` with size 4, that the LOCAL symbol `lbl`
/// sorted ahead of every GLOBAL symbol, and that `.text.a` carries the one reloc at
/// offset 1 targeting `b` at `b`'s post-sort index, not its input index.
fn checkParsed(parsed: *const @import("vulcan-link").elf.ParsedObject) !void {
    try testing.expectEqual(@as(usize, 4), parsed.sections.len);

    const a = parsed.sections[0];
    const b = parsed.sections[1];
    const k = parsed.sections[2];
    const z = parsed.sections[3];
    try testing.expectEqualStrings(".text.a", a.name);
    try testing.expectEqualStrings(".text.b", b.name);
    try testing.expectEqualStrings(".rodata.k", k.name);
    try testing.expectEqualStrings(".bss.z", z.name);

    try testing.expectEqual(SHF_ALLOC | SHF_EXECINSTR, a.flags);
    try testing.expectEqual(false, a.is_nobits);
    try testing.expectEqual(@as(u64, 4), a.size);
    try testing.expectEqual(SHF_ALLOC, k.flags);
    try testing.expectEqual(SHF_ALLOC | SHF_WRITE, z.flags);
    try testing.expectEqual(true, z.is_nobits);
    try testing.expectEqual(@as(u64, 8), z.size);

    // Five input symbols plus the reserved null symbol at index 0: `a`, `b`, `lbl`, `k`,
    // `ext`. Find each parsed symbol by name, so the check holds regardless of table
    // order, then pin the post-sort slot each one must land in.
    try testing.expectEqual(@as(usize, 6), parsed.symbols.len);

    var lbl_index: ?u32 = null;
    var b_index: ?u32 = null;
    for (parsed.symbols, 0..) |s, i| {
        // Index 0 is the reserved null symbol (empty name); skip it, it carries no
        // binding this test cares about.
        if (s.name.len == 0) continue;
        if (std.mem.eql(u8, s.name, "lbl")) {
            lbl_index = @intCast(i);
            try testing.expectEqual(true, s.local);
        } else {
            // `a`, `b`, `k`, and `ext` are all GLOBAL: none may carry `lbl`'s binding.
            try testing.expectEqual(false, s.local);
        }
        if (std.mem.eql(u8, s.name, "b")) {
            b_index = @intCast(i);
            try testing.expectEqual(@as(u32, 1), s.section_index);
            try testing.expectEqual(@as(u64, 4), s.size);
        }
    }
    try testing.expect(lbl_index != null);
    try testing.expect(b_index != null);

    // `lbl` was input index 2, after `a` (index 0) and `b` (index 1). The sort must move
    // it to symbol-table slot 1, right after the reserved null symbol and ahead of every
    // GLOBAL symbol: proof the locals-before-globals sort ran, not just that `lbl` is
    // present.
    try testing.expectEqual(@as(u32, 1), lbl_index.?);
    // `b` was input index 1. With `lbl` sorted ahead of it, `b` lands at post-sort slot 3,
    // not slot 2 (its slot if the sort were a no-op): proof the sort actually reordered
    // `b`, not only the newly added local.
    try testing.expectEqual(@as(u32, 3), b_index.?);

    // The one `.text.a` reloc sits at offset 1 and targets `b` at `b`'s NEW, remapped
    // index. A remap bug (for example, forgetting to apply the sort permutation to
    // `OutReloc.symbol`) would leave this pointing at the wrong symbol.
    try testing.expectEqual(@as(usize, 1), a.relocs.len);
    try testing.expectEqual(@as(u64, 1), a.relocs[0].offset);
    try testing.expectEqual(b_index.?, a.relocs[0].symbol);
}
