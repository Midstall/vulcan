//! This file emits an ELF64 relocatable object (`ET_REL`, `EM_AARCH64`). Each function
//! becomes an `STT_FUNC` global symbol in `.text`. Each call (`bl`) becomes an
//! `R_AARCH64_CALL26` relocation against the callee symbol. The symbol is undefined if
//! external. Each data global (`module.data`) becomes an `STT_OBJECT` symbol in
//! `.rodata`, `.data`, or `.bss`. A `global_addr`'s `adrp`/`add` pair becomes an
//! `R_AARCH64_ADR_PREL_PG_HI21` / `R_AARCH64_ADD_ABS_LO12_NC` relocation pair against
//! the symbol. `readelf` and a system AArch64 linker accept the output. `ld.zig` is
//! Vulcan's own linker for this object format.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");
const dwarf = @import("../dwarf.zig");

const Function = ir.function.Function;

pub const Error = isel.Error;

/// A symbol's binding. Locals must precede globals in the symbol table.
pub const Binding = enum { local, global };

/// A symbol's type. `func` is an entry point. `object` is a data object. `notype` is
/// unknown.
pub const SymKind = enum { notype, func, object };

/// The allocatable output sections for a symbol. `.text` holds code. `.rodata` holds
/// read-only data. `.data` holds writable data. `.bss` holds zero-initialized data.
/// `.bss` uses memory but no file bytes.
pub const SectionKind = enum { text, rodata, data, bss };

/// One symbol table entry. A defined symbol lives in `section` at `value`. An
/// undefined symbol (`defined = false`) is external. The linker must resolve it.
pub const Symbol = struct {
    name: []const u8,
    value: u64 = 0,
    size: u64 = 0,
    binding: Binding = .global,
    kind: SymKind = .func,
    defined: bool = true,
    section: SectionKind = .text,
};

/// The AArch64 relocation types this file emits (the architectural `R_AARCH64_*` codes).
pub const RelocType = enum(u32) {
    /// `R_AARCH64_ADR_PREL_PG_HI21`. It patches an `adrp` instruction's page-relative
    /// immediate. This is the high half of a `global_addr`'s adrp/add pair.
    adr_prel_pg_hi21 = 275,
    /// `R_AARCH64_ADD_ABS_LO12_NC`. It patches an `add` instruction's 12-bit page-offset
    /// immediate. This is the low half of a `global_addr`'s adrp/add pair.
    add_abs_lo12_nc = 277,
    /// `R_AARCH64_CALL26`. It patches a `bl` or `b` instruction's 26-bit immediate. The
    /// call range is plus or minus 128 MiB.
    call26 = 283,
    /// `R_AARCH64_ADR_GOT_PAGE`. It patches an `adrp` instruction's page-relative
    /// immediate to the page of the symbol's GOT entry. This is the high half of a
    /// GOT-indirect `adrp`/`ldr` pair.
    adr_got_page = 311,
    /// `R_AARCH64_LD64_GOT_LO12_NC`. It patches an `ldr` instruction's 12-bit unsigned
    /// offset to the GOT entry's lo12 value shifted right by 3. This is the low half of a
    /// GOT-indirect `adrp`/`ldr` pair.
    ld64_got_lo12_nc = 312,
};

/// A relocation applied to a `.text` byte offset against a symbol.
pub const Reloc = struct {
    offset: u64,
    symbol: u32,
    type: RelocType,
    addend: i64 = 0,
};

/// `R_AARCH64_ABS64`: a 64-bit absolute address (`S + A`) written into a data section
/// slot. A pointer-initialized global (`int *p = &g;`) carries this relocation in
/// `.rela.data` or `.rela.rodata`. The linker must patch the 8-byte slot that holds the
/// pointer to the target symbol's runtime address. The linker turns it into an
/// `R_AARCH64_RELATIVE` dynamic relocation for a PIE or shared object, or a direct
/// absolute write for a non-PIE executable.
pub const R_AARCH64_ABS64: u32 = 257;

/// A relocation applied to a data section (`.data` or `.rodata`) slot against a symbol.
/// At byte `offset` within `section`, an `R_AARCH64_ABS64` relocation writes `symbol`'s
/// runtime address plus `addend`. This is emitted into `.rela.data` or `.rela.rodata`,
/// keyed by `section`. A data global's own pointer initializers are now carried in the
/// object, not dropped.
pub const DataRelocEntry = struct {
    section: SectionKind,
    offset: u64,
    symbol: u32,
    addend: i64 = 0,
};

/// A non-alloc PROGBITS section carried without change (for example a DWARF `.debug_*`
/// blob).
pub const DebugSection = struct { name: []const u8, bytes: []const u8 };

/// A relocatable object: code and data section blobs, the symbol table, and the
/// relocations that apply to `.text`. `.bss` has no bytes, only a size. The `debug`
/// sections (DWARF) are appended as plain PROGBITS, so a compiled object can carry
/// debug info.
pub const Object = struct {
    text: []const u8,
    rodata: []const u8 = &.{},
    data: []const u8 = &.{},
    bss_size: u64 = 0,
    symbols: []const Symbol,
    relocs: []const Reloc,
    /// Relocations applied to the `.data` and `.rodata` sections themselves (pointer
    /// initializers), emitted as `.rela.data`/`.rela.rodata`. This list is empty for a
    /// code-only or const-only object.
    data_relocs: []const DataRelocEntry = &.{},
    debug: []const DebugSection = &.{},
};

const ET_REL: u16 = 1;
const EM_AARCH64: u16 = 183;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_INFO_LINK: u64 = 0x40;
const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const SHN_UNDEF: u16 = 0;

const ehsize: u64 = 64;
const shentsize: u64 = 64;
const symentsize: u64 = 24;
const relaentsize: u64 = 24;

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

/// One emitted section header.
const Shdr = struct {
    name: []const u8,
    typ: u32,
    flags: u64,
    addralign: u64,
    entsize: u64 = 0,
    link: u16 = 0,
    info: u32 = 0,
    bytes: ?[]const u8,
    size: u64,
};

/// Serialize `obj` into an ELF64 AArch64 relocatable object. The output has these
/// sections: `.text`. `.rodata`, `.data`, and `.bss` when they are non-empty.
/// `.rela.text` when there are relocations. `.symtab`. `.strtab`. `.shstrtab`. The
/// caller owns the result.
pub fn write(allocator: std.mem.Allocator, obj: Object) Error![]u8 {
    var local_count: u32 = 0;
    var seen_global = false;
    for (obj.symbols) |s| switch (s.binding) {
        .local => {
            if (seen_global) return error.Unsupported; // a local symbol after a global one
            local_count += 1;
        },
        .global => seen_global = true,
    };
    const first_global: u32 = 1 + local_count; // the null symbol at 0 is local

    const has_rodata = obj.rodata.len > 0;
    const has_data = obj.data.len > 0;
    const has_bss = obj.bss_size > 0;
    const has_rela = obj.relocs.len > 0;
    // Split the data-section relocations by the section they modify: `.rela.data` or
    // `.rela.rodata`. Each group needs its own `SHT_RELA` header. Its `sh_info` field
    // names the target section.
    var rela_data_count: usize = 0;
    var rela_rodata_count: usize = 0;
    for (obj.data_relocs) |dr| switch (dr.section) {
        .data => rela_data_count += 1,
        .rodata => rela_rodata_count += 1,
        .text, .bss => {}, // a data relocation never lives in .text or .bss
    };
    const has_rela_data = rela_data_count > 0;
    const has_rela_rodata = rela_rodata_count > 0;
    var next: u16 = 1;
    const text_ndx = next;
    next += 1;
    const rodata_ndx = if (has_rodata) blk: {
        defer next += 1;
        break :blk next;
    } else 0;
    const data_ndx = if (has_data) blk: {
        defer next += 1;
        break :blk next;
    } else 0;
    const bss_ndx = if (has_bss) blk: {
        defer next += 1;
        break :blk next;
    } else 0;
    if (has_rela) next += 1; // .rela.text
    if (has_rela_data) next += 1; // .rela.data
    if (has_rela_rodata) next += 1; // .rela.rodata
    const symtab_ndx = next;
    next += 1;
    const strtab_ndx = next;
    next += 1;
    // The .shstrtab index and the total section count are set after the header list is
    // built. The `debug` sections are inserted before it.

    const shndxOf = struct {
        fn f(kind: SectionKind, t: u16, ro: u16, d: u16, b: u16) u16 {
            return switch (kind) {
                .text => t,
                .rodata => ro,
                .data => d,
                .bss => b,
            };
        }
    }.f;

    // The symbol-name string table.
    var strtab = try StrTab.init(allocator);
    defer strtab.deinit(allocator);
    var name_offsets = try allocator.alloc(u32, obj.symbols.len);
    defer allocator.free(name_offsets);
    for (obj.symbols, 0..) |s, i| name_offsets[i] = try strtab.add(allocator, s.name);

    // The symbol table.
    const sym_bytes = (1 + obj.symbols.len) * symentsize;
    var symtab = try allocator.alloc(u8, sym_bytes);
    defer allocator.free(symtab);
    @memset(symtab, 0);
    for (obj.symbols, 0..) |s, i| {
        const e = symtab[(i + 1) * symentsize ..][0..symentsize];
        const binding: u8 = switch (s.binding) {
            .local => STB_LOCAL,
            .global => STB_GLOBAL,
        };
        const typ: u8 = switch (s.kind) {
            .notype => STT_NOTYPE,
            .func => STT_FUNC,
            .object => STT_OBJECT,
        };
        putInt(e[0..4], u32, name_offsets[i]); // st_name
        e[4] = (binding << 4) | typ; // st_info
        e[5] = 0; // st_other
        const shndx: u16 = if (s.defined) shndxOf(s.section, text_ndx, rodata_ndx, data_ndx, bss_ndx) else SHN_UNDEF;
        putInt(e[6..8], u16, shndx); // st_shndx
        putInt(e[8..16], u64, s.value); // st_value
        putInt(e[16..24], u64, s.size); // st_size
    }

    // Relocation table. It applies to `.text`.
    const rela_bytes = obj.relocs.len * relaentsize;
    var rela = try allocator.alloc(u8, rela_bytes);
    defer allocator.free(rela);
    for (obj.relocs, 0..) |r, i| {
        const e = rela[i * relaentsize ..][0..relaentsize];
        const sym_index: u64 = @as(u64, r.symbol) + 1; // null entry at 0
        putInt(e[0..8], u64, r.offset); // r_offset
        putInt(e[8..16], u64, (sym_index << 32) | @intFromEnum(r.type)); // r_info
        putInt(e[16..24], i64, r.addend); // r_addend
    }

    // `.rela.data` and `.rela.rodata` hold the data-section pointer-init relocations.
    // Each one is an `R_AARCH64_ABS64` relocation against the target symbol. They are
    // grouped by the section they modify, so each group can name its target section in
    // `sh_info`.
    var rela_data = try allocator.alloc(u8, rela_data_count * relaentsize);
    defer allocator.free(rela_data);
    var rela_rodata = try allocator.alloc(u8, rela_rodata_count * relaentsize);
    defer allocator.free(rela_rodata);
    {
        var di: usize = 0;
        var ri: usize = 0;
        for (obj.data_relocs) |dr| {
            const dst = switch (dr.section) {
                .data => blk: {
                    const e = rela_data[di * relaentsize ..][0..relaentsize];
                    di += 1;
                    break :blk e;
                },
                .rodata => blk: {
                    const e = rela_rodata[ri * relaentsize ..][0..relaentsize];
                    ri += 1;
                    break :blk e;
                },
                .text, .bss => continue,
            };
            const sym_index: u64 = @as(u64, dr.symbol) + 1; // null entry at 0
            putInt(dst[0..8], u64, dr.offset); // r_offset
            putInt(dst[8..16], u64, (sym_index << 32) | R_AARCH64_ABS64); // r_info
            putInt(dst[16..24], i64, dr.addend); // r_addend
        }
    }

    var shstrtab = try StrTab.init(allocator);
    defer shstrtab.deinit(allocator);

    var headers: std.ArrayList(Shdr) = .empty;
    defer headers.deinit(allocator);
    try headers.append(allocator, .{ .name = ".text", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_EXECINSTR, .addralign = 4, .bytes = obj.text, .size = obj.text.len });
    if (has_rodata) try headers.append(allocator, .{ .name = ".rodata", .typ = SHT_PROGBITS, .flags = SHF_ALLOC, .addralign = 8, .bytes = obj.rodata, .size = obj.rodata.len });
    if (has_data) try headers.append(allocator, .{ .name = ".data", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_WRITE, .addralign = 8, .bytes = obj.data, .size = obj.data.len });
    if (has_bss) try headers.append(allocator, .{ .name = ".bss", .typ = SHT_NOBITS, .flags = SHF_ALLOC | SHF_WRITE, .addralign = 8, .bytes = null, .size = obj.bss_size });
    if (has_rela) try headers.append(allocator, .{ .name = ".rela.text", .typ = SHT_RELA, .flags = SHF_INFO_LINK, .addralign = 8, .entsize = relaentsize, .link = symtab_ndx, .info = text_ndx, .bytes = rela, .size = rela_bytes });
    if (has_rela_data) try headers.append(allocator, .{ .name = ".rela.data", .typ = SHT_RELA, .flags = SHF_INFO_LINK, .addralign = 8, .entsize = relaentsize, .link = symtab_ndx, .info = data_ndx, .bytes = rela_data, .size = rela_data.len });
    if (has_rela_rodata) try headers.append(allocator, .{ .name = ".rela.rodata", .typ = SHT_RELA, .flags = SHF_INFO_LINK, .addralign = 8, .entsize = relaentsize, .link = symtab_ndx, .info = rodata_ndx, .bytes = rela_rodata, .size = rela_rodata.len });
    try headers.append(allocator, .{ .name = ".symtab", .typ = SHT_SYMTAB, .flags = 0, .addralign = 8, .entsize = symentsize, .link = strtab_ndx, .info = first_global, .bytes = symtab, .size = sym_bytes });
    try headers.append(allocator, .{ .name = ".strtab", .typ = SHT_STRTAB, .flags = 0, .addralign = 1, .bytes = strtab.bytes.items, .size = strtab.bytes.items.len });
    // DWARF (or other) debug sections. They are plain PROGBITS, and no other section
    // refers to them.
    for (obj.debug) |d| try headers.append(allocator, .{ .name = d.name, .typ = SHT_PROGBITS, .flags = 0, .addralign = 1, .bytes = d.bytes, .size = d.bytes.len });
    try headers.append(allocator, .{ .name = ".shstrtab", .typ = SHT_STRTAB, .flags = 0, .addralign = 1, .bytes = null, .size = 0 });

    // `.shstrtab` is the last header. The section count includes the leading null
    // section.
    const shstrtab_ndx: u16 = @intCast(headers.items.len);
    const section_count: u16 = @intCast(headers.items.len + 1);

    var offsets = try allocator.alloc(u64, headers.items.len);
    defer allocator.free(offsets);
    var off: u64 = ehsize;
    for (headers.items, 0..) |h, i| {
        off = alignUp(off, 8);
        offsets[i] = off;
        if (h.bytes != null) off += h.size;
    }
    var name_in_shstr = try allocator.alloc(u32, headers.items.len);
    defer allocator.free(name_in_shstr);
    for (headers.items, 0..) |h, i| name_in_shstr[i] = try shstrtab.add(allocator, h.name);
    const shstr_idx = headers.items.len - 1;
    offsets[shstr_idx] = alignUp(off, 8);
    off = offsets[shstr_idx] + shstrtab.bytes.items.len;

    off = alignUp(off, 8);
    const shoff = off;
    const total = shoff + @as(u64, section_count) * shentsize;

    var buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);
    @memset(buf, 0);

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    putInt(buf[16..18], u16, ET_REL);
    putInt(buf[18..20], u16, EM_AARCH64);
    putInt(buf[20..24], u32, 1); // e_version
    putInt(buf[40..48], u64, shoff); // e_shoff
    putInt(buf[52..54], u16, @intCast(ehsize)); // e_ehsize
    putInt(buf[58..60], u16, @intCast(shentsize)); // e_shentsize
    putInt(buf[60..62], u16, section_count); // e_shnum
    putInt(buf[62..64], u16, shstrtab_ndx); // e_shstrndx

    putShdr(buf, shoff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (headers.items, 0..) |h, i| {
        const ndx: u16 = @intCast(i + 1);
        const content = if (i == shstr_idx) shstrtab.bytes.items else (h.bytes orelse &.{});
        const size = if (i == shstr_idx) shstrtab.bytes.items.len else h.size;
        if (content.len > 0) @memcpy(buf[offsets[i]..][0..content.len], content);
        putShdr(buf, shoff, ndx, name_in_shstr[i], h.typ, h.flags, offsets[i], size, h.link, h.info, h.addralign, h.entsize);
    }

    return buf;
}

fn putShdr(buf: []u8, shoff: u64, idx: u16, name: u32, typ: u32, flags: u64, offset: u64, size: u64, sh_link: u32, sh_info: u32, addralign: u64, entsize: u64) void {
    const e = buf[shoff + idx * shentsize ..][0..shentsize];
    putInt(e[0..4], u32, name);
    putInt(e[4..8], u32, typ);
    putInt(e[8..16], u64, flags);
    putInt(e[16..24], u64, 0); // sh_addr
    putInt(e[24..32], u64, offset);
    putInt(e[32..40], u64, size);
    putInt(e[40..44], u32, sh_link);
    putInt(e[44..48], u32, sh_info);
    putInt(e[48..56], u64, addralign);
    putInt(e[56..64], u64, entsize);
}

const Pending = struct { offset: u64, name: []const u8, kind: isel.Kind };

/// Map an isel relocation's `kind` to the matching ELF relocation type. The `kind` says
/// which half of a call, or an adrp/add pair, the relocation patches.
fn relocTypeOf(kind: isel.Kind) RelocType {
    return switch (kind) {
        .call => .call26,
        .adrp_pg => .adr_prel_pg_hi21,
        .add_pgoff => .add_abs_lo12_nc,
        .got_pg => .adr_got_page,
        .got_lo12 => .ld64_got_lo12_nc,
    };
}

/// A data-section pointer-init relocation before its target name resolves to a symbol
/// index. It holds the section and byte offset of the slot, and the target symbol's
/// name.
const PendingDataReloc = struct { section: SectionKind, offset: u64, symbol: []const u8 };

/// Lay out `module.data` into `.rodata`/`.data` byte buffers and a `.bss` size. For each
/// data global, append an `STT_OBJECT` symbol (its section and offset) to `globals`. Each
/// data global's own `DataReloc`s (pointer initializers) are recorded in `data_relocs` at
/// their absolute section offset. This offset is the global's placement plus the
/// relocation's offset within the object. The function returns the accumulated `.bss`
/// size. It grows `rodata` and `data` in place.
fn layoutData(allocator: std.mem.Allocator, module: *const link.Module, globals: *std.ArrayList(Symbol), rodata: *std.ArrayList(u8), data: *std.ArrayList(u8), data_relocs: *std.ArrayList(PendingDataReloc)) Error!u64 {
    var bss_size: u64 = 0;
    for (module.data.items) |d| {
        const section: SectionKind, const value: u64 = switch (d.kind) {
            .rodata => blk: {
                const start = rodata.items.len;
                try rodata.appendSlice(allocator, d.bytes);
                break :blk .{ .rodata, start };
            },
            .data => blk: {
                const start = data.items.len;
                try data.appendSlice(allocator, d.bytes);
                break :blk .{ .data, start };
            },
            .bss => blk: {
                const start = bss_size;
                bss_size += d.size;
                break :blk .{ .bss, start };
            },
        };
        for (d.relocs) |r| try data_relocs.append(allocator, .{ .section = section, .offset = value + r.off, .symbol = r.symbol });
        // An anonymous compiler-internal object has internal linkage. This includes a
        // string literal named `.str.N`, and any other `.`-prefixed assembler-local
        // name. Emit it with LOCAL binding. Every object numbers its own string literals
        // from zero, so two objects can both define `.str.2`. A local symbol keeps each
        // one private and avoids a link-time collision.
        const binding: Binding = if (std.mem.startsWith(u8, d.name, ".")) .local else .global;
        try globals.append(allocator, .{ .name = d.name, .value = value, .size = d.size, .kind = .object, .defined = true, .section = section, .binding = binding });
    }
    return bss_size;
}

/// Compile every function in `module`, and serialize them and its data globals into one
/// ELF relocatable object. Each function becomes a defined `STT_FUNC` symbol in `.text`.
/// Each data global becomes an `STT_OBJECT` symbol in `.rodata`, `.data`, or `.bss`. Each
/// `bl` becomes an `R_AARCH64_CALL26` relocation. Each `global_addr`'s `adrp`/`add` pair
/// becomes an `R_AARCH64_ADR_PREL_PG_HI21`/`R_AARCH64_ADD_ABS_LO12_NC` pair. Both kinds
/// of relocation target the referenced symbol, which is undefined if external. The
/// caller owns the returned ELF bytes.
pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    var globals: std.ArrayList(Symbol) = .empty;
    defer globals.deinit(allocator);
    var pending: std.ArrayList(Pending) = .empty;
    defer pending.deinit(allocator);

    const caps: isel.ModelCaps = if (module.model) |m| isel.capsForModel(m) else .{};
    for (module.functions.items) |entry| {
        const start: u64 = text.items.len;
        var compiled = try isel.compileFunction(allocator, entry.func, caps);
        defer compiled.deinit(allocator);
        for (compiled.code) |word| {
            var w: [4]u8 = undefined;
            putInt(&w, u32, word);
            try text.appendSlice(allocator, &w);
        }
        try globals.append(allocator, .{ .name = entry.name, .value = start, .size = @as(u64, compiled.code.len) * 4, .kind = .func, .defined = true, .binding = if (entry.func.is_local) .local else .global });
        for (compiled.relocs) |r| {
            try pending.append(allocator, .{ .offset = start + @as(u64, r.offset) * 4, .name = r.symbol, .kind = r.kind });
        }
    }

    var rodata: std.ArrayList(u8) = .empty;
    defer rodata.deinit(allocator);
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(allocator);
    var pending_data: std.ArrayList(PendingDataReloc) = .empty;
    defer pending_data.deinit(allocator);
    const bss_size = try layoutData(allocator, module, &globals, &rodata, &data, &pending_data);

    // The symbol table lists defined globals first, then any undefined external callees.
    var symbols: std.ArrayList(Symbol) = .empty;
    defer symbols.deinit(allocator);
    try symbols.appendSlice(allocator, globals.items);
    for (pending.items) |p| {
        if (symbolIndex(symbols.items, p.name) == null) {
            try symbols.append(allocator, .{ .name = p.name, .size = 0, .kind = .notype, .defined = false });
        }
    }
    // ELF requires every LOCAL symbol to precede every GLOBAL one. Since `static`
    // functions are local, the function list can interleave the two kinds. Stable-sort
    // locals ahead of globals before the relocations below resolve their target symbol
    // indices. Those indices must match the emitted order.
    std.mem.sort(Symbol, symbols.items, {}, localsFirst);

    var relocs = try allocator.alloc(Reloc, pending.items.len);
    defer allocator.free(relocs);
    for (pending.items, 0..) |p, i| {
        relocs[i] = .{ .offset = p.offset, .symbol = symbolIndex(symbols.items, p.name).?, .type = relocTypeOf(p.kind) };
    }

    // Resolve each data-section relocation's target name to a symbol index. The target
    // of a pointer initializer is an internally defined data or function global, added
    // above.
    var data_relocs = try allocator.alloc(DataRelocEntry, pending_data.items.len);
    defer allocator.free(data_relocs);
    for (pending_data.items, 0..) |p, i| {
        data_relocs[i] = .{ .section = p.section, .offset = p.offset, .symbol = symbolIndex(symbols.items, p.symbol) orelse return error.Unsupported };
    }

    return write(allocator, .{ .text = text.items, .rodata = rodata.items, .data = data.items, .bss_size = bss_size, .symbols = symbols.items, .relocs = relocs, .data_relocs = data_relocs });
}

fn symbolIndex(symbols: []const Symbol, name: []const u8) ?u32 {
    for (symbols, 0..) |s, i| if (std.mem.eql(u8, s.name, name)) return @intCast(i);
    return null;
}

/// Order a symbol table so every LOCAL binding sorts before every GLOBAL one
/// (`Binding.local` = 0, `.global` = 1). This is the ELF requirement that `write`
/// validates. Used with the stable `std.mem.sort`, so functions and data keep their
/// relative order within each binding group.
fn localsFirst(_: void, a: Symbol, b: Symbol) bool {
    return @intFromEnum(a.binding) < @intFromEnum(b.binding);
}

/// Like `writeModule`, but also emits inline DWARF (`.debug_abbrev`, `.debug_info`,
/// `.debug_line`) describing each function's name and PC range. It also maps code
/// offsets to `source_file` line numbers, taken from the `debug.line` IR attributes. The
/// result is a real relocatable object that carries debug info for objdump or gdb. The
/// caller owns the bytes.
pub fn writeModuleWithDebug(allocator: std.mem.Allocator, module: *const link.Module, source_file: []const u8) Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    var globals: std.ArrayList(Symbol) = .empty;
    defer globals.deinit(allocator);
    var pending: std.ArrayList(Pending) = .empty;
    defer pending.deinit(allocator);
    var rows: std.ArrayList(dwarf.LineRow) = .empty;
    defer rows.deinit(allocator);

    const caps: isel.ModelCaps = if (module.model) |m| isel.capsForModel(m) else .{};
    for (module.functions.items) |entry| {
        const start: u64 = text.items.len;
        var compiled = try isel.compileFunction(allocator, entry.func, caps);
        defer compiled.deinit(allocator);
        for (compiled.code) |word| {
            var w: [4]u8 = undefined;
            putInt(&w, u32, word);
            try text.appendSlice(allocator, &w);
        }
        try globals.append(allocator, .{ .name = entry.name, .value = start, .size = @as(u64, compiled.code.len) * 4, .kind = .func, .defined = true, .binding = if (entry.func.is_local) .local else .global });
        for (compiled.relocs) |r| {
            try pending.append(allocator, .{ .offset = start + @as(u64, r.offset) * 4, .name = r.symbol, .kind = r.kind });
        }
        // Line rows are function-relative. Shift each one to the module-relative .text
        // offset.
        for (compiled.lines) |e| try rows.append(allocator, .{ .address = start + e.offset, .line = e.line });
    }

    var symbols: std.ArrayList(Symbol) = .empty;
    defer symbols.deinit(allocator);
    try symbols.appendSlice(allocator, globals.items);
    for (pending.items) |p| {
        if (symbolIndex(symbols.items, p.name) == null) {
            try symbols.append(allocator, .{ .name = p.name, .size = 0, .kind = .notype, .defined = false });
        }
    }
    // See `writeModule`. Locals must precede globals, sorted before the relocations
    // resolve indices. The DWARF `subs` list below indexes `globals.items` in function
    // order, unsorted, so it is unaffected.
    std.mem.sort(Symbol, symbols.items, {}, localsFirst);
    var relocs = try allocator.alloc(Reloc, pending.items.len);
    defer allocator.free(relocs);
    for (pending.items, 0..) |p, i| {
        relocs[i] = .{ .offset = p.offset, .symbol = symbolIndex(symbols.items, p.name).?, .type = relocTypeOf(p.kind) };
    }

    // DWARF: one subprogram DIE per function. Its PC range is the function's .text
    // placement. It carries the function's IR return type as a base-type reference, so a
    // debugger can show a typed signature.
    const subs = try allocator.alloc(dwarf.Subprogram, globals.items.len);
    defer allocator.free(subs);
    for (globals.items, 0..) |g, i| subs[i] = .{
        .name = g.name,
        .low_pc = g.value,
        .high_pc = g.value + g.size,
        .ret_type = returnBaseType(module.functions.items[i].func),
    };

    const abbrev = try dwarf.emitAbbrev(allocator);
    defer allocator.free(abbrev);
    // The object carries one line program at offset 0 of .debug_line. Link the
    // compilation unit to it with DW_AT_stmt_list. A debugger can now go from a
    // subprogram DIE straight to its source lines.
    const info = try dwarf.emitInfo(allocator, .{ .name = source_file, .low_pc = 0, .high_pc = text.items.len, .subprograms = subs, .stmt_list = 0 });
    defer allocator.free(info);
    const line = try dwarf.emitLine(allocator, source_file, rows.items, text.items.len);
    defer allocator.free(line);

    return write(allocator, .{
        .text = text.items,
        .symbols = symbols.items,
        .relocs = relocs,
        .debug = &.{
            .{ .name = ".debug_abbrev", .bytes = abbrev },
            .{ .name = ".debug_info", .bytes = info },
            .{ .name = ".debug_line", .bytes = line },
        },
    });
}

/// Map a function's IR return type, the type of its `ret` value, to a DWARF base type.
/// Return null for a void return or a non-primitive (aggregate) return. The names are
/// C-like, so a debugger prints a natural signature. Distinct primitives get distinct
/// names, so the `.debug_info` base-type dedup keeps them apart.
fn returnBaseType(func: *const Function) ?dwarf.BaseType {
    const ret_val = for (0..func.blocks.items.len) |bi| {
        const term = func.terminator(@enumFromInt(bi)) orelse continue;
        switch (term) {
            .ret => |r| switch (r.count) {
                0 => return null,
                1 => break r.values[0],
                else => return null, // A multi-value return has no DWARF representation yet.
            },
            else => {},
        }
    } else return null;

    return switch (func.types.type_kind(func.valueType(ret_val))) {
        .bool => .{ .name = "bool", .encoding = .boolean, .byte_size = 1 },
        .float => |f| switch (f) {
            .f32 => .{ .name = "float", .encoding = .float, .byte_size = 4 },
            .f64 => .{ .name = "double", .encoding = .float, .byte_size = 8 },
            // This name is for debug info only. It does not affect lowering. AArch64 f16
            // codegen is future work.
            .f16 => .{ .name = "half", .encoding = .float, .byte_size = 2 },
        },
        .int => |i| blk: {
            const bytes: u8 = @intCast((i.bits + 7) / 8);
            const signed = i.signedness == .signed;
            const name: []const u8 = switch (i.bits) {
                8 => if (signed) "i8" else "u8",
                16 => if (signed) "i16" else "u16",
                32 => if (signed) "int" else "unsigned int",
                64 => if (signed) "long" else "unsigned long",
                else => if (signed) "int" else "unsigned",
            };
            break :blk .{ .name = name, .encoding = if (signed) .signed else .unsigned, .byte_size = bytes };
        },
        else => null, // A pointer, vector, or aggregate type has no base-type DIE here.
    };
}

test "returnBaseType reads the ret value's IR type" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, i32_t);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    const bt = returnBaseType(&func).?;
    try std.testing.expectEqualStrings("int", bt.name);
    try std.testing.expectEqual(dwarf.Encoding.signed, bt.encoding);
    try std.testing.expectEqual(@as(u8, 4), bt.byte_size);
}

test "returnBaseType is null for a void return" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const b = try func.appendBlock();
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    try std.testing.expectEqual(@as(?dwarf.BaseType, null), returnBaseType(&func));
}

test "writes an ELF64 AArch64 relocatable header" {
    const allocator = std.testing.allocator;
    const text = [_]u8{ 0, 0, 0, 0 };
    const symbols = [_]Symbol{
        .{ .name = "f", .value = 0, .size = text.len, .kind = .func },
        .{ .name = "ext", .defined = false },
    };
    const relocs = [_]Reloc{.{ .offset = 0, .symbol = 1, .type = .call26 }};
    const bytes = try write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
    defer allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, "\x7fELF", bytes[0..4]);
    try std.testing.expectEqual(@as(u8, 2), bytes[4]); // ELFCLASS64
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, bytes[16..18], .little)); // ET_REL
    try std.testing.expectEqual(@as(u16, 183), std.mem.readInt(u16, bytes[18..20], .little)); // EM_AARCH64
}

test "readelf accepts the emitted AArch64 object (cross-check)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a, and caller(x) = dbl(x) + 1. This call produces a CALL26 relocation.
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(i32k);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const d = try caller.appendCall(b, t, "dbl", &.{x});
        const r = try caller.appendArithImm(b, t, .add, d, 1);
        caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "caller", &caller);
    try module.addFunction(allocator, "dbl", &dbl);

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "out.o", .data = obj });

    // Run readelf with its cwd set to the temp dir, so the path is the bare file.
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "readelf", "-hr", "out.o" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // readelf unavailable
        else => return e,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    // A standard tool recognizes the machine and the relocation type.
    try std.testing.expect(std.mem.indexOf(u8, result.stdout, "AArch64") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.stdout, "R_AARCH64_CALL26") != null);
}

test "readelf shows a .rodata section and ADRP/ADD relocations for a global_addr load" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() returns *(&K), where K is an i32 rodata constant. This is the exact shape
    // that isel's `.global_addr` arm, the adrp/add pair, lowers to an
    // ADR_PREL_PG_HI21/ADD_ABS_LO12_NC pair.
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.intern(.ptr);
        const b = try entry.appendBlock();
        const g = try entry.appendGlobalAddr(b, ptr_t, "K");
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = g } });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }
    const k_bytes = [_]u8{ 42, 0, 0, 0 };
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addData(allocator, "K", &k_bytes); // .rodata

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "gd.o", .data = obj });

    const secs = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-S", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(secs.stdout);
    defer allocator.free(secs.stderr);
    if (secs.term != .exited or secs.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, ".rodata") != null);

    // The `-W` (wide) flag stops readelf from truncating the long `R_AARCH64_*` names to
    // fit its default column width.
    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-rW", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADR_PREL_PG_HI21") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADD_ABS_LO12_NC") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_CALL26") == null);

    const syms = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-s", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(syms.stdout);
    defer allocator.free(syms.stderr);
    try std.testing.expect(std.mem.indexOf(u8, syms.stdout, "OBJECT") != null);
}

test "readelf shows GOT relocations for a via_got global_addr load (data import)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() returns *(&G), where G is an imported data symbol in another shared
    // object. This is the exact shape that isel's `.global_addr` GOT arm, the adrp/ldr
    // pair, lowers to an ADR_GOT_PAGE/LD64_GOT_LO12_NC pair. G is left undefined (no
    // addData). Its address loads from the GOT at run time.
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.intern(.ptr);
        const b = try entry.appendBlock();
        const g = try entry.appendGlobalAddrGot(b, ptr_t, "G");
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = g } });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "got.o", .data = obj });

    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-rW", "got.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    if (rels.term != .exited or rels.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADR_GOT_PAGE") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_LD64_GOT_LO12_NC") != null);
    // The GOT path replaces the direct pair, so those must NOT appear for G.
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADR_PREL_PG_HI21") == null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADD_ABS_LO12_NC") == null);
}
