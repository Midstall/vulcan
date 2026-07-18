//! ELF64 relocatable object (`ET_REL`, `EM_RISCV`) emission. Turns a `.text`
//! blob, a symbol table, and a list of relocations into a real `.o` that a
//! RISC-V linker (and `readelf`) accepts. Calls are emitted as `R_RISCV_JAL`
//! relocations, matching the single-`jal` call lowering in isel.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");
const ld = @import("ld.zig");
const encode = @import("encode.zig");
const dwarf = @import("../dwarf.zig");
const harness = @import("tests/harness.zig");

pub const Error = std.mem.Allocator.Error || isel.Error;

/// A symbol's binding. Locals must be listed before globals (ELF requires the
/// local symbols to form a prefix of the symbol table).
pub const Binding = enum { local, global };

/// A symbol's type. `func` marks a function entry, `object` a data object,
/// `notype` an unknown (e.g. a local PC-relative label).
pub const SymKind = enum { notype, func, object };

/// The allocatable output sections an object can carry. `.text` is code,
/// `.rodata` read-only data, `.data` writable data, `.bss` zero-initialized
/// data (occupies memory but no file bytes).
pub const SectionKind = enum { text, rodata, data, bss };

/// One entry in the object's symbol table. A defined symbol lives in `section`
/// at `value`. An undefined one (`defined = false`) is an external reference the
/// linker must resolve.
pub const Symbol = struct {
    name: []const u8,
    value: u64 = 0,
    size: u64 = 0,
    binding: Binding = .global,
    kind: SymKind = .func,
    defined: bool = true,
    section: SectionKind = .text,
};

/// The emitted RISC-V relocation types. Values are the architectural
/// `R_RISCV_*` codes that land in the high half of `r_info`.
pub const RelocType = enum(u32) {
    /// `R_RISCV_JAL`: patch a `jal`'s 20-bit immediate (a +/-1MiB call).
    jal = 17,
    /// `R_RISCV_CALL`: patch an `auipc`+`jalr` pair (a long call).
    call = 18,
    /// `R_RISCV_PCREL_HI20`: the high 20 bits of a PC-relative address.
    pcrel_hi20 = 23,
    /// `R_RISCV_PCREL_LO12_I`: the low 12 bits, for an I-type form.
    pcrel_lo12_i = 24,
};

/// A relocation applied to a `.text` offset against a symbol.
pub const Reloc = struct {
    /// Byte offset within `.text` of the instruction to patch.
    offset: u64,
    /// Index into the `Object.symbols` array.
    symbol: u32,
    type: RelocType,
    addend: i64 = 0,
};

/// The pieces of a relocatable object: code and data section blobs, the symbol
/// table, and the relocations (which apply to `.text`). `.bss` has no bytes,
/// just a size. Empty sections are omitted from the output.
/// A non-allocatable metadata section to append verbatim (e.g. `.debug_info`/`.debug_line`).
pub const DebugSection = struct { name: []const u8, bytes: []const u8 };

pub const Object = struct {
    text: []const u8,
    rodata: []const u8 = &.{},
    data: []const u8 = &.{},
    bss_size: u64 = 0,
    symbols: []const Symbol,
    relocs: []const Reloc,
    /// Extra PROGBITS metadata sections (DWARF) placed after the allocatable sections. Symbols do
    /// not reference them, so they need no section-index bookkeeping beyond the header count.
    debug: []const DebugSection = &.{},
};

// ELF constants.
const ET_REL: u16 = 1;
const EM_RISCV: u16 = 243;
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
    return std.mem.alignForward(u64, v, a);
}

/// A growable string table: a leading NUL, then NUL-terminated names. Returns
/// each appended name's byte offset.
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

fn putInt(buf: []u8, comptime T: type, value: T) void {
    std.mem.writeInt(T, buf[0..@sizeOf(T)], value, .little);
}

/// One laid-out section header: where it points and what symbols map to it.
const Shdr = struct {
    name: []const u8,
    typ: u32,
    flags: u64,
    addralign: u64,
    entsize: u64 = 0,
    link: u16 = 0,
    info: u32 = 0,
    /// File bytes, or null for `SHT_NOBITS` (.bss): occupies memory, not the file.
    bytes: ?[]const u8,
    /// `sh_size` (equals `bytes.len` unless NOBITS).
    size: u64,
};

/// Serialize `obj` into an ELF64 relocatable object. Allocatable sections
/// (`.text`/`.rodata`/`.data`/`.bss`) are emitted only when non-empty. Data
/// lives in its own section, not in `.text`. The caller owns the bytes.
pub fn write(allocator: std.mem.Allocator, obj: Object) Error![]u8 {
    // Locals must form a prefix of the symbol table. Count them and verify.
    var local_count: u32 = 0;
    var seen_global = false;
    for (obj.symbols) |s| {
        switch (s.binding) {
            .local => {
                if (seen_global) return error.Unsupported; // local after global
                local_count += 1;
            },
            .global => seen_global = true,
        }
    }
    const first_global: u32 = 1 + local_count; // null symbol at 0 is local

    // Assign section header indices in output order. 0 is the null section, then
    // the allocatable sections that exist, then the metadata sections.
    const has_rodata = obj.rodata.len > 0;
    const has_data = obj.data.len > 0;
    const has_bss = obj.bss_size > 0;
    const has_rela = obj.relocs.len > 0;
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
    next += @intCast(obj.debug.len); // .debug_* metadata sections (no symbol references them)
    const symtab_ndx = next;
    next += 1;
    const strtab_ndx = next;
    next += 1;
    const shstrtab_ndx = next;
    next += 1;
    const section_count = next;

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

    // String table of symbol names.
    var strtab = try StrTab.init(allocator);
    defer strtab.deinit(allocator);
    var name_offsets = try allocator.alloc(u32, obj.symbols.len);
    defer allocator.free(name_offsets);
    for (obj.symbols, 0..) |s, i| name_offsets[i] = try strtab.add(allocator, s.name);

    // Symbol table: a null entry, then each symbol pointing at its section.
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

    // Relocation table (applies to `.text`).
    const rela_bytes = obj.relocs.len * relaentsize;
    var rela = try allocator.alloc(u8, rela_bytes);
    defer allocator.free(rela);
    for (obj.relocs, 0..) |r, i| {
        const e = rela[i * relaentsize ..][0..relaentsize];
        const sym_index: u64 = @as(u64, r.symbol) + 1; // null entry at 0
        const r_info: u64 = (sym_index << 32) | @intFromEnum(r.type);
        putInt(e[0..8], u64, r.offset); // r_offset
        putInt(e[8..16], u64, r_info); // r_info
        putInt(e[16..24], i64, r.addend); // r_addend
    }

    // Section header strings, added in index order.
    var shstrtab = try StrTab.init(allocator);
    defer shstrtab.deinit(allocator);

    // Build the section header descriptors (excluding the null section).
    var headers: std.ArrayList(Shdr) = .empty;
    defer headers.deinit(allocator);
    try headers.append(allocator, .{ .name = ".text", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_EXECINSTR, .addralign = 4, .bytes = obj.text, .size = obj.text.len });
    if (has_rodata) try headers.append(allocator, .{ .name = ".rodata", .typ = SHT_PROGBITS, .flags = SHF_ALLOC, .addralign = 8, .bytes = obj.rodata, .size = obj.rodata.len });
    if (has_data) try headers.append(allocator, .{ .name = ".data", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_WRITE, .addralign = 8, .bytes = obj.data, .size = obj.data.len });
    if (has_bss) try headers.append(allocator, .{ .name = ".bss", .typ = SHT_NOBITS, .flags = SHF_ALLOC | SHF_WRITE, .addralign = 8, .bytes = null, .size = obj.bss_size });
    if (has_rela) try headers.append(allocator, .{ .name = ".rela.text", .typ = SHT_RELA, .flags = SHF_INFO_LINK, .addralign = 8, .entsize = relaentsize, .link = symtab_ndx, .info = text_ndx, .bytes = rela, .size = rela_bytes });
    // DWARF metadata sections (appended in the same order they were counted above).
    for (obj.debug) |d| try headers.append(allocator, .{ .name = d.name, .typ = SHT_PROGBITS, .flags = 0, .addralign = 1, .bytes = d.bytes, .size = d.bytes.len });
    try headers.append(allocator, .{ .name = ".symtab", .typ = SHT_SYMTAB, .flags = 0, .addralign = 8, .entsize = symentsize, .link = strtab_ndx, .info = first_global, .bytes = symtab, .size = sym_bytes });
    try headers.append(allocator, .{ .name = ".strtab", .typ = SHT_STRTAB, .flags = 0, .addralign = 1, .bytes = strtab.bytes.items, .size = strtab.bytes.items.len });
    try headers.append(allocator, .{ .name = ".shstrtab", .typ = SHT_STRTAB, .flags = 0, .addralign = 1, .bytes = null, .size = 0 }); // filled in below

    // Lay out file offsets for every section with file content, 8-aligned.
    var offsets = try allocator.alloc(u64, headers.items.len);
    defer allocator.free(offsets);
    var off: u64 = ehsize;
    for (headers.items, 0..) |h, i| {
        off = alignUp(off, 8);
        offsets[i] = off;
        if (h.bytes != null) off += h.size; // NOBITS occupies no file space
    }
    // The shstrtab content is the section names themselves. Intern them now that
    // all are known, then place it last.
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

    // ELF header.
    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    putInt(buf[16..18], u16, ET_REL);
    putInt(buf[18..20], u16, EM_RISCV);
    putInt(buf[20..24], u32, 1); // e_version
    putInt(buf[40..48], u64, shoff); // e_shoff
    putInt(buf[52..54], u16, @intCast(ehsize)); // e_ehsize
    putInt(buf[58..60], u16, @intCast(shentsize)); // e_shentsize
    putInt(buf[60..62], u16, section_count); // e_shnum
    putInt(buf[62..64], u16, shstrtab_ndx); // e_shstrndx

    // Section contents and headers (index 0 stays the null section).
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

/// Fill one 64-byte `Elf64_Shdr` at section index `idx`.
fn putShdr(buf: []u8, shoff: u64, idx: u16, name: u32, typ: u32, flags: u64, offset: u64, size: u64, sh_link: u32, sh_info: u32, addralign: u64, entsize: u64) void {
    const e = buf[shoff + idx * shentsize ..][0..shentsize];
    putInt(e[0..4], u32, name); // sh_name
    putInt(e[4..8], u32, typ); // sh_type
    putInt(e[8..16], u64, flags); // sh_flags
    putInt(e[16..24], u64, 0); // sh_addr
    putInt(e[24..32], u64, offset); // sh_offset
    putInt(e[32..40], u64, size); // sh_size
    putInt(e[40..44], u32, sh_link); // sh_link
    putInt(e[44..48], u32, sh_info); // sh_info
    putInt(e[48..56], u64, addralign); // sh_addralign
    putInt(e[56..64], u64, entsize); // sh_entsize
}

/// A relocation gathered during layout, before symbol indices are known. `name`
/// is the target symbol (for jal/hi20). `pair` is the paired `auipc`'s byte
/// offset (for lo12, whose target is a synthesized local label there).
const PendingKind = enum { jal, hi20, lo12 };
const PendingReloc = struct { offset: u64, kind: PendingKind, name: []const u8 = "", pair: u64 = 0 };

/// Compile every function in `module` and serialize them, plus its data globals,
/// into a single ELF relocatable object. Functions become defined `STT_FUNC`
/// globals and data blobs `STT_OBJECT` globals, both placed in `.text`. Calls
/// become `R_RISCV_JAL`. A `global_addr` becomes a `PCREL_HI20`/`PCREL_LO12_I`
/// pair (the lo12 targeting a local label at its `auipc`). Undefined targets
/// become undefined globals. The caller owns the returned ELF bytes.
pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);

    // Defined globals (functions then data), plus the relocations to resolve.
    var globals: std.ArrayList(Symbol) = .empty;
    defer globals.deinit(allocator);
    var pending: std.ArrayList(PendingReloc) = .empty;
    defer pending.deinit(allocator);

    // Functions, laid out back to back in `.text`.
    for (module.entries.items) |entry| {
        const start: u64 = text.items.len;
        var compiled = try isel.compileFunction(allocator, entry.func, .{});
        defer compiled.deinit(allocator);
        for (compiled.code) |word| {
            var w: [4]u8 = undefined;
            putInt(&w, u32, word);
            try text.appendSlice(allocator, &w);
        }
        try globals.append(allocator, .{ .name = entry.name, .value = start, .size = @as(u64, compiled.code.len) * 4, .kind = .func, .defined = true });
        for (compiled.relocs) |r| {
            const off = start + @as(u64, r.offset) * 4;
            switch (r.kind) {
                .call => try pending.append(allocator, .{ .offset = off, .kind = .jal, .name = r.symbol }),
                .pcrel_hi20 => try pending.append(allocator, .{ .offset = off, .kind = .hi20, .name = r.symbol }),
                .pcrel_lo12 => try pending.append(allocator, .{ .offset = off, .kind = .lo12, .pair = start + @as(u64, r.pair) * 4 }),
            }
        }
    }
    // Data globals go into their own sections (read-only, writable, or zero-init).
    var rodata: std.ArrayList(u8) = .empty;
    defer rodata.deinit(allocator);
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(allocator);
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
        try globals.append(allocator, .{ .name = d.name, .value = value, .size = d.size, .kind = .object, .defined = true, .section = section });
    }

    // Synthesize a local label at each lo12's paired `auipc`. Locals must come
    // first in the symbol table, so build them up front. Their names are owned
    // here and freed after `write` copies them into the string table.
    var locals: std.ArrayList(Symbol) = .empty;
    defer locals.deinit(allocator);
    var local_names: std.ArrayList([]u8) = .empty;
    defer {
        for (local_names.items) |n| allocator.free(n);
        local_names.deinit(allocator);
    }
    for (pending.items) |*p| {
        if (p.kind != .lo12) continue;
        const name = try std.fmt.allocPrint(allocator, ".Lpcrel_hi{d}", .{p.pair});
        try local_names.append(allocator, name);
        try locals.append(allocator, .{ .name = name, .value = p.pair, .size = 0, .binding = .local, .kind = .notype, .defined = true });
    }

    // Final symbol table: locals, then defined globals, then undefined externs.
    var symbols: std.ArrayList(Symbol) = .empty;
    defer symbols.deinit(allocator);
    try symbols.appendSlice(allocator, locals.items);
    try symbols.appendSlice(allocator, globals.items);
    for (pending.items) |p| {
        if (p.kind == .lo12) continue;
        if (symbolIndex(symbols.items, p.name) == null) {
            try symbols.append(allocator, .{ .name = p.name, .size = 0, .kind = .notype, .defined = false });
        }
    }

    // Resolve each pending relocation to its symbol index and ELF type.
    var relocs = try allocator.alloc(Reloc, pending.items.len);
    defer allocator.free(relocs);
    for (pending.items, 0..) |p, i| {
        relocs[i] = switch (p.kind) {
            .jal => .{ .offset = p.offset, .symbol = symbolIndex(symbols.items, p.name).?, .type = .jal },
            .hi20 => .{ .offset = p.offset, .symbol = symbolIndex(symbols.items, p.name).?, .type = .pcrel_hi20 },
            .lo12 => .{ .offset = p.offset, .symbol = localIndexAt(symbols.items, locals.items.len, p.pair), .type = .pcrel_lo12_i },
        };
    }

    return write(allocator, .{
        .text = text.items,
        .rodata = rodata.items,
        .data = data.items,
        .bss_size = bss_size,
        .symbols = symbols.items,
        .relocs = relocs,
    });
}

/// Like `writeModule`, but also emits inline DWARF: `.debug_abbrev` + `.debug_info` (a subprogram
/// DIE per function with its PC range and IR return type) + `.debug_line` (address -> source line,
/// from the functions' `debug.line` attributes), with the CU linked to the line program. So a
/// debugger reads function names, ranges, typed signatures, and source lines on real RISC-V objects.
pub fn writeModuleWithDebug(allocator: std.mem.Allocator, module: *const link.Module, source_file: []const u8) Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    var globals: std.ArrayList(Symbol) = .empty;
    defer globals.deinit(allocator);
    var pending: std.ArrayList(PendingReloc) = .empty;
    defer pending.deinit(allocator);
    var rows: std.ArrayList(dwarf.LineRow) = .empty;
    defer rows.deinit(allocator);
    // Per-function DWARF info, in .text layout order.
    const FnDie = struct { name: []const u8, low: u64, high: u64, func: *const Function };
    var fndies: std.ArrayList(FnDie) = .empty;
    defer fndies.deinit(allocator);

    for (module.entries.items) |entry| {
        const start: u64 = text.items.len;
        var compiled = try isel.compileFunction(allocator, entry.func, .{});
        defer compiled.deinit(allocator);
        for (compiled.code) |word| {
            var w: [4]u8 = undefined;
            putInt(&w, u32, word);
            try text.appendSlice(allocator, &w);
        }
        const size = @as(u64, compiled.code.len) * 4;
        try globals.append(allocator, .{ .name = entry.name, .value = start, .size = size, .kind = .func, .defined = true });
        try fndies.append(allocator, .{ .name = entry.name, .low = start, .high = start + size, .func = entry.func });
        for (compiled.relocs) |r| {
            const off = start + @as(u64, r.offset) * 4;
            switch (r.kind) {
                .call => try pending.append(allocator, .{ .offset = off, .kind = .jal, .name = r.symbol }),
                .pcrel_hi20 => try pending.append(allocator, .{ .offset = off, .kind = .hi20, .name = r.symbol }),
                .pcrel_lo12 => try pending.append(allocator, .{ .offset = off, .kind = .lo12, .pair = start + @as(u64, r.pair) * 4 }),
            }
        }
        // Line rows are function-relative; shift to the module-relative .text offset.
        for (compiled.lines) |e| try rows.append(allocator, .{ .address = start + e.offset, .line = e.line });
    }

    // Data globals (rodata/data/bss).
    var rodata: std.ArrayList(u8) = .empty;
    defer rodata.deinit(allocator);
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(allocator);
    var bss_size: u64 = 0;
    for (module.data.items) |d| {
        const section: SectionKind, const value: u64 = switch (d.kind) {
            .rodata => blk: {
                const s = rodata.items.len;
                try rodata.appendSlice(allocator, d.bytes);
                break :blk .{ .rodata, s };
            },
            .data => blk: {
                const s = data.items.len;
                try data.appendSlice(allocator, d.bytes);
                break :blk .{ .data, s };
            },
            .bss => blk: {
                const s = bss_size;
                bss_size += d.size;
                break :blk .{ .bss, s };
            },
        };
        try globals.append(allocator, .{ .name = d.name, .value = value, .size = d.size, .kind = .object, .defined = true, .section = section });
    }

    // Local labels for each lo12's paired auipc (locals must precede globals in .symtab).
    var locals: std.ArrayList(Symbol) = .empty;
    defer locals.deinit(allocator);
    var local_names: std.ArrayList([]u8) = .empty;
    defer {
        for (local_names.items) |n| allocator.free(n);
        local_names.deinit(allocator);
    }
    for (pending.items) |*p| {
        if (p.kind != .lo12) continue;
        const name = try std.fmt.allocPrint(allocator, ".Lpcrel_hi{d}", .{p.pair});
        try local_names.append(allocator, name);
        try locals.append(allocator, .{ .name = name, .value = p.pair, .size = 0, .binding = .local, .kind = .notype, .defined = true });
    }

    var symbols: std.ArrayList(Symbol) = .empty;
    defer symbols.deinit(allocator);
    try symbols.appendSlice(allocator, locals.items);
    try symbols.appendSlice(allocator, globals.items);
    for (pending.items) |p| {
        if (p.kind == .lo12) continue;
        if (symbolIndex(symbols.items, p.name) == null) {
            try symbols.append(allocator, .{ .name = p.name, .size = 0, .kind = .notype, .defined = false });
        }
    }

    var relocs = try allocator.alloc(Reloc, pending.items.len);
    defer allocator.free(relocs);
    for (pending.items, 0..) |p, i| {
        relocs[i] = switch (p.kind) {
            .jal => .{ .offset = p.offset, .symbol = symbolIndex(symbols.items, p.name).?, .type = .jal },
            .hi20 => .{ .offset = p.offset, .symbol = symbolIndex(symbols.items, p.name).?, .type = .pcrel_hi20 },
            .lo12 => .{ .offset = p.offset, .symbol = localIndexAt(symbols.items, locals.items.len, p.pair), .type = .pcrel_lo12_i },
        };
    }

    // DWARF sections: a subprogram DIE per function (with its typed return), plus the line program.
    const subs = try allocator.alloc(dwarf.Subprogram, fndies.items.len);
    defer allocator.free(subs);
    for (fndies.items, 0..) |fd, i| subs[i] = .{ .name = fd.name, .low_pc = fd.low, .high_pc = fd.high, .ret_type = returnBaseType(fd.func) };

    const abbrev = try dwarf.emitAbbrev(allocator);
    defer allocator.free(abbrev);
    // One line program at offset 0 of .debug_line, so link the CU to it via DW_AT_stmt_list.
    const info = try dwarf.emitInfo(allocator, .{ .name = source_file, .low_pc = 0, .high_pc = text.items.len, .subprograms = subs, .stmt_list = 0 });
    defer allocator.free(info);
    const line = try dwarf.emitLine(allocator, source_file, rows.items, text.items.len);
    defer allocator.free(line);

    return write(allocator, .{
        .text = text.items,
        .rodata = rodata.items,
        .data = data.items,
        .bss_size = bss_size,
        .symbols = symbols.items,
        .relocs = relocs,
        .debug = &.{
            .{ .name = ".debug_abbrev", .bytes = abbrev },
            .{ .name = ".debug_info", .bytes = info },
            .{ .name = ".debug_line", .bytes = line },
        },
    });
}

/// Map a function's IR return type to a DWARF base type (C-like names), or null for a void /
/// non-primitive return. Distinct primitives get distinct names so the base-type dedup keeps them apart.
fn returnBaseType(func: *const Function) ?dwarf.BaseType {
    const ret_val = for (0..func.blocks.items.len) |bi| {
        const term = func.terminator(@enumFromInt(bi)) orelse continue;
        switch (term) {
            .ret => |maybe| if (maybe) |v| break v else return null,
            else => {},
        }
    } else return null;

    return switch (func.types.type_kind(func.valueType(ret_val))) {
        .bool => .{ .name = "bool", .encoding = .boolean, .byte_size = 1 },
        .float => |f| switch (f) {
            .f32 => .{ .name = "float", .encoding = .float, .byte_size = 4 },
            .f64 => .{ .name = "double", .encoding = .float, .byte_size = 8 },
            // Debug-info naming only, not lowering: riscv64 has no f16 codegen yet.
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
        else => null, // ptr / vector / aggregate
    };
}

/// Find the local label symbol covering `.text` byte offset `value` among the
/// first `local_count` (local) symbols.
fn localIndexAt(symbols: []const Symbol, local_count: usize, value: u64) u32 {
    for (symbols[0..local_count], 0..) |s, i| {
        if (s.value == value) return @intCast(i);
    }
    unreachable; // every lo12 has a matching local label
}

fn symbolIndex(symbols: []const Symbol, name: []const u8) ?u32 {
    for (symbols, 0..) |s, i| {
        if (std.mem.eql(u8, s.name, name)) return @intCast(i);
    }
    return null;
}

test "writes an ELF64 RISC-V relocatable header" {
    const allocator = std.testing.allocator;

    // One function "add" of two words, with a call relocation to "helper".
    const text = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const symbols = [_]Symbol{
        .{ .name = "add", .value = 0, .size = text.len, .kind = .func },
        .{ .name = "helper", .defined = false },
    };
    const relocs = [_]Reloc{
        .{ .offset = 0, .symbol = 1, .type = .jal },
    };

    const obj = Object{ .text = &text, .symbols = &symbols, .relocs = &relocs };
    const bytes = try write(allocator, obj);
    defer allocator.free(bytes);

    // ELF magic and a 64-bit little-endian RISC-V relocatable object.
    try std.testing.expectEqualSlices(u8, "\x7fELF", bytes[0..4]);
    try std.testing.expectEqual(@as(u8, 2), bytes[4]); // ELFCLASS64
    try std.testing.expectEqual(@as(u8, 1), bytes[5]); // ELFDATA2LSB
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, bytes[16..18], .little)); // ET_REL
    try std.testing.expectEqual(@as(u16, 243), std.mem.readInt(u16, bytes[18..20], .little)); // EM_RISCV
}

const Function = ir.function.Function;

/// Run `readelf flag` on `bytes` and return its stdout. Skips the test when
/// `readelf` is not on PATH.
fn readelf(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8, flag: []const u8) ![]u8 {
    const Nonce = struct {
        var counter: usize = 0;
    };
    Nonce.counter += 1;
    const name = try std.fmt.allocPrint(allocator, "vulcan-obj-{d}.o", .{Nonce.counter});
    defer allocator.free(name);
    // A unique temp dir with the child cwd set there, so relative names resolve and the
    // process cwd is never written to (which was flaky).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data = bytes });

    const argv = [_][]const u8{ "readelf", flag, name };
    const result = std.process.run(allocator, io, .{ .argv = &argv, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(result.stderr);
    return result.stdout;
}

test "readelf accepts the object and sees its symbols and relocations" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee: fn(x) -> x.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const x = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = x });
    }
    // caller: fn(x) -> external(callee(x)). One intra-module call, one external.
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        const r2 = try caller.appendCall(b, t, "external", &.{r});
        caller.setTerminator(b, .{ .ret = r2 });
    }

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "callee", &callee);
    try module.addFunction(allocator, "caller", &caller);

    const bytes = try writeModule(allocator, &module);
    defer allocator.free(bytes);

    // The symbol table: callee/caller are defined functions, external is UND.
    const syms = try readelf(allocator, io, bytes, "-s");
    defer allocator.free(syms);
    try std.testing.expect(std.mem.indexOf(u8, syms, "callee") != null);
    try std.testing.expect(std.mem.indexOf(u8, syms, "caller") != null);
    try std.testing.expect(std.mem.indexOf(u8, syms, "external") != null);
    try std.testing.expect(std.mem.indexOf(u8, syms, "FUNC") != null);

    // The relocations: every call is an R_RISCV_JAL.
    const rels = try readelf(allocator, io, bytes, "-r");
    defer allocator.free(rels);
    try std.testing.expect(std.mem.indexOf(u8, rels, "R_RISCV_JAL") != null);
}

test "readelf shows separate .rodata, .data, and .bss sections" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const b = try entry.appendBlock();
        const x = try entry.appendBlockParam(b, t);
        entry.setTerminator(b, .{ .ret = x });
    }
    const ro = [_]u8{ 1, 0, 0, 0 };
    const da = [_]u8{ 2, 0, 0, 0 };
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addData(allocator, "RO", &ro);
    try module.addWritable(allocator, "DA", &da);
    try module.addBss(allocator, "BS", 8);

    const bytes = try writeModule(allocator, &module);
    defer allocator.free(bytes);

    // The section headers carry the three data sections, the BSS one as NOBITS.
    const secs = try readelf(allocator, io, bytes, "-S");
    defer allocator.free(secs);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".rodata") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".data") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".bss") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, "NOBITS") != null);

    // The data symbols are OBJECT-typed.
    const syms = try readelf(allocator, io, bytes, "-s");
    defer allocator.free(syms);
    try std.testing.expect(std.mem.indexOf(u8, syms, "OBJECT") != null);
}

/// Link `obj_bytes` (a relocatable object) with the real RISC-V `ld.lld` and
/// return the raw `.text` image (via `--oformat binary`). Entry `entry` is
/// placed at `0x80000000`, the input objects laid out in the order given. Skips
/// the test when `ld.lld` is not on PATH. lld is only a comparison oracle. The
/// shipping linker is ld.zig.
fn lldLink(allocator: std.mem.Allocator, io: std.Io, objs: []const []const u8, entry: []const u8) ![]u8 {
    const Nonce = struct {
        var counter: usize = 0;
    };
    // Work in a unique temp directory with the child's cwd set there, so the argv just
    // names the files. Writing to the process cwd made this flaky (the files could be
    // missing or unwritable depending on where the test binary ran).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = tmp.dir;

    // Write each object to its own file, collecting the names for the argv.
    var obj_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (obj_names.items) |n| {
            dir.deleteFile(io, n) catch {};
            allocator.free(n);
        }
        obj_names.deinit(allocator);
    }
    for (objs) |obj_bytes| {
        Nonce.counter += 1;
        const name = try std.fmt.allocPrint(allocator, "vulcan-link-{d}.o", .{Nonce.counter});
        try obj_names.append(allocator, name);
        try dir.writeFile(io, .{ .sub_path = name, .data = obj_bytes });
    }

    Nonce.counter += 1;
    const bin_name = try std.fmt.allocPrint(allocator, "vulcan-link-{d}.bin", .{Nonce.counter});
    defer allocator.free(bin_name);
    defer dir.deleteFile(io, bin_name) catch {};

    const entry_arg = try std.fmt.allocPrint(allocator, "-e{s}", .{entry});
    defer allocator.free(entry_arg);

    // ld.lld -m elf64lriscv -e<entry> -Ttext=0x80000000 --no-dynamic-linker
    //   --oformat binary <objs...> -o <bin>
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    try argv.appendSlice(allocator, &.{ "ld.lld", "-m", "elf64lriscv", entry_arg, "-Ttext=0x80000000", "--no-dynamic-linker", "--oformat", "binary" });
    try argv.appendSlice(allocator, obj_names.items);
    try argv.appendSlice(allocator, &.{ "-o", bin_name });

    const result = std.process.run(allocator, io, .{ .argv = argv.items, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (result.term != .exited or result.term.exited != 0) {
        std.debug.print("ld.lld failed:\n{s}\n", .{result.stderr});
        return error.LinkFailed;
    }

    // 1 MiB cap is plenty for these tiny test images.
    return dir.readFileAlloc(io, bin_name, allocator, .limited(1 << 20));
}

test "real ld.lld links the object to the same bytes as the in-memory linker" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // A self-contained module (no external symbols) so the link fully resolves.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const x = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = x });
    }
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        caller.setTerminator(b, .{ .ret = r });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "callee", &callee);
    try module.addFunction(allocator, "caller", &caller);

    // The in-memory linker's resolved code (already River-validated elsewhere).
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);

    // The object, linked by the real RISC-V lld, as a raw .text image.
    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);
    const image = try lldLink(allocator, io, &.{obj}, "caller");
    defer allocator.free(image);

    // lld must produce exactly the bytes the in-memory linker does, word for word.
    try std.testing.expectEqual(linked.code.len * 4, image.len);
    for (linked.code, 0..) |word, i| {
        const got = std.mem.readInt(u32, image[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(word, got);
    }
}

test "lld resolves a call across two separately compiled objects" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee compiled entirely on its own (one object).
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
    const callee_obj = try writeModule(allocator, &callee_mod);
    defer allocator.free(callee_obj);

    // caller compiled on its own: "callee" stays an undefined external symbol.
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
    const caller_obj = try writeModule(allocator, &caller_mod);
    defer allocator.free(caller_obj);

    // The combined in-memory link is the reference layout: callee then caller.
    var combined: link.Module = .{};
    defer combined.deinit(allocator);
    try combined.addFunction(allocator, "callee", &callee);
    try combined.addFunction(allocator, "caller", &caller);
    var linked = try link.compileModule(allocator, &combined);
    defer linked.deinit(allocator);

    // lld links the two objects (callee first) and resolves the cross-object
    // call to the same bytes the in-memory linker produces.
    const image = try lldLink(allocator, io, &.{ callee_obj, caller_obj }, "caller");
    defer allocator.free(image);

    try std.testing.expectEqual(linked.code.len * 4, image.len);
    for (linked.code, 0..) |word, i| {
        const got = std.mem.readInt(u32, image[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(word, got);
    }
}

test "our PCREL global-data resolution matches lld byte for byte" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&K), K an i32 constant. Same module as the River test in ld.zig.
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
    const k_bytes = [_]u8{ 42, 0, 0, 0 };
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addData(allocator, "K", &k_bytes);

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);

    // The in-memory linker and lld must resolve the PCREL_HI20/LO12 pair to
    // exactly the same bytes (entry at base, K right after it).
    var ours = try ld.linkObjects(allocator, &.{obj}, 0x80000000);
    defer ours.deinit(allocator);
    const theirs = try lldLink(allocator, io, &.{obj}, "entry");
    defer allocator.free(theirs);

    try std.testing.expectEqualSlices(u8, ours.code, theirs);
}

test "linker resolves an R_RISCV_CALL far call (matches lld, runs on River)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // A hand-built object using the standard far-call sequence `auipc ra, 0`
    // then `jalr ra, ra, 0` with one R_RISCV_CALL relocation, exactly what gcc/clang
    // emit by default. entry() far-calls callee() which returns 7.
    const words = [_]u32{
        encode.addi(.x2, .x2, -16), // entry: open frame
        encode.sd(.x1, .x2, 0), //          save ra
        encode.auipc(.x1, 0), //            R_RISCV_CALL callee (byte offset 8)
        encode.jalr(.x1, .x1, 0), //        call
        encode.ld(.x1, .x2, 0), //          restore ra
        encode.addi(.x2, .x2, 16), //       close frame
        encode.jalr(.x0, .x1, 0), //        ret
        encode.addi(.x10, .x0, 7), // callee: a0 = 7
        encode.jalr(.x0, .x1, 0), //        ret
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| putInt(text[i * 4 ..][0..4], u32, w);

    const symbols = [_]Symbol{
        .{ .name = "entry", .value = 0, .size = 7 * 4, .kind = .func },
        .{ .name = "callee", .value = 7 * 4, .size = 2 * 4, .kind = .func },
    };
    const relocs = [_]Reloc{
        .{ .offset = 8, .symbol = 1, .type = .call }, // the auipc, against callee
    };
    const obj = try write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
    defer allocator.free(obj);

    // The linker and lld must agree, and the result must run on River.
    var ours = try ld.linkObjects(allocator, &.{obj}, harness.load_address);
    defer ours.deinit(allocator);
    const theirs = try lldLink(allocator, io, &.{obj}, "entry");
    defer allocator.free(theirs);
    try std.testing.expectEqualSlices(u8, ours.code, theirs);

    const img_words = try allocator.alloc(u32, ours.code.len / 4);
    defer allocator.free(img_words);
    for (img_words, 0..) |*w, i| w.* = std.mem.readInt(u32, ours.code[i * 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(i64, 7), try harness.runCode(io, allocator, img_words, &.{}, harness.river));
}

test "writeModuleWithDebug emits DWARF readelf reads (subprograms + CU line linkage)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // helper(x) -> x*3 ; main(x) -> helper(x). Two functions, a real intra-module call.
    var helper = Function.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(i32k);
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        const m = try helper.appendArithImm(b, t, .add, x, 5);
        helper.setTerminator(b, .{ .ret = m });
    }
    var main_f = Function.init(allocator);
    defer main_f.deinit();
    {
        const t = try main_f.types.intern(i32k);
        const b = try main_f.appendBlock();
        const x = try main_f.appendBlockParam(b, t);
        const r = try main_f.appendCall(b, t, "helper", &.{x});
        main_f.setTerminator(b, .{ .ret = r });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "helper", &helper);
    try module.addFunction(allocator, "main", &main_f);

    const obj = try writeModuleWithDebug(allocator, &module, "mod.glsl");
    defer allocator.free(obj);

    // The three DWARF sections are present in the object.
    const secs = readelf(allocator, io, obj, "-S") catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(secs);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".debug_abbrev") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".debug_info") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".debug_line") != null);

    // The CU decodes: both functions as subprograms, the CU linked to its line program, typed return.
    const info = readelf(allocator, io, obj, "--debug-dump=info") catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(info);
    try std.testing.expect(std.mem.indexOf(u8, info, "DW_TAG_subprogram") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "helper") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "main") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "DW_AT_stmt_list") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "int") != null); // i32 return -> "int"
}
