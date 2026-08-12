//! ELF64 relocatable object (ET_REL, EM_X86_64) emission. Each function becomes an
//! STT_FUNC global in .text. Every `call` becomes an R_X86_64_PLT32 relocation, with addend
//! -4 for the implicit displacement adjustment, against the callee symbol (undefined if
//! external). readelf and a system x86-64 linker both accept the output. link.zig is the
//! in-memory linker for the same data.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");
const dwarf = @import("../dwarf.zig");

/// A non-allocatable metadata section (DWARF) to append verbatim.
pub const DebugSection = struct { name: []const u8, bytes: []const u8 };

pub const Error = isel.Error;

const ET_REL: u16 = 1;
const EM_X86_64: u16 = 62;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_INFO_LINK: u64 = 0x40;
const R_X86_64_PC32: u32 = 2;
const R_X86_64_PLT32: u32 = 4;
/// `R_X86_64_GOTPCREL`: a PC-relative reference to a symbol's GOT slot, from a GOT-indirect
/// `global_addr`'s `mov rd, [rip+disp32]` (`via_got`). It uses the same disp32 field and
/// addend as PC32/PLT32, but the target is the symbol's GOT slot. The dynamic linker
/// synthesizes the slot plus a GLOB_DAT relocation.
const R_X86_64_GOTPCREL: u32 = 9;
/// `R_X86_64_64`: a 64-bit absolute address (`S + A`) written into a data section slot. This
/// is the relocation a pointer-initialized global (`int *p = &g;`) carries in
/// `.rela.data`/`.rela.rodata`. The 8-byte slot holding the pointer must be patched to the
/// target symbol's runtime address. The linker turns it into an `R_X86_64_RELATIVE` dynamic
/// relocation for a PIE or a `.so`, or a direct absolute write for a non-PIE executable. This
/// mirrors `aarch64/object.zig`'s `R_AARCH64_ABS64`.
pub const R_X86_64_64: u32 = 1;
const STB_GLOBAL: u8 = 1;
const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const SHN_UNDEF: u16 = 0;

/// Which allocatable data section a global lands in, matching `link.DataKind`.
pub const SectionKind = enum { rodata, data, bss };

fn put(buf: []u8, comptime T: type, off: usize, v: T) void {
    std.mem.writeInt(T, buf[off..][0..@sizeOf(T)], v, .little);
}

/// A string table that interns names and returns their byte offsets.
const StrTab = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn init(allocator: std.mem.Allocator) !StrTab {
        var s: StrTab = .{};
        try s.bytes.append(allocator, 0); // index 0 is the empty string
        return s;
    }
    fn add(self: *StrTab, allocator: std.mem.Allocator, name: []const u8) !u32 {
        const off: u32 = @intCast(self.bytes.items.len);
        try self.bytes.appendSlice(allocator, name);
        try self.bytes.append(allocator, 0);
        return off;
    }
};

/// A data global's placement within its own section blob, gathered before symbol
/// indices (and hence `st_shndx`) are known.
const DataPlace = struct { name: []const u8, section: SectionKind, value: u64, size: u64 };

/// A data-section pointer-init relocation before its target name is resolved to a symbol
/// index: the section and byte offset of the slot, and the target symbol's name. This
/// mirrors `aarch64/object.zig`'s `PendingDataReloc`.
const PendingDataReloc = struct { section: SectionKind, offset: u64, symbol: []const u8 };

/// A data-section pointer-init relocation applied to a `.data`/`.rodata` slot against a
/// symbol. At byte `offset` within `section`, an `R_X86_64_64` writes `symbol`'s runtime
/// address plus `addend`. This is emitted into `.rela.data`/`.rela.rodata`, keyed by
/// `section`. It carries a data global's own pointer inits in the object, instead of
/// dropping them.
pub const DataRelocEntry = struct { section: SectionKind, offset: u64, symbol: u32, addend: i64 = 0 };

/// Lay out `module.data` into three section blobs: rodata and data are appended verbatim,
/// bss is tracked only as a size. This records each global's placement. Each data global's
/// own `DataReloc`s (pointer inits) are recorded in `data_relocs` at their absolute section
/// offset, the global's placement plus the reloc's in-object offset.
fn layoutData(allocator: std.mem.Allocator, module: *const link.Module, rodata: *std.ArrayList(u8), data_sec: *std.ArrayList(u8), data_relocs: *std.ArrayList(PendingDataReloc)) Error!struct { places: []DataPlace, bss_size: u64 } {
    var places: std.ArrayList(DataPlace) = .empty;
    errdefer places.deinit(allocator);
    var bss_size: u64 = 0;
    for (module.data.items) |d| {
        const section: SectionKind, const value: u64 = switch (d.kind) {
            .rodata => blk: {
                const v = rodata.items.len;
                try rodata.appendSlice(allocator, d.bytes);
                break :blk .{ .rodata, v };
            },
            .data => blk: {
                const v = data_sec.items.len;
                try data_sec.appendSlice(allocator, d.bytes);
                break :blk .{ .data, v };
            },
            .bss => blk: {
                const v = bss_size;
                bss_size += d.size;
                break :blk .{ .bss, v };
            },
        };
        for (d.relocs) |r| try data_relocs.append(allocator, .{ .section = section, .offset = value + r.off, .symbol = r.symbol });
        try places.append(allocator, .{ .name = d.name, .section = section, .value = value, .size = d.size });
    }
    return .{ .places = try places.toOwnedSlice(allocator), .bss_size = bss_size };
}

/// Compile and emit `module` as an ELF64 relocatable object. Caller owns the bytes.
pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error![]u8 {
    const funcs = module.funcs.items;
    const compiled = try allocator.alloc(isel.Compiled, funcs.len);
    var compiled_n: usize = 0;
    defer {
        for (compiled[0..compiled_n]) |*c| c.deinit(allocator);
        allocator.free(compiled);
    }
    const caps: isel.ModelCaps = if (module.model) |m| isel.capsForModel(m) else .{};
    for (funcs, 0..) |e, i| {
        compiled[i] = try isel.compileWithCaps(allocator, e.func, caps);
        compiled_n = i + 1;
    }

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    const offsets = try allocator.alloc(usize, funcs.len);
    defer allocator.free(offsets);
    for (compiled, 0..) |c, i| {
        offsets[i] = text.items.len;
        try text.appendSlice(allocator, c.code);
        while (text.items.len % 16 != 0) try text.append(allocator, 0x90);
    }

    // Data globals (`global_addr` targets) are laid out into their own rodata, data, and
    // bss blobs, placed in the module's own sections, not `.text`.
    var rodata: std.ArrayList(u8) = .empty;
    defer rodata.deinit(allocator);
    var data_sec: std.ArrayList(u8) = .empty;
    defer data_sec.deinit(allocator);
    var pending_data_relocs: std.ArrayList(PendingDataReloc) = .empty;
    defer pending_data_relocs.deinit(allocator);
    const laid_out = try layoutData(allocator, module, &rodata, &data_sec, &pending_data_relocs);
    defer allocator.free(laid_out.places);

    // Section indices matching the layout `assemble` builds: NULL(0), .text(1), then
    // whichever of .rodata, .data, and .bss are non-empty, in that order.
    const text_ndx: u16 = 1;
    const has_rodata = rodata.items.len > 0;
    const has_data = data_sec.items.len > 0;
    const has_bss = laid_out.bss_size > 0;
    var next_ndx: u16 = 2;
    const rodata_ndx: u16 = if (has_rodata) blk: {
        defer next_ndx += 1;
        break :blk next_ndx;
    } else 0;
    const data_ndx: u16 = if (has_data) blk: {
        defer next_ndx += 1;
        break :blk next_ndx;
    } else 0;
    const bss_ndx: u16 = if (has_bss) blk: {
        defer next_ndx += 1;
        break :blk next_ndx;
    } else 0;

    // Symbols: the null symbol, then each function and data global as a defined global,
    // then any external callee or global that a relocation references but that this
    // module does not define, as an undefined global.
    var strtab = try StrTab.init(allocator);
    defer strtab.bytes.deinit(allocator);
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    var sym_index = std.StringHashMapUnmanaged(u32){};
    defer sym_index.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 24); // null symbol (index 0)

    for (funcs, 0..) |e, i| {
        try sym_index.put(allocator, e.name, @intCast(symtab.items.len / 24));
        try appendSym(allocator, &symtab, try strtab.add(allocator, e.name), STT_FUNC, STB_GLOBAL, text_ndx, offsets[i], compiled[i].code.len);
    }
    for (laid_out.places) |dp| {
        const shndx: u16 = switch (dp.section) {
            .rodata => rodata_ndx,
            .data => data_ndx,
            .bss => bss_ndx,
        };
        try sym_index.put(allocator, dp.name, @intCast(symtab.items.len / 24));
        try appendSym(allocator, &symtab, try strtab.add(allocator, dp.name), STT_OBJECT, STB_GLOBAL, shndx, dp.value, dp.size);
    }
    for (compiled) |c| for (c.relocs) |r| {
        if (!sym_index.contains(r.symbol)) {
            try sym_index.put(allocator, r.symbol, @intCast(symtab.items.len / 24));
            try appendSym(allocator, &symtab, try strtab.add(allocator, r.symbol), STT_NOTYPE, 0, SHN_UNDEF, 0, 0); // undefined (binding=0, as before)
        }
    };

    // Relocations against .text: `.call` becomes PLT32, `.pcrel_lea` (`global_addr`)
    // becomes PC32. Both share the same addend, -4, the disp32 field's own width. The
    // PC-relative math is identical for both: S + A - P, for a field whose P is its own
    // start and whose runtime read point is P+4.
    var rela: std.ArrayList(u8) = .empty;
    defer rela.deinit(allocator);
    for (compiled, 0..) |c, i| for (c.relocs) |r| {
        var ent: [24]u8 = undefined;
        put(&ent, u64, 0, offsets[i] + r.offset); // r_offset
        const rtype: u32 = switch (r.kind) {
            .call => R_X86_64_PLT32,
            .pcrel_lea => R_X86_64_PC32,
            .got_pcrel => R_X86_64_GOTPCREL,
        };
        put(&ent, u64, 8, (@as(u64, sym_index.get(r.symbol).?) << 32) | rtype); // r_info
        put(&ent, i64, 16, -4); // r_addend
        try rela.appendSlice(allocator, &ent);
    };

    // Resolve each data-section reloc's target name to a symbol index. The target of a
    // pointer init is an internally defined data/function global (added above).
    var data_relocs = try allocator.alloc(DataRelocEntry, pending_data_relocs.items.len);
    defer allocator.free(data_relocs);
    for (pending_data_relocs.items, 0..) |p, i| {
        data_relocs[i] = .{ .section = p.section, .offset = p.offset, .symbol = sym_index.get(p.symbol) orelse return error.Unsupported };
    }

    // `.rela.data` / `.rela.rodata` hold the data-section pointer-init relocations, each an
    // `R_X86_64_64` against the target symbol, grouped by the section each one modifies.
    var rela_data: std.ArrayList(u8) = .empty;
    defer rela_data.deinit(allocator);
    var rela_rodata: std.ArrayList(u8) = .empty;
    defer rela_rodata.deinit(allocator);
    for (data_relocs) |dr| {
        var ent: [24]u8 = undefined;
        put(&ent, u64, 0, dr.offset); // r_offset
        put(&ent, u64, 8, (@as(u64, dr.symbol) << 32) | R_X86_64_64); // r_info
        put(&ent, i64, 16, dr.addend); // r_addend
        const dst: *std.ArrayList(u8) = switch (dr.section) {
            .data => &rela_data,
            .rodata => &rela_rodata,
            .bss => unreachable, // a data reloc never targets .bss
        };
        try dst.appendSlice(allocator, &ent);
    }

    return assemble(allocator, text.items, rodata.items, data_sec.items, laid_out.bss_size, rela.items, rela_data.items, rela_rodata.items, symtab.items, strtab.bytes.items, funcs.len + laid_out.places.len + 1, &.{});
}

/// Like `writeModule`, but also emits inline DWARF: `.debug_abbrev`, `.debug_info` (a
/// subprogram DIE per function, with PC range and typed return), and `.debug_line` (built
/// from the functions' `debug.line` attributes), with the compile unit linked to the line
/// program. A debugger can then read names, ranges, typed signatures, and source lines on
/// real x86-64 objects.
pub fn writeModuleWithDebug(allocator: std.mem.Allocator, module: *const link.Module, source_file: []const u8) Error![]u8 {
    const funcs = module.funcs.items;
    const compiled = try allocator.alloc(isel.Compiled, funcs.len);
    var compiled_n: usize = 0;
    defer {
        for (compiled[0..compiled_n]) |*c| c.deinit(allocator);
        allocator.free(compiled);
    }
    const caps: isel.ModelCaps = if (module.model) |m| isel.capsForModel(m) else .{};
    for (funcs, 0..) |e, i| {
        compiled[i] = try isel.compileWithCaps(allocator, e.func, caps);
        compiled_n = i + 1;
    }

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    const offsets = try allocator.alloc(usize, funcs.len);
    defer allocator.free(offsets);
    var rows: std.ArrayList(dwarf.LineRow) = .empty;
    defer rows.deinit(allocator);
    for (compiled, 0..) |c, i| {
        offsets[i] = text.items.len;
        for (c.lines) |ln| try rows.append(allocator, .{ .address = offsets[i] + ln.offset, .line = ln.line });
        try text.appendSlice(allocator, c.code);
        while (text.items.len % 16 != 0) try text.append(allocator, 0x90);
    }

    var strtab = try StrTab.init(allocator);
    defer strtab.bytes.deinit(allocator);
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    var sym_index = std.StringHashMapUnmanaged(u32){};
    defer sym_index.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 24); // null symbol
    for (funcs, 0..) |e, i| {
        try sym_index.put(allocator, e.name, @intCast(symtab.items.len / 24));
        try appendSym(allocator, &symtab, try strtab.add(allocator, e.name), STT_FUNC, STB_GLOBAL, 1, offsets[i], compiled[i].code.len);
    }
    for (compiled) |c| for (c.relocs) |r| {
        if (!sym_index.contains(r.symbol)) {
            try sym_index.put(allocator, r.symbol, @intCast(symtab.items.len / 24));
            try appendSym(allocator, &symtab, try strtab.add(allocator, r.symbol), STT_NOTYPE, 0, SHN_UNDEF, 0, 0); // undefined (binding=0, as before)
        }
    };

    var rela: std.ArrayList(u8) = .empty;
    defer rela.deinit(allocator);
    for (compiled, 0..) |c, i| for (c.relocs) |r| {
        var ent: [24]u8 = undefined;
        put(&ent, u64, 0, offsets[i] + r.offset);
        const rtype: u32 = switch (r.kind) {
            .call => R_X86_64_PLT32,
            .pcrel_lea => R_X86_64_PC32,
            .got_pcrel => R_X86_64_GOTPCREL,
        };
        put(&ent, u64, 8, (@as(u64, sym_index.get(r.symbol).?) << 32) | rtype);
        put(&ent, i64, 16, -4);
        try rela.appendSlice(allocator, &ent);
    };

    // DWARF: a subprogram DIE per function, with a typed return, plus the line program.
    const subs = try allocator.alloc(dwarf.Subprogram, funcs.len);
    defer allocator.free(subs);
    for (funcs, 0..) |e, i| subs[i] = .{
        .name = e.name,
        .low_pc = offsets[i],
        .high_pc = offsets[i] + compiled[i].code.len,
        .ret_type = returnBaseType(e.func),
    };
    const abbrev = try dwarf.emitAbbrev(allocator);
    defer allocator.free(abbrev);
    const info = try dwarf.emitInfo(allocator, .{ .name = source_file, .low_pc = 0, .high_pc = text.items.len, .subprograms = subs, .stmt_list = 0 });
    defer allocator.free(info);
    const line = try dwarf.emitLine(allocator, source_file, rows.items, text.items.len);
    defer allocator.free(line);

    return assemble(allocator, text.items, &.{}, &.{}, 0, rela.items, &.{}, &.{}, symtab.items, strtab.bytes.items, funcs.len + 1, &.{
        .{ .name = ".debug_abbrev", .bytes = abbrev },
        .{ .name = ".debug_info", .bytes = info },
        .{ .name = ".debug_line", .bytes = line },
    });
}

/// Map a function's IR return type to a DWARF base type, using C-like names. Returns null
/// for a void return or a non-primitive return.
fn returnBaseType(func: *const ir.function.Function) ?dwarf.BaseType {
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
            // A 2-byte IEEE half in memory. x86_64 lowers f16 via F16C, and holds it as its
            // widened f32 form in registers.
            .f16 => .{ .name = "half", .encoding = .float, .byte_size = 2 },
        },
        .int => |it| blk: {
            const bytes: u8 = @intCast((it.bits + 7) / 8);
            const signed = it.signedness == .signed;
            const name: []const u8 = switch (it.bits) {
                8 => if (signed) "i8" else "u8",
                16 => if (signed) "i16" else "u16",
                32 => if (signed) "int" else "unsigned int",
                64 => if (signed) "long" else "unsigned long",
                else => if (signed) "int" else "unsigned",
            };
            break :blk .{ .name = name, .encoding = if (signed) .signed else .unsigned, .byte_size = bytes };
        },
        else => null,
    };
}

fn appendSym(allocator: std.mem.Allocator, symtab: *std.ArrayList(u8), name_off: u32, typ: u8, binding: u8, shndx: u16, value: usize, size: usize) !void {
    var ent: [24]u8 = undefined;
    @memset(&ent, 0);
    put(&ent, u32, 0, name_off); // st_name
    ent[4] = (binding << 4) | typ; // st_info
    put(&ent, u16, 6, shndx); // st_shndx
    put(&ent, u64, 8, value); // st_value
    put(&ent, u64, 16, size); // st_size
    try symtab.appendSlice(allocator, ent[0..24]);
}

/// A section header descriptor for the flexible ELF writer below.
const Sec = struct {
    name: []const u8,
    typ: u32,
    flags: u64 = 0,
    data: []const u8,
    link: u32 = 0,
    info: u32 = 0,
    addralign: u64 = 1,
    entsize: u64 = 0,
    /// An `sh_size` override for `SHT_NOBITS` (`.bss`). This section occupies no file
    /// bytes, so `data` stays empty, but it still reports its zero-initialized size. Null
    /// means `sh_size` is just `data.len`, which holds for every other section.
    size: ?usize = null,
};

/// Lay out the ELF file: header, section data, and the section header table. The order is
/// NULL, .text, [.rodata], [.data], [.bss], [.rela.text], [.rela.data], [.rela.rodata],
/// [debug...], .symtab, .strtab, .shstrtab. `first_global` is the index of the first global
/// symbol. `debug` holds extra, non-allocatable metadata sections placed before .symtab.
fn assemble(allocator: std.mem.Allocator, text: []const u8, rodata: []const u8, data: []const u8, bss_size: u64, rela: []const u8, rela_data: []const u8, rela_rodata: []const u8, symtab: []const u8, strtab: []const u8, first_global: usize, debug: []const DebugSection) Error![]u8 {
    const has_rodata = rodata.len > 0;
    const has_data = data.len > 0;
    const has_bss = bss_size > 0;
    const has_rela = rela.len > 0;
    // `.rela.data`/`.rela.rodata` hold the data-section pointer-init relocations. They stay
    // empty for every object that carries none, so this is additive and byte-identical
    // when absent.
    const has_rela_data = rela_data.len > 0;
    const has_rela_rodata = rela_rodata.len > 0;

    // Compute section indices up front (NULL is 0), so the link and info fields are correct.
    // `rodata_ndx` and `data_ndx` are tracked explicitly, because `.rela.data` and
    // `.rela.rodata` need their target section's index for `sh_info`.
    const text_ndx: u32 = 1;
    var next_ndx: u32 = 2;
    const rodata_ndx: u32 = if (has_rodata) blk: {
        defer next_ndx += 1;
        break :blk next_ndx;
    } else 0;
    const data_ndx: u32 = if (has_data) blk: {
        defer next_ndx += 1;
        break :blk next_ndx;
    } else 0;
    if (has_bss) next_ndx += 1;
    if (has_rela) next_ndx += 1; // .rela.text
    if (has_rela_data) next_ndx += 1; // .rela.data
    if (has_rela_rodata) next_ndx += 1; // .rela.rodata
    const debug0_ndx: u32 = next_ndx;
    const symtab_ndx: u32 = debug0_ndx + @as(u32, @intCast(debug.len));
    const strtab_ndx: u32 = symtab_ndx + 1;

    // Assemble the ordered section list (excluding the NULL section at index 0).
    var secs: std.ArrayList(Sec) = .empty;
    defer secs.deinit(allocator);
    try secs.append(allocator, .{ .name = ".text", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_EXECINSTR, .data = text, .addralign = 16 });
    if (has_rodata) try secs.append(allocator, .{ .name = ".rodata", .typ = SHT_PROGBITS, .flags = SHF_ALLOC, .data = rodata, .addralign = 8 });
    if (has_data) try secs.append(allocator, .{ .name = ".data", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_WRITE, .data = data, .addralign = 8 });
    if (has_bss) try secs.append(allocator, .{ .name = ".bss", .typ = SHT_NOBITS, .flags = SHF_ALLOC | SHF_WRITE, .data = &.{}, .size = bss_size, .addralign = 8 });
    if (has_rela) try secs.append(allocator, .{ .name = ".rela.text", .typ = SHT_RELA, .flags = SHF_INFO_LINK, .data = rela, .link = symtab_ndx, .info = text_ndx, .addralign = 8, .entsize = 24 });
    if (has_rela_data) try secs.append(allocator, .{ .name = ".rela.data", .typ = SHT_RELA, .flags = SHF_INFO_LINK, .data = rela_data, .link = symtab_ndx, .info = data_ndx, .addralign = 8, .entsize = 24 });
    if (has_rela_rodata) try secs.append(allocator, .{ .name = ".rela.rodata", .typ = SHT_RELA, .flags = SHF_INFO_LINK, .data = rela_rodata, .link = symtab_ndx, .info = rodata_ndx, .addralign = 8, .entsize = 24 });
    for (debug) |d| try secs.append(allocator, .{ .name = d.name, .typ = SHT_PROGBITS, .data = d.bytes, .addralign = 1 });
    try secs.append(allocator, .{ .name = ".symtab", .typ = SHT_SYMTAB, .data = symtab, .link = strtab_ndx, .info = @intCast(first_global), .addralign = 8, .entsize = 24 });
    try secs.append(allocator, .{ .name = ".strtab", .typ = SHT_STRTAB, .data = strtab, .addralign = 1 });

    // The section-name string table, including its own name, built in index order.
    var names: std.ArrayList(u8) = .empty;
    defer names.deinit(allocator);
    try names.append(allocator, 0); // the NULL section's empty name
    const name_offs = try allocator.alloc(u32, secs.items.len + 1); // +1 for .shstrtab
    defer allocator.free(name_offs);
    for (secs.items, 0..) |s, i| {
        name_offs[i] = @intCast(names.items.len);
        try names.appendSlice(allocator, s.name);
        try names.append(allocator, 0);
    }
    name_offs[secs.items.len] = @intCast(names.items.len);
    try names.appendSlice(allocator, ".shstrtab\x00");

    // File offsets: each section's data (8-aligned), then the shstrtab, then the header table.
    var off: usize = 64;
    const data_offs = try allocator.alloc(usize, secs.items.len);
    defer allocator.free(data_offs);
    for (secs.items, 0..) |s, i| {
        off = std.mem.alignForward(usize, off, if (s.addralign > 0) s.addralign else 1);
        data_offs[i] = off;
        off += s.data.len;
    }
    off = std.mem.alignForward(usize, off, 1);
    const shstr_off = off;
    off += names.items.len;
    off = std.mem.alignForward(usize, off, 8);
    const sh_off = off;
    const nsections = secs.items.len + 2; // + NULL + .shstrtab
    const total = sh_off + nsections * 64;

    const buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    put(buf, u16, 16, ET_REL);
    put(buf, u16, 18, EM_X86_64);
    put(buf, u32, 20, 1); // e_version
    put(buf, u64, 40, sh_off); // e_shoff
    put(buf, u16, 52, 64); // e_ehsize
    put(buf, u16, 58, 64); // e_shentsize
    put(buf, u16, 60, @intCast(nsections)); // e_shnum
    put(buf, u16, 62, @intCast(nsections - 1)); // e_shstrndx (.shstrtab is last)

    for (secs.items, 0..) |s, i| @memcpy(buf[data_offs[i]..][0..s.data.len], s.data);
    @memcpy(buf[shstr_off..][0..names.items.len], names.items);

    // Section headers: NULL, each section, then .shstrtab.
    const sh = buf[sh_off..];
    for (secs.items, 0..) |s, i| {
        putShdr(sh[(i + 1) * 64 ..], name_offs[i], s.typ, s.flags, data_offs[i], s.size orelse s.data.len, s.link, s.info, s.addralign, s.entsize);
    }
    putShdr(sh[(nsections - 1) * 64 ..], name_offs[secs.items.len], SHT_STRTAB, 0, shstr_off, names.items.len, 0, 0, 1, 0);
    return buf;
}

fn putShdr(e: []u8, name: u32, typ: u32, flags: u64, offset: usize, size: usize, sh_link: u32, sh_info: u32, addralign: u64, entsize: u64) void {
    put(e, u32, 0, name);
    put(e, u32, 4, typ);
    put(e, u64, 8, flags);
    put(e, u64, 24, offset);
    put(e, u64, 32, size);
    put(e, u32, 40, sh_link);
    put(e, u32, 44, sh_info);
    put(e, u64, 48, addralign);
    put(e, u64, 56, entsize);
}

fn nameOff(names: []const u8, name: []const u8) u32 {
    return @intCast(std.mem.indexOf(u8, names, name).?);
}

test "emits an ELF64 x86-64 relocatable object with a call relocation" {
    const allocator = std.testing.allocator;
    const Function = ir.function.Function;
    var helper = Function.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        helper.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }
    var main = Function.init(allocator);
    defer main.deinit();
    {
        const t = try main.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try main.appendBlock();
        const x = try main.appendBlockParam(b, t);
        const r = try main.appendCall(b, t, "helper", &.{x});
        main.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "helper", &helper);

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);
    try std.testing.expectEqualSlices(u8, "\x7fELF", obj[0..4]);
    try std.testing.expectEqual(@as(u16, ET_REL), std.mem.readInt(u16, obj[16..18], .little));
    try std.testing.expectEqual(@as(u16, EM_X86_64), std.mem.readInt(u16, obj[18..20], .little));
}

test "isel collects source-line rows from debug.line attributes" {
    const allocator = std.testing.allocator;
    const F = ir.function.Function;
    var func = F.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const add_idx = func.instCount();
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    try func.addAttr(.{ .inst = @enumFromInt(@as(u32, @intCast(add_idx))) }, .{ .custom = .{ .namespace = "debug", .key = "line", .value = .{ .int = 7 } } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });

    var compiled = try isel.compile(allocator, &func);
    defer compiled.deinit(allocator);
    var found = false;
    for (compiled.lines) |ln| if (ln.line == 7) {
        found = true;
    };
    try std.testing.expect(found);
}

test "writeModuleWithDebug emits DWARF (readelf + self-decoded line program)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const elf_read = @import("../elf_read.zig");
    const dwarf_mod = @import("../dwarf.zig");
    const F = ir.function.Function;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // helper(x) -> x + 5, and main(x) -> helper(x).
    var helper = F.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(i32k);
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        const s = try helper.appendArithImm(b, t, .add, x, 5);
        helper.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });
    }
    var main_f = F.init(allocator);
    defer main_f.deinit();
    {
        const t = try main_f.types.intern(i32k);
        const b = try main_f.appendBlock();
        const x = try main_f.appendBlockParam(b, t);
        const r = try main_f.appendCall(b, t, "helper", &.{x});
        main_f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "helper", &helper);
    try module.addFunction(allocator, "main", &main_f);

    const obj = try writeModuleWithDebug(allocator, &module, "mod.c");
    defer allocator.free(obj);

    // Self-contained check: our own decoder reads back the object's .debug_line as a valid program.
    const dl = (try elf_read.sectionByName(obj, ".debug_line")) orelse return error.NoLine;
    const rows = try dwarf_mod.decodeLine(allocator, dl);
    defer allocator.free(rows);
    var saw_end = false;
    for (rows) |r| if (r.end_sequence) {
        saw_end = true;
    };
    try std.testing.expect(saw_end);

    // And readelf agrees: both subprograms, the CU-to-line link, and the typed return.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "m.o", .data = obj });
    const res = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "--debug-dump=info", "m.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    if (res.term != .exited or res.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_TAG_subprogram") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "helper") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "main") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_AT_stmt_list") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "int") != null);
}

test "readelf shows .rodata/.data/.bss and a PC32 relocation for a global_addr load" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const F = ir.function.Function;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&K), where K is an i32 rodata constant. This is the exact shape that
    // isel's `.global_addr` arm (`lea rd, [rip+disp32]`) lowers to a PC32 relocation for.
    var entry = F.init(allocator);
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
    try module.addWritable(allocator, "W", &k_bytes); // .data
    try module.addBss(allocator, "B", 8); // .bss

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
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, ".data") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, ".bss") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, "NOBITS") != null);

    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-r", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_X86_64_PC32") != null);

    const syms = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-s", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(syms.stdout);
    defer allocator.free(syms.stderr);
    try std.testing.expect(std.mem.indexOf(u8, syms.stdout, "OBJECT") != null);
}
