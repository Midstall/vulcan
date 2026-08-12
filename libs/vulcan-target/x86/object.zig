//! ELF32 relocatable object (ET_REL, EM_386) emission. Each function becomes an STT_FUNC
//! global in .text; each data global (`global_addr` target) becomes an STT_OBJECT global
//! in .rodata/.data/.bss. Every `call` becomes an R_386_PC32 relocation and every
//! `global_addr` becomes an R_386_32 relocation against its symbol. i386 uses SHT_REL, so
//! the addend is stored implicitly in the relocated field itself (-4 for a call's
//! PC-relative displacement, 0 for an absolute address). link.zig is the in-memory linker.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");

pub const Error = isel.Error;

const ET_REL: u16 = 1;
const EM_386: u16 = 3;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_REL: u32 = 9;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u32 = 0x1;
const SHF_ALLOC: u32 = 0x2;
const SHF_EXECINSTR: u32 = 0x4;
const SHF_INFO_LINK: u32 = 0x40;
/// `R_386_32`: a plain 32-bit absolute address (`S + A`). Doubles as BOTH a `.text` reloc (a
/// `global_addr` load's `mov rd, imm32`) AND a DATA-section reloc (`.rel.data`/`.rel.rodata`,
/// a pointer-initialized global's `int *p = &g;` slot) - which list a given entry feeds is
/// decided by which section its `SHT_REL` table targets (`elf.zig`'s `parseObject32`), not by
/// this numeric code. i386 uses `SHT_REL` (no addend field): the slot itself holds 0 (the
/// linker turns an internal-target one into a `R_386_RELATIVE` dyn reloc for a PIE/`.so`, pre-
/// writing the nominal target vaddr into the slot, or a direct absolute write for a non-PIE
/// exec - mirrors `x86_64/object.zig`'s `R_X86_64_64`, minus the addend field).
const R_386_32: u32 = 1;
const R_386_PC32: u32 = 2;
/// `R_386_GOT32`: a GOT-indirect `global_addr`'s `mov rd, [abs32]` (`via_got`) reference to a
/// symbol's GOT slot (a data import). The dynamic linker synthesizes the GOT slot + a
/// `R_386_GLOB_DAT` and patches the abs32 field to the slot's vaddr.
const R_386_GOT32: u32 = 3;
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

const StrTab = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn init(allocator: std.mem.Allocator) !StrTab {
        var s: StrTab = .{};
        try s.bytes.append(allocator, 0);
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
/// index: the section + byte offset of the slot, and the target symbol's name (mirrors
/// `x86_64/object.zig`'s `PendingDataReloc`).
const PendingDataReloc = struct { section: SectionKind, offset: u64, symbol: []const u8 };

/// A data-section pointer-init relocation applied to a `.data`/`.rodata` slot against a
/// symbol: at byte `offset` within `section`, an `R_386_32` writes `symbol`'s runtime address
/// (no addend field - i386 `SHT_REL`). Emitted into `.rel.data`/`.rel.rodata` (keyed by
/// `section`). This closes the data-section-relocation gap on i386: a data global's own
/// pointer inits are now carried in the object, not dropped.
pub const DataRelocEntry = struct { section: SectionKind, offset: u64, symbol: u32 };

/// Lay out `module.data` into three section blobs (rodata/data appended verbatim,
/// bss tracked as a size only) and record each global's placement. Each data global's own
/// `DataReloc`s (pointer inits) are recorded in `data_relocs` at their absolute section offset
/// (the global's placement plus the reloc's in-object offset).
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

pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error![]u8 {
    const funcs = module.funcs.items;
    const compiled = try allocator.alloc(isel.Compiled, funcs.len);
    var compiled_n: usize = 0;
    defer {
        for (compiled[0..compiled_n]) |*c| c.deinit(allocator);
        allocator.free(compiled);
    }
    for (funcs, 0..) |e, i| {
        compiled[i] = try isel.compile(allocator, e.func);
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
    // i386 REL: the addend lives in the field itself. `.call` (R_386_PC32) is -4 (the
    // rel32 field's own width); `.abs32` (R_386_32) is 0 (isel already emits a zero
    // imm32 placeholder, so its field needs no rewrite - the linker just adds S to it).
    for (compiled, 0..) |c, i| for (c.relocs) |r| switch (r.kind) {
        .call => put(text.items, i32, offsets[i] + r.offset, -4),
        // `.abs32`/`.got_abs` both emit a zero placeholder (the linker adds S / the GOT slot
        // vaddr to it), so their fields need no rewrite here.
        .abs32, .got_abs => {},
    };

    // Data globals (`global_addr` targets): laid out into their own rodata/data/bss
    // blobs, placed in the module's own sections (not `.text`).
    var rodata: std.ArrayList(u8) = .empty;
    defer rodata.deinit(allocator);
    var data_sec: std.ArrayList(u8) = .empty;
    defer data_sec.deinit(allocator);
    var pending_data_relocs: std.ArrayList(PendingDataReloc) = .empty;
    defer pending_data_relocs.deinit(allocator);
    const laid_out = try layoutData(allocator, module, &rodata, &data_sec, &pending_data_relocs);
    defer allocator.free(laid_out.places);

    // Section indices matching the layout `assemble` builds: NULL(0), .text(1), then
    // whichever of .rodata/.data/.bss are non-empty, in that order.
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

    var strtab = try StrTab.init(allocator);
    defer strtab.bytes.deinit(allocator);
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    var sym_index = std.StringHashMapUnmanaged(u32){};
    defer sym_index.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 16); // null symbol

    for (funcs, 0..) |e, i| {
        try sym_index.put(allocator, e.name, @intCast(symtab.items.len / 16));
        try appendSym(allocator, &symtab, try strtab.add(allocator, e.name), STT_FUNC, STB_GLOBAL, text_ndx, offsets[i], compiled[i].code.len);
    }
    for (laid_out.places) |dp| {
        const shndx: u16 = switch (dp.section) {
            .rodata => rodata_ndx,
            .data => data_ndx,
            .bss => bss_ndx,
        };
        try sym_index.put(allocator, dp.name, @intCast(symtab.items.len / 16));
        try appendSym(allocator, &symtab, try strtab.add(allocator, dp.name), STT_OBJECT, STB_GLOBAL, shndx, dp.value, dp.size);
    }
    for (compiled) |c| for (c.relocs) |r| {
        if (!sym_index.contains(r.symbol)) {
            try sym_index.put(allocator, r.symbol, @intCast(symtab.items.len / 16));
            try appendSym(allocator, &symtab, try strtab.add(allocator, r.symbol), STT_NOTYPE, 0, SHN_UNDEF, 0, 0); // undefined
        }
    };

    var rel: std.ArrayList(u8) = .empty;
    defer rel.deinit(allocator);
    for (compiled, 0..) |c, i| for (c.relocs) |r| {
        var ent: [8]u8 = undefined;
        put(&ent, u32, 0, @intCast(offsets[i] + r.offset)); // r_offset
        const rtype: u32 = switch (r.kind) {
            .call => R_386_PC32,
            .abs32 => R_386_32,
            .got_abs => R_386_GOT32,
        };
        put(&ent, u32, 4, (sym_index.get(r.symbol).? << 8) | rtype); // r_info
        try rel.appendSlice(allocator, &ent);
    };

    // Resolve each data-section reloc's target name to a symbol index. The target of a
    // pointer init is an internally defined data/function global (added above).
    var data_relocs = try allocator.alloc(DataRelocEntry, pending_data_relocs.items.len);
    defer allocator.free(data_relocs);
    for (pending_data_relocs.items, 0..) |p, i| {
        data_relocs[i] = .{ .section = p.section, .offset = p.offset, .symbol = sym_index.get(p.symbol) orelse return error.Unsupported };
    }

    // `.rel.data` / `.rel.rodata`: the data-section pointer-init relocations, each an
    // `R_386_32` against the target symbol (no addend field - i386 `SHT_REL`; the object
    // already wrote 0 into the slot), grouped by the section they modify.
    var rel_data: std.ArrayList(u8) = .empty;
    defer rel_data.deinit(allocator);
    var rel_rodata: std.ArrayList(u8) = .empty;
    defer rel_rodata.deinit(allocator);
    for (data_relocs) |dr| {
        var ent: [8]u8 = undefined;
        put(&ent, u32, 0, @intCast(dr.offset)); // r_offset
        put(&ent, u32, 4, (dr.symbol << 8) | R_386_32); // r_info
        const dst: *std.ArrayList(u8) = switch (dr.section) {
            .data => &rel_data,
            .rodata => &rel_rodata,
            .bss => unreachable, // a data reloc never targets .bss
        };
        try dst.appendSlice(allocator, &ent);
    }

    return assemble(allocator, text.items, rodata.items, data_sec.items, laid_out.bss_size, rel.items, rel_data.items, rel_rodata.items, symtab.items, strtab.bytes.items, funcs.len + laid_out.places.len + 1);
}

fn appendSym(allocator: std.mem.Allocator, symtab: *std.ArrayList(u8), name_off: u32, typ: u8, binding: u8, shndx: u16, value: u64, size: u64) !void {
    var ent: [16]u8 = undefined;
    @memset(&ent, 0);
    put(&ent, u32, 0, name_off); // st_name
    put(&ent, u32, 4, @intCast(value)); // st_value
    put(&ent, u32, 8, @intCast(size)); // st_size
    ent[12] = (binding << 4) | typ; // st_info
    put(&ent, u16, 14, shndx); // st_shndx
    try symtab.appendSlice(allocator, ent[0..16]);
}

/// A section header descriptor for the flexible ELF writer below.
const Sec = struct {
    name: []const u8,
    typ: u32,
    flags: u32 = 0,
    data: []const u8,
    link: u32 = 0,
    info: u32 = 0,
    addralign: u32 = 1,
    entsize: u32 = 0,
    /// `sh_size` override for `SHT_NOBITS` (`.bss`), which occupies no file bytes
    /// (`data` stays empty) but still reports its zero-initialized size. Null (every
    /// other section) means `sh_size` is just `data.len`. `u64` (not `usize`) so a
    /// `link.Data.size`-derived `bss_size` assigns cleanly on a 32-bit `usize` target.
    size: ?u64 = null,
};

/// Lay out the ELF: header, section data, and the section header table. Order: NULL, .text,
/// [.rodata], [.data], [.bss], [.rel.text], [.rel.data], [.rel.rodata], .symtab, .strtab,
/// .shstrtab. `first_global` is the index of the first global symbol.
fn assemble(allocator: std.mem.Allocator, text: []const u8, rodata: []const u8, data: []const u8, bss_size: u64, rel: []const u8, rel_data: []const u8, rel_rodata: []const u8, symtab: []const u8, strtab: []const u8, first_global: usize) Error![]u8 {
    const has_rodata = rodata.len > 0;
    const has_data = data.len > 0;
    const has_bss = bss_size > 0;
    const has_rel = rel.len > 0;
    // `.rel.data`/`.rel.rodata`: the data-section pointer-init relocations (empty for every
    // object that carries none, so this is additive and byte-identical when absent).
    const has_rel_data = rel_data.len > 0;
    const has_rel_rodata = rel_rodata.len > 0;

    // Compute section indices up front (NULL is 0) so link/info fields are right.
    // `rodata_ndx`/`data_ndx` are tracked explicitly (unlike before this change) because
    // `.rel.data`/`.rel.rodata` need their target section's index for `sh_info`.
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
    if (has_rel) next_ndx += 1; // .rel.text
    if (has_rel_data) next_ndx += 1; // .rel.data
    if (has_rel_rodata) next_ndx += 1; // .rel.rodata
    const symtab_ndx: u32 = next_ndx;
    const strtab_ndx: u32 = symtab_ndx + 1;

    // Assemble the ordered section list (excluding the NULL section at index 0).
    var secs: std.ArrayList(Sec) = .empty;
    defer secs.deinit(allocator);
    try secs.append(allocator, .{ .name = ".text", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_EXECINSTR, .data = text, .addralign = 16 });
    if (has_rodata) try secs.append(allocator, .{ .name = ".rodata", .typ = SHT_PROGBITS, .flags = SHF_ALLOC, .data = rodata, .addralign = 4 });
    if (has_data) try secs.append(allocator, .{ .name = ".data", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_WRITE, .data = data, .addralign = 4 });
    if (has_bss) try secs.append(allocator, .{ .name = ".bss", .typ = SHT_NOBITS, .flags = SHF_ALLOC | SHF_WRITE, .data = &.{}, .size = bss_size, .addralign = 4 });
    if (has_rel) try secs.append(allocator, .{ .name = ".rel.text", .typ = SHT_REL, .flags = SHF_INFO_LINK, .data = rel, .link = symtab_ndx, .info = text_ndx, .addralign = 4, .entsize = 8 });
    if (has_rel_data) try secs.append(allocator, .{ .name = ".rel.data", .typ = SHT_REL, .flags = SHF_INFO_LINK, .data = rel_data, .link = symtab_ndx, .info = data_ndx, .addralign = 4, .entsize = 8 });
    if (has_rel_rodata) try secs.append(allocator, .{ .name = ".rel.rodata", .typ = SHT_REL, .flags = SHF_INFO_LINK, .data = rel_rodata, .link = symtab_ndx, .info = rodata_ndx, .addralign = 4, .entsize = 8 });
    try secs.append(allocator, .{ .name = ".symtab", .typ = SHT_SYMTAB, .data = symtab, .link = strtab_ndx, .info = @intCast(first_global), .addralign = 4, .entsize = 16 });
    try secs.append(allocator, .{ .name = ".strtab", .typ = SHT_STRTAB, .data = strtab, .addralign = 1 });

    // The section-name string table (its own name included), built in index order.
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

    // File offsets for each section's data (4-aligned), then the shstrtab, then the header table.
    var off: usize = 52;
    const data_offs = try allocator.alloc(usize, secs.items.len);
    defer allocator.free(data_offs);
    for (secs.items, 0..) |s, i| {
        off = std.mem.alignForward(usize, off, if (s.addralign > 0) s.addralign else 1);
        data_offs[i] = off;
        off += s.data.len;
    }
    const shstr_off = off;
    off += names.items.len;
    off = std.mem.alignForward(usize, off, 4);
    const sh_off = off;
    const nsections = secs.items.len + 2; // + NULL + .shstrtab
    const total = sh_off + nsections * 40;

    const buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 1; // ELFCLASS32
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    put(buf, u16, 16, ET_REL);
    put(buf, u16, 18, EM_386);
    put(buf, u32, 20, 1); // e_version
    put(buf, u32, 32, @intCast(sh_off)); // e_shoff
    put(buf, u16, 40, 52); // e_ehsize
    put(buf, u16, 46, 40); // e_shentsize
    put(buf, u16, 48, @intCast(nsections)); // e_shnum
    put(buf, u16, 50, @intCast(nsections - 1)); // e_shstrndx (.shstrtab is last)

    for (secs.items, 0..) |s, i| @memcpy(buf[data_offs[i]..][0..s.data.len], s.data);
    @memcpy(buf[shstr_off..][0..names.items.len], names.items);

    // Section headers: NULL, each section, then .shstrtab.
    const sh = buf[sh_off..];
    for (secs.items, 0..) |s, i| {
        putShdr(sh[(i + 1) * 40 ..], name_offs[i], s.typ, s.flags, data_offs[i], s.size orelse @as(u64, s.data.len), s.link, s.info, s.addralign, s.entsize);
    }
    putShdr(sh[(nsections - 1) * 40 ..], name_offs[secs.items.len], SHT_STRTAB, 0, shstr_off, names.items.len, 0, 0, 1, 0);
    return buf;
}

fn putShdr(e: []u8, name: u32, typ: u32, flags: u32, offset: usize, size: u64, sh_link: u32, sh_info: u32, addralign: u32, entsize: u32) void {
    put(e, u32, 0, name);
    put(e, u32, 4, typ);
    put(e, u32, 8, flags);
    put(e, u32, 16, @intCast(offset));
    put(e, u32, 20, @intCast(size));
    put(e, u32, 24, sh_link);
    put(e, u32, 28, sh_info);
    put(e, u32, 32, addralign);
    put(e, u32, 36, entsize);
}

test "emits an ELF32 i386 relocatable object" {
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
    try std.testing.expectEqual(@as(u16, EM_386), std.mem.readInt(u16, obj[18..20], .little));
    try std.testing.expectEqual(@as(u8, 1), obj[4]); // ELFCLASS32
}

test "readelf shows .rodata/.data/.bss and an R_386_32 relocation for a global_addr load" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Function = ir.function.Function;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&K), K an i32 rodata constant - the exact shape isel's `.global_addr`
    // arm (`mov rd, imm32`) lowers to an R_386_32 reloc for.
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
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_386_32") != null);

    const syms = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-s", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(syms.stdout);
    defer allocator.free(syms.stderr);
    try std.testing.expect(std.mem.indexOf(u8, syms.stdout, "OBJECT") != null);
}
