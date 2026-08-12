//! The shared default (no-linker-script) placer for the static linker. Every
//! architecture backend's `computeDefaultPlacement` now calls this one routine: it
//! packs each object's arbitrary named `ObjSection`s into four concatenated regions
//! (text, then rodata, then data, then bss), builds the single loadable segment, and
//! resolves the defined symbols. The per-architecture differences it takes as inputs:
//! an `ArchDesc` of the per-class alignments (text is 4 on riscv64/aarch64, 16 on the
//! x86 backends), and an optional `StubHook` for riscv64's extern-stub/GOT region, the
//! one placement piece no other backend has.
//!
//! It fills the `section_places` map (one place per (object, section)). Two same-class
//! sections keep distinct places, so a symbol resolves to the section that defines it. As
//! long as each object carries at most one section per class (the case for every object
//! the linker sees today), every address matches the old per-architecture placers byte
//! for byte.

const std = @import("std");
const elf = @import("elf.zig");

const Error = elf.Error;
const ObjSection = elf.ObjSection;
const ParsedObject = elf.ParsedObject;
const ResolvedSymbol = elf.ResolvedSymbol;
const Resolver = elf.Resolver;
const Segment = elf.Segment;
const SecPlace = elf.SecPlace;
const Placement = elf.Placement;
const alignUp = elf.alignUp;
const findSymbol = elf.findSymbol;
const not_placed = elf.not_placed;

/// The broad placement class one allocatable section packs into: text first, then rodata,
/// then data, then bss. Every `ObjSection` is allocatable (the parser drops the rest), so
/// `classOf` always returns one of these.
const Class = enum { text, rodata, data, bss };

/// The per-architecture placement parameters: the alignment each section class packs
/// to within its region, and the alignment each region concatenates at. The x86
/// backends want a 16-byte text alignment; riscv64 and aarch64 want 4. Everything else
/// is 8 on every architecture today.
pub const ArchDesc = struct {
    text_align: u64,
    ro_align: u64 = 8,
    rw_align: u64 = 8,
    bss_align: u64 = 8,
    region_align: u64 = 8,
};

/// riscv64's extern-stub/GOT region hook, the one non-uniform placement piece. It runs
/// in two phases so the shared placer can size the file image around it. `reserve` runs
/// after the data region end (with the resolved symbols and the object relocs in hand):
/// it collects the externs it must synthesize and returns the new file size, which the
/// stub + GOT bytes extend past `data_end`. `emit` runs once the code buffer exists: it
/// writes each stub and GOT slot into `code` and appends each stub's symbol. The other
/// three backends pass no hook, so the file size stays `data_end`.
pub const StubHook = struct {
    context: *anyopaque,
    reserve: *const fn (
        context: *anyopaque,
        allocator: std.mem.Allocator,
        objs: []const ParsedObject,
        symbols: []const ResolvedSymbol,
        data_end: u64,
        resolver: ?Resolver,
        // The GC liveness mask (`section_place_base[oi] + si`), or null when GC is off. A
        // dead section must not seed a stub. `section_place_base` gives the flat index.
        live: ?[]const bool,
        section_place_base: []const usize,
    ) Error!u64,
    emit: *const fn (
        context: *anyopaque,
        allocator: std.mem.Allocator,
        code: []u8,
        symbols: *std.ArrayList(ResolvedSymbol),
        base: u64,
    ) Error!void,
};

/// The placement class of one allocatable section, from its flags.
fn classOf(sec: ObjSection) Class {
    if (sec.is_nobits) return .bss;
    if ((sec.flags & elf.SHF_EXECINSTR) != 0) return .text;
    if ((sec.flags & elf.SHF_WRITE) != 0) return .data;
    return .rodata;
}

/// The runtime offset (from `base`) at which a section class's region starts. Text is
/// first at `base` (offset 0), then rodata, then data, then bss.
fn regionBase(kind: Class, rodata_region: u64, data_region: u64, bss_region: u64) u64 {
    return switch (kind) {
        .text => 0,
        .rodata => rodata_region,
        .data => data_region,
        .bss => bss_region,
    };
}

/// Assign addresses for a set of parsed relocatable objects (the default, no
/// linker-script layout). Objects lay out in the order given, `.text` first at `base`,
/// then `.rodata`, `.data`, an optional per-`StubHook` region, and `.bss` (memory only).
/// Produces a single loadable `Segment` (the whole image at `base`, R|W|X), the
/// `section_places` map (one place per (object, section)), and the resolved symbol table
/// (`base + region + value`, rejecting a duplicate, skipping undef/local/ABS symbols).
/// No relocations are applied here (that is each backend's `applyRelocs`). The caller
/// owns the returned placement; it does not own `objs`.
/// `live` is the GC liveness mask (`section_place_base[oi] + si`), or null when GC is off.
/// A DEAD section (mask false) is packed exactly like a zero-size not-present section: it
/// takes no space, copies no bytes, resolves no symbol, and gets a `not_placed` place. When
/// `live == null` every section is live, so the output is byte-identical to before.
pub fn computeDefaultPlacement(
    allocator: std.mem.Allocator,
    objs: []const ParsedObject,
    base: u64,
    desc: ArchDesc,
    stub: ?StubHook,
    resolver: ?Resolver,
    live: ?[]const bool,
) Error!Placement {
    const nobj = objs.len;

    // A per-object base offset into the flat `section_places` array (the last entry is
    // the total section count). `sectionPlace(oi, si)` reads `base[oi] + si`.
    var section_place_base = try allocator.alloc(usize, nobj + 1);
    errdefer allocator.free(section_place_base);
    var total_secs: usize = 0;
    for (0..nobj) |oi| {
        section_place_base[oi] = total_secs;
        total_secs += objs[oi].sections.len;
    }
    section_place_base[nobj] = total_secs;

    // Each section's class, its class-relative (within-region) offset, and whether it
    // has any bytes to place.
    var sec_kind = try allocator.alloc(Class, total_secs);
    defer allocator.free(sec_kind);
    var sec_off = try allocator.alloc(u64, total_secs);
    defer allocator.free(sec_off);
    var sec_present = try allocator.alloc(bool, total_secs);
    defer allocator.free(sec_present);

    // Pack every object's sections within their class regions, keeping the four class
    // totals. A zero-size section aligns the running total but is not placed, matching
    // the old placers (which aligned each region per object regardless of size).
    var text_total: u64 = 0;
    var rodata_total: u64 = 0;
    var data_total: u64 = 0;
    var bss_total: u64 = 0;
    for (0..nobj) |oi| {
        const obase = section_place_base[oi];
        for (objs[oi].sections, 0..) |sec, si| {
            const kind = classOf(sec);
            const gi = obase + si;
            // A GC-dead section counts as zero-size, so it takes no room and is not placed,
            // the same as a zero-size not-present section. `sz` is 0 for a dead section.
            const dead = if (live) |lv| !lv[gi] else false;
            const sz = if (dead) 0 else sec.size;
            const present = sz > 0;
            const off = switch (kind) {
                .text => blk: {
                    const o = alignUp(text_total, desc.text_align);
                    text_total = o + sz;
                    break :blk o;
                },
                .rodata => blk: {
                    const o = alignUp(rodata_total, desc.ro_align);
                    rodata_total = o + sz;
                    break :blk o;
                },
                .data => blk: {
                    const o = alignUp(data_total, desc.rw_align);
                    data_total = o + sz;
                    break :blk o;
                },
                .bss => blk: {
                    const o = alignUp(bss_total, desc.bss_align);
                    bss_total = o + sz;
                    break :blk o;
                },
            };
            sec_kind[gi] = kind;
            sec_off[gi] = off;
            sec_present[gi] = present;
        }
    }

    // Concatenate the regions: text at `base`, then rodata, then data, each aligned. The
    // file image ends after the last non-empty region (`data_end`).
    const rodata_region = alignUp(text_total, desc.region_align);
    const data_region = alignUp(rodata_region + rodata_total, desc.region_align);
    var data_end: u64 = text_total;
    if (rodata_total > 0) data_end = rodata_region + rodata_total;
    if (data_total > 0) data_end = data_region + data_total;

    // The bss region used to resolve `.bss` symbols. A `StubHook` region (riscv64) can
    // still push the file image past `data_end`, but the old riscv64 placer resolved bss
    // symbols against `alignUp(data_end, region_align)` (before the stub region) while
    // placing bss after it, so keep that split exactly.
    const bss_region_pre = alignUp(data_end, desc.region_align);

    // Resolve each defined, non-local, named symbol to its final address, through the
    // symbol's OWN section (by `section_index`), not its class region. The section's runtime
    // address is `base + regionBase(kind) + sec_off[gi]`, the same value `sectionPlace(oi,
    // section_index)` yields (bss still resolves against `bss_region_pre`, before the stub
    // region, matching the placed section address for every non-riscv64 object and the old
    // riscv64 bss split). With one section per class this is the old class-region address;
    // with two same-class sections a symbol now resolves to the section that defines it,
    // closing the global-symbol last-wins gap.
    var symbols: std.ArrayList(ResolvedSymbol) = .empty;
    errdefer {
        for (symbols.items) |s| allocator.free(s.name);
        symbols.deinit(allocator);
    }
    for (0..nobj) |oi| {
        const obase = section_place_base[oi];
        for (objs[oi].symbols) |sym| {
            // A symbol can be "defined" (st_shndx != UNDEF) yet resolve to no
            // allocatable section (an ABS symbol, or one naming a reserved section):
            // it has no region offset, so skip it.
            if (!sym.defined or sym.local or sym.name.len == 0 or sym.section_index == std.math.maxInt(u32)) continue;
            // A symbol in a GC-dead section is dropped along with its section, so it never
            // enters the resolved table (it would resolve into the not-placed region).
            if (live) |lv| {
                if (!lv[obase + sym.section_index]) continue;
            }
            if (findSymbol(symbols.items, sym.name) != null) return error.DuplicateSymbol;
            const gi = obase + sym.section_index;
            const sec_vaddr = base + regionBase(sec_kind[gi], rodata_region, data_region, bss_region_pre) + sec_off[gi];
            const name = try allocator.dupe(u8, sym.name);
            errdefer allocator.free(name);
            try symbols.append(allocator, .{ .name = name, .address = sec_vaddr + sym.value, .is_exec = sec_kind[gi] == .text });
        }
    }

    // Reserve the optional stub/GOT region after the data end, which can extend the file
    // image. Without a hook the file image ends at `data_end`.
    var filesz = data_end;
    if (stub) |h| filesz = try h.reserve(h.context, allocator, objs, symbols.items, data_end, resolver, live, section_place_base);
    const bss_region = alignUp(filesz, desc.region_align);
    const memsz = if (bss_total > 0) bss_region + bss_total else filesz;

    // The loadable file image: every non-bss section copied into place. Bss is zero and
    // lives only in `memsz`, beyond the file image.
    var code = try allocator.alloc(u8, @intCast(filesz));
    errdefer allocator.free(code);
    @memset(code, 0);
    for (0..nobj) |oi| {
        const obase = section_place_base[oi];
        for (objs[oi].sections, 0..) |sec, si| {
            if (sec.is_nobits or sec.bytes.len == 0) continue;
            const gi = obase + si;
            // A GC-dead section copies no bytes (it is not placed).
            if (live) |lv| {
                if (!lv[gi]) continue;
            }
            const off = regionBase(sec_kind[gi], rodata_region, data_region, bss_region) + sec_off[gi];
            @memcpy(code[@intCast(off)..][0..sec.bytes.len], sec.bytes);
        }
    }

    // Let the hook synthesize its stubs and GOT slots into `code` and append their
    // symbols, now that the code buffer exists.
    if (stub) |h| try h.emit(h.context, allocator, code, &symbols, base);

    // One segment: the whole image at `base` (R|W|X), the bss tail living only in memsz.
    var segments = try allocator.alloc(Segment, 1);
    errdefer allocator.free(segments);
    segments[0] = .{ .vaddr = base, .paddr = base, .bytes = code, .memsz = memsz, .flags = 7 };

    // The per-(object, section) place map: one segment (index 0), the region offset being
    // both the in-segment offset and (added to `base`) the runtime address. Two same-class
    // sections keep distinct places here, so a symbol resolves to the section that defines
    // it rather than a merged per-class base.
    var section_places = try allocator.alloc(SecPlace, total_secs);
    errdefer allocator.free(section_places);
    for (0..nobj) |oi| {
        const obase = section_place_base[oi];
        for (0..objs[oi].sections.len) |si| {
            const gi = obase + si;
            if (sec_present[gi]) {
                const off = regionBase(sec_kind[gi], rodata_region, data_region, bss_region) + sec_off[gi];
                section_places[gi] = .{ .vaddr = base + off, .seg = 0, .seg_off = off };
            } else {
                section_places[gi] = .{ .vaddr = 0, .seg = not_placed, .seg_off = 0 };
            }
        }
    }

    return .{
        .segments = segments,
        .symbols = try symbols.toOwnedSlice(allocator),
        .entry = 0,
        .section_places = section_places,
        .section_place_base = section_place_base,
    };
}
