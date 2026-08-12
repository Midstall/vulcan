//! The `--gc-sections` mark pass. It walks the relocation graph over the collected
//! object set and reports which sections are LIVE, that is reachable from the program
//! entry and the exported symbols. The caller drops the DEAD sections. Correctness is
//! the priority here. A dropped live section breaks the program in a silent way, so on
//! ANY doubt the pass keeps the section (a conservative-keep rule, see `markObject`).
//!
//! Nothing calls this pass yet. The caller runs it only when GC is enabled. When GC is
//! off the caller skips the pass and treats every section as live.

const std = @import("std");
const elf = @import("elf.zig");

const Error = elf.Error;

/// The sentinel `section_index` for a symbol with no defining section in this link (an
/// undefined or external symbol).
const no_section = std.math.maxInt(u32);

/// A reference to one section: the object index and the section index inside it.
const SecRef = struct { oi: usize, si: u32 };

/// Compute a per-(object, section) liveness mask. The mask is indexed `base[oi] + si`,
/// where `base[oi]` is the sum of `objs[0..oi].sections.len` (the same flat scheme
/// `place.zig` uses). `true` means the section is live and must be kept. `entry_name`
/// is the program entry (for example `"_start"`). `is_shared` roots every global-defined
/// symbol, because a shared object exports them all.
///
/// The caller owns the returned slice and must free it with `allocator`.
pub fn computeLiveSections(
    allocator: std.mem.Allocator,
    objs: []const elf.ParsedObject,
    entry_name: []const u8,
    is_shared: bool,
) Error![]bool {
    const nobj = objs.len;

    // The per-object base offset into the flat mask. `base[oi] + si` is the mask index
    // for section `si` of object `oi`. The last entry is the total section count.
    const base = try allocator.alloc(usize, nobj + 1);
    defer allocator.free(base);
    var total_secs: usize = 0;
    for (0..nobj) |oi| {
        base[oi] = total_secs;
        total_secs += objs[oi].sections.len;
    }
    base[nobj] = total_secs;

    const live = try allocator.alloc(bool, total_secs);
    errdefer allocator.free(live);
    @memset(live, false);

    // A name -> defining (object, section) map over every GLOBAL DEFINED symbol. A
    // reloc to a global symbol resolves its target section through this map. First
    // definition wins. A duplicate global must not happen in a valid link.
    var globals = std.StringHashMap(SecRef).init(allocator);
    defer globals.deinit();
    for (objs, 0..) |obj, oi| {
        for (obj.symbols) |sym| {
            if (sym.defined and !sym.local) {
                const gop = try globals.getOrPut(sym.name);
                if (!gop.found_existing) {
                    gop.value_ptr.* = .{ .oi = oi, .si = sym.section_index };
                }
            }
        }
    }

    var work: std.ArrayList(SecRef) = .empty;
    defer work.deinit(allocator);

    var marker: Marker = .{
        .allocator = allocator,
        .objs = objs,
        .base = base,
        .live = live,
        .work = &work,
        .globals = &globals,
    };

    // Seed the entry. It has no inbound reloc, so it MUST be seeded by name. Take the
    // first defined symbol named `entry_name`. `entry_seeded` records that the root landed
    // on a resolvable section.
    var entry_seeded = false;
    for (objs, 0..) |obj, oi| {
        var found = false;
        for (obj.symbols) |sym| {
            if (sym.defined and std.mem.eql(u8, sym.name, entry_name)) {
                if (sym.section_index != no_section and sym.section_index < obj.sections.len) {
                    try marker.markSection(oi, sym.section_index);
                    entry_seeded = true;
                }
                // Found but with no resolvable section. The root is unseeded, so the
                // keep-all fallback below runs.
                found = true;
                break;
            }
        }
        if (found) break;
    }

    // The safe fallback for a static link. If the entry root can not be seeded (no defined
    // symbol named `entry_name`, or its section is unresolvable) the worklist would start
    // empty and the mask would come back all-false, so a caller would drop EVERY section.
    // Keep everything instead. A shared object roots every global export below, so it does
    // not need this fallback.
    if (!entry_seeded and !is_shared) {
        @memset(live, true);
        return live;
    }

    // Seed every global-defined symbol when the output is a shared object, because a
    // shared object exports them all.
    if (is_shared) {
        for (objs, 0..) |obj, oi| {
            for (obj.symbols) |sym| {
                if (sym.defined and !sym.local) {
                    if (sym.section_index != no_section and sym.section_index < obj.sections.len) {
                        try marker.markSection(oi, sym.section_index);
                    } else {
                        try marker.markObject(oi);
                    }
                }
            }
        }
    }

    // Walk the reachable graph. Each popped section reveals its relocs, and each reloc
    // pulls its target section live.
    while (work.pop()) |cur| {
        const secs = objs[cur.oi].sections;
        if (cur.si >= secs.len) continue; // Defensive. Enqueued indexes are in range.
        for (secs[cur.si].relocs) |r| {
            try marker.resolveReloc(cur.oi, r);
        }
    }

    return live;
}

/// The mutable state the mark walk carries. It marks sections live and pushes the newly
/// live ones onto the worklist.
const Marker = struct {
    allocator: std.mem.Allocator,
    objs: []const elf.ParsedObject,
    base: []const usize,
    live: []bool,
    work: *std.ArrayList(SecRef),
    globals: *const std.StringHashMap(SecRef),

    /// Mark section `si` of object `oi` live and enqueue it if it was not live yet. The
    /// caller must pass an in-range `si`.
    fn markSection(self: *Marker, oi: usize, si: u32) Error!void {
        const idx = self.base[oi] + si;
        if (!self.live[idx]) {
            self.live[idx] = true;
            try self.work.append(self.allocator, .{ .oi = oi, .si = si });
        }
    }

    /// The conservative-keep fallback. It marks EVERY section of object `oi` live. The
    /// pass calls it when a reloc targets a symbol that is defined but its section can
    /// not be resolved. A blunt but safe choice, it keeps too much rather than drop a
    /// live section.
    fn markObject(self: *Marker, oi: usize) Error!void {
        for (0..self.objs[oi].sections.len) |si| {
            try self.markSection(oi, @intCast(si));
        }
    }

    /// Pull the target section of one reloc live. `oi` is the object that owns the reloc.
    ///
    /// This mirrors the real linker's target resolution order (`resolve.zig` applyRelocs
    /// and the arch appliers, for example `arch/aarch64.zig`). The linker resolves a reloc
    /// target by GLOBAL NAME FIRST through `findSymbol` over the merged table, and only
    /// falls back to the symbol's own local section when the name is not a defined global.
    /// GC follows the same order, so its liveness is a superset-or-equal of the sections
    /// the linker actually references. It can NOT drop a referenced section.
    fn resolveReloc(self: *Marker, oi: usize, r: elf.Reloc) Error!void {
        const syms = self.objs[oi].symbols;
        if (r.symbol >= syms.len) {
            // A malformed reloc symbol index. Keep the whole object to stay safe.
            try self.markObject(oi);
            return;
        }
        const sym = syms[r.symbol];

        // Global name first. A named symbol that resolves to a defined global takes the
        // global's defining section, the same as the linker's `findSymbol`. An empty name
        // never matches a named global, so it skips straight to the local fallback.
        if (sym.name.len != 0) {
            if (self.globals.get(sym.name)) |ref| {
                if (ref.si != no_section and ref.si < self.objs[ref.oi].sections.len) {
                    try self.markSection(ref.oi, ref.si);
                } else {
                    // Present in the map but with no resolvable section. Keep its owner.
                    try self.markObject(ref.oi);
                }
                return;
            }
        }

        // Not a defined global by name. Fall back to the symbol's OWN local section, the
        // same fallback the linker takes for a local defined symbol (for example a `.str.N`
        // string literal or a section symbol).
        if (sym.section_index != no_section and sym.section_index < self.objs[oi].sections.len) {
            try self.markSection(oi, sym.section_index);
        } else if (sym.defined) {
            // A defined symbol with no resolvable section. Keep the reloc's owning object.
            try self.markObject(oi);
        }
        // An undefined or import symbol has no section in this link. This is normal. A
        // shared object resolves it at run time, so skip it.
    }
};

test "gc marks reachable sections and drops the unused one" {
    const a = std.testing.allocator;

    // One aarch64 object. Four executable sections, each 4 bytes.
    //   section 0 .text._start  reloc -> main
    //   section 1 .text.main    reloc -> used
    //   section 2 .text.used    no reloc
    //   section 3 .text.unused  no reloc, no inbound reference
    // Symbols index the symbol table: 0 _start, 1 main, 2 used, 3 unused.
    var start_relocs = [_]elf.Reloc{.{ .offset = 0, .symbol = 1, .type = .abs64 }};
    var main_relocs = [_]elf.Reloc{.{ .offset = 0, .symbol = 2, .type = .abs64 }};

    var sections = [_]elf.ObjSection{
        .{ .name = ".text._start", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false, .relocs = &start_relocs },
        .{ .name = ".text.main", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false, .relocs = &main_relocs },
        .{ .name = ".text.used", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false },
        .{ .name = ".text.unused", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false },
    };

    var symbols = [_]elf.ObjSymbol{
        .{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 },
        .{ .name = "main", .value = 0, .defined = true, .local = false, .section_index = 1 },
        .{ .name = "used", .value = 0, .defined = true, .local = false, .section_index = 2 },
        .{ .name = "unused", .value = 0, .defined = true, .local = false, .section_index = 3 },
    };

    const objs = [_]elf.ParsedObject{.{ .arch = .aarch64, .symbols = &symbols, .sections = &sections }};

    // Non-shared. _start -> main -> used are live. unused is dead.
    {
        const live = try computeLiveSections(a, &objs, "_start", false);
        defer a.free(live);
        try std.testing.expectEqual(4, live.len);
        try std.testing.expect(live[0]); // _start
        try std.testing.expect(live[1]); // main
        try std.testing.expect(live[2]); // used
        try std.testing.expect(!live[3]); // unused, dropped
    }

    // Shared. Every global export is a root, so unused is live too.
    {
        const live = try computeLiveSections(a, &objs, "_start", true);
        defer a.free(live);
        try std.testing.expect(live[0]);
        try std.testing.expect(live[1]);
        try std.testing.expect(live[2]);
        try std.testing.expect(live[3]); // unused now exported, kept
    }
}

test "gc handles a reloc to an undefined symbol without crashing" {
    const a = std.testing.allocator;

    // One section .text._start with a reloc to an UNDEFINED external symbol. The pass
    // must not crash and must not mark anything extra live.
    var start_relocs = [_]elf.Reloc{.{ .offset = 0, .symbol = 1, .type = .abs64 }};
    var sections = [_]elf.ObjSection{
        .{ .name = ".text._start", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false, .relocs = &start_relocs },
    };
    var symbols = [_]elf.ObjSymbol{
        .{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 },
        .{ .name = "external", .value = 0, .defined = false, .local = false, .section_index = no_section },
    };
    const objs = [_]elf.ParsedObject{.{ .arch = .aarch64, .symbols = &symbols, .sections = &sections }};

    const live = try computeLiveSections(a, &objs, "_start", false);
    defer a.free(live);
    try std.testing.expectEqual(1, live.len);
    try std.testing.expect(live[0]); // _start only, the undefined reloc adds nothing
}

test "gc marks a cross-object global reference and drops the unreferenced peer section" {
    const a = std.testing.allocator;

    // Object A holds `_start`. Its reloc names `helper`, a symbol that is UNDEFINED inside
    // A (an inbound global) but DEFINED in object B. GC must resolve `helper` by global
    // name to B's section and keep it, exactly as the linker's `findSymbol` does. B also
    // owns `.text.bonus`, which nothing references, so it must stay dead.
    var a_start_relocs = [_]elf.Reloc{.{ .offset = 0, .symbol = 1, .type = .abs64 }};
    var a_sections = [_]elf.ObjSection{
        .{ .name = ".text._start", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false, .relocs = &a_start_relocs },
    };
    var a_symbols = [_]elf.ObjSymbol{
        .{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 },
        .{ .name = "helper", .value = 0, .defined = false, .local = false, .section_index = no_section },
    };

    var b_sections = [_]elf.ObjSection{
        .{ .name = ".text.helper", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false },
        .{ .name = ".text.bonus", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false },
    };
    var b_symbols = [_]elf.ObjSymbol{
        .{ .name = "helper", .value = 0, .defined = true, .local = false, .section_index = 0 },
    };

    const objs = [_]elf.ParsedObject{
        .{ .arch = .aarch64, .symbols = &a_symbols, .sections = &a_sections },
        .{ .arch = .aarch64, .symbols = &b_symbols, .sections = &b_sections },
    };

    const live = try computeLiveSections(a, &objs, "_start", false);
    defer a.free(live);
    // Mask layout: A.section0 = 0, B.section0 = 1, B.section1 = 2.
    try std.testing.expectEqual(3, live.len);
    try std.testing.expect(live[0]); // A ._start, the seeded root
    try std.testing.expect(live[1]); // B .text.helper, reached by the cross-object global
    try std.testing.expect(!live[2]); // B .text.bonus, unreferenced, dropped
}

test "gc keeps the section of a local-symbol reloc target" {
    const a = std.testing.allocator;

    // `_start` relocs to a DEFINED LOCAL symbol (a `.str.0` string literal in the same
    // object). Its name is not a global, so GC must fall back to the symbol's own section
    // and keep it. This is the `char *p = "x";` pattern.
    var start_relocs = [_]elf.Reloc{.{ .offset = 0, .symbol = 1, .type = .abs64 }};
    var sections = [_]elf.ObjSection{
        .{ .name = ".text._start", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false, .relocs = &start_relocs },
        .{ .name = ".rodata.str0", .flags = 0, .bytes = &.{ 'x', 0, 0, 0 }, .size = 4, .is_nobits = false },
    };
    var symbols = [_]elf.ObjSymbol{
        .{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 },
        .{ .name = ".str.0", .value = 0, .defined = true, .local = true, .section_index = 1 },
    };
    const objs = [_]elf.ParsedObject{.{ .arch = .aarch64, .symbols = &symbols, .sections = &sections }};

    const live = try computeLiveSections(a, &objs, "_start", false);
    defer a.free(live);
    try std.testing.expectEqual(2, live.len);
    try std.testing.expect(live[0]); // _start
    try std.testing.expect(live[1]); // the local string literal's section, kept
}

test "gc conservatively keeps an object on a defined-but-unresolvable reloc target" {
    const a = std.testing.allocator;

    // `_start` relocs to a DEFINED LOCAL symbol whose `section_index` is out of range. GC
    // must not crash and must not drop. It keeps the whole owning object instead.
    var start_relocs = [_]elf.Reloc{.{ .offset = 0, .symbol = 1, .type = .abs64 }};
    var sections = [_]elf.ObjSection{
        .{ .name = ".text._start", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false, .relocs = &start_relocs },
        .{ .name = ".text.other", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false },
    };
    var symbols = [_]elf.ObjSymbol{
        .{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 },
        .{ .name = "weird", .value = 0, .defined = true, .local = true, .section_index = 99 },
    };
    const objs = [_]elf.ParsedObject{.{ .arch = .aarch64, .symbols = &symbols, .sections = &sections }};

    const live = try computeLiveSections(a, &objs, "_start", false);
    defer a.free(live);
    try std.testing.expectEqual(2, live.len);
    try std.testing.expect(live[0]); // _start
    try std.testing.expect(live[1]); // .text.other, kept by the conservative fallback
}

test "gc keeps every section when the entry root can not be seeded (static link)" {
    const a = std.testing.allocator;

    // No symbol named `_start` exists, so the entry root can not be seeded. A static link
    // must NOT drop everything. GC returns an all-true mask instead.
    var sections = [_]elf.ObjSection{
        .{ .name = ".text.a", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false },
        .{ .name = ".text.b", .flags = 0, .bytes = &.{ 0, 0, 0, 0 }, .size = 4, .is_nobits = false },
    };
    var symbols = [_]elf.ObjSymbol{
        .{ .name = "foo", .value = 0, .defined = true, .local = false, .section_index = 0 },
    };
    const objs = [_]elf.ParsedObject{.{ .arch = .aarch64, .symbols = &symbols, .sections = &sections }};

    // Entry not found, non-shared. Keep everything.
    {
        const live = try computeLiveSections(a, &objs, "_start", false);
        defer a.free(live);
        try std.testing.expectEqual(2, live.len);
        try std.testing.expect(live[0]);
        try std.testing.expect(live[1]);
    }

    // Entry present by name but with an unresolvable section. Still keep everything.
    {
        var bad_syms = [_]elf.ObjSymbol{
            .{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = no_section },
        };
        const bad_objs = [_]elf.ParsedObject{.{ .arch = .aarch64, .symbols = &bad_syms, .sections = &sections }};
        const live = try computeLiveSections(a, &bad_objs, "_start", false);
        defer a.free(live);
        try std.testing.expect(live[0]);
        try std.testing.expect(live[1]);
    }
}
