//! The generic, script-driven layout engine. It consumes a parsed linker
//! script (`script.Script`, the AST) plus a set of parsed relocatable objects
//! (`elf.ParsedObject`) and produces an `elf.Placement`. This is the same address-assignment
//! interface the default (non-script) `arch/<a>.zig` `computeDefaultPlacement` produces,
//! so the same per-arch `applyRelocs` patches the result. A linker script therefore
//! controls section placement, addresses, and boundary symbols while every downstream
//! stage, relocation and ELF emission, stays unchanged.
//!
//! The core engine holds a single default region with VMA equal to LMA. It walks the
//! script's commands in order, maintaining a location counter `.` and a script-symbol
//! environment, gathers each output section's `*(...)` input patterns across every object
//! in input order, assembles one loadable image segment, and resolves both the objects'
//! own symbols and the script-defined symbols into the placement's symbol table. MEMORY
//! regions and the LMA/AT split (ORIGIN/LENGTH, `>region`, `AT>`) are not yet supported.
//! They surface as `error.ScriptUndefinedRegion`.
//!
//! Depends only on `std`, `elf.zig`, and `script.zig`, keeping `vulcan-link` std-only.

const std = @import("std");
const elf = @import("elf.zig");
const script = @import("script.zig");

const ObjSection = elf.ObjSection;
const ParsedObject = elf.ParsedObject;
const ResolvedSymbol = elf.ResolvedSymbol;
const Segment = elf.Segment;
const SecPlace = elf.SecPlace;
const Placement = elf.Placement;
const Arch = elf.Arch;
const alignUp = elf.alignUp;
const findSymbol = elf.findSymbol;
const not_placed = elf.not_placed;

/// The broad placement class of one section (text/rodata/data/bss), used for the default
/// alignment and bss detection. The script layout places sections by NAME, not class, so
/// this only classifies for those two decisions.
const Class = enum { text, rodata, data, bss };

/// Every failure `computeScriptPlacement`/`linkInputsScript` can report: the generic
/// linker errors (`elf.Error`, including `DuplicateSymbol`/`UndefinedSymbol`), script-parse
/// errors (so a caller can parse-then-lay-out through one error set), plus the
/// layout-specific ones. `ScriptRegionOverflow`/`ScriptUndefinedRegion` are the
/// MEMORY-region diagnostics for a future feature (ORIGIN/LENGTH/`>region` are not wired
/// yet, so they fail closed here); `ScriptUndefinedSymbol` is an expression naming an
/// undefined script symbol (or an `ENTRY(sym)` that resolves to nothing);
/// `ScriptDivByZero` guards a `/` or `%` by zero in a script expression.
pub const ScriptError = elf.Error || script.ParseError || error{
    ScriptRegionOverflow,
    ScriptUndefinedRegion,
    ScriptUndefinedSymbol,
    ScriptDivByZero,
};

/// One (object, section) the layout has placed. `vaddr` is its runtime address (relocs,
/// symbols, and `p_vaddr` all use this); `laddr` is its load address (the file byte
/// placement and `p_paddr` use this). With no `AT`/`AT>` they are equal. An
/// `AT(expr)`/`AT>region` splits them by a constant per-section delta. `flags` is the ELF
/// `p_flags` the enclosing segment should carry (from its `>region`, else the default 7).
/// `si` is the index into the owning object's `sections`, and `class` its broad placement
/// class (used for byte copying and bss detection).
const Placed = struct {
    oi: usize,
    si: usize,
    class: Class,
    vaddr: u64,
    laddr: u64,
    len: u64,
    flags: u8,
};

/// One MEMORY region resolved to concrete bounds, with a live allocation cursor. `flags`
/// carries the region's r/w/x attributes (mapped to ELF `p_flags` for any segment placed in
/// it). `cursor` starts at `origin` and advances as `>region` sections allocate VMA (or
/// `AT>region` sections allocate LMA) from it.
const RegionInfo = struct {
    origin: u64,
    length: u64,
    cursor: u64,
    flags: script.RegionFlags,
};

/// ELF `p_flags` (R=4, W=2, X=1) for a region's r/w/x attributes. A region that declares no
/// access flags defaults to R|W|X (7), matching the default-segment flags, so a
/// flag-less region still yields a runnable segment.
fn regionPFlags(f: script.RegionFlags) u8 {
    var v: u8 = 0;
    if (f.r) v |= 4;
    if (f.w) v |= 2;
    if (f.x) v |= 1;
    return if (v == 0) 7 else v;
}

/// The broad placement class of one `ObjSection`, from its flags: a NOBITS section is
/// bss, an executable section text, a writable section data, else read-only. This is the
/// section-model twin of `elf.sectionClass` (which needs the raw `sh_type`); here the
/// NOBITS case is already flagged by `is_nobits`.
fn classForSection(sec: *const ObjSection) Class {
    if (sec.is_nobits) return .bss;
    if ((sec.flags & elf.SHF_EXECINSTR) != 0) return .text;
    if ((sec.flags & elf.SHF_WRITE) != 0) return .data;
    return .rodata;
}

/// The bytes one section contributes to the file image. A NOBITS (bss) section has none
/// (it is memory-only), and its `bytes` is already empty.
fn sectionBytes(sec: *const ObjSection) []const u8 {
    return sec.bytes;
}

/// The in-memory length of a section (its `sh_size`: bytes for a progbits section, the
/// reserved size for bss).
fn sectionLen(sec: *const ObjSection) u64 {
    return sec.size;
}

/// True when a section has something to place (a non-zero in-memory size).
fn sectionPresent(sec: *const ObjSection) bool {
    return sec.size > 0;
}

/// The default alignment applied to the location counter before a section is placed:
/// an instruction (text-class) section is 4-byte aligned, everything else 8-byte aligned.
/// The class comes from the section flags, so the alignment matches the old per-class rule.
fn defaultAlign(sec: *const ObjSection) u64 {
    return if (classForSection(sec) == .text) 4 else 8;
}

/// Round `v` up to a multiple of `a`, tolerating `a == 0` (a no-op) so a script's
/// `ALIGN(., 0)` cannot trap.
fn alignUpTo(v: u64, a: u64) u64 {
    if (a == 0) return v;
    return alignUp(v, a);
}

/// Sort predicate: order placed sections by ascending VMA (for segment grouping).
fn placedVaddrLess(_: void, a: Placed, b: Placed) bool {
    return a.vaddr < b.vaddr;
}

/// The evaluation context threaded through `eval`: the current location counter, the
/// script-defined symbol environment, and the MEMORY region table (for ORIGIN/LENGTH).
const EvalCtx = struct {
    dot: u64,
    env: *const std.StringHashMapUnmanaged(u64),
    regions: *const std.StringHashMapUnmanaged(RegionInfo),

    fn eval(self: *const EvalCtx, expr: *const script.Expr) ScriptError!u64 {
        return switch (expr.*) {
            .number => |n| n,
            .dot => self.dot,
            .symbol => |name| self.env.get(name) orelse error.ScriptUndefinedSymbol,
            .binary => |bin| blk: {
                const lhs = try self.eval(bin.lhs);
                const rhs = try self.eval(bin.rhs);
                break :blk switch (bin.op) {
                    .add => lhs +% rhs,
                    .sub => lhs -% rhs,
                    .mul => lhs *% rhs,
                    .div => if (rhs == 0) error.ScriptDivByZero else lhs / rhs,
                    .mod => if (rhs == 0) error.ScriptDivByZero else lhs % rhs,
                    .band => lhs & rhs,
                    .bor => lhs | rhs,
                    .shl => if (rhs >= 64) 0 else lhs << @intCast(rhs),
                    .shr => if (rhs >= 64) 0 else lhs >> @intCast(rhs),
                };
            },
            .align_ => |al| alignUpTo(try self.eval(al.value), try self.eval(al.boundary)),
            // ORIGIN(name)/LENGTH(name) resolve against the MEMORY region table; an unknown
            // region name fails closed. A script with no MEMORY block has an empty table, so
            // any ORIGIN/LENGTH there still errors, preserving the original behavior.
            .origin => |name| if (self.regions.get(name)) |r| r.origin else error.ScriptUndefinedRegion,
            .length => |name| if (self.regions.get(name)) |r| r.length else error.ScriptUndefinedRegion,
        };
    }
};

/// Assign addresses for a set of parsed objects under the control of a linker script,
/// producing a `Placement` the per-arch `applyRelocs` then patches. This handles MEMORY
/// regions (ORIGIN/LENGTH + `>region` VMA allocation), the LMA/AT split (`AT(expr)`/`AT>region`), and
/// multiple `PT_LOAD` segments grouped by (VMA-to-LMA delta, flags).
///
/// The algorithm: a location counter `dot` (starting 0), a script-symbol environment, and a
/// MEMORY region table (name -> {origin, length, live cursor, flags}) are threaded through
/// the script's commands in source order. `. = expr` sets the counter; `sym = expr` /
/// `PROVIDE(sym = expr)` defines a script symbol at the counter's current value; an output
/// section starts its VMA at an explicit address, else its `>region`'s cursor, else the
/// counter, then gathers each `*(pattern)` input spec's matching section (in input order,
/// each placed at most once) at the aligned counter, advancing it (and the region cursor).
/// `AT(expr)`/`AT>region` gives the section a load address distinct from its VMA by a
/// constant delta (the LMA drives file byte placement + `p_paddr`; the VMA drives relocs,
/// symbols, and `p_vaddr`). After the walk, the placed sections are grouped by contiguous
/// (delta, flags) into loadable segments (bss contributing memory size only), the
/// per-(object, section) `section_places` map is filled, and both the objects' own defined
/// symbols and the script-defined symbols are resolved into the symbol table (script symbols
/// override an object symbol of the same name). `ENTRY(sym)` sets the entry (0 if none). The
/// caller owns the returned placement (`placement.deinit`); it does not own `parsed`/`scr`.
pub fn computeScriptPlacement(allocator: std.mem.Allocator, parsed: []ParsedObject, scr: *const script.Script, arch: Arch) ScriptError!Placement {
    _ = arch; // The default alignment is arch-independent; kept for the region-align seam.
    const nobj = parsed.len;

    // The script-defined symbol environment. Keys borrow the script arena's duped names
    // (the script outlives this call), so nothing here needs its own duping.
    var env: std.StringHashMapUnmanaged(u64) = .empty;
    defer env.deinit(allocator);

    // The MEMORY region table, keyed by the script arena's duped region names. Origins and
    // lengths are evaluated in source order, so a later region may reference an earlier one
    // via ORIGIN/LENGTH; the env is (normally) still empty here since MEMORY precedes
    // SECTIONS.
    var regions: std.StringHashMapUnmanaged(RegionInfo) = .empty;
    defer regions.deinit(allocator);
    for (scr.memory) |region| {
        var ctx = EvalCtx{ .dot = 0, .env = &env, .regions = &regions };
        const origin = try ctx.eval(&region.origin);
        const length = try ctx.eval(&region.length);
        try regions.put(allocator, region.name, .{ .origin = origin, .length = length, .cursor = origin, .flags = region.flags });
    }

    // The placed (object, section) list.
    var placed_list: std.ArrayList(Placed) = .empty;
    defer placed_list.deinit(allocator);

    // A per-object base offset into the flat per-section arrays (also returned as the
    // placement's `section_place_base`). Entry `nobj` holds the total section count.
    var section_place_base = try allocator.alloc(usize, nobj + 1);
    errdefer allocator.free(section_place_base);
    var total_secs: usize = 0;
    for (0..nobj) |oi| {
        section_place_base[oi] = total_secs;
        total_secs += parsed[oi].sections.len;
    }
    section_place_base[nobj] = total_secs;

    // A per-(object, section) placed flag: a section is gathered at most once, even when
    // it matches more than one input pattern.
    var sec_placed = try allocator.alloc(bool, total_secs);
    defer allocator.free(sec_placed);
    @memset(sec_placed, false);

    var dot: u64 = 0;

    for (scr.commands) |cmd| {
        switch (cmd) {
            .set_dot => |e| {
                var ctx = EvalCtx{ .dot = dot, .env = &env, .regions = &regions };
                dot = try ctx.eval(&e);
            },
            .assign => |a| {
                var ctx = EvalCtx{ .dot = dot, .env = &env, .regions = &regions };
                const v = try ctx.eval(&a.value);
                try env.put(allocator, a.name, v);
            },
            .output => |sec| {
                // The section's VMA start: an explicit address, else its `>region` cursor,
                // else the running counter (the default behavior).
                if (sec.vma) |e| {
                    var ctx = EvalCtx{ .dot = dot, .env = &env, .regions = &regions };
                    dot = try ctx.eval(&e);
                } else if (sec.region) |rname| {
                    const ri = regions.getPtr(rname) orelse return error.ScriptUndefinedRegion;
                    dot = ri.cursor;
                }

                // The ELF `p_flags` for this section's segment: from its `>region`, else the
                // default R|W|X. (Resolving `sec.region` already errored above if unknown.)
                const section_flags: u8 = if (sec.region) |rname| regionPFlags(regions.getPtr(rname).?.flags) else 7;

                // The VMA-to-LMA delta for this section, computed once at its first placement
                // (an `AT(expr)`/`AT>region` shifts the load address by a constant amount;
                // absent, LMA == VMA). `at_region`, when set, is the `AT>region` whose cursor
                // this section's LMA is allocated from (advanced after the section is placed).
                var lma_delta: u64 = 0;
                var section_started = false;
                var at_region: ?*RegionInfo = null;

                for (sec.body) |scmd| switch (scmd) {
                    .input => |spec| {
                        for (spec.sections) |pat| {
                            for (parsed, 0..) |*obj, oi| {
                                for (obj.sections, 0..) |*isec, si| {
                                    if (!sectionMatchesPattern(pat, isec)) continue;
                                    const gi = section_place_base[oi] + si;
                                    if (sec_placed[gi]) continue;
                                    if (!sectionPresent(isec)) continue;
                                    const class = classForSection(isec);
                                    dot = alignUpTo(dot, defaultAlign(isec));
                                    if (!section_started) {
                                        section_started = true;
                                        if (sec.lma) |lma| {
                                            var lma_base: u64 = undefined;
                                            switch (lma) {
                                                .addr => |ae| {
                                                    var ctx = EvalCtx{ .dot = dot, .env = &env, .regions = &regions };
                                                    lma_base = try ctx.eval(&ae);
                                                },
                                                .region => |lrn| {
                                                    const lri = regions.getPtr(lrn) orelse return error.ScriptUndefinedRegion;
                                                    lma_base = alignUpTo(lri.cursor, defaultAlign(isec));
                                                    at_region = lri;
                                                },
                                            }
                                            lma_delta = lma_base -% dot;
                                        }
                                    }
                                    const len = sectionLen(isec);
                                    try placed_list.append(allocator, .{ .oi = oi, .si = si, .class = class, .vaddr = dot, .laddr = dot +% lma_delta, .len = len, .flags = section_flags });
                                    sec_placed[gi] = true;
                                    dot += len;
                                }
                            }
                        }
                    },
                    .assign => |a| {
                        var ctx = EvalCtx{ .dot = dot, .env = &env, .regions = &regions };
                        const v = try ctx.eval(&a.value);
                        try env.put(allocator, a.name, v);
                    },
                    .set_dot => |e| {
                        var ctx = EvalCtx{ .dot = dot, .env = &env, .regions = &regions };
                        dot = try ctx.eval(&e);
                    },
                };

                // Persist the region cursors past this section and enforce the region bounds.
                // The VMA region advances to `dot`; the `AT>region` advances to this section's
                // LMA end (`dot + delta`, since the delta is constant across the section).
                if (sec.region) |rname| {
                    const ri = regions.getPtr(rname).?;
                    ri.cursor = dot;
                    if (dot > ri.origin +% ri.length) return error.ScriptRegionOverflow;
                }
                if (at_region) |lri| {
                    const lma_end = dot +% lma_delta;
                    lri.cursor = lma_end;
                    if (lma_end > lri.origin +% lri.length) return error.ScriptRegionOverflow;
                }
            },
        }
    }

    // Group the placed sections into loadable segments. Sort by VMA, then start a new
    // segment at every change of (LMA-VMA delta, flags). Sections sharing both stay in one
    // segment whose span covers any alignment gap between them (zero-filled), reproducing
    // the original single-segment image. A different delta (an `AT`/LMA split) or different
    // flags (a different region) begins a new segment. Two same-(delta, flags) regions with
    // a VMA gap would merge into one gap-spanning segment, a known limit. Distinct region
    // flags keep them apart in practice.
    const placements = placed_list.items;
    std.sort.pdq(Placed, placements, {}, placedVaddrLess);

    var seg_of_placement = try allocator.alloc(u32, placements.len);
    defer allocator.free(seg_of_placement);
    var nseg: u32 = 0;
    {
        var i: usize = 0;
        while (i < placements.len) {
            const delta = placements[i].laddr -% placements[i].vaddr;
            const flags = placements[i].flags;
            var j = i;
            while (j < placements.len and (placements[j].laddr -% placements[j].vaddr) == delta and placements[j].flags == flags) : (j += 1) {
                seg_of_placement[j] = nseg;
            }
            nseg += 1;
            i = j;
        }
    }

    // Per-group extents: `gvma`/`glma` the group's lowest VMA/LMA, `gfile_end` the highest
    // end of a byte-carrying section, `gmem_end` the highest end including bss.
    const seg_count: usize = if (nseg == 0) 1 else nseg;
    var gvma = try allocator.alloc(u64, seg_count);
    defer allocator.free(gvma);
    var glma = try allocator.alloc(u64, seg_count);
    defer allocator.free(glma);
    var gfile_end = try allocator.alloc(u64, seg_count);
    defer allocator.free(gfile_end);
    var gmem_end = try allocator.alloc(u64, seg_count);
    defer allocator.free(gmem_end);
    var gflags = try allocator.alloc(u8, seg_count);
    defer allocator.free(gflags);
    @memset(gvma, std.math.maxInt(u64));
    @memset(glma, std.math.maxInt(u64));
    @memset(gfile_end, 0);
    @memset(gmem_end, 0);
    @memset(gflags, 7);
    for (placements, 0..) |p, idx| {
        const g = seg_of_placement[idx];
        if (p.vaddr < gvma[g]) gvma[g] = p.vaddr;
        if (p.laddr < glma[g]) glma[g] = p.laddr;
        const end = p.vaddr + p.len;
        if (end > gmem_end[g]) gmem_end[g] = end;
        if (p.class != .bss and end > gfile_end[g]) gfile_end[g] = end;
        gflags[g] = p.flags;
    }
    // An empty placement set: one empty segment at 0 (matching the historical empty image).
    if (nseg == 0) {
        gvma[0] = 0;
        glma[0] = 0;
    }

    // Build one `Segment` per group: byte-carrying sections copied to (vaddr - gvma), bss
    // contributing only to `memsz`. `built` tracks how many segments already own their
    // bytes so the errdefer frees exactly those on a later failure.
    var segments = try allocator.alloc(Segment, seg_count);
    var built: usize = 0;
    errdefer {
        for (segments[0..built]) |s| allocator.free(s.bytes);
        allocator.free(segments);
    }
    for (0..seg_count) |g| {
        const file_end = if (gfile_end[g] > gvma[g]) gfile_end[g] else gvma[g];
        const filesz: usize = @intCast(file_end - gvma[g]);
        const memsz: u64 = gmem_end[g] - gvma[g];
        var bytes = try allocator.alloc(u8, filesz);
        @memset(bytes, 0);
        for (placements, 0..) |p, idx| {
            if (seg_of_placement[idx] != g) continue;
            if (p.class == .bss) continue;
            const src = sectionBytes(&parsed[p.oi].sections[p.si]);
            if (src.len == 0) continue;
            const off: usize = @intCast(p.vaddr - gvma[g]);
            @memcpy(bytes[off..][0..src.len], src);
        }
        segments[g] = .{ .vaddr = gvma[g], .paddr = glma[g], .bytes = bytes, .memsz = memsz, .flags = gflags[g] };
        built = g + 1;
    }

    // Per-(object, section) placement: each placed slot maps to its group's segment at
    // (vaddr - gvma); absent slots get the not-placed sentinel. The precise per-section
    // `section_places` is keyed by section index, so the appliers read a section's place
    // uniformly on the script path too.
    var section_places = try allocator.alloc(SecPlace, total_secs);
    errdefer allocator.free(section_places);
    for (section_places) |*pl| pl.* = .{ .vaddr = 0, .seg = not_placed, .seg_off = 0 };
    for (placements, 0..) |p, idx| {
        const g = seg_of_placement[idx];
        const sp = SecPlace{ .vaddr = p.vaddr, .seg = g, .seg_off = p.vaddr - gvma[g] };
        section_places[section_place_base[p.oi] + p.si] = sp;
    }

    // The resolved symbol table. Script-defined symbols are authoritative: an object symbol
    // whose name a script assignment also defines is skipped here (the script value is
    // injected below), so `findSymbol` (and thus reloc resolution) sees the script's value.
    // This is the SAME rule `ENTRY` uses (`env.get(name) orelse findSymbol(...)`). It resolves
    // an earlier inconsistency where reloc resolution preferred the object while `ENTRY`
    // preferred the script. Duplicate DEFINED object symbols across objects (with no script
    // override) remain an error.
    var symbols: std.ArrayList(ResolvedSymbol) = .empty;
    errdefer {
        for (symbols.items) |s| allocator.free(s.name);
        symbols.deinit(allocator);
    }
    for (parsed, 0..) |*obj, oi| {
        for (obj.symbols) |sym| {
            // Skip undefined/local/anonymous/ABS symbols and any whose section the script
            // did not place (nothing to anchor an address to). Resolve through the symbol's
            // OWN section (by `section_index`): its placed VMA plus the in-section value.
            // With one section per class this is the old class VMA; with two same-class
            // sections a symbol now resolves to the section that defines it.
            if (!sym.defined or sym.local or sym.name.len == 0 or sym.section_index == std.math.maxInt(u32)) continue;
            const sp = section_places[section_place_base[oi] + sym.section_index];
            if (sp.seg == not_placed) continue;
            if (env.contains(sym.name)) continue; // a script symbol of this name overrides it
            if (findSymbol(symbols.items, sym.name) != null) return error.DuplicateSymbol;
            const name = try allocator.dupe(u8, sym.name);
            errdefer allocator.free(name);
            const is_exec = (obj.sections[sym.section_index].flags & elf.SHF_EXECINSTR) != 0;
            try symbols.append(allocator, .{ .name = name, .address = sp.vaddr + sym.value, .is_exec = is_exec });
        }
    }
    var it = env.iterator();
    while (it.next()) |entry| {
        const name = try allocator.dupe(u8, entry.key_ptr.*);
        errdefer allocator.free(name);
        try symbols.append(allocator, .{ .name = name, .address = entry.value_ptr.* });
    }

    // The entry: the address of `ENTRY(sym)` (a script symbol, else an object symbol), or
    // 0 when the script names none.
    var entry_addr: u64 = 0;
    if (scr.entry) |name| {
        entry_addr = env.get(name) orelse findSymbol(symbols.items, name) orelse return error.ScriptUndefinedSymbol;
    }

    return .{
        .segments = segments,
        .symbols = try symbols.toOwnedSlice(allocator),
        .entry = entry_addr,
        .section_places = section_places,
        .section_place_base = section_place_base,
    };
}

/// Decide whether a linker-script input pattern selects one object section, matching on
/// the section NAME (so an arbitrary named section places, not only the four standard
/// families). A pattern with a trailing `*` matches any name that starts with the text
/// before the `*` (so `.text*` selects `.text`, `.text.startup`, and so on); a pattern
/// with no `*` matches the name exactly. The special `COMMON` pseudo-section selects any
/// bss-class section (tentative definitions), as before. The four standard families
/// (`.text*`/`.rodata*`/`.data*`/`.bss*`) stay a subset of this rule, so an existing
/// script places byte-for-byte as before.
pub fn sectionMatchesPattern(pat: []const u8, sec: *const ObjSection) bool {
    if (std.mem.eql(u8, pat, "COMMON")) return classForSection(sec) == .bss;
    if (std.mem.endsWith(u8, pat, "*")) return std.mem.startsWith(u8, sec.name, pat[0 .. pat.len - 1]);
    return std.mem.eql(u8, sec.name, pat);
}

test "sectionMatchesPattern matches by section name, arbitrary sections, and COMMON" {
    const A = elf.SHF_ALLOC;
    const X = elf.SHF_EXECINSTR;
    const W = elf.SHF_WRITE;
    const text = ObjSection{ .name = ".text", .flags = A | X, .size = 4, .is_nobits = false };
    const text_startup = ObjSection{ .name = ".text.startup", .flags = A | X, .size = 4, .is_nobits = false };
    const rodata = ObjSection{ .name = ".rodata", .flags = A, .size = 4, .is_nobits = false };
    const rodata_str = ObjSection{ .name = ".rodata.str1.1", .flags = A, .size = 4, .is_nobits = false };
    const data = ObjSection{ .name = ".data", .flags = A | W, .size = 4, .is_nobits = false };
    const bss = ObjSection{ .name = ".bss", .flags = A | W, .size = 4, .is_nobits = true };
    const init_array = ObjSection{ .name = ".init_array", .flags = A | W, .size = 8, .is_nobits = false };
    const custom = ObjSection{ .name = ".mycustom", .flags = A, .size = 4, .is_nobits = false };

    // The standard families still place through their wildcard patterns.
    try std.testing.expect(sectionMatchesPattern(".text*", &text));
    try std.testing.expect(sectionMatchesPattern(".text*", &text_startup));
    try std.testing.expect(sectionMatchesPattern(".rodata*", &rodata));
    try std.testing.expect(sectionMatchesPattern(".rodata*", &rodata_str));
    try std.testing.expect(sectionMatchesPattern(".data*", &data));
    try std.testing.expect(sectionMatchesPattern(".bss*", &bss));
    // A no-wildcard pattern is an exact name match now, not a family merge.
    try std.testing.expect(sectionMatchesPattern(".text", &text));
    try std.testing.expect(!sectionMatchesPattern(".text", &text_startup));
    // COMMON selects any bss-class section, and only those.
    try std.testing.expect(sectionMatchesPattern("COMMON", &bss));
    try std.testing.expect(!sectionMatchesPattern("COMMON", &text));
    // Arbitrary named sections now MATCH their own pattern (the old mapper dropped them).
    try std.testing.expect(sectionMatchesPattern(".init_array", &init_array));
    try std.testing.expect(sectionMatchesPattern(".init_array*", &init_array));
    try std.testing.expect(sectionMatchesPattern(".mycustom", &custom));
    try std.testing.expect(!sectionMatchesPattern(".text*", &custom));
}

test "computeScriptPlacement: location counter, output sections, ALIGN, and boundary symbols" {
    const allocator = std.testing.allocator;

    // Two objects, each with a `.text` and one with a `.data` blob, exercised through a
    // representative script.
    var t0 = [_]u8{ 1, 2, 3, 4 }; // obj0 .text (4 bytes)
    var t1 = [_]u8{ 5, 6, 7, 8, 9, 10 }; // obj1 .text (6 bytes)
    var d1 = [_]u8{ 0xAA, 0xBB, 0xCC, 0xDD }; // obj1 .data (4 bytes)

    var syms0 = [_]elf.ObjSymbol{.{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 }};
    var syms1 = [_]elf.ObjSymbol{
        .{ .name = "main", .value = 0, .defined = true, .local = false, .section_index = 0 },
        .{ .name = "g", .value = 0, .defined = true, .local = false, .section_index = 1 },
    };
    var secs0 = [_]elf.ObjSection{
        .{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t0, .size = t0.len, .is_nobits = false },
    };
    var secs1 = [_]elf.ObjSection{
        .{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t1, .size = t1.len, .is_nobits = false },
        .{ .name = ".data", .flags = elf.SHF_ALLOC | elf.SHF_WRITE, .bytes = &d1, .size = d1.len, .is_nobits = false },
    };
    var parsed = [_]ParsedObject{
        .{ .arch = .aarch64, .symbols = &syms0, .sections = &secs0 },
        .{ .arch = .aarch64, .symbols = &syms1, .sections = &secs1 },
    };

    const src =
        \\ENTRY(_start)
        \\SECTIONS {
        \\  . = 0x400000;
        \\  .text : { *(.text*) }
        \\  . = ALIGN(16);
        \\  .data : { *(.data*) }
        \\  __data_end = .;
        \\}
    ;
    var scr = try script.parse(allocator, src, null);
    defer scr.deinit();

    var placement = try computeScriptPlacement(allocator, &parsed, &scr, .aarch64);
    defer placement.deinit(allocator);

    // One segment starting at the script base.
    try std.testing.expectEqual(@as(usize, 1), placement.segments.len);
    try std.testing.expectEqual(@as(u64, 0x400000), placement.segments[0].vaddr);

    // `_start` at the base, `main` right after obj0's 4-byte text (4-aligned already).
    try std.testing.expectEqual(@as(u64, 0x400000), findSymbol(placement.symbols, "_start").?);
    try std.testing.expectEqual(@as(u64, 0x400004), findSymbol(placement.symbols, "main").?);

    // text ends at 0x40000A (4 + 6); ALIGN(16) rounds the counter to 0x400010, where
    // `.data`'s `g` lands.
    try std.testing.expectEqual(@as(u64, 0x400010), findSymbol(placement.symbols, "g").?);
    // `__data_end` (a script symbol) sits just past the 4-byte `.data`.
    try std.testing.expectEqual(@as(u64, 0x400014), findSymbol(placement.symbols, "__data_end").?);

    // Entry resolves to `_start`.
    try std.testing.expectEqual(@as(u64, 0x400000), placement.entry);

    // The image bytes: obj0/obj1 text laid contiguously, obj1 data at offset 0x10.
    const bytes = placement.segments[0].bytes;
    try std.testing.expectEqual(@as(usize, 0x14), bytes.len);
    try std.testing.expectEqualSlices(u8, &t0, bytes[0..4]);
    try std.testing.expectEqualSlices(u8, &t1, bytes[4..10]);
    try std.testing.expectEqualSlices(u8, &d1, bytes[0x10..0x14]);
}

test "computeScriptPlacement: a duplicate defined symbol across objects errors" {
    const allocator = std.testing.allocator;
    var t0 = [_]u8{ 0, 0, 0, 0 };
    var t1 = [_]u8{ 0, 0, 0, 0 };
    var syms0 = [_]elf.ObjSymbol{.{ .name = "dup", .value = 0, .defined = true, .local = false, .section_index = 0 }};
    var syms1 = [_]elf.ObjSymbol{.{ .name = "dup", .value = 0, .defined = true, .local = false, .section_index = 0 }};
    var secs0 = [_]elf.ObjSection{.{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t0, .size = t0.len, .is_nobits = false }};
    var secs1 = [_]elf.ObjSection{.{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t1, .size = t1.len, .is_nobits = false }};
    var parsed = [_]ParsedObject{
        .{ .arch = .aarch64, .symbols = &syms0, .sections = &secs0 },
        .{ .arch = .aarch64, .symbols = &syms1, .sections = &secs1 },
    };
    var scr = try script.parse(allocator, "SECTIONS { . = 0x1000; .text : { *(.text*) } }", null);
    defer scr.deinit();
    try std.testing.expectError(error.DuplicateSymbol, computeScriptPlacement(allocator, &parsed, &scr, .aarch64));
}

test "computeScriptPlacement: ENTRY naming an unresolved symbol errors" {
    const allocator = std.testing.allocator;
    var t0 = [_]u8{ 0, 0, 0, 0 };
    var syms0 = [_]elf.ObjSymbol{.{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 }};
    var secs0 = [_]elf.ObjSection{.{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t0, .size = t0.len, .is_nobits = false }};
    var parsed = [_]ParsedObject{.{ .arch = .aarch64, .symbols = &syms0, .sections = &secs0 }};
    var scr = try script.parse(allocator, "ENTRY(nope) SECTIONS { . = 0x1000; .text : { *(.text*) } }", null);
    defer scr.deinit();
    try std.testing.expectError(error.ScriptUndefinedSymbol, computeScriptPlacement(allocator, &parsed, &scr, .aarch64));
}

test "computeScriptPlacement: an ORIGIN of an undefined region (no MEMORY block) fails closed" {
    const allocator = std.testing.allocator;
    var t0 = [_]u8{ 0, 0, 0, 0 };
    var syms0 = [_]elf.ObjSymbol{.{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 }};
    var secs0 = [_]elf.ObjSection{.{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t0, .size = t0.len, .is_nobits = false }};
    var parsed = [_]ParsedObject{.{ .arch = .aarch64, .symbols = &syms0, .sections = &secs0 }};
    var scr = try script.parse(allocator, "SECTIONS { . = ORIGIN(rom); .text : { *(.text*) } }", null);
    defer scr.deinit();
    try std.testing.expectError(error.ScriptUndefinedRegion, computeScriptPlacement(allocator, &parsed, &scr, .aarch64));
}

test "computeScriptPlacement: MEMORY regions place VMA from a region; AT>region splits LMA into a second segment (p_vaddr != p_paddr)" {
    const allocator = std.testing.allocator;

    // One object with a `.text` blob (8 bytes) and a `.data` blob (4 bytes). `.text >rom`
    // (VMA == LMA in rom), `.data >ram AT>rom` (VMA in ram, LMA loaded from rom).
    var t0 = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var d0 = [_]u8{ 0xAA, 0xBB, 0xCC, 0xDD };
    var syms0 = [_]elf.ObjSymbol{
        .{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 },
        .{ .name = "g", .value = 0, .defined = true, .local = false, .section_index = 1 },
    };
    var secs0 = [_]elf.ObjSection{
        .{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t0, .size = t0.len, .is_nobits = false },
        .{ .name = ".data", .flags = elf.SHF_ALLOC | elf.SHF_WRITE, .bytes = &d0, .size = d0.len, .is_nobits = false },
    };
    var parsed = [_]ParsedObject{.{ .arch = .aarch64, .symbols = &syms0, .sections = &secs0 }};

    const src =
        \\MEMORY {
        \\  rom (rx) : ORIGIN = 0x08000000, LENGTH = 256K
        \\  ram (rwx): ORIGIN = 0x20000000, LENGTH = 64K
        \\}
        \\SECTIONS {
        \\  .text : { *(.text*) } >rom
        \\  .data : { *(.data*) } >ram AT>rom
        \\}
    ;
    var scr = try script.parse(allocator, src, null);
    defer scr.deinit();

    var placement = try computeScriptPlacement(allocator, &parsed, &scr, .aarch64);
    defer placement.deinit(allocator);

    // Two PT_LOADs: a rom-mapped text segment and a ram-mapped (rom-loaded) data segment.
    try std.testing.expectEqual(@as(usize, 2), placement.segments.len);

    // Segment 0: `.text` in rom, VMA == LMA (== paddr), flags = R|X (5).
    const text_seg = placement.segments[0];
    try std.testing.expectEqual(@as(u64, 0x08000000), text_seg.vaddr);
    try std.testing.expectEqual(@as(u64, 0x08000000), text_seg.paddr);
    try std.testing.expectEqual(@as(u8, 5), text_seg.flags);

    // Segment 1: `.data`'s VMA is in ram, but it LOADS from rom (right after `.text`, at
    // rom's cursor 0x08000008). So p_vaddr != p_paddr, p_vaddr in ram, p_paddr in rom.
    const data_seg = placement.segments[1];
    try std.testing.expectEqual(@as(u64, 0x20000000), data_seg.vaddr);
    try std.testing.expectEqual(@as(u64, 0x08000008), data_seg.paddr);
    try std.testing.expect(data_seg.vaddr != data_seg.paddr);
    try std.testing.expectEqual(@as(u8, 7), data_seg.flags); // ram is rwx
    // The data segment carries the `.data` bytes (its file image is LMA-loaded content).
    try std.testing.expectEqualSlices(u8, &d0, data_seg.bytes[0..4]);

    // The `.data` global `g` resolves to its runtime VMA in ram, not its LMA in rom.
    try std.testing.expectEqual(@as(u64, 0x20000000), findSymbol(placement.symbols, "g").?);
    // `_start` resolves to its rom VMA.
    try std.testing.expectEqual(@as(u64, 0x08000000), findSymbol(placement.symbols, "_start").?);
}

test "computeScriptPlacement: a region too small for its sections errors with ScriptRegionOverflow" {
    const allocator = std.testing.allocator;
    // rom LENGTH = 4, but `.text` is 8 bytes: allocating it overruns the region.
    var t0 = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var syms0 = [_]elf.ObjSymbol{.{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 }};
    var secs0 = [_]elf.ObjSection{.{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t0, .size = t0.len, .is_nobits = false }};
    var parsed = [_]ParsedObject{.{ .arch = .aarch64, .symbols = &syms0, .sections = &secs0 }};
    const src =
        \\MEMORY { rom (rx) : ORIGIN = 0x08000000, LENGTH = 4 }
        \\SECTIONS { .text : { *(.text*) } >rom }
    ;
    var scr = try script.parse(allocator, src, null);
    defer scr.deinit();
    try std.testing.expectError(error.ScriptRegionOverflow, computeScriptPlacement(allocator, &parsed, &scr, .aarch64));
}

test "computeScriptPlacement: a script-defined symbol overrides an object symbol of the same name" {
    const allocator = std.testing.allocator;
    // The object defines `g` in `.data`; the script also assigns `g = 0x1234`. The script
    // value wins (consistent with ENTRY), and no DuplicateSymbol is raised.
    var t0 = [_]u8{ 1, 2, 3, 4 };
    var d0 = [_]u8{ 9, 9, 9, 9 };
    var syms0 = [_]elf.ObjSymbol{
        .{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 },
        .{ .name = "g", .value = 0, .defined = true, .local = false, .section_index = 1 },
    };
    var secs0 = [_]elf.ObjSection{
        .{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t0, .size = t0.len, .is_nobits = false },
        .{ .name = ".data", .flags = elf.SHF_ALLOC | elf.SHF_WRITE, .bytes = &d0, .size = d0.len, .is_nobits = false },
    };
    var parsed = [_]ParsedObject{.{ .arch = .aarch64, .symbols = &syms0, .sections = &secs0 }};
    const src =
        \\ENTRY(_start)
        \\SECTIONS {
        \\  . = 0x1000;
        \\  .text : { *(.text*) }
        \\  .data : { *(.data*) }
        \\  g = 0x1234;
        \\}
    ;
    var scr = try script.parse(allocator, src, null);
    defer scr.deinit();
    var placement = try computeScriptPlacement(allocator, &parsed, &scr, .aarch64);
    defer placement.deinit(allocator);
    // The script's `g` (0x1234) wins over the object's `.data`-relative `g`.
    try std.testing.expectEqual(@as(u64, 0x1234), findSymbol(placement.symbols, "g").?);
}

test "computeScriptPlacement: two regions with VMA == LMA yield two runnable segments" {
    const allocator = std.testing.allocator;
    // `.text >rom` and `.data >ram`, no AT: each segment has p_vaddr == p_paddr, but they
    // are distinct regions (distinct flags), so two PT_LOADs result.
    var t0 = [_]u8{ 1, 2, 3, 4 };
    var d0 = [_]u8{ 5, 6, 7, 8 };
    var syms0 = [_]elf.ObjSymbol{
        .{ .name = "_start", .value = 0, .defined = true, .local = false, .section_index = 0 },
        .{ .name = "g", .value = 0, .defined = true, .local = false, .section_index = 1 },
    };
    var secs0 = [_]elf.ObjSection{
        .{ .name = ".text", .flags = elf.SHF_ALLOC | elf.SHF_EXECINSTR, .bytes = &t0, .size = t0.len, .is_nobits = false },
        .{ .name = ".data", .flags = elf.SHF_ALLOC | elf.SHF_WRITE, .bytes = &d0, .size = d0.len, .is_nobits = false },
    };
    var parsed = [_]ParsedObject{.{ .arch = .aarch64, .symbols = &syms0, .sections = &secs0 }};
    const src =
        \\MEMORY {
        \\  rom (rx) : ORIGIN = 0x400000, LENGTH = 64K
        \\  ram (rw) : ORIGIN = 0x500000, LENGTH = 64K
        \\}
        \\SECTIONS {
        \\  .text : { *(.text*) } >rom
        \\  .data : { *(.data*) } >ram
        \\}
    ;
    var scr = try script.parse(allocator, src, null);
    defer scr.deinit();
    var placement = try computeScriptPlacement(allocator, &parsed, &scr, .aarch64);
    defer placement.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), placement.segments.len);
    try std.testing.expectEqual(placement.segments[0].vaddr, placement.segments[0].paddr);
    try std.testing.expectEqual(placement.segments[1].vaddr, placement.segments[1].paddr);
    try std.testing.expectEqual(@as(u64, 0x400000), placement.segments[0].vaddr);
    try std.testing.expectEqual(@as(u64, 0x500000), placement.segments[1].vaddr);
    try std.testing.expectEqual(@as(u64, 0x500000), findSymbol(placement.symbols, "g").?);
}
