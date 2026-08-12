//! x86-64 link backend for the shared static linker. The first real x86-64 object
//! linker: previously `libs/vulcan-target/x86_64` had no real linker at all, only a
//! flat-blob `elf.zig` writer whose test harness hand-resolved cross-object relocs
//! itself (see `x86_64/tests/harness.zig`'s `runModuleData`). This module owns the
//! x86-64 reloc math (`R_X86_64_PC32`/`R_X86_64_PLT32`, both a 4-byte PC-relative
//! displacement patched into a `lea`/`call` site) and the executable parameters.
//! Both reloc kinds resolve identically here: our objects (`x86_64/object.zig`)
//! never emit a real PLT (no shared libraries, no lazy binding), so a `.call`'s
//! PLT32 reloc against a symbol defined in this link set behaves exactly like a
//! PC32 one - `disp32 = symbol_addr + addend - site_addr`, where `object.zig`
//! always emits `addend = -4` (the disp32 field's own width, since the runtime
//! PC-relative read point is the end of the 4-byte field, one past its start).
//! Unlike riscv64, x86-64 has no external resolver/stub/GOT mechanism here: an
//! undefined symbol is always `error.UndefinedSymbol`. Depends only on `std` and
//! the generic `elf.zig`.

const std = @import("std");
const elf = @import("../elf.zig");

const Error = elf.Error;
const SecKind = elf.SecKind;
const ParsedObject = elf.ParsedObject;
const ResolvedSymbol = elf.ResolvedSymbol;
const Image = elf.Image;
const Resolver = elf.Resolver;
const Segment = elf.Segment;
const SecPlace = elf.SecPlace;
const Placement = elf.Placement;
const alignUp = elf.alignUp;
const findSymbol = elf.findSymbol;
const secIndex = elf.secIndex;
const places_per_object = elf.places_per_object;
const not_placed = elf.not_placed;

/// Executable parameters for wrapping an x86-64 image in a static ELF64 (`ET_EXEC`).
/// Only the code range `[code_offset, code_offset+code.len)` is mapped at `base`
/// (headers are not part of the mapped segment, matching riscv64 - x86-64's real
/// ELF loader does not need them mapped to execute). The default `p_flags = 7`
/// (R|W|X) gives one segment that can hold `.text`/`.rodata`/`.data`/`.bss`
/// together, and `writeElfExec` already maps `p_memsz >= code.len` to cover any
/// trailing `.bss`.
pub const exec_params: elf.ExecParams = .{ .e_machine = elf.EM_X86_64 };

/// Assign addresses for a set of parsed x86-64 relocatable objects (the default, no
/// linker-script layout), laid out in the order given with `.text` first at `base`, then
/// `.rodata`, `.data`, and `.bss` (memory only). Produces a single loadable `Segment`
/// (the whole image at `base`, R|W|X), a `places` map recording where each (object,
/// section) landed, and the resolved symbol table (`base + region + value`, rejecting a
/// duplicate, skipping undef/local/ABS symbols). No relocations are applied here (that is
/// `applyRelocs`). x86-64 has no resolver/stub/GOT mechanism, so `resolver` is unused;
/// `compress_text` is unused too (no compressed-instruction form). The caller owns the
/// returned placement; it does not own `parsed`.
pub fn computeDefaultPlacement(allocator: std.mem.Allocator, parsed: []ParsedObject, base: u64, resolver: ?Resolver, compress_text: bool) Error!Placement {
    _ = resolver;
    _ = compress_text;
    const nobj = parsed.len;

    // Pack each object's sections within its region, recording per-object offsets.
    var text_at = try allocator.alloc(u64, nobj);
    defer allocator.free(text_at);
    var rodata_at = try allocator.alloc(u64, nobj);
    defer allocator.free(rodata_at);
    var data_at = try allocator.alloc(u64, nobj);
    defer allocator.free(data_at);
    var bss_at = try allocator.alloc(u64, nobj);
    defer allocator.free(bss_at);
    var text_total: u64 = 0;
    var rodata_total: u64 = 0;
    var data_total: u64 = 0;
    var bss_total: u64 = 0;
    for (0..nobj) |oi| {
        text_at[oi] = alignUp(text_total, 16);
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
    const filesz = data_end;
    const bss_region = alignUp(filesz, 8);
    const memsz = if (bss_total > 0) bss_region + bss_total else filesz;

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

    // The loadable file image: text, rodata, data laid into place (bss is zero,
    // beyond the file image entirely - it lives only in memsz).
    var code = try allocator.alloc(u8, @intCast(filesz));
    errdefer allocator.free(code);
    @memset(code, 0);
    for (0..nobj) |oi| {
        if (parsed[oi].text.len > 0) @memcpy(code[@intCast(text_at[oi])..][0..parsed[oi].text.len], parsed[oi].text);
        if (parsed[oi].rodata.len > 0) @memcpy(code[@intCast(rodata_region + rodata_at[oi])..][0..parsed[oi].rodata.len], parsed[oi].rodata);
        if (parsed[oi].data.len > 0) @memcpy(code[@intCast(data_region + data_at[oi])..][0..parsed[oi].data.len], parsed[oi].data);
    }

    // One segment: the whole image at `base` (R|W|X), the bss tail living only in memsz.
    var segments = try allocator.alloc(Segment, 1);
    errdefer allocator.free(segments);
    segments[0] = .{ .vaddr = base, .paddr = base, .bytes = code, .memsz = memsz, .flags = 7 };

    // Record where each (object, section) landed: one segment (index 0), the region
    // offset being both the in-segment offset and (added to `base`) the runtime address.
    // A section with no bytes is not placed.
    var places = try allocator.alloc(SecPlace, nobj * places_per_object);
    errdefer allocator.free(places);
    for (0..nobj) |oi| {
        const present = [places_per_object]bool{
            parsed[oi].text.len > 0,
            parsed[oi].rodata.len > 0,
            parsed[oi].data.len > 0,
            parsed[oi].bss_size > 0,
        };
        inline for (.{ SecKind.text, SecKind.rodata, SecKind.data, SecKind.bss }) |kind| {
            const idx = oi * places_per_object + secIndex(kind);
            if (present[secIndex(kind)]) {
                const off = regionOffset(kind, oi, rodata_region, data_region, bss_region, text_at, rodata_at, data_at, bss_at);
                places[idx] = .{ .vaddr = base + off, .seg = 0, .seg_off = off };
            } else {
                places[idx] = .{ .vaddr = 0, .seg = not_placed, .seg_off = 0 };
            }
        }
    }

    // Resolve each defined, non-local symbol to its final address.
    var symbols: std.ArrayList(ResolvedSymbol) = .empty;
    errdefer {
        for (symbols.items) |s| allocator.free(s.name);
        symbols.deinit(allocator);
    }
    for (0..nobj) |oi| {
        for (parsed[oi].symbols) |sym| {
            // A symbol can be "defined" (st_shndx != UNDEF) yet resolve to no
            // allocatable section (e.g. an ABS symbol); such symbols have no
            // region offset, so skip them rather than hitting the
            // `.undef => unreachable` in regionOffset.
            if (!sym.defined or sym.local or sym.name.len == 0 or sym.section == .undef) continue;
            if (findSymbol(symbols.items, sym.name) != null) return error.DuplicateSymbol;
            const region = regionOffset(sym.section, oi, rodata_region, data_region, bss_region, text_at, rodata_at, data_at, bss_at);
            const name = try allocator.dupe(u8, sym.name);
            errdefer allocator.free(name);
            try symbols.append(allocator, .{ .name = name, .address = base + region + sym.value, .section = sym.section });
        }
    }

    return .{
        .segments = segments,
        .places = places,
        .symbols = try symbols.toOwnedSlice(allocator),
        .entry = 0,
    };
}

/// Apply every relocation into `placement`'s segment bytes, sourcing each site's address
/// and each target from the placement. Both `.pc32` and `.plt32` sites are always in
/// `.text` and carry the same math: `disp32 = symbol_addr + addend - site_addr`. The bit
/// math is unchanged from the old single-image path (only the address *source* moved to
/// the placement). It does not own `parsed`.
pub fn applyRelocs(allocator: std.mem.Allocator, placement: *Placement, parsed: []ParsedObject) Error!void {
    _ = allocator;
    for (parsed, 0..) |*obj, oi| {
        const tp = placement.places[oi * places_per_object + secIndex(.text)];
        for (obj.relocs) |r| {
            if (r.symbol >= obj.symbols.len) return error.MalformedObject;
            if (tp.seg == not_placed or tp.seg >= placement.segments.len) return error.MalformedObject;
            const seg = &placement.segments[tp.seg];
            const name = obj.symbols[r.symbol].name;
            const target = findSymbol(placement.symbols, name) orelse return error.UndefinedSymbol;
            // Offset of the patched word within this segment's bytes.
            // Derive both the in-segment write offset and the site's runtime address
            // from the segment actually being patched, so they stay in lockstep (see the
            // aarch64 backend for the rationale; `tp.vaddr` is a redundant `base +
            // seg_off` restatement that the placement refactor introduced).
            const site = std.math.add(u64, tp.seg_off, r.offset) catch return error.MalformedObject;
            const site_addr = seg.vaddr + site;
            switch (r.type) {
                .pc32, .plt32 => {
                    const resolved = @as(i64, @intCast(target)) + r.addend - @as(i64, @intCast(site_addr));
                    try applyDisp32(seg.bytes, site, resolved);
                },
                else => return error.UnsupportedReloc,
            }
        }
    }
}

/// Patch a 4-byte PC-relative displacement (`R_X86_64_PC32`/`R_X86_64_PLT32`) at
/// image offset `site` with the already-computed `resolved` value.
fn applyDisp32(code: []u8, site: u64, resolved: i64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const disp = std.math.cast(i32, resolved) orelse return error.RelocationOutOfRange;
    std.mem.writeInt(i32, code[s..][0..4], disp, .little);
}

/// `R_X86_64_JUMP_SLOT`: the dynamic relocation a `.rela.plt` entry carries so a real
/// `ld.so` binds an imported function's `.got.plt` slot to the resolved target address
/// (eager under `DF_BIND_NOW`). The dynamic emitter writes these; the static path never does.
pub const R_X86_64_JUMP_SLOT: u32 = 7;

/// `R_X86_64_GLOB_DAT`: the dynamic relocation a `.rela.dyn` entry carries so a real `ld.so`
/// writes an imported DATA global's runtime address into its `.got` slot at load. The dynamic
/// emitter writes these for GOT-indirect data imports; the static path never does.
pub const R_X86_64_GLOB_DAT: u32 = 6;

/// `R_X86_64_RELATIVE`: the dynamic relocation a PIE/`.so` carries for a pointer-initialized
/// data slot whose target is defined INTERNALLY. The loader writes `load_bias + r_addend` (the
/// target's biased runtime address) into the slot at `r_offset` - no symbol lookup, just a base
/// fixup. The dynamic emitter synthesizes these in `.rela.dyn`; a non-PIE `ET_EXEC` resolves the
/// same slot directly at link time (no dyn reloc). Mirrors `arch/aarch64.zig`'s
/// `R_AARCH64_RELATIVE`.
pub const R_X86_64_RELATIVE: u32 = 8;

/// Patch the disp32 of a GOT-indirect `mov rd, [rip+disp32]` (`R_X86_64_GOTPCREL`) at image
/// offset `site` (the disp32 field, runtime address `site_vaddr`) so it addresses the data
/// import's `.got` slot at `got_slot_vaddr`. The CPU's RIP at the read point is the byte after
/// the 4-byte field, so `disp32 = got_slot_vaddr - (site_vaddr + 4)`. Shares `applyDisp32` with
/// the static PC32/PLT32 path (the same rip-relative disp32 write). A public seam for the
/// dynamic linker's data-import patch.
pub fn patchGotPcRel(code: []u8, site: u64, site_vaddr: u64, got_slot_vaddr: u64) Error!void {
    const disp: i64 = @as(i64, @intCast(got_slot_vaddr)) - @as(i64, @intCast(site_vaddr + 4));
    return applyDisp32(code, site, disp);
}

/// Encode a 16-byte x86-64 PLT entry that tail-jumps through the `.got.plt` slot at
/// `got_slot_vaddr`, given the entry's own runtime address `plt_vaddr`:
///
///   ff 25 <disp32>   jmp *disp32(%rip)   ; jump to *got_slot (ld.so wrote the target there)
///   90 90 ...        nop padding         ; pad the 6-byte stub to 16 bytes
///
/// The RIP-relative `jmp` reads its target through the GOT slot: the CPU's RIP at the read
/// point is the byte after the 6-byte instruction (`plt_vaddr + 6`), so the disp32 is
/// `got_slot_vaddr - (plt_vaddr + 6)`. The GOT slot (8 bytes, initially 0) is filled by the
/// loader under BIND_NOW before `_start` runs. The GOT and PLT sit within a couple of pages
/// of each other, so the displacement always fits an i32. Used only by the dynamic import
/// path; the returned bytes are copied into the `.plt` (an RX segment).
pub fn pltEntry(got_slot_vaddr: u64, plt_vaddr: u64) [16]u8 {
    var e: [16]u8 = @splat(0x90); // nop padding
    e[0] = 0xff;
    e[1] = 0x25;
    const disp: i64 = @as(i64, @intCast(got_slot_vaddr)) - @as(i64, @intCast(plt_vaddr + 6));
    std.mem.writeInt(i32, e[2..6], @intCast(disp), .little);
    return e;
}

/// Redirect a `call` at image offset `site` (the rel32 field's runtime address `site_vaddr`)
/// to call `target_plt_vaddr` (a synthesized PLT entry), via the PC-relative rel32 the
/// `E8 <rel32>` call carries. The CPU's RIP at the branch is the byte after the 4-byte
/// field, so `rel32 = target_plt_vaddr - (site_vaddr + 4)`. A public seam for the dynamic
/// linker's import redirect, sharing `applyDisp32` with the static PC32/PLT32 path so the
/// disp32 write stays single-sourced. `site`/`site_vaddr` name the rel32 field itself (the
/// import call's reloc offset), exactly as `applyRelocs` sources a PLT32 site.
pub fn redirectCall(code: []u8, site: u64, site_vaddr: u64, target_plt_vaddr: u64) Error!void {
    const rel: i64 = @as(i64, @intCast(target_plt_vaddr)) - @as(i64, @intCast(site_vaddr + 4));
    return applyDisp32(code, site, rel);
}
