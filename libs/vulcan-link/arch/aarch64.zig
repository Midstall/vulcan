//! AArch64 link backend for the shared static linker. Ported faithfully from the old
//! `libs/vulcan-target/aarch64/ld.zig`: it owns the AArch64 reloc math (`R_AARCH64_CALL26`
//! for a `bl`/`b`, and the `R_AARCH64_ADR_PREL_PG_HI21`/`R_AARCH64_ADD_ABS_LO12_NC` pair
//! for a `global_addr`'s `adrp`+`add`) and the executable parameters. Calls are
//! PC-relative, so `.text` alone is position-independent; the ADRP/ADD pair addresses
//! page-relative to the `adrp` site, so it too works at any load `base` once resolved
//! against the final image. Unlike riscv64, AArch64 has no external resolver/stub/GOT
//! mechanism: an undefined symbol is always `error.UndefinedSymbol`, and there is no
//! RVC-equivalent compressed form. Depends only on `std` and the generic `elf.zig`.

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

/// Executable parameters for wrapping an AArch64 image in a static ELF64 (`ET_EXEC`).
/// The whole file (headers included) maps one page below `base` (64 KiB, AArch64's
/// max page size) so the code itself lands exactly at `base`.
pub const exec_params: elf.ExecParams = .{
    .e_machine = elf.EM_AARCH64,
    .code_offset = 0x10000,
    .p_align = 0x10000,
    .map_headers = true,
};

/// Assign addresses for a set of parsed AArch64 relocatable objects (the default, no
/// linker-script layout), laid out in the order given with `.text` first at `base`, then
/// `.rodata`, `.data`, and `.bss` (memory only). Produces a single loadable `Segment`
/// (the whole image at `base`, R|W|X), a `places` map recording where each (object,
/// section) landed, and the resolved symbol table (`base + region + value`, rejecting a
/// duplicate, skipping undef/local/ABS symbols). No relocations are applied here (that is
/// `applyRelocs`). AArch64 has no resolver/stub/GOT mechanism, so `resolver` is unused;
/// `compress_text` is unused too (no RVC-equivalent form). The caller owns the returned
/// placement; it does not own `parsed`.
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

/// `R_AARCH64_JUMP_SLOT`: the dynamic relocation a `.rela.plt` entry carries so a real
/// `ld.so` binds an imported function's `.got.plt` slot to the resolved target address
/// (eager under `DF_BIND_NOW`). The dynamic emitter writes these; the static path never does.
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;

/// `R_AARCH64_GLOB_DAT`: the dynamic relocation a `.rela.dyn` entry carries so a real
/// `ld.so` writes an imported DATA symbol's runtime address into the executable's GOT
/// slot at load. The dynamic emitter writes these for GOT-indirect data imports; the
/// static path never does.
pub const R_AARCH64_GLOB_DAT: u32 = 1025;

/// `R_AARCH64_RELATIVE`: the dynamic relocation a PIE/`.so` carries for a pointer-initialized
/// data slot whose target is defined INTERNALLY. The loader writes `load_bias + r_addend`
/// (the target's biased runtime address) into the slot at `r_offset` - no symbol lookup, just
/// a base fixup. The dynamic emitter synthesizes these in `.rela.dyn`; a non-PIE `ET_EXEC`
/// resolves the same slot directly at link time (no dyn reloc).
pub const R_AARCH64_RELATIVE: u32 = 1027;

/// Patch the `adrp` of a GOT-indirect data import (`R_AARCH64_ADR_GOT_PAGE`) at image
/// offset `site` (runtime address `site_vaddr`) to the 4KiB page of the GOT slot at
/// `got_slot_vaddr`. Same page math as the ADRP `global_addr` reloc (`applyAdrpPg`), but
/// the target is the synthesized GOT slot, not the symbol itself. A public seam for the
/// dynamic linker's data-import path.
pub fn patchGotPage(code: []u8, site: u64, site_vaddr: u64, got_slot_vaddr: u64) Error!void {
    return applyAdrpPg(code, site, site_vaddr, got_slot_vaddr);
}

/// Patch the 64-bit `ldr` of a GOT-indirect data import (`R_AARCH64_LD64_GOT_LO12_NC`) at
/// image offset `site` to the GOT slot's low-12 scaled by 8 (the LD64 unsigned-offset
/// scale). The GOT slot is 8-aligned, so the shift is exact. Mirrors the `ldr` imm patch
/// in `pltEntry`; a public seam for the dynamic linker's data-import path.
pub fn patchGotLo12(code: []u8, site: u64, got_slot_vaddr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const lo12: u32 = @intCast(got_slot_vaddr & 0xFFF);
    const scaled: u32 = lo12 >> 3;
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    const patched = (word & ~(@as(u32, 0xFFF) << 10)) | (scaled << 10);
    std.mem.writeInt(u32, code[s..][0..4], patched, .little);
}

/// Encode a 16-byte AArch64 PLT entry that tail-jumps through the `.got.plt` slot at
/// `got_slot_vaddr`, given the entry's own runtime address `plt_entry_vaddr`:
///
///   adrp x16, page(got_slot)          ; x16 = the GOT slot's 4KiB page
///   ldr  x17, [x16, #lo12(got_slot)]  ; x17 = *got_slot (ld.so wrote the target here)
///   add  x16, x16, #lo12(got_slot)    ; x16 = &got_slot (scratch, matches the ABI PLT0-less stub)
///   br   x17                          ; jump to the resolved target
///
/// The page delta comes from `applyAdrpPg` (the same math the ADRP `global_addr` reloc uses)
/// and the `add`'s low-12 from `applyAddPgoff`; the 64-bit `ldr`'s unsigned immediate is the
/// GOT slot's low-12 scaled by 8 (the slot is 8-aligned, so this is exact). Used only by the
/// dynamic import path; the returned bytes are copied into the `.plt` (an RX segment).
pub fn pltEntry(got_slot_vaddr: u64, plt_entry_vaddr: u64) [16]u8 {
    var e: [16]u8 = undefined;
    const w = std.mem.writeInt;
    // adrp x16, #0        (Rd = 16); page delta patched below.
    w(u32, e[0..4], 0x90000000 | 16, .little);
    // ldr x17, [x16, #0]  (64-bit unsigned-offset load; Rn = 16, Rt = 17); imm12 patched below.
    w(u32, e[4..8], 0xF9400000 | (16 << 5) | 17, .little);
    // add x16, x16, #0    (Rn = 16, Rd = 16); imm12 patched below.
    w(u32, e[8..12], 0x91000000 | (16 << 5) | 16, .little);
    // br x17
    w(u32, e[12..16], 0xD61F0000 | (17 << 5), .little);

    // ADRP page-relative to the entry's own address (the adrp is the first instruction).
    applyAdrpPg(e[0..], 0, plt_entry_vaddr, got_slot_vaddr) catch unreachable;
    // The 64-bit LDR's unsigned imm12 is scaled by 8; the GOT slot is 8-aligned.
    const lo12: u32 = @intCast(got_slot_vaddr & 0xFFF);
    const scaled: u32 = lo12 >> 3;
    const ldr_w = std.mem.readInt(u32, e[4..8], .little);
    w(u32, e[4..8], (ldr_w & ~(@as(u32, 0xFFF) << 10)) | (scaled << 10), .little);
    // The ADD's imm12 is the raw low-12 bits of the GOT slot address.
    applyAddPgoff(e[0..], 8, got_slot_vaddr) catch unreachable;
    return e;
}

/// Redirect a `bl` at image offset `site` (runtime address `site_vaddr`) to call
/// `target_vaddr` (a synthesized PLT entry), via the CALL26 PC-relative displacement. A
/// public seam for the dynamic linker's import redirect; the static path patches CALL26
/// inline in `applyRelocs`. Both share `applyCall26`, so the bit math stays single-sourced.
pub fn patchCall26(code: []u8, site: u64, site_vaddr: u64, target_vaddr: u64) Error!void {
    const delta = @as(i64, @intCast(target_vaddr)) - @as(i64, @intCast(site_vaddr));
    return applyCall26(code, site, delta);
}

/// Apply every relocation into `placement`'s segment bytes, sourcing each site's address
/// and each target from the placement. `.call26` sites/targets are always in `.text` and
/// are PC-relative (the segment `vaddr` cancels). The ADRP/ADD pair is page/pgoff-relative
/// to the final absolute addresses; each reloc is self-contained (no hi/lo pairing needed,
/// unlike RISC-V's AUIPC/lo12 scheme), so one pass resolves both halves. The bit math is
/// unchanged from the old single-image path (only the address *source* moved to the
/// placement). It does not own `parsed`.
pub fn applyRelocs(allocator: std.mem.Allocator, placement: *Placement, parsed: []ParsedObject) Error!void {
    _ = allocator;
    for (parsed, 0..) |*obj, oi| {
        const tp = placement.places[oi * places_per_object + secIndex(.text)];
        for (obj.relocs) |r| {
            if (r.symbol >= obj.symbols.len) return error.MalformedObject;
            if (tp.seg == not_placed or tp.seg >= placement.segments.len) return error.MalformedObject;
            const seg = &placement.segments[tp.seg];
            const sym = obj.symbols[r.symbol];
            // Resolve the reloc target. A GLOBAL symbol comes from the merged symbol table by
            // name. A LOCAL DEFINED symbol (a real glibc `crt1.o` addresses a `.text`-local
            // trampoline through a `.text` SECTION symbol, whose name is empty and so is not in
            // the global table) is resolved through THIS object's own placement of the symbol's
            // section, plus the symbol's in-section value. The global lookup is tried FIRST, so
            // every pre-existing (global-name) reloc resolves exactly as before.
            const target = findSymbol(placement.symbols, sym.name) orelse blk: {
                if (sym.defined and sym.local and sym.section != .undef) {
                    const sp = placement.places[oi * places_per_object + secIndex(sym.section)];
                    if (sp.seg == not_placed or sp.seg >= placement.segments.len) return error.MalformedObject;
                    break :blk placement.segments[sp.seg].vaddr + sp.seg_off + sym.value;
                }
                return error.UndefinedSymbol;
            };
            // Offset of the patched word within this segment's bytes, and its runtime
            // address. Both derive from the segment actually being patched (`seg`), so
            // the page-relative ADRP/ADD math sees the same load address the bytes land
            // at. (`tp.vaddr` is a redundant `base + seg_off` restatement of the same
            // value; sourcing `site_addr` from `seg.vaddr + site` keeps the site address
            // and the patched buffer in lockstep - matching the pre-placement single
            // image path's `base + site`.)
            const site = std.math.add(u64, tp.seg_off, r.offset) catch return error.MalformedObject;
            const site_addr = seg.vaddr + site;
            switch (r.type) {
                .call26, .jump26 => {
                    // CALL26 and JUMP26 share one encoding field and one PC-relative math.
                    // `applyCall26` rebuilds the word keeping the B-vs-BL opcode bit it found,
                    // so a JUMP26 (a `b` tail branch, from a real `crt1.o`'s trampoline to
                    // `main`) patches correctly the same way a `bl` CALL26 does.
                    const delta = (@as(i64, @intCast(target)) - @as(i64, @intCast(seg.vaddr))) - @as(i64, @intCast(site)) + r.addend;
                    try applyCall26(seg.bytes, site, delta);
                },
                .adr_prel_pg_hi21 => {
                    const target_addr: u64 = @intCast(@as(i64, @intCast(target)) + r.addend);
                    try applyAdrpPg(seg.bytes, site, site_addr, target_addr);
                },
                .add_abs_lo12_nc => {
                    const target_addr: u64 = @intCast(@as(i64, @intCast(target)) + r.addend);
                    try applyAddPgoff(seg.bytes, site, target_addr);
                },
                .adr_got_page => {
                    // A GOT-indirect page reference (`adrp xN, :got:sym`) to a symbol DEFINED in
                    // this image needs no GOT slot: the address is fixed at link time. Relax it to
                    // a direct `adrp xN, sym` page reference - the same instruction, the symbol's
                    // page in place of the GOT slot's page, exactly like an `adr_prel_pg_hi21`. An
                    // UNDEFINED symbol's GOT ref never reaches here (it was diverted to a real GOT
                    // slot in the dynamic-import path), so any GOT reloc left for `applyRelocs` is
                    // in-image and safe to relax.
                    const target_addr: u64 = @intCast(@as(i64, @intCast(target)) + r.addend);
                    try applyAdrpPg(seg.bytes, site, site_addr, target_addr);
                },
                .ld64_got_lo12_nc => {
                    // The paired GOT load (`ldr xT, [xN, :got_lo12:sym]`) for an in-image symbol.
                    // With no GOT slot to load from, REWRITE the `ldr` into `add xT, xN, :lo12:sym`
                    // so the `adrp`+`add` pair computes the symbol address directly.
                    const target_addr: u64 = @intCast(@as(i64, @intCast(target)) + r.addend);
                    try relaxGotLoadToAdd(seg.bytes, site, target_addr);
                },
                .ldst64_abs_lo12_nc => {
                    // The low half of a direct `adrp`/`ldr` pair addressing a symbol's storage
                    // (`ldr xT, [xN, #:lo12:sym]`). The 64-bit access scales its 12-bit immediate
                    // by 8, so the field holds `lo12(sym) >> 3`.
                    const target_addr: u64 = @intCast(@as(i64, @intCast(target)) + r.addend);
                    try applyLdst64Lo12(seg.bytes, site, target_addr);
                },
                else => return error.UnsupportedReloc,
            }
        }
    }
}

/// Patch a `bl`/`b` (R_AARCH64_CALL26) at image offset `site` with the byte
/// displacement `delta` (+/-128MiB, multiple of 4). Rebuilds the instruction word
/// from scratch (keeping only which of BL/B it was), mirroring `encode.bl`/`encode.b`.
fn applyCall26(code: []u8, site: u64, delta: i64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    if (delta < -(1 << 27) or delta >= (1 << 27) or (delta & 3) != 0) return error.RelocationOutOfRange;
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    const is_bl = (word & 0xFC000000) == 0x94000000; // BL vs B share the encoding but for bit 31
    const off: i28 = @intCast(delta);
    const imm26: u32 = @as(u32, @bitCast(@as(i32, off) >> 2)) & 0x3FFFFFF;
    const patched = (if (is_bl) @as(u32, 0x94000000) else @as(u32, 0x14000000)) | imm26;
    std.mem.writeInt(u32, code[s..][0..4], patched, .little);
}

/// Patch an `adrp` (R_AARCH64_ADR_PREL_PG_HI21) at image offset `site`: the
/// 21-bit page delta `(page(target_addr) - page(site_addr)) >> 12` (page = address
/// with its low 12 bits cleared) split across the immhi:immlo fields, matching
/// AArch64's ADRP encoding. Bit math ported verbatim from `link.zig`'s
/// `applyGlobalReloc` (proven correct there for the JIT path).
fn applyAdrpPg(code: []u8, site: u64, site_addr: u64, target_addr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const site_page = site_addr & ~@as(u64, 0xFFF);
    const target_page = target_addr & ~@as(u64, 0xFFF);
    const delta_pages = @divExact(@as(i64, @intCast(target_page)) - @as(i64, @intCast(site_page)), 4096);
    if (delta_pages < -(1 << 20) or delta_pages >= (1 << 20)) return error.RelocationOutOfRange;
    const imm: i21 = @intCast(delta_pages);
    const u: u21 = @bitCast(imm);
    const immlo: u32 = u & 0x3;
    const immhi: u32 = u >> 2;
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    const patched = (word & ~((@as(u32, 0x3) << 29) | (@as(u32, 0x7FFFF) << 5))) | (immlo << 29) | (immhi << 5);
    std.mem.writeInt(u32, code[s..][0..4], patched, .little);
}

/// Patch an `add` (R_AARCH64_ADD_ABS_LO12_NC) at image offset `site`: the low 12
/// bits of `target_addr` go in the imm12 field. Bit math ported verbatim from
/// `link.zig`'s `applyGlobalReloc`.
fn applyAddPgoff(code: []u8, site: u64, target_addr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const lo12: u32 = @intCast(target_addr & 0xFFF);
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    const patched = (word & ~(@as(u32, 0xFFF) << 10)) | (lo12 << 10);
    std.mem.writeInt(u32, code[s..][0..4], patched, .little);
}

/// Relax an aarch64 GOT load (`ldr Xt, [Xn, #:got_lo12:sym]`) into `add Xt, Xn, #:lo12:sym`
/// for a symbol defined in this image, then fill the low-12-bit immediate. The `ldr` word's
/// destination register (Rt, bits 0-4) and base register (Rn, bits 5-9) are kept; only the
/// opcode becomes the 64-bit ADD-immediate form. Pairs with the relaxed `adrp` so the two
/// together compute `&sym` directly, no GOT slot involved.
fn relaxGotLoadToAdd(code: []u8, site: u64, target_addr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const ldr = std.mem.readInt(u32, code[s..][0..4], .little);
    const rt = ldr & 0x1F;
    const rn = (ldr >> 5) & 0x1F;
    const add = @as(u32, 0x91000000) | (rn << 5) | rt; // `add Xt, Xn, #0`; the imm12 is set next.
    std.mem.writeInt(u32, code[s..][0..4], add, .little);
    try applyAddPgoff(code, site, target_addr);
}

/// Patch a 64-bit `ldr`/`str` (R_AARCH64_LDST64_ABS_LO12_NC): the low 12 bits of `target_addr`,
/// scaled DOWN by 8 (the access size), go in the imm12 field. The symbol's storage is 8-aligned,
/// so the low 3 bits are zero and the shift is exact.
fn applyLdst64Lo12(code: []u8, site: u64, target_addr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const lo12: u32 = @intCast((target_addr & 0xFFF) >> 3);
    const word = std.mem.readInt(u32, code[s..][0..4], .little);
    const patched = (word & ~(@as(u32, 0xFFF) << 10)) | (lo12 << 10);
    std.mem.writeInt(u32, code[s..][0..4], patched, .little);
}

test "applyRelocs sources the ADRP/ADD site address from the segment, not the redundant SecPlace.vaddr" {
    // Regression for a placement-refactor miscompile: the `adrp`/`add` (global_addr)
    // path computed the site's runtime address as `tp.vaddr + r.offset`, reading the
    // redundant `SecPlace.vaddr` field, while the `call26` path (correctly) used
    // `seg.vaddr + site`. When that redundant field read came back one page high (a
    // build/heap-layout-dependent misread observed in the field), the ADRP page delta
    // landed one page off and the global read a zero page - even though the resolved
    // symbol and the byte placement were both correct. This test pins the invariant that
    // `applyRelocs` must derive the site address from the segment actually being patched,
    // by handing it a placement whose `places[text].vaddr` is deliberately inconsistent
    // with the segment (as the misread made it) and requiring correct resolution anyway.
    const allocator = std.testing.allocator;
    const base: u64 = 0x400000;

    // `adrp x0, g@PAGE` / `add x0, x0, g@PAGEOFF` placeholders, then a 4-byte `.data` `g`.
    const g_off: u64 = 8;
    var code = try allocator.alloc(u8, 12);
    defer allocator.free(code);
    std.mem.writeInt(u32, code[0..4], 0x90000000, .little); // adrp x0, #0 (placeholder)
    std.mem.writeInt(u32, code[4..8], 0x91000000, .little); // add  x0, x0, #0 (placeholder)
    std.mem.writeInt(u32, code[8..12], 5, .little); // g = i32 5

    var syms = [_]elf.ObjSymbol{.{ .name = "g", .value = 0, .defined = true, .local = false, .section = .data }};
    var relocs = [_]elf.Reloc{
        .{ .offset = 0, .symbol = 0, .type = .adr_prel_pg_hi21, .addend = 0 },
        .{ .offset = 4, .symbol = 0, .type = .add_abs_lo12_nc, .addend = 0 },
    };
    var parsed = [_]ParsedObject{.{
        .arch = .aarch64,
        .text = code[0..8],
        .data = code[8..12],
        .symbols = &syms,
        .relocs = &relocs,
    }};

    var segments = [_]Segment{.{ .vaddr = base, .paddr = base, .bytes = code, .memsz = code.len, .flags = 7 }};
    // The `.text` place's `vaddr` is set one page high on purpose (base + 0x1000) while
    // its `seg_off` stays 0 - exactly the inconsistency the miscompile produced. A linker
    // that (wrongly) trusts `tp.vaddr` resolves the ADRP one page low; the fix ignores it.
    var places = [_]SecPlace{
        .{ .vaddr = base + 0x1000, .seg = 0, .seg_off = 0 }, // text (poisoned vaddr)
        .{ .vaddr = 0, .seg = not_placed, .seg_off = 0 }, // rodata
        .{ .vaddr = base + g_off, .seg = 0, .seg_off = g_off }, // data
        .{ .vaddr = 0, .seg = not_placed, .seg_off = 0 }, // bss
    };
    var symbols = [_]ResolvedSymbol{.{ .name = "g", .address = base + g_off }};
    var placement: Placement = .{ .segments = &segments, .places = &places, .symbols = &symbols, .entry = 0 };

    try applyRelocs(allocator, &placement, &parsed);

    // Decode the patched adrp/add and reconstruct the effective address the CPU computes,
    // using the site's TRUE runtime page (the adrp lives at `base`). It must equal `g`.
    const adrp_w = std.mem.readInt(u32, code[0..4], .little);
    const add_w = std.mem.readInt(u32, code[4..8], .little);
    const immlo: u32 = (adrp_w >> 29) & 0x3;
    const immhi: u32 = (adrp_w >> 5) & 0x7FFFF;
    const imm21: i64 = @as(i21, @bitCast(@as(u21, @intCast((immhi << 2) | immlo))));
    const site_page: u64 = base & ~@as(u64, 0xFFF);
    const adrp_target_page: u64 = @intCast(@as(i64, @intCast(site_page)) + imm21 * 4096);
    const lo12: u64 = (add_w >> 10) & 0xFFF;
    try std.testing.expectEqual(base + g_off, adrp_target_page + lo12);
}
