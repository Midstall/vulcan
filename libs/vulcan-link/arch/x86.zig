//! i386 (32-bit x86) link backend for the shared static linker. The first real i386
//! object linker: previously `libs/vulcan-target/x86` had no real linker at all, only a
//! flat-blob `elf.zig` writer whose test harness hand-resolved cross-object relocs
//! itself (see `x86/tests/harness.zig`'s `runModuleData`). This module owns the i386
//! reloc math (`R_386_PC32` for a `call rel32`, `R_386_32` for a `global_addr`'s plain
//! `mov rd, imm32`) and the executable parameters.
//!
//! i386 objects use `SHT_REL` (`Elf32_Rel`), which carries no addend field: the addend
//! is pre-written into the relocated field itself by `x86/object.zig` (`-4` for a
//! `.call` site, `0` for a `.abs32` site - see that file's `put(text, i32, ..., -4)`).
//! So unlike every other arch here (all `SHT_RELA`, addend read straight off the reloc
//! entry), this backend reads the *existing* 4 bytes at each site as the addend before
//! computing the resolved value, then overwrites those same bytes with the result.
//!
//! `R_386_PC32` is PC-relative (`disp32 = target - site_addr + in_field_addend`, exactly
//! `x86_64.zig`'s `pc32`/`plt32` math with the addend sourced differently);
//! `R_386_32` is a plain absolute store (`resolved = target + in_field_addend`, no
//! subtraction) - this is how a `global_addr` load gets the symbol's true runtime
//! address baked directly into the `mov`'s immediate.
//!
//! Like x86-64/AArch64, i386 has no external resolver/stub/GOT mechanism here: an
//! undefined symbol is always `error.UndefinedSymbol`. Depends only on `std` and the
//! generic `elf.zig`.

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

/// Executable parameters for wrapping an i386 image in a static ELF32 (`ET_EXEC`).
/// Matches the classic i386 Linux base (`vulcan-target/x86/elf.zig`'s `load_addr`).
/// `writeElfExec`/`writeElfSegments` in `resolve.zig` branch on `arch == .x86` to emit
/// the ELF32 header/program-header layout (different field offsets than ELF64, not just
/// widths) but still read these same `ExecParams`.
pub const exec_params: elf.ExecParams = .{
    .e_machine = elf.EM_386,
    .code_offset = 0x1000,
    .p_align = 0x1000,
};

/// Assign addresses for a set of parsed i386 relocatable objects (the default, no
/// linker-script layout), laid out in the order given with `.text` first at `base`, then
/// `.rodata`, `.data`, and `.bss` (memory only). Produces a single loadable `Segment`
/// (the whole image at `base`, R|W|X), a `places` map recording where each (object,
/// section) landed, and the resolved symbol table (`base + region + value`, rejecting a
/// duplicate, skipping undef/local/ABS symbols). No relocations are applied here (that is
/// `applyRelocs`). i386 has no resolver/stub/GOT mechanism, so `resolver` is unused;
/// `compress_text` is unused too (no compressed-instruction form). The caller owns the
/// returned placement; it does not own `parsed`. The segment this produces is
/// ELFCLASS32-sized (`Placement`/`Segment` are architecture-generic - `writeElfSegments`
/// is what branches to the ELF32 writer for `.x86`).
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
/// and each target from the placement. Both `.pc32` and `.abs32` sites are always in
/// `.text`. `SHT_REL` carries no addend field, so `applyPc32`/`applyAbs32` read the
/// existing 4 bytes at each site as the addend before overwriting them with the resolved
/// value - unchanged from the old single-image path (only the address *source* moved to
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
                .pc32 => try applyPc32(seg.bytes, site, site_addr, target),
                .abs32 => try applyAbs32(seg.bytes, site, target),
                else => return error.UnsupportedReloc,
            }
        }
    }
}

/// Patch an `R_386_PC32` site (a `call rel32`) at image offset `site`: read the
/// existing 4 LE bytes as the REL in-field addend (`object.zig` writes `-4` for a
/// call, the disp32 field's own width), then store
/// `disp32 = target_addr - site_addr + in_field_addend` back into the same field.
fn applyPc32(code: []u8, site: u64, site_addr: u64, target_addr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const in_field = std.mem.readInt(i32, code[s..][0..4], .little);
    const resolved = @as(i64, @intCast(target_addr)) - @as(i64, @intCast(site_addr)) + in_field;
    const disp = std.math.cast(i32, resolved) orelse return error.RelocationOutOfRange;
    std.mem.writeInt(i32, code[s..][0..4], disp, .little);
}

/// Patch an `R_386_32` site (a `mov rd, imm32` reading a `global_addr`) at image
/// offset `site`: read the existing 4 LE bytes as the REL in-field addend
/// (`object.zig` always writes 0 here), then store the plain absolute
/// `resolved = target_addr + in_field_addend` back into the same field. No
/// PC-relative subtraction - unlike `applyPc32`, this is the symbol's true runtime
/// address.
fn applyAbs32(code: []u8, site: u64, target_addr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const in_field = std.mem.readInt(i32, code[s..][0..4], .little);
    const resolved = @as(i64, @intCast(target_addr)) + in_field;
    const val = std.math.cast(u32, resolved) orelse return error.RelocationOutOfRange;
    std.mem.writeInt(u32, code[s..][0..4], val, .little);
}

/// `R_386_JMP_SLOT`: the dynamic relocation a `.rel.plt` entry carries so a real `ld.so`
/// binds an imported function's `.got.plt` slot to the resolved target address (eager under
/// `DF_BIND_NOW`). i386 uses `SHT_REL` (`Elf32_Rel`, no addend field), so the JUMP_SLOT
/// reloc has no addend at all - the loader writes the symbol's resolved address straight into
/// the slot. The ELF32 dynamic emitter writes these; the static path never does.
pub const R_386_JMP_SLOT: u32 = 7;

/// `R_386_GLOB_DAT`: the dynamic relocation an ELF32 `.rel.dyn` entry carries so a real
/// `ld.so` writes an imported DATA global's runtime address into its `.got` slot at load.
/// i386 uses `SHT_REL` (`Elf32_Rel`, no addend field), so the GLOB_DAT reloc has no addend -
/// the loader writes the symbol's resolved address straight into the slot. The ELF32 dynamic
/// emitter writes these for GOT-indirect data imports; the static path never does.
pub const R_386_GLOB_DAT: u32 = 6;

/// `R_386_RELATIVE`: the dynamic relocation a PIE/`.so` carries for a pointer-initialized data
/// global whose target is defined INTERNALLY (`int *p = &g;`, not imported). Unlike ELF64's
/// RELA `R_X86_64_RELATIVE`/`R_AARCH64_RELATIVE` (which carry the target's base-0 vaddr in
/// `r_addend`, so the loader OVERWRITES the slot with `bias + r_addend`), i386's `Elf32_Rel`
/// has NO addend field: the loader instead does `*slot += load_bias` IN PLACE. So the linker
/// must PRE-WRITE the target's nominal (base-relative) vaddr into the slot itself before
/// emitting this reloc - the loader's in-place add then turns it into the final runtime
/// address. See `dynamic.zig`'s `emit32` for the pre-write + reloc-entry pairing.
pub const R_386_RELATIVE: u32 = 8;

/// Patch the abs32 field of a GOT-indirect `mov rd, [abs32]` (`R_386_GOT32`) at image offset
/// `site` to the data import's `.got` slot at `got_slot_vaddr`. i386 has no rip-relative form,
/// so the `mov` reads through an ABSOLUTE 32-bit address: the abs32 field IS the GOT slot's
/// runtime vaddr, written directly (mirroring the x86 PLT's `ff 25 <abs32>` slot address in
/// `pltEntry`). A public seam for the dynamic linker's data-import patch.
pub fn patchGotAbs(code: []u8, site: u64, got_slot_vaddr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const val = std.math.cast(u32, got_slot_vaddr) orelse return error.RelocationOutOfRange;
    std.mem.writeInt(u32, code[s..][0..4], val, .little);
}

/// Encode a 16-byte i386 PLT entry that tail-jumps through the `.got.plt` slot at
/// `got_slot_vaddr`, given the entry's own runtime address `plt_vaddr` (unused here):
///
///   ff 25 <abs32>   jmp *abs32          ; jump to *[got_slot_vaddr] (ld.so wrote the target)
///   90 90 ...       nop padding         ; pad the 6-byte stub to 16 bytes
///
/// Unlike x86-64's `ff 25 <disp32>` (RIP-relative: the disp is measured from the next
/// instruction), i386's `ff 25 <abs32>` reads through an ABSOLUTE 32-bit address - correct
/// for a non-PIE dynamic executable loaded at a fixed base, where the GOT slot's runtime
/// address is known at link time. So `abs32` is the GOT slot's VADDR itself, not a
/// displacement. The GOT slot (4 bytes, initially 0) is filled by the loader under BIND_NOW
/// before `_start` runs. Used only by the dynamic import path; the returned bytes are copied
/// into the `.plt` (an RX segment).
pub fn pltEntry(got_slot_vaddr: u64, plt_vaddr: u64) [16]u8 {
    _ = plt_vaddr;
    var e: [16]u8 = @splat(0x90); // nop padding
    e[0] = 0xff;
    e[1] = 0x25;
    std.mem.writeInt(u32, e[2..6], @intCast(got_slot_vaddr), .little); // absolute GOT slot addr
    return e;
}

/// The PIE counterpart of `pltEntry`: `pltEntry`'s absolute `ff 25 <abs32>` is only correct
/// when the executable loads at the exact base it was linked at (a non-PIE `ET_EXEC` - the
/// GOT slot's runtime address is then a link-time constant). A PIE loads at whatever base
/// `ld.so` chooses, and i386 has no RIP-relative addressing mode to fall back on (unlike
/// x86-64's `ff 25 <disp32(%rip)>`, which is already position-independent and needs no PIE
/// variant), so this stub instead computes its OWN runtime address via the classic PIC
/// "call-next; pop" idiom, then reaches the GOT slot through a register-relative `jmp`, whose
/// displacement is the LINK-TIME constant `got_slot_vaddr - (plt_vaddr + 5)` - a difference
/// between two positions in the SAME image, invariant under any uniform load-address shift:
///
///   e8 00 00 00 00   call next           ; self-referential (rel32=0), no reloc needed
/// next:
///   58               pop eax             ; eax = this stub's OWN runtime address of `next`
///   ff a0 <disp32>   jmp [eax+disp32]    ; jump to *[eax+disp32] == *[got_slot_vaddr] at runtime
///   90 90 90 90      nop padding         ; pad the 12-byte stub to 16 bytes
///
/// `jmp [eax+disp32]` (`ff /4`, ModRM `10_100_000` = 0xA0) reads the same GOT-slot contents
/// `pltEntry`'s absolute form does, so the loader's `JUMP_SLOT` fill (BIND_NOW) is unchanged;
/// only how the stub ADDRESSES that slot differs. Used only when `DynOptions.pie` is set.
pub fn pltEntryPie(got_slot_vaddr: u64, plt_vaddr: u64) [16]u8 {
    var e: [16]u8 = @splat(0x90); // nop padding
    e[0] = 0xe8;
    e[1] = 0x00;
    e[2] = 0x00;
    e[3] = 0x00;
    e[4] = 0x00; // call next (next == plt_vaddr + 5, self-referential, no reloc)
    e[5] = 0x58; // pop eax
    e[6] = 0xff;
    e[7] = 0xa0; // jmp [eax+disp32]
    const next = plt_vaddr + 5;
    const disp: i64 = @as(i64, @intCast(got_slot_vaddr)) - @as(i64, @intCast(next));
    std.mem.writeInt(i32, e[8..12], @intCast(disp), .little);
    return e;
}

/// Redirect a `call` at image offset `site` (the rel32 field's runtime address `site_vaddr`)
/// to call `target_plt_vaddr` (a synthesized PLT entry), via the PC-relative rel32 the
/// `E8 <rel32>` call carries. The CPU's EIP at the branch is the byte after the 4-byte
/// field, so `rel32 = target_plt_vaddr - site_vaddr - 4`. Written directly into the field
/// (the `-4` is the field's own width, mirroring x86-64's `redirectCall`); the object's
/// pre-written REL in-field addend at this site is overwritten. `site`/`site_vaddr` name the
/// rel32 field itself (the import call's reloc offset), exactly as `applyRelocs` sources a
/// PC32 site.
pub fn redirectCall(code: []u8, site: u64, site_vaddr: u64, target_plt_vaddr: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s > code.len or 4 > code.len - s) return error.MalformedObject;
    const rel = @as(i64, @intCast(target_plt_vaddr)) - @as(i64, @intCast(site_vaddr)) - 4;
    const disp = std.math.cast(i32, rel) orelse return error.RelocationOutOfRange;
    std.mem.writeInt(i32, code[s..][0..4], disp, .little);
}
