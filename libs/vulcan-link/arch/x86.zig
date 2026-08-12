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
const place = @import("../place.zig");

const Error = elf.Error;
const ParsedObject = elf.ParsedObject;
const ResolvedSymbol = elf.ResolvedSymbol;
const Image = elf.Image;
const Resolver = elf.Resolver;
const Segment = elf.Segment;
const Placement = elf.Placement;
const alignUp = elf.alignUp;
const findSymbol = elf.findSymbol;
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
pub fn computeDefaultPlacement(allocator: std.mem.Allocator, parsed: []ParsedObject, base: u64, resolver: ?Resolver, compress_text: bool, live: ?[]const bool) Error!Placement {
    _ = compress_text;
    // i386 packs `.text` to 16 bytes and has no stub/GOT region, so it passes no hook.
    // The shared placer ignores `resolver` without a hook. `live` is the GC mask (null when
    // GC is off, so the layout is byte-identical).
    return place.computeDefaultPlacement(allocator, parsed, base, .{ .text_align = 16 }, null, resolver, live);
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
        // Patch every relocation the object's EXECUTABLE (`SHF_EXECINSTR`) sections carry.
        // Each exec section owns its own reloc list and its own placement, so a site's image
        // offset comes from that section's place. Non-exec (data-pointer) relocs are handled
        // only by the dynamic path's `collectDataFixups`, never here.
        for (obj.sections, 0..) |*isec, si| {
            if ((isec.flags & elf.SHF_EXECINSTR) == 0) continue;
            if (isec.relocs.len == 0) continue;
            const sp = placement.sectionPlace(oi, si);
            // A GC-dropped (not-placed) section takes no bytes in the image, so its own relocs
            // are dropped with it. A real dropped section always has a nonzero size (it held
            // actual code), so a ZERO-size section that still carries relocs cannot be a GC
            // drop: it is a malformed object, and stays fail-closed. With GC off no
            // reloc-bearing section is ever not-placed, so this is byte-identical to the plain
            // error path.
            if (sp.seg == not_placed) {
                if (isec.size == 0) return error.MalformedObject;
                continue;
            }
            if (sp.seg >= placement.segments.len) return error.MalformedObject;
            const seg = &placement.segments[sp.seg];
            for (isec.relocs) |r| {
                if (r.symbol >= obj.symbols.len) return error.MalformedObject;
                const name = obj.symbols[r.symbol].name;
                const target = findSymbol(placement.symbols, name) orelse return error.UndefinedSymbol;
                // Offset of the patched word within this segment's bytes. Derive both the
                // in-segment write offset and the site's runtime address from the segment
                // actually being patched, so they stay in lockstep (see the aarch64 backend
                // for the rationale). The SHT_REL addend read-back inside the `apply*` helpers
                // is unchanged.
                const site = std.math.add(u64, sp.seg_off, r.offset) catch return error.MalformedObject;
                const site_addr = seg.vaddr + site;
                switch (r.type) {
                    .pc32 => try applyPc32(seg.bytes, site, site_addr, target),
                    .abs32 => try applyAbs32(seg.bytes, site, target),
                    else => return error.UnsupportedReloc,
                }
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
