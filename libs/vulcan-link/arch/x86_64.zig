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
pub fn computeDefaultPlacement(allocator: std.mem.Allocator, parsed: []ParsedObject, base: u64, resolver: ?Resolver, compress_text: bool, live: ?[]const bool) Error!Placement {
    _ = compress_text;
    // x86-64 packs `.text` to 16 bytes and has no stub/GOT region, so it passes no hook.
    // The shared placer ignores `resolver` without a hook. `live` is the GC mask (null when
    // GC is off, so the layout is byte-identical).
    return place.computeDefaultPlacement(allocator, parsed, base, .{ .text_align = 16 }, null, resolver, live);
}

/// Apply every relocation into `placement`'s segment bytes, sourcing each site's address
/// and each target from the placement. Both `.pc32` and `.plt32` sites are always in
/// `.text` and carry the same math: `disp32 = symbol_addr + addend - site_addr`. The bit
/// math is unchanged from the old single-image path (only the address *source* moved to
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
                const sym = obj.symbols[r.symbol];
                // Resolve the reloc target. A GLOBAL symbol comes from the merged symbol table by
                // name. A LOCAL DEFINED symbol (a `.str.N` string literal, made local so per-object
                // duplicate numbering never collides at link time; a `lea rd, [rip+.str.N]` becomes
                // a PC32 to it) is not in the global table, so it resolves through THIS object's own
                // placement of the symbol's defining section plus its in-section value - the same
                // fallback the aarch64 backend and `collectDataFixups` use. The global lookup is
                // tried FIRST, so every pre-existing (global-name) reloc resolves exactly as before.
                const target = findSymbol(placement.symbols, sym.name) orelse blk: {
                    if (sym.defined and sym.local and sym.section_index != std.math.maxInt(u32)) {
                        const sp2 = placement.sectionPlace(oi, sym.section_index);
                        if (sp2.seg == not_placed or sp2.seg >= placement.segments.len) return error.MalformedObject;
                        break :blk placement.segments[sp2.seg].vaddr + sp2.seg_off + sym.value;
                    }
                    return error.UndefinedSymbol;
                };
                // Offset of the patched word within this segment's bytes. Derive both the
                // in-segment write offset and the site's runtime address from the segment
                // actually being patched, so they stay in lockstep (see the aarch64 backend
                // for the rationale).
                const site = std.math.add(u64, sp.seg_off, r.offset) catch return error.MalformedObject;
                const site_addr = seg.vaddr + site;
                switch (r.type) {
                    .pc32, .plt32 => {
                        const resolved = @as(i64, @intCast(target)) + r.addend - @as(i64, @intCast(site_addr));
                        try applyDisp32(seg.bytes, site, resolved);
                    },
                    // A GOTPCRELX/REX_GOTPCRELX site only reaches `applyRelocs` when its target
                    // is defined IN-IMAGE: a shared-export one is diverted to a `.got` slot (a
                    // data import) and filtered out before this pass. For an in-image,
                    // non-preemptible symbol the psABI lets the linker relax the GOT indirection
                    // away, exactly as a real linker does - rewrite `mov sym@GOTPCREL(%rip), %reg`
                    // to `lea sym(%rip), %reg` and patch the disp32 as a plain PC32 to the symbol.
                    .gotpcrelx, .rex_gotpcrelx => {
                        try relaxGotMovToLea(seg.bytes, site);
                        const resolved = @as(i64, @intCast(target)) + r.addend - @as(i64, @intCast(site_addr));
                        try applyDisp32(seg.bytes, site, resolved);
                    },
                    else => return error.UnsupportedReloc,
                }
            }
        }
    }
}

/// Relax a GOT-indirect `mov sym@GOTPCREL(%rip), %reg` at disp32 image offset `site` into a
/// direct `lea sym(%rip), %reg`, the x86-64 psABI GOTPCRELX/REX_GOTPCRELX relaxation for a
/// locally-bound target. The `mov r64, m64` opcode byte (`8B`) sits two bytes ahead of the
/// disp32 (opcode, ModRM, then disp32), and `lea r64, m` (`8D`) shares the exact ModRM (a
/// `mod=00 rm=101` RIP-relative operand), so only the opcode byte flips; the caller then writes
/// the disp32 as a plain PC32 to the symbol. Only the `mov` form is relaxed here - the sole one
/// a real `crt1.o` uses for an in-image target. Any other opcode (a `call`/`jmp` indirect form,
/// which never targets an in-image symbol) is fail-closed as unsupported, never mis-patched.
fn relaxGotMovToLea(code: []u8, site: u64) Error!void {
    const s = std.math.cast(usize, site) orelse return error.MalformedObject;
    if (s < 2 or s > code.len or 4 > code.len - s) return error.MalformedObject;
    if (code[s - 2] != 0x8b) return error.UnsupportedReloc;
    code[s - 2] = 0x8d;
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
