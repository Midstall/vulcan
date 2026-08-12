//! Dynamic-ELF emitter + `.so` reader for the shared linker. This is the
//! foundation for dynamic linking: it produces a loader-valid ET_DYN shared object (a
//! `.so`) or a minimal dynamically-linked ET_EXEC, laying out the dynamic sections
//! (`.hash`, `.dynsym`, `.dynstr`, `.dynamic`) and the matching program headers
//! (PT_DYNAMIC, and PT_INTERP for an exec) around the linked code image. It also reads an
//! ET_DYN back (`readSharedExports`) so a later link can bind against a `.so`'s exports.
//! It also synthesizes function imports (PLT/GOT) and data imports (GOT).
//!
//! Layout model: the whole file maps at `opts.base`, so a byte at file offset `o` has
//! runtime address `base + o`. That makes every `DT_*` address (which is a VADDR, not a
//! file offset) simply `base + <its file offset>`, and keeps `p_offset`/`p_vaddr`
//! congruent modulo the page size for free. The code image is placed page-aligned so the
//! backend's PC-relative / page-relative relocations (patched by the caller at a
//! page-aligned load) stay valid at this final address.
//!
//! `std`-only; it defines its own dynamic ELF constants and imports only `elf.zig` for the
//! shared bounds-checked readers.

const std = @import("std");
const elf = @import("elf.zig");
const aarch64 = @import("arch/aarch64.zig");
const x86_64 = @import("arch/x86_64.zig");
const x86 = @import("arch/x86.zig");
const riscv64 = @import("arch/riscv64.zig");

pub const Error = elf.Error;

/// The per-architecture PLT/GOT specifics the import synthesis in `emit` needs: the
/// `JUMP_SLOT` relocation code, the PLT-stub encoder, and the call-site redirect. Extracting
/// these behind one small vtable keeps `emit`'s layout/relocation bookkeeping architecture-
/// generic; each backend (`arch/<a>.zig`) owns its own bit math. `pltEntry` returns a
/// 16-byte stub jumping through the GOT slot at `got_slot_vaddr` (its own address is
/// `plt_vaddr`); `redirectCall` rewrites the import call at `site` (runtime address
/// `site_vaddr`) to reach `target_plt_vaddr`. The signatures match each backend's existing
/// `pltEntry`/`patchCall26`/`redirectCall`, so aarch64 stays byte-identical.
/// `redirectCallNear` is riscv64-only (null everywhere else): it redirects a single `.jal`
/// import site (an `ImportCall` with `near = true`), the near-call analog of `redirectCall`'s
/// `.call` auipc+jalr pair patch.
const DynArch = struct {
    jump_slot: u32,
    pltEntry: *const fn (got_slot_vaddr: u64, plt_vaddr: u64) [16]u8,
    redirectCall: *const fn (code: []u8, site: u64, site_vaddr: u64, target_plt_vaddr: u64) Error!void,
    redirectCallNear: ?*const fn (code: []u8, site: u64, site_vaddr: u64, target_plt_vaddr: u64) Error!void = null,
};

/// The `DynArch` for `arch`. All four backends now supply a PLT-stub encoder, a JUMP_SLOT
/// reloc code, and a call redirect; aarch64's entry names the exact functions its PLT
/// loop used to call directly, so its emitted bytes are unchanged. riscv64 reuses its own
/// static extern stub (`auipc`/`ld`/`jr`) for the PLT entry.
fn dynArch(arch: elf.Arch) Error!DynArch {
    return switch (arch) {
        .aarch64 => .{
            .jump_slot = aarch64.R_AARCH64_JUMP_SLOT,
            .pltEntry = aarch64.pltEntry,
            .redirectCall = aarch64.patchCall26,
        },
        .x86_64 => .{
            .jump_slot = x86_64.R_X86_64_JUMP_SLOT,
            .pltEntry = x86_64.pltEntry,
            .redirectCall = x86_64.redirectCall,
        },
        .x86 => .{
            .jump_slot = x86.R_386_JMP_SLOT,
            .pltEntry = x86.pltEntry,
            .redirectCall = x86.redirectCall,
        },
        .riscv64 => .{
            .jump_slot = riscv64.R_RISCV_JUMP_SLOT,
            .pltEntry = riscv64.pltEntry,
            .redirectCall = riscv64.redirectCall,
            .redirectCallNear = riscv64.redirectCallNear,
        },
    };
}

// ELF object types (`e_type`).
pub const ET_EXEC: u16 = 2;
pub const ET_DYN: u16 = 3;

// Program header types (`p_type`).
pub const PT_LOAD: u32 = 1;
pub const PT_DYNAMIC: u32 = 2;
pub const PT_INTERP: u32 = 3;
// `PT_PHDR` locates the program header table itself in the runtime image. A PIE (ET_DYN exec)
// needs it: glibc's `ld.so` derives the main executable's load bias from it (`l_addr =
// AT_PHDR - PT_PHDR.p_vaddr`); without it the bias defaults to 0 and every base-0 vaddr is
// dereferenced raw (SIGSEGV at a low address). Per the ELF spec it must precede any PT_LOAD.
pub const PT_PHDR: u32 = 6;

// Program header flags (`p_flags`): R=4, W=2, X=1.
pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

// Symbol binding/type nibbles and the undefined section index.
pub const SHN_UNDEF: u16 = 0;
pub const STB_GLOBAL: u8 = 1;
/// A WEAK binding. glibc exports many standard functions WEAK (e.g. `vsnprintf`), so a shared
/// object's exports must count WEAK defined symbols too, not only STB_GLOBAL ones.
pub const STB_WEAK: u8 = 2;
pub const STT_FUNC: u8 = 2;
pub const STT_OBJECT: u8 = 1;

// `.dynamic` tags (`d_tag`). Their `d_val` payloads are VADDRs for the address tags.
pub const DT_NULL: i64 = 0;
pub const DT_NEEDED: i64 = 1;
pub const DT_HASH: i64 = 4;
pub const DT_STRTAB: i64 = 5;
pub const DT_SYMTAB: i64 = 6;
pub const DT_STRSZ: i64 = 10;
pub const DT_SYMENT: i64 = 11;
pub const DT_SONAME: i64 = 14;
// PLT/GOT import tags. `DT_PLTGOT`/`DT_JMPREL` payloads are VADDRs; `DT_PLTRELSZ`
// is a byte size; `DT_PLTREL` names the reloc form (`DT_RELA`); the two `DT_FLAGS*` request
// eager binding so the loader resolves every JUMP_SLOT before entering `_start`.
pub const DT_PLTRELSZ: i64 = 2;
pub const DT_PLTGOT: i64 = 3;
pub const DT_RELA: i64 = 7;
// `.rela.dyn` (GOT `GLOB_DAT` data-import relocations) tags: `DT_RELA` is the table's VADDR
// (defined above, and doubling as `DT_PLTREL`'s value on ELF64), `DT_RELASZ` its byte size,
// `DT_RELAENT` one entry's size (`Elf64_Rela` = 24).
pub const DT_RELASZ: i64 = 8;
pub const DT_RELAENT: i64 = 9;
// `DT_REL` (17) is the ELF32/i386 `.rel.plt` reloc form (`SHT_REL`, no addend): `DT_PLTREL`
// carries this value on x86 in place of ELF64's `DT_RELA`. As a TAG, `DT_REL` also names the
// `.rel.dyn` table VADDR (the GOT `GLOB_DAT` data-import relocations on ELF32), with `DT_RELSZ`
// its byte size and `DT_RELENT` one `Elf32_Rel` (8) - the REL analog of ELF64's
// `DT_RELA`/`DT_RELASZ`/`DT_RELAENT`.
pub const DT_REL: i64 = 17;
pub const DT_RELSZ: i64 = 18;
pub const DT_RELENT: i64 = 19;
pub const DT_PLTREL: i64 = 20;
pub const DT_JMPREL: i64 = 23;
pub const DT_FLAGS: i64 = 30;
pub const DT_FLAGS_1: i64 = 0x6ffffffb;
pub const DF_BIND_NOW: u64 = 0x8;
pub const DF_1_NOW: u64 = 0x1;

/// Sizes of the PLT/GOT structures the import path synthesizes.
const rela_size: u64 = 24; // Elf64_Rela
const plt_entry_size: u64 = 16; // one PLT stub (aarch64 4-insn / x86-64 jmp+nop pad)
const got_slot_size: u64 = 8; // one `.got.plt` slot

/// Sizes of the fixed ELF64 structures this emitter writes.
const ehdr_size: u64 = 64;
const phdr_size: u64 = 56;
const dyn_size: u64 = 16; // Elf64_Dyn
const sym_size: u64 = 24; // Elf64_Sym

/// Sizes of the fixed ELF32 (i386) structures the `emit32` path writes. `Elf32_Rel` is 8
/// bytes with NO addend field (`SHT_REL`, unlike ELF64's 24-byte `Elf32_Rela`); the GOT slot
/// is 4 bytes; the PLT stub stays 16 bytes (`ff 25 <abs32>` + nop pad). `Elf32_Sym` (16B)
/// orders its fields differently from `Elf64_Sym` (st_value/st_size come BEFORE st_info).
const rel_size32: u64 = 8; // Elf32_Rel (r_offset:u32, r_info:u32)
const plt_entry_size32: u64 = 16; // one i386 PLT stub
const got_slot_size32: u64 = 4; // one `.got.plt` slot
const ehdr_size32: u64 = 52;
const phdr_size32: u64 = 32;
const dyn_size32: u64 = 8; // Elf32_Dyn (d_tag:i32, d_val:u32)
const sym_size32: u64 = 16; // Elf32_Sym

/// Which kind of dynamic image to emit.
pub const DynMode = enum { shared, exec };

/// The knobs `linkDynamic` accepts. `base` is the runtime address the whole file maps at
/// (and therefore the base every `DT_*` VADDR is measured from). `interp` is required for
/// `mode == .exec` (the PT_INTERP path); `soname` names a `mode == .shared` object;
/// `needed` lists the shared objects an exec depends on (emitted as `DT_NEEDED`).
///
/// `pie` (only meaningful with `mode == .exec`) requests a position-independent executable:
/// the emitter marks the file `ET_DYN` (not `ET_EXEC`) and lays it out at base 0, so every
/// `DT_*` VADDR, every reloc `r_offset`, and `e_entry` become base-0 vaddrs (== file offsets)
/// that the real `ld.so` biases by its chosen (ASLR) load address. The PT_INTERP, entry,
/// `DT_NEEDED`, and PLT/GOT machinery are all still emitted, since a PIE is a dynamic exec that
/// happens to be relocatable. The aarch64/x86-64/riscv64/i386 code is position-independent
/// (PC-relative calls, page-relative GOT/PLT), so a function-only PIE needs no extra RELATIVE
/// relocations. `R_*_RELATIVE` relocations are added for pointer-initialized data.
pub const DynOptions = struct {
    mode: DynMode,
    interp: ?[]const u8 = null,
    soname: ?[]const u8 = null,
    needed: []const []const u8 = &.{},
    base: u64 = 0x400000,
    entry: []const u8 = "_start",
    pie: bool = false,
};

/// One link input for the dynamic path: a relocatable object, an `.a` archive, or a
/// shared object to link against. `.shared` is parsed (validated via
/// `readSharedExports`), and the resolver binds imports against it.
pub const DynInput = union(enum) {
    object: []const u8,
    archive: []const u8,
    shared: []const u8,
};

/// One imported function bound against a `.shared` input: the undefined symbol's
/// `name` (added to `.dynsym` as UNDEF + `.dynstr`) and the providing object's `soname`
/// (added to `DT_NEEDED`, deduped). Each import gets a `.plt` entry, a `.got.plt` slot, and
/// a `JUMP_SLOT` relocation in the PLT reloc table (ELF64 `.rela.plt` / ELF32 `.rel.plt`).
pub const Import = struct {
    name: []const u8,
    soname: []const u8,
};

/// One imported data global bound against a `.shared` input (aarch64): the undefined
/// symbol's `name` (added to `.dynsym` as UNDEF STT_OBJECT + `.dynstr`) and the providing
/// object's `soname` (added to `DT_NEEDED`, deduped alongside function imports). Each data
/// import gets an 8-byte `.got` slot (zero-init, in the writable segment) and a
/// `R_AARCH64_GLOB_DAT` relocation in `.rela.dyn` so a real `ld.so` writes the symbol's
/// runtime address into that slot at load; the GOT-indirect `adrp`/`ldr` refs are patched to
/// address the slot.
pub const DataImport = struct {
    name: []const u8,
    soname: []const u8,
};

/// Which GOT-indirect instruction field a `DataImportRef` patches. aarch64 uses a two-part
/// `adrp`/`ldr` sequence: `.got_pg` patches the `adrp` (to the GOT slot's page), `.got_lo12`
/// the `ldr`'s imm12 (the slot's lo12>>3). x86-64 uses a single `mov rd, [rip+disp32]`:
/// `.got_pcrel` patches its disp32 (rip-relative to the GOT slot). i386 uses a single
/// `mov rd, [abs32]`: `.got_abs` patches its abs32 field to the GOT slot's ABSOLUTE vaddr (no
/// rip-relative form on i386). riscv64 uses a two-part `auipc`/`ld` sequence: `.got_hi20`
/// patches the `auipc` (to the GOT slot's page-relative high 20 bits), `.got_lo12_i` the `ld`'s
/// I-immediate (the low 12 bits of the SAME PC-relative delta the `auipc` used, so the `ld`
/// loads the GOT slot). Each arch produces only its own kind(s) (the resolve layer classifies
/// by reloc type), so the patch dispatch below stays per-arch.
pub const DataRefKind = enum { got_pg, got_lo12, got_pcrel, got_abs, got_hi20, got_lo12_i };

/// A GOT-indirect reference in the code image that must be patched to address a data
/// import's `.got` slot: `site` is its byte offset within the code image, `import_index`
/// selects which data import (hence which GOT slot), `kind` which instruction half.
pub const DataImportRef = struct {
    site: u64,
    import_index: u32,
    kind: DataRefKind,
};

/// A call site in the code image that must be redirected to an import's PLT entry: `site`
/// is the call site's byte offset within the code image, `import_index` selects which import
/// (hence which PLT entry) it calls. `near` marks a riscv64 `.jal` site (a single-instruction
/// call, +/-1MiB reach) so the emitter patches just that one `jal` instead of the `.call`
/// auipc+jalr pair; always `false` on the other three arches.
pub const ImportCall = struct {
    site: u64,
    import_index: u32,
    near: bool = false,
};

/// A data-section pointer-init relocation whose target is defined INTERNALLY (a
/// `R_*_ABS64` in `.data`/`.rodata`): `site` is the pointer slot's byte offset within the
/// code image, `target` the byte offset within the code image of the value it must hold (the
/// referenced symbol). The emitter turns each into a `R_*_RELATIVE` in `.rela.dyn` for a
/// PIE/`.so` (the loader writes `base + target` into the slot), or writes the target's final
/// absolute vaddr straight into the slot for a non-PIE `ET_EXEC` (no dyn reloc). Both `site`
/// and `target` are measured the same way as `Export.offset` (relative to the code image).
pub const DataFixup = struct {
    site: u64,
    target: u64,
};

/// Whether an `Export` names a callable function (`STT_FUNC`) or a data global
/// (`STT_OBJECT`). Defaults to `.func` so existing callers/literals that predate this field
/// (and any symbol whose defining section is ambiguous) keep emitting the historical
/// STT_FUNC byte.
pub const SymKind = enum { func, object };

/// One defined global from the linked code image: its name, its byte OFFSET within the
/// code image (i.e. its address when the image was linked at base 0), and whether it is a
/// function or a data object (drives the `.dynsym` `st_info` type bits for a `-shared`
/// output: a `.text` global is `.func`/STT_FUNC, a `.rodata`/`.data`/`.bss` global is
/// `.object`/STT_OBJECT). `emit` turns the offset into the final VADDR
/// `base + code_off + offset`.
pub const Export = struct {
    name: []const u8,
    offset: u64,
    kind: SymKind = .func,
};

/// The exports read back out of an ET_DYN: its `DT_SONAME` (if any) and every global,
/// defined symbol name in its `.dynsym`. Owns its strings; `deinit` frees them.
pub const SharedExports = struct {
    soname: ?[]const u8,
    symbols: [][]const u8,

    pub fn deinit(self: *SharedExports, allocator: std.mem.Allocator) void {
        if (self.soname) |s| allocator.free(s);
        for (self.symbols) |s| allocator.free(s);
        allocator.free(self.symbols);
    }
};

/// The classic SysV ELF hash of a symbol name, as `DT_HASH` bucket selection uses. With
/// today's single bucket (`nbucket == 1`) every symbol lands in bucket 0 regardless of
/// this value, but a real dynamic loader still relies on the algorithm, and a future
/// multi-bucket table will bucket by `elf_hash(name) % nbucket`.
pub fn elfHash(name: []const u8) u32 {
    var h: u32 = 0;
    for (name) |c| {
        h = (h << 4) +% c;
        const g = h & 0xf0000000;
        if (g != 0) h ^= g >> 24;
        h &= ~g;
    }
    return h;
}

/// Append `s` plus a terminating NUL to the string table `list`, returning the offset the
/// string starts at.
fn addStr(allocator: std.mem.Allocator, list: *std.ArrayList(u8), s: []const u8) Error!u32 {
    const off: u32 = @intCast(list.items.len);
    try list.appendSlice(allocator, s);
    try list.append(allocator, 0);
    return off;
}

/// One `.dynamic` entry to write.
const Dyn = struct { tag: i64, val: u64 };

/// True iff `list` already holds a string equal to `s` (used to dedup `DT_NEEDED`).
fn containsStr(list: []const []const u8, s: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, s)) return true;
    return false;
}

/// Emit a dynamic ELF64 for `params` (`e_machine` + page size) wrapping the already-linked
/// `code` image (in-memory size `memsz` >= `code.len`, the tail zero `.bss`). `image_syms`
/// are the code image's defined globals (name + image offset); for `mode == .shared` they
/// become the `.dynsym` exports, and for `mode == .exec` they resolve the entry symbol. The
/// caller owns the returned bytes.
pub fn emit(
    allocator: std.mem.Allocator,
    params: elf.ExecParams,
    code: []const u8,
    memsz: u64,
    opts: DynOptions,
    image_syms: []const Export,
    imports: []const Import,
    import_calls: []const ImportCall,
    data_imports: []const DataImport,
    data_refs: []const DataImportRef,
    data_fixups: []const DataFixup,
) Error![]u8 {
    // i386 is ELFCLASS32 + REL (not RELA): a distinct byte layout the parallel `emit32`
    // owns. Dispatch before touching any ELF64 field so the ELF64 path below is unchanged.
    if (elf.fromEMachine(params.e_machine) == .x86) {
        return emit32(allocator, params, code, memsz, opts, image_syms, imports, import_calls, data_imports, data_refs, data_fixups);
    }

    const page = params.p_align;
    const is_exec = opts.mode == .exec;
    // A PIE exec is an `ET_DYN` laid out at base 0 (the loader adds its chosen bias to every
    // vaddr). A non-PIE exec / a `.shared` keep `opts.base`. Because the whole layout below is
    // `base + <file offset>`, base 0 makes every `DT_*`/reloc `r_offset`/`e_entry` a base-0
    // vaddr (== its file offset), exactly what a real `ld.so` expects to bias for a PIE. The
    // dynamic layout maps headers at `p_vaddr = base` (`p_offset = 0`), so base 0 puts the
    // lowest vaddr at 0 with no underflow (unlike the static path's map-headers-below-base).
    const is_pie = is_exec and opts.pie;
    const base = if (is_pie) @as(u64, 0) else opts.base;
    const ni: u64 = imports.len;
    const nd: u64 = data_imports.len;
    // A PIE or a `.so` is loaded at a runtime-chosen bias, so an internal-target data pointer
    // needs a `R_*_RELATIVE` in `.rela.dyn` (the loader biases it). A non-PIE `ET_EXEC` has a
    // fixed load address, so its data pointers are resolved directly here (no dyn reloc).
    const emit_relative = is_pie or opts.mode == .shared;
    const n_reldyn: u64 = if (emit_relative) data_fixups.len else 0;
    // `.rela.dyn` holds the GOT `GLOB_DAT` data-import relocs and (PIE/`.so`) the RELATIVE
    // pointer-init relocs together.
    const total_reldyn: u64 = nd + n_reldyn;
    // The architecture (for the PLT/GOT dispatch below); every `e_machine` this emitter is
    // handed maps to a known backend. Only consumed when there are imports (`ni > 0`).
    const arch = elf.fromEMachine(params.e_machine) orelse return error.MalformedObject;

    // The `DT_NEEDED` list: `opts.needed` unioned with each import's providing soname,
    // order-preserving and deduped (the caller may pass a soname both explicitly and via an
    // import). Only an exec emits `DT_NEEDED`.
    var needed_names: std.ArrayList([]const u8) = .empty;
    defer needed_names.deinit(allocator);
    if (is_exec) {
        for (opts.needed) |n| {
            if (!containsStr(needed_names.items, n)) try needed_names.append(allocator, n);
        }
        for (imports) |imp| {
            if (!containsStr(needed_names.items, imp.soname)) try needed_names.append(allocator, imp.soname);
        }
        for (data_imports) |imp| {
            if (!containsStr(needed_names.items, imp.soname)) try needed_names.append(allocator, imp.soname);
        }
    }

    // --- Build the string table, recording every offset we will need. ---
    var dynstr: std.ArrayList(u8) = .empty;
    defer dynstr.deinit(allocator);
    try dynstr.append(allocator, 0); // index 0 is the empty string

    var soname_off: u32 = 0;
    if (!is_exec) {
        if (opts.soname) |s| soname_off = try addStr(allocator, &dynstr, s);
    }

    var needed_offs: []u32 = &.{};
    defer if (needed_offs.len > 0) allocator.free(needed_offs);
    if (needed_names.items.len > 0) {
        needed_offs = try allocator.alloc(u32, needed_names.items.len);
        for (needed_names.items, 0..) |n, i| needed_offs[i] = try addStr(allocator, &dynstr, n);
    }

    // The `.dynsym` exports: shared objects export their globals; an exec exports nothing.
    const dynsyms: []const Export = if (is_exec) &.{} else image_syms;
    var sym_name_offs: []u32 = &.{};
    defer if (sym_name_offs.len > 0) allocator.free(sym_name_offs);
    if (dynsyms.len > 0) {
        sym_name_offs = try allocator.alloc(u32, dynsyms.len);
        for (dynsyms, 0..) |e, i| sym_name_offs[i] = try addStr(allocator, &dynstr, e.name);
    }

    // The imported functions become UNDEF `.dynsym` entries (indices after the exports); a
    // real `ld.so` resolves them against the `DT_NEEDED` `.so`s and binds each JUMP_SLOT.
    var import_name_offs: []u32 = &.{};
    defer if (import_name_offs.len > 0) allocator.free(import_name_offs);
    if (ni > 0) {
        import_name_offs = try allocator.alloc(u32, imports.len);
        for (imports, 0..) |imp, i| import_name_offs[i] = try addStr(allocator, &dynstr, imp.name);
    }
    const import_dynsym_base: u64 = 1 + dynsyms.len; // first import's `.dynsym` index

    // The imported DATA globals become UNDEF STT_OBJECT `.dynsym` entries (indices after the
    // function imports); a real `ld.so` resolves each against the `DT_NEEDED` `.so`s and writes
    // its runtime address into the matching `.got` slot via a `R_AARCH64_GLOB_DAT` `.rela.dyn`.
    var data_name_offs: []u32 = &.{};
    defer if (data_name_offs.len > 0) allocator.free(data_name_offs);
    if (nd > 0) {
        data_name_offs = try allocator.alloc(u32, data_imports.len);
        for (data_imports, 0..) |imp, i| data_name_offs[i] = try addStr(allocator, &dynstr, imp.name);
    }
    const data_import_dynsym_base: u64 = 1 + dynsyms.len + ni; // first data import's index

    // Symbol count includes the mandatory null symbol at index 0, the exports, the function
    // imports, and the data imports.
    const sym_count: u64 = 1 + dynsyms.len + ni + nd;
    const dynsym_bytes: u64 = sym_count * sym_size;
    // SysV `.hash`: nbucket + nchain + bucket[nbucket] + chain[nchain]; nbucket = 1 (MVP).
    const nbucket: u64 = 1;
    const hash_bytes: u64 = 8 + (nbucket + sym_count) * 4;

    // --- Assign file offsets (runtime address = base + offset). ---
    // PT_LOAD #0 (code) + the writable PT_LOAD (`.got.plt`/`.dynamic`) + PT_DYNAMIC, plus
    // PT_INTERP for an exec and a dedicated executable PT_LOAD for `.plt` when there are imports.
    // A PIE adds a leading PT_PHDR (so `ld.so` can compute the load bias).
    var nph: u16 = 3;
    if (is_exec) nph += 1;
    if (ni > 0) nph += 1;
    if (is_pie) nph += 1;
    var off: u64 = ehdr_size + @as(u64, nph) * phdr_size;

    var interp_off: u64 = 0;
    var interp_len: u64 = 0;
    if (is_exec) {
        const ip = opts.interp orelse return error.MalformedObject;
        interp_off = off;
        interp_len = ip.len + 1; // include the terminating NUL
        off += interp_len;
    }

    off = elf.alignUp(off, 8);
    const hash_off = off;
    off += hash_bytes;

    off = elf.alignUp(off, 8);
    const dynsym_off = off;
    off += dynsym_bytes;

    const dynstr_off = off;
    off += dynstr.items.len;

    // `.rela.plt` (the eager JUMP_SLOT relocations) lives in the read-only front tables of
    // PT_LOAD #0, 8-aligned after `.dynstr`.
    off = elf.alignUp(off, 8);
    const relaplt_off = off;
    if (ni > 0) off += ni * rela_size;

    // `.rela.dyn` (the GOT `GLOB_DAT` data-import relocations) follows `.rela.plt`, also in
    // the read-only front tables, 8-aligned. When `nd == 0` this adds nothing and `code_off`
    // is unchanged (`.rela.plt` leaves `off` already 8-aligned), so the no-data-import path is
    // byte-identical.
    off = elf.alignUp(off, 8);
    const reladyn_off = off;
    if (total_reldyn > 0) off += total_reldyn * rela_size;

    // The code image maps page-aligned so the backend's page-relative relocations (patched
    // at a page-aligned load) stay valid at `base + code_off`.
    off = elf.alignUp(off, page);
    const code_off = off;
    off += code.len;

    // Past the code image's whole in-memory extent: the executable `.plt` (its own R+X
    // segment, so it never shares a page with the writable segment), then the writable
    // segment holding `.got.plt` (function imports) + `.got` (data imports) + `.dynamic`. The
    // `.plt` (when present) and the writable segment each start page-aligned so no two
    // segments with different flags share a page (nor overlap PT_LOAD #0's `.bss` tail).
    const code_mem_end = code_off + @max(@as(u64, code.len), memsz);
    var plt_off: u64 = 0;
    var gotplt_off: u64 = undefined;
    if (ni > 0) {
        plt_off = elf.alignUp(code_mem_end, page);
        gotplt_off = elf.alignUp(plt_off + ni * plt_entry_size, page);
    } else {
        // No `.plt`; the writable segment (which still holds `.got`/`.dynamic`) starts here.
        gotplt_off = elf.alignUp(code_mem_end, page);
    }
    // The data-import GOT slots follow the `.got.plt` slots (8-aligned: page base + 8*ni),
    // then `.dynamic`. With `ni == nd == 0` this collapses to `dynamic_off == gotplt_off`
    // (page-aligned), reproducing the historical no-import layout exactly.
    const got_off = gotplt_off + ni * got_slot_size;
    const dynamic_off = got_off + nd * got_slot_size; // 8-aligned
    // The writable segment spans `.got.plt` + `.got` (when present) through `.dynamic`.
    const writable_off = gotplt_off;

    // --- Build the `.dynamic` entries now that every VADDR is known. ---
    var dyns: std.ArrayList(Dyn) = .empty;
    defer dyns.deinit(allocator);
    if (is_exec) {
        for (needed_offs) |no| try dyns.append(allocator, .{ .tag = DT_NEEDED, .val = no });
    }
    try dyns.append(allocator, .{ .tag = DT_HASH, .val = base + hash_off });
    try dyns.append(allocator, .{ .tag = DT_STRTAB, .val = base + dynstr_off });
    try dyns.append(allocator, .{ .tag = DT_SYMTAB, .val = base + dynsym_off });
    try dyns.append(allocator, .{ .tag = DT_STRSZ, .val = dynstr.items.len });
    try dyns.append(allocator, .{ .tag = DT_SYMENT, .val = sym_size });
    if (!is_exec and opts.soname != null) {
        try dyns.append(allocator, .{ .tag = DT_SONAME, .val = soname_off });
    }
    // PLT/GOT import tags. `DT_PLTGOT`/`DT_JMPREL` are VADDRs; `DT_PLTREL = DT_RELA` names
    // the reloc form; `DT_FLAGS`/`DT_FLAGS_1` request eager binding (the loader resolves
    // every JUMP_SLOT before entering the entry point, so the GOT slots are live at `_start`).
    if (ni > 0) {
        try dyns.append(allocator, .{ .tag = DT_PLTGOT, .val = base + gotplt_off });
        try dyns.append(allocator, .{ .tag = DT_PLTRELSZ, .val = ni * rela_size });
        try dyns.append(allocator, .{ .tag = DT_PLTREL, .val = @intCast(DT_RELA) });
        try dyns.append(allocator, .{ .tag = DT_JMPREL, .val = base + relaplt_off });
        try dyns.append(allocator, .{ .tag = DT_FLAGS, .val = DF_BIND_NOW });
        try dyns.append(allocator, .{ .tag = DT_FLAGS_1, .val = DF_1_NOW });
    }
    // GOT data-import tags: `DT_RELA` is the `.rela.dyn` VADDR, `DT_RELASZ` its byte size,
    // `DT_RELAENT` one `Elf64_Rela` (24). The loader applies each `R_AARCH64_GLOB_DAT` (writes
    // the imported symbol's runtime address into its `.got` slot) at load, so the slot is live
    // at `_start`. Coexists with the `.rela.plt` JUMP_SLOT tags above (a program can have both).
    if (total_reldyn > 0) {
        try dyns.append(allocator, .{ .tag = DT_RELA, .val = base + reladyn_off });
        try dyns.append(allocator, .{ .tag = DT_RELASZ, .val = total_reldyn * rela_size });
        try dyns.append(allocator, .{ .tag = DT_RELAENT, .val = rela_size });
    }
    try dyns.append(allocator, .{ .tag = DT_NULL, .val = 0 });

    const dynamic_bytes: u64 = @as(u64, dyns.items.len) * dyn_size;
    off = dynamic_off + dynamic_bytes;
    const file_size = off;

    // Entry address (exec only): the entry symbol's final VADDR.
    var entry_addr: u64 = 0;
    if (is_exec) {
        entry_addr = for (image_syms) |e| {
            if (std.mem.eql(u8, e.name, opts.entry)) break base + code_off + e.offset;
        } else return error.UndefinedSymbol;
    }

    // --- Emit. ---
    const buf = try allocator.alloc(u8, @intCast(file_size));
    errdefer allocator.free(buf);
    @memset(buf, 0);
    const w = std.mem.writeInt;

    // ELF header.
    buf[0] = 0x7f;
    buf[1] = 'E';
    buf[2] = 'L';
    buf[3] = 'F';
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    // A non-PIE exec is `ET_EXEC`; a `.shared` and a PIE exec are both `ET_DYN` (a PIE is a
    // relocatable executable the loader places at a chosen base).
    w(u16, buf[16..18], if (is_exec and !opts.pie) ET_EXEC else ET_DYN, .little);
    w(u16, buf[18..20], params.e_machine, .little);
    w(u32, buf[20..24], 1, .little); // e_version
    w(u64, buf[24..32], entry_addr, .little); // e_entry
    w(u64, buf[32..40], ehdr_size, .little); // e_phoff
    w(u64, buf[40..48], 0, .little); // e_shoff (no section headers)
    w(u16, buf[52..54], @intCast(ehdr_size), .little); // e_ehsize
    w(u16, buf[54..56], @intCast(phdr_size), .little); // e_phentsize
    w(u16, buf[56..58], nph, .little); // e_phnum
    w(u32, buf[48..52], params.e_flags, .little); // e_flags (riscv64 float ABI; 0 elsewhere)
    // e_shentsize / e_shnum / e_shstrndx stay 0 (no section header table).

    // Program headers.
    var ph: u64 = ehdr_size;
    const code_filesz = code_off + code.len;
    const code_memsz = code_off + @max(@as(u64, code.len), memsz);
    // PT_PHDR (PIE only, and it MUST come first): the program header table's own location.
    // `ld.so` reads it to derive the load bias for the whole ET_DYN executable.
    if (is_pie) {
        writePhdr(buf, ph, .{
            .p_type = PT_PHDR,
            .p_flags = PF_R,
            .p_offset = ehdr_size,
            .p_vaddr = base + ehdr_size,
            .p_filesz = @as(u64, nph) * phdr_size,
            .p_memsz = @as(u64, nph) * phdr_size,
            .p_align = 8,
        });
        ph += phdr_size;
    }
    // PT_LOAD #0: headers + dynamic read-only tables + the code image (R+W+X). The code
    // image packs `.text`/`.rodata`/`.data`/`.bss` into one blob (see
    // `computeDefaultPlacement`), so this single segment must stay writable for the same
    // reason the static path's single-segment `PT_LOAD` is R+W+X (`elf.zig`'s
    // `p_flags: u32 = 7` default): dropping W here would land any writable global
    // (`.data`/`.bss`) in a read+execute-only mapping and SIGSEGV on its first write under
    // a real `ld.so`. A proper W^X split of the text/data image is a later refinement, not
    // needed for a loadable, correct dynamic ELF matching the static path's convention.
    writePhdr(buf, ph, .{
        .p_type = PT_LOAD,
        .p_flags = PF_R | PF_W | PF_X,
        .p_offset = 0,
        .p_vaddr = base,
        .p_filesz = code_filesz,
        .p_memsz = code_memsz,
        .p_align = page,
    });
    ph += phdr_size;
    // PT_LOAD (imports only): the executable `.plt` (R+X), its own page-aligned segment.
    if (ni > 0) {
        writePhdr(buf, ph, .{
            .p_type = PT_LOAD,
            .p_flags = PF_R | PF_X,
            .p_offset = plt_off,
            .p_vaddr = base + plt_off,
            .p_filesz = ni * plt_entry_size,
            .p_memsz = ni * plt_entry_size,
            .p_align = page,
        });
        ph += phdr_size;
    }
    // PT_LOAD: the writable segment (`.got.plt` when present, then `.dynamic`) (R+W).
    const writable_bytes = (dynamic_off + dynamic_bytes) - writable_off;
    writePhdr(buf, ph, .{
        .p_type = PT_LOAD,
        .p_flags = PF_R | PF_W,
        .p_offset = writable_off,
        .p_vaddr = base + writable_off,
        .p_filesz = writable_bytes,
        .p_memsz = writable_bytes,
        .p_align = page,
    });
    ph += phdr_size;
    // PT_DYNAMIC: the same `.dynamic` region.
    writePhdr(buf, ph, .{
        .p_type = PT_DYNAMIC,
        .p_flags = PF_R | PF_W,
        .p_offset = dynamic_off,
        .p_vaddr = base + dynamic_off,
        .p_filesz = dynamic_bytes,
        .p_memsz = dynamic_bytes,
        .p_align = 8,
    });
    ph += phdr_size;
    if (is_exec) {
        // PT_INTERP: the interpreter path string.
        writePhdr(buf, ph, .{
            .p_type = PT_INTERP,
            .p_flags = PF_R,
            .p_offset = interp_off,
            .p_vaddr = base + interp_off,
            .p_filesz = interp_len,
            .p_memsz = interp_len,
            .p_align = 1,
        });
        ph += phdr_size;
    }

    // PT_INTERP string content.
    if (is_exec) {
        const ip = opts.interp.?;
        @memcpy(buf[@intCast(interp_off)..][0..ip.len], ip);
        // The trailing NUL is already zero.
    }

    // `.hash` (SysV): all symbols chain from the single bucket.
    {
        const h: usize = @intCast(hash_off);
        w(u32, buf[h..][0..4], @intCast(nbucket), .little); // nbucket
        w(u32, buf[h + 4 ..][0..4], @intCast(sym_count), .little); // nchain
        const bucket0: u32 = if (sym_count > 1) 1 else 0; // head of the chain (or STN_UNDEF)
        w(u32, buf[h + 8 ..][0..4], bucket0, .little);
        const chain_base = h + 12;
        var ci: u64 = 0;
        while (ci < sym_count) : (ci += 1) {
            // Symbol 0 is the null symbol; real symbols link to the next, last terminates.
            const next: u32 = if (ci == 0) 0 else if (ci + 1 < sym_count) @intCast(ci + 1) else 0;
            w(u32, buf[chain_base + @as(usize, @intCast(ci)) * 4 ..][0..4], next, .little);
        }
    }

    // `.dynsym`: entry 0 is the null symbol (already zero); then the exports. `st_info`'s
    // type bits follow each export's `kind` (STT_OBJECT for a data global, STT_FUNC for a
    // function) rather than assuming every export is callable.
    for (dynsyms, 0..) |e, i| {
        const s: usize = @intCast(dynsym_off + (i + 1) * sym_size);
        w(u32, buf[s..][0..4], sym_name_offs[i], .little); // st_name
        buf[s + 4] = (STB_GLOBAL << 4) | (if (e.kind == .object) STT_OBJECT else STT_FUNC); // st_info
        buf[s + 5] = 0; // st_other
        w(u16, buf[s + 6 ..][0..2], 1, .little); // st_shndx (any non-UNDEF: "defined")
        w(u64, buf[s + 8 ..][0..8], base + code_off + e.offset, .little); // st_value
        w(u64, buf[s + 16 ..][0..8], 0, .little); // st_size
    }

    // `.dynsym` imports: UNDEF (`st_shndx = SHN_UNDEF`, `st_value = 0`) globals the loader binds.
    for (0..@intCast(ni)) |i| {
        const s: usize = @intCast(dynsym_off + (import_dynsym_base + i) * sym_size);
        w(u32, buf[s..][0..4], import_name_offs[i], .little); // st_name
        buf[s + 4] = (STB_GLOBAL << 4) | STT_FUNC; // st_info
        buf[s + 5] = 0; // st_other
        w(u16, buf[s + 6 ..][0..2], SHN_UNDEF, .little); // st_shndx (an import)
        w(u64, buf[s + 8 ..][0..8], 0, .little); // st_value
        w(u64, buf[s + 16 ..][0..8], 0, .little); // st_size
    }

    // `.dynsym` data imports: UNDEF STT_OBJECT globals the loader binds via `GLOB_DAT`. Same
    // UNDEF shape as a function import, but `st_info` marks STT_OBJECT (data) so `ld.so` looks
    // up a data symbol in the `DT_NEEDED` `.so`s.
    for (0..@intCast(nd)) |i| {
        const s: usize = @intCast(dynsym_off + (data_import_dynsym_base + i) * sym_size);
        w(u32, buf[s..][0..4], data_name_offs[i], .little); // st_name
        buf[s + 4] = (STB_GLOBAL << 4) | STT_OBJECT; // st_info (a data import)
        buf[s + 5] = 0; // st_other
        w(u16, buf[s + 6 ..][0..2], SHN_UNDEF, .little); // st_shndx (an import)
        w(u64, buf[s + 8 ..][0..8], 0, .little); // st_value
        w(u64, buf[s + 16 ..][0..8], 0, .little); // st_size
    }

    // `.dynstr`.
    @memcpy(buf[@intCast(dynstr_off)..][0..dynstr.items.len], dynstr.items);

    // The code image.
    @memcpy(buf[@intCast(code_off)..][0..code.len], code);

    // PLT/GOT import synthesis (arch-dispatched via `dynArch`). Each import gets: a
    // `.rela.plt` JUMP_SLOT (so `ld.so` writes the resolved address into the `.got.plt` slot
    // at load), a 16-byte `.plt` stub jumping through that slot, and a redirected call (to the
    // stub). The `.got.plt` slots stay zero (the buffer is memset; `ld.so` overwrites them
    // under BIND_NOW). The layout/relocation bookkeeping is architecture-generic; only the
    // JUMP_SLOT code, the stub encoding, and the call redirect come from the backend.
    if (ni > 0) {
        const da = try dynArch(arch);
        for (0..@intCast(ni)) |i| {
            const got_slot_vaddr = base + gotplt_off + @as(u64, i) * got_slot_size;
            const plt_vaddr = base + plt_off + @as(u64, i) * plt_entry_size;

            // `.rela.plt[i]`: r_offset = the GOT slot VADDR; r_info = (dynsym_index << 32) | JUMP_SLOT.
            const r: usize = @intCast(relaplt_off + @as(u64, i) * rela_size);
            w(u64, buf[r..][0..8], got_slot_vaddr, .little); // r_offset
            const sym_index: u64 = import_dynsym_base + i;
            w(u64, buf[r + 8 ..][0..8], (sym_index << 32) | da.jump_slot, .little); // r_info
            w(u64, buf[r + 16 ..][0..8], 0, .little); // r_addend

            // `.plt[i]`: the backend's stub jumping through the GOT slot.
            const stub = da.pltEntry(got_slot_vaddr, plt_vaddr);
            @memcpy(buf[@intCast(plt_off + @as(u64, i) * plt_entry_size)..][0..plt_entry_size], &stub);
        }
        // Redirect each import call (in the code image) to its PLT entry. A `near` site (a
        // riscv64 `.jal`) goes through `redirectCallNear`; every other site (the `.call`
        // auipc+jalr pair, or aarch64/x86-64's single-instruction call) through `redirectCall`.
        for (import_calls) |ic| {
            const site = code_off + ic.site;
            const site_vaddr = base + site;
            const target_vaddr = base + plt_off + @as(u64, ic.import_index) * plt_entry_size;
            if (ic.near) {
                const redirectNear = da.redirectCallNear orelse return error.UnsupportedReloc;
                try redirectNear(buf, site, site_vaddr, target_vaddr);
            } else {
                try da.redirectCall(buf, site, site_vaddr, target_vaddr);
            }
        }
    }

    // GOT data-import synthesis (aarch64 + x86-64; riscv64/x86 are later slices). Each data
    // import gets: a `.rela.dyn` `R_*_GLOB_DAT` (so `ld.so` writes the resolved symbol address
    // into its `.got` slot at load), and every GOT-indirect ref in the code image patched to
    // address that slot. The GLOB_DAT type and the per-instruction patch are arch-specific (an
    // aarch64 `adrp`/`ldr` pair vs an x86-64 rip-relative `mov`); the `.got` slot size (8) and
    // the `.rela.dyn`/RELA machinery are shared. The `.got` slots stay zero (memset; `ld.so`
    // fills them). The resolve layer only produces `nd > 0` for these two arches.
    if (nd > 0) {
        const glob_dat: u32 = switch (arch) {
            .aarch64 => aarch64.R_AARCH64_GLOB_DAT,
            .x86_64 => x86_64.R_X86_64_GLOB_DAT,
            // RISC-V has no distinct GLOB_DAT code; `R_RISCV_64` (S + A into the slot) fills it.
            .riscv64 => riscv64.R_RISCV_64,
            else => return error.UnsupportedReloc,
        };
        for (0..@intCast(nd)) |i| {
            const got_slot_vaddr = base + got_off + @as(u64, i) * got_slot_size;
            // `.rela.dyn[i]`: r_offset = the GOT slot VADDR; r_info = (dynsym_index << 32) | GLOB_DAT.
            const r: usize = @intCast(reladyn_off + @as(u64, i) * rela_size);
            w(u64, buf[r..][0..8], got_slot_vaddr, .little); // r_offset
            const sym_index: u64 = data_import_dynsym_base + i;
            w(u64, buf[r + 8 ..][0..8], (sym_index << 32) | glob_dat, .little); // r_info
            w(u64, buf[r + 16 ..][0..8], 0, .little); // r_addend
        }
        // Patch each GOT-indirect ref in the code image to its data import's `.got` slot.
        for (data_refs) |dr| {
            const got_slot_vaddr = base + got_off + @as(u64, dr.import_index) * got_slot_size;
            const site = code_off + dr.site;
            const site_vaddr = base + site;
            switch (dr.kind) {
                .got_pg => try aarch64.patchGotPage(buf, site, site_vaddr, got_slot_vaddr),
                .got_lo12 => try aarch64.patchGotLo12(buf, site, got_slot_vaddr),
                // x86-64's `mov rd, [rip+disp32]`: disp32 = got_slot_vaddr - (site_vaddr + 4)
                // (rip-relative from the end of the 4-byte disp32 field).
                .got_pcrel => try x86_64.patchGotPcRel(buf, site, site_vaddr, got_slot_vaddr),
                // riscv64's `auipc`/`ld` GOT pair: the `auipc` (got_hi20) takes the high 20 bits
                // of `got_slot_vaddr - site_vaddr`, the adjacent `ld` (got_lo12_i, at site+4) the
                // low 12 of the SAME auipc-relative delta, so the `ld` loads the GOT slot.
                .got_hi20 => try riscv64.patchGotHi20(buf, site, site_vaddr, got_slot_vaddr),
                .got_lo12_i => try riscv64.patchGotLo12(buf, site, site_vaddr, got_slot_vaddr),
                // `.got_abs` is i386-only (ELF32/REL), handled in `emit32`; it never reaches
                // this ELF64 path (the resolve layer produces it only for `arch == .x86`).
                .got_abs => return error.UnsupportedReloc,
            }
        }
    }

    // Data-section pointer-init relocations (internal targets: `int *p = &g;`). In a PIE/`.so`
    // each becomes a `R_*_RELATIVE` in `.rela.dyn` (after any GLOB_DAT entries) so the loader
    // writes `load_bias + target` into the slot; in a non-PIE `ET_EXEC` the slot is resolved
    // directly here (the target's fixed absolute vaddr, no dyn reloc).
    if (data_fixups.len > 0) {
        for (data_fixups, 0..) |fx, i| {
            const target_vaddr = base + code_off + fx.target;
            if (emit_relative) {
                const relative_code: u32 = switch (arch) {
                    .aarch64 => aarch64.R_AARCH64_RELATIVE,
                    .x86_64 => x86_64.R_X86_64_RELATIVE,
                    .riscv64 => riscv64.R_RISCV_RELATIVE,
                    // x86 (i386, ELFCLASS32/REL) never reaches this ELF64 path (dispatched to
                    // `emit32` above); only aarch64/x86_64/riscv64 produce ELF64 data fixups.
                    else => return error.UnsupportedReloc,
                };
                const slot_vaddr = base + code_off + fx.site;
                const r: usize = @intCast(reladyn_off + (nd + @as(u64, i)) * rela_size);
                w(u64, buf[r..][0..8], slot_vaddr, .little); // r_offset
                w(u64, buf[r + 8 ..][0..8], relative_code, .little); // r_info (sym 0 | RELATIVE)
                w(u64, buf[r + 16 ..][0..8], target_vaddr, .little); // r_addend (target vaddr)
            } else {
                const s: usize = @intCast(code_off + fx.site);
                w(u64, buf[s..][0..8], target_vaddr, .little); // fixed absolute pointer value
            }
        }
    }

    // `.dynamic`.
    for (dyns.items, 0..) |d, i| {
        const s: usize = @intCast(dynamic_off + i * dyn_size);
        w(i64, buf[s..][0..8], d.tag, .little);
        w(u64, buf[s + 8 ..][0..8], d.val, .little);
    }

    return buf;
}

/// The fields of one program header `writePhdr` emits.
const PhdrDesc = struct {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
};

/// Write one `Elf64_Phdr` at byte offset `at` in `buf` (p_paddr mirrors p_vaddr).
fn writePhdr(buf: []u8, at: u64, d: PhdrDesc) void {
    const w = std.mem.writeInt;
    const p: usize = @intCast(at);
    w(u32, buf[p + 0 ..][0..4], d.p_type, .little);
    w(u32, buf[p + 4 ..][0..4], d.p_flags, .little);
    w(u64, buf[p + 8 ..][0..8], d.p_offset, .little);
    w(u64, buf[p + 16 ..][0..8], d.p_vaddr, .little);
    w(u64, buf[p + 24 ..][0..8], d.p_vaddr, .little); // p_paddr
    w(u64, buf[p + 32 ..][0..8], d.p_filesz, .little);
    w(u64, buf[p + 40 ..][0..8], d.p_memsz, .little);
    w(u64, buf[p + 48 ..][0..8], d.p_align, .little);
}

/// Write one `Elf32_Phdr` (32 bytes) at byte offset `at` in `buf`. The ELF32 program header
/// orders its fields differently from `Elf64_Phdr`: `p_flags` sits at byte 24 (AFTER
/// `p_offset`/`p_vaddr`/`p_paddr`/`p_filesz`/`p_memsz`), not right after `p_type`. p_paddr
/// mirrors p_vaddr.
fn writePhdr32(buf: []u8, at: u64, d: PhdrDesc) void {
    const w = std.mem.writeInt;
    const p: usize = @intCast(at);
    w(u32, buf[p + 0 ..][0..4], d.p_type, .little);
    w(u32, buf[p + 4 ..][0..4], @intCast(d.p_offset), .little);
    w(u32, buf[p + 8 ..][0..4], @intCast(d.p_vaddr), .little);
    w(u32, buf[p + 12 ..][0..4], @intCast(d.p_vaddr), .little); // p_paddr
    w(u32, buf[p + 16 ..][0..4], @intCast(d.p_filesz), .little);
    w(u32, buf[p + 20 ..][0..4], @intCast(d.p_memsz), .little);
    w(u32, buf[p + 24 ..][0..4], d.p_flags, .little);
    w(u32, buf[p + 28 ..][0..4], @intCast(d.p_align), .little);
}

/// The ELFCLASS32 (i386) counterpart of `emit`: same section/segment layout model, but every
/// structure is the 32-bit width and the PLT/data-import/RELATIVE reloc table is `SHT_REL`
/// (`Elf32_Rel`, no addend) tagged `DT_PLTREL = DT_REL`, not ELF64's `SHT_RELA`. The GOT slot
/// is 4 bytes; the PLT stub is i386's `ff 25 <abs32>` (absolute, via `x86.pltEntry`). Like
/// `emit`, a PIE (`opts.pie`, exec only) is an `ET_DYN` laid out at base 0 with a leading
/// `PT_PHDR` (`ld.so` derives the load bias from it); a function-only PIE needs no extra
/// relocations (the code is already position-independent), but a pointer-initialized data
/// global needs `R_386_RELATIVE` - see the data-fixup synthesis below for why the REL (no
/// addend) form needs the slot PRE-WRITTEN, unlike RELA. `emit` dispatches here for
/// `arch == .x86`; the ELF64 path is left untouched. The caller owns the returned bytes.
fn emit32(
    allocator: std.mem.Allocator,
    params: elf.ExecParams,
    code: []const u8,
    memsz: u64,
    opts: DynOptions,
    image_syms: []const Export,
    imports: []const Import,
    import_calls: []const ImportCall,
    data_imports: []const DataImport,
    data_refs: []const DataImportRef,
    data_fixups: []const DataFixup,
) Error![]u8 {
    const page = params.p_align;
    const is_exec = opts.mode == .exec;
    // A PIE exec is an `ET_DYN` at base 0 (the loader biases every vaddr); see `emit`'s note.
    const is_pie = is_exec and opts.pie;
    const base = if (is_pie) @as(u64, 0) else opts.base;
    const ni: u64 = imports.len;
    const nd: u64 = data_imports.len;
    // A PIE or a `.so` is loaded at a runtime-chosen bias, so an internal-target data pointer
    // needs a `R_386_RELATIVE` in `.rel.dyn` (the loader biases it in place - see the
    // pre-write/reloc pairing below). A non-PIE `ET_EXEC` has a fixed load address, so its
    // data pointer is resolved directly (no dyn reloc). Mirrors `emit`'s `emit_relative`.
    const emit_relative = is_pie or opts.mode == .shared;
    const n_reldyn: u64 = if (emit_relative) data_fixups.len else 0;
    // `.rel.dyn` holds the GOT `GLOB_DAT` data-import relocs and (PIE/`.so`) the RELATIVE
    // pointer-init relocs together, RELATIVE entries appended after any GLOB_DAT entries.
    const total_reldyn: u64 = nd + n_reldyn;
    const arch = elf.fromEMachine(params.e_machine) orelse return error.MalformedObject;

    // The `DT_NEEDED` list: `opts.needed` unioned with each import's providing soname (deduped,
    // order-preserving). Only an exec emits `DT_NEEDED`.
    var needed_names: std.ArrayList([]const u8) = .empty;
    defer needed_names.deinit(allocator);
    if (is_exec) {
        for (opts.needed) |n| {
            if (!containsStr(needed_names.items, n)) try needed_names.append(allocator, n);
        }
        for (imports) |imp| {
            if (!containsStr(needed_names.items, imp.soname)) try needed_names.append(allocator, imp.soname);
        }
        for (data_imports) |imp| {
            if (!containsStr(needed_names.items, imp.soname)) try needed_names.append(allocator, imp.soname);
        }
    }

    // --- Build the string table, recording every offset we will need. ---
    var dynstr: std.ArrayList(u8) = .empty;
    defer dynstr.deinit(allocator);
    try dynstr.append(allocator, 0); // index 0 is the empty string

    var soname_off: u32 = 0;
    if (!is_exec) {
        if (opts.soname) |s| soname_off = try addStr(allocator, &dynstr, s);
    }

    var needed_offs: []u32 = &.{};
    defer if (needed_offs.len > 0) allocator.free(needed_offs);
    if (needed_names.items.len > 0) {
        needed_offs = try allocator.alloc(u32, needed_names.items.len);
        for (needed_names.items, 0..) |n, i| needed_offs[i] = try addStr(allocator, &dynstr, n);
    }

    const dynsyms: []const Export = if (is_exec) &.{} else image_syms;
    var sym_name_offs: []u32 = &.{};
    defer if (sym_name_offs.len > 0) allocator.free(sym_name_offs);
    if (dynsyms.len > 0) {
        sym_name_offs = try allocator.alloc(u32, dynsyms.len);
        for (dynsyms, 0..) |e, i| sym_name_offs[i] = try addStr(allocator, &dynstr, e.name);
    }

    var import_name_offs: []u32 = &.{};
    defer if (import_name_offs.len > 0) allocator.free(import_name_offs);
    if (ni > 0) {
        import_name_offs = try allocator.alloc(u32, imports.len);
        for (imports, 0..) |imp, i| import_name_offs[i] = try addStr(allocator, &dynstr, imp.name);
    }
    const import_dynsym_base: u64 = 1 + dynsyms.len; // first import's `.dynsym` index

    // The imported DATA globals become UNDEF STT_OBJECT `.dynsym` entries (indices after the
    // function imports); a real `ld.so` resolves each against the `DT_NEEDED` `.so`s and writes
    // its runtime address into the matching `.got` slot via an `R_386_GLOB_DAT` `.rel.dyn`.
    var data_name_offs: []u32 = &.{};
    defer if (data_name_offs.len > 0) allocator.free(data_name_offs);
    if (nd > 0) {
        data_name_offs = try allocator.alloc(u32, data_imports.len);
        for (data_imports, 0..) |imp, i| data_name_offs[i] = try addStr(allocator, &dynstr, imp.name);
    }
    const data_import_dynsym_base: u64 = 1 + dynsyms.len + ni; // first data import's index

    const sym_count: u64 = 1 + dynsyms.len + ni + nd;
    const dynsym_bytes: u64 = sym_count * sym_size32;
    // SysV `.hash` (width-independent u32 words): nbucket + nchain + bucket[nbucket] + chain[nchain].
    const nbucket: u64 = 1;
    const hash_bytes: u64 = 8 + (nbucket + sym_count) * 4;

    // --- Assign file offsets (runtime address = base + offset). ---
    // A PIE adds a leading PT_PHDR (so `ld.so` can compute the load bias); see `emit`.
    var nph: u16 = 3;
    if (is_exec) nph += 1;
    if (ni > 0) nph += 1;
    if (is_pie) nph += 1;
    var off: u64 = ehdr_size32 + @as(u64, nph) * phdr_size32;

    var interp_off: u64 = 0;
    var interp_len: u64 = 0;
    if (is_exec) {
        const ip = opts.interp orelse return error.MalformedObject;
        interp_off = off;
        interp_len = ip.len + 1; // include the terminating NUL
        off += interp_len;
    }

    off = elf.alignUp(off, 4);
    const hash_off = off;
    off += hash_bytes;

    off = elf.alignUp(off, 4);
    const dynsym_off = off;
    off += dynsym_bytes;

    const dynstr_off = off;
    off += dynstr.items.len;

    // `.rel.plt` (the eager JUMP_SLOT relocations) lives in the read-only front tables of
    // PT_LOAD #0, 4-aligned after `.dynstr`.
    off = elf.alignUp(off, 4);
    const relplt_off = off;
    if (ni > 0) off += ni * rel_size32;

    // `.rel.dyn` (the GOT `GLOB_DAT` data-import relocations, plus (PIE/`.so`) the RELATIVE
    // pointer-init relocations) follows `.rel.plt`, also in the read-only front tables,
    // 4-aligned. When `total_reldyn == 0` this adds nothing and `code_off` is unchanged, so
    // the no-data-import/no-data-fixup path is byte-identical.
    off = elf.alignUp(off, 4);
    const reldyn_off = off;
    if (total_reldyn > 0) off += total_reldyn * rel_size32;

    // The code image maps page-aligned so the backend's PC-relative relocations stay valid.
    off = elf.alignUp(off, page);
    const code_off = off;
    off += code.len;

    // The executable `.plt` (its own R+X segment, functions only), then the writable segment
    // holding `.got.plt` (function imports) + `.got` (data imports) + `.dynamic`.
    const code_mem_end = code_off + @max(@as(u64, code.len), memsz);
    var plt_off: u64 = 0;
    var gotplt_off: u64 = undefined;
    if (ni > 0) {
        plt_off = elf.alignUp(code_mem_end, page);
        gotplt_off = elf.alignUp(plt_off + ni * plt_entry_size32, page);
    } else {
        // No `.plt`; the writable segment (which still holds `.got`/`.dynamic`) starts here.
        gotplt_off = elf.alignUp(code_mem_end, page);
    }
    // The data-import GOT slots follow the `.got.plt` slots (4-aligned: page base + 4*ni), then
    // `.dynamic`. With `ni == nd == 0` this collapses to `dynamic_off == gotplt_off`
    // (page-aligned), reproducing the historical no-import layout exactly.
    const got_off = gotplt_off + ni * got_slot_size32;
    const dynamic_off = got_off + nd * got_slot_size32; // 4-aligned
    // The writable segment spans `.got.plt` + `.got` (when present) through `.dynamic`.
    const writable_off = gotplt_off;

    // --- Build the `.dynamic` entries now that every VADDR is known. ---
    var dyns: std.ArrayList(Dyn) = .empty;
    defer dyns.deinit(allocator);
    if (is_exec) {
        for (needed_offs) |no| try dyns.append(allocator, .{ .tag = DT_NEEDED, .val = no });
    }
    try dyns.append(allocator, .{ .tag = DT_HASH, .val = base + hash_off });
    try dyns.append(allocator, .{ .tag = DT_STRTAB, .val = base + dynstr_off });
    try dyns.append(allocator, .{ .tag = DT_SYMTAB, .val = base + dynsym_off });
    try dyns.append(allocator, .{ .tag = DT_STRSZ, .val = dynstr.items.len });
    try dyns.append(allocator, .{ .tag = DT_SYMENT, .val = sym_size32 });
    if (!is_exec and opts.soname != null) {
        try dyns.append(allocator, .{ .tag = DT_SONAME, .val = soname_off });
    }
    // PLT/GOT import tags. `DT_PLTGOT`/`DT_JMPREL` are VADDRs; `DT_PLTREL = DT_REL` names the
    // i386 REL reloc form (no addend, unlike ELF64's `DT_RELA`); `DT_FLAGS`/`DT_FLAGS_1`
    // request eager binding (every JUMP_SLOT resolved before entering the entry point).
    if (ni > 0) {
        try dyns.append(allocator, .{ .tag = DT_PLTGOT, .val = base + gotplt_off });
        try dyns.append(allocator, .{ .tag = DT_PLTRELSZ, .val = ni * rel_size32 });
        try dyns.append(allocator, .{ .tag = DT_PLTREL, .val = @intCast(DT_REL) });
        try dyns.append(allocator, .{ .tag = DT_JMPREL, .val = base + relplt_off });
        try dyns.append(allocator, .{ .tag = DT_FLAGS, .val = DF_BIND_NOW });
        try dyns.append(allocator, .{ .tag = DT_FLAGS_1, .val = DF_1_NOW });
    }
    // `.rel.dyn` tags (ELF32/REL): `DT_REL` is the table's VADDR, `DT_RELSZ` its byte size,
    // `DT_RELENT` one `Elf32_Rel` (8). It holds each GOT `R_386_GLOB_DAT` (a data import; the
    // loader writes the imported symbol's runtime address into its `.got` slot) and each
    // internal-target `R_386_RELATIVE` (a PIE/`.so` pointer init; the loader adds its load bias
    // in place) together, so either alone (or both) makes `total_reldyn > 0`. Coexists with the
    // `.rel.plt` JUMP_SLOT tags above (a program can have both); the two use distinct tags
    // (`DT_REL` here vs `DT_JMPREL`/`DT_PLTREL` for functions).
    if (total_reldyn > 0) {
        try dyns.append(allocator, .{ .tag = DT_REL, .val = base + reldyn_off });
        try dyns.append(allocator, .{ .tag = DT_RELSZ, .val = total_reldyn * rel_size32 });
        try dyns.append(allocator, .{ .tag = DT_RELENT, .val = rel_size32 });
    }
    try dyns.append(allocator, .{ .tag = DT_NULL, .val = 0 });

    const dynamic_bytes: u64 = @as(u64, dyns.items.len) * dyn_size32;
    off = dynamic_off + dynamic_bytes;
    const file_size = off;

    // Entry address (exec only): the entry symbol's final VADDR.
    var entry_addr: u64 = 0;
    if (is_exec) {
        entry_addr = for (image_syms) |e| {
            if (std.mem.eql(u8, e.name, opts.entry)) break base + code_off + e.offset;
        } else return error.UndefinedSymbol;
    }

    // --- Emit. ---
    const buf = try allocator.alloc(u8, @intCast(file_size));
    errdefer allocator.free(buf);
    @memset(buf, 0);
    const w = std.mem.writeInt;

    // Elf32_Ehdr (52 bytes).
    buf[0] = 0x7f;
    buf[1] = 'E';
    buf[2] = 'L';
    buf[3] = 'F';
    buf[4] = 1; // ELFCLASS32
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    // A non-PIE exec is `ET_EXEC`; a `.shared` and a PIE exec are both `ET_DYN` (see `emit`).
    w(u16, buf[16..18], if (is_exec and !opts.pie) ET_EXEC else ET_DYN, .little);
    w(u16, buf[18..20], params.e_machine, .little);
    w(u32, buf[20..24], 1, .little); // e_version
    w(u32, buf[24..28], @intCast(entry_addr), .little); // e_entry
    w(u32, buf[28..32], @intCast(ehdr_size32), .little); // e_phoff
    w(u32, buf[32..36], 0, .little); // e_shoff (no section headers)
    w(u16, buf[40..42], @intCast(ehdr_size32), .little); // e_ehsize
    w(u16, buf[42..44], @intCast(phdr_size32), .little); // e_phentsize
    w(u16, buf[44..46], nph, .little); // e_phnum
    // e_shentsize / e_shnum / e_shstrndx stay 0 (no section header table).

    // Program headers.
    var ph: u64 = ehdr_size32;
    const code_filesz = code_off + code.len;
    const code_memsz = code_off + @max(@as(u64, code.len), memsz);
    // PT_PHDR (PIE only, and it MUST come first): the program header table's own location, so
    // `ld.so` can derive the ET_DYN executable's load bias. See `emit`.
    if (is_pie) {
        writePhdr32(buf, ph, .{
            .p_type = PT_PHDR,
            .p_flags = PF_R,
            .p_offset = ehdr_size32,
            .p_vaddr = base + ehdr_size32,
            .p_filesz = @as(u64, nph) * phdr_size32,
            .p_memsz = @as(u64, nph) * phdr_size32,
            .p_align = 4,
        });
        ph += phdr_size32;
    }
    // PT_LOAD #0: headers + dynamic read-only tables + the code image (R+W+X, matching the
    // static ELF32 path's single-segment convention so writable globals never land read-only).
    writePhdr32(buf, ph, .{
        .p_type = PT_LOAD,
        .p_flags = PF_R | PF_W | PF_X,
        .p_offset = 0,
        .p_vaddr = base,
        .p_filesz = code_filesz,
        .p_memsz = code_memsz,
        .p_align = page,
    });
    ph += phdr_size32;
    // PT_LOAD (imports only): the executable `.plt` (R+X), its own page-aligned segment.
    if (ni > 0) {
        writePhdr32(buf, ph, .{
            .p_type = PT_LOAD,
            .p_flags = PF_R | PF_X,
            .p_offset = plt_off,
            .p_vaddr = base + plt_off,
            .p_filesz = ni * plt_entry_size32,
            .p_memsz = ni * plt_entry_size32,
            .p_align = page,
        });
        ph += phdr_size32;
    }
    // PT_LOAD: the writable segment (`.got.plt` when present, then `.dynamic`) (R+W).
    const writable_bytes = (dynamic_off + dynamic_bytes) - writable_off;
    writePhdr32(buf, ph, .{
        .p_type = PT_LOAD,
        .p_flags = PF_R | PF_W,
        .p_offset = writable_off,
        .p_vaddr = base + writable_off,
        .p_filesz = writable_bytes,
        .p_memsz = writable_bytes,
        .p_align = page,
    });
    ph += phdr_size32;
    // PT_DYNAMIC: the same `.dynamic` region.
    writePhdr32(buf, ph, .{
        .p_type = PT_DYNAMIC,
        .p_flags = PF_R | PF_W,
        .p_offset = dynamic_off,
        .p_vaddr = base + dynamic_off,
        .p_filesz = dynamic_bytes,
        .p_memsz = dynamic_bytes,
        .p_align = 4,
    });
    ph += phdr_size32;
    if (is_exec) {
        // PT_INTERP: the interpreter path string.
        writePhdr32(buf, ph, .{
            .p_type = PT_INTERP,
            .p_flags = PF_R,
            .p_offset = interp_off,
            .p_vaddr = base + interp_off,
            .p_filesz = interp_len,
            .p_memsz = interp_len,
            .p_align = 1,
        });
        ph += phdr_size32;
    }

    // PT_INTERP string content.
    if (is_exec) {
        const ip = opts.interp.?;
        @memcpy(buf[@intCast(interp_off)..][0..ip.len], ip);
        // The trailing NUL is already zero.
    }

    // `.hash` (SysV): all symbols chain from the single bucket.
    {
        const h: usize = @intCast(hash_off);
        w(u32, buf[h..][0..4], @intCast(nbucket), .little); // nbucket
        w(u32, buf[h + 4 ..][0..4], @intCast(sym_count), .little); // nchain
        const bucket0: u32 = if (sym_count > 1) 1 else 0; // head of the chain (or STN_UNDEF)
        w(u32, buf[h + 8 ..][0..4], bucket0, .little);
        const chain_base = h + 12;
        var ci: u64 = 0;
        while (ci < sym_count) : (ci += 1) {
            const next: u32 = if (ci == 0) 0 else if (ci + 1 < sym_count) @intCast(ci + 1) else 0;
            w(u32, buf[chain_base + @as(usize, @intCast(ci)) * 4 ..][0..4], next, .little);
        }
    }

    // `.dynsym`: entry 0 is the null symbol (already zero); then the exports. NOTE the
    // `Elf32_Sym` field order: st_name(u32), st_value(u32), st_size(u32), st_info(u8),
    // st_other(u8), st_shndx(u16) - st_value/st_size come BEFORE st_info, unlike `Elf64_Sym`.
    for (dynsyms, 0..) |e, i| {
        const s: usize = @intCast(dynsym_off + (i + 1) * sym_size32);
        w(u32, buf[s..][0..4], sym_name_offs[i], .little); // st_name
        w(u32, buf[s + 4 ..][0..4], @intCast(base + code_off + e.offset), .little); // st_value
        w(u32, buf[s + 8 ..][0..4], 0, .little); // st_size
        buf[s + 12] = (STB_GLOBAL << 4) | (if (e.kind == .object) STT_OBJECT else STT_FUNC); // st_info
        buf[s + 13] = 0; // st_other
        w(u16, buf[s + 14 ..][0..2], 1, .little); // st_shndx (any non-UNDEF: "defined")
    }

    // `.dynsym` imports: UNDEF (`st_shndx = SHN_UNDEF`, `st_value = 0`) globals the loader binds.
    for (0..@intCast(ni)) |i| {
        const s: usize = @intCast(dynsym_off + (import_dynsym_base + i) * sym_size32);
        w(u32, buf[s..][0..4], import_name_offs[i], .little); // st_name
        w(u32, buf[s + 4 ..][0..4], 0, .little); // st_value
        w(u32, buf[s + 8 ..][0..4], 0, .little); // st_size
        buf[s + 12] = (STB_GLOBAL << 4) | STT_FUNC; // st_info
        buf[s + 13] = 0; // st_other
        w(u16, buf[s + 14 ..][0..2], SHN_UNDEF, .little); // st_shndx (an import)
    }

    // `.dynsym` data imports: UNDEF STT_OBJECT globals the loader binds via `GLOB_DAT`. Same
    // UNDEF shape as a function import, but `st_info` marks STT_OBJECT (data) so `ld.so` looks
    // up a data symbol in the `DT_NEEDED` `.so`s. NOTE the `Elf32_Sym` field order (st_value /
    // st_size BEFORE st_info), unlike `Elf64_Sym`.
    for (0..@intCast(nd)) |i| {
        const s: usize = @intCast(dynsym_off + (data_import_dynsym_base + i) * sym_size32);
        w(u32, buf[s..][0..4], data_name_offs[i], .little); // st_name
        w(u32, buf[s + 4 ..][0..4], 0, .little); // st_value
        w(u32, buf[s + 8 ..][0..4], 0, .little); // st_size
        buf[s + 12] = (STB_GLOBAL << 4) | STT_OBJECT; // st_info (a data import)
        buf[s + 13] = 0; // st_other
        w(u16, buf[s + 14 ..][0..2], SHN_UNDEF, .little); // st_shndx (an import)
    }

    // `.dynstr`.
    @memcpy(buf[@intCast(dynstr_off)..][0..dynstr.items.len], dynstr.items);

    // The code image.
    @memcpy(buf[@intCast(code_off)..][0..code.len], code);

    // PLT/GOT import synthesis (i386 via `dynArch`). Each import gets: a `.rel.plt` JUMP_SLOT
    // (`Elf32_Rel`, no addend - the loader writes the resolved address straight into the
    // `.got.plt` slot), a 16-byte `.plt` stub, and a redirected call. The `.got.plt` slots
    // stay zero (memset; `ld.so` fills under BIND_NOW). A PIE uses `pltEntryPie` (PC-relative,
    // via the "call-next; pop" idiom): `pltEntry`'s absolute `ff 25 <abs32>` bakes in the GOT
    // slot's LINK-TIME address, which is only correct when the executable loads at the exact
    // base it was linked at (a non-PIE `ET_EXEC`) - a PIE's `ld.so`-chosen runtime base would
    // make that absolute address stale, and i386 has no RIP-relative addressing to fall back
    // on (unlike x86-64's already-position-independent `ff 25 <disp32(%rip)>`, which needs no
    // PIE variant). Non-PIE keeps the historical stub byte-for-byte.
    if (ni > 0) {
        const da = try dynArch(arch);
        for (0..@intCast(ni)) |i| {
            const got_slot_vaddr = base + gotplt_off + @as(u64, i) * got_slot_size32;
            const plt_vaddr = base + plt_off + @as(u64, i) * plt_entry_size32;

            // `.rel.plt[i]`: r_offset = the GOT slot VADDR; r_info = (dynsym_index << 8) | JUMP_SLOT.
            const r: usize = @intCast(relplt_off + @as(u64, i) * rel_size32);
            w(u32, buf[r..][0..4], @intCast(got_slot_vaddr), .little); // r_offset
            const sym_index: u32 = @intCast(import_dynsym_base + i);
            w(u32, buf[r + 4 ..][0..4], (sym_index << 8) | da.jump_slot, .little); // r_info

            // `.plt[i]`: the stub jumping through the GOT slot (PIE-safe form under `.pie`).
            const stub = if (is_pie) x86.pltEntryPie(got_slot_vaddr, plt_vaddr) else da.pltEntry(got_slot_vaddr, plt_vaddr);
            @memcpy(buf[@intCast(plt_off + @as(u64, i) * plt_entry_size32)..][0..plt_entry_size32], &stub);
        }
        // Redirect each import call (in the code image) to its PLT entry.
        for (import_calls) |ic| {
            const site = code_off + ic.site;
            const site_vaddr = base + site;
            const target_vaddr = base + plt_off + @as(u64, ic.import_index) * plt_entry_size32;
            try da.redirectCall(buf, site, site_vaddr, target_vaddr);
        }
    }

    // GOT data-import synthesis (i386, ELF32/REL). Each data import gets: a `.rel.dyn`
    // `R_386_GLOB_DAT` (so `ld.so` writes the resolved symbol address into its `.got` slot at
    // load), and every GOT-indirect `mov rd, [abs32]` in the code image patched so its abs32
    // field IS the slot's absolute vaddr (i386 has no rip-relative form; mirrors the PLT's
    // `ff 25 <abs32>`). `Elf32_Rel` (8B, no addend) unlike ELF64's `.rela.dyn`. The `.got`
    // slots stay zero (memset; `ld.so` fills them). The resolve layer only produces `nd > 0`
    // here for i386.
    if (nd > 0) {
        for (0..@intCast(nd)) |i| {
            const got_slot_vaddr = base + got_off + @as(u64, i) * got_slot_size32;
            // `.rel.dyn[i]`: r_offset = the GOT slot VADDR; r_info = (dynsym_index << 8) | GLOB_DAT.
            const r: usize = @intCast(reldyn_off + @as(u64, i) * rel_size32);
            w(u32, buf[r..][0..4], @intCast(got_slot_vaddr), .little); // r_offset
            const sym_index: u32 = @intCast(data_import_dynsym_base + i);
            w(u32, buf[r + 4 ..][0..4], (sym_index << 8) | x86.R_386_GLOB_DAT, .little); // r_info
        }
        // Patch each GOT-indirect ref in the code image to its data import's `.got` slot.
        for (data_refs) |dr| {
            const got_slot_vaddr = base + got_off + @as(u64, dr.import_index) * got_slot_size32;
            const site = code_off + dr.site;
            switch (dr.kind) {
                .got_abs => try x86.patchGotAbs(buf, site, got_slot_vaddr),
                else => return error.UnsupportedReloc, // i386 produces only `.got_abs`
            }
        }
    }

    // Data-section pointer-init relocations (internal targets: `int *p = &g;`), ELF32/REL.
    // Unlike ELF64's RELA (which carries the target vaddr in `r_addend`, so the loader
    // OVERWRITES the slot with `bias + r_addend`), `Elf32_Rel` has no addend field - the
    // loader instead does `*slot += load_bias` IN PLACE. So the slot must ALWAYS hold the
    // nominal (base-relative) target vaddr: for a non-PIE `ET_EXEC` that value already IS the
    // final address (no dyn reloc needed, base is fixed); for a PIE/`.so` the loader's
    // in-place add turns it into the final address. Both branches write the identical value
    // into the slot; only the PIE/`.so` branch ALSO appends the `R_386_RELATIVE` entry (after
    // any GLOB_DAT entries, per `total_reldyn`'s layout above).
    if (data_fixups.len > 0) {
        for (data_fixups, 0..) |fx, i| {
            const target_vaddr = base + code_off + fx.target;
            const s: usize = @intCast(code_off + fx.site);
            w(u32, buf[s..][0..4], @intCast(target_vaddr), .little); // pre-write: slot += bias (PIE/`.so`) or already final (non-PIE)
            if (emit_relative) {
                const slot_vaddr = base + code_off + fx.site;
                const r: usize = @intCast(reldyn_off + (nd + @as(u64, i)) * rel_size32);
                w(u32, buf[r..][0..4], @intCast(slot_vaddr), .little); // r_offset
                w(u32, buf[r + 4 ..][0..4], (@as(u32, 0) << 8) | x86.R_386_RELATIVE, .little); // r_info (sym 0 | RELATIVE)
            }
        }
    }

    // `.dynamic` (Elf32_Dyn: d_tag:i32, d_val:u32).
    for (dyns.items, 0..) |d, i| {
        const s: usize = @intCast(dynamic_off + i * dyn_size32);
        w(i32, buf[s..][0..4], @intCast(d.tag), .little);
        w(u32, buf[s + 4 ..][0..4], @intCast(d.val), .little);
    }

    return buf;
}

/// Map a runtime VADDR to its file offset using the ET_DYN's PT_LOAD program headers
/// (the containing segment's `p_offset + (vaddr - p_vaddr)`).
fn vaddrToOffset(so: []const u8, e_phoff: u64, e_phentsize: u16, e_phnum: u16, vaddr: u64) Error!u64 {
    var i: u16 = 0;
    while (i < e_phnum) : (i += 1) {
        const ph = try elf.tableOffset(e_phoff, i, e_phentsize);
        if (try elf.rdInt(u32, so, ph) != PT_LOAD) continue;
        const p_offset = try elf.rdInt(u64, so, ph + 8);
        const p_vaddr = try elf.rdInt(u64, so, ph + 16);
        const p_filesz = try elf.rdInt(u64, so, ph + 32);
        if (vaddr < p_vaddr) continue;
        const rel = vaddr - p_vaddr;
        if (rel >= p_filesz) continue;
        return std.math.add(u64, p_offset, rel) catch return error.MalformedObject;
    }
    return error.MalformedObject;
}

/// The ELF32 counterpart of `vaddrToOffset`: map a runtime VADDR to its file offset using
/// the ELF32 PT_LOAD program headers (`Elf32_Phdr`: p_offset:u32@4, p_vaddr:u32@8,
/// p_filesz:u32@16).
fn vaddrToOffset32(so: []const u8, e_phoff: u64, e_phentsize: u16, e_phnum: u16, vaddr: u64) Error!u64 {
    var i: u16 = 0;
    while (i < e_phnum) : (i += 1) {
        const ph = try elf.tableOffset(e_phoff, i, e_phentsize);
        if (try elf.rdInt(u32, so, ph) != PT_LOAD) continue;
        const p_offset = try elf.rdInt(u32, so, ph + 4);
        const p_vaddr = try elf.rdInt(u32, so, ph + 8);
        const p_filesz = try elf.rdInt(u32, so, ph + 16);
        if (vaddr < p_vaddr) continue;
        const rel = vaddr - p_vaddr;
        if (rel >= p_filesz) continue;
        return std.math.add(u64, p_offset, rel) catch return error.MalformedObject;
    }
    return error.MalformedObject;
}

/// The ELFCLASS32 (i386) counterpart of `readSharedExports`'s ELF64 body: the same
/// PT_DYNAMIC walk + defined-global collection, but over the 32-bit header/program-header/
/// dynamic/symbol layouts (`Elf32_Ehdr` e_phoff@28, `Elf32_Phdr` p_offset@4/p_filesz@16,
/// `Elf32_Dyn` 8B, `Elf32_Sym` 16B with st_info@12/st_shndx@14). `readSharedExports`
/// dispatches here for an ELFCLASS32 object.
fn readSharedExports32(allocator: std.mem.Allocator, so: []const u8) Error!SharedExports {
    if (so.len < 52) return error.MalformedObject;
    if (try elf.rdInt(u16, so, 16) != ET_DYN) return error.MalformedObject;

    const e_phoff = try elf.rdInt(u32, so, 28);
    const e_phentsize = try elf.rdInt(u16, so, 42);
    const e_phnum = try elf.rdInt(u16, so, 44);

    // Locate PT_DYNAMIC.
    var dyn_file_off: ?u64 = null;
    var dyn_file_size: u64 = 0;
    var i: u16 = 0;
    while (i < e_phnum) : (i += 1) {
        const ph = try elf.tableOffset(e_phoff, i, e_phentsize);
        if (try elf.rdInt(u32, so, ph) != PT_DYNAMIC) continue;
        dyn_file_off = try elf.rdInt(u32, so, ph + 4);
        dyn_file_size = try elf.rdInt(u32, so, ph + 16);
    }
    const doff = dyn_file_off orelse return error.MalformedObject;

    // Walk the `.dynamic` array (Elf32_Dyn: d_tag:i32@0, d_val:u32@4).
    var symtab_va: ?u64 = null;
    var strtab_va: ?u64 = null;
    var hash_va: ?u64 = null;
    var strsz: u64 = 0;
    var syment: u64 = sym_size32;
    var soname_off: ?u64 = null;
    const dyn_n = dyn_file_size / dyn_size32;
    var n: u64 = 0;
    while (n < dyn_n) : (n += 1) {
        const e = try elf.tableOffset(doff, n, dyn_size32);
        const tag = try elf.rdInt(i32, so, e);
        const val = try elf.rdInt(u32, so, e + 4);
        if (tag == DT_NULL) break;
        switch (tag) {
            DT_SYMTAB => symtab_va = val,
            DT_STRTAB => strtab_va = val,
            DT_HASH => hash_va = val,
            DT_STRSZ => strsz = val,
            DT_SYMENT => syment = val,
            DT_SONAME => soname_off = val,
            else => {},
        }
    }
    const sym_va = symtab_va orelse return error.MalformedObject;
    const str_va = strtab_va orelse return error.MalformedObject;
    const h_va = hash_va orelse return error.MalformedObject;
    if (syment == 0) return error.MalformedObject;

    // The string table.
    const str_off = try vaddrToOffset32(so, e_phoff, e_phentsize, e_phnum, str_va);
    const strtab = try elf.secSlice(so, str_off, strsz);

    // The symbol count is `.hash`'s nchain (u32@+4, width-independent).
    const hash_off = try vaddrToOffset32(so, e_phoff, e_phentsize, e_phnum, h_va);
    const sym_count = try elf.rdInt(u32, so, hash_off + 4);

    // Collect the defined global exports (skip the null symbol at index 0). `Elf32_Sym`:
    // st_name@0, st_value@4, st_size@8, st_info@12, st_other@13, st_shndx@14.
    const sym_off = try vaddrToOffset32(so, e_phoff, e_phentsize, e_phnum, sym_va);
    var symbols: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (symbols.items) |s| allocator.free(s);
        symbols.deinit(allocator);
    }
    var k: u64 = 1;
    while (k < sym_count) : (k += 1) {
        const e = try elf.tableOffset(sym_off, k, syment);
        const st_name = try elf.rdInt(u32, so, e);
        const st_info = try elf.rdInt(u8, so, e + 12);
        const st_shndx = try elf.rdInt(u16, so, e + 14);
        const bind = st_info >> 4;
        if (bind != STB_GLOBAL and bind != STB_WEAK) continue; // local symbols are not exports
        if (st_shndx == SHN_UNDEF) continue; // an import, not an export
        const name = try cstrAt(strtab, st_name);
        if (name.len == 0) continue;
        try symbols.append(allocator, try allocator.dupe(u8, name));
    }

    var soname: ?[]const u8 = null;
    errdefer if (soname) |s| allocator.free(s);
    if (soname_off) |so_off| {
        soname = try allocator.dupe(u8, try cstrAt(strtab, so_off));
    }

    return .{ .soname = soname, .symbols = try symbols.toOwnedSlice(allocator) };
}

/// Read a NUL-terminated string at `off` within a string table.
fn cstrAt(strtab: []const u8, off: u64) Error![]const u8 {
    const o = std.math.cast(usize, off) orelse return error.MalformedObject;
    if (o >= strtab.len) return error.MalformedObject;
    const end = std.mem.indexOfScalarPos(u8, strtab, o, 0) orelse return error.MalformedObject;
    return strtab[o..end];
}

/// Parse an ET_DYN shared object's exports: locate PT_DYNAMIC, read `DT_SYMTAB`/`DT_STRTAB`
/// (+ `DT_STRSZ`) and `DT_HASH` (whose `nchain` is the symbol count), then collect every
/// global, defined (`st_shndx != SHN_UNDEF`) symbol name plus `DT_SONAME`. `DT_*` values are
/// VADDRs, translated to file offsets via the PT_LOAD headers. Both GLOBAL and WEAK defined
/// symbols count as exports (glibc exports much of the C library WEAK). A malformed object
/// surfaces as `error.MalformedObject`. The caller owns the returned `SharedExports`.
pub fn readSharedExports(allocator: std.mem.Allocator, so: []const u8) Error!SharedExports {
    if (so.len < 16) return error.MalformedObject;
    if (!std.mem.eql(u8, so[0..4], "\x7fELF")) return error.MalformedObject;
    if (so[5] != 1) return error.MalformedObject; // ELFDATA2LSB
    // i386 shared objects are ELFCLASS32 with a distinct header/dynamic byte layout.
    if (so[4] == 1) return readSharedExports32(allocator, so);
    if (so[4] != 2) return error.MalformedObject; // ELFCLASS64
    if (so.len < 64) return error.MalformedObject;
    if (try elf.rdInt(u16, so, 16) != ET_DYN) return error.MalformedObject;

    const e_phoff = try elf.rdInt(u64, so, 32);
    const e_phentsize = try elf.rdInt(u16, so, 54);
    const e_phnum = try elf.rdInt(u16, so, 56);

    // Locate PT_DYNAMIC.
    var dyn_file_off: ?u64 = null;
    var dyn_file_size: u64 = 0;
    var i: u16 = 0;
    while (i < e_phnum) : (i += 1) {
        const ph = try elf.tableOffset(e_phoff, i, e_phentsize);
        if (try elf.rdInt(u32, so, ph) != PT_DYNAMIC) continue;
        dyn_file_off = try elf.rdInt(u64, so, ph + 8);
        dyn_file_size = try elf.rdInt(u64, so, ph + 32);
    }
    const doff = dyn_file_off orelse return error.MalformedObject;

    // Walk the `.dynamic` array.
    var symtab_va: ?u64 = null;
    var strtab_va: ?u64 = null;
    var hash_va: ?u64 = null;
    var strsz: u64 = 0;
    var syment: u64 = sym_size;
    var soname_off: ?u64 = null;
    const dyn_n = dyn_file_size / dyn_size;
    var n: u64 = 0;
    while (n < dyn_n) : (n += 1) {
        const e = try elf.tableOffset(doff, n, dyn_size);
        const tag = try elf.rdInt(i64, so, e);
        const val = try elf.rdInt(u64, so, e + 8);
        if (tag == DT_NULL) break;
        switch (tag) {
            DT_SYMTAB => symtab_va = val,
            DT_STRTAB => strtab_va = val,
            DT_HASH => hash_va = val,
            DT_STRSZ => strsz = val,
            DT_SYMENT => syment = val,
            DT_SONAME => soname_off = val,
            else => {},
        }
    }
    const sym_va = symtab_va orelse return error.MalformedObject;
    const str_va = strtab_va orelse return error.MalformedObject;
    const h_va = hash_va orelse return error.MalformedObject;
    if (syment == 0) return error.MalformedObject;

    // The string table.
    const str_off = try vaddrToOffset(so, e_phoff, e_phentsize, e_phnum, str_va);
    const strtab = try elf.secSlice(so, str_off, strsz);

    // The symbol count is `.hash`'s nchain.
    const hash_off = try vaddrToOffset(so, e_phoff, e_phentsize, e_phnum, h_va);
    const sym_count = try elf.rdInt(u32, so, hash_off + 4);

    // Collect the defined global exports (skip the null symbol at index 0).
    const sym_off = try vaddrToOffset(so, e_phoff, e_phentsize, e_phnum, sym_va);
    var symbols: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (symbols.items) |s| allocator.free(s);
        symbols.deinit(allocator);
    }
    var k: u64 = 1;
    while (k < sym_count) : (k += 1) {
        const e = try elf.tableOffset(sym_off, k, syment);
        const st_name = try elf.rdInt(u32, so, e);
        const st_info = try elf.rdInt(u8, so, e + 4);
        const st_shndx = try elf.rdInt(u16, so, e + 6);
        const bind = st_info >> 4;
        if (bind != STB_GLOBAL and bind != STB_WEAK) continue; // local symbols are not exports
        if (st_shndx == SHN_UNDEF) continue; // an import, not an export
        const name = try cstrAt(strtab, st_name);
        if (name.len == 0) continue;
        try symbols.append(allocator, try allocator.dupe(u8, name));
    }

    var soname: ?[]const u8 = null;
    errdefer if (soname) |s| allocator.free(s);
    if (soname_off) |so_off| {
        soname = try allocator.dupe(u8, try cstrAt(strtab, so_off));
    }

    return .{ .soname = soname, .symbols = try symbols.toOwnedSlice(allocator) };
}

test "elfHash matches the classic SysV algorithm on known vectors" {
    try std.testing.expectEqual(@as(u32, 0), elfHash(""));
    try std.testing.expectEqual(@as(u32, 0x61), elfHash("a"));
    try std.testing.expectEqual(@as(u32, 0x672), elfHash("ab"));
    // A longer name exercising the high-nibble fold branch stays within u32.
    _ = elfHash("a_much_longer_symbol_name_that_folds_high_bits");
}

test "readSharedExports rejects a non-ELF blob" {
    const bad = "not an elf file at all!!";
    try std.testing.expectError(error.MalformedObject, readSharedExports(std.testing.allocator, bad));
}
