//! Generic ELF relocatable-object parsing for the shared static linker. This layer
//! is architecture-agnostic and depends only on `std`: it reads raw ELF bytes into
//! the generic `ParsedObject` the link driver consumes. The per-architecture reloc
//! math, stub/GOT layout, and executable parameters live in `arch/<a>.zig`; the
//! generic driver in `resolve.zig` dispatches on `ParsedObject.arch`.

const std = @import("std");

pub const Error = std.mem.Allocator.Error || error{
    MalformedObject,
    UndefinedSymbol,
    DuplicateSymbol,
    RelocationOutOfRange,
    UnsupportedReloc,
};

// Recognized ELF section/symbol type codes.
pub const SHT_PROGBITS: u32 = 1;
pub const SHT_SYMTAB: u32 = 2;
pub const SHT_RELA: u32 = 4;
pub const SHT_NOBITS: u32 = 8;
/// i386 (and every other 32-bit-legacy ELF32 target) carries no addend field in its
/// relocation entries: the addend is embedded in the instruction/data field itself,
/// already written there by the object emitter (`x86/object.zig`). `elf.zig`'s ELF32
/// parse path leaves `Reloc.addend` at its default 0 for these; `arch/x86.zig` reads
/// the real addend straight out of the relocated field before patching it.
pub const SHT_REL: u32 = 9;
pub const SHF_WRITE: u64 = 0x1;
pub const SHF_ALLOC: u64 = 0x2;
pub const SHF_EXECINSTR: u64 = 0x4;
pub const SHN_UNDEF: u16 = 0;

// ELF machine identifiers, keyed off `e_machine`.
pub const EM_RISCV: u16 = 243;
pub const EM_AARCH64: u16 = 183;
pub const EM_X86_64: u16 = 62;
pub const EM_386: u16 = 3;

/// The architectures this linker can target. The concrete reloc appliers, stub
/// mechanism, and executable parameters for each live in `arch/<a>.zig`.
pub const Arch = enum { riscv64, aarch64, x86_64, x86 };

/// Map an ELF `e_machine` to the linker's architecture, or null if unsupported.
pub fn fromEMachine(e_machine: u16) ?Arch {
    return switch (e_machine) {
        EM_RISCV => .riscv64,
        EM_AARCH64 => .aarch64,
        EM_X86_64 => .x86_64,
        EM_386 => .x86,
        else => null,
    };
}

/// The relocation kinds the linker understands. The numeric tags are the
/// architectural `R_<ARCH>_*` codes (RISC-V and AArch64 codes never collide, so one
/// enum serves every architecture); each `arch/<a>.zig` backend only ever sees the
/// variants its own objects produce.
pub const RelocType = enum(u32) {
    /// `R_RISCV_JAL`: patch a `jal`'s 20-bit immediate (a +/-1MiB call).
    jal = 17,
    /// `R_RISCV_CALL`: patch an `auipc`+`jalr` pair (a long call).
    call = 18,
    /// `R_RISCV_PCREL_HI20`: the high 20 bits of a PC-relative address.
    pcrel_hi20 = 23,
    /// `R_RISCV_PCREL_LO12_I`: the low 12 bits, for an I-type form.
    pcrel_lo12_i = 24,
    /// `R_RISCV_GOT_HI20`: the high 20 bits of the PC-relative address of a symbol's GOT
    /// entry (the `auipc` half of a GOT-indirect `auipc`/`ld` data-import pair). The paired
    /// `ld`'s low 12 bits reuse `pcrel_lo12_i` (its target is the `auipc` label). The dynamic
    /// linker synthesizes the GOT slot + a `R_RISCV_64` in `.rela.dyn` and patches the `auipc`/
    /// `ld` to the slot; the static path never sees this reloc.
    got_hi20 = 20,
    /// `R_AARCH64_ADR_PREL_PG_HI21`: patch an `adrp`'s page-relative immediate
    /// (a `global_addr`'s high half).
    adr_prel_pg_hi21 = 275,
    /// `R_AARCH64_ADD_ABS_LO12_NC`: patch an `add`'s 12-bit page-offset immediate
    /// (a `global_addr`'s low half).
    add_abs_lo12_nc = 277,
    /// `R_AARCH64_CALL26`: patch a `bl`/`b`'s 26-bit immediate (a +/-128MiB call).
    call26 = 283,
    /// `R_AARCH64_JUMP26`: patch a `b`'s 26-bit immediate (a +/-128MiB tail branch). It
    /// shares CALL26's encoding field, so the same bit math applies (`applyCall26` keeps the
    /// B-vs-BL opcode bit as it found it). A real glibc `crt1.o` uses this for `_start`'s tail
    /// branch into a local trampoline that jumps to `main`, so the autolink path
    /// needs it. It appears only in `.rela.text`, never a data section.
    jump26 = 282,
    /// `R_AARCH64_PREL32`: a 32-bit PC-relative value (`S + A - P`). A real glibc `crt1.o`
    /// carries two of these in `.rela.eh_frame` (each FDE's `initial_location` field points
    /// at a `.text` address PC-relative), so the autolink path must parse it. VCC
    /// does not register the crt's unwind tables (the autolink omits `crtbegin.o`, so nothing
    /// ever reads `.eh_frame`), so `collectDataFixups` LEAVES a `.prel32` site unrelocated
    /// rather than mis-applying it as a 64-bit pointer init. It appears only in a
    /// `.rela.rodata`/`.rela.eh_frame` (a `.rodata`-classified section), never `.rela.text`.
    prel32 = 261,
    /// `R_AARCH64_ABS64` (and, sharing this generic tag, x86-64's numerically-distinct
    /// `R_X86_64_64` and riscv64's numerically-distinct `R_RISCV_64`): a 64-bit absolute
    /// address (`S + A`) in a DATA section (`.data`/`.rodata`), from a pointer-initialized
    /// global (`int *p = &g;`). The dynamic linker turns an internal-target one into a
    /// `R_*_RELATIVE` (PIE/`.so`) or a direct absolute write (non-PIE exec). It appears only
    /// in `.rela.data`/`.rela.rodata`, never in `.rela.text`. The parse-time raw numeric code
    /// differs per arch (257 for AArch64, 1 for x86-64, 2 for riscv64 - see `parseObject64`'s
    /// arch-gated match), but all three map to this one semantic tag.
    abs64 = 257,
    /// `R_AARCH64_ADR_GOT_PAGE`: patch an `adrp`'s page-relative immediate to the page
    /// of the symbol's GOT entry (the high half of a GOT-indirect `adrp`/`ldr` pair, a
    /// data import). The dynamic linker synthesizes the GOT slot; the static path never
    /// sees this reloc.
    adr_got_page = 311,
    /// `R_AARCH64_LD64_GOT_LO12_NC`: patch a 64-bit `ldr`'s 12-bit unsigned offset to the
    /// GOT entry's lo12>>3 (the low half of a GOT-indirect `adrp`/`ldr` pair, a data import).
    ld64_got_lo12_nc = 312,
    /// `R_AARCH64_LDST64_ABS_LO12_NC`: patch a 64-bit `ldr`/`str`'s 12-bit unsigned
    /// offset to `(S + A) & 0xFFF >> 3` (the load/store is size-scaled by 8), the low half of
    /// a direct `adrp`/`ldr` pair addressing a symbol's storage. glibc's `libc_nonshared.a`
    /// `atexit` wrapper uses it to load `*__dso_handle`.
    ldst64_abs_lo12_nc = 286,
    /// `R_X86_64_PC32` (and, sharing the same numeric code, i386's `R_386_PC32`):
    /// patch a 4-byte PC-relative displacement (a `global_addr`'s
    /// `lea rd, [rip+disp32]`, or i386's `call rel32`/`mov`-based PC32 forms). The
    /// x86-64 (RELA) and i386 (REL) backends differ only in where the addend comes
    /// from: RELA carries it in the reloc entry; REL has it pre-written into the
    /// field itself (see `arch/x86.zig`).
    pc32 = 2,
    /// `R_X86_64_PLT32`: patch a 4-byte PC-relative displacement (a `call rel32`).
    /// Resolves identically to `pc32` here (no real PLT is ever generated).
    plt32 = 4,
    /// `R_386_32`: patch a plain 32-bit absolute address (i386's `global_addr`
    /// `mov rd, imm32`). No PC-relative subtraction; like `pc32` on i386, the REL
    /// addend is pre-written into the field (`object.zig` always writes 0 here).
    abs32 = 1,
    /// `R_386_GOT32`: an i386 reference to a symbol's GOT slot (a GOT-indirect
    /// `global_addr`'s `mov rd, [abs32]`, a DATA import). The dynamic linker synthesizes the
    /// GOT slot + an `R_386_GLOB_DAT` (in `.rel.dyn`) and patches the `mov`'s abs32 field to
    /// the slot's absolute vaddr; the static path never sees this reloc. The ELF32/REL analog
    /// of x86-64's `gotpcrel` (which is rip-relative), so it gets its own numeric code (3).
    got32 = 3,
    /// `R_X86_64_GOTPCREL`: a PC-relative reference to a symbol's GOT slot (a
    /// GOT-indirect `global_addr`'s `mov rd, [rip+disp32]`, a DATA import). The dynamic
    /// linker synthesizes the GOT slot + a `R_X86_64_GLOB_DAT` and patches the disp32 to
    /// the slot; the static path never sees this reloc (it targets no in-image symbol).
    gotpcrel = 9,
};

/// A relocation applied to a `.text` offset against a symbol.
pub const Reloc = struct {
    /// Byte offset within `.text` of the instruction to patch.
    offset: u64,
    /// Index into the object's `symbols` array.
    symbol: u32,
    type: RelocType,
    addend: i64 = 0,
};

/// A symbol parsed out of one object's symbol table. `section_index` says which
/// allocatable section `value` is an offset into, as an index into
/// `ParsedObject.sections` (the arbitrary-named-section model).
pub const ObjSymbol = struct {
    name: []const u8,
    value: u64,
    defined: bool,
    local: bool,
    /// The symbol's size in bytes (`st_size`). 0 for a symbol with no known size
    /// (a label, or a fixture built before this field existed).
    size: u64 = 0,
    /// Index into the owning `ParsedObject.sections`, or `std.math.maxInt(u32)` for
    /// an undefined (external) symbol, which has no defining section.
    section_index: u32 = std.math.maxInt(u32),
};

/// One allocatable section from a parsed object. `ParsedObject.sections` keeps
/// same-class sections (for example two PROGBITS|EXECINSTR sections named `.text`
/// and `.text.extra`) as separate entries, so a symbol resolves to the exact
/// section that defines it, not a merged per-class blob.
pub const ObjSection = struct {
    name: []const u8,
    flags: u64,
    /// The file bytes of this section. Empty for a NOBITS section (`is_nobits`).
    /// `size` still gives its true in-memory extent.
    bytes: []const u8 = &.{},
    /// The section's in-memory size (`sh_size`).
    size: u64,
    is_nobits: bool,
    /// The relocations whose target is this section, `offset` a byte offset
    /// within it. `ParsedObject.deinit` owns and frees this slice.
    relocs: []Reloc = &.{},
};

/// The pieces lifted out of a single relocatable object: each allocatable
/// section's bytes (`.bss` has only a size), its symbols, and each section's
/// relocations (in `ObjSection.relocs`).
pub const ParsedObject = struct {
    arch: Arch,
    symbols: []ObjSymbol,
    /// Every `SHF_ALLOC` section of the object, one entry each, in section-header
    /// order, with no same-class collapsing. This is the arbitrary-named-section
    /// linker model: each section owns its own bytes and per-section relocs, so two
    /// same-class sections (for example two `.text` inputs) stay separate.
    sections: []ObjSection = &.{},

    pub fn deinit(self: *ParsedObject, allocator: std.mem.Allocator) void {
        allocator.free(self.symbols);
        for (self.sections) |sec| allocator.free(sec.relocs);
        allocator.free(self.sections);
    }
};

/// A resolved symbol in the linked image: its name, final absolute address, and whether the
/// section that defined it is executable (`is_exec` true for a function in a `SHF_EXECINSTR`
/// section, false for a data object in `.rodata`/`.data`/`.bss`). The dynamic exporter turns
/// this into the `.dynsym` type bits (`STT_FUNC` vs `STT_OBJECT`). Defaults to `true` so
/// existing construction sites that predate this field (literal test fixtures, synthesized
/// script symbols with no natural section) keep their historical "function" classification.
pub const ResolvedSymbol = struct { name: []const u8, address: u64, is_exec: bool = true };

/// A linked code image: the relocated `.text` (in input order) plus the symbol
/// table giving each defined symbol's absolute address. The code is meant to
/// load at `base`.
pub const Image = struct {
    code: []u8,
    symbols: []ResolvedSymbol,
    base: u64,
    /// Total in-memory size including `.bss` (>= `code.len`).
    memsz: u64,

    pub fn deinit(self: *Image, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        for (self.symbols) |s| allocator.free(s.name);
        allocator.free(self.symbols);
    }

    pub fn addressOf(self: *const Image, name: []const u8) ?u64 {
        for (self.symbols) |s| {
            if (std.mem.eql(u8, s.name, name)) return s.address;
        }
        return null;
    }
};

/// One loadable segment of the final image: a contiguous run of file bytes mapped at
/// `vaddr` (physical `paddr`), zero-extended to `memsz` (>= `bytes.len`, the tail being
/// `.bss`), with ELF `p_flags` in `flags` (R=4, W=2, X=1). A default (no-linker-script)
/// link produces exactly one segment; a linker script can place several. The bytes are
/// owned and mutable so relocation appliers can patch them in place.
pub const Segment = struct {
    vaddr: u64,
    paddr: u64,
    bytes: []u8,
    memsz: u64,
    flags: u8,
};

/// Where one (object, section) landed once addresses were assigned: its runtime address
/// `vaddr`, which `Placement.segments` index it lives in, and its byte offset within that
/// segment. A `seg == not_placed` marks an absent section (nothing to patch or reference).
pub const SecPlace = struct {
    vaddr: u64,
    seg: u32,
    seg_off: u64,
};

/// `SecPlace.seg` sentinel: this (object, section) has no bytes and was not placed.
pub const not_placed: u32 = std.math.maxInt(u32);

/// The interface between address assignment (LAYOUT) and relocation (bit math): a list
/// of loadable `Segment`s, a per-(object, section) address map (`section_places`), the
/// resolved symbol table, and the entry address (0 = unset). A per-arch
/// `computeDefaultPlacement` fills this (today's single contiguous image as one segment);
/// a per-arch `applyRelocs` patches the segment bytes; a shared `writeElfSegments` emits
/// one `PT_LOAD` per segment. Owns its segment bytes and symbol names.
pub const Placement = struct {
    segments: []Segment,
    symbols: []ResolvedSymbol,
    entry: u64 = 0,
    /// The per-(object, section) place map: a flat `SecPlace` array with one entry per
    /// section of every object, indexed `section_place_base[oi] + section_index`. Every
    /// placer (the shared default placer and the script layout engine) fills it. See
    /// `sectionPlace`.
    section_places: []SecPlace = &.{},
    /// Per-object start offset into `section_places` (length = object count + 1, the
    /// last entry the total section count). Empty when `section_places` is.
    section_place_base: []usize = &.{},

    /// The place of one (object, section) in the section-indexed map.
    pub fn sectionPlace(self: *const Placement, oi: usize, section_index: usize) SecPlace {
        return self.section_places[self.section_place_base[oi] + section_index];
    }

    pub fn deinit(self: *Placement, allocator: std.mem.Allocator) void {
        for (self.segments) |seg| allocator.free(seg.bytes);
        allocator.free(self.segments);
        allocator.free(self.section_places);
        allocator.free(self.section_place_base);
        for (self.symbols) |s| allocator.free(s.name);
        allocator.free(self.symbols);
    }
};

/// Resolves an undefined (external) symbol name to its absolute runtime address,
/// or null if unknown. The JIT uses this to bind calls to host/runtime functions.
pub const Resolver = struct {
    context: *anyopaque,
    func: *const fn (context: *anyopaque, name: []const u8) ?u64,

    pub fn resolve(self: Resolver, name: []const u8) ?u64 {
        return self.func(self.context, name);
    }
};

/// Per-architecture parameters for wrapping a code image in a static ELF executable.
/// `writeElfExec` in `resolve.zig` fills the fixed ELF64 layout from these.
pub const ExecParams = struct {
    e_machine: u16,
    /// The code is placed at this page-aligned file offset so a real loader's mmap
    /// keeps p_offset and p_vaddr congruent modulo the page size.
    code_offset: u64 = 0x1000,
    p_align: u64 = 0x1000,
    p_flags: u32 = 7,
    /// When false (the default), only the code itself (file range
    /// `[code_offset, code_offset+code.len)`) is mapped, at `p_vaddr = base`. When
    /// true, the whole file is mapped from offset 0 (headers included) one page
    /// below `base`, so `base` still lands exactly `code_offset` bytes into the
    /// mapping while a real loader still sees the ELF/program headers mapped
    /// (AArch64 wants this; RISC-V does not).
    map_headers: bool = false,
    /// The ELF header's `e_flags`. Zero for most targets, but riscv64 encodes its
    /// float ABI here: a real riscv64 glibc `ld.so` is `lp64d` and rejects a shared
    /// object/exe whose float ABI does not match (`elf_machine_matches_host` fails and
    /// the loader reports the library as not found), so riscv64 sets
    /// `EF_RISCV_FLOAT_ABI_DOUBLE` (0x4). Consumed by the dynamic emitter; the static
    /// path leaves `e_flags` zero (its executables run bare under `qemu-user`, not a
    /// strict `ld.so`, so this stays byte-identical there).
    e_flags: u32 = 0,
};

pub fn alignUp(v: u64, a: u64) u64 {
    return std.mem.alignForward(u64, v, a);
}

/// `base + index*stride`, rejecting the overflow that would otherwise wrap a bounds
/// check on attacker-controlled ELF offset/size fields into an out-of-bounds access.
pub fn tableOffset(base: u64, index: u64, stride: u64) Error!u64 {
    const scaled = std.math.mul(u64, index, stride) catch return error.MalformedObject;
    return std.math.add(u64, base, scaled) catch return error.MalformedObject;
}

/// The `[off, off+size)` slice of `buf`, bounds-checked without overflowing.
pub fn secSlice(buf: []const u8, off: u64, size: u64) Error![]const u8 {
    const o = std.math.cast(usize, off) orelse return error.MalformedObject;
    const s = std.math.cast(usize, size) orelse return error.MalformedObject;
    if (o > buf.len or s > buf.len - o) return error.MalformedObject;
    return buf[o..][0..s];
}

pub fn rdInt(comptime T: type, buf: []const u8, off: u64) Error!T {
    const o = std.math.cast(usize, off) orelse return error.MalformedObject;
    if (o > buf.len or @sizeOf(T) > buf.len - o) return error.MalformedObject;
    return std.mem.readInt(T, buf[o..][0..@sizeOf(T)], .little);
}

/// Classify a section header by its flags into a broad placement class, for the
/// arbitrary-named-section model. Any allocatable section type gets a class, so
/// `ParsedObject.sections` can hold a kind a strict PROGBITS filter would drop (for
/// example `.init_array`). The placement and symbol-kind passes reuse this.
pub fn sectionClass(sh_type: u32, sh_flags: u64) enum { exec, ro, rw, nobits, ignore } {
    if ((sh_flags & SHF_ALLOC) == 0) return .ignore;
    if (sh_type == SHT_NOBITS) return .nobits;
    if ((sh_flags & SHF_EXECINSTR) != 0) return .exec;
    if ((sh_flags & SHF_WRITE) != 0) return .rw;
    return .ro;
}

/// Read a NUL-terminated string from a string table at `off`.
fn strAt(strtab: []const u8, off: u32) Error![]const u8 {
    if (off >= strtab.len) return error.MalformedObject;
    const end = std.mem.indexOfScalarPos(u8, strtab, off, 0) orelse return error.MalformedObject;
    return strtab[off..end];
}

/// Parse one relocatable object into its `.text`, symbols, and relocations. Dispatches
/// on the ELF class byte: ELFCLASS64 (riscv64/aarch64/x86_64, all `SHT_RELA`) goes
/// through `parseObject64`; ELFCLASS32 (i386, `SHT_REL`) goes through `parseObject32`.
/// The architecture is read from `e_machine`.
pub fn parseObject(allocator: std.mem.Allocator, buf: []const u8) Error!ParsedObject {
    if (buf.len < 20) return error.MalformedObject;
    if (!std.mem.eql(u8, buf[0..4], "\x7fELF")) return error.MalformedObject;
    if (buf[5] != 1) return error.MalformedObject; // ELFDATA2LSB
    return switch (buf[4]) {
        2 => parseObject64(allocator, buf),
        1 => parseObject32(allocator, buf),
        else => error.MalformedObject,
    };
}

/// Parse one ELF64/RELA relocatable object (riscv64, aarch64, x86_64) into its
/// `.text`, symbols, and relocations. Section roles are identified by type (not by
/// name) for robustness. Names borrow from `buf`.
fn parseObject64(allocator: std.mem.Allocator, buf: []const u8) Error!ParsedObject {
    if (buf.len < 64) return error.MalformedObject;
    const arch = fromEMachine(try rdInt(u16, buf, 18)) orelse return error.MalformedObject;

    const shoff = try rdInt(u64, buf, 40);
    const shentsize = try rdInt(u16, buf, 58);
    const shnum = try rdInt(u16, buf, 60);

    // The section-name string table (`.shstrtab`), located via `e_shstrndx`. It feeds
    // the `sections` table below. A missing/out-of-range index (never happens for a real
    // object) just leaves every section entry unnamed rather than failing the whole parse.
    const e_shstrndx = try rdInt(u16, buf, 62);
    const shstrtab: []const u8 = blk: {
        if (e_shstrndx == 0 or e_shstrndx >= shnum) break :blk &.{};
        const shstr_hdr = try tableOffset(shoff, e_shstrndx, shentsize);
        const shstr_off = try rdInt(u64, buf, shstr_hdr + 24);
        const shstr_size = try rdInt(u64, buf, shstr_hdr + 32);
        break :blk try secSlice(buf, shstr_off, shstr_size);
    };

    // Pick up the symbol/relocation tables.
    var symtab_ndx: ?u16 = null;
    var rela_ndxs: std.ArrayList(u16) = .empty;
    defer rela_ndxs.deinit(allocator);

    // The arbitrary-named-section table (every `SHF_ALLOC` section, no same-class
    // collapsing), plus a map from the raw section-header index to its index within
    // `sections` (`maxInt` for a section `sections` does not carry).
    var sections: std.ArrayList(ObjSection) = .empty;
    errdefer {
        for (sections.items) |sec| allocator.free(sec.relocs);
        sections.deinit(allocator);
    }
    var sec_map = try allocator.alloc(u32, shnum);
    defer allocator.free(sec_map);
    @memset(sec_map, std.math.maxInt(u32));

    var i: u16 = 0;
    while (i < shnum) : (i += 1) {
        const hdr = try tableOffset(shoff, i, shentsize);
        const sh_name = try rdInt(u32, buf, hdr + 0);
        const typ = try rdInt(u32, buf, hdr + 4);
        const flags = try rdInt(u64, buf, hdr + 8);
        const sh_off = try rdInt(u64, buf, hdr + 24);
        const sh_size = try rdInt(u64, buf, hdr + 32);
        const class = sectionClass(typ, flags);
        if (class != .ignore) {
            const is_nobits = class == .nobits;
            const bytes: []const u8 = if (is_nobits) &.{} else try secSlice(buf, sh_off, sh_size);
            sec_map[i] = @intCast(sections.items.len);
            try sections.append(allocator, .{
                .name = if (shstrtab.len == 0) "" else try strAt(shstrtab, sh_name),
                .flags = flags,
                .bytes = bytes,
                .size = sh_size,
                .is_nobits = is_nobits,
            });
        }
        if (typ == SHT_SYMTAB) symtab_ndx = i;
        if (typ == SHT_RELA) try rela_ndxs.append(allocator, i);
    }
    const si = symtab_ndx orelse return error.MalformedObject;

    // Symbol table and its string table (via sh_link).
    const sym_hdr = try tableOffset(shoff, si, shentsize);
    const sym_off = try rdInt(u64, buf, sym_hdr + 24);
    const sym_size = try rdInt(u64, buf, sym_hdr + 32);
    const sym_link = try rdInt(u32, buf, sym_hdr + 40);
    const str_hdr = try tableOffset(shoff, sym_link, shentsize);
    const str_off = try rdInt(u64, buf, str_hdr + 24);
    const str_size = try rdInt(u64, buf, str_hdr + 32);
    const strtab = try secSlice(buf, str_off, str_size);

    const sym_count: usize = @intCast(sym_size / 24);
    var symbols = try allocator.alloc(ObjSymbol, sym_count);
    errdefer allocator.free(symbols);
    var k: usize = 0;
    while (k < sym_count) : (k += 1) {
        const e = try tableOffset(sym_off, k, 24);
        const st_name = try rdInt(u32, buf, e + 0);
        const st_info = try rdInt(u8, buf, e + 4);
        const st_shndx = try rdInt(u16, buf, e + 6);
        const st_value = try rdInt(u64, buf, e + 8);
        const st_size = try rdInt(u64, buf, e + 16);
        const section_index: u32 = if (st_shndx != SHN_UNDEF and st_shndx < shnum) sec_map[st_shndx] else std.math.maxInt(u32);
        symbols[k] = .{
            .name = if (st_name == 0) "" else try strAt(strtab, st_name),
            .value = st_value,
            .defined = st_shndx != SHN_UNDEF,
            .local = (st_info >> 4) == 0, // STB_LOCAL
            .size = st_size,
            .section_index = section_index,
        };
    }

    // Relocations. Each `SHT_RELA` section applies to the section named by its `sh_info`:
    // route each reloc into that target section's `ObjSection.relocs`, by section-table
    // index. A `.rela.text` feeds the `.text` section's relocs, a `.rela.data` the `.data`
    // section's, and so on. A RELA whose target is not an allocatable section (for example
    // `.rela.debug`) has no `sections` entry and its relocs are dropped (the linker never
    // patches a non-loadable section).
    var sec_relocs = try allocator.alloc(std.ArrayList(Reloc), sections.items.len);
    defer allocator.free(sec_relocs);
    for (sec_relocs) |*sr| sr.* = .empty;
    errdefer for (sec_relocs) |*sr| sr.deinit(allocator);
    for (rela_ndxs.items) |ri| {
        const rela_hdr = try tableOffset(shoff, ri, shentsize);
        const rela_off = try rdInt(u64, buf, rela_hdr + 24);
        const rela_size = try rdInt(u64, buf, rela_hdr + 32);
        const sh_info = try rdInt(u32, buf, rela_hdr + 44);
        // The section this RELA modifies, and hence which `ObjSection.relocs` list it feeds.
        const target_sec: u32 = if (sh_info < shnum) sec_map[sh_info] else std.math.maxInt(u32);
        const rela_count: usize = @intCast(rela_size / 24);
        var r: usize = 0;
        while (r < rela_count) : (r += 1) {
            const e = try tableOffset(rela_off, r, 24);
            const r_offset = try rdInt(u64, buf, e + 0);
            const r_info = try rdInt(u64, buf, e + 8);
            const r_addend = try rdInt(i64, buf, e + 16);
            const typ: u32 = @truncate(r_info & 0xffffffff);
            const sym_index: u32 = @intCast(r_info >> 32);
            const rt: RelocType = switch (typ) {
                @intFromEnum(RelocType.jal) => .jal,
                @intFromEnum(RelocType.call) => .call,
                @intFromEnum(RelocType.pcrel_hi20) => .pcrel_hi20,
                @intFromEnum(RelocType.pcrel_lo12_i) => .pcrel_lo12_i,
                @intFromEnum(RelocType.got_hi20) => .got_hi20,
                @intFromEnum(RelocType.adr_prel_pg_hi21) => .adr_prel_pg_hi21,
                @intFromEnum(RelocType.add_abs_lo12_nc) => .add_abs_lo12_nc,
                @intFromEnum(RelocType.call26) => .call26,
                @intFromEnum(RelocType.jump26) => .jump26,
                @intFromEnum(RelocType.prel32) => .prel32,
                @intFromEnum(RelocType.abs64) => .abs64,
                @intFromEnum(RelocType.adr_got_page) => .adr_got_page,
                @intFromEnum(RelocType.ld64_got_lo12_nc) => .ld64_got_lo12_nc,
                @intFromEnum(RelocType.ldst64_abs_lo12_nc) => .ldst64_abs_lo12_nc,
                // `R_RISCV_64` (numeric 2) ALSO collides with x86_64's `R_X86_64_PC32`
                // (numeric 2, `RelocType.pc32` above). riscv64 never emits a numeric-2 TEXT
                // reloc (its text relocs are jal/call/pcrel_hi20/pcrel_lo12_i/got_hi20), so a
                // riscv64 numeric-2 RELA entry appears only in `.rela.data`/`.rela.rodata` (a
                // pointer-init slot, riscv64's analog of aarch64's ABS64) - gate on `arch`,
                // mirroring the `1 =>` x86-64/i386 gate below.
                @intFromEnum(RelocType.pc32) => if (arch == .riscv64) .abs64 else .pc32,
                @intFromEnum(RelocType.plt32) => .plt32,
                @intFromEnum(RelocType.gotpcrel) => .gotpcrel,
                // `R_X86_64_64` (numeric 1): a 64-bit absolute address in a DATA section
                // (a pointer-init slot), x86-64's analog of aarch64's ABS64. Gated on
                // `arch == .x86_64` since numeric 1 is ALSO i386's `R_386_32` - but that is
                // parsed by the wholly separate ELF32 `parseObject32` path (SHT_REL, never
                // SHT_RELA), so this arm only ever fires for a real x86-64 RELA object.
                1 => if (arch == .x86_64) .abs64 else return error.UnsupportedReloc,
                else => return error.UnsupportedReloc,
            };
            const reloc: Reloc = .{
                .offset = r_offset,
                .symbol = sym_index,
                .type = rt,
                .addend = r_addend,
            };
            if (target_sec != std.math.maxInt(u32)) try sec_relocs[target_sec].append(allocator, reloc);
        }
    }

    for (sections.items, 0..) |*sec, idx| sec.relocs = try sec_relocs[idx].toOwnedSlice(allocator);

    return .{
        .arch = arch,
        .symbols = symbols,
        .sections = try sections.toOwnedSlice(allocator),
    };
}

/// Parse one ELF32/REL relocatable object (i386) into its `.text`, symbols, and
/// relocations. Structurally the same walk as `parseObject64`, but every ELF
/// structure is the 32-bit variant with different field widths *and offsets*
/// (`Elf32_Shdr`/`Elf32_Sym` are not simply narrowed `Elf64_*` layouts), and the
/// relocation section is `SHT_REL` (`Elf32_Rel`, no addend field): `Reloc.addend`
/// stays 0 here, since i386's addend is pre-written into the relocated field itself
/// (`arch/x86.zig` reads it back out before patching).
fn parseObject32(allocator: std.mem.Allocator, buf: []const u8) Error!ParsedObject {
    if (buf.len < 52) return error.MalformedObject;
    const arch = fromEMachine(try rdInt(u16, buf, 18)) orelse return error.MalformedObject;

    const shoff = try rdInt(u32, buf, 32);
    const shentsize = try rdInt(u16, buf, 46);
    const shnum = try rdInt(u16, buf, 48);

    // The section-name string table (`.shstrtab`), located via `e_shstrndx`. It feeds
    // the `sections` table below. A missing/out-of-range index (never happens for a real
    // object) just leaves every section entry unnamed rather than failing the whole parse.
    const e_shstrndx = try rdInt(u16, buf, 50);
    const shstrtab: []const u8 = blk: {
        if (e_shstrndx == 0 or e_shstrndx >= shnum) break :blk &.{};
        const shstr_hdr = try tableOffset(shoff, e_shstrndx, shentsize);
        const shstr_off = try rdInt(u32, buf, shstr_hdr + 16);
        const shstr_size = try rdInt(u32, buf, shstr_hdr + 20);
        break :blk try secSlice(buf, shstr_off, shstr_size);
    };

    // Pick up the symbol/relocation tables.
    var symtab_ndx: ?u16 = null;
    var rel_ndxs: std.ArrayList(u16) = .empty;
    defer rel_ndxs.deinit(allocator);

    // The arbitrary-named-section table (every `SHF_ALLOC` section, no same-class
    // collapsing), plus a map from the raw section-header index to its index within
    // `sections` (`maxInt` for a section `sections` does not carry).
    var sections: std.ArrayList(ObjSection) = .empty;
    errdefer {
        for (sections.items) |sec| allocator.free(sec.relocs);
        sections.deinit(allocator);
    }
    var sec_map = try allocator.alloc(u32, shnum);
    defer allocator.free(sec_map);
    @memset(sec_map, std.math.maxInt(u32));

    var i: u16 = 0;
    while (i < shnum) : (i += 1) {
        const hdr = try tableOffset(shoff, i, shentsize);
        const sh_name = try rdInt(u32, buf, hdr + 0);
        const typ = try rdInt(u32, buf, hdr + 4);
        const flags: u64 = try rdInt(u32, buf, hdr + 8);
        const sh_off = try rdInt(u32, buf, hdr + 16);
        const sh_size = try rdInt(u32, buf, hdr + 20);
        const class = sectionClass(typ, flags);
        if (class != .ignore) {
            const is_nobits = class == .nobits;
            const bytes: []const u8 = if (is_nobits) &.{} else try secSlice(buf, sh_off, sh_size);
            sec_map[i] = @intCast(sections.items.len);
            try sections.append(allocator, .{
                .name = if (shstrtab.len == 0) "" else try strAt(shstrtab, sh_name),
                .flags = flags,
                .bytes = bytes,
                .size = sh_size,
                .is_nobits = is_nobits,
            });
        }
        if (typ == SHT_SYMTAB) symtab_ndx = i;
        if (typ == SHT_REL) try rel_ndxs.append(allocator, i);
    }
    const si = symtab_ndx orelse return error.MalformedObject;

    // Symbol table and its string table (via sh_link). Elf32_Shdr: sh_offset@16,
    // sh_size@20, sh_link@24 (all u32, unlike Elf64_Shdr's offsets 24/32/40).
    const sym_hdr = try tableOffset(shoff, si, shentsize);
    const sym_off = try rdInt(u32, buf, sym_hdr + 16);
    const sym_size = try rdInt(u32, buf, sym_hdr + 20);
    const sym_link = try rdInt(u32, buf, sym_hdr + 24);
    const str_hdr = try tableOffset(shoff, sym_link, shentsize);
    const str_off = try rdInt(u32, buf, str_hdr + 16);
    const str_size = try rdInt(u32, buf, str_hdr + 20);
    const strtab = try secSlice(buf, str_off, str_size);

    // Elf32_Sym is 16 bytes: st_name@0(u32), st_value@4(u32), st_size@8(u32),
    // st_info@12(u8), st_other@13(u8), st_shndx@14(u16) - a different field order
    // than Elf64_Sym (which has st_info right after st_name).
    const sym_count: usize = @intCast(sym_size / 16);
    var symbols = try allocator.alloc(ObjSymbol, sym_count);
    errdefer allocator.free(symbols);
    var k: usize = 0;
    while (k < sym_count) : (k += 1) {
        const e = try tableOffset(sym_off, k, 16);
        const st_name = try rdInt(u32, buf, e + 0);
        const st_value = try rdInt(u32, buf, e + 4);
        const st_size = try rdInt(u32, buf, e + 8);
        const st_info = try rdInt(u8, buf, e + 12);
        const st_shndx = try rdInt(u16, buf, e + 14);
        const section_index: u32 = if (st_shndx != SHN_UNDEF and st_shndx < shnum) sec_map[st_shndx] else std.math.maxInt(u32);
        symbols[k] = .{
            .name = if (st_name == 0) "" else try strAt(strtab, st_name),
            .value = st_value,
            .defined = st_shndx != SHN_UNDEF,
            .local = (st_info >> 4) == 0, // STB_LOCAL
            .size = st_size,
            .section_index = section_index,
        };
    }

    // Relocations (if any). `SHT_REL` (`Elf32_Rel`) is 8 bytes: r_offset@0(u32),
    // r_info@4(u32) with `r_info = (sym_index << 8) | type` and NO addend field -
    // the addend lives in the relocated field itself, left as `Reloc.addend = 0`
    // here (the default) for `arch/x86.zig` to read back out of the field.
    //
    // Each `SHT_REL` section applies to the section named by its `sh_info`: route each
    // reloc into that target section's `ObjSection.relocs`, by section-table index - the
    // same per-section routing `parseObject64` applies to `SHT_RELA`. A `.abs32` (`R_386_32`,
    // numeric 1) reloc can appear in EITHER `.text` (a `global_addr` load's `mov reg, imm32`)
    // or `.data`/`.rodata` (a pointer-init slot); which section's relocs it lands in is
    // decided purely by the section it is found in, not by its numeric type.
    var sec_relocs = try allocator.alloc(std.ArrayList(Reloc), sections.items.len);
    defer allocator.free(sec_relocs);
    for (sec_relocs) |*sr| sr.* = .empty;
    errdefer for (sec_relocs) |*sr| sr.deinit(allocator);
    for (rel_ndxs.items) |ri| {
        const rel_hdr = try tableOffset(shoff, ri, shentsize);
        const rel_off = try rdInt(u32, buf, rel_hdr + 16);
        const rel_size = try rdInt(u32, buf, rel_hdr + 20);
        const sh_info = try rdInt(u32, buf, rel_hdr + 28);
        // The section this REL modifies, and hence which `ObjSection.relocs` list it feeds.
        const target_sec: u32 = if (sh_info < shnum) sec_map[sh_info] else std.math.maxInt(u32);
        const rel_count: usize = @intCast(rel_size / 8);
        var r: usize = 0;
        while (r < rel_count) : (r += 1) {
            const e = try tableOffset(rel_off, r, 8);
            const r_offset = try rdInt(u32, buf, e + 0);
            const r_info = try rdInt(u32, buf, e + 4);
            const typ: u32 = r_info & 0xff;
            const sym_index: u32 = r_info >> 8;
            const rt: RelocType = switch (typ) {
                @intFromEnum(RelocType.abs32) => .abs32,
                @intFromEnum(RelocType.pc32) => .pc32,
                @intFromEnum(RelocType.got32) => .got32,
                else => return error.UnsupportedReloc,
            };
            const reloc: Reloc = .{
                .offset = r_offset,
                .symbol = sym_index,
                .type = rt,
            };
            if (target_sec != std.math.maxInt(u32)) try sec_relocs[target_sec].append(allocator, reloc);
        }
    }

    for (sections.items, 0..) |*sec, idx| sec.relocs = try sec_relocs[idx].toOwnedSlice(allocator);

    return .{
        .arch = arch,
        .symbols = symbols,
        .sections = try sections.toOwnedSlice(allocator),
    };
}

pub fn findSymbol(symbols: []const ResolvedSymbol, name: []const u8) ?u64 {
    for (symbols) |s| {
        if (std.mem.eql(u8, s.name, name)) return s.address;
    }
    return null;
}

test "fromEMachine maps known machines and rejects the rest" {
    try std.testing.expectEqual(@as(?Arch, .riscv64), fromEMachine(EM_RISCV));
    try std.testing.expectEqual(@as(?Arch, .aarch64), fromEMachine(EM_AARCH64));
    try std.testing.expectEqual(@as(?Arch, .x86_64), fromEMachine(EM_X86_64));
    try std.testing.expectEqual(@as(?Arch, .x86), fromEMachine(EM_386));
    try std.testing.expectEqual(@as(?Arch, null), fromEMachine(0));
}

test "parseObject64: two same-class sections stay separate in the new sections table" {
    const allocator = std.testing.allocator;

    // A minimal, hand-built ELF64/RELA relocatable object carrying TWO
    // `SHT_PROGBITS | SHF_ALLOC|SHF_EXECINSTR` sections (".text" and ".text.extra"),
    // a ".symtab" with one FUNC symbol in each (nonzero `st_size`), a ".strtab", and a
    // ".shstrtab". Proves the `sections` table holds both separately, each symbol keyed
    // to the section that defines it.
    const shstrtab = "\x00.text\x00.text.extra\x00.symtab\x00.strtab\x00.shstrtab\x00";
    const strtab = "\x00func1\x00func2\x00";

    const text_off: u64 = 64;
    const text_extra_off: u64 = text_off + 4;
    const symtab_off: u64 = text_extra_off + 4;
    const symtab_size: u64 = 3 * 24; // the null entry plus two symbols
    const strtab_off: u64 = symtab_off + symtab_size;
    const shstrtab_off: u64 = strtab_off + strtab.len;
    const shoff: u64 = alignUp(shstrtab_off + shstrtab.len, 8);
    const shnum: u16 = 6;
    const total: usize = @intCast(shoff + @as(u64, shnum) * 64);

    var buf = try allocator.alloc(u8, total);
    defer allocator.free(buf);
    @memset(buf, 0);

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    std.mem.writeInt(u16, buf[16..18], 1, .little); // e_type = ET_REL
    std.mem.writeInt(u16, buf[18..20], EM_X86_64, .little);
    std.mem.writeInt(u32, buf[20..24], 1, .little); // e_version
    std.mem.writeInt(u64, buf[40..48], shoff, .little); // e_shoff
    std.mem.writeInt(u16, buf[52..54], 64, .little); // e_ehsize
    std.mem.writeInt(u16, buf[58..60], 64, .little); // e_shentsize
    std.mem.writeInt(u16, buf[60..62], shnum, .little); // e_shnum
    std.mem.writeInt(u16, buf[62..64], 5, .little); // e_shstrndx (.shstrtab is index 5)

    @memcpy(buf[text_off..][0..4], &[_]u8{ 0xaa, 0xbb, 0xcc, 0xdd });
    @memcpy(buf[text_extra_off..][0..4], &[_]u8{ 0x11, 0x22, 0x33, 0x44 });
    @memcpy(buf[strtab_off..][0..strtab.len], strtab);
    @memcpy(buf[shstrtab_off..][0..shstrtab.len], shstrtab);

    const writeSym = struct {
        fn f(b: []u8, off: u64, idx: usize, name: u32, info: u8, shndx: u16, value: u64, size: u64) void {
            const e = b[off + idx * 24 ..][0..24];
            std.mem.writeInt(u32, e[0..4], name, .little);
            e[4] = info;
            e[5] = 0;
            std.mem.writeInt(u16, e[6..8], shndx, .little);
            std.mem.writeInt(u64, e[8..16], value, .little);
            std.mem.writeInt(u64, e[16..24], size, .little);
        }
    }.f;
    // func1 @ .text (shndx=1), func2 @ .text.extra (shndx=2). Both STB_GLOBAL|STT_FUNC.
    writeSym(buf, symtab_off, 1, 1, 0x12, 1, 0, 4);
    writeSym(buf, symtab_off, 2, 7, 0x12, 2, 0, 4);

    const writeShdr = struct {
        fn f(b: []u8, sh: u64, idx: u16, name: u32, typ: u32, flags: u64, off: u64, size: u64, link: u32, info: u32) void {
            const e = b[sh + @as(u64, idx) * 64 ..][0..64];
            std.mem.writeInt(u32, e[0..4], name, .little);
            std.mem.writeInt(u32, e[4..8], typ, .little);
            std.mem.writeInt(u64, e[8..16], flags, .little);
            std.mem.writeInt(u64, e[16..24], 0, .little); // sh_addr
            std.mem.writeInt(u64, e[24..32], off, .little);
            std.mem.writeInt(u64, e[32..40], size, .little);
            std.mem.writeInt(u32, e[40..44], link, .little);
            std.mem.writeInt(u32, e[44..48], info, .little);
            std.mem.writeInt(u64, e[48..56], 1, .little); // sh_addralign
            std.mem.writeInt(u64, e[56..64], 0, .little); // sh_entsize
        }
    }.f;
    writeShdr(buf, shoff, 0, 0, 0, 0, 0, 0, 0, 0); // the reserved NULL section
    writeShdr(buf, shoff, 1, 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, 4, 0, 0);
    writeShdr(buf, shoff, 2, 7, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_extra_off, 4, 0, 0);
    writeShdr(buf, shoff, 3, 19, SHT_SYMTAB, 0, symtab_off, symtab_size, 4, 1); // sh_link -> .strtab
    writeShdr(buf, shoff, 4, 27, 3, 0, strtab_off, strtab.len, 0, 0); // SHT_STRTAB
    writeShdr(buf, shoff, 5, 35, 3, 0, shstrtab_off, shstrtab.len, 0, 0); // SHT_STRTAB

    var parsed = try parseObject(allocator, buf);
    defer parsed.deinit(allocator);

    // Both sections show up, kept separate, with their names/sizes/flags intact.
    try std.testing.expectEqual(@as(usize, 2), parsed.sections.len);
    try std.testing.expectEqualStrings(".text", parsed.sections[0].name);
    try std.testing.expectEqualStrings(".text.extra", parsed.sections[1].name);
    try std.testing.expectEqual(@as(u64, 4), parsed.sections[0].size);
    try std.testing.expectEqual(@as(u64, 4), parsed.sections[1].size);
    try std.testing.expectEqual(SHF_ALLOC | SHF_EXECINSTR, parsed.sections[0].flags);
    try std.testing.expectEqual(false, parsed.sections[0].is_nobits);

    // func1 -> sections[0], func2 -> sections[1], each carrying its real st_size.
    var seen_func1 = false;
    var seen_func2 = false;
    for (parsed.symbols) |s| {
        if (std.mem.eql(u8, s.name, "func1")) {
            seen_func1 = true;
            try std.testing.expectEqual(@as(u32, 0), s.section_index);
            try std.testing.expectEqual(@as(u64, 4), s.size);
        }
        if (std.mem.eql(u8, s.name, "func2")) {
            seen_func2 = true;
            try std.testing.expectEqual(@as(u32, 1), s.section_index);
            try std.testing.expectEqual(@as(u64, 4), s.size);
        }
    }
    try std.testing.expect(seen_func1);
    try std.testing.expect(seen_func2);
}
