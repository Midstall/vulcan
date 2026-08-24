//! This file emits an ELF64 relocatable object (`ET_REL`, `EM_RISCV`). Each function
//! becomes its own `.text.<name>` section with an `STT_FUNC` symbol at offset 0. Each
//! data global (`module.data`) becomes its own `.rodata.<name>`, `.data.<name>`, or
//! `.bss.<name>` section with an `STT_OBJECT` symbol at offset 0. Each call becomes an
//! `R_RISCV_JAL` (or `R_RISCV_CALL`) relocation against the callee symbol. The symbol is
//! undefined if external. A `global_addr`'s `auipc`/`addi` pair becomes an
//! `R_RISCV_PCREL_HI20`/`R_RISCV_PCREL_LO12_I` relocation pair. The high half targets the
//! referenced symbol. The low half targets a synthesized local `.Lpcrel_hi` label placed
//! at the `auipc` inside the same function's `.text.<name>` section, so the paired high
//! reloc resolves through the section-relative label. The shared `object_emit.emit`
//! serializes the neutral section, symbol, and relocation lists into the ELF bytes.
//! `readelf` and a system RISC-V linker accept the output. `ld.zig` is Vulcan's own linker
//! for this object format.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");
const ld = @import("vulcan-link");
const encode = @import("encode.zig");
const dwarf = @import("../dwarf.zig");
const object_emit = @import("../object_emit.zig");
const harness = @import("tests/harness.zig");

pub const Error = std.mem.Allocator.Error || isel.Error;

/// A symbol's binding. Locals must be listed before globals (ELF requires the
/// local symbols to form a prefix of the symbol table). The shared emitter owns
/// that sort, so this file passes symbols in their natural order.
pub const Binding = enum { local, global };

/// A symbol's type. `func` marks a function entry, `object` a data object,
/// `notype` an unknown (e.g. a local PC-relative label).
pub const SymKind = enum { notype, func, object };

/// The allocatable output sections an object can carry. `.text` is code,
/// `.rodata` read-only data, `.data` writable data, `.bss` zero-initialized
/// data (occupies memory but no file bytes).
pub const SectionKind = enum { text, rodata, data, bss };

/// One entry in the object's symbol table. A defined symbol lives in `section`
/// at `value`. An undefined one (`defined = false`) is an external reference the
/// linker must resolve.
pub const Symbol = struct {
    name: []const u8,
    value: u64 = 0,
    size: u64 = 0,
    binding: Binding = .global,
    kind: SymKind = .func,
    defined: bool = true,
    section: SectionKind = .text,
};

/// The emitted RISC-V relocation types. Values are the architectural
/// `R_RISCV_*` codes that land in the high half of `r_info`.
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
    /// entry (the `auipc` half of a GOT-indirect `auipc`/`ld` pair, a DATA import). The paired
    /// `ld`'s low 12 bits reuse `pcrel_lo12_i` (its target is the `auipc` label, as for the
    /// direct `addi` form). The dynamic linker synthesizes the GOT slot. The static path never
    /// sees this reloc.
    got_hi20 = 20,
};

/// A relocation applied to a `.text` offset against a symbol.
pub const Reloc = struct {
    /// Byte offset within `.text` of the instruction to patch.
    offset: u64,
    /// Index into the `Object.symbols` array.
    symbol: u32,
    type: RelocType,
    addend: i64 = 0,
};

/// `R_RISCV_64`: a 64-bit absolute address (`S + A`) written into a DATA section slot. This is
/// the relocation a pointer-initialized global (`int *p = &g;`) carries in `.rela.data`/
/// `.rela.rodata`: the 8-byte slot holding the pointer must be patched to the target symbol's
/// runtime address. The linker turns it into a `R_RISCV_RELATIVE` dyn reloc (PIE/`.so`) or a
/// direct absolute write (non-PIE exec). Mirrors `aarch64/object.zig`'s `R_AARCH64_ABS64` /
/// `x86_64/object.zig`'s `R_X86_64_64`. Numerically the same code `arch/riscv64.zig` uses for
/// the GOT `GLOB_DAT` role (both are the architectural "64-bit absolute" reloc), but this one
/// lives in `.rela.data`/`.rela.rodata`, never `.rela.dyn`.
pub const R_RISCV_64: u32 = 2;

/// A relocation applied to a DATA section (`.data` or `.rodata`) slot against a symbol: at
/// byte `offset` within `section`, an `R_RISCV_64` writes `symbol`'s runtime address plus
/// `addend`. Emitted into `.rela.data`/`.rela.rodata` (keyed by `section`). This closes the
/// long-deferred data-section-relocation gap on riscv64 (the last of the four arches): a data
/// global's own pointer inits are now carried in the object, not dropped.
pub const DataRelocEntry = struct {
    section: SectionKind,
    offset: u64,
    symbol: u32,
    addend: i64 = 0,
};

/// A non-alloc PROGBITS section carried without change (for example a DWARF `.debug_*`
/// blob).
pub const DebugSection = struct { name: []const u8, bytes: []const u8 };

pub const Object = struct {
    text: []const u8,
    rodata: []const u8 = &.{},
    data: []const u8 = &.{},
    bss_size: u64 = 0,
    symbols: []const Symbol,
    relocs: []const Reloc,
    /// Relocations applied to the `.data`/`.rodata` sections themselves (pointer inits),
    /// emitted as `.rela.data`/`.rela.rodata`. Empty for a code-only or const-only object.
    data_relocs: []const DataRelocEntry = &.{},
    /// Extra PROGBITS metadata sections (DWARF) placed after the allocatable sections. Symbols do
    /// not reference them, so they need no section-index bookkeeping beyond the header count.
    debug: []const DebugSection = &.{},
};

// The ELF constants this file still uses. The shared emitter owns the rest (the header,
// symbol-table, string-table, and reloc-section layout).
const EM_RISCV: u16 = 243;
const SHT_PROGBITS: u32 = 1;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

fn putInt(buf: []u8, comptime T: type, value: T) void {
    std.mem.writeInt(T, buf[0..@sizeOf(T)], value, .little);
}

/// Map this file's `Binding` to the shared emitter's binding.
fn toBinding(b: Binding) object_emit.Binding {
    return switch (b) {
        .local => .local,
        .global => .global,
    };
}

/// Map this file's `SymKind` to the shared emitter's symbol type.
fn toSymType(k: SymKind) object_emit.SymType {
    return switch (k) {
        .notype => .notype,
        .func => .func,
        .object => .object,
    };
}

/// Serialize a single-section `Object` into an ELF64 RISC-V relocatable object. This is the
/// raw entry point that hand-built test objects (and the assembler in `vulcan-ld`) use. It
/// maps the `.text`, `.rodata`, `.data`, and `.bss` blobs and the symbol and relocation
/// lists into the neutral form, then lets `object_emit.emit` write the ELF bytes. The
/// emitter owns the symbol sort and the reloc-index remap, so the symbols pass in their
/// natural order. The caller owns the result.
pub fn write(allocator: std.mem.Allocator, obj: Object) Error![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    const has_rodata = obj.rodata.len > 0;
    const has_data = obj.data.len > 0;
    const has_bss = obj.bss_size > 0;

    // The text relocations, section-relative offsets already. Each `symbol` is the index
    // into `obj.symbols`. The emitter remaps it after its symbol sort.
    const text_relocs = try a.alloc(object_emit.OutReloc, obj.relocs.len);
    for (obj.relocs, 0..) |r, i| text_relocs[i] = .{ .offset = r.offset, .symbol = r.symbol, .r_type = @intFromEnum(r.type), .addend = r.addend };

    // The data pointer-init relocations, grouped by the section they modify. Each one is an
    // `R_RISCV_64` against the target symbol.
    var rodata_relocs: std.ArrayList(object_emit.OutReloc) = .empty;
    var data_relocs: std.ArrayList(object_emit.OutReloc) = .empty;
    for (obj.data_relocs) |dr| {
        const e: object_emit.OutReloc = .{ .offset = dr.offset, .symbol = dr.symbol, .r_type = R_RISCV_64, .addend = dr.addend };
        switch (dr.section) {
            .rodata => try rodata_relocs.append(a, e),
            .data => try data_relocs.append(a, e),
            .text, .bss => {}, // a data relocation never lives in .text or .bss
        }
    }

    // The section list. `.text` is index 0, always present. Track each present section's
    // index, so a symbol resolves its owning section.
    var sections: std.ArrayList(object_emit.OutSection) = .empty;
    var text_idx: u32 = 0;
    var rodata_idx: u32 = 0;
    var data_idx: u32 = 0;
    var bss_idx: u32 = 0;
    text_idx = @intCast(sections.items.len);
    try sections.append(a, .{ .name = ".text", .sh_type = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_EXECINSTR, .bytes = obj.text, .size = obj.text.len, .addralign = 4, .relocs = text_relocs });
    if (has_rodata) {
        rodata_idx = @intCast(sections.items.len);
        try sections.append(a, .{ .name = ".rodata", .sh_type = SHT_PROGBITS, .flags = SHF_ALLOC, .bytes = obj.rodata, .size = obj.rodata.len, .addralign = 8, .relocs = rodata_relocs.items });
    }
    if (has_data) {
        data_idx = @intCast(sections.items.len);
        try sections.append(a, .{ .name = ".data", .sh_type = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_WRITE, .bytes = obj.data, .size = obj.data.len, .addralign = 8, .relocs = data_relocs.items });
    }
    if (has_bss) {
        bss_idx = @intCast(sections.items.len);
        try sections.append(a, .{ .name = ".bss", .sh_type = SHT_NOBITS, .flags = SHF_ALLOC | SHF_WRITE, .size = obj.bss_size, .addralign = 8 });
    }
    // DWARF (or other) debug sections. They are plain PROGBITS, non-alloc, and no other
    // section refers to them.
    for (obj.debug) |d| try sections.append(a, .{ .name = d.name, .sh_type = SHT_PROGBITS, .flags = 0, .bytes = d.bytes, .size = d.bytes.len, .addralign = 1 });

    // The symbols, in their natural order. Each defined symbol names its own section.
    const symbols = try a.alloc(object_emit.OutSymbol, obj.symbols.len);
    for (obj.symbols, 0..) |s, i| {
        const sec: u32 = switch (s.section) {
            .text => text_idx,
            .rodata => rodata_idx,
            .data => data_idx,
            .bss => bss_idx,
        };
        symbols[i] = .{ .name = s.name, .section = sec, .value = s.value, .size = s.size, .binding = toBinding(s.binding), .sym_type = toSymType(s.kind), .defined = s.defined };
    }

    return object_emit.emit(allocator, sections.items, symbols, .{ .class = .elf64, .machine = EM_RISCV, .use_rela = true });
}

/// Map an isel relocation's `kind` to the matching ELF relocation type. A `pcrel_lo12` is
/// handled apart, because its target is a synthesized local label, not a named symbol.
fn relocTypeOf(kind: isel.RelocKind) RelocType {
    return switch (kind) {
        .call => .jal,
        .pcrel_hi20 => .pcrel_hi20,
        .got_hi20 => .got_hi20,
        .pcrel_lo12 => .pcrel_lo12_i,
    };
}

/// Find the index of the symbol named `name` in the neutral symbol list, or null.
fn oeIndex(symbols: []const object_emit.OutSymbol, name: []const u8) ?u32 {
    for (symbols, 0..) |s, i| if (std.mem.eql(u8, s.name, name)) return @intCast(i);
    return null;
}

/// The section-name class for a data global's kind. A `.rodata` global lands in
/// `.rodata.<name>`, a `.data` global in `.data.<name>`, and a `.bss` global in
/// `.bss.<name>`.
fn dataClass(kind: link.DataKind) []const u8 {
    return switch (kind) {
        .rodata => "rodata",
        .data => "data",
        .bss => "bss",
    };
}

/// Build a data global's own section name. A leading `.` is stripped from the symbol name,
/// so a local `.str.N` becomes `.rodata.str.N`, not `..rodata..str.N`.
fn dataSectionName(a: std.mem.Allocator, class: []const u8, name: []const u8) Error![]u8 {
    const bare = if (std.mem.startsWith(u8, name, ".")) name[1..] else name;
    return std.fmt.allocPrint(a, ".{s}.{s}", .{ class, bare });
}

/// Compile every function in `module`, and serialize them and its data globals into one ELF
/// relocatable object. Each function becomes its own `.text.<name>` section with a defined
/// `STT_FUNC` symbol at offset 0. Each data global becomes its own `.rodata.<name>`,
/// `.data.<name>`, or `.bss.<name>` section with an `STT_OBJECT` symbol at offset 0. Each
/// call becomes an `R_RISCV_JAL` relocation, rebased to its own section. A `global_addr`'s
/// `auipc`/`addi` pair becomes an `R_RISCV_PCREL_HI20`/`R_RISCV_PCREL_LO12_I` pair, the low
/// half targeting a local `.Lpcrel_hi` label at the `auipc` inside the same section.
/// Undefined targets become undefined globals. The shared `object_emit.emit` writes the ELF
/// bytes. The caller owns the result.
pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var sections: std.ArrayList(object_emit.OutSection) = .empty;
    var symbols: std.ArrayList(object_emit.OutSymbol) = .empty;
    // One relocation list per section. Its index matches `sections`.
    var reloc_lists: std.ArrayList(std.ArrayList(object_emit.OutReloc)) = .empty;

    // A text relocation resolved by name (a call, a `pcrel_hi20`, or a `got_hi20`). Its
    // target symbol index resolves after every symbol is known.
    const NamedReloc = struct { sec: usize, offset: u64, name: []const u8, r_type: u32 };
    var named_relocs: std.ArrayList(NamedReloc) = .empty;
    // A `pcrel_lo12` text relocation. Its target is a synthesized local label, so its symbol
    // index is already known and stored directly.
    const Lo12Reloc = struct { sec: usize, offset: u64, sym: u32 };
    var lo12_relocs: std.ArrayList(Lo12Reloc) = .empty;
    // A data pointer-init relocation, resolved by name after every symbol exists.
    const DataR = struct { sec: usize, offset: u64, name: []const u8 };
    var data_pending: std.ArrayList(DataR) = .empty;

    // A running counter, so each synthesized `.Lpcrel_hi` label has a distinct name. The
    // paired low reloc targets its label by symbol index, so the name only needs to be
    // unique for the string table.
    var aux_counter: usize = 0;

    const caps: isel.ModelCaps = if (module.model) |m| isel.capsForModel(m) else .{};
    for (module.entries.items) |entry| {
        var compiled = try isel.compileFunction(allocator, entry.func, caps);
        defer compiled.deinit(allocator);
        const code = try a.alloc(u8, compiled.code.len * 4);
        for (compiled.code, 0..) |word, wi| putInt(code[wi * 4 ..][0..4], u32, word);
        const sec_index = sections.items.len;
        try sections.append(a, .{
            .name = try std.fmt.allocPrint(a, ".text.{s}", .{entry.name}),
            .sh_type = SHT_PROGBITS,
            .flags = SHF_ALLOC | SHF_EXECINSTR,
            .bytes = code,
            .size = code.len,
            .addralign = 4,
        });
        try reloc_lists.append(a, .empty);
        // A `static` function has internal linkage. So does a `.`-prefixed compiler-local
        // name. Both get LOCAL binding.
        const binding: object_emit.Binding = if (entry.func.is_local or std.mem.startsWith(u8, entry.name, ".")) .local else .global;
        try symbols.append(a, .{ .name = entry.name, .section = @intCast(sec_index), .value = 0, .size = code.len, .binding = binding, .sym_type = .func, .defined = true });
        // Rebase each relocation section-relative. Its offset was a word index into this
        // function's own code, so multiply by 4.
        for (compiled.relocs) |r| {
            const off = @as(u64, r.offset) * 4;
            switch (r.kind) {
                .call, .pcrel_hi20, .got_hi20 => try named_relocs.append(a, .{ .sec = sec_index, .offset = off, .name = r.symbol, .r_type = @intFromEnum(relocTypeOf(r.kind)) }),
                .pcrel_lo12 => {
                    // Synthesize a local label at the paired `auipc`. It lives in THIS
                    // function's section, at the `auipc`'s within-section byte offset (the
                    // paired word index times 4), NOT at 0. The low reloc below targets it by
                    // this symbol index, so the high/low pair resolves through the label once
                    // the relocs are section-relative.
                    const aux_index: u32 = @intCast(symbols.items.len);
                    const name = try std.fmt.allocPrint(a, ".Lpcrel_hi{d}", .{aux_counter});
                    aux_counter += 1;
                    try symbols.append(a, .{ .name = name, .section = @intCast(sec_index), .value = @as(u64, r.pair) * 4, .size = 0, .binding = .local, .sym_type = .notype, .defined = true });
                    try lo12_relocs.append(a, .{ .sec = sec_index, .offset = off, .sym = aux_index });
                },
            }
        }
    }

    // One section per data global.
    for (module.data.items) |d| {
        const sec_index = sections.items.len;
        const sec_name = try dataSectionName(a, dataClass(d.kind), d.name);
        switch (d.kind) {
            .rodata => try sections.append(a, .{ .name = sec_name, .sh_type = SHT_PROGBITS, .flags = SHF_ALLOC, .bytes = d.bytes, .size = d.bytes.len, .addralign = 8 }),
            .data => try sections.append(a, .{ .name = sec_name, .sh_type = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_WRITE, .bytes = d.bytes, .size = d.bytes.len, .addralign = 8 }),
            .bss => try sections.append(a, .{ .name = sec_name, .sh_type = SHT_NOBITS, .flags = SHF_ALLOC | SHF_WRITE, .size = d.size, .addralign = 8 }),
        }
        try reloc_lists.append(a, .empty);
        // An anonymous compiler-internal object (a string literal `.str.N` or any other
        // `.`-prefixed name) has internal linkage, so it takes LOCAL binding.
        const binding: object_emit.Binding = if (std.mem.startsWith(u8, d.name, ".")) .local else .global;
        try symbols.append(a, .{ .name = d.name, .section = @intCast(sec_index), .value = 0, .size = d.size, .binding = binding, .sym_type = .object, .defined = true });
        // A data global is its own section, so its pointer-init offset is already
        // section-relative.
        for (d.relocs) |r| try data_pending.append(a, .{ .sec = sec_index, .offset = r.off, .name = r.symbol });
    }

    // An undefined external callee. A named relocation whose target names no defined symbol
    // is an import. Add it once as an undefined `notype` global.
    for (named_relocs.items) |p| {
        if (oeIndex(symbols.items, p.name) == null) try symbols.append(a, .{ .name = p.name, .section = 0, .value = 0, .size = 0, .binding = .global, .sym_type = .notype, .defined = false });
    }

    // Resolve every named text relocation to its symbol index. The emitter remaps it after
    // its own symbol sort.
    for (named_relocs.items) |p| {
        const sym = oeIndex(symbols.items, p.name).?;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = p.r_type });
    }
    // Attach each low reloc. Its symbol index (the local label) is already known.
    for (lo12_relocs.items) |p| {
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = p.sym, .r_type = @intFromEnum(RelocType.pcrel_lo12_i) });
    }
    // Resolve every data pointer-init relocation. Its target is an internally defined
    // global. It is an error if the target is missing.
    for (data_pending.items) |p| {
        const sym = oeIndex(symbols.items, p.name) orelse return error.Unsupported;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = R_RISCV_64 });
    }

    // Attach each section's relocation list.
    for (sections.items, 0..) |*sec, i| sec.relocs = reloc_lists.items[i].items;

    return object_emit.emit(allocator, sections.items, symbols.items, .{ .class = .elf64, .machine = EM_RISCV, .use_rela = true });
}

/// Like `writeModule`, but also emits inline DWARF: `.debug_abbrev` + `.debug_info` (a subprogram
/// DIE per function with its PC range and IR return type) + `.debug_line` (address -> source line,
/// from the functions' `debug.line` attributes), with the CU linked to the line program. The DWARF
/// program numbers PC ranges from a single running text offset, so the line and range tables stay
/// self-consistent across the per-function sections. So a debugger reads function names, ranges,
/// typed signatures, and source lines on real RISC-V objects.
pub fn writeModuleWithDebug(allocator: std.mem.Allocator, module: *const link.Module, source_file: []const u8) Error![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var sections: std.ArrayList(object_emit.OutSection) = .empty;
    var symbols: std.ArrayList(object_emit.OutSymbol) = .empty;
    var reloc_lists: std.ArrayList(std.ArrayList(object_emit.OutReloc)) = .empty;

    const NamedReloc = struct { sec: usize, offset: u64, name: []const u8, r_type: u32 };
    var named_relocs: std.ArrayList(NamedReloc) = .empty;
    const Lo12Reloc = struct { sec: usize, offset: u64, sym: u32 };
    var lo12_relocs: std.ArrayList(Lo12Reloc) = .empty;
    const DataR = struct { sec: usize, offset: u64, name: []const u8 };
    var data_pending: std.ArrayList(DataR) = .empty;
    var aux_counter: usize = 0;

    var rows: std.ArrayList(dwarf.LineRow) = .empty;
    // Each function's DWARF PC range, numbered from one running text offset.
    var func_low: std.ArrayList(u64) = .empty;
    var func_high: std.ArrayList(u64) = .empty;
    var text_off: u64 = 0;

    const caps: isel.ModelCaps = if (module.model) |m| isel.capsForModel(m) else .{};
    for (module.entries.items) |entry| {
        var compiled = try isel.compileFunction(allocator, entry.func, caps);
        defer compiled.deinit(allocator);
        const code = try a.alloc(u8, compiled.code.len * 4);
        for (compiled.code, 0..) |word, wi| putInt(code[wi * 4 ..][0..4], u32, word);
        const sec_index = sections.items.len;
        try sections.append(a, .{
            .name = try std.fmt.allocPrint(a, ".text.{s}", .{entry.name}),
            .sh_type = SHT_PROGBITS,
            .flags = SHF_ALLOC | SHF_EXECINSTR,
            .bytes = code,
            .size = code.len,
            .addralign = 4,
        });
        try reloc_lists.append(a, .empty);
        const binding: object_emit.Binding = if (entry.func.is_local or std.mem.startsWith(u8, entry.name, ".")) .local else .global;
        try symbols.append(a, .{ .name = entry.name, .section = @intCast(sec_index), .value = 0, .size = code.len, .binding = binding, .sym_type = .func, .defined = true });
        for (compiled.relocs) |r| {
            const off = @as(u64, r.offset) * 4;
            switch (r.kind) {
                .call, .pcrel_hi20, .got_hi20 => try named_relocs.append(a, .{ .sec = sec_index, .offset = off, .name = r.symbol, .r_type = @intFromEnum(relocTypeOf(r.kind)) }),
                .pcrel_lo12 => {
                    const aux_index: u32 = @intCast(symbols.items.len);
                    const name = try std.fmt.allocPrint(a, ".Lpcrel_hi{d}", .{aux_counter});
                    aux_counter += 1;
                    try symbols.append(a, .{ .name = name, .section = @intCast(sec_index), .value = @as(u64, r.pair) * 4, .size = 0, .binding = .local, .sym_type = .notype, .defined = true });
                    try lo12_relocs.append(a, .{ .sec = sec_index, .offset = off, .sym = aux_index });
                },
            }
        }
        // The DWARF PC range and line rows use the running text offset.
        try func_low.append(a, text_off);
        try func_high.append(a, text_off + code.len);
        for (compiled.lines) |e| try rows.append(a, .{ .address = text_off + e.offset, .line = e.line });
        text_off += code.len;
    }

    // One section per data global (mirrors `writeModule`).
    for (module.data.items) |d| {
        const sec_index = sections.items.len;
        const sec_name = try dataSectionName(a, dataClass(d.kind), d.name);
        switch (d.kind) {
            .rodata => try sections.append(a, .{ .name = sec_name, .sh_type = SHT_PROGBITS, .flags = SHF_ALLOC, .bytes = d.bytes, .size = d.bytes.len, .addralign = 8 }),
            .data => try sections.append(a, .{ .name = sec_name, .sh_type = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_WRITE, .bytes = d.bytes, .size = d.bytes.len, .addralign = 8 }),
            .bss => try sections.append(a, .{ .name = sec_name, .sh_type = SHT_NOBITS, .flags = SHF_ALLOC | SHF_WRITE, .size = d.size, .addralign = 8 }),
        }
        try reloc_lists.append(a, .empty);
        const binding: object_emit.Binding = if (std.mem.startsWith(u8, d.name, ".")) .local else .global;
        try symbols.append(a, .{ .name = d.name, .section = @intCast(sec_index), .value = 0, .size = d.size, .binding = binding, .sym_type = .object, .defined = true });
        for (d.relocs) |r| try data_pending.append(a, .{ .sec = sec_index, .offset = r.off, .name = r.symbol });
    }

    // An undefined external callee, added once.
    for (named_relocs.items) |p| {
        if (oeIndex(symbols.items, p.name) == null) try symbols.append(a, .{ .name = p.name, .section = 0, .value = 0, .size = 0, .binding = .global, .sym_type = .notype, .defined = false });
    }
    for (named_relocs.items) |p| {
        const sym = oeIndex(symbols.items, p.name).?;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = p.r_type });
    }
    for (lo12_relocs.items) |p| {
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = p.sym, .r_type = @intFromEnum(RelocType.pcrel_lo12_i) });
    }
    // The debug variant must not drop a data global's own pointer inits: a second reconstruct
    // site is exactly where an additive field like this one silently goes missing if it is not
    // threaded through both.
    for (data_pending.items) |p| {
        const sym = oeIndex(symbols.items, p.name) orelse return error.Unsupported;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = R_RISCV_64 });
    }
    // Attach each function/data section's relocation list, before the debug sections append.
    for (sections.items, 0..) |*sec, i| sec.relocs = reloc_lists.items[i].items;

    // DWARF sections: a subprogram DIE per function (with its typed return), plus the line program.
    const subs = try a.alloc(dwarf.Subprogram, module.entries.items.len);
    for (module.entries.items, 0..) |entry, i| subs[i] = .{
        .name = entry.name,
        .low_pc = func_low.items[i],
        .high_pc = func_high.items[i],
        .ret_type = returnBaseType(entry.func),
    };

    const abbrev = try dwarf.emitAbbrev(a);
    // One line program at offset 0 of .debug_line, so link the CU to it via DW_AT_stmt_list.
    const info = try dwarf.emitInfo(a, .{ .name = source_file, .low_pc = 0, .high_pc = text_off, .subprograms = subs, .stmt_list = 0 });
    const line = try dwarf.emitLine(a, source_file, rows.items, text_off);

    // The debug sections are plain non-alloc PROGBITS. No other section refers to them.
    try sections.append(a, .{ .name = ".debug_abbrev", .sh_type = SHT_PROGBITS, .flags = 0, .bytes = abbrev, .size = abbrev.len, .addralign = 1 });
    try sections.append(a, .{ .name = ".debug_info", .sh_type = SHT_PROGBITS, .flags = 0, .bytes = info, .size = info.len, .addralign = 1 });
    try sections.append(a, .{ .name = ".debug_line", .sh_type = SHT_PROGBITS, .flags = 0, .bytes = line, .size = line.len, .addralign = 1 });

    return object_emit.emit(allocator, sections.items, symbols.items, .{ .class = .elf64, .machine = EM_RISCV, .use_rela = true });
}

/// Map a function's IR return type to a DWARF base type (C-like names), or null for a void /
/// non-primitive return. Distinct primitives get distinct names so the base-type dedup keeps them apart.
fn returnBaseType(func: *const Function) ?dwarf.BaseType {
    const ret_val = for (0..func.blocks.items.len) |bi| {
        const term = func.terminator(@enumFromInt(bi)) orelse continue;
        switch (term) {
            .ret => |r| switch (r.count) {
                0 => return null,
                1 => break r.values[0],
                else => return null, // multi-value return not representable in DWARF yet
            },
            else => {},
        }
    } else return null;

    return switch (func.types.type_kind(func.valueType(ret_val))) {
        .bool => .{ .name = "bool", .encoding = .boolean, .byte_size = 1 },
        .float => |f| switch (f) {
            .f32 => .{ .name = "float", .encoding = .float, .byte_size = 4 },
            .f64 => .{ .name = "double", .encoding = .float, .byte_size = 8 },
            // Debug-info naming only, not lowering: riscv64 has no f16 codegen yet.
            .f16 => .{ .name = "half", .encoding = .float, .byte_size = 2 },
            // Same rule for f128: debug naming only, no lowering claim.
            .f128 => .{ .name = "__float128", .encoding = .float, .byte_size = 16 },
        },
        .int => |i| blk: {
            const bytes: u8 = @intCast((i.bits + 7) / 8);
            const signed = i.signedness == .signed;
            const name: []const u8 = switch (i.bits) {
                8 => if (signed) "i8" else "u8",
                16 => if (signed) "i16" else "u16",
                32 => if (signed) "int" else "unsigned int",
                64 => if (signed) "long" else "unsigned long",
                else => if (signed) "int" else "unsigned",
            };
            break :blk .{ .name = name, .encoding = if (signed) .signed else .unsigned, .byte_size = bytes };
        },
        else => null, // ptr / vector / aggregate
    };
}

test "writes an ELF64 RISC-V relocatable header" {
    const allocator = std.testing.allocator;

    // One function "add" of two words, with a call relocation to "helper".
    const text = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const symbols = [_]Symbol{
        .{ .name = "add", .value = 0, .size = text.len, .kind = .func },
        .{ .name = "helper", .defined = false },
    };
    const relocs = [_]Reloc{
        .{ .offset = 0, .symbol = 1, .type = .jal },
    };

    const obj = Object{ .text = &text, .symbols = &symbols, .relocs = &relocs };
    const bytes = try write(allocator, obj);
    defer allocator.free(bytes);

    // ELF magic and a 64-bit little-endian RISC-V relocatable object.
    try std.testing.expectEqualSlices(u8, "\x7fELF", bytes[0..4]);
    try std.testing.expectEqual(@as(u8, 2), bytes[4]); // ELFCLASS64
    try std.testing.expectEqual(@as(u8, 1), bytes[5]); // ELFDATA2LSB
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, bytes[16..18], .little)); // ET_REL
    try std.testing.expectEqual(@as(u16, 243), std.mem.readInt(u16, bytes[18..20], .little)); // EM_RISCV
}

const Function = ir.function.Function;

/// Run `readelf flag` on `bytes` and return its stdout. Skips the test when
/// `readelf` is not on PATH.
fn readelf(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8, flag: []const u8) ![]u8 {
    const Nonce = struct {
        var counter: usize = 0;
    };
    Nonce.counter += 1;
    const name = try std.fmt.allocPrint(allocator, "vulcan-obj-{d}.o", .{Nonce.counter});
    defer allocator.free(name);
    // A unique temp dir with the child cwd set there, so relative names resolve and the
    // process cwd is never written to (which was flaky).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data = bytes });

    const argv = [_][]const u8{ "readelf", flag, name };
    const result = std.process.run(allocator, io, .{ .argv = &argv, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(result.stderr);
    return result.stdout;
}

test "readelf accepts the object and sees its symbols and relocations" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee: fn(x) -> x.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const x = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }
    // caller: fn(x) -> external(callee(x)). One intra-module call, one external.
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        const r2 = try caller.appendCall(b, t, "external", &.{r});
        caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r2) });
    }

    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "callee", &callee);
    try module.addFunction(allocator, "caller", &caller);

    const bytes = try writeModule(allocator, &module);
    defer allocator.free(bytes);

    // The symbol table: callee/caller are defined functions, external is UND.
    const syms = try readelf(allocator, io, bytes, "-s");
    defer allocator.free(syms);
    try std.testing.expect(std.mem.indexOf(u8, syms, "callee") != null);
    try std.testing.expect(std.mem.indexOf(u8, syms, "caller") != null);
    try std.testing.expect(std.mem.indexOf(u8, syms, "external") != null);
    try std.testing.expect(std.mem.indexOf(u8, syms, "FUNC") != null);

    // The relocations: every call is an R_RISCV_JAL.
    const rels = try readelf(allocator, io, bytes, "-r");
    defer allocator.free(rels);
    try std.testing.expect(std.mem.indexOf(u8, rels, "R_RISCV_JAL") != null);
}

test "readelf shows separate .rodata, .data, and .bss sections" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const b = try entry.appendBlock();
        const x = try entry.appendBlockParam(b, t);
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }
    const ro = [_]u8{ 1, 0, 0, 0 };
    const da = [_]u8{ 2, 0, 0, 0 };
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addData(allocator, "RO", &ro);
    try module.addWritable(allocator, "DA", &da);
    try module.addBss(allocator, "BS", 8);

    const bytes = try writeModule(allocator, &module);
    defer allocator.free(bytes);

    // The section headers carry the three data sections, the BSS one as NOBITS.
    const secs = try readelf(allocator, io, bytes, "-S");
    defer allocator.free(secs);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".rodata") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".data") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".bss") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, "NOBITS") != null);

    // The data symbols are OBJECT-typed.
    const syms = try readelf(allocator, io, bytes, "-s");
    defer allocator.free(syms);
    try std.testing.expect(std.mem.indexOf(u8, syms, "OBJECT") != null);
}

/// Link `obj_bytes` (a relocatable object) with the real RISC-V `ld.lld` and
/// return the raw `.text` image (via `--oformat binary`). Entry `entry` is
/// placed at `0x80000000`, the input objects laid out in the order given. Skips
/// the test when `ld.lld` is not on PATH. lld is only a comparison oracle. The
/// shipping linker is ld.zig.
fn lldLink(allocator: std.mem.Allocator, io: std.Io, objs: []const []const u8, entry: []const u8) ![]u8 {
    const Nonce = struct {
        var counter: usize = 0;
    };
    // Work in a unique temp directory with the child's cwd set there, so the argv just
    // names the files. Writing to the process cwd made this flaky (the files could be
    // missing or unwritable depending on where the test binary ran).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = tmp.dir;

    // Write each object to its own file, collecting the names for the argv.
    var obj_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (obj_names.items) |n| {
            dir.deleteFile(io, n) catch {};
            allocator.free(n);
        }
        obj_names.deinit(allocator);
    }
    for (objs) |obj_bytes| {
        Nonce.counter += 1;
        const name = try std.fmt.allocPrint(allocator, "vulcan-link-{d}.o", .{Nonce.counter});
        try obj_names.append(allocator, name);
        try dir.writeFile(io, .{ .sub_path = name, .data = obj_bytes });
    }

    Nonce.counter += 1;
    const bin_name = try std.fmt.allocPrint(allocator, "vulcan-link-{d}.bin", .{Nonce.counter});
    defer allocator.free(bin_name);
    defer dir.deleteFile(io, bin_name) catch {};

    const entry_arg = try std.fmt.allocPrint(allocator, "-e{s}", .{entry});
    defer allocator.free(entry_arg);

    // ld.lld -m elf64lriscv -e<entry> -Ttext=0x80000000 --no-dynamic-linker
    //   --oformat binary <objs...> -o <bin>
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    try argv.appendSlice(allocator, &.{ "ld.lld", "-m", "elf64lriscv", entry_arg, "-Ttext=0x80000000", "--no-dynamic-linker", "--oformat", "binary" });
    try argv.appendSlice(allocator, obj_names.items);
    try argv.appendSlice(allocator, &.{ "-o", bin_name });

    const result = std.process.run(allocator, io, .{ .argv = argv.items, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (result.term != .exited or result.term.exited != 0) {
        std.debug.print("ld.lld failed:\n{s}\n", .{result.stderr});
        return error.LinkFailed;
    }

    // 1 MiB cap is plenty for these tiny test images.
    return dir.readFileAlloc(io, bin_name, allocator, .limited(1 << 20));
}

test "real ld.lld links the object to the same bytes as the in-memory linker" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // A self-contained module (no external symbols) so the link fully resolves.
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const x = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "callee", &callee);
    try module.addFunction(allocator, "caller", &caller);

    // The in-memory linker's resolved code (already River-validated elsewhere).
    var linked = try link.compileModule(allocator, &module);
    defer linked.deinit(allocator);

    // The object, linked by the real RISC-V lld, as a raw .text image.
    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);
    const image = try lldLink(allocator, io, &.{obj}, "caller");
    defer allocator.free(image);

    // lld must produce exactly the bytes the in-memory linker does, word for word.
    try std.testing.expectEqual(linked.code.len * 4, image.len);
    for (linked.code, 0..) |word, i| {
        const got = std.mem.readInt(u32, image[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(word, got);
    }
}

test "lld resolves a call across two separately compiled objects" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee compiled entirely on its own (one object).
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32k);
        const b = try callee.appendBlock();
        const x = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }
    var callee_mod: link.Module = .{};
    defer callee_mod.deinit(allocator);
    try callee_mod.addFunction(allocator, "callee", &callee);
    const callee_obj = try writeModule(allocator, &callee_mod);
    defer allocator.free(callee_obj);

    // caller compiled on its own: "callee" stays an undefined external symbol.
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var caller_mod: link.Module = .{};
    defer caller_mod.deinit(allocator);
    try caller_mod.addFunction(allocator, "caller", &caller);
    const caller_obj = try writeModule(allocator, &caller_mod);
    defer allocator.free(caller_obj);

    // The combined in-memory link is the reference layout: callee then caller.
    var combined: link.Module = .{};
    defer combined.deinit(allocator);
    try combined.addFunction(allocator, "callee", &callee);
    try combined.addFunction(allocator, "caller", &caller);
    var linked = try link.compileModule(allocator, &combined);
    defer linked.deinit(allocator);

    // lld links the two objects (callee first) and resolves the cross-object
    // call to the same bytes the in-memory linker produces.
    const image = try lldLink(allocator, io, &.{ callee_obj, caller_obj }, "caller");
    defer allocator.free(image);

    try std.testing.expectEqual(linked.code.len * 4, image.len);
    for (linked.code, 0..) |word, i| {
        const got = std.mem.readInt(u32, image[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(word, got);
    }
}

test "our PCREL global-data resolution matches lld byte for byte" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&K), K an i32 constant. Same module as the River test in ld.zig.
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.intern(.ptr);
        const b = try entry.appendBlock();
        const p = try entry.appendGlobalAddr(b, ptr_t, "K");
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = p } });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }
    const k_bytes = [_]u8{ 42, 0, 0, 0 };
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addData(allocator, "K", &k_bytes);

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);

    // The in-memory linker and lld must resolve the PCREL_HI20/LO12 pair to
    // exactly the same bytes (entry at base, K right after it).
    var ours = try ld.linkObjects(allocator, &.{obj}, 0x80000000);
    defer ours.deinit(allocator);
    const theirs = try lldLink(allocator, io, &.{obj}, "entry");
    defer allocator.free(theirs);

    try std.testing.expectEqualSlices(u8, ours.code, theirs);
}

test "linker resolves an R_RISCV_CALL far call (matches lld, runs on River)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // A hand-built object using the standard far-call sequence `auipc ra, 0`
    // then `jalr ra, ra, 0` with one R_RISCV_CALL relocation, exactly what gcc/clang
    // emit by default. entry() far-calls callee() which returns 7.
    const words = [_]u32{
        encode.addi(.x2, .x2, -16), // entry: open frame
        encode.sd(.x1, .x2, 0), //          save ra
        encode.auipc(.x1, 0), //            R_RISCV_CALL callee (byte offset 8)
        encode.jalr(.x1, .x1, 0), //        call
        encode.ld(.x1, .x2, 0), //          restore ra
        encode.addi(.x2, .x2, 16), //       close frame
        encode.jalr(.x0, .x1, 0), //        ret
        encode.addi(.x10, .x0, 7), // callee: a0 = 7
        encode.jalr(.x0, .x1, 0), //        ret
    };
    var text: [words.len * 4]u8 = undefined;
    for (words, 0..) |w, i| putInt(text[i * 4 ..][0..4], u32, w);

    const symbols = [_]Symbol{
        .{ .name = "entry", .value = 0, .size = 7 * 4, .kind = .func },
        .{ .name = "callee", .value = 7 * 4, .size = 2 * 4, .kind = .func },
    };
    const relocs = [_]Reloc{
        .{ .offset = 8, .symbol = 1, .type = .call }, // the auipc, against callee
    };
    const obj = try write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
    defer allocator.free(obj);

    // The linker and lld must agree, and the result must run on River.
    var ours = try ld.linkObjects(allocator, &.{obj}, harness.load_address);
    defer ours.deinit(allocator);
    const theirs = try lldLink(allocator, io, &.{obj}, "entry");
    defer allocator.free(theirs);
    try std.testing.expectEqualSlices(u8, ours.code, theirs);

    const img_words = try allocator.alloc(u32, ours.code.len / 4);
    defer allocator.free(img_words);
    for (img_words, 0..) |*w, i| w.* = std.mem.readInt(u32, ours.code[i * 4 ..][0..4], .little);
    try std.testing.expectEqual(@as(i64, 7), try harness.runCode(io, allocator, img_words, &.{}, harness.river));
}

test "writeModuleWithDebug emits DWARF readelf reads (subprograms + CU line linkage)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // helper(x) -> x*3. main(x) -> helper(x). Two functions, a real intra-module call.
    var helper = Function.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(i32k);
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        const m = try helper.appendArithImm(b, t, .add, x, 5);
        helper.setTerminator(b, .{ .ret = ir.function.Ret.one(m) });
    }
    var main_f = Function.init(allocator);
    defer main_f.deinit();
    {
        const t = try main_f.types.intern(i32k);
        const b = try main_f.appendBlock();
        const x = try main_f.appendBlockParam(b, t);
        const r = try main_f.appendCall(b, t, "helper", &.{x});
        main_f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "helper", &helper);
    try module.addFunction(allocator, "main", &main_f);

    const obj = try writeModuleWithDebug(allocator, &module, "mod.glsl");
    defer allocator.free(obj);

    // The three DWARF sections are present in the object.
    const secs = readelf(allocator, io, obj, "-S") catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(secs);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".debug_abbrev") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".debug_info") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs, ".debug_line") != null);

    // The CU decodes: both functions as subprograms, the CU linked to its line program, typed return.
    const info = readelf(allocator, io, obj, "--debug-dump=info") catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(info);
    try std.testing.expect(std.mem.indexOf(u8, info, "DW_TAG_subprogram") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "helper") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "main") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "DW_AT_stmt_list") != null);
    try std.testing.expect(std.mem.indexOf(u8, info, "int") != null); // i32 return -> "int"
}
