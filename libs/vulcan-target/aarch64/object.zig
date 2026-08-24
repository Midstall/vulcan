//! This file emits an ELF64 relocatable object (`ET_REL`, `EM_AARCH64`). Each function
//! becomes its own `.text.<name>` section with an `STT_FUNC` symbol at offset 0. Each
//! data global (`module.data`) becomes its own `.rodata.<name>`, `.data.<name>`, or
//! `.bss.<name>` section with an `STT_OBJECT` symbol at offset 0. Each call (`bl`)
//! becomes an `R_AARCH64_CALL26` relocation against the callee symbol. The symbol is
//! undefined if external. A `global_addr`'s `adrp`/`add` pair becomes an
//! `R_AARCH64_ADR_PREL_PG_HI21` / `R_AARCH64_ADD_ABS_LO12_NC` relocation pair against
//! the symbol. The shared `object_emit.emit` serializes the neutral section, symbol, and
//! relocation lists into the ELF bytes. `readelf` and a system AArch64 linker accept the
//! output. `ld.zig` is Vulcan's own linker for this object format.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");
const dwarf = @import("../dwarf.zig");
const object_emit = @import("../object_emit.zig");

const Function = ir.function.Function;

pub const Error = isel.Error;

/// A symbol's binding. Locals must precede globals in the symbol table.
pub const Binding = enum { local, global };

/// A symbol's type. `func` is an entry point. `object` is a data object. `notype` is
/// unknown.
pub const SymKind = enum { notype, func, object };

/// The allocatable output sections for a symbol. `.text` holds code. `.rodata` holds
/// read-only data. `.data` holds writable data. `.bss` holds zero-initialized data.
/// `.bss` uses memory but no file bytes.
pub const SectionKind = enum { text, rodata, data, bss };

/// One symbol table entry. A defined symbol lives in `section` at `value`. An
/// undefined symbol (`defined = false`) is external. The linker must resolve it.
pub const Symbol = struct {
    name: []const u8,
    value: u64 = 0,
    size: u64 = 0,
    binding: Binding = .global,
    kind: SymKind = .func,
    defined: bool = true,
    section: SectionKind = .text,
};

/// The AArch64 relocation types this file emits (the architectural `R_AARCH64_*` codes).
pub const RelocType = enum(u32) {
    /// `R_AARCH64_ADR_PREL_PG_HI21`. It patches an `adrp` instruction's page-relative
    /// immediate. This is the high half of a `global_addr`'s adrp/add pair.
    adr_prel_pg_hi21 = 275,
    /// `R_AARCH64_ADD_ABS_LO12_NC`. It patches an `add` instruction's 12-bit page-offset
    /// immediate. This is the low half of a `global_addr`'s adrp/add pair.
    add_abs_lo12_nc = 277,
    /// `R_AARCH64_CALL26`. It patches a `bl` or `b` instruction's 26-bit immediate. The
    /// call range is plus or minus 128 MiB.
    call26 = 283,
    /// `R_AARCH64_ADR_GOT_PAGE`. It patches an `adrp` instruction's page-relative
    /// immediate to the page of the symbol's GOT entry. This is the high half of a
    /// GOT-indirect `adrp`/`ldr` pair.
    adr_got_page = 311,
    /// `R_AARCH64_LD64_GOT_LO12_NC`. It patches an `ldr` instruction's 12-bit unsigned
    /// offset to the GOT entry's lo12 value shifted right by 3. This is the low half of a
    /// GOT-indirect `adrp`/`ldr` pair.
    ld64_got_lo12_nc = 312,
};

/// A relocation applied to a `.text` byte offset against a symbol.
pub const Reloc = struct {
    offset: u64,
    symbol: u32,
    type: RelocType,
    addend: i64 = 0,
};

/// `R_AARCH64_ABS64`: a 64-bit absolute address (`S + A`) written into a data section
/// slot. A pointer-initialized global (`int *p = &g;`) carries this relocation in
/// `.rela.data` or `.rela.rodata`. The linker must patch the 8-byte slot that holds the
/// pointer to the target symbol's runtime address. The linker turns it into an
/// `R_AARCH64_RELATIVE` dynamic relocation for a PIE or shared object, or a direct
/// absolute write for a non-PIE executable.
pub const R_AARCH64_ABS64: u32 = 257;

/// A relocation applied to a data section (`.data` or `.rodata`) slot against a symbol.
/// At byte `offset` within `section`, an `R_AARCH64_ABS64` relocation writes `symbol`'s
/// runtime address plus `addend`. This is emitted into `.rela.data` or `.rela.rodata`,
/// keyed by `section`. A data global's own pointer initializers are now carried in the
/// object, not dropped.
pub const DataRelocEntry = struct {
    section: SectionKind,
    offset: u64,
    symbol: u32,
    addend: i64 = 0,
};

/// A non-alloc PROGBITS section carried without change (for example a DWARF `.debug_*`
/// blob).
pub const DebugSection = struct { name: []const u8, bytes: []const u8 };

/// A relocatable object: code and data section blobs, the symbol table, and the
/// relocations that apply to `.text`. `.bss` has no bytes, only a size. The `debug`
/// sections (DWARF) are appended as plain PROGBITS, so a compiled object can carry
/// debug info.
pub const Object = struct {
    text: []const u8,
    rodata: []const u8 = &.{},
    data: []const u8 = &.{},
    bss_size: u64 = 0,
    symbols: []const Symbol,
    relocs: []const Reloc,
    /// Relocations applied to the `.data` and `.rodata` sections themselves (pointer
    /// initializers), emitted as `.rela.data`/`.rela.rodata`. This list is empty for a
    /// code-only or const-only object.
    data_relocs: []const DataRelocEntry = &.{},
    debug: []const DebugSection = &.{},
};

const EM_AARCH64: u16 = 183;
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

/// Serialize a single-section `Object` into an ELF64 AArch64 relocatable object. This is
/// the raw entry point that hand-built test objects use. It maps the `.text`, `.rodata`,
/// `.data`, and `.bss` blobs and the symbol and relocation lists into the neutral form,
/// then lets `object_emit.emit` write the ELF bytes. The emitter owns the symbol sort and
/// the reloc-index remap, so the symbols pass in their natural order. The caller owns the
/// result.
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

    // The data pointer-init relocations, grouped by the section they modify. Each one is
    // an `R_AARCH64_ABS64` against the target symbol.
    var rodata_relocs: std.ArrayList(object_emit.OutReloc) = .empty;
    var data_relocs: std.ArrayList(object_emit.OutReloc) = .empty;
    for (obj.data_relocs) |dr| {
        const e: object_emit.OutReloc = .{ .offset = dr.offset, .symbol = dr.symbol, .r_type = R_AARCH64_ABS64, .addend = dr.addend };
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

    return object_emit.emit(allocator, sections.items, symbols, .{ .class = .elf64, .machine = EM_AARCH64, .use_rela = true });
}

/// Map an isel relocation's `kind` to the matching ELF relocation type. The `kind` says
/// which half of a call, or an adrp/add pair, the relocation patches.
fn relocTypeOf(kind: isel.Kind) RelocType {
    return switch (kind) {
        .call => .call26,
        .adrp_pg => .adr_prel_pg_hi21,
        .add_pgoff => .add_abs_lo12_nc,
        .got_pg => .adr_got_page,
        .got_lo12 => .ld64_got_lo12_nc,
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

/// Build a data global's own section name. A leading `.` is stripped from the symbol
/// name, so a local `.str.N` becomes `.rodata.str.N`, not `..rodata..str.N`.
fn dataSectionName(a: std.mem.Allocator, class: []const u8, name: []const u8) Error![]u8 {
    const bare = if (std.mem.startsWith(u8, name, ".")) name[1..] else name;
    return std.fmt.allocPrint(a, ".{s}.{s}", .{ class, bare });
}

/// Compile every function in `module`, and serialize them and its data globals into one
/// ELF relocatable object. Each function becomes its own `.text.<name>` section with a
/// defined `STT_FUNC` symbol at offset 0. Each data global becomes its own
/// `.rodata.<name>`, `.data.<name>`, or `.bss.<name>` section with an `STT_OBJECT` symbol
/// at offset 0. Each `bl` becomes an `R_AARCH64_CALL26` relocation, rebased to its own
/// section. Each `global_addr`'s `adrp`/`add` pair becomes an
/// `R_AARCH64_ADR_PREL_PG_HI21`/`R_AARCH64_ADD_ABS_LO12_NC` pair. Both kinds of
/// relocation target the referenced symbol, which is undefined if external. The shared
/// `object_emit.emit` writes the ELF bytes. The caller owns the result.
pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var sections: std.ArrayList(object_emit.OutSection) = .empty;
    var symbols: std.ArrayList(object_emit.OutSymbol) = .empty;
    // One relocation list per section. Its index matches `sections`.
    var reloc_lists: std.ArrayList(std.ArrayList(object_emit.OutReloc)) = .empty;

    // A text relocation recorded by its owning section and target name. Its target symbol
    // index resolves after every symbol is known.
    const TextReloc = struct { sec: usize, offset: u64, name: []const u8, r_type: u32 };
    var text_relocs: std.ArrayList(TextReloc) = .empty;
    // A data pointer-init relocation, likewise resolved by name after every symbol exists.
    const DataR = struct { sec: usize, offset: u64, name: []const u8 };
    var data_pending: std.ArrayList(DataR) = .empty;

    const caps: isel.ModelCaps = if (module.model) |m| isel.capsForModel(m) else .{};
    for (module.functions.items) |entry| {
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
        for (compiled.relocs) |r| try text_relocs.append(a, .{ .sec = sec_index, .offset = @as(u64, r.offset) * 4, .name = r.symbol, .r_type = @intFromEnum(relocTypeOf(r.kind)) });
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

    // An undefined external callee. A text relocation whose target names no defined
    // symbol is an import. Add it once as an undefined `notype` global.
    for (text_relocs.items) |p| {
        if (oeIndex(symbols.items, p.name) == null) try symbols.append(a, .{ .name = p.name, .section = 0, .value = 0, .size = 0, .binding = .global, .sym_type = .notype, .defined = false });
    }

    // Resolve every text relocation to its symbol index. Its `symbol` is the index into
    // this symbol list. The emitter remaps it after its own symbol sort.
    for (text_relocs.items) |p| {
        const sym = oeIndex(symbols.items, p.name).?;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = p.r_type });
    }
    // Resolve every data pointer-init relocation. Its target is an internally defined
    // global. It is an error if the target is missing.
    for (data_pending.items) |p| {
        const sym = oeIndex(symbols.items, p.name) orelse return error.Unsupported;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = R_AARCH64_ABS64 });
    }

    // Attach each section's relocation list.
    for (sections.items, 0..) |*sec, i| sec.relocs = reloc_lists.items[i].items;

    return object_emit.emit(allocator, sections.items, symbols.items, .{ .class = .elf64, .machine = EM_AARCH64, .use_rela = true });
}

/// Like `writeModule`, but also emits inline DWARF (`.debug_abbrev`, `.debug_info`,
/// `.debug_line`) describing each function's name and PC range. It also maps code offsets
/// to `source_file` line numbers, taken from the `debug.line` IR attributes. The DWARF
/// program numbers PC ranges from a single running text offset, so the line and range
/// tables stay self-consistent across the per-function sections. The result is a real
/// relocatable object that carries debug info for objdump or gdb. The caller owns the
/// bytes.
pub fn writeModuleWithDebug(allocator: std.mem.Allocator, module: *const link.Module, source_file: []const u8) Error![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var sections: std.ArrayList(object_emit.OutSection) = .empty;
    var symbols: std.ArrayList(object_emit.OutSymbol) = .empty;
    var reloc_lists: std.ArrayList(std.ArrayList(object_emit.OutReloc)) = .empty;

    const TextReloc = struct { sec: usize, offset: u64, name: []const u8, r_type: u32 };
    var text_relocs: std.ArrayList(TextReloc) = .empty;

    var rows: std.ArrayList(dwarf.LineRow) = .empty;
    // Each function's DWARF PC range, numbered from one running text offset.
    var func_low: std.ArrayList(u64) = .empty;
    var func_high: std.ArrayList(u64) = .empty;
    var text_off: u64 = 0;

    const caps: isel.ModelCaps = if (module.model) |m| isel.capsForModel(m) else .{};
    for (module.functions.items) |entry| {
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
        for (compiled.relocs) |r| try text_relocs.append(a, .{ .sec = sec_index, .offset = @as(u64, r.offset) * 4, .name = r.symbol, .r_type = @intFromEnum(relocTypeOf(r.kind)) });
        // The DWARF PC range and line rows use the running text offset.
        try func_low.append(a, text_off);
        try func_high.append(a, text_off + code.len);
        for (compiled.lines) |e| try rows.append(a, .{ .address = text_off + e.offset, .line = e.line });
        text_off += code.len;
    }

    // An undefined external callee, added once.
    for (text_relocs.items) |p| {
        if (oeIndex(symbols.items, p.name) == null) try symbols.append(a, .{ .name = p.name, .section = 0, .value = 0, .size = 0, .binding = .global, .sym_type = .notype, .defined = false });
    }
    for (text_relocs.items) |p| {
        const sym = oeIndex(symbols.items, p.name).?;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = p.r_type });
    }
    // Attach each function section's relocation list, before the debug sections append.
    for (sections.items, 0..) |*sec, i| sec.relocs = reloc_lists.items[i].items;

    // DWARF: one subprogram DIE per function. Its PC range is the function's running
    // text placement. It carries the function's IR return type as a base-type reference,
    // so a debugger can show a typed signature.
    const subs = try a.alloc(dwarf.Subprogram, module.functions.items.len);
    for (module.functions.items, 0..) |entry, i| subs[i] = .{
        .name = entry.name,
        .low_pc = func_low.items[i],
        .high_pc = func_high.items[i],
        .ret_type = returnBaseType(entry.func),
    };

    const abbrev = try dwarf.emitAbbrev(a);
    // The object carries one line program at offset 0 of .debug_line. Link the
    // compilation unit to it with DW_AT_stmt_list. A debugger can now go from a
    // subprogram DIE straight to its source lines.
    const info = try dwarf.emitInfo(a, .{ .name = source_file, .low_pc = 0, .high_pc = text_off, .subprograms = subs, .stmt_list = 0 });
    const line = try dwarf.emitLine(a, source_file, rows.items, text_off);

    // The debug sections are plain non-alloc PROGBITS. No other section refers to them.
    try sections.append(a, .{ .name = ".debug_abbrev", .sh_type = SHT_PROGBITS, .flags = 0, .bytes = abbrev, .size = abbrev.len, .addralign = 1 });
    try sections.append(a, .{ .name = ".debug_info", .sh_type = SHT_PROGBITS, .flags = 0, .bytes = info, .size = info.len, .addralign = 1 });
    try sections.append(a, .{ .name = ".debug_line", .sh_type = SHT_PROGBITS, .flags = 0, .bytes = line, .size = line.len, .addralign = 1 });

    return object_emit.emit(allocator, sections.items, symbols.items, .{ .class = .elf64, .machine = EM_AARCH64, .use_rela = true });
}

/// Map a function's IR return type, the type of its `ret` value, to a DWARF base type.
/// Return null for a void return or a non-primitive (aggregate) return. The names are
/// C-like, so a debugger prints a natural signature. Distinct primitives get distinct
/// names, so the `.debug_info` base-type dedup keeps them apart.
fn returnBaseType(func: *const Function) ?dwarf.BaseType {
    const ret_val = for (0..func.blocks.items.len) |bi| {
        const term = func.terminator(@enumFromInt(bi)) orelse continue;
        switch (term) {
            .ret => |r| switch (r.count) {
                0 => return null,
                1 => break r.values[0],
                else => return null, // A multi-value return has no DWARF representation yet.
            },
            else => {},
        }
    } else return null;

    return switch (func.types.type_kind(func.valueType(ret_val))) {
        .bool => .{ .name = "bool", .encoding = .boolean, .byte_size = 1 },
        .float => |f| switch (f) {
            .f32 => .{ .name = "float", .encoding = .float, .byte_size = 4 },
            .f64 => .{ .name = "double", .encoding = .float, .byte_size = 8 },
            // This name is for debug info only. It does not affect lowering. AArch64 f16
            // codegen is future work.
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
        else => null, // A pointer, vector, or aggregate type has no base-type DIE here.
    };
}

test "returnBaseType reads the ret value's IR type" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, i32_t);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    const bt = returnBaseType(&func).?;
    try std.testing.expectEqualStrings("int", bt.name);
    try std.testing.expectEqual(dwarf.Encoding.signed, bt.encoding);
    try std.testing.expectEqual(@as(u8, 4), bt.byte_size);
}

test "returnBaseType is null for a void return" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const b = try func.appendBlock();
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    try std.testing.expectEqual(@as(?dwarf.BaseType, null), returnBaseType(&func));
}

test "writes an ELF64 AArch64 relocatable header" {
    const allocator = std.testing.allocator;
    const text = [_]u8{ 0, 0, 0, 0 };
    const symbols = [_]Symbol{
        .{ .name = "f", .value = 0, .size = text.len, .kind = .func },
        .{ .name = "ext", .defined = false },
    };
    const relocs = [_]Reloc{.{ .offset = 0, .symbol = 1, .type = .call26 }};
    const bytes = try write(allocator, .{ .text = &text, .symbols = &symbols, .relocs = &relocs });
    defer allocator.free(bytes);
    try std.testing.expectEqualSlices(u8, "\x7fELF", bytes[0..4]);
    try std.testing.expectEqual(@as(u8, 2), bytes[4]); // ELFCLASS64
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, bytes[16..18], .little)); // ET_REL
    try std.testing.expectEqual(@as(u16, 183), std.mem.readInt(u16, bytes[18..20], .little)); // EM_AARCH64
}

test "readelf accepts the emitted AArch64 object (cross-check)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // dbl(a) = a + a, and caller(x) = dbl(x) + 1. This call produces a CALL26 relocation.
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(i32k);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const d = try caller.appendCall(b, t, "dbl", &.{x});
        const r = try caller.appendArithImm(b, t, .add, d, 1);
        caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "caller", &caller);
    try module.addFunction(allocator, "dbl", &dbl);

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "out.o", .data = obj });

    // Run readelf with its cwd set to the temp dir, so the path is the bare file.
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "readelf", "-hr", "out.o" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // readelf unavailable
        else => return e,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    // A standard tool recognizes the machine and the relocation type.
    try std.testing.expect(std.mem.indexOf(u8, result.stdout, "AArch64") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.stdout, "R_AARCH64_CALL26") != null);
}

test "readelf shows a .rodata section and ADRP/ADD relocations for a global_addr load" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() returns *(&K), where K is an i32 rodata constant. This is the exact shape
    // that isel's `.global_addr` arm, the adrp/add pair, lowers to an
    // ADR_PREL_PG_HI21/ADD_ABS_LO12_NC pair.
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.intern(.ptr);
        const b = try entry.appendBlock();
        const g = try entry.appendGlobalAddr(b, ptr_t, "K");
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = g } });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }
    const k_bytes = [_]u8{ 42, 0, 0, 0 };
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    try module.addData(allocator, "K", &k_bytes); // .rodata

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "gd.o", .data = obj });

    const secs = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-S", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(secs.stdout);
    defer allocator.free(secs.stderr);
    if (secs.term != .exited or secs.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, ".rodata") != null);

    // The `-W` (wide) flag stops readelf from truncating the long `R_AARCH64_*` names to
    // fit its default column width.
    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-rW", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADR_PREL_PG_HI21") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADD_ABS_LO12_NC") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_CALL26") == null);

    const syms = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-s", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(syms.stdout);
    defer allocator.free(syms.stderr);
    try std.testing.expect(std.mem.indexOf(u8, syms.stdout, "OBJECT") != null);
}

test "readelf shows GOT relocations for a via_got global_addr load (data import)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() returns *(&G), where G is an imported data symbol in another shared
    // object. This is the exact shape that isel's `.global_addr` GOT arm, the adrp/ldr
    // pair, lowers to an ADR_GOT_PAGE/LD64_GOT_LO12_NC pair. G is left undefined (no
    // addData). Its address loads from the GOT at run time.
    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.intern(.ptr);
        const b = try entry.appendBlock();
        const g = try entry.appendGlobalAddrGot(b, ptr_t, "G");
        const v = try entry.appendInst(b, t, .{ .load = .{ .ptr = g } });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "got.o", .data = obj });

    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-rW", "got.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    if (rels.term != .exited or rels.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADR_GOT_PAGE") != null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_LD64_GOT_LO12_NC") != null);
    // The GOT path replaces the direct pair, so those must NOT appear for G.
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADR_PREL_PG_HI21") == null);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_AARCH64_ADD_ABS_LO12_NC") == null);
}
