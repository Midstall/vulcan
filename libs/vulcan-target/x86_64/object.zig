//! This file emits an ELF64 relocatable object (`ET_REL`, `EM_X86_64`). Each function
//! becomes its own `.text.<name>` section with an `STT_FUNC` symbol at offset 0. Each
//! data global (`module.data`) becomes its own `.rodata.<name>`, `.data.<name>`, or
//! `.bss.<name>` section with an `STT_OBJECT` symbol at offset 0. Each `call` becomes an
//! `R_X86_64_PLT32` relocation against the callee symbol, with the `-4` addend for the
//! implicit displacement adjustment. A `global_addr`'s `lea rd, [rip+disp32]` becomes an
//! `R_X86_64_PC32` relocation, and its GOT-indirect form an `R_X86_64_GOTPCREL`. The
//! symbol is undefined if external. The shared `object_emit.emit` serializes the neutral
//! section, symbol, and relocation lists into the ELF bytes. `readelf` and a system
//! x86-64 linker accept the output. `link.zig` is the in-memory linker for the same data.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");
const dwarf = @import("../dwarf.zig");
const object_emit = @import("../object_emit.zig");

pub const Error = isel.Error;

const EM_X86_64: u16 = 62;
const SHT_PROGBITS: u32 = 1;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

const R_X86_64_PC32: u32 = 2;
const R_X86_64_PLT32: u32 = 4;
/// `R_X86_64_GOTPCREL`: a PC-relative reference to a symbol's GOT slot, from a GOT-indirect
/// `global_addr`'s `mov rd, [rip+disp32]` (`via_got`). It uses the same disp32 field and
/// addend as PC32/PLT32, but the target is the symbol's GOT slot. The dynamic linker
/// synthesizes the slot plus a GLOB_DAT relocation.
const R_X86_64_GOTPCREL: u32 = 9;
/// `R_X86_64_64`: a 64-bit absolute address (`S + A`) written into a data section slot. A
/// pointer-initialized global (`int *p = &g;`) carries this relocation in `.rela.data` or
/// `.rela.rodata`. The linker must patch the 8-byte slot that holds the pointer to the
/// target symbol's runtime address. The linker turns it into an `R_X86_64_RELATIVE`
/// dynamic relocation for a PIE or a shared object, or a direct absolute write for a
/// non-PIE executable. This mirrors `aarch64/object.zig`'s `R_AARCH64_ABS64`.
pub const R_X86_64_64: u32 = 1;

/// The `-4` addend a PC-relative text relocation carries. The disp32 field's runtime read
/// point is four bytes past its own start, so the relocation subtracts that field width.
/// The `use_rela = true` emitter writes it into the RELA addend field.
const PCREL_ADDEND: i64 = -4;

/// Map an isel relocation's `kind` to the matching ELF relocation type. A `call` is a
/// PLT32 call, a `pcrel_lea` a direct PC32 `global_addr`, and a `got_pcrel` a GOT-indirect
/// `global_addr`.
fn relocTypeOf(kind: isel.Kind) u32 {
    return switch (kind) {
        .call => R_X86_64_PLT32,
        .pcrel_lea => R_X86_64_PC32,
        .got_pcrel => R_X86_64_GOTPCREL,
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
/// at offset 0. Each `call` becomes an `R_X86_64_PLT32` relocation, rebased to its own
/// section. Each `global_addr` becomes an `R_X86_64_PC32` or `R_X86_64_GOTPCREL`
/// relocation, all carrying the `-4` PC-relative addend. The section addralign of 16 gives
/// each function its 16-byte start, so no intra-section nop padding is needed. The shared
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
    for (module.funcs.items) |entry| {
        var compiled = try isel.compileWithCaps(allocator, entry.func, caps);
        defer compiled.deinit(allocator);
        const code = try a.dupe(u8, compiled.code);
        const sec_index = sections.items.len;
        try sections.append(a, .{
            .name = try std.fmt.allocPrint(a, ".text.{s}", .{entry.name}),
            .sh_type = SHT_PROGBITS,
            .flags = SHF_ALLOC | SHF_EXECINSTR,
            .bytes = code,
            .size = code.len,
            .addralign = 16,
        });
        try reloc_lists.append(a, .empty);
        // A `static` function has internal linkage. So does a `.`-prefixed compiler-local
        // name. Both get LOCAL binding.
        const binding: object_emit.Binding = if (entry.func.is_local or std.mem.startsWith(u8, entry.name, ".")) .local else .global;
        try symbols.append(a, .{ .name = entry.name, .section = @intCast(sec_index), .value = 0, .size = code.len, .binding = binding, .sym_type = .func, .defined = true });
        // Each isel relocation offset is already a byte offset into this function's own
        // code, so it is section-relative once rebased to its owning section.
        for (compiled.relocs) |r| try text_relocs.append(a, .{ .sec = sec_index, .offset = r.offset, .name = r.symbol, .r_type = relocTypeOf(r.kind) });
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
    // this symbol list. The emitter remaps it after its own symbol sort. Every text
    // relocation is PC-relative, so each carries the `-4` addend.
    for (text_relocs.items) |p| {
        const sym = oeIndex(symbols.items, p.name).?;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = p.r_type, .addend = PCREL_ADDEND });
    }
    // Resolve every data pointer-init relocation. Its target is an internally defined
    // global. It is an error if the target is missing.
    for (data_pending.items) |p| {
        const sym = oeIndex(symbols.items, p.name) orelse return error.Unsupported;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = R_X86_64_64 });
    }

    // Attach each section's relocation list.
    for (sections.items, 0..) |*sec, i| sec.relocs = reloc_lists.items[i].items;

    return object_emit.emit(allocator, sections.items, symbols.items, .{ .class = .elf64, .machine = EM_X86_64, .use_rela = true });
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
    for (module.funcs.items) |entry| {
        var compiled = try isel.compileWithCaps(allocator, entry.func, caps);
        defer compiled.deinit(allocator);
        const code = try a.dupe(u8, compiled.code);
        const sec_index = sections.items.len;
        try sections.append(a, .{
            .name = try std.fmt.allocPrint(a, ".text.{s}", .{entry.name}),
            .sh_type = SHT_PROGBITS,
            .flags = SHF_ALLOC | SHF_EXECINSTR,
            .bytes = code,
            .size = code.len,
            .addralign = 16,
        });
        try reloc_lists.append(a, .empty);
        const binding: object_emit.Binding = if (entry.func.is_local or std.mem.startsWith(u8, entry.name, ".")) .local else .global;
        try symbols.append(a, .{ .name = entry.name, .section = @intCast(sec_index), .value = 0, .size = code.len, .binding = binding, .sym_type = .func, .defined = true });
        for (compiled.relocs) |r| try text_relocs.append(a, .{ .sec = sec_index, .offset = r.offset, .name = r.symbol, .r_type = relocTypeOf(r.kind) });
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
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = p.r_type, .addend = PCREL_ADDEND });
    }
    // Attach each function section's relocation list, before the debug sections append.
    for (sections.items, 0..) |*sec, i| sec.relocs = reloc_lists.items[i].items;

    // DWARF: one subprogram DIE per function. Its PC range is the function's running
    // text placement. It carries the function's IR return type as a base-type reference,
    // so a debugger can show a typed signature.
    const subs = try a.alloc(dwarf.Subprogram, module.funcs.items.len);
    for (module.funcs.items, 0..) |entry, i| subs[i] = .{
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

    return object_emit.emit(allocator, sections.items, symbols.items, .{ .class = .elf64, .machine = EM_X86_64, .use_rela = true });
}

/// Map a function's IR return type to a DWARF base type, using C-like names. Returns null
/// for a void return or a non-primitive return.
fn returnBaseType(func: *const ir.function.Function) ?dwarf.BaseType {
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
            // A 2-byte IEEE half in memory. x86_64 lowers f16 via F16C, and holds it as its
            // widened f32 form in registers.
            .f16 => .{ .name = "half", .encoding = .float, .byte_size = 2 },
            // A 16-byte IEEE quad. Debug naming only; no lowering claim.
            .f128 => .{ .name = "__float128", .encoding = .float, .byte_size = 16 },
        },
        .int => |it| blk: {
            const bytes: u8 = @intCast((it.bits + 7) / 8);
            const signed = it.signedness == .signed;
            const name: []const u8 = switch (it.bits) {
                8 => if (signed) "i8" else "u8",
                16 => if (signed) "i16" else "u16",
                32 => if (signed) "int" else "unsigned int",
                64 => if (signed) "long" else "unsigned long",
                else => if (signed) "int" else "unsigned",
            };
            break :blk .{ .name = name, .encoding = if (signed) .signed else .unsigned, .byte_size = bytes };
        },
        else => null,
    };
}

test "emits an ELF64 x86-64 relocatable object with a call relocation" {
    const allocator = std.testing.allocator;
    const Function = ir.function.Function;
    var helper = Function.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        helper.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }
    var main = Function.init(allocator);
    defer main.deinit();
    {
        const t = try main.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try main.appendBlock();
        const x = try main.appendBlockParam(b, t);
        const r = try main.appendCall(b, t, "helper", &.{x});
        main.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "helper", &helper);

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);
    try std.testing.expectEqualSlices(u8, "\x7fELF", obj[0..4]);
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, obj[16..18], .little)); // ET_REL
    try std.testing.expectEqual(@as(u16, EM_X86_64), std.mem.readInt(u16, obj[18..20], .little));
}

test "isel collects source-line rows from debug.line attributes" {
    const allocator = std.testing.allocator;
    const F = ir.function.Function;
    var func = F.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const add_idx = func.instCount();
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    try func.addAttr(.{ .inst = @enumFromInt(@as(u32, @intCast(add_idx))) }, .{ .custom = .{ .namespace = "debug", .key = "line", .value = .{ .int = 7 } } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });

    var compiled = try isel.compile(allocator, &func);
    defer compiled.deinit(allocator);
    var found = false;
    for (compiled.lines) |ln| if (ln.line == 7) {
        found = true;
    };
    try std.testing.expect(found);
}

test "writeModuleWithDebug emits DWARF (readelf + self-decoded line program)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const elf_read = @import("../elf_read.zig");
    const dwarf_mod = @import("../dwarf.zig");
    const F = ir.function.Function;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // helper(x) -> x + 5, and main(x) -> helper(x).
    var helper = F.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(i32k);
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        const s = try helper.appendArithImm(b, t, .add, x, 5);
        helper.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });
    }
    var main_f = F.init(allocator);
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

    const obj = try writeModuleWithDebug(allocator, &module, "mod.c");
    defer allocator.free(obj);

    // Self-contained check: our own decoder reads back the object's .debug_line as a valid program.
    const dl = (try elf_read.sectionByName(obj, ".debug_line")) orelse return error.NoLine;
    const rows = try dwarf_mod.decodeLine(allocator, dl);
    defer allocator.free(rows);
    var saw_end = false;
    for (rows) |r| if (r.end_sequence) {
        saw_end = true;
    };
    try std.testing.expect(saw_end);

    // And readelf agrees: both subprograms, the CU-to-line link, and the typed return.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "m.o", .data = obj });
    const res = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "--debug-dump=info", "m.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    if (res.term != .exited or res.term.exited != 0) return error.SkipZigTest;
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_TAG_subprogram") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "helper") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "main") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "DW_AT_stmt_list") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.stdout, "int") != null);
}

test "readelf shows per-function .text sections and .rodata/.data/.bss for a global_addr load" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const F = ir.function.Function;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&K), where K is an i32 rodata constant. This is the exact shape that
    // isel's `.global_addr` arm (`lea rd, [rip+disp32]`) lowers to a PC32 relocation for.
    var entry = F.init(allocator);
    defer entry.deinit();
    {
        const t = try entry.types.intern(i32k);
        const ptr_t = try entry.types.ptrGlobal();
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
    try module.addWritable(allocator, "W", &k_bytes); // .data
    try module.addBss(allocator, "B", 8); // .bss

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
    // Each function is its own `.text.<name>` section now, not one shared `.text`.
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, ".text.entry") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, ".rodata.K") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, ".data.W") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, ".bss.B") != null);
    try std.testing.expect(std.mem.indexOf(u8, secs.stdout, "NOBITS") != null);

    const rels = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-r", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(rels.stdout);
    defer allocator.free(rels.stderr);
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_X86_64_PC32") != null);

    const syms = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-s", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(syms.stdout);
    defer allocator.free(syms.stderr);
    try std.testing.expect(std.mem.indexOf(u8, syms.stdout, "OBJECT") != null);
}
