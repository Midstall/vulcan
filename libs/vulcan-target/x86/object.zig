//! ELF32 relocatable object (ET_REL, EM_386) emission. Each function becomes its own
//! `.text.<name>` section with an STT_FUNC symbol at offset 0. Each data global
//! (`global_addr` target) becomes its own `.rodata.<name>`, `.data.<name>`, or
//! `.bss.<name>` section with an STT_OBJECT symbol at offset 0. Every `call` becomes an
//! R_386_PC32 relocation and every `global_addr` becomes an R_386_32 (or R_386_GOT32 for a
//! GOT-indirect load) relocation against its symbol. i386 uses SHT_REL, so the addend is
//! stored implicitly in the relocated field itself (-4 for a call's PC-relative
//! displacement, 0 for an absolute address). The addend stays baked into the section bytes
//! here, because the shared `object_emit.emit` with `use_rela = false` writes SHT_REL
//! entries that carry no addend field. `link.zig` is the in-memory linker.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");
const object_emit = @import("../object_emit.zig");

pub const Error = isel.Error;

const EM_386: u16 = 3;
const SHT_PROGBITS: u32 = 1;
const SHT_NOBITS: u32 = 8;
const SHF_WRITE: u64 = 0x1;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;

/// `R_386_32`: a plain 32-bit absolute address (`S + A`). It doubles as a `.text`
/// relocation (a `global_addr` load's `mov rd, imm32`) AND a data-section relocation (a
/// pointer-initialized global's `int *p = &g;` slot). Which list a given entry feeds is
/// decided by which section its SHT_REL table targets (`elf.zig`'s `parseObject32`), not by
/// this numeric code. i386 uses SHT_REL (no addend field): the slot itself holds 0.
const R_386_32: u32 = 1;
/// `R_386_PC32`: a PC-relative call displacement (`S + A - P`). The addend is -4, baked
/// into the rel32 field of the `call` instruction.
const R_386_PC32: u32 = 2;
/// `R_386_GOT32`: a GOT-indirect `global_addr`'s `mov rd, [abs32]` (`via_got`) reference to
/// a symbol's GOT slot (a data import). The dynamic linker synthesizes the GOT slot plus an
/// R_386_GLOB_DAT and patches the abs32 field to the slot's vaddr.
const R_386_GOT32: u32 = 3;

/// Map an isel relocation's `kind` to the matching ELF relocation type. A `call` is a
/// PC-relative call, an `abs32` a direct absolute `global_addr`, and a `got_abs` a
/// GOT-indirect `global_addr`.
fn relocTypeOf(kind: isel.Kind) u32 {
    return switch (kind) {
        .call => R_386_PC32,
        .abs32 => R_386_32,
        .got_abs => R_386_GOT32,
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

fn put(buf: []u8, comptime T: type, off: usize, v: T) void {
    std.mem.writeInt(T, buf[off..][0..@sizeOf(T)], v, .little);
}

/// Compile every function in `module`, and serialize them and its data globals into one
/// ELF32 relocatable object. Each function becomes its own `.text.<name>` section with a
/// defined STT_FUNC symbol at offset 0. Each data global becomes its own `.rodata.<name>`,
/// `.data.<name>`, or `.bss.<name>` section with an STT_OBJECT symbol at offset 0. Each
/// `call` becomes an R_386_PC32 relocation, rebased to its own section. Each `global_addr`
/// becomes an R_386_32 or R_386_GOT32 relocation. The section addralign of 16 gives each
/// function its 16-byte start, so no intra-section nop padding is needed. i386 uses SHT_REL,
/// so the addend is baked into the section bytes here (-4 into a call's rel32 field, 0 for
/// an absolute field), and the shared `object_emit.emit` (with `use_rela = false`) writes
/// the ELF bytes with no reloc addend field. The caller owns the result.
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

    for (module.funcs.items) |entry| {
        var compiled = try isel.compile(allocator, entry.func);
        defer compiled.deinit(allocator);
        const code = try a.dupe(u8, compiled.code);
        // i386 SHT_REL: the addend lives in the field itself. Bake -4 into each `call`'s
        // rel32 field at its section-relative offset. `.abs32`/`.got_abs` keep the zero
        // placeholder isel already emitted, so the linker just adds S (or the GOT slot
        // vaddr) to it. The bake offset is the same section-relative `r.offset` the emitted
        // relocation reports below.
        for (compiled.relocs) |r| switch (r.kind) {
            .call => put(code, i32, r.offset, -4),
            .abs32, .got_abs => {},
        };
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

    // One section per data global. Its bytes already carry the zero placeholder for any
    // pointer-init slot (i386 SHT_REL keeps the addend in the field), so they pass through
    // verbatim.
    for (module.data.items) |d| {
        const sec_index = sections.items.len;
        const sec_name = try dataSectionName(a, dataClass(d.kind), d.name);
        switch (d.kind) {
            .rodata => try sections.append(a, .{ .name = sec_name, .sh_type = SHT_PROGBITS, .flags = SHF_ALLOC, .bytes = d.bytes, .size = d.bytes.len, .addralign = 4 }),
            .data => try sections.append(a, .{ .name = sec_name, .sh_type = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_WRITE, .bytes = d.bytes, .size = d.bytes.len, .addralign = 4 }),
            .bss => try sections.append(a, .{ .name = sec_name, .sh_type = SHT_NOBITS, .flags = SHF_ALLOC | SHF_WRITE, .size = d.size, .addralign = 4 }),
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

    // An undefined external callee. A text relocation whose target names no defined symbol
    // is an import. Add it once as an undefined `notype` global.
    for (text_relocs.items) |p| {
        if (oeIndex(symbols.items, p.name) == null) try symbols.append(a, .{ .name = p.name, .section = 0, .value = 0, .size = 0, .binding = .global, .sym_type = .notype, .defined = false });
    }

    // Resolve every text relocation to its symbol index. Its `symbol` is the index into this
    // symbol list. The emitter remaps it after its own symbol sort. The addend is baked into
    // the bytes above, so `OutReloc.addend` is unused (SHT_REL carries no addend field).
    for (text_relocs.items) |p| {
        const sym = oeIndex(symbols.items, p.name).?;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = p.r_type });
    }
    // Resolve every data pointer-init relocation. Its target is an internally defined global.
    // It is an error if the target is missing.
    for (data_pending.items) |p| {
        const sym = oeIndex(symbols.items, p.name) orelse return error.Unsupported;
        try reloc_lists.items[p.sec].append(a, .{ .offset = p.offset, .symbol = sym, .r_type = R_386_32 });
    }

    // Attach each section's relocation list.
    for (sections.items, 0..) |*sec, i| sec.relocs = reloc_lists.items[i].items;

    return object_emit.emit(allocator, sections.items, symbols.items, .{ .class = .elf32, .machine = EM_386, .use_rela = false });
}

test "emits an ELF32 i386 relocatable object" {
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
    try std.testing.expectEqual(@as(u16, EM_386), std.mem.readInt(u16, obj[18..20], .little));
    try std.testing.expectEqual(@as(u8, 1), obj[4]); // ELFCLASS32
}

test "readelf shows per-function .text sections and .rodata/.data/.bss for a global_addr load" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Function = ir.function.Function;
    const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // entry() -> *(&K), K an i32 rodata constant. This is the exact shape isel's
    // `.global_addr` arm (`mov rd, imm32`) lowers to an R_386_32 reloc for.
    var entry = Function.init(allocator);
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
    try std.testing.expect(std.mem.indexOf(u8, rels.stdout, "R_386_32") != null);

    const syms = std.process.run(allocator, io, .{ .argv = &.{ "readelf", "-s", "gd.o" }, .cwd = .{ .dir = tmp.dir } }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(syms.stdout);
    defer allocator.free(syms.stderr);
    try std.testing.expect(std.mem.indexOf(u8, syms.stdout, "OBJECT") != null);
}
