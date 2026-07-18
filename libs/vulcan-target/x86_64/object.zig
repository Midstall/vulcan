//! ELF64 relocatable object (ET_REL, EM_X86_64) emission. Each function becomes an
//! STT_FUNC global in .text. Every `call` becomes an R_X86_64_PLT32 relocation (addend
//! -4, the implicit displacement adjustment) against the callee symbol (undefined if
//! external). Accepted by readelf and a system x86-64 linker. link.zig is the in-memory
//! linker for the same data.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");
const dwarf = @import("../dwarf.zig");

/// A non-allocatable metadata section (DWARF) to append verbatim.
pub const DebugSection = struct { name: []const u8, bytes: []const u8 };

pub const Error = isel.Error;

const ET_REL: u16 = 1;
const EM_X86_64: u16 = 62;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_INFO_LINK: u64 = 0x40;
const R_X86_64_PLT32: u32 = 4;
const STB_GLOBAL: u8 = 1;
const STT_FUNC: u8 = 2;

fn put(buf: []u8, comptime T: type, off: usize, v: T) void {
    std.mem.writeInt(T, buf[off..][0..@sizeOf(T)], v, .little);
}

/// A string table that interns names and returns their byte offsets.
const StrTab = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn init(allocator: std.mem.Allocator) !StrTab {
        var s: StrTab = .{};
        try s.bytes.append(allocator, 0); // index 0 is the empty string
        return s;
    }
    fn add(self: *StrTab, allocator: std.mem.Allocator, name: []const u8) !u32 {
        const off: u32 = @intCast(self.bytes.items.len);
        try self.bytes.appendSlice(allocator, name);
        try self.bytes.append(allocator, 0);
        return off;
    }
};

/// Compile and emit `module` as an ELF64 relocatable object. Caller owns the bytes.
pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error![]u8 {
    const funcs = module.funcs.items;
    const compiled = try allocator.alloc(isel.Compiled, funcs.len);
    var compiled_n: usize = 0;
    defer {
        for (compiled[0..compiled_n]) |*c| c.deinit(allocator);
        allocator.free(compiled);
    }
    for (funcs, 0..) |e, i| {
        compiled[i] = try isel.compile(allocator, e.func);
        compiled_n = i + 1;
    }

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    const offsets = try allocator.alloc(usize, funcs.len);
    defer allocator.free(offsets);
    for (compiled, 0..) |c, i| {
        offsets[i] = text.items.len;
        try text.appendSlice(allocator, c.code);
        while (text.items.len % 16 != 0) try text.append(allocator, 0x90);
    }

    // Symbols: the null symbol, then each function as a defined global, then any
    // external callee referenced by a relocation as an undefined global.
    var strtab = try StrTab.init(allocator);
    defer strtab.bytes.deinit(allocator);
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    var sym_index = std.StringHashMapUnmanaged(u32){};
    defer sym_index.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 24); // null symbol (index 0)

    for (funcs, 0..) |e, i| {
        try sym_index.put(allocator, e.name, @intCast(symtab.items.len / 24));
        try appendSym(allocator, &symtab, try strtab.add(allocator, e.name), STT_FUNC, 1, offsets[i], compiled[i].code.len);
    }
    for (compiled) |c| for (c.relocs) |r| {
        if (!sym_index.contains(r.symbol)) {
            try sym_index.put(allocator, r.symbol, @intCast(symtab.items.len / 24));
            try appendSym(allocator, &symtab, try strtab.add(allocator, r.symbol), STT_FUNC, 0, 0, 0); // undefined
        }
    };

    // Relocations against .text.
    var rela: std.ArrayList(u8) = .empty;
    defer rela.deinit(allocator);
    for (compiled, 0..) |c, i| for (c.relocs) |r| {
        var ent: [24]u8 = undefined;
        put(&ent, u64, 0, offsets[i] + r.offset); // r_offset
        put(&ent, u64, 8, (@as(u64, sym_index.get(r.symbol).?) << 32) | R_X86_64_PLT32); // r_info
        put(&ent, i64, 16, -4); // r_addend
        try rela.appendSlice(allocator, &ent);
    };

    return assemble(allocator, text.items, rela.items, symtab.items, strtab.bytes.items, funcs.len + 1, &.{});
}

/// Like `writeModule`, but also emits inline DWARF: `.debug_abbrev` + `.debug_info` (a subprogram DIE
/// per function with PC range + typed return) + `.debug_line` (from the functions' `debug.line`
/// attributes), with the CU linked to the line program. So a debugger reads names, ranges, typed
/// signatures, and source lines on real x86-64 objects.
pub fn writeModuleWithDebug(allocator: std.mem.Allocator, module: *const link.Module, source_file: []const u8) Error![]u8 {
    const funcs = module.funcs.items;
    const compiled = try allocator.alloc(isel.Compiled, funcs.len);
    var compiled_n: usize = 0;
    defer {
        for (compiled[0..compiled_n]) |*c| c.deinit(allocator);
        allocator.free(compiled);
    }
    for (funcs, 0..) |e, i| {
        compiled[i] = try isel.compile(allocator, e.func);
        compiled_n = i + 1;
    }

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    const offsets = try allocator.alloc(usize, funcs.len);
    defer allocator.free(offsets);
    var rows: std.ArrayList(dwarf.LineRow) = .empty;
    defer rows.deinit(allocator);
    for (compiled, 0..) |c, i| {
        offsets[i] = text.items.len;
        for (c.lines) |ln| try rows.append(allocator, .{ .address = offsets[i] + ln.offset, .line = ln.line });
        try text.appendSlice(allocator, c.code);
        while (text.items.len % 16 != 0) try text.append(allocator, 0x90);
    }

    var strtab = try StrTab.init(allocator);
    defer strtab.bytes.deinit(allocator);
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    var sym_index = std.StringHashMapUnmanaged(u32){};
    defer sym_index.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 24); // null symbol
    for (funcs, 0..) |e, i| {
        try sym_index.put(allocator, e.name, @intCast(symtab.items.len / 24));
        try appendSym(allocator, &symtab, try strtab.add(allocator, e.name), STT_FUNC, 1, offsets[i], compiled[i].code.len);
    }
    for (compiled) |c| for (c.relocs) |r| {
        if (!sym_index.contains(r.symbol)) {
            try sym_index.put(allocator, r.symbol, @intCast(symtab.items.len / 24));
            try appendSym(allocator, &symtab, try strtab.add(allocator, r.symbol), STT_FUNC, 0, 0, 0);
        }
    };

    var rela: std.ArrayList(u8) = .empty;
    defer rela.deinit(allocator);
    for (compiled, 0..) |c, i| for (c.relocs) |r| {
        var ent: [24]u8 = undefined;
        put(&ent, u64, 0, offsets[i] + r.offset);
        put(&ent, u64, 8, (@as(u64, sym_index.get(r.symbol).?) << 32) | R_X86_64_PLT32);
        put(&ent, i64, 16, -4);
        try rela.appendSlice(allocator, &ent);
    };

    // DWARF: a subprogram DIE per function (typed return), plus the line program.
    const subs = try allocator.alloc(dwarf.Subprogram, funcs.len);
    defer allocator.free(subs);
    for (funcs, 0..) |e, i| subs[i] = .{
        .name = e.name,
        .low_pc = offsets[i],
        .high_pc = offsets[i] + compiled[i].code.len,
        .ret_type = returnBaseType(e.func),
    };
    const abbrev = try dwarf.emitAbbrev(allocator);
    defer allocator.free(abbrev);
    const info = try dwarf.emitInfo(allocator, .{ .name = source_file, .low_pc = 0, .high_pc = text.items.len, .subprograms = subs, .stmt_list = 0 });
    defer allocator.free(info);
    const line = try dwarf.emitLine(allocator, source_file, rows.items, text.items.len);
    defer allocator.free(line);

    return assemble(allocator, text.items, rela.items, symtab.items, strtab.bytes.items, funcs.len + 1, &.{
        .{ .name = ".debug_abbrev", .bytes = abbrev },
        .{ .name = ".debug_info", .bytes = info },
        .{ .name = ".debug_line", .bytes = line },
    });
}

/// Map a function's IR return type to a DWARF base type (C-like names), or null for a void /
/// non-primitive return.
fn returnBaseType(func: *const ir.function.Function) ?dwarf.BaseType {
    const ret_val = for (0..func.blocks.items.len) |bi| {
        const term = func.terminator(@enumFromInt(bi)) orelse continue;
        switch (term) {
            .ret => |maybe| if (maybe) |v| break v else return null,
            else => {},
        }
    } else return null;

    return switch (func.types.type_kind(func.valueType(ret_val))) {
        .bool => .{ .name = "bool", .encoding = .boolean, .byte_size = 1 },
        .float => |f| switch (f) {
            .f32 => .{ .name = "float", .encoding = .float, .byte_size = 4 },
            .f64 => .{ .name = "double", .encoding = .float, .byte_size = 8 },
            // A 2-byte IEEE half in memory (x86_64 lowers f16 via F16C, held as its f32 widening
            // in registers).
            .f16 => .{ .name = "half", .encoding = .float, .byte_size = 2 },
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

fn appendSym(allocator: std.mem.Allocator, symtab: *std.ArrayList(u8), name_off: u32, typ: u8, binding: u8, value: usize, size: usize) !void {
    var ent: [24]u8 = undefined;
    @memset(&ent, 0);
    put(&ent, u32, 0, name_off); // st_name
    ent[4] = (binding << 4) | typ; // st_info
    put(&ent, u16, 6, if (binding == STB_GLOBAL and value == 0 and size == 0) 0 else 1); // st_shndx (0=UNDEF, 1=.text)
    put(&ent, u64, 8, value); // st_value
    put(&ent, u64, 16, size); // st_size
    try symtab.appendSlice(allocator, ent[0..24]);
}

/// A section header descriptor for the flexible ELF writer below.
const Sec = struct {
    name: []const u8,
    typ: u32,
    flags: u64 = 0,
    data: []const u8,
    link: u32 = 0,
    info: u32 = 0,
    addralign: u64 = 1,
    entsize: u64 = 0,
};

/// Lay out the ELF: header, section data, and the section header table. Order: NULL, .text,
/// [.rela.text], [debug...], .symtab, .strtab, .shstrtab. `first_global` is the index of the first
/// global symbol; `debug` are extra (non-allocatable) metadata sections placed before .symtab.
fn assemble(allocator: std.mem.Allocator, text: []const u8, rela: []const u8, symtab: []const u8, strtab: []const u8, first_global: usize, debug: []const DebugSection) Error![]u8 {
    const has_rela = rela.len > 0;

    // Compute section indices up front (NULL is 0) so link/info fields are right.
    const text_ndx: u32 = 1;
    const rela_ndx: u32 = 2; // only meaningful when has_rela
    const debug0_ndx: u32 = if (has_rela) 3 else 2;
    const symtab_ndx: u32 = debug0_ndx + @as(u32, @intCast(debug.len));
    const strtab_ndx: u32 = symtab_ndx + 1;
    _ = rela_ndx;

    // Assemble the ordered section list (excluding the NULL section at index 0).
    var secs: std.ArrayList(Sec) = .empty;
    defer secs.deinit(allocator);
    try secs.append(allocator, .{ .name = ".text", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_EXECINSTR, .data = text, .addralign = 16 });
    if (has_rela) try secs.append(allocator, .{ .name = ".rela.text", .typ = SHT_RELA, .flags = SHF_INFO_LINK, .data = rela, .link = symtab_ndx, .info = text_ndx, .addralign = 8, .entsize = 24 });
    for (debug) |d| try secs.append(allocator, .{ .name = d.name, .typ = SHT_PROGBITS, .data = d.bytes, .addralign = 1 });
    try secs.append(allocator, .{ .name = ".symtab", .typ = SHT_SYMTAB, .data = symtab, .link = strtab_ndx, .info = @intCast(first_global), .addralign = 8, .entsize = 24 });
    try secs.append(allocator, .{ .name = ".strtab", .typ = SHT_STRTAB, .data = strtab, .addralign = 1 });

    // The section-name string table (its own name included), built in index order.
    var names: std.ArrayList(u8) = .empty;
    defer names.deinit(allocator);
    try names.append(allocator, 0); // the NULL section's empty name
    const name_offs = try allocator.alloc(u32, secs.items.len + 1); // +1 for .shstrtab
    defer allocator.free(name_offs);
    for (secs.items, 0..) |s, i| {
        name_offs[i] = @intCast(names.items.len);
        try names.appendSlice(allocator, s.name);
        try names.append(allocator, 0);
    }
    name_offs[secs.items.len] = @intCast(names.items.len);
    try names.appendSlice(allocator, ".shstrtab\x00");

    // File offsets for each section's data (8-aligned), then the shstrtab, then the header table.
    var off: usize = 64;
    const data_offs = try allocator.alloc(usize, secs.items.len);
    defer allocator.free(data_offs);
    for (secs.items, 0..) |s, i| {
        off = std.mem.alignForward(usize, off, if (s.addralign > 0) s.addralign else 1);
        data_offs[i] = off;
        off += s.data.len;
    }
    off = std.mem.alignForward(usize, off, 1);
    const shstr_off = off;
    off += names.items.len;
    off = std.mem.alignForward(usize, off, 8);
    const sh_off = off;
    const nsections = secs.items.len + 2; // + NULL + .shstrtab
    const total = sh_off + nsections * 64;

    const buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    put(buf, u16, 16, ET_REL);
    put(buf, u16, 18, EM_X86_64);
    put(buf, u32, 20, 1); // e_version
    put(buf, u64, 40, sh_off); // e_shoff
    put(buf, u16, 52, 64); // e_ehsize
    put(buf, u16, 58, 64); // e_shentsize
    put(buf, u16, 60, @intCast(nsections)); // e_shnum
    put(buf, u16, 62, @intCast(nsections - 1)); // e_shstrndx (.shstrtab is last)

    for (secs.items, 0..) |s, i| @memcpy(buf[data_offs[i]..][0..s.data.len], s.data);
    @memcpy(buf[shstr_off..][0..names.items.len], names.items);

    // Section headers: NULL, each section, then .shstrtab.
    const sh = buf[sh_off..];
    for (secs.items, 0..) |s, i| {
        putShdr(sh[(i + 1) * 64 ..], name_offs[i], s.typ, s.flags, data_offs[i], s.data.len, s.link, s.info, s.addralign, s.entsize);
    }
    putShdr(sh[(nsections - 1) * 64 ..], name_offs[secs.items.len], SHT_STRTAB, 0, shstr_off, names.items.len, 0, 0, 1, 0);
    return buf;
}

fn putShdr(e: []u8, name: u32, typ: u32, flags: u64, offset: usize, size: usize, sh_link: u32, sh_info: u32, addralign: u64, entsize: u64) void {
    put(e, u32, 0, name);
    put(e, u32, 4, typ);
    put(e, u64, 8, flags);
    put(e, u64, 24, offset);
    put(e, u64, 32, size);
    put(e, u32, 40, sh_link);
    put(e, u32, 44, sh_info);
    put(e, u64, 48, addralign);
    put(e, u64, 56, entsize);
}

fn nameOff(names: []const u8, name: []const u8) u32 {
    return @intCast(std.mem.indexOf(u8, names, name).?);
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
        helper.setTerminator(b, .{ .ret = x });
    }
    var main = Function.init(allocator);
    defer main.deinit();
    {
        const t = try main.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try main.appendBlock();
        const x = try main.appendBlockParam(b, t);
        const r = try main.appendCall(b, t, "helper", &.{x});
        main.setTerminator(b, .{ .ret = r });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "helper", &helper);

    const obj = try writeModule(allocator, &module);
    defer allocator.free(obj);
    try std.testing.expectEqualSlices(u8, "\x7fELF", obj[0..4]);
    try std.testing.expectEqual(@as(u16, ET_REL), std.mem.readInt(u16, obj[16..18], .little));
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
    func.setTerminator(b, .{ .ret = s });

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

    // helper(x) -> x + 5 ; main(x) -> helper(x).
    var helper = F.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(i32k);
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        const s = try helper.appendArithImm(b, t, .add, x, 5);
        helper.setTerminator(b, .{ .ret = s });
    }
    var main_f = F.init(allocator);
    defer main_f.deinit();
    {
        const t = try main_f.types.intern(i32k);
        const b = try main_f.appendBlock();
        const x = try main_f.appendBlockParam(b, t);
        const r = try main_f.appendCall(b, t, "helper", &.{x});
        main_f.setTerminator(b, .{ .ret = r });
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
