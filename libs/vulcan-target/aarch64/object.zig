//! ELF64 relocatable object (`ET_REL`, `EM_AARCH64`) emission. Each function becomes
//! an `STT_FUNC` global in `.text`. Every call (`bl`) becomes an `R_AARCH64_CALL26`
//! relocation against the callee symbol (undefined if external). Data sections are
//! not emitted (the selector produces no global data). The output is accepted by
//! `readelf` and a system AArch64 linker. `ld.zig` is Vulcan's own linker for it.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");

const Function = ir.function.Function;

pub const Error = isel.Error;

/// A symbol's binding. Locals must precede globals in the symbol table.
pub const Binding = enum { local, global };

/// A symbol's type: `func` an entry, `object` a data object, `notype` unknown.
pub const SymKind = enum { notype, func, object };

/// One symbol-table entry. A defined symbol lives in `.text` at `value`. An
/// undefined one (`defined = false`) is an external the linker must resolve.
pub const Symbol = struct {
    name: []const u8,
    value: u64 = 0,
    size: u64 = 0,
    binding: Binding = .global,
    kind: SymKind = .func,
    defined: bool = true,
};

/// The emitted AArch64 relocation types (architectural `R_AARCH64_*` codes).
pub const RelocType = enum(u32) {
    /// `R_AARCH64_CALL26`: patch a `bl`/`b`'s 26-bit immediate (a +/-128MiB call).
    call26 = 283,
};

/// A relocation applied to a `.text` byte offset against a symbol.
pub const Reloc = struct {
    offset: u64,
    symbol: u32,
    type: RelocType,
    addend: i64 = 0,
};

/// A relocatable object: code plus the symbol table and the relocations on it.
pub const Object = struct {
    text: []const u8,
    symbols: []const Symbol,
    relocs: []const Reloc,
};

const ET_REL: u16 = 1;
const EM_AARCH64: u16 = 183;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const SHF_INFO_LINK: u64 = 0x40;
const STB_LOCAL: u8 = 0;
const STB_GLOBAL: u8 = 1;
const STT_NOTYPE: u8 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const SHN_UNDEF: u16 = 0;

const ehsize: u64 = 64;
const shentsize: u64 = 64;
const symentsize: u64 = 24;
const relaentsize: u64 = 24;

fn alignUp(v: u64, a: u64) u64 {
    return (v + a - 1) & ~(a - 1);
}

fn putInt(buf: []u8, comptime T: type, value: T) void {
    std.mem.writeInt(T, buf[0..@sizeOf(T)], value, .little);
}

/// A growable string table: a leading NUL, then NUL-terminated names. Returns each
/// appended name's byte offset.
const StrTab = struct {
    bytes: std.ArrayList(u8) = .empty,

    fn init(allocator: std.mem.Allocator) Error!StrTab {
        var t: StrTab = .{};
        try t.bytes.append(allocator, 0);
        return t;
    }
    fn deinit(self: *StrTab, allocator: std.mem.Allocator) void {
        self.bytes.deinit(allocator);
    }
    fn add(self: *StrTab, allocator: std.mem.Allocator, name: []const u8) Error!u32 {
        const off: u32 = @intCast(self.bytes.items.len);
        try self.bytes.appendSlice(allocator, name);
        try self.bytes.append(allocator, 0);
        return off;
    }
};

/// One emitted section header.
const Shdr = struct {
    name: []const u8,
    typ: u32,
    flags: u64,
    addralign: u64,
    entsize: u64 = 0,
    link: u16 = 0,
    info: u32 = 0,
    bytes: ?[]const u8,
    size: u64,
};

/// Serialize `obj` into an ELF64 AArch64 relocatable object. Sections: `.text`,
/// `.rela.text` (if any relocs), `.symtab`, `.strtab`, `.shstrtab`. Caller owns it.
pub fn write(allocator: std.mem.Allocator, obj: Object) Error![]u8 {
    var local_count: u32 = 0;
    var seen_global = false;
    for (obj.symbols) |s| switch (s.binding) {
        .local => {
            if (seen_global) return error.Unsupported; // local after global
            local_count += 1;
        },
        .global => seen_global = true,
    };
    const first_global: u32 = 1 + local_count; // null symbol at 0 is local

    const has_rela = obj.relocs.len > 0;
    var next: u16 = 1;
    const text_ndx = next;
    next += 1;
    if (has_rela) next += 1; // .rela.text
    const symtab_ndx = next;
    next += 1;
    const strtab_ndx = next;
    next += 1;
    const shstrtab_ndx = next;
    next += 1;
    const section_count = next;

    // Symbol-name string table.
    var strtab = try StrTab.init(allocator);
    defer strtab.deinit(allocator);
    var name_offsets = try allocator.alloc(u32, obj.symbols.len);
    defer allocator.free(name_offsets);
    for (obj.symbols, 0..) |s, i| name_offsets[i] = try strtab.add(allocator, s.name);

    // Symbol table.
    const sym_bytes = (1 + obj.symbols.len) * symentsize;
    var symtab = try allocator.alloc(u8, sym_bytes);
    defer allocator.free(symtab);
    @memset(symtab, 0);
    for (obj.symbols, 0..) |s, i| {
        const e = symtab[(i + 1) * symentsize ..][0..symentsize];
        const binding: u8 = switch (s.binding) {
            .local => STB_LOCAL,
            .global => STB_GLOBAL,
        };
        const typ: u8 = switch (s.kind) {
            .notype => STT_NOTYPE,
            .func => STT_FUNC,
            .object => STT_OBJECT,
        };
        putInt(e[0..4], u32, name_offsets[i]); // st_name
        e[4] = (binding << 4) | typ; // st_info
        e[5] = 0; // st_other
        putInt(e[6..8], u16, if (s.defined) text_ndx else SHN_UNDEF); // st_shndx
        putInt(e[8..16], u64, s.value); // st_value
        putInt(e[16..24], u64, s.size); // st_size
    }

    // Relocation table (applies to `.text`).
    const rela_bytes = obj.relocs.len * relaentsize;
    var rela = try allocator.alloc(u8, rela_bytes);
    defer allocator.free(rela);
    for (obj.relocs, 0..) |r, i| {
        const e = rela[i * relaentsize ..][0..relaentsize];
        const sym_index: u64 = @as(u64, r.symbol) + 1; // null entry at 0
        putInt(e[0..8], u64, r.offset); // r_offset
        putInt(e[8..16], u64, (sym_index << 32) | @intFromEnum(r.type)); // r_info
        putInt(e[16..24], i64, r.addend); // r_addend
    }

    var shstrtab = try StrTab.init(allocator);
    defer shstrtab.deinit(allocator);

    var headers: std.ArrayList(Shdr) = .empty;
    defer headers.deinit(allocator);
    try headers.append(allocator, .{ .name = ".text", .typ = SHT_PROGBITS, .flags = SHF_ALLOC | SHF_EXECINSTR, .addralign = 4, .bytes = obj.text, .size = obj.text.len });
    if (has_rela) try headers.append(allocator, .{ .name = ".rela.text", .typ = SHT_RELA, .flags = SHF_INFO_LINK, .addralign = 8, .entsize = relaentsize, .link = symtab_ndx, .info = text_ndx, .bytes = rela, .size = rela_bytes });
    try headers.append(allocator, .{ .name = ".symtab", .typ = SHT_SYMTAB, .flags = 0, .addralign = 8, .entsize = symentsize, .link = strtab_ndx, .info = first_global, .bytes = symtab, .size = sym_bytes });
    try headers.append(allocator, .{ .name = ".strtab", .typ = SHT_STRTAB, .flags = 0, .addralign = 1, .bytes = strtab.bytes.items, .size = strtab.bytes.items.len });
    try headers.append(allocator, .{ .name = ".shstrtab", .typ = SHT_STRTAB, .flags = 0, .addralign = 1, .bytes = null, .size = 0 });

    var offsets = try allocator.alloc(u64, headers.items.len);
    defer allocator.free(offsets);
    var off: u64 = ehsize;
    for (headers.items, 0..) |h, i| {
        off = alignUp(off, 8);
        offsets[i] = off;
        if (h.bytes != null) off += h.size;
    }
    var name_in_shstr = try allocator.alloc(u32, headers.items.len);
    defer allocator.free(name_in_shstr);
    for (headers.items, 0..) |h, i| name_in_shstr[i] = try shstrtab.add(allocator, h.name);
    const shstr_idx = headers.items.len - 1;
    offsets[shstr_idx] = alignUp(off, 8);
    off = offsets[shstr_idx] + shstrtab.bytes.items.len;

    off = alignUp(off, 8);
    const shoff = off;
    const total = shoff + @as(u64, section_count) * shentsize;

    var buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);
    @memset(buf, 0);

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2; // ELFCLASS64
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1; // EV_CURRENT
    putInt(buf[16..18], u16, ET_REL);
    putInt(buf[18..20], u16, EM_AARCH64);
    putInt(buf[20..24], u32, 1); // e_version
    putInt(buf[40..48], u64, shoff); // e_shoff
    putInt(buf[52..54], u16, @intCast(ehsize)); // e_ehsize
    putInt(buf[58..60], u16, @intCast(shentsize)); // e_shentsize
    putInt(buf[60..62], u16, section_count); // e_shnum
    putInt(buf[62..64], u16, shstrtab_ndx); // e_shstrndx

    putShdr(buf, shoff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (headers.items, 0..) |h, i| {
        const ndx: u16 = @intCast(i + 1);
        const content = if (i == shstr_idx) shstrtab.bytes.items else (h.bytes orelse &.{});
        const size = if (i == shstr_idx) shstrtab.bytes.items.len else h.size;
        if (content.len > 0) @memcpy(buf[offsets[i]..][0..content.len], content);
        putShdr(buf, shoff, ndx, name_in_shstr[i], h.typ, h.flags, offsets[i], size, h.link, h.info, h.addralign, h.entsize);
    }

    return buf;
}

fn putShdr(buf: []u8, shoff: u64, idx: u16, name: u32, typ: u32, flags: u64, offset: u64, size: u64, sh_link: u32, sh_info: u32, addralign: u64, entsize: u64) void {
    const e = buf[shoff + idx * shentsize ..][0..shentsize];
    putInt(e[0..4], u32, name);
    putInt(e[4..8], u32, typ);
    putInt(e[8..16], u64, flags);
    putInt(e[16..24], u64, 0); // sh_addr
    putInt(e[24..32], u64, offset);
    putInt(e[32..40], u64, size);
    putInt(e[40..44], u32, sh_link);
    putInt(e[44..48], u32, sh_info);
    putInt(e[48..56], u64, addralign);
    putInt(e[56..64], u64, entsize);
}

const Pending = struct { offset: u64, name: []const u8 };

/// Compile every function in `module` and serialize them into one ELF relocatable
/// object: each function a defined `STT_FUNC` global in `.text`, each `bl` an
/// `R_AARCH64_CALL26` relocation (its callee an undefined global if external). The
/// caller owns the returned ELF bytes.
pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error![]u8 {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    var globals: std.ArrayList(Symbol) = .empty;
    defer globals.deinit(allocator);
    var pending: std.ArrayList(Pending) = .empty;
    defer pending.deinit(allocator);

    for (module.functions.items) |entry| {
        const start: u64 = text.items.len;
        var compiled = try isel.compileFunction(allocator, entry.func);
        defer compiled.deinit(allocator);
        for (compiled.code) |word| {
            var w: [4]u8 = undefined;
            putInt(&w, u32, word);
            try text.appendSlice(allocator, &w);
        }
        try globals.append(allocator, .{ .name = entry.name, .value = start, .size = @as(u64, compiled.code.len) * 4, .kind = .func, .defined = true });
        for (compiled.relocs) |r| {
            try pending.append(allocator, .{ .offset = start + @as(u64, r.offset) * 4, .name = r.symbol });
        }
    }

    // Symbol table: defined globals, then any undefined external callees.
    var symbols: std.ArrayList(Symbol) = .empty;
    defer symbols.deinit(allocator);
    try symbols.appendSlice(allocator, globals.items);
    for (pending.items) |p| {
        if (symbolIndex(symbols.items, p.name) == null) {
            try symbols.append(allocator, .{ .name = p.name, .size = 0, .kind = .notype, .defined = false });
        }
    }

    var relocs = try allocator.alloc(Reloc, pending.items.len);
    defer allocator.free(relocs);
    for (pending.items, 0..) |p, i| {
        relocs[i] = .{ .offset = p.offset, .symbol = symbolIndex(symbols.items, p.name).?, .type = .call26 };
    }

    return write(allocator, .{ .text = text.items, .symbols = symbols.items, .relocs = relocs });
}

fn symbolIndex(symbols: []const Symbol, name: []const u8) ?u32 {
    for (symbols, 0..) |s, i| if (std.mem.eql(u8, s.name, name)) return @intCast(i);
    return null;
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

    // dbl(a) = a + a, caller(x) = dbl(x) + 1 (yields a CALL26 relocation).
    var dbl = Function.init(allocator);
    defer dbl.deinit();
    {
        const t = try dbl.types.intern(i32k);
        const b = try dbl.appendBlock();
        const a = try dbl.appendBlockParam(b, t);
        const r = try dbl.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
        dbl.setTerminator(b, .{ .ret = r });
    }
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32k);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const d = try caller.appendCall(b, t, "dbl", &.{x});
        const r = try caller.appendArithImm(b, t, .add, d, 1);
        caller.setTerminator(b, .{ .ret = r });
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
