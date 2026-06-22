//! ELF64 relocatable object (ET_REL, EM_X86_64) emission. Each function becomes an
//! STT_FUNC global in .text. Every `call` becomes an R_X86_64_PLT32 relocation (addend
//! -4, the implicit displacement adjustment) against the callee symbol (undefined if
//! external). Accepted by readelf and a system x86-64 linker. link.zig is the in-memory
//! linker for the same data.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");

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

    return assemble(allocator, text.items, rela.items, symtab.items, strtab.bytes.items, funcs.len + 1);
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

/// Lay out the ELF: header, the four sections' data, and the section header table.
/// Sections (indices): 0 NULL, 1 .text, 2 .rela.text, 3 .symtab, 4 .strtab,
/// 5 .shstrtab. `first_global` is the index of the first global symbol.
fn assemble(allocator: std.mem.Allocator, text: []const u8, rela: []const u8, symtab: []const u8, strtab: []const u8, first_global: usize) Error![]u8 {
    const has_rela = rela.len > 0;

    // Section-name string table.
    const names = if (has_rela)
        "\x00.text\x00.rela.text\x00.symtab\x00.strtab\x00.shstrtab\x00"
    else
        "\x00.text\x00.symtab\x00.strtab\x00.shstrtab\x00";
    const nsections: usize = if (has_rela) 6 else 5;

    var off: usize = 64; // after the ELF header
    const text_off = off;
    off += text.len;
    off = std.mem.alignForward(usize, off, 8);
    const rela_off = off;
    if (has_rela) off += rela.len;
    off = std.mem.alignForward(usize, off, 8);
    const sym_off = off;
    off += symtab.len;
    const str_off = off;
    off += strtab.len;
    const shstr_off = off;
    off += names.len;
    off = std.mem.alignForward(usize, off, 8);
    const sh_off = off;
    const total = sh_off + nsections * 64;

    const buf = try allocator.alloc(u8, total);
    @memset(buf, 0);

    // ELF header.
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

    @memcpy(buf[text_off..][0..text.len], text);
    if (has_rela) @memcpy(buf[rela_off..][0..rela.len], rela);
    @memcpy(buf[sym_off..][0..symtab.len], symtab);
    @memcpy(buf[str_off..][0..strtab.len], strtab);
    @memcpy(buf[shstr_off..][0..names.len], names);

    // Section headers. Name offsets index into `names`.
    const sym_ndx: u32 = if (has_rela) 3 else 2;
    const str_ndx: u32 = if (has_rela) 4 else 3;
    const shstr_ndx: u32 = if (has_rela) 5 else 4;
    var sh = buf[sh_off..];
    // 0: NULL (already zero).
    putShdr(sh[64..], nameOff(names, ".text"), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    var idx: usize = 2;
    if (has_rela) {
        putShdr(sh[idx * 64 ..], nameOff(names, ".rela.text"), SHT_RELA, SHF_INFO_LINK, rela_off, rela.len, sym_ndx, 1, 8, 24);
        idx += 1;
    }
    putShdr(sh[idx * 64 ..], nameOff(names, ".symtab"), SHT_SYMTAB, 0, sym_off, symtab.len, str_ndx, @intCast(first_global), 8, 24);
    putShdr(sh[(idx + 1) * 64 ..], nameOff(names, ".strtab"), SHT_STRTAB, 0, str_off, strtab.len, 0, 0, 1, 0);
    putShdr(sh[(idx + 2) * 64 ..], nameOff(names, ".shstrtab"), SHT_STRTAB, 0, shstr_off, names.len, 0, 0, 1, 0);
    _ = shstr_ndx;
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
