//! ELF32 relocatable object (ET_REL, EM_386) emission. Each function becomes an STT_FUNC
//! global in .text. Every `call` becomes an R_386_PC32 relocation against the callee. i386
//! uses SHT_REL, so the addend -4 is stored implicitly in the call's displacement field.
//! link.zig is the in-memory linker.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const link = @import("link.zig");

pub const Error = isel.Error;

const ET_REL: u16 = 1;
const EM_386: u16 = 3;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_REL: u32 = 9;
const SHF_ALLOC: u32 = 0x2;
const SHF_EXECINSTR: u32 = 0x4;
const SHF_INFO_LINK: u32 = 0x40;
const R_386_PC32: u32 = 2;

fn put(buf: []u8, comptime T: type, off: usize, v: T) void {
    std.mem.writeInt(T, buf[off..][0..@sizeOf(T)], v, .little);
}

const StrTab = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn init(allocator: std.mem.Allocator) !StrTab {
        var s: StrTab = .{};
        try s.bytes.append(allocator, 0);
        return s;
    }
    fn add(self: *StrTab, allocator: std.mem.Allocator, name: []const u8) !u32 {
        const off: u32 = @intCast(self.bytes.items.len);
        try self.bytes.appendSlice(allocator, name);
        try self.bytes.append(allocator, 0);
        return off;
    }
};

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
    // i386 REL: store the addend (-4 for a call's PC-relative displacement) in the field.
    for (compiled, 0..) |c, i| for (c.relocs) |r| put(text.items, i32, offsets[i] + r.offset, -4);

    var strtab = try StrTab.init(allocator);
    defer strtab.bytes.deinit(allocator);
    var symtab: std.ArrayList(u8) = .empty;
    defer symtab.deinit(allocator);
    var sym_index = std.StringHashMapUnmanaged(u32){};
    defer sym_index.deinit(allocator);
    try symtab.appendNTimes(allocator, 0, 16); // null symbol

    for (funcs, 0..) |e, i| {
        try sym_index.put(allocator, e.name, @intCast(symtab.items.len / 16));
        try appendSym(allocator, &symtab, try strtab.add(allocator, e.name), 2, 1, offsets[i], compiled[i].code.len);
    }
    for (compiled) |c| for (c.relocs) |r| {
        if (!sym_index.contains(r.symbol)) {
            try sym_index.put(allocator, r.symbol, @intCast(symtab.items.len / 16));
            try appendSym(allocator, &symtab, try strtab.add(allocator, r.symbol), 2, 0, 0, 0);
        }
    };

    var rel: std.ArrayList(u8) = .empty;
    defer rel.deinit(allocator);
    for (compiled, 0..) |c, i| for (c.relocs) |r| {
        var ent: [8]u8 = undefined;
        put(&ent, u32, 0, @intCast(offsets[i] + r.offset)); // r_offset
        put(&ent, u32, 4, (sym_index.get(r.symbol).? << 8) | R_386_PC32); // r_info
        try rel.appendSlice(allocator, &ent);
    };

    return assemble(allocator, text.items, rel.items, symtab.items, strtab.bytes.items, funcs.len + 1);
}

fn appendSym(allocator: std.mem.Allocator, symtab: *std.ArrayList(u8), name_off: u32, typ: u8, binding: u8, value: usize, size: usize) !void {
    var ent: [16]u8 = undefined;
    @memset(&ent, 0);
    put(&ent, u32, 0, name_off); // st_name
    put(&ent, u32, 4, @intCast(value)); // st_value
    put(&ent, u32, 8, @intCast(size)); // st_size
    ent[12] = (binding << 4) | typ; // st_info
    put(&ent, u16, 14, if (binding == 1 and value == 0 and size == 0) 0 else 1); // st_shndx (0=UNDEF,1=.text)
    try symtab.appendSlice(allocator, ent[0..16]);
}

fn assemble(allocator: std.mem.Allocator, text: []const u8, rel: []const u8, symtab: []const u8, strtab: []const u8, first_global: usize) Error![]u8 {
    const has_rel = rel.len > 0;
    const names = if (has_rel)
        "\x00.text\x00.rel.text\x00.symtab\x00.strtab\x00.shstrtab\x00"
    else
        "\x00.text\x00.symtab\x00.strtab\x00.shstrtab\x00";
    const nsections: usize = if (has_rel) 6 else 5;

    var off: usize = 52;
    const text_off = off;
    off += text.len;
    const rel_off = off;
    if (has_rel) off += rel.len;
    const sym_off = off;
    off += symtab.len;
    const str_off = off;
    off += strtab.len;
    const shstr_off = off;
    off += names.len;
    off = std.mem.alignForward(usize, off, 4);
    const sh_off = off;
    const total = sh_off + nsections * 40;

    const buf = try allocator.alloc(u8, total);
    @memset(buf, 0);
    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 1; // ELFCLASS32
    buf[5] = 1; // ELFDATA2LSB
    buf[6] = 1;
    put(buf, u16, 16, ET_REL);
    put(buf, u16, 18, EM_386);
    put(buf, u32, 20, 1); // e_version
    put(buf, u32, 32, @intCast(sh_off)); // e_shoff
    put(buf, u16, 40, 52); // e_ehsize
    put(buf, u16, 46, 40); // e_shentsize
    put(buf, u16, 48, @intCast(nsections)); // e_shnum
    put(buf, u16, 50, @intCast(nsections - 1)); // e_shstrndx

    @memcpy(buf[text_off..][0..text.len], text);
    if (has_rel) @memcpy(buf[rel_off..][0..rel.len], rel);
    @memcpy(buf[sym_off..][0..symtab.len], symtab);
    @memcpy(buf[str_off..][0..strtab.len], strtab);
    @memcpy(buf[shstr_off..][0..names.len], names);

    const sym_ndx: u32 = if (has_rel) 3 else 2;
    const str_ndx: u32 = if (has_rel) 4 else 3;
    var sh = buf[sh_off..];
    putShdr(sh[40..], nameOff(names, ".text"), SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_off, text.len, 0, 0, 16, 0);
    var idx: usize = 2;
    if (has_rel) {
        putShdr(sh[idx * 40 ..], nameOff(names, ".rel.text"), SHT_REL, SHF_INFO_LINK, rel_off, rel.len, sym_ndx, 1, 4, 8);
        idx += 1;
    }
    putShdr(sh[idx * 40 ..], nameOff(names, ".symtab"), SHT_SYMTAB, 0, sym_off, symtab.len, str_ndx, @intCast(first_global), 4, 16);
    putShdr(sh[(idx + 1) * 40 ..], nameOff(names, ".strtab"), SHT_STRTAB, 0, str_off, strtab.len, 0, 0, 1, 0);
    putShdr(sh[(idx + 2) * 40 ..], nameOff(names, ".shstrtab"), SHT_STRTAB, 0, shstr_off, names.len, 0, 0, 1, 0);
    return buf;
}

fn putShdr(e: []u8, name: u32, typ: u32, flags: u32, offset: usize, size: usize, sh_link: u32, sh_info: u32, addralign: u32, entsize: u32) void {
    put(e, u32, 0, name);
    put(e, u32, 4, typ);
    put(e, u32, 8, flags);
    put(e, u32, 16, @intCast(offset));
    put(e, u32, 20, @intCast(size));
    put(e, u32, 24, sh_link);
    put(e, u32, 28, sh_info);
    put(e, u32, 32, addralign);
    put(e, u32, 36, entsize);
}

fn nameOff(names: []const u8, name: []const u8) u32 {
    return @intCast(std.mem.indexOf(u8, names, name).?);
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
    try std.testing.expectEqual(@as(u16, EM_386), std.mem.readInt(u16, obj[18..20], .little));
    try std.testing.expectEqual(@as(u8, 1), obj[4]); // ELFCLASS32
}
