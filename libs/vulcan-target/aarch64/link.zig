//! This file links AArch64 modules in memory. It compiles a set of named functions,
//! lays them out in one code image, and resolves every intra-module call (`bl`)
//! relocation to its callee. The image is position-independent, since each `bl` is
//! PC-relative, so it maps at any address. Data globals (`global_addr`) are laid out
//! into named `DataSym`s. Their `adrp`/`add` relocations are carried forward in
//! `Linked.relocs` instead of resolved here. Their targets are runtime addresses,
//! unknown until the image is mapped. See `applyGlobalReloc`, called after mapping.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const encode = @import("encode.zig");
const jit_platform = @import("../jit_platform.zig");

const Function = ir.function.Function;

pub const Error = isel.Error || error{ UndefinedSymbol, UnsupportedReloc };

/// The section a data global lands in: read-only, writable, or zero-initialized.
pub const DataKind = enum { rodata, data, bss };

/// An internal relocation within a data object. At byte offset `off` within the
/// object, the linker must write a pointer-sized absolute runtime address of
/// `symbol` (another data object or a function) once the module is mapped. This is
/// how a pointer-valued global (for example `char *s = "hi";`) gets patched to point
/// at another object's runtime address.
pub const DataReloc = struct { off: usize, symbol: []const u8 };

/// A named data global placed in the linked output. For `.bss`, `bytes` is empty
/// and `size` gives the zero-initialized length. Otherwise `size == bytes.len`.
/// The module borrows the bytes, and the relocations if any.
pub const Data = struct { name: []const u8, bytes: []const u8, kind: DataKind, size: u64, relocs: []const DataReloc = &.{} };

/// A set of named functions and data globals to link together. The first added
/// function is the entry point. It sits at offset 0 of the linked image.
pub const Module = struct {
    functions: std.ArrayListUnmanaged(Entry) = .empty,
    data: std.ArrayListUnmanaged(Data) = .empty,

    pub const Entry = struct { name: []const u8, func: *const Function };

    pub fn addFunction(self: *Module, allocator: std.mem.Allocator, name: []const u8, func: *const Function) std.mem.Allocator.Error!void {
        try self.functions.append(allocator, .{ .name = name, .func = func });
    }

    /// Add a named read-only data blob, a global constant, into `.rodata`. The
    /// bytes are borrowed and must outlive the module.
    pub fn addData(self: *Module, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = bytes, .kind = .rodata, .size = bytes.len });
    }

    /// Add a named read-only data blob that carries internal relocations. For
    /// example, a `const char *s = "hi"` global whose bytes are a pointer, patched
    /// to another object's runtime address at map time. The bytes and relocations
    /// are borrowed.
    pub fn addDataRelocs(self: *Module, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8, relocs: []const DataReloc) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = bytes, .kind = .rodata, .size = bytes.len, .relocs = relocs });
    }

    /// Add a named writable data global into `.data`.
    pub fn addWritable(self: *Module, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = bytes, .kind = .data, .size = bytes.len });
    }

    /// Add a named writable data global that carries internal relocations, into
    /// `.data`. The bytes and relocations are borrowed.
    pub fn addWritableRelocs(self: *Module, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8, relocs: []const DataReloc) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = bytes, .kind = .data, .size = bytes.len, .relocs = relocs });
    }

    /// Add a named zero-initialized data global of `size` bytes into `.bss`.
    pub fn addBss(self: *Module, allocator: std.mem.Allocator, name: []const u8, size: u64) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = &.{}, .kind = .bss, .size = size });
    }

    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        self.functions.deinit(allocator);
        self.data.deinit(allocator);
    }
};

/// A function's byte offset within the linked image.
pub const Symbol = struct { name: []const u8, offset: usize };

/// A data global's placement within its section. `off` is the byte offset within
/// that section. `.rodata`, `.data`, and `.bss` are laid out separately, each
/// aligned up to the object's natural alignment (see `alignOfData`). `bytes` is
/// empty for `.bss`. `relocs` is carried forward from `Data` (see `DataReloc`), and
/// is also empty for `.bss`.
pub const DataSym = struct { name: []const u8, kind: DataKind, off: usize, size: usize, bytes: []const u8, relocs: []const DataReloc = &.{} };

/// A linked code image: the machine words, each function's byte offset, each
/// data global's per-section placement, and any still-unresolved `global_addr`
/// relocations. Each relocation holds a word index into `code`, the symbol, and
/// which half of the adrp/add pair it patches. Intra-module `.call` relocations are
/// always resolved by `compileModule` and never appear here.
pub const Linked = struct {
    code: []u32,
    symbols: []Symbol,
    data: []DataSym,
    relocs: []isel.Reloc,

    pub fn deinit(self: *Linked, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.symbols);
        allocator.free(self.data);
        allocator.free(self.relocs);
    }

    /// The byte offset of the function named `name`, or null if absent.
    pub fn addressOf(self: *const Linked, name: []const u8) ?usize {
        for (self.symbols) |s| if (std.mem.eql(u8, s.name, name)) return s.offset;
        return null;
    }
};

/// A data object's natural alignment, derived from its size since `Data` has no
/// explicit align field yet. It is the next power of two up to `size`, capped at 8.
/// This covers scalars and pointers up to a 64-bit word. It is conservative and
/// correct for this case. A future explicit per-symbol align field will replace
/// this.
fn alignOfData(d: Data) usize {
    const size: usize = @intCast(@max(@as(u64, 1), d.size));
    return @min(8, std.math.ceilPowerOfTwo(usize, size) catch 8);
}

/// Compile every function, concatenate the code, resolve each intra-module `bl`
/// call relocation, lay out data globals into per-section `DataSym`s, and carry
/// every `global_addr` relocation forward unresolved. The caller owns the result.
pub fn compileModule(allocator: std.mem.Allocator, module: *const Module) Error!Linked {
    const funcs = module.functions.items;
    var compiled = try allocator.alloc(isel.Compiled, funcs.len);
    var n: usize = 0;
    defer {
        for (0..n) |i| compiled[i].deinit(allocator);
        allocator.free(compiled);
    }

    const word_off = try allocator.alloc(usize, funcs.len);
    defer allocator.free(word_off);
    var total: usize = 0;
    for (funcs, 0..) |e, i| {
        compiled[i] = try isel.compileFunction(allocator, e.func, .{});
        n = i + 1;
        word_off[i] = total;
        total += compiled[i].code.len;
    }

    var code = try allocator.alloc(u32, total);
    errdefer allocator.free(code);
    for (0..funcs.len) |i| @memcpy(code[word_off[i]..][0..compiled[i].code.len], compiled[i].code);

    // Resolve each intra-module call relocation, a PC-relative `bl`, to its callee's
    // word. Carry every `global_addr` relocation (`.adrp_pg`/`.add_pgoff`) forward,
    // and turn its offset into a module-wide word index. Their targets are runtime
    // data addresses, unknown until the image is mapped.
    var global_relocs: std.ArrayList(isel.Reloc) = .empty;
    errdefer global_relocs.deinit(allocator);
    for (0..funcs.len) |i| {
        for (compiled[i].relocs) |r| {
            const at = word_off[i] + r.offset;
            switch (r.kind) {
                .call => {
                    const target = symbolWord(funcs, word_off, r.symbol) orelse return error.UndefinedSymbol;
                    code[at] = encode.bl(@intCast((@as(i64, @intCast(target)) - @as(i64, @intCast(at))) * 4));
                },
                .adrp_pg, .add_pgoff => {
                    try global_relocs.append(allocator, .{ .offset = at, .symbol = r.symbol, .kind = r.kind });
                },
                // GOT-indirect relocations, a data import from another shared object, are
                // resolved by the real linker or loader, not this in-process static or JIT
                // linker.
                .got_pg, .got_lo12 => return error.UnsupportedReloc,
            }
        }
    }

    const symbols = try allocator.alloc(Symbol, funcs.len);
    errdefer allocator.free(symbols);
    for (funcs, 0..) |e, i| symbols[i] = .{ .name = e.name, .offset = word_off[i] * 4 };

    // Lay out data globals into their own logical sections: rodata, data, and bss.
    // Each object's `off` is aligned up to its natural alignment within that section.
    const data_syms = try allocator.alloc(DataSym, module.data.items.len);
    errdefer allocator.free(data_syms);
    var rodata_off: usize = 0;
    var data_off: usize = 0;
    var bss_off: usize = 0;
    for (module.data.items, 0..) |d, i| {
        const a = alignOfData(d);
        const size: usize = @intCast(d.size);
        data_syms[i] = switch (d.kind) {
            .rodata => blk: {
                rodata_off = std.mem.alignForward(usize, rodata_off, a);
                const sym = DataSym{ .name = d.name, .kind = d.kind, .off = rodata_off, .size = size, .bytes = d.bytes, .relocs = d.relocs };
                rodata_off += size;
                break :blk sym;
            },
            .data => blk: {
                data_off = std.mem.alignForward(usize, data_off, a);
                const sym = DataSym{ .name = d.name, .kind = d.kind, .off = data_off, .size = size, .bytes = d.bytes, .relocs = d.relocs };
                data_off += size;
                break :blk sym;
            },
            .bss => blk: {
                bss_off = std.mem.alignForward(usize, bss_off, a);
                const sym = DataSym{ .name = d.name, .kind = d.kind, .off = bss_off, .size = size, .bytes = &.{} };
                bss_off += size;
                break :blk sym;
            },
        };
    }

    return .{
        .code = code,
        .symbols = symbols,
        .data = data_syms,
        .relocs = try global_relocs.toOwnedSlice(allocator),
    };
}

fn symbolWord(funcs: []const Module.Entry, word_off: []const usize, name: []const u8) ?usize {
    for (funcs, 0..) |e, i| if (std.mem.eql(u8, e.name, name)) return word_off[i];
    return null;
}

/// Patch a carried-forward `global_addr` relocation now that the symbol's runtime
/// address is known. `image` is the mapped image whose code section holds the
/// instruction at `r.offset`, a word index that matches `Linked.relocs` and
/// `Linked.code`. `site_addr` is the runtime address of the `adrp` instruction. Its
/// own page is the base of the page-relative delta. `target_addr` is the symbol's
/// resolved runtime address. `image` is `anytype`, rather than a concrete
/// `jit_platform.MappedImage`, because `MappedImage` is a generic type constructor
/// parameterized by the architecture's instruction-cache-sync hook. This accepts any
/// of its instantiations, so `link.zig` stays independent of which sync hook the
/// caller chose, for example aarch64/jit.zig's `syncICache`, or a test's no-op hook.
///
/// `.adrp_pg` rewrites the immhi:immlo page-delta fields to
/// `(target_page - site_page) >> 12`. A page is an address with the low 12 bits
/// cleared. `.add_pgoff` rewrites the imm12 field to `target_addr`'s low 12 bits.
/// Both are edited in place with `std.mem.readInt`/`writeInt` over the
/// instruction's 4 bytes, preserving every other bit, including the destination
/// and source register fields.
pub fn applyGlobalReloc(image: anytype, r: isel.Reloc, site_addr: usize, target_addr: usize) void {
    const word_bytes = image.ptr(image.code_off + r.offset * 4)[0..4];
    var word = std.mem.readInt(u32, word_bytes, .little);
    switch (r.kind) {
        .adrp_pg => {
            const site_page = site_addr & ~@as(usize, 0xFFF);
            const target_page = target_addr & ~@as(usize, 0xFFF);
            const delta_pages = @divExact(@as(i64, @intCast(target_page)) - @as(i64, @intCast(site_page)), 4096);
            const imm: i21 = @intCast(delta_pages);
            const u: u21 = @bitCast(imm);
            const immlo: u32 = u & 0x3;
            const immhi: u32 = u >> 2;
            word = (word & ~((@as(u32, 0x3) << 29) | (@as(u32, 0x7FFFF) << 5))) | (immlo << 29) | (immhi << 5);
        },
        .add_pgoff => {
            const lo12: u32 = @intCast(target_addr & 0xFFF);
            word = (word & ~(@as(u32, 0xFFF) << 10)) | (lo12 << 10);
        },
        .call => unreachable, // `.call` relocations are always resolved by `compileModule` and never carried forward.
        // GOT-indirect relocations are never carried into this in-process linker.
        // `compileModule` rejects them with error.UnsupportedReloc. The real linker or
        // loader resolves them.
        .got_pg, .got_lo12 => unreachable,
    }
    std.mem.writeInt(u32, word_bytes, word, .little);
}

test "links an intra-module call to a real bl offset" {
    const allocator = std.testing.allocator;
    const i32_t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32_t_kind);
        const b = try callee.appendBlock();
        const a = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(a) });
    }

    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32_t_kind);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }

    var module: Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "callee", &callee);
    try module.addFunction(allocator, "caller", &caller);

    var linked = try compileModule(allocator, &module);
    defer linked.deinit(allocator);

    try std.testing.expectEqual(@as(?usize, 0), linked.addressOf("callee"));
    try std.testing.expect(linked.addressOf("caller") != null);
    try std.testing.expectEqual(@as(usize, 0), linked.relocs.len);
    try std.testing.expectEqual(@as(usize, 0), linked.data.len);
}

test "compileModule lays out a global_addr's data object and carries its relocs forward" {
    const allocator = std.testing.allocator;
    const ptr_t_kind = ir.types.TypeKind.ptr;
    const i8_t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 8 } };

    var entry = Function.init(allocator);
    defer entry.deinit();
    {
        const ptr_t = try entry.types.intern(ptr_t_kind);
        const i8_t = try entry.types.intern(i8_t_kind);
        const b = try entry.appendBlock();
        const g = try entry.appendGlobalAddr(b, ptr_t, "g");
        const r = try entry.appendInst(b, i8_t, .{ .load = .{ .ptr = g } });
        entry.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }

    var module: Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "entry", &entry);
    const bytes = [_]u8{ 1, 2, 3, 4 };
    try module.addData(allocator, "g", &bytes);

    var linked = try compileModule(allocator, &module);
    defer linked.deinit(allocator);

    // The data object landed in `.data` at offset 0, the first and only rodata symbol.
    try std.testing.expectEqual(@as(usize, 1), linked.data.len);
    try std.testing.expectEqualStrings("g", linked.data[0].name);
    try std.testing.expectEqual(DataKind.rodata, linked.data[0].kind);
    try std.testing.expectEqual(@as(usize, 0), linked.data[0].off);
    try std.testing.expectEqual(@as(usize, 4), linked.data[0].size);
    try std.testing.expectEqualSlices(u8, &bytes, linked.data[0].bytes);

    // Both global relocations (adrp_pg and add_pgoff) were carried forward, naming
    // "g". They were not resolved into a `bl`, unlike a `.call` relocation.
    try std.testing.expectEqual(@as(usize, 2), linked.relocs.len);
    var saw_pg = false;
    var saw_pgoff = false;
    for (linked.relocs) |r| {
        try std.testing.expectEqualStrings("g", r.symbol);
        switch (r.kind) {
            .adrp_pg => saw_pg = true,
            .add_pgoff => saw_pgoff = true,
            .call, .got_pg, .got_lo12 => unreachable,
        }
    }
    try std.testing.expect(saw_pg);
    try std.testing.expect(saw_pgoff);
}

test "compileModule aligns data objects per-symbol within a section" {
    const allocator = std.testing.allocator;

    var module: Module = .{};
    defer module.deinit(allocator);
    // A 1-byte object is followed by an 8-byte one. The 8-byte object must land at
    // offset 8, aligned up from 1, not at offset 1 packed by byte.
    const one = [_]u8{0xAA};
    const eight = [_]u8{0} ** 8;
    try module.addData(allocator, "small", &one);
    try module.addData(allocator, "big", &eight);

    var linked = try compileModule(allocator, &module);
    defer linked.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), linked.data[0].off);
    try std.testing.expectEqual(@as(usize, 8), linked.data[1].off);
}

test "compileModule carries bss size with no bytes" {
    const allocator = std.testing.allocator;

    var module: Module = .{};
    defer module.deinit(allocator);
    try module.addBss(allocator, "buf", 16);

    var linked = try compileModule(allocator, &module);
    defer linked.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), linked.data.len);
    try std.testing.expectEqual(DataKind.bss, linked.data[0].kind);
    try std.testing.expectEqual(@as(usize, 16), linked.data[0].size);
    try std.testing.expectEqual(@as(usize, 0), linked.data[0].bytes.len);
}

fn noSync(_: []const u8) void {}

test "applyGlobalReloc patches adrp/add immediates to the resolved page delta and lo12" {
    const provider = jit_platform.default_provider;
    const Image = jit_platform.MappedImage(noSync);

    // A code image with a placeholder adrp/add pair, register x3, at word 0 and 1.
    var code = [_]u32{ encode.adrp(.x3, 0), encode.addImm64(.x3, .x3, 0) };
    var img = try Image.map(provider, std.mem.sliceAsBytes(&code), &.{}, &.{}, 0);
    defer img.deinit();

    const site_addr = @intFromPtr(img.ptr(img.code_off));
    // Target 3 pages up plus a 0x123 page offset, well within adrp/add's reach.
    const target_addr = (site_addr & ~@as(usize, 0xFFF)) + 3 * 4096 + 0x123;

    applyGlobalReloc(&img, .{ .offset = 0, .symbol = "g", .kind = .adrp_pg }, site_addr, target_addr);
    applyGlobalReloc(&img, .{ .offset = 1, .symbol = "g", .kind = .add_pgoff }, site_addr, target_addr);

    const adrp_word = std.mem.readInt(u32, img.ptr(img.code_off)[0..4], .little);
    const add_word = std.mem.readInt(u32, img.ptr(img.code_off + 4)[0..4], .little);

    // Decode the adrp's immhi:immlo back into a signed page delta and confirm it
    // reproduces the target page relative to the site's own page.
    const immlo: u32 = (adrp_word >> 29) & 0x3;
    const immhi: u32 = (adrp_word >> 5) & 0x7FFFF;
    const u: u21 = @intCast((immhi << 2) | immlo);
    const imm: i21 = @bitCast(u);
    const decoded_target_page: i64 = @as(i64, @intCast(site_addr & ~@as(usize, 0xFFF))) + @as(i64, imm) * 4096;
    try std.testing.expectEqual(@as(i64, @intCast(target_addr & ~@as(usize, 0xFFF))), decoded_target_page);
    // The rd field, bits [4:0], is untouched by the patch.
    try std.testing.expectEqual(@as(u32, @intFromEnum(encode.Reg.x3)), adrp_word & 0x1F);

    // The add's imm12 (bits [21:10]) decodes to the target's low 12 bits.
    const lo12 = (add_word >> 10) & 0xFFF;
    try std.testing.expectEqual(@as(u32, @intCast(target_addr & 0xFFF)), lo12);
    // The rd and rn fields, bits [4:0] and [9:5], are untouched.
    try std.testing.expectEqual(@as(u32, @intFromEnum(encode.Reg.x3)), add_word & 0x1F);
    try std.testing.expectEqual(@as(u32, @intFromEnum(encode.Reg.x3)), (add_word >> 5) & 0x1F);
}
