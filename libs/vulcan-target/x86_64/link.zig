//! In-memory linker for x86-64. Compiles named functions, lays them out consecutively
//! in one blob, and resolves each `call` relocation (a rel32 displacement) to the
//! target symbol's offset. The result is a position-independent blob entered at any
//! function's offset. object.zig emits the same data as an ELF .o.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const mm = @import("vulcan-opt").microarch;

const Function = ir.function.Function;

pub const Error = isel.Error || error{UndefinedSymbol};

const Entry = struct { name: []const u8, func: *const Function };

/// Which section a data global lands in: read-only, writable, or zero-init. Uniform
/// with the other backends' `link.DataKind` (native.zig links against whichever the
/// host arch selects).
pub const DataKind = enum { rodata, data, bss };

/// An internal relocation within a data object: at byte offset `off` WITHIN the
/// object, a pointer-sized absolute runtime address of `symbol` (another data
/// object or a function) must be written once the module is mapped. This is how a
/// pointer-valued global (e.g. `char *s = "hi";`) gets patched to point at another
/// object's runtime address.
pub const DataReloc = struct { off: usize, symbol: []const u8 };

/// A named data global placed in the linked output. For `.bss`, `bytes` is empty
/// and `size` gives the zero-initialized length. Otherwise `size == bytes.len`.
/// The module borrows the bytes (and `relocs`, if any).
pub const Data = struct { name: []const u8, bytes: []const u8, kind: DataKind, size: u64, relocs: []const DataReloc = &.{} };

pub const Module = struct {
    funcs: std.ArrayListUnmanaged(Entry) = .empty,
    data: std.ArrayListUnmanaged(Data) = .empty,
    /// When set, `object.writeModule` selects code for this microarch model
    /// instead of the generic default. Null keeps the generic path.
    model: ?*const mm.Model = null,

    pub fn addFunction(self: *Module, allocator: std.mem.Allocator, name: []const u8, func: *const Function) std.mem.Allocator.Error!void {
        try self.funcs.append(allocator, .{ .name = name, .func = func });
    }

    /// Add a named read-only data blob (a global constant, into `.rodata`). The
    /// bytes are borrowed and must outlive the module.
    pub fn addData(self: *Module, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = bytes, .kind = .rodata, .size = bytes.len });
    }

    /// Add a named read-only data blob carrying internal relocations (e.g. a
    /// `const char *s = "hi"` global whose bytes are a pointer patched to another
    /// object's runtime address at map time). Bytes/relocs are borrowed.
    pub fn addDataRelocs(self: *Module, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8, relocs: []const DataReloc) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = bytes, .kind = .rodata, .size = bytes.len, .relocs = relocs });
    }

    /// Add a named writable data global (into `.data`).
    pub fn addWritable(self: *Module, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = bytes, .kind = .data, .size = bytes.len });
    }

    /// Add a named writable data global carrying internal relocations (into
    /// `.data`). Bytes/relocs are borrowed.
    pub fn addWritableRelocs(self: *Module, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8, relocs: []const DataReloc) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = bytes, .kind = .data, .size = bytes.len, .relocs = relocs });
    }

    /// Add a named zero-initialized data global of `size` bytes (into `.bss`).
    pub fn addBss(self: *Module, allocator: std.mem.Allocator, name: []const u8, size: u64) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = &.{}, .kind = .bss, .size = size });
    }

    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        self.funcs.deinit(allocator);
        self.data.deinit(allocator);
    }
};

pub const Symbol = struct { name: []const u8, offset: usize };

/// A data global's placement within its section: `off` is the byte offset within
/// that section (rodata/data/bss laid out separately), aligned up to the object's
/// natural alignment (see `alignOfData`). `bytes` is empty for `.bss`. `relocs` is
/// carried forward from `Data` (see `DataReloc`); empty for `.bss`.
pub const DataSym = struct { name: []const u8, kind: DataKind, off: usize, size: usize, bytes: []const u8, relocs: []const DataReloc = &.{} };

pub const Linked = struct {
    code: []u8,
    symbols: []Symbol,
    /// Per-section placement of each data global (see `DataSym`).
    data: []DataSym,
    /// Carried-forward `global_addr` (`.pcrel_lea`) relocations: byte offset (module-
    /// wide, into `code`) of each `lea rd, [rip+disp32]`'s disp32 field, plus the
    /// symbol it targets. Resolved once the image is mapped (see `applyGlobalReloc`);
    /// intra-module `.call` relocs are always resolved above and never appear here.
    relocs: []isel.Reloc,

    pub fn deinit(self: *Linked, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.symbols);
        allocator.free(self.data);
        allocator.free(self.relocs);
    }
    pub fn addressOf(self: *const Linked, name: []const u8) ?usize {
        for (self.symbols) |s| if (std.mem.eql(u8, s.name, name)) return s.offset;
        return null;
    }
};

/// A data object's natural alignment, derived from its size (no explicit align field
/// yet): the next power of two up to `size`, capped at 8. Uniform with the other
/// backends.
fn alignOfData(d: Data) usize {
    const size: usize = @intCast(@max(@as(u64, 1), d.size));
    return @min(8, std.math.ceilPowerOfTwo(usize, size) catch 8);
}

/// Lay out `module.data` into per-section `DataSym`s, each object's `off` aligned up
/// to its natural alignment within its (rodata/data/bss) section. Caller owns the result.
fn layoutData(allocator: std.mem.Allocator, module: *const Module) std.mem.Allocator.Error![]DataSym {
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
    return data_syms;
}

/// Patch a carried-forward `global_addr` (`.pcrel_lea`) relocation now that the
/// symbol's runtime address is known. `image` is the mapped image whose code section
/// holds the `lea`'s disp32 field at BYTE offset `r.offset` (matching `Linked.relocs`/
/// `Linked.code`, both byte-addressed unlike aarch64/riscv64's word-indexed code);
/// `site_addr` is the runtime address of that disp32 field itself and `target_addr`
/// is the symbol's resolved runtime address. `image` is `anytype` (rather than the
/// concrete `jit_platform.MappedImage` instantiation) so this stays independent of
/// which arch's instruction-cache-sync hook parameterized it - see aarch64/link.zig's
/// `applyGlobalReloc` for the same rationale.
///
/// `lea rd, [rip+disp32]`'s RIP is the address of the NEXT instruction, i.e. the end
/// of the disp32 field (`site_addr + 4`): `disp32 = target_addr - (site_addr + 4)`,
/// written little-endian into the 4 bytes at the site.
pub fn applyGlobalReloc(image: anytype, r: isel.Reloc, site_addr: usize, target_addr: usize) void {
    switch (r.kind) {
        .pcrel_lea => {
            const disp: i32 = @intCast(@as(i64, @intCast(target_addr)) - @as(i64, @intCast(site_addr + 4)));
            const bytes = image.ptr(image.code_off + r.offset)[0..4];
            std.mem.writeInt(u32, bytes, @bitCast(disp), .little);
        },
        .call => unreachable, // `.call` relocs are always resolved by `compileModule`, never carried forward
        // `.got_pcrel` (a GOT-indirect data import) is never carried forward here:
        // `compileModule` rejects it (the JIT path has no dynamic GOT), so it cannot reach
        // this map-time patch.
        .got_pcrel => unreachable,
    }
}

/// Compile and link `module`'s functions into one blob. Caller owns the result.
pub fn compileModule(allocator: std.mem.Allocator, module: *const Module) Error!Linked {
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

    // Lay out functions consecutively, 16-byte aligned starts.
    const offsets = try allocator.alloc(usize, funcs.len);
    defer allocator.free(offsets);
    var total: usize = 0;
    for (compiled, 0..) |c, i| {
        offsets[i] = total;
        total += (c.code.len + 15) & ~@as(usize, 15);
    }

    const symbols = try allocator.alloc(Symbol, funcs.len);
    errdefer allocator.free(symbols);
    for (funcs, 0..) |e, i| symbols[i] = .{ .name = e.name, .offset = offsets[i] };

    const code = try allocator.alloc(u8, total);
    errdefer allocator.free(code);
    @memset(code, 0x90); // pad with NOPs
    for (compiled, 0..) |c, i| @memcpy(code[offsets[i]..][0..c.code.len], c.code);

    // Resolve each intra-module `.call` relocation (rel32 = target - (reloc_site + 4));
    // carry every `.pcrel_lea` (`global_addr`) relocation forward, globalizing its
    // offset to a module-wide byte index. Its target is a runtime data address,
    // unknown until the image is mapped.
    var global_relocs: std.ArrayList(isel.Reloc) = .empty;
    errdefer global_relocs.deinit(allocator);
    for (compiled, 0..) |c, i| {
        for (c.relocs) |r| {
            const site = offsets[i] + r.offset;
            switch (r.kind) {
                .call => {
                    const target = addressBySymbol(symbols, r.symbol) orelse return error.UndefinedSymbol;
                    const rel: i32 = @intCast(@as(i64, @intCast(target)) - @as(i64, @intCast(site + 4)));
                    std.mem.writeInt(u32, code[site..][0..4], @bitCast(rel), .little);
                },
                .pcrel_lea => try global_relocs.append(allocator, .{ .offset = site, .symbol = r.symbol, .kind = .pcrel_lea }),
                // A GOT-indirect data import (`via_got`) needs a dynamic GOT + GLOB_DAT the
                // real `ld.so` fills, which this in-memory JIT linker cannot synthesize. It is
                // only produced on the object/dynamic-link path (see `object.zig` +
                // `vulcan-link`), so it never reaches here; reject it fail-closed.
                .got_pcrel => return error.Unsupported,
            }
        }
    }

    const data_syms = try layoutData(allocator, module);
    errdefer allocator.free(data_syms);

    return .{ .code = code, .symbols = symbols, .data = data_syms, .relocs = try global_relocs.toOwnedSlice(allocator) };
}

fn addressBySymbol(symbols: []const Symbol, name: []const u8) ?usize {
    for (symbols) |s| if (std.mem.eql(u8, s.name, name)) return s.offset;
    return null;
}

test "links two functions and resolves the call" {
    const allocator = std.testing.allocator;
    var helper = Function.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        const r = try helper.appendArithImm(b, t, .mul, x, 2);
        helper.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var main = Function.init(allocator);
    defer main.deinit();
    {
        const t = try main.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try main.appendBlock();
        const x = try main.appendBlockParam(b, t);
        const r = try main.appendCall(b, t, "helper", &.{x}); // helper(x) = x*2
        main.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var module: Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "helper", &helper);

    var linked = try compileModule(allocator, &module);
    defer linked.deinit(allocator);
    try std.testing.expect(linked.addressOf("main") != null);
    try std.testing.expect(linked.addressOf("helper").? != linked.addressOf("main").?);
    // The call site holds a non-zero (resolved) displacement.
    var any_call = false;
    for (linked.code, 0..) |byte, i| if (byte == 0xE8 and i + 4 < linked.code.len) {
        if (std.mem.readInt(u32, linked.code[i + 1 ..][0..4], .little) != 0) any_call = true;
    };
    try std.testing.expect(any_call);
}
