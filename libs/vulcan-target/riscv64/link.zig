//! Module linking for RISC-V: compile a set of named functions, lay them out
//! sequentially, and resolve each intra-module call to a real `jal` offset.
//! Calls to symbols outside the module stay as relocations for a later linker.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const isel = @import("isel.zig");
const mm = @import("vulcan-opt").microarch;

const Function = ir.function.Function;

pub const Error = isel.Error;

/// A named function gathered for compilation. The module borrows the function.
const Entry = struct { name: []const u8, func: *const Function };

/// Which section a data global lands in: read-only, writable, or zero-init.
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

/// A grouping of named functions (and data) compiled and linked together.
pub const Module = struct {
    entries: std.ArrayList(Entry) = .empty,
    data: std.ArrayList(Data) = .empty,
    /// When set, `object.writeModule` selects code for this microarch model
    /// instead of the generic default. Null keeps the generic path.
    model: ?*const mm.Model = null,

    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        self.entries.deinit(allocator);
        self.data.deinit(allocator);
    }

    pub fn addFunction(self: *Module, allocator: std.mem.Allocator, name: []const u8, func: *const Function) std.mem.Allocator.Error!void {
        try self.entries.append(allocator, .{ .name = name, .func = func });
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
};

/// A symbol's resolved location: the word index where its function begins.
pub const Symbol = struct { name: []const u8, offset: usize };

/// A data global's placement within its section: `off` is the byte offset within
/// that section (rodata/data/bss laid out separately), aligned up to the object's
/// natural alignment (see `alignOfData`). `bytes` is empty for `.bss`. `relocs` is
/// carried forward from `Data` (see `DataReloc`); empty for `.bss`. Uniform with
/// the other backends so native.zig links against it regardless of host arch.
pub const DataSym = struct { name: []const u8, kind: DataKind, off: usize, size: usize, bytes: []const u8, relocs: []const DataReloc = &.{} };

/// A linked module: concatenated code, a symbol table, per-section data placement,
/// and any still-unresolved (external) relocations, with offsets in module-global
/// word indices.
pub const Linked = struct {
    code: []u32,
    symbols: []Symbol,
    /// Per-section placement of each data global (see `DataSym`).
    data: []DataSym,
    relocs: []isel.Reloc,

    pub fn deinit(self: *Linked, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.symbols);
        allocator.free(self.data);
        allocator.free(self.relocs);
    }

    pub fn symbolOffset(self: *const Linked, name: []const u8) ?usize {
        for (self.symbols) |s| {
            if (std.mem.eql(u8, s.name, name)) return s.offset;
        }
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

/// Patch a carried-forward `global_addr` relocation once its runtime address is
/// known. `global_addr` lowers to an `auipc rd, %pcrel_hi(sym)` / `addi rd, rd,
/// %pcrel_lo(.Lhi)` pair (see `isel.zig`'s `.global_addr` arm), so two relocations
/// arrive here, one at a time: the `.pcrel_hi20` on the `auipc` and the
/// `.pcrel_lo12` on the `addi`. `site_addr`/`target_addr` are already runtime byte
/// addresses (native.zig scales `r.offset`'s word index by 4 for `[]u32` code
/// before calling in); `image.ptr` also takes a byte offset into the mapped image.
///
/// The RISC-V paired-reloc rule (mirrors `ld.zig`'s `applyReloc`/`patchLo12` and its
/// two-pass hi/lo driver): the `.pcrel_lo12`'s 12 low bits must reproduce the SAME
/// PC-relative delta its paired `auipc` encoded, i.e. `target_addr - auipc_site`,
/// NOT `target_addr - this_site` (the `addi` sits one word after its `auipc`, so
/// the two deltas differ and using the wrong one corrupts the low bits). Rather
/// than have native.zig resolve the pair and pass the `auipc`'s site alongside,
/// `r.pair` already carries the `auipc`'s module-global word index (`compileModule`
/// globalizes it the same way it globalizes `r.offset`), so the `auipc`'s runtime
/// site address is recomputed here from `image`/`r.pair` directly - self-contained,
/// no native.zig change needed.
pub fn applyGlobalReloc(image: anytype, r: isel.Reloc, site_addr: usize, target_addr: usize) void {
    switch (r.kind) {
        .pcrel_hi20 => {
            // The high 20 bits of the PC-relative delta go in the `auipc`'s
            // U-immediate (bits 31:12). The +0x800 pre-rounds for the lo12 sign,
            // matching `ld.zig`'s `applyReloc(.pcrel_hi20)`.
            const delta = @as(i64, @intCast(target_addr)) - @as(i64, @intCast(site_addr));
            const hi: u32 = @truncate(@as(u64, @bitCast(delta +% 0x800)) >> 12);
            const word_bytes = image.ptr(image.code_off + r.offset * 4)[0..4];
            const word = std.mem.readInt(u32, word_bytes, .little);
            std.mem.writeInt(u32, word_bytes, (word & 0x0000_0fff) | (hi << 12), .little);
        },
        .pcrel_lo12 => {
            const auipc_site_addr = @intFromPtr(image.ptr(image.code_off + r.pair * 4));
            const pcrel = @as(i64, @intCast(target_addr)) - @as(i64, @intCast(auipc_site_addr));
            const lo: u32 = @as(u12, @truncate(@as(u64, @bitCast(pcrel))));
            const word_bytes = image.ptr(image.code_off + r.offset * 4)[0..4];
            const word = std.mem.readInt(u32, word_bytes, .little);
            std.mem.writeInt(u32, word_bytes, (word & 0x000f_ffff) | (lo << 20), .little);
        },
        .call => unreachable, // `.call` relocs are always resolved by `compileModule`, never carried forward
        .got_hi20 => unreachable, // `via_got` (GOT-indirect) is only emitted on the dynamic-link path, never JIT
    }
}

fn findSymbol(symbols: []const Symbol, name: []const u8) ?usize {
    for (symbols) |s| {
        if (std.mem.eql(u8, s.name, name)) return s.offset;
    }
    return null;
}

/// Compile and link every function in `module`.
pub fn compileModule(allocator: std.mem.Allocator, module: *const Module) Error!Linked {
    var code: std.ArrayList(u32) = .empty;
    errdefer code.deinit(allocator);
    var symbols: std.ArrayList(Symbol) = .empty;
    errdefer symbols.deinit(allocator);
    var all_relocs: std.ArrayList(isel.Reloc) = .empty;
    defer all_relocs.deinit(allocator);

    // Lay out each function, recording its start and globalizing its relocs.
    for (module.entries.items) |entry| {
        const start = code.items.len;
        var compiled = try isel.compileFunction(allocator, entry.func, .{});
        defer compiled.deinit(allocator);
        try code.appendSlice(allocator, compiled.code);
        try symbols.append(allocator, .{ .name = entry.name, .offset = start });
        for (compiled.relocs) |r| {
            // `r.pair` (meaningful only for `.pcrel_lo12`) is a function-local word
            // index into the same function's code, just like `r.offset` - globalize
            // it the same way so `applyGlobalReloc` can find the paired `auipc`'s
            // runtime site directly from the module-global reloc.
            //
            // isel emits the `.pcrel_lo12` with an EMPTY `.symbol` (its target is
            // implied by the paired `auipc`, `isel.zig`'s `.global_addr` arm). But
            // native.zig's generic in-process reloc loop resolves EVERY carried
            // reloc's `target_addr` via `map.get(r.symbol)` and errors on a miss, so
            // a `""` symbol would fail before `applyGlobalReloc` ever runs. Copy the
            // paired hi20's data-symbol name onto the lo12 here (matching the lo12's
            // function-local `r.pair` against the paired hi20's `.offset` within this
            // function's relocs): both hi and lo12 then resolve to the same target T,
            // and `applyGlobalReloc` computes `(T - auipc_site)` for each. This only
            // shapes the in-process `Linked.relocs` list; the object.zig/ld.zig qemu
            // path consumes isel's original relocs, so it is unaffected.
            const symbol = if (r.kind == .pcrel_lo12) blk: {
                for (compiled.relocs) |h| {
                    if (h.kind == .pcrel_hi20 and h.offset == r.pair) break :blk h.symbol;
                }
                break :blk r.symbol;
            } else r.symbol;
            try all_relocs.append(allocator, .{ .offset = start + r.offset, .symbol = symbol, .kind = r.kind, .pair = start + r.pair });
        }
    }

    // Resolve intra-module calls to a real `jal`, keeping external calls AND every
    // `global_addr` pcrel_hi20/pcrel_lo12 pair as relocations (they name data
    // symbols, resolved once the module is mapped - see `applyGlobalReloc`). Only
    // `.call` relocs are eligible for intra-module resolution: a `.pcrel_lo12`
    // carries no symbol name (`findSymbol("")` never matches) and a `.pcrel_hi20`
    // names a data symbol, which `findSymbol` (function symbols only) also never
    // matches - the `.call` guard just documents that invariant.
    var external: std.ArrayList(isel.Reloc) = .empty;
    errdefer external.deinit(allocator);
    for (all_relocs.items) |r| {
        if (r.kind == .call) {
            if (findSymbol(symbols.items, r.symbol)) |target| {
                const delta = (@as(i64, @intCast(target)) - @as(i64, @intCast(r.offset))) * 4;
                code.items[r.offset] = encode.jal(.x1, @intCast(delta));
                continue;
            }
        }
        try external.append(allocator, r);
    }

    const data_syms = try layoutData(allocator, module);
    errdefer allocator.free(data_syms);

    return .{
        .code = try code.toOwnedSlice(allocator),
        .symbols = try symbols.toOwnedSlice(allocator),
        .data = data_syms,
        .relocs = try external.toOwnedSlice(allocator),
    };
}

test "links an intra-module call to a real jal offset" {
    const allocator = std.testing.allocator;
    const i32_t_kind = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

    // callee: fn(a) -> a   (just `ret a`, which is a bare `jalr`).
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(i32_t_kind);
        const b = try callee.appendBlock();
        const a = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(a) });
    }

    // caller: fn(x) -> callee(x).
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

    // callee is a leaf (one `jalr`). Caller is non-leaf, so it opens a frame and saves/restores ra
    // around the call. The shared Wimmer allocator parks the call result in callee-saved x9, so the
    // caller also saves/restores x9, which pushes its `jal` (now at word 4) to a -16 byte backward
    // offset to callee at word 0. No external relocs remain.
    try std.testing.expectEqualSlices(u32, &.{
        encode.jalr(.x0, .x1, 0), // callee: ret
        encode.addi(.x2, .x2, -16), // caller: open frame
        encode.sd(.x1, .x2, 8), // caller: save ra
        encode.sd(.x9, .x2, 0), // caller: save x9 (holds the result)
        encode.jal(.x1, -16), // caller: call callee  (resolved)
        encode.addi(.x9, .x10, 0), // r = a0  (into callee-saved x9)
        encode.addi(.x10, .x9, 0), // mv a0, r
        encode.ld(.x1, .x2, 8), // caller: restore ra
        encode.ld(.x9, .x2, 0), // caller: restore x9
        encode.addi(.x2, .x2, 16), // caller: close frame
        encode.jalr(.x0, .x1, 0), // caller: ret
    }, linked.code);
    try std.testing.expectEqual(@as(?usize, 0), linked.symbolOffset("callee"));
    try std.testing.expectEqual(@as(?usize, 1), linked.symbolOffset("caller"));
    try std.testing.expectEqual(@as(usize, 0), linked.relocs.len);
}

test "an unresolved external call stays a relocation" {
    const allocator = std.testing.allocator;
    var caller = Function.init(allocator);
    defer caller.deinit();
    const t = try caller.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try caller.appendBlock();
    const x = try caller.appendBlockParam(b, t);
    const r = try caller.appendCall(b, t, "external", &.{x});
    caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    var module: Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "caller", &caller);

    var linked = try compileModule(allocator, &module);
    defer linked.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), linked.relocs.len);
    try std.testing.expectEqualStrings("external", linked.relocs[0].symbol);
    // caller@0 then prologue (frame open + save ra + save the callee-saved x9 that the shared Wimmer
    // allocator parks the result in), so the `jal` lands at word 3 and its relocation is unresolved.
    try std.testing.expectEqual(@as(usize, 3), linked.relocs[0].offset);
}

test "compileModule carries a global_addr's pcrel_hi20/pcrel_lo12 pair and lays out its data" {
    // Non-executing structural test (runs on any host, unlike the in-process JIT
    // tests in native.zig which are riscv64-host-gated): proves the carry logic -
    // `Linked.relocs` holds exactly the hi20/lo12 pair (not resolved to a `jal`,
    // since `findSymbol` only searches function symbols) and `Linked.data` holds
    // the aligned rodata object - without executing anything.
    const allocator = std.testing.allocator;

    var f = Function.init(allocator);
    defer f.deinit();
    const ptr_t = try f.types.intern(.ptr);
    const i8_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const b = try f.appendBlock();
    const g = try f.appendGlobalAddr(b, ptr_t, "g");
    const v = try f.appendInst(b, i8_t, .{ .load = .{ .ptr = g } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var module: Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "f", &f);
    const bytes = [_]u8{42};
    try module.addData(allocator, "g", &bytes);

    var linked = try compileModule(allocator, &module);
    defer linked.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), linked.data.len);
    try std.testing.expectEqualStrings("g", linked.data[0].name);
    try std.testing.expectEqual(DataKind.rodata, linked.data[0].kind);
    try std.testing.expectEqual(@as(usize, 0), linked.data[0].off);
    try std.testing.expectEqual(@as(usize, 1), linked.data[0].size);

    try std.testing.expectEqual(@as(usize, 2), linked.relocs.len);
    try std.testing.expectEqual(isel.RelocKind.pcrel_hi20, linked.relocs[0].kind);
    try std.testing.expectEqualStrings("g", linked.relocs[0].symbol);
    try std.testing.expectEqual(isel.RelocKind.pcrel_lo12, linked.relocs[1].kind);
    // The lo12's `.pair` names the hi20's (globalized) word offset.
    try std.testing.expectEqual(linked.relocs[0].offset, linked.relocs[1].pair);
    // CRITICAL: the carried lo12's `.symbol` MUST be populated with the paired
    // hi20's data-symbol name (isel emits it empty). native.zig's generic reloc
    // loop resolves every carried reloc's target via `map.get(r.symbol)` and errors
    // on a miss, so an empty lo12 symbol would fail `jitModuleData` for ANY
    // `global_addr` on a real riscv64 host before `applyGlobalReloc` ever runs. This
    // asserts the fix on any host (incl. this aarch64 one, where execution skips).
    try std.testing.expectEqualStrings("g", linked.relocs[1].symbol);
}
