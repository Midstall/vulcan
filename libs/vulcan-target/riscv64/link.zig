//! Module linking for RISC-V: compile a set of named functions, lay them out
//! sequentially, and resolve each intra-module call to a real `jal` offset.
//! Calls to symbols outside the module stay as relocations for a later linker.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const isel = @import("isel.zig");

const Function = ir.function.Function;

pub const Error = isel.Error;

/// A named function gathered for compilation. The module borrows the function.
const Entry = struct { name: []const u8, func: *const Function };

/// Which section a data global lands in: read-only, writable, or zero-init.
pub const DataKind = enum { rodata, data, bss };

/// A named data global placed in the linked output. For `.bss`, `bytes` is empty
/// and `size` gives the zero-initialized length. Otherwise `size == bytes.len`.
/// The module borrows the bytes.
pub const Data = struct { name: []const u8, bytes: []const u8, kind: DataKind, size: u64 };

/// A grouping of named functions (and data) compiled and linked together.
pub const Module = struct {
    entries: std.ArrayList(Entry) = .empty,
    data: std.ArrayList(Data) = .empty,

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

    /// Add a named writable data global (into `.data`).
    pub fn addWritable(self: *Module, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = bytes, .kind = .data, .size = bytes.len });
    }

    /// Add a named zero-initialized data global of `size` bytes (into `.bss`).
    pub fn addBss(self: *Module, allocator: std.mem.Allocator, name: []const u8, size: u64) std.mem.Allocator.Error!void {
        try self.data.append(allocator, .{ .name = name, .bytes = &.{}, .kind = .bss, .size = size });
    }
};

/// A symbol's resolved location: the word index where its function begins.
pub const Symbol = struct { name: []const u8, offset: usize };

/// A linked module: concatenated code, a symbol table, and any still-unresolved
/// (external) relocations, with offsets in module-global word indices.
pub const Linked = struct {
    code: []u32,
    symbols: []Symbol,
    relocs: []isel.Reloc,

    pub fn deinit(self: *Linked, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.symbols);
        allocator.free(self.relocs);
    }

    pub fn symbolOffset(self: *const Linked, name: []const u8) ?usize {
        for (self.symbols) |s| {
            if (std.mem.eql(u8, s.name, name)) return s.offset;
        }
        return null;
    }
};

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
            try all_relocs.append(allocator, .{ .offset = start + r.offset, .symbol = r.symbol });
        }
    }

    // Resolve intra-module calls, keeping external ones as relocations.
    var external: std.ArrayList(isel.Reloc) = .empty;
    errdefer external.deinit(allocator);
    for (all_relocs.items) |r| {
        if (findSymbol(symbols.items, r.symbol)) |target| {
            const delta = (@as(i64, @intCast(target)) - @as(i64, @intCast(r.offset))) * 4;
            code.items[r.offset] = encode.jal(.x1, @intCast(delta));
        } else {
            try external.append(allocator, r);
        }
    }

    return .{
        .code = try code.toOwnedSlice(allocator),
        .symbols = try symbols.toOwnedSlice(allocator),
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
        callee.setTerminator(b, .{ .ret = a });
    }

    // caller: fn(x) -> callee(x).
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(i32_t_kind);
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        caller.setTerminator(b, .{ .ret = r });
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
    caller.setTerminator(b, .{ .ret = r });

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
