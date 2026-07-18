//! In-memory module linking for AArch64: compile a set of named functions, lay them
//! out in one code image, and resolve every call (`bl`) relocation to its callee.
//! The image is position-independent (each `bl` is PC-relative), so it maps at any
//! address. Data sections are not handled (the selector emits no `global_addr`).

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");
const encode = @import("encode.zig");

const Function = ir.function.Function;

pub const Error = isel.Error || error{UndefinedSymbol};

/// A set of named functions to link together. The first added function is the
/// entry (it sits at offset 0 of the linked image).
pub const Module = struct {
    functions: std.ArrayListUnmanaged(Entry) = .empty,

    pub const Entry = struct { name: []const u8, func: *const Function };

    pub fn addFunction(self: *Module, allocator: std.mem.Allocator, name: []const u8, func: *const Function) std.mem.Allocator.Error!void {
        try self.functions.append(allocator, .{ .name = name, .func = func });
    }

    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        self.functions.deinit(allocator);
    }
};

/// A function's byte offset within the linked image.
pub const Symbol = struct { name: []const u8, offset: usize };

/// A linked code image: the machine words plus each function's byte offset.
pub const Linked = struct {
    code: []u32,
    symbols: []Symbol,

    pub fn deinit(self: *Linked, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.symbols);
    }

    /// The byte offset of the function named `name`, or null if absent.
    pub fn addressOf(self: *const Linked, name: []const u8) ?usize {
        for (self.symbols) |s| if (std.mem.eql(u8, s.name, name)) return s.offset;
        return null;
    }
};

/// Compile every function, concatenate the code, and resolve each `bl` relocation
/// to the callee's location. The caller owns the result.
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

    // Resolve each call relocation (a PC-relative `bl`) to its callee's word.
    for (0..funcs.len) |i| {
        for (compiled[i].relocs) |r| {
            const at = word_off[i] + r.offset;
            const target = symbolWord(funcs, word_off, r.symbol) orelse return error.UndefinedSymbol;
            code[at] = encode.bl(@intCast((@as(i64, @intCast(target)) - @as(i64, @intCast(at))) * 4));
        }
    }

    const symbols = try allocator.alloc(Symbol, funcs.len);
    errdefer allocator.free(symbols);
    for (funcs, 0..) |e, i| symbols[i] = .{ .name = e.name, .offset = word_off[i] * 4 };

    return .{ .code = code, .symbols = symbols };
}

fn symbolWord(funcs: []const Module.Entry, word_off: []const usize, name: []const u8) ?usize {
    for (funcs, 0..) |e, i| if (std.mem.eql(u8, e.name, name)) return word_off[i];
    return null;
}
