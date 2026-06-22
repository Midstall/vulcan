//! In-memory linker for i386. Compiles named functions, lays them out, and resolves each
//! `call` relocation (a rel32 displacement) to the target symbol's offset. Mirrors the
//! x86-64 linker. object.zig emits the same data as an ELF32 .o.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("isel.zig");

const Function = ir.function.Function;

pub const Error = isel.Error || error{UndefinedSymbol};

const Entry = struct { name: []const u8, func: *const Function };

pub const Module = struct {
    funcs: std.ArrayListUnmanaged(Entry) = .empty,

    pub fn addFunction(self: *Module, allocator: std.mem.Allocator, name: []const u8, func: *const Function) std.mem.Allocator.Error!void {
        try self.funcs.append(allocator, .{ .name = name, .func = func });
    }
    pub fn deinit(self: *Module, allocator: std.mem.Allocator) void {
        self.funcs.deinit(allocator);
    }
};

pub const Symbol = struct { name: []const u8, offset: usize };

pub const Linked = struct {
    code: []u8,
    symbols: []Symbol,

    pub fn deinit(self: *Linked, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.symbols);
    }
    pub fn addressOf(self: *const Linked, name: []const u8) ?usize {
        for (self.symbols) |s| if (std.mem.eql(u8, s.name, name)) return s.offset;
        return null;
    }
};

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
    @memset(code, 0x90);
    for (compiled, 0..) |c, i| @memcpy(code[offsets[i]..][0..c.code.len], c.code);

    for (compiled, 0..) |c, i| {
        for (c.relocs) |r| {
            const site = offsets[i] + r.offset;
            const target = addressBySymbol(symbols, r.symbol) orelse return error.UndefinedSymbol;
            const rel: i32 = @intCast(@as(i64, @intCast(target)) - @as(i64, @intCast(site + 4)));
            std.mem.writeInt(u32, code[site..][0..4], @bitCast(rel), .little);
        }
    }
    return .{ .code = code, .symbols = symbols };
}

fn addressBySymbol(symbols: []const Symbol, name: []const u8) ?usize {
    for (symbols) |s| if (std.mem.eql(u8, s.name, name)) return s.offset;
    return null;
}

test "links two i386 functions and resolves the call" {
    const allocator = std.testing.allocator;
    var helper = Function.init(allocator);
    defer helper.deinit();
    {
        const t = try helper.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        const r = try helper.appendArithImm(b, t, .mul, x, 2);
        helper.setTerminator(b, .{ .ret = r });
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
    var module: Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "helper", &helper);
    var linked = try compileModule(allocator, &module);
    defer linked.deinit(allocator);
    try std.testing.expect(linked.addressOf("helper") != null);
}
