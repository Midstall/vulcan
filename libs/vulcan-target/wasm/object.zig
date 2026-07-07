//! Wasm object: wraps a compiled Wasm module with section metadata.
//! Unlike native targets that produce ELF/PE objects, Wasm IS the object format.
//! This module provides a thin wrapper for accessing module sections.

const std = @import("std");
const ir = @import("vulcan-ir");
const link = @import("link.zig");
const isel = @import("isel.zig");

const Function = ir.function.Function;

pub const Error = link.Error;

/// A Wasm relocatable object: in Wasm's case, this is just the module bytes
/// with metadata about functions and exports.
pub const Object = struct {
    /// The Wasm module bytes.
    module: []u8,
    /// Function symbols with their indices in the Wasm function table.
    symbols: []const link.Symbol,
    /// Number of imports (functions defined outside this module).
    import_count: u32,

    pub fn deinit(self: *Object, allocator: std.mem.Allocator) void {
        allocator.free(self.module);
        allocator.free(self.symbols);
    }

    /// Find the function index for a given symbol name.
    pub fn funcIndex(self: *const Object, name: []const u8) ?u32 {
        for (self.symbols) |s| {
            if (std.mem.eql(u8, s.name, name)) return s.index;
        }
        return null;
    }
};

/// Write a linked module to a Wasm object file.
pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error!Object {
    const linked = try link.compileModule(allocator, module);
    defer allocator.free(linked.symbols);
    return .{
        .module = linked.module,
        .symbols = try allocator.dupe(link.Symbol, linked.symbols),
        .import_count = linked.import_count,
    };
}

/// Compile a single function and return its Wasm bytecode.
pub fn compileFunction(allocator: std.mem.Allocator, func: *const Function) isel.Error!isel.Compiled {
    return isel.selectFunction(allocator, func, null);
}

test "object: compiles a simple add function to Wasm" {
    const allocator = std.testing.allocator;

    // Build IR: fn add(i32, i32) -> i32 returning a + b.
    var func = Function.init(allocator);
    defer func.deinit();

    const t_i32 = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    const b0 = try func.appendBlock();
    const a = try func.appendBlockParam(b0, t_i32);
    const b = try func.appendBlockParam(b0, t_i32);

    const sum = try func.appendInst(b0, t_i32, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(b0, .{ .ret = sum });

    // Compile to Wasm.
    var m = link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("add", &func);

    var obj = try writeModule(allocator, &m);
    defer obj.deinit(allocator);

    // Verify Wasm magic header.
    try std.testing.expectEqualSlices(u8, "\x00asm\x01\x00\x00\x00", obj.module[0..8]);

    // Verify the function is exported.
    try std.testing.expectEqual(@as(u32, 0), obj.funcIndex("add").?);
    try std.testing.expectEqual(@as(u32, 0), obj.import_count);
}

test "object: compiles a function with no params and no result" {
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();

    const b0 = try func.appendBlock();
    func.setTerminator(b0, .{ .ret = null });

    var m = link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("void_fn", &func);

    var obj = try writeModule(allocator, &m);
    defer obj.deinit(allocator);

    try std.testing.expectEqualSlices(u8, "\x00asm\x01\x00\x00\x00", obj.module[0..8]);
}

test "object: compiles a call_indirect module (table + element sections)" {
    const allocator = std.testing.allocator;

    // double(x)=x*2 and a dispatch(sel, x)=table[sel](x) that calls it indirectly.
    var f_double = Function.init(allocator);
    defer f_double.deinit();
    {
        const t = try f_double.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try f_double.appendBlock();
        const x = try f_double.appendBlockParam(b, t);
        const r = try f_double.appendInst(b, t, .{ .arith_imm = .{ .op = .mul, .lhs = x, .imm = 2 } });
        f_double.setTerminator(b, .{ .ret = r });
    }

    var f_disp = Function.init(allocator);
    defer f_disp.deinit();
    {
        const t = try f_disp.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try f_disp.appendBlock();
        const sel = try f_disp.appendBlockParam(b, t);
        const x = try f_disp.appendBlockParam(b, t);
        const r = try f_disp.appendCallIndirect(b, t, sel, &.{x});
        f_disp.setTerminator(b, .{ .ret = r });
    }

    var m = link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("double", &f_double);
    try m.addFunction("dispatch", &f_disp);

    var obj = try writeModule(allocator, &m);
    defer obj.deinit(allocator);

    try std.testing.expectEqualSlices(u8, "\x00asm\x01\x00\x00\x00", obj.module[0..8]);
    try std.testing.expectEqual(@as(u32, 1), obj.funcIndex("dispatch").?);

    // A funcref table (section id 4) and an element segment (section id 9) must be
    // present for the indirect call to resolve at runtime.
    try std.testing.expect(hasSection(obj.module, 0x04));
    try std.testing.expect(hasSection(obj.module, 0x09));
}

/// Walk the module's section headers looking for `id`. Sections are id then a
/// LEB length then that many content bytes, so this skips content correctly rather
/// than scanning for a raw byte.
fn hasSection(module: []const u8, id: u8) bool {
    var pos: usize = 8; // past magic + version
    while (pos < module.len) {
        const sec_id = module[pos];
        pos += 1;
        var len: u32 = 0;
        var shift: u5 = 0;
        while (pos < module.len) {
            const b = module[pos];
            pos += 1;
            len |= @as(u32, b & 0x7F) << shift;
            if (b & 0x80 == 0) break;
            shift += 7;
        }
        if (sec_id == id) return true;
        pos += len;
    }
    return false;
}

test "object: compiles two functions sharing a signature" {
    const allocator = std.testing.allocator;

    // First function: fn add(i32, i32) -> i32
    var func1 = Function.init(allocator);
    defer func1.deinit();
    const t_i32 = try func1.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b1 = try func1.appendBlock();
    const a1 = try func1.appendBlockParam(b1, t_i32);
    const b1_val = try func1.appendBlockParam(b1, t_i32);
    const s1 = try func1.appendInst(b1, t_i32, .{ .arith = .{ .op = .add, .lhs = a1, .rhs = b1_val } });
    func1.setTerminator(b1, .{ .ret = s1 });

    // Second function: fn sub(i32, i32) -> i32 (same signature)
    var func2 = Function.init(allocator);
    defer func2.deinit();
    const t_i32_2 = try func2.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b2 = try func2.appendBlock();
    const a2 = try func2.appendBlockParam(b2, t_i32_2);
    const b2_val = try func2.appendBlockParam(b2, t_i32_2);
    const s2 = try func2.appendInst(b2, t_i32_2, .{ .arith = .{ .op = .sub, .lhs = a2, .rhs = b2_val } });
    func2.setTerminator(b2, .{ .ret = s2 });

    var m = link.Module.init(allocator);
    defer m.deinit();
    try m.addFunction("add", &func1);
    try m.addFunction("sub", &func2);

    var obj = try writeModule(allocator, &m);
    defer obj.deinit(allocator);

    // Should have two exports.
    try std.testing.expectEqual(@as(u32, 0), obj.funcIndex("add").?);
    try std.testing.expectEqual(@as(u32, 1), obj.funcIndex("sub").?);
}
