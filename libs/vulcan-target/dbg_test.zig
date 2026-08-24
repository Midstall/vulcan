const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const x86_64 = target.x86_64.isel;
const Function = ir.function.Function;

test "debug entry_fixed count" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const p0 = try func.appendBlockParam(b, i32t);
    const p1 = try func.appendBlockParam(b, i32t);
    const p2 = try func.appendBlockParam(b, i32t);
    const p3 = try func.appendBlockParam(b, i32t);
    const quot = try func.appendInst(b, i32t, .{ .arith = .{ .op = .div, .lhs = p0, .rhs = p1 } });
    var acc = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = quot, .rhs = p2 } });
    acc = try func.appendInst(b, i32t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p3 } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(acc) });

    var desc = try x86_64.x86_64RegDescription(allocator, &func);
    defer desc.deinit(allocator);
    std.debug.print("entry_fixed len={d}\n", .{desc.entry_fixed.len});
    for (desc.entry_fixed) |ef| std.debug.print("  value={} class={d} reg={d}\n", .{ ef.value, ef.class, ef.reg });
    return error.SkipZigTest;
}
