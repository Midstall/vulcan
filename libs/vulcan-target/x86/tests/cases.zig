//! Shared i386 execution test cases, parameterized by a `harness.Backend`. The runner
//! files (qemu.zig, native.zig) call `runAll` with their backend.

const std = @import("std");
const ir = @import("vulcan-ir");
const h = @import("harness.zig");

const Function = ir.function.Function;
const expectRun = h.expectRun;

pub fn runAll(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    try arithmetic(io, allocator, backend);
    try immediates(io, allocator, backend);
    try controlFlow(io, allocator, backend);
    try spilling(io, allocator, backend);
    try calls(io, allocator, backend);
}

fn arithmetic(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // a*b + a
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const y = try f.appendBlockParam(b, t);
        const p = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
        const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = x } });
        f.setTerminator(b, .{ .ret = s });
        try expectRun(io, allocator, &f, &.{ 3, 4 }, 15, backend);
    }
    inline for (.{ .{ .op = .div, .want = 6 }, .{ .op = .rem, .want = 2 } }) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const y = try f.appendBlockParam(b, t);
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = c.op, .lhs = x, .rhs = y } });
        f.setTerminator(b, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{ 20, 3 }, c.want, backend);
    }
    { // shift by a register count: x << k
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const k = try f.appendBlockParam(b, t);
        const s = try f.appendInst(b, t, .{ .arith = .{ .op = .shl, .lhs = x, .rhs = k } });
        f.setTerminator(b, .{ .ret = s });
        try expectRun(io, allocator, &f, &.{ 3, 4 }, 48, backend);
    }
    { // 3 args with division: (a + b) / c. The EAX/EDX reservation leaves only a
        // 2-register pool, so the third parameter must spill.
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, t);
        const bb = try f.appendBlockParam(b, t);
        const c = try f.appendBlockParam(b, t);
        const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bb } });
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = .div, .lhs = s, .rhs = c } });
        f.setTerminator(b, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{ 20, 1, 3 }, 7, backend); // 21 / 3
    }
}

fn immediates(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    const cases = .{
        .{ .op = .add, .imm = 7, .want = 17 },
        .{ .op = .mul, .imm = 6, .want = 60 },
        .{ .op = .shl, .imm = 4, .want = 160 },
        .{ .op = .div, .imm = 3, .want = 3 },
    };
    inline for (cases) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const r = try f.appendArithImm(b, t, c.op, x, c.imm);
        f.setTerminator(b, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{10}, c.want, backend);
    }
}

fn controlFlow(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // max(a, b)
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const bool_t = try f.types.intern(.bool);
        const entry = try f.appendBlock();
        const a = try f.appendBlockParam(entry, t);
        const b = try f.appendBlockParam(entry, t);
        const merge = try f.appendBlock();
        const r = try f.appendBlockParam(merge, t);
        const c = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
        try f.appendIf(entry, c, .{ .target = merge, .args = &.{a} }, .{ .target = merge, .args = &.{b} });
        f.setTerminator(merge, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{ 3, 4 }, 4, backend);
        try expectRun(io, allocator, &f, &.{ 7, 2 }, 7, backend);
    }
    { // sum 1..n loop
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const bool_t = try f.types.intern(.bool);
        const entry = try f.appendBlock();
        const n = try f.appendBlockParam(entry, t);
        const header = try f.appendBlock();
        const sum = try f.appendBlockParam(header, t);
        const i = try f.appendBlockParam(header, t);
        const body = try f.appendBlock();
        const exit = try f.appendBlock();
        const r = try f.appendBlockParam(exit, t);
        const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
        const one = try f.appendInst(entry, t, .{ .iconst = 1 });
        try f.setJump(entry, header, &.{ zero, one });
        const cont = try f.appendInst(header, bool_t, .{ .icmp = .{ .op = .le, .lhs = i, .rhs = n } });
        try f.appendIf(header, cont, .{ .target = body, .args = &.{} }, .{ .target = exit, .args = &.{sum} });
        const sum2 = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = i } });
        const ob = try f.appendInst(body, t, .{ .iconst = 1 });
        const inext = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = i, .rhs = ob } });
        try f.setJump(body, header, &.{ sum2, inext });
        f.setTerminator(exit, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{5}, 15, backend);
        try expectRun(io, allocator, &f, &.{10}, 55, backend);
    }
}

fn spilling(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // 6 temporaries live until summed -> forces spilling (the x86-32 pool is small).
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try h.i32type(&f);
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, t);
    var temps: [6]ir.function.Value = undefined;
    for (0..6) |k| temps[k] = try f.appendArithImm(b, t, .add, a, @intCast(k + 1));
    var acc = temps[0];
    for (1..6) |k| acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = temps[k] } });
    f.setTerminator(b, .{ .ret = acc });
    try expectRun(io, allocator, &f, &.{10}, 81, backend); // 6a + 21
}

fn calls(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    const link = @import("../link.zig");
    // main(x) = helper(x), helper(x) = x*x + 1 -> linked, the call resolved.
    var helper = Function.init(allocator);
    defer helper.deinit();
    {
        const t = try h.i32type(&helper);
        const b = try helper.appendBlock();
        const x = try helper.appendBlockParam(b, t);
        const d = try helper.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = x } });
        const r = try helper.appendArithImm(b, t, .add, d, 1);
        helper.setTerminator(b, .{ .ret = r });
    }
    var main = Function.init(allocator);
    defer main.deinit();
    {
        const t = try h.i32type(&main);
        const b = try main.appendBlock();
        const x = try main.appendBlockParam(b, t);
        const r = try main.appendCall(b, t, "helper", &.{x});
        main.setTerminator(b, .{ .ret = r });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "helper", &helper);
    try h.expectRunModule(io, allocator, &module, &.{6}, 37, backend); // 6*6 + 1

    // A value live across the call (x used after) survives via a spill slot:
    // main2(x) = helper(x) + x = (x*x + 1) + x. For x=5: 26 + 5 = 31.
    var helper2 = Function.init(allocator);
    defer helper2.deinit();
    {
        const t = try h.i32type(&helper2);
        const b = try helper2.appendBlock();
        const x = try helper2.appendBlockParam(b, t);
        const d = try helper2.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = x } });
        const r = try helper2.appendArithImm(b, t, .add, d, 1);
        helper2.setTerminator(b, .{ .ret = r });
    }
    var main2 = Function.init(allocator);
    defer main2.deinit();
    {
        const t = try h.i32type(&main2);
        const b = try main2.appendBlock();
        const x = try main2.appendBlockParam(b, t);
        const called = try main2.appendCall(b, t, "helper", &.{x});
        const r = try main2.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = x } });
        main2.setTerminator(b, .{ .ret = r });
    }
    var module2: link.Module = .{};
    defer module2.deinit(allocator);
    try module2.addFunction(allocator, "main", &main2);
    try module2.addFunction(allocator, "helper", &helper2);
    try h.expectRunModule(io, allocator, &module2, &.{5}, 31, backend); // (5*5+1) + 5
}
