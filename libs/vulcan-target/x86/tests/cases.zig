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
    try memory(io, allocator, backend);
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

fn memory(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // x86-32 stores then loads a 32-bit value through a pointer: alloca an i32, store a
        // constant, load it back, return it. A small sweep of constants (positive, negative,
        // and one with every byte distinct) exercises the same [reg+0] 32-bit load/store path.
        const consts = [_]i64{ 0x5A, -1, 0x12345678 };
        inline for (consts) |c| {
            var f = Function.init(allocator);
            defer f.deinit();
            const t = try h.i32type(&f);
            const ptr_t = try f.types.intern(.ptr);
            const b = try f.appendBlock();
            const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
            const v = try f.appendInst(b, t, .{ .iconst = c });
            try f.appendStore(b, v, slot);
            const r = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
            f.setTerminator(b, .{ .ret = r });
            try expectRun(io, allocator, &f, &.{}, c, backend);
        }
    }
    // x86-32 loads and stores 8-bit and 16-bit values (zero and sign extended). Both a signed
    // and an unsigned alloca read back the same top-bit-set byte pattern. A shift right by
    // exactly the extension width (32 - value width) moves the fill bits (all 1s for a sign
    // extension, all 0s for a zero extension) down into the observable low byte, so the two
    // extension kinds are distinguishable even though the process exit code only reports one
    // byte. -1 means "all 1s survived" (sign-extended), 0 means "zero-filled".
    try memorySubWord(io, allocator, backend);
    { // x86-32 copies consecutive words: alloca two i32 slots, store the argument into the
        // first, copy it into the second via a load+store, return the second. This is the
        // adjacent-slot shape a later folding pass will fuse.
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const ptr_t = try f.types.intern(.ptr);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const slot0 = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
        const slot1 = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
        try f.appendStore(b, x, slot0);
        const v = try f.appendInst(b, t, .{ .load = .{ .ptr = slot0 } });
        try f.appendStore(b, v, slot1);
        const r = try f.appendInst(b, t, .{ .load = .{ .ptr = slot1 } });
        f.setTerminator(b, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{77}, 77, backend);
    }
}

fn memorySubWord(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    const ptr_kind: ir.types.TypeKind = .ptr;
    { // signed i8 load: 0x81 sign-extends, so shr(v, 24) leaves the extension fill (-1).
        var f = Function.init(allocator);
        defer f.deinit();
        const i8_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
        const ptr_t = try f.types.intern(ptr_kind);
        const b = try f.appendBlock();
        const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i8_t } });
        const c = try f.appendInst(b, i8_t, .{ .iconst = -127 }); // 0x81
        try f.appendStore(b, c, slot);
        const v = try f.appendInst(b, i8_t, .{ .load = .{ .ptr = slot } });
        const r = try f.appendArithImm(b, i8_t, .shr, v, 24);
        f.setTerminator(b, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{}, -1, backend);
    }
    { // unsigned u8 load: the same 0x81 pattern zero-extends, so shr(v, 24) leaves 0.
        var f = Function.init(allocator);
        defer f.deinit();
        const u8_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
        const ptr_t = try f.types.intern(ptr_kind);
        const b = try f.appendBlock();
        const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = u8_t } });
        const c = try f.appendInst(b, u8_t, .{ .iconst = 0x81 });
        try f.appendStore(b, c, slot);
        const v = try f.appendInst(b, u8_t, .{ .load = .{ .ptr = slot } });
        const r = try f.appendArithImm(b, u8_t, .shr, v, 24);
        f.setTerminator(b, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{}, 0, backend);
    }
    { // signed i16 load: 0x8001 sign-extends, so shr(v, 16) leaves the extension fill (-1).
        var f = Function.init(allocator);
        defer f.deinit();
        const i16_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 16 } });
        const ptr_t = try f.types.intern(ptr_kind);
        const b = try f.appendBlock();
        const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = i16_t } });
        const c = try f.appendInst(b, i16_t, .{ .iconst = -32767 }); // 0x8001
        try f.appendStore(b, c, slot);
        const v = try f.appendInst(b, i16_t, .{ .load = .{ .ptr = slot } });
        const r = try f.appendArithImm(b, i16_t, .shr, v, 16);
        f.setTerminator(b, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{}, -1, backend);
    }
    { // unsigned u16 load: the same 0x8001 pattern zero-extends, so shr(v, 16) leaves 0.
        var f = Function.init(allocator);
        defer f.deinit();
        const u16_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
        const ptr_t = try f.types.intern(ptr_kind);
        const b = try f.appendBlock();
        const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = u16_t } });
        const c = try f.appendInst(b, u16_t, .{ .iconst = 0x8001 });
        try f.appendStore(b, c, slot);
        const v = try f.appendInst(b, u16_t, .{ .load = .{ .ptr = slot } });
        const r = try f.appendArithImm(b, u16_t, .shr, v, 16);
        f.setTerminator(b, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{}, 0, backend);
    }
}
