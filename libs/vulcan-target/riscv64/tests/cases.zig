//! Shared RISC-V execution test cases, parameterized by a `harness.Backend`. Each
//! runner (river.zig, spike.zig, qemu.zig, native.zig) calls `runAll` with its
//! backend, so every backend validates the same codegen. A backend that is
//! unavailable or incompatible makes the first case skip the whole test.

const std = @import("std");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const link = @import("../link.zig");
const object = @import("../object.zig");
const ld = @import("../ld.zig");
const h = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

/// Run every execution case on `backend`.
pub fn runAll(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    try arithmetic(io, allocator, backend);
    try controlFlow(io, allocator, backend);
    try memory(io, allocator, backend);
    try calls(io, allocator, backend);
    try spilling(io, allocator, backend);
    try optimized(io, allocator, backend);
    try bitcodeAndLto(io, allocator, backend);
    try pgo(io, allocator, backend);
}

fn arithmetic(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // a*b + a
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const e = try f.appendBlock();
        const a = try f.appendBlockParam(e, t);
        const b = try f.appendBlockParam(e, t);
        const p = try f.appendInst(e, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
        const s = try f.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = a } });
        f.setTerminator(e, .{ .ret = s });
        try h.expectRun(io, allocator, &f, &.{ 3, 4 }, 15, backend);
    }
    { // subtraction (negative result)
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const e = try f.appendBlock();
        const a = try f.appendBlockParam(e, t);
        const b = try f.appendBlockParam(e, t);
        const d = try f.appendInst(e, t, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } });
        f.setTerminator(e, .{ .ret = d });
        try h.expectRun(io, allocator, &f, &.{ 3, 10 }, -7, backend);
    }
    { // signed division
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const e = try f.appendBlock();
        const a = try f.appendBlockParam(e, t);
        const b = try f.appendBlockParam(e, t);
        const q = try f.appendInst(e, t, .{ .arith = .{ .op = .div, .lhs = a, .rhs = b } });
        f.setTerminator(e, .{ .ret = q });
        try h.expectRun(io, allocator, &f, &.{ 20, 3 }, 6, backend);
    }
    { // strength-reduced multiply (a * 8 -> a << 3)
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const e = try f.appendBlock();
        const a = try f.appendBlockParam(e, t);
        const eight = try f.appendInst(e, t, .{ .iconst = 8 });
        const p = try f.appendInst(e, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = eight } });
        f.setTerminator(e, .{ .ret = p });
        try h.expectRun(io, allocator, &f, &.{5}, 40, backend);
    }
    { // immediate arithmetic (a + 100)
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const e = try f.appendBlock();
        const a = try f.appendBlockParam(e, t);
        const r = try f.appendArithImm(e, t, .add, a, 100);
        f.setTerminator(e, .{ .ret = r });
        try h.expectRun(io, allocator, &f, &.{5}, 105, backend);
    }
    { // comparison returns a boolean
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const bool_t = try f.types.intern(.bool);
        const e = try f.appendBlock();
        const a = try f.appendBlockParam(e, t);
        const b = try f.appendBlockParam(e, t);
        const c = try f.appendInst(e, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
        f.setTerminator(e, .{ .ret = c });
        try h.expectRun(io, allocator, &f, &.{ 3, 7 }, 1, backend);
    }
}

fn controlFlow(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // select picks the smaller operand
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const bool_t = try f.types.intern(.bool);
        const e = try f.appendBlock();
        const a = try f.appendBlockParam(e, t);
        const b = try f.appendBlockParam(e, t);
        const c = try f.appendInst(e, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
        const m = try f.appendInst(e, t, .{ .select = .{ .cond = c, .then = a, .@"else" = b } });
        f.setTerminator(e, .{ .ret = m });
        try h.expectRun(io, allocator, &f, &.{ 7, 3 }, 3, backend);
    }
    { // if computes a max via a merge block
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const bool_t = try f.types.intern(.bool);
        const e = try f.appendBlock();
        const a = try f.appendBlockParam(e, t);
        const b = try f.appendBlockParam(e, t);
        const exit = try f.appendBlock();
        const r = try f.appendBlockParam(exit, t);
        const c = try f.appendInst(e, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
        try f.appendIf(e, c, .{ .target = exit, .args = &.{a} }, .{ .target = exit, .args = &.{b} });
        f.setTerminator(exit, .{ .ret = r });
        try h.expectRun(io, allocator, &f, &.{ 4, 9 }, 9, backend);
    }
    { // counted loop summing 0..n
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const bool_t = try f.types.intern(.bool);
        const entry = try f.appendBlock();
        const loop = try f.appendBlock();
        const body = try f.appendBlock();
        const done = try f.appendBlock();
        const n = try f.appendBlockParam(entry, t);
        const i = try f.appendBlockParam(loop, t);
        const acc = try f.appendBlockParam(loop, t);
        const bi = try f.appendBlockParam(body, t);
        const bacc = try f.appendBlockParam(body, t);
        const racc = try f.appendBlockParam(done, t);
        const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
        try f.setJump(entry, loop, &.{ zero, zero });
        const cmp = try f.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
        try f.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
        const ni = try f.appendArithImm(body, t, .add, bi, 1);
        const nacc = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
        try f.setJump(body, loop, &.{ ni, nacc });
        f.setTerminator(done, .{ .ret = racc });
        try h.expectRun(io, allocator, &f, &.{5}, 10, backend);
    }
}

fn memory(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // alloca store/reload
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const ptr_t = try f.types.intern(.ptr);
        const e = try f.appendBlock();
        const x = try f.appendBlockParam(e, t);
        const slot = try f.appendInst(e, ptr_t, .{ .alloca = .{ .elem = t } });
        try f.appendStore(e, x, slot);
        const v = try f.appendInst(e, t, .{ .load = .{ .ptr = slot } });
        f.setTerminator(e, .{ .ret = v });
        try h.expectRun(io, allocator, &f, &.{42}, 42, backend);
    }
    { // sub-word store + sign-extending load (i8): 200 -> 0xC8 -> -56
        var f = Function.init(allocator);
        defer f.deinit();
        const i8_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
        const ptr_t = try f.types.intern(.ptr);
        const e = try f.appendBlock();
        const a = try f.appendBlockParam(e, i8_t);
        const slot = try f.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i8_t } });
        try f.appendStore(e, a, slot);
        const v = try f.appendInst(e, i8_t, .{ .load = .{ .ptr = slot } });
        f.setTerminator(e, .{ .ret = v });
        try h.expectRun(io, allocator, &f, &.{200}, -56, backend);
    }
}

fn calls(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // ten-argument call (stack arguments)
        var callee = Function.init(allocator);
        defer callee.deinit();
        const t = try callee.types.intern(i32k);
        const cb = try callee.appendBlock();
        var ps: [10]Value = undefined;
        for (&ps) |*p| p.* = try callee.appendBlockParam(cb, t);
        var sum = ps[0];
        for (ps[1..]) |p| sum = try callee.appendInst(cb, t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = p } });
        callee.setTerminator(cb, .{ .ret = sum });

        var caller = Function.init(allocator);
        defer caller.deinit();
        const ct = try caller.types.intern(i32k);
        const cb2 = try caller.appendBlock();
        var args: [10]Value = undefined;
        for (&args, 0..) |*a, i| a.* = try caller.appendInst(cb2, ct, .{ .iconst = @intCast(i + 1) });
        const r = try caller.appendCall(cb2, ct, "callee", &args);
        caller.setTerminator(cb2, .{ .ret = r });

        var module: link.Module = .{};
        defer module.deinit(allocator);
        try module.addFunction(allocator, "caller", &caller);
        try module.addFunction(allocator, "callee", &callee);
        try std.testing.expectEqual(@as(i64, 55), try h.runModule(io, allocator, &module, &.{}, backend));
    }
    { // swapped-argument call (parallel move) + result+1 through the linker
        var callee = Function.init(allocator);
        defer callee.deinit();
        const t = try callee.types.intern(i32k);
        const cb = try callee.appendBlock();
        const a = try callee.appendBlockParam(cb, t);
        const bb = try callee.appendBlockParam(cb, t);
        const d = try callee.appendInst(cb, t, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = bb } });
        callee.setTerminator(cb, .{ .ret = d });

        var caller = Function.init(allocator);
        defer caller.deinit();
        const ct = try caller.types.intern(i32k);
        const cb2 = try caller.appendBlock();
        const x = try caller.appendBlockParam(cb2, ct);
        const y = try caller.appendBlockParam(cb2, ct);
        const r = try caller.appendCall(cb2, ct, "callee", &.{ y, x });
        caller.setTerminator(cb2, .{ .ret = r });

        var module: link.Module = .{};
        defer module.deinit(allocator);
        try module.addFunction(allocator, "caller", &caller);
        try module.addFunction(allocator, "callee", &callee);
        try std.testing.expectEqual(@as(i64, 7), try h.runModule(io, allocator, &module, &.{ 3, 10 }, backend));
    }
}

fn spilling(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(i32k);
    const e = try f.appendBlock();
    const p0 = try f.appendBlockParam(e, t);
    const p1 = try f.appendBlockParam(e, t);
    var vals: [20]Value = undefined;
    for (&vals) |*v| v.* = try f.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = p0, .rhs = p1 } });
    var acc = vals[0];
    for (vals[1..]) |v| acc = try f.appendInst(e, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    f.setTerminator(e, .{ .ret = acc });
    try h.expectRun(io, allocator, &f, &.{ 1, 1 }, 40, backend);
}

fn optimized(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // constant folding + DCE: (2+3)*4 + x, with dead x*x
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const c2 = try f.appendInst(b, t, .{ .iconst = 2 });
        const c3 = try f.appendInst(b, t, .{ .iconst = 3 });
        const c4 = try f.appendInst(b, t, .{ .iconst = 4 });
        const a = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c2, .rhs = c3 } });
        const prod = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c4 } });
        _ = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = x } });
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
        f.setTerminator(b, .{ .ret = r });
        try std.testing.expect(try opt.optimize(allocator, &f));
        try h.expectRun(io, allocator, &f, &.{22}, 42, backend);
    }
    { // GVN reuses a dominating subexpression across blocks
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const bool_t = try f.types.intern(.bool);
        const entry = try f.appendBlock();
        const x = try f.appendBlockParam(entry, t);
        const y = try f.appendBlockParam(entry, t);
        const then_b = try f.appendBlock();
        const else_b = try f.appendBlock();
        const base = try f.appendInst(entry, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
        const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
        const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = zero } });
        try f.appendIf(entry, cond, .{ .target = then_b }, .{ .target = else_b });
        const d = try f.appendInst(then_b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
        const r = try f.appendInst(then_b, t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = d } });
        f.setTerminator(then_b, .{ .ret = r });
        f.setTerminator(else_b, .{ .ret = base });
        try std.testing.expect(try opt.optimize(allocator, &f));
        try h.expectRun(io, allocator, &f, &.{ 3, 4 }, 24, backend);
    }
    { // LICM hoists a loop-invariant product
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const bool_t = try f.types.intern(.bool);
        const entry = try f.appendBlock();
        const loop = try f.appendBlock();
        const body = try f.appendBlock();
        const done = try f.appendBlock();
        const x = try f.appendBlockParam(entry, t);
        const y = try f.appendBlockParam(entry, t);
        const n = try f.appendBlockParam(entry, t);
        const i = try f.appendBlockParam(loop, t);
        const acc = try f.appendBlockParam(loop, t);
        const bi = try f.appendBlockParam(body, t);
        const bacc = try f.appendBlockParam(body, t);
        const racc = try f.appendBlockParam(done, t);
        const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
        try f.setJump(entry, loop, &.{ zero, zero });
        const cmp = try f.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
        try f.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
        const p = try f.appendInst(body, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
        const nacc = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = p } });
        const ni = try f.appendArithImm(body, t, .add, bi, 1);
        try f.setJump(body, loop, &.{ ni, nacc });
        f.setTerminator(done, .{ .ret = racc });
        try std.testing.expect(try opt.optimize(allocator, &f));
        try h.expectRun(io, allocator, &f, &.{ 2, 3, 4 }, 24, backend);
    }
}

fn bitcodeAndLto(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // a function round-tripped through bitcode runs identically
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const bool_t = try f.types.intern(.bool);
        const entry = try f.appendBlock();
        const merge = try f.appendBlock();
        const a = try f.appendBlockParam(entry, t);
        const b = try f.appendBlockParam(entry, t);
        const z = try f.appendBlockParam(merge, t);
        const c = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
        const prod = try f.appendInst(entry, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
        try f.appendIf(entry, c, .{ .target = merge, .args = &.{prod} }, .{ .target = merge, .args = &.{b} });
        const r = try f.appendArithImm(merge, t, .add, z, 1);
        f.setTerminator(merge, .{ .ret = r });
        const bytes = try ir.bitcode.encode(allocator, &f);
        defer allocator.free(bytes);
        var decoded = try ir.bitcode.decode(allocator, bytes);
        defer decoded.deinit();
        try h.expectRun(io, allocator, &decoded, &.{ 3, 4 }, 13, backend);
    }
    { // LTO across two bitcode units: inline + prune, then run
        var src = opt.lto.Module.init(allocator);
        {
            var f = Function.init(allocator);
            const t = try f.types.intern(i32k);
            const b = try f.appendBlock();
            const a = try f.appendBlockParam(b, t);
            const bb = try f.appendBlockParam(b, t);
            const prod = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bb } });
            const sum = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = a } });
            f.setTerminator(b, .{ .ret = sum });
            try src.add("helper", f);
            var g = Function.init(allocator);
            const gt = try g.types.intern(i32k);
            const gb = try g.appendBlock();
            const x = try g.appendBlockParam(gb, gt);
            const call = try g.appendCall(gb, gt, "helper", &.{ x, x });
            const r = try g.appendArithImm(gb, gt, .add, call, 1);
            g.setTerminator(gb, .{ .ret = r });
            try src.add("entry", g);
        }
        const blob = try opt.lto.encode(allocator, &src);
        src.deinit();
        defer allocator.free(blob);
        var module = try opt.lto.decode(allocator, blob);
        defer module.deinit();
        _ = try opt.lto.link(allocator, &module, &.{"entry"});
        try std.testing.expectEqual(@as(usize, 1), module.count());
        try h.expectRun(io, allocator, module.get("entry").?, &.{4}, 21, backend);
    }
}

fn pgo(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // instrumentation is transparent: f(x)=x*x with .bss counters still gives 36
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(i32k);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = x } });
        f.setTerminator(b, .{ .ret = r });
        const nblocks = try opt.pgo.instrument(allocator, &f, "pgo_counters");

        var module: link.Module = .{};
        defer module.deinit(allocator);
        try module.addFunction(allocator, "f", &f);
        try module.addBss(allocator, "pgo_counters", nblocks * 8);
        const obj = try object.writeModule(allocator, &module);
        defer allocator.free(obj);
        var image = try ld.linkObjects(allocator, &.{obj}, h.load_address);
        defer image.deinit(allocator);
        const words = try allocator.alloc(u32, image.code.len / 4);
        defer allocator.free(words);
        for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, image.code[i * 4 ..][0..4], .little);
        try std.testing.expectEqual(@as(i64, 36), try h.runCode(io, allocator, words, &.{6}, backend));
    }
    { // profile-guided inlining of a hot call, then run
        var module = opt.lto.Module.init(allocator);
        defer module.deinit();
        {
            var f = Function.init(allocator);
            const t = try f.types.intern(i32k);
            const b = try f.appendBlock();
            const a = try f.appendBlockParam(b, t);
            const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
            f.setTerminator(b, .{ .ret = s });
            try module.add("helper", f);
            var g = Function.init(allocator);
            const gt = try g.types.intern(i32k);
            const gb = try g.appendBlock();
            const x = try g.appendBlockParam(gb, gt);
            const call = try g.appendCall(gb, gt, "helper", &.{x});
            g.setTerminator(gb, .{ .ret = call });
            try module.add("caller", g);
        }
        var profile = opt.pgo.Profile.init(allocator);
        defer profile.deinit();
        try profile.add("caller", &.{1000});
        try std.testing.expect(try opt.pgo.guidedInline(allocator, &module, &profile, 100));
        try h.expectRun(io, allocator, module.get("caller").?, &.{21}, 42, backend);
    }
}
