//! Shared x86-64 execution test cases, parameterized by a `harness.Backend`. Each case
//! builds an IR function (or a linked module), runs it through the backend, and asserts
//! the result (mod 256). The runner files (qemu.zig, native.zig) call `runAll` with their
//! backend.

const std = @import("std");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const h = @import("harness.zig");
const isel = @import("../isel.zig");

const Function = ir.function.Function;
const expectRun = h.expectRun;

pub fn runAll(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    try arithmetic(io, allocator, backend);
    try immediates(io, allocator, backend);
    try controlFlow(io, allocator, backend);
    try fallthrough(io, allocator, backend);
    try spilling(io, allocator, backend);
    try calls(io, allocator, backend);
    try floats(io, allocator, backend);
    try doubles(io, allocator, backend);
    try vectors(io, allocator, backend);
    try memory(io, allocator, backend);
}

fn doubles(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    const ops = [_]struct { op: ir.function.BinOp, x: f64, y: f64 }{
        .{ .op = .add, .x = 1.1, .y = 2.2 },
        .{ .op = .sub, .x = 5.5, .y = 1.1 },
        .{ .op = .mul, .x = 1.7, .y = 2.3 },
        .{ .op = .div, .x = 10.0, .y = 3.0 },
    };
    for (ops) |c| { // scalar SSE2 double arithmetic
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f64 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const y = try f.appendBlockParam(b, t);
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = c.op, .lhs = x, .rhs = y } });
        f.setTerminator(b, .{ .ret = r });
        const expected: f64 = switch (c.op) {
            .add => c.x + c.y,
            .sub => c.x - c.y,
            .mul => c.x * c.y,
            .div => c.x / c.y,
            else => unreachable,
        };
        try h.expectRunDouble(io, allocator, &f, &.{ c.x, c.y }, expected, backend);
    }
    { // f64 constants: x * 2.0 + 1.0 (movq-materialized 64-bit constants)
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f64 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const two = try f.appendInst(b, t, .{ .fconst = 2.0 });
        const one = try f.appendInst(b, t, .{ .fconst = 1.0 });
        const p = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = two } });
        const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = one } });
        f.setTerminator(b, .{ .ret = s });
        const xv: f64 = 3.3;
        try h.expectRunDouble(io, allocator, &f, &.{xv}, xv * 2.0 + 1.0, backend);
    }
    { // f64 compare + select: min(x, y) = (x < y) ? x : y
        const Case = struct { x: f64, y: f64 };
        for ([_]Case{ .{ .x = 5.5, .y = 3.3 }, .{ .x = 1.1, .y = 7.7 } }) |c| {
            var f = Function.init(allocator);
            defer f.deinit();
            const t = try f.types.intern(.{ .float = .f64 });
            const bool_t = try f.types.intern(.bool);
            const b = try f.appendBlock();
            const x = try f.appendBlockParam(b, t);
            const y = try f.appendBlockParam(b, t);
            const lt = try f.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
            const m = try f.appendInst(b, t, .{ .select = .{ .cond = lt, .then = x, .@"else" = y } });
            f.setTerminator(b, .{ .ret = m });
            try h.expectRunDouble(io, allocator, &f, &.{ c.x, c.y }, @min(c.x, c.y), backend);
        }
    }
    { // f64 conversions: (double)(int)(x * 100) / 100 and an f32 round-trip (double)(float)x
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f64 });
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const narrowed = try f.appendInst(b, f32_t, .{ .convert = .{ .value = x } }); // f64 -> f32
        const widened = try f.appendInst(b, t, .{ .convert = .{ .value = narrowed } }); // f32 -> f64
        const c100 = try f.appendInst(b, t, .{ .fconst = 100.0 });
        const scaled = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = widened, .rhs = c100 } });
        const i = try f.appendInst(b, i32_t, .{ .convert = .{ .value = scaled } }); // f64 -> int
        const back = try f.appendInst(b, t, .{ .convert = .{ .value = i } }); // int -> f64
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = .div, .lhs = back, .rhs = c100 } });
        f.setTerminator(b, .{ .ret = r });
        const xv: f64 = 1.234;
        const rt: f64 = @as(f32, @floatCast(xv)); // (double)(float)x
        const back_e: f64 = @floatFromInt(@as(i32, @intFromFloat(rt * 100.0)));
        try h.expectRunDouble(io, allocator, &f, &.{xv}, back_e / 100.0, backend);
    }
}

fn memory(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    { // an int slot: alloca, store a value, load it back, return it.
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const ptr_t = try f.types.intern(.ptr);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
        try f.appendStore(b, x, slot);
        const r = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
        f.setTerminator(b, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{0x5A}, 0x5A, backend); // round-trips the byte 0x5A
    }
    { // a float-typed integer constant (how the frontend zero-inits float locals):
        // x + (0.0 typed as a float iconst) == x. The const must land in an xmm register,
        // not a gpr.
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f32 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const z = try f.appendInst(b, t, .{ .iconst = 0 });
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = z } });
        f.setTerminator(b, .{ .ret = r });
        try h.expectRunFloat(io, allocator, &f, &.{@as(f32, 3.5)}, 3.5, backend);
    }
    { // a float slot: alloca f32, store x + y, load it back, return it.
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f32 });
        const ptr_t = try f.types.intern(.ptr);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const y = try f.appendBlockParam(b, t);
        const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
        const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
        try f.appendStore(b, s, slot);
        const r = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
        f.setTerminator(b, .{ .ret = r });
        const xv: f32 = 1.7;
        const yv: f32 = 2.3;
        try h.expectRunFloat(io, allocator, &f, &.{ xv, yv }, xv + yv, backend); // f32-precision sum
    }
    { // an f64 slot: alloca f64, store x + y, load it back (movsd, all 8 bytes), return it.
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f64 });
        const ptr_t = try f.types.intern(.ptr);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const y = try f.appendBlockParam(b, t);
        const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
        const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
        try f.appendStore(b, s, slot);
        const r = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
        f.setTerminator(b, .{ .ret = r });
        const xv: f64 = 1.7;
        const yv: f64 = 2.3;
        try h.expectRunDouble(io, allocator, &f, &.{ xv, yv }, xv + yv, backend);
    }
}

fn vectors(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // The vectorizer's output shape: pack two <4 x f32> from scalar params, add with one
    // SSE addps, extract the four lanes, and reduce to a scalar.
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f32 });
    const v4 = try f.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const b = try f.appendBlock();
    var ap: [4]ir.function.Value = undefined;
    var bp: [4]ir.function.Value = undefined;
    for (0..4) |i| ap[i] = try f.appendBlockParam(b, t);
    for (0..4) |i| bp[i] = try f.appendBlockParam(b, t);
    const va = try f.appendInst(b, v4, .{ .struct_new = .{ .fields = try f.internValueList(&ap) } });
    const vb = try f.appendInst(b, v4, .{ .struct_new = .{ .fields = try f.internValueList(&bp) } });
    const vc = try f.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    var c: [4]ir.function.Value = undefined;
    for (0..4) |i| c[i] = try f.appendInst(b, t, .{ .extract = .{ .aggregate = vc, .index = @intCast(i) } });
    const s01 = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c[0], .rhs = c[1] } });
    const s012 = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = c[2] } });
    const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = c[3] } });
    f.setTerminator(b, .{ .ret = s });

    const av = [4]f32{ 1.1, 2.2, 3.3, 4.4 };
    const bv = [4]f32{ 5.5, 6.6, 7.7, 8.8 };
    var cc: [4]f32 = undefined;
    for (0..4) |i| cc[i] = av[i] + bv[i];
    const expected = ((cc[0] + cc[1]) + cc[2]) + cc[3]; // same order as the IR reduction
    try h.expectRunFloat(io, allocator, &f, &.{ av[0], av[1], av[2], av[3], bv[0], bv[1], bv[2], bv[3] }, expected, backend);

    try vectorSpill(io, allocator, backend);
    try widevectors(io, allocator, backend);
    try vectorParamCall(io, allocator, backend);
    try vectorSqrtSelect(io, allocator, backend);
}

fn vectorSqrtSelect(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // Two per-lane SSE paths the graphics quad FS needs: packed sqrt (sqrtps, every lane, not
    // just lane 0) and a per-lane max built from a vector compare + vector select (cmpps mask
    // then and/andn/or). max(sqrt(a), b) reduced over the 4 lanes.
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f32 });
    const v4 = try f.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const b = try f.appendBlock();
    var ap: [4]ir.function.Value = undefined;
    var bp: [4]ir.function.Value = undefined;
    for (0..4) |i| ap[i] = try f.appendBlockParam(b, t);
    for (0..4) |i| bp[i] = try f.appendBlockParam(b, t);
    const va = try f.appendInst(b, v4, .{ .struct_new = .{ .fields = try f.internValueList(&ap) } });
    const vb = try f.appendInst(b, v4, .{ .struct_new = .{ .fields = try f.internValueList(&bp) } });
    const sq = try f.appendInst(b, v4, .{ .unary = .{ .op = .sqrt, .value = va } }); // sqrt(a) per lane
    const mask = try f.appendInst(b, v4, .{ .icmp = .{ .op = .gt, .lhs = sq, .rhs = vb } }); // sqrt(a) > b
    const mx = try f.appendInst(b, v4, .{ .select = .{ .cond = mask, .then = sq, .@"else" = vb } }); // per-lane max
    var c: [4]ir.function.Value = undefined;
    for (0..4) |i| c[i] = try f.appendInst(b, t, .{ .extract = .{ .aggregate = mx, .index = @intCast(i) } });
    const s01 = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c[0], .rhs = c[1] } });
    const s012 = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = c[2] } });
    const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = c[3] } });
    f.setTerminator(b, .{ .ret = s });

    // Non-round lanes so every one flows through sqrtps + the mask select and the reduced sum
    // has a distinctive low byte (the harness compares only the low byte of the f32 result).
    const av = [4]f32{ 2.0, 8.0, 5.0, 3.0 }; // sqrt -> 1.414, 2.828, 2.236, 1.732 (all > b)
    const bv = [4]f32{ 1.0, 1.0, 1.0, 1.0 };
    var expect: f32 = 0;
    for (0..4) |i| expect += @max(@sqrt(av[i]), bv[i]);
    try h.expectRunFloat(io, allocator, &f, &.{ av[0], av[1], av[2], av[3], bv[0], bv[1], bv[2], bv[3] }, expect, backend);
}

fn vectorParamCall(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // A callee receives a <4 x f32> PARAMETER (not one built from scalars inside it): main packs
    // a vector from its scalar args and calls helper(v), whose prologue must accept a vector-typed
    // parameter arriving in an xmm register. Guards the entry-param classification (a vector param
    // lives in xmm, so it must be handled like a float param, not routed to the gpr path).
    const link = @import("../link.zig");
    var fhelper = Function.init(allocator);
    defer fhelper.deinit();
    {
        const t = try fhelper.types.intern(.{ .float = .f32 });
        const v4 = try fhelper.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
        const b = try fhelper.appendBlock();
        const v = try fhelper.appendBlockParam(b, v4);
        var e: [4]ir.function.Value = undefined;
        for (0..4) |i| e[i] = try fhelper.appendInst(b, t, .{ .extract = .{ .aggregate = v, .index = @intCast(i) } });
        const s01 = try fhelper.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = e[0], .rhs = e[1] } });
        const s012 = try fhelper.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = e[2] } });
        const s = try fhelper.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = e[3] } });
        fhelper.setTerminator(b, .{ .ret = s });
    }
    var fmain = Function.init(allocator);
    defer fmain.deinit();
    {
        const t = try fmain.types.intern(.{ .float = .f32 });
        const v4 = try fmain.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
        const b = try fmain.appendBlock();
        var p: [4]ir.function.Value = undefined;
        for (0..4) |i| p[i] = try fmain.appendBlockParam(b, t);
        const v = try fmain.appendInst(b, v4, .{ .struct_new = .{ .fields = try fmain.internValueList(&p) } });
        const called = try fmain.appendCall(b, t, "helper", &.{v});
        fmain.setTerminator(b, .{ .ret = called });
    }
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &fmain);
    try module.addFunction(allocator, "helper", &fhelper);
    // Non-round lane values so the reduced sum has a distinctive low byte (the harness compares
    // only the low byte of the f32 result). A dropped or misread lane changes that byte.
    const pv = [4]f32{ 1.1, 2.2, 3.3, 4.5 };
    try h.expectRunFloatModule(io, allocator, &module, &.{ pv[0], pv[1], pv[2], pv[3] }, ((pv[0] + pv[1]) + pv[2]) + pv[3], backend);
}

fn widevectors(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // AVX <8 x f32>: pack two 8-lane vectors from constants, add with one 256-bit vaddps,
    // extract all eight lanes (the high four via vextractf128) and reduce. Needs AVX, so it
    // runs only under qemu (-cpu max).
    if (backend.qemu_cmd == null) return;
    const ca = [8]f32{ 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8 };
    const da = [8]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f32 });
    const v8 = try f.types.intern(.{ .vector = .{ .len = 8, .elem = t } });
    const b = try f.appendBlock();
    // Build each vector's lanes right before packing it, so at most one set of eight scalars
    // is live at once (whole-vector spill of a 256-bit ymm is not implemented yet).
    var af: [8]ir.function.Value = undefined;
    for (0..8) |i| af[i] = try f.appendInst(b, t, .{ .fconst = ca[i] });
    const va = try f.appendInst(b, v8, .{ .struct_new = .{ .fields = try f.internValueList(&af) } });
    var bf: [8]ir.function.Value = undefined;
    for (0..8) |i| bf[i] = try f.appendInst(b, t, .{ .fconst = da[i] });
    const vb = try f.appendInst(b, v8, .{ .struct_new = .{ .fields = try f.internValueList(&bf) } });
    const vc = try f.appendInst(b, v8, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    var e: [8]ir.function.Value = undefined;
    for (0..8) |i| e[i] = try f.appendInst(b, t, .{ .extract = .{ .aggregate = vc, .index = @intCast(i) } });
    var s = e[0];
    for (1..8) |i| s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = e[i] } });
    f.setTerminator(b, .{ .ret = s });

    var cc: [8]f32 = undefined;
    for (0..8) |i| cc[i] = ca[i] + da[i];
    var expected: f32 = cc[0];
    for (1..8) |i| expected += cc[i]; // same reduction order as the IR
    try h.expectRunFloat(io, allocator, &f, &.{}, expected, backend);

    try wideSpill(io, allocator, backend);
}

fn wideSpill(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // High AVX pressure: 16 live <8 x f32> (> 13 allocatable xmm) force whole-ymm spills to
    // 32-byte slots (256-bit vmovups, no truncation). Each v_i has all 8 lanes = (i + 0.1).
    // Sum them, then extract and reduce all eight lanes.
    if (backend.qemu_cmd == null) return;
    const N = 16;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f32 });
    const v8 = try f.types.intern(.{ .vector = .{ .len = 8, .elem = t } });
    const b = try f.appendBlock();
    var vs: [N]ir.function.Value = undefined;
    for (0..N) |i| {
        const c = try f.appendInst(b, t, .{ .fconst = @as(f64, @floatFromInt(i)) + 0.1 });
        vs[i] = try f.appendInst(b, v8, .{ .struct_new = .{ .fields = try f.internValueList(&.{ c, c, c, c, c, c, c, c }) } });
    }
    var acc = vs[0];
    for (1..N) |i| acc = try f.appendInst(b, v8, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vs[i] } });
    var e: [8]ir.function.Value = undefined;
    for (0..8) |i| e[i] = try f.appendInst(b, t, .{ .extract = .{ .aggregate = acc, .index = @intCast(i) } });
    var s = e[0];
    for (1..8) |i| s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = e[i] } });
    f.setTerminator(b, .{ .ret = s });

    var lane: f32 = @floatCast(@as(f64, 0.1)); // v_0
    for (1..N) |i| lane += @as(f32, @floatCast(@as(f64, @floatFromInt(i)) + 0.1));
    var expected: f32 = lane; // e[0]
    for (1..8) |_| expected += lane; // all 8 lanes equal, same reduction order
    try h.expectRunFloat(io, allocator, &f, &.{}, expected, backend);

    try autoWide(io, allocator, backend);
}

fn autoWide(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // End to end: eight contiguous scalar f32 adds are auto-vectorized at 8-wide into one
    // <8 x f32> add (runLanes(8)), which then lowers to a 256-bit vaddps and runs on AVX.
    if (backend.qemu_cmd == null) return;
    const ca = [8]f32{ 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8 };
    const da = [8]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f32 });
    const b = try f.appendBlock();
    var av: [8]ir.function.Value = undefined;
    var bv: [8]ir.function.Value = undefined;
    for (0..8) |i| av[i] = try f.appendInst(b, t, .{ .fconst = ca[i] });
    for (0..8) |i| bv[i] = try f.appendInst(b, t, .{ .fconst = da[i] });
    var c: [8]ir.function.Value = undefined;
    for (0..8) |i| c[i] = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = av[i], .rhs = bv[i] } }); // 8 parallel adds
    var s = c[0];
    for (1..8) |i| s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = c[i] } });
    f.setTerminator(b, .{ .ret = s });

    try std.testing.expect(try opt.vectorize.runLanes(allocator, &f, 8)); // fuse the eight adds

    var cc: [8]f32 = undefined;
    for (0..8) |i| cc[i] = ca[i] + da[i];
    var expected: f32 = cc[0];
    for (1..8) |i| expected += cc[i]; // same reduction order as the IR
    try h.expectRunFloat(io, allocator, &f, &.{}, expected, backend);

    try wideBlockEdge(io, allocator, backend);
}

fn wideBlockEdge(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // A <8 x f32> crosses a block edge as a merge-block parameter (a 256-bit vmovups move).
    // (x < 5) ? sum(va) : sum(vb) picks va or vb at runtime, so both edges are exercised.
    if (backend.qemu_cmd == null) return;
    const ca = [8]f32{ 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8 };
    const da = [8]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    var sc: f32 = ca[0];
    for (1..8) |i| sc += ca[i];
    var sd: f32 = da[0];
    for (1..8) |i| sd += da[i];
    const Case = struct { x: f32, want: f32 };
    for ([_]Case{ .{ .x = 1.0, .want = sc }, .{ .x = 9.0, .want = sd } }) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f32 });
        const v8 = try f.types.intern(.{ .vector = .{ .len = 8, .elem = t } });
        const bool_t = try f.types.intern(.bool);
        const entry = try f.appendBlock();
        const x = try f.appendBlockParam(entry, t);
        var af: [8]ir.function.Value = undefined;
        for (0..8) |i| af[i] = try f.appendInst(entry, t, .{ .fconst = ca[i] });
        const va = try f.appendInst(entry, v8, .{ .struct_new = .{ .fields = try f.internValueList(&af) } });
        var bf: [8]ir.function.Value = undefined;
        for (0..8) |i| bf[i] = try f.appendInst(entry, t, .{ .fconst = da[i] });
        const vb = try f.appendInst(entry, v8, .{ .struct_new = .{ .fields = try f.internValueList(&bf) } });
        const five = try f.appendInst(entry, t, .{ .fconst = 5.0 });
        const lt = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = five } });
        const then_b = try f.appendBlock();
        const else_b = try f.appendBlock();
        const merge = try f.appendBlock();
        const m = try f.appendBlockParam(merge, v8);
        try f.appendIf(entry, lt, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
        f.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try f.internValueList(&.{va}) } });
        f.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try f.internValueList(&.{vb}) } });
        var e: [8]ir.function.Value = undefined;
        for (0..8) |i| e[i] = try f.appendInst(merge, t, .{ .extract = .{ .aggregate = m, .index = @intCast(i) } });
        var s = e[0];
        for (1..8) |i| s = try f.appendInst(merge, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = e[i] } });
        f.setTerminator(merge, .{ .ret = s });
        try h.expectRunFloat(io, allocator, &f, &.{c.x}, c.want, backend);
    }
}

fn vectorSpill(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // High vector pressure: 16 live <4 x f32> (> 13 allocatable xmm) force whole-vector spills
    // to 16-byte slots (movups). Each v_i has all lanes = (i + 0.1). Sum them, reduce a lane.
    const N = 16;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f32 });
    const v4 = try f.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const b = try f.appendBlock();
    var vs: [N]ir.function.Value = undefined;
    for (0..N) |i| {
        const c = try f.appendInst(b, t, .{ .fconst = @as(f64, @floatFromInt(i)) + 0.1 });
        vs[i] = try f.appendInst(b, v4, .{ .struct_new = .{ .fields = try f.internValueList(&.{ c, c, c, c }) } });
    }
    var acc = vs[0];
    for (1..N) |i| acc = try f.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vs[i] } });
    var e: [4]ir.function.Value = undefined;
    for (0..4) |i| e[i] = try f.appendInst(b, t, .{ .extract = .{ .aggregate = acc, .index = @intCast(i) } });
    const s01 = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = e[0], .rhs = e[1] } });
    const s012 = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = e[2] } });
    const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = e[3] } });
    f.setTerminator(b, .{ .ret = s });

    var lane: f32 = @floatCast(@as(f64, 0.1)); // v_0
    for (1..N) |i| lane += @as(f32, @floatCast(@as(f64, @floatFromInt(i)) + 0.1));
    const expected = ((lane + lane) + lane) + lane; // all 4 lanes equal, same reduction order
    try h.expectRunFloat(io, allocator, &f, &.{}, expected, backend);
}

fn floats(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    const ops = [_]struct { op: ir.function.BinOp, x: f32, y: f32 }{
        .{ .op = .add, .x = 1.1, .y = 2.2 },
        .{ .op = .sub, .x = 5.5, .y = 1.1 },
        .{ .op = .mul, .x = 1.7, .y = 2.3 },
        .{ .op = .div, .x = 10.0, .y = 3.0 },
    };
    for (ops) |c| { // each scalar SSE op, validated against the same f32-precision arithmetic
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f32 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const y = try f.appendBlockParam(b, t);
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = c.op, .lhs = x, .rhs = y } });
        f.setTerminator(b, .{ .ret = r });
        // Compute the expected from the f32 operands (not comptime) so the rounding matches.
        const expected: f32 = switch (c.op) {
            .add => c.x + c.y,
            .sub => c.x - c.y,
            .mul => c.x * c.y,
            .div => c.x / c.y,
            else => unreachable,
        };
        try h.expectRunFloat(io, allocator, &f, &.{ c.x, c.y }, expected, backend);
    }
    { // a small chain: (x + y) * x
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f32 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const y = try f.appendBlockParam(b, t);
        const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
        const p = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = s, .rhs = x } });
        f.setTerminator(b, .{ .ret = p });
        const xv: f32 = 1.1;
        const yv: f32 = 2.2;
        try h.expectRunFloat(io, allocator, &f, &.{ xv, yv }, (xv + yv) * xv, backend);
    }
    { // float constants: x * 2.0 + 1.0
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f32 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const two = try f.appendInst(b, t, .{ .fconst = 2.0 });
        const one = try f.appendInst(b, t, .{ .fconst = 1.0 });
        const p = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = two } });
        const s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = one } });
        f.setTerminator(b, .{ .ret = s });
        const xv: f32 = 3.3;
        try h.expectRunFloat(io, allocator, &f, &.{xv}, xv * 2.0 + 1.0, backend);
    }
    { // high fp pressure: 14 live temporaries (> 13 allocatable xmm) force spilling
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f32 });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        var temps: [14]ir.function.Value = undefined;
        for (0..14) |i| {
            const k = try f.appendInst(b, t, .{ .fconst = @floatFromInt(i + 1) });
            temps[i] = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = k } });
        }
        var s = temps[0];
        for (1..14) |i| s = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = temps[i] } });
        f.setTerminator(b, .{ .ret = s });

        const xv: f32 = 1.1;
        var et: [14]f32 = undefined;
        for (0..14) |i| et[i] = xv + @as(f32, @floatFromInt(i + 1));
        var es: f32 = et[0];
        for (1..14) |i| es += et[i];
        try h.expectRunFloat(io, allocator, &f, &.{xv}, es, backend);
    }
    { // float compare + select: min(x, y) = (x < y) ? x : y, exercising both branches
        const Case = struct { x: f32, y: f32 };
        for ([_]Case{ .{ .x = 5.5, .y = 3.3 }, .{ .x = 1.1, .y = 7.7 } }) |c| {
            var f = Function.init(allocator);
            defer f.deinit();
            const t = try f.types.intern(.{ .float = .f32 });
            const bool_t = try f.types.intern(.bool);
            const b = try f.appendBlock();
            const x = try f.appendBlockParam(b, t);
            const y = try f.appendBlockParam(b, t);
            const lt = try f.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
            const m = try f.appendInst(b, t, .{ .select = .{ .cond = lt, .then = x, .@"else" = y } });
            f.setTerminator(b, .{ .ret = m });
            try h.expectRunFloat(io, allocator, &f, &.{ c.x, c.y }, @min(c.x, c.y), backend);
        }
    }
    { // conversions: round to 2 decimals via (float)(int)(x * 100) / 100 (both directions)
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .float = .f32 });
        const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const c100 = try f.appendInst(b, t, .{ .fconst = 100.0 });
        const scaled = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = c100 } });
        const i = try f.appendInst(b, i32_t, .{ .convert = .{ .value = scaled } }); // float -> int (truncate)
        const back = try f.appendInst(b, t, .{ .convert = .{ .value = i } }); // int -> float
        const r = try f.appendInst(b, t, .{ .arith = .{ .op = .div, .lhs = back, .rhs = c100 } });
        f.setTerminator(b, .{ .ret = r });
        const xv: f32 = 1.234;
        const scaled_e: f32 = xv * 100.0;
        const back_e: f32 = @floatFromInt(@as(i32, @intFromFloat(scaled_e)));
        try h.expectRunFloat(io, allocator, &f, &.{xv}, back_e / 100.0, backend);
    }
    { // float equality compare feeding a select: (x == y) ? x : y, equal and not-equal
        const Case = struct { x: f32, y: f32 };
        for ([_]Case{ .{ .x = 3.3, .y = 3.3 }, .{ .x = 3.3, .y = 7.7 } }) |c| {
            var f = Function.init(allocator);
            defer f.deinit();
            const t = try f.types.intern(.{ .float = .f32 });
            const bool_t = try f.types.intern(.bool);
            const b = try f.appendBlock();
            const x = try f.appendBlockParam(b, t);
            const y = try f.appendBlockParam(b, t);
            const eq = try f.appendInst(b, bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = y } });
            const m = try f.appendInst(b, t, .{ .select = .{ .cond = eq, .then = x, .@"else" = y } });
            f.setTerminator(b, .{ .ret = m });
            try h.expectRunFloat(io, allocator, &f, &.{ c.x, c.y }, if (c.x == c.y) c.x else c.y, backend);
        }
    }
    { // int select via cmov-free branch: select(cond, a, b), both branches
        for ([_]u8{ 1, 0 }) |cond| {
            var f = Function.init(allocator);
            defer f.deinit();
            const i32_t = try h.i32type(&f);
            const bool_t = try f.types.intern(.bool);
            const b = try f.appendBlock();
            const cp = try f.appendBlockParam(b, bool_t);
            const a = try f.appendBlockParam(b, i32_t);
            const bb = try f.appendBlockParam(b, i32_t);
            const m = try f.appendInst(b, i32_t, .{ .select = .{ .cond = cp, .then = a, .@"else" = bb } });
            f.setTerminator(b, .{ .ret = m });
            try expectRun(io, allocator, &f, &.{ cond, 0x42, 0x99 }, if (cond == 1) 0x42 else 0x99, backend);
        }
    }
    { // fp block parameters: min via a diamond (if -> then/else -> merge(float param))
        const Case = struct { x: f32, y: f32 };
        for ([_]Case{ .{ .x = 5.5, .y = 3.3 }, .{ .x = 1.1, .y = 7.7 } }) |c| {
            var f = Function.init(allocator);
            defer f.deinit();
            const t = try f.types.intern(.{ .float = .f32 });
            const bool_t = try f.types.intern(.bool);
            const entry = try f.appendBlock();
            const x = try f.appendBlockParam(entry, t);
            const y = try f.appendBlockParam(entry, t);
            const lt = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = y } });
            const then_b = try f.appendBlock();
            const else_b = try f.appendBlock();
            const merge = try f.appendBlock();
            const m = try f.appendBlockParam(merge, t);
            try f.appendIf(entry, lt, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
            f.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try f.internValueList(&.{x}) } });
            f.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try f.internValueList(&.{y}) } });
            f.setTerminator(merge, .{ .ret = m });
            try h.expectRunFloat(io, allocator, &f, &.{ c.x, c.y }, @min(c.x, c.y), backend);
        }
    }
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
    { // subtraction (negative result, low byte)
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const b = try f.appendBlock();
        const x = try f.appendBlockParam(b, t);
        const y = try f.appendBlockParam(b, t);
        const d = try f.appendInst(b, t, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = y } });
        f.setTerminator(b, .{ .ret = d });
        try expectRun(io, allocator, &f, &.{ 3, 10 }, -7, backend);
    }
    inline for (.{ .{ .op = .div, .want = 6 }, .{ .op = .rem, .want = 2 } }) |c| { // signed div/rem
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
    { // max(a, b) via if + merge
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
    { // counting loop: sum 1..n
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

fn fallthrough(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // Fall-through branch elision. x86_64 emits blocks in append order, so a successor whose block
    // is appended immediately after its predecessor is the emitted-NEXT block and its branch is
    // elided (the code just falls into it). These cases execute the elided layouts end to end.

    { // x86_64 fallthrough: unconditional jump to the next block elides the jmp, block param on
        // the fall-through edge still arrives. entry -> next(x + 100), next(p) -> ret p.
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const entry = try f.appendBlock();
        const x = try f.appendBlockParam(entry, t);
        const next = try f.appendBlock(); // appended right after entry: the emitted-NEXT block
        const p = try f.appendBlockParam(next, t);
        const k = try f.appendArithImm(entry, t, .add, x, 100); // carried across the fall-through edge
        try f.setJump(entry, next, &.{k});
        f.setTerminator(next, .{ .ret = p });
        for ([_]i32{ 0, 1, -1, 7, 55 }) |xv| {
            try expectRun(io, allocator, &f, &.{xv}, xv +% 100, backend);
        }
    }

    { // x86_64 fallthrough: conditional then-edge next elides jmp-then (then_next layout), block
        // param on the then fall-through edge arrives. entry: if a>b -> then_b(a) else else_b(b),
        // with then_b appended NEXT. Sweep the condition so both edges are taken. Result = max(a,b).
        const Case = struct { a: i32, b: i32 };
        for ([_]Case{ .{ .a = 3, .b = 4 }, .{ .a = 7, .b = 2 }, .{ .a = -5, .b = -9 }, .{ .a = 6, .b = 6 } }) |c| {
            var f = Function.init(allocator);
            defer f.deinit();
            const t = try h.i32type(&f);
            const bool_t = try f.types.intern(.bool);
            const entry = try f.appendBlock();
            const a = try f.appendBlockParam(entry, t);
            const b = try f.appendBlockParam(entry, t);
            const then_b = try f.appendBlock(); // NEXT: the then edge falls through
            const pt = try f.appendBlockParam(then_b, t);
            const else_b = try f.appendBlock();
            const pe = try f.appendBlockParam(else_b, t);
            const merge = try f.appendBlock();
            const r = try f.appendBlockParam(merge, t);
            const cnd = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
            try f.appendIf(entry, cnd, .{ .target = then_b, .args = &.{a} }, .{ .target = else_b, .args = &.{b} });
            try f.setJump(then_b, merge, &.{pt});
            try f.setJump(else_b, merge, &.{pe});
            f.setTerminator(merge, .{ .ret = r });
            try expectRun(io, allocator, &f, &.{ c.a, c.b }, @max(c.a, c.b), backend);
        }
    }

    { // x86_64 fallthrough: conditional else-edge next inverts (jnz->jz) and falls through
        // (else_next layout), block param on the else fall-through edge arrives. Same diamond but
        // else_b is appended NEXT, forcing the inverted branch. Sweep the condition. Result = max(a,b).
        const Case = struct { a: i32, b: i32 };
        for ([_]Case{ .{ .a = 3, .b = 4 }, .{ .a = 7, .b = 2 }, .{ .a = -5, .b = -9 }, .{ .a = 6, .b = 6 } }) |c| {
            var f = Function.init(allocator);
            defer f.deinit();
            const t = try h.i32type(&f);
            const bool_t = try f.types.intern(.bool);
            const entry = try f.appendBlock();
            const a = try f.appendBlockParam(entry, t);
            const b = try f.appendBlockParam(entry, t);
            const else_b = try f.appendBlock(); // NEXT: the else edge falls through (branch inverted)
            const pe = try f.appendBlockParam(else_b, t);
            const then_b = try f.appendBlock();
            const pt = try f.appendBlockParam(then_b, t);
            const merge = try f.appendBlock();
            const r = try f.appendBlockParam(merge, t);
            const cnd = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
            try f.appendIf(entry, cnd, .{ .target = then_b, .args = &.{a} }, .{ .target = else_b, .args = &.{b} });
            try f.setJump(then_b, merge, &.{pt});
            try f.setJump(else_b, merge, &.{pe});
            f.setTerminator(merge, .{ .ret = r });
            try expectRun(io, allocator, &f, &.{ c.a, c.b }, @max(c.a, c.b), backend);
        }
    }

    { // x86_64 fallthrough: a diamond and a loop compute correctly. The loop chains every elided
        // layout: entry -> header falls through with block params, header's then-edge (body) falls
        // through, body jumps BACKWARD to header (a real jmp, not elided). Counting sum 1..n.
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
        try f.setJump(entry, header, &.{ zero, one }); // fall-through with two block params
        const cont = try f.appendInst(header, bool_t, .{ .icmp = .{ .op = .le, .lhs = i, .rhs = n } });
        try f.appendIf(header, cont, .{ .target = body, .args = &.{} }, .{ .target = exit, .args = &.{sum} });
        const sum2 = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = i } });
        const ob = try f.appendInst(body, t, .{ .iconst = 1 });
        const inext = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = i, .rhs = ob } });
        try f.setJump(body, header, &.{ sum2, inext }); // backward jump: a real jmp, not elided
        f.setTerminator(exit, .{ .ret = r });
        try expectRun(io, allocator, &f, &.{5}, 15, backend);
        try expectRun(io, allocator, &f, &.{10}, 55, backend);
    }
}

fn spilling(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // 8 temporaries (a+1..a+8) all live until summed -> forces register spilling.
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try h.i32type(&f);
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, t);
    var temps: [8]ir.function.Value = undefined;
    for (0..8) |k| temps[k] = try f.appendArithImm(b, t, .add, a, @intCast(k + 1));
    var acc = temps[0];
    for (1..8) |k| acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = temps[k] } });
    f.setTerminator(b, .{ .ret = acc });
    try expectRun(io, allocator, &f, &.{10}, 116, backend); // 8a + 36
    try intPressure(io, allocator, backend);
    try tailSplit(io, allocator, backend);
    try secondChanceReHome(io, allocator, backend);
    try secondChanceDecline(io, allocator, backend);
    try terminatorReHome(io, allocator, backend);
}

fn secondChanceReHome(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // f(a, b): t0 = a*b is defined FIRST and used again only at the very end. Under the 20-term
    // pressure block it TAIL-SPLITS (register prefix + slot tail). As the reduction consumes the
    // terms the register pressure DROPS, so by t0's late use a register is free again and
    // second-chance RE-HOMES t0 into it: its final use reads that register instead of reloading from
    // the slot on every use. The result is correct only if the re-home reload round-trips the right
    // bits, and the differential is meaningful only if a re-home actually fired.
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try h.i32type(&f);
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, t);
    const bp = try f.appendBlockParam(b, t);
    const t0 = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    var terms: [20]ir.function.Value = undefined;
    var k: i64 = 1;
    while (k <= 20) : (k += 1) {
        const kc = try f.appendInst(b, t, .{ .iconst = k });
        const ak = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
    }
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    const res = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = t0 } }); // t0's sole late use
    f.setTerminator(b, .{ .ret = res });

    // Meaningful-differential gate: a second-chance re-home MUST have fired (a `.reg` segment after a
    // `.spill` segment). Without it this would only exercise the tail-split + per-use reload path.
    try std.testing.expect(try isel.reHomeCountForTest(allocator, &f) > 0);

    // Sweep inputs (zero, unit, negatives, larger magnitudes), i32 wrapping arithmetic to match the
    // target's 32-bit muls/adds exactly. res = a*b + sum_{k=1..20}(a*k + b), read mod 256.
    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const av = in[0];
        const bv = in[1];
        var expected: i32 = av *% bv;
        var kk: i32 = 1;
        while (kk <= 20) : (kk += 1) expected +%= (av *% kk) +% bv;
        try expectRun(io, allocator, &f, &.{ av, bv }, expected, backend);
    }
}

fn secondChanceDecline(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // A sustained-pressure kernel where second-chance CANNOT save every split value. Six early
    // `sp[i] = a*(i+1)` products are defined first and used only late, so under pressure they
    // TAIL-SPLIT (their far next use makes them the Belady victims). Twelve `res[i] = b + const`
    // values are then defined and used TWICE (once before and once after the sp uses), so they stay
    // register-resident and occupy every register ACROSS the sp uses. With the pool full at those
    // uses, second-chance has no free register to re-home some split sp values, so they fall back to
    // per-use slot reloads. This exercises the DECLINE path (`free` empty) alongside the re-homes,
    // and the result is correct only if every reload (re-homed or per-use) is right.
    const nspill = 6;
    const nres = 12;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try h.i32type(&f);
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, t);
    const bp = try f.appendBlockParam(b, t);
    var sp: [nspill]ir.function.Value = undefined;
    for (0..nspill) |i| {
        const c = try f.appendInst(b, t, .{ .iconst = @intCast(i + 1) });
        sp[i] = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    }
    var res: [nres]ir.function.Value = undefined;
    for (0..nres) |i| {
        const c = try f.appendInst(b, t, .{ .iconst = @intCast(100 + i) });
        res[i] = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = bp, .rhs = c } });
    }
    var acc = res[0];
    for (1..nres) |i| acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = res[i] } });
    for (0..nspill) |i| acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = sp[i] } });
    for (0..nres) |i| acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = res[i] } });
    f.setTerminator(b, .{ .ret = acc });

    // Meaningful gate: more values TAIL-SPLIT than were RE-HOMED, i.e. at least one split value found
    // no free register at its tail use and DECLINED (stayed in its slot, reloading per use). Both the
    // re-home path and the decline path are thus exercised in one kernel.
    const segs = try isel.splitCountForTest(allocator, &f);
    const rehomes = try isel.reHomeCountForTest(allocator, &f);
    try std.testing.expect(segs > rehomes);

    // result = 2*sum_i(b + 100 + i) + sum_i(a*(i+1)), i32 wrapping arithmetic, read mod 256.
    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const av = in[0];
        const bv = in[1];
        var expected: i32 = bv +% 100; // res[0]
        var i: i32 = 1;
        while (i < nres) : (i += 1) expected +%= bv +% (100 + i);
        i = 0;
        while (i < nspill) : (i += 1) expected +%= av *% (i + 1);
        i = 0;
        while (i < nres) : (i += 1) expected +%= bv +% (100 + i);
        try expectRun(io, allocator, &f, &.{ av, bv }, expected, backend);
    }
}

fn terminatorReHome(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // Regression for the terminator-position reload drain. t0 = a*b is defined FIRST and used ONLY by
    // the `ret`, whose operand is a NON-edge-arg, so t0 is intra-splittable. Under the 20-term
    // pressure block t0 TAIL-SPLITS. As the reduction drains the terms a register frees, and because
    // t0's SOLE remaining use is the terminator, second-chance re-homes t0 with a `.reload` recorded
    // AT the terminator position (block_end). The per-instruction drain never reaches that position,
    // so without the terminator drain the reload is dropped and `ret` returns the stale, never-loaded
    // register (WRONG value). The result is correct only if the terminator drain emits the reload.
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try h.i32type(&f);
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, t);
    const bp = try f.appendBlockParam(b, t);
    const t0 = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    var terms: [20]ir.function.Value = undefined;
    var k: i64 = 1;
    while (k <= 20) : (k += 1) {
        const kc = try f.appendInst(b, t, .{ .iconst = k });
        const ak = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
    }
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    // The reduction only exists to create then relieve pressure (its final `acc` is deliberately
    // unreturned). The RETURN value is t0 itself, so t0's last use is the terminator and any
    // second-chance re-home of it lands there.
    f.setTerminator(b, .{ .ret = t0 });

    // Meaningful-differential gate: t0 must actually tail-split, else a whole-life register would also
    // return the right value and the terminator drain would not be exercised.
    try std.testing.expect(try isel.reHomeCountForTest(allocator, &f) > 0);

    // Sweep inputs (zero, unit, negatives, larger magnitudes). result = a*b in i32 wrapping, mod 256.
    const inputs = [_][2]i32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const av = in[0];
        const bv = in[1];
        const expected: i32 = av *% bv;
        try expectRun(io, allocator, &f, &.{ av, bv }, expected, backend);
    }
}

fn tailSplit(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // Force an intra-block tail split: `v = a + 100` is defined early and used ONLY at the very
    // end, so its next use is the furthest of any live value. Ten temporaries `t_k = a + k` all
    // live until summed drive pressure past the seven-register GPR pool, so eviction picks `v`
    // (furthest next use) and TAIL-SPLITS it: its register serves the hot prefix, a stack slot the
    // cold tail, with a store at the split point. The final `res = (sum t_k) + v` reloads `v` from
    // that slot, so a mis-drained store (saving after the taker overwrote the register) would give
    // the wrong result. Swept over inputs and checked bit-identical against a Zig reference:
    //   res = (11 * a) + 155  for ten temporaries (sum_{k=1..10}(a+k) = 10a + 55, plus a + 100).
    for ([_]i64{ 3, 5, 7 }) |av| {
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, t);
        const v = try f.appendArithImm(b, t, .add, a, 100); // long-lived, used only at the end
        var terms: [10]ir.function.Value = undefined;
        for (0..10) |k| terms[k] = try f.appendArithImm(b, t, .add, a, @intCast(k + 1));
        var acc = terms[0];
        for (1..10) |k| acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[k] } });
        const res = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } }); // v's sole use
        f.setTerminator(b, .{ .ret = res });

        // The case must actually exercise the splitter, else it proves nothing about the store drain.
        try std.testing.expect(try isel.splitCountForTest(allocator, &f) > 0);

        var expected: i64 = av + 100; // v
        for (1..11) |k| expected += av + @as(i64, @intCast(k)); // sum of the temporaries
        try expectRun(io, allocator, &f, &.{av}, expected, backend);
    }
}

fn intPressure(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // Twelve temporaries (a*k + b for k in 1..=12) all live until summed, far past the seven-entry
    // GPR pool, so several forced spills run through the furthest-next-use eviction and its
    // reload/spill machinery. Swept over inputs and checked bit-identical against a Zig reference
    // (result read mod 256 by the harness): sum = a*78 + 12*b.
    const Case = struct { a: i64, b: i64 };
    for ([_]Case{ .{ .a = 3, .b = 7 }, .{ .a = 5, .b = 1 }, .{ .a = 10, .b = 2 } }) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try h.i32type(&f);
        const b = try f.appendBlock();
        const a = try f.appendBlockParam(b, t);
        const bp = try f.appendBlockParam(b, t);
        var terms: [12]ir.function.Value = undefined;
        for (0..12) |k| {
            const kc = try f.appendInst(b, t, .{ .iconst = @as(i64, @intCast(k + 1)) });
            const ak = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
            terms[k] = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = bp } });
        }
        var acc = terms[0];
        for (1..12) |k| acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[k] } });
        f.setTerminator(b, .{ .ret = acc });

        var expected: i64 = 0;
        for (1..13) |k| expected += c.a * @as(i64, @intCast(k)) + c.b; // same order as the IR reduction
        try expectRun(io, allocator, &f, &.{ c.a, c.b }, expected, backend);
    }
}

fn calls(io: std.Io, allocator: std.mem.Allocator, backend: h.Backend) !void {
    // main(x) = helper(x), helper(x) = x*2 + 1 -> linked, the call resolved.
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
    const link = @import("../link.zig");
    var module: link.Module = .{};
    defer module.deinit(allocator);
    try module.addFunction(allocator, "main", &main);
    try module.addFunction(allocator, "helper", &helper);
    try h.expectRunModule(io, allocator, &module, &.{6}, 37, backend); // 6*6 + 1

    // A value live across the call (x used after) must survive the clobber via a spill
    // slot: main2(x) = helper(x) + x = (x*x + 1) + x. For x=5: 26 + 5 = 31.
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

    // Float call arguments + float result + a float value live across the call:
    // main(x, y) = helper(x, y) + x, helper(a, b) = a + b. So main = (x + y) + x.
    var fhelper = Function.init(allocator);
    defer fhelper.deinit();
    {
        const t = try fhelper.types.intern(.{ .float = .f32 });
        const b = try fhelper.appendBlock();
        const a = try fhelper.appendBlockParam(b, t);
        const bb = try fhelper.appendBlockParam(b, t);
        const r = try fhelper.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bb } });
        fhelper.setTerminator(b, .{ .ret = r });
    }
    var fmain = Function.init(allocator);
    defer fmain.deinit();
    {
        const t = try fmain.types.intern(.{ .float = .f32 });
        const b = try fmain.appendBlock();
        const x = try fmain.appendBlockParam(b, t);
        const y = try fmain.appendBlockParam(b, t);
        const called = try fmain.appendCall(b, t, "helper", &.{ x, y }); // x is live across the call
        const r = try fmain.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = x } });
        fmain.setTerminator(b, .{ .ret = r });
    }
    var fmodule: link.Module = .{};
    defer fmodule.deinit(allocator);
    try fmodule.addFunction(allocator, "main", &fmain);
    try fmodule.addFunction(allocator, "helper", &fhelper);
    const xv: f32 = 1.7;
    const yv: f32 = 2.3;
    try h.expectRunFloatModule(io, allocator, &fmodule, &.{ xv, yv }, (xv + yv) + xv, backend);
}
