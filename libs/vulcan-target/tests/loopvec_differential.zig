//! Differential JIT oracle for the structural loop vectorizer. For each map-loop shape we build two
//! identical functions, run loopvec (and, in the second test group, the full microarch pipeline so
//! SLP widens it) on one, JIT both on the host, run them over real arrays, and require bit-identical
//! output across trip counts chosen to exercise the remainder: 0, below V, exactly V, V-multiples,
//! and non-multiples. Any divergence is a miscompile. Runs where the native JIT has a backend.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;

fn ampere() *const opt.microarch.Model {
    return opt.microarch.modelFor(.@"ampere-altra");
}

fn hasJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .riscv64, .x86 => true,
        else => false,
    };
}

/// `for (i = 0; i < n; i += 1) y[i] = a*x[i] + y[i];` over f32 arrays.
fn buildSaxpy(func: *Function) !void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const x = try func.appendBlockParam(entry, ptr_t);
    const y = try func.appendBlockParam(entry, ptr_t);
    const a = try func.appendBlockParam(entry, f32_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const i = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const xaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = off } });
    const xv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = xaddr } });
    const yaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = off } });
    const yv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = yaddr } });
    const ax = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = xv } });
    const res = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .add, .lhs = ax, .rhs = yv } });
    try func.appendStore(body, res, yaddr);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ni});
    func.setTerminator(done, .{ .ret = null });
}

/// `for (i = 0; i < n; i += 1) y[i] = x[i] * x[i];` (a pure map, no read of the output).
fn buildSquare(func: *Function) !void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const x = try func.appendBlockParam(entry, ptr_t);
    const y = try func.appendBlockParam(entry, ptr_t);
    const a = try func.appendBlockParam(entry, f32_t); // unused, keeps the 4-arg signature uniform
    _ = a;
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const i = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const xaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = off } });
    const xv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = xaddr } });
    const sq = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .mul, .lhs = xv, .rhs = xv } });
    const yaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = off } });
    try func.appendStore(body, sq, yaddr);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ni});
    func.setTerminator(done, .{ .ret = null });
}

/// `s = 0; for (i = 0; i < n; i += 1) s += a[i]; return s;` over f32. Marked fast_math so the FP
/// reduction may reassociate (vector partial sums + horizontal reduce).
fn buildSumReduction(func: *Function) !void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan", .key = "fast_math", .value = .flag } });
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero_i = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const zero_f = try func.appendInst(entry, f32_t, .{ .fconst = 0 });
    try func.setJump(entry, loop, &.{ zero_i, zero_f });
    const i = try func.appendBlockParam(loop, i32_t);
    const s = try func.appendBlockParam(loop, f32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done, .args = &.{s} });
    const bi = try func.appendBlockParam(body, i32_t);
    const bs = try func.appendBlockParam(body, f32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const addr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = off } });
    const v = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = addr } });
    const ns = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = v } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    const rs = try func.appendBlockParam(done, f32_t);
    func.setTerminator(done, .{ .ret = rs });
}

const Builder = *const fn (*Function) anyerror!void;
const SaxpyFn = *const fn (x: [*]f32, y: [*]f32, a: f32, n: i32) callconv(.c) void;
const SumFn = *const fn (a: [*]f32, n: i32) callconv(.c) f32;

/// Transform: apply loopvec only, or the full microarch pipeline (so SLP widens the body too).
const Mode = enum { loopvec_only, full };

fn expectMapMatches(build: Builder, mode: Mode) !void {
    const allocator = std.testing.allocator;
    const trip_counts = [_]i32{ 0, 1, 3, 4, 5, 7, 8, 12, 13, 17, 100 };

    var orig = Function.init(allocator);
    defer orig.deinit();
    try build(&orig);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try build(&tuned);
    switch (mode) {
        .loopvec_only => try std.testing.expect(try opt.microarch.loopvec.run(allocator, &tuned, ampere())),
        .full => _ = try opt.microarch.optimize(allocator, &tuned, ampere()),
    }

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_o = try target.native.jitFunction(allocator, &orig);
    defer buf_o.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const f_o = buf_o.entry(SaxpyFn, 0);
    const f_t = buf_t.entry(SaxpyFn, 0);

    for (trip_counts) |n| {
        var x: [128]f32 = undefined;
        var y_o: [128]f32 = undefined;
        var y_t: [128]f32 = undefined;
        for (0..128) |k| {
            x[k] = @floatFromInt(k + 1);
            y_o[k] = @floatFromInt(200 - @as(i32, @intCast(k)));
            y_t[k] = y_o[k];
        }
        f_o(&x, &y_o, 2.5, n);
        f_t(&x, &y_t, 2.5, n);
        if (!std.mem.eql(f32, &y_o, &y_t)) {
            std.debug.print("\nn={d}: y_o[0..8]={any}\n       y_t[0..8]={any}\n", .{ n, y_o[0..8], y_t[0..8] });
        }
        try std.testing.expectEqualSlices(f32, &y_o, &y_t); // whole buffer: no over/under-run either
    }
}

test "loopvec differential: saxpy, scalar V-unroll, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectMapMatches(buildSaxpy, .loopvec_only);
}

test "loopvec differential: pure square map, scalar V-unroll, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectMapMatches(buildSquare, .loopvec_only);
}

test "loopvec differential: saxpy through the full pipeline (SLP-widened), all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectMapMatches(buildSaxpy, .full);
}

test "loopvec differential: pure square map through the full pipeline, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectMapMatches(buildSquare, .full);
}

fn hasVectorValue(func: *const Function) bool {
    var i: usize = 0;
    while (i < func.valueCount()) : (i += 1) {
        const v: ir.function.Value = @enumFromInt(@as(u32, @intCast(i)));
        if (func.types.type_kind(func.valueType(v)) == .vector) return true;
    }
    return false;
}

test "loopvec output is SLP-fusable: force-fusing the main body produces vector ops" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSaxpy(&func);
    // loopvec makes a 4-wide op-major main body; the SLP vectorizer's correctness (force) path then
    // fuses it into wide loads, a vector fmul, a vector fadd, and a wide store. This proves the
    // structure loopvec produces is exactly what SLP consumes.
    try std.testing.expect(try opt.microarch.loopvec.run(allocator, &func, ampere()));
    try std.testing.expect(try opt.vectorize.runLanes(allocator, &func, 4));
    try std.testing.expect(hasVectorValue(&func));
}

test "loopvec differential: f32 sum reduction (vector accumulator), all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const trip_counts = [_]i32{ 0, 1, 3, 4, 5, 7, 8, 12, 13, 17, 100 };

    var orig = Function.init(allocator);
    defer orig.deinit();
    try buildSumReduction(&orig);
    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try buildSumReduction(&tuned);
    try std.testing.expect(try opt.microarch.loopvec.run(allocator, &tuned, ampere()));
    try std.testing.expect(hasVectorValue(&tuned)); // a genuine vector accumulator was emitted

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_o = try target.native.jitFunction(allocator, &orig);
    defer buf_o.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const f_o = buf_o.entry(SumFn, 0);
    const f_t = buf_t.entry(SumFn, 0);

    for (trip_counts) |n| {
        // Small integers as f32: their partial sums stay exact (< 2^24), so serial and vector-partial
        // reductions agree bit-for-bit despite the reassociation.
        var arr: [128]f32 = undefined;
        for (0..128) |k| arr[k] = @floatFromInt((k % 7) + 1);
        try std.testing.expectEqual(f_o(&arr, n), f_t(&arr, n));
    }
}

test "the full ampere pipeline vectorizes saxpy (splat-aware cost model)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSaxpy(&func);
    _ = try opt.microarch.optimize(allocator, &func, ampere());
    // The invariant multiplier `a` is a splat (one `dup`, not a 4-lane pack), so the SLP cost model
    // judges the saxpy group profitable on the wide OoO ampere and vectorizes it, like gcc -O3's fmla.
    try std.testing.expect(hasVectorValue(&func));
}
