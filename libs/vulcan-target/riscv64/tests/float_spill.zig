//! Scalar-float register spilling, executed on qemu-riscv64 (the oracle) for the non-vpu path and
//! compiled for the et-soc VPU path. When the scalar float register file is exhausted, isel.zig
//! spills a float value to an 8-byte stack slot and reloads it on use, exactly as it already does
//! for integers and vectors. These tests force that pressure and prove the spilled computation is
//! still correct.
//!
//! Non-vpu (f32 and f64): build a function that keeps far more scalar floats simultaneously live
//! than the allocatable float file holds (22 registers: ten caller-saved temporaries plus twelve
//! callee-saved, with f30/f31 reserved as the two spill scratches), so several must spill. The
//! values are exact small integers in float form, so their sum is exact in both f32 and f64
//! regardless of evaluation order, and the qemu-executed result must equal the closed-form
//! reference bit-for-bit. Compiled with `selectFunction` directly (no scheduler) so the wide live
//! range is preserved: a pressure-tolerant scheduler could otherwise fold the reduction and dodge
//! the spill this test exists to exercise.
//!
//! VPU: the et-soc SLP shape `out[i] = a[i]*b[i] + a[i]` packs two different freshly-loaded 8-wide
//! operands (a and b) for its first group, needing 16 live scalar floats at once - more than the
//! vpu-mode scalar pool (f0..f7). Before float spilling this failed register allocation with
//! `error.Unsupported`; now it spills and compiles. The test asserts it compiles (and runs it under
//! sw-sysemu when present).

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("vulcan-opt").microarch;
const isel = @import("../isel.zig");
const harness = @import("harness.zig");
const etsoc = @import("etsoc_sysemu.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

/// The number of simultaneously-live derived float values. 30 exceeds the 22 allocatable scalar
/// float registers, so at least eight must spill to the stack.
const n_live = 30;

/// Build `f(a) = sum over i in 0..n_live of (a + i)`, where every `a + i` is computed up front (so
/// all `n_live` results are live at once, forcing spills) and then summed. `dbl` selects f64 vs
/// f32. Each `i` is materialized as an integer constant and converted to float (fconst has no f64
/// form), so the builder is identical across precisions. With `a` an exact integer and small `i`,
/// every intermediate is an exact integer in float, so the sum is order-independent and exact.
fn buildSumFunc(allocator: std.mem.Allocator, dbl: bool) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ft = try func.types.intern(.{ .float = if (dbl) .f64 else .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, ft);

    var vals: [n_live]Value = undefined;
    for (0..n_live) |i| {
        const ci = try func.appendInst(b, i32_t, .{ .iconst = @intCast(i) });
        const cf = try func.appendInst(b, ft, .{ .convert = .{ .value = ci } });
        vals[i] = try func.appendInst(b, ft, .{ .arith = .{ .op = .add, .lhs = a, .rhs = cf } });
    }
    var acc = vals[0];
    for (vals[1..]) |v| acc = try func.appendInst(b, ft, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(b, .{ .ret = acc });
    return func;
}

/// The exact reference: sum_{i=0}^{n_live-1} (a + i) = n_live*a + n_live*(n_live-1)/2.
fn reference(comptime T: type, a: T) T {
    var acc: T = a + 0.0;
    var i: usize = 1;
    while (i < n_live) : (i += 1) acc += a + @as(T, @floatFromInt(i));
    return acc;
}

fn checkSpillSum(io: std.Io, comptime T: type) !void {
    const allocator = std.testing.allocator;
    const dbl = T == f64;

    // Compile with selectFunction ALONE (no scheduler) so the source-order wide live range - and
    // thus the spill - is preserved. Also proves the codegen path itself does not error under
    // float pressure, independent of whether qemu is present.
    var func = try buildSumFunc(allocator, dbl);
    defer func.deinit();
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);

    const a_val: T = 100.0;
    const a_bits: u64 = if (dbl) @bitCast(@as(f64, a_val)) else @as(u32, @bitCast(@as(f32, a_val)));
    const fargs = [_]u64{a_bits};
    const got_bits = harness.runCompiledFloat(io, allocator, code, dbl, &fargs, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    const got: T = if (dbl) @bitCast(got_bits) else @bitCast(@as(u32, @truncate(got_bits)));
    const want = reference(T, a_val);
    try std.testing.expectEqual(@as(std.meta.Int(.unsigned, @bitSizeOf(T)), @bitCast(want)), @as(std.meta.Int(.unsigned, @bitSizeOf(T)), @bitCast(got)));
}

test "float-spill: f32 sum of 30 simultaneously-live values spills and computes correctly (qemu-riscv64)" {
    try checkSpillSum(std.testing.io, f32);
}

test "float-spill: f64 sum of 30 simultaneously-live values spills and computes correctly (qemu-riscv64)" {
    try checkSpillSum(std.testing.io, f64);
}

/// Build `f(a) = sum over i in 0..n_live of (a*i + a)`, where each `a*i` multiply immediately
/// precedes and is single-used by its `+ a` add - the exact scalar fma shape isel.zig fuses to
/// `fmadd`. All `n_live` results stay live and force spills, so the fma candidates whose result (or
/// an operand) spills cannot use the three-source R4 form (only two float spill scratches exist) and
/// fall back to a standalone spilled `fmul` + `fadd`; the low-pressure early ones still fuse. This
/// exercises both the fusion-under-pressure fallback and the standalone spilled multiply. With `a`
/// an exact small integer, every intermediate `a*i + a = a*(i+1)` is exact, so the sum is
/// order-independent and bit-exact in f32.
fn buildFmaSumFunc(allocator: std.mem.Allocator, dbl: bool) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ft = try func.types.intern(.{ .float = if (dbl) .f64 else .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, ft);

    var vals: [n_live]Value = undefined;
    for (0..n_live) |i| {
        const ci = try func.appendInst(b, i32_t, .{ .iconst = @intCast(i) });
        const cf = try func.appendInst(b, ft, .{ .convert = .{ .value = ci } });
        const prod = try func.appendInst(b, ft, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = cf } });
        vals[i] = try func.appendInst(b, ft, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = a } });
    }
    var acc = vals[0];
    for (vals[1..]) |v| acc = try func.appendInst(b, ft, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(b, .{ .ret = acc });
    return func;
}

test "float-spill: fused multiply-add under register pressure falls back correctly (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = try buildFmaSumFunc(allocator, false);
    defer func.deinit();
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);

    const a_val: f32 = 2.0;
    const fargs = [_]u64{@as(u32, @bitCast(a_val))};
    const got_bits = harness.runCompiledFloat(std.testing.io, allocator, code, false, &fargs, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    const got: f32 = @bitCast(@as(u32, @truncate(got_bits)));
    // Reference: sum_i (a*i + a), accumulated in the same order (exact for small integer a).
    var want: f32 = a_val * 0.0 + a_val;
    var i: usize = 1;
    while (i < n_live) : (i += 1) want += a_val * @as(f32, @floatFromInt(i)) + a_val;
    try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(got)));
}

/// Build the et-soc SLP kernel `out[i] = a[i]*b[i] + a[i]` for i in 0..8: 8 scalar loads of `a`, 8
/// scalar loads of `b`, 8 `a[i]*b[i]` muls, 8 `mul + a[i]` adds, 8 stores. The first arith group
/// (`a[i]*b[i]`) packs two DIFFERENT freshly-loaded 8-wide operands, needing 16 live scalar floats
/// at once - the exact shape isel.zig's own note flags as previously failing with
/// `error.Unsupported`. Mirrors `buildSquareAddKernel` in etsoc_sysemu.zig but with distinct mul
/// operands so the pressure is real.
fn buildMulAddSelfKernel(func: *Function) !void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(b, ptr_t);
    const ptr_b = try func.appendBlockParam(b, ptr_t);
    const ptr_out = try func.appendBlockParam(b, ptr_t);

    var av: [8]Value = undefined;
    for (0..8) |i| {
        const addr_a = try func.appendArithImm(b, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_a } });
    }
    var bv: [8]Value = undefined;
    for (0..8) |i| {
        const addr_b = try func.appendArithImm(b, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_b } });
    }
    var mulv: [8]Value = undefined;
    for (0..8) |i| {
        mulv[i] = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = av[i], .rhs = bv[i] } });
    }
    var addv: [8]Value = undefined;
    for (0..8) |i| {
        addv[i] = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = mulv[i], .rhs = av[i] } });
    }
    for (0..8) |i| {
        const addr_out = try func.appendArithImm(b, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(b, addv[i], addr_out);
    }
    func.setTerminator(b, .{ .ret = null });
}

test "float-spill: et-soc VPU a[i]*b[i]+a[i] now compiles (previously error.Unsupported), runs on sw-sysemu when present" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMulAddSelfKernel(&func);

    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // SLP-vectorize the scalar kernel to 8-lane VPU arith.
    const changed = try mm.optimize(allocator, &func, model);
    try std.testing.expect(changed);
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    // The load-bearing assertion: this shape (16 live scalars during the a*b pack) used to fail
    // register allocation. Float spilling makes it compile.
    const code = try isel.selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);
    try std.testing.expect(code.len != 0);

    // If sw-sysemu is on PATH, execute it and check the result bit-for-bit; else skip the run.
    const in_a = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const in_b = [8]f32{ 10.5, 20.25, -3.5, 0.0, 100.0, -0.5, 42.0, 1000.0 };
    const lanes = etsoc.runVpuKernel(std.testing.io, allocator, code, in_a, in_b) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    for (0..8) |i| {
        const expected = in_a[i] * in_b[i] + in_a[i];
        try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(lanes[i])));
    }
}
