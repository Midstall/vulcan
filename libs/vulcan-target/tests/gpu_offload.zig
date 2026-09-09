//! Execution tests for the GPU offload pass. Each test builds a kernel, lowers it to a host
//! loop nest with `vulcan-gpu.lowerToLoopNest`, JITs the result for the host CPU, calls it over
//! a real buffer, and asserts every element of that buffer.
//!
//! This is the point where a kernel stops being a shape and becomes an answer. Every other
//! kernel test in this repository counts opcodes or walks the IR, so a nest that computes the
//! wrong index looks correct to them. These tests run the grid and read the memory back, so a
//! wrong index, a wrong bound, a swapped axis, or a lost branch shows up as wrong numbers. The
//! result is the reference a GPU backend gets checked against.
//!
//! The nest the pass builds is six loops deep:
//!
//! ```
//! for (block_id_z in 0..grid_z)
//!   for (block_id_y in 0..grid_y)
//!     for (block_id_x in 0..grid_x)
//!       for (thread_id_z in 0..block[2])
//!         for (thread_id_y in 0..block[1])
//!           for (thread_id_x in 0..block[0]) body
//! ```
//!
//! So a launch runs `grid_x * grid_y * grid_z * block[0] * block[1] * block[2]` threads, with
//! `global_id_a = block_id_a * block_dim_a + thread_id_a` on each axis, and it visits x fastest
//! and z slowest. The bound tests and the order test below pin all of that.
//!
//! Runs where the native JIT has a backend for the host architecture, and skips elsewhere.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");
const target = @import("vulcan-target");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Type = ir.types.Type;

const i32_kind: ir.types.TypeKind = .{ .int = .{ .signedness = .signed, .bits = 32 } };

/// Whether the native JIT has a backend for the host architecture.
fn hasJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .x86, .riscv64 => true,
        else => false,
    };
}

/// The size in bytes of one `i32` element, which is the stride of every buffer here.
const elem_bytes: i64 = 4;

/// The address of the element `index` of `base`, where `index` is a runtime value.
fn elemPtr(
    func: *Function,
    block: Block,
    ptr_t: Type,
    i32_t: Type,
    base: Value,
    index: Value,
) !Value {
    const off = try func.appendArithImm(block, i32_t, .mul, index, elem_bytes);
    return func.appendInst(block, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = off } });
}

/// The address of the element `index` of `base`, where `index` is known when the kernel is
/// built.
fn constElemPtr(
    func: *Function,
    block: Block,
    ptr_t: Type,
    i32_t: Type,
    base: Value,
    index: i64,
) !Value {
    const off = try func.appendInst(block, i32_t, .{ .iconst = index * elem_bytes });
    return func.appendInst(block, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = off } });
}

/// The kernel `buf[gid] = gid * 3`, where `gid` is `global_id_x` and `buf` is the only real
/// parameter. `block` becomes the declared workgroup size, which the caller reads back with
/// `attrs.localSize` and hands to the pass.
fn tripleKernel(allocator: std.mem.Allocator, block: [3]u32) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const gid = try func.appendBlockParam(entry, i32_t);
    const buf = try func.appendBlockParam(entry, ptr_t);
    try gpu.attrs.setBuiltin(&func, gid, .global_id_x);
    try gpu.attrs.setLocalSize(&func, block);

    const slot = try elemPtr(&func, entry, ptr_t, i32_t, buf, gid);
    const tripled = try func.appendArithImm(entry, i32_t, .mul, gid, 3);
    try func.appendStore(entry, tripled, slot);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// The kernel `buf[gid] = if (gid & 1 != 0) gid else 0`. The two arms pass their value to a
/// merge block as a block argument, so the body is a diamond of four blocks. The splice must
/// keep every edge and every block argument, or the merge stores the wrong value.
fn parityKernel(allocator: std.mem.Allocator, block: [3]u32) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const then_block = try func.appendBlock();
    const else_block = try func.appendBlock();
    const merge = try func.appendBlock();

    const gid = try func.appendBlockParam(entry, i32_t);
    const buf = try func.appendBlockParam(entry, ptr_t);
    try gpu.attrs.setBuiltin(&func, gid, .global_id_x);
    try gpu.attrs.setLocalSize(&func, block);

    const slot = try elemPtr(&func, entry, ptr_t, i32_t, buf, gid);
    const low_bit = try func.appendArithImm(entry, i32_t, .bit_and, gid, 1);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const is_odd = try func.appendInst(entry, bool_t, .{
        .icmp = .{ .op = .ne, .lhs = low_bit, .rhs = zero },
    });
    try func.appendIf(
        entry,
        is_odd,
        .{ .target = then_block, .args = &.{gid} },
        .{ .target = else_block, .args = &.{zero} },
    );

    const then_value = try func.appendBlockParam(then_block, i32_t);
    try func.setJump(then_block, merge, &.{then_value});
    const else_value = try func.appendBlockParam(else_block, i32_t);
    try func.setJump(else_block, merge, &.{else_value});

    const merged = try func.appendBlockParam(merge, i32_t);
    try func.appendStore(merge, merged, slot);
    func.setTerminator(merge, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// The kernel `trace[gid] = 1; if (gid < n) out[gid] = gid * 3;`.
///
/// The guard is what a real kernel writes when the data length is not a multiple of the
/// workgroup size: the launch rounds the thread count up, and the extra threads must do
/// nothing. The unconditional store to `trace` records which threads the nest really started,
/// so the test can tell "the nest ran too few threads" apart from "the guard rejected them".
fn guardedKernel(allocator: std.mem.Allocator, block: [3]u32) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const store_block = try func.appendBlock();
    const done = try func.appendBlock();

    const gid = try func.appendBlockParam(entry, i32_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const out = try func.appendBlockParam(entry, ptr_t);
    const trace = try func.appendBlockParam(entry, ptr_t);
    try gpu.attrs.setBuiltin(&func, gid, .global_id_x);
    try gpu.attrs.setLocalSize(&func, block);

    const off = try func.appendArithImm(entry, i32_t, .mul, gid, elem_bytes);
    const trace_slot = try func.appendInst(entry, ptr_t, .{
        .arith = .{ .op = .add, .lhs = trace, .rhs = off },
    });
    const one = try func.appendInst(entry, i32_t, .{ .iconst = 1 });
    try func.appendStore(entry, one, trace_slot);
    const in_range = try func.appendInst(entry, bool_t, .{
        .icmp = .{ .op = .lt, .lhs = gid, .rhs = n },
    });
    try func.appendIf(entry, in_range, .{ .target = store_block }, .{ .target = done });

    const out_slot = try func.appendInst(store_block, ptr_t, .{
        .arith = .{ .op = .add, .lhs = out, .rhs = off },
    });
    const tripled = try func.appendArithImm(store_block, i32_t, .mul, gid, 3);
    try func.appendStore(store_block, tripled, out_slot);
    try func.setJump(store_block, done, &.{});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// The kernel `buf[gz * ex * ey + gy * ex + gx] = gx * 100 + gy * 10 + gz`, where the three
/// coordinates are `global_id_x`, `global_id_y` and `global_id_z`.
///
/// `ex` and `ey` are the x and y extents of the launch in threads, so the index is the usual
/// row-major linear address of the thread's own point in the grid. Each thread owns exactly one
/// element and writes a number that names all three of its coordinates, so a swapped axis, a
/// repeated thread, and a missing thread are each visible per element.
fn coordKernel(allocator: std.mem.Allocator, block: [3]u32, ex: i64, ey: i64) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const gx = try func.appendBlockParam(entry, i32_t);
    const gy = try func.appendBlockParam(entry, i32_t);
    const gz = try func.appendBlockParam(entry, i32_t);
    const buf = try func.appendBlockParam(entry, ptr_t);
    try gpu.attrs.setBuiltin(&func, gx, .global_id_x);
    try gpu.attrs.setBuiltin(&func, gy, .global_id_y);
    try gpu.attrs.setBuiltin(&func, gz, .global_id_z);
    try gpu.attrs.setLocalSize(&func, block);

    const plane = try func.appendArithImm(entry, i32_t, .mul, gz, ex * ey);
    const row = try func.appendArithImm(entry, i32_t, .mul, gy, ex);
    const partial = try func.appendInst(entry, i32_t, .{
        .arith = .{ .op = .add, .lhs = plane, .rhs = row },
    });
    const linear = try func.appendInst(entry, i32_t, .{
        .arith = .{ .op = .add, .lhs = partial, .rhs = gx },
    });
    const slot = try elemPtr(&func, entry, ptr_t, i32_t, buf, linear);

    const hundreds = try func.appendArithImm(entry, i32_t, .mul, gx, 100);
    const tens = try func.appendArithImm(entry, i32_t, .mul, gy, 10);
    const upper = try func.appendInst(entry, i32_t, .{
        .arith = .{ .op = .add, .lhs = hundreds, .rhs = tens },
    });
    const coded = try func.appendInst(entry, i32_t, .{
        .arith = .{ .op = .add, .lhs = upper, .rhs = gz },
    });
    try func.appendStore(entry, coded, slot);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// The kernel that writes its own thread index and its own workgroup index, each coded as
/// `x * 100 + y * 10 + z`, to `buf[linear]` and `buf[half + linear]`. `linear` is the same
/// row-major address `coordKernel` uses, so the two halves of the buffer say which thread of
/// which workgroup landed on each point of the grid.
fn idKernel(allocator: std.mem.Allocator, block: [3]u32, ex: i64, ey: i64, half: i64) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const gx = try func.appendBlockParam(entry, i32_t);
    const gy = try func.appendBlockParam(entry, i32_t);
    const gz = try func.appendBlockParam(entry, i32_t);
    const tx = try func.appendBlockParam(entry, i32_t);
    const ty = try func.appendBlockParam(entry, i32_t);
    const tz = try func.appendBlockParam(entry, i32_t);
    const bx = try func.appendBlockParam(entry, i32_t);
    const by = try func.appendBlockParam(entry, i32_t);
    const bz = try func.appendBlockParam(entry, i32_t);
    const buf = try func.appendBlockParam(entry, ptr_t);
    try gpu.attrs.setBuiltin(&func, gx, .global_id_x);
    try gpu.attrs.setBuiltin(&func, gy, .global_id_y);
    try gpu.attrs.setBuiltin(&func, gz, .global_id_z);
    try gpu.attrs.setBuiltin(&func, tx, .thread_id_x);
    try gpu.attrs.setBuiltin(&func, ty, .thread_id_y);
    try gpu.attrs.setBuiltin(&func, tz, .thread_id_z);
    try gpu.attrs.setBuiltin(&func, bx, .block_id_x);
    try gpu.attrs.setBuiltin(&func, by, .block_id_y);
    try gpu.attrs.setBuiltin(&func, bz, .block_id_z);
    try gpu.attrs.setLocalSize(&func, block);

    const plane = try func.appendArithImm(entry, i32_t, .mul, gz, ex * ey);
    const row = try func.appendArithImm(entry, i32_t, .mul, gy, ex);
    const partial = try func.appendInst(entry, i32_t, .{
        .arith = .{ .op = .add, .lhs = plane, .rhs = row },
    });
    const linear = try func.appendInst(entry, i32_t, .{
        .arith = .{ .op = .add, .lhs = partial, .rhs = gx },
    });
    const shifted = try func.appendArithImm(entry, i32_t, .add, linear, half);

    const thread_code = try code3(&func, entry, i32_t, tx, ty, tz);
    const thread_slot = try elemPtr(&func, entry, ptr_t, i32_t, buf, linear);
    try func.appendStore(entry, thread_code, thread_slot);

    const block_code = try code3(&func, entry, i32_t, bx, by, bz);
    const block_slot = try elemPtr(&func, entry, ptr_t, i32_t, buf, shifted);
    try func.appendStore(entry, block_code, block_slot);

    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// `x * 100 + y * 10 + z`, which packs a three-axis index into one number a test can read.
fn code3(func: *Function, block: Block, i32_t: Type, x: Value, y: Value, z: Value) !Value {
    const hundreds = try func.appendArithImm(block, i32_t, .mul, x, 100);
    const tens = try func.appendArithImm(block, i32_t, .mul, y, 10);
    const upper = try func.appendInst(block, i32_t, .{
        .arith = .{ .op = .add, .lhs = hundreds, .rhs = tens },
    });
    return func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = upper, .rhs = z } });
}

/// The kernel that writes the six size builtins to `buf[0..5]` and counts itself into
/// `buf[6]`. Every thread writes the same six numbers, because a size is uniform over the
/// launch, and every thread adds one to the count, so the count is the number of threads the
/// nest really started.
fn dimKernel(allocator: std.mem.Allocator, block: [3]u32) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const sizes = [_]gpu.Builtin{
        .block_dim_x, .block_dim_y, .block_dim_z,
        .grid_dim_x,  .grid_dim_y,  .grid_dim_z,
    };
    var values: [sizes.len]Value = undefined;
    for (&values, sizes) |*slot, tag| {
        slot.* = try func.appendBlockParam(entry, i32_t);
        try gpu.attrs.setBuiltin(&func, slot.*, tag);
    }
    const buf = try func.appendBlockParam(entry, ptr_t);
    try gpu.attrs.setLocalSize(&func, block);

    for (values, 0..) |v, i| {
        const slot = try constElemPtr(&func, entry, ptr_t, i32_t, buf, @intCast(i));
        try func.appendStore(entry, v, slot);
    }

    const count_slot = try constElemPtr(&func, entry, ptr_t, i32_t, buf, sizes.len);
    const count = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = count_slot } });
    const next = try func.appendArithImm(entry, i32_t, .add, count, 1);
    try func.appendStore(entry, next, count_slot);

    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// The kernel that appends its own `global_id` triple to `out` at a running counter kept in
/// `counter[0]`, then advances the counter.
///
/// The nest runs one thread at a time, so the counter needs no atomic and the buffer ends up
/// holding the visitation ORDER rather than a per-thread answer. That is the only way to see
/// which of the six loops runs fastest.
fn traceKernel(allocator: std.mem.Allocator, block: [3]u32) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const gx = try func.appendBlockParam(entry, i32_t);
    const gy = try func.appendBlockParam(entry, i32_t);
    const gz = try func.appendBlockParam(entry, i32_t);
    const out = try func.appendBlockParam(entry, ptr_t);
    const counter = try func.appendBlockParam(entry, ptr_t);
    try gpu.attrs.setBuiltin(&func, gx, .global_id_x);
    try gpu.attrs.setBuiltin(&func, gy, .global_id_y);
    try gpu.attrs.setBuiltin(&func, gz, .global_id_z);
    try gpu.attrs.setLocalSize(&func, block);

    const seq = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = counter } });
    const base = try func.appendArithImm(entry, i32_t, .mul, seq, 3);

    const coords = [_]Value{ gx, gy, gz };
    for (coords, 0..) |v, i| {
        const index = try func.appendArithImm(entry, i32_t, .add, base, @intCast(i));
        const slot = try elemPtr(&func, entry, ptr_t, i32_t, out, index);
        try func.appendStore(entry, v, slot);
    }

    const next = try func.appendArithImm(entry, i32_t, .add, seq, 1);
    try func.appendStore(entry, next, counter);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// The signature of a lowered one-buffer kernel: the grid size in workgroups on all three
/// axes, then the kernel's real parameters. The builtin parameters are gone, because the nest
/// computes them.
const GridFn = *const fn (grid_x: i32, grid_y: i32, grid_z: i32, buf: [*]i32) callconv(.c) void;

/// The signature of the lowered guarded kernel.
const GuardedFn = *const fn (
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    n: i32,
    out: [*]i32,
    trace: [*]i32,
) callconv(.c) void;

/// The signature of the lowered trace kernel.
const TraceFn = *const fn (
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    out: [*]i32,
    counter: *i32,
) callconv(.c) void;

test "offload: a lowered kernel runs the whole grid and writes buf[gid] = gid * 3" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var kernel = try tripleKernel(allocator, .{ 4, 1, 1 });
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    // 5 workgroups of 4 threads is 20 threads, with gid 0..19. The buffer has 4 spare slots
    // that no thread may touch, so an overrun fails here too.
    var buf: [24]i32 = @splat(-1);
    code.entry(GridFn, 0)(5, 1, 1, &buf);

    const want: [24]i32 = .{
        0,  3,  6,  9,  12, 15, 18, 21,
        24, 27, 30, 33, 36, 39, 42, 45,
        48, 51, 54, 57, -1, -1, -1, -1,
    };
    try std.testing.expectEqualSlices(i32, &want, &buf);
}

test "offload: a multi-block kernel body keeps its control flow when it runs" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var kernel = try parityKernel(allocator, .{ 4, 1, 1 });
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    // 4 workgroups of 4 threads is 16 threads, with gid 0..15. An odd thread stores its own
    // index and an even one stores 0, so a lost arm or a lost block argument is visible per
    // element rather than as one wrong total.
    var buf: [20]i32 = @splat(-1);
    code.entry(GridFn, 0)(4, 1, 1, &buf);

    const want: [20]i32 = .{
        0, 1,  0, 3,  0, 5,  0,  7,  0,  9,
        0, 11, 0, 13, 0, 15, -1, -1, -1, -1,
    };
    try std.testing.expectEqualSlices(i32, &want, &buf);
}

test "offload: 3 workgroups of 4 run exactly 12 threads over 10 elements" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var kernel = try guardedKernel(allocator, .{ 4, 1, 1 });
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    // 10 elements do not fill a workgroup of 4, so the launch takes 3 workgroups. The nest
    // runs `grid_x * block[0]` threads, which is 3 * 4 = 12, not 10 and not 16. Threads 0..9
    // pass the guard and write, threads 10 and 11 start and do nothing.
    var out: [20]i32 = @splat(-1);
    var trace: [20]i32 = @splat(0);
    code.entry(GuardedFn, 0)(3, 1, 1, 10, &out, &trace);

    const want_out: [20]i32 = .{
        0,  3,  6,  9,  12, 15, 18, 21, 24, 27,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    };
    const want_trace: [20]i32 = .{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    try std.testing.expectEqualSlices(i32, &want_out, &out);
    try std.testing.expectEqualSlices(i32, &want_trace, &trace);
}

test "offload: 2 workgroups of 8 run exactly 16 threads over the same 10 elements" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var kernel = try guardedKernel(allocator, .{ 8, 1, 1 });
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    // The same 10 elements with a workgroup of 8 take 2 workgroups, so 2 * 8 = 16 threads.
    // The output is identical to the 3-by-4 launch, but 4 more threads start. That is what
    // separates the thread count from the answer.
    var out: [20]i32 = @splat(-1);
    var trace: [20]i32 = @splat(0);
    code.entry(GuardedFn, 0)(2, 1, 1, 10, &out, &trace);

    const want_out: [20]i32 = .{
        0,  3,  6,  9,  12, 15, 18, 21, 24, 27,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    };
    const want_trace: [20]i32 = .{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    };
    try std.testing.expectEqualSlices(i32, &want_out, &out);
    try std.testing.expectEqualSlices(i32, &want_trace, &trace);
}

test "offload: an empty grid starts no threads at all" {
    // The lower bound of the same off-by-one. `block_id < grid_x` is false on the first test,
    // so the nest must leave every byte alone.
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var kernel = try guardedKernel(allocator, .{ 4, 1, 1 });
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    var out: [20]i32 = @splat(-1);
    var trace: [20]i32 = @splat(0);
    code.entry(GuardedFn, 0)(0, 1, 1, 10, &out, &trace);

    const untouched_out: [20]i32 = @splat(-1);
    const untouched_trace: [20]i32 = @splat(0);
    try std.testing.expectEqualSlices(i32, &untouched_out, &out);
    try std.testing.expectEqualSlices(i32, &untouched_trace, &trace);
}

test "offload: a 3-D launch gives every thread its own global_id triple" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Workgroup 2 by 3 by 1, grid 3 by 1 by 2. The three extents in threads are 6, 3 and 2,
    // which are all different, so a lowering that swapped two axes cannot land on the same
    // buffer by symmetry. The x axis gets 3 workgroups of 2, the y axis gets 1 workgroup of 3,
    // and the z axis gets 2 workgroups of 1.
    var kernel = try coordKernel(allocator, .{ 2, 3, 1 }, 6, 3);
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    var buf: [40]i32 = @splat(-1);
    code.entry(GridFn, 0)(3, 1, 2, &buf);

    // buf[gz * 18 + gy * 6 + gx] = gx * 100 + gy * 10 + gz, for gx in 0..5, gy in 0..2 and
    // gz in 0..1. The last 4 slots belong to no thread.
    const want: [40]i32 = .{
        0,  100, 200, 300, 400, 500,
        10, 110, 210, 310, 410, 510,
        20, 120, 220, 320, 420, 520,
        1,  101, 201, 301, 401, 501,
        11, 111, 211, 311, 411, 511,
        21, 121, 221, 321, 421, 521,
        -1, -1,  -1,  -1,
    };
    try std.testing.expectEqualSlices(i32, &want, &buf);
}

test "offload: global_id_y and global_id_z use their own axis, not the x axis" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // The roles of the previous launch are rotated: now the y axis is the one with several
    // workgroups AND several threads, the x axis is workgroups only, and the z axis is threads
    // only. Between the two tests every axis gets exercised on both of its loops. The extents
    // in threads are 2, 6 and 3, again all different.
    var kernel = try coordKernel(allocator, .{ 1, 2, 3 }, 2, 6);
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    var buf: [40]i32 = @splat(-1);
    code.entry(GridFn, 0)(2, 3, 1, &buf);

    // buf[gz * 12 + gy * 2 + gx] = gx * 100 + gy * 10 + gz, for gx in 0..1, gy in 0..5 and
    // gz in 0..2.
    const want: [40]i32 = .{
        0,  100, 10, 110, 20, 120, 30, 130, 40, 140, 50, 150,
        1,  101, 11, 111, 21, 121, 31, 131, 41, 141, 51, 151,
        2,  102, 12, 112, 22, 122, 32, 132, 42, 142, 52, 152,
        -1, -1,  -1, -1,
    };
    try std.testing.expectEqualSlices(i32, &want, &buf);
}

test "offload: each grid point names the thread and the workgroup that reached it" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // The same launch as the first 3-D test: workgroup 2 by 3 by 1 over a grid of 3 by 1 by 2,
    // so the extents in threads are 6, 3 and 2. The first half of the buffer holds the thread
    // index of the thread that reached each point, the second half its workgroup index. That
    // splits `global_id` back into the two values it fuses, so a thread index that leaked into
    // a workgroup index, or an axis that took another axis's induction variable, is visible.
    var kernel = try idKernel(allocator, .{ 2, 3, 1 }, 6, 3, 36);
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    var buf: [76]i32 = @splat(-1);
    code.entry(GridFn, 0)(3, 1, 2, &buf);

    // Thread index, coded as tx * 100 + ty * 10 + tz. Along a row of 6 the x index alternates
    // 0 and 1, because a workgroup is 2 threads wide. The y index is the row, and the z index
    // is always 0, because a workgroup is 1 thread deep. The second plane repeats the first:
    // the thread index says nothing about which workgroup it is in.
    const want_thread: [36]i32 = .{
        0,  100, 0,  100, 0,  100,
        10, 110, 10, 110, 10, 110,
        20, 120, 20, 120, 20, 120,
        0,  100, 0,  100, 0,  100,
        10, 110, 10, 110, 10, 110,
        20, 120, 20, 120, 20, 120,
    };
    // Workgroup index, coded as bx * 100 + by * 10 + bz. Along a row of 6 the x index steps
    // every 2 threads. The y index is always 0, because the grid is 1 workgroup tall, and the
    // z index is the plane.
    const want_block: [36]i32 = .{
        0, 0, 100, 100, 200, 200,
        0, 0, 100, 100, 200, 200,
        0, 0, 100, 100, 200, 200,
        1, 1, 101, 101, 201, 201,
        1, 1, 101, 101, 201, 201,
        1, 1, 101, 101, 201, 201,
    };
    const want_spare: [4]i32 = .{ -1, -1, -1, -1 };
    try std.testing.expectEqualSlices(i32, &want_thread, buf[0..36]);
    try std.testing.expectEqualSlices(i32, &want_block, buf[36..72]);
    try std.testing.expectEqualSlices(i32, &want_spare, buf[72..76]);
}

test "offload: the six size builtins report the launch shape and the whole nest runs" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Six different numbers, so no two of the six builtins can be confused for each other.
    var kernel = try dimKernel(allocator, .{ 2, 3, 4 });
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    var buf: [10]i32 = @splat(0);
    code.entry(GridFn, 0)(5, 6, 7, &buf);

    // The launch runs 2 * 3 * 4 * 5 * 6 * 7 = 5040 threads, so a loop with the wrong bound
    // shows up in the count as well as in the six sizes.
    const want: [10]i32 = .{ 2, 3, 4, 5, 6, 7, 5040, 0, 0, 0 };
    try std.testing.expectEqualSlices(i32, &want, &buf);
}

test "offload: the nest visits x fastest, then y, then z" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Each thread appends its own global_id triple at a running counter, so the buffer holds
    // the ORDER the nest visited the grid in and not a per-thread answer. Workgroup 2 by 1 by
    // 1 over a grid of 2 by 2 by 2 gives 16 threads, and every axis moves, so any other
    // ordering of the six loops writes a different sequence.
    var kernel = try traceKernel(allocator, .{ 2, 1, 1 });
    defer kernel.deinit();
    var lowered = try gpu.lowerToLoopNest(allocator, &kernel, gpu.attrs.localSize(&kernel));
    defer lowered.deinit();

    var code = try target.native.jitFunction(allocator, &lowered);
    defer code.deinit();

    var trace: [48]i32 = @splat(-1);
    var counter: i32 = 0;
    code.entry(TraceFn, 0)(2, 2, 2, &trace, &counter);

    // x runs through 0, 1, 2, 3 before y moves, and y runs through 0 and 1 before z moves.
    const want: [48]i32 = .{
        0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0,
        0, 1, 0, 1, 1, 0, 2, 1, 0, 3, 1, 0,
        0, 0, 1, 1, 0, 1, 2, 0, 1, 3, 0, 1,
        0, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1,
    };
    try std.testing.expectEqualSlices(i32, &want, &trace);
    // Every thread appended exactly once, so the counter is the thread count.
    try std.testing.expectEqual(@as(i32, 16), counter);
}
