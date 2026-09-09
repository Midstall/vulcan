//! Execution tests for the GPU offload pass. Each test builds a kernel, lowers it to a host
//! loop nest with `vulcan-gpu.lowerToLoopNest`, JITs the result for the host CPU, calls it over
//! a real buffer, and asserts every element of that buffer.
//!
//! This is the point where a kernel stops being a shape and becomes an answer. Every other
//! kernel test in this repository counts opcodes or walks the IR, so a nest that computes the
//! wrong index looks correct to them. These tests run the grid and read the memory back, so a
//! wrong index, a wrong bound, or a lost branch shows up as wrong numbers. The result is the
//! reference a GPU backend gets checked against.
//!
//! The nest the pass builds is `for (block_id in 0..grid_x) for (thread_id in 0..block[0])`,
//! with `gid = block_id * block[0] + thread_id`. So a launch runs `grid_x * block[0]` threads,
//! with `gid` covering `0..grid_x * block[0] - 1` once each. The bound tests below pin that.
//!
//! Runs where the native JIT has a backend for the host architecture, and skips elsewhere.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");
const target = @import("vulcan-target");

const Function = ir.function.Function;

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

    const off = try func.appendArithImm(entry, i32_t, .mul, gid, elem_bytes);
    const slot = try func.appendInst(entry, ptr_t, .{
        .arith = .{ .op = .add, .lhs = buf, .rhs = off },
    });
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

    const off = try func.appendArithImm(entry, i32_t, .mul, gid, elem_bytes);
    const slot = try func.appendInst(entry, ptr_t, .{
        .arith = .{ .op = .add, .lhs = buf, .rhs = off },
    });
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

/// The signature of a lowered one-buffer kernel: the grid size in workgroups, then the
/// kernel's real parameters. The builtin parameter is gone, because the nest computes it.
const GridFn = *const fn (grid_x: i32, buf: [*]i32) callconv(.c) void;

/// The signature of the lowered guarded kernel.
const GuardedFn = *const fn (grid_x: i32, n: i32, out: [*]i32, trace: [*]i32) callconv(.c) void;

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
    code.entry(GridFn, 0)(5, &buf);

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
    code.entry(GridFn, 0)(4, &buf);

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
    code.entry(GuardedFn, 0)(3, 10, &out, &trace);

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
    code.entry(GuardedFn, 0)(2, 10, &out, &trace);

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
    code.entry(GuardedFn, 0)(0, 10, &out, &trace);

    const untouched_out: [20]i32 = @splat(-1);
    const untouched_trace: [20]i32 = @splat(0);
    try std.testing.expectEqualSlices(i32, &untouched_out, &out);
    try std.testing.expectEqualSlices(i32, &untouched_trace, &trace);
}
