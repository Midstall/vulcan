//! End-to-end validation harness for the microarch optimizer pipeline (spec chunk 10, Task 3). This
//! is the capstone correctness oracle for chunks 1-9: it runs the FULL pipeline, `microarch.optimize`
//! (unroll, vectorize, prefetch, schedule, in that order) followed by the model-aware backend compile
//! that fires the loop-header alignment hook, against a plain compile of an identically-built,
//! untouched copy of the same function. Both are JIT'd in-process on this host and called with a
//! spread of inputs; any divergence is a miscompile somewhere in the pipeline, never a value to fudge.
//!
//! Convention (stated once, applies to every kernel below): the BASELINE gets neither the IR passes
//! nor the alignment hook (`isel.selectFunction`, today's plain compile). The TUNED copy gets both
//! (`microarch.optimize` then `isel.selectFunctionForModel`). Comparing the two is exactly "not
//! calling the optimizer" (today's behavior) vs "the whole pipeline wired up for ampere-altra".
//!
//! Cycle counts are also measured ON LINUX, via perf_event_open (PERF_COUNT_HW_CPU_CYCLES). That path
//! is Linux-only, so on any other OS the whole perf section is comptime-eliminated and only the
//! correctness differential runs (the minimal open/reset/read sequence mirrors the OS-guarded pattern
//! in tools/uarch-bench.zig; it is replicated rather than shared because that tool and this test live
//! in different modules). Cycle counts are NOT logged: printing from a unit test makes the test runner
//! surface spurious-looking output, and a human-readable cycle report belongs in the dedicated
//! benchmark (vulcan-uarch-bench). Here they are only checked against a loose sanity bound
//! (tuned < 10x baseline, see checkBound), to catch a gross pipeline regression, not to assert "never
//! slower": cycle counts jitter in a unit-test context, and a strict no-slower gate belongs in that
//! benchmark. Correctness (the JIT differential) is the hard assert.
//!
//! The correctness differential needs the aarch64 JIT backend, so it skips on other arches; the cycle
//! measurement additionally needs Linux, so it is skipped (not run) on non-Linux aarch64. Everything
//! skips cleanly so `zig build test` stays green on every host; on this dev Altra box it runs for real.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;
const linux = std.os.linux;

/// The model this harness tunes for: the dev box (Ampere Altra / Neoverse N1), a wide out-of-order
/// part with a non-trivial fetch_align, so both the IR passes and the alignment hook have real work
/// to consider.
fn ampere() *const opt.microarch.Model {
    return opt.microarch.modelFor(.@"ampere-altra");
}

/// Kernel 1: a dependent-mul chain, straight-line (no loop). `r = x; r = r*x` repeated 5 times, so
/// `f(x) = x^6`. Every multiply depends on the previous one: this is the shape that exercises
/// scheduling (arranging a dependency chain for the model's ports/latency) with nothing for
/// unroll/vectorize/prefetch to do (no loop, no independent lanes), so it also proves the pipeline is
/// a safe no-op-ish pass over straight-line code, not just loops.
fn buildMulChain(func: *Function) anyerror!void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i64_t);
    var r = x;
    var k: usize = 0;
    while (k < 5) : (k += 1) {
        r = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = r, .rhs = x } });
    }
    func.setTerminator(entry, .{ .ret = r });
}

/// Kernel 2: a counted sum-reduction loop, `s = 0; for (i = 0; i < n; i += 1) s += i; return s`. Two
/// carried values (i, s); the accumulator escapes and is read directly at the loop-exit block, the
/// same loop-closed-SSA shape proven under the unroller in unroll_differential.zig's `buildSum`, here
/// widened to i64. Exercises unroll (expose ILP across iterations) and schedule together.
fn buildSumLoop(func: *Function) anyerror!void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);
    const i = try func.appendBlockParam(loop, i64_t);
    const s = try func.appendBlockParam(loop, i64_t);
    const bi = try func.appendBlockParam(body, i64_t);
    const bs = try func.appendBlockParam(body, i64_t);
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done });
    const ns = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = bi } });
    const ni = try func.appendArithImm(body, i64_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    func.setTerminator(done, .{ .ret = s });
}

/// Kernel 3: a strided pointer-walking reduction over a real backing array. `s = 0; p = arr; for (i =
/// 0; i < n; i += 1) { s += *p; p += 8; } return s`. `p` is a pointer-typed loop-carried value that
/// steps by a constant 8-byte stride and is dereferenced each iteration: the same affine-strided-load
/// shape as prefetch_differential.zig's `buildStridedSum`, reused here because it's exactly the real
/// oracle Task 3 needs (the prefetch pass reads genuine loop/load structure, not a hand-placed hint).
/// Exercises prefetch + unroll + schedule together, over real memory.
fn buildStridedSum(func: *Function) anyerror!void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i64_t);
    const arr = try func.appendBlockParam(entry, ptr_t);
    const i = try func.appendBlockParam(loop, i64_t);
    const p = try func.appendBlockParam(loop, ptr_t);
    const s = try func.appendBlockParam(loop, i64_t);
    const bi = try func.appendBlockParam(body, i64_t);
    const bp = try func.appendBlockParam(body, ptr_t);
    const bs = try func.appendBlockParam(body, i64_t);

    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, arr, zero });

    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, p, s } }, .{ .target = done });

    const val = try func.appendInst(body, i64_t, .{ .load = .{ .ptr = bp } });
    const ns = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = val } });
    const np = try func.appendArithImm(body, ptr_t, .add, bp, 8);
    const ni = try func.appendArithImm(body, i64_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, np, ns });

    func.setTerminator(done, .{ .ret = s });
}

fn perfOpenCycles() !i32 {
    var attr = std.mem.zeroes(linux.perf_event_attr);
    attr.type = .HARDWARE;
    attr.size = @sizeOf(linux.perf_event_attr);
    attr.config = @intFromEnum(linux.PERF.COUNT.HW.CPU_CYCLES);
    attr.flags.disabled = true;
    attr.flags.exclude_kernel = true;
    attr.flags.exclude_hv = true;
    const rc = linux.perf_event_open(&attr, 0, -1, -1, 0);
    if (linux.errno(rc) != .SUCCESS) return error.PerfUnavailable;
    return @intCast(rc);
}

fn readCycles(fd: i32) u64 {
    var buf: [8]u8 = undefined;
    _ = linux.read(fd, &buf, 8);
    return std.mem.readInt(u64, &buf, builtin.cpu.arch.endian());
}

const outer_runs: usize = 7;
const hot_iters: u64 = 100_000;

/// Cycles-per-call for `f(arg)`, MIN over `outer_runs` hot loops of `hot_iters` calls each.
fn measureUnary(fd: i32, f: *const fn (i64) callconv(.c) i64, arg: i64) f64 {
    var best: f64 = std.math.inf(f64);
    var run: usize = 0;
    while (run < outer_runs) : (run += 1) {
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.RESET, 0);
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.ENABLE, 0);
        var acc: i64 = 0;
        var i: u64 = 0;
        while (i < hot_iters) : (i += 1) acc +%= f(arg);
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.DISABLE, 0);
        std.mem.doNotOptimizeAway(acc);
        const cycles = readCycles(fd);
        const cpi = @as(f64, @floatFromInt(cycles)) / @as(f64, @floatFromInt(hot_iters));
        if (cpi < best) best = cpi;
    }
    return best;
}

/// Cycles-per-call for `f(n, arr)`, MIN over `outer_runs` hot loops of `hot_iters` calls each.
fn measureStrided(fd: i32, f: *const fn (i64, [*]i64) callconv(.c) i64, n: i64, arr: [*]i64) f64 {
    var best: f64 = std.math.inf(f64);
    var run: usize = 0;
    while (run < outer_runs) : (run += 1) {
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.RESET, 0);
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.ENABLE, 0);
        var acc: i64 = 0;
        var i: u64 = 0;
        while (i < hot_iters) : (i += 1) acc +%= f(n, arr);
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.DISABLE, 0);
        std.mem.doNotOptimizeAway(acc);
        const cycles = readCycles(fd);
        const cpi = @as(f64, @floatFromInt(cycles)) / @as(f64, @floatFromInt(hot_iters));
        if (cpi < best) best = cpi;
    }
    return best;
}

/// Enforce only a very generous gross-regression bound (tuned not more than 10x baseline): enough to
/// catch a genuinely broken pipeline (a pass gone wrong, a bad alignment computation making the tuned
/// build wildly slower), not a precise performance claim. Cycle counts jitter under parallel test load
/// (scheduler noise, cache/TLB state, other tests sharing the core), so this bound is intentionally
/// loose and MUST NOT be tightened back down to chase a "faster" result here; a strict, repeatable
/// no-regression gate belongs in a dedicated benchmark (vulcan-uarch-bench), not a unit test. The
/// measured values are deliberately NOT printed (see the file header on why). Correctness (the JIT
/// differential above) is the hard gate; this is only a sanity check that nothing catastrophic happened.
fn checkBound(base_cpi: f64, tuned_cpi: f64) !void {
    try std.testing.expect(tuned_cpi < base_cpi * 10.0);
}

test "microarch e2e: dependent mul chain (straight-line, scheduling)" {
    if (comptime builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var baseline = Function.init(allocator);
    defer baseline.deinit();
    try buildMulChain(&baseline);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try buildMulChain(&tuned);
    _ = try opt.microarch.optimize(allocator, &tuned, ampere());

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    const base_code = try target.aarch64.isel.selectFunction(allocator, &baseline);
    defer allocator.free(base_code);
    const tuned_code = try target.aarch64.isel.selectFunctionForModel(allocator, &tuned, ampere());
    defer allocator.free(tuned_code);

    var base_buf = try target.aarch64.jit.CodeBuffer.map(std.mem.sliceAsBytes(base_code));
    defer base_buf.deinit();
    var tuned_buf = try target.aarch64.jit.CodeBuffer.map(std.mem.sliceAsBytes(tuned_code));
    defer tuned_buf.deinit();

    const Fn = *const fn (i64) callconv(.c) i64;
    const f_base = base_buf.entry(Fn, 0);
    const f_tuned = tuned_buf.entry(Fn, 0);

    // The fully-optimized, model-compiled build must compute exactly what the
    // untouched baseline computes, for every input. Any divergence is a miscompile in the pipeline;
    // do not fudge the expected value, go find the pass that broke it.
    const inputs = [_]i64{ -5, -3, -2, -1, 0, 1, 2, 3, 5, 7 };
    for (inputs) |x| try std.testing.expectEqual(f_base(x), f_tuned(x));

    // Cycle measurement is Linux-only (perf_event_open); the whole block is comptime-eliminated on
    // other OSes, where the correctness differential above is the entire test.
    if (comptime builtin.os.tag == .linux) {
        const fd = perfOpenCycles() catch return; // no perf access (permissions/sandbox): skip, not a failure
        defer _ = linux.close(fd);
        const base_cpi = measureUnary(fd, f_base, 5);
        const tuned_cpi = measureUnary(fd, f_tuned, 5);
        try checkBound(base_cpi, tuned_cpi);
    }
}

test "microarch e2e: counted sum-reduction loop (unroll + schedule)" {
    if (comptime builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var baseline = Function.init(allocator);
    defer baseline.deinit();
    try buildSumLoop(&baseline);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try buildSumLoop(&tuned);
    _ = try opt.microarch.optimize(allocator, &tuned, ampere());

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    const base_code = try target.aarch64.isel.selectFunction(allocator, &baseline);
    defer allocator.free(base_code);
    const tuned_code = try target.aarch64.isel.selectFunctionForModel(allocator, &tuned, ampere());
    defer allocator.free(tuned_code);

    var base_buf = try target.aarch64.jit.CodeBuffer.map(std.mem.sliceAsBytes(base_code));
    defer base_buf.deinit();
    var tuned_buf = try target.aarch64.jit.CodeBuffer.map(std.mem.sliceAsBytes(tuned_code));
    defer tuned_buf.deinit();

    const Fn = *const fn (i64) callconv(.c) i64;
    const f_base = base_buf.entry(Fn, 0);
    const f_tuned = tuned_buf.entry(Fn, 0);

    // Covers n = 0 (loop never taken), small n, and larger n (several unroll
    // iterations of the tuned build's factor).
    const inputs = [_]i64{ 0, 1, 2, 3, 7, 16, 100, 1000 };
    for (inputs) |n| try std.testing.expectEqual(f_base(n), f_tuned(n));

    // Cycle measurement is Linux-only (perf_event_open); the whole block is comptime-eliminated on
    // other OSes, where the correctness differential above is the entire test.
    if (comptime builtin.os.tag == .linux) {
        const fd = perfOpenCycles() catch return; // no perf access (permissions/sandbox): skip, not a failure
        defer _ = linux.close(fd);
        const base_cpi = measureUnary(fd, f_base, 1000);
        const tuned_cpi = measureUnary(fd, f_tuned, 1000);
        try checkBound(base_cpi, tuned_cpi);
    }
}

test "microarch e2e: strided pointer-walking reduction over a real array (unroll + prefetch + schedule)" {
    if (comptime builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var baseline = Function.init(allocator);
    defer baseline.deinit();
    try buildStridedSum(&baseline);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try buildStridedSum(&tuned);
    _ = try opt.microarch.optimize(allocator, &tuned, ampere());

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    const base_code = try target.aarch64.isel.selectFunction(allocator, &baseline);
    defer allocator.free(base_code);
    const tuned_code = try target.aarch64.isel.selectFunctionForModel(allocator, &tuned, ampere());
    defer allocator.free(tuned_code);

    var base_buf = try target.aarch64.jit.CodeBuffer.map(std.mem.sliceAsBytes(base_code));
    defer base_buf.deinit();
    var tuned_buf = try target.aarch64.jit.CodeBuffer.map(std.mem.sliceAsBytes(tuned_code));
    defer tuned_buf.deinit();

    const Fn = *const fn (i64, [*]i64) callconv(.c) i64;
    const f_base = base_buf.entry(Fn, 0);
    const f_tuned = tuned_buf.entry(Fn, 0);

    // A real backing array, long enough for the largest n exercised below (128).
    var backing: [128]i64 = undefined;
    for (&backing, 0..) |*v, idx| v.* = @as(i64, @intCast(idx)) * 3 - 17; // an arbitrary, non-trivial pattern

    const inputs = [_]i64{ 0, 1, 2, 5, 16, 64, 128 };
    for (inputs) |n| try std.testing.expectEqual(f_base(n, &backing), f_tuned(n, &backing));

    // Cycle measurement is Linux-only (perf_event_open); the whole block is comptime-eliminated on
    // other OSes, where the correctness differential above is the entire test.
    if (comptime builtin.os.tag == .linux) {
        const fd = perfOpenCycles() catch return; // no perf access (permissions/sandbox): skip, not a failure
        defer _ = linux.close(fd);
        const base_cpi = measureStrided(fd, f_base, 128, &backing);
        const tuned_cpi = measureStrided(fd, f_tuned, 128, &backing);
        try checkBound(base_cpi, tuned_cpi);
    }
}
