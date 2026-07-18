//! Pins the ampere-altra model latencies to a fresh on-host measurement of the real silicon. The
//! Ampere Altra's microarchitecture is Neoverse N1, shared by both the Altra M and Q model families
//! (M128 etc are models, not microarchitectures: M is the family, 128 the core count), so the model is
//! one set of N1 latencies. Skips unless the detected host is an Ampere Altra; only built on Linux
//! (it uses perf_event_open). Measures the hardware cycle counter over dependency chains and
//! independent-op streams.

const std = @import("std");
const builtin = @import("builtin");
const registry = @import("registry.zig");
const linux = std.os.linux;

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

fn measureMulLatencyCycles() !f64 {
    const fd = try perfOpenCycles();
    defer _ = linux.close(fd);
    const loops: u64 = 2_000_000;
    const unroll: u64 = 16;
    var best: f64 = std.math.inf(f64);
    var run: usize = 0;
    while (run < 7) : (run += 1) {
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.RESET, 0);
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.ENABLE, 0);
        var x: u64 = 1;
        const one: u64 = 1;
        var i: u64 = 0;
        while (i < loops) : (i += 1) {
            asm volatile (
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                \\mul %[x], %[x], %[one]
                : [x] "+r" (x),
                : [one] "r" (one),
            );
        }
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.DISABLE, 0);
        std.mem.doNotOptimizeAway(x);
        var buf: [8]u8 = undefined;
        _ = linux.read(fd, &buf, 8);
        const cycles: u64 = std.mem.readInt(u64, &buf, builtin.cpu.arch.endian());
        const cpo = @as(f64, @floatFromInt(cycles)) / @as(f64, @floatFromInt(loops * unroll));
        if (cpo < best) best = cpo;
    }
    return best;
}

// Reciprocal THROUGHPUT of the f32 MULTIPLY: cycles between two back-to-back INDEPENDENT f32 muls.
// This, not the integer mul below, is the multiply the SLP cost model weights on ampere, because on
// a NEON core the vectorizer only ever fuses f32 groups (there is no <N x i32> lowering, so the
// integer SLP path is gated off, see vectorize.runModel's allow_i32). The N1's FP/NEON multiplier is
// fully pipelined, so eight INDEPENDENT f32 muls (different destinations, one shared invariant
// source, round-robin, unrolled 16 deep) issue at well under one cycle each even though each result
// takes ~3-4 cycles. The `:s` modifier names each `w`-class operand as its `sN` scalar-f32 register.
fn measureF32MulThroughputCycles() !f64 {
    const fd = try perfOpenCycles();
    defer _ = linux.close(fd);
    const loops: u64 = 2_000_000;
    const unroll: u64 = 16;
    var best: f64 = std.math.inf(f64);
    var run: usize = 0;
    while (run < 7) : (run += 1) {
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.RESET, 0);
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.ENABLE, 0);
        // Eight independent accumulators, all just above 1.0 so repeated `*1.0`... actually `* one`
        // keeps them finite and non-denormal; the value is irrelevant, only the mul stream is timed.
        var a0: f32 = 1.001;
        var a1: f32 = 1.001;
        var a2: f32 = 1.001;
        var a3: f32 = 1.001;
        var a4: f32 = 1.001;
        var a5: f32 = 1.001;
        var a6: f32 = 1.001;
        var a7: f32 = 1.001;
        const one: f32 = 1.0;
        var i: u64 = 0;
        while (i < loops) : (i += 1) {
            asm volatile (
                \\fmul %[a0:s], %[a0:s], %[one:s]
                \\fmul %[a1:s], %[a1:s], %[one:s]
                \\fmul %[a2:s], %[a2:s], %[one:s]
                \\fmul %[a3:s], %[a3:s], %[one:s]
                \\fmul %[a4:s], %[a4:s], %[one:s]
                \\fmul %[a5:s], %[a5:s], %[one:s]
                \\fmul %[a6:s], %[a6:s], %[one:s]
                \\fmul %[a7:s], %[a7:s], %[one:s]
                \\fmul %[a0:s], %[a0:s], %[one:s]
                \\fmul %[a1:s], %[a1:s], %[one:s]
                \\fmul %[a2:s], %[a2:s], %[one:s]
                \\fmul %[a3:s], %[a3:s], %[one:s]
                \\fmul %[a4:s], %[a4:s], %[one:s]
                \\fmul %[a5:s], %[a5:s], %[one:s]
                \\fmul %[a6:s], %[a6:s], %[one:s]
                \\fmul %[a7:s], %[a7:s], %[one:s]
                : [a0] "+w" (a0),
                  [a1] "+w" (a1),
                  [a2] "+w" (a2),
                  [a3] "+w" (a3),
                  [a4] "+w" (a4),
                  [a5] "+w" (a5),
                  [a6] "+w" (a6),
                  [a7] "+w" (a7),
                : [one] "w" (one),
            );
        }
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.DISABLE, 0);
        std.mem.doNotOptimizeAway(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
        var buf: [8]u8 = undefined;
        _ = linux.read(fd, &buf, 8);
        const cycles: u64 = std.mem.readInt(u64, &buf, builtin.cpu.arch.endian());
        const cpo = @as(f64, @floatFromInt(cycles)) / @as(f64, @floatFromInt(loops * unroll));
        if (cpo < best) best = cpo;
    }
    return best;
}

// Reciprocal throughput of the scalar 64-bit INTEGER multiply, measured the same way (eight
// independent `mul xN, xN, one`). Kept for the record: on this N1 it is only PARTIALLY pipelined
// (measures ~3 cycles/mul, far above the f32 mul), which is why the cost model must NOT weight a mul
// by the integer-mul throughput. It is never used to gate ampere SLP (the integer path is off for
// NEON), so it is documented here, not baked into the model's `.mul` weight.
fn measureIntMulThroughputCycles() !f64 {
    const fd = try perfOpenCycles();
    defer _ = linux.close(fd);
    const loops: u64 = 2_000_000;
    const unroll: u64 = 16;
    var best: f64 = std.math.inf(f64);
    var run: usize = 0;
    while (run < 7) : (run += 1) {
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.RESET, 0);
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.ENABLE, 0);
        var a0: u64 = 1;
        var a1: u64 = 2;
        var a2: u64 = 3;
        var a3: u64 = 4;
        var a4: u64 = 5;
        var a5: u64 = 6;
        var a6: u64 = 7;
        var a7: u64 = 8;
        const one: u64 = 1;
        var i: u64 = 0;
        while (i < loops) : (i += 1) {
            asm volatile (
                \\mul %[a0], %[a0], %[one]
                \\mul %[a1], %[a1], %[one]
                \\mul %[a2], %[a2], %[one]
                \\mul %[a3], %[a3], %[one]
                \\mul %[a4], %[a4], %[one]
                \\mul %[a5], %[a5], %[one]
                \\mul %[a6], %[a6], %[one]
                \\mul %[a7], %[a7], %[one]
                \\mul %[a0], %[a0], %[one]
                \\mul %[a1], %[a1], %[one]
                \\mul %[a2], %[a2], %[one]
                \\mul %[a3], %[a3], %[one]
                \\mul %[a4], %[a4], %[one]
                \\mul %[a5], %[a5], %[one]
                \\mul %[a6], %[a6], %[one]
                \\mul %[a7], %[a7], %[one]
                : [a0] "+r" (a0),
                  [a1] "+r" (a1),
                  [a2] "+r" (a2),
                  [a3] "+r" (a3),
                  [a4] "+r" (a4),
                  [a5] "+r" (a5),
                  [a6] "+r" (a6),
                  [a7] "+r" (a7),
                : [one] "r" (one),
            );
        }
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.DISABLE, 0);
        std.mem.doNotOptimizeAway(a0 +% a1 +% a2 +% a3 +% a4 +% a5 +% a6 +% a7);
        var buf: [8]u8 = undefined;
        _ = linux.read(fd, &buf, 8);
        const cycles: u64 = std.mem.readInt(u64, &buf, builtin.cpu.arch.endian());
        const cpo = @as(f64, @floatFromInt(cycles)) / @as(f64, @floatFromInt(loops * unroll));
        if (cpo < best) best = cpo;
    }
    return best;
}

test "ampere-altra (Neoverse N1) int mul latency matches the on-host measured value" {
    if (registry.detectHost() != .@"ampere-altra") return error.SkipZigTest;
    const measured = measureMulLatencyCycles() catch return error.SkipZigTest;
    const model_mul: f64 = @floatFromInt(registry.modelFor(.@"ampere-altra").latency(
        .{ .arith = .{ .op = .mul, .lhs = undefined, .rhs = undefined } },
    ));
    try std.testing.expect(@abs(measured - model_mul) < 0.6);
}

test "ampere-altra per-type mul THROUGHPUT matches the model: f32 mul pipelined (1), int mul partial (3)" {
    // Keeps the cost model's PER-TYPE mul weights honest against the silicon. The model now prices a
    // mul by its reciprocal THROUGHPUT (not its latency 4), and that throughput is TYPE-AWARE:
    //   - the f32 FP/vector multiply is FULLY PIPELINED, so throughput(mul, elem_float=true) = 1;
    //   - the 64-bit integer multiply is only PARTIALLY pipelined (~3), so throughput(mul, false) = 3.
    // Both are asserted here against the on-host probes, so the per-type split is test-driven against
    // this silicon (and a stepping drift flags it). Same skip-off-N1 and best-of-runs discipline as
    // the latency probe.
    if (registry.detectHost() != .@"ampere-altra") return error.SkipZigTest;
    const altra = registry.modelFor(.@"ampere-altra");
    const mul_oc: @import("vulcan-ir").function.Opcode = .{ .arith = .{ .op = .mul, .lhs = undefined, .rhs = undefined } };

    // FP path: measured f32 fmul throughput vs the model's f32-mul weight (1) and its latency.
    const f32_tput = measureF32MulThroughputCycles() catch return error.SkipZigTest;
    const model_fp_tput: f64 = @floatFromInt(altra.throughput(mul_oc, true)); // 1
    const model_lat: f64 = @floatFromInt(altra.latency(mul_oc)); // 4
    // Pipelined: throughput strictly below latency, and matching the model's f32-mul weight within a lane.
    try std.testing.expect(f32_tput < model_lat);
    try std.testing.expect(@abs(f32_tput - model_fp_tput) < 0.6);

    // Integer path: measured 64-bit int mul throughput vs the model's int-mul weight (3). The integer
    // multiplier is measurably less pipelined than the f32 one (~3 vs ~1), which is exactly why the
    // model prices the two types apart: weighting an f32 mul by the integer 3 would leave the
    // register-input f32 mul SLP group wrongly profitable (the flagged bug).
    const int_tput = measureIntMulThroughputCycles() catch return error.SkipZigTest;
    const model_int_tput: f64 = @floatFromInt(altra.throughput(mul_oc, false)); // 3
    try std.testing.expect(int_tput >= f32_tput);
    try std.testing.expect(@abs(int_tput - model_int_tput) < 0.6);
}
