//! vulcan-uarch-bench: microarchitecture benchmark harness for the microarch optimizer
//! (libs/vulcan-opt/microarch.zig). For a chosen CPU model, it JIT-compiles a fixed set of
//! kernels twice: once left alone (today's plain isel) and once run through
//! `microarch.optimize` (unroll/vectorize/prefetch/schedule) then compiled with the
//! model-aware isel entry point, and measures the cycle difference on THIS host. This answers
//! "what does the microarch tuning actually buy", not "does it compile": a benchmark that skips
//! the correctness check is worthless, so every JIT'd pair is required to agree bit-for-bit
//! before its cycle numbers are trusted.
//!
//! Usage:
//!   vulcan-uarch-bench                 detect the host microarch and benchmark it
//!   vulcan-uarch-bench --model <tag>   benchmark a specific predefined part (see --list)
//!   vulcan-uarch-bench --list          print every predefined Microarch tag
//!   vulcan-uarch-bench --custom        benchmark the hand-built example model (see below)
//!
//! Three ways to hand this tool a model (the design is meant to make plugging in a fourth,
//! custom model an obvious, documented edit rather than a new code path):
//!   1. A predefined tag: `--model ampere-altra` (any `Microarch` enum tag; `--list` prints them
//!      all). Resolved via `Microarch.parse` + `microarch.modelFor`.
//!   2. Host detection: no `--model` at all calls `microarch.detectHost()`. Errors with a clean
//!      message if the running CPU is not one Vulcan recognizes (an aarch64 box that is not a
//!      Neoverse N1, or any riscv64/x86_64 host today: riscv64 has no architectural part
//!      register to probe, and x86_64 detection has not been wired up yet).
//!   3. A hand-built `Model` literal: `Model` is a plain, public, user-constructible struct (see
//!      libs/vulcan-opt/microarch/model.zig), so a caller with a part Vulcan does not ship just
//!      builds one and passes `&my_model` to `benchModel`. `custom_model` and
//!      `customModelExample` below are a fully worked, runnable instance of this (wired to
//!      `--custom` so the example stays real, exercised code instead of a comment that bit-rots).
//!
//! When the chosen model's `arch` matches the actual host CPU (`builtin.cpu.arch`), every kernel
//! is JIT'd and run for real: baseline and tuned code are compiled, executed, checked for
//! identical results, then timed (perf_event_open cycles on Linux when available, wall-clock
//! elsewhere or when perf is denied). When the model targets a different architecture (e.g.
//! benchmarking an et-soc/riscv64 model from this aarch64 box), on-CPU execution is impossible
//! (riscv64 machine code cannot run on an aarch64 core), so the tool instead reports the
//! IR-level transform stats (instruction/block counts before and after `microarch.optimize`) and
//! says plainly that cycle measurement needs a matching host.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Model = opt.microarch.Model;
const Microarch = opt.microarch.Microarch;
const linux = std.os.linux;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip(); // argv0

    var model_arg: ?[]const u8 = null;
    var custom = false;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--list")) return printList(io);
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return printUsage(io);
        if (std.mem.eql(u8, arg, "--custom")) {
            custom = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--model")) {
            model_arg = it.next() orelse {
                std.debug.print("vulcan-uarch-bench: --model requires a value; run --list for valid tags\n", .{});
                return error.Usage;
            };
            continue;
        }
        std.debug.print("vulcan-uarch-bench: unrecognized argument '{s}'; run --help\n", .{arg});
        return error.Usage;
    }

    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);

    if (custom) {
        try customModelExample(allocator, io, &w.interface);
        return;
    }

    const model: *const Model = if (model_arg) |name| blk: {
        const tag = Microarch.parse(name) orelse {
            std.debug.print("vulcan-uarch-bench: unknown model '{s}'; run --list for valid tags\n", .{name});
            return error.UnknownModel;
        };
        break :blk opt.microarch.modelFor(tag);
    } else blk: {
        const tag = opt.microarch.detectHost() orelse {
            std.debug.print(
                "vulcan-uarch-bench: could not detect a known microarch for this host (arch {s}).\n" ++
                    "Pass --model <tag> explicitly (see --list), or hand-build a custom Model (see\n" ++
                    "customModelExample in this tool's source, exercised via --custom).\n",
                .{@tagName(builtin.cpu.arch)},
            );
            return error.UnknownHost;
        };
        break :blk opt.microarch.modelFor(tag);
    };

    try printHeader(&w.interface, model);
    try benchModel(allocator, io, model, &w.interface);
}

fn printUsage(io: std.Io) !void {
    var buf: [1024]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.print(
        "usage: vulcan-uarch-bench [--model <tag> | --list | --custom]\n" ++
            "  (no args)      detect the host microarch and benchmark it\n" ++
            "  --model <tag>  benchmark a specific predefined part (see --list)\n" ++
            "  --list         print every predefined Microarch tag\n" ++
            "  --custom       benchmark the hand-built example model (see customModelExample)\n",
        .{},
    );
    try w.interface.flush();
}

fn printList(io: std.Io) !void {
    var buf: [1024]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.print("predefined Microarch tags:\n", .{});
    inline for (std.meta.tags(Microarch)) |t| try w.interface.print("  {s}\n", .{t.name()});
    try w.interface.flush();
}

fn printHeader(w: *std.Io.Writer, model: *const Model) !void {
    const host_tag = opt.microarch.detectHost();
    try w.print("vulcan-uarch-bench\n", .{});
    try w.print("  host arch:      {s}\n", .{@tagName(builtin.cpu.arch)});
    try w.print("  host microarch: {s}\n", .{if (host_tag) |t| t.name() else "unrecognized"});
    try w.print("  chosen model:   {s}  (arch {s}, {s}, issue_width {d})\n", .{
        model.tag.name(),
        @tagName(model.arch),
        if (model.reorders()) "out-of-order" else "in-order",
        model.issue_width,
    });
    try w.print("\n", .{});
    try w.flush();
}

/// A dependent-mul chain, straight-line (no loop): `r = x; r = r*x` five times, so `f(x) = x^6`.
/// Every multiply depends on the previous one, exercising scheduling (arranging a dependency
/// chain for the model's ports/latency) with nothing for unroll/vectorize/prefetch to do.
fn buildMulChain(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i64_t);
    var r = x;
    var k: usize = 0;
    while (k < 5) : (k += 1) r = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = r, .rhs = x } });
    func.setTerminator(entry, .{ .ret = r });
    return func;
}

/// A register-pressure kernel: fn(a, b) = sum over k in 1..=20 of (a*k + b). All 20 products are live
/// simultaneously until the final sum, so the whole-interval linear-scan allocator MUST spill (the GPR
/// pool is ~12 registers). Used to confirm spilling exists and to measure the live-range-splitting win.
fn buildPressureKernel(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    var terms: [20]Value = undefined;
    var k: i64 = 1;
    while (k <= 20) : (k += 1) {
        const kc = try func.appendInst(entry, i32_t, .{ .iconst = k });
        const ak = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        terms[@intCast(k - 1)] = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = b } });
    }
    // Left-leaning sum tree so every term stays live until it is folded in.
    var acc = terms[0];
    var j: usize = 1;
    while (j < terms.len) : (j += 1) {
        acc = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = terms[j] } });
    }
    func.setTerminator(entry, .{ .ret = acc });
    return func;
}

/// A counted sum-reduction loop: `s = 0; for (i = 0; i < n; i += 1) s += i; return s`. Exercises
/// unroll (expose ILP across iterations) and schedule together.
fn buildSumLoop(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
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
    return func;
}

/// A SAXPY map loop over real f32 arrays, writing a SEPARATE output so it is non-accumulating (the
/// bench runs baseline and tuned over the same buffers): `for (i = 0; i < n; i += 1) out[i] = a*x[i]
/// + y[i]; return out[0];`. Exercises the loop vectorizer (main body unrolled by the SIMD width, SLP
/// widened to a wide load / vector fmul / vector fadd / wide store, the invariant `a` a splat `dup`).
fn buildSaxpyLoop(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
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
    const out = try func.appendBlockParam(entry, ptr_t);
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
    const oaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = out, .rhs = off } });
    try func.appendStore(body, res, oaddr);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ni});
    const r = try func.appendInst(done, f32_t, .{ .load = .{ .ptr = out } }); // out[0], data-dependent
    func.setTerminator(done, .{ .ret = r });
    return func;
}

/// A contiguous f32 sum reduction over a real array: `s = 0; for (i = 0; i < n; i += 1) s += a[i];
/// return s`. Marked fast_math, so the loop vectorizer builds a vector accumulator (wide loads +
/// vector fadd) and a horizontal reduce, like gcc -O3's vector-accumulator reduction.
fn buildFsumLoop(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
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
    return func;
}

/// A strided pointer-walking reduction over a real backing array: `s = 0; p = arr; for (i = 0;
/// i < n; i += 1) { s += *p; p += 8; } return s`. Exercises prefetch + unroll + schedule together
/// over real memory.
fn buildStridedSum(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
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
    return func;
}

/// An FMA-friendly `a*b+c` chain, straight-line: `r = a; r = r*b + c` five times. Each `mul`
/// feeds only the immediately following `add` and is otherwise dead, exactly the pattern the
/// aarch64 backend's fused-multiply-add isel hook (see aarch64/isel.zig's `tryFuseMulAdd`-style
/// comment block) collapses to one `fmadd` instead of a separate `fmul`+`fadd`.
fn buildFmaChain(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);
    const b = try func.appendBlockParam(entry, f32_t);
    const c = try func.appendBlockParam(entry, f32_t);
    var r = a;
    var k: usize = 0;
    while (k < 5) : (k += 1) {
        const t = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = r, .rhs = b } });
        r = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = t, .rhs = c } });
    }
    func.setTerminator(entry, .{ .ret = r });
    return func;
}

/// Four independent, parallel f32 adds: `r0 = a0+b0; r1 = a1+b1; r2 = a2+b2; r3 = a3+b3; return
/// r0`. The exact shape `vectorize.zig`'s SLP pass fuses (see its `parallelAdds` test helper): a
/// contiguous run of same-op scalar f32 arith, four lanes because Ampere's NEON is 128-bit
/// (128/32 = 4). `microarch.optimize` collapses these four scalar adds into one vector add for
/// any model with a wide-enough vector unit and an ISA vector feature bit set.
fn buildSlpAdds(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    var a: [4]Value = undefined;
    var b: [4]Value = undefined;
    for (0..4) |lane| a[lane] = try func.appendBlockParam(entry, f32_t);
    for (0..4) |lane| b[lane] = try func.appendBlockParam(entry, f32_t);
    var r: [4]Value = undefined;
    for (0..4) |lane| r[lane] = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = a[lane], .rhs = b[lane] } });
    func.setTerminator(entry, .{ .ret = r[0] });
    return func;
}

/// A memory-based elementwise kernel: `out[i] = a[i] + b[i]` for i in 0..8, over three contiguous
/// f32 arrays passed by pointer, returning `out[0]` so the harness has a return value to check.
/// UNLIKE `buildSlpAdds` (whose operands are block-parameter registers, so its pack/unpack overhead
/// makes SLP unprofitable and it stays declined at 1.0x), every operand here is a contiguous load
/// and every result a contiguous store, so the vectorizer's memory coalescing turns the eight scalar
/// loads + eight scalar stores into two wide vector loads + one wide vector store per 4-lane group
/// (aarch64 NEON), fusing the adds. That is a real instruction-count win: the tuned build measures
/// FASTER than the baseline, the payoff coalescing exists to deliver. The eight adds are independent
/// (each reads its own two loaded lanes), so the SLP scan fuses them; the returned `out[0]` is a
/// single lane, never a fused-away dependent reduction.
fn buildMemAdd(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const pa = try func.appendBlockParam(entry, ptr_t);
    const pb = try func.appendBlockParam(entry, ptr_t);
    const pout = try func.appendBlockParam(entry, ptr_t);

    var av: [8]Value = undefined;
    for (0..8) |i| {
        const addr = try func.appendArithImm(entry, ptr_t, .add, pa, @intCast(i * 4));
        av[i] = try func.appendInst(entry, f32_t, .{ .load = .{ .ptr = addr } });
    }
    var bv: [8]Value = undefined;
    for (0..8) |i| {
        const addr = try func.appendArithImm(entry, ptr_t, .add, pb, @intCast(i * 4));
        bv[i] = try func.appendInst(entry, f32_t, .{ .load = .{ .ptr = addr } });
    }
    var cv: [8]Value = undefined;
    for (0..8) |i| cv[i] = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = av[i], .rhs = bv[i] } });
    for (0..8) |i| {
        const addr = try func.appendArithImm(entry, ptr_t, .add, pout, @intCast(i * 4));
        try func.appendStore(entry, cv[i], addr);
    }
    func.setTerminator(entry, .{ .ret = cv[0] });
    return func;
}

/// A memory-based SAXPY-style kernel: `out[i] = a[i]*b[i] + a[i]` for i in 0..8, over three
/// contiguous f32 arrays passed by pointer, returning `out[0]`. This is the collateral the
/// result-chaining credit recovers: the intermediate mul results are NOT stored, they chain
/// lane-for-lane into the following add (the mul result vector rides a register into the add), so the
/// greedy per-group cost model would otherwise charge the mul group a full per-lane unpack and
/// decline it, leaving the whole kernel scalar. With result-chaining credited, the mul group is
/// profitable via its two coalesced-load operands (unpack free), and the add group is profitable via
/// operand-chaining (mul result reused) plus its coalesced `a` reload and coalesced `out` store. The
/// eight muls and eight adds each fuse into two 4-lane NEON groups, a real instruction-count win the
/// tuned build measures FASTER than the scalar baseline.
fn buildMemMulAdd(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const pa = try func.appendBlockParam(entry, ptr_t);
    const pb = try func.appendBlockParam(entry, ptr_t);
    const pout = try func.appendBlockParam(entry, ptr_t);

    var av: [8]Value = undefined;
    for (0..8) |i| {
        const addr = try func.appendArithImm(entry, ptr_t, .add, pa, @intCast(i * 4));
        av[i] = try func.appendInst(entry, f32_t, .{ .load = .{ .ptr = addr } });
    }
    var bv: [8]Value = undefined;
    for (0..8) |i| {
        const addr = try func.appendArithImm(entry, ptr_t, .add, pb, @intCast(i * 4));
        bv[i] = try func.appendInst(entry, f32_t, .{ .load = .{ .ptr = addr } });
    }
    // Two contiguous same-op runs: the muls, then the adds. The mul result vector chains into the
    // add (vectorize.zig's chain reuse), while `a` is reloaded for the add's rhs.
    var mv: [8]Value = undefined;
    for (0..8) |i| mv[i] = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = av[i], .rhs = bv[i] } });
    var cv: [8]Value = undefined;
    for (0..8) |i| cv[i] = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = mv[i], .rhs = av[i] } });
    for (0..8) |i| {
        const addr = try func.appendArithImm(entry, ptr_t, .add, pout, @intCast(i * 4));
        try func.appendStore(entry, cv[i], addr);
    }
    func.setTerminator(entry, .{ .ret = cv[0] });
    return func;
}

/// A straight-line consecutive-word memory copy: load `words` contiguous i64 from `in`, then store
/// them to `out`, then read `out[0]` and `out[words - 1]` back and return their sum so the check in
/// `benchGeneric` actually exercises the store half of the copy (a return value built only from the
/// loaded values, as before, would pass its baseline/tuned equivalence check even if the stores were
/// wrong). This is the ldp/stp peephole's shape: adjacent same-base loads and adjacent same-base
/// stores. The always-on `pairMemory` pass runs inside compileFunction for BOTH the baseline and
/// tuned builds, so the bench reports this memory-heavy kernel's cycles either way and confirms it
/// runs. With address folding (Task 3) the constant-index loads/stores now fold to `[base, #off]`,
/// their address-adds die, and the resulting adjacent same-base runs fuse into ldp/stp, so the copy
/// body itself is paired here (in both builds equally, so the tuned/baseline ratio stays near 1.0
/// while the absolute cycles drop versus the pre-fold, one-add-per-access shape).
fn buildMemPair(allocator: std.mem.Allocator) anyerror!Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.intern(.ptr);
    const words = 16;
    const entry = try func.appendBlock();
    const in = try func.appendBlockParam(entry, ptr_t);
    const out = try func.appendBlockParam(entry, ptr_t);
    var loaded: [words]Value = undefined;
    for (0..words) |i| {
        const addr = if (i == 0) in else try func.appendArithImm(entry, ptr_t, .add, in, @intCast(i * 8));
        loaded[i] = try func.appendInst(entry, i64_t, .{ .load = .{ .ptr = addr } });
    }
    var out_addrs: [words]Value = undefined;
    for (0..words) |i| {
        out_addrs[i] = if (i == 0) out else try func.appendArithImm(entry, ptr_t, .add, out, @intCast(i * 8));
        try func.appendStore(entry, loaded[i], out_addrs[i]);
    }
    const back0 = try func.appendInst(entry, i64_t, .{ .load = .{ .ptr = out_addrs[0] } });
    const backN = try func.appendInst(entry, i64_t, .{ .load = .{ .ptr = out_addrs[words - 1] } });
    const sum = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = back0, .rhs = backN } });
    func.setTerminator(entry, .{ .ret = sum });
    return func;
}

fn customArith(op: ir.function.BinOp) u32 {
    return switch (op) {
        .mul, .mulh => 3,
        .div, .rem => 20,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn customLatency(op: ir.function.Opcode) u32 {
    return switch (op) {
        .arith => |a| customArith(a.op),
        .arith_imm => |a| customArith(a.op),
        .load => 3,
        .convert, .unary => 2,
        // A dot is a multiply-class op (4-way multiply-accumulate), grouped with `mul`.
        .dot => 3,
        // This fictional part carries no tensor unit; a placeholder in case one is added.
        .matmul => 64,
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

// This fictional part is a simple in-order core with an unpipelined multiplier and no FPU, so mul/div
// keep their latency as reciprocal throughput (non-pipelined) for BOTH element types (elem_float is
// ignored); the single-cycle ops issue at 1, and the pipelined load-to-use issues at 1 despite its
// 3-cycle latency. All values satisfy throughput <= latency (see Model.validate).
fn customArithThroughput(op: ir.function.BinOp, elem_float: bool) u32 {
    _ = elem_float; // unpipelined multiplier, no FPU: FP and integer mul share the non-pipelined price
    return switch (op) {
        .mul, .mulh => 3,
        .div, .rem => 20,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
    };
}

fn customThroughput(op: ir.function.Opcode, elem_float: bool) u32 {
    return switch (op) {
        .arith => |a| customArithThroughput(a.op, elem_float),
        .arith_imm => |a| customArithThroughput(a.op, elem_float),
        .load => 1,
        .convert, .unary => 1,
        .dot => 3,
        .matmul => 64, // no tensor unit here; non-pipelined placeholder
        .iconst, .fconst, .icmp, .select, .struct_new, .extract, .alloca, .call, .call_indirect, .global_addr, .store, .prefetch, .@"if" => 1,
    };
}

fn customUnit(op: ir.function.Opcode) opt.microarch.UnitClass {
    return switch (op) {
        .arith => |a| customArithUnit(a.op),
        .arith_imm => |a| customArithUnit(a.op),
        .icmp, .select, .iconst, .fconst, .global_addr => .alu,
        .convert => .fpsimd,
        .unary => |u| switch (u.op) {
            .reinterpret => .alu,
            .sqrt, .ceil, .floor, .trunc, .nearest => .fpsimd,
        },
        .load, .store, .prefetch, .alloca => .mem,
        .@"if" => .branch,
        .call, .call_indirect => .branch,
        .struct_new, .extract => .none,
        // dot runs on the SIMD/vector unit, like the vector-shaped fpsimd ops above.
        .dot => .fpsimd,
        // matmul runs on the tensor/VPU unit, modeled as fpsimd like dot.
        .matmul => .fpsimd,
    };
}

fn customArithUnit(op: ir.function.BinOp) opt.microarch.UnitClass {
    return switch (op) {
        .mul, .mulh, .div, .rem => .muldiv,
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => .alu,
    };
}

// FIXME: allow the `--custom` flag to override this
const custom_model: Model = .{
    .tag = .@"ampere-altra",
    .arch = .aarch64,
    .exec = .in_order,
    .issue_width = 2,
    .rob_size = 0,
    .units = .{ .alu = 2, .muldiv = 1, .mem = 1, .branch = 1, .fpsimd = 0 },
    .vector_bits = 0,
    .cache_line = 64,
    .fetch_align = 64,
    .features = .{ .aarch64 = .{} },
    .latency = customLatency,
    .throughput = customThroughput,
    .unitOf = customUnit,
    .fusion = &.{},
};

comptime {
    Model.validate(custom_model);
}

/// Worked example: run the tool's full benchmark against the hand-built `custom_model` above,
/// exactly like `main` does for a predefined tag or `detectHost()`. Wired to `--custom` so it
/// stays real, exercised code rather than a comment that bit-rots.
fn customModelExample(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer) !void {
    try printHeader(w, &custom_model);
    try benchModel(allocator, io, &custom_model, w);
}

/// Whether this tool can actually JIT and run `model`'s arch on this host. aarch64, riscv64, and
/// x86_64 each have a model-aware `selectFunctionForModel` today, and JIT output only runs on its
/// own ISA, so both conditions are required (anything else always falls to the IR
/// transform-stats path, regardless of host).
fn canJit(model: *const Model) bool {
    return switch (model.arch) {
        .aarch64 => builtin.cpu.arch == .aarch64,
        .riscv64 => builtin.cpu.arch == .riscv64,
        .x86_64 => builtin.cpu.arch == .x86_64,
    };
}

/// One of the three backends' mapped executable buffers. A tagged union rather than a common type
/// because `target.aarch64.jit.CodeBuffer`, `target.riscv64.jit.CodeBuffer`, and
/// `target.x86_64.jit.CodeBuffer` are distinct instantiations of the same generic (see
/// jit_platform.zig's `Buffer`), each carrying its own arch-specific instruction-cache sync
/// function baked in at comptime.
const HostBuffer = union(enum) {
    aarch64: target.aarch64.jit.CodeBuffer,
    riscv64: target.riscv64.jit.CodeBuffer,
    x86_64: target.x86_64.jit.CodeBuffer,

    fn deinit(self: *HostBuffer) void {
        switch (self.*) {
            inline else => |*b| b.deinit(),
        }
    }

    fn entry(self: *const HostBuffer, comptime Fn: type, offset: usize) Fn {
        return switch (self.*) {
            inline else => |b| b.entry(Fn, offset),
        };
    }
};

/// Compile `func` for `arch` and map it into executable memory. `model` is null for the
/// untouched baseline (plain `selectFunction`) and set for the tuned build
/// (`selectFunctionForModel`, which fires the backend's alignment/fusion hooks for that model).
/// Only ever called after `canJit` confirms `arch` matches the host, so the mapped code is always
/// safe to call.
fn selectAndMap(allocator: std.mem.Allocator, func: *const Function, model: ?*const Model, arch: opt.microarch.Arch) !HostBuffer {
    return switch (arch) {
        .aarch64 => blk: {
            const code = if (model) |m|
                try target.aarch64.isel.selectFunctionForModel(allocator, func, m)
            else
                try target.aarch64.isel.selectFunction(allocator, func);
            defer allocator.free(code);
            break :blk .{ .aarch64 = try target.aarch64.jit.CodeBuffer.map(std.mem.sliceAsBytes(code)) };
        },
        .riscv64 => blk: {
            const code = if (model) |m|
                try target.riscv64.isel.selectFunctionForModel(allocator, func, m)
            else
                try target.riscv64.isel.selectFunction(allocator, func);
            defer allocator.free(code);
            break :blk .{ .riscv64 = try target.riscv64.jit.CodeBuffer.map(std.mem.sliceAsBytes(code)) };
        },
        .x86_64 => blk: {
            const code = if (model) |m|
                try target.x86_64.isel.selectFunctionForModel(allocator, func, m)
            else
                try target.x86_64.isel.selectFunction(allocator, func);
            defer allocator.free(code);
            break :blk .{ .x86_64 = try target.x86_64.jit.CodeBuffer.map(std.mem.sliceAsBytes(code)) };
        },
    };
}

/// Block/instruction totals, for the transform-stats path (no JIT possible).
const IrStats = struct { blocks: usize, insts: usize };

fn irStats(func: *const Function) IrStats {
    var insts: usize = 0;
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) insts += func.blockInsts(@enumFromInt(bi)).len;
    return .{ .blocks = func.blockCount(), .insts = insts };
}

// perf_event_open cycle counting, mirroring libs/vulcan-target/tests/microarch_e2e.zig's harness
// (that file does not export a reusable helper, hence the small intentional duplicate here).

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

const Measurement = struct { per_call: f64, unit: []const u8 };

/// Best-of-`outer_runs` cost per call to `f(args)`: cycles/call via perf_event_open on Linux when
/// available, else wall-clock ns/call (non-Linux hosts, or perf denied e.g. no CAP_PERFMON).
/// Never fails: the wall-clock path is always usable, so a benchmark run never crashes for lack
/// of a perf counter, it just reports a coarser unit.
fn measure(comptime PtrFn: type, io: std.Io, f: PtrFn, args: std.meta.ArgsTuple(@typeInfo(PtrFn).pointer.child)) Measurement {
    if (comptime builtin.os.tag == .linux) {
        if (perfOpenCycles()) |fd| {
            defer _ = linux.close(fd);
            return .{ .per_call = measureCyclesLoop(PtrFn, fd, f, args), .unit = "cycles/call" };
        } else |_| {} // perf unavailable: fall through to wall-clock below
    }
    return .{ .per_call = measureWallLoop(PtrFn, io, f, args), .unit = "ns/call" };
}

fn measureCyclesLoop(comptime PtrFn: type, fd: i32, f: PtrFn, args: std.meta.ArgsTuple(@typeInfo(PtrFn).pointer.child)) f64 {
    var best: f64 = std.math.inf(f64);
    var run: usize = 0;
    while (run < outer_runs) : (run += 1) {
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.RESET, 0);
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.ENABLE, 0);
        var i: u64 = 0;
        while (i < hot_iters) : (i += 1) std.mem.doNotOptimizeAway(@call(.auto, f, args));
        _ = linux.ioctl(fd, linux.PERF.EVENT_IOC.DISABLE, 0);
        const cycles = readCycles(fd);
        const cpi = @as(f64, @floatFromInt(cycles)) / @as(f64, @floatFromInt(hot_iters));
        if (cpi < best) best = cpi;
    }
    return best;
}

fn measureWallLoop(comptime PtrFn: type, io: std.Io, f: PtrFn, args: std.meta.ArgsTuple(@typeInfo(PtrFn).pointer.child)) f64 {
    var best: f64 = std.math.inf(f64);
    var run: usize = 0;
    while (run < outer_runs) : (run += 1) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        var i: u64 = 0;
        while (i < hot_iters) : (i += 1) std.mem.doNotOptimizeAway(@call(.auto, f, args));
        const t1 = std.Io.Timestamp.now(io, .awake);
        const dur = t0.durationTo(t1);
        const npc = @as(f64, @floatFromInt(dur.nanoseconds)) / @as(f64, @floatFromInt(hot_iters));
        if (npc < best) best = npc;
    }
    return best;
}

/// Bench one kernel under `model`: build baseline + tuned copies, run `microarch.optimize` on
/// the tuned one, then either JIT-and-measure (host arch matches) or report IR transform stats
/// (it does not). `PtrFn` is the kernel's native calling-convention signature (e.g. `*const fn
/// (i64) callconv(.c) i64`); `checks` is a spread of representative inputs the baseline and
/// tuned JIT must agree on bit-for-bit before any cycle number is trusted; `measure_args` is the
/// single input used for the timed hot loop.
fn benchGeneric(
    comptime PtrFn: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    model: *const Model,
    w: *std.Io.Writer,
    name: []const u8,
    build: *const fn (std.mem.Allocator) anyerror!Function,
    checks: []const std.meta.ArgsTuple(@typeInfo(PtrFn).pointer.child),
    measure_args: std.meta.ArgsTuple(@typeInfo(PtrFn).pointer.child),
) !void {
    var baseline = try build(allocator);
    defer baseline.deinit();
    var tuned = try build(allocator);
    defer tuned.deinit();

    const changed = try opt.microarch.optimize(allocator, &tuned, model);

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    if (!diags.ok()) {
        try w.print("{s:<12} FAIL  microarch.optimize produced a function that fails IR verify\n", .{name});
        try w.flush();
        return error.Miscompile;
    }

    if (canJit(model)) {
        var base_buf = try selectAndMap(allocator, &baseline, null, model.arch);
        defer base_buf.deinit();
        var tuned_buf = try selectAndMap(allocator, &tuned, model, model.arch);
        defer tuned_buf.deinit();

        const f_base = base_buf.entry(PtrFn, 0);
        const f_tuned = tuned_buf.entry(PtrFn, 0);

        // The tuned build must compute exactly what the untouched baseline computes, for
        // every representative input. A benchmark that skips this is worthless, so a mismatch
        // aborts with a clear diagnostic instead of printing a bogus number.
        for (checks) |c| {
            const r_base = @call(.auto, f_base, c);
            const r_tuned = @call(.auto, f_tuned, c);
            if (r_base != r_tuned) {
                try w.print("{s:<12} FAIL  baseline and tuned diverge (miscompile in the pipeline)\n", .{name});
                try w.flush();
                return error.Miscompile;
            }
        }

        const base_m = measure(PtrFn, io, f_base, measure_args);
        const tuned_m = measure(PtrFn, io, f_tuned, measure_args);
        const speedup = base_m.per_call / tuned_m.per_call;
        try w.print("{s:<12} base {d:>10.2} {s:<12} tuned {d:>10.2} {s:<12} speedup {d:>5.2}x  [OK]\n", .{
            name, base_m.per_call, base_m.unit, tuned_m.per_call, tuned_m.unit, speedup,
        });
    } else {
        const base_stats = irStats(&baseline);
        const tuned_stats = irStats(&tuned);
        try w.print(
            "{s:<12} model {s} targets {s}, host is {s}: on-CPU cycle measurement needs a " ++
                "matching host.\n{s:<12} IR transform stats -- insts {d} -> {d}, blocks {d} -> {d}, changed={}\n",
            .{
                name,               model.tag.name(), @tagName(model.arch), @tagName(builtin.cpu.arch),
                "",                 base_stats.insts, tuned_stats.insts,    base_stats.blocks,
                tuned_stats.blocks, changed,
            },
        );
    }
    try w.flush();
}

/// Run every kernel against `model`, printing one (or two, for the transform-stats path) rows
/// per kernel. See the module doc comment for the three ways to obtain a `model`.
pub fn benchModel(allocator: std.mem.Allocator, io: std.Io, model: *const Model, w: *std.Io.Writer) !void {
    // Shared backing array for the strided-sum kernel: long enough for the largest n exercised
    // below (128), filled with an arbitrary non-trivial pattern so the reduction is not all-zero.
    var backing: [128]i64 = undefined;
    for (&backing, 0..) |*v, idx| v.* = @as(i64, @intCast(idx)) * 3 - 17;

    try benchGeneric(
        *const fn (i64) callconv(.c) i64,
        allocator,
        io,
        model,
        w,
        "mul-chain",
        buildMulChain,
        &.{ .{-5}, .{-3}, .{-2}, .{-1}, .{0}, .{1}, .{2}, .{3}, .{5}, .{7} },
        .{5},
    );

    try benchGeneric(
        *const fn (i32, i32) callconv(.c) i32,
        allocator,
        io,
        model,
        w,
        "pressure",
        buildPressureKernel,
        &.{ .{ 1, 2 }, .{ 3, 5 }, .{ -2, 7 }, .{ 0, 0 }, .{ 100, -100 } },
        .{ 7, 11 },
    );

    try benchGeneric(
        *const fn (i64) callconv(.c) i64,
        allocator,
        io,
        model,
        w,
        "sum-loop",
        buildSumLoop,
        &.{ .{0}, .{1}, .{2}, .{3}, .{7}, .{16}, .{100}, .{1000} },
        .{1000},
    );

    try benchGeneric(
        *const fn (i64, [*]i64) callconv(.c) i64,
        allocator,
        io,
        model,
        w,
        "strided-sum",
        buildStridedSum,
        &.{ .{ 0, &backing }, .{ 1, &backing }, .{ 2, &backing }, .{ 5, &backing }, .{ 16, &backing }, .{ 64, &backing }, .{ 128, &backing } },
        .{ 128, &backing },
    );

    try benchGeneric(
        *const fn (f32, f32, f32) callconv(.c) f32,
        allocator,
        io,
        model,
        w,
        "fma-chain",
        buildFmaChain,
        &.{ .{ 1.5, 0.5, 2.0 }, .{ -2.0, 3.0, 1.0 }, .{ 0.0, 1.0, 1.0 }, .{ 3.25, -1.25, 0.5 } },
        .{ 1.5, 0.5, 2.0 },
    );

    try benchGeneric(
        *const fn (f32, f32, f32, f32, f32, f32, f32, f32) callconv(.c) f32,
        allocator,
        io,
        model,
        w,
        "slp-adds",
        buildSlpAdds,
        &.{
            .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 },
            .{ -1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0 },
            .{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        },
        .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 },
    );

    // The memory-coalescing payoff kernel: contiguous loads/stores let SLP coalesce and win, unlike
    // the register-input slp-adds above which correctly stays declined (1.0x).
    var mem_a = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var mem_b = [8]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    var mem_out = [_]f32{0} ** 8;
    try benchGeneric(
        *const fn ([*]f32, [*]f32, [*]f32) callconv(.c) f32,
        allocator,
        io,
        model,
        w,
        "mem-add",
        buildMemAdd,
        &.{.{ &mem_a, &mem_b, &mem_out }},
        .{ &mem_a, &mem_b, &mem_out },
    );

    // The SAXPY / multiply-add memory kernel: `out[i] = a[i]*b[i] + a[i]`. Its intermediate mul
    // results chain into the following add (never stored), so it only wins once the cost model credits
    // that result-chaining (the mul group's unpack is elided). With the credit it coalesces like
    // mem-add and measures FASTER; without it the mul group would decline and the kernel stay scalar.
    var mma_a = [8]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var mma_b = [8]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    var mma_out = [_]f32{0} ** 8;
    try benchGeneric(
        *const fn ([*]f32, [*]f32, [*]f32) callconv(.c) f32,
        allocator,
        io,
        model,
        w,
        "mem-mul-add",
        buildMemMulAdd,
        &.{.{ &mma_a, &mma_b, &mma_out }},
        .{ &mma_a, &mma_b, &mma_out },
    );

    // The consecutive-word memory copy: the ldp/stp peephole's adjacent-load/adjacent-store shape.
    // Both baseline and tuned go through the always-on pairMemory pass, so this reports the copy
    // kernel's cycles and confirms it runs correctly (baseline == tuned) under fusion.
    var pair_in: [16]i64 = undefined;
    for (&pair_in, 0..) |*v, idx| v.* = @as(i64, @intCast(idx)) * 7 - 3;
    var pair_out = [_]i64{0} ** 16;
    try benchGeneric(
        *const fn ([*]i64, [*]i64) callconv(.c) i64,
        allocator,
        io,
        model,
        w,
        "mem-pair",
        buildMemPair,
        &.{.{ &pair_in, &pair_out }},
        .{ &pair_in, &pair_out },
    );

    // The SAXPY map LOOP over real f32 arrays: exercises the loop vectorizer (main body unrolled by the
    // SIMD width, SLP-widened to wide load / vector fmul / vector fadd / wide store, `a` a splat dup).
    var sax_x = [_]f32{0} ** 1024;
    var sax_y = [_]f32{0} ** 1024;
    var sax_out = [_]f32{0} ** 1024;
    for (0..1024) |k| {
        sax_x[k] = @floatFromInt(k + 1);
        sax_y[k] = @floatFromInt(1024 - @as(i32, @intCast(k)));
    }
    try benchGeneric(
        *const fn ([*]f32, [*]f32, [*]f32, f32, i32) callconv(.c) f32,
        allocator,
        io,
        model,
        w,
        "saxpy-loop",
        buildSaxpyLoop,
        &.{.{ &sax_x, &sax_y, &sax_out, 2.5, 1024 }},
        .{ &sax_x, &sax_y, &sax_out, 2.5, 1024 },
    );

    // The contiguous f32 sum reduction: the loop vectorizer builds a vector accumulator (wide loads +
    // vector fadd) with a horizontal reduce at the end.
    var fsum_a = [_]f32{0} ** 1024;
    for (0..1024) |k| fsum_a[k] = @floatFromInt((k % 13) + 1);
    try benchGeneric(
        *const fn ([*]f32, i32) callconv(.c) f32,
        allocator,
        io,
        model,
        w,
        "fsum-loop",
        buildFsumLoop,
        &.{.{ &fsum_a, 1024 }},
        .{ &fsum_a, 1024 },
    );
}
