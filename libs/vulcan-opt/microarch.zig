//! Microarchitecture-aware optimization. This substrate exposes the model types, the predefined
//! part models, and host detection. The passes (scheduling, vectorization width, unrolling,
//! prefetch) and the backend hooks (fusion, alignment) are added in later plans.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const vectorize = @import("vectorize.zig"); // vulcan-opt root module

const model_mod = @import("microarch/model.zig");
const registry = @import("microarch/registry.zig");

pub const Arch = model_mod.Arch;
pub const ExecMode = model_mod.ExecMode;
pub const UnitClass = model_mod.UnitClass;
pub const Units = model_mod.Units;
pub const Features = model_mod.Features;
pub const FuseKind = model_mod.FuseKind;
pub const FusionRule = model_mod.FusionRule;
pub const Microarch = model_mod.Microarch;
pub const Model = model_mod.Model;

pub const modelFor = registry.modelFor;
pub const detectHost = registry.detectHost;
pub const schedule = @import("microarch/schedule.zig");
pub const cost = @import("microarch/cost.zig");
pub const unroll = @import("microarch/unroll.zig");
pub const splitunroll = @import("microarch/splitunroll.zig");
pub const prefetch = @import("microarch/prefetch.zig");
pub const dotprod = @import("microarch/dotprod.zig");
// Matmul-nest recognition: raises a naive fp32 triply-nested matmul loop to the et-soc tensor `matmul`
// op. Wired into `optimize` below; exposed here too so its canonical nest builder is reachable from
// the sysemu differential in libs/vulcan-target/riscv64/tests/etsoc_sysemu.zig.
pub const matmul_recog = @import("microarch/matmul_recog.zig");

/// Convenience: the model for a predefined tag. Same as modelFor, reads better at call sites.
pub fn model(tag: Microarch) *const Model {
    return registry.modelFor(tag);
}

/// Run the microarchitecture-aware IR passes for `model`, in spec order: unroll (expose ILP), then
/// vectorize (over the unrolled bodies), then dotprod/matmul_recog (recognize reduction/nest idioms
/// while the scalar shape is still intact), then prefetch (knows the strides), then schedule (arrange
/// for the ports). Each pass is evaluated into its own const first, so every reporting pass ALWAYS
/// runs: no `or` short-circuit is used to decide whether to call the next one. Returns whether the
/// reporting passes (unroll/vectorize/dotprod/matmul_recog/prefetch) changed the IR. Scheduling always
/// runs and only reorders instructions within a block, so it is not counted in the return: it never
/// changes results, only order. Not calling this function leaves today's behavior exactly as it is:
/// the whole feature is opt-in.
pub fn optimize(allocator: std.mem.Allocator, func: *ir.function.Function, m: *const Model) pass.Error!bool {
    // Accumulator-splitting unroll of counted reduction loops runs FIRST: it rewrites a reduction
    // loop into a main loop carrying K independent partial accumulators plus a remainder loop, so the
    // loop-carried dependency is one op instead of K. The general guarded unroller then handles
    // whatever loops this did not (non-reduction loops, and the small remainder loops it leaves).
    const split = try splitunroll.run(allocator, func, m);
    const unrolled = try unroll.run(allocator, func, m);
    const vectorized = try vectorize.runModel(allocator, func, m);
    // INT8 dot-product recognition runs after the general vectorizer (it needs the scalar reduction
    // loop intact) and before scheduling. It is internally gated on aarch64+dotprod, so non-aarch64
    // models are unaffected.
    const dotted = try dotprod.run(allocator, func, m);
    // Matmul-nest recognition runs after the general vectorizer and before scheduling too, mirroring
    // dotprod: it needs the scalar loop nest intact to recognize it, and gates internally on
    // `model.vpu()` (et-soc only), so it is a no-op for every other model.
    const matmuled = try matmul_recog.run(allocator, func, m);
    const prefetched = try prefetch.run(allocator, func, m);
    try schedule.run(allocator, func, m);
    return split or unrolled or vectorized or dotted or matmuled or prefetched;
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(@import("microarch/schedule.zig"));
    std.testing.refAllDecls(@import("microarch/cost.zig"));
    std.testing.refAllDecls(@import("microarch/unroll.zig"));
    std.testing.refAllDecls(@import("microarch/prefetch.zig"));
    std.testing.refAllDecls(@import("microarch/dotprod.zig"));
    std.testing.refAllDecls(@import("microarch/matmul_recog.zig"));
    std.testing.refAllDecls(@import("vectorize.zig"));
    // The Ampere measurement test uses aarch64 inline asm (mul/fmul chains) AND Linux perf_event_open,
    // and refAllDecls forces its Linux-using helpers to compile, so only reference it on aarch64-Linux.
    // On any other host it is left out entirely (it would only runtime-skip there anyway), which keeps
    // `zig build test` compiling on non-aarch64 and on non-Linux aarch64 (macOS/etc).
    if (comptime builtin.cpu.arch == .aarch64 and builtin.os.tag == .linux) {
        std.testing.refAllDecls(@import("microarch/altra_measure_test.zig"));
    }
}

/// Build a small straight-line function with no loops and no float ops: `fn(a: i32, b: i32) i32 {
/// let c1 = a + b; let c2 = c1 * a; let c3 = c2 - b; ret c3 }`. Used by both structural tests below.
/// Purely a dependent chain (no independent pure ops, no matching float-arith runs), so it gives
/// every pass nothing to do under an inert model.
fn buildStraightLine(func: *ir.function.Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const c1 = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const c2 = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .mul, .lhs = c1, .rhs = a } });
    const c3 = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .sub, .lhs = c2, .rhs = b } });
    func.setTerminator(entry, .{ .ret = c3 });
}

test "optimize is a no-op for a straight-line function under a single-issue, no-prefetch, no-SLP model" {
    // et-soc: issue_width 1 (unroll factor collapses to 1, and there is no loop here anyway),
    // prefetches() false (riscv64, so prefetch.run bails immediately), and this function has no
    // float arith at all, so vectorize's SLP scan finds no group regardless of vector width.
    // Scheduling may still reorder within the block (not counted in the return), so correctness is
    // checked via verify, not via printed/structural equality.
    const allocator = std.testing.allocator;
    var func = ir.function.Function.init(allocator);
    defer func.deinit();
    try buildStraightLine(&func);

    const changed = try optimize(allocator, &func, modelFor(.@"et-soc"));
    try std.testing.expect(!changed);

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "not calling optimize leaves a function exactly as built: today's behavior is unchanged" {
    // Guards that merely importing/linking this driver changes nothing: two identically-built
    // functions, neither ever passed to `optimize`, must stay structurally identical (same
    // block/inst counts and printed form).
    const allocator = std.testing.allocator;

    var a = ir.function.Function.init(allocator);
    defer a.deinit();
    try buildStraightLine(&a);

    var b = ir.function.Function.init(allocator);
    defer b.deinit();
    try buildStraightLine(&b);

    try std.testing.expectEqual(a.blockCount(), b.blockCount());
    var bi: usize = 0;
    while (bi < a.blockCount()) : (bi += 1) {
        const block: ir.function.Block = @enumFromInt(bi);
        try std.testing.expectEqual(a.blockInsts(block).len, b.blockInsts(block).len);
    }

    const text_a = try std.fmt.allocPrint(allocator, "{f}", .{a});
    defer allocator.free(text_a);
    const text_b = try std.fmt.allocPrint(allocator, "{f}", .{b});
    defer allocator.free(text_b);
    try std.testing.expectEqualStrings(text_a, text_b);
}
