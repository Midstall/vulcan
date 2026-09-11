//! Target-independent optimization framework: pass manager with cached analyses,
//! plus the analyses and transforms over Vulcan IR. Freestanding (no libc/OS,
//! allocator per call).

const std = @import("std");
const ir = @import("vulcan-ir");

pub const cfg = @import("vulcan-opt/cfg.zig");
pub const dominators = @import("vulcan-opt/dominators.zig");
pub const loops = @import("vulcan-opt/loops.zig");
pub const uniform = @import("vulcan-opt/uniform.zig");
pub const pass = @import("vulcan-opt/pass.zig");
pub const mem2reg = @import("vulcan-opt/mem2reg.zig");
pub const knownbits = @import("vulcan-opt/knownbits.zig");
pub const loadfwd = @import("vulcan-opt/loadfwd.zig");
pub const jumpthread = @import("vulcan-opt/jumpthread.zig");
pub const constfold = @import("vulcan-opt/constfold.zig");
pub const simplify = @import("vulcan-opt/simplify.zig");
pub const strength = @import("vulcan-opt/strength.zig");
pub const immform = @import("vulcan-opt/immform.zig");
pub const branchfold = @import("vulcan-opt/branchfold.zig");
pub const gvn = @import("vulcan-opt/gvn.zig");
pub const licm = @import("vulcan-opt/licm.zig");
pub const ivsr = @import("vulcan-opt/ivsr.zig");
pub const inlining = @import("vulcan-opt/inline.zig");
pub const dce = @import("vulcan-opt/dce.zig");
pub const lto = @import("vulcan-opt/lto.zig");
pub const pgo = @import("vulcan-opt/pgo.zig");
pub const lowerdiv = @import("vulcan-opt/lowerdiv.zig");
pub const vectorize = @import("vulcan-opt/vectorize.zig");
pub const blocklayout = @import("vulcan-opt/blocklayout.zig");
pub const microarch = @import("vulcan-opt/microarch.zig");

/// Default pipeline: constant folding, algebraic simplification, strength reduction, GVN/CSE, LICM,
/// then DCE, to a fixpoint.
pub const default_pipeline = [_]pass.Pass{
    mem2reg.pass_def,
    loadfwd.pass_def,
    constfold.pass_def,
    simplify.pass_def,
    knownbits.pass_def,
    strength.pass_def,
    immform.pass_def,
    branchfold.pass_def,
    jumpthread.pass_def,
    gvn.pass_def,
    licm.pass_def,
    dce.pass_def,
};

/// The late pipeline: induction-variable strength reduction, then just enough clean-up to settle
/// what it leaves behind. It is SEPARATE from the default pipeline for a phase-ordering reason.
///
/// `ivsr` rewrites a loop's addresses into loop-carried pointers, which grows the header's
/// parameter list and turns index arithmetic into a back-edge step. Four microarch idiom
/// recognizers read the SHAPE it replaces and match on an exact parameter count:
/// `loopvec.recognize` wants exactly one header parameter, `loopvec.recognizeReduction` and
/// `splitunroll` want the induction plus reduction accumulators and nothing else, and `dotprod`
/// wants exactly four. Running `ivsr` before them costs vectorization, which is worth more than
/// the multiply it removes. So it runs LAST, after those recognizers have had their look. See
/// `optimizeEarly` and `optimizeLate`, and `frontends/vcc.zig`, which is where both halves meet.
pub const late_pipeline = [_]pass.Pass{
    ivsr.pass_def,
    constfold.pass_def,
    simplify.pass_def,
    strength.pass_def,
    immform.pass_def,
    dce.pass_def,
};

/// Optimize `func` in place: the default pipeline, then the late one. Returns whether anything
/// changed. A caller that runs `microarch.optimize` over the same function calls the two halves
/// separately instead, with the microarch layer in between.
pub fn optimize(allocator: std.mem.Allocator, func: *ir.function.Function) pass.Error!bool {
    const early = try optimizeEarly(allocator, func);
    const late = try optimizeLate(allocator, func);
    return early or late;
}

/// The default pipeline to a fixpoint, then block layout. This is the whole of `optimize` except
/// the late pipeline, and it leaves every loop in the shape the microarch idiom recognizers read.
pub fn optimizeEarly(allocator: std.mem.Allocator, func: *ir.function.Function) pass.Error!bool {
    const optimized = try pass.runToFixpoint(allocator, func, &default_pipeline, 16);
    // Block layout is a ONE-SHOT run after the pipeline fixpoint, not an iterated pass: it computes a
    // single fall-through-friendly linearization and permutes the blocks into it once. It is
    // dominance-respecting by construction and falls back to the original order if unsure, so it is
    // always safe on the machine-backend path.
    const laid_out = try blocklayout.layout(allocator, func, null);
    return optimized or laid_out;
}

/// The late pipeline to a fixpoint. Safe to run after `microarch.optimize`: it adds no block and
/// moves none, so the layout `optimizeEarly` settled still holds. It does append to the preheader
/// and the latch of each loop it reduces, which `microarch.schedule` has by then already ordered,
/// so those few instructions land unscheduled at the end of their blocks.
pub fn optimizeLate(allocator: std.mem.Allocator, func: *ir.function.Function) pass.Error!bool {
    return pass.runToFixpoint(allocator, func, &late_pipeline, 16);
}

test {
    std.testing.refAllDecls(@This());
}
