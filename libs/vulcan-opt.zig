//! Target-independent optimization framework: pass manager with cached analyses,
//! plus the analyses and transforms over Vulcan IR. Freestanding (no libc/OS,
//! allocator per call).

const std = @import("std");
const ir = @import("vulcan-ir");

pub const cfg = @import("vulcan-opt/cfg.zig");
pub const dominators = @import("vulcan-opt/dominators.zig");
pub const loops = @import("vulcan-opt/loops.zig");
pub const pass = @import("vulcan-opt/pass.zig");
pub const mem2reg = @import("vulcan-opt/mem2reg.zig");
pub const knownbits = @import("vulcan-opt/knownbits.zig");
pub const loadfwd = @import("vulcan-opt/loadfwd.zig");
pub const jumpthread = @import("vulcan-opt/jumpthread.zig");
pub const constfold = @import("vulcan-opt/constfold.zig");
pub const simplify = @import("vulcan-opt/simplify.zig");
pub const strength = @import("vulcan-opt/strength.zig");
pub const branchfold = @import("vulcan-opt/branchfold.zig");
pub const gvn = @import("vulcan-opt/gvn.zig");
pub const licm = @import("vulcan-opt/licm.zig");
pub const inlining = @import("vulcan-opt/inline.zig");
pub const dce = @import("vulcan-opt/dce.zig");
pub const lto = @import("vulcan-opt/lto.zig");
pub const pgo = @import("vulcan-opt/pgo.zig");
pub const lowerdiv = @import("vulcan-opt/lowerdiv.zig");
pub const vectorize = @import("vulcan-opt/vectorize.zig");
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
    branchfold.pass_def,
    jumpthread.pass_def,
    gvn.pass_def,
    licm.pass_def,
    dce.pass_def,
};

/// Optimize `func` in place with the default pipeline. Returns whether anything
/// changed.
pub fn optimize(allocator: std.mem.Allocator, func: *ir.function.Function) pass.Error!bool {
    return pass.runToFixpoint(allocator, func, &default_pipeline, 16);
}

test {
    std.testing.refAllDecls(@This());
}
