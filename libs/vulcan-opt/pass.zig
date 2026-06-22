//! Pass manager: runs a sequence of function passes, lazily computing and caching
//! analyses, invalidating them whenever a pass reports a change.

const std = @import("std");
const ir = @import("vulcan-ir");
const dom = @import("dominators.zig");
const loops_mod = @import("loops.zig");

const Function = ir.function.Function;

pub const Error = std.mem.Allocator.Error;

/// Lazily computes and caches a function's analyses. A pass queries it for what
/// it needs. The manager recomputes on demand after an invalidation.
pub const Analyses = struct {
    allocator: std.mem.Allocator,
    func: *const Function,
    doms: ?dom.Dominators = null,
    loop_info: ?loops_mod.LoopInfo = null,

    pub fn deinit(self: *Analyses) void {
        self.invalidate();
    }

    /// The dominator analysis, computed once and cached until invalidation.
    pub fn dominators(self: *Analyses) Error!*const dom.Dominators {
        if (self.doms == null) self.doms = try dom.compute(self.allocator, self.func);
        return &self.doms.?;
    }

    /// The natural-loop analysis, computed once and cached until invalidation.
    pub fn loops(self: *Analyses) Error!*const loops_mod.LoopInfo {
        if (self.loop_info == null) self.loop_info = try loops_mod.analyze(self.allocator, self.func);
        return &self.loop_info.?;
    }

    /// Drop every cached analysis (call after the function is modified).
    pub fn invalidate(self: *Analyses) void {
        if (self.doms) |*d| d.deinit(self.allocator);
        self.doms = null;
        if (self.loop_info) |*l| l.deinit(self.allocator);
        self.loop_info = null;
    }
};

/// A function pass: transforms `func`, returning whether it changed it, and may
/// query `analyses` for cached analyses.
pub const Pass = struct {
    name: []const u8,
    run: *const fn (allocator: std.mem.Allocator, func: *Function, analyses: *Analyses) Error!bool,
};

/// Run each pass once in order. Analyses are invalidated after any modifying
/// pass. Returns whether anything changed.
pub fn run(allocator: std.mem.Allocator, func: *Function, passes: []const Pass) Error!bool {
    var analyses = Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();

    var changed = false;
    for (passes) |pass| {
        if (try pass.run(allocator, func, &analyses)) {
            analyses.invalidate();
            changed = true;
        }
    }
    return changed;
}

/// Run the passes repeatedly until a full sweep makes no change (a fixpoint), up
/// to `max_iters` sweeps. Returns whether anything changed overall.
pub fn runToFixpoint(allocator: std.mem.Allocator, func: *Function, passes: []const Pass, max_iters: usize) Error!bool {
    var analyses = Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();

    var changed_overall = false;
    var iter: usize = 0;
    while (iter < max_iters) : (iter += 1) {
        var changed = false;
        for (passes) |pass| {
            if (try pass.run(allocator, func, &analyses)) {
                analyses.invalidate();
                changed = true;
            }
        }
        if (!changed) break;
        changed_overall = true;
    }
    return changed_overall;
}
