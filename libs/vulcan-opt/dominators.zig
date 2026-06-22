//! Dominator analysis over a function's CFG. Block A dominates B when every path
//! from entry to B passes through A. Iterative data-flow fixpoint. Exposes the
//! full dominance relation and each block's immediate dominator (dominator tree).

const std = @import("std");
const ir = @import("vulcan-ir");
const cfg_mod = @import("cfg.zig");

const Function = ir.function.Function;

pub const Dominators = struct {
    n: usize,
    /// `dom[b * n + a]` is true when block `a` dominates block `b`.
    dom: []bool,
    /// The immediate dominator of each block (itself for the entry and for
    /// unreachable blocks).
    idom: []u32,

    pub fn deinit(self: *Dominators, allocator: std.mem.Allocator) void {
        allocator.free(self.dom);
        allocator.free(self.idom);
    }

    pub fn dominates(self: *const Dominators, a: usize, b: usize) bool {
        return self.dom[b * self.n + a];
    }

    /// Whether `a` strictly dominates `b` (dominates and is not `b`).
    pub fn strictlyDominates(self: *const Dominators, a: usize, b: usize) bool {
        return a != b and self.dominates(a, b);
    }

    pub fn immediateDominator(self: *const Dominators, b: usize) u32 {
        return self.idom[b];
    }
};

/// Compute the dominators of `func` (entry is block 0). The caller owns the
/// result (`deinit`).
pub fn compute(allocator: std.mem.Allocator, func: *const Function) std.mem.Allocator.Error!Dominators {
    var cfg = try cfg_mod.build(allocator, func);
    defer cfg.deinit(allocator);
    const n = cfg.blockCount();

    const dom = try allocator.alloc(bool, n * n);
    errdefer allocator.free(dom);
    @memset(dom, true);
    // The entry is dominated only by itself.
    for (0..n) |a| dom[a] = (a == 0);

    const tmp = try allocator.alloc(bool, n);
    defer allocator.free(tmp);

    var changed = true;
    while (changed) {
        changed = false;
        for (1..n) |b| {
            if (cfg.predecessors(b).len == 0) continue; // unreachable
            // dom(b) = {b} U (intersection of dom(p) over predecessors p).
            for (0..n) |a| {
                var all = true;
                for (cfg.predecessors(b)) |p| {
                    if (!dom[@as(usize, p) * n + a]) {
                        all = false;
                        break;
                    }
                }
                tmp[a] = all;
            }
            tmp[b] = true;
            for (0..n) |a| {
                if (dom[b * n + a] != tmp[a]) {
                    dom[b * n + a] = tmp[a];
                    changed = true;
                }
            }
        }
    }

    // Immediate dominators: idom(b) is the strict dominator of b that every other
    // strict dominator of b also dominates (the closest one).
    const idom = try allocator.alloc(u32, n);
    errdefer allocator.free(idom);
    for (0..n) |b| {
        idom[b] = @intCast(b); // entry and unreachable map to themselves
        if (b == 0) continue;
        for (0..n) |d| {
            if (d == b or !dom[b * n + d]) continue; // d must strictly dominate b
            var lowest = true;
            for (0..n) |o| {
                if (o == b or o == d or !dom[b * n + o]) continue;
                // Another strict dominator o that d does not dominate => d is not idom.
                if (!dom[d * n + o]) {
                    lowest = false;
                    break;
                }
            }
            if (lowest) {
                idom[b] = @intCast(d);
                break;
            }
        }
    }

    return .{ .n = n, .dom = dom, .idom = idom };
}

test "dominators of a diamond" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const c = try func.appendBlockParam(b0, bool_t);
    const b1 = try func.appendBlock();
    const b2 = try func.appendBlock();
    const b3 = try func.appendBlock();
    const v = try func.appendBlockParam(b3, i32_t);
    try func.appendIf(b0, c, .{ .target = b1 }, .{ .target = b2 });
    const x = try func.appendInst(b1, i32_t, .{ .iconst = 1 });
    try func.setJump(b1, b3, &.{x});
    const y = try func.appendInst(b2, i32_t, .{ .iconst = 2 });
    try func.setJump(b2, b3, &.{y});
    func.setTerminator(b3, .{ .ret = v });

    var doms = try compute(allocator, &func);
    defer doms.deinit(allocator);

    // b0 dominates all. b3 is dominated by b0 and b3 only (not b1 or b2).
    try std.testing.expect(doms.dominates(0, 3));
    try std.testing.expect(!doms.dominates(1, 3));
    try std.testing.expect(!doms.dominates(2, 3));
    try std.testing.expect(doms.strictlyDominates(0, 3));
    try std.testing.expect(!doms.strictlyDominates(3, 3));
    // The dominator tree: idom(b1)=idom(b2)=idom(b3)=b0.
    try std.testing.expectEqual(@as(u32, 0), doms.immediateDominator(1));
    try std.testing.expectEqual(@as(u32, 0), doms.immediateDominator(2));
    try std.testing.expectEqual(@as(u32, 0), doms.immediateDominator(3));
}
