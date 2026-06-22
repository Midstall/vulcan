//! Control-flow graph: per-block successor and predecessor lists, derived from
//! the high-IR `if` instruction edges and the block's jump terminator. The basis
//! for the dominator analysis and any control-flow-aware transform.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Block = ir.function.Block;

pub const Cfg = struct {
    /// Successor and predecessor block indices, one list per block.
    succ: []std.ArrayList(u32),
    pred: []std.ArrayList(u32),

    pub fn deinit(self: *Cfg, allocator: std.mem.Allocator) void {
        for (self.succ) |*s| s.deinit(allocator);
        for (self.pred) |*p| p.deinit(allocator);
        allocator.free(self.succ);
        allocator.free(self.pred);
    }

    pub fn successors(self: *const Cfg, block: usize) []const u32 {
        return self.succ[block].items;
    }

    pub fn predecessors(self: *const Cfg, block: usize) []const u32 {
        return self.pred[block].items;
    }

    pub fn blockCount(self: *const Cfg) usize {
        return self.succ.len;
    }

    /// Reverse postorder of the blocks reachable from the entry (block 0). A
    /// definition's block precedes every block it dominates, so processing values
    /// in this order numbers operands before their uses. The caller owns the slice.
    pub fn reversePostorder(self: *const Cfg, allocator: std.mem.Allocator) std.mem.Allocator.Error![]u32 {
        const n = self.blockCount();
        var order: std.ArrayList(u32) = .empty;
        errdefer order.deinit(allocator);
        if (n == 0) return order.toOwnedSlice(allocator);

        const visited = try allocator.alloc(bool, n);
        defer allocator.free(visited);
        @memset(visited, false);

        const Frame = struct { block: u32, next: usize };
        var stack: std.ArrayList(Frame) = .empty;
        defer stack.deinit(allocator);
        visited[0] = true;
        try stack.append(allocator, .{ .block = 0, .next = 0 });
        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            const succs = self.successors(top.block);
            if (top.next < succs.len) {
                const s = succs[top.next];
                top.next += 1;
                if (!visited[s]) {
                    visited[s] = true;
                    try stack.append(allocator, .{ .block = s, .next = 0 });
                }
            } else {
                try order.append(allocator, top.block); // postorder on the way up
                _ = stack.pop();
            }
        }
        std.mem.reverse(u32, order.items);
        return order.toOwnedSlice(allocator);
    }
};

/// Build the CFG for `func`. The caller owns the result (`deinit`).
pub fn build(allocator: std.mem.Allocator, func: *const Function) std.mem.Allocator.Error!Cfg {
    const n = func.blockCount();
    const succ = try allocator.alloc(std.ArrayList(u32), n);
    errdefer allocator.free(succ);
    for (succ) |*s| s.* = .empty;
    const pred = try allocator.alloc(std.ArrayList(u32), n);
    errdefer allocator.free(pred);
    for (pred) |*p| p.* = .empty;

    // A block's successors come from its `if` edges (high IR) and its terminator.
    for (0..n) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => |cf| {
                    try succ[bi].append(allocator, @intFromEnum(cf.then.target));
                    try succ[bi].append(allocator, @intFromEnum(cf.@"else".target));
                },
                else => {},
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .jump => |j| try succ[bi].append(allocator, @intFromEnum(j.target)),
            .ret => {},
        };
    }
    for (0..n) |bi| {
        for (succ[bi].items) |s| try pred[s].append(allocator, @intCast(bi));
    }

    return .{ .succ = succ, .pred = pred };
}

test "cfg of a diamond has the right edges" {
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
    // b0: if c -> b1 else b2, b1 -> b3(x), b2 -> b3(y), b3: ret v
    try func.appendIf(b0, c, .{ .target = b1 }, .{ .target = b2 });
    const x = try func.appendInst(b1, i32_t, .{ .iconst = 1 });
    try func.setJump(b1, b3, &.{x});
    const y = try func.appendInst(b2, i32_t, .{ .iconst = 2 });
    try func.setJump(b2, b3, &.{y});
    func.setTerminator(b3, .{ .ret = v });

    var cfg = try build(allocator, &func);
    defer cfg.deinit(allocator);

    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, cfg.successors(0));
    try std.testing.expectEqualSlices(u32, &.{3}, cfg.successors(1));
    try std.testing.expectEqualSlices(u32, &.{3}, cfg.successors(2));
    try std.testing.expectEqualSlices(u32, &.{}, cfg.successors(3));
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, cfg.predecessors(3));
}
