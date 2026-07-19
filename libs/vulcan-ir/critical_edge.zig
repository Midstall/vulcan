//! Splits every critical control-flow edge by inserting a forwarding block. A
//! critical edge runs from a block with more than one successor (one ending in
//! an `if`) to a block with more than one predecessor. The Wimmer-Franz
//! allocator's resolution phase needs a block on every such edge to place its
//! shuffle moves, since neither endpoint can host them without affecting the
//! other edges. The pass is minimal (only genuinely critical edges are split)
//! and idempotent (a split edge is never critical again).

const std = @import("std");
const function = @import("function.zig");

const Function = function.Function;
const Block = function.Block;
const Value = function.Value;

pub const Error = std.mem.Allocator.Error;

/// Which side of an `if` an edge belongs to.
const Side = enum { then, @"else" };

/// Split every critical edge in `func` in place by inserting a parameter-less
/// forwarding block on it. A function with no critical edge is left unchanged.
pub fn splitCriticalEdges(allocator: std.mem.Allocator, func: *Function) Error!void {
    const original_count = func.blockCount();
    if (original_count == 0) return;

    // Predecessor count per (original) block, derived from the CFG exactly as
    // verification does: an `if` contributes one predecessor to each of its two
    // targets, a `jump` terminator one to its target. Snapshot it up front so
    // that inserting forwarding blocks cannot change a critical decision midway.
    const pred_count = try allocator.alloc(u32, original_count);
    defer allocator.free(pred_count);
    @memset(pred_count, 0);

    for (0..original_count) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => |cf| {
                    pred_count[@intFromEnum(cf.then.target)] += 1;
                    pred_count[@intFromEnum(cf.@"else".target)] += 1;
                },
                else => {},
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .jump => |j| pred_count[@intFromEnum(j.target)] += 1,
            .ret => {},
        };
    }

    // A block ending in an `if` is the only multi-successor block, so its two
    // outgoing edges are the only critical-edge candidates. Split each edge
    // whose target has more than one predecessor.
    for (0..original_count) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => {
                    try splitEdge(allocator, func, inst, pred_count, .then);
                    try splitEdge(allocator, func, inst, pred_count, .@"else");
                },
                else => {},
            }
        }
    }
}

/// Split one side of an `if` edge if it is critical. Inserts a forwarding block
/// that jumps to the original target with the original edge arguments, then
/// retargets the `if` branch to the forwarding block with no arguments.
fn splitEdge(allocator: std.mem.Allocator, func: *Function, inst: function.Inst, pred_count: []const u32, side: Side) Error!void {
    const cf = func.opcode(inst).@"if";
    const edge = switch (side) {
        .then => cf.then,
        .@"else" => cf.@"else",
    };

    // The pred already has two successors (it ends in `if`); the edge is
    // critical only when the target also has more than one predecessor.
    if (pred_count[@intFromEnum(edge.target)] <= 1) return;

    // Copy the edge arguments before any interning can reallocate the value-list
    // pool the slice points into.
    const orig_args = try allocator.dupe(Value, func.blockArgs(edge));
    defer allocator.free(orig_args);

    const fwd = try func.appendBlock();
    try func.setJump(fwd, edge.target, orig_args);

    // Retarget the branch to the forwarding block, which has no parameters.
    const empty = try func.internValueList(&.{});
    const op = func.opcodeMut(inst);
    switch (side) {
        .then => op.@"if".then = .{ .target = fwd, .args = empty },
        .@"else" => op.@"if".@"else" = .{ .target = fwd, .args = empty },
    }
}

const verify = @import("verify.zig");

/// Find the `if` instruction of a block, or null if it has none.
fn findIf(func: *const Function, block: Block) ?function.Inst {
    for (func.blockInsts(block)) |inst| {
        switch (func.opcode(inst)) {
            .@"if" => return inst,
            else => {},
        }
    }
    return null;
}

test "critical edges: a diamond with a critical edge gets a forwarding block" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);

    // entry -[if]-> a, b ; a -[jump]-> b ; b: ret. The entry->b edge is critical
    // (entry has two successors, b has two predecessors {entry, a}).
    const entry = try func.appendBlock();
    const a = try func.appendBlock();
    const b = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.appendIf(entry, cond, .{ .target = a }, .{ .target = b });
    try func.setJump(a, b, &.{});
    func.setTerminator(b, .{ .ret = null });

    const before = func.blockCount();
    try splitCriticalEdges(std.testing.allocator, &func);

    // Exactly one forwarding block was added for the one critical edge.
    try std.testing.expectEqual(before + 1, func.blockCount());
    const fwd: Block = @enumFromInt(before);

    // The forwarding block jumps to b with no parameters of its own.
    try std.testing.expectEqual(@as(usize, 0), func.blockParams(fwd).len);
    const term = func.terminator(fwd).?;
    try std.testing.expectEqual(b, term.jump.target);

    // entry's `if` now branches then->a (unchanged) and else->fwd.
    const cf = func.opcode(findIf(&func, entry).?).@"if";
    try std.testing.expectEqual(a, cf.then.target);
    try std.testing.expectEqual(fwd, cf.@"else".target);

    var d = try verify.verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(d.ok());
}

test "critical edges: a CFG with no critical edge is unchanged" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);

    // A plain diamond: entry -[if]-> a, b ; a -> m ; b -> m ; m: ret. No edge is
    // critical: a and b each have one predecessor, and the two edges into m come
    // from single-successor `jump` blocks.
    const entry = try func.appendBlock();
    const a = try func.appendBlock();
    const b = try func.appendBlock();
    const m = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.appendIf(entry, cond, .{ .target = a }, .{ .target = b });
    try func.setJump(a, m, &.{});
    try func.setJump(b, m, &.{});
    func.setTerminator(m, .{ .ret = null });

    const before = func.blockCount();
    try splitCriticalEdges(std.testing.allocator, &func);

    try std.testing.expectEqual(before, func.blockCount());

    var d = try verify.verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(d.ok());
}

test "critical edges: forwarding block forwards the original edge arguments" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    // entry(x) -[if]-> a, b(x) ; a -[jump]-> b(x) ; b(p): ret p. The entry->b
    // edge is critical and carries the argument x.
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const a = try func.appendBlock();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, i32_t);

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.appendIf(entry, cond, .{ .target = a }, .{ .target = b, .args = &.{x} });
    try func.setJump(a, b, &.{x});
    func.setTerminator(b, .{ .ret = p });

    const before = func.blockCount();
    try splitCriticalEdges(std.testing.allocator, &func);

    try std.testing.expectEqual(before + 1, func.blockCount());
    const fwd: Block = @enumFromInt(before);

    // The forwarding block's jump carries exactly the original edge argument x,
    // satisfying b's single parameter.
    const term = func.terminator(fwd).?;
    try std.testing.expectEqual(b, term.jump.target);
    const args = func.blockArgs(term.jump);
    try std.testing.expectEqual(@as(usize, 1), args.len);
    try std.testing.expectEqual(x, args[0]);

    // entry's else edge now targets fwd and passes no arguments.
    const cf = func.opcode(findIf(&func, entry).?).@"if";
    try std.testing.expectEqual(fwd, cf.@"else".target);
    try std.testing.expectEqual(@as(usize, 0), func.blockArgs(cf.@"else").len);

    var d = try verify.verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(d.ok());
}

test "critical edges: splitting is idempotent" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const a = try func.appendBlock();
    const b = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.appendIf(entry, cond, .{ .target = a }, .{ .target = b });
    try func.setJump(a, b, &.{});
    func.setTerminator(b, .{ .ret = null });

    try splitCriticalEdges(std.testing.allocator, &func);
    const after_first = func.blockCount();

    // A second run finds no critical edge (the inserted block has one
    // predecessor and one successor), so the block count does not grow.
    try splitCriticalEdges(std.testing.allocator, &func);
    try std.testing.expectEqual(after_first, func.blockCount());

    var d = try verify.verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(d.ok());
}
