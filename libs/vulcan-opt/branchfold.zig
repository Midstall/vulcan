//! Branch folding fires when a structured `if`'s condition is a compile-time constant, which
//! constant folding produces from an `icmp`. It replaces the `if` with an unconditional `jump` to
//! the taken arm, removing the branch and leaving the not-taken arm dead.
//!
//! The high-IR `if` is a non-terminating instruction that acts as the block's exit, so folding it
//! means dropping that instruction and setting the block's terminator to the taken `jump`. Blocks
//! that become unreachable stay in place as dead code. The dominator and loop analyses are
//! reachability-aware and ignore them, see dominators.isReachable, so this pass does no block
//! removal of its own.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "branchfold", .run = run };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;

    // Constant values, to spot a constant `if` condition.
    var consts = try allocator.alloc(?i64, func.valueCount());
    defer allocator.free(consts);
    @memset(consts, null);
    for (0..func.instCount()) |i| {
        const inst: Inst = @enumFromInt(i);
        if (func.opcode(inst) == .iconst) {
            if (func.instResult(inst)) |r| consts[@intFromEnum(r)] = func.opcode(inst).iconst;
        }
    }

    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        // Find this block's structured `if` exit (if any).
        var if_at: ?usize = null;
        for (func.blockInsts(block), 0..) |inst, j| {
            if (func.opcode(inst) == .@"if") {
                if_at = j;
                break;
            }
        }
        const j = if_at orelse continue;
        const cf = func.opcode(func.blockInsts(block)[j]).@"if";

        // A constant condition jumps straight to the taken arm; a branch whose two arms are the same
        // edge (same target and same arguments) is pointless whatever the condition, so it collapses
        // to that single jump. Either way the `if` is dropped and the block gets an unconditional
        // terminator.
        const taken: ir.function.Jump = if (consts[@intFromEnum(cf.cond)]) |cv|
            (if (cv != 0) cf.then else cf.@"else")
        else if (cf.then.target == cf.@"else".target and sameArgs(func, cf.then, cf.@"else"))
            cf.then
        else
            continue;
        _ = func.blockInstsMut(block).orderedRemove(j);
        func.setTerminator(block, .{ .jump = taken });
        changed = true;
    }
    return changed;
}

/// Whether two edges pass identical argument lists (so their only difference could be the target).
fn sameArgs(func: *const Function, a: ir.function.Jump, b: ir.function.Jump) bool {
    return std.mem.eql(ir.function.Value, func.blockArgs(a), func.blockArgs(b));
}

const testing = std.testing;

test "branchfold: a constant condition becomes an unconditional jump to the taken arm" {
    const allocator = testing.allocator;
    inline for (.{ .{ @as(i64, 1), true }, .{ @as(i64, 0), false } }) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const bool_t = try func.types.intern(.bool);
        const entry = try func.appendBlock();
        const a = try func.appendBlock();
        const b = try func.appendBlock();
        const x = try func.appendBlockParam(entry, t);
        const av = try func.appendBlockParam(a, t);
        const bv = try func.appendBlockParam(b, t);
        const cond = try func.appendInst(entry, bool_t, .{ .iconst = case[0] });
        try func.appendIf(entry, cond, .{ .target = a, .args = &.{x} }, .{ .target = b, .args = &.{x} });
        func.setTerminator(a, .{ .ret = av });
        func.setTerminator(b, .{ .ret = bv });

        var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
        defer analyses.deinit();
        try testing.expect(try run(allocator, &func, &analyses));
        const term = func.terminator(entry).?;
        try testing.expect(term == .jump);
        try testing.expectEqual(if (case[1]) a else b, term.jump.target);
        for (func.blockInsts(entry)) |inst| try testing.expect(func.opcode(inst) != .@"if"); // if is gone
    }
}

test "branchfold: a runtime branch whose arms are the same edge collapses to a jump" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const target = try func.appendBlock();
    const cond = try func.appendBlockParam(entry, bool_t); // runtime condition
    const tv = try func.appendBlockParam(target, t);
    const x = try func.appendInst(entry, t, .{ .iconst = 5 });
    // if cond -> target(x) else target(x): both arms identical.
    try func.appendIf(entry, cond, .{ .target = target, .args = &.{x} }, .{ .target = target, .args = &.{x} });
    func.setTerminator(target, .{ .ret = tv });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(try run(allocator, &func, &analyses));
    const term = func.terminator(entry).?;
    try testing.expect(term == .jump);
    try testing.expectEqual(target, term.jump.target);
    for (func.blockInsts(entry)) |inst| try testing.expect(func.opcode(inst) != .@"if");
}

test "branchfold: a runtime condition is left alone" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlock();
    const b = try func.appendBlock();
    const cond = try func.appendBlockParam(entry, bool_t); // runtime
    const av = try func.appendBlockParam(a, t);
    const bv = try func.appendBlockParam(b, t);
    const x = try func.appendInst(entry, t, .{ .iconst = 5 });
    try func.appendIf(entry, cond, .{ .target = a, .args = &.{x} }, .{ .target = b, .args = &.{x} });
    func.setTerminator(a, .{ .ret = av });
    func.setTerminator(b, .{ .ret = bv });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(!try run(allocator, &func, &analyses)); // nothing folded
}
