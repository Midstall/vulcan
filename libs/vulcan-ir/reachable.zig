//! Neutralizes the blocks unreachable from the entry so downstream passes see
//! exactly the reachable CFG. A pass that raises a loop nest into a single op
//! (matmul recognition) can orphan its original blocks, and the IR has no
//! block-delete, so the orphaned nest stays physically present and still holds
//! instructions that USE values defined in reachable blocks. The Wimmer-Franz
//! allocator's `buildIntervals` walks ALL blocks, so such a value would get a
//! live range built entirely from the dead region, one that does not contain
//! its def, tripping the allocator's SSA "def lies in the earliest range"
//! invariant. Emptying every unreachable block removes those spurious uses.
//!
//! The block is kept PHYSICALLY PRESENT (its index and enum handle stay stable)
//! so branch targets and relocations, which reference blocks by handle, are
//! undisturbed. This mirrors the transform proven in production on riscv64.

const std = @import("std");
const function = @import("function.zig");

const Function = function.Function;
const Block = function.Block;

pub const Error = std.mem.Allocator.Error;

/// Compute the block set reachable from the entry (block 0) and neutralize every
/// unreachable block in place: its parameters are emptied, its instruction list
/// is cleared, and its terminator is set to null. Returns the reachable set,
/// which a backend that skips dead blocks in emission can reuse.
///
/// The caller OWNS the returned slice and must release it with
/// `allocator.free`. Its length is `func.blockCount()` and index `bi` is true
/// when block `bi` is reachable from the entry.
///
/// For a function with every block already reachable this is a strict no-op:
/// nothing is emptied and the result is all-true.
pub fn neutralizeUnreachable(allocator: std.mem.Allocator, func: *Function) Error![]bool {
    const nblocks = func.blockCount();

    const reachable = try allocator.alloc(bool, nblocks);
    errdefer allocator.free(reachable);
    @memset(reachable, false);

    // BFS from the entry over the successor edges: an `if` instruction branches
    // to both of its targets, a `jump` terminator to its single target. This is
    // the SAME edge set the allocator's liveness fixpoint and the dominator
    // analysis walk, so the reachable set matches theirs exactly.
    if (nblocks > 0) {
        reachable[0] = true;
        var stack: std.ArrayList(u32) = .empty;
        defer stack.deinit(allocator);
        try stack.append(allocator, 0);
        // Each block is pushed at most once (guarded by `reachable`), so the
        // outer walk is bounded by `nblocks`.
        while (stack.pop()) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockInsts(block)) |inst| {
                switch (func.opcode(inst)) {
                    .@"if" => |cf| {
                        try visitSucc(allocator, reachable, &stack, cf.then.target);
                        try visitSucc(allocator, reachable, &stack, cf.@"else".target);
                    },
                    else => {},
                }
            }
            if (func.terminator(block)) |term| switch (term) {
                .jump => |j| try visitSucc(allocator, reachable, &stack, j.target),
                .ret => {},
            };
        }
    }

    // The entry (block 0) is reachable by construction, it seeds the walk above.

    // Neutralize every unreachable block. Bounded by `nblocks`.
    for (0..nblocks) |bi| {
        if (reachable[bi]) continue;
        const dead: Block = @enumFromInt(bi);
        try func.setBlockParams(dead, &.{});
        func.blockInstsMut(dead).clearRetainingCapacity();
        func.terminatorPtr(dead).* = null;
    }

    return reachable;
}

/// Mark `target` reachable and queue it if this is the first time it is seen.
fn visitSucc(allocator: std.mem.Allocator, reachable: []bool, stack: *std.ArrayList(u32), target: Block) Error!void {
    const ti = @intFromEnum(target);
    if (reachable[ti]) return;
    reachable[ti] = true;
    try stack.append(allocator, ti);
}

test "neutralizeUnreachable empties exactly the unreachable block" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    // entry -[jump]-> b1 -[ret]. b2 is unreachable (no block branches to it) yet it carries a
    // parameter, an instruction, and a terminator.
    const entry = try func.appendBlock();
    const b1 = try func.appendBlock();
    const b2 = try func.appendBlock();

    try func.setJump(entry, b1, &.{});
    func.setTerminator(b1, .{ .ret = null });

    _ = try func.appendBlockParam(b2, i32_t);
    _ = try func.appendInst(b2, i32_t, .{ .iconst = 7 });
    func.setTerminator(b2, .{ .ret = null });

    const reachable = try neutralizeUnreachable(std.testing.allocator, &func);
    defer std.testing.allocator.free(reachable);

    try std.testing.expectEqual(@as(usize, 3), reachable.len);
    try std.testing.expect(reachable[0]);
    try std.testing.expect(reachable[1]);
    try std.testing.expect(!reachable[2]);

    // b2 is fully neutralized: params empty, instructions cleared, terminator null.
    try std.testing.expectEqual(@as(usize, 0), func.blockParams(b2).len);
    try std.testing.expectEqual(@as(usize, 0), func.blockInsts(b2).len);
    try std.testing.expect(func.terminator(b2) == null);

    // The reachable blocks are untouched: b1 still returns.
    try std.testing.expect(func.terminator(b1) != null);
}

test "neutralizeUnreachable is a no-op for an all-reachable function" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);

    // entry -[if]-> a or b. a -[jump]-> b. b -[ret]. Every block reachable from the entry.
    const entry = try func.appendBlock();
    const a = try func.appendBlock();
    const b = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.appendIf(entry, cond, .{ .target = a }, .{ .target = b });
    try func.setJump(a, b, &.{});
    func.setTerminator(b, .{ .ret = null });

    const reachable = try neutralizeUnreachable(std.testing.allocator, &func);
    defer std.testing.allocator.free(reachable);

    try std.testing.expectEqual(@as(usize, 3), reachable.len);
    for (reachable) |r| try std.testing.expect(r);

    // Nothing was emptied: entry keeps its condition and `if` instructions, a keeps its jump, b its ret.
    try std.testing.expectEqual(@as(usize, 2), func.blockInsts(entry).len);
    try std.testing.expect(func.terminator(a) != null);
    try std.testing.expect(func.terminator(b) != null);
}
