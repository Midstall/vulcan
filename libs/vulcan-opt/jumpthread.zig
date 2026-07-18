//! Jump threading over identity-forwarding blocks: an empty block that does nothing but pass its own
//! parameters straight to a single successor is a pure detour, so every edge into it is redirected
//! to that successor. These forwarding blocks are what critical-edge splitting and structured
//! lowering leave behind; removing them shortens branch chains and helps fallthrough. The block
//! itself is left in place as dead code (the reachability-aware analyses ignore it), matching how
//! branchfold leaves an unreachable arm. Only the identity case is threaded (the forwarded arguments
//! are exactly the block's parameters), so no argument substitution is needed and it cannot go
//! wrong; condition-implication threading and tail duplication are separate, heavier transforms.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "jumpthread", .run = run };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    const n = func.blockCount();
    if (n == 0) return false;

    // forward_to[b] = the successor to which block b is a pure identity forwarder, else null.
    const forward_to = try allocator.alloc(?Block, n);
    defer allocator.free(forward_to);
    for (0..n) |bi| forward_to[bi] = identityForwardTarget(func, @enumFromInt(bi));

    var changed = false;
    for (0..n) |bi| {
        const block: Block = @enumFromInt(bi);
        // Redirect this block's out-edges that land on a forwarder to the forwarder's target.
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => |cf| {
                    if (retarget(forward_to, cf.then.target)) |dest| {
                        func.opcodeMut(inst).@"if".then.target = dest;
                        changed = true;
                    }
                    if (retarget(forward_to, cf.@"else".target)) |dest| {
                        func.opcodeMut(inst).@"if".@"else".target = dest;
                        changed = true;
                    }
                },
                else => {},
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .jump => |jmp| if (retarget(forward_to, jmp.target)) |dest| {
                func.terminatorPtr(block).*.?.jump.target = dest;
                changed = true;
            },
            .ret => {},
        };
    }
    return changed;
}

/// The final destination for an edge into `target`: follows a chain of identity forwarders (bounded
/// by the block count to avoid looping on a forwarder cycle), or null if `target` is not a forwarder.
fn retarget(forward_to: []const ?Block, target: Block) ?Block {
    var dest = forward_to[@intFromEnum(target)] orelse return null;
    var steps: usize = 0;
    while (steps < forward_to.len) : (steps += 1) {
        const next = forward_to[@intFromEnum(dest)] orelse return dest;
        if (next == dest) return dest; // self-forwarder guard (should not arise)
        dest = next;
    }
    return dest;
}

/// If `block` is a pure identity forwarder (not the entry, no instructions, terminated by a jump to a
/// different block whose arguments are exactly this block's parameters in order), return that
/// successor. Otherwise null.
fn identityForwardTarget(func: *const Function, block: Block) ?Block {
    if (@intFromEnum(block) == 0) return null; // never redirect away from the entry
    if (func.blockInsts(block).len != 0) return null; // must do nothing but forward
    const term = func.terminator(block) orelse return null;
    const jmp = switch (term) {
        .jump => |j| j,
        .ret => return null,
    };
    if (jmp.target == block) return null; // a self-loop is not a forwarder
    const params = func.blockParams(block);
    const args = func.blockArgs(jmp);
    if (!std.mem.eql(ir.function.Value, params, args)) return null; // only the identity case
    return jmp.target;
}

const testing = std.testing;

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

fn i32Ty(func: *Function) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}

test "a jump through an identity-forwarding block is redirected to its target" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const entry = try func.appendBlock();
    const mid = try func.appendBlock();
    const dest = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const mp = try func.appendBlockParam(mid, t);
    const dp = try func.appendBlockParam(dest, t);
    try func.setJump(entry, mid, &.{x});
    try func.setJump(mid, dest, &.{mp}); // mid forwards its param straight through
    func.setTerminator(dest, .{ .ret = dp });

    try testing.expect(try runOnce(allocator, &func));
    const term = func.terminator(entry).?;
    try testing.expectEqual(dest, term.jump.target); // entry now jumps past mid
    try testing.expectEqual(x, func.blockArgs(term.jump)[0]);
}

test "an if edge into a forwarding block is redirected" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const mid = try func.appendBlock();
    const other = try func.appendBlock();
    const dest = try func.appendBlock();
    const c = try func.appendBlockParam(entry, bool_t);
    const x = try func.appendBlockParam(entry, t);
    const mp = try func.appendBlockParam(mid, t);
    const dp = try func.appendBlockParam(dest, t);
    const op = try func.appendBlockParam(other, t);
    try func.appendIf(entry, c, .{ .target = mid, .args = &.{x} }, .{ .target = other, .args = &.{x} });
    try func.setJump(mid, dest, &.{mp});
    func.setTerminator(other, .{ .ret = op });
    func.setTerminator(dest, .{ .ret = dp });

    try testing.expect(try runOnce(allocator, &func));
    const cf = func.opcode(func.blockInsts(entry)[0]).@"if";
    try testing.expectEqual(dest, cf.then.target); // then edge threaded past mid
    try testing.expectEqual(other, cf.@"else".target); // else edge unchanged
}

test "a block that computes something is not a forwarder" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const entry = try func.appendBlock();
    const mid = try func.appendBlock();
    const dest = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const mp = try func.appendBlockParam(mid, t);
    const dp = try func.appendBlockParam(dest, t);
    try func.setJump(entry, mid, &.{x});
    const doubled = try func.appendArithImm(mid, t, .mul, mp, 2); // mid does real work
    try func.setJump(mid, dest, &.{doubled});
    func.setTerminator(dest, .{ .ret = dp });

    try testing.expect(!try runOnce(allocator, &func)); // mid is not a pure forwarder
    try testing.expectEqual(mid, func.terminator(entry).?.jump.target);
}
