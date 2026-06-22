//! Loop-invariant code motion. Within each natural loop that has a preheader, an
//! instruction is invariant when it is a safe-to-speculate pure op whose operands
//! are all defined outside the loop (or themselves invariant). Such instructions
//! move to the preheader, which dominates the loop and runs once before it.
//!
//! Only non-trapping pure ops are hoisted, since hoisting speculates them ahead of
//! the loop body: no loads (may fault), no `div`/`rem` (may trap), no `alloca`
//! (distinct addresses), nothing impure.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;
const BinOp = ir.function.BinOp;

pub const pass_def = pass.Pass{ .name = "licm", .run = run };

/// An instruction to hoist, with the loop block it currently lives in.
const Hoist = struct { inst: Inst, from: u32 };

fn nonTrapping(op: BinOp) bool {
    return op != .div and op != .rem;
}

/// Whether an instruction may be hoisted out of a loop if it is invariant.
fn hoistable(opcode: ir.function.Opcode) bool {
    return switch (opcode) {
        .iconst, .fconst, .icmp, .select, .convert, .unary, .extract, .global_addr => true,
        .arith => |a| nonTrapping(a.op),
        .arith_imm => |a| nonTrapping(a.op),
        .alloca, .struct_new, .load, .store, .call, .call_indirect, .@"if" => false,
    };
}

/// Whether every value operand of `inst` is currently invariant.
fn operandsInvariant(func: *const Function, inst: Inst, invariant: []const bool) bool {
    const inv = struct {
        fn f(i: []const bool, v: Value) bool {
            return i[@intFromEnum(v)];
        }
    }.f;
    return switch (func.opcode(inst)) {
        .iconst, .fconst, .global_addr => true,
        .arith => |a| inv(invariant, a.lhs) and inv(invariant, a.rhs),
        .arith_imm => |a| inv(invariant, a.lhs),
        .icmp => |c| inv(invariant, c.lhs) and inv(invariant, c.rhs),
        .select => |s| inv(invariant, s.cond) and inv(invariant, s.then) and inv(invariant, s.@"else"),
        .convert => |cv| inv(invariant, cv.value),
        .unary => |u| inv(invariant, u.value),
        .extract => |e| inv(invariant, e.aggregate),
        else => false,
    };
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const info = try analyses.loops();
    if (info.loops.len == 0) return false;

    // Which block defines each value (recomputed per loop, since hoisting in an
    // earlier loop moves instructions between blocks).
    const def_block = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(def_block);

    // Instructions to move, paired with their source block. Collected in
    // dependency-respecting order (program order, fixpoint), then applied.
    var to_hoist: std.ArrayList(Hoist) = .empty;
    defer to_hoist.deinit(allocator);

    const invariant = try allocator.alloc(bool, func.valueCount());
    defer allocator.free(invariant);

    var changed = false;
    for (info.loops) |loop| {
        const pre = loop.preheader orelse continue;

        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockParams(block)) |p| def_block[@intFromEnum(p)] = @intCast(bi);
            for (func.blockInsts(block)) |inst| {
                if (func.instResult(inst)) |r| def_block[@intFromEnum(r)] = @intCast(bi);
            }
        }

        // A value is invariant if it is defined outside the loop. Loop block
        // params stay variant. Loop instructions become invariant once proven.
        for (invariant, 0..) |*v, i| v.* = !loop.contains(def_block[i]);

        to_hoist.clearRetainingCapacity();
        var again = true;
        while (again) {
            again = false;
            for (0..func.blockCount()) |bi| {
                if (!loop.contains(bi)) continue;
                for (func.blockInsts(@enumFromInt(bi))) |inst| {
                    const result = func.instResult(inst) orelse continue;
                    if (invariant[@intFromEnum(result)]) continue; // already hoisted/invariant
                    if (!hoistable(func.opcode(inst))) continue;
                    if (!operandsInvariant(func, inst, invariant)) continue;
                    invariant[@intFromEnum(result)] = true;
                    try to_hoist.append(allocator, .{ .inst = inst, .from = @intCast(bi) });
                    again = true;
                }
            }
        }
        if (to_hoist.items.len == 0) continue;

        // Remove the hoisted instructions from their loop blocks.
        for (0..func.blockCount()) |bi| {
            if (!loop.contains(bi)) continue;
            const insts = func.blockInstsMut(@enumFromInt(bi));
            var w: usize = 0;
            for (insts.items) |inst| {
                if (isHoisted(to_hoist.items, inst)) continue;
                insts.items[w] = inst;
                w += 1;
            }
            insts.shrinkRetainingCapacity(w);
        }
        // Append them to the preheader (before its terminator), in order.
        const pre_insts = func.blockInstsMut(@enumFromInt(pre));
        for (to_hoist.items) |h| try pre_insts.append(allocator, h.inst);
        changed = true;
    }
    return changed;
}

fn isHoisted(list: []const Hoist, inst: Inst) bool {
    for (list) |h| if (h.inst == inst) return true;
    return false;
}

test "hoists a loop-invariant product to the preheader" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const y = try func.appendBlockParam(entry, i32_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);

    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    // inv = x * y is loop-invariant (x, y come from outside the loop).
    const inv = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    _ = inv;
    const next = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{next});
    func.setTerminator(done, .{ .ret = n });

    const entry_before = func.blockInsts(entry).len;
    const body_before = func.blockInsts(body).len;

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));

    // The invariant `x * y` left the body and landed in the preheader (entry).
    try std.testing.expectEqual(entry_before + 1, func.blockInsts(entry).len);
    try std.testing.expectEqual(body_before - 1, func.blockInsts(body).len);
}

test "does not hoist a loop-variant value" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);

    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    // `bi * bi` depends on the loop induction value, so it is not invariant.
    _ = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = bi, .rhs = bi } });
    const next = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{next});
    func.setTerminator(done, .{ .ret = n });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
}
