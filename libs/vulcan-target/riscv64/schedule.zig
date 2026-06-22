//! Microarchitecture-aware list scheduling for the River CPU. Reorders a block's
//! independent instructions to hide functional-unit latency: a multi-cycle
//! producer (multiply, load) issues early so its result is ready when a consumer
//! needs it, with independent work filling the gap. Driven by per-opcode
//! latencies from target data. Memory ops and the `if` control statement act as
//! barriers. Movable (pure value) ops reorder within the regions between them.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;

/// River functional-unit latencies, in issue cycles, keyed by opcode. Multiply
/// and divide occupy the multiplier for several cycles. Loads have use-latency.
/// everything else is single-cycle. These are the target-data knob the scheduler
/// turns, so a different microarchitecture only swaps this table.
pub fn riverLatency(op: ir.function.Opcode) u32 {
    return switch (op) {
        .arith => |a| switch (a.op) {
            .mul => 3,
            .div, .rem => 6,
            else => 1,
        },
        .load => 2,
        .convert => 2,
        .unary => 2,
        else => 1,
    };
}

/// A pure value op may be freely reordered within its block. Memory ops and the
/// `if` control statement are pinned (reordering across them needs barriers).
fn movable(op: ir.function.Opcode) bool {
    return switch (op) {
        .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .struct_new, .extract, .convert, .unary, .alloca, .global_addr => true,
        .load, .store, .@"if", .call, .call_indirect => false,
    };
}

/// Append the value operands an instruction reads into `buf`.
fn collectOperands(
    allocator: std.mem.Allocator,
    func: *const Function,
    inst: Inst,
    buf: *std.ArrayList(Value),
) std.mem.Allocator.Error!void {
    buf.clearRetainingCapacity();
    switch (func.opcode(inst)) {
        .iconst, .fconst, .alloca, .global_addr => {},
        .arith => |a| {
            try buf.append(allocator, a.lhs);
            try buf.append(allocator, a.rhs);
        },
        .arith_imm => |a| try buf.append(allocator, a.lhs),
        .icmp => |c| {
            try buf.append(allocator, c.lhs);
            try buf.append(allocator, c.rhs);
        },
        .select => |s| {
            try buf.append(allocator, s.cond);
            try buf.append(allocator, s.then);
            try buf.append(allocator, s.@"else");
        },
        .extract => |e| try buf.append(allocator, e.aggregate),
        .convert => |cv| try buf.append(allocator, cv.value),
        .unary => |u| try buf.append(allocator, u.value),
        .struct_new => |sn| for (func.valueList(sn.fields)) |f| try buf.append(allocator, f),
        .call => |c| for (func.valueList(c.args)) |a| try buf.append(allocator, a),
        .call_indirect => |c| {
            try buf.append(allocator, c.target);
            for (func.valueList(c.args)) |a| try buf.append(allocator, a);
        },
        .load => |l| try buf.append(allocator, l.ptr),
        .store => |st| {
            try buf.append(allocator, st.value);
            try buf.append(allocator, st.ptr);
        },
        .@"if" => |cf| {
            try buf.append(allocator, cf.cond);
            for (func.valueList(cf.then.args)) |a| try buf.append(allocator, a);
            for (func.valueList(cf.@"else".args)) |a| try buf.append(allocator, a);
        },
    }
}

/// Schedule every block of `func` in place.
pub fn scheduleFunction(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!void {
    for (0..func.blockCount()) |bi| {
        try scheduleBlock(allocator, func, @enumFromInt(bi));
    }
}

/// List-schedule a single block, reordering its instructions to hide latency.
/// Only blocks whose instructions are all movable are touched for now.
fn scheduleBlock(allocator: std.mem.Allocator, func: *Function, block: Block) std.mem.Allocator.Error!void {
    const insts = func.blockInsts(block);
    const n = insts.len;
    if (n < 2) return;

    const none = std.math.maxInt(usize);

    // Pinned instructions (memory ops, `if`) act as barriers: a movable
    // instruction may not cross one, and pinned instructions keep their relative
    // order. This is enforced below as extra ordering constraints, so the regions
    // between barriers still get reordered to hide latency.
    const pinned = try allocator.alloc(bool, n);
    defer allocator.free(pinned);
    for (insts, 0..) |inst, i| pinned[i] = !movable(func.opcode(inst));

    // Map each value defined by an instruction in this block to its list index.
    // Values not found here (block params, values from other blocks) are inputs,
    // ready from cycle zero.
    const local_of = try allocator.alloc(usize, func.valueCount());
    defer allocator.free(local_of);
    @memset(local_of, none);
    for (insts, 0..) |inst, i| {
        if (func.instResult(inst)) |r| local_of[@intFromEnum(r)] = i;
    }

    const latency = try allocator.alloc(u32, n);
    defer allocator.free(latency);
    const scheduled = try allocator.alloc(bool, n);
    defer allocator.free(scheduled);
    const avail_at = try allocator.alloc(u32, n); // result-ready cycle once issued
    defer allocator.free(avail_at);
    for (insts, 0..) |inst, i| {
        latency[i] = riverLatency(func.opcode(inst));
        scheduled[i] = false;
    }

    var order: std.ArrayList(Inst) = .empty;
    defer order.deinit(allocator);
    var operands: std.ArrayList(Value) = .empty;
    defer operands.deinit(allocator);

    var cycle: u32 = 0;
    while (order.items.len < n) {
        var best: ?usize = null;
        var best_lat: u32 = 0;
        var soonest: ?u32 = null; // earliest ready cycle among stalled candidates

        for (0..n) |i| {
            if (scheduled[i]) continue;

            // Barrier ordering. A pinned instruction waits for everything before
            // it. Any instruction waits for every pinned instruction before it.
            // (Checking the nearest preceding barrier suffices, since a barrier
            // itself waits for all earlier ones.) This both pins barriers in
            // program order and confines movable instructions to their region.
            var barrier_ok = true;
            var j = i;
            while (j > 0) {
                j -= 1;
                if (pinned[i] or pinned[j]) {
                    if (!scheduled[j]) {
                        barrier_ok = false;
                        break;
                    }
                    if (pinned[j] and !pinned[i]) break; // nearest barrier satisfied
                }
            }
            if (!barrier_ok) continue;

            try collectOperands(allocator, func, insts[i], &operands);
            var deps_ready = true;
            var ready_cycle: u32 = 0;
            for (operands.items) |v| {
                const li = local_of[@intFromEnum(v)];
                if (li == none) continue; // external input: ready at cycle 0
                if (!scheduled[li]) {
                    deps_ready = false;
                    break;
                }
                ready_cycle = @max(ready_cycle, avail_at[li]);
            }
            if (!deps_ready) continue;
            if (ready_cycle <= cycle) {
                // Ready now: prefer the highest latency, breaking ties by program
                // order (the strict `>` keeps the earlier index on a tie).
                if (best == null or latency[i] > best_lat) {
                    best = i;
                    best_lat = latency[i];
                }
            } else {
                soonest = if (soonest) |s| @min(s, ready_cycle) else ready_cycle;
            }
        }

        if (best) |i| {
            scheduled[i] = true;
            avail_at[i] = cycle + latency[i];
            try order.append(allocator, insts[i]);
            cycle += 1; // single-issue: one instruction per cycle
        } else {
            // Nothing ready: skip ahead to when the soonest dependency lands.
            cycle = soonest orelse (cycle + 1);
        }
    }

    try func.setBlockInsts(block, order.items);
}

const expectEqualSlices = std.testing.expectEqualSlices;

test "a dependent chain keeps its order and stays verifiable" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const v0 = try func.appendBlockParam(block, i32_t);
    const v1 = try func.appendBlockParam(block, i32_t);
    // Each step needs the previous, so no reordering is possible.
    const v2 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v0, .rhs = v1 } });
    const v3 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v2, .rhs = v0 } });
    const v4 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v3, .rhs = v0 } });
    func.setTerminator(block, .{ .ret = v4 });

    try scheduleFunction(std.testing.allocator, &func);

    try expectEqualSlices(Inst, &.{
        func.definingInst(v2).?,
        func.definingInst(v3).?,
        func.definingInst(v4).?,
    }, func.blockInsts(block));

    // Every operand is still defined before its use.
    var diags = try @import("vulcan-ir").verify.verify(std.testing.allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "a later load never reorders ahead of an earlier store" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const block = try func.appendBlock();
    const p = try func.appendBlockParam(block, ptr_t);
    const x = try func.appendBlockParam(block, i32_t);
    try func.appendStore(block, x, p);
    const v = try func.appendInst(block, i32_t, .{ .load = .{ .ptr = p } }); // higher latency than the store
    _ = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    func.setTerminator(block, .{ .ret = v });

    const before = func.blockInsts(block);
    const i_store = before[0];
    const i_load = before[1];
    const i_add = before[2];

    try scheduleFunction(std.testing.allocator, &func);

    // Despite the load's higher latency, the store stays first (memory order) and
    // the independent add cannot hop ahead of either barrier.
    try expectEqualSlices(Inst, &.{ i_store, i_load, i_add }, func.blockInsts(block));
}

test "fills the load-use gap with independent work, keeping the load pinned" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const block = try func.appendBlock();
    const p = try func.appendBlockParam(block, ptr_t);
    const x = try func.appendBlockParam(block, i32_t);
    const v2 = try func.appendInst(block, i32_t, .{ .load = .{ .ptr = p } }); // latency 2
    const v3 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v2, .rhs = x } }); // uses the load
    const v4 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } }); // independent
    const v5 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v4, .rhs = x } });
    const v6 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v3, .rhs = v5 } });
    func.setTerminator(block, .{ .ret = v6 });

    try scheduleFunction(std.testing.allocator, &func);

    // The load stays first (memory order pinned). The independent `v4` is hoisted
    // into the load's two-cycle shadow so `v3` is not stalled waiting on it.
    try expectEqualSlices(Inst, &.{
        func.definingInst(v2).?,
        func.definingInst(v4).?,
        func.definingInst(v3).?,
        func.definingInst(v5).?,
        func.definingInst(v6).?,
    }, func.blockInsts(block));
}

test "schedules independent high-latency ops early to hide latency" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const v0 = try func.appendBlockParam(block, i32_t);
    const v1 = try func.appendBlockParam(block, i32_t);
    const v2 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .mul, .lhs = v0, .rhs = v1 } });
    const v3 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v2, .rhs = v0 } });
    const v4 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .mul, .lhs = v0, .rhs = v0 } });
    const v5 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v4, .rhs = v1 } });
    const v6 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v3, .rhs = v5 } });
    func.setTerminator(block, .{ .ret = v6 });

    try scheduleFunction(std.testing.allocator, &func);

    // Both multiplies (latency 3) get hoisted ahead of the adds that depend on
    // them, so their latencies overlap instead of stalling the pipeline.
    try expectEqualSlices(Inst, &.{
        func.definingInst(v2).?,
        func.definingInst(v4).?,
        func.definingInst(v3).?,
        func.definingInst(v5).?,
        func.definingInst(v6).?,
    }, func.blockInsts(block));
}
