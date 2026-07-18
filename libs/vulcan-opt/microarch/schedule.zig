//! Microarchitecture-aware list scheduling, shared and model-driven. Reorders each block's
//! independent instructions to hide functional-unit latency, driven by a Model's per-opcode latency
//! and functional-unit data. Memory ops and the `if` control statement are barriers, movable pure
//! ops reorder within the regions between them. A single-issue model keeps one instruction per
//! cycle, a wider model co-issues up to issue_width per cycle bounded by the per-class port counts,
//! so the pass exploits a superscalar core's width. Generalized from the original River-only
//! scheduler (riscv64/schedule.zig), which now delegates here. An in-order model gets the full
//! latency-driven schedule over the whole block, since the static schedule is the only thing hiding
//! functional-unit latency there. An out-of-order model bounds the reorder window to about
//! `rob_size` and, within that window, keeps close to program order instead of aggressively
//! hoisting by latency: the hardware itself reorders and hides latency, so a wide static hoist over
//! the whole block buys little and only inflates register pressure. This is a heuristic refinement,
//! not a correctness concern, dependencies and barriers are enforced identically either way.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Model = mm.Model;
const UnitClass = mm.UnitClass;

/// A pure value op may be freely reordered within its block. Memory ops, `if`, and calls are pinned
/// (reordering across them needs barriers). This is IR structure, not microarch, so it is not in the
/// Model.
fn movable(op: ir.function.Opcode) bool {
    return switch (op) {
        .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .struct_new, .extract, .convert, .unary, .alloca, .global_addr, .dot => true,
        // matmul writes the `c` memory: a barrier, like store/prefetch, not reordered.
        .load, .store, .prefetch, .matmul, .@"if", .call, .call_indirect => false,
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
        .prefetch => |pf| try buf.append(allocator, pf.ptr),
        .dot => |d| {
            try buf.append(allocator, d.acc);
            try buf.append(allocator, d.a);
            try buf.append(allocator, d.b);
        },
        .matmul => |mmv| {
            try buf.append(allocator, mmv.a);
            try buf.append(allocator, mmv.b);
            try buf.append(allocator, mmv.c);
        },
        .@"if" => |cf| {
            try buf.append(allocator, cf.cond);
            for (func.valueList(cf.then.args)) |a| try buf.append(allocator, a);
            for (func.valueList(cf.@"else".args)) |a| try buf.append(allocator, a);
        },
    }
}

/// The per-cycle port capacity for a unit class. A class with a 0 count (or `.none`) is treated as
/// unmodeled and does not constrain issue, so a model that omits a class never deadlocks the loop.
/// Public so the profitability cost model (cost.zig) reads the same port data the scheduler does.
pub fn classPorts(model: *const Model, class: UnitClass) u32 {
    const n: u32 = switch (class) {
        .alu => model.units.alu,
        .muldiv => model.units.muldiv,
        .mem => model.units.mem,
        .branch => model.units.branch,
        .fpsimd => model.units.fpsimd,
        .none => 0,
    };
    return if (n == 0) std.math.maxInt(u32) else n;
}

/// Schedule every block of `func` in place for `model`.
pub fn run(allocator: std.mem.Allocator, func: *Function, model: *const Model) std.mem.Allocator.Error!void {
    for (0..func.blockCount()) |bi| {
        try scheduleBlock(allocator, func, @enumFromInt(bi), model);
    }
}

fn scheduleBlock(allocator: std.mem.Allocator, func: *Function, block: Block, model: *const Model) std.mem.Allocator.Error!void {
    const insts = func.blockInsts(block);
    const n = insts.len;
    if (n < 2) return;

    const none = std.math.maxInt(usize);

    const pinned = try allocator.alloc(bool, n);
    defer allocator.free(pinned);
    for (insts, 0..) |inst, i| pinned[i] = !movable(func.opcode(inst));

    const local_of = try allocator.alloc(usize, func.valueCount());
    defer allocator.free(local_of);
    @memset(local_of, none);
    for (insts, 0..) |inst, i| {
        if (func.instResult(inst)) |r| local_of[@intFromEnum(r)] = i;
    }

    const latency = try allocator.alloc(u32, n);
    defer allocator.free(latency);
    const class = try allocator.alloc(UnitClass, n);
    defer allocator.free(class);
    const scheduled = try allocator.alloc(bool, n);
    defer allocator.free(scheduled);
    const avail_at = try allocator.alloc(u32, n);
    defer allocator.free(avail_at);
    for (insts, 0..) |inst, i| {
        latency[i] = model.latency(func.opcode(inst));
        class[i] = model.unitOf(func.opcode(inst));
        scheduled[i] = false;
    }

    var order: std.ArrayList(Inst) = .empty;
    defer order.deinit(allocator);
    var operands: std.ArrayList(Value) = .empty;
    defer operands.deinit(allocator);

    // The reorder window: how many positions past the earliest unscheduled instruction the search
    // may look. An in-order model (rob_size 0) keeps window == n, today's full-block schedule. An
    // out-of-order model bounds it to about its reorder-buffer size, since hardware past that
    // distance cannot actually see the instruction yet. Always >= 1, so the earliest unscheduled
    // instruction itself is always in-window (see the deadlock argument below).
    const window: usize = if (model.reorders() and model.rob_size > 0) @min(n, model.rob_size) else n;

    const width: u32 = @max(1, model.issue_width);
    var cycle: u32 = 0;
    while (order.items.len < n) {
        var issued_this_cycle: u32 = 0;
        var used = [_]u32{0} ** @typeInfo(UnitClass).@"enum".fields.len;
        var soonest: ?u32 = null;

        while (issued_this_cycle < width) {
            var best: ?usize = null;
            var best_lat: u32 = 0;
            soonest = null;

            // The earliest unscheduled instruction. Everything before it is scheduled, so its
            // barrier and operand constraints already hold, only its latency (`ready_cycle`) can
            // defer it, which the soonest/cycle-advance logic below handles. It is always within
            // the window (lo < lo + window for window >= 1), so this loop always makes progress:
            // no deadlock.
            var lo: usize = 0;
            while (lo < n and scheduled[lo]) lo += 1;

            for (0..n) |i| {
                if (i >= lo + window) break;
                if (scheduled[i]) continue;

                // Barrier ordering: a pinned instruction waits for everything before it, and any
                // instruction waits for every pinned instruction before it. Checking the nearest
                // preceding barrier suffices, since a barrier itself waits for all earlier ones.
                var barrier_ok = true;
                var j = i;
                while (j > 0) {
                    j -= 1;
                    if (pinned[i] or pinned[j]) {
                        if (!scheduled[j]) {
                            barrier_ok = false;
                            break;
                        }
                        if (pinned[j] and !pinned[i]) break;
                    }
                }
                if (!barrier_ok) continue;

                try collectOperands(allocator, func, insts[i], &operands);
                var deps_ready = true;
                var ready_cycle: u32 = 0;
                for (operands.items) |v| {
                    const li = local_of[@intFromEnum(v)];
                    if (li == none) continue;
                    if (!scheduled[li]) {
                        deps_ready = false;
                        break;
                    }
                    ready_cycle = @max(ready_cycle, avail_at[li]);
                }
                if (!deps_ready) continue;

                if (ready_cycle <= cycle) {
                    const cap = classPorts(model, class[i]);
                    if (cap != std.math.maxInt(u32) and used[@intFromEnum(class[i])] >= cap) continue;
                    if (best == null) {
                        best = i;
                        best_lat = latency[i];
                    } else if (!model.reorders() and latency[i] > best_lat) {
                        // In-order: aggressively hoist the highest-latency ready op to hide its
                        // latency. An OoO core reorders in hardware, so it keeps the earliest ready
                        // op (lower register pressure, near program order), which the `best == null`
                        // branch above already selected, since the scan visits `i` in increasing
                        // order.
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
                used[@intFromEnum(class[i])] += 1;
                issued_this_cycle += 1;
            } else break;
        }

        if (issued_this_cycle == 0) {
            cycle = soonest orelse (cycle + 1);
        } else {
            cycle += 1;
        }
    }

    try func.setBlockInsts(block, order.items);
}

const registry = @import("registry.zig");
const expectEqualSlices = std.testing.expectEqualSlices;

test "issues independent high-latency ops early to hide latency (single-issue river model)" {
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

    try run(std.testing.allocator, &func, registry.modelFor(.@"river-rc1.s"));

    // Both multiplies get hoisted ahead of the adds that depend on them.
    try expectEqualSlices(Inst, &.{
        func.definingInst(v2).?, func.definingInst(v4).?, func.definingInst(v3).?,
        func.definingInst(v5).?, func.definingInst(v6).?,
    }, func.blockInsts(block));
}

test "a dependent chain stays in order and verifies for a wide out-of-order model" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const v0 = try func.appendBlockParam(block, i32_t);
    const v1 = try func.appendBlockParam(block, i32_t);
    const v2 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v0, .rhs = v1 } });
    const v3 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v2, .rhs = v0 } });
    const v4 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v3, .rhs = v0 } });
    func.setTerminator(block, .{ .ret = v4 });

    try run(std.testing.allocator, &func, registry.modelFor(.@"ampere-altra"));

    try expectEqualSlices(Inst, &.{
        func.definingInst(v2).?, func.definingInst(v3).?, func.definingInst(v4).?,
    }, func.blockInsts(block));
    var diags = try ir.verify.verify(std.testing.allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "priority follows the model latency: the higher-latency independent op schedules first" {
    // Two independent movable ops (a multiply and an add) feeding one use. The latency-driven ready
    // priority issues the higher-latency op first. Proven with the ET-SOC model, whose multiply
    // latency (8) far outweighs the add (1), so the multiply leads.
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, i32_t);
    const b = try func.appendBlockParam(block, i32_t);
    const mul = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const add = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const use = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = mul, .rhs = add } });
    func.setTerminator(block, .{ .ret = use });

    try run(std.testing.allocator, &func, registry.modelFor(.@"et-soc"));
    try expectEqualSlices(Inst, &.{
        func.definingInst(mul).?, func.definingInst(add).?, func.definingInst(use).?,
    }, func.blockInsts(block));
}

/// Builds an add (program-order first) and a mul (program-order second), both independent and both
/// feeding one use. Shared by the in-order-vs-out-of-order priority test below.
fn buildAddThenMul(func: *Function) !struct { block: Block, add: Value, mul: Value, use: Value } {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, i32_t);
    const b = try func.appendBlockParam(block, i32_t);
    const add = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const mul = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const use = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = add, .rhs = mul } });
    func.setTerminator(block, .{ .ret = use });
    return .{ .block = block, .add = add, .mul = mul, .use = use };
}

test "out-of-order keeps program order while in-order hoists the higher-latency op" {
    // Same two independent ops, add first and mul second in program order, both feeding one use.
    // In-order (et-soc, mul latency 8 >> add latency 1) aggressively hoists the mul: mul, add, use.
    // Out-of-order (ampere-altra) prefers the earliest-ready op regardless of latency, so program
    // order survives: add, mul, use. This is the lightened OoO reordering weight from the module
    // doc comment, directly contrasted against the in-order aggressive hoist.
    var func_io = Function.init(std.testing.allocator);
    defer func_io.deinit();
    const io = try buildAddThenMul(&func_io);
    try run(std.testing.allocator, &func_io, registry.modelFor(.@"et-soc"));
    try expectEqualSlices(Inst, &.{
        func_io.definingInst(io.mul).?, func_io.definingInst(io.add).?, func_io.definingInst(io.use).?,
    }, func_io.blockInsts(io.block));

    var func_ooo = Function.init(std.testing.allocator);
    defer func_ooo.deinit();
    const ooo = try buildAddThenMul(&func_ooo);
    try run(std.testing.allocator, &func_ooo, registry.modelFor(.@"ampere-altra"));
    try expectEqualSlices(Inst, &.{
        func_ooo.definingInst(ooo.add).?, func_ooo.definingInst(ooo.mul).?, func_ooo.definingInst(ooo.use).?,
    }, func_ooo.blockInsts(ooo.block));
}

// A tiny out-of-order model with a fixed unit class and latency table, used only to isolate the
// reorder window's effect (a small vs. a large rob_size) from the priority rule, which is already
// covered above. Both models share the same latency/unitOf functions and out-of-order priority, so
// the only variable is rob_size.
fn windowTestLatency(op: ir.function.Opcode) u32 {
    return switch (op) {
        .arith => |a| switch (a.op) {
            .mul => 5,
            .div, .rem, .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => 1,
        },
        .arith_imm, .iconst, .fconst, .icmp, .select, .struct_new, .extract, .convert, .unary, .alloca, .global_addr, .load, .store, .prefetch, .dot, .matmul, .@"if", .call, .call_indirect => 1,
    };
}
fn windowTestUnit(op: ir.function.Opcode) UnitClass {
    _ = op;
    return .alu;
}
// This model isolates the reorder-window effect, not throughput, so a pipelined (1) throughput for
// every op is fine and satisfies throughput <= latency (mul latency 5, everything else 1).
fn windowTestThroughput(op: ir.function.Opcode, elem_float: bool) u32 {
    _ = op;
    _ = elem_float;
    return 1;
}
const window_small_rob = Model{
    .tag = .@"ampere-altra",
    .arch = .aarch64,
    .exec = .out_of_order,
    .issue_width = 1,
    .rob_size = 2,
    .units = .{},
    .vector_bits = 0,
    .cache_line = 64,
    .fetch_align = 0,
    .features = .{ .aarch64 = .{} },
    .latency = windowTestLatency,
    .throughput = windowTestThroughput,
    .unitOf = windowTestUnit,
    .fusion = &.{},
};
const window_large_rob = Model{
    .tag = .@"ampere-altra",
    .arch = .aarch64,
    .exec = .out_of_order,
    .issue_width = 1,
    .rob_size = 100,
    .units = .{},
    .vector_bits = 0,
    .cache_line = 64,
    .fetch_align = 0,
    .features = .{ .aarch64 = .{} },
    .latency = windowTestLatency,
    .throughput = windowTestThroughput,
    .unitOf = windowTestUnit,
    .fusion = &.{},
};

/// v0..v4 is a dependent chain (v2 stalls it with a high latency); `far` is independent of the
/// chain, at index 5, feeding `use` alongside the chain's tail.
fn buildWindowBlock(func: *Function) !struct {
    block: Block,
    v4: Value,
    far: Value,
    use: Value,
} {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, i32_t);
    const b = try func.appendBlockParam(block, i32_t);
    const v0 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } });
    const v1 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v0, .rhs = a } });
    const v2 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .mul, .lhs = v1, .rhs = a } });
    const v3 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v2, .rhs = a } });
    const v4 = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v3, .rhs = a } });
    const far = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = b, .rhs = b } });
    const use = try func.appendInst(block, i32_t, .{ .arith = .{ .op = .add, .lhs = v4, .rhs = far } });
    func.setTerminator(block, .{ .ret = use });
    return .{ .block = block, .v4 = v4, .far = far, .use = use };
}

test "a small reorder window bounds how far an independent op can move; a large window does not" {
    // Both models are out-of-order with the identical earliest-ready priority, only rob_size (hence
    // the window) differs, so any difference in where `far` lands is the window's doing, not the
    // priority rule's (that is covered by the test above). The dependent chain's stall (v2's high
    // latency) opens an idle issue slot; a large window lets the scheduler fill it with the
    // independent `far`, a small window excludes `far` from the search until the frontier reaches it.
    var func_small = Function.init(std.testing.allocator);
    defer func_small.deinit();
    const small = try buildWindowBlock(&func_small);
    try run(std.testing.allocator, &func_small, &window_small_rob);
    const order_small = func_small.blockInsts(small.block);
    const far_at_small = std.mem.indexOfScalar(Inst, order_small, func_small.definingInst(small.far).?).?;

    var func_large = Function.init(std.testing.allocator);
    defer func_large.deinit();
    const large = try buildWindowBlock(&func_large);
    try run(std.testing.allocator, &func_large, &window_large_rob);
    const order_large = func_large.blockInsts(large.block);
    const far_at_large = std.mem.indexOfScalar(Inst, order_large, func_large.definingInst(large.far).?).?;

    // The large window pulls `far` strictly earlier than the small window does.
    try std.testing.expect(far_at_large < far_at_small);
    // The small window keeps `far` at or past its own program-order index (5): bounded motion.
    try std.testing.expect(far_at_small >= 5);
}
