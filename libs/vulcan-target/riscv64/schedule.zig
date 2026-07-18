//! Microarchitecture-aware list scheduling for the River CPU. Delegates to the shared model-driven
//! scheduler in vulcan-opt, giving it the River in-order model so its latency table (multiply 3,
//! divide 6, load 2) and single-issue cadence match this backend's pipeline exactly.

const std = @import("std");
const ir = @import("vulcan-ir");
const microarch = @import("vulcan-opt").microarch;

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;

/// Schedule every block of `func` for the River microarchitecture. Delegates to the shared
/// model-driven scheduler in vulcan-opt with a River in-order model, whose latency table (multiply
/// 3, divide 6, load 2) matches this backend's pipeline. The scheduler moved to vulcan-opt so every
/// target can share it, this shim keeps the River call sites and tests unchanged.
pub fn scheduleFunction(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!void {
    return microarch.schedule.run(allocator, func, microarch.modelFor(.@"river-rc1.s"));
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
