//! Dead-code elimination: remove pure instructions whose result is never used,
//! iterating to a fixpoint (removing one dead value can make its operands dead).
//! Impure instructions (loads, stores, calls, `if`) are always kept.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;

pub const pass_def = pass.Pass{ .name = "dce", .run = run };

/// Whether an instruction has no side effects, so it may be dropped when unused.
fn isPure(op: ir.function.Opcode) bool {
    return switch (op) {
        .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .struct_new, .extract, .convert, .unary, .alloca, .global_addr => true,
        .load, .store, .@"if", .call, .call_indirect => false,
    };
}

/// Count uses of each value across live instructions, `if` edges, and terminators.
fn countUses(func: *const Function, uses: []u32) void {
    @memset(uses, 0);
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .iconst, .fconst, .alloca, .global_addr => {},
                .arith => |a| {
                    uses[@intFromEnum(a.lhs)] += 1;
                    uses[@intFromEnum(a.rhs)] += 1;
                },
                .arith_imm => |a| uses[@intFromEnum(a.lhs)] += 1,
                .icmp => |c| {
                    uses[@intFromEnum(c.lhs)] += 1;
                    uses[@intFromEnum(c.rhs)] += 1;
                },
                .select => |s| {
                    uses[@intFromEnum(s.cond)] += 1;
                    uses[@intFromEnum(s.then)] += 1;
                    uses[@intFromEnum(s.@"else")] += 1;
                },
                .extract => |e| uses[@intFromEnum(e.aggregate)] += 1,
                .convert => |cv| uses[@intFromEnum(cv.value)] += 1,
                .unary => |u| uses[@intFromEnum(u.value)] += 1,
                .load => |l| uses[@intFromEnum(l.ptr)] += 1,
                .store => |st| {
                    uses[@intFromEnum(st.value)] += 1;
                    uses[@intFromEnum(st.ptr)] += 1;
                },
                .struct_new => |sn| for (func.valueList(sn.fields)) |f| {
                    uses[@intFromEnum(f)] += 1;
                },
                .call => |c| for (func.valueList(c.args)) |arg| {
                    uses[@intFromEnum(arg)] += 1;
                },
                .call_indirect => |c| {
                    uses[@intFromEnum(c.target)] += 1;
                    for (func.valueList(c.args)) |arg| uses[@intFromEnum(arg)] += 1;
                },
                .@"if" => |cf| {
                    uses[@intFromEnum(cf.cond)] += 1;
                    for (func.blockArgs(cf.then)) |arg| uses[@intFromEnum(arg)] += 1;
                    for (func.blockArgs(cf.@"else")) |arg| uses[@intFromEnum(arg)] += 1;
                },
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .ret => |v| if (v) |vv| {
                uses[@intFromEnum(vv)] += 1;
            },
            .jump => |j| for (func.blockArgs(j)) |arg| {
                uses[@intFromEnum(arg)] += 1;
            },
        };
    }
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);

    var changed = false;
    while (true) {
        countUses(func, uses);
        var removed = false;
        for (0..func.blockCount()) |bi| {
            const insts = func.blockInstsMut(@enumFromInt(bi));
            var w: usize = 0;
            for (insts.items) |inst| {
                const dead = isPure(func.opcode(inst)) and
                    if (func.instResult(inst)) |r| uses[@intFromEnum(r)] == 0 else false;
                if (dead) {
                    removed = true;
                    continue;
                }
                insts.items[w] = inst;
                w += 1;
            }
            insts.shrinkRetainingCapacity(w);
        }
        if (!removed) break;
        changed = true;
    }
    return changed;
}

test "removes a chain of dead pure instructions" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    // dead1 = x + x, dead2 = dead1 * x, (neither used), ret x
    const dead1 = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    _ = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .mul, .lhs = dead1, .rhs = x } });
    func.setTerminator(b, .{ .ret = x });

    try std.testing.expectEqual(@as(usize, 2), func.blockInsts(b).len);

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));

    // Both dead instructions are gone.
    try std.testing.expectEqual(@as(usize, 0), func.blockInsts(b).len);
}

test "keeps an impure call even if its result is unused" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    _ = try func.appendCall(b, i32_t, "sink", &.{x}); // result unused, but a call has effects
    func.setTerminator(b, .{ .ret = x });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(b).len);
}
