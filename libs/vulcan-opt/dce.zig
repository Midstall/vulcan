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
        .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .select, .struct_new, .extract, .convert, .unary, .alloca, .global_addr, .dot => true,
        // A prefetch hint has no result but must be kept, like a store. A
        // matmul writes the `c` memory, likewise kept.
        .load, .store, .prefetch, .matmul, .@"if", .call, .call_indirect => false,
        // SM12 T3: mutate/read the `va_list` object at `list`, like `load`/`store` above.
        .va_start, .va_arg, .va_end => false,
        // An atomic is a STORE as well as a load. Deleting one whose old value nobody
        // reads loses the write, so the reduction form is kept as firmly as the reading
        // form. This is the guard the optional result makes necessary.
        .atomic_rmw => false,
        // A barrier synchronizes threads and fences memory. It produces no result, so a
        // purity rule keyed on an unused result would delete every one of them.
        .barrier => false,
    };
}

/// Count uses of each value across live instructions, `if` edges, and terminators.
pub fn countUses(func: *const Function, uses: []u32) void {
    @memset(uses, 0);
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .atomic_rmw => |a| {
                    uses[@intFromEnum(a.ptr)] += 1;
                    uses[@intFromEnum(a.value)] += 1;
                    if (a.compare) |c| uses[@intFromEnum(c)] += 1;
                },
                // A barrier uses no Value, so it adds no use count.
                .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
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
                .prefetch => |pf| uses[@intFromEnum(pf.ptr)] += 1,
                .va_start => |vs| uses[@intFromEnum(vs.list)] += 1,
                .va_arg => |va| uses[@intFromEnum(va.list)] += 1,
                .va_end => |ve| uses[@intFromEnum(ve.list)] += 1,
                .dot => |d| {
                    uses[@intFromEnum(d.acc)] += 1;
                    uses[@intFromEnum(d.a)] += 1;
                    uses[@intFromEnum(d.b)] += 1;
                },
                .matmul => |mm| {
                    uses[@intFromEnum(mm.a)] += 1;
                    uses[@intFromEnum(mm.b)] += 1;
                    uses[@intFromEnum(mm.c)] += 1;
                },
                .struct_new => |sn| for (func.valueList(sn.fields)) |f| {
                    uses[@intFromEnum(f)] += 1;
                },
                .call => |c| {
                    for (func.valueList(c.args)) |arg| uses[@intFromEnum(arg)] += 1;
                    if (c.ret_dest) |rd| uses[@intFromEnum(rd)] += 1; // SM14 M4d-c T1: dest kept alive for the post-call store
                },
                .call_indirect => |c| {
                    uses[@intFromEnum(c.target)] += 1;
                    for (func.valueList(c.args)) |arg| uses[@intFromEnum(arg)] += 1;
                    if (c.ret_dest) |rd| uses[@intFromEnum(rd)] += 1; // SM14 M4d-c T1: dest kept alive for the post-call store
                },
                .@"if" => |cf| {
                    uses[@intFromEnum(cf.cond)] += 1;
                    for (func.blockArgs(cf.then)) |arg| uses[@intFromEnum(arg)] += 1;
                    for (func.blockArgs(cf.@"else")) |arg| uses[@intFromEnum(arg)] += 1;
                },
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .ret => |r| for (r.slice()) |vv| {
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
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });

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
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(b).len);
}

test "keeps a matmul even though it has no result to be used" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, ptr_t);
    const bp = try func.appendBlockParam(b, ptr_t);
    const c = try func.appendBlockParam(b, ptr_t);
    try func.appendMatmul(b, a, bp, c, 4, 4, 4, .fp32, false);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(b).len);
}

test "keeps a barrier even though it has no result to be used" {
    // The whole reason a barrier is an opcode and not a call: a purity rule keyed on an
    // unused result would delete it, and the deletion would be silent.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    try func.appendBarrier(b, .workgroup);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(b).len);
    try std.testing.expect(func.opcode(func.blockInsts(b)[0]) == .barrier);
}

test "keeps both forms of an atomic, read result or not" {
    // An atomic is a STORE as well as a load. The reduction form has no result to be used,
    // and the reading form's result is deliberately left unread here: a purity rule keyed
    // on an unused result would delete BOTH and lose two writes.
    //
    // The pure `arith` beside them is the control: it is also unused, and it IS deleted, so
    // the pass really ran on this block.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const x = try func.appendBlockParam(b, i32_t);
    // `used` feeds the atomic and nothing else, so `countUses` must count an atomic's
    // operands. If it does not, this pure multiply looks dead and its deletion leaves the
    // atomic naming a value nothing defines.
    const used = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = x } });
    try func.appendAtomicRmwStmt(b, .{ .op = .add, .ptr = p, .value = used, .ordering = .relaxed, .scope = .device });
    _ = try func.appendAtomicRmw(b, .{ .op = .bit_or, .ptr = p, .value = x, .ordering = .relaxed, .scope = .device });
    _ = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = x } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses)); // the dead subtract went

    const insts = func.blockInsts(b);
    try std.testing.expectEqual(@as(usize, 3), insts.len);
    try std.testing.expect(func.opcode(insts[0]) == .arith); // the multiply the atomic uses
    try std.testing.expectEqual(used, func.instResult(insts[0]).?);
    try std.testing.expect(func.opcode(insts[1]) == .atomic_rmw);
    try std.testing.expect(func.opcode(insts[2]) == .atomic_rmw);
}
