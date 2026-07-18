//! Global value numbering / common-subexpression elimination. Pure instructions
//! computing the same value (same opcode over congruent operands) are numbered
//! together. A later occurrence is replaced by an earlier one whose definition
//! dominates it, so the value is always available. Uses are rewritten to the
//! leader, leaving the redundant instructions dead for DCE.
//!
//! Numbering walks blocks in reverse postorder so an operand is numbered before
//! its uses.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const cfg_mod = @import("cfg.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;

pub const pass_def = pass.Pass{ .name = "gvn", .run = run };

const ExprKind = enum(u8) { iconst, fconst, arith, arith_imm, icmp, select, convert, unary, extract, global_addr, dot };

/// A canonical key for a pure expression: its kind, a sub-opcode (BinOp/CmpOp,
/// result type, or field index), and up to three operand value-numbers/literals.
const Key = struct {
    kind: ExprKind,
    sub: u32 = 0,
    a: u64 = 0,
    b: u64 = 0,
    c: u64 = 0,
};

fn isCommutative(op: ir.function.BinOp) bool {
    return switch (op) {
        .add, .mul, .mulh, .bit_and, .bit_or, .bit_xor => true,
        .sub, .div, .rem, .shl, .shr => false,
    };
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const n = func.blockCount();
    if (n == 0) return false;

    const doms = try analyses.dominators();
    var cfg = try cfg_mod.build(allocator, func);
    defer cfg.deinit(allocator);
    const rpo = try cfg.reversePostorder(allocator);
    defer allocator.free(rpo);

    // The canonical (leader) value for each value, and the block each is defined
    // in. Both start as the identity.
    const canon = try allocator.alloc(Value, func.valueCount());
    defer allocator.free(canon);
    for (canon, 0..) |*c, i| c.* = @enumFromInt(i);
    const def_block = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(def_block);
    for (0..n) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| def_block[@intFromEnum(p)] = @intCast(bi);
        for (func.blockInsts(block)) |inst| {
            if (func.instResult(inst)) |r| def_block[@intFromEnum(r)] = @intCast(bi);
        }
    }

    var table: std.AutoHashMapUnmanaged(Key, Value) = .empty;
    defer table.deinit(allocator);

    var changed = false;
    for (rpo) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            const result = func.instResult(inst) orelse continue;
            const key = keyOf(func, canon, inst, result) orelse continue; // not numberable
            if (table.get(key)) |leader| {
                if (doms.dominates(def_block[@intFromEnum(leader)], bi)) {
                    canon[@intFromEnum(result)] = leader; // redundant: reuse the leader
                    changed = true;
                    continue;
                }
            }
            // First occurrence here, or the prior one does not dominate: this
            // instruction becomes the leader for blocks it dominates.
            try table.put(allocator, key, result);
        }
    }

    if (changed) rewriteOperands(func, canon);
    return changed;
}

/// Build the canonical key for `inst` (whose result is `result`), or null if the
/// instruction is impure or otherwise not subject to value numbering. `alloca`
/// is excluded: each yields a distinct address.
fn keyOf(func: *const Function, canon: []const Value, inst: Inst, result: Value) ?Key {
    const vn = struct {
        fn of(c: []const Value, v: Value) u64 {
            return @intFromEnum(c[@intFromEnum(v)]);
        }
    }.of;
    return switch (func.opcode(inst)) {
        .iconst => |v| .{ .kind = .iconst, .a = @bitCast(v) },
        .fconst => |v| .{ .kind = .fconst, .a = @bitCast(v) },
        .arith => |x| blk: {
            var a = vn(canon, x.lhs);
            var b = vn(canon, x.rhs);
            if (isCommutative(x.op) and a > b) std.mem.swap(u64, &a, &b);
            break :blk .{ .kind = .arith, .sub = @intFromEnum(x.op), .a = a, .b = b };
        },
        .arith_imm => |x| .{ .kind = .arith_imm, .sub = @intFromEnum(x.op), .a = vn(canon, x.lhs), .b = @bitCast(x.imm) },
        .icmp => |x| blk: {
            var a = vn(canon, x.lhs);
            var b = vn(canon, x.rhs);
            if ((x.op == .eq or x.op == .ne) and a > b) std.mem.swap(u64, &a, &b);
            break :blk .{ .kind = .icmp, .sub = @intFromEnum(x.op), .a = a, .b = b };
        },
        .select => |x| .{ .kind = .select, .a = vn(canon, x.cond), .b = vn(canon, x.then), .c = vn(canon, x.@"else") },
        .convert => |x| .{ .kind = .convert, .sub = @intFromEnum(func.valueType(result)), .a = vn(canon, x.value) },
        .unary => |x| .{ .kind = .unary, .sub = @intFromEnum(func.valueType(result)), .a = vn(canon, x.value), .b = @intFromEnum(x.op) },
        .extract => |x| .{ .kind = .extract, .sub = x.index, .a = vn(canon, x.aggregate) },
        .global_addr => |x| .{ .kind = .global_addr, .a = x.symbol },
        // dot is pure, like arith, and keyed on all three operands (not commutative:
        // acc is the accumulator, distinct from a/b).
        .dot => |x| .{ .kind = .dot, .a = vn(canon, x.acc), .b = vn(canon, x.a), .c = vn(canon, x.b) },
        // alloca (distinct addresses), struct_new (variadic), and the impure
        // load/store/prefetch/matmul/call/if are not numbered.
        .alloca, .struct_new, .load, .store, .prefetch, .matmul, .call, .call_indirect, .@"if" => null,
    };
}

fn sub(canon: []const Value, v: Value) Value {
    return canon[@intFromEnum(v)];
}

/// Rewrite every value operand (in instructions, `if` edges, and terminators) to
/// its canonical leader, so redundant definitions fall out of use.
fn rewriteOperands(func: *Function, canon: []const Value) void {
    for (0..func.instCount()) |i| {
        const op = func.opcodeMut(@enumFromInt(i));
        switch (op.*) {
            .iconst, .fconst, .alloca, .global_addr => {},
            .arith => |*a| {
                a.lhs = sub(canon, a.lhs);
                a.rhs = sub(canon, a.rhs);
            },
            .arith_imm => |*a| a.lhs = sub(canon, a.lhs),
            .icmp => |*c| {
                c.lhs = sub(canon, c.lhs);
                c.rhs = sub(canon, c.rhs);
            },
            .select => |*s| {
                s.cond = sub(canon, s.cond);
                s.then = sub(canon, s.then);
                s.@"else" = sub(canon, s.@"else");
            },
            .extract => |*e| e.aggregate = sub(canon, e.aggregate),
            .convert => |*cv| cv.value = sub(canon, cv.value),
            .unary => |*u| u.value = sub(canon, u.value),
            .load => |*l| l.ptr = sub(canon, l.ptr),
            .store => |*st| {
                st.value = sub(canon, st.value);
                st.ptr = sub(canon, st.ptr);
            },
            .prefetch => |*pf| pf.ptr = sub(canon, pf.ptr),
            .dot => |*d| {
                d.acc = sub(canon, d.acc);
                d.a = sub(canon, d.a);
                d.b = sub(canon, d.b);
            },
            .matmul => |*mm| {
                mm.a = sub(canon, mm.a);
                mm.b = sub(canon, mm.b);
                mm.c = sub(canon, mm.c);
            },
            .struct_new => |sn| for (func.valueListMut(sn.fields)) |*f| {
                f.* = sub(canon, f.*);
            },
            .call => |c| for (func.valueListMut(c.args)) |*arg| {
                arg.* = sub(canon, arg.*);
            },
            .call_indirect => |*c| {
                c.target = sub(canon, c.target);
                for (func.valueListMut(c.args)) |*arg| arg.* = sub(canon, arg.*);
            },
            .@"if" => |*cf| {
                cf.cond = sub(canon, cf.cond);
                for (func.valueListMut(cf.then.args)) |*arg| arg.* = sub(canon, arg.*);
                for (func.valueListMut(cf.@"else".args)) |*arg| arg.* = sub(canon, arg.*);
            },
        }
    }
    for (0..func.blockCount()) |bi| {
        const term = func.terminatorPtr(@enumFromInt(bi));
        if (term.*) |*t| switch (t.*) {
            .ret => |*v| if (v.*) |vv| {
                v.* = sub(canon, vv);
            },
            .jump => |*j| for (func.valueListMut(j.args)) |*arg| {
                arg.* = sub(canon, arg.*);
            },
        };
    }
}

test "cse reuses a redundant arithmetic expression" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const y = try func.appendBlockParam(b, i32_t);
    // e1 = x + y, e2 = y + x (congruent, commutative), r = e1 + e2, ret r
    const e1 = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    const e2 = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = x } });
    const r = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = e1, .rhs = e2 } });
    func.setTerminator(b, .{ .ret = r });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));

    // e2 was recognized as congruent to e1. r now adds e1 to itself.
    const rdef = func.definingInst(r).?;
    try std.testing.expectEqual(e1, func.opcode(rdef).arith.lhs);
    try std.testing.expectEqual(e1, func.opcode(rdef).arith.rhs);
}

test "cse does not reuse across a non-dominating block" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const cnd = try func.appendBlockParam(b0, bool_t);
    const x = try func.appendBlockParam(b0, i32_t);
    const b1 = try func.appendBlock();
    const b2 = try func.appendBlock();
    // b0: if cnd -> b1 else b2, b1: e1 = x+x -> ret e1, b2: e2 = x+x -> ret e2
    try func.appendIf(b0, cnd, .{ .target = b1 }, .{ .target = b2 });
    const e1 = try func.appendInst(b1, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    func.setTerminator(b1, .{ .ret = e1 });
    const e2 = try func.appendInst(b2, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    func.setTerminator(b2, .{ .ret = e2 });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    // b1 does not dominate b2, so e2 cannot reuse e1: nothing changes.
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expectEqual(e2, func.terminator(b2).?.ret.?);
}
