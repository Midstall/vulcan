//! Lowers high-profile IR toward the low profile. The main transform is scalar
//! replacement of aggregates: an `extract` of a `struct` construction forwards
//! to the field value, then the dead aggregate instructions are dropped. A
//! function that only used composites this way then passes the low profile.

const std = @import("std");
const function = @import("function.zig");
const types = @import("types.zig");

const Function = function.Function;
const Value = function.Value;
const ValueList = function.ValueList;

const Block = function.Block;
const Subst = std.AutoHashMapUnmanaged(Value, Value);
/// Maps a struct parameter that was split to the scalar field parameters that
/// replaced it.
const ParamFields = std.AutoHashMapUnmanaged(Value, []Value);
/// Maps a block whose parameters were split to its original parameter list, so
/// predecessor edges can be expanded to match.
const OldParams = std.AutoHashMapUnmanaged(Block, []Value);

/// Run legalization in place over `func`.
pub fn legalize(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!void {
    var param_fields: ParamFields = .empty;
    defer {
        var it = param_fields.valueIterator();
        while (it.next()) |slice| allocator.free(slice.*);
        param_fields.deinit(allocator);
    }
    var old_params: OldParams = .empty;
    defer {
        var it = old_params.valueIterator();
        while (it.next()) |slice| allocator.free(slice.*);
        old_params.deinit(allocator);
    }

    try splitStructParams(allocator, func, &param_fields, &old_params);
    try rewriteJumpEdges(allocator, func, &param_fields, &old_params);
    try rewriteIfEdges(allocator, func, &param_fields, &old_params);

    var subst: Subst = .empty;
    defer subst.deinit(allocator);

    try buildSubst(allocator, func, &subst, &param_fields);
    applySubst(func, &subst);
    foldConstants(func);
    foldImmediates(func);
    try dropDeadInstructions(allocator, func);
}

/// Split every struct-typed block parameter into its scalar field parameters,
/// recording the field mapping and the original parameter list per block.
fn splitStructParams(allocator: std.mem.Allocator, func: *Function, param_fields: *ParamFields, old_params: *OldParams) std.mem.Allocator.Error!void {
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        const old = try allocator.dupe(Value, func.blockParams(block));

        var new_params: std.ArrayList(Value) = .empty;
        defer new_params.deinit(allocator);

        var changed = false;
        for (old) |p| {
            switch (func.types.type_kind(func.valueType(p))) {
                .@"struct" => |flds| {
                    changed = true;
                    var fps: std.ArrayList(Value) = .empty;
                    for (flds) |ft| {
                        const fp = try func.newParam(block, ft);
                        try fps.append(allocator, fp);
                        try new_params.append(allocator, fp);
                    }
                    try param_fields.put(allocator, p, try fps.toOwnedSlice(allocator));
                },
                else => try new_params.append(allocator, p),
            }
        }

        if (changed) {
            try func.setBlockParams(block, new_params.items);
            try old_params.put(allocator, block, old);
        } else {
            allocator.free(old);
        }
    }
}

/// Rewrite jump edges that target a split block so they pass the scalar fields
/// (extracted from the struct argument in the source block) instead of the whole
/// struct. Edges from `if` instructions are not handled here yet.
fn rewriteJumpEdges(allocator: std.mem.Allocator, func: *Function, param_fields: *const ParamFields, old_params: *const OldParams) std.mem.Allocator.Error!void {
    for (0..func.blockCount()) |bi| {
        const src: Block = @enumFromInt(bi);
        const jump = switch (func.terminator(src) orelse continue) {
            .jump => |j| j,
            .ret => continue,
        };
        const olds = old_params.get(jump.target) orelse continue;

        var new_args: std.ArrayList(Value) = .empty;
        defer new_args.deinit(allocator);

        const args = func.blockArgs(jump);
        for (olds, args) |old_param, arg| {
            if (param_fields.get(old_param)) |flds| {
                for (flds, 0..) |fp, i| {
                    const e = try func.appendInst(src, func.valueType(fp), .{ .extract = .{ .aggregate = arg, .index = @intCast(i) } });
                    try new_args.append(allocator, e);
                }
            } else {
                try new_args.append(allocator, arg);
            }
        }

        try func.setJump(src, jump.target, new_args.items);
    }
}

/// Rewrite `if`-instruction edges that target a split block. The field extracts
/// are spliced into the block immediately before the `if`, since the `if` is not
/// the block's terminator.
fn rewriteIfEdges(allocator: std.mem.Allocator, func: *Function, param_fields: *const ParamFields, old_params: *const OldParams) std.mem.Allocator.Error!void {
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);

        const insts = try allocator.dupe(function.Inst, func.blockInsts(block));
        defer allocator.free(insts);

        var new_list: std.ArrayList(function.Inst) = .empty;
        defer new_list.deinit(allocator);

        var changed = false;
        for (insts) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => |cf| {
                    var then_args: std.ArrayList(Value) = .empty;
                    defer then_args.deinit(allocator);
                    var else_args: std.ArrayList(Value) = .empty;
                    defer else_args.deinit(allocator);
                    var extracts: std.ArrayList(function.Inst) = .empty;
                    defer extracts.deinit(allocator);

                    const then_split = try expandEdge(allocator, func, cf.then, param_fields, old_params, &extracts, &then_args);
                    const else_split = try expandEdge(allocator, func, cf.@"else", param_fields, old_params, &extracts, &else_args);

                    if (then_split or else_split) {
                        changed = true;
                        try new_list.appendSlice(allocator, extracts.items);
                        const op = func.opcodeMut(inst);
                        if (then_split) op.@"if".then.args = try func.internValueList(then_args.items);
                        if (else_split) op.@"if".@"else".args = try func.internValueList(else_args.items);
                    }
                    try new_list.append(allocator, inst);
                },
                else => try new_list.append(allocator, inst),
            }
        }

        if (changed) try func.setBlockInsts(block, new_list.items);
    }
}

/// Expand an edge's arguments for any split target parameters, creating field
/// extracts (collected in `extracts`) and the new argument list (`out_args`).
/// Returns whether the edge actually changed.
fn expandEdge(
    allocator: std.mem.Allocator,
    func: *Function,
    edge: function.Jump,
    param_fields: *const ParamFields,
    old_params: *const OldParams,
    extracts: *std.ArrayList(function.Inst),
    out_args: *std.ArrayList(Value),
) std.mem.Allocator.Error!bool {
    const olds = old_params.get(edge.target) orelse return false;
    const args = func.blockArgs(edge);

    var changed = false;
    for (olds, args) |old_param, arg| {
        if (param_fields.get(old_param)) |flds| {
            changed = true;
            for (flds, 0..) |fp, i| {
                const e = try func.createInst(func.valueType(fp), .{ .extract = .{ .aggregate = arg, .index = @intCast(i) } });
                try extracts.append(allocator, func.definingInst(e).?);
                try out_args.append(allocator, e);
            }
        } else {
            try out_args.append(allocator, arg);
        }
    }
    return changed;
}

/// Replace `arith` instructions with two constant operands by the folded
/// constant, in place (the result value is unchanged).
fn foldConstants(func: *Function) void {
    for (0..func.instCount()) |i| {
        const op = func.opcodeMut(@enumFromInt(i));
        switch (op.*) {
            .arith => |a| {
                const lhs = constValue(func, a.lhs) orelse continue;
                const rhs = constValue(func, a.rhs) orelse continue;
                const folded = foldArith(a.op, lhs, rhs) orelse continue;
                op.* = .{ .iconst = folded };
            },
            else => {},
        }
    }
}

/// Rewrite an `arith` with one constant operand into `arith_imm`, so codegen uses
/// an immediate instruction rather than materializing the constant. The constant
/// is then dropped if this was its last use. Ranges are RISC-V immediate widths.
fn foldImmediates(func: *Function) void {
    for (0..func.instCount()) |i| {
        const op = func.opcodeMut(@enumFromInt(i));
        const a = switch (op.*) {
            .arith => |a| a,
            else => continue,
        };
        // Strength-reduce `x * 2^k` into `x << k` (valid for both signednesses).
        if (a.op == .mul) {
            if (constValue(func, a.rhs)) |c| {
                if (powerOfTwoShift(c)) |k| op.* = .{ .arith_imm = .{ .op = .shl, .lhs = a.lhs, .imm = k } };
            } else if (constValue(func, a.lhs)) |c| {
                if (powerOfTwoShift(c)) |k| op.* = .{ .arith_imm = .{ .op = .shl, .lhs = a.rhs, .imm = k } };
            }
            continue;
        }
        // Unsigned `x / 2^k` is a logical shift, `x % 2^k` is a mask. (Signed needs
        // rounding adjustment, so it is left alone.)
        if ((a.op == .div or a.op == .rem) and isUnsignedType(func, func.valueType(a.lhs))) {
            if (constValue(func, a.rhs)) |c| {
                if (powerOfTwoShift(c)) |k| {
                    if (a.op == .div)
                        op.* = .{ .arith_imm = .{ .op = .shr, .lhs = a.lhs, .imm = k } }
                    else if (c <= 2048) // mask `c - 1` must fit a 12-bit andi
                        op.* = .{ .arith_imm = .{ .op = .bit_and, .lhs = a.lhs, .imm = c - 1 } };
                }
            }
            continue;
        }
        if (!immHasForm(a.op)) continue;
        // Prefer folding the right operand, the left only for commutative ops.
        if (constValue(func, a.rhs)) |c| {
            if (immFits(a.op, c)) op.* = .{ .arith_imm = .{ .op = a.op, .lhs = a.lhs, .imm = c } };
        } else if (immCommutative(a.op)) {
            if (constValue(func, a.lhs)) |c| {
                if (immFits(a.op, c)) op.* = .{ .arith_imm = .{ .op = a.op, .lhs = a.rhs, .imm = c } };
            }
        }
    }
}

fn isUnsignedType(func: *const Function, ty: types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .int => |i| i.signedness == .unsigned,
        else => false,
    };
}

/// If `c` is a power of two `2^k` with `2 <= c`, the shift amount `k`, else null.
fn powerOfTwoShift(c: i64) ?i64 {
    if (c < 2) return null;
    const u: u64 = @intCast(c);
    if (u & (u - 1) != 0) return null;
    const k = @ctz(u);
    return if (k <= 63) @intCast(k) else null;
}

fn immHasForm(op: function.BinOp) bool {
    return switch (op) {
        .add, .sub, .bit_and, .bit_or, .bit_xor, .shl, .shr => true,
        .mul, .mulh, .div, .rem => false,
    };
}

fn immCommutative(op: function.BinOp) bool {
    return switch (op) {
        .add, .bit_and, .bit_or, .bit_xor => true,
        else => false,
    };
}

fn immFits(op: function.BinOp, c: i64) bool {
    return switch (op) {
        .shl, .shr => c >= 0 and c <= 63,
        .sub => c >= -2047 and c <= 2048, // negated into an addi
        else => c >= -2048 and c <= 2047,
    };
}

/// The constant an instruction-defined value holds, if it is an `iconst`.
fn constValue(func: *const Function, value: Value) ?i64 {
    const inst = func.definingInst(value) orelse return null;
    return switch (func.opcode(inst)) {
        .iconst => |c| c,
        else => null,
    };
}

/// Fold a binary operation on two constants. Returns null for division by zero,
/// which is left for runtime. Arithmetic wraps (the result type sets the width).
fn foldArith(op: function.BinOp, lhs: i64, rhs: i64) ?i64 {
    return switch (op) {
        .add => lhs +% rhs,
        .sub => lhs -% rhs,
        .mul => lhs *% rhs,
        .div => if (rhs == 0) null else @divTrunc(lhs, rhs),
        .rem => if (rhs == 0) null else @rem(lhs, rhs),
        .bit_and => lhs & rhs,
        .bit_or => lhs | rhs,
        .bit_xor => lhs ^ rhs,
        // Shifts and the high-multiply depend on signedness/width, left for runtime, not folded.
        .shl, .shr, .mulh => null,
    };
}

/// Whether an instruction has no side effects, so it may be dropped when unused.
fn isPure(op: function.Opcode) bool {
    return switch (op) {
        .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .struct_new, .extract, .convert, .unary, .alloca, .global_addr, .dot => true,
        // Loads are kept conservatively. Stores, `if`, and calls have effects.
        // A prefetch hint behaves like store here (effectful, not droppable).
        // A matmul writes the `c` memory, likewise effectful.
        .load, .store, .prefetch, .matmul, .@"if", .call, .call_indirect => false,
    };
}

/// Map each `extract` of a `struct` construction to the field value it selects.
fn buildSubst(allocator: std.mem.Allocator, func: *const Function, subst: *Subst, param_fields: *const ParamFields) std.mem.Allocator.Error!void {
    for (0..func.instCount()) |i| {
        const inst: function.Inst = @enumFromInt(i);
        switch (func.opcode(inst)) {
            .extract => |ex| {
                // Resolve the aggregate through the substitution first, so an
                // extract of an extracted nested struct chains through.
                const agg = resolve(subst, ex.aggregate);

                // An extract of a split struct parameter forwards to its field.
                if (param_fields.get(agg)) |flds| {
                    try subst.put(allocator, func.instResult(inst).?, resolve(subst, flds[ex.index]));
                    continue;
                }

                const agg_inst = func.definingInst(agg) orelse continue;
                switch (func.opcode(agg_inst)) {
                    .struct_new => |sn| {
                        const field = func.valueList(sn.fields)[ex.index];
                        try subst.put(allocator, func.instResult(inst).?, resolve(subst, field));
                    },
                    else => {},
                }
            },
            else => {},
        }
    }
}

/// Follow a substitution chain to its final value.
fn resolve(subst: *const Subst, value: Value) Value {
    var cur = value;
    while (subst.get(cur)) |next| cur = next;
    return cur;
}

fn sub(subst: *const Subst, value: Value) Value {
    return subst.get(value) orelse value;
}

/// Rewrite every operand through the substitution.
fn applySubst(func: *Function, subst: *const Subst) void {
    for (0..func.instCount()) |i| {
        const op = func.opcodeMut(@enumFromInt(i));
        switch (op.*) {
            .iconst, .fconst, .alloca, .global_addr => {},
            .arith => |*a| {
                a.lhs = sub(subst, a.lhs);
                a.rhs = sub(subst, a.rhs);
            },
            .arith_imm => |*a| a.lhs = sub(subst, a.lhs),
            .icmp => |*cmp| {
                cmp.lhs = sub(subst, cmp.lhs);
                cmp.rhs = sub(subst, cmp.rhs);
            },
            .select => |*sel| {
                sel.cond = sub(subst, sel.cond);
                sel.then = sub(subst, sel.then);
                sel.@"else" = sub(subst, sel.@"else");
            },
            .extract => |*ex| ex.aggregate = sub(subst, ex.aggregate),
            .convert => |*cv| cv.value = sub(subst, cv.value),
            .unary => |*u| u.value = sub(subst, u.value),
            .load => |*ld| ld.ptr = sub(subst, ld.ptr),
            .store => |*st| {
                st.value = sub(subst, st.value);
                st.ptr = sub(subst, st.ptr);
            },
            .prefetch => |*pf| pf.ptr = sub(subst, pf.ptr),
            .dot => |*d| {
                d.acc = sub(subst, d.acc);
                d.a = sub(subst, d.a);
                d.b = sub(subst, d.b);
            },
            .matmul => |*mm| {
                mm.a = sub(subst, mm.a);
                mm.b = sub(subst, mm.b);
                mm.c = sub(subst, mm.c);
            },
            .struct_new => |sn| substList(func, subst, sn.fields),
            .call => |c| substList(func, subst, c.args),
            .call_indirect => |*c| {
                c.target = sub(subst, c.target);
                substList(func, subst, c.args);
            },
            .@"if" => |*cf| {
                cf.cond = sub(subst, cf.cond);
                substList(func, subst, cf.then.args);
                substList(func, subst, cf.@"else".args);
            },
        }
    }
    for (0..func.blockCount()) |bi| {
        const term = func.terminatorPtr(@enumFromInt(bi));
        if (term.*) |*t| switch (t.*) {
            .ret => |*v| {
                if (v.*) |vv| v.* = sub(subst, vv);
            },
            .jump => |*j| substList(func, subst, j.args),
        };
    }
}

fn substList(func: *Function, subst: *const Subst, list: ValueList) void {
    for (func.valueListMut(list)) |*arg| arg.* = sub(subst, arg.*);
}

/// Drop pure instructions whose result is unused, to a fixpoint. Removes the
/// aggregate ops left dead by forwarding and any other dead pure computation.
fn dropDeadInstructions(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!void {
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);

    while (true) {
        @memset(uses, 0);
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
    }
}

/// Count uses of each value across the instructions still live in blocks and
/// their terminators.
fn countUses(func: *const Function, uses: []u32) void {
    for (0..func.blockCount()) |bi| {
        const block: function.Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .iconst, .fconst, .alloca, .global_addr => {},
                .arith => |a| {
                    uses[@intFromEnum(a.lhs)] += 1;
                    uses[@intFromEnum(a.rhs)] += 1;
                },
                .arith_imm => |a| uses[@intFromEnum(a.lhs)] += 1,
                .icmp => |cmp| {
                    uses[@intFromEnum(cmp.lhs)] += 1;
                    uses[@intFromEnum(cmp.rhs)] += 1;
                },
                .select => |sel| {
                    uses[@intFromEnum(sel.cond)] += 1;
                    uses[@intFromEnum(sel.then)] += 1;
                    uses[@intFromEnum(sel.@"else")] += 1;
                },
                .extract => |ex| uses[@intFromEnum(ex.aggregate)] += 1,
                .convert => |cv| uses[@intFromEnum(cv.value)] += 1,
                .unary => |u| uses[@intFromEnum(u.value)] += 1,
                .load => |ld| uses[@intFromEnum(ld.ptr)] += 1,
                .store => |st| {
                    uses[@intFromEnum(st.value)] += 1;
                    uses[@intFromEnum(st.ptr)] += 1;
                },
                .prefetch => |pf| uses[@intFromEnum(pf.ptr)] += 1,
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

test "legalize splits struct params across an if edge" {
    const verify = @import("verify.zig");

    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const st = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });

    const block0 = try func.appendBlock();
    const c = try func.appendBlockParam(block0, bool_t);
    const v0 = try func.appendBlockParam(block0, i32_t);
    const v1 = try func.appendBlockParam(block0, i32_t);
    const block1 = try func.appendBlock();
    const p = try func.appendBlockParam(block1, st);
    const block2 = try func.appendBlock();

    const s = try func.appendStructNew(block0, st, &.{ v0, v1 });
    try func.appendIf(block0, c, .{ .target = block1, .args = &.{s} }, .{ .target = block2 });

    const f0 = try func.appendInst(block1, i32_t, .{ .extract = .{ .aggregate = p, .index = 0 } });
    func.setTerminator(block1, .{ .ret = f0 });
    func.setTerminator(block2, .{ .ret = null });

    var before = try verify.verify(std.testing.allocator, &func, .low);
    defer before.deinit();
    try std.testing.expect(!before.ok());

    try legalize(std.testing.allocator, &func);

    var after = try verify.verify(std.testing.allocator, &func, .low);
    defer after.deinit();
    try std.testing.expect(after.ok());
    try std.testing.expectEqual(@as(usize, 2), func.blockParams(block1).len);
}

test "legalize splits struct params across a jump edge" {
    const verify = @import("verify.zig");

    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const st = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });

    const block0 = try func.appendBlock();
    const v0 = try func.appendBlockParam(block0, i32_t);
    const v1 = try func.appendBlockParam(block0, i32_t);
    const block1 = try func.appendBlock();
    const p = try func.appendBlockParam(block1, st);

    const s = try func.appendStructNew(block0, st, &.{ v0, v1 });
    try func.setJump(block0, block1, &.{s});

    const f0 = try func.appendInst(block1, i32_t, .{ .extract = .{ .aggregate = p, .index = 0 } });
    func.setTerminator(block1, .{ .ret = f0 });

    var before = try verify.verify(std.testing.allocator, &func, .low);
    defer before.deinit();
    try std.testing.expect(!before.ok());

    try legalize(std.testing.allocator, &func);

    var after = try verify.verify(std.testing.allocator, &func, .low);
    defer after.deinit();
    try std.testing.expect(after.ok());
    try std.testing.expectEqual(@as(usize, 2), func.blockParams(block1).len);
}

test "legalize splits a struct entry parameter into scalars" {
    const verify = @import("verify.zig");

    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const st = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, st);
    const f0 = try func.appendInst(entry, i32_t, .{ .extract = .{ .aggregate = p, .index = 0 } });
    func.setTerminator(entry, .{ .ret = f0 });

    var before = try verify.verify(std.testing.allocator, &func, .low);
    defer before.deinit();
    try std.testing.expect(!before.ok());

    try legalize(std.testing.allocator, &func);

    var after = try verify.verify(std.testing.allocator, &func, .low);
    defer after.deinit();
    try std.testing.expect(after.ok());

    // The struct parameter became two scalar parameters. The return forwards to
    // the first field parameter.
    try std.testing.expectEqual(@as(usize, 2), func.blockParams(entry).len);
    try std.testing.expectEqual(function.Terminator{ .ret = func.blockParams(entry)[0] }, func.terminator(entry).?);
}

test "legalize folds constant arithmetic" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const x = try func.appendInst(entry, i32_t, .{ .iconst = 10 });
    const y = try func.appendInst(entry, i32_t, .{ .iconst = 20 });
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(entry, .{ .ret = sum });

    try legalize(std.testing.allocator, &func);

    // The addition folded to a constant, and the operand constants are gone.
    try std.testing.expectEqual(@as(i64, 30), func.opcode(func.definingInst(sum).?).iconst);
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(entry).len);
}

test "legalize folds a constant arith operand into arith_imm" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const c = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = c } });
    func.setTerminator(entry, .{ .ret = sum });

    try legalize(std.testing.allocator, &func);

    // The add became `x + 7` and the now-dead constant was dropped, leaving just
    // the single arith_imm instruction.
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(entry).len);
    const op = func.opcode(func.definingInst(sum).?);
    try std.testing.expectEqual(@as(i64, 7), op.arith_imm.imm);
    try std.testing.expectEqual(function.BinOp.add, op.arith_imm.op);
}

test "legalize strength-reduces multiply by a power of two to a shift" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const c = try func.appendInst(entry, i32_t, .{ .iconst = 8 });
    const prod = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = c } });
    func.setTerminator(entry, .{ .ret = prod });

    try legalize(std.testing.allocator, &func);

    // `x * 8` became `x << 3`, the constant 8 is gone.
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(entry).len);
    const op = func.opcode(func.definingInst(prod).?);
    try std.testing.expectEqual(function.BinOp.shl, op.arith_imm.op);
    try std.testing.expectEqual(@as(i64, 3), op.arith_imm.imm);
}

test "legalize strength-reduces unsigned divide and remainder by a power of two" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, u32_t);
    const c = try func.appendInst(entry, u32_t, .{ .iconst = 8 });
    const q = try func.appendInst(entry, u32_t, .{ .arith = .{ .op = .div, .lhs = x, .rhs = c } });
    const d = try func.appendInst(entry, u32_t, .{ .iconst = 8 });
    const m = try func.appendInst(entry, u32_t, .{ .arith = .{ .op = .rem, .lhs = x, .rhs = d } });
    _ = try func.appendInst(entry, u32_t, .{ .arith = .{ .op = .add, .lhs = q, .rhs = m } });
    const sum = func.blockInsts(entry)[func.blockInsts(entry).len - 1];
    func.setTerminator(entry, .{ .ret = func.instResult(sum).? });

    try legalize(std.testing.allocator, &func);

    // x / 8 -> x >> 3 (logical), x % 8 -> x & 7.
    const qop = func.opcode(func.definingInst(q).?);
    try std.testing.expectEqual(function.BinOp.shr, qop.arith_imm.op);
    try std.testing.expectEqual(@as(i64, 3), qop.arith_imm.imm);
    const mop = func.opcode(func.definingInst(m).?);
    try std.testing.expectEqual(function.BinOp.bit_and, mop.arith_imm.op);
    try std.testing.expectEqual(@as(i64, 7), mop.arith_imm.imm);
}

test "legalize removes dead pure instructions" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    _ = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a } }); // dead
    func.setTerminator(entry, .{ .ret = a });

    try legalize(std.testing.allocator, &func);

    try std.testing.expectEqual(@as(usize, 0), func.blockInsts(entry).len);
}

test "legalize forwards through nested structs" {
    const verify = @import("verify.zig");

    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);

    const inner_t = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });
    const inner = try func.appendStructNew(entry, inner_t, &.{ a, b });
    const outer_t = try func.types.intern(.{ .@"struct" = &.{inner_t} });
    const outer = try func.appendStructNew(entry, outer_t, &.{inner});

    const got_inner = try func.appendInst(entry, inner_t, .{ .extract = .{ .aggregate = outer, .index = 0 } });
    const got_b = try func.appendInst(entry, i32_t, .{ .extract = .{ .aggregate = got_inner, .index = 1 } });
    func.setTerminator(entry, .{ .ret = got_b });

    try legalize(std.testing.allocator, &func);

    var after = try verify.verify(std.testing.allocator, &func, .low);
    defer after.deinit();
    try std.testing.expect(after.ok());
    try std.testing.expectEqual(function.Terminator{ .ret = b }, func.terminator(entry).?);
}

test "legalize eliminates struct/extract and passes the low profile" {
    const verify = @import("verify.zig");

    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const st = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });
    const s = try func.appendStructNew(entry, st, &.{ a, b });
    const f0 = try func.appendInst(entry, i32_t, .{ .extract = .{ .aggregate = s, .index = 0 } });
    func.setTerminator(entry, .{ .ret = f0 });

    // Before: the struct type makes it fail the low profile.
    var before = try verify.verify(std.testing.allocator, &func, .low);
    defer before.deinit();
    try std.testing.expect(!before.ok());

    try legalize(std.testing.allocator, &func);

    // After: the aggregate is gone, the return forwards to `a`, low profile passes.
    var after = try verify.verify(std.testing.allocator, &func, .low);
    defer after.deinit();
    try std.testing.expect(after.ok());
    try std.testing.expectEqual(function.Terminator{ .ret = a }, func.terminator(entry).?);
}
