//! Checks a function is well-formed for a given profile, returning structured
//! diagnostics rather than a bare pass/fail. The high profile allows composite
//! types and structured control. The low profile (consumed by codegen) is
//! primitive-only. Legalization moves a function from high to low.

const std = @import("std");
const function = @import("function.zig");
const types = @import("types.zig");

const Function = function.Function;
const Block = function.Block;
const Value = function.Value;
const Jump = function.Jump;
const Opcode = function.Opcode;
const AttrTarget = function.AttrTarget;
const Type = types.Type;

/// Which profile to verify against.
pub const Profile = enum {
    /// Frontend IR: composite types and structured control allowed.
    high,
    /// Codegen IR: primitive types only.
    low,
};

/// A single well-formedness problem.
pub const Diagnostic = union(enum) {
    /// A value has a composite type, which is illegal in the low profile.
    composite_type_in_low_profile: Value,
    /// An edge passes the wrong number of arguments to its target's parameters.
    arg_count_mismatch: struct { target: Block, expected: u32, found: u32 },
    /// An edge argument's type does not match the target parameter's type.
    arg_type_mismatch: struct { target: Block, index: u32, expected: Type, found: Type },
    /// A value is used in a block its definition does not dominate.
    not_dominated: struct { value: Value, block: Block },
    /// An `endian` attribute is attached to something that is not a memory op.
    misplaced_endian: AttrTarget,
    /// An instruction's operands have mismatched types (named by its result).
    operand_type_mismatch: Value,
};

/// The collected diagnostics of a verification run.
pub const Diagnostics = struct {
    allocator: std.mem.Allocator,
    list: std.ArrayList(Diagnostic),

    pub fn deinit(self: *Diagnostics) void {
        self.list.deinit(self.allocator);
    }

    /// Whether the function passed (no diagnostics).
    pub fn ok(self: *const Diagnostics) bool {
        return self.list.items.len == 0;
    }

    pub fn count(self: *const Diagnostics) usize {
        return self.list.items.len;
    }

    pub fn items(self: *const Diagnostics) []const Diagnostic {
        return self.list.items;
    }

    fn add(self: *Diagnostics, diag: Diagnostic) std.mem.Allocator.Error!void {
        try self.list.append(self.allocator, diag);
    }
};

/// Verify `func` against `profile`, returning the diagnostics. The caller owns
/// and must `deinit` the result.
pub fn verify(allocator: std.mem.Allocator, func: *const Function, profile: Profile) std.mem.Allocator.Error!Diagnostics {
    var diags: Diagnostics = .{ .allocator = allocator, .list = .empty };
    errdefer diags.deinit();

    try checkEdges(func, &diags);
    try checkDominance(func, &diags);
    try checkAttributes(func, &diags);
    try checkOperandTypes(func, &diags);
    if (profile == .low) try checkPrimitiveTypes(func, &diags);

    return diags;
}

/// Binary arithmetic and comparison operands must have matching types.
fn checkOperandTypes(func: *const Function, diags: *Diagnostics) std.mem.Allocator.Error!void {
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            const mismatch = switch (func.opcode(inst)) {
                .arith => |a| func.valueType(a.lhs) != func.valueType(a.rhs),
                .icmp => |c| func.valueType(c.lhs) != func.valueType(c.rhs),
                else => false,
            };
            if (mismatch) {
                if (func.instResult(inst)) |result| try diags.add(.{ .operand_type_mismatch = result });
            }
        }
    }
}

/// Attribute shape/placement checks. Currently: `endian` only belongs on a
/// memory operation (a load result value, or a load/store instruction).
fn checkAttributes(func: *const Function, diags: *Diagnostics) std.mem.Allocator.Error!void {
    for (func.attributeEntries()) |entry| {
        switch (entry.attr) {
            .endian => if (!endianTargetOk(func, entry.target)) {
                try diags.add(.{ .misplaced_endian = entry.target });
            },
            else => {},
        }
    }
}

fn endianTargetOk(func: *const Function, target: AttrTarget) bool {
    return switch (target) {
        .value => |v| if (func.definingInst(v)) |inst| isMemoryOp(func.opcode(inst)) else false,
        .inst => |i| isMemoryOp(func.opcode(i)),
        .func, .block => false,
    };
}

fn isMemoryOp(op: Opcode) bool {
    return switch (op) {
        .load, .store => true,
        else => false,
    };
}

/// Dominator sets over a function's control-flow graph, computed by the standard
/// iterative algorithm. `dom[b*n + a]` is true when block `a` dominates block `b`.
const Dominance = struct {
    allocator: std.mem.Allocator,
    n: usize,
    dom: []bool,

    fn deinit(self: *Dominance) void {
        self.allocator.free(self.dom);
    }

    fn dominates(self: *const Dominance, a: usize, b: usize) bool {
        return self.dom[b * self.n + a];
    }

    fn compute(allocator: std.mem.Allocator, func: *const Function) std.mem.Allocator.Error!Dominance {
        const n = func.blockCount();

        // Successor lists, derived from `if` edges and the jump terminator.
        const succ = try allocator.alloc(std.ArrayList(u32), n);
        defer {
            for (succ) |*s| s.deinit(allocator);
            allocator.free(succ);
        }
        for (succ) |*s| s.* = .empty;

        for (0..n) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockInsts(block)) |inst| {
                switch (func.opcode(inst)) {
                    .@"if" => |cond| {
                        try succ[bi].append(allocator, @intFromEnum(cond.then.target));
                        try succ[bi].append(allocator, @intFromEnum(cond.@"else".target));
                    },
                    else => {},
                }
            }
            if (func.terminator(block)) |term| switch (term) {
                .jump => |j| try succ[bi].append(allocator, @intFromEnum(j.target)),
                .ret => {},
            };
        }

        // Predecessor lists.
        const preds = try allocator.alloc(std.ArrayList(u32), n);
        defer {
            for (preds) |*p| p.deinit(allocator);
            allocator.free(preds);
        }
        for (preds) |*p| p.* = .empty;
        for (0..n) |bi| {
            for (succ[bi].items) |s| try preds[s].append(allocator, @intCast(bi));
        }

        // dom[b][a]: block 0 is the entry, dominated only by itself, all others
        // start dominated by everything, then shrink to a fixpoint.
        const dom = try allocator.alloc(bool, n * n);
        errdefer allocator.free(dom);
        @memset(dom, true);
        if (n > 0) {
            for (0..n) |a| dom[a] = (a == 0);
        }

        const tmp = try allocator.alloc(bool, n);
        defer allocator.free(tmp);

        var changed = true;
        while (changed) {
            changed = false;
            for (1..n) |b| {
                if (preds[b].items.len == 0) continue; // unreachable
                for (0..n) |a| {
                    var all = true;
                    for (preds[b].items) |p| {
                        if (!dom[p * n + a]) {
                            all = false;
                            break;
                        }
                    }
                    tmp[a] = all;
                }
                tmp[b] = true;
                for (0..n) |a| {
                    if (dom[b * n + a] != tmp[a]) {
                        dom[b * n + a] = tmp[a];
                        changed = true;
                    }
                }
            }
        }

        return .{ .allocator = allocator, .n = n, .dom = dom };
    }
};

/// Every use of a value must be in a block dominated by the value's definition.
fn checkDominance(func: *const Function, diags: *Diagnostics) std.mem.Allocator.Error!void {
    const n = func.blockCount();
    if (n == 0) return;

    var dominance = try Dominance.compute(diags.allocator, func);
    defer dominance.deinit();

    // The block that defines each value.
    const def_block = try diags.allocator.alloc(u32, func.valueCount());
    defer diags.allocator.free(def_block);
    for (0..n) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |param| def_block[@intFromEnum(param)] = @intCast(bi);
        for (func.blockInsts(block)) |inst| {
            if (func.instResult(inst)) |result| def_block[@intFromEnum(result)] = @intCast(bi);
        }
    }

    for (0..n) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .iconst, .fconst, .alloca, .global_addr => {},
                .arith => |a| {
                    try checkUse(&dominance, def_block, diags, a.lhs, bi);
                    try checkUse(&dominance, def_block, diags, a.rhs, bi);
                },
                .arith_imm => |a| try checkUse(&dominance, def_block, diags, a.lhs, bi),
                .icmp => |cmp| {
                    try checkUse(&dominance, def_block, diags, cmp.lhs, bi);
                    try checkUse(&dominance, def_block, diags, cmp.rhs, bi);
                },
                .select => |sel| {
                    try checkUse(&dominance, def_block, diags, sel.cond, bi);
                    try checkUse(&dominance, def_block, diags, sel.then, bi);
                    try checkUse(&dominance, def_block, diags, sel.@"else", bi);
                },
                .struct_new => |sn| for (func.valueList(sn.fields)) |field| {
                    try checkUse(&dominance, def_block, diags, field, bi);
                },
                .call => |c| for (func.valueList(c.args)) |arg| {
                    try checkUse(&dominance, def_block, diags, arg, bi);
                },
                .call_indirect => |c| {
                    try checkUse(&dominance, def_block, diags, c.target, bi);
                    for (func.valueList(c.args)) |arg| try checkUse(&dominance, def_block, diags, arg, bi);
                },
                .extract => |ex| try checkUse(&dominance, def_block, diags, ex.aggregate, bi),
                .convert => |cv| try checkUse(&dominance, def_block, diags, cv.value, bi),
                .unary => |u| try checkUse(&dominance, def_block, diags, u.value, bi),
                .load => |ld| try checkUse(&dominance, def_block, diags, ld.ptr, bi),
                .store => |st| {
                    try checkUse(&dominance, def_block, diags, st.value, bi);
                    try checkUse(&dominance, def_block, diags, st.ptr, bi);
                },
                .@"if" => |cond| {
                    try checkUse(&dominance, def_block, diags, cond.cond, bi);
                    for (func.blockArgs(cond.then)) |arg| try checkUse(&dominance, def_block, diags, arg, bi);
                    for (func.blockArgs(cond.@"else")) |arg| try checkUse(&dominance, def_block, diags, arg, bi);
                },
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .ret => |value| if (value) |v| try checkUse(&dominance, def_block, diags, v, bi),
            .jump => |j| for (func.blockArgs(j)) |arg| try checkUse(&dominance, def_block, diags, arg, bi),
        };
    }
}

fn checkUse(
    dominance: *const Dominance,
    def_block: []const u32,
    diags: *Diagnostics,
    value: Value,
    use_block: usize,
) std.mem.Allocator.Error!void {
    const db = def_block[@intFromEnum(value)];
    if (db != use_block and !dominance.dominates(db, use_block)) {
        try diags.add(.{ .not_dominated = .{ .value = value, .block = @enumFromInt(use_block) } });
    }
}

/// Every edge must pass arguments matching its target block's parameters. This
/// holds in both profiles.
fn checkEdges(func: *const Function, diags: *Diagnostics) std.mem.Allocator.Error!void {
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => |cond| {
                    try checkEdge(func, diags, cond.then);
                    try checkEdge(func, diags, cond.@"else");
                },
                else => {},
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .jump => |j| try checkEdge(func, diags, j),
            .ret => {},
        };
    }
}

fn checkEdge(func: *const Function, diags: *Diagnostics, jump: Jump) std.mem.Allocator.Error!void {
    const args = func.blockArgs(jump);
    const params = func.blockParams(jump.target);
    if (args.len != params.len) {
        try diags.add(.{ .arg_count_mismatch = .{
            .target = jump.target,
            .expected = @intCast(params.len),
            .found = @intCast(args.len),
        } });
        return; // types are unverifiable once the counts disagree
    }
    for (args, params, 0..) |arg, param, i| {
        const arg_ty = func.valueType(arg);
        const param_ty = func.valueType(param);
        if (arg_ty != param_ty) {
            try diags.add(.{ .arg_type_mismatch = .{
                .target = jump.target,
                .index = @intCast(i),
                .expected = param_ty,
                .found = arg_ty,
            } });
        }
    }
}

/// In the low profile, every value must have a primitive type.
fn checkPrimitiveTypes(func: *const Function, diags: *Diagnostics) std.mem.Allocator.Error!void {
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |param| try checkValueType(func, diags, param);
        for (func.blockInsts(block)) |inst| {
            if (func.instResult(inst)) |result| try checkValueType(func, diags, result);
        }
    }
}

fn checkValueType(func: *const Function, diags: *Diagnostics, value: Value) std.mem.Allocator.Error!void {
    switch (func.types.type_kind(func.valueType(value))) {
        .@"struct", .array, .slice => try diags.add(.{ .composite_type_in_low_profile = value }),
        else => {},
    }
}

test "using a value not dominated by its definition is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const left = try func.appendBlock();
    const right = try func.appendBlock();
    const merge = try func.appendBlock();

    const cond = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.appendIf(entry, cond, .{ .target = left }, .{ .target = right });

    // x is defined only on the left arm, so it does not dominate the merge.
    const x = try func.appendInst(left, i32_t, .{ .iconst = 7 });
    try func.setJump(left, merge, &.{});
    try func.setJump(right, merge, &.{});
    func.setTerminator(merge, .{ .ret = x });

    var d = try verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(@as(usize, 1), d.count());
    try std.testing.expectEqual(Diagnostic{ .not_dominated = .{ .value = x, .block = merge } }, d.items()[0]);
}

test "endian is rejected when not on a memory op" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const loaded = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = loaded, .rhs = loaded } });
    func.setTerminator(entry, .{ .ret = sum });

    try func.addAttr(.{ .value = loaded }, .{ .endian = .big }); // ok: load result
    try func.addAttr(.{ .value = sum }, .{ .endian = .big }); // misplaced: iadd result

    var d = try verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(@as(usize, 1), d.count());
    try std.testing.expectEqual(Diagnostic{ .misplaced_endian = .{ .value = sum } }, d.items()[0]);
}

test "arith with mismatched operand types is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i64_t);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    var d = try verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(Diagnostic{ .operand_type_mismatch = sum }, d.items()[0]);
}

test "a well-formed function passes verification in both profiles" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    var high = try verify(std.testing.allocator, &func, .high);
    defer high.deinit();
    try std.testing.expect(high.ok());

    var low = try verify(std.testing.allocator, &func, .low);
    defer low.deinit();
    try std.testing.expect(low.ok());
}

test "low profile rejects composite types, high profile allows them" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const st = try func.types.intern(.{ .@"struct" = &.{i32_t} });
    const entry = try func.appendBlock();
    _ = try func.appendBlockParam(entry, st);
    func.setTerminator(entry, .{ .ret = null });

    var high = try verify(std.testing.allocator, &func, .high);
    defer high.deinit();
    try std.testing.expect(high.ok());

    var low = try verify(std.testing.allocator, &func, .low);
    defer low.deinit();
    try std.testing.expect(!low.ok());
    try std.testing.expectEqual(@as(usize, 1), low.count());
}

test "an edge passing the wrong number of arguments is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const target = try func.appendBlock();
    _ = try func.appendBlockParam(target, i32_t);
    _ = try func.appendBlockParam(target, i32_t);

    const v = try func.appendInst(entry, i32_t, .{ .iconst = 1 });
    try func.setJump(entry, target, &.{v}); // passes 1 arg, target wants 2
    func.setTerminator(target, .{ .ret = null });

    var d = try verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(@as(usize, 1), d.count());
    try std.testing.expectEqual(Diagnostic{ .arg_count_mismatch = .{ .target = target, .expected = 2, .found = 1 } }, d.items()[0]);
}

test "an edge passing a mismatched argument type is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const target = try func.appendBlock();
    _ = try func.appendBlockParam(target, i64_t); // wants i64

    const v = try func.appendInst(entry, i32_t, .{ .iconst = 1 }); // produces i32
    try func.setJump(entry, target, &.{v});
    func.setTerminator(target, .{ .ret = null });

    var d = try verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(@as(usize, 1), d.count());
    try std.testing.expectEqual(
        Diagnostic{ .arg_type_mismatch = .{ .target = target, .index = 0, .expected = i64_t, .found = i32_t } },
        d.items()[0],
    );
}
