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

/// Pointer arithmetic: `add`/`sub` of a pointer and an integer offset (either order) is a valid
/// address computation, so its operands are allowed to differ in type. Every frontend emits this for
/// indexed loads/stores and the backends lower it directly, so it is not a type mismatch.
fn pointerArith(func: *const Function, a: function.Arith) bool {
    if (a.op != .add and a.op != .sub) return false;
    const lt = func.types.type_kind(func.valueType(a.lhs));
    const rt = func.types.type_kind(func.valueType(a.rhs));
    const l_ptr = lt == .ptr;
    const r_ptr = rt == .ptr;
    return (l_ptr and rt == .int) or (r_ptr and lt == .int);
}

/// Binary arithmetic and comparison operands must have matching types.
fn checkOperandTypes(func: *const Function, diags: *Diagnostics) std.mem.Allocator.Error!void {
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .arith => |a| if (func.valueType(a.lhs) != func.valueType(a.rhs) and !pointerArith(func, a)) {
                    if (func.instResult(inst)) |result| try diags.add(.{ .operand_type_mismatch = result });
                },
                .icmp => |c| if (func.valueType(c.lhs) != func.valueType(c.rhs)) {
                    if (func.instResult(inst)) |result| try diags.add(.{ .operand_type_mismatch = result });
                },
                .dot => |d| if (dotOperandsMismatch(func, inst, d)) {
                    if (func.instResult(inst)) |result| try diags.add(.{ .operand_type_mismatch = result });
                },
                // matmul has no result value, so a mismatch is reported against
                // `c` (the pointer the tile is written to).
                .matmul => |mm| if (matmulOperandsMismatch(func, mm)) {
                    try diags.add(.{ .operand_type_mismatch = mm.c });
                },
                else => {},
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
        .load, .store, .prefetch, .matmul => true,
        else => false,
    };
}

/// `dot`'s accumulator (and result) must be `<4 x i32>`; `a` and `b` must be
/// the same `<16 x i8>` (signed) or `<16 x u8>` (unsigned) type.
fn dotOperandsMismatch(func: *const Function, inst: function.Inst, d: function.Dot) bool {
    if (!isDotAccType(func, func.valueType(d.acc))) return true;
    if (func.instResult(inst)) |result| {
        if (func.valueType(result) != func.valueType(d.acc)) return true;
    }
    const a_ty = func.valueType(d.a);
    const b_ty = func.valueType(d.b);
    if (a_ty != b_ty) return true;
    return !isDotDataType(func, a_ty);
}

/// `<4 x i32>`, the required accumulator/result type of `dot`.
fn isDotAccType(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .vector => |v| v.len == 4 and isIntOfBits(func, v.elem, 32),
        else => false,
    };
}

/// `<16 x i8>` or `<16 x u8>`, the required `a`/`b` type of `dot`.
fn isDotDataType(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .vector => |v| v.len == 16 and isIntOfBits(func, v.elem, 8),
        else => false,
    };
}

/// `matmul`'s `a`, `b`, and `c` must all be `ptr` values (`a`/`b` are read
/// from memory, `c` is where the tile is written). A `quant` epilogue is
/// int8-only: the et-soc requantize path (scale -> saturate -> pack) only
/// exists for the int32 accumulator that `dtype == .int8` produces, so any
/// other dtype paired with a non-null `quant` is rejected here too (reusing
/// the existing "matmul mismatch reported against c" diagnostic path). A
/// `per_column` scale is one fp32 scale per output column, so its interned
/// length must equal `n`; a mismatched length is rejected the same way. A
/// `bias`, when present, is one int32 per output column added to the int32
/// accumulator before scaling, so its interned length must equal `n` too;
/// `zero_point` is a single per-tensor constant so it has no length to check.
/// `accumulate` (real `C += A*B`) is fp32/fp16-only, so it may not be paired
/// with a `quant` epilogue (which requantizes an int32 accumulator); that
/// combination is rejected here too.
fn matmulOperandsMismatch(func: *const Function, mm: function.MatMul) bool {
    const quant_mismatch = if (mm.quant) |q| blk: {
        // The quant epilogue requantizes the int32 accumulator, which only int8 and uint8 inputs
        // produce (fp32/fp16 accumulate in fp32 TenC that the transform chain cannot consume). Both
        // signednesses are valid: asymmetric quantization uses unsigned uint8 activations, and the
        // lowering handles either (both map to the same 8-bit tensor path, di.tt == .int8).
        if (mm.dtype != .int8 and mm.dtype != .uint8) break :blk true;
        if (q.bias) |bh| {
            if (func.biasList(bh).len != mm.n) break :blk true;
        }
        break :blk switch (q.scale) {
            .scalar => false,
            .per_column => |h| func.scaleList(h).len != mm.n,
        };
    } else false;
    // A per-operand signedness override is only meaningful for the 8-bit integer hardware type:
    // fp32/fp16 have no signedness to override, and uint8 already spells "both unsigned" via
    // dtype alone, so requiring dtype == .int8 keeps one canonical spelling per configuration
    // (symmetric-signed = int8+null, symmetric-unsigned = uint8+null, mixed = int8+input_signs).
    const signs_mismatch = mm.input_signs != null and mm.dtype != .int8;
    // `accumulate` (real `C += A*B`) preloads the existing fp32 C tile into the fp32 TenC before the
    // fma passes. The quant epilogue instead consumes an int32 TenC and writes packed bytes, so an
    // fp32 C preload is meaningless there. Forbid the pairing uniformly at verify time (the isel
    // lowering also guards it defensively, for IR that skipped verify).
    const accumulate_quant_mismatch = mm.accumulate and mm.quant != null;
    return !isPtrType(func, func.valueType(mm.a)) or
        !isPtrType(func, func.valueType(mm.b)) or
        !isPtrType(func, func.valueType(mm.c)) or
        quant_mismatch or
        signs_mismatch or
        accumulate_quant_mismatch;
}

fn isPtrType(func: *const Function, ty: Type) bool {
    return func.types.type_kind(ty) == .ptr;
}

fn isIntOfBits(func: *const Function, ty: Type, bits: u16) bool {
    return switch (func.types.type_kind(ty)) {
        .int => |i| i.bits == bits,
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
                .prefetch => |pf| try checkUse(&dominance, def_block, diags, pf.ptr, bi),
                .dot => |d| {
                    try checkUse(&dominance, def_block, diags, d.acc, bi);
                    try checkUse(&dominance, def_block, diags, d.a, bi);
                    try checkUse(&dominance, def_block, diags, d.b, bi);
                },
                .matmul => |mm| {
                    try checkUse(&dominance, def_block, diags, mm.a, bi);
                    try checkUse(&dominance, def_block, diags, mm.b, bi);
                    try checkUse(&dominance, def_block, diags, mm.c, bi);
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

test "f16 value with f16<->f32 width-changing converts verifies clean" {
    // convert has no dedicated operand/result type check (unlike arith/icmp,
    // which require exact operand-type equality), so a width-changing
    // float<->float convert was never rejected in the first place; this pins
    // that behavior now that f16 is a real FloatKind member.
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const half = try func.appendBlockParam(entry, f16_t);
    const widened = try func.appendInst(entry, f32_t, .{ .convert = .{ .value = half } }); // f16 -> f32
    const narrowed = try func.appendInst(entry, f16_t, .{ .convert = .{ .value = widened } }); // f32 -> f16
    const back = try func.appendInst(entry, f32_t, .{ .convert = .{ .value = narrowed } }); // f16 -> f32 again
    func.setTerminator(entry, .{ .ret = back });

    var d = try verify(std.testing.allocator, &func, .low);
    defer d.deinit();
    try std.testing.expect(d.ok());
}

test "f16 vs f32 operand type mismatch on arith is reported" {
    // Float types are checked by interned-handle equality, same as any other
    // type: f16 and f32 are distinct handles, so mixing them in a binary op
    // is rejected the same way an i32/i64 mismatch already is.
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f16_t);
    const b = try func.appendBlockParam(entry, f32_t);
    const sum = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    var d = try verify(std.testing.allocator, &func, .high);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(Diagnostic{ .operand_type_mismatch = sum }, d.items()[0]);
}

test "dot with matching operand types verifies clean" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const v16i8 = try func.types.intern(.{ .vector = .{ .len = 16, .elem = i8_t } });
    const v4i32 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const entry = try func.appendBlock();
    const acc = try func.appendBlockParam(entry, v4i32);
    const a = try func.appendBlockParam(entry, v16i8);
    const b = try func.appendBlockParam(entry, v16i8);
    const result = try func.appendDot(entry, acc, a, b);
    func.setTerminator(entry, .{ .ret = result });

    var d = try verify(std.testing.allocator, &func, .low);
    defer d.deinit();
    try std.testing.expect(d.ok());
}

test "dot with mismatched a/b types is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const i16_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 16 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const v16i8 = try func.types.intern(.{ .vector = .{ .len = 16, .elem = i8_t } });
    const v16i16 = try func.types.intern(.{ .vector = .{ .len = 16, .elem = i16_t } });
    const v4i32 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const entry = try func.appendBlock();
    const acc = try func.appendBlockParam(entry, v4i32);
    const a = try func.appendBlockParam(entry, v16i8);
    const b = try func.appendBlockParam(entry, v16i16); // wrong: does not match a's type
    const result = try func.appendDot(entry, acc, a, b);
    func.setTerminator(entry, .{ .ret = result });

    var d = try verify(std.testing.allocator, &func, .low);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(Diagnostic{ .operand_type_mismatch = result }, d.items()[0]);
}

test "dot with a non-<4 x i32> accumulator is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const v16i8 = try func.types.intern(.{ .vector = .{ .len = 16, .elem = i8_t } });
    const v4i32 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const entry = try func.appendBlock();
    const acc = try func.appendBlockParam(entry, i32_t); // wrong: scalar, not <4 x i32>
    const a = try func.appendBlockParam(entry, v16i8);
    const b = try func.appendBlockParam(entry, v16i8);
    const result = try func.appendInst(entry, v4i32, .{ .dot = .{ .acc = acc, .a = a, .b = b } });
    func.setTerminator(entry, .{ .ret = result });

    var d = try verify(std.testing.allocator, &func, .low);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(Diagnostic{ .operand_type_mismatch = result }, d.items()[0]);
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

test "a prefetch hint verifies clean in the low profile and prints as a hint" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    try func.appendPrefetch(entry, p);
    func.setTerminator(entry, .{ .ret = null });

    var low = try verify(std.testing.allocator, &func, .low);
    defer low.deinit();
    try std.testing.expect(low.ok());

    const text = try std.fmt.allocPrint(std.testing.allocator, "{f}", .{func});
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "prefetch v0") != null);
}

test "a matmul over pointer operands verifies clean and prints the tile" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmul(entry, a, b, c, 4, 4, 4, .fp32, false);
    func.setTerminator(entry, .{ .ret = null });

    var low = try verify(std.testing.allocator, &func, .low);
    defer low.deinit();
    try std.testing.expect(low.ok());

    const text = try std.fmt.allocPrint(std.testing.allocator, "{f}", .{func});
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "matmul c=v2, a=v0, b=v1 [4 x 4 x 4] fp32") != null);
}

test "matmul with a non-pointer operand is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, i32_t); // wrong: not a pointer
    try func.appendMatmul(entry, a, b, c, 4, 4, 4, .fp32, false);
    func.setTerminator(entry, .{ .ret = null });

    var d = try verify(std.testing.allocator, &func, .low);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(Diagnostic{ .operand_type_mismatch = c }, d.items()[0]);
}

test "an int8 matmul with a mixed input_signs override verifies clean" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulSigned(entry, a, b, c, 4, 4, 4, .int8, false, .{ .a_unsigned = true, .b_unsigned = false });
    func.setTerminator(entry, .{ .ret = null });

    var low = try verify(std.testing.allocator, &func, .low);
    defer low.deinit();
    try std.testing.expect(low.ok());
}

test "a non-int8 matmul with an input_signs override is rejected" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    // fp32 has no signedness to override; input_signs is only meaningful paired with .int8.
    try func.appendMatmulSigned(entry, a, b, c, 4, 4, 4, .fp32, false, .{ .a_unsigned = true, .b_unsigned = false });
    func.setTerminator(entry, .{ .ret = null });

    var d = try verify(std.testing.allocator, &func, .low);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(Diagnostic{ .operand_type_mismatch = c }, d.items()[0]);
}

test "a matmul quant epilogue on int8 verifies clean" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulQuant(entry, a, b, c, 4, 4, 4, .int8, false, .{ .scale = .{ .scalar = 0x3F000000 }, .relu = true });
    func.setTerminator(entry, .{ .ret = null });

    var low = try verify(std.testing.allocator, &func, .low);
    defer low.deinit();
    try std.testing.expect(low.ok());
}

test "a matmul quant epilogue with a per_column scale of len==n verifies clean" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulQuantPerColumn(entry, a, b, c, 4, 4, 4, .int8, false, true, .i8, &.{ 0x3F800000, 0x3F000000, 0x3E800000, 0x40000000 });
    func.setTerminator(entry, .{ .ret = null });

    var low = try verify(std.testing.allocator, &func, .low);
    defer low.deinit();
    try std.testing.expect(low.ok());
}

test "a matmul quant epilogue with a per_column scale of len!=n is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulQuantPerColumn(entry, a, b, c, 4, 4, 4, .int8, false, true, .i8, &.{ 0x3F800000, 0x3F000000, 0x3E800000 });
    func.setTerminator(entry, .{ .ret = null });

    var d = try verify(std.testing.allocator, &func, .low);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(Diagnostic{ .operand_type_mismatch = c }, d.items()[0]);
}

test "a matmul quant epilogue on a non-int8 dtype is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulQuant(entry, a, b, c, 4, 4, 4, .fp32, false, .{ .scale = .{ .scalar = 0x3F000000 }, .relu = true });
    func.setTerminator(entry, .{ .ret = null });

    var d = try verify(std.testing.allocator, &func, .low);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(Diagnostic{ .operand_type_mismatch = c }, d.items()[0]);
}

test "a matmul quant epilogue with a per-column bias of len==n and a nonzero zero_point verifies clean" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulQuantSpec(entry, a, b, c, 4, 4, 4, .int8, false, .{
        .scale_scalar = 0x3F000000,
        .bias = &.{ 1, -2, 3, -4 },
        .zero_point = 17, // nonzero zero-point has no length constraint, must not be flagged
        .relu = true,
        .out = .u8,
    });
    func.setTerminator(entry, .{ .ret = null });

    var low = try verify(std.testing.allocator, &func, .low);
    defer low.deinit();
    try std.testing.expect(low.ok());
}

test "a matmul quant epilogue with a per-column bias of len!=n is reported" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    try func.appendMatmulQuantSpec(entry, a, b, c, 4, 4, 4, .int8, false, .{
        .scale_scalar = 0x3F000000,
        .bias = &.{ 1, -2, 3 }, // len 3, n is 4: mismatch
        .relu = true,
    });
    func.setTerminator(entry, .{ .ret = null });

    var d = try verify(std.testing.allocator, &func, .low);
    defer d.deinit();
    try std.testing.expect(!d.ok());
    try std.testing.expectEqual(Diagnostic{ .operand_type_mismatch = c }, d.items()[0]);
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
