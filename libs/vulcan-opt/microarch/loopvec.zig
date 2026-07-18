//! Structural loop vectorizer. Unrolls a counted loop's main body by the SIMD width V into one block,
//! regenerating each array access as `base_i + m*stride` (the `arith_imm add base, CONST` form the SLP
//! pass recognizes), and keeps the original loop as a scalar remainder. The subsequent SLP pass
//! (`vectorize.runModel`) then fuses the V-wide body into vector loads/ops/stores. So this pass owns
//! only the STRUCTURE and the strided-address analysis; SLP owns the arithmetic widening. See
//! loop-vectorizer-spec in memory.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const loops = @import("../loops.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Opcode = ir.function.Opcode;
const BinOp = ir.function.BinOp;
const CmpOp = ir.function.CmpOp;

pub const Error = std.mem.Allocator.Error;

/// One contiguous (unit-stride) array access in the body: `addr = base + i*stride`, stride == the
/// element's byte size. `addr_inst` is the `base + off` instruction (regenerated per copy);
/// `scale_inst` is the `i*stride` computation (skipped when cloning, since addresses are regenerated).
const ArrayAccess = struct { base: Value, stride: i64, addr_inst: Inst, scale_inst: Inst, is_store: bool };

const Plan = struct {
    header: Block,
    body: Block,
    preheader: Block,
    exit_cond: CmpOp,
    bound: Value,
    induction: usize,
    step: i64,
    accesses: []ArrayAccess, // owned
    factor: u32, // V
};

pub fn run(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) Error!bool {
    var info = try loops.analyze(allocator, func);
    defer info.deinit(allocator);
    if (info.loops.len == 0) return false;
    const fast_math = functionHasFastMath(func);

    var plans: std.ArrayList(Plan) = .empty;
    defer {
        for (plans.items) |*p| allocator.free(p.accesses);
        plans.deinit(allocator);
    }
    var reductions: std.ArrayList(RedPlan) = .empty;
    defer reductions.deinit(allocator);
    for (info.loops) |*loop| {
        if (try recognize(allocator, func, model, loop)) |plan| {
            try plans.append(allocator, plan);
        } else if (recognizeReduction(func, model, loop, fast_math)) |rplan| {
            try reductions.append(allocator, rplan);
        }
    }
    for (plans.items) |*plan| try apply(allocator, func, plan);
    for (reductions.items) |*plan| try applyReduction(allocator, func, plan);
    return plans.items.len != 0 or reductions.items.len != 0;
}

fn functionHasFastMath(func: *const Function) bool {
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan") and std.mem.eql(u8, c.key, "fast_math")) return true,
        else => {},
    };
    return false;
}

fn apply(allocator: std.mem.Allocator, func: *Function, plan: *const Plan) Error!void {
    const V = plan.factor;
    const ind_ty = func.valueType(func.blockParams(plan.header)[plan.induction]);

    const mainheader = try func.appendBlock();
    const mainbody = try func.appendBlock();
    const combine = try func.appendBlock();
    const mi = try func.appendBlockParam(mainheader, ind_ty);

    // Preheader -> main header, carrying the induction's initial value.
    const init_i = func.blockArgs(func.terminator(plan.preheader).?.jump)[plan.induction];
    try func.setJump(plan.preheader, mainheader, &.{init_i});

    // Main header: run a full group only while the highest index in it still passes the test.
    const highest = try func.appendArithImm(mainheader, ind_ty, .add, mi, @as(i64, @intCast(V - 1)));
    const bool_t = try func.types.intern(.bool);
    const cont = try func.appendInst(mainheader, bool_t, .{ .icmp = .{ .op = plan.exit_cond, .lhs = highest, .rhs = plan.bound } });
    try func.appendIf(mainheader, cont, .{ .target = mainbody, .args = &.{mi} }, .{ .target = combine, .args = &.{mi} });

    // Main body: V copies emitted OP-MAJOR (all V copies of each instruction adjacently), which is
    // the run of contiguous same-op instructions the SLP pass fuses into vectors. Per unique address,
    // base_i = base + bmi*stride computed once; copy m reads/writes base_i + m*stride. The induction
    // maps to bmi (it appears only inside addresses, regenerated here); the address/scale/induction-
    // update instructions are skipped when cloning.
    const bmi = try func.appendBlockParam(mainbody, ind_ty);
    const back_args = func.blockArgs(func.terminator(plan.body).?.jump);
    const ind_update = func.definingInst(back_args[plan.induction]).?;
    const base_ty = func.valueType(plan.accesses[0].base);

    // Per-copy value maps, pre-seeded with the induction and every regenerated address. Deduplicate by
    // the address instruction (a load and a store to the same slot share it).
    const vmaps = try allocator.alloc(std.AutoHashMapUnmanaged(Value, Value), V);
    defer {
        for (vmaps) |*vm| vm.deinit(allocator);
        allocator.free(vmaps);
    }
    for (vmaps) |*vm| vm.* = .empty;
    const i_alias = func.blockParams(plan.body)[plan.induction];
    var skip: std.AutoHashMapUnmanaged(Inst, void) = .empty;
    defer skip.deinit(allocator);
    try skip.put(allocator, ind_update, {});
    for (vmaps) |*vm| try vm.put(allocator, i_alias, bmi);
    for (plan.accesses) |acc| {
        try skip.put(allocator, acc.addr_inst, {});
        try skip.put(allocator, acc.scale_inst, {});
        const addr_res = func.instResult(acc.addr_inst).?;
        if (vmaps[0].contains(addr_res)) continue; // already handled via a shared address instruction
        const scaled = try func.appendArithImm(mainbody, base_ty, .mul, bmi, acc.stride);
        const base_i = try func.appendInst(mainbody, base_ty, .{ .arith = .{ .op = .add, .lhs = acc.base, .rhs = scaled } });
        for (vmaps, 0..) |*vm, m| {
            const addr = try func.appendArithImm(mainbody, base_ty, .add, base_i, @as(i64, @intCast(m)) * acc.stride);
            try vm.put(allocator, addr_res, addr);
        }
    }

    // Clone the body op-major: for each instruction, emit its V copies (one per value map) adjacently.
    const body_insts = try allocator.dupe(Inst, func.blockInsts(plan.body));
    defer allocator.free(body_insts);
    for (body_insts) |inst| {
        if (skip.contains(inst)) continue;
        if (func.instResult(inst)) |result| {
            if (vmaps[0].contains(result)) continue; // a pre-seeded address
            for (vmaps) |*vm| {
                const op = try remapOp(func, func.opcode(inst), vm, allocator);
                const clone = try func.appendInst(mainbody, func.valueType(result), op);
                try vm.put(allocator, result, clone);
            }
        } else {
            for (vmaps) |*vm| {
                const op = try remapOp(func, func.opcode(inst), vm, allocator);
                _ = try func.appendStmtRaw(mainbody, op);
            }
        }
    }
    const nmi = try func.appendArithImm(mainbody, ind_ty, .add, bmi, @as(i64, @intCast(V)));
    try func.setJump(mainbody, mainheader, &.{nmi});

    // Combine: nothing to reduce for a map loop; fall into the ORIGINAL loop (remainder) at the current
    // index. The original loop's header params were exactly [induction].
    const cmi = try func.appendBlockParam(combine, ind_ty);
    try func.setJump(combine, plan.header, &.{cmi});
}

fn rv(vmap: *const std.AutoHashMapUnmanaged(Value, Value), v: Value) Value {
    return vmap.get(v) orelse v;
}

fn remapOp(func: *Function, op: Opcode, vmap: *const std.AutoHashMapUnmanaged(Value, Value), allocator: std.mem.Allocator) Error!Opcode {
    return switch (op) {
        .iconst, .fconst, .alloca, .global_addr => op,
        .arith => |a| .{ .arith = .{ .op = a.op, .lhs = rv(vmap, a.lhs), .rhs = rv(vmap, a.rhs) } },
        .arith_imm => |a| .{ .arith_imm = .{ .op = a.op, .lhs = rv(vmap, a.lhs), .imm = a.imm } },
        .icmp => |c| .{ .icmp = .{ .op = c.op, .lhs = rv(vmap, c.lhs), .rhs = rv(vmap, c.rhs) } },
        .select => |s| .{ .select = .{ .cond = rv(vmap, s.cond), .then = rv(vmap, s.then), .@"else" = rv(vmap, s.@"else") } },
        .convert => |c| .{ .convert = .{ .value = rv(vmap, c.value) } },
        .unary => |u| .{ .unary = .{ .op = u.op, .value = rv(vmap, u.value) } },
        .extract => |e| .{ .extract = .{ .aggregate = rv(vmap, e.aggregate), .index = e.index } },
        .load => |l| .{ .load = .{ .ptr = rv(vmap, l.ptr) } },
        .store => |s| .{ .store = .{ .value = rv(vmap, s.value), .ptr = rv(vmap, s.ptr) } },
        .prefetch => |p| .{ .prefetch = .{ .ptr = rv(vmap, p.ptr) } },
        .struct_new => |sn| blk: {
            var fields: std.ArrayList(Value) = .empty;
            defer fields.deinit(allocator);
            for (func.valueList(sn.fields)) |v| try fields.append(allocator, rv(vmap, v));
            break :blk .{ .struct_new = .{ .fields = try func.internValues(fields.items) } };
        },
        else => unreachable, // recognition excludes call/dot/matmul/if
    };
}

/// SLP fuses 32-bit elements, so loopvec targets the same width and gates on the same vector feature.
const ELEM_BITS: u16 = 32;
const MAX_LANES: u32 = 8;

fn recognize(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model, loop: *const loops.Loop) Error!?Plan {
    const vec_ok = switch (model.features) {
        .aarch64 => |f| f.neon,
        .riscv64 => |f| f.v or f.vpu,
        .x86_64 => false,
    };
    if (!vec_ok or model.vector_bits < 2 * ELEM_BITS) return null;
    const V: u32 = @min(model.vector_bits / ELEM_BITS, MAX_LANES);
    if (V < 2) return null;

    const header: Block = @enumFromInt(loop.header);
    const preheader = loop.preheader orelse return null;

    // Single-block straight-line body that is the only latch, no `if`/`matmul`.
    var body: ?Block = null;
    var in_loop_blocks: usize = 0;
    for (0..func.blockCount()) |bi| {
        if (bi >= loop.body.len or !loop.body[bi]) continue;
        in_loop_blocks += 1;
        const b: Block = @enumFromInt(bi);
        if (b == header) continue;
        switch (func.terminator(b) orelse return null) {
            .jump => |j| if (j.target == header) {
                if (body != null) return null;
                body = b;
            } else return null,
            .ret => return null,
        }
    }
    if (in_loop_blocks != 2) return null;
    const bodyb = body orelse return null;

    // Header: pure test ending in an `if`, body on the then edge, passing the header params through.
    var if_inst: ?Inst = null;
    for (func.blockInsts(header)) |inst| switch (func.opcode(inst)) {
        .@"if" => {
            if (if_inst != null) return null;
            if_inst = inst;
        },
        .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .convert, .unary => {},
        else => return null,
    };
    const cf = func.opcode(if_inst orelse return null).@"if";
    if (cf.then.target != bodyb) return null;
    const hparams = func.blockParams(header);
    const body_args = func.blockArgs(cf.then);
    if (body_args.len != hparams.len) return null;
    for (body_args, hparams) |arg, hp| if (arg != hp) return null;

    // Phase 1: only the induction is carried (no reductions).
    if (hparams.len != 1) return null;
    const cmp = switch (func.opcode(func.definingInst(cf.cond) orelse return null)) {
        .icmp => |c| c,
        else => return null,
    };
    if (cmp.op != .lt and cmp.op != .le) return null;
    const induction = paramIndex(hparams, cmp.lhs) orelse return null;
    if (definedInLoop(func, loop, cmp.rhs)) return null;
    const bound = cmp.rhs;

    const bparams = func.blockParams(bodyb);
    const i_alias = bparams[induction];
    const back_args = func.blockArgs(func.terminator(bodyb).?.jump);
    const step = constAddend(func, back_args[induction], i_alias) orelse return null;
    if (step != 1) return null; // unit induction step: consecutive iterations touch consecutive elements
    const ind_update = func.definingInst(back_args[induction]).?;

    // Every load/store address must be unit-stride contiguous; other ops must be pure compute.
    var accesses: std.ArrayList(ArrayAccess) = .empty;
    errdefer accesses.deinit(allocator);
    var scale_insts: std.ArrayList(Inst) = .empty;
    defer scale_insts.deinit(allocator);
    for (func.blockInsts(bodyb)) |inst| {
        switch (func.opcode(inst)) {
            .load => |l| {
                const sa = stridedAddress(func, l.ptr, i_alias) orelse return bail(&accesses, allocator);
                if (sa.stride != byteSize(func, func.valueType(func.instResult(inst).?))) return bail(&accesses, allocator);
                try accesses.append(allocator, .{ .base = sa.base, .stride = sa.stride, .addr_inst = sa.addr_inst, .scale_inst = sa.scale_inst, .is_store = false });
                try scale_insts.append(allocator, sa.scale_inst);
            },
            .store => |s| {
                const sa = stridedAddress(func, s.ptr, i_alias) orelse return bail(&accesses, allocator);
                if (sa.stride != byteSize(func, func.valueType(s.value))) return bail(&accesses, allocator);
                try accesses.append(allocator, .{ .base = sa.base, .stride = sa.stride, .addr_inst = sa.addr_inst, .scale_inst = sa.scale_inst, .is_store = true });
                try scale_insts.append(allocator, sa.scale_inst);
            },
            .arith, .arith_imm, .iconst, .fconst, .icmp, .select, .convert, .unary => {},
            else => return bail(&accesses, allocator), // call/alloca/dot/matmul/etc: not handled
        }
    }
    if (accesses.items.len == 0) return bail(&accesses, allocator);

    // The induction alias must be used ONLY inside the address scales and the induction update, so the
    // main body can regenerate addresses and map the induction to the group base without changing a
    // compute that observed the raw index.
    if (!inductionOnlyInAddresses(func, bodyb, i_alias, scale_insts.items, ind_update)) return bail(&accesses, allocator);

    return Plan{
        .header = header,
        .body = bodyb,
        .preheader = @enumFromInt(preheader),
        .exit_cond = cmp.op,
        .bound = bound,
        .induction = induction,
        .step = step,
        .accesses = try accesses.toOwnedSlice(allocator),
        .factor = V,
    };
}

fn bail(accesses: *std.ArrayList(ArrayAccess), allocator: std.mem.Allocator) ?Plan {
    accesses.deinit(allocator);
    return null;
}

fn paramIndex(params: []const Value, v: Value) ?usize {
    for (params, 0..) |p, i| if (p == v) return i;
    return null;
}

const Strided = struct { base: Value, stride: i64, addr_inst: Inst, scale_inst: Inst };

/// Decompose `addr` as `base + i_alias*stride` (base loop-invariant), or null.
fn stridedAddress(func: *const Function, addr: Value, i_alias: Value) ?Strided {
    const addr_inst = func.definingInst(addr) orelse return null;
    const add = switch (func.opcode(addr_inst)) {
        .arith => |a| if (a.op == .add) a else return null,
        else => return null,
    };
    if (scaleOf(func, add.rhs, i_alias)) |sc| return .{ .base = add.lhs, .stride = sc.stride, .addr_inst = addr_inst, .scale_inst = sc.inst };
    if (scaleOf(func, add.lhs, i_alias)) |sc| return .{ .base = add.rhs, .stride = sc.stride, .addr_inst = addr_inst, .scale_inst = sc.inst };
    return null;
}

/// Recognize `v == i_alias * C` (via `mul` or `shl`), returning the constant stride and the inst.
fn scaleOf(func: *const Function, v: Value, i_alias: Value) ?struct { stride: i64, inst: Inst } {
    const di = func.definingInst(v) orelse return null;
    switch (func.opcode(di)) {
        .arith_imm => |a| {
            if (a.lhs != i_alias) return null;
            return switch (a.op) {
                .mul => .{ .stride = a.imm, .inst = di },
                .shl => .{ .stride = @as(i64, 1) << @intCast(a.imm), .inst = di },
                else => null,
            };
        },
        .arith => |a| {
            if (a.op == .mul) {
                if (a.lhs == i_alias) {
                    if (constOf(func, a.rhs)) |c| return .{ .stride = c, .inst = di };
                } else if (a.rhs == i_alias) {
                    if (constOf(func, a.lhs)) |c| return .{ .stride = c, .inst = di };
                }
            } else if (a.op == .shl and a.lhs == i_alias) {
                if (constOf(func, a.rhs)) |c| return .{ .stride = @as(i64, 1) << @intCast(c), .inst = di };
            }
            return null;
        },
        else => return null,
    }
}

fn constOf(func: *const Function, v: Value) ?i64 {
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .iconst => |c| c,
        else => null,
    };
}

fn constAddend(func: *const Function, v: Value, base: Value) ?i64 {
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .arith_imm => |a| if (a.op == .add and a.lhs == base) a.imm else null,
        .arith => |a| if (a.op == .add and a.lhs == base) constOf(func, a.rhs) else if (a.op == .add and a.rhs == base) constOf(func, a.lhs) else null,
        else => null,
    };
}

/// The element byte size for a scalar type (unit stride equals this).
fn byteSize(func: *const Function, ty: ir.types.Type) i64 {
    return switch (func.types.type_kind(ty)) {
        .int => |i| @intCast((i.bits + 7) / 8),
        .float => |f| switch (f) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
        },
        .ptr => 8,
        else => 0,
    };
}

/// True when `i_alias` is used only by the given address-scale instructions and the induction update
/// (so no compute or memory value observes the raw index, which the main body would compute wrong).
fn inductionOnlyInAddresses(func: *const Function, body: Block, i_alias: Value, scales: []const Inst, ind_update: Inst) bool {
    for (func.blockInsts(body)) |inst| {
        if (usesValue(func, inst, i_alias)) {
            if (inst == ind_update) continue;
            var is_scale = false;
            for (scales) |s| if (s == inst) {
                is_scale = true;
            };
            if (!is_scale) return false;
        }
    }
    // The back-edge/terminator may reference i only through the induction update (already an inst use).
    return true;
}

fn usesValue(func: *const Function, inst: Inst, v: Value) bool {
    return switch (func.opcode(inst)) {
        .arith => |a| a.lhs == v or a.rhs == v,
        .arith_imm => |a| a.lhs == v,
        .icmp => |c| c.lhs == v or c.rhs == v,
        .select => |s| s.cond == v or s.then == v or s.@"else" == v,
        .convert => |c| c.value == v,
        .unary => |u| u.value == v,
        .load => |l| l.ptr == v,
        .store => |s| s.value == v or s.ptr == v,
        else => false,
    };
}

fn definedInLoop(func: *const Function, loop: *const loops.Loop, v: Value) bool {
    if (func.definingInst(v)) |di| {
        for (0..func.blockCount()) |bi| {
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (inst == di) return bi < loop.body.len and loop.body[bi];
            }
        }
        return false;
    }
    for (0..func.blockCount()) |bi| {
        for (func.blockParams(@enumFromInt(bi))) |p| {
            if (p == v) return bi < loop.body.len and loop.body[bi];
        }
    }
    return false;
}

/// A recognized `s op= a[i]` reduction over contiguous memory, vectorizable to a vector accumulator.
const RedPlan = struct {
    header: Block,
    preheader: Block,
    exit_cond: CmpOp,
    bound: Value,
    induction: usize,
    accumulator: usize,
    op: BinOp,
    elem_ty: ir.types.Type, // the reduction/element type (32-bit)
    load_base: Value, // the contiguous load's base pointer
    stride: i64, // element byte size (unit stride)
    factor: u32, // V
};

fn recognizeReduction(func: *const Function, model: *const mm.Model, loop: *const loops.Loop, fast_math: bool) ?RedPlan {
    const vec_ok = switch (model.features) {
        .aarch64 => |f| f.neon,
        .riscv64 => |f| f.v or f.vpu,
        .x86_64 => false,
    };
    if (!vec_ok or model.vector_bits < 2 * ELEM_BITS) return null;
    const V: u32 = @min(model.vector_bits / ELEM_BITS, MAX_LANES);
    if (V < 2) return null;

    const header: Block = @enumFromInt(loop.header);
    const preheader = loop.preheader orelse return null;

    // Single-block straight-line body, only latch, no if/matmul.
    var body: ?Block = null;
    var in_loop_blocks: usize = 0;
    for (0..func.blockCount()) |bi| {
        if (bi >= loop.body.len or !loop.body[bi]) continue;
        in_loop_blocks += 1;
        const b: Block = @enumFromInt(bi);
        if (b == header) continue;
        switch (func.terminator(b) orelse return null) {
            .jump => |j| if (j.target == header) {
                if (body != null) return null;
                body = b;
            } else return null,
            .ret => return null,
        }
    }
    if (in_loop_blocks != 2) return null;
    const bodyb = body orelse return null;
    for (func.blockInsts(bodyb)) |inst| switch (func.opcode(inst)) {
        .@"if", .matmul => return null,
        else => {},
    };

    // Header: pure test, body on then, params passed through.
    var if_inst: ?Inst = null;
    for (func.blockInsts(header)) |inst| switch (func.opcode(inst)) {
        .@"if" => {
            if (if_inst != null) return null;
            if_inst = inst;
        },
        .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .convert, .unary => {},
        else => return null,
    };
    const cf = func.opcode(if_inst orelse return null).@"if";
    if (cf.then.target != bodyb) return null;
    const hparams = func.blockParams(header);
    if (hparams.len != 2) return null; // exactly [induction, accumulator]
    const body_args = func.blockArgs(cf.then);
    if (body_args.len != hparams.len) return null;
    for (body_args, hparams) |arg, hp| if (arg != hp) return null;

    const cmp = switch (func.opcode(func.definingInst(cf.cond) orelse return null)) {
        .icmp => |c| c,
        else => return null,
    };
    if (cmp.op != .lt and cmp.op != .le) return null;
    const induction = paramIndex(hparams, cmp.lhs) orelse return null;
    if (definedInLoop(func, loop, cmp.rhs)) return null;
    const accumulator = 1 - induction; // the other of the two params
    const bparams = func.blockParams(bodyb);
    const back_args = func.blockArgs(func.terminator(bodyb).?.jump);

    // Induction steps by 1.
    if ((constAddend(func, back_args[induction], bparams[induction]) orelse return null) != 1) return null;

    // The accumulator's next value is `acc_body op load(a[i])`, op associative & reorderable, and the
    // increment is a contiguous unit-stride load of a 32-bit element.
    const acc_body = bparams[accumulator];
    const upd = func.definingInst(back_args[accumulator]) orelse return null;
    const ar = switch (func.opcode(upd)) {
        .arith => |a| a,
        else => return null,
    };
    const incr = if (ar.lhs == acc_body) ar.rhs else if (ar.rhs == acc_body) ar.lhs else return null;
    if (!reorderable(func, back_args[accumulator], ar.op, fast_math)) return null;
    const load_inst = func.definingInst(incr) orelse return null;
    const load = switch (func.opcode(load_inst)) {
        .load => |l| l,
        else => return null,
    };
    const elem_ty = func.valueType(incr);
    if (byteSize(func, elem_ty) * 8 != ELEM_BITS) return null; // 32-bit element for the vector path
    const sa = stridedAddress(func, load.ptr, bparams[induction]) orelse return null;
    if (sa.stride != byteSize(func, elem_ty)) return null;
    // The accumulator body-alias is used only by its update.
    if (countUses(func, bodyb, acc_body) != 1) return null;

    return RedPlan{
        .header = header,
        .preheader = @enumFromInt(preheader),
        .exit_cond = cmp.op,
        .bound = cmp.rhs,
        .induction = induction,
        .accumulator = accumulator,
        .op = ar.op,
        .elem_ty = elem_ty,
        .load_base = sa.base,
        .stride = sa.stride,
        .factor = V,
    };
}

fn countUses(func: *const Function, block: Block, v: Value) usize {
    var c: usize = 0;
    for (func.blockInsts(block)) |inst| {
        if (usesValue(func, inst, v)) c += 1;
    }
    if (func.terminator(block)) |t| switch (t) {
        .jump => |j| for (func.blockArgs(j)) |a| {
            if (a == v) c += 1;
        },
        .ret => |x| if (x) |xx| {
            if (xx == v) c += 1;
        },
    };
    return c;
}

fn reorderable(func: *const Function, v: Value, op: BinOp, fast_math: bool) bool {
    const is_float = switch (func.types.type_kind(func.valueType(v))) {
        .float => true,
        else => false,
    };
    return switch (op) {
        .add, .mul => if (is_float) fast_math else true,
        .bit_and, .bit_or, .bit_xor => !is_float,
        .sub, .div, .rem, .shl, .shr, .mulh => false,
    };
}

fn identityConst(func: *Function, block: Block, op: BinOp, ty: ir.types.Type) Error!Value {
    const is_float = switch (func.types.type_kind(ty)) {
        .float => true,
        else => false,
    };
    return switch (op) {
        .add, .bit_or, .bit_xor => if (is_float) func.appendInst(block, ty, .{ .fconst = 0 }) else func.appendInst(block, ty, .{ .iconst = 0 }),
        .mul => if (is_float) func.appendInst(block, ty, .{ .fconst = 1 }) else func.appendInst(block, ty, .{ .iconst = 1 }),
        .bit_and => func.appendInst(block, ty, .{ .iconst = -1 }),
        else => unreachable,
    };
}

fn applyReduction(allocator: std.mem.Allocator, func: *Function, plan: *const RedPlan) Error!void {
    const V = plan.factor;
    const hparams = try allocator.dupe(Value, func.blockParams(plan.header));
    defer allocator.free(hparams);
    const ind_ty = func.valueType(hparams[plan.induction]);
    const base_ty = func.valueType(plan.load_base);
    const vt = try func.types.intern(.{ .vector = .{ .len = @intCast(V), .elem = plan.elem_ty } });

    const mainheader = try func.appendBlock();
    const mainbody = try func.appendBlock();
    const combine = try func.appendBlock();

    // Preheader: build a splat-identity vector accumulator, redirect into the main header.
    const pre_args = try allocator.dupe(Value, func.blockArgs(func.terminator(plan.preheader).?.jump));
    defer allocator.free(pre_args);
    const init_i = pre_args[plan.induction];
    const init_acc = pre_args[plan.accumulator];
    const ident = try identityConst(func, plan.preheader, plan.op, plan.elem_ty);
    var idlanes: std.ArrayList(Value) = .empty;
    defer idlanes.deinit(allocator);
    for (0..V) |_| try idlanes.append(allocator, ident);
    const vacc0 = try func.appendInst(plan.preheader, vt, .{ .struct_new = .{ .fields = try func.internValues(idlanes.items) } });
    try func.setJump(plan.preheader, mainheader, &.{ init_i, vacc0 });

    // Main header: [mi, vacc]. Run a full V-group while the highest index passes the test.
    const mi = try func.appendBlockParam(mainheader, ind_ty);
    const vacc = try func.appendBlockParam(mainheader, vt);
    const highest = try func.appendArithImm(mainheader, ind_ty, .add, mi, @as(i64, @intCast(V - 1)));
    const bool_t = try func.types.intern(.bool);
    const cont = try func.appendInst(mainheader, bool_t, .{ .icmp = .{ .op = plan.exit_cond, .lhs = highest, .rhs = plan.bound } });
    try func.appendIf(mainheader, cont, .{ .target = mainbody, .args = &.{ mi, vacc } }, .{ .target = combine, .args = &.{ mi, vacc } });

    // Main body: wide-load V contiguous elements at base + mi*stride, add into the vector accumulator.
    const bmi = try func.appendBlockParam(mainbody, ind_ty);
    const bvacc = try func.appendBlockParam(mainbody, vt);
    const scaled = try func.appendArithImm(mainbody, base_ty, .mul, bmi, plan.stride);
    const base_i = try func.appendInst(mainbody, base_ty, .{ .arith = .{ .op = .add, .lhs = plan.load_base, .rhs = scaled } });
    const wv = try func.appendInst(mainbody, vt, .{ .load = .{ .ptr = base_i } });
    const nvacc = try func.appendInst(mainbody, vt, .{ .arith = .{ .op = plan.op, .lhs = bvacc, .rhs = wv } });
    const nmi = try func.appendArithImm(mainbody, ind_ty, .add, bmi, @as(i64, @intCast(V)));
    try func.setJump(mainbody, mainheader, &.{ nmi, nvacc });

    // Combine: horizontally reduce the vector accumulator (extract each lane, tree-sum), fold in the
    // original init, then fall into the ORIGINAL loop (remainder) for the leftover scalar iterations.
    const cmi = try func.appendBlockParam(combine, ind_ty);
    const cvacc = try func.appendBlockParam(combine, vt);
    var lanes: std.ArrayList(Value) = .empty;
    defer lanes.deinit(allocator);
    for (0..V) |k| try lanes.append(allocator, try func.appendInst(combine, plan.elem_ty, .{ .extract = .{ .aggregate = cvacc, .index = @intCast(k) } }));
    const hsum = try buildTree(func, combine, plan.op, plan.elem_ty, lanes.items, allocator);
    const sacc = try func.appendInst(combine, plan.elem_ty, .{ .arith = .{ .op = plan.op, .lhs = init_acc, .rhs = hsum } });
    const rem_args = try allocator.alloc(Value, hparams.len);
    defer allocator.free(rem_args);
    rem_args[plan.induction] = cmi;
    rem_args[plan.accumulator] = sacc;
    try func.setJump(combine, plan.header, rem_args);
}

fn buildTree(func: *Function, block: Block, op: BinOp, ty: ir.types.Type, items: []const Value, allocator: std.mem.Allocator) Error!Value {
    var cur: std.ArrayList(Value) = .empty;
    defer cur.deinit(allocator);
    try cur.appendSlice(allocator, items);
    while (cur.items.len > 1) {
        var next: std.ArrayList(Value) = .empty;
        var i: usize = 0;
        while (i + 1 < cur.items.len) : (i += 2) {
            try next.append(allocator, try func.appendInst(block, ty, .{ .arith = .{ .op = op, .lhs = cur.items[i], .rhs = cur.items[i + 1] } }));
        }
        if (cur.items.len % 2 == 1) try next.append(allocator, cur.items[cur.items.len - 1]);
        cur.deinit(allocator);
        cur = next;
    }
    return cur.items[0];
}

const testing = std.testing;
const registry = @import("registry.zig");

/// `for (i = 0; i < n; i += 1) y[i] = a*x[i] + y[i];` over f32 arrays. Induction only used in the two
/// contiguous addresses; the classic saxpy the vectorizer should recognize.
fn buildSaxpy(func: *Function) !void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const x = try func.appendBlockParam(entry, ptr_t);
    const y = try func.appendBlockParam(entry, ptr_t);
    const a = try func.appendBlockParam(entry, f32_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const i = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4); // i*4 (f32 elem bytes)
    const xaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = off } });
    const xv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = xaddr } });
    const yaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = off } });
    const yv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = yaddr } });
    const ax = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = xv } });
    const res = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .add, .lhs = ax, .rhs = yv } });
    try func.appendStore(body, res, yaddr);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ni});
    func.setTerminator(done, .{ .ret = null });
}

test "recognizes a saxpy map loop: two contiguous loads, one contiguous store, V lanes" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSaxpy(&func);

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    const plan = (try recognize(allocator, &func, registry.modelFor(.@"ampere-altra"), &info.loops[0])) orelse return error.NotRecognized;
    defer allocator.free(plan.accesses);

    try testing.expectEqual(@as(u32, 4), plan.factor); // 128-bit / 32-bit f32 = 4 lanes
    var loads: usize = 0;
    var stores: usize = 0;
    for (plan.accesses) |acc| {
        if (acc.is_store) stores += 1 else loads += 1;
        try testing.expectEqual(@as(i64, 4), acc.stride); // unit stride for f32
    }
    try testing.expectEqual(@as(usize, 2), loads);
    try testing.expectEqual(@as(usize, 1), stores);
}

test "declines a non-unit-stride access (a[2*i])" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const arr = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const i = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 8); // 2*i for a 4-byte element: stride 8 != 4
    const addr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = arr, .rhs = off } });
    _ = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = addr } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ni});
    func.setTerminator(done, .{ .ret = null });

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    try testing.expect((try recognize(allocator, &func, registry.modelFor(.@"ampere-altra"), &info.loops[0])) == null);
}
