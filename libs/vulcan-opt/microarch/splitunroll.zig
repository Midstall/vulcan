//! Accumulator-splitting unroll for counted reduction loops (the doc's "reduction splitting"). A loop
//! `for (i = i0; i < n; i += step) { acc = acc op f(i); ... }` is latency-bound on `acc`: each
//! iteration's `acc` depends on the previous one. This pass rewrites it into a main loop that runs
//! full groups of K iterations while carrying K INDEPENDENT partial accumulators (each updated by
//! every K-th iteration, so each is its own short dependency chain), a combine block that sums the
//! partials, and the ORIGINAL loop kept as a remainder for the leftover `< K` iterations. The
//! loop-carried dependency drops from K ops to one, which is the whole point; the original loop being
//! reused as the remainder means no loop-closed-SSA surgery and no change to how the accumulator
//! escapes. Integer reductions are always safe (associative); floating-point add/mul reassociation
//! changes rounding, so it is gated on the function's `vulcan.fast_math` attribute.
//!
//! Scope is deliberately the canonical shape: a single-block straight-line body that is the loop's
//! only latch, header params that are exactly the induction variable plus reduction accumulators
//! (nothing else carried), an `i < n` / `i <= n` test with a loop-invariant bound, a constant
//! positive step, and each accumulator used only by its own reduction. Anything else is skipped
//! unchanged. Correctness is proven by the differential JIT oracle in splitunroll_differential.zig.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const loops = @import("../loops.zig");
const unroll = @import("unroll.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Opcode = ir.function.Opcode;
const BinOp = ir.function.BinOp;
const CmpOp = ir.function.CmpOp;

pub const Error = std.mem.Allocator.Error;

const Reduction = struct { index: usize, op: BinOp }; // header-param index of the accumulator, its op

/// A recognized, eligible loop, captured before any mutation (only appends happen afterward, so the
/// handles stay valid).
const Plan = struct {
    header: Block,
    body: Block, // the single-block body, also the latch
    preheader: Block,
    exit_cond: CmpOp, // the header test `icmp(exit_cond, i, bound)`, body taken when true
    bound: Value, // the loop-invariant upper bound `n`
    induction: usize, // header-param index of the induction variable
    step: i64, // its positive constant increment
    reductions: []Reduction, // owned
    factor: u32, // K
};

pub fn run(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) Error!bool {
    if (model.exec == .in_order and model.issue_width <= 1) return false;
    const fast_math = functionHasFastMath(func);

    var info = try loops.analyze(allocator, func);
    defer info.deinit(allocator);
    if (info.loops.len == 0) return false;

    // Snapshot eligible plans before mutating; mutation only appends, so handles stay valid.
    var plans: std.ArrayList(Plan) = .empty;
    defer {
        for (plans.items) |*p| allocator.free(p.reductions);
        plans.deinit(allocator);
    }
    for (info.loops) |*loop| {
        if (try recognize(allocator, func, model, loop, fast_math)) |plan| try plans.append(allocator, plan);
    }
    for (plans.items) |*plan| try apply(allocator, func, plan);
    return plans.items.len != 0;
}

fn functionHasFastMath(func: *const Function) bool {
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan") and std.mem.eql(u8, c.key, "fast_math")) return true,
        else => {},
    };
    return false;
}

fn recognize(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model, loop: *const loops.Loop, fast_math: bool) Error!?Plan {
    const header: Block = @enumFromInt(loop.header);
    const preheader = loop.preheader orelse return null;

    // Innermost only: no other loop header inside this loop's body.
    // (Single-block body below already implies this, but keep the intent explicit via the body check.)

    // The body is the single in-loop block whose terminator jumps back to the header; require it to
    // be the ONLY other in-loop block (single-block straight-line body) with no `if`.
    var body: ?Block = null;
    var in_loop_blocks: usize = 0;
    for (0..func.blockCount()) |bi| {
        const b: Block = @enumFromInt(bi);
        if (@intFromEnum(b) >= loop.body.len or !loop.body[bi]) continue;
        in_loop_blocks += 1;
        if (b == header) continue;
        const term = func.terminator(b) orelse return null;
        switch (term) {
            .jump => |j| if (j.target == header) {
                if (body != null) return null;
                body = b;
            } else return null,
            .ret => return null,
        }
    }
    if (in_loop_blocks != 2) return null; // exactly header + one body block
    const bodyb = body orelse return null;
    for (func.blockInsts(bodyb)) |inst| switch (func.opcode(inst)) {
        .@"if", .matmul => return null, // straight-line body only
        else => {},
    };

    // Header: pure test ending in `if cond -> {body} else {exit}` (or the reverse), cond a comparison.
    var if_inst: ?Inst = null;
    for (func.blockInsts(header)) |inst| switch (func.opcode(inst)) {
        .@"if" => {
            if (if_inst != null) return null;
            if_inst = inst;
        },
        .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .convert, .unary => {},
        else => return null, // impure/memory op in the header
    };
    const iff = if_inst orelse return null;
    const cf = func.opcode(iff).@"if";
    const body_is_then = cf.then.target == bodyb;
    if (!body_is_then and cf.@"else".target != bodyb) return null;
    // The body edge must pass the header params through unchanged (so body param k aliases header
    // param k), which is what lets us reason about the accumulators positionally.
    const body_edge = if (body_is_then) cf.then else cf.@"else";
    const hparams = func.blockParams(header);
    const body_args = func.blockArgs(body_edge);
    if (body_args.len != hparams.len) return null;
    for (body_args, hparams) |arg, hp| if (arg != hp) return null;

    // The condition must be `icmp(op, i, bound)` with `i` a header param and `bound` loop-invariant,
    // and it must gate CONTINUING the loop (true -> body). If body is the else edge, we would need the
    // negated sense; keep it simple and require body on the then edge.
    if (!body_is_then) return null;
    const cond_def = func.definingInst(cf.cond) orelse return null;
    const cmp = switch (func.opcode(cond_def)) {
        .icmp => |c| c,
        else => return null,
    };
    const bparams = func.blockParams(bodyb);
    const induction = paramIndex(hparams, cmp.lhs) orelse return null; // require `i` on the left
    if (definedInLoop(func, loop, cmp.rhs)) return null; // bound must be loop-invariant
    const bound = cmp.rhs;
    if (cmp.op != .lt and cmp.op != .le) return null;

    // The back-edge (body's jump) args, positionally matching the header params.
    const back_args = func.blockArgs(func.terminator(bodyb).?.jump);
    if (back_args.len != hparams.len) return null;

    // The induction: its back-edge value is `i_body + step` (constant step > 0).
    const step = constAddend(func, back_args[induction], bparams[induction]) orelse return null;
    if (step <= 0) return null;

    // Every other header param must be a reduction accumulator used only by its own update.
    const uses = try useCounts(allocator, func);
    defer allocator.free(uses);
    var reductions: std.ArrayList(Reduction) = .empty;
    errdefer reductions.deinit(allocator);
    for (hparams, 0..) |_, k| {
        if (k == induction) continue;
        const op = reductionOp(func, back_args[k], bparams[k], fast_math) orelse {
            reductions.deinit(allocator);
            return null;
        };
        // The accumulator body-alias must be used ONLY by its update (so splitting cannot change a
        // side effect or another value that observed the running total).
        if (uses[@intFromEnum(bparams[k])] != 1) {
            reductions.deinit(allocator);
            return null;
        }
        try reductions.append(allocator, .{ .index = k, .op = op });
    }
    if (reductions.items.len == 0) {
        reductions.deinit(allocator);
        return null; // no accumulator to split
    }

    const factor = unroll.unrollFactor(model, @intCast(func.blockInsts(bodyb).len));
    if (factor < 2) {
        reductions.deinit(allocator);
        return null;
    }

    return Plan{
        .header = header,
        .body = bodyb,
        .preheader = @enumFromInt(preheader),
        .exit_cond = cmp.op,
        .bound = bound,
        .induction = induction,
        .step = step,
        .reductions = try reductions.toOwnedSlice(allocator),
        .factor = factor,
    };
}

fn paramIndex(params: []const Value, v: Value) ?usize {
    for (params, 0..) |p, i| if (p == v) return i;
    return null;
}

/// If `v` is `base + c` (a constant addend on `base`), return `c`. Handles `arith_imm add` and an
/// `arith add` with a constant operand.
fn constAddend(func: *const Function, v: Value, base: Value) ?i64 {
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .arith_imm => |a| if (a.op == .add and a.lhs == base) a.imm else null,
        .arith => |a| if (a.op == .add and a.lhs == base) constOf(func, a.rhs) else if (a.op == .add and a.rhs == base) constOf(func, a.lhs) else null,
        else => null,
    };
}

fn constOf(func: *const Function, v: Value) ?i64 {
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .iconst => |c| c,
        else => null,
    };
}

/// If `v` is `base op x` for an associative, reorderable op (with `base` as one operand), return the
/// op. `x` (the increment) may be anything; associativity makes the split value-preserving.
fn reductionOp(func: *const Function, v: Value, base: Value, fast_math: bool) ?BinOp {
    const di = func.definingInst(v) orelse return null;
    const a = switch (func.opcode(di)) {
        .arith => |ar| ar,
        else => return null,
    };
    if (a.lhs != base and a.rhs != base) return null;
    if (!reorderable(func, v, a.op, fast_math)) return null;
    return a.op;
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

fn definedInLoop(func: *const Function, loop: *const loops.Loop, v: Value) bool {
    const di = func.definingInst(v) orelse {
        // A block parameter: in-loop if its block is in the loop.
        return paramBlockInLoop(func, loop, v);
    };
    // Find the defining instruction's block.
    for (0..func.blockCount()) |bi| {
        if (@intFromEnum(@as(Block, @enumFromInt(bi))) >= loop.body.len) break;
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (inst == di) return bi < loop.body.len and loop.body[bi];
        }
    }
    return false;
}

fn paramBlockInLoop(func: *const Function, loop: *const loops.Loop, v: Value) bool {
    for (0..func.blockCount()) |bi| {
        for (func.blockParams(@enumFromInt(bi))) |p| {
            if (p == v) return bi < loop.body.len and loop.body[bi];
        }
    }
    return false;
}

fn apply(allocator: std.mem.Allocator, func: *Function, plan: *const Plan) Error!void {
    const K = plan.factor;
    const hparams = try allocator.dupe(Value, func.blockParams(plan.header));
    defer allocator.free(hparams);
    const ind_ty = func.valueType(hparams[plan.induction]);

    // New blocks: main header, main body, combine.
    const mainheader = try func.appendBlock();
    const mainbody = try func.appendBlock();
    const combine = try func.appendBlock();

    // Main header params: [mi, then K partials per reduction, in reduction order].
    const mi = try func.appendBlockParam(mainheader, ind_ty);
    // partials[r][m] is the main-header param for reduction r, copy m.
    const partials = try allocator.alloc([]Value, plan.reductions.len);
    defer {
        for (partials) |row| allocator.free(row);
        allocator.free(partials);
    }
    for (plan.reductions, 0..) |red, r| {
        partials[r] = try allocator.alloc(Value, K);
        const ty = func.valueType(hparams[red.index]);
        for (0..K) |m| partials[r][m] = try func.appendBlockParam(mainheader, ty);
    }

    // --- Preheader: redirect its jump from the original header to the main header ---
    // Its current args (init values for i and each accumulator), positionally by header param.
    const pre_jump = func.terminator(plan.preheader).?.jump;
    const init_args = try allocator.dupe(Value, func.blockArgs(pre_jump));
    defer allocator.free(init_args);
    var main_init: std.ArrayList(Value) = .empty;
    defer main_init.deinit(allocator);
    try main_init.append(allocator, init_args[plan.induction]);
    for (plan.reductions, 0..) |red, r| {
        try main_init.append(allocator, init_args[red.index]); // p_r_0 starts at the real init
        const ty = func.valueType(hparams[red.index]);
        for (1..K) |_| {
            const id = try identityConst(func, plan.preheader, red.op, ty);
            try main_init.append(allocator, id); // p_r_1..p_r_{K-1} start at the op identity
        }
        _ = partials[r];
    }
    try func.setJump(plan.preheader, mainheader, main_init.items);

    // --- Main header: run a full group only while the highest index in it still passes the test ---
    // t = mi + step*(K-1); c = icmp(exit_cond, t, bound); if c -> mainbody(...) else combine(...).
    const highest = try func.appendArithImm(mainheader, ind_ty, .add, mi, plan.step * @as(i64, @intCast(K - 1)));
    const bool_t = try func.types.intern(.bool);
    const cont = try func.appendInst(mainheader, bool_t, .{ .icmp = .{ .op = plan.exit_cond, .lhs = highest, .rhs = plan.bound } });
    var carry: std.ArrayList(Value) = .empty; // [mi, all partials] passed to mainbody and combine
    defer carry.deinit(allocator);
    try carry.append(allocator, mi);
    for (partials) |row| try carry.appendSlice(allocator, row);
    try func.appendIf(mainheader, cont, .{ .target = mainbody, .args = carry.items }, .{ .target = combine, .args = carry.items });

    // Main body params mirror the carry: [bmi, then K partials per reduction].
    const bmi = try func.appendBlockParam(mainbody, ind_ty);
    const bpartials = try allocator.alloc([]Value, plan.reductions.len);
    defer {
        for (bpartials) |row| allocator.free(row);
        allocator.free(bpartials);
    }
    for (plan.reductions, 0..) |red, r| {
        bpartials[r] = try allocator.alloc(Value, K);
        const ty = func.valueType(hparams[red.index]);
        for (0..K) |m| bpartials[r][m] = try func.appendBlockParam(mainbody, ty);
    }

    const bparams = try allocator.dupe(Value, func.blockParams(plan.body));
    defer allocator.free(bparams);
    const back_args = try allocator.dupe(Value, func.blockArgs(func.terminator(plan.body).?.jump));
    defer allocator.free(back_args);

    const nexts = try allocator.alloc([]Value, plan.reductions.len); // nexts[r][m] = updated partial
    defer {
        for (nexts) |row| allocator.free(row);
        allocator.free(nexts);
    }
    for (nexts) |*row| row.* = try allocator.alloc(Value, K);

    for (0..K) |m| {
        const idx_m = if (m == 0) bmi else try func.appendArithImm(mainbody, ind_ty, .add, bmi, plan.step * @as(i64, @intCast(m)));
        var vmap: std.AutoHashMapUnmanaged(Value, Value) = .empty;
        defer vmap.deinit(allocator);
        try vmap.put(allocator, bparams[plan.induction], idx_m);
        for (plan.reductions, 0..) |red, r| try vmap.put(allocator, bparams[red.index], bpartials[r][m]);
        try cloneBodyInsts(func, mainbody, plan.body, &vmap, allocator);
        for (plan.reductions, 0..) |red, r| nexts[r][m] = vmap.get(back_args[red.index]).?;
    }
    const nmi = try func.appendArithImm(mainbody, ind_ty, .add, bmi, plan.step * @as(i64, @intCast(K)));
    var back: std.ArrayList(Value) = .empty;
    defer back.deinit(allocator);
    try back.append(allocator, nmi);
    for (nexts) |row| try back.appendSlice(allocator, row);
    try func.setJump(mainbody, mainheader, back.items);

    const cmi = try func.appendBlockParam(combine, ind_ty);
    const cpartials = try allocator.alloc([]Value, plan.reductions.len);
    defer {
        for (cpartials) |row| allocator.free(row);
        allocator.free(cpartials);
    }
    for (plan.reductions, 0..) |red, r| {
        cpartials[r] = try allocator.alloc(Value, K);
        const ty = func.valueType(hparams[red.index]);
        for (0..K) |m| cpartials[r][m] = try func.appendBlockParam(combine, ty);
    }
    const rem_args = try allocator.alloc(Value, hparams.len);
    defer allocator.free(rem_args);
    rem_args[plan.induction] = cmi;
    for (plan.reductions, 0..) |red, r| {
        const ty = func.valueType(hparams[red.index]);
        rem_args[red.index] = try buildTree(func, combine, red.op, ty, cpartials[r], allocator);
    }
    try func.setJump(combine, plan.header, rem_args);
}

/// The identity element of `op` for type `ty`, materialized as a constant in `block`.
fn identityConst(func: *Function, block: Block, op: BinOp, ty: ir.types.Type) Error!Value {
    const is_float = switch (func.types.type_kind(ty)) {
        .float => true,
        else => false,
    };
    return switch (op) {
        .add, .bit_or, .bit_xor => if (is_float) func.appendInst(block, ty, .{ .fconst = 0 }) else func.appendInst(block, ty, .{ .iconst = 0 }),
        .mul => if (is_float) func.appendInst(block, ty, .{ .fconst = 1 }) else func.appendInst(block, ty, .{ .iconst = 1 }),
        .bit_and => func.appendInst(block, ty, .{ .iconst = -1 }), // all ones
        else => unreachable, // recognition only accepts the ops above
    };
}

/// A balanced `op`-tree over `items` (len >= 1), appended to `block`.
fn buildTree(func: *Function, block: Block, op: BinOp, ty: ir.types.Type, items: []const Value, allocator: std.mem.Allocator) Error!Value {
    var cur: std.ArrayList(Value) = .empty;
    defer cur.deinit(allocator);
    try cur.appendSlice(allocator, items);
    while (cur.items.len > 1) {
        var next: std.ArrayList(Value) = .empty;
        var i: usize = 0;
        while (i + 1 < cur.items.len) : (i += 2) {
            const n = try func.appendInst(block, ty, .{ .arith = .{ .op = op, .lhs = cur.items[i], .rhs = cur.items[i + 1] } });
            try next.append(allocator, n);
        }
        if (cur.items.len % 2 == 1) try next.append(allocator, cur.items[cur.items.len - 1]);
        cur.deinit(allocator);
        cur = next;
    }
    return cur.items[0];
}

/// Clone every instruction of `body` (not its terminator) into `dest`, remapping operands through
/// `vmap` and recording each result -> clone. Straight-line opcodes only (recognition guarantees no
/// `if`/`matmul`); memory and call ops are cloned as-is (they run once per original iteration, in
/// order, so their side effects are preserved).
fn cloneBodyInsts(func: *Function, dest: Block, body: Block, vmap: *std.AutoHashMapUnmanaged(Value, Value), allocator: std.mem.Allocator) Error!void {
    const insts = try allocator.dupe(Inst, func.blockInsts(body));
    defer allocator.free(insts);
    for (insts) |inst| {
        const op = try remapOp(func, func.opcode(inst), vmap, allocator);
        if (func.instResult(inst)) |result| {
            const clone = try func.appendInst(dest, func.valueType(result), op);
            try vmap.put(allocator, result, clone);
        } else {
            _ = try func.appendStmtRaw(dest, op);
        }
    }
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
        .dot => |d| .{ .dot = .{ .acc = rv(vmap, d.acc), .a = rv(vmap, d.a), .b = rv(vmap, d.b) } },
        .struct_new => |sn| blk: {
            var fields: std.ArrayList(Value) = .empty;
            defer fields.deinit(allocator);
            for (func.valueList(sn.fields)) |v| try fields.append(allocator, rv(vmap, v));
            break :blk .{ .struct_new = .{ .fields = try func.internValues(fields.items) } };
        },
        .call => |c| blk: {
            var args: std.ArrayList(Value) = .empty;
            defer args.deinit(allocator);
            for (func.valueList(c.args)) |v| try args.append(allocator, rv(vmap, v));
            break :blk .{ .call = .{ .symbol = c.symbol, .args = try func.internValues(args.items) } };
        },
        .call_indirect => |c| blk: {
            var args: std.ArrayList(Value) = .empty;
            defer args.deinit(allocator);
            for (func.valueList(c.args)) |v| try args.append(allocator, rv(vmap, v));
            break :blk .{ .call_indirect = .{ .target = rv(vmap, c.target), .args = try func.internValues(args.items) } };
        },
        .@"if", .matmul => unreachable, // excluded by recognition
    };
}

/// Count how many times each value is used as an operand (instructions, `if` edges, terminators).
fn useCounts(allocator: std.mem.Allocator, func: *const Function) Error![]u32 {
    const counts = try allocator.alloc(u32, func.valueCount());
    @memset(counts, 0);
    const bump = struct {
        fn f(c: []u32, v: Value) void {
            c[@intFromEnum(v)] += 1;
        }
    }.f;
    for (0..func.instCount()) |i| {
        switch (func.opcode(@enumFromInt(i))) {
            .iconst, .fconst, .alloca, .global_addr => {},
            .arith => |x| {
                bump(counts, x.lhs);
                bump(counts, x.rhs);
            },
            .arith_imm => |x| bump(counts, x.lhs),
            .icmp => |x| {
                bump(counts, x.lhs);
                bump(counts, x.rhs);
            },
            .select => |x| {
                bump(counts, x.cond);
                bump(counts, x.then);
                bump(counts, x.@"else");
            },
            .extract => |x| bump(counts, x.aggregate),
            .convert => |x| bump(counts, x.value),
            .unary => |x| bump(counts, x.value),
            .load => |x| bump(counts, x.ptr),
            .store => |x| {
                bump(counts, x.value);
                bump(counts, x.ptr);
            },
            .prefetch => |x| bump(counts, x.ptr),
            .dot => |x| {
                bump(counts, x.acc);
                bump(counts, x.a);
                bump(counts, x.b);
            },
            .matmul => |x| {
                bump(counts, x.a);
                bump(counts, x.b);
                bump(counts, x.c);
            },
            .struct_new => |x| for (func.valueList(x.fields)) |v| bump(counts, v),
            .call => |x| for (func.valueList(x.args)) |v| bump(counts, v),
            .call_indirect => |x| {
                bump(counts, x.target);
                for (func.valueList(x.args)) |v| bump(counts, v);
            },
            .@"if" => |x| {
                bump(counts, x.cond);
                for (func.valueList(x.then.args)) |v| bump(counts, v);
                for (func.valueList(x.@"else".args)) |v| bump(counts, v);
            },
        }
    }
    for (0..func.blockCount()) |bi| {
        if (func.terminator(@enumFromInt(bi))) |term| switch (term) {
            .ret => |v| if (v) |vv| bump(counts, vv),
            .jump => |jm| for (func.blockArgs(jm)) |v| bump(counts, v),
        };
    }
    return counts;
}
