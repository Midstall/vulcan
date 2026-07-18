//! Model-driven INT8 dot-product vectorization (Altra SDOT/UDOT). Recognizes a scalar INT8
//! multiply-accumulate reduction loop, `acc += convert_i32(a[i]) * convert_i32(b[i])` over unit-stride
//! int8 (or u8) arrays, and rewrites it to process 16 elements per iteration with the `dot` op (which
//! lowers to SDOT for signed i8, UDOT for unsigned u8 on aarch64+dotprod). A lane-reduction seeds the
//! partial vector sum back into the ORIGINAL scalar loop, which is reused unchanged as the remainder
//! (handling the trailing `n mod 16` elements). SKIP-IF-UNSURE: only a loop proven to match the exact
//! shape is transformed; anything else is left scalar, which is always correct. Identical integer
//! results to the scalar loop are proven by the differential JIT oracle in
//! libs/vulcan-target/tests/dotprod_differential.zig.
//!
//! Addressing is verify-clean: the scalar loop carries its two element pointers as loop values and
//! advances them by one byte each iteration (arith_imm on a ptr, which verify permits, unlike a
//! runtime `arith add(ptr, i32)`). The vector loop carries the same two pointers and advances them by
//! 16 bytes per iteration, so at the remainder entry they already point at element `16*floor(n/16)`.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const loops = @import("../loops.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Type = ir.types.Type;

pub const Error = std.mem.Allocator.Error;

/// A vetted, eligible dot-reduction loop plus everything the transform needs. Snapshotted before any
/// mutation (we only ever append blocks/values and rewrite the preheader's jump, so these handles
/// stay valid across the transform of other loops).
const Plan = struct {
    /// The original scalar loop header (H) and its preheader (P). H is reused as the remainder loop.
    header: Block,
    preheader: Block,
    /// Which of the four header params is the induction variable, the accumulator, and the two element
    /// pointers. The four indices are a permutation of {0,1,2,3}.
    ivi: usize,
    acci: usize,
    pax: usize,
    pby: usize,
    /// The loop-invariant bound `n` (the icmp rhs), the initial pointer bases and initial accumulator
    /// (the preheader's edge args). All dominate the inserted blocks.
    n: Value,
    base_a: Value,
    base_b: Value,
    acc0: Value,
    /// The int8 element signedness: signed -> SDOT, unsigned -> UDOT.
    sign: std.builtin.Signedness,
};

/// Recognize and vectorize every INT8 dot-reduction loop `func` contains, when `model` is
/// aarch64+dotprod (the only target the `dot` op lowers on). Returns whether anything changed. Gathers
/// all eligible loops before mutating (mutation invalidates the loop analysis; only appends and a
/// preheader-jump rewrite happen, so the snapshot's handles stay valid), mirroring unroll.zig.
pub fn run(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) Error!bool {
    // The dot op only lowers on aarch64 with the dotprod feature; never emit it elsewhere.
    if (model.arch != .aarch64) return false;
    if (!model.features.aarch64.dotprod) return false;

    var info = try loops.analyze(allocator, func);
    defer info.deinit(allocator);

    // Value -> defining block index, for loop-invariance checks during recognition.
    const def_block = try computeDefBlocks(allocator, func);
    defer allocator.free(def_block);

    var plans: std.ArrayList(Plan) = .empty;
    defer plans.deinit(allocator);

    for (info.loops) |*loop| {
        if (try recognize(allocator, func, loop, def_block)) |plan| {
            try plans.append(allocator, plan);
        }
    }

    for (plans.items) |*plan| try apply(func, plan);
    return plans.items.len != 0;
}

/// The block index that defines each value (a block param's block, or the block holding the defining
/// instruction). Indexed by `@intFromEnum(value)`.
fn computeDefBlocks(allocator: std.mem.Allocator, func: *const Function) Error![]u32 {
    const def_block = try allocator.alloc(u32, func.valueCount());
    errdefer allocator.free(def_block);
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| def_block[@intFromEnum(p)] = @intCast(bi);
        for (func.blockInsts(block)) |inst| {
            if (func.instResult(inst)) |r| def_block[@intFromEnum(r)] = @intCast(bi);
        }
    }
    return def_block;
}

/// Whether block index `idx` is inside the loop described by `body` (its membership bitset).
fn inLoop(body: []const bool, idx: u32) bool {
    return idx < body.len and body[idx];
}

/// Step A: prove `loop` is the exact INT8 dot-reduction shape and gather a Plan, or return null to
/// skip it. The recognized shape (verify-clean, unit-stride, advancing pointers):
///
///   H(i, acc, pa, pb): if i < n { B(i, acc, pa, pb) } else exit(acc)   // 4 params, pure test header
///   B(bi, bacc, bpa, bpb):
///     la = load i8/u8, bpa; lb = load i8/u8, bpb   (both same 8-bit signedness)
///     ca = convert i32, la; cb = convert i32, lb
///     prod = ca * cb;  nacc = bacc + prod
///     ni = bi + 1;  npa = bpa + 1;  npb = bpb + 1  (unit stride)
///     jump H(ni, nacc, npa, npb)
///
/// Any deviation (different stride, non-i8 loads, mixed signedness, extra body ops, multiple
/// accumulators, wrong param count, non-zero initial index, ...) returns null.
fn recognize(allocator: std.mem.Allocator, func: *Function, loop: *const loops.Loop, def_block: []const u32) Error!?Plan {
    if (loop.preheader == null) return null;
    const header: Block = @enumFromInt(loop.header);
    const body_bits = loop.body;
    const h_idx: u32 = loop.header;

    // Exactly one body block (this also rejects nested loops and multi-block bodies).
    var body_opt: ?Block = null;
    for (0..func.blockCount()) |bi| {
        if (!inLoop(body_bits, @intCast(bi)) or bi == h_idx) continue;
        if (body_opt != null) return null;
        body_opt = @enumFromInt(bi);
    }
    const body = body_opt orelse return null;
    const b_idx: u32 = @intFromEnum(body);

    // Pure test header: exactly [icmp, if], the `if` testing the icmp, no explicit branching
    // terminator. Matches the loop-header idiom unroll.zig also requires.
    const h_insts = func.blockInsts(header);
    if (h_insts.len != 2) return null;
    if (func.opcode(h_insts[0]) != .icmp) return null;
    if (func.opcode(h_insts[1]) != .@"if") return null;
    const cmp = func.opcode(h_insts[0]).icmp;
    const iff = func.opcode(h_insts[1]).@"if";
    if (iff.cond != func.instResult(h_insts[0]).?) return null;
    if (cmp.op != .lt) return null; // canonical `i < n`
    if (func.terminator(header)) |t| switch (t) {
        .ret => |v| if (v != null) return null,
        .jump => return null,
    };

    // Exactly four header params. Identify the induction variable (the icmp lhs) and the bound n.
    const hparams = func.blockParams(header);
    if (hparams.len != 4) return null;
    var ivi: ?usize = null;
    for (hparams, 0..) |p, idx| {
        if (p == cmp.lhs) ivi = idx;
    }
    const iv_i = ivi orelse return null;
    const n = cmp.rhs;
    if (inLoop(body_bits, def_block[@intFromEnum(n)])) return null; // n must be loop-invariant
    if (!isI32(func, hparams[iv_i])) return null;

    // The header's `if` has exactly one in-loop edge (to the body) and one exit edge; the in-loop edge
    // passes the header params straight through, so body param k corresponds to header param k.
    const then_in = inLoop(body_bits, @intFromEnum(iff.then.target));
    const else_in = inLoop(body_bits, @intFromEnum(iff.@"else".target));
    const in_edge = if (then_in and !else_in) iff.then else if (else_in and !then_in) iff.@"else" else return null;
    if (in_edge.target != body) return null;
    const in_args = func.blockArgs(in_edge);
    if (in_args.len != 4) return null;
    for (in_args, hparams) |arg, hp| {
        if (arg != hp) return null;
    }

    const bparams = func.blockParams(body);
    if (bparams.len != 4) return null;

    // The latch: the body's terminator jumps back to the header, one back-edge arg per header param.
    const back_args = switch (func.terminator(body) orelse return null) {
        .jump => |j| blk: {
            if (j.target != header) return null;
            break :blk func.blockArgs(j);
        },
        .ret => return null,
    };
    if (back_args.len != 4) return null;

    // --- Match the body dataflow backward from the accumulator update. ---
    // The accumulator: the header param whose back-edge value is `arith add(bacc, prod)`. Scan all four
    // to find the single one matching that shape.
    var acci: ?usize = null;
    var prod: Value = undefined;
    for (0..4) |k| {
        const upd = back_args[k];
        if (def_block[@intFromEnum(upd)] != b_idx) continue;
        const di = func.definingInst(upd) orelse continue;
        const op = func.opcode(di);
        if (op != .arith) continue;
        const a = op.arith;
        if (a.op != .add) continue;
        const bacc = bparams[k];
        if (a.lhs == bacc) {
            if (acci != null) return null; // more than one accumulator-shaped update
            acci = k;
            prod = a.rhs;
        } else if (a.rhs == bacc) {
            if (acci != null) return null;
            acci = k;
            prod = a.lhs;
        }
    }
    const acc_i = acci orelse return null;
    if (acc_i == iv_i) return null;
    if (!isI32(func, hparams[acc_i])) return null;
    const nacc_inst = func.definingInst(back_args[acc_i]).?;

    // prod = ca * cb (both defined in the body).
    if (def_block[@intFromEnum(prod)] != b_idx) return null;
    const prod_inst = func.definingInst(prod) orelse return null;
    const mulop = switch (func.opcode(prod_inst)) {
        .arith => |a| a,
        else => return null,
    };
    if (mulop.op != .mul) return null;

    // ca = convert i32, la ; cb = convert i32, lb.
    const la = try convertSource(func, def_block, b_idx, mulop.lhs) orelse return null;
    const lb = try convertSource(func, def_block, b_idx, mulop.rhs) orelse return null;
    const ca_inst = func.definingInst(mulop.lhs).?;
    const cb_inst = func.definingInst(mulop.rhs).?;

    // la = load(bp_x), lb = load(bp_y): 8-bit loads from two distinct body pointer params, same
    // signedness. bp_x and bp_y identify the pointer param indices.
    if (def_block[@intFromEnum(la)] != b_idx or def_block[@intFromEnum(lb)] != b_idx) return null;
    const la_inst = func.definingInst(la) orelse return null;
    const lb_inst = func.definingInst(lb) orelse return null;
    const bp_x = switch (func.opcode(la_inst)) {
        .load => |l| l.ptr,
        else => return null,
    };
    const bp_y = switch (func.opcode(lb_inst)) {
        .load => |l| l.ptr,
        else => return null,
    };
    const sign_a = int8Sign(func, func.valueType(la)) orelse return null;
    const sign_b = int8Sign(func, func.valueType(lb)) orelse return null;
    if (sign_a != sign_b) return null; // mixed signedness is not a single SDOT/UDOT

    // The load pointers must be two distinct body params (not the same reused pointer).
    const pax = paramIndex(bparams, bp_x) orelse return null;
    const pby = paramIndex(bparams, bp_y) orelse return null;
    if (pax == pby) return null;
    if (pax == iv_i or pax == acc_i or pby == iv_i or pby == acc_i) return null;
    if (!isPtr(func, hparams[pax]) or !isPtr(func, hparams[pby])) return null;

    // Unit stride: iv, pa, pb each advance by exactly 1 via arith_imm add.
    const ni_inst = try stepByOne(func, def_block, b_idx, back_args[iv_i], bparams[iv_i]) orelse return null;
    const npa_inst = try stepByOne(func, def_block, b_idx, back_args[pax], bparams[pax]) orelse return null;
    const npb_inst = try stepByOne(func, def_block, b_idx, back_args[pby], bparams[pby]) orelse return null;

    // Exactly these nine instructions form the body, nothing else (no extra ops, no second reduction,
    // no side-effecting ops beyond the two loads).
    const nine = [_]Inst{ la_inst, lb_inst, ca_inst, cb_inst, prod_inst, nacc_inst, ni_inst, npa_inst, npb_inst };
    var seen: std.AutoHashMapUnmanaged(Inst, void) = .empty;
    defer seen.deinit(allocator);
    for (nine) |i| try seen.put(allocator, i, {});
    if (seen.count() != nine.len) return null; // some instruction coincided (e.g. a[i]*a[i]); skip
    if (func.blockInsts(body).len != nine.len) return null;

    // The preheader supplies the initial index (must be 0), accumulator, and the two pointer bases.
    const preheader: Block = @enumFromInt(loop.preheader.?);
    const p_args = switch (func.terminator(preheader) orelse return null) {
        .jump => |j| blk: {
            if (j.target != header) return null;
            break :blk func.blockArgs(j);
        },
        .ret => return null,
    };
    if (p_args.len != 4) return null;
    if (!isIconstZero(func, p_args[iv_i])) return null; // the induction variable must start at 0

    return Plan{
        .header = header,
        .preheader = preheader,
        .ivi = iv_i,
        .acci = acc_i,
        .pax = pax,
        .pby = pby,
        .n = n,
        .base_a = p_args[pax],
        .base_b = p_args[pby],
        .acc0 = p_args[acc_i],
        .sign = sign_a,
    };
}

/// If `v` is a `convert` to i32 defined in block `b_idx`, return its source value; else null.
fn convertSource(func: *const Function, def_block: []const u32, b_idx: u32, v: Value) Error!?Value {
    if (def_block[@intFromEnum(v)] != b_idx) return null;
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .convert => |c| if (isI32(func, v)) c.value else null,
        else => null,
    };
}

/// If `upd` is `arith_imm add(base, 1)` defined in block `b_idx`, return its instruction; else null.
fn stepByOne(func: *const Function, def_block: []const u32, b_idx: u32, upd: Value, base: Value) Error!?Inst {
    if (def_block[@intFromEnum(upd)] != b_idx) return null;
    const di = func.definingInst(upd) orelse return null;
    return switch (func.opcode(di)) {
        .arith_imm => |a| if (a.op == .add and a.lhs == base and a.imm == 1) di else null,
        else => null,
    };
}

/// The index of `v` within `params`, or null if absent.
fn paramIndex(params: []const Value, v: Value) ?usize {
    for (params, 0..) |p, i| {
        if (p == v) return i;
    }
    return null;
}

fn isI32(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.signedness == .signed and i.bits == 32,
        else => false,
    };
}

fn isPtr(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .ptr;
}

/// The signedness of an 8-bit integer type, or null if `ty` is not an 8-bit integer.
fn int8Sign(func: *const Function, ty: Type) ?std.builtin.Signedness {
    return switch (func.types.type_kind(ty)) {
        .int => |i| if (i.bits == 8) i.signedness else null,
        else => null,
    };
}

fn isIconstZero(func: *const Function, v: Value) bool {
    const di = func.definingInst(v) orelse return false;
    return switch (func.opcode(di)) {
        .iconst => |c| c == 0,
        else => false,
    };
}

/// Step B: build the vector loop, lane reduction, and remainder wiring for one recognized loop. The
/// original scalar loop (header/body/exit) is left untouched and reused as the remainder; only the
/// preheader's jump is redirected into the new vector prologue.
fn apply(func: *Function, plan: *const Plan) Error!void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = plan.sign, .bits = 8 } });
    const v16 = try func.types.intern(.{ .vector = .{ .len = 16, .elem = i8_t } });

    // The four new blocks. Appending never disturbs the entry (block 0) or the reused loop.
    const seed = try func.appendBlock();
    const vh = try func.appendBlock();
    const vb = try func.appendBlock();
    const red = try func.appendBlock();

    // Vector-header carries (vi, vacc, pa, pb); the remainder needs the advanced pointers too.
    const vi = try func.appendBlockParam(vh, i32_t);
    const vacc = try func.appendBlockParam(vh, v4);
    const vpa = try func.appendBlockParam(vh, ptr_t);
    const vpb = try func.appendBlockParam(vh, ptr_t);

    const bvi = try func.appendBlockParam(vb, i32_t);
    const bvacc = try func.appendBlockParam(vb, v4);
    const bvpa = try func.appendBlockParam(vb, ptr_t);
    const bvpb = try func.appendBlockParam(vb, ptr_t);

    const rvi = try func.appendBlockParam(red, i32_t);
    const rvacc = try func.appendBlockParam(red, v4);
    const rvpa = try func.appendBlockParam(red, ptr_t);
    const rvpb = try func.appendBlockParam(red, ptr_t);

    // --- Seed block: materialize a zero <4 x i32> and enter the vector loop with i=0 and the bases. ---
    // A vector zero constant does not lower directly, so build it via an alloca zeroed with scalar
    // stores, then a vector load (the same store/reload the reduction uses in reverse).
    const zero = try func.appendInst(seed, i32_t, .{ .iconst = 0 });
    const zslot = try func.appendInst(seed, ptr_t, .{ .alloca = .{ .elem = v4 } });
    try func.appendStore(seed, zero, zslot);
    const z4 = try func.appendArithImm(seed, ptr_t, .add, zslot, 4);
    try func.appendStore(seed, zero, z4);
    const z8 = try func.appendArithImm(seed, ptr_t, .add, zslot, 8);
    try func.appendStore(seed, zero, z8);
    const z12 = try func.appendArithImm(seed, ptr_t, .add, zslot, 12);
    try func.appendStore(seed, zero, z12);
    const vzero = try func.appendInst(seed, v4, .{ .load = .{ .ptr = zslot } });
    const vi0 = try func.appendInst(seed, i32_t, .{ .iconst = 0 });
    try func.setJump(seed, vh, &.{ vi0, vzero, plan.base_a, plan.base_b });

    // --- Vector header: while i + 16 <= n, run the vector body; else fall through to the reduction. ---
    const vend = try func.appendArithImm(vh, i32_t, .add, vi, 16);
    const vcmp = try func.appendInst(vh, bool_t, .{ .icmp = .{ .op = .le, .lhs = vend, .rhs = plan.n } });
    try func.appendIf(
        vh,
        vcmp,
        .{ .target = vb, .args = &.{ vi, vacc, vpa, vpb } },
        .{ .target = red, .args = &.{ vi, vacc, vpa, vpb } },
    );

    // --- Vector body: load 16 int8 lanes from each pointer, dot-accumulate, advance i/pointers by 16. ---
    const va = try func.appendInst(vb, v16, .{ .load = .{ .ptr = bvpa } });
    const vbv = try func.appendInst(vb, v16, .{ .load = .{ .ptr = bvpb } });
    const ndot = try func.appendDot(vb, bvacc, va, vbv);
    const nvi = try func.appendArithImm(vb, i32_t, .add, bvi, 16);
    const nvpa = try func.appendArithImm(vb, ptr_t, .add, bvpa, 16);
    const nvpb = try func.appendArithImm(vb, ptr_t, .add, bvpb, 16);
    try func.setJump(vb, vh, &.{ nvi, ndot, nvpa, nvpb });

    // --- Reduction: sum vacc's four i32 lanes and seed the scalar accumulator (acc0 + lane sum). ---
    const rslot = try func.appendInst(red, ptr_t, .{ .alloca = .{ .elem = v4 } });
    try func.appendStore(red, rvacc, rslot);
    const l0 = try func.appendInst(red, i32_t, .{ .load = .{ .ptr = rslot } });
    const r4 = try func.appendArithImm(red, ptr_t, .add, rslot, 4);
    const l1 = try func.appendInst(red, i32_t, .{ .load = .{ .ptr = r4 } });
    const r8 = try func.appendArithImm(red, ptr_t, .add, rslot, 8);
    const l2 = try func.appendInst(red, i32_t, .{ .load = .{ .ptr = r8 } });
    const r12 = try func.appendArithImm(red, ptr_t, .add, rslot, 12);
    const l3 = try func.appendInst(red, i32_t, .{ .load = .{ .ptr = r12 } });
    const s01 = try func.appendInst(red, i32_t, .{ .arith = .{ .op = .add, .lhs = l0, .rhs = l1 } });
    const s23 = try func.appendInst(red, i32_t, .{ .arith = .{ .op = .add, .lhs = l2, .rhs = l3 } });
    const sred = try func.appendInst(red, i32_t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = s23 } });
    const seed_acc = try func.appendInst(red, i32_t, .{ .arith = .{ .op = .add, .lhs = plan.acc0, .rhs = sred } });

    // Enter the original scalar loop with the leftover index, seeded accumulator, and advanced
    // pointers, threaded back into the header's original parameter positions (loop-closed SSA: the
    // header's `acc` param now carries the vector partial sum, so its exit read is the full result).
    var hargs: [4]Value = undefined;
    hargs[plan.ivi] = rvi;
    hargs[plan.acci] = seed_acc;
    hargs[plan.pax] = rvpa;
    hargs[plan.pby] = rvpb;
    try func.setJump(red, plan.header, &hargs);

    // Redirect the preheader into the vector prologue. Its former index/accumulator inits become dead;
    // the pointer bases and acc0 it defined stay live (used by the seed and reduction blocks).
    try func.setJump(plan.preheader, seed, &.{});
}

const registry = @import("registry.zig");

/// Build the canonical INT8 dot-reduction `fn(a: ptr, b: ptr, n: i32) i32` returning
/// `sum_{i<n} convert(a[i]) * convert(b[i])`, with the given element signedness and pointer stride
/// (a stride other than 1, or an 8/32-bit element override, produces a deliberately non-matching loop).
const LoopSpec = struct {
    sign: std.builtin.Signedness = .signed,
    stride: i64 = 1,
    elem_bits: u16 = 8,
    mixed_sign: bool = false, // make b's element the opposite signedness of a's
    start: i64 = 0, // the induction variable's initial value (must be 0 to match)
};

fn buildDotLoop(func: *Function, spec: LoopSpec) Error!void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ea = try func.types.intern(.{ .int = .{ .signedness = spec.sign, .bits = spec.elem_bits } });
    const eb_sign: std.builtin.Signedness = if (spec.mixed_sign) (if (spec.sign == .signed) .unsigned else .signed) else spec.sign;
    const eb = try func.types.intern(.{ .int = .{ .signedness = eb_sign, .bits = spec.elem_bits } });

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const a_ptr = try func.appendBlockParam(entry, ptr_t);
    const b_ptr = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);

    const i = try func.appendBlockParam(header, i32_t);
    const acc = try func.appendBlockParam(header, i32_t);
    const pa = try func.appendBlockParam(header, ptr_t);
    const pb = try func.appendBlockParam(header, ptr_t);

    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const bpa = try func.appendBlockParam(body, ptr_t);
    const bpb = try func.appendBlockParam(body, ptr_t);

    const iv0 = try func.appendInst(entry, i32_t, .{ .iconst = spec.start });
    const acc0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ iv0, acc0, a_ptr, b_ptr });

    const cmp = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(header, cmp, .{ .target = body, .args = &.{ i, acc, pa, pb } }, .{ .target = exit, .args = &.{} });

    const la = try func.appendInst(body, ea, .{ .load = .{ .ptr = bpa } });
    const lb = try func.appendInst(body, eb, .{ .load = .{ .ptr = bpb } });
    const ca = try func.appendInst(body, i32_t, .{ .convert = .{ .value = la } });
    const cb = try func.appendInst(body, i32_t, .{ .convert = .{ .value = lb } });
    const prod = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = ca, .rhs = cb } });
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = prod } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    const npa = try func.appendArithImm(body, ptr_t, .add, bpa, spec.stride);
    const npb = try func.appendArithImm(body, ptr_t, .add, bpb, spec.stride);
    try func.setJump(body, header, &.{ ni, nacc, npa, npb });

    func.setTerminator(exit, .{ .ret = acc });
}

fn countDots(func: *const Function) usize {
    var count: usize = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .dot) count += 1;
        }
    }
    return count;
}

test "recognizes a signed int8 dot loop, inserts a dot op, stays verifiable" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildDotLoop(&func, .{ .sign = .signed });

    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(usize, 1), countDots(&func));

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "recognizes an unsigned u8 dot loop and stays verifiable" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildDotLoop(&func, .{ .sign = .unsigned });

    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(usize, 1), countDots(&func));

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "skips a non-unit-stride loop" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildDotLoop(&func, .{ .stride = 2 });
    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(!changed);
    try std.testing.expectEqual(@as(usize, 0), countDots(&func));
}

test "skips a loop whose loads are not int8" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildDotLoop(&func, .{ .elem_bits = 32 });
    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(!changed);
}

test "skips a loop with mixed load signedness" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildDotLoop(&func, .{ .sign = .signed, .mixed_sign = true });
    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(!changed);
}

test "skips every loop on a non-aarch64 model" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildDotLoop(&func, .{ .sign = .signed });
    const changed = try run(allocator, &func, registry.modelFor(.@"et-soc"));
    try std.testing.expect(!changed);
    try std.testing.expectEqual(@as(usize, 0), countDots(&func));
}

/// Asserts `run` rejects `func` (`changed == false`) and leaves its block/instruction counts
/// exactly as they were. Shared by every false-accept regression test below.
fn expectRejectedUnchanged(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) !void {
    const blocks_before = func.blockCount();
    const insts_before = func.instCount();
    const changed = try run(allocator, func, model);
    try std.testing.expect(!changed);
    try std.testing.expectEqual(blocks_before, func.blockCount());
    try std.testing.expectEqual(insts_before, func.instCount());
}

/// The canonical dot loop plus one extra, unrelated arith instruction in the body (`junk = bi +
/// bi`, unused by anything), so the body is ten instructions rather than the exact
/// nine-instruction shape `recognize` requires.
fn buildDotLoopExtraOp(func: *Function) Error!void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const a_ptr = try func.appendBlockParam(entry, ptr_t);
    const b_ptr = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);

    const i = try func.appendBlockParam(header, i32_t);
    const acc = try func.appendBlockParam(header, i32_t);
    const pa = try func.appendBlockParam(header, ptr_t);
    const pb = try func.appendBlockParam(header, ptr_t);

    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const bpa = try func.appendBlockParam(body, ptr_t);
    const bpb = try func.appendBlockParam(body, ptr_t);

    const iv0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const acc0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ iv0, acc0, a_ptr, b_ptr });

    const cmp = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(header, cmp, .{ .target = body, .args = &.{ i, acc, pa, pb } }, .{ .target = exit, .args = &.{} });

    const la = try func.appendInst(body, i8_t, .{ .load = .{ .ptr = bpa } });
    const lb = try func.appendInst(body, i8_t, .{ .load = .{ .ptr = bpb } });
    const ca = try func.appendInst(body, i32_t, .{ .convert = .{ .value = la } });
    const cb = try func.appendInst(body, i32_t, .{ .convert = .{ .value = lb } });
    const prod = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = ca, .rhs = cb } });
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = prod } });
    // The extra op: unrelated to the reduction, not part of the recognized nine-instruction shape.
    _ = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bi, .rhs = bi } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    const npa = try func.appendArithImm(body, ptr_t, .add, bpa, 1);
    const npb = try func.appendArithImm(body, ptr_t, .add, bpb, 1);
    try func.setJump(body, header, &.{ ni, nacc, npa, npb });

    func.setTerminator(exit, .{ .ret = acc });
}

test "skips a loop whose body has an extra unrelated instruction" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildDotLoopExtraOp(&func);
    try expectRejectedUnchanged(allocator, &func, registry.modelFor(.@"ampere-altra"));
}

/// Otherwise-canonical dot loop where `pb` is ALSO stepped by `arith add(bpb, nacc)` instead of the
/// canonical `arith_imm add(bpb, 1)` unit stride: two of the four back-edge args (`acc` and `pb`)
/// now match the "accumulator-shaped update" pattern the recognizer looks for.
fn buildMultiAccDotLoop(func: *Function) Error!void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const a_ptr = try func.appendBlockParam(entry, ptr_t);
    const b_ptr = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);

    const i = try func.appendBlockParam(header, i32_t);
    const acc = try func.appendBlockParam(header, i32_t);
    const pa = try func.appendBlockParam(header, ptr_t);
    const pb = try func.appendBlockParam(header, ptr_t);

    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const bpa = try func.appendBlockParam(body, ptr_t);
    const bpb = try func.appendBlockParam(body, ptr_t);

    const iv0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const acc0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ iv0, acc0, a_ptr, b_ptr });

    const cmp = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(header, cmp, .{ .target = body, .args = &.{ i, acc, pa, pb } }, .{ .target = exit, .args = &.{} });

    const la = try func.appendInst(body, i8_t, .{ .load = .{ .ptr = bpa } });
    const lb = try func.appendInst(body, i8_t, .{ .load = .{ .ptr = bpb } });
    const ca = try func.appendInst(body, i32_t, .{ .convert = .{ .value = la } });
    const cb = try func.appendInst(body, i32_t, .{ .convert = .{ .value = lb } });
    const prod = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = ca, .rhs = cb } });
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = prod } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    const npa = try func.appendArithImm(body, ptr_t, .add, bpa, 1);
    // The second accumulator-shaped update: `bpb + nacc`, not the canonical unit-stride step.
    const npb = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = bpb, .rhs = nacc } });
    try func.setJump(body, header, &.{ ni, nacc, npa, npb });

    func.setTerminator(exit, .{ .ret = acc });
}

test "skips a loop with two accumulator-shaped updates in the latch args" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMultiAccDotLoop(&func);
    try expectRejectedUnchanged(allocator, &func, registry.modelFor(.@"ampere-altra"));
}

/// A dot-shaped reduction whose body spans TWO blocks (`body1` then `body2`, forwarding straight
/// through to the header), so the loop is not the single-body-block shape `recognize` requires.
fn buildNestedBodyDotLoop(func: *Function) Error!void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body1 = try func.appendBlock();
    const body2 = try func.appendBlock();
    const exit = try func.appendBlock();

    const a_ptr = try func.appendBlockParam(entry, ptr_t);
    const b_ptr = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);

    const i = try func.appendBlockParam(header, i32_t);
    const acc = try func.appendBlockParam(header, i32_t);
    const pa = try func.appendBlockParam(header, ptr_t);
    const pb = try func.appendBlockParam(header, ptr_t);

    const bi = try func.appendBlockParam(body1, i32_t);
    const bacc = try func.appendBlockParam(body1, i32_t);
    const bpa = try func.appendBlockParam(body1, ptr_t);
    const bpb = try func.appendBlockParam(body1, ptr_t);

    const ci = try func.appendBlockParam(body2, i32_t);
    const cacc = try func.appendBlockParam(body2, i32_t);
    const cpa = try func.appendBlockParam(body2, ptr_t);
    const cpb = try func.appendBlockParam(body2, ptr_t);

    const iv0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const acc0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ iv0, acc0, a_ptr, b_ptr });

    const cmp = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(header, cmp, .{ .target = body1, .args = &.{ i, acc, pa, pb } }, .{ .target = exit, .args = &.{} });

    const la = try func.appendInst(body1, i8_t, .{ .load = .{ .ptr = bpa } });
    const lb = try func.appendInst(body1, i8_t, .{ .load = .{ .ptr = bpb } });
    const ca = try func.appendInst(body1, i32_t, .{ .convert = .{ .value = la } });
    const cb = try func.appendInst(body1, i32_t, .{ .convert = .{ .value = lb } });
    const prod = try func.appendInst(body1, i32_t, .{ .arith = .{ .op = .mul, .lhs = ca, .rhs = cb } });
    const nacc = try func.appendInst(body1, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = prod } });
    const ni = try func.appendArithImm(body1, i32_t, .add, bi, 1);
    const npa = try func.appendArithImm(body1, ptr_t, .add, bpa, 1);
    const npb = try func.appendArithImm(body1, ptr_t, .add, bpb, 1);
    try func.setJump(body1, body2, &.{ ni, nacc, npa, npb });

    // body2 does nothing but forward straight back to the header: the body is still one
    // reduction's worth of work, just split across two blocks.
    try func.setJump(body2, header, &.{ ci, cacc, cpa, cpb });

    func.setTerminator(exit, .{ .ret = acc });
}

test "skips a loop whose body spans more than one block" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildNestedBodyDotLoop(&func);
    try expectRejectedUnchanged(allocator, &func, registry.modelFor(.@"ampere-altra"));
}

test "skips a loop whose induction variable starts at a non-zero value" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildDotLoop(&func, .{ .sign = .signed, .start = 5 });
    try expectRejectedUnchanged(allocator, &func, registry.modelFor(.@"ampere-altra"));
}

/// A reduction loop shaped like a plain array sum, with only THREE header/body params (`i`, `acc`,
/// `pa`), not the four the dot-reduction shape requires.
fn buildWrongParamCountLoop(func: *Function) Error!void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const a_ptr = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);

    const i = try func.appendBlockParam(header, i32_t);
    const acc = try func.appendBlockParam(header, i32_t);
    const pa = try func.appendBlockParam(header, ptr_t);

    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const bpa = try func.appendBlockParam(body, ptr_t);

    const iv0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const acc0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ iv0, acc0, a_ptr });

    const cmp = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(header, cmp, .{ .target = body, .args = &.{ i, acc, pa } }, .{ .target = exit, .args = &.{} });

    const la = try func.appendInst(body, i8_t, .{ .load = .{ .ptr = bpa } });
    const ca = try func.appendInst(body, i32_t, .{ .convert = .{ .value = la } });
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = ca } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    const npa = try func.appendArithImm(body, ptr_t, .add, bpa, 1);
    try func.setJump(body, header, &.{ ni, nacc, npa });

    func.setTerminator(exit, .{ .ret = acc });
}

test "skips a loop whose header has the wrong param count" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildWrongParamCountLoop(&func);
    try expectRejectedUnchanged(allocator, &func, registry.modelFor(.@"ampere-altra"));
}

/// A dot-shaped loop where BOTH multiply operands load from the same `pa` pointer param (`pb` is
/// carried and stepped but never read): the aliased-base-pointer case, `sum(a[i] * a[i])`.
fn buildAliasedDotLoop(func: *Function) Error!void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const a_ptr = try func.appendBlockParam(entry, ptr_t);
    const b_ptr = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);

    const i = try func.appendBlockParam(header, i32_t);
    const acc = try func.appendBlockParam(header, i32_t);
    const pa = try func.appendBlockParam(header, ptr_t);
    const pb = try func.appendBlockParam(header, ptr_t);

    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const bpa = try func.appendBlockParam(body, ptr_t);
    const bpb = try func.appendBlockParam(body, ptr_t);

    const iv0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const acc0 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ iv0, acc0, a_ptr, b_ptr });

    const cmp = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(header, cmp, .{ .target = body, .args = &.{ i, acc, pa, pb } }, .{ .target = exit, .args = &.{} });

    // Both loads read from `bpa`; `bpb` is carried and stepped but never dereferenced.
    const la = try func.appendInst(body, i8_t, .{ .load = .{ .ptr = bpa } });
    const lb = try func.appendInst(body, i8_t, .{ .load = .{ .ptr = bpa } });
    const ca = try func.appendInst(body, i32_t, .{ .convert = .{ .value = la } });
    const cb = try func.appendInst(body, i32_t, .{ .convert = .{ .value = lb } });
    const prod = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = ca, .rhs = cb } });
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = prod } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    const npa = try func.appendArithImm(body, ptr_t, .add, bpa, 1);
    const npb = try func.appendArithImm(body, ptr_t, .add, bpb, 1);
    try func.setJump(body, header, &.{ ni, nacc, npa, npb });

    func.setTerminator(exit, .{ .ret = acc });
}

test "skips a loop whose two loads alias the same base pointer" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildAliasedDotLoop(&func);
    try expectRejectedUnchanged(allocator, &func, registry.modelFor(.@"ampere-altra"));
}

test "skips an otherwise-matching loop on an aarch64 model with dotprod disabled" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildDotLoop(&func, .{ .sign = .signed });

    // Same aarch64 model set as every accepting test above, with only the dotprod feature bit
    // flipped off: the `dot` op has nowhere to lower to, so `run` must refuse even a perfectly
    // matching loop.
    var no_dotprod = registry.modelFor(.@"ampere-altra").*;
    try std.testing.expect(no_dotprod.features.aarch64.dotprod); // sanity: altra ships it on
    no_dotprod.features.aarch64.dotprod = false;
    try expectRejectedUnchanged(allocator, &func, &no_dotprod);
}
