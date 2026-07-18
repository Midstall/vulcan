//! SLP auto-vectorizer: fuses runs of `lanes` parallel scalar f32 arith ops into one vector
//! arith (the shape the GLSL/SPIR-V frontends emit when scalarizing a vecN op). Chain reuse
//! keeps intermediates in vector registers via a scalar-to-lane map, so a chain like (a+b)*c
//! does not re-pack between groups. Per block, contiguous same-op runs only. lanes is 4 for
//! NEON/SSE/RVV, 8 for AVX. `run` defaults to 4.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("microarch/model.zig");
const cost = @import("microarch/cost.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const BinOp = ir.function.BinOp;

pub const Error = std.mem.Allocator.Error;

/// The widest lane count any target asks for (AVX `<8 x f32>`), for fixed-size scratch buffers.
const MAX_LANES = 8;

/// Which lane of which vector a scalar value holds (recorded on extract).
const LaneOf = struct { vec: Value, lane: u8 };
const VMap = std.AutoHashMapUnmanaged(Value, LaneOf);

/// Bytes stepped per lane in a contiguous memory run. The vectorizer only fuses 32-bit elements
/// (f32 or i32), so a run of adjacent element loads/stores steps by 4 bytes per lane.
const ELEM_BYTES: i64 = ELEM_BITS / 8;

/// The set of scalar `load` results this run replaced with wide vector loads (coalescing). A load in
/// this set whose result ends up unused (its only consumers were the fused-away scalar arith) is
/// removed by `cleanup`. Loads are memory ops that a general DCE keeps, so coalesced-away loads are
/// tracked explicitly here rather than being left to fall to DCE.
const CoalescedLoads = std.AutoHashMapUnmanaged(Value, void);

/// A base pointer and byte offset an address value decomposes to.
const AddrParts = struct { base: Value, off: i64 };

/// A recognized run of `lanes` contiguous ascending scalar loads feeding one operand vector: lane k
/// is a `load` from `ptr0`'s base + off0 + k*ELEM_BYTES. `ptr0` is lane 0's own address value (the
/// run's base address), reused directly as the wide vector load's pointer. `loads[k]` is lane k's
/// load instruction, whose result is recorded for removal once dead.
const LoadRun = struct {
    ptr0: Value,
    loads: [MAX_LANES]Inst,
};

/// Vectorize every eligible group in `func` at the NEON/SSE/RVV width (4). Returns true if
/// anything changed. This model-agnostic entry fuses only scalar-f32 groups (see `runModel` for
/// the model-gated integer path): with no model in hand there is no target that can lower an
/// `<N x i32>`, so producing one would be unsound.
pub fn run(allocator: std.mem.Allocator, func: *Function) Error!bool {
    return runLanesGated(allocator, func, 4, false, null);
}

/// Vectorize `func` fusing runs of `lanes` scalars (4 for 128-bit SIMD, 8 for AVX). Returns
/// true if anything changed. f32-only, same reasoning as `run`.
pub fn runLanes(allocator: std.mem.Allocator, func: *Function, lanes: u8) Error!bool {
    return runLanesGated(allocator, func, lanes, false, null);
}

/// The shared driver. `allow_i32` opens the model-gated integer path: when true, contiguous
/// scalar-i32 arith runs whose shared BinOp the packed-integer (pi) backend can lower are ALSO
/// fused into `<lanes x i32>` ops. When false, only scalar-f32 runs are fused (the f32 path is
/// unchanged for every caller). `runModel` sets it only for a riscv64 vpu (et-soc) model.
///
/// `model` gates PROFITABILITY: when non-null, each structurally-eligible group is fused only if
/// `cost.slpProfitable` judges the vector form cheaper than the scalar form for that model (so a
/// cheap op on a wide OoO core is left scalar). When null (the `run`/`runLanes` correctness path,
/// which has no model in hand) every eligible group is force-fused, so pure-correctness callers keep
/// vectorizing regardless of profitability.
fn runLanesGated(allocator: std.mem.Allocator, func: *Function, lanes: u8, allow_i32: bool, model: ?*const mm.Model) Error!bool {
    std.debug.assert(lanes >= 2 and lanes <= MAX_LANES);
    var vmap: VMap = .empty;
    defer vmap.deinit(allocator);
    // Scalar loads replaced by wide vector loads during operand coalescing, removed once dead.
    var coalesced_loads: CoalescedLoads = .empty;
    defer coalesced_loads.deinit(allocator);
    var changed = false;
    // Whether any load- or store-coalescing fired. Only then does the block need the cleanup pass
    // (dead address arithmetic, dead extracts/packs, coalesced-away loads). With no coalescing the
    // IR is left exactly as the pre-coalescing vectorizer produced it, so every non-memory caller is
    // byte-for-byte unaffected.
    var coalesced = false;
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        const block: Block = @enumFromInt(bi);
        while (try vectorizeOne(allocator, func, block, &vmap, lanes, allow_i32, model, &coalesced_loads, &coalesced)) changed = true;
        // Store coalescing is a per-block scan run after this block's arith groups are fused, so the
        // stored values are the fused groups' result lanes (or hand-built extracts). It rewrites each
        // run of `lanes` contiguous scalar stores of one vector's lanes into a single wide store.
        if (try coalesceStores(allocator, func, block, &vmap, lanes)) {
            changed = true;
            coalesced = true;
        }
    }
    if (coalesced) try cleanup(allocator, func, &coalesced_loads);
    return changed;
}

/// The f32 element width this vectorizer fuses. The lane count is derived from the model's vector
/// register width divided by this.
const ELEM_BITS: u16 = 32;

/// Vectorize `func` at the width the microarchitecture supports: model.vector_bits / 32 lanes of
/// f32, capped at MAX_LANES. Vectorizing requires BOTH a vector width of at least two lanes AND
/// an ISA vector feature flag for the model's arch (aarch64 `neon`, riscv64 `v` OR `vpu`). A scalar
/// model (vector_bits 0), one too narrow for two lanes, or one with no vector feature bit set
/// vectorizes nothing and returns false. ET-SOC has a custom, non-RVV 8-lane f32 SIMD unit (the
/// CORE-ET Erbium packed-single VPU): its `v` bit is off (it is not RVV) but its `vpu` bit is on, so
/// it vectorizes here same as any RVV or NEON part, at its model.vector_bits/32 = 8 lanes. The
/// riscv64 backend reads the same `vpu` capability to lower those 8-lane vector ops to VPU
/// instructions instead of RVV (see riscv64/isel.zig's `vpu` parameter).
pub fn runModel(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) Error!bool {
    const vec_ok = switch (model.features) {
        .aarch64 => |f| f.neon,
        .riscv64 => |f| f.v or f.vpu,
        .x86_64 => false,
    };
    if (!vec_ok) return false;
    if (model.vector_bits < 2 * ELEM_BITS) return false;
    const lanes_full = model.vector_bits / ELEM_BITS;
    const lanes: u8 = @intCast(@min(lanes_full, @as(u16, MAX_LANES)));
    // The integer (`<N x i32>`) path is produced ONLY for a model whose backend can lower it: the
    // CORE-ET packed-integer (pi) ops, which the riscv64 backend selects for a vpu (et-soc) model
    // (see riscv64/isel.zig). RVV (`v`) and NEON have no `<8 x i32>` lowering here, so they keep
    // the f32-only path: an i32 vector there would reach isel as error.Unsupported. Same lane count
    // as f32 (vector_bits/32 = 8 on et-soc), since the pi element is also 32 bits wide.
    const allow_i32 = model.vpu();
    return runLanesGated(allocator, func, lanes, allow_i32, model);
}

fn isScalarF32(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f32,
        else => false,
    };
}

/// A scalar 32-bit integer of either signedness (the pi element width). The vector's element type
/// is interned from the operands' own result type, so the group's signedness is preserved (it
/// drives the backend's logical-vs-arithmetic right-shift choice, `fsrl.pi` vs `fsra.pi`).
fn isScalarI32(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.bits == ELEM_BITS,
        else => false,
    };
}

/// Whether the packed-integer (pi) backend can lower a vector arith with this BinOp. The pi unit
/// has add/sub/mul, the three bitwise ops, and both shifts, but NO integer divide or remainder
/// (div/rem return error.Unsupported in riscv64/isel.zig). A group on an unsupported op is left
/// scalar so the vectorizer never emits something the backend rejects.
fn piSupportsOp(op: BinOp) bool {
    return switch (op) {
        .add, .sub, .mul, .bit_and, .bit_or, .bit_xor, .shl, .shr => true,
        .div, .rem, .mulh => false,
    };
}

/// Find and fuse the first eligible group of `lanes` instructions in `block`. Returns true if
/// it did. A group is `lanes` contiguous `.arith` sharing one BinOp and one scalar element type,
/// where the element is either f32 (always eligible) or, when `allow_i32`, a 32-bit integer whose
/// shared BinOp the pi backend lowers. The packed vector's element type is the group's own scalar
/// type, so an i32 group's signedness carries through to the vector (and thus to the shift lowering).
fn vectorizeOne(
    allocator: std.mem.Allocator,
    func: *Function,
    block: Block,
    vmap: *VMap,
    lanes: u8,
    allow_i32: bool,
    model: ?*const mm.Model,
    coalesced_loads: *CoalescedLoads,
    coalesced: *bool,
) Error!bool {
    // Scan (read-only) for `lanes` contiguous `arith` instructions sharing a BinOp AND a scalar
    // element type. The head fixes the element type (its result type handle); every lane must
    // match it exactly, so signedness and width are uniform across the group.
    var pos: usize = 0;
    var op: BinOp = undefined;
    var elem_t: ir.types.Type = undefined; // the group's scalar element type (f32 or i32)
    var group: [MAX_LANES]Inst = undefined;
    var found = false;
    // Coalescing analysis for the SELECTED group, carried from the gate to emission so the memory
    // shape is recognized once. `sel_a`/`sel_b` are the operands' contiguous-load runs (null when the
    // operand is not a coalesceable load run); `sel_a_eq_b` is set when both operands are the same
    // value in every lane (one operand vector, e.g. `a*a`).
    var sel_a: ?LoadRun = null;
    var sel_b: ?LoadRun = null;
    var sel_a_eq_b = false;
    {
        const insts = func.blockInsts(block);
        scan: for (0..(if (insts.len >= lanes) insts.len - lanes + 1 else 0)) |g| {
            const head = func.opcodeMut(insts[g]).*;
            if (head != .arith) continue;
            const head_res = func.instResult(insts[g]).?;
            const is_f32 = isScalarF32(func, head_res);
            // The integer path is off unless the caller allowed it (a vpu model) AND the shared op
            // is one the pi backend lowers. A value is never both f32 and i32, so this stays
            // mutually exclusive with the f32 path.
            const is_i32 = allow_i32 and isScalarI32(func, head_res) and piSupportsOp(head.arith.op);
            if (!is_f32 and !is_i32) continue;
            const head_ty = func.valueType(head_res);
            for (1..lanes) |k| {
                const o = func.opcodeMut(insts[g + k]).*;
                if (o != .arith or o.arith.op != head.arith.op) continue :scan;
                // Same interned scalar type as the head: identical element kind, signedness, width.
                if (func.valueType(func.instResult(insts[g + k]).?) != head_ty) continue :scan;
            }
            // Capture this candidate's operand and result lanes to analyze its memory shape.
            var ca: [MAX_LANES]Value = undefined;
            var cb: [MAX_LANES]Value = undefined;
            var cc: [MAX_LANES]Value = undefined;
            for (0..lanes) |k| {
                const arith = func.opcodeMut(insts[g + k]).*.arith;
                ca[k] = arith.lhs;
                cb[k] = arith.rhs;
                cc[k] = func.instResult(insts[g + k]).?;
            }
            // `a_eq_b` when both sides are the same value in every lane: one operand vector to build.
            var a_eq_b = true;
            for (0..lanes) |k| {
                if (ca[k] != cb[k]) {
                    a_eq_b = false;
                    break;
                }
            }
            // How each operand vector will be built, matching `buildOperand`'s priority: chain reuse
            // (already-live vector, free) first, then a contiguous-load coalesce (one wide load),
            // else a pack. These classifications feed the profitability price AND drive emission.
            const chain_a = chainVector(vmap, ca[0..lanes]) != null;
            const chain_b = if (a_eq_b) chain_a else chainVector(vmap, cb[0..lanes]) != null;
            const run_a = if (chain_a) null else analyzeLoadRun(func, block, ca[0..lanes], g);
            const run_b = if (a_eq_b) run_a else if (chain_b) null else analyzeLoadRun(func, block, cb[0..lanes], g);
            const result_store = resultsAreCoalesceableStores(func, block, cc[0..lanes], lanes);
            // Result-chaining credit: when the group's result is NOT stored, its lanes may instead
            // feed a fusable consumer group (e.g. the SAXPY mul feeding the add), staying in a vector
            // register so the chain-reuse vmap elides the extracts. This is the producer-side analogue
            // of `chained_ops` (operand-side reuse). Only computed when the result is not a coalesced
            // store: a group's result leaves EITHER as a wide store OR chained, never both.
            // Pass the group's LAST index (g + lanes - 1) so a consumer must sit after the WHOLE group,
            // not just after its head (a group member reading an earlier lane's result is not a chain).
            const result_chained = if (result_store) false else resultsChainIntoGroup(func, block, cc[0..lanes], g + lanes - 1);

            // Profitability gate (model-driven path only). A structurally-eligible group is fused
            // only when the cost model judges the vector form cheaper than the serialized scalar
            // form for this microarch; otherwise it is left scalar and the scan moves on to look for
            // a later, profitable group. The correctness path (model == null) skips the gate and
            // force-fuses. `distinct` counts the operand vectors the group must build: 1 when both
            // sides are the same value in every lane (e.g. `a*a`), 2 otherwise. `coalesced_ops` is
            // how many of those the loads coalesce (wide loads, not packs); `chained_ops` is how many
            // are reused for free from an earlier group (no pack, no load). `result_chained` credits
            // the producer side: the result rides a vector into a consumer group, so no unpack.
            if (model) |m| {
                const distinct: u8 = if (a_eq_b) 1 else 2;
                const coalesced_ops: u8 = if (a_eq_b)
                    (if (run_a != null) @as(u8, 1) else 0)
                else
                    (if (run_a != null) @as(u8, 1) else 0) + (if (run_b != null) @as(u8, 1) else 0);
                const chained_ops: u8 = if (a_eq_b)
                    (if (chain_a) @as(u8, 1) else 0)
                else
                    (if (chain_a) @as(u8, 1) else 0) + (if (chain_b) @as(u8, 1) else 0);
                if (!cost.slpProfitableMem(m, head.arith.op, is_f32, lanes, distinct, coalesced_ops, chained_ops, result_store, result_chained)) continue :scan;
            }
            pos = g;
            op = head.arith.op;
            elem_t = head_ty;
            for (0..lanes) |k| group[k] = insts[g + k];
            sel_a = run_a;
            sel_b = run_b;
            sel_a_eq_b = a_eq_b;
            found = true;
            break;
        }
    }
    if (!found) return false;

    // Capture each lane's operands and result before appending.
    var a: [MAX_LANES]Value = undefined;
    var b: [MAX_LANES]Value = undefined;
    var c: [MAX_LANES]Value = undefined;
    for (0..lanes) |k| {
        const arith = func.opcodeMut(group[k]).*.arith;
        a[k] = arith.lhs;
        b[k] = arith.rhs;
        c[k] = func.instResult(group[k]).?;
    }

    const vt = try func.types.intern(.{ .vector = .{ .len = lanes, .elem = elem_t } });

    const old_len = func.blockInsts(block).len; // boundary between old and appended insts

    const va = try buildOperand(allocator, func, block, vt, vmap, a[0..lanes], sel_a, coalesced_loads, coalesced);
    // When both sides are the same operand vector, build it once and reuse it (avoids a second
    // identical pack or wide load for `a*a`-shaped groups).
    const vb = if (sel_a_eq_b) va else try buildOperand(allocator, func, block, vt, vmap, b[0..lanes], sel_b, coalesced_loads, coalesced);
    const vc = try func.appendInst(block, vt, .{ .arith = .{ .op = op, .lhs = va, .rhs = vb } });
    var x: [MAX_LANES]Value = undefined;
    try unpack(allocator, func, block, elem_t, vc, x[0..lanes], vmap);

    // Redirect every downstream use of the scalar results to the extracted lanes. When those uses
    // are contiguous stores, the later `coalesceStores` scan collapses them to one wide store.
    for (0..lanes) |k| func.replaceAllUses(c[k], x[k]);

    // Move the appended sequence to the group's position and drop the now-dead scalar ops.
    try splice(allocator, func, block, pos, old_len, lanes);
    return true;
}

/// The vector to feed as an operand. In priority: (1) chain reuse when `scalars` are already lanes
/// 0..N-1 of one vector from an earlier group; (2) one wide vector load when `run` recognized them as
/// a contiguous ascending load run (the coalesced-away scalar loads are recorded for removal); (3) a
/// pack (`struct_new`, one lane-insert apiece), which is always correct.
fn buildOperand(
    allocator: std.mem.Allocator,
    func: *Function,
    block: Block,
    vt: ir.types.Type,
    vmap: *VMap,
    scalars: []const Value,
    load_run: ?LoadRun,
    coalesced_loads: *CoalescedLoads,
    coalesced: *bool,
) Error!Value {
    if (chainVector(vmap, scalars)) |v| return v; // the chain stays in a vector register
    if (load_run) |r| {
        // One wide load from the run's base address replaces `lanes` scalar loads + a pack. Record
        // each replaced load's result so `cleanup` removes it once its scalar uses are gone.
        const v = try func.appendInst(block, vt, .{ .load = .{ .ptr = r.ptr0 } });
        for (0..scalars.len) |k| try coalesced_loads.put(allocator, func.instResult(r.loads[k]).?, {});
        coalesced.* = true;
        return v;
    }
    return pack(func, block, vt, scalars);
}

/// If `scalars` are exactly lanes 0..N-1 of one existing vector, return that vector (chain reuse).
fn chainVector(vmap: *const VMap, scalars: []const Value) ?Value {
    const l0 = vmap.get(scalars[0]) orelse return null;
    if (l0.lane != 0) return null;
    for (1..scalars.len) |k| {
        const lk = vmap.get(scalars[k]) orelse return null;
        if (lk.vec != l0.vec or lk.lane != k) return null;
    }
    return l0.vec;
}

/// Pack scalars into a vector with a `struct_new` (lowered to one insert per lane). A pure
/// register build, so a pack rendered dead by chain reuse falls to DCE.
fn pack(func: *Function, block: Block, vt: ir.types.Type, scalars: []const Value) Error!Value {
    const list = try func.internValueList(scalars);
    return func.appendInst(block, vt, .{ .struct_new = .{ .fields = list } });
}

/// Extract `out.len` scalars from `vec` (one `extract` op per lane), recording each as a known
/// lane of `vec` so a later group can reuse it. The extracts are pure, so chain-dead ones DCE.
fn unpack(allocator: std.mem.Allocator, func: *Function, block: Block, elem_t: ir.types.Type, vec: Value, out: []Value, vmap: *VMap) Error!void {
    for (0..out.len) |k| {
        out[k] = try func.appendInst(block, elem_t, .{ .extract = .{ .aggregate = vec, .index = @intCast(k) } });
        try vmap.put(allocator, out[k], .{ .vec = vec, .lane = @intCast(k) });
    }
}

/// Reorder the block so the appended vector sequence (insts[old_len..]) sits where the
/// scalar group was, and the `lanes` dead scalar instructions at `pos` are removed.
fn splice(allocator: std.mem.Allocator, func: *Function, block: Block, pos: usize, old_len: usize, lanes: u8) Error!void {
    const insts = func.blockInstsMut(block);
    var rebuilt: std.ArrayList(Inst) = .empty;
    defer rebuilt.deinit(allocator);
    try rebuilt.appendSlice(allocator, insts.items[0..pos]); // before the group
    try rebuilt.appendSlice(allocator, insts.items[old_len..]); // the vectorized sequence
    try rebuilt.appendSlice(allocator, insts.items[pos + lanes .. old_len]); // after the group
    insts.clearRetainingCapacity();
    try insts.appendSlice(allocator, rebuilt.items);
}

/// Decompose an address value into a base pointer and a byte offset. Recognizes the `+ imm` address
/// form the frontends emit (`arith_imm add base, imm`); any other producer is treated as base + 0.
fn addrParts(func: *const Function, ptr: Value) AddrParts {
    if (func.definingInst(ptr)) |inst| {
        const opcode = func.opcode(inst);
        if (opcode == .arith_imm and opcode.arith_imm.op == .add) {
            return .{ .base = opcode.arith_imm.lhs, .off = opcode.arith_imm.imm };
        }
    }
    return .{ .base = ptr, .off = 0 };
}

/// The position of `target` in `block`'s instruction list, or null if it is not in this block.
fn instPos(func: *const Function, block: Block, target: Inst) ?usize {
    for (func.blockInsts(block), 0..) |inst, i| {
        if (inst == target) return i;
    }
    return null;
}

/// Whether any memory-writing op (a store or a call, which may write through a pointer) sits in the
/// half-open block-position window `[lo, hi)`. Loads are pure readers and are allowed. Used to prove
/// that fusing a set of loads into one wide load placed at `hi` observes the same memory each scalar
/// load did (no intervening write could have changed it).
fn hasWriteBetween(func: *const Function, block: Block, lo: usize, hi: usize) bool {
    const insts = func.blockInsts(block);
    var i = lo;
    while (i < hi) : (i += 1) {
        switch (func.opcode(insts[i])) {
            .store, .call, .call_indirect => return true,
            else => {},
        }
    }
    return false;
}

/// Recognize `scalars` (an operand's lane values, in lane order) as a contiguous ascending run of
/// scalar loads safe to fuse into one wide load placed at `group_pos`. Returns the run when: every
/// lane is the result of a `load` in `block`; the lane-k address is base + (off0 + k*ELEM_BYTES)
/// sharing one base with lane 0; and no store/call sits between the earliest of these loads and
/// `group_pos` (so the single wide load, which executes at `group_pos`, sees the same memory each
/// scalar load saw). Any failure returns null and the caller packs, which is always correct. A miss
/// is a missed optimization; a false positive would be a miscompile, so every condition is exact.
fn analyzeLoadRun(func: *const Function, block: Block, scalars: []const Value, group_pos: usize) ?LoadRun {
    std.debug.assert(scalars.len >= 2 and scalars.len <= MAX_LANES);
    var loads: [MAX_LANES]Inst = undefined;
    var min_pos: usize = std.math.maxInt(usize);
    var base0: Value = undefined;
    var off0: i64 = undefined;
    var ptr0: Value = undefined;
    for (scalars, 0..) |s, k| {
        const inst = func.definingInst(s) orelse return null;
        if (func.opcode(inst) != .load) return null;
        const p = instPos(func, block, inst) orelse return null; // the load must live in this block
        loads[k] = inst;
        if (p < min_pos) min_pos = p;
        const parts = addrParts(func, func.opcode(inst).load.ptr);
        if (k == 0) {
            base0 = parts.base;
            off0 = parts.off;
            ptr0 = func.opcode(inst).load.ptr;
        } else {
            if (parts.base != base0) return null;
            // Lane k must sit exactly one element above lane k-1: base + off0 + k*ELEM_BYTES. This
            // also rules out a repeated address (a splat), whose offsets would not be distinct.
            if (parts.off != off0 + @as(i64, @intCast(k)) * ELEM_BYTES) return null;
        }
    }
    // A load's result is defined before it is used, so every load precedes the group; guard anyway.
    if (min_pos >= group_pos) return null;
    if (hasWriteBetween(func, block, min_pos, group_pos)) return null;
    return .{ .ptr0 = ptr0, .loads = loads };
}

/// Whether the group's `lanes` scalar results are each consumed by exactly one store, those stores
/// writing contiguous ascending addresses (base + k*ELEM_BYTES) with no aliasing memory op between
/// the earliest and latest of them. A pre-vectorization predictor used only to PRICE the group (the
/// actual rewrite is `coalesceStores`, which runs on the post-fusion extracts). Conservative: any
/// non-store memory op, or a store not part of this run, inside the window forbids coalescing.
fn resultsAreCoalesceableStores(func: *const Function, block: Block, results: []const Value, lanes: u8) bool {
    std.debug.assert(results.len == lanes);
    const insts = func.blockInsts(block);
    var positions: [MAX_LANES]usize = undefined;
    var base0: Value = undefined;
    var off0: i64 = undefined;
    for (results, 0..) |res, k| {
        // The unique store whose value is this result. Two stores of one result, or none, disqualify.
        var store_pos: ?usize = null;
        for (insts, 0..) |inst, i| {
            const opcode = func.opcode(inst);
            if (opcode == .store and opcode.store.value == res) {
                if (store_pos != null) return false;
                store_pos = i;
            }
        }
        const si = store_pos orelse return false;
        positions[k] = si;
        const parts = addrParts(func, func.opcode(insts[si]).store.ptr);
        if (k == 0) {
            base0 = parts.base;
            off0 = parts.off;
        } else {
            if (parts.base != base0) return false;
            if (parts.off != off0 + @as(i64, @intCast(k)) * ELEM_BYTES) return false;
        }
    }
    var lo = positions[0];
    var hi = positions[0];
    for (positions[0..lanes]) |p| {
        lo = @min(lo, p);
        hi = @max(hi, p);
    }
    // No load/call inside the window (they could alias), and no store other than our own `lanes`.
    var store_count: usize = 0;
    var i = lo;
    while (i <= hi) : (i += 1) {
        switch (func.opcode(insts[i])) {
            .store => store_count += 1,
            .load, .call, .call_indirect => return false,
            else => {},
        }
    }
    return store_count == lanes;
}

/// One `.arith` consumer of a group result: the instruction, its block position, and whether the
/// result sits in the consumer's `lhs` (else `rhs`) slot.
const ArithConsumer = struct { inst: Inst, pos: usize, c_is_lhs: bool };

/// Total number of times `v` is used as an operand anywhere in `func` (every instruction operand
/// slot, `if` block-arg edges, and terminators). Kept exhaustive over the opcode set so a new
/// operand slot cannot silently go uncounted and let a many-use value pass the exactly-once test.
/// Used to prove a group result feeds exactly one consumer before crediting a chained (free) unpack.
fn valueUseCount(func: *const Function, v: Value) usize {
    var n: usize = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .iconst, .fconst, .alloca, .global_addr => {},
                .arith => |x| {
                    if (x.lhs == v) n += 1;
                    if (x.rhs == v) n += 1;
                },
                .arith_imm => |x| if (x.lhs == v) {
                    n += 1;
                },
                .icmp => |x| {
                    if (x.lhs == v) n += 1;
                    if (x.rhs == v) n += 1;
                },
                .select => |x| {
                    if (x.cond == v) n += 1;
                    if (x.then == v) n += 1;
                    if (x.@"else" == v) n += 1;
                },
                .extract => |x| if (x.aggregate == v) {
                    n += 1;
                },
                .convert => |x| if (x.value == v) {
                    n += 1;
                },
                .unary => |x| if (x.value == v) {
                    n += 1;
                },
                .load => |x| if (x.ptr == v) {
                    n += 1;
                },
                .store => |x| {
                    if (x.value == v) n += 1;
                    if (x.ptr == v) n += 1;
                },
                .prefetch => |x| if (x.ptr == v) {
                    n += 1;
                },
                .dot => |x| {
                    if (x.acc == v) n += 1;
                    if (x.a == v) n += 1;
                    if (x.b == v) n += 1;
                },
                .matmul => |x| {
                    if (x.a == v) n += 1;
                    if (x.b == v) n += 1;
                    if (x.c == v) n += 1;
                },
                .struct_new => |x| for (func.valueList(x.fields)) |f| {
                    if (f == v) n += 1;
                },
                .call => |x| for (func.valueList(x.args)) |arg| {
                    if (arg == v) n += 1;
                },
                .call_indirect => |x| {
                    if (x.target == v) n += 1;
                    for (func.valueList(x.args)) |arg| {
                        if (arg == v) n += 1;
                    }
                },
                .@"if" => |x| {
                    if (x.cond == v) n += 1;
                    for (func.blockArgs(x.then)) |arg| {
                        if (arg == v) n += 1;
                    }
                    for (func.blockArgs(x.@"else")) |arg| {
                        if (arg == v) n += 1;
                    }
                },
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .ret => |rv| if (rv) |vv| {
                if (vv == v) n += 1;
            },
            .jump => |j| for (func.blockArgs(j)) |arg| {
                if (arg == v) n += 1;
            },
        };
    }
    return n;
}

/// The single `.arith` consumer of `v` in `block`, or null. Requires `v` to be used EXACTLY ONCE in
/// the whole function and that one use to be an `.arith` in `block` after `group_pos` (a result used
/// by a store, a return, an `if`, a non-arith op, or more than once returns null, so the unpack is
/// charged as real). Reports which operand slot `v` occupies so the caller can require a consistent
/// slot across lanes (the condition the chain-reuse vmap needs to elide the extracts).
fn singleArithConsumer(func: *const Function, block: Block, v: Value, group_pos: usize) ?ArithConsumer {
    if (valueUseCount(func, v) != 1) return null;
    const insts = func.blockInsts(block);
    for (insts, 0..) |inst, i| {
        if (i <= group_pos) continue; // the consumer is defined after the producing group
        const op = func.opcode(inst);
        if (op != .arith) continue;
        if (op.arith.lhs == v) return .{ .inst = inst, .pos = i, .c_is_lhs = true };
        if (op.arith.rhs == v) return .{ .inst = inst, .pos = i, .c_is_lhs = false };
    }
    return null;
}

/// Whether the group's `lanes` results (`results[k]` is lane k's scalar result, in lane order) each
/// feed a fusable consumer that keeps them in a vector register, so the group's result never needs a
/// real unpack: a later pass fuses the consumers, and the chain-reuse vmap reuses the producer's
/// result vector directly (the extracts go dead and DCE away). Conservative: returns true ONLY when
///   - every result is used EXACTLY ONCE, by an `.arith` (any store/return/other use -> false),
///   - those `lanes` consumers occupy `lanes` CONTIGUOUS block positions in lane order (the consumer
///     of result k sits exactly k slots past the consumer of result 0), which is the shape a later
///     `vectorizeOne` recognizes and fuses,
///   - all consumers share one BinOp and one scalar result type (a fusable group), and
///   - each consumer reads its result in the SAME operand slot (lhs or rhs) as lane 0, so the fused
///     consumer sees results[0..lanes] as lanes 0..N-1 of the producer vector (chain reuse fires).
/// Any other shape -> false, and the caller charges the real per-lane unpack (always safe: declining
/// the credit at worst leaves a genuine chain mispriced-conservative, never miscompiles). This is the
/// producer-side analogue of `buildOperand`'s operand chain reuse.
fn resultsChainIntoGroup(func: *const Function, block: Block, results: []const Value, group_pos: usize) bool {
    std.debug.assert(results.len >= 2 and results.len <= MAX_LANES);
    // Lane 0 fixes the consumer op, result type, base position, and operand slot every lane matches.
    const c0 = singleArithConsumer(func, block, results[0], group_pos) orelse return false;
    const op0 = func.opcode(c0.inst).arith.op;
    const ty0 = func.valueType(func.instResult(c0.inst).?);
    for (1..results.len) |k| {
        const ck = singleArithConsumer(func, block, results[k], group_pos) orelse return false;
        // Contiguous and lane-ordered: consumer of lane k sits exactly k positions past lane 0's, so
        // the later fusion captures them as lanes 0..N-1 in the same order the producer emitted them.
        if (ck.pos != c0.pos + k) return false;
        // Same fusable op and same scalar result type across every consumer lane.
        if (func.opcode(ck.inst).arith.op != op0) return false;
        if (func.valueType(func.instResult(ck.inst).?) != ty0) return false;
        // Same operand slot, so the chained operand vector is results[0..lanes] in lane order (a
        // differing slot would leave results split across both operands, breaking chain reuse).
        if (ck.c_is_lhs != c0.c_is_lhs) return false;
    }
    return true;
}

/// Which vector lane a scalar value is, if any: via the chain map (a fused group's extract) or, for
/// hand-built or not-yet-mapped values, directly via an `extract` producer. Both express the same
/// fact ("this scalar is lane N of vector V"); the direct-extract fallback is the superset.
fn laneOfValue(func: *const Function, vmap: *const VMap, v: Value) ?LaneOf {
    if (vmap.get(v)) |l| return l;
    if (func.definingInst(v)) |inst| {
        const opcode = func.opcode(inst);
        if (opcode == .extract) return .{ .vec = opcode.extract.aggregate, .lane = @intCast(opcode.extract.index) };
    }
    return null;
}

/// Rewrite every run of `lanes` contiguous scalar stores that write lanes 0..N-1 of ONE vector to
/// contiguous ascending addresses into a single wide vector store. Returns whether anything changed.
/// Runs after this block's arith groups are fused, so a group's result lanes are the extracts feeding
/// the stores. Conservative: the run must be barrier-free (no load/call/if or foreign store between
/// its stores), lane-ordered, and exactly contiguous, or it is left as scalar stores.
fn coalesceStores(allocator: std.mem.Allocator, func: *Function, block: Block, vmap: *const VMap, lanes: u8) Error!bool {
    var changed = false;
    // Re-scan from the top after each rewrite: removing stores shifts positions, and a block may hold
    // more than one store run (multiple SLP groups). Bounded by the store count, so it terminates.
    while (try coalesceStoreRun(allocator, func, block, vmap, lanes)) changed = true;
    return changed;
}

/// Find and coalesce the first eligible store run in `block`. Returns true if it rewrote one.
fn coalesceStoreRun(allocator: std.mem.Allocator, func: *Function, block: Block, vmap: *const VMap, lanes: u8) Error!bool {
    const insts = func.blockInsts(block);
    var start: usize = 0;
    while (start < insts.len) : (start += 1) {
        if (func.opcode(insts[start]) != .store) continue;
        // Grow a barrier-free window of `lanes` stores from `start` (pure ops between are fine; any
        // load/call/if breaks the window, since a wide store moves the later lanes' writes earlier).
        var store_positions: [MAX_LANES]usize = undefined;
        var cnt: usize = 0;
        var i = start;
        var broke = false;
        while (i < insts.len and cnt < lanes) : (i += 1) {
            switch (func.opcode(insts[i])) {
                .store => {
                    store_positions[cnt] = i;
                    cnt += 1;
                },
                .load, .call, .call_indirect, .@"if" => {
                    broke = true;
                    break;
                },
                else => {}, // a pure op (e.g. the address arith_imm) may sit between stores
            }
        }
        if (broke or cnt < lanes) continue;
        // Validate: store j writes lane j of one shared vector to base + j*ELEM_BYTES.
        var vec: Value = undefined;
        var base0: Value = undefined;
        var off0: i64 = undefined;
        var ok = true;
        for (0..lanes) |j| {
            const st = func.opcode(insts[store_positions[j]]).store;
            const lane = laneOfValue(func, vmap, st.value) orelse {
                ok = false;
                break;
            };
            if (lane.lane != j) {
                ok = false;
                break;
            }
            const parts = addrParts(func, st.ptr);
            if (j == 0) {
                vec = lane.vec;
                base0 = parts.base;
                off0 = parts.off;
            } else {
                if (lane.vec != vec or parts.base != base0 or parts.off != off0 + @as(i64, @intCast(j)) * ELEM_BYTES) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) continue;
        // Rewrite store 0 in place to store the whole vector to its (lane-0) address, and drop the
        // other `lanes-1` scalar stores. Their now-dead extracts fall to `cleanup`.
        const ptr0 = func.opcode(insts[store_positions[0]]).store.ptr;
        func.opcodeMut(insts[store_positions[0]]).* = .{ .store = .{ .value = vec, .ptr = ptr0 } };
        var drop: [MAX_LANES]Inst = undefined;
        for (1..lanes) |j| drop[j - 1] = insts[store_positions[j]];
        try removeInsts(allocator, func, block, drop[0 .. lanes - 1]);
        return true;
    }
    return false;
}

/// Remove the given instructions from `block`'s instruction list (order preserved for the rest).
fn removeInsts(allocator: std.mem.Allocator, func: *Function, block: Block, drop: []const Inst) Error!void {
    const insts = func.blockInstsMut(block);
    var rebuilt: std.ArrayList(Inst) = .empty;
    defer rebuilt.deinit(allocator);
    outer: for (insts.items) |inst| {
        for (drop) |d| {
            if (inst == d) continue :outer;
        }
        try rebuilt.append(allocator, inst);
    }
    insts.clearRetainingCapacity();
    try insts.appendSlice(allocator, rebuilt.items);
}

/// Post-coalescing cleanup: iterate to a fixpoint removing dead pure instructions (address
/// arithmetic and packs/extracts left dead by coalescing) and the specific scalar loads that
/// coalescing replaced with wide loads. A general DCE keeps every load (memory op); the coalesced
/// loads in `coalesced_loads` are known-replaced, so a coalesced load whose result is now unused is
/// safe to drop. Only ever runs when coalescing fired, so non-memory functions are untouched.
fn cleanup(allocator: std.mem.Allocator, func: *Function, coalesced_loads: *const CoalescedLoads) Error!void {
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    while (true) {
        countUses(func, uses);
        var removed = false;
        for (0..func.blockCount()) |bi| {
            const insts = func.blockInstsMut(@enumFromInt(bi));
            var w: usize = 0;
            for (insts.items) |inst| {
                const result = func.instResult(inst);
                const unused = if (result) |r| uses[@intFromEnum(r)] == 0 else false;
                const is_coalesced_load = func.opcode(inst) == .load and result != null and
                    coalesced_loads.contains(result.?);
                const dead = unused and (isPure(func.opcode(inst)) or is_coalesced_load);
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

/// Whether an instruction has no side effects (may be dropped when unused). Mirrors dce.zig's rule:
/// loads/stores/prefetch/calls/`if` are impure and kept (coalesced loads are handled separately).
fn isPure(op: ir.function.Opcode) bool {
    return switch (op) {
        .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .struct_new, .extract, .convert, .unary, .alloca, .global_addr, .dot => true,
        .load, .store, .prefetch, .matmul, .@"if", .call, .call_indirect => false,
    };
}

/// Count uses of every value across all instructions, `if` edges, and terminators, into `uses`.
/// Local to the cleanup pass (dce.zig's equivalent is private); kept exhaustive so a new operand
/// slot cannot silently go uncounted and let a still-used instruction be dropped.
fn countUses(func: *const Function, uses: []u32) void {
    @memset(uses, 0);
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .iconst, .fconst, .alloca, .global_addr => {},
                .arith => |x| {
                    uses[@intFromEnum(x.lhs)] += 1;
                    uses[@intFromEnum(x.rhs)] += 1;
                },
                .arith_imm => |x| uses[@intFromEnum(x.lhs)] += 1,
                .icmp => |x| {
                    uses[@intFromEnum(x.lhs)] += 1;
                    uses[@intFromEnum(x.rhs)] += 1;
                },
                .select => |x| {
                    uses[@intFromEnum(x.cond)] += 1;
                    uses[@intFromEnum(x.then)] += 1;
                    uses[@intFromEnum(x.@"else")] += 1;
                },
                .extract => |x| uses[@intFromEnum(x.aggregate)] += 1,
                .convert => |x| uses[@intFromEnum(x.value)] += 1,
                .unary => |x| uses[@intFromEnum(x.value)] += 1,
                .load => |x| uses[@intFromEnum(x.ptr)] += 1,
                .store => |x| {
                    uses[@intFromEnum(x.value)] += 1;
                    uses[@intFromEnum(x.ptr)] += 1;
                },
                .prefetch => |x| uses[@intFromEnum(x.ptr)] += 1,
                .dot => |x| {
                    uses[@intFromEnum(x.acc)] += 1;
                    uses[@intFromEnum(x.a)] += 1;
                    uses[@intFromEnum(x.b)] += 1;
                },
                .matmul => |x| {
                    uses[@intFromEnum(x.a)] += 1;
                    uses[@intFromEnum(x.b)] += 1;
                    uses[@intFromEnum(x.c)] += 1;
                },
                .struct_new => |x| for (func.valueList(x.fields)) |f| {
                    uses[@intFromEnum(f)] += 1;
                },
                .call => |x| for (func.valueList(x.args)) |arg| {
                    uses[@intFromEnum(arg)] += 1;
                },
                .call_indirect => |x| {
                    uses[@intFromEnum(x.target)] += 1;
                    for (func.valueList(x.args)) |arg| uses[@intFromEnum(arg)] += 1;
                },
                .@"if" => |x| {
                    uses[@intFromEnum(x.cond)] += 1;
                    for (func.blockArgs(x.then)) |arg| uses[@intFromEnum(arg)] += 1;
                    for (func.blockArgs(x.@"else")) |arg| uses[@intFromEnum(arg)] += 1;
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

const registry = @import("microarch/registry.zig");

/// Build a block of `n` parallel `ai + bi` f32 adds and return the function. The first result feeds
/// the terminator so it stays live. Caller deinits.
fn parallelAdds(n: usize) !Function {
    var func = Function.init(std.testing.allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    var params_a: [8]Value = undefined;
    var params_b: [8]Value = undefined;
    for (0..n) |i| params_a[i] = try func.appendBlockParam(block, f32_t);
    for (0..n) |i| params_b[i] = try func.appendBlockParam(block, f32_t);
    var first: ?Value = null;
    for (0..n) |i| {
        const c = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = params_a[i], .rhs = params_b[i] } });
        if (first == null) first = c;
    }
    func.setTerminator(block, .{ .ret = first.? });
    return func;
}

/// True if some arith instruction in block 0 produces a vector of exactly `want` lanes.
fn hasVectorArith(func: *const Function, want: u32) bool {
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .arith) continue;
        const r = func.instResult(inst) orelse continue;
        switch (func.types.type_kind(func.valueType(r))) {
            .vector => |v| if (v.len == want) return true,
            else => {},
        }
    }
    return false;
}

/// True if some arith in block 0 produces a `<want x iN>` (an integer-element vector) of exactly
/// `want` lanes. Distinguishes the pi integer path from the f32 path, which `hasVectorArith`
/// cannot on its own.
fn hasIntVectorArith(func: *const Function, want: u32) bool {
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .arith) continue;
        const r = func.instResult(inst) orelse continue;
        switch (func.types.type_kind(func.valueType(r))) {
            .vector => |v| if (v.len == want and func.types.type_kind(v.elem) == .int) return true,
            else => {},
        }
    }
    return false;
}

/// Build a block of `n` parallel `op(ai, bi)` i32 ops (signed) and return the function. The first
/// result feeds the terminator so it stays live. Mirrors `parallelAdds` but on a 32-bit int
/// element, so it is the exact shape the pi integer SLP path scans for. Caller deinits.
fn parallelIntOps(n: usize, op: BinOp) !Function {
    var func = Function.init(std.testing.allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    var params_a: [8]Value = undefined;
    var params_b: [8]Value = undefined;
    for (0..n) |i| params_a[i] = try func.appendBlockParam(block, i32_t);
    for (0..n) |i| params_b[i] = try func.appendBlockParam(block, i32_t);
    var first: ?Value = null;
    for (0..n) |i| {
        const c = try func.appendInst(block, i32_t, .{ .arith = .{ .op = op, .lhs = params_a[i], .rhs = params_b[i] } });
        if (first == null) first = c;
    }
    func.setTerminator(block, .{ .ret = first.? });
    return func;
}

test "runModel fuses contiguous i32 add/mul/xor/shl groups to <8 x i32> under et-soc (pi)" {
    // Each of these ops is one the packed-integer (pi) backend lowers, so under et-soc's vpu model
    // the 8 contiguous scalar-i32 arith fuse into a single 8-lane integer vector arith.
    for ([_]BinOp{ .add, .mul, .bit_xor, .shl }) |op| {
        var func = try parallelIntOps(8, op);
        defer func.deinit();
        const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"et-soc"));
        try std.testing.expect(changed);
        try std.testing.expect(hasIntVectorArith(&func, 8));
        // A total transform, not a half-applied one: the function still verifies.
        var diags = try ir.verify.verify(std.testing.allocator, &func, .low);
        defer diags.deinit();
        try std.testing.expect(diags.ok());
    }
}

test "runModel leaves an i32 group scalar under a non-vpu riscv64 model" {
    // river-rc1.n is riscv64 but has neither `v` nor `vpu`, so it vectorizes nothing at all (its
    // vec_ok is false). Even a vpu-supported op like add stays scalar: there is no <8 x i32>
    // lowering for it, so producing one would be unsound.
    var func = try parallelIntOps(8, .add);
    defer func.deinit();
    const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"river-rc1.n"));
    try std.testing.expect(!changed);
    try std.testing.expect(!hasIntVectorArith(&func, 8));
    try std.testing.expect(!hasVectorArith(&func, 4));
}

test "runModel leaves an i32 group scalar under aarch64 (neon has no <8 x i32> pi lowering)" {
    // ampere-altra is a vector-capable aarch64 (neon) part, so its f32 path is live, but the
    // integer path is gated on a riscv64 vpu model. An i32 group must NOT be vectorized: neon has
    // no `<8 x i32>` pi lowering in this backend, only et-soc does.
    var func = try parallelIntOps(8, .add);
    defer func.deinit();
    const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(!changed);
    try std.testing.expect(!hasIntVectorArith(&func, 8));
    try std.testing.expect(!hasIntVectorArith(&func, 4));
}

test "runModel leaves an i32 .div group scalar even under et-soc (unsupported pi op)" {
    // Integer divide has no packed-integer op (div/rem return error.Unsupported in riscv64 isel),
    // so the vectorizer must skip the group and leave the 8 scalar divides alone, never emitting a
    // vector the backend would reject.
    var func = try parallelIntOps(8, .div);
    defer func.deinit();
    const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"et-soc"));
    try std.testing.expect(!changed);
    try std.testing.expect(!hasIntVectorArith(&func, 8));
    // The untouched function still verifies.
    var diags = try ir.verify.verify(std.testing.allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "runModel vectorizes ET-SOC at 8 lanes via its vpu capability (not RVV)" {
    var func = try parallelAdds(8);
    defer func.deinit();
    // ET-SOC's 8-lane unit is a custom SIMD block, not RVV: features.riscv64.v is false, but `vpu`
    // is true, and that now suffices to vectorize here (the riscv64 backend lowers the result to
    // VPU packed-single instructions, not RVV; see riscv64/isel.zig's `vpu` parameter).
    const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"et-soc"));
    try std.testing.expect(changed);
    try std.testing.expect(hasVectorArith(&func, 8));
    // The transform is total, not just producing a vector op: the function still verifies.
    var diags = try ir.verify.verify(std.testing.allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "runModel vectorizes at the model's width: 4 lanes for a 128-bit part" {
    // Uses a memory-COALESCED elementwise add over eight contiguous f32 elements: under the
    // throughput cost model, a register-input group (mul or add) is unprofitable on the wide
    // out-of-order ampere core (its independent scalar ops already fly across the ports), so the only
    // shapes that fuse there are memory-coalesced ones whose wide loads/stores pay for the pack. This
    // test is about WIDTH derivation (vector_bits/32 = 4 lanes on a 128-bit part), so it just needs a
    // shape that fuses there: the eight adds become 4-lane vectors, never an 8-lane one.
    var func = try buildMemElementwise(.add, 8, false);
    defer func.deinit();
    const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(changed);
    // A 128-bit NEON part fuses four lanes at a time, so the eight adds become 4-lane vectors, never 8.
    try std.testing.expect(hasVectorArith(&func, 4));
    try std.testing.expect(!hasVectorArith(&func, 8));
}

test "runModel declines a cheap f32 add SLP group on the wide out-of-order ampere core" {
    // The profitability gate in action: the exact `slp-adds` shape (parallel f32 adds, 2 distinct
    // operand vectors) is left scalar on ampere, because the independent scalar adds already
    // parallelize across the ALU ports and the pack/unpack overhead of the vector form loses. This
    // removes the measured 0.94x uarch-bench regression.
    var func = try parallelAdds(8);
    defer func.deinit();
    const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(!changed);
    try std.testing.expect(!hasVectorArith(&func, 4));
    try std.testing.expect(!hasVectorArith(&func, 8));
}

test "runModel leaves a scalar model alone" {
    var func = try parallelAdds(8);
    defer func.deinit();
    const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"river-rc1.s"));
    try std.testing.expect(!changed);
    try std.testing.expect(!hasVectorArith(&func, 4));
    try std.testing.expect(!hasVectorArith(&func, 8));
    // And the scheduled function still verifies (no half-applied transform).
    var diags = try ir.verify.verify(std.testing.allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

/// Count block-0 instructions whose opcode tag equals `tag`.
fn countOpcode(func: *const Function, tag: std.meta.Tag(ir.function.Opcode)) usize {
    var n: usize = 0;
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) == tag) n += 1;
    }
    return n;
}

/// True if block 0 has a `load` whose result is a vector (a coalesced wide load).
fn hasVectorLoad(func: *const Function) bool {
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .load) continue;
        const r = func.instResult(inst).?;
        if (func.types.type_kind(func.valueType(r)) == .vector) return true;
    }
    return false;
}

/// True if block 0 has a `store` whose stored value is a vector (a coalesced wide store).
fn hasVectorStore(func: *const Function) bool {
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .store) continue;
        if (func.types.type_kind(func.valueType(func.opcode(inst).store.value)) == .vector) return true;
    }
    return false;
}

/// True if block 0 has a `store` whose stored value is a scalar (i.e. not coalesced away).
fn hasScalarStore(func: *const Function) bool {
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .store) continue;
        if (func.types.type_kind(func.valueType(func.opcode(inst).store.value)) != .vector) return true;
    }
    return false;
}

/// Build a SCALAR elementwise kernel `out[i] = op(a[i], b[i])` over `n` f32 elements: `n` contiguous
/// loads of `a`, `n` of `b`, `n` contiguous `op` arith, `n` contiguous stores. The exact shape the
/// SLP scan fuses, with every operand a contiguous load and every result a contiguous store, so
/// coalescing has both sides to work on. `store_between` inserts a store to a scratch cell in the
/// MIDDLE of the `a` loads, which must force load coalescing to decline (a possibly-aliasing write
/// between the loads), proving the safety guard.
fn buildMemElementwise(op: BinOp, n: usize, store_between: bool) !Function {
    var func = Function.init(std.testing.allocator);
    errdefer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const block = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(block, ptr_t);
    const ptr_b = try func.appendBlockParam(block, ptr_t);
    const ptr_out = try func.appendBlockParam(block, ptr_t);
    const scratch = try func.appendBlockParam(block, ptr_t);

    var av: [8]Value = undefined;
    for (0..n) |i| {
        const addr = try func.appendArithImm(block, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = addr } });
        // A store landing between the `a` loads: an alias hazard that must block load coalescing.
        if (store_between and i == n / 2) try func.appendStore(block, av[0], scratch);
    }
    var bv: [8]Value = undefined;
    for (0..n) |i| {
        const addr = try func.appendArithImm(block, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = addr } });
    }
    var cv: [8]Value = undefined;
    for (0..n) |i| cv[i] = try func.appendInst(block, f32_t, .{ .arith = .{ .op = op, .lhs = av[i], .rhs = bv[i] } });
    for (0..n) |i| {
        const addr = try func.appendArithImm(block, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(block, cv[i], addr);
    }
    func.setTerminator(block, .{ .ret = null });
    return func;
}

test "coalescing turns a declined cheap-add mem kernel into a profitable vectorized one on ampere" {
    // A 4-lane f32 add is declined on ampere when it must pack/unpack (the register `slp-adds`
    // shape). The SAME arithmetic over contiguous memory now coalesces: four scalar `a` loads and
    // four `b` loads become two wide loads, four scalar stores become one wide store, and the group
    // is fused. The scalar loads/stores are gone; the wide memory ops and a `<4 x f32>` add remain.
    var func = try buildMemElementwise(.add, 4, false);
    defer func.deinit();

    const before_loads = countOpcode(&func, .load);
    try std.testing.expectEqual(@as(usize, 8), before_loads); // 4 of a + 4 of b, all scalar

    const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(changed);
    try std.testing.expect(hasVectorArith(&func, 4)); // the add fused
    try std.testing.expect(hasVectorLoad(&func)); // operands coalesced to wide loads
    try std.testing.expect(hasVectorStore(&func)); // results coalesced to a wide store
    // Two wide loads replaced the eight scalar loads; one wide store replaced the four scalar stores.
    try std.testing.expectEqual(@as(usize, 2), countOpcode(&func, .load));
    try std.testing.expectEqual(@as(usize, 1), countOpcode(&func, .store));
    try std.testing.expectEqual(@as(usize, 0), countOpcode(&func, .struct_new)); // no packing left
    try std.testing.expectEqual(@as(usize, 0), countOpcode(&func, .extract)); // no unpacking left

    var diags = try ir.verify.verify(std.testing.allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "safety: a store between the operand loads blocks THAT operand's load coalescing" {
    // The per-operand coalescing safety guard. This uses the et-soc model, not ampere: under the
    // throughput cost model an f32 mul is PIPELINED on ampere (weight 1), so an ampere group in which
    // one operand cannot coalesce (must pack) is no longer profitable and declines entirely, which
    // would not exercise the mixed coalesced/packed path. et-soc's mul is a non-pipelined async
    // MulDiv (weight 8) on a single-issue in-order core, so an 8-lane mul stays strongly profitable
    // even when one operand falls back to a pack, keeping the partial-coalesce path live. (Correctness
    // of the ampere decline is covered by the differential harness; declining is always safe.)
    //
    // Without the hazard, both operands coalesce and no pack survives. WITH a store dropped between
    // the `a` loads (a possible alias), the `a` side must fall back to packing (a `struct_new`
    // appears), proving the wide load did not form across the write. The `b` side, hazard-free, still
    // coalesces. Both forms stay verifiable, and the intervening store is never removed.
    const etsoc = registry.modelFor(.@"et-soc");
    var clean = try buildMemElementwise(.mul, 8, false);
    defer clean.deinit();
    try std.testing.expect(try runModel(std.testing.allocator, &clean, etsoc));
    try std.testing.expectEqual(@as(usize, 0), countOpcode(&clean, .struct_new)); // both operands coalesced
    try std.testing.expect(hasVectorLoad(&clean));

    var hazard = try buildMemElementwise(.mul, 8, true);
    defer hazard.deinit();
    try std.testing.expect(try runModel(std.testing.allocator, &hazard, etsoc));
    try std.testing.expect(hasVectorArith(&hazard, 8)); // the mul still fused
    try std.testing.expect(countOpcode(&hazard, .struct_new) >= 1); // the `a` side fell back to a pack
    // The intervening scratch store survives: it stores a scalar f32 (not the wide out vector), so a
    // scalar-valued store must remain, proving it was respected and never coalesced across.
    try std.testing.expect(hasScalarStore(&hazard));

    var diags = try ir.verify.verify(std.testing.allocator, &hazard, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "coalescing fires on et-soc at 8 lanes: wide load + wide store for an i32 mem kernel" {
    // The 8-lane vpu path: an 8-element i32 add over contiguous memory coalesces to two wide loads,
    // one `<8 x i32>` pi add, and one wide store, staying verifiable.
    var func = try buildMemElementwiseInt(.add, 8);
    defer func.deinit();

    const changed = try runModel(std.testing.allocator, &func, registry.modelFor(.@"et-soc"));
    try std.testing.expect(changed);
    try std.testing.expect(hasIntVectorArith(&func, 8));
    try std.testing.expect(hasVectorLoad(&func));
    try std.testing.expect(hasVectorStore(&func));
    try std.testing.expectEqual(@as(usize, 2), countOpcode(&func, .load));
    try std.testing.expectEqual(@as(usize, 1), countOpcode(&func, .store));

    var diags = try ir.verify.verify(std.testing.allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

/// i32 sibling of `buildMemElementwise` (no store-between variant), for the vpu integer path.
fn buildMemElementwiseInt(op: BinOp, n: usize) !Function {
    var func = Function.init(std.testing.allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const block = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(block, ptr_t);
    const ptr_b = try func.appendBlockParam(block, ptr_t);
    const ptr_out = try func.appendBlockParam(block, ptr_t);

    var av: [8]Value = undefined;
    for (0..n) |i| {
        const addr = try func.appendArithImm(block, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(block, i32_t, .{ .load = .{ .ptr = addr } });
    }
    var bv: [8]Value = undefined;
    for (0..n) |i| {
        const addr = try func.appendArithImm(block, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(block, i32_t, .{ .load = .{ .ptr = addr } });
    }
    var cv: [8]Value = undefined;
    for (0..n) |i| cv[i] = try func.appendInst(block, i32_t, .{ .arith = .{ .op = op, .lhs = av[i], .rhs = bv[i] } });
    for (0..n) |i| {
        const addr = try func.appendArithImm(block, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(block, cv[i], addr);
    }
    func.setTerminator(block, .{ .ret = null });
    return func;
}
