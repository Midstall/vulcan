//! SLP profitability cost model. The vectorizer's eligibility/legality checks (vectorize.zig) say
//! WHETHER a group of `lanes` parallel scalar arith ops CAN be fused; this file says whether it
//! SHOULD be. It estimates the steady-state THROUGHPUT cost (in issue-cycles, lower is better) of
//! the group left scalar versus its vectorized form, and reports profitable only when the vector
//! form is strictly cheaper.
//!
//! The one insight it captures: a cheap op on a WIDE out-of-order core is a bad SLP candidate, a
//! cheap op on a NARROW in-order core is a good one. On a wide OoO part (ampere: issue_width 4,
//! 3 alu / 2 fpsimd ports) the `lanes` independent scalar ops already parallelize across the
//! class's ports, so their scalar throughput cost is low, and the pack/unpack overhead of the
//! vector form outweighs the single vector op: SLP LOSES (this is the measured 0.94x ampere
//! `slp-adds` regression). On a single-issue in-order part (et-soc: issue_width 1) the scalar ops
//! SERIALIZE, so `lanes` of them cost `lanes` issue slots, and collapsing them to one vector op is
//! a real win even after pack/unpack: SLP wins. An expensive op (mul) strengthens the win further,
//! since each serialized scalar op costs more.
//!
//! Everything here is deterministic and target-independent: it reads only the Model (the same
//! resource data schedule.zig uses via `classPorts`, plus `throughput`/`unitOf`/`issue_width`/`exec`).
//! It weights ops by reciprocal THROUGHPUT, not latency: independent SLP lanes issue at the unit's
//! throughput, so latency would overstate a pipelined multi-cycle op's steady-state cost.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const sched = @import("schedule.zig");

const Model = mm.Model;
const UnitClass = mm.UnitClass;
const BinOp = ir.function.BinOp;

// Per-lane cost of a pack (one lane insert into an operand vector) and an unpack (one lane extract
// from the result vector), expressed as a fraction of a full arithmetic op's reciprocal-throughput.
// These micro-ops are lane moves, cheaper than arithmetic, and on the vector/mem unit they overlap
// the surrounding load/store stream rather than adding full serial cost, so they are weighted below
// unity. This fraction is the model's one tuning knob: it must stay sub-unity for a narrow-core
// cheap-op group (where the 8:1 issue collapse is the whole win) to come out profitable, while the
// overhead still grows with `lanes` and with the number of distinct operand vectors, which is what
// makes a wide-core cheap-op group (low scalar cost to begin with) come out unprofitable.
const PACK_COST_PER_LANE: f64 = 0.2;
const UNPACK_COST_PER_LANE: f64 = 0.2;

// Cost of ONE coalesced contiguous vector load (in place of a pack) or vector store (in place of an
// unpack). When the operand scalars are adjacent loads, the vectorizer emits a single wide load
// instead of `lanes` lane-inserts; when the result lanes are stored to adjacent memory it emits a
// single wide store instead of `lanes` lane-extracts. A wide memory op is one instruction, not N
// lane moves, so it is priced at a single lane's worth of the pack/unpack knob rather than `lanes`
// of them. This is what turns a wide-core cheap-op group (declined when it must pack/unpack) into a
// profitable one when its operands/results ride contiguous memory, without ever making a
// register-input group (no coalesceable memory) profitable.
const VECTOR_MEM_COST: f64 = PACK_COST_PER_LANE;

/// Build the throwaway opcode used only to query the model's per-op `throughput`/`unitOf`. Both
/// functions switch on `.arith.op` alone and never read the operands, so placeholder operand
/// handles are safe (and never dereferenced: Value is an opaque index).
fn arithOpcode(op: BinOp) ir.function.Opcode {
    return .{ .arith = .{ .op = op, .lhs = @enumFromInt(0), .rhs = @enumFromInt(0) } };
}

/// The op's per-port cost weight: its reciprocal THROUGHPUT, i.e. the cycles between two back-to-back
/// INDEPENDENT issues on one port. This is the right metric for the independent SLP lanes this model
/// prices: latency would double-count a pipelined multi-cycle op. The weight is TYPE-AWARE via
/// `elem_is_float`, because the same BinOp can have very different reciprocal throughput per element
/// type on one core (see `Model.throughput`): a cheap op (add/sub/bitwise/shift) weighs 1 for both
/// types; a MUL splits, e.g. on et-soc an f32 mul weighs 1 (pipelined VPU TXFMA) but an i32 mul
/// weighs 8 (async MulDiv), and on ampere an f32 mul weighs 1 (pipelined FP) but an i32 mul weighs 3
/// (partially-pipelined integer multiplier). A non-pipelined divide weighs its latency for both. The
/// scalar side then divides by the class's port count for cross-port parallelism, so this is the cost
/// on ONE port. Using latency here was the flagged bug: it made a pipelined-but-expensive op (e.g. an
/// ampere register f32 mul) look 4x its true steady-state cost and biased the model toward
/// over-vectorizing it.
///
/// One throughput value per (op, elem_float) prices BOTH forms of the op the profitability compare
/// weighs: the scalar op in the scalar cost and the corresponding vector op in the vector cost. That
/// assumes the scalar and vector execution of an element type share a throughput class (e.g. et-soc's
/// scalar fmul.s and its VPU fmul.ps are both pipelined at 1), which holds for the parts modeled; a
/// part whose scalar and vector FP throughput diverge would need the two costs weighed separately.
fn opWeight(model: *const Model, op: BinOp, elem_is_float: bool) f64 {
    return @floatFromInt(model.throughput(arithOpcode(op), elem_is_float));
}

/// The functional-unit class the scalar form of this op contends for. Float arith runs on the
/// FP/SIMD unit (its port count, e.g. 2 on ampere / 1 on et-soc), integer arith on the model's own
/// binding (alu for add-class, muldiv for mul-class). This is why a float and an integer add on the
/// same core can have different scalar parallelism.
fn scalarClass(model: *const Model, op: BinOp, elem_is_float: bool) UnitClass {
    if (elem_is_float) return .fpsimd;
    return model.unitOf(arithOpcode(op));
}

/// Effective scalar parallelism: how many of the `lanes` independent scalar ops actually retire per
/// cycle. An out-of-order core overlaps them up to min(issue_width, class ports); a single-issue
/// in-order core cannot reorder to fill idle slots, so they serialize at one per cycle.
fn scalarParallelism(model: *const Model, class: UnitClass) f64 {
    if (model.exec == .in_order) return 1.0;
    const ports = sched.classPorts(model, class); // maxInt(u32) when the class is unmodeled
    const eff = @min(@as(u32, model.issue_width), ports);
    return @floatFromInt(@max(@as(u32, 1), eff));
}

/// Throughput cost of leaving the group scalar: `lanes` independent ops of one class, retiring
/// `scalarParallelism` per cycle, each weighing `opWeight`.
pub fn scalarCost(model: *const Model, op: BinOp, elem_is_float: bool, lanes: u8) f64 {
    const class = scalarClass(model, op, elem_is_float);
    const par = scalarParallelism(model, class);
    return (@as(f64, @floatFromInt(lanes)) / par) * opWeight(model, op, elem_is_float);
}

/// Throughput cost of the vectorized form: one vector op, plus packing each distinct operand vector
/// (`lanes` lane-inserts apiece) and unpacking the result (`lanes` lane-extracts). This is the exact
/// SLP lowering vectorize.zig emits (a `struct_new` per operand, one `extract` per result lane), so
/// the pack overhead scales with both `lanes` and how many distinct operand vectors must be built.
pub fn vectorCost(model: *const Model, op: BinOp, elem_is_float: bool, lanes: u8, distinct_operand_vectors: u8) f64 {
    // The plain (no-coalescing) form: every operand vector is packed, the result is unpacked.
    return vectorCostMem(model, op, elem_is_float, lanes, distinct_operand_vectors, 0, 0, false, false);
}

/// Throughput cost of the vectorized form WITH memory coalescing and chain reuse accounted for. Of
/// the `distinct_operand_vectors`, `coalesced_operand_vectors` are contiguous loads (one wide load
/// each, `VECTOR_MEM_COST`, instead of `lanes` lane-inserts) and `chained_operand_vectors` are
/// already-live vectors reused from an earlier group (free, no pack and no load); the remainder are
/// packed. The result side is charged one of three ways: `result_chained` (the result stays live in
/// a vector register because it feeds a fusable consumer group at the same lanes, so the chain-reuse
/// vmap elides the extracts and they DCE away) costs NOTHING; else `result_coalesced_store` (the
/// result lanes go to contiguous memory as one wide store) costs `VECTOR_MEM_COST`; else the result
/// is unpacked with `lanes` lane-extracts. With everything zero/false this is exactly the plain
/// `vectorCost` above.
pub fn vectorCostMem(
    model: *const Model,
    op: BinOp,
    elem_is_float: bool,
    lanes: u8,
    distinct_operand_vectors: u8,
    coalesced_operand_vectors: u8,
    chained_operand_vectors: u8,
    result_coalesced_store: bool,
    result_chained: bool,
) f64 {
    std.debug.assert(coalesced_operand_vectors + chained_operand_vectors <= distinct_operand_vectors);
    // A result cannot both stay in a register (chained into a consumer) and be written to memory (a
    // coalesced store); the vectorizer picks one, preferring the store when the result is stored.
    std.debug.assert(!(result_chained and result_coalesced_store));
    const lanes_f: f64 = @floatFromInt(lanes);
    // The one vector op, same per-type weight as its scalar sibling (a <N x f32> mul is priced by the
    // FP throughput, a <N x i32> mul by the integer throughput).
    const vec_op = opWeight(model, op, elem_is_float);
    // Operand vectors that must still be packed (lane-insert per lane), after subtracting those that
    // coalesce to one wide load apiece and those reused for free from an earlier group.
    const packed_vectors: f64 = @floatFromInt(distinct_operand_vectors - coalesced_operand_vectors - chained_operand_vectors);
    const pack = packed_vectors * lanes_f * PACK_COST_PER_LANE;
    const loads = @as(f64, @floatFromInt(coalesced_operand_vectors)) * VECTOR_MEM_COST;
    // Chained results cost nothing (no extract, no store, they ride a vector register into the next
    // group); a coalesced store is one wide store; otherwise `lanes` lane-extracts.
    const unpack = if (result_chained)
        0.0
    else if (result_coalesced_store)
        VECTOR_MEM_COST
    else
        lanes_f * UNPACK_COST_PER_LANE;
    return vec_op + pack + loads + unpack;
}

/// The gate: is SLP-vectorizing this group profitable for `model`? True only when the vector form is
/// strictly cheaper than the scalar form (a near-tie stays scalar, so the transform never runs on a
/// break-even shape). `distinct_operand_vectors` is how many operand vectors the group must pack
/// (2 for a binary op with distinct sides, 1 when both sides are the same value across all lanes,
/// e.g. `a*a`). `elem_is_float` routes the scalar cost to the FP/SIMD ports.
pub fn slpProfitable(
    model: *const Model,
    group_op: BinOp,
    elem_is_float: bool,
    lanes: u8,
    distinct_operand_vectors: u8,
) bool {
    std.debug.assert(lanes >= 2);
    std.debug.assert(distinct_operand_vectors >= 1 and distinct_operand_vectors <= 2);
    const scalar = scalarCost(model, group_op, elem_is_float, lanes);
    const vector = vectorCost(model, group_op, elem_is_float, lanes, distinct_operand_vectors);
    return vector < scalar;
}

/// The gate WITH memory coalescing: like `slpProfitable`, but the vector form's cost reflects that
/// `coalesced_operand_vectors` of the operands are contiguous loads (one wide load each, not a pack),
/// that when `result_coalesced_store` the result lanes go to contiguous memory as one wide store (not
/// an unpack), and that when `result_chained` the result stays live in a vector register feeding a
/// fusable consumer group (no unpack at all, the chain-reuse vmap elides the extracts). This is what
/// makes a group whose operands/results ride adjacent memory OR chain into a consumer profitable on a
/// wide core that would decline the same group when it must pack/unpack. A register-input group whose
/// result is stored or returned has `coalesced_operand_vectors == 0`, `result_coalesced_store` per
/// its store, and `result_chained == false`, so it stays declined exactly as the throughput fix
/// intends; only a producer whose result genuinely chains into a fusable consumer earns the free
/// unpack (the SAXPY `a*b+a` mul group).
pub fn slpProfitableMem(
    model: *const Model,
    group_op: BinOp,
    elem_is_float: bool,
    lanes: u8,
    distinct_operand_vectors: u8,
    coalesced_operand_vectors: u8,
    chained_operand_vectors: u8,
    result_coalesced_store: bool,
    result_chained: bool,
) bool {
    std.debug.assert(lanes >= 2);
    std.debug.assert(distinct_operand_vectors >= 1 and distinct_operand_vectors <= 2);
    std.debug.assert(coalesced_operand_vectors + chained_operand_vectors <= distinct_operand_vectors);
    const scalar = scalarCost(model, group_op, elem_is_float, lanes);
    const vector = vectorCostMem(model, group_op, elem_is_float, lanes, distinct_operand_vectors, coalesced_operand_vectors, chained_operand_vectors, result_coalesced_store, result_chained);
    return vector < scalar;
}

const registry = @import("registry.zig");

test "ampere cheap-add SLP is declined: the wide OoO core parallelizes the scalar adds" {
    // The measured 0.94x regression shape: an f32 ADD group, 2 distinct operand vectors, on the
    // 4-wide out-of-order ampere core (2 fpsimd ports). The scalar adds fly across the ports, so the
    // pack/unpack overhead of the vector form loses. Checked at both the 4-lane NEON width and a
    // hypothetical wider width: a cheap op on a wide core stays unprofitable as lanes grow, since
    // the pack cost grows with lanes too.
    const altra = registry.modelFor(.@"ampere-altra");
    try std.testing.expect(!slpProfitable(altra, .add, true, 4, 2));
    try std.testing.expect(!slpProfitable(altra, .add, true, 8, 2));
    // Integer cheap add on the wide core is likewise unprofitable.
    try std.testing.expect(!slpProfitable(altra, .add, false, 4, 2));
}

test "et-soc SLP stays profitable: the single-issue core serializes the scalar ops" {
    // The narrow in-order et-soc core (issue_width 1): 8 serialized scalar ops collapse to one
    // 8-lane VPU op, a real win even with pack overhead. Both the cheap shapes (add/xor, needed by
    // the i32 differential kernels) and the expensive mul shape (the square-add / mul-add kernels)
    // must come out profitable.
    const etsoc = registry.modelFor(.@"et-soc");
    try std.testing.expect(slpProfitable(etsoc, .add, true, 8, 2)); // f32 add group
    try std.testing.expect(slpProfitable(etsoc, .add, false, 8, 2)); // i32 add group
    try std.testing.expect(slpProfitable(etsoc, .bit_xor, false, 8, 2)); // i32 xor group
    try std.testing.expect(slpProfitable(etsoc, .mul, true, 8, 2)); // f32 mul group
    try std.testing.expect(slpProfitable(etsoc, .mul, false, 8, 2)); // i32 mul group
    // Same-operand mul (a*a, one distinct operand vector) is profitable with even more margin.
    try std.testing.expect(slpProfitable(etsoc, .mul, true, 8, 1));
}

test "et-soc prices an i32 mul and an f32 mul differently, and the i32 mul SLP group is strongly profitable" {
    // The whole point of the per-type throughput. On et-soc the integer MulDiv is async multicycle
    // (throughput 8 == latency, FE-Intpipe 3.4.2) while the VPU FP multiply-add is pipelined
    // (throughput 1, Minion VPU Spec 2.1). So the SAME BinOp `.mul` weighs 8x more as an i32 op than
    // as an f32 op. This is a direct query of the model, pinning the split.
    const etsoc = registry.modelFor(.@"et-soc");
    try std.testing.expectEqual(@as(f64, 8.0), opWeight(etsoc, .mul, false)); // i32 mul: async MulDiv
    try std.testing.expectEqual(@as(f64, 1.0), opWeight(etsoc, .mul, true)); // f32 mul: pipelined VPU
    try std.testing.expect(opWeight(etsoc, .mul, false) > opWeight(etsoc, .mul, true));

    // The i32 mul SLP group is STRONGLY profitable despite the correct high price: on the single-issue
    // in-order core the 8 lanes serialize at 8 cyc each, scalar cost = (8 lanes / 1 port) * 8 = 64,
    // versus a vector cost of one i32 vector mul (8) plus pack/unpack, far below 64. The high per-lane
    // price makes the win LARGER here, it does not flip the decision.
    try std.testing.expectEqual(@as(f64, 64.0), scalarCost(etsoc, .mul, false, 8));
    try std.testing.expect(slpProfitable(etsoc, .mul, false, 8, 2));
    // And the vector cost of the i32 group is priced by the i32 weight (8), strictly above the f32
    // group's vector cost (priced by 1): type-correct, not type-blind.
    try std.testing.expect(vectorCost(etsoc, .mul, false, 8, 2) > vectorCost(etsoc, .mul, true, 8, 2));
}

test "the flagged bug fixed: a register-input mul SLP group on ampere is declined under throughput" {
    // The regression this whole change fixes. Ampere's Neoverse N1 has a PIPELINED multiplier: an
    // independent mul issues every cycle (throughput 1) though its result takes 4 cycles (latency).
    // Under the old latency weight (4), a register-input mul SLP group looked expensive-per-lane and
    // came out PROFITABLE, so the vectorizer packed 8 muls into a NEON group whose pack/unpack
    // overhead was not actually paid for (the scalar muls already fly across the ports). Under the
    // throughput weight (1) the scalar cost drops and the group is correctly DECLINED: register-input
    // mul SLP is not worth it on this core. We assert both the fixed verdict AND that the OLD latency
    // weight would have judged it profitable, pinning the bug in place.
    const altra = registry.modelFor(.@"ampere-altra");

    // Fixed behavior: the f32 mul group (the ONLY mul SLP path ampere ever runs, since NEON has no
    // <N x i32> lowering so the integer path is gated off) binds to the 2-port FP/SIMD unit and is
    // now DECLINED at the NEON width and a wider hypothetical width.
    try std.testing.expect(!slpProfitable(altra, .mul, true, 4, 2)); // f32, 4 lanes
    try std.testing.expect(!slpProfitable(altra, .mul, true, 8, 2)); // f32, 8 lanes
    // Same-operand (a*a, one operand vector) is likewise declined even with less pack overhead.
    try std.testing.expect(!slpProfitable(altra, .mul, true, 4, 1));

    // A cheap add is declined too (unchanged: add throughput == latency == 1).
    try std.testing.expect(!slpProfitable(altra, .add, true, 4, 2));

    // Note the i32 mul group is NOT declined by the cost model: an integer mul binds to the single
    // muldiv port (1), so `lanes` scalar muls serialize on that one port and vectorizing is a genuine
    // win even at throughput weight 1. This is moot on ampere (the i32 SLP path is gated off for NEON,
    // see vectorize.runModel's allow_i32), but it shows the cost model is reasoning about ports, not
    // just the op weight.
    try std.testing.expect(slpProfitable(altra, .mul, false, 4, 2)); // i32, 1 muldiv port: profitable

    // Proof the fix matters: recomputing the same group with the LATENCY weight (mul latency 4)
    // instead of the throughput weight (1) flips the mul group to profitable, which is the old bug.
    // scalar = (lanes/ports) * weight; vector = weight + pack + unpack. With ports = 2 (fpsimd):
    const lanes: f64 = 4;
    const ports: f64 = 2;
    const pack = 2.0 * lanes * PACK_COST_PER_LANE;
    const unpack = lanes * UNPACK_COST_PER_LANE;
    const mul_latency: f64 = @floatFromInt(altra.latency(arithOpcode(.mul))); // 4
    const scalar_lat = (lanes / ports) * mul_latency; // 8
    const vector_lat = mul_latency + pack + unpack; // 4 + 1.6 + 0.8 = 6.4
    try std.testing.expect(vector_lat < scalar_lat); // old weight: profitable (the bug)
    const mul_tput: f64 = @floatFromInt(altra.throughput(arithOpcode(.mul), true)); // 1 (f32 mul, pipelined FP path)
    const scalar_tput = (lanes / ports) * mul_tput; // 2
    const vector_tput = mul_tput + pack + unpack; // 1 + 1.6 + 0.8 = 3.4
    try std.testing.expect(vector_tput > scalar_tput); // new weight: declined (fixed)
}

test "memory coalescing flips a wide-core cheap-add group from declined to profitable" {
    // The exact ampere `slp-adds` shape (f32 ADD, 2 distinct operand vectors, 4 lanes) is declined
    // when it must pack/unpack (register inputs). The SAME group becomes profitable once its two
    // operands are contiguous loads AND its results go to contiguous stores: the pack/unpack
    // overhead collapses to three cheap wide memory ops.
    const altra = registry.modelFor(.@"ampere-altra");
    try std.testing.expect(!slpProfitableMem(altra, .add, true, 4, 2, 0, 0, false, false)); // register inputs: declined
    try std.testing.expect(slpProfitableMem(altra, .add, true, 4, 2, 2, 0, true, false)); // both loads + store: profitable
    // Load coalescing alone (result still unpacked) is NOT enough on this cheap-add shape; the win
    // needs the result store to coalesce too. This documents why the vectorizer prices both sides.
    try std.testing.expect(!slpProfitableMem(altra, .add, true, 4, 2, 2, 0, false, false));
    // A chained (already-live) operand plus one coalesced load plus a coalesced store is profitable:
    // this is the multiply-add shape's second group, where the mul result feeds the add for free.
    try std.testing.expect(slpProfitableMem(altra, .add, true, 4, 2, 1, 1, true, false));
}

test "result-chaining credits the free unpack: the SAXPY mul group becomes profitable on ampere" {
    // The collateral the result-chaining credit recovers. The ampere `a[i]*b[i]+a[i]` (SAXPY) mul
    // group has two contiguous-load operands (a, b) but its result is NOT stored: it feeds the
    // following add group lane-for-lane, so the chain-reuse vmap elides the extracts (they DCE away).
    // The greedy per-group model would otherwise charge that mul group a full per-lane unpack and
    // decline it, killing a real NEON SAXPY. Crediting the chained result (unpack cost 0) flips it
    // profitable: scalar = (4/2 ports)*1 = 2.0, vector = 1(mul) + 0(pack, both operands coalesce) +
    // 2*0.2(loads) + 0(chained result) = 1.4 < 2.0.
    const altra = registry.modelFor(.@"ampere-altra");
    try std.testing.expect(slpProfitableMem(altra, .mul, true, 4, 2, 2, 0, false, true)); // two loads + chained result

    // The throughput fix is preserved: the same register-input mul group (no coalesceable memory) whose
    // result feeds a STORE (result_coalesced_store) or a RETURN (neither store nor chain) stays
    // DECLINED. The credit only removes the unpack, never the packs, so a register mul whose two
    // operand vectors must be packed loses regardless of how its result leaves the group.
    try std.testing.expect(!slpProfitableMem(altra, .mul, true, 4, 2, 0, 0, true, false)); // register mul -> coalesced store: declined
    try std.testing.expect(!slpProfitableMem(altra, .mul, true, 4, 2, 0, 0, false, false)); // register mul -> return: declined
    // Even a register mul whose result CHAINS is still declined: the two packs (1.6) dominate. Only
    // when the operands ALSO ride memory (wide loads, 0.4) does the chained mul group win. This is why
    // the credit re-enables the memory SAXPY without ever rescuing a register-input mul SLP group.
    try std.testing.expect(!slpProfitableMem(altra, .mul, true, 4, 2, 0, 0, false, true)); // register mul -> chained: still declined
    // And the register cheap-add stays declined under every result disposition.
    try std.testing.expect(!slpProfitableMem(altra, .add, true, 4, 2, 0, 0, false, true));
}

test "memory coalescing never rescues a register-input group (no coalesceable memory)" {
    // With zero coalesced operands and no coalesced store, `slpProfitableMem` is identical to the
    // plain gate, so a register-input cheap add stays declined on the wide core at every width.
    const altra = registry.modelFor(.@"ampere-altra");
    try std.testing.expectEqual(slpProfitable(altra, .add, true, 4, 2), slpProfitableMem(altra, .add, true, 4, 2, 0, 0, false, false));
    try std.testing.expectEqual(slpProfitable(altra, .add, true, 8, 2), slpProfitableMem(altra, .add, true, 8, 2, 0, 0, false, false));
    try std.testing.expect(!slpProfitableMem(altra, .add, true, 8, 2, 0, 0, false, false));
}

test "coalesced vector cost is strictly below the packed cost for the same group" {
    // One wide load beats `lanes` lane-inserts; one wide store beats `lanes` lane-extracts.
    const etsoc = registry.modelFor(.@"et-soc");
    try std.testing.expect(vectorCostMem(etsoc, .add, false, 8, 2, 2, 0, true, false) < vectorCostMem(etsoc, .add, false, 8, 2, 0, 0, false, false));
    // A chained result (no unpack at all) is cheaper still than a coalesced store (one wide store).
    try std.testing.expect(vectorCostMem(etsoc, .add, false, 8, 2, 2, 0, false, true) < vectorCostMem(etsoc, .add, false, 8, 2, 2, 0, true, false));
    // And the zero-coalescing form matches the plain vectorCost exactly.
    try std.testing.expectEqual(vectorCost(etsoc, .add, false, 8, 2), vectorCostMem(etsoc, .add, false, 8, 2, 0, 0, false, false));
}

test "cost ordering is sane: vector cost rises with distinct operand vectors and with lanes" {
    const etsoc = registry.modelFor(.@"et-soc");
    try std.testing.expect(vectorCost(etsoc, .add, false, 8, 2) > vectorCost(etsoc, .add, false, 8, 1));
    try std.testing.expect(vectorCost(etsoc, .add, false, 8, 2) > vectorCost(etsoc, .add, false, 4, 2));
    // Scalar cost of a serial in-order core scales straight with lanes.
    try std.testing.expectEqual(
        scalarCost(etsoc, .add, false, 8),
        2.0 * scalarCost(etsoc, .add, false, 4),
    );
}
