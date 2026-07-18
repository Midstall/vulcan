//! Model-driven matmul-nest recognition (et-soc VPU tensor tile). Detects the naive triply-nested
//! `for i,j,k: C[i][j] += A[i][k] * B[k][j]` loop and raises it to the fixed-tile `matmul` IR op, which
//! only lowers on the et-soc VPU. Task 1 matched ONLY the loop SKELETON: three perfectly-nested loops
//! with pure-test headers, constant tile bounds read from the icmp rhs, unit-stepped 0-based induction
//! variables, and a whole-function gate proving the nest is the entire function. Task 2 additionally
//! proved the k-loop body actually computes an fp32 multiply-accumulate reduction from two distinct
//! element pointers, with nothing else in the body, and that the finished accumulator is stored to a
//! pointer above the k-loop (the C write), otherwise a same-shaped but differently-bodied 3-nest would
//! also match. Task 3 traced the k-loop element pointers outward to recover the A/B/C base pointers and
//! prove a single consistent row-major fp32 layout. Task 4 (this file, now) adds the transform: `apply`
//! replaces the recognized nest with the `matmul` op, and `recognizeNest` gates on the same tile-count
//! cap the isel backend enforces, so recognition never raises a matmul the backend would reject. Not
//! yet wired into the `optimize` pipeline (a later task does that plus the sysemu differential).
//!
//! Plan 19 generalized the body match from fp32-only to also recognize int8/uint8 (and MIXED operand
//! signedness) matmul nests, raising each to the `matmul` op with the correct `dtype` and, for mixed
//! signedness, the plan-16 `input_signs` override. The int8 body is the 2D analogue of dotprod.zig's
//! int8 reduction: an i32 accumulator over `convert_i32(load_i8(pa)) * convert_i32(load_i8(pb))` with
//! INTEGER mul/add (fp32's body has no converts and floating mul/add). Per-operand signedness is read
//! from the two 8-bit load types (both signed -> .int8, both unsigned -> .uint8, one of each -> .int8
//! with `input_signs`). C is always a 32-bit accumulator, so the A/B element size (`input_elem`: 1 for
//! int8, 2 for fp16, 4 for fp32) is split from the fixed 4-byte C `output_elem`.
//!
//! This revision (the fp16 follow-up, now that the IR's `FloatKind` has grown an `f16`) adds a THIRD
//! body shape: an f32 accumulator over `convert_f32(load_f16(pa)) * convert_f32(load_f16(pb))` with
//! FLOATING mul/add. It is the floating analogue of the int8 body (converts feed the mul, rather than
//! direct loads) but shares the int8/uint8 body's f32-vs-i32 accumulator disambiguation problem in
//! reverse: both the real fp32 body and the fp16 body have an f32 accumulator, so `matchBody` peeks at
//! whether the mul's operands are direct loads (fp32) or converts (fp16) to tell them apart, then the
//! fp16 arm additionally confirms each convert's source is genuinely f16-typed (`input_signs` stays
//! null: floats have no signedness).
//!
//! SKIP-IF-UNSURE: recognition is conservative. Any deviation from the exact expected shape returns
//! null (a missed optimization, never a false positive), because raising a non-matmul nest to the
//! tensor op would miscompile. Every `return null` is commented with the reason it is not the shape.
//!
//! The recognizer is the 2D generalization of `dotprod.zig`: `run` -> `loops.analyze` ->
//! `computeDefBlocks` -> `recognizeNest` -> collect Plans -> `apply`. It reuses that file's
//! pure-test-header idiom, latch/back-edge matching, unit-step matcher, and small type helpers; Task 2's
//! `matchBody` is the 2D analogue of dotprod.zig's backward accumulator-update match (dotprod.zig:182-263).

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const loops = @import("../loops.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Type = ir.types.Type;
const MatMulType = ir.function.MatMulType;
const InputSigns = ir.function.InputSigns;

pub const Error = std.mem.Allocator.Error;

/// The most outer-header params a surrounded nest may thread out on its loop-exit edge and still be
/// reconstructed. A wider exit edge is left as loops (a missed optimization, never a miscompile). Real
/// matmul nests thread out little or nothing, so this bound is generous.
const max_exit_args = 8;

/// How `apply` reconstructs one arg of the outer loop's exit edge when it redirects the preheader
/// straight to `outer.exit`. Both cases are provably the exit value of a matched outer-header param
/// (see the reconstruction in `recognizeNest`), so the redirect passes exactly what `outer.exit`
/// expected without executing the (now orphaned) loop.
const ExitArg = union(enum) {
    /// The outer induction variable at loop exit. A 0-based unit-stepped loop bounded by `i < m` exits
    /// exactly when the iv reaches `m`, so `apply` materializes `iconst(m)` in the preheader for it.
    iv_bound,
    /// A loop-invariant outer-header param (the latch threads it back unchanged), so its exit value is
    /// its loop-entry value: the preheader's initial jump arg at this param index.
    initial: usize,
};

/// A vetted matmul nest. Task 1 fills the loop skeleton (the three loop headers, their
/// induction-variable header-param indices, the constant tile bounds, the three latches, the immediate
/// body blocks, and the outermost preheader). Task 2 fills the body dataflow: which k-loop param is the
/// fp32 accumulator, which two are the distinct A/B element pointers, and the store that writes the
/// finished accumulator to C. Task 3 extends it with the A/B/C base pointers (below); Task 4 (`apply`)
/// only reads those fields, adding nothing new to the Plan itself.
const Plan = struct {
    /// The three loop headers, outermost (i) to innermost (k). Each is a pure-test header.
    i_header: Block,
    j_header: Block,
    k_header: Block,
    /// The immediate in-loop body block each header branches to (the block the header params are
    /// threaded straight through into). `k_body` is the innermost body Tasks 2-3 analyze.
    i_body: Block,
    j_body: Block,
    k_body: Block,
    /// The single back-edge block of each loop (the latch that steps the induction variable and
    /// jumps back to its header).
    i_latch: Block,
    j_latch: Block,
    k_latch: Block,
    /// Which header param is the induction variable at each level (the icmp lhs position).
    i_iv: usize,
    j_iv: usize,
    k_iv: usize,
    /// The three compile-time tile bounds (the icmp rhs iconsts), 1..=65535.
    m: u16,
    n: u16,
    k: u16,
    /// The outermost loop's preheader (the block that enters the i-loop with the initial state).
    preheader: Block,

    /// Task 2: the k-loop's fp32 multiply-accumulate reduction. Which k_header/k_body param (the two
    /// blocks share a param index space, per the in-loop edge's straight-through pass) is the running
    /// accumulator, and which two are the distinct element pointers the product is loaded from (`pa`
    /// the multiply's left operand, `pb` its right).
    acc_k_param: usize,
    pa_k_param: usize,
    pb_k_param: usize,
    /// The `store` instruction, in the k-loop's exit block, that writes the finished accumulator out.
    c_store: Inst,
    /// The pointer the store writes through: a pointer-typed block param living above the k-loop (it is
    /// NOT threaded through the k-loop's own params). Task 3 traces this to the C base and stride.
    c_ptr: Value,

    /// Task 3: the recovered row-major matmul base pointers and A/B element size. `a`/`b`/`c` are the
    /// A/B/C base pointers, each a value that dominates the whole nest (a function param or a preheader
    /// instruction), recovered by tracing the k-loop element pointers outward through their in-loop steps
    /// and per-loop resets. `input_elem` is the A/B element size in bytes proven from the A inner stride
    /// (4 for fp32, 2 for fp16, 1 for int8/uint8); C is always a 32-bit accumulator (a fixed 4-byte
    /// `output_elem`).
    /// The layout proven consistent: A(m x k), B(k x n), C(m x n) row-major.
    a: Value,
    b: Value,
    c: Value,
    input_elem: u32,

    /// Plan 19 (+ the fp16 follow-up): the recognized matmul element dtype and, for MIXED operand
    /// signedness, the plan-16 per-operand override. `dtype` is `.fp32` for the direct-load floating
    /// body, `.fp16` for the convert-then-float-multiply body, `.int8` for both-signed or mixed int8,
    /// `.uint8` for both-unsigned int8. `input_signs` is non-null ONLY for the mixed int8 case (one
    /// operand signed, one unsigned), and then `dtype` is `.int8` per the plan-16 spelling; `apply`
    /// routes on it to `appendMatmulSigned` vs `appendMatmul`. fp16 never carries `input_signs` (floats
    /// have no signedness).
    dtype: MatMulType,
    input_signs: ?InputSigns,

    /// Task 2 (memory accumulator): whether the nest accumulates INTO the existing C tile
    /// (`C += A*B`, from a `load(C[i][j])`-seeded reduction) and so lowers to matmul(accumulate=true),
    /// vs a fresh zero-seeded reduction (accumulate=false). Independent of `embedded`: a memory
    /// accumulator can be whole-function (embedded=false) or surrounded (embedded=true).
    accumulate: bool,

    /// Task 2 (non-whole-function): where `apply` redirects the preheader (`outer.exit`, the
    /// continuation reached when the outer loop finishes), the reconstructed args it must pass there,
    /// and whether the matmul must be `embedded`. `embedded` is true ONLY when the nest is surrounded
    /// by other code (a non-trivial continuation): then live values may straddle the matmul, so the
    /// self-contained save/restore lowering is required. A whole-function nest (bare `ret void`
    /// continuation, nothing live across) keeps `embedded=false`, the cheaper lowering, so the existing
    /// whole-function sysemu differentials stay byte-for-byte identical.
    outer_exit: Block,
    exit_args_buf: [max_exit_args]ExitArg,
    exit_args_len: usize,
    embedded: bool,
};

/// Recognize the matmul nest `func` contains, when `model` is an et-soc VPU target (the only place the
/// `matmul` op lowers), and transform it. Returns whether a nest was matched and applied. Recognition
/// runs to completion (collect) before `apply` mutates anything (kept as a collect-then-apply structure
/// so the mutation lands after recognition, mirroring `dotprod.zig`).
pub fn run(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) Error!bool {
    // The matmul op only lowers on the et-soc VPU; never raise a nest to it on any other target.
    if (!model.vpu()) return false;

    var info = try loops.analyze(allocator, func);
    defer info.deinit(allocator);

    // Value -> defining block index, for the loop-invariance checks during recognition.
    const def_block = try computeDefBlocks(allocator, func);
    defer allocator.free(def_block);

    var plans: std.ArrayList(Plan) = .empty;
    defer plans.deinit(allocator);

    // A single well-formed function holds at most one whole-function matmul nest; recognizeNest
    // scans the flat loop list for the exact 3-loop chain and gates on the whole function.
    if (try recognizeNest(allocator, func, &info, def_block)) |plan| {
        try plans.append(allocator, plan);
    }

    // Task 4: apply every recognized plan. The whole-function gate means there is at most one plan in
    // practice, but keep the loop (mirrors dotprod.zig's collect-then-apply structure).
    for (plans.items) |*plan| try apply(func, plan);
    return plans.items.len != 0;
}

/// Task 4 (+ Task 2 non-whole-function): replace the recognized nest with a single `matmul` op. The
/// transform mutates only the preheader: append the matmul (its `a`/`b`/`c` operands dominate the
/// preheader, proven by Task 3's base-dominance checks, so they are in scope), then replace the
/// preheader's `jump i_header(...)` terminator with a jump straight to the continuation `outer.exit`,
/// passing the reconstructed exit args. This orphans the entire nest (`i_header` and every block below
/// it become unreachable from the preheader); orphan-and-redirect is the supported idiom here (the IR
/// has no block-deletion primitive), and reachability-aware analyses (dominators, loops) and codegen
/// already ignore unreachable blocks. The preheader's own setup instructions (the `iconst` bounds, the
/// initial `a`/`c` pointer args) become dead but harmless; they are not deleted either.
///
/// ONE code path serves both cases (recognizeNest picked `embedded` and the exit args): a whole-function
/// nest's `outer.exit` is the bare `ret void` block, so jumping to it with no args is equivalent to the
/// old `ret null` in the preheader and keeps the whole-function lowering byte-identical; a surrounded
/// nest jumps to real continuation code with its live values reconstructed and uses the `embedded`
/// matmul so those values survive the tensor unit's register clobber.
///
/// `plan`'s stored Block/Value handles are all stable across this mutation: the matmul builders and the
/// `iconst` materialization only APPEND (they do not touch existing blocks/values), and `setJump`
/// replaces the preheader's terminator in place, so none of the Plan's other blocks or the a/b/c values
/// move.
fn apply(func: *Function, plan: *const Plan) Error!void {
    // Emit the matmul into the preheader. A surrounded nest needs the self-contained `embedded` lowering
    // so a value live across the matmul survives the tensor unit's full register clobber;
    // `appendMatmulEmbedded` handles BOTH the plain (input_signs null) and mixed-signedness cases. The
    // whole-function case keeps the cheaper plain lowering: `appendMatmulSigned` for a mixed-signedness
    // nest (dtype == .int8 plus the plan-16 override), else the symmetric `appendMatmul`.
    if (plan.embedded) {
        try func.appendMatmulEmbedded(plan.preheader, plan.a, plan.b, plan.c, plan.m, plan.n, plan.k, plan.dtype, plan.accumulate, plan.input_signs);
    } else if (plan.input_signs) |signs| {
        try func.appendMatmulSigned(plan.preheader, plan.a, plan.b, plan.c, plan.m, plan.n, plan.k, plan.dtype, plan.accumulate, signs);
    } else {
        try func.appendMatmul(plan.preheader, plan.a, plan.b, plan.c, plan.m, plan.n, plan.k, plan.dtype, plan.accumulate);
    }

    // The preheader still holds its `jump i_header(initial state)`; recognizeNest (via matchLoop) proved
    // it jumps into the outer header, so the initial arg at a param index is that param's loop-entry
    // value. Read it BEFORE `setJump` overwrites the terminator; the reconstructed Values are copied into
    // `args_buf` in the loop, so the later `setJump` (which re-interns the arg pool) cannot invalidate them.
    const init_args = switch (func.terminator(plan.preheader) orelse unreachable) {
        .jump => |j| func.blockArgs(j),
        // recognizeNest proved the preheader jumps into the header, so a ret here is impossible.
        .ret => unreachable,
    };
    var args_buf: [max_exit_args]Value = undefined;
    // The iv is an i32 counter (matchLoop proved `isI32`), so the exit-value constant is an i32 iconst.
    const i32_t = if (plan.exit_args_len != 0) try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }) else undefined;
    for (plan.exit_args_buf[0..plan.exit_args_len], 0..) |ea, i| {
        args_buf[i] = switch (ea) {
            // The outer iv equals the constant bound m at loop exit (0-based unit-stepped, `i < m`).
            .iv_bound => try func.appendInst(plan.preheader, i32_t, .{ .iconst = plan.m }),
            // A loop-invariant outer-header param: its exit value is its loop-entry (initial) value.
            .initial => |j| init_args[j],
        };
    }
    try func.setJump(plan.preheader, plan.outer_exit, args_buf[0..plan.exit_args_len]);
}

/// The block index that defines each value (a block param's block, or the block holding the defining
/// instruction). Indexed by `@intFromEnum(value)`. Verbatim from `dotprod.zig`.
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

/// One structurally-matched loop level: its header, immediate body, latch, exit target, induction
/// variable param index, and constant bound. Purely local (no allocation).
const LoopMatch = struct {
    header: Block,
    body: Block,
    latch: Block,
    exit: Block,
    preheader: Block,
    iv: usize,
    bound: u16,
    param_count: usize,
};

/// Match a 3-perfectly-nested loop and return the partial Plan, or null on ANY deviation. The expected
/// shape (each level a pure-test header, 0-based unit-stepped iv, params threaded straight through):
///
///   preheader:              ... -> i_header(0, ...)
///   i_header(i, ...):       if i < m -> i_body(i, ...) else <exit>
///   i_body:                 ... -> j_header(0, ...)                 // preheader of the j-loop
///   j_header(j, ...):       if j < n -> j_body(j, ...) else i_latch(...)
///   j_body:                 ... -> k_header(0, ...)                 // preheader of the k-loop
///   k_header(kk, ...):      if kk < k -> k_body(kk, ...) else j_latch(...)
///   k_body:                 ... ; kk+1 ; -> k_header(...)           // the innermost body + k-latch
///   j_latch:                j+1 ; -> j_header(...)
///   i_latch:                i+1 ; -> i_header(...)
///
/// `allocator` is unused in Task 1 (recognition is allocation-free) but is kept in the signature so
/// Tasks 2-4 can grow the analysis without a churny signature change.
fn recognizeNest(allocator: std.mem.Allocator, func: *const Function, info: *const loops.LoopInfo, def_block: []const u32) Error!?Plan {
    const l = info.loops;
    // A matmul is a perfect nest of EXACTLY three loops; any other count is not this shape.
    if (l.len != 3) return null;

    // Order the three loops by body size, largest first. In a proper chain the outer body strictly
    // contains the middle, which strictly contains the inner, so popcount orders them uniquely.
    var idx = [3]usize{ 0, 1, 2 };
    const pc = [3]usize{ popcount(l[0].body), popcount(l[1].body), popcount(l[2].body) };
    if (pc[idx[0]] < pc[idx[1]]) std.mem.swap(usize, &idx[0], &idx[1]);
    if (pc[idx[1]] < pc[idx[2]]) std.mem.swap(usize, &idx[1], &idx[2]);
    if (pc[idx[0]] < pc[idx[1]]) std.mem.swap(usize, &idx[0], &idx[1]);
    const outer_l = &l[idx[0]];
    const middle_l = &l[idx[1]];
    const inner_l = &l[idx[2]];

    // The three loops must form a single containment chain outer > middle > inner. If any two are
    // siblings (neither strictly contains the other) this is not a perfect 3-nest.
    if (!strictSubset(inner_l.body, middle_l.body)) return null;
    if (!strictSubset(middle_l.body, outer_l.body)) return null;

    const outer = matchLoop(func, outer_l, def_block) orelse return null;
    const middle = matchLoop(func, middle_l, def_block) orelse return null;
    const inner = matchLoop(func, inner_l, def_block) orelse return null;

    // Task 2 REGION-INTEGRITY gate (replaces Task 1's whole-function gate). The nest need no longer be
    // the entire function: arbitrary LOOP-FREE code may sit before the preheader and after the exit, with
    // live values straddling the nest. What we must prove instead is that the nest is a single-entry /
    // single-exit REGION, so that orphaning it (redirecting the preheader straight to outer.exit) can
    // neither strand surrounding code nor let surrounding code fall into the now-dead nest.
    //
    // The REGION INTERIOR is the scaffolding EXCEPT outer.exit: outer.exit is the CONTINUATION (the first
    // block of the code after the nest, which in the whole-function case is the bare `ret void`). Prove:
    //   (a) SINGLE ENTRY  no NON-interior block targets an interior block other than the preheader.
    //   (b) SINGLE EXIT   no interior block leaves the interior except outer.header -> outer.exit.
    //   (c) NO CALLS      no call inside the interior (the tensor lowering cannot absorb a side effect;
    //                     a call in the SURROUNDING code is fine and left untouched).
    // Surrounding code is loop-free by construction: recognizeNest already required info.loops.len == 3
    // (exactly the three nest loops), so no other loop can exist. Hence no separate acyclicity proof is
    // needed; the two sanctioned edges are the only way in and out.
    const known = [_]Block{
        outer.preheader,
        outer.header,
        outer.body,
        outer.latch,
        outer.exit,
        middle.header,
        middle.body,
        middle.latch,
        middle.exit,
        inner.header,
        inner.body,
        inner.latch,
        inner.exit,
    };
    // interior[bi]: block bi is a region-interior scaffolding block (everything in `known` but outer.exit).
    const interior = try allocator.alloc(bool, func.blockCount());
    defer allocator.free(interior);
    @memset(interior, false);
    for (known) |b| interior[@intFromEnum(b)] = true;
    interior[@intFromEnum(outer.exit)] = false; // outer.exit is the continuation, not interior

    for (0..func.blockCount()) |bi| {
        const b: Block = @enumFromInt(bi);
        const src_interior = interior[bi];
        const src_is_outer_header = b == outer.header;
        if (src_interior) {
            for (func.blockInsts(b)) |inst| switch (func.opcode(inst)) {
                // (c) a call inside the nest is an arbitrary side effect the tensor lowering cannot absorb.
                .call, .call_indirect => return null,
                else => {},
            };
        }
        // A block's successors are its `if` edges plus its jump terminator target (the exact edges
        // cfg.zig derives). Every one must satisfy the region rule for `b`'s side of the boundary.
        for (func.blockInsts(b)) |inst| switch (func.opcode(inst)) {
            .@"if" => |cf| {
                // A stray branch into the nest interior, or out of it, breaks orphan-and-redirect.
                if (!regionEdgeOk(cf.then.target, src_interior, src_is_outer_header, interior, outer.preheader, outer.exit)) return null;
                if (!regionEdgeOk(cf.@"else".target, src_interior, src_is_outer_header, interior, outer.preheader, outer.exit)) return null;
            },
            else => {},
        };
        if (func.terminator(b)) |t| switch (t) {
            // A jump into the nest interior, or out of it, breaks orphan-and-redirect.
            .jump => |j| if (!regionEdgeOk(j.target, src_interior, src_is_outer_header, interior, outer.preheader, outer.exit)) return null,
            .ret => {},
        };
    }

    // Task 2: the loop SKELETON matches, but that alone is not "this is a matmul". A differently-bodied
    // 3-nest with this exact scaffolding would also reach here. Verify the k-loop body actually computes
    // an fp32 multiply-accumulate reduction and that the result is stored to C before accepting the nest.
    const body = matchBody(func, def_block, &inner) orelse return null;

    // Task 3 (THE CRUX): the body is a multiply-accumulate that stores, but the ADDRESSES are still
    // unproven. Trace the k-loop element pointers outward through their in-loop steps and per-loop resets
    // to prove the stepping is a single row-major fp32 matmul and recover the A/B/C base pointers. A wrong
    // stride or base here would raise the nest to a `matmul` op computing the wrong result, so this is
    // strict: any deviation returns null.
    const strides = matchStrides(func, def_block, outer_l.body, &outer, &middle, &inner, &body) orelse return null;

    // Task 4 CAP (final gate before accepting the nest): recognition must never raise a matmul the
    // isel backend will refuse, because doing so destroys the loop nest and then fails to lower the
    // op, turning a program that compiled as scalar loops into a hard compile error. Mirror EVERY
    // riscv64/isel.zig `.matmul` rejection for the recognized dtype (isel.zig around lines 2639-2657):
    //   1. N must be a multiple of 4 (the fma b_cols field encodes cols/4 - 1; all dtypes).
    //   2. K must be a multiple of `factor` (the fma acols field encodes K/factor; a partial packed
    //      column group has no representation). factor = 4 for int8/uint8 (64 int8 per SCP line / 16),
    //      2 for fp16 (32 f16 per SCP line / 16), 1 for fp32.
    //   3. the tile-count cap: TILE=16 rows/cols, K_TILE = 16*factor (fp32 16, fp16 32, int8/uint8 64),
    //      m_tiles*n_tiles*k_tiles <= 64.
    // An ineligible nest is left as loops.
    const factor: u32 = switch (body.dtype) {
        .fp32 => 1, // one fp32 element per 4-byte column slot
        .fp16 => 2, // two fp16 elements packed per 4-byte column slot
        .int8, .uint8 => 4, // four int8 elements packed per 4-byte column slot
    };
    if (middle.bound % 4 != 0) return null; // N not a multiple of 4: isel rejects, keep the loops
    if (inner.bound % factor != 0) return null; // K not a whole packed column group: isel rejects
    const k_tile: u32 = 16 * factor;
    const m_tiles: u32 = (@as(u32, outer.bound) + 15) / 16;
    const n_tiles: u32 = (@as(u32, middle.bound) + 15) / 16;
    const k_tiles: u32 = (@as(u32, inner.bound) + k_tile - 1) / k_tile;
    if (@as(u64, m_tiles) * n_tiles * k_tiles > 64) return null; // too big for the compile-time-unrolled op

    // Task 2 EXIT-ARG RECONSTRUCTION (done here, in recognition, so `apply` can never fail): when the
    // preheader is redirected straight to outer.exit, it must pass exactly the block args the outer
    // loop's own exit edge passed. Reconstruct each from a matched outer-header param, and refuse the
    // whole nest (keep the loops) if any exit arg is not reconstructible. The outer header is the proven
    // [icmp, if] shape, so re-read its `if` to recover the exit edge (LoopMatch kept only the target).
    const oh_insts = func.blockInsts(outer.header);
    const oiff = func.opcode(oh_insts[1]).@"if";
    const outer_exit_edge = if (oiff.then.target == outer.body)
        oiff.@"else"
    else if (oiff.@"else".target == outer.body)
        oiff.then
    else
        // Neither `if` successor is the matched outer body: the header changed shape underneath us.
        return null;
    const exit_args = func.blockArgs(outer_exit_edge);
    if (exit_args.len > max_exit_args) return null; // more live-out args than the reconstruction buffer holds: keep the loops
    const oh_params = func.blockParams(outer.header);
    // The outer latch back-edge args (one per outer-header param), used to prove a threaded-out param is
    // loop-invariant. matchLoop proved the latch ends in a jump into the header, so a null is impossible.
    const outer_back = jumpArgsOf(func, outer.latch) orelse return null;
    if (outer_back.len != oh_params.len) return null; // malformed back-edge: refuse
    var exit_args_buf: [max_exit_args]ExitArg = undefined;
    for (exit_args, 0..) |v, i| {
        const pj = paramIndex(oh_params, v) orelse return null; // exit arg is not an outer-header param: unreconstructible
        if (pj == outer.iv) {
            // The iv holds m at loop exit (0-based, unit-stepped, `i < m`); apply materializes iconst(m).
            exit_args_buf[i] = .iv_bound;
        } else {
            // A non-iv param is reconstructible only if it is loop-invariant, so its exit value equals
            // its initial value. Prove invariance directly (the latch threads this header param straight
            // back unchanged) rather than trusting the skeleton, since matchLoop only pins the iv step:
            // a stepped/recomputed param threaded out would otherwise be reconstructed to the wrong value.
            if (outer_back[pj] != oh_params[pj]) return null; // threaded-out param is not loop-invariant: keep the loops
            exit_args_buf[i] = .{ .initial = pj };
        }
    }

    // Task 2 EMBEDDED DISCRIMINATOR: a whole-function nest's continuation is a bare `ret void` with
    // nothing live across the matmul, so it keeps the cheaper plain lowering (and the existing
    // whole-function sysemu differentials stay byte-identical). Any other continuation (surrounding code
    // after the nest) may hold live values straddling the matmul, so it needs the self-contained
    // `embedded` lowering that saves/restores the tensor unit's clobbered registers.
    const embedded = !isTrivialVoidExit(func, outer.exit, exit_args);

    // Mirror riscv64/isel.zig's embedded-matmul reject (isel.zig ~line 3251: `if (mmv.embedded and
    // functionHasWideFloatValue(func)) return error.Unsupported`). The embedded lowering saves each live
    // float as 32 bits, so an f64 or 256-bit vector live across the matmul cannot be preserved. Raising an
    // embedded matmul into such a function would destroy the loops and THEN fail to lower, a hard compile
    // error where the scalar loops compiled fine. Gate on `embedded` exactly as isel does, so a
    // whole-function nest (never embedded) is unaffected. Same "never raise a matmul the backend refuses"
    // contract as the cap gate above.
    if (embedded and functionHasWideFloatValue(func)) return null;

    // Mirror riscv64/isel.zig's OTHER embedded-matmul reject (isel.zig ~line 3246: `if (mmv.embedded
    // and uses_f16) return error.Unsupported`, uses_f16 = `functionUsesF16(func)`). The embedded lowering
    // reserves the f16 software-convert scratch registers (x28..x31) as its own save/holder set, so an
    // embedded matmul in a function that uses f16 anywhere collides. Raising one would destroy the loops
    // and THEN fail to lower, the same hard compile error the wide-float gate above guards against. Gate
    // on `embedded` exactly as isel does, so a whole-function nest (never embedded, e.g. the recognized
    // fp16 whole-function matmul) is unaffected.
    if (embedded and ir.function.functionUsesF16(func)) return null;

    return Plan{
        .i_header = outer.header,
        .j_header = middle.header,
        .k_header = inner.header,
        .i_body = outer.body,
        .j_body = middle.body,
        .k_body = inner.body,
        .i_latch = outer.latch,
        .j_latch = middle.latch,
        .k_latch = inner.latch,
        .i_iv = outer.iv,
        .j_iv = middle.iv,
        .k_iv = inner.iv,
        .m = outer.bound,
        .n = middle.bound,
        .k = inner.bound,
        .preheader = outer.preheader,
        .acc_k_param = body.acc_k_param,
        .pa_k_param = body.pa_k_param,
        .pb_k_param = body.pb_k_param,
        .c_store = body.c_store,
        .c_ptr = body.c_ptr,
        .a = strides.a,
        .b = strides.b,
        .c = strides.c,
        .input_elem = body.input_elem,
        .dtype = body.dtype,
        .input_signs = body.input_signs,
        .accumulate = body.accumulate,
        .outer_exit = outer.exit,
        .exit_args_buf = exit_args_buf,
        .exit_args_len = exit_args.len,
        .embedded = embedded,
    };
}

/// One region-integrity check over a single control edge `src -> dst`, for the Task 2 region gate.
/// Returns false when the edge crosses the region boundary illegally. `src_interior` is whether the
/// source block is region-interior, `src_is_outer_header` whether it is outer.header, and `interior`
/// the interior membership bitset. The two sanctioned boundary edges are: any NON-interior block ->
/// preheader (the single entry), and outer.header -> outer.exit (the single exit).
fn regionEdgeOk(dst: Block, src_interior: bool, src_is_outer_header: bool, interior: []const bool, preheader: Block, outer_exit: Block) bool {
    const dst_interior = interior[@intFromEnum(dst)];
    if (src_interior) {
        // (b) SINGLE EXIT: an interior block may reach interior blocks freely; the only sanctioned way
        // out of the region is outer.header's edge to outer.exit (the loop-done continuation).
        if (dst_interior) return true;
        return src_is_outer_header and dst == outer_exit;
    }
    // (a) SINGLE ENTRY: a non-interior block may target other non-interior blocks (surrounding code)
    // or the preheader (the one entry), but never a deeper interior block (a header/body/latch), which
    // orphaning the nest would strand.
    if (!dst_interior) return true;
    return dst == preheader;
}

/// Whether `exit` is a bare `ret void` continuation with nothing threaded out: the whole-function nest's
/// continuation. Then no value is live across the matmul, so recognition keeps the cheaper non-embedded
/// lowering (and the existing whole-function sysemu differentials stay byte-identical). Any other
/// continuation (surrounding code, or threaded-out live values) is treated as surrounded -> embedded.
fn isTrivialVoidExit(func: *const Function, exit: Block, exit_args: []const Value) bool {
    if (exit_args.len != 0) return false; // live values threaded out: surrounded
    if (func.blockInsts(exit).len != 0) return false; // the continuation does real work: surrounded
    return switch (func.terminator(exit) orelse return false) {
        .ret => |v| v == null, // exactly `ret void`
        // A jump means more code follows the exit: surrounded, not a bare terminal return.
        .jump => false,
    };
}

/// Whether `func` holds any f64 or vector value. Mirrors riscv64/isel.zig's `functionHasWideFloatValue`
/// exactly: the embedded-matmul lowering saves each live scalar float as 32 bits, so an f64 or 256-bit
/// VPU vector live across the matmul cannot be preserved and isel rejects the embedded op. Recognition
/// must not raise an embedded matmul into such a function (it would destroy the loops then fail to lower).
fn functionHasWideFloatValue(func: *const Function) bool {
    var i: usize = 0;
    while (i < func.valueCount()) : (i += 1) {
        const v: Value = @enumFromInt(@as(u32, @intCast(i)));
        switch (func.types.type_kind(func.valueType(v))) {
            .float => |f| if (f == .f64) return true,
            .vector => return true,
            else => {},
        }
    }
    return false;
}

/// Prove one loop level has the pure-test-header idiom, a constant bound, a 0-based unit-stepped
/// induction variable, and a single back-edge latch. Returns the LoopMatch or null on ANY deviation.
fn matchLoop(func: *const Function, loop: *const loops.Loop, def_block: []const u32) ?LoopMatch {
    // Need a single preheader to read the induction variable's initial value.
    if (loop.preheader == null) return null;
    const header: Block = @enumFromInt(loop.header);
    const body_bits = loop.body;

    // Pure test header: exactly [icmp, if], the `if` testing the icmp, comparing `iv < bound`, and no
    // explicit branching terminator (the `if` is the only control transfer out of the header).
    const h_insts = func.blockInsts(header);
    if (h_insts.len != 2) return null;
    if (func.opcode(h_insts[0]) != .icmp) return null;
    if (func.opcode(h_insts[1]) != .@"if") return null;
    const cmp = func.opcode(h_insts[0]).icmp;
    const iff = func.opcode(h_insts[1]).@"if";
    // The `if` must branch on the header's icmp, not some other condition.
    if (iff.cond != func.instResult(h_insts[0]).?) return null;
    // Canonical ascending bound test.
    if (cmp.op != .lt) return null;
    if (func.terminator(header)) |t| switch (t) {
        // A value-returning header is not the loop-test idiom.
        .ret => |v| if (v != null) return null,
        // An explicit jump terminator would make the header more than a pure test.
        .jump => return null,
    };

    // The induction variable is the icmp lhs, and it must be one of the header params.
    const hparams = func.blockParams(header);
    var iv: ?usize = null;
    for (hparams, 0..) |p, i| {
        if (p == cmp.lhs) iv = i;
    }
    const iv_i = iv orelse return null;
    // The induction variable is an i32 counter (matches the 0-based tile index the lowering emits).
    if (!isI32(func, hparams[iv_i])) return null;

    // The bound is the icmp rhs: a loop-invariant iconst tile dimension in 1..=65535.
    const bound = constBound(func, def_block, body_bits, cmp.rhs) orelse return null;

    // Exactly one of the `if` successors is in-loop (the immediate body); the other exits the loop.
    const then_in = inLoop(body_bits, @intFromEnum(iff.then.target));
    const else_in = inLoop(body_bits, @intFromEnum(iff.@"else".target));
    const in_edge = if (then_in and !else_in) iff.then else if (else_in and !then_in) iff.@"else" else return null;
    const exit_edge = if (then_in) iff.@"else" else iff.then;
    const body_blk = in_edge.target;

    // The in-loop edge threads the header params straight through, so body param k is header param k.
    const in_args = func.blockArgs(in_edge);
    if (in_args.len != hparams.len) return null;
    for (in_args, hparams) |arg, hp| {
        if (arg != hp) return null;
    }
    const bparams = func.blockParams(body_blk);
    if (iv_i >= bparams.len) return null;

    // The latch: the single block inside the loop whose terminator jumps back to the header.
    var latch: ?Block = null;
    for (0..func.blockCount()) |bi| {
        if (!inLoop(body_bits, @intCast(bi))) continue;
        const b: Block = @enumFromInt(bi);
        switch (func.terminator(b) orelse continue) {
            .jump => |j| if (j.target == header) {
                // More than one back-edge is not the single-latch shape the lowering assumes.
                if (latch != null) return null;
                latch = b;
            },
            .ret => {},
        }
    }
    const latch_blk = latch orelse return null;

    // The latch steps the induction variable by exactly 1 off the value threaded into the body param.
    const back_args = switch (func.terminator(latch_blk).?) {
        .jump => |j| func.blockArgs(j),
        // The latch was found as a jump above; a ret here is impossible, but stay total.
        .ret => return null,
    };
    if (back_args.len != hparams.len) return null;
    if (!stepsByOne(func, def_block, @intFromEnum(latch_blk), back_args[iv_i], bparams[iv_i])) return null;

    // The induction variable starts at 0 (the loop is 0-based over the tile dimension).
    const preheader: Block = @enumFromInt(loop.preheader.?);
    const p_args = switch (func.terminator(preheader) orelse return null) {
        .jump => |j| blk: {
            if (j.target != header) return null;
            break :blk func.blockArgs(j);
        },
        // A preheader that returns instead of jumping into the header is not a preheader.
        .ret => return null,
    };
    if (p_args.len != hparams.len) return null;
    if (!isIconstZero(func, p_args[iv_i])) return null;

    return LoopMatch{
        .header = header,
        .body = body_blk,
        .latch = latch_blk,
        .exit = exit_edge.target,
        .preheader = preheader,
        .iv = iv_i,
        .bound = bound,
        .param_count = hparams.len,
    };
}

/// The Task 2 body-dataflow facts `matchBody` extracts, folded into `Plan` by `recognizeNest`. Plan 19
/// adds the recognized element dtype, the mixed-signedness override, and the A/B element size in bytes;
/// the fp16 follow-up adds the `.fp16` dtype alongside them.
const BodyMatch = struct {
    acc_k_param: usize,
    pa_k_param: usize,
    pb_k_param: usize,
    c_store: Inst,
    c_ptr: Value,
    /// `.fp32` (direct-load floating body), `.fp16` (convert-then-float-multiply body), `.int8`
    /// (both-signed or mixed int8), or `.uint8` (both-unsigned int8). `input_signs` is non-null only for
    /// mixed int8 signedness (then `dtype == .int8`). `input_elem` is the A/B element size the strides
    /// must confirm (4 for fp32, 2 for fp16, 1 for int8/uint8).
    dtype: MatMulType,
    input_signs: ?InputSigns,
    input_elem: u32,
    /// Task 2: whether the k-loop reduction seeds its accumulator from `load(C[i][j])` (a MEMORY
    /// accumulator, `C += A*B`, matmul(accumulate=true)) rather than a fresh dtype zero
    /// (matmul(accumulate=false)). Proven true only when the init load's pointer is the exact same SSA
    /// Value as the store target c_ptr.
    accumulate: bool,
};

/// Prove the k-loop's body (`inner`, already Task-1-matched as the innermost loop) computes a
/// multiply-accumulate reduction over two distinct element pointers, with nothing else in the body, and
/// that the finished accumulator is stored to a pointer living above the k-loop. Returns null on ANY
/// deviation: a non-matmul body raised to the `matmul` op would silently compute the wrong answer, so
/// recognition must be conservative here exactly as `matchLoop` is conservative about the skeleton.
///
/// Three body shapes are accepted. The accumulator's TYPE narrows it to fp32-or-fp16 (f32) vs int8-family
/// (i32); an f32 accumulator additionally needs the mul operands PEEKED (converts select fp16, direct
/// loads select fp32) since both share the same accumulator type:
///   fp32 : f32 accumulator, `prod = fmul(la, lb)` of two DIRECT f32 loads, `fadd` accumulate. No
///          converts; dtype .fp32, input_signs null, input_elem 4.
///   fp16 : f32 accumulator, `prod = fmul(ca, cb)` (FLOAT mul) where `ca = convert(f32, la)`,
///          `cb = convert(f32, lb)` are converts of two f16 loads, `fadd` accumulate. dtype .fp16,
///          input_signs null (floats have no signedness), input_elem 2.
///   int8 : i32 accumulator, `prod = mul(ca, cb)` (integer mul) where `ca = convert(i32, la)`,
///          `cb = convert(i32, lb)` are converts of two 8-bit loads, `add` accumulate. Per-operand
///          signedness from the two 8-bit load types (both signed -> .int8, both unsigned -> .uint8, one
///          of each -> .int8 with `input_signs`, unlike dotprod which rejects mixed). input_elem 1.
///
/// Mirrors dotprod.zig's backward accumulator match (dotprod.zig:182-263): find the accumulator from the
/// back-edge args, walk backward through the product to the two loads (through the converts for
/// fp16/int8), then check nothing else is in the body, then walk forward from the k-loop's exit edge to
/// the store.
fn matchBody(func: *const Function, def_block: []const u32, inner: *const LoopMatch) ?BodyMatch {
    const hparams = func.blockParams(inner.header);
    const bparams = func.blockParams(inner.body);
    const b_idx: u32 = @intFromEnum(inner.body);

    // The k-loop's back-edge args (its latch's jump back to the header), one per header param. matchLoop
    // already proved the latch jumps to the header, so a `ret` here cannot happen; the switch stays total.
    const back_args = switch (func.terminator(inner.latch) orelse return null) {
        .jump => |j| func.blockArgs(j),
        .ret => return null,
    };
    if (back_args.len != hparams.len) return null;

    // The accumulator is the single header param whose back-edge value is `arith add(acc, prod)` or
    // `arith add(prod, acc)`, `acc` being that same param threaded into the body (a self-recurrent
    // reduction). This one `arith .add` matches BOTH the fp32 (floating operands) and int8 (i32 operands)
    // bodies; the accumulator's TYPE below decides which. Scan every param; more than one such update is
    // ambiguous, none means there is no reduction here.
    var acci: ?usize = null;
    var prod: Value = undefined;
    for (0..hparams.len) |k| {
        const upd = back_args[k];
        // An update not computed in this body (e.g. threaded straight through) cannot be the reduction.
        if (def_block[@intFromEnum(upd)] != b_idx) continue;
        const di = func.definingInst(upd) orelse continue;
        const op = func.opcode(di);
        if (op != .arith) continue;
        const a = op.arith;
        if (a.op != .add) continue; // only an add is an accumulate (float add for fp32, integer add for int8/uint8)
        const bacc = bparams[k];
        if (a.lhs == bacc) {
            if (acci != null) return null; // a second accumulator-shaped update: ambiguous, refuse
            acci = k;
            prod = a.rhs;
        } else if (a.rhs == bacc) {
            if (acci != null) return null;
            acci = k;
            prod = a.lhs;
        }
    }
    const acc_i = acci orelse return null; // no self-recurrent add: this is not a reduction loop
    // The accumulator's type selects the body shape: an f32 accumulator is the fp32 matmul, an i32
    // accumulator is the int8/uint8 matmul (C is always a 32-bit accumulator). Any other type (f64, i8,
    // i16, ...) is a reduction the tensor op cannot represent.
    const acc_is_f32 = isF32(func, hparams[acc_i]);
    const acc_is_i32 = isI32(func, hparams[acc_i]);
    if (!acc_is_f32 and !acc_is_i32) return null; // accumulator is neither fp32 nor i32: not a tensor matmul
    const nacc_inst = func.definingInst(back_args[acc_i]).?;

    // The accumulator's initial value comes from j_body, which doubles as the k-loop's preheader (the topology contract every
    // buildMatmulNest-shaped nest relies on), so its jump into k_header carries the fresh value. Anything
    // but a 0.0 fconst is a pre-seeded accumulator, which this first slice does not handle.
    const init_args = switch (func.terminator(inner.preheader) orelse return null) {
        .jump => |j| blk: {
            // matchLoop already required this preheader to jump into the header with the iv at 0; reject
            // defensively anyway rather than trust that unchecked here.
            if (j.target != inner.header) return null;
            break :blk func.blockArgs(j);
        },
        .ret => return null,
    };
    if (init_args.len != hparams.len) return null;
    // The accumulator's INIT selects accumulate=false vs accumulate=true. A FRESH accumulator starts at the
    // dtype's zero (fconst 0.0 for fp32/fp16, iconst 0 for the i32 int8/uint8 accumulator) and lowers to
    // matmul(accumulate=false), the existing path. A MEMORY accumulator instead seeds the reduction with
    // `load(C[i][j])`, the SAME element the k_exit store writes back, which is exactly `C[i][j] += sum_k A*B`
    // and lowers to matmul(accumulate=true). Capture the load's pointer here and prove it is the very C
    // pointer the reduction stores to once c_ptr is recovered below; only that SSA identity licenses
    // accumulate=true. Anything else (a nonzero constant, or a load of an unrelated pointer) is not a
    // provable C-memory accumulation, so it stays rejected exactly as the old fresh-only slice did.
    const init_acc = init_args[acc_i];
    var accumulate = false;
    var acc_init_load_ptr: ?Value = null;
    const init_is_zero = if (acc_is_f32) isFconstZero(func, init_acc) else isIconstZero(func, init_acc);
    if (init_is_zero) {
        accumulate = false; // fresh accumulator, matmul(accumulate=false)
    } else if (loadPtrOf(func, init_acc)) |p| {
        // Pending: this is a memory accumulator ONLY if `p` is the exact C pointer the reduction stores back
        // to. Recorded now, proven against c_ptr after c_ptr is determined below.
        acc_init_load_ptr = p;
    } else {
        // A nonzero, non-load init (some other pre-seeded value): not a C-memory accumulation we can prove.
        return null;
    }

    // The product `prod` must be a mul of two per-element values, each ultimately loaded from a distinct
    // k_header param. For fp32 the operands are the two f32 loads directly; for int8 they are converts of
    // the two 8-bit loads. The `arith .mul` opcode is shared; the operand type (guaranteed by the
    // accumulator type, since verify pins the mul result to the add operand type) makes it a floating or
    // integer multiply.
    if (def_block[@intFromEnum(prod)] != b_idx) return null; // the product must be computed in this body
    const prod_inst = func.definingInst(prod) orelse return null;
    const mulop = switch (func.opcode(prod_inst)) {
        .arith => |a| a,
        // The accumulate's other operand is not itself a multiply: `acc += x` alone is not a product.
        else => return null,
    };
    if (mulop.op != .mul) return null;

    // Recover the two load instructions, their pointers, and (for int8) the two convert instructions plus
    // per-operand signedness. `mulop.lhs` feeds the A operand, `mulop.rhs` the B operand; a commuted mul
    // is disambiguated (and, if genuinely swapped, rejected) by matchStrides' A-inner/B-inner checks.
    var la_inst: Inst = undefined;
    var lb_inst: Inst = undefined;
    var la_ptr: Value = undefined;
    var lb_ptr: Value = undefined;
    var dtype: MatMulType = undefined;
    var input_signs: ?InputSigns = null;
    var input_elem: u32 = undefined;
    // The fp16 and int8 bodies each have two extra convert instructions in the allow-list; fp32 has
    // none. Collect them here so the "no extra body ops" count below stays exact per dtype.
    var extra: [2]Inst = undefined;
    var extra_len: usize = 0;
    if (acc_is_f32) {
        // An f32 accumulator is EITHER the real fp32 body (two DIRECT f32 loads, no converts) or the
        // fp16 body (two `convert(f32, load_f16)` operands): both share the same accumulator type, so
        // peek at whether the mul's LHS is itself a convert to tell them apart before committing to
        // either shape's checks. A non-convert LHS (including one not even computed in this body) falls
        // through to the fp32 arm, whose own checks fail closed if it turns out not to be a load either.
        const lhs_is_convert = blk: {
            if (def_block[@intFromEnum(mulop.lhs)] != b_idx) break :blk false;
            const di = func.definingInst(mulop.lhs) orelse break :blk false;
            break :blk func.opcode(di) == .convert;
        };
        if (lhs_is_convert) {
            // fp16: each mul operand is `convert(f32, load_f16)`. The mul is a FLOAT mul (its f32
            // operands are guaranteed by the f32 accumulator, which verify pins the mul result type to).
            const la_val = convertSourceF32(func, def_block, b_idx, mulop.lhs) orelse return null; // A operand not convert(f32, _)
            const lb_val = convertSourceF32(func, def_block, b_idx, mulop.rhs) orelse return null; // B operand not convert(f32, _)
            const ca_inst = func.definingInst(mulop.lhs).?; // the convert, existence proven by convertSourceF32
            const cb_inst = func.definingInst(mulop.rhs).?;
            if (def_block[@intFromEnum(la_val)] != b_idx or def_block[@intFromEnum(lb_val)] != b_idx) return null;
            la_inst = func.definingInst(la_val) orelse return null;
            lb_inst = func.definingInst(lb_val) orelse return null;
            la_ptr = switch (func.opcode(la_inst)) {
                .load => |l| l.ptr,
                // A convert whose source is not a load is not the element-fetch idiom.
                else => return null,
            };
            lb_ptr = switch (func.opcode(lb_inst)) {
                .load => |l| l.ptr,
                else => return null,
            };
            // Each converted value must be an f16 load; any other width/kind (e.g. the bad_convert_src
            // negative, whose loads are f32) is not an fp16 tensor element.
            if (!isF16(func, la_val) or !isF16(func, lb_val)) return null;
            dtype = .fp16;
            input_signs = null; // floats have no signedness
            input_elem = 2;
            extra[0] = ca_inst;
            extra[1] = cb_inst;
            extra_len = 2;
        } else {
            // fp32: both mul operands are DIRECT f32 loads (no converts).
            if (def_block[@intFromEnum(mulop.lhs)] != b_idx or def_block[@intFromEnum(mulop.rhs)] != b_idx) return null;
            la_inst = func.definingInst(mulop.lhs) orelse return null;
            lb_inst = func.definingInst(mulop.rhs) orelse return null;
            la_ptr = switch (func.opcode(la_inst)) {
                .load => |l| l.ptr,
                // A multiply operand that is not a load is not the two-element-fetch idiom (this also
                // rejects an int8-style body whose product is a FLOAT mul of converts: the operand is a
                // convert, not a load).
                else => return null,
            };
            lb_ptr = switch (func.opcode(lb_inst)) {
                .load => |l| l.ptr,
                else => return null,
            };
            // Both loaded elements must be fp32; the fp32 matmul never mixes widths or reduces integers.
            if (!isF32(func, mulop.lhs) or !isF32(func, mulop.rhs)) return null;
            dtype = .fp32;
            input_signs = null;
            input_elem = 4;
        }
    } else {
        // int8/uint8: each mul operand is `convert(i32, load_i8)`. The mul is an integer mul (its i32
        // operands are guaranteed by the i32 accumulator, which verify pins the mul result width to).
        const la_val = convertSource(func, def_block, b_idx, mulop.lhs) orelse return null; // A operand not convert(i32, _)
        const lb_val = convertSource(func, def_block, b_idx, mulop.rhs) orelse return null; // B operand not convert(i32, _)
        const ca_inst = func.definingInst(mulop.lhs).?; // the convert, existence proven by convertSource
        const cb_inst = func.definingInst(mulop.rhs).?;
        if (def_block[@intFromEnum(la_val)] != b_idx or def_block[@intFromEnum(lb_val)] != b_idx) return null;
        la_inst = func.definingInst(la_val) orelse return null;
        lb_inst = func.definingInst(lb_val) orelse return null;
        la_ptr = switch (func.opcode(la_inst)) {
            .load => |l| l.ptr,
            // A convert whose source is not a load is not the element-fetch idiom.
            else => return null,
        };
        lb_ptr = switch (func.opcode(lb_inst)) {
            .load => |l| l.ptr,
            else => return null,
        };
        // Each converted value must be an 8-bit integer load; a wider (e.g. i16) or non-integer source is
        // not an int8 tensor element. int8Sign yields the per-operand signedness.
        const sign_a = int8Sign(func, func.valueType(la_val)) orelse return null;
        const sign_b = int8Sign(func, func.valueType(lb_val)) orelse return null;
        const a_unsigned = sign_a == .unsigned;
        const b_unsigned = sign_b == .unsigned;
        if (!a_unsigned and !b_unsigned) {
            dtype = .int8; // both operands signed: symmetric int8, no override
            input_signs = null;
        } else if (a_unsigned and b_unsigned) {
            dtype = .uint8; // both operands unsigned: symmetric uint8, no override
            input_signs = null;
        } else {
            // Mixed signedness (one signed, one unsigned): plan-16 spells this as .int8 plus the explicit
            // per-operand override. Unlike dotprod, matmul KEEPS this rather than rejecting it.
            dtype = .int8;
            input_signs = .{ .a_unsigned = a_unsigned, .b_unsigned = b_unsigned };
        }
        input_elem = 1;
        extra[0] = ca_inst;
        extra[1] = cb_inst;
        extra_len = 2;
    }

    const pa_i = paramIndex(bparams, la_ptr) orelse return null; // must be a k_header/k_body param
    const pb_i = paramIndex(bparams, lb_ptr) orelse return null;
    if (pa_i == pb_i) return null; // the same pointer feeding both loads is not two distinct operands
    if (pa_i == acc_i or pb_i == acc_i) return null; // the accumulator reused as a pointer is not this shape
    if (!isPtr(func, hparams[pa_i]) or !isPtr(func, hparams[pb_i])) return null;

    // The induction variable, and each of the two element pointers, must step by an `arith_imm add` off
    // the value threaded into the body (any immediate; Task 3 checks the exact strides). This also
    // recovers the three step instructions for the "no extra body ops" count below.
    const nkk_inst = stepInst(func, def_block, b_idx, back_args[inner.iv], bparams[inner.iv]) orelse return null;
    const npa_inst = stepInst(func, def_block, b_idx, back_args[pa_i], bparams[pa_i]) orelse return null;
    const npb_inst = stepInst(func, def_block, b_idx, back_args[pb_i], bparams[pb_i]) orelse return null;

    // The two loads, the mul, the add, the three arith_imm steps, and (int8 only) the
    // two converts must be the WHOLE body. Any other instruction (another load/store/call/if/select/second
    // reduction/...) is an uncaptured side effect or extra computation the tensor lowering cannot absorb.
    // The exact count differs by dtype: 7 for fp32 (no converts), 9 for int8/uint8 AND fp16 (two extra converts).
    var idiom_buf: [9]Inst = undefined;
    var idiom_len: usize = 0;
    for ([_]Inst{ la_inst, lb_inst, prod_inst, nacc_inst, nkk_inst, npa_inst, npb_inst }) |i| {
        idiom_buf[idiom_len] = i;
        idiom_len += 1;
    }
    for (extra[0..extra_len]) |i| {
        idiom_buf[idiom_len] = i;
        idiom_len += 1;
    }
    const idiom = idiom_buf[0..idiom_len];
    for (0..idiom.len) |x| {
        for (idiom[x + 1 ..]) |y| {
            // Two "distinct" roles resolving to the same instruction means the body has fewer real
            // operations than the idiom needs (e.g. a stride instruction doing double duty); refuse.
            if (idiom[x] == y) return null;
        }
    }
    if (func.blockInsts(inner.body).len != idiom.len) return null; // extra instructions present: refuse

    // The k-loop's exit edge (the `if`'s other successor, not the in-loop edge to the body)
    // carries the finished accumulator out. Find which k_exit param it lands in, then require k_exit to
    // store exactly that value to a pointer living above the k-loop (the C write).
    const h_insts = func.blockInsts(inner.header);
    // matchLoop already proved the header is exactly [icmp, if]; re-read the `if` here only to recover
    // the exit edge's args, which LoopMatch does not carry (it kept only the exit block, not its args).
    const iff = func.opcode(h_insts[1]).@"if";
    const exit_edge = if (iff.then.target == inner.body)
        iff.@"else"
    else if (iff.@"else".target == inner.body)
        iff.then
    else
        // Neither edge targets the body matchLoop found: the header changed shape underneath us, refuse.
        return null;
    const exit_args = func.blockArgs(exit_edge);

    var exit_i: ?usize = null;
    for (exit_args, 0..) |v, i| {
        if (v == hparams[acc_i]) exit_i = i;
    }
    const ei = exit_i orelse return null; // the accumulator is not threaded out on the exit edge: no result

    const k_exit = exit_edge.target;
    const kexit_params = func.blockParams(k_exit);
    if (ei >= kexit_params.len) return null; // args/params length mismatch: malformed edge, refuse
    const acc_out = kexit_params[ei];

    // Exactly one store of the finished accumulator. Other instructions may be present in k_exit (the
    // canonical shape also does the j-step and pointer advances there); those are not this Task's concern.
    var store_inst: ?Inst = null;
    for (func.blockInsts(k_exit)) |inst| switch (func.opcode(inst)) {
        .store => |s| {
            if (s.value != acc_out) continue; // a store of something else is not the C write we need
            if (store_inst != null) return null; // more than one store of the accumulator: ambiguous
            store_inst = inst;
        },
        else => {},
    };
    const c_store = store_inst orelse return null; // the k-loop's result is never stored: not this shape
    const c_ptr = func.opcode(c_store).store.ptr;

    // The stored accumulator must match the accumulator's own type (fp32 stores f32, int8/uint8 store the
    // i32 accumulator into the 32-bit C). A mismatch means the store is not this reduction's result.
    if (acc_is_f32) {
        if (!isF32(func, acc_out)) return null;
    } else {
        if (!isI32(func, acc_out)) return null;
    }
    if (!isPtr(func, c_ptr)) return null; // the store target must be a pointer
    // The C pointer must be a block param living above the k-loop (a "j-level pointer"), not a value
    // computed in k_exit itself; Task 3 traces it to the C base and stride once this shape is confirmed.
    if (func.definingInst(c_ptr) != null) return null;

    // Task 2 (memory accumulator): if the init was a load, PROVE it read this very output element. c_ptr is
    // the j-level C block param (def-less, checked above); the memory-accumulator init loaded some pointer.
    // Require SSA Value equality: same param, same C[i][j], so the net effect is `C[i][j] = C[i][j] + sum_k
    // A*B`, i.e. matmul(accumulate=true). A different pointer would accumulate into other memory, not this
    // output, so refuse (keep the loops). This reorder is sound: the matmul op reads the whole C tile then
    // writes it, while the loop reads/reduces/writes each distinct C[i][j]; C's m*n elements are distinct
    // (row-major, matchStrides) and A/B are inputs, so read-all-then-write-all equals per-element
    // read-reduce-write (the matmul op already assumes A/B/C do not alias).
    if (acc_init_load_ptr) |p| {
        if (p != c_ptr) return null; // init loaded a different pointer than the store target: not a C accumulation
        accumulate = true;
        // Task 1's isel REJECTS accumulate=true with an int8/uint8 dtype (its C-preload path is fp32/fp16
        // only). Raising a memory-accumulator int8/uint8 nest would destroy the loops and THEN fail to
        // lower, a hard compile error where the scalar loops compiled fine. So keep such nests as loops,
        // the same "never raise a matmul the backend refuses" contract as recognizeNest's cap and
        // embedded-wide-float gates. fp32/fp16 memory accumulation is the supported set, matching Task 1.
        switch (dtype) {
            .int8, .uint8 => return null, // accumulate=true + int8/uint8: isel refuses, keep the loops
            .fp32, .fp16 => {},
        }
    }

    return BodyMatch{
        .acc_k_param = acc_i,
        .pa_k_param = pa_i,
        .pb_k_param = pb_i,
        .c_store = c_store,
        .c_ptr = c_ptr,
        .dtype = dtype,
        .input_signs = input_signs,
        .input_elem = input_elem,
        .accumulate = accumulate,
    };
}

/// The Task 3 pointer-choreography facts `matchStrides` proves: the three base pointers of the single
/// row-major matmul the nest computes. The A/B element size (`input_elem`) is an INPUT here (from
/// `matchBody`, which read it off the load/accumulator dtype); the strides only confirm it.
const StrideMatch = struct {
    a: Value,
    b: Value,
    c: Value,
};

/// Prove the pointer stepping is a single row-major matmul and recover the A/B/C base pointers. This is
/// THE CRUX: Task 2 proved the body multiplies-accumulates two loads and stores the result, but not that
/// the ADDRESSES are `A[i][k]`, `B[k][j]`, `C[i][j]` in one consistent row-major layout. A wrong stride or
/// base would raise a `matmul` op computing the wrong result, so every check is strict and any deviation
/// returns null.
///
/// The trace works OUTWARD from the two k-loop element pointers (`pa_k_param`, `pb_k_param`), reading
/// each pointer's carried value through the loop-closed SSA Tasks 1/2 established: its in-loop step is the
/// latch back-edge arg at that param index (an `arith_imm add(base, imm)`), and its per-loop reset is the
/// preheader/entry edge arg at that param index. No pointer final is recomputed; every value is read from
/// a header/body param or a jump arg. The layout proven: A(m x k), B(k x n), C(m x n), all row-major.
///
/// A and B step in units of the A/B element size IE = `body.input_elem` (4 for fp32, 2 for fp16, 1 for
/// int8/uint8);
/// C steps in units of the 32-bit accumulator size OE = 4 (C is always 32-bit, so its element size is
/// independent of the A/B dtype). The A-inner step also CONFIRMS IE matches what matchBody found.
///
///   IE check  = A inner step  (pa_k += IE per k)                   -> A contiguous in k, IE bytes/elem
///   B inner   = pb_k += n*IE per k                                 -> B strides one full n-row per k
///   A outer   = pa_row += k*IE per i (pa_row invariant across j)   -> A row length k
///   B outer   = pb_col += IE per j, reset to `b` (i-invariant)     -> B contiguous in j
///   C         = c_ptr += OE per j, continues across i (no reset)   -> C contiguous, row stride n implied
///
/// `outer_bits` is the i-loop (whole nest) membership bitset, used to prove each base dominates the nest.
fn matchStrides(
    func: *const Function,
    def_block: []const u32,
    outer_bits: []const bool,
    outer: *const LoopMatch,
    middle: *const LoopMatch,
    inner: *const LoopMatch,
    body: *const BodyMatch,
) ?StrideMatch {
    // Param/arg views at each level. Bodies are straight-through of their headers (Task 1 proved the
    // in-edge threads header params into body params 1:1), so a body param index is also the header index.
    const k_bparams = func.blockParams(inner.body);
    const j_hparams = func.blockParams(middle.header);
    const j_bparams = func.blockParams(middle.body);
    const i_bparams = func.blockParams(outer.body);

    // The three loop back-edges (latch jump args, one per header param) and the three entry edges (each
    // loop's preheader jump args). matchLoop proved every latch/preheader ends in a jump, so a null here
    // means the shape changed underneath us.
    const k_back = jumpArgsOf(func, inner.latch) orelse return null; // k-loop step (kk += 1, pointers)
    const j_back = jumpArgsOf(func, middle.latch) orelse return null; // j-loop step (in k_exit)
    const i_back = jumpArgsOf(func, outer.latch) orelse return null; // i-loop step (in j_exit)
    const k_init = jumpArgsOf(func, inner.preheader) orelse return null; // j_body -> k_header resets
    const j_init = jumpArgsOf(func, middle.preheader) orelse return null; // i_body -> j_header resets
    const i_init = jumpArgsOf(func, outer.preheader) orelse return null; // preheader -> i_header init
    const k_idx: u32 = @intFromEnum(inner.body);
    const j_latch_idx: u32 = @intFromEnum(middle.latch);
    const i_latch_idx: u32 = @intFromEnum(outer.latch);

    const pa_k = body.pa_k_param;
    const pb_k = body.pb_k_param;
    // These param indices index every level's param/arg list; a matmul lowering keeps them in range, but
    // stay total against a malformed edge whose arg list is shorter than the header's param list.
    if (pa_k >= k_back.len or pb_k >= k_back.len) return null;
    if (pa_k >= k_init.len or pb_k >= k_init.len) return null;

    // The A/B element size matchBody proved (4 fp32, 1 int8/uint8) and the fixed 32-bit C element size.
    const input_elem_i: i64 = body.input_elem;
    const output_elem_i: i64 = 4; // C is always a 32-bit accumulator, independent of the A/B dtype
    const n_i: i64 = middle.bound;
    const k_i: i64 = inner.bound;

    // 1. A INNER STRIDE confirms the A/B element size: pa_k advances by a clean immediate each k, and it
    //    must equal the `input_elem` matchBody read off the load dtype; any other stride is not a
    //    contiguous A row of that element size (a mismatch means the loads and the stride disagree).
    const a_inner = stepImm(func, def_block, k_idx, k_back[pa_k], k_bparams[pa_k]) orelse return null; // A ptr not a clean per-k add
    if (a_inner != input_elem_i) return null; // A element stride disagrees with the load dtype

    // 2. B INNER STRIDE: pb_k advances one full B row (n elements) per k. Require exactly n*input_elem, or
    //    the B stepping is not a row-major `B[k][j]` walk down column j.
    const b_inner = stepImm(func, def_block, k_idx, k_back[pb_k], k_bparams[pb_k]) orelse return null; // B ptr not a clean per-k add
    if (b_inner != n_i * input_elem_i) return null; // B per-k stride is not one n-element row

    // 3. A OUTER STRIDE: trace pa_k's k-loop reset to the j-level row pointer, prove it is invariant across
    //    j (the A row is fixed for the whole j sweep), then to the i-level row pointer that steps one full
    //    A row (k elements) per i, and finally to the A base at the outermost entry.
    const pa_j = paramIndex(j_bparams, k_init[pa_k]) orelse return null; // A's k-entry value is not a j-level param
    if (pa_j >= j_back.len or pa_j >= j_init.len) return null;
    if (k_init[pa_k] != j_bparams[pa_j]) return null; // reset must be the straight-through j-body param, not a recompute
    // Invariant across j: the j-latch threads the same j-body param back unchanged (no step).
    if (j_back[pa_j] != j_bparams[pa_j]) return null; // A row pointer changes across j: not a fixed row
    const pa_i = paramIndex(i_bparams, j_init[pa_j]) orelse return null; // A row's j-entry value is not an i-level param
    if (pa_i >= i_back.len or pa_i >= i_init.len) return null;
    const a_outer = stepImm(func, def_block, i_latch_idx, i_back[pa_i], i_bparams[pa_i]) orelse return null; // A row not a clean per-i add
    if (a_outer != k_i * input_elem_i) return null; // A row length is not k elements
    const a = i_init[pa_i];
    if (inLoop(outer_bits, def_block[@intFromEnum(a)])) return null; // A base defined inside the nest: not a real base

    // 4. B OUTER STRIDE: trace pb_k's k-loop reset to the j-level column pointer, prove it advances one
    //    element per j (B contiguous in j), and that it resets each i to an i-invariant base (the B base).
    const pb_j = paramIndex(j_bparams, k_init[pb_k]) orelse return null; // B's k-entry value is not a j-level param
    if (pb_j >= j_back.len or pb_j >= j_init.len) return null;
    if (k_init[pb_k] != j_bparams[pb_j]) return null; // reset must be the straight-through j-body param
    const b_jstep = stepImm(func, def_block, j_latch_idx, j_back[pb_j], j_bparams[pb_j]) orelse return null; // B column not a clean per-j add
    if (b_jstep != input_elem_i) return null; // B is not contiguous in j
    const b = j_init[pb_j];
    if (inLoop(outer_bits, def_block[@intFromEnum(b)])) return null; // B base recomputed inside the i-loop: not i-invariant

    // 5. C: the store pointer is a j-level pointer that advances one 32-bit element per j and CONTINUES
    //    across i with no reset (contiguous C, so the i-stride is n*OE implied by n j-steps per row). C is
    //    always 32-bit, so it steps by OE (4), NOT input_elem. Prove the i-latch carries out exactly the
    //    j-loop's advanced C pointer, never a per-row recomputed base.
    const pc_j = paramIndex(j_bparams, body.c_ptr) orelse return null; // C store pointer is not a j-level param
    if (pc_j >= j_back.len or pc_j >= j_init.len) return null;
    // The three j-level pointer roles (A row pa_j, B column pb_j, C store pc_j) must be DISTINCT params.
    // Aliasing is already rejected implicitly by the stride contradictions above (a shared param cannot
    // be both j-invariant and step by elem, etc.), but make the disjointness explicit and local so this
    // load-bearing safety is obvious rather than a matter of tracing the interacting gates: two roles
    // sharing a pointer would mean two matrices read/written through one address, not a real matmul.
    if (pa_j == pb_j or pa_j == pc_j or pb_j == pc_j) return null;
    const c_jstep = stepImm(func, def_block, j_latch_idx, j_back[pc_j], j_bparams[pc_j]) orelse return null; // C not a clean per-j add
    if (c_jstep != output_elem_i) return null; // C is not contiguous in j (32-bit accumulator elements)
    const pc_i = paramIndex(i_bparams, j_init[pc_j]) orelse return null; // C's j-entry value is not an i-level param
    if (pc_i >= i_back.len or pc_i >= i_init.len) return null;
    // The i-loop's carried-in C value must be a plain block param (the continuation), not an `arith_imm`
    // recomputation of a per-row base. A step here would mean C is reset each i, which this conservative
    // slice refuses even though a matching n*ELEM step would compute the same addresses.
    const cont = i_back[pc_i];
    if (func.definingInst(cont) != null) return null; // C is reset/recomputed each i, not a single advance
    // Airtight: the continuation is a param of the i-latch (which is the j-loop's exit block), and the
    // j-loop threads its advanced C pointer (j_hparams[pc_j]) into exactly that param position.
    if (middle.exit != outer.latch) return null; // j-loop exit is not the i-latch: not the standard fused nest
    const jexit = jExitEdge(func, middle) orelse return null; // j-header exit edge is not the [icmp, if] shape
    if (jexit.target != outer.latch) return null; // j-exit does not target the i-latch
    const jexit_args = func.blockArgs(jexit);
    const cont_q = paramIndex(func.blockParams(outer.latch), cont) orelse return null; // continuation is not an i-latch param
    if (cont_q >= jexit_args.len) return null; // malformed exit edge: fewer args than the i-latch has params
    if (jexit_args[cont_q] != j_hparams[pc_j]) return null; // i-latch carries out something other than the j-loop's C
    const c = i_init[pc_i];
    if (inLoop(outer_bits, def_block[@intFromEnum(c)])) return null; // C base defined inside the nest: not a real base

    return StrideMatch{ .a = a, .b = b, .c = c };
}

/// The arguments the plain-jump terminator of `from` passes to its successor, or null if `from` does not
/// end in a jump. Every non-header block in the nest is a plain jump (Task 1's scaffolding), so a null
/// here means the shape is not the one Task 1 matched.
fn jumpArgsOf(func: *const Function, from: Block) ?[]const Value {
    return switch (func.terminator(from) orelse return null) {
        .jump => |j| func.blockArgs(j),
        // A ret where a loop's latch/preheader jump is expected means the shape changed; refuse.
        .ret => null,
    };
}

/// The exit edge (the `if`'s non-body successor) of `loop`'s header, or null if the header is not the
/// proven [icmp, if] shape or neither successor is the matched body.
fn jExitEdge(func: *const Function, loop: *const LoopMatch) ?ir.function.Jump {
    const h = func.blockInsts(loop.header);
    if (h.len != 2) return null; // header is not [icmp, if]: Task 1 proved it is, so this is defensive
    const iff = switch (func.opcode(h[1])) {
        .@"if" => |x| x,
        else => return null,
    };
    if (iff.then.target == loop.body) return iff.@"else";
    if (iff.@"else".target == loop.body) return iff.then;
    // Neither `if` successor is the matched body: the header changed shape underneath us.
    return null;
}

/// The number of blocks a loop-body bitset contains.
fn popcount(body: []const bool) usize {
    var c: usize = 0;
    for (body) |x| {
        if (x) c += 1;
    }
    return c;
}

/// Whether `a` is a STRICT subset of `b` (every block in `a` is in `b`, and `b` has at least one block
/// `a` does not). Used to prove proper loop containment.
fn strictSubset(a: []const bool, b: []const bool) bool {
    var proper = false;
    for (a, b) |x, y| {
        if (x and !y) return false; // `a` has a block outside `b`, so not a subset
        if (y and !x) proper = true; // `b` is strictly larger somewhere
    }
    return proper;
}

/// If `v` is a loop-invariant `iconst` in 1..=65535, return its value as the tile bound; else null.
fn constBound(func: *const Function, def_block: []const u32, body: []const bool, v: Value) ?u16 {
    // The bound must not be defined inside the loop, or it is not a fixed tile dimension.
    if (inLoop(body, def_block[@intFromEnum(v)])) return null;
    // A block-param bound (no defining instruction) is a runtime value, not a compile-time tile size.
    const di = func.definingInst(v) orelse return null;
    const val = switch (func.opcode(di)) {
        .iconst => |c| c,
        // Any non-iconst bound is not a compile-time tile dimension.
        else => return null,
    };
    // Tile dimensions are positive and must fit the u16 the matmul op carries.
    if (val < 1 or val > 65535) return null;
    return @intCast(val);
}

/// Whether `upd` is `arith_imm add(base, 1)` defined in block index `blk_idx` (the unit induction step).
fn stepsByOne(func: *const Function, def_block: []const u32, blk_idx: u32, upd: Value, base: Value) bool {
    if (def_block[@intFromEnum(upd)] != blk_idx) return false;
    const di = func.definingInst(upd) orelse return false;
    return switch (func.opcode(di)) {
        .arith_imm => |a| a.op == .add and a.lhs == base and a.imm == 1,
        else => false,
    };
}

fn isI32(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.signedness == .signed and i.bits == 32,
        else => false,
    };
}

/// If `v` is a `convert` to i32 defined in block index `b_idx`, return its source value; else null. The
/// int8 body multiplies `convert(i32, load_i8)` operands, so recognition steps through the convert to
/// reach the underlying 8-bit load (the 2D analogue of dotprod.zig's `convertSource`).
fn convertSource(func: *const Function, def_block: []const u32, b_idx: u32, v: Value) ?Value {
    if (def_block[@intFromEnum(v)] != b_idx) return null;
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .convert => |c| if (isI32(func, v)) c.value else null,
        // A non-convert operand is not the int8 element idiom.
        else => null,
    };
}

/// If `v` is a `convert` to f32 defined in block index `b_idx`, return its source value; else null. The
/// fp16 body multiplies `convert(f32, load_f16)` operands, so recognition steps through the convert to
/// reach the underlying 16-bit float load (the floating analogue of `convertSource`).
fn convertSourceF32(func: *const Function, def_block: []const u32, b_idx: u32, v: Value) ?Value {
    if (def_block[@intFromEnum(v)] != b_idx) return null;
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .convert => |c| if (isF32(func, v)) c.value else null,
        // A non-convert operand is not the fp16 element idiom.
        else => null,
    };
}

/// The signedness of an 8-bit integer type, or null if `ty` is not an 8-bit integer. Distinguishes the
/// per-operand int8 (signed) vs uint8 (unsigned) element type driving the matmul dtype and `input_signs`.
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

fn isF32(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f32,
        else => false,
    };
}

/// Distinguishes the fp16 body's element loads (16-bit float) from the fp32 body's (32-bit float) and
/// from any non-float source (e.g. the int8 family's 8-bit int loads).
fn isF16(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f16,
        else => false,
    };
}

fn isPtr(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .ptr;
}

fn isFconstZero(func: *const Function, v: Value) bool {
    const di = func.definingInst(v) orelse return false;
    return switch (func.opcode(di)) {
        .fconst => |c| c == 0.0,
        else => false,
    };
}

/// If `v` is defined by a `load` instruction, its pointer operand; else null. Used to recognize a
/// memory-accumulator init (`acc0 = load(C[i][j])`), whose pointer must then equal the store target.
fn loadPtrOf(func: *const Function, v: Value) ?Value {
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .load => |l| l.ptr,
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

/// If `upd` is `arith_imm add(base, imm)` (any immediate; the caller checks the exact stride, Task 3's
/// job for the two element pointers) defined in block index `blk_idx`, return its instruction; else null.
fn stepInst(func: *const Function, def_block: []const u32, blk_idx: u32, upd: Value, base: Value) ?Inst {
    if (def_block[@intFromEnum(upd)] != blk_idx) return null;
    const di = func.definingInst(upd) orelse return null;
    return switch (func.opcode(di)) {
        .arith_imm => |a| if (a.op == .add and a.lhs == base) di else null,
        else => null,
    };
}

/// If `stepped` is `arith_imm add(base, imm)` defined in block index `blk_idx`, return `imm` (the byte
/// stride); else null. Task 3 reads each pointer's per-loop byte stride off its latch back-edge value.
fn stepImm(func: *const Function, def_block: []const u32, blk_idx: u32, stepped: Value, base: Value) ?i64 {
    if (def_block[@intFromEnum(stepped)] != blk_idx) return null;
    const di = func.definingInst(stepped) orelse return null;
    return switch (func.opcode(di)) {
        .arith_imm => |a| if (a.op == .add and a.lhs == base) a.imm else null,
        else => null,
    };
}

const registry = @import("registry.zig");
const dominators = @import("../dominators.zig");

/// The number of `matmul` instructions anywhere in `func`. Used by the Task 4 tests to confirm `apply`
/// added exactly one (positive) or none at all (every rejection/regression case).
fn countMatmuls(func: *const Function) usize {
    var count: usize = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .matmul) count += 1;
        }
    }
    return count;
}

/// Knobs that turn the canonical nest into a deliberately non-matching shape for the negative tests.
/// `pub` so the Task 5 sysemu differential (libs/vulcan-target/riscv64/tests/etsoc_sysemu.zig) can
/// build the exact canonical shape `recognizeNest` matches, rather than an equivalent-but-different
/// hand-rolled nest that might drift from the recognizer's contract.
pub const NestSpec = struct {
    m: i64 = 2,
    n: i64 = 4,
    k: i64 = 3,
    /// The element dtype of the built nest. `.fp32` is the original floating body (f32 accumulator,
    /// fmul/fadd, direct f32 loads, 4-byte A/B elements, no converts). `.fp16` is the floating-with-
    /// converts body (f32 accumulator, `convert(f32, load_f16)` operands, FLOAT mul/add, 2-byte A/B
    /// elements). The int8 family (i32 accumulator, integer mul/add, `convert(i32, load_i8)` operands,
    /// 1-byte A/B elements, 4-byte i32 C) covers `.int8` (both operands signed), `.uint8` (both
    /// unsigned), and `.mixed` (A unsigned x B signed, which recognition raises as `.int8` plus the
    /// plan-16 `input_signs` override).
    elem_dtype: enum { fp32, int8, uint8, mixed, fp16 } = .fp32,
    /// int8 family: make the two loads 16-bit instead of 8-bit (still converted to i32), so the convert
    /// source is not an 8-bit load (rejects matchBody's `int8Sign` check). fp16: make the two loads f32
    /// instead of f16 (still converted to f32, still multiplied as float), so the convert source is not
    /// f16 (rejects matchBody's `isF16` check).
    bad_convert_src: bool = false,
    /// int8 family only: convert the two 8-bit loads to f32 and multiply as a FLOAT into an f32
    /// accumulator, so the product is a floating mul of converts (rejects matchBody's fp32-path "a
    /// multiply operand is not a direct load" check: the operand is a convert, not a load).
    float_product: bool = false,
    /// fp16 only: convert the two f16 loads to i32 and multiply as an INTEGER into an i32 accumulator
    /// (the reverse of `float_product`), so the product is an integer mul of converts from f16. Rejected
    /// two ways at once: matchBody's f32-accumulator path never sees this body (the accumulator is i32),
    /// and its int8-accumulator path's `int8Sign` check rejects the convert source (f16 is not an 8-bit
    /// integer), so this nest matches neither fp16 nor int8/uint8.
    int_product: bool = false,
    /// Make the j-loop bound a runtime function param instead of an iconst (rejects at constBound).
    nonconst_bound: bool = false,
    /// Put a void call in the innermost body (rejects at the whole-function call gate).
    extra_call: bool = false,
    /// Insert a stray reachable block before the real preheader (rejects at the whole-function gate).
    extra_block: bool = false,
    /// Accumulate a bare load instead of a product: `acc += load(a_k)`, no fmul (rejects Task 2's
    /// "the accumulate's other operand is a product" gate).
    no_product: bool = false,
    /// Both loads read the SAME element pointer instead of two distinct ones (rejects Task 2's "two
    /// distinct pointers" gate).
    alias_pointers: bool = false,
    /// The k-loop accumulator's fresh value is 1.0, not 0.0 (rejects Task 2's "fresh accumulator" gate).
    nonzero_acc_init: bool = false,
    /// An extra, unrelated fp instruction sits in the k-loop body (rejects Task 2's "no extra body ops"
    /// allow-list).
    extra_body_op: bool = false,
    /// k_exit stores a fresh constant instead of the finished accumulator (rejects Task 2's "the result
    /// is stored" gate).
    bad_store: bool = false,
    /// The B element pointer steps by `n*4 + 4` per k instead of a clean `n*4` row stride (rejects Task
    /// 3's B inner stride check: not one n-element B row per k).
    bad_b_inner: bool = false,
    /// The A row pointer steps by `k*4 + 4` per i instead of `k*4` (rejects Task 3's A outer stride
    /// check: the A row length is not k elements).
    bad_a_outer: bool = false,
    /// The C pointer is reset each i to a per-row base stepped in the i-latch, instead of continuing
    /// contiguously across i (rejects Task 3's "C is a single advance with no reset" check). Same
    /// addresses, but a shape this conservative slice refuses to prove is row-major.
    c_resets: bool = false,
    /// The B base handed into the j-loop is recomputed inside i_body (`b_base + 0`) instead of being the
    /// invariant function param, so it is defined INSIDE the nest (rejects Task 3's "base dominates the
    /// nest" check).
    base_inside_nest: bool = false,
    /// int8 family only: emit the k-loop product as `mul(B, A)` instead of `mul(A, B)` (the two multiply
    /// operands swapped). `arith .mul` is commutative, so matchBody itself does not reject this (Task 2
    /// does not care which operand comes first); the swap is instead caught downstream by matchStrides'
    /// A-inner-step check, which requires the pointer feeding the mul's LHS to step by IE (one element)
    /// per k - under this swap that pointer is actually the true B pointer (which steps by n*IE), so the
    /// check fails whenever n > 1. Used by the "commuted mul is not recognized" negative.
    commuted_mul: bool = false,
    /// Task 2 (memory accumulator): seed the k-loop reduction with `load(C[i][j])` (the SAME j-level C
    /// pointer k_exit stores back to) instead of the fresh dtype zero, so the nest computes `C += A*B` and
    /// recognition raises it to matmul(accumulate=true). With this off the fresh (zero-init) path is
    /// byte-identical to before, so the existing accumulate=false differentials stay green.
    mem_accumulate: bool = false,
};

/// Build the canonical matmul nest `fn(A, B, C) void` computing `C[i*n+j] = sum_k A[i*k+kk] * B[kk*n+j]`
/// over compile-time tile sizes, with all addresses carried as advancing element pointers stepped by
/// `arith_imm` (verify-clean, unlike a runtime `arith add(ptr, index)`). The three loops carry 0-based
/// i32 induction variables stepped by 1, the k-loop carries the accumulator, and the store lands after
/// the k-loop. `spec.elem_dtype` selects the fp32 (f32 accumulator, direct f32 loads, 4-byte A/B stride),
/// fp16 (f32 accumulator, `convert(f32, load_f16)` operands, 2-byte A/B stride), or int8-family (i32
/// accumulator, `convert(i32, load_i8)` operands, 1-byte A/B stride, 4-byte i32 C) body. Reused by Tasks
/// 2-5, and `pub` for the same reason as `NestSpec` above: it is the one nest builder the sysemu
/// differential can trust to match `recognizeNest`.
pub fn buildMatmulNest(func: *Function, spec: NestSpec) Error!void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const f16_t = try func.types.intern(.{ .float = .f16 });

    // int8-family (int8/uint8/mixed) uses 8-bit loads converted to i32 and integer arithmetic; fp16
    // uses f16 loads converted to f32 and floating arithmetic; fp32 uses direct f32 loads and floating
    // arithmetic with no converts at all. `float_product` is an int8-shaped body that instead converts
    // to f32 and multiplies as a float (a deliberate negative); `int_product` is the fp16 mirror, an
    // fp16-shaped body that instead converts to i32 and multiplies as an integer (also a deliberate
    // negative). `acc_is_float` decides the accumulator/convert-target type for every case.
    const int8_family = switch (spec.elem_dtype) {
        .int8, .uint8, .mixed => true,
        .fp32, .fp16 => false,
    };
    const fp16_family = spec.elem_dtype == .fp16;
    const has_convert = int8_family or fp16_family; // real fp32 is the only shape with no convert at all
    const acc_is_float = switch (spec.elem_dtype) {
        .fp32 => true,
        .fp16 => !spec.int_product, // int_product negative: integer-multiply the f16-shaped loads
        .int8, .uint8, .mixed => spec.float_product, // float_product negative: float-multiply the int8-shaped loads
    };
    const acc_t = if (acc_is_float) f32_t else i32_t;
    const convert_t = acc_t; // the convert's target is always the accumulator's type (verify pins the mul's operand type to what add expects)
    // Per-operand load element types. fp32 loads f32 directly; fp16 loads f16 (f32 under
    // `bad_convert_src`, so the convert source is not f16); int8-family loads 8-bit ints (16-bit under
    // `bad_convert_src`), signed for int8/B-side-of-mixed, unsigned for uint8/A-side-of-mixed.
    const load_bits: u16 = if (spec.bad_convert_src) 16 else 8;
    const a_unsigned = switch (spec.elem_dtype) {
        .fp32, .fp16, .int8 => false,
        .uint8, .mixed => true, // mixed = A unsigned x B signed
    };
    const b_unsigned = switch (spec.elem_dtype) {
        .fp32, .fp16, .int8, .mixed => false,
        .uint8 => true,
    };
    const a_elem_t = if (int8_family)
        try func.types.intern(.{ .int = .{ .signedness = if (a_unsigned) .unsigned else .signed, .bits = load_bits } })
    else if (fp16_family)
        (if (spec.bad_convert_src) f32_t else f16_t)
    else
        f32_t;
    const b_elem_t = if (int8_family)
        try func.types.intern(.{ .int = .{ .signedness = if (b_unsigned) .unsigned else .signed, .bits = load_bits } })
    else if (fp16_family)
        (if (spec.bad_convert_src) f32_t else f16_t)
    else
        f32_t;
    // A/B element byte size (`ie`) drives the pointer strides; C is always a 4-byte accumulator (`oe`).
    const ie: i64 = if (int8_family) 1 else if (fp16_family) 2 else 4;
    const oe: i64 = 4;

    const entry = try func.appendBlock();
    // When `extra_block`, a stray reachable block sits between entry and the real preheader.
    const mid = if (spec.extra_block) try func.appendBlock() else entry;
    const i_header = try func.appendBlock();
    const i_body = try func.appendBlock();
    const j_header = try func.appendBlock();
    const j_body = try func.appendBlock();
    const k_header = try func.appendBlock();
    const k_body = try func.appendBlock();
    const k_exit = try func.appendBlock();
    const j_exit = try func.appendBlock();
    const ret_block = try func.appendBlock();

    // Function params: base pointers of A, B, C (and, for the negative, a runtime j-bound).
    const a_base = try func.appendBlockParam(entry, ptr_t);
    const b_base = try func.appendBlockParam(entry, ptr_t);
    const c_base = try func.appendBlockParam(entry, ptr_t);
    const n_param = if (spec.nonconst_bound) try func.appendBlockParam(entry, i32_t) else undefined;

    // Loop-invariant constants live in entry, which dominates every header (so the headers stay the
    // pure [icmp, if] idiom with no constant materialization of their own).
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const m_c = try func.appendInst(entry, i32_t, .{ .iconst = spec.m });
    const n_c = if (spec.nonconst_bound) n_param else try func.appendInst(entry, i32_t, .{ .iconst = spec.n });
    const k_c = try func.appendInst(entry, i32_t, .{ .iconst = spec.k });
    // The fresh accumulator: fconst 0.0 for a floating accumulator, iconst 0 for the i32 int8 accumulator
    // (1.0 / 1 under `nonzero_acc_init`, the pre-seeded-accumulator negative).
    const facc0 = if (acc_is_float)
        try func.appendInst(entry, acc_t, .{ .fconst = if (spec.nonzero_acc_init) 1.0 else 0.0 })
    else
        try func.appendInst(entry, acc_t, .{ .iconst = if (spec.nonzero_acc_init) 1 else 0 });
    const b_row: i64 = spec.n * ie; // B strides one full row (n elements) per k step
    const a_row_stride: i64 = spec.k * ie; // A strides one full row (k elements) per i step
    const c_row: i64 = spec.n * oe; // one full C row (used only by the `c_resets` negative)

    if (spec.extra_block) {
        // The stray block: entry does an extra store then hands off to `mid`, the real preheader.
        try func.appendStore(entry, zero, c_base);
        try func.setJump(entry, mid, &.{});
    }
    // Preheader edge: enter the i-loop with i=0, a_row=A, c_ptr=C.
    try func.setJump(mid, i_header, &.{ zero, a_base, c_base });

    // i-loop: for i in 0..m, carrying the row pointer into A and the running C write pointer.
    const i = try func.appendBlockParam(i_header, i32_t);
    const a_row = try func.appendBlockParam(i_header, ptr_t);
    const c_ptr = try func.appendBlockParam(i_header, ptr_t);
    const cmp_i = try func.appendInst(i_header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = m_c } });
    try func.appendIf(i_header, cmp_i, .{ .target = i_body, .args = &.{ i, a_row, c_ptr } }, .{ .target = ret_block, .args = &.{} });

    // i_body doubles as the j-loop preheader: reset j=0, set the B column pointer to B, thread A row
    // and C pointer on. Its params mirror i_header's (the straight-through the recognizer checks).
    const bi = try func.appendBlockParam(i_body, i32_t);
    const ib_a_row = try func.appendBlockParam(i_body, ptr_t);
    const ib_c_ptr = try func.appendBlockParam(i_body, ptr_t);
    // `base_inside_nest`: hand the j-loop a B base recomputed here (b_base + 0, same address) instead of
    // the invariant function param, so Task 3's "the base dominates the nest" check sees it defined inside
    // the i-loop. Verify-clean (b_base dominates i_body); it is simply not a real, hoistable base.
    const b_col_base = if (spec.base_inside_nest) try func.appendArithImm(i_body, ptr_t, .add, b_base, 0) else b_base;
    try func.setJump(i_body, j_header, &.{ zero, ib_a_row, b_col_base, ib_c_ptr });

    // j-loop: for j in 0..n, carrying the A row (invariant across j), the B column pointer, and C.
    const j = try func.appendBlockParam(j_header, i32_t);
    const ja_row = try func.appendBlockParam(j_header, ptr_t);
    const jb_col = try func.appendBlockParam(j_header, ptr_t);
    const jc_ptr = try func.appendBlockParam(j_header, ptr_t);
    const cmp_j = try func.appendInst(j_header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = j, .rhs = n_c } });
    try func.appendIf(j_header, cmp_j, .{ .target = j_body, .args = &.{ j, ja_row, jb_col, jc_ptr } }, .{ .target = j_exit, .args = &.{jc_ptr} });

    // j_body doubles as the k-loop preheader: reset kk=0 and acc to its zero, seed the two element pointers
    // (a_k from the A row, b_k from the B column). j's own carried values (bj, row, col, c) dominate
    // k_exit and are used there for the store and the j-step, so they need not thread through the k-loop.
    const bj = try func.appendBlockParam(j_body, i32_t);
    const jba_row = try func.appendBlockParam(j_body, ptr_t);
    const jbb_col = try func.appendBlockParam(j_body, ptr_t);
    const jbc_ptr = try func.appendBlockParam(j_body, ptr_t);
    // `mem_accumulate`: seed the k-loop accumulator with `load(C[i][j])` (the same `jbc_ptr` k_exit stores
    // back to) instead of the fresh `facc0` zero, so the nest computes `C += A*B`. jbc_ptr is a j_body
    // block param, so it dominates this load; the load reads the accumulator width (f32/i32, matching C).
    const k_acc_init = if (spec.mem_accumulate)
        try func.appendInst(j_body, acc_t, .{ .load = .{ .ptr = jbc_ptr } })
    else
        facc0;
    try func.setJump(j_body, k_header, &.{ zero, k_acc_init, jba_row, jbb_col });

    // k-loop: for kk in 0..k, acc += A[i*k+kk] * B[kk*n+j], advancing a_k by one element and b_k by
    // one row each step.
    const kk = try func.appendBlockParam(k_header, i32_t);
    const acc = try func.appendBlockParam(k_header, acc_t);
    const a_k = try func.appendBlockParam(k_header, ptr_t);
    const b_k = try func.appendBlockParam(k_header, ptr_t);
    const cmp_k = try func.appendInst(k_header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = kk, .rhs = k_c } });
    try func.appendIf(k_header, cmp_k, .{ .target = k_body, .args = &.{ kk, acc, a_k, b_k } }, .{ .target = k_exit, .args = &.{acc} });

    const bk = try func.appendBlockParam(k_body, i32_t);
    const bacc = try func.appendBlockParam(k_body, acc_t);
    const ba_k = try func.appendBlockParam(k_body, ptr_t);
    const bb_k = try func.appendBlockParam(k_body, ptr_t);
    const va = try func.appendInst(k_body, a_elem_t, .{ .load = .{ .ptr = ba_k } });
    // `alias_pointers` reads the SECOND load from `ba_k` too, instead of `bb_k`: both multiply operands
    // come from the same pointer, which Task 2's "two distinct pointers" gate must reject.
    const vb = try func.appendInst(k_body, b_elem_t, .{ .load = .{ .ptr = if (spec.alias_pointers) ba_k else bb_k } });
    // int8 family: convert each 8-bit load to the accumulator width before the multiply. fp32 multiplies
    // the direct loads. `oa`/`ob` are the multiply operands either way.
    const oa = if (has_convert) try func.appendInst(k_body, convert_t, .{ .convert = .{ .value = va } }) else va;
    const ob = if (has_convert) try func.appendInst(k_body, convert_t, .{ .convert = .{ .value = vb } }) else vb;
    const nacc = if (spec.no_product) blk: {
        // `no_product`: accumulate the bare (converted) load directly, with no multiply at all. Task 2's
        // "the accumulate's other operand is a product" gate must reject this (it is a sum, not a matmul).
        break :blk try func.appendInst(k_body, acc_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = oa } });
    } else blk: {
        // `commuted_mul`: swap the two multiply operands (the "B, A" negative). Verify-clean either way
        // (both operands share `convert_t`); only matchStrides' A-inner-step check can tell them apart.
        const mul_lhs = if (spec.commuted_mul) ob else oa;
        const mul_rhs = if (spec.commuted_mul) oa else ob;
        const prod = try func.appendInst(k_body, acc_t, .{ .arith = .{ .op = .mul, .lhs = mul_lhs, .rhs = mul_rhs } });
        break :blk try func.appendInst(k_body, acc_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = prod } });
    };
    if (spec.extra_call) {
        // A side effect the tensor lowering cannot absorb; the whole-function gate must reject it.
        try func.appendVoidCall(k_body, "sink", &.{});
    }
    if (spec.extra_body_op) {
        // An extra, unrelated instruction (dead, fed by nothing downstream): Task 2's "no extra body ops"
        // allow-list must reject it, since a real extra computation could be anything. Use the accumulator
        // width so it is verify-clean for both the fp32 (f32) and int8 (i32) accumulator.
        _ = try func.appendInst(k_body, acc_t, .{ .arith = .{ .op = .add, .lhs = oa, .rhs = ob } });
    }
    const nkk = try func.appendArithImm(k_body, i32_t, .add, bk, 1);
    const nba_k = try func.appendArithImm(k_body, ptr_t, .add, ba_k, ie);
    // `bad_b_inner`: step B by `n*ie + ie` per k, not a clean n-element row, so Task 3's B inner stride
    // check rejects it (the B stepping is no longer a row-major `B[k][j]` column walk).
    const nbb_k = try func.appendArithImm(k_body, ptr_t, .add, bb_k, if (spec.bad_b_inner) b_row + ie else b_row);
    try func.setJump(k_body, k_header, &.{ nkk, nacc, nba_k, nbb_k });

    // k_exit (the j-loop latch): store the accumulated element, step j, advance the B column and C.
    const kacc = try func.appendBlockParam(k_exit, acc_t);
    if (spec.bad_store) {
        // `bad_store`: store a fresh constant instead of `kacc`, so the k-loop's real answer is silently
        // dropped. Task 2's "the result is stored" gate must reject this (no store matches the finished
        // accumulator's value). `kacc` stays a legitimate (if now-unused-by-Zig-here) k_exit param in the
        // IR; the `appendBlockParam` call above already registered it regardless of this branch. The bogus
        // value matches the accumulator width (f32 or i32) so the nest stays verify-clean.
        const bogus = if (acc_is_float)
            try func.appendInst(k_exit, acc_t, .{ .fconst = 9.0 })
        else
            try func.appendInst(k_exit, acc_t, .{ .iconst = 9 });
        try func.appendStore(k_exit, bogus, jbc_ptr);
    } else {
        try func.appendStore(k_exit, kacc, jbc_ptr);
    }
    const nj = try func.appendArithImm(k_exit, i32_t, .add, bj, 1);
    const nb_col = try func.appendArithImm(k_exit, ptr_t, .add, jbb_col, ie);
    const nc_ptr = try func.appendArithImm(k_exit, ptr_t, .add, jbc_ptr, oe);
    try func.setJump(k_exit, j_header, &.{ nj, jba_row, nb_col, nc_ptr });

    // j_exit (the i-loop latch): step i and advance the A row by one full row. C normally CONTINUES across
    // i (the j-loop's carried-out `jx_c_ptr` threads straight back into i_header, a single monotonic
    // advance with no reset).
    const jx_c_ptr = try func.appendBlockParam(j_exit, ptr_t);
    const ni = try func.appendArithImm(j_exit, i32_t, .add, bi, 1);
    // `bad_a_outer`: advance the A row by `k*4 + 4` per i, so Task 3's A outer stride check rejects it
    // (the A row length is no longer k elements).
    const na_row = try func.appendArithImm(j_exit, ptr_t, .add, ib_a_row, if (spec.bad_a_outer) a_row_stride + ie else a_row_stride);
    // `c_resets`: instead of continuing the j-loop's C pointer across i, step a per-row C base by one full
    // C row (`n*oe`) here and reset the j-loop to it each i. Same addresses, but Task 3 refuses the reset
    // shape (C must be a single advance the whole nest, so its i-latch value must be the carried
    // continuation, not an `arith_imm` recomputation).
    const nc_ptr_i = if (spec.c_resets) try func.appendArithImm(j_exit, ptr_t, .add, ib_c_ptr, c_row) else jx_c_ptr;
    try func.setJump(j_exit, i_header, &.{ ni, na_row, nc_ptr_i });

    func.setTerminator(ret_block, .{ .ret = null });
}

/// Build a 2-deep (i, j) loop nest, verify-clean, for the "not exactly three loops" negative.
fn buildTwoDeepNest(func: *Function) Error!void {
    const ptr_t = try func.types.intern(.ptr);
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });

    const entry = try func.appendBlock();
    const i_header = try func.appendBlock();
    const i_body = try func.appendBlock();
    const j_header = try func.appendBlock();
    const j_body = try func.appendBlock();
    const j_exit = try func.appendBlock();
    const ret_block = try func.appendBlock();

    const a_base = try func.appendBlockParam(entry, ptr_t);
    const c_base = try func.appendBlockParam(entry, ptr_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const m_c = try func.appendInst(entry, i32_t, .{ .iconst = 2 });
    const n_c = try func.appendInst(entry, i32_t, .{ .iconst = 4 });
    try func.setJump(entry, i_header, &.{ zero, a_base, c_base });

    const i = try func.appendBlockParam(i_header, i32_t);
    const a_row = try func.appendBlockParam(i_header, ptr_t);
    const c_ptr = try func.appendBlockParam(i_header, ptr_t);
    const cmp_i = try func.appendInst(i_header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = m_c } });
    try func.appendIf(i_header, cmp_i, .{ .target = i_body, .args = &.{ i, a_row, c_ptr } }, .{ .target = ret_block, .args = &.{} });

    const bi = try func.appendBlockParam(i_body, i32_t);
    const ib_a_row = try func.appendBlockParam(i_body, ptr_t);
    const ib_c_ptr = try func.appendBlockParam(i_body, ptr_t);
    try func.setJump(i_body, j_header, &.{ zero, ib_a_row, ib_c_ptr });

    const j = try func.appendBlockParam(j_header, i32_t);
    const ja_row = try func.appendBlockParam(j_header, ptr_t);
    const jc_ptr = try func.appendBlockParam(j_header, ptr_t);
    const cmp_j = try func.appendInst(j_header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = j, .rhs = n_c } });
    try func.appendIf(j_header, cmp_j, .{ .target = j_body, .args = &.{ j, ja_row, jc_ptr } }, .{ .target = j_exit, .args = &.{jc_ptr} });

    const bj = try func.appendBlockParam(j_body, i32_t);
    const jba_row = try func.appendBlockParam(j_body, ptr_t);
    const jbc_ptr = try func.appendBlockParam(j_body, ptr_t);
    const v = try func.appendInst(j_body, f32_t, .{ .load = .{ .ptr = jba_row } });
    try func.appendStore(j_body, v, jbc_ptr);
    const nj = try func.appendArithImm(j_body, i32_t, .add, bj, 1);
    const nc_ptr = try func.appendArithImm(j_body, ptr_t, .add, jbc_ptr, 4);
    try func.setJump(j_body, j_header, &.{ nj, jba_row, nc_ptr });

    const jx_c_ptr = try func.appendBlockParam(j_exit, ptr_t);
    const ni = try func.appendArithImm(j_exit, i32_t, .add, bi, 1);
    const na_row = try func.appendArithImm(j_exit, ptr_t, .add, ib_a_row, 16);
    try func.setJump(j_exit, i_header, &.{ ni, na_row, jx_c_ptr });

    func.setTerminator(ret_block, .{ .ret = null });
}

/// Run recognizeNest end to end (analyze + def-blocks + match), returning the Plan or null.
fn recognizeIn(allocator: std.mem.Allocator, func: *const Function) Error!?Plan {
    var info = try loops.analyze(allocator, func);
    defer info.deinit(allocator);
    const def_block = try computeDefBlocks(allocator, func);
    defer allocator.free(def_block);
    return recognizeNest(allocator, func, &info, def_block);
}

test "the canonical matmul nest is well-formed and recognized with the right bounds" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .m = 2, .n = 4, .k = 3 });

    // The hand-built nest the recognizer matches must itself be a real, well-formed function.
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    const plan = (try recognizeIn(allocator, &func)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u16, 2), plan.m);
    try std.testing.expectEqual(@as(u16, 4), plan.n);
    try std.testing.expectEqual(@as(u16, 3), plan.k);
    try std.testing.expectEqual(@as(usize, 0), plan.i_iv);
    try std.testing.expectEqual(@as(usize, 0), plan.j_iv);
    try std.testing.expectEqual(@as(usize, 0), plan.k_iv);
    // The three headers are distinct blocks in outer-to-inner order.
    try std.testing.expect(plan.i_header != plan.j_header);
    try std.testing.expect(plan.j_header != plan.k_header);
    try std.testing.expect(plan.i_header != plan.k_header);

    // Task 2: the k_header params are (kk, acc, a_k, b_k) in that order in the builder, so the
    // accumulator is param 1 and the two element pointers are params 2 and 3, matching the multiply's
    // (A-side, B-side) operand order.
    try std.testing.expectEqual(@as(usize, 1), plan.acc_k_param);
    try std.testing.expectEqual(@as(usize, 2), plan.pa_k_param);
    try std.testing.expectEqual(@as(usize, 3), plan.pb_k_param);
    // The recorded store really does write the recorded C pointer.
    switch (func.opcode(plan.c_store)) {
        .store => |s| try std.testing.expectEqual(plan.c_ptr, s.ptr),
        else => return error.TestUnexpectedResult,
    }

    // Task 3: the recovered base pointers are exactly the function's A/B/C params (entry params 0/1/2 in
    // builder order), and the element size is the fp32 4 bytes proven from the A inner stride.
    const entry_params = func.blockParams(@as(Block, @enumFromInt(0)));
    try std.testing.expectEqual(entry_params[0], plan.a);
    try std.testing.expectEqual(entry_params[1], plan.b);
    try std.testing.expectEqual(entry_params[2], plan.c);
    try std.testing.expectEqual(@as(u32, 4), plan.input_elem);
    // Plan 19: the default nest is fp32 with no signedness override.
    try std.testing.expectEqual(MatMulType.fp32, plan.dtype);
    try std.testing.expectEqual(@as(?InputSigns, null), plan.input_signs);
}

test "run reports a match on the et-soc vpu model" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{});
    try std.testing.expect(try run(allocator, &func, registry.modelFor(.@"et-soc")));
}

test "run skips every nest on a non-vpu model" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{});
    // Ampere Altra is aarch64 (no VPU); the matmul op has nowhere to lower, so run must refuse.
    try std.testing.expect(!try run(allocator, &func, registry.modelFor(.@"ampere-altra")));
    try std.testing.expectEqual(@as(usize, 0), countMatmuls(&func));
}

test "run transforms the canonical nest into a single matmul, orphaning the loop nest" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .m = 2, .n = 4, .k = 3 });

    // Recover the preheader and i_header handles before `run` mutates anything, the same way the
    // Task 1-3 test recovers the Plan, so the assertions below check the actual transformed IR rather
    // than trusting `run`'s boolean alone.
    const plan = (try recognizeIn(allocator, &func)) orelse return error.TestUnexpectedResult;

    try std.testing.expect(try run(allocator, &func, registry.modelFor(.@"et-soc")));
    try std.testing.expectEqual(@as(usize, 1), countMatmuls(&func));

    // The preheader's terminator is now a jump straight to the continuation (outer.exit). For this
    // whole-function nest that continuation is the bare `ret void` block, and the exit edge carried no
    // args, so the redirect is the empty-arg jump `apply` reconstructs (equivalent to the old `ret null`
    // but on the unified jump-to-exit path). It holds a matmul with the right operands.
    const term = func.terminator(plan.preheader) orelse return error.TestUnexpectedResult;
    switch (term) {
        .jump => |j| {
            try std.testing.expectEqual(plan.outer_exit, j.target);
            try std.testing.expectEqual(@as(usize, 0), func.blockArgs(j).len);
        },
        .ret => return error.TestUnexpectedResult,
    }
    // The whole-function nest keeps the cheaper non-embedded lowering (nothing is live across it).
    try std.testing.expectEqual(false, plan.embedded);
    var found: ?ir.function.MatMul = null;
    for (func.blockInsts(plan.preheader)) |inst| {
        switch (func.opcode(inst)) {
            .matmul => |mm2| found = mm2,
            else => {},
        }
    }
    const mm2 = found orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(plan.a, mm2.a);
    try std.testing.expectEqual(plan.b, mm2.b);
    try std.testing.expectEqual(plan.c, mm2.c);
    try std.testing.expectEqual(@as(u16, 2), mm2.m);
    try std.testing.expectEqual(@as(u16, 4), mm2.n);
    try std.testing.expectEqual(@as(u16, 3), mm2.k);
    try std.testing.expectEqual(ir.function.MatMulType.fp32, mm2.dtype);
    try std.testing.expectEqual(false, mm2.accumulate);

    // The orphaned nest: i_header (and everything below it) is unreachable from the entry now that the
    // preheader no longer jumps into it.
    var doms = try dominators.compute(allocator, &func);
    defer doms.deinit(allocator);
    try std.testing.expect(!doms.isReachable(@intFromEnum(plan.i_header)));

    // The transformed function, nest orphaned and all, must still be a well-formed program.
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "run does not transform a nest whose tile count exceeds the isel cap" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // 80x80x80: ceil(80/16)=5 per axis, 5*5*5 = 125 > 64, over the cap `matmul_recog` mirrors from
    // riscv64/isel.zig. The nest is otherwise perfectly canonical, so this fails ONLY the cap gate.
    try buildMatmulNest(&func, .{ .m = 80, .n = 80, .k = 80 });
    const blocks_before = func.blockCount();
    const insts_before = func.instCount();

    try std.testing.expect(!try run(allocator, &func, registry.modelFor(.@"et-soc")));
    try std.testing.expectEqual(@as(usize, 0), countMatmuls(&func));
    try std.testing.expectEqual(blocks_before, func.blockCount());
    try std.testing.expectEqual(insts_before, func.instCount());
}

test "run does not transform a nest whose N is not a multiple of 4" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // N=6 is not a multiple of 4, which the isel fma b_cols field (cols/4 - 1) cannot encode, so isel
    // rejects it. Recognition must refuse too: raising it would destroy the loops and then fail to
    // lower the op, a hard compile error where scalar loops would have worked. Otherwise canonical, so
    // this fails ONLY the N%4 gate.
    try buildMatmulNest(&func, .{ .m = 2, .n = 6, .k = 3 });
    const blocks_before = func.blockCount();
    const insts_before = func.instCount();

    try std.testing.expect(!try run(allocator, &func, registry.modelFor(.@"et-soc")));
    try std.testing.expectEqual(@as(usize, 0), countMatmuls(&func));
    try std.testing.expectEqual(blocks_before, func.blockCount());
    try std.testing.expectEqual(insts_before, func.instCount());
}

test "run does not transform a non-matmul loop nest" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // extra_call: a Task-1 whole-function-gate rejection, reused here to prove `run` (not just
    // `recognizeIn`) leaves a non-matmul nest completely untouched.
    try buildMatmulNest(&func, .{ .extra_call = true });
    const blocks_before = func.blockCount();
    const insts_before = func.instCount();

    try std.testing.expect(!try run(allocator, &func, registry.modelFor(.@"et-soc")));
    try std.testing.expectEqual(@as(usize, 0), countMatmuls(&func));
    try std.testing.expectEqual(blocks_before, func.blockCount());
    try std.testing.expectEqual(insts_before, func.instCount());
}

test "skips a two-deep nest" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildTwoDeepNest(&func);

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest whose middle bound is a runtime param" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .nonconst_bound = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest with a call in the innermost body" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .extra_call = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "recognizes a nest with loop-free setup code before the preheader (Task 2 surrounded region)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // `extra_block` inserts a stray reachable block before the real preheader (it stores then jumps into
    // the preheader). Task 1's whole-function gate rejected this; Task 2's region gate ACCEPTS it: a
    // loop-free block whose only nest-facing edge is into the preheader is exactly the newly-allowed
    // surrounding setup code (single-entry preserved). The continuation is still the bare `ret void`, so
    // nothing is live across the matmul and recognition keeps the non-embedded lowering.
    try buildMatmulNest(&func, .{ .extra_block = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    const plan = (try recognizeIn(allocator, &func)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(false, plan.embedded); // bare `ret void` continuation: no live-across value
    try std.testing.expectEqual(@as(usize, 0), plan.exit_args_len); // the outer exit edge threads nothing out

    // Running it raises exactly one matmul and leaves a verify-clean function (orphaned nest and all).
    try std.testing.expect(try run(allocator, &func, registry.modelFor(.@"et-soc")));
    try std.testing.expectEqual(@as(usize, 1), countMatmuls(&func));
    var diags2 = try ir.verify.verify(allocator, &func, .low);
    defer diags2.deinit();
    try std.testing.expect(diags2.ok());
}

test "skips a nest whose k-body accumulates a bare load (no product)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .no_product = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest whose two loads alias the same element pointer" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .alias_pointers = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest with a nonzero accumulator init" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .nonzero_acc_init = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest with an extra instruction in the k-loop body" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .extra_body_op = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest whose k-loop result is not stored" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .bad_store = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest whose B pointer steps by the wrong inner stride" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .bad_b_inner = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest whose A row steps by the wrong outer stride" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .bad_a_outer = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest whose C pointer resets each i instead of advancing contiguously" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .c_resets = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips a nest whose base pointer is defined inside the nest" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .base_inside_nest = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

// --- Plan 19: int8/uint8/mixed dtype recognition. ---

/// The single `matmul` op `func` contains, or null. Used by the Plan-19 positive tests to check `apply`
/// raised the op with the right dtype and (for mixed) `input_signs`.
fn firstMatmul(func: *const Function) ?ir.function.MatMul {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .matmul => |m| return m,
                else => {},
            }
        }
    }
    return null;
}

/// Build an int8-family nest, assert it is verify-clean, recognize it, and assert the Plan carries the
/// expected dtype/input_signs/input_elem plus the m/n/k bounds. Then `run` it on the et-soc VPU and assert
/// exactly one matmul was raised with the same dtype/input_signs and that the transformed IR is still
/// verify-clean. Shared by the three positive int8/uint8/mixed tests.
fn expectInt8Recognized(
    allocator: std.mem.Allocator,
    spec: NestSpec,
    dtype: MatMulType,
    input_signs: ?InputSigns,
) !void {
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, spec);

    // The hand-built int8 nest must itself be a real, well-formed function.
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    const plan = (try recognizeIn(allocator, &func)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(dtype, plan.dtype);
    try std.testing.expectEqual(input_signs, plan.input_signs);
    try std.testing.expectEqual(@as(u32, 1), plan.input_elem); // int8 A/B elements are 1 byte
    try std.testing.expectEqual(@as(u16, @intCast(spec.m)), plan.m);
    try std.testing.expectEqual(@as(u16, @intCast(spec.n)), plan.n);
    try std.testing.expectEqual(@as(u16, @intCast(spec.k)), plan.k);

    try std.testing.expect(try run(allocator, &func, registry.modelFor(.@"et-soc")));
    try std.testing.expectEqual(@as(usize, 1), countMatmuls(&func));
    const op = firstMatmul(&func) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(dtype, op.dtype);
    try std.testing.expectEqual(input_signs, op.input_signs);
    try std.testing.expectEqual(false, op.accumulate);
    try std.testing.expectEqual(plan.a, op.a);
    try std.testing.expectEqual(plan.b, op.b);
    try std.testing.expectEqual(plan.c, op.c);

    // The transformed function (orphaned nest and all) must still verify.
    var diags2 = try ir.verify.verify(allocator, &func, .low);
    defer diags2.deinit();
    try std.testing.expect(diags2.ok());
}

test "recognizes a signed int8 matmul nest and raises an int8 matmul" {
    // K must be a multiple of 4 for the int8 gate (the fma acols field encodes K/4).
    try expectInt8Recognized(std.testing.allocator, .{ .elem_dtype = .int8, .m = 2, .n = 4, .k = 4 }, .int8, null);
}

test "recognizes an unsigned uint8 matmul nest and raises a uint8 matmul" {
    try expectInt8Recognized(std.testing.allocator, .{ .elem_dtype = .uint8, .m = 2, .n = 4, .k = 4 }, .uint8, null);
}

test "recognizes a mixed uint8-x-int8 matmul nest and raises an int8 matmul with an input_signs override" {
    // A uint8 (unsigned) x B int8 (signed): plan-16 spells this as .int8 plus the per-operand override.
    try expectInt8Recognized(
        std.testing.allocator,
        .{ .elem_dtype = .mixed, .m = 2, .n = 4, .k = 4 },
        .int8,
        .{ .a_unsigned = true, .b_unsigned = false },
    );
}

test "rejects a mixed matmul nest whose inner product is commuted (mul(B, A) instead of mul(A, B))" {
    // Reviewer follow-up: prove the swapped-operand orientation is genuinely rejected, not merely
    // untested. matchBody's mul is commutative in itself (it does not care which operand is lhs), so
    // this is caught downstream: matchStrides' A-inner-step check requires the pointer bound to the
    // mul's LHS to step by IE (one int8 element) per k. Commuted, that pointer is really the B pointer
    // (steps by n*IE); with n=4 > 1 this is a genuine mismatch, so the whole nest is left as loops.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, .{ .elem_dtype = .mixed, .m = 2, .n = 4, .k = 4, .commuted_mul = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips an int8-shaped nest whose product is a float mul" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // int8 loads converted to f32 and multiplied as a float into an f32 accumulator: the fp32 body match
    // sees converts (not direct loads) feeding the mul and refuses; the int8 body match is not reached
    // because the accumulator is f32.
    try buildMatmulNest(&func, .{ .elem_dtype = .int8, .m = 2, .n = 4, .k = 4, .float_product = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips an int8-shaped nest whose convert source is not an 8-bit load" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // The loads are i16 (converted to i32), so `int8Sign` on the convert source is null: not an int8
    // tensor element.
    try buildMatmulNest(&func, .{ .elem_dtype = .int8, .m = 2, .n = 4, .k = 4, .bad_convert_src = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips an int8 nest with an extra instruction in the k-loop body" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // The int8 allow-list is exactly nine instructions (two loads, two converts, mul, add, three steps);
    // an extra op makes it ten, which the "no extra body ops" count rejects.
    try buildMatmulNest(&func, .{ .elem_dtype = .int8, .m = 2, .n = 4, .k = 4, .extra_body_op = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "run does not transform an int8 nest whose K is not a multiple of 4" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // K=6 is not a multiple of the int8 factor 4 (the fma acols field encodes K/4), which isel rejects,
    // so recognition must refuse too. N=4 keeps the N%4 gate satisfied, so this fails ONLY the K%4 gate.
    try buildMatmulNest(&func, .{ .elem_dtype = .int8, .m = 2, .n = 4, .k = 6 });
    const blocks_before = func.blockCount();
    const insts_before = func.instCount();

    try std.testing.expect(!try run(allocator, &func, registry.modelFor(.@"et-soc")));
    try std.testing.expectEqual(@as(usize, 0), countMatmuls(&func));
    try std.testing.expectEqual(blocks_before, func.blockCount());
    try std.testing.expectEqual(insts_before, func.instCount());
}

// --- fp16 dtype recognition (the fp16 follow-up to plan 19). ---

/// Build an fp16 nest, assert it is verify-clean, recognize it, and assert the Plan carries dtype .fp16,
/// null input_signs, input_elem 2, and the expected m/n/k. Then `run` it on the et-soc VPU and assert
/// exactly one matmul was raised with the same dtype and that the transformed IR is still verify-clean.
/// Mirrors `expectInt8Recognized` above, minus the input_signs axis (floats have none).
fn expectFp16Recognized(allocator: std.mem.Allocator, spec: NestSpec) !void {
    var func = Function.init(allocator);
    defer func.deinit();
    try buildMatmulNest(&func, spec);

    // The hand-built fp16 nest must itself be a real, well-formed function.
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    const plan = (try recognizeIn(allocator, &func)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(MatMulType.fp16, plan.dtype);
    try std.testing.expectEqual(@as(?InputSigns, null), plan.input_signs);
    try std.testing.expectEqual(@as(u32, 2), plan.input_elem); // fp16 A/B elements are 2 bytes
    try std.testing.expectEqual(@as(u16, @intCast(spec.m)), plan.m);
    try std.testing.expectEqual(@as(u16, @intCast(spec.n)), plan.n);
    try std.testing.expectEqual(@as(u16, @intCast(spec.k)), plan.k);

    try std.testing.expect(try run(allocator, &func, registry.modelFor(.@"et-soc")));
    try std.testing.expectEqual(@as(usize, 1), countMatmuls(&func));
    const op = firstMatmul(&func) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(MatMulType.fp16, op.dtype);
    try std.testing.expectEqual(@as(?InputSigns, null), op.input_signs);
    try std.testing.expectEqual(false, op.accumulate);
    try std.testing.expectEqual(plan.a, op.a);
    try std.testing.expectEqual(plan.b, op.b);
    try std.testing.expectEqual(plan.c, op.c);

    // The transformed function (orphaned nest and all) must still verify.
    var diags2 = try ir.verify.verify(allocator, &func, .low);
    defer diags2.deinit();
    try std.testing.expect(diags2.ok());
}

test "recognizes an fp16 matmul nest and raises an fp16 matmul" {
    // K must be a multiple of 2 for the fp16 gate (the fma acols field encodes K/2).
    try expectFp16Recognized(std.testing.allocator, .{ .elem_dtype = .fp16, .m = 2, .n = 4, .k = 4 });
}

test "skips an fp16-shaped nest whose product is an integer mul" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // f16 loads converted to i32 and multiplied as an INTEGER into an i32 accumulator: the fp16 body
    // match is never reached (that arm only fires under an f32 accumulator), and the int8 body match's
    // `int8Sign` check rejects the convert source (f16 is not an 8-bit integer type). Matches neither
    // dtype, proving the distinction is on the mul's kind (float vs integer), not just the load width.
    try buildMatmulNest(&func, .{ .elem_dtype = .fp16, .m = 2, .n = 4, .k = 4, .int_product = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "skips an fp16-shaped nest whose convert source is not an f16 load" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // The loads are f32 instead of f16 (still converted to f32, still multiplied as float), so `isF16`
    // on the convert source is false: not an fp16 tensor element. The fp32 arm does not catch this
    // either (its operands must be DIRECT loads, and these are converts), so the nest matches nothing.
    try buildMatmulNest(&func, .{ .elem_dtype = .fp16, .m = 2, .n = 4, .k = 4, .bad_convert_src = true });

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    try std.testing.expect((try recognizeIn(allocator, &func)) == null);
}

test "run does not transform an fp16 nest whose K is not a multiple of 2" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // K=5 is not a multiple of the fp16 factor 2 (the fma acols field encodes K/2), which isel rejects,
    // so recognition must refuse too. N=4 keeps the N%4 gate satisfied, so this fails ONLY the K%2 gate.
    try buildMatmulNest(&func, .{ .elem_dtype = .fp16, .m = 2, .n = 4, .k = 5 });
    const blocks_before = func.blockCount();
    const insts_before = func.instCount();

    try std.testing.expect(!try run(allocator, &func, registry.modelFor(.@"et-soc")));
    try std.testing.expectEqual(@as(usize, 0), countMatmuls(&func));
    try std.testing.expectEqual(blocks_before, func.blockCount());
    try std.testing.expectEqual(insts_before, func.instCount());
}
