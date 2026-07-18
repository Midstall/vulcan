//! SIMD widening for fragment shaders: turn a scalar (one-fragment) lowered graphics
//! function into a 4-wide (2x2-quad) one, where every f32 value becomes a `<4 x f32>`
//! whose lanes are the 4 fragments of a quad. The existing aarch64 NEON vector isel
//! (lane-wise fadd/fsub/fmul/fdiv, fsqrt, fcmp->mask, bsl masked-select) then executes
//! all 4 fragments per instruction. This is target-INDEPENDENT codegen and lives in
//! Vulcan at the root (a GPU backend would treat the wider type as a wider warp). The
//! host backend that interprets the tagged output stores as packed-quad memory stores
//! stays in Prism.
//!
//! Two tiers:
//!
//!  1. STRAIGHT-LINE tier (`widenSingleBlock`): a single-entry-block, branch-free,
//!     buffer-free FS (a passthrough / channel-rotate / arithmetic / clamp FS). Every f32
//!     value retypes to `<4 x f32>`, constants splat, the tagged output stores hold the
//!     vector. This is the original, narrow, fast path.
//!
//!  2. HEAVY tier (`widenHeavy`): the vkcube-class FS - it loads per-triangle gradients
//!     (`grad_buf`) and uniforms (descriptors), samples a texture and calls a host math
//!     function (`sampler_fn` / `math_fn`), and has the inlined `linearToSrgb`'s if/else
//!     diamonds. It is widened by SCALARIZING the non-vectorizable ops across the 4 lanes:
//!       - a `load` from a lane-invariant pointer (grad_buf / a descriptor / a sampler
//!         out-slot fed by per-lane samples) keeps ONE scalar load and SPLATS it (broadcast).
//!       - a `call_indirect` (sampler_fn / math_fn) is GATHERED: extract lane k from each
//!         vector arg, do 4 scalar calls, pack the 4 scalar results back into a `<4 x f32>`.
//!       - the sampler's void `call_indirect(desc, u, v, lod, out_ptr)` + reload pattern is
//!         scalarized into 4 per-lane (desc, u_k, v_k, lod_k, out_k) calls so each lane samples its
//!         own uv (exactly the scalar path's per-fragment sample).
//!       - the if/else merge-phi diamonds are FLATTENED to `select` (both side-effect-free
//!         arms execute for all lanes, then a per-lane masked blend picks the result).
//!     Anything still outside this set returns error.NotWidenable and the caller keeps the
//!     proven scalar per-fragment path. CORRECTNESS: each lane's computation is bit-for-bit
//!     the scalar path's computation for that fragment (same ops, same host calls, same
//!     per-lane uv), just packed - the Prism equivalence tests check this lane-by-lane.

const std = @import("std");
const ir = @import("vulcan-ir");
const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Type = ir.types.Type;

pub const Error = std.mem.Allocator.Error || error{NotWidenable};

/// The SIMD width: a 2x2 quad = 4 fragments shaded at once.
pub const lanes: u32 = 4;

/// Whether a type is scalar f32 (the values a widened FS turns into `<4 x f32>`).
fn isF32(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .float => |f| f == .f32,
        else => false,
    };
}

fn isF32Vec(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .vector => |v| v.len == lanes and isF32(func, v.elem),
        else => false,
    };
}

fn isPtr(func: *const Function, ty: Type) bool {
    return func.types.type_kind(ty) == .ptr;
}

/// Widen a lowered fragment `Function` in place to 4-wide SIMD. Tries the straight-line
/// single-block path first. If the FS has the buffer/sampler/branch patterns of the heavy
/// (vkcube-class) FS, falls through to the scalarizing heavy widener. Returns
/// error.NotWidenable for anything outside both subsets (the caller keeps the scalar path).
pub fn widenGraphics(func: *Function) Error!void {
    if (func.blockCount() == 1 and singleBlockWidenable(func)) {
        return widenSingleBlock(func);
    }
    return widenHeavy(func);
}

// Tier 1: the straight-line, buffer-free single-block widener (original path).

/// Whether the single entry block is in the original straight-line, buffer-free,
/// float-only vectorizable subset. (A `load`/`call`/`alloca`/`if`/ptr-param FS is NOT -
/// it goes to the heavy widener instead.)
fn singleBlockWidenable(func: *const Function) bool {
    const entry: Block = @enumFromInt(0);
    for (func.blockParams(entry)) |pv| {
        if (isPtr(func, func.valueType(pv))) return false;
    }
    for (func.blockInsts(entry)) |inst| {
        switch (func.opcode(inst)) {
            .fconst, .iconst => {},
            .arith => |a| if (!isF32(func, func.valueType(a.lhs))) return false,
            .icmp => |c| if (!isF32(func, func.valueType(c.lhs))) return false,
            .select => |s| if (!isF32(func, func.valueType(s.then))) return false,
            .unary => |u| if (u.op != .sqrt or !isF32(func, func.valueType(u.value))) return false,
            .struct_new => |sn| for (func.valueList(sn.fields)) |fv| {
                if (!isF32(func, func.valueType(fv))) return false;
            },
            .extract => if (!isF32(func, func.valueType(func.instResult(inst).?))) return false,
            .store => |st| if (!isF32(func, func.valueType(st.value))) return false,
            .prefetch => return false, // scalar-address hint only, not lane-widenable
            else => return false,
        }
    }
    return true;
}

/// Widen a single-entry-block, straight-line, buffer-free fragment `Function` in place so
/// every f32 value carries 4 fragments (lanes) at once.
fn widenSingleBlock(func: *Function) Error!void {
    const entry: Block = @enumFromInt(0);
    const vec_ty = try f32VecType(func);

    for (func.blockParams(entry)) |pv| {
        if (isF32(func, func.valueType(pv))) func.setValueType(pv, vec_ty);
    }

    const orig = try func.allocator.dupe(Inst, func.blockInsts(entry));
    defer func.allocator.free(orig);

    var new_insts = std.ArrayListUnmanaged(Inst).empty;
    defer new_insts.deinit(func.allocator);

    for (orig) |inst| {
        try new_insts.append(func.allocator, inst);
        switch (func.opcode(inst)) {
            .fconst => {
                const cval = func.instResult(inst).?; // scalar f32 constant (kept scalar)
                const fields = try func.internValues(&.{ cval, cval, cval, cval });
                const splat = try func.createInst(vec_ty, .{ .struct_new = .{ .fields = fields } });
                func.replaceAllUses(cval, splat);
                for (func.valueListMut(fields)) |*f| f.* = cval;
                try new_insts.append(func.allocator, func.definingInst(splat).?);
            },
            .icmp => |c| {
                if (func.types.type_kind(func.valueType(c.lhs)) == .vector) {
                    if (func.instResult(inst)) |rv| func.setValueType(rv, vec_ty);
                }
            },
            else => {
                if (func.instResult(inst)) |rv| {
                    if (isF32(func, func.valueType(rv))) func.setValueType(rv, vec_ty);
                }
            },
        }
    }
    try func.setBlockInsts(entry, new_insts.items);
}

fn f32VecType(func: *Function) Error!Type {
    return func.types.intern(.{ .vector = .{ .len = lanes, .elem = try func.types.intern(.{ .float = .f32 }) } });
}

// Tier 2: the heavy (vkcube-class) widener.
//
// Step A: FLATTEN the CFG to a single block. The lowered FS is a reducible chain of
// straight jumps and if/else "diamonds" (an `if cond {A} else {B}` where A and B each
// jump to a merge block M whose block-params are the SSA "phi" values). With both arms
// side-effect-free, the diamond flattens: emit A's and B's instructions unconditionally,
// then replace each merge param with `select(cond, A_arg, B_arg)`. Straight jumps just
// concatenate the target's instructions. Loops / back-edges / multi-pred-non-diamond
// merges are rejected (-> scalar fallback).
//
// Step B: WIDEN the single flattened block, scalarizing loads (broadcast) and
// call_indirects (gather), as documented at the top of the file.

fn widenHeavy(func: *Function) Error!void {
    // A single-block heavy FS (a textured / derivative / pow FS: buffer/sampler/call but no
    // branches) needs no flattening - widen it directly. A multi-block FS (vkcube's inlined
    // linearToSrgb diamonds) is first flattened to one block.
    if (func.blockCount() > 1) try flattenToSingleBlock(func);
    try widenFlattened(func);
}

/// Flatten the function's CFG into a single block (block0), turning if/else merge-phi
/// diamonds into `select`. Rejects (NotWidenable) anything that is not a reducible chain of
/// straight jumps + side-effect-free diamonds (loops, back-edges, calls/stores inside an
/// arm that must be predicated, etc).
fn flattenToSingleBlock(func: *Function) Error!void {
    const nblocks = func.blockCount();
    if (nblocks < 2) return error.NotWidenable;

    // We walk the CFG from block0 following the chain of straight jumps + if/else diamonds and
    // build the flattened instruction list. Rather than RAUW eagerly during the walk (whose
    // ordering across a chain of merges is hazardous - a merge param can be bound to another
    // merge's param), we record every rewrite in a deferred substitution table `subst` (merge
    // param -> select, straight-jump target param -> passed arg), then resolve it transitively
    // and apply it ONCE at the end. This makes the result independent of walk order.
    var out_insts = std.ArrayListUnmanaged(Inst).empty;
    defer out_insts.deinit(func.allocator);

    var visited = try func.allocator.alloc(bool, nblocks);
    defer func.allocator.free(visited);
    @memset(visited, false);

    // subst[v] = the value v should become (identity until recorded). Sized to the value count
    // BEFORE we create any selects. Selects we create are never themselves substituted, so they
    // need no entry (resolve() treats out-of-range / identity as themselves).
    const nval0 = func.valueCount();
    var subst = try func.allocator.alloc(Value, nval0);
    defer func.allocator.free(subst);
    for (0..nval0) |i| subst[i] = @enumFromInt(i);

    var cur: usize = 0;
    while (true) {
        if (cur >= nblocks) return error.NotWidenable;
        if (visited[cur]) return error.NotWidenable; // back-edge / loop
        visited[cur] = true;
        const block: Block = @enumFromInt(cur);

        var if_inst: ?Inst = null;
        for (func.blockInsts(block)) |inst| {
            if (func.opcode(inst) == .@"if") {
                if (if_inst != null) return error.NotWidenable; // two ifs in a block
                if_inst = inst;
            }
        }

        if (if_inst) |ifi| {
            for (func.blockInsts(block)) |inst| {
                if (inst == ifi) continue;
                if (func.opcode(inst) == .@"if") continue;
                try out_insts.append(func.allocator, inst);
            }
            const cf = func.opcode(ifi).@"if";
            const merge = try flattenDiamond(func, &out_insts, visited, subst, cf);
            cur = @intFromEnum(merge);
            continue;
        }

        for (func.blockInsts(block)) |inst| {
            try out_insts.append(func.allocator, inst);
        }
        const term = func.terminator(block) orelse break; // implicit ret void: final block
        switch (term) {
            .ret => break, // the final block of the chain
            .jump => |j| {
                // A straight jump may carry args to a block with params (a non-diamond merge,
                // e.g. vkcube's block17 -> block18(v88,v95,v102) collecting the 3 channels):
                // record each target param -> passed arg in `subst` (deferred).
                const args = func.blockArgs(j);
                const mparams = func.blockParams(j.target);
                if (args.len != mparams.len) return error.NotWidenable;
                for (mparams, args) |mp, arg| recordSubst(subst, mp, arg);
                cur = @intFromEnum(j.target);
            },
        }
    }

    // Resolve the substitution transitively (a param may map to another param that maps on),
    // then apply it across every instruction operand + terminator. We apply by RAUW per entry.
    for (0..nval0) |i| {
        const from: Value = @enumFromInt(i);
        const to = resolveSubst(subst, from);
        if (to != from) func.replaceAllUses(from, to);
    }

    const entry: Block = @enumFromInt(0);
    try func.setBlockInsts(entry, out_insts.items);
    func.setTerminator(entry, .{ .ret = null });

    // Neutralize every other block (empty insts + params + ret) so the widen pass below only
    // sees block0.
    var bi: usize = 1;
    while (bi < nblocks) : (bi += 1) {
        try func.setBlockInsts(@enumFromInt(bi), &.{});
        try func.setBlockParams(@enumFromInt(bi), &.{});
        func.setTerminator(@enumFromInt(bi), .{ .ret = null });
    }
}

/// Record `from -> to` in the substitution table (no-op if `from` is out of range, which only
/// happens for a freshly-created select, which is never substituted).
fn recordSubst(subst: []Value, from: Value, to: Value) void {
    const i = @intFromEnum(from);
    if (i < subst.len) subst[i] = to;
}

/// Resolve a value through the substitution table transitively (following chains of param ->
/// arg -> arg). A select (out-of-range index) resolves to itself.
fn resolveSubst(subst: []const Value, v: Value) Value {
    var cur = v;
    var guard: usize = 0;
    while (@intFromEnum(cur) < subst.len) {
        const next = subst[@intFromEnum(cur)];
        if (next == cur) break;
        cur = next;
        guard += 1;
        if (guard > subst.len) break; // defensive: no infinite loop on a cycle
    }
    return cur;
}

/// Flatten one if/else diamond. `cf` is the `if`. Emits both arms' side-effect-free
/// instructions into `out`, then for each of the merge block's params creates a
/// `select(cond, then_arg, else_arg)` (appended to `out`) and records merge_param -> select
/// in `subst`. Returns the merge block to continue the walk from. Rejects shapes it cannot
/// prove equivalent. The select args are resolved through `subst` first (an arm may pass an
/// already-substituted param).
fn flattenDiamond(
    func: *Function,
    out: *std.ArrayListUnmanaged(Inst),
    visited: []bool,
    subst: []Value,
    cf: ir.function.If,
) Error!Block {
    const then_b = cf.then.target;
    const else_b = cf.@"else".target;
    const then_merge = try emitArm(func, out, visited, then_b);
    const else_merge = try emitArm(func, out, visited, else_b);
    if (then_merge.target != else_merge.target) return error.NotWidenable;
    const merge = then_merge.target;

    // Snapshot the arm args (createInst may realloc the value-list pool the slices point into).
    const then_args = try func.allocator.dupe(Value, then_merge.args);
    defer func.allocator.free(then_args);
    const else_args = try func.allocator.dupe(Value, else_merge.args);
    defer func.allocator.free(else_args);
    const mparams = func.blockParams(merge);
    if (then_args.len != mparams.len or else_args.len != mparams.len) return error.NotWidenable;

    // Snapshot the merge params too (setBlockParams clears them).
    const mparams_copy = try func.allocator.dupe(Value, mparams);
    defer func.allocator.free(mparams_copy);

    for (mparams_copy, 0..) |mp, i| {
        const sel = try func.createInst(func.valueType(mp), .{ .select = .{
            .cond = resolveSubst(subst, cf.cond),
            .then = resolveSubst(subst, then_args[i]),
            .@"else" = resolveSubst(subst, else_args[i]),
        } });
        try out.append(func.allocator, func.definingInst(sel).?);
        recordSubst(subst, mp, sel);
    }
    try func.setBlockParams(merge, &.{});
    return merge;
}

const ArmResult = struct { target: Block, args: []const Value };

/// Emit a diamond arm's side-effect-free instructions into `out` and return its jump target
/// + args (the phi values it passes to the merge). The arm must end in a `jump M(args)` and
/// contain no `if`. We allow `call_indirect` (the pow math_fn call) and `arith`/`select`
/// etc - they are pure for our FSes (the math_fn is a pure transcendental, we never predicate
/// a store or a sampler write inside an arm). A `store` inside an arm IS rejected (it would
/// need real predication).
fn emitArm(func: *Function, out: *std.ArrayListUnmanaged(Inst), visited: []bool, arm: Block) Error!ArmResult {
    const bi = @intFromEnum(arm);
    if (visited[bi]) return error.NotWidenable;
    visited[bi] = true;
    for (func.blockInsts(arm)) |inst| {
        switch (func.opcode(inst)) {
            .@"if" => return error.NotWidenable,
            .store => return error.NotWidenable, // a predicated store needs real masking
            .prefetch => return error.NotWidenable, // conservative: no prefetch reaches this shader path
            else => try out.append(func.allocator, inst),
        }
    }
    const term = func.terminator(arm) orelse return error.NotWidenable;
    switch (term) {
        .jump => |j| return .{ .target = j.target, .args = func.blockArgs(j) },
        .ret => return error.NotWidenable,
    }
}

// Step B: widen the single flattened block, scalarizing loads (broadcast) and
// call_indirects (gather). Processes the block linearly, maintaining a map from each
// sampler-out alloca to its 4 per-lane out slots so the following component reloads gather
// across lanes instead of broadcasting.

/// Maps a sampler out-slot alloca (Value) to its 4 per-lane replacement slots.
const SamplerSlots = struct { base: Value, slots: [lanes]Value };

fn widenFlattened(func: *Function) Error!void {
    const entry: Block = @enumFromInt(0);
    const vec_ty = try f32VecType(func);
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const u128_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 128 } });

    // Retype f32 PARAMS to <4 x f32>. Ptr params stay scalar (broadcast-invariant pointers).
    for (func.blockParams(entry)) |pv| {
        if (isF32(func, func.valueType(pv))) func.setValueType(pv, vec_ty);
    }

    const orig = try func.allocator.dupe(Inst, func.blockInsts(entry));
    defer func.allocator.free(orig);

    var new_insts = std.ArrayListUnmanaged(Inst).empty;
    defer new_insts.deinit(func.allocator);

    // Sampler out-slot bookkeeping (small: a handful of samples per FS).
    var sampler_map = std.ArrayListUnmanaged(SamplerSlots).empty;
    defer sampler_map.deinit(func.allocator);

    for (orig) |inst| {
        const op = func.opcode(inst);
        switch (op) {
            // ptr-offset selectors / op codes: stay scalar.
            .iconst => try new_insts.append(func.allocator, inst),
            .fconst => {
                const cval = func.instResult(inst).?;
                try new_insts.append(func.allocator, inst);
                try splatAndReplace(func, &new_insts, cval, vec_ty);
            },
            .arith => |a| {
                const lt = func.valueType(a.lhs);
                if (isPtr(func, lt)) {
                    // A sampler out-slot offset-compute (`out_ptr + k*4`) is dead: the gather
                    // creates its own per-lane address computes. Drop it (re-emitting it would
                    // reference the dropped original out_ptr alloca - a dangling value).
                    if (isSamplerBase(sampler_map.items, a.lhs)) continue;
                    try new_insts.append(func.allocator, inst); // a live ptr + offset
                } else {
                    // Float arith. The lhs is f32 (not yet widened) or already <4 x f32> (an
                    // operand a prior gather/broadcast replaced with a vector). Either way the
                    // result is a vector.
                    if (!isF32(func, lt) and !isF32Vec(func, lt)) return error.NotWidenable;
                    if (func.instResult(inst)) |rv| func.setValueType(rv, vec_ty);
                    try new_insts.append(func.allocator, inst);
                }
            },
            .arith_imm => try new_insts.append(func.allocator, inst), // grad_buf + N (ptr)
            .icmp => |c| {
                const lt = func.valueType(c.lhs);
                if (!isF32(func, lt) and !isF32Vec(func, lt)) return error.NotWidenable;
                if (func.instResult(inst)) |rv| func.setValueType(rv, vec_ty); // per-lane mask
                try new_insts.append(func.allocator, inst);
            },
            .select => |s| {
                const tt = func.valueType(s.then);
                if (!isF32(func, tt) and !isF32Vec(func, tt)) return error.NotWidenable;
                if (func.instResult(inst)) |rv| func.setValueType(rv, vec_ty);
                try new_insts.append(func.allocator, inst);
            },
            .unary => |u| {
                if (u.op != .sqrt) return error.NotWidenable;
                if (func.instResult(inst)) |rv| func.setValueType(rv, vec_ty);
                try new_insts.append(func.allocator, inst);
            },
            .struct_new, .extract => {
                if (func.instResult(inst)) |rv| {
                    if (isF32(func, func.valueType(rv))) func.setValueType(rv, vec_ty);
                }
                try new_insts.append(func.allocator, inst);
            },
            .alloca => {
                // A sampler out-slot alloca: replaced by 4 per-lane slots created at the call.
                // We DROP the original alloca (it becomes dead once the call gather replaces
                // the reloads). If some FS used an alloca for something else, that is outside
                // our subset - reject to stay correct.
                const rv = func.instResult(inst).?;
                if (!isSamplerOutSlot(func, rv)) return error.NotWidenable;
                // Defer creating slots to the call site. Record the base here as pending by
                // emitting nothing now (the call gather looks it up by base value).
                // We still need the base value to exist for the call's arg lookup, but the
                // call uses our per-lane slots, not this alloca, so dropping is safe. Record a
                // placeholder entry. Slots get filled at the call.
                try sampler_map.append(func.allocator, .{ .base = rv, .slots = undefined });
            },
            .load => |l| {
                const rv = func.instResult(inst).?;
                if (!isF32(func, func.valueType(rv))) return error.NotWidenable;
                // Is this a reload off a sampler out-slot? Gather across lanes if so.
                if (samplerReloadFor(func, &sampler_map, l.ptr)) |gather| {
                    const packed_val = try gatherSamplerComp(func, &new_insts, gather.slots, gather.comp, vec_ty, f32_t, ptr_t);
                    func.replaceAllUses(rv, packed_val);
                } else {
                    // A lane-invariant load (grad_buf / descriptor): one scalar load + splat.
                    const scalar_load = try func.createInst(f32_t, .{ .load = .{ .ptr = l.ptr } });
                    try new_insts.append(func.allocator, func.definingInst(scalar_load).?);
                    const fields = try func.internValues(&.{ scalar_load, scalar_load, scalar_load, scalar_load });
                    const splat = try func.createInst(vec_ty, .{ .struct_new = .{ .fields = fields } });
                    func.replaceAllUses(rv, splat);
                    try new_insts.append(func.allocator, func.definingInst(splat).?);
                }
            },
            .call_indirect => |c| {
                try gatherCall(func, &new_insts, &sampler_map, inst, c, vec_ty, f32_t, ptr_t, u128_t);
            },
            .store => try new_insts.append(func.allocator, inst), // tagged output store of <4 x f32>
            .prefetch => try new_insts.append(func.allocator, inst), // scalar-address hint, passes through unchanged
            // dot is an aarch64+dotprod-only INT8 op; a shader function never contains one. Reject
            // conservatively rather than assume a lane-widening it has never been proven correct for.
            .dot => return error.NotWidenable,
            // matmul is an et-soc tensor-tile op; a shader function never contains one either.
            .matmul => return error.NotWidenable,
            .convert, .call, .global_addr, .@"if" => return error.NotWidenable,
        }
    }
    try func.setBlockInsts(entry, new_insts.items);
}

/// Splat scalar f32 `sv` to <4 x f32> and re-point every use of `sv` to the splat. (The
/// splat's own 4 fields stay pointed at `sv`.)
fn splatAndReplace(func: *Function, out: *std.ArrayListUnmanaged(Inst), sv: Value, vec_ty: Type) Error!void {
    const fields = try func.internValues(&.{ sv, sv, sv, sv });
    const splat = try func.createInst(vec_ty, .{ .struct_new = .{ .fields = fields } });
    func.replaceAllUses(sv, splat);
    for (func.valueListMut(fields)) |*f| f.* = sv;
    try out.append(func.allocator, func.definingInst(splat).?);
}

/// Whether `rv` (an alloca result) is a sampler out-slot: it is the `out_ptr` arg of a
/// `call_indirect(desc, u, v, out_ptr)` (the void sampler call). Pattern-matched over block0.
fn isSamplerOutSlot(func: *const Function, rv: Value) bool {
    const entry: Block = @enumFromInt(0);
    for (func.blockInsts(entry)) |inst| {
        if (func.opcode(inst) == .call_indirect) {
            const c = func.opcode(inst).call_indirect;
            if (func.instResult(inst) != null) continue; // value call (math_fn), not sampler
            const args = func.valueList(c.args);
            if (args.len == 5 and args[4] == rv) return true;
        }
    }
    return false;
}

/// Whether `v` is a recorded sampler out-slot base (the original out_ptr alloca).
fn isSamplerBase(map: []const SamplerSlots, v: Value) bool {
    for (map) |e| if (e.base == v) return true;
    return false;
}

const SamplerReload = struct { slots: [lanes]Value, comp: u32 };

/// If `ptr` reads a sampler out-slot (base or base+k*4) of a recorded sampler call, return
/// the per-lane slots + the component index.
fn samplerReloadFor(func: *const Function, map: *const std.ArrayListUnmanaged(SamplerSlots), ptr: Value) ?SamplerReload {
    for (map.items) |entry| {
        if (ptr == entry.base) return .{ .slots = entry.slots, .comp = 0 };
        if (offsetFromBase(func, ptr, entry.base)) |off| {
            if (off >= 0 and @rem(off, 4) == 0) {
                const c = @divTrunc(off, 4);
                if (c >= 1 and c < 4) return .{ .slots = entry.slots, .comp = @intCast(c) };
            }
        }
    }
    return null;
}

/// Gather RGBA component `comp` across the 4 per-lane sampler out slots into a <4 x f32>.
fn gatherSamplerComp(func: *Function, out: *std.ArrayListUnmanaged(Inst), slots: [lanes]Value, comp: u32, vec_ty: Type, f32_t: Type, ptr_t: Type) Error!Value {
    var lane_vals: [lanes]Value = undefined;
    var lk: u32 = 0;
    while (lk < lanes) : (lk += 1) {
        const base = slots[lk];
        const eptr = if (comp == 0) base else blk: {
            const off = try func.createInst(ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = @intCast(comp * 4) } });
            try out.append(func.allocator, func.definingInst(off).?);
            break :blk off;
        };
        const ld = try func.createInst(f32_t, .{ .load = .{ .ptr = eptr } });
        try out.append(func.allocator, func.definingInst(ld).?);
        lane_vals[lk] = ld;
    }
    const fields = try func.internValues(&lane_vals);
    const packed_comp = try func.createInst(vec_ty, .{ .struct_new = .{ .fields = fields } });
    try out.append(func.allocator, func.definingInst(packed_comp).?);
    return packed_comp;
}

/// Emit one arithmetic instruction into `out` and return its result value (widen-path helper).
fn wArith(func: *Function, out: *std.ArrayListUnmanaged(Inst), ty: Type, op: ir.function.BinOp, a: Value, b: Value) Error!Value {
    const r = try func.createInst(ty, .{ .arith = .{ .op = op, .lhs = a, .rhs = b } });
    try out.append(func.allocator, func.definingInst(r).?);
    return r;
}

/// Emit a single instruction (payload) into `out` and return its result value.
fn wInst(func: *Function, out: *std.ArrayListUnmanaged(Inst), ty: Type, payload: ir.function.Opcode) Error!Value {
    const r = try func.createInst(ty, payload);
    try out.append(func.allocator, func.definingInst(r).?);
    return r;
}

/// Compute the per-quad IMPLICIT level-of-detail for a texture sample from the 2x2 quad's
/// coordinate lanes + the bound descriptor's dimensions. Lane layout 0=TL,1=TR,2=BL,3=BR, so
/// d/dx = lane1-lane0 and d/dy = lane2-lane0. LOD = 0.5*log2(rho2), rho2 = max over the two screen
/// axes of |d(uv)/d*|^2 in TEXEL space (uv scaled by the descriptor width/height, loaded at byte
/// offsets 8/12 of TexDesc). max(a,b) uses the |a-b| sign-bit trick (no fcmp/select needed); log2
/// is a fast bit-trick (exact log2 is not an IR primitive) - accurate to ~0.03, ample for mip
/// selection. Returns a scalar f32 (the caller broadcasts it to all lanes). A non-mipmapped
/// texture ignores any LOD (the sampler clamps to the base level), so this never perturbs the
/// existing non-mip texturing path.
fn computeQuadLod(func: *Function, out: *std.ArrayListUnmanaged(Inst), desc: Value, u: Value, v: Value, f32_t: Type) Error!Value {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_ty = func.valueType(desc);
    const cu0 = try laneOf(func, out, u, 0, f32_t);
    const cu1 = try laneOf(func, out, u, 1, f32_t);
    const cu2 = try laneOf(func, out, u, 2, f32_t);
    const cv0 = try laneOf(func, out, v, 0, f32_t);
    const cv1 = try laneOf(func, out, v, 1, f32_t);
    const cv2 = try laneOf(func, out, v, 2, f32_t);
    const dudx = try wArith(func, out, f32_t, .sub, cu1, cu0);
    const dvdx = try wArith(func, out, f32_t, .sub, cv1, cv0);
    const dudy = try wArith(func, out, f32_t, .sub, cu2, cu0);
    const dvdy = try wArith(func, out, f32_t, .sub, cv2, cv0);
    // Load width/height (u32 at TexDesc offsets 8/12: pixels ptr is 8 bytes, then width, height)
    // and numerically convert to f32.
    const wptr = try wInst(func, out, ptr_ty, .{ .arith_imm = .{ .op = .add, .lhs = desc, .imm = 8 } });
    const wf = try wInst(func, out, f32_t, .{ .convert = .{ .value = try wInst(func, out, i32_t, .{ .load = .{ .ptr = wptr } }) } });
    const hptr = try wInst(func, out, ptr_ty, .{ .arith_imm = .{ .op = .add, .lhs = desc, .imm = 12 } });
    const hf = try wInst(func, out, f32_t, .{ .convert = .{ .value = try wInst(func, out, i32_t, .{ .load = .{ .ptr = hptr } }) } });
    // Texel-space gradient squared-lengths along each screen axis.
    const ax = try wArith(func, out, f32_t, .mul, dudx, wf);
    const bx = try wArith(func, out, f32_t, .mul, dvdx, hf);
    const ay = try wArith(func, out, f32_t, .mul, dudy, wf);
    const by = try wArith(func, out, f32_t, .mul, dvdy, hf);
    const lenX2 = try wArith(func, out, f32_t, .add, try wArith(func, out, f32_t, .mul, ax, ax), try wArith(func, out, f32_t, .mul, bx, bx));
    const lenY2 = try wArith(func, out, f32_t, .add, try wArith(func, out, f32_t, .mul, ay, ay), try wArith(func, out, f32_t, .mul, by, by));
    // rho2 = max(lenX2, lenY2) = 0.5*(a+b+|a-b|); |x| clears the IEEE sign bit.
    const diff = try wArith(func, out, f32_t, .sub, lenX2, lenY2);
    const absmask = try wInst(func, out, i32_t, .{ .arith_imm = .{ .op = .bit_and, .lhs = try wInst(func, out, i32_t, .{ .unary = .{ .op = .reinterpret, .value = diff } }), .imm = 0x7fffffff } });
    const absdiff = try wInst(func, out, f32_t, .{ .unary = .{ .op = .reinterpret, .value = absmask } });
    const sum = try wArith(func, out, f32_t, .add, lenX2, lenY2);
    const half = try wInst(func, out, f32_t, .{ .fconst = 0.5 });
    const rho2 = try wArith(func, out, f32_t, .mul, try wArith(func, out, f32_t, .add, sum, absdiff), half);
    // Isotropic LOD = 0.5*log2(rho2) via the IEEE-bit trick (no log2 IR primitive): full log2(x) ~=
    // bits(x)*2^-23 - 126.94269504, so the half-scale form of a squared length uses 2^-24 and the
    // halved bias. Avoids DIVISION, which the host scalar path does not lower correctly here.
    const rf = try wInst(func, out, f32_t, .{ .convert = .{ .value = try wInst(func, out, i32_t, .{ .unary = .{ .op = .reinterpret, .value = rho2 } }) } });
    const scale = try wInst(func, out, f32_t, .{ .fconst = 5.9604644775390625e-8 }); // 2^-24
    const bias = try wInst(func, out, f32_t, .{ .fconst = 63.47134752 });
    return try wArith(func, out, f32_t, .sub, try wArith(func, out, f32_t, .mul, rf, scale), bias);
}

/// Scalarize (GATHER) a `call_indirect` across the 4 lanes.
fn gatherCall(
    func: *Function,
    out: *std.ArrayListUnmanaged(Inst),
    sampler_map: *std.ArrayListUnmanaged(SamplerSlots),
    inst: Inst,
    c: ir.function.CallIndirect,
    vec_ty: Type,
    f32_t: Type,
    ptr_t: Type,
    u128_t: Type,
) Error!void {
    // OWN a copy of the args: `func.valueList` returns a slice INTO the function's
    // `value_lists` pool, and the per-lane loop below calls `internValues` (which
    // `appendSlice`s to that same pool and can REALLOCATE it). A borrowed slice would
    // dangle after the first lane's intern - reading garbage args on later lanes. In a
    // debug build that 0xAA-faults. In release it reads stale/reused memory, which is the
    // data-dependent per-quad speckle this gather produced (e.g. glmark2 phong).
    const args = try func.allocator.dupe(Value, func.valueList(c.args));
    defer func.allocator.free(args);

    if (func.instResult(inst)) |rv| {
        // Value-returning gather (math_fn: f32 fn(op:i32, a:f32, b:f32)).
        if (!isF32(func, func.valueType(rv))) return error.NotWidenable;
        var lane_results: [lanes]Value = undefined;
        var lane: u32 = 0;
        while (lane < lanes) : (lane += 1) {
            const lane_args = try func.allocator.alloc(Value, args.len);
            defer func.allocator.free(lane_args);
            for (args, 0..) |a, ai| lane_args[ai] = try laneOf(func, out, a, lane, f32_t);
            const list = try func.internValues(lane_args);
            const call_res = try func.createInst(f32_t, .{ .call_indirect = .{ .target = c.target, .args = list } });
            try out.append(func.allocator, func.definingInst(call_res).?);
            lane_results[lane] = call_res;
        }
        const fields = try func.internValues(&lane_results);
        const packed_res = try func.createInst(vec_ty, .{ .struct_new = .{ .fields = fields } });
        func.replaceAllUses(rv, packed_res);
        try out.append(func.allocator, func.definingInst(packed_res).?);
        return;
    }

    // Void gather (sampler_fn(desc, u, v, lod, out_ptr)). Create 4 per-lane out slots + 4 per-lane
    // calls (desc broadcast, u_k/v_k/lod_k extracted). Record the slots so the following component
    // reloads gather across them.
    if (args.len != 5) return error.NotWidenable;
    const desc = args[0];
    const u = args[1];
    const v = args[2];
    const out_ptr = args[4];
    // LOD selection. lower.zig passes fconst 0 as the lod for an IMPLICIT sample (texture()) and
    // the real LOD value for an EXPLICIT sample (textureLod). For implicit we compute the per-quad
    // level-of-detail from the coordinate lanes + the descriptor's dimensions (this is what makes a
    // plain texture() auto-select mip levels). For explicit we KEEP the given LOD (the app chose
    // the level directly - derivatives must not override it). A non-mipmapped texture ignores the
    // LOD (the sampler clamps to base), so neither path disturbs non-mip texturing.
    const lod = if (isFconstZero(func, args[3]))
        try computeQuadLod(func, out, desc, u, v, f32_t)
    else
        args[3];

    var slots: [lanes]Value = undefined;
    var lane: u32 = 0;
    while (lane < lanes) : (lane += 1) {
        const slot = try func.createInst(ptr_t, .{ .alloca = .{ .elem = u128_t } });
        try out.append(func.allocator, func.definingInst(slot).?);
        slots[lane] = slot;
        const u_k = try laneOf(func, out, u, lane, f32_t);
        const v_k = try laneOf(func, out, v, lane, f32_t);
        const lod_k = try laneOf(func, out, lod, lane, f32_t);
        const list = try func.internValues(&.{ desc, u_k, v_k, lod_k, slot });
        // A result-less (void) call. createInst makes a result value we never read. The isel
        // simply leaves it dead. Emitting a dead f32 result is harmless and keeps us off the
        // function.zig API. Mark it by giving it a zero-width-equivalent: we use f32_t.
        const call = try func.createInst(f32_t, .{ .call_indirect = .{ .target = c.target, .args = list } });
        try out.append(func.allocator, func.definingInst(call).?);
    }
    // Record the slots against the original out_ptr alloca base.
    for (sampler_map.items) |*e| {
        if (e.base == out_ptr) {
            e.slots = slots;
            return;
        }
    }
    try sampler_map.append(func.allocator, .{ .base = out_ptr, .slots = slots });
}

/// Get lane `lane` of `v` as a scalar f32. Scalar `v` (a ptr or i32 selector) is returned
/// unchanged. A `<4 x f32>` gets an `extract`.
fn laneOf(func: *Function, out: *std.ArrayListUnmanaged(Inst), v: Value, lane: u32, f32_t: Type) Error!Value {
    if (isF32Vec(func, func.valueType(v))) {
        const ex = try func.createInst(f32_t, .{ .extract = .{ .aggregate = v, .index = lane } });
        try out.append(func.allocator, func.definingInst(ex).?);
        return ex;
    }
    return v;
}

/// Whether `v` is the constant float 0 - the placeholder lod lower.zig passes for an IMPLICIT
/// texture() sample (an explicit textureLod passes a real value instead). Distinguishes the two
/// so the widener computes the quad LOD only for implicit samples. A prior pass may already have
/// SPLATTED the scalar fconst into a `<4 x f32>` (struct_new of the same scalar), so unwrap that.
/// LIMITATION: a literal `textureLod(s, uv, 0.0)` is indistinguishable from the implicit
/// placeholder and is treated as implicit; a uniform/computed LOD of 0 is a real load and is
/// correctly explicit. Real shaders that want the base level via an explicit LOD use a uniform.
fn isFconstZero(func: *const Function, v: Value) bool {
    const di = func.definingInst(v) orelse return false;
    return switch (func.opcode(di)) {
        .fconst => |c| c == 0,
        .struct_new => |s| blk: {
            const fields = func.valueList(s.fields);
            break :blk fields.len > 0 and isFconstZero(func, fields[0]);
        },
        else => false,
    };
}

/// If `ptr` is `base + N` (an `arith_imm`, or an `arith` with an iconst rhs), return N.
fn offsetFromBase(func: *const Function, ptr: Value, base: Value) ?i64 {
    const di = func.definingInst(ptr) orelse return null;
    switch (func.opcode(di)) {
        .arith_imm => |a| if (a.lhs == base and a.op == .add) return a.imm,
        .arith => |a| if (a.lhs == base and a.op == .add) {
            const ri = func.definingInst(a.rhs) orelse return null;
            switch (func.opcode(ri)) {
                .iconst => |k| return k,
                else => return null,
            }
        },
        else => {},
    }
    return null;
}

// Step A structural tests: prove the heavy widener produces well-formed single-block IR
// with the expected broadcast (splat), gather (4 calls + pack), and flatten (select) shapes.
// (End-to-end lane-by-lane correctness vs the scalar golden is proven in Prism's spirv_jit
// tests, which JIT + EXECUTE both paths. Here we assert IR structure only.)

const testing = std.testing;

fn countOp(func: *const Function, block: Block, tag: std.meta.Tag(ir.function.Opcode)) usize {
    var n: usize = 0;
    for (func.blockInsts(block)) |inst| {
        if (std.meta.activeTag(func.opcode(inst)) == tag) n += 1;
    }
    return n;
}

test "widen heavy: a grad_buf load BROADCASTS (one scalar load + a 4-splat), result vector" {
    const gpa = testing.allocator;
    var func = Function.init(gpa);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const vin = try func.appendBlockParam(entry, f32_t); // one f32 varying input
    const gbuf = try func.appendBlockParam(entry, ptr_t); // a grad_buf-like pointer param
    // r = load(gbuf) + vin. Store r to a slot (color_out 0).
    const g = try func.appendInst(entry, f32_t, .{ .load = .{ .ptr = gbuf } });
    const r = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = g, .rhs = vin } });
    const slot = try func.appendInst(entry, ptr_t, .{ .iconst = 0 });
    try func.appendStore(entry, r, slot);

    try widenGraphics(&func);

    // Single block. The load stays scalar (its result f32) but is splatted by a struct_new to
    // a <4 x f32>. The input + result are vectors.
    try testing.expectEqual(@as(usize, 1), func.blockCount());
    try testing.expect(isF32Vec(&func, func.valueType(vin)));
    // Exactly one scalar load (broadcast), and at least one struct_new (the splat).
    try testing.expectEqual(@as(usize, 1), countOp(&func, entry, .load));
    try testing.expect(countOp(&func, entry, .struct_new) >= 1);
    // The arith result is a vector now.
    try testing.expect(isF32Vec(&func, func.valueType(r)));
}

test "widen heavy: a math_fn call_indirect GATHERS to 4 scalar calls + a pack" {
    const gpa = testing.allocator;
    var func = Function.init(gpa);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, f32_t);
    const mathfn = try func.appendBlockParam(entry, ptr_t);
    const sel = try func.appendInst(entry, i32_t, .{ .iconst = 0 }); // op selector
    const two = try func.appendInst(entry, f32_t, .{ .fconst = 2.0 });
    const p = try func.appendInst(entry, f32_t, .{ .call_indirect = .{ .target = mathfn, .args = try func.internValues(&.{ sel, x, two }) } });
    const slot = try func.appendInst(entry, ptr_t, .{ .iconst = 0 });
    try func.appendStore(entry, p, slot);

    try widenGraphics(&func);

    try testing.expectEqual(@as(usize, 1), func.blockCount());
    // The single value call became 4 lane calls (gather).
    try testing.expectEqual(@as(usize, 4), countOp(&func, entry, .call_indirect));
    // 4 lane extracts of x + a final pack (struct_new). The 2.0 splat is also a struct_new.
    try testing.expect(countOp(&func, entry, .extract) >= 4);
    try testing.expect(countOp(&func, entry, .struct_new) >= 1);
    // The store now holds a <4 x f32> (the packed gather result re-pointed every use of the
    // original f32 call result `p`, which is itself dropped).
    const last = func.blockInsts(entry)[func.blockInsts(entry).len - 1];
    try testing.expect(isF32Vec(&func, func.valueType(func.opcode(last).store.value)));
}

test "widen heavy: an if/else merge-phi diamond FLATTENS to one block with a select" {
    const gpa = testing.allocator;
    var func = Function.init(gpa);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);

    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();

    const x = try func.appendBlockParam(entry, f32_t);
    // cond = x > 0.5, then arm = x*2, else arm = x*3, merge picks via phi.
    const half = try func.appendInst(entry, f32_t, .{ .fconst = 0.5 });
    const cond = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = half } });
    try func.appendIf(entry, cond, .{ .target = then_b }, .{ .target = else_b });

    const two = try func.appendInst(then_b, f32_t, .{ .fconst = 2.0 });
    const tval = try func.appendInst(then_b, f32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = two } });
    try func.setJump(then_b, merge, &.{tval});

    const three = try func.appendInst(else_b, f32_t, .{ .fconst = 3.0 });
    const eval = try func.appendInst(else_b, f32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = three } });
    try func.setJump(else_b, merge, &.{eval});

    const mp = try func.appendBlockParam(merge, f32_t);
    const slot = try func.appendInst(merge, ptr_t, .{ .iconst = 0 });
    try func.appendStore(merge, mp, slot);

    try widenGraphics(&func);

    // Flattened: ALL content collapses into block0 (sibling blocks are emptied, not removed -
    // value handles are dense indices, so blocks cannot be deleted). block0 holds a `select`
    // replacing the merge phi, and every other block is empty with no params.
    try testing.expect(countOp(&func, @enumFromInt(0), .select) >= 1);
    var bi: usize = 1;
    while (bi < func.blockCount()) : (bi += 1) {
        try testing.expectEqual(@as(usize, 0), func.blockInsts(@enumFromInt(bi)).len);
        try testing.expectEqual(@as(usize, 0), func.blockParams(@enumFromInt(bi)).len);
    }
    // Both arms' arithmetic execute unconditionally (vectorized): the result is a vector.
    try testing.expect(isF32Vec(&func, func.valueType(x)));
}

test "widen single-block straight-line still works (the original fast path)" {
    const gpa = testing.allocator;
    var func = Function.init(gpa);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);
    const b = try func.appendBlockParam(entry, f32_t);
    const s = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const slot = try func.appendInst(entry, ptr_t, .{ .iconst = 0 });
    try func.appendStore(entry, s, slot);

    try widenGraphics(&func);
    try testing.expectEqual(@as(usize, 1), func.blockCount());
    try testing.expect(isF32Vec(&func, func.valueType(a)));
    try testing.expect(isF32Vec(&func, func.valueType(s)));
}
