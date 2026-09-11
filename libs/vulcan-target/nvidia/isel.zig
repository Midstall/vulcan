//! NVIDIA SASS instruction selection. This module lowers a Vulcan IR function to
//! a compute kernel or a graphics shader.
//!
//! Kernels are leaf functions. The isel inlines calls before it runs, so there
//! is no call stack. The GPU has about 255 GPRs, so register allocation stays
//! simple: a pointer takes an even-aligned register pair, and a boolean takes a
//! predicate register (P0 to P5, where P6 holds the 64-bit-add carry). Kernel
//! ABI: parameters arrive in constant bank 0 at the caller's `Abi.param_base`,
//! and `vulcan-gpu` places them. A kernel that
//! returns a value reads a 64-bit output pointer first (its `ret` stores the
//! result there). A void compute kernel has no output pointer. Each parameter
//! then loads in order: the tagged invocation ID comes from the hardware thread
//! ID (S2R), a pointer loads as a 64-bit pair from the constant bank, and a
//! scalar loads as one value. Memory load and store use LDG and STG through a
//! 64-bit pointer pair. Pointer arithmetic uses a 64-bit IADD3 carry chain: the
//! low add carries out, and the high `.X` add carries in. Control flow uses BRA
//! with block-parameter edge moves. schedule.zig then assigns write barriers to
//! the variable-latency ops (LDG, S2R) and adds waits before their consumers.
//!
//! Validation checks the structure of the emitted instruction stream. Live
//! execution happens later, in prism's compute dispatch. Unsupported IR (calls,
//! aggregates, integer divide) makes this module return `error.Unsupported`.

const std = @import("std");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");
const opt = @import("vulcan-opt");
const encode = @import("encode.zig");
const schedule = @import("schedule.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Terminator = ir.function.Terminator;
const Inst = encode.Inst;

pub const Error = std.mem.Allocator.Error || gpu.abi.Error || error{Unsupported};

/// The default NVIDIA parameter ABI. `param_base` 0x160 is the CUDA driver convention, where a
/// driver-owned block sits in front of the kernel parameters. A runtime that binds its own
/// parameter buffer as the base of constant bank 0 passes 0 instead, which is what the
/// hardware-verified sm_120 dispatch in nvidia.zig does. Both values are correct for their own
/// binding convention, so the caller chooses.
pub const nvidia_abi: gpu.Abi = .{
    .param_base = 0x160,
    .pointer_bytes = 8,
    .param_align = 4,
    .max_shared_bytes = 48 * 1024,
    // The hardware holds a thread index and a workgroup index per AXIS, in SR_TID_X/Y/Z and
    // SR_CTAID_X/Y/Z, so this backend never splits a linear identifier. Only the grid size
    // comes out of the parameter block.
    .linear_thread_id = false,
};
const bank0: u5 = 0;

/// Graphics prologue padding. These are throwaway instructions emitted before
/// the first attribute fetch or color write. They wait out the asynchronous
/// hardware delivery of sysvals and barycentrics into the low registers (a
/// clean threshold of 4, with 6 used for margin). Each pad instruction writes
/// to a dedicated high scratch register, not RZ, because a write to RZ can
/// retire instantly and skip the cycles the delivery window needs. The register
/// allocator excludes this register from its pool (see assignLocs).
const graphics_prologue_pad: u32 = 6;
const graphics_pad_reg: u8 = 40;

/// Reserved registers: R0 and R1 are scratch, and R2:R3 hold the 64-bit output
/// pointer. Values get GPRs starting at R4.
const r_scratch: u8 = 0;
const r_scratch2: u8 = 1; // second prologue scratch register, for invocation-ID computation
const r_outptr: u8 = 2; // pair R2:R3
const value_reg_base: u8 = 4;

/// The GPR the ROP reads gl_FragDepth from at EXIT. NAK lays out fragment
/// outputs as a fixed contiguous block: [RT0 c0..c3, ..., RT(N-1) c0..c3,
/// sample-mask, depth], pinned to R0, R1, and so on (OpRegOut src[i] maps to
/// R[i]). N color targets occupy R0..R[4N-1], the always-reserved sample-mask
/// slot is R[4N], and the depth value lands at R[4N+1]. A fragment shader that
/// writes depth reserves that register, excluding it from the allocator pool,
/// and moves the depth value into it. The SPH's OMAP_DEPTH flag tells the ROP
/// to read the fragment depth from there. The ROP derives the same register
/// from omap_targets, so MRT and gl_FragDepth work together correctly.
fn fragDepthReg(func: *const Function) u8 {
    return 4 * colorTargetCount(func) + 1;
}

/// Whether the fragment shader stores gl_FragDepth (a store the frontend tagged
/// `frag_depth`). Such a shader routes the depth into `frag_depth_out_reg` and
/// sets SPH OMAP_DEPTH, so the ROP reads the depth from the shader instead of
/// the interpolated z value.
fn writesFragDepth(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .store and attrTag(func, func.opcode(inst).store.ptr, "frag_depth") != null) return true;
        }
    }
    return false;
}

/// The number of render targets a fragment shader writes (MRT). The frontend
/// tags each color store `color_out` = target*4 + component, so the highest
/// such tag gives the target count. The ROP reads target T's RGBA from
/// R[T*4 .. T*4+3] (the fixed fragment-shader-output register block), so N
/// targets occupy R0..R[4N-1]. assignLocs reserves those registers when N > 1.
/// When N == 1, the existing R0..R3 color path stays unchanged. Returns 1 for a
/// single-RT or non-fragment shader (the default), up to 8.
fn colorTargetCount(func: *const Function) u8 {
    var max_comp: i32 = -1;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) != .store) continue;
            if (attrTag(func, func.opcode(inst).store.ptr, "color_out")) |comp| {
                if (@as(i32, comp) > max_comp) max_comp = comp;
            }
        }
    }
    if (max_comp < 0) return 1;
    return @intCast(@min(8, @as(u32, @intCast(max_comp)) / 4 + 1));
}

/// A compiled kernel: the SASS instruction stream and the register count the
/// launch descriptor needs.
pub const Kernel = struct {
    code: []u32,
    reg_count: u32,
    /// Whether a fragment shader writes gl_FragDepth (this routes it to
    /// frag_depth_out_reg). When true, the nvidia pipeline sets the SPH
    /// OMAP_DEPTH bit, so the ROP reads the fragment depth from the shader.
    /// False for vertex shaders and fragment shaders that do not write depth.
    writes_depth: bool = false,
    /// The number of render targets a fragment shader writes (MRT). 1 for a
    /// single-RT shader or a vertex shader. The nvidia pipeline declares this
    /// many color targets in the SPH omap and binds that many color surfaces
    /// to the ROP.
    color_targets: u8 = 1,
    /// What a runtime needs to launch this kernel. `params` is owned by this Kernel.
    launch: gpu.LaunchInfo,

    pub fn deinit(self: *Kernel, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.launch.params);
    }
};

/// Where each IR value lives: a general register, or a predicate register (for
/// booleans produced by a compare).
const Loc = union(enum) { gpr: u8, pred: u8 };

/// A texture-sample result block. The SPIR-V image-sample lowering uses an
/// alloca as the host-sampler out-pointer. This gives that alloca a block of 4
/// consecutive registers (RGBA). The NVIDIA TEX instruction writes its result
/// there. The lowering's 4 reload `load`s resolve to those registers. Keyed by
/// the alloca Value, mapped to its base register.
const TexResult = struct { base: u8 };

// A BRA to patch. `at` is the branch instruction's index. The destination is
// either a block (`target`, resolved through block_start) or a direct
// instruction index (`target_inst`, a local label inside emitIf). Exactly one
// of the two is set. `is_bssy` marks a BSSY convergence-barrier setup: a
// forward branch to the reconvergence block, with the barrier register kept
// from the original encoding.
const Fixup = struct { at: usize, target: u32 = 0, target_inst: ?usize = null, is_bssy: bool = false };

/// Emit Volta-and-later convergence barriers (BSSY/BSYNC) around divergent
/// `if` regions. This lets a quad-dependent op (TEX or a derivative SHFL)
/// after the merge run with the warp reconverged. See computeConvergence and
/// encode.{bclear,bssy,bsync}.
const emit_convergence_barriers = true;

/// Convergence-barrier plan. For each block that ends in a divergent `if`,
/// this records the reconvergence (post-dominator) block where the warp must
/// meet, and which hardware barrier register (B0..B15) to use. On Volta and
/// later, a divergent branch splits the warp. A quad-dependent op (TEX or a
/// derivative SHFL) executed afterward, without reconverging, reads garbage
/// from the lanes that took the other path. NAK wraps every divergent region
/// in BSSY (set a reconvergence point before the branch) and BSYNC (meet at
/// the join). This backend does the same: for each `if` block, it finds the
/// immediate post-dominator (the merge block both arms reach), emits
/// BCLEAR and BSSY before the branch, and emits BSYNC at the start of the
/// merge block.
const Convergence = struct {
    // bar_at_if[bi]: the barrier register if block bi ends in a divergent if (else null).
    bar_at_if: []?u4,
    // merge_of_if[bi]: the reconvergence (ipdom) block of block bi's if (else undefined).
    merge_of_if: []Block,
    // syncs_at[bi]: barrier registers whose BSYNC must be emitted at the start of block bi.
    syncs_at: [][]u4,

    fn deinit(self: *Convergence, allocator: std.mem.Allocator) void {
        allocator.free(self.bar_at_if);
        allocator.free(self.merge_of_if);
        for (self.syncs_at) |s| allocator.free(s);
        allocator.free(self.syncs_at);
    }
};

/// Successors of a block: the `if` then/else targets, or the jump target, or none.
fn blockSuccessors(func: *const Function, bi: usize, buf: *[2]usize) []const usize {
    const block: Block = @enumFromInt(bi);
    for (func.blockInsts(block)) |inst| {
        if (func.opcode(inst) == .@"if") {
            const cf = func.opcode(inst).@"if";
            buf[0] = @intFromEnum(cf.then.target);
            buf[1] = @intFromEnum(cf.@"else".target);
            return buf[0..2];
        }
    }
    switch (func.terminator(block) orelse Terminator{ .ret = ir.function.Ret.none() }) {
        .ret => return buf[0..0],
        .jump => |j| {
            buf[0] = @intFromEnum(j.target);
            return buf[0..1];
        },
    }
}

/// Whether block bi ends in a divergent `if` (a conditional branch whose two
/// arms reach different blocks). A degenerate `if` whose then and else target
/// the same block is not divergent and needs no barrier.
fn divergentIf(func: *const Function, bi: usize) ?ir.function.If {
    const block: Block = @enumFromInt(bi);
    for (func.blockInsts(block)) |inst| {
        if (func.opcode(inst) == .@"if") {
            const cf = func.opcode(inst).@"if";
            if (cf.then.target == cf.@"else".target) return null;
            return cf;
        }
    }
    return null;
}

/// Compute the convergence-barrier plan. This builds the block CFG, computes
/// post-dominators with the standard iterative dataflow (the reverse of the
/// dominator algorithm), and finds each divergent `if`'s immediate
/// post-dominator, which is its reconvergence block. Barrier registers are
/// assigned by region nesting depth, so nested divergent regions use distinct
/// barriers. This matches how NAK's allocator keeps overlapping convergence
/// barriers in distinct Bar registers. Returns a plan with no barriers (all
/// null) when there are no divergent ifs.
fn computeConvergence(allocator: std.mem.Allocator, func: *const Function) Error!Convergence {
    const n = func.blockCount();
    const bar_at_if = try allocator.alloc(?u4, n);
    @memset(bar_at_if, null);
    errdefer allocator.free(bar_at_if);
    const merge_of_if = try allocator.alloc(Block, n);
    @memset(merge_of_if, @enumFromInt(0));
    errdefer allocator.free(merge_of_if);
    const syncs_at = try allocator.alloc([]u4, n);
    @memset(syncs_at, &.{});
    errdefer allocator.free(syncs_at);

    // Any divergent ifs at all?
    var any = false;
    for (0..n) |bi| {
        if (divergentIf(func, bi) != null) {
            any = true;
            break;
        }
    }
    if (!any or !emit_convergence_barriers) return .{ .bar_at_if = bar_at_if, .merge_of_if = merge_of_if, .syncs_at = syncs_at };

    // Post-dominators: pdom[b] is the set of blocks that post-dominate b. Exit
    // blocks (no successors) post-dominate only themselves. Every other
    // block's pdom set is {b} union the intersection, over successors s, of
    // pdom[s]. Iterate to a fixpoint. The block order from the frontend is
    // close to a topological order, so iterating in reverse converges quickly
    // for these small shaders (under about 30 blocks).
    const word_count = (n + 63) / 64;
    const pdom = try allocator.alloc(u64, n * word_count);
    defer allocator.free(pdom);
    const tmp = try allocator.alloc(u64, word_count);
    defer allocator.free(tmp);

    // init: exit blocks -> {self}. Others -> universe (all bits set).
    for (0..n) |b| {
        const row = pdom[b * word_count ..][0..word_count];
        var succ_buf: [2]usize = undefined;
        const succs = blockSuccessors(func, b, &succ_buf);
        if (succs.len == 0) {
            @memset(row, 0);
            row[b / 64] |= @as(u64, 1) << @intCast(b % 64);
        } else {
            @memset(row, ~@as(u64, 0));
        }
    }

    var changed = true;
    var guard: usize = 0;
    while (changed and guard < n + 4) : (guard += 1) {
        changed = false;
        var bi: usize = n;
        while (bi > 0) {
            bi -= 1;
            var succ_buf: [2]usize = undefined;
            const succs = blockSuccessors(func, bi, &succ_buf);
            if (succs.len == 0) continue; // exit block fixed at {self}
            // tmp = intersection of pdom[s] over successors.
            @memset(tmp, ~@as(u64, 0));
            for (succs) |s| {
                const srow = pdom[s * word_count ..][0..word_count];
                for (tmp, srow) |*t, sv| t.* &= sv;
            }
            // add self.
            tmp[bi / 64] |= @as(u64, 1) << @intCast(bi % 64);
            const row = pdom[bi * word_count ..][0..word_count];
            if (!std.mem.eql(u64, row, tmp)) {
                @memcpy(row, tmp);
                changed = true;
            }
        }
    }

    // The immediate post-dominator of an if block is the closest strict
    // post-dominator, the merge block right after the if. Among the strict
    // post-dominators of `bi` (its pdom set minus itself), the ipdom is the one
    // that every other strict pdom also post-dominates. In other words, it is
    // the one nearest to `bi`. The closer a strict pdom `p` is to `bi`, the more
    // blocks post-dominate-chain through it, so its own pdom set is the
    // largest: it includes itself plus every farther merge or exit block it
    // leads toward. So this code picks the strict pdom with the largest
    // pdom-set size. For the diamond `if a else b -> merge -> ...`, the
    // merge's pdom set is {merge} union all later blocks, which is the
    // largest. The final exit's pdom set is {exit}, the smallest. An earlier
    // version picked the smallest set and wrongly chose the function exit.
    // That over-extended every region to the final block and nested EXIT
    // instructions inside live barriers, causing an illegal-instruction-
    // encoding warp fault. Ties cannot occur in a reducible CFG's
    // post-dominator tree.
    var depth: u4 = 0;
    for (0..n) |bi| {
        if (divergentIf(func, bi) == null) continue;
        const row = pdom[bi * word_count ..][0..word_count];
        var best: ?usize = null;
        var best_size: usize = 0;
        for (0..n) |p| {
            if (p == bi) continue;
            if ((row[p / 64] >> @intCast(p % 64)) & 1 == 0) continue; // p not a pdom of bi
            // size of pdom[p].
            const prow = pdom[p * word_count ..][0..word_count];
            var sz: usize = 0;
            for (prow) |w| sz += @popCount(w);
            if (sz > best_size) {
                best_size = sz;
                best = p;
            }
        }
        if (best) |m| {
            // Assign a barrier register. Cycle through B0..B15 by the count of
            // ifs seen. These regions are mostly sequential in the inlined leaf
            // shaders. A distinct barrier for each region is always safe, more
            // so than reuse.
            bar_at_if[bi] = depth;
            depth = (depth + 1) & 0xf;
            merge_of_if[bi] = @enumFromInt(m);
        }
    }

    // Collect, per merge block, the barrier registers whose BSYNC fires there.
    for (0..n) |bi| {
        if (bar_at_if[bi]) |bar| {
            const m = @intFromEnum(merge_of_if[bi]);
            const old = syncs_at[m];
            const grown = try allocator.alloc(u4, old.len + 1);
            @memcpy(grown[0..old.len], old);
            grown[old.len] = bar;
            if (old.len != 0) allocator.free(old);
            syncs_at[m] = grown;
        }
    }

    return .{ .bar_at_if = bar_at_if, .merge_of_if = merge_of_if, .syncs_at = syncs_at };
}

/// Whether `block` holds a `barrier` instruction.
fn blockHasBarrier(func: *const Function, bi: usize) bool {
    for (func.blockInsts(@as(Block, @enumFromInt(bi)))) |inst| {
        if (func.opcode(inst) == .barrier) return true;
    }
    return false;
}

/// Whether `target` is reachable from a SUCCESSOR of block `from`, never entering `avoid` and
/// never leaving `stop`. A path that arrives at `stop` ends there, because `stop` is a region
/// join and everything past it is outside the region. `avoid` and `stop` are optional.
fn reachesFromSuccessors(
    allocator: std.mem.Allocator,
    func: *const Function,
    from: usize,
    target: usize,
    stop: ?usize,
    avoid: ?usize,
) Error!bool {
    const n = func.blockCount();
    const seen = try allocator.alloc(bool, n);
    defer allocator.free(seen);
    @memset(seen, false);

    var stack: std.ArrayList(usize) = .empty;
    defer stack.deinit(allocator);

    var buf: [2]usize = undefined;
    for (blockSuccessors(func, from, &buf)) |s| {
        if (avoid != null and s == avoid.?) continue;
        if (seen[s]) continue;
        seen[s] = true;
        try stack.append(allocator, s);
    }
    while (stack.pop()) |b| {
        if (b == target) return true;
        if (stop != null and b == stop.?) continue;
        var sbuf: [2]usize = undefined;
        for (blockSuccessors(func, b, &sbuf)) |s| {
            if (avoid != null and s == avoid.?) continue;
            if (seen[s]) continue;
            seen[s] = true;
            try stack.append(allocator, s);
        }
    }
    return false;
}

/// Whether the branch at block `ai` leaves a loop that holds block `b`, and leaves it on a
/// UNIFORM condition. Such a branch never splits the workgroup, so a barrier in `b` runs the
/// same number of times in every thread.
///
/// Four things must all hold, and each one closes a way to be wrong:
///   - `ai` and `b` belong to the same natural loop. An irreducible loop is not a natural loop,
///     so it finds none here and keeps the refusal.
///   - The branch is an EXIT of that loop: one edge stays inside, the other leaves. A branch
///     with both edges inside is an ordinary `if` in the body, and the region check owns it.
///   - The exit condition is uniform.
///   - Neither block runs under a divergent branch itself. A uniform condition proves the
///     threads agree at each visit of `ai`, not that every thread visits it.
///
/// The fourth condition is DEFENSE IN DEPTH, not the load-bearing guard. An outer divergent
/// branch that can reach the barrier already fails the region check on its own pass, because a
/// block strictly inside a region that the join cannot be reached around WOULD BE the join.
/// Removing this line therefore breaks no test today. It stays because it makes the exemption
/// stop where its own reasoning stops, instead of resting on another check to catch it.
fn isUniformLoopExit(
    func: *const Function,
    uni: *const opt.uniform.Uniformity,
    info: *const opt.loops.LoopInfo,
    ai: usize,
    b: usize,
) bool {
    if (uni.blockIsSplit(ai) or uni.blockIsSplit(b)) return false;
    const cf = opt.uniform.twoWayIf(func, ai) orelse return false;
    if (uni.isDivergent(cf.cond)) return false;
    const then_i: usize = @intFromEnum(cf.then.target);
    const else_i: usize = @intFromEnum(cf.@"else".target);
    for (info.loops) |*l| {
        if (!l.contains(ai) or !l.contains(b)) continue;
        if (l.contains(then_i) == l.contains(else_i)) continue; // not an exit branch
        return true;
    }
    return false;
}

/// Refuse a `barrier` the warp can split around. Returns `error.Unsupported` for one, and
/// nothing when every barrier in `func` is safely placed.
///
/// The hardware fact, measured on the GB10 (Blackwell): a divergent branch AROUND a BAR.SYNC
/// corrupts a staged shared-memory tile. The BSSY/BSYNC pair this backend emits does NOT save
/// such a barrier. BSYNC sits at the JOIN, which is AFTER the arm's body, so a BAR.SYNC inside
/// an arm still executes with the warp split. A barrier placed AFTER a divergent region is
/// genuinely safe here, and this check accepts that one.
///
/// The rule: for a divergent `if` at block A whose join is M, a barrier in a block B inside that
/// region must lie on EVERY path from A to M. Two shapes fail it:
///   - A barrier in one arm of a diamond. The other arm reaches M without running it.
///   - A barrier in a loop body whose trip count the compiler cannot prove uniform. A thread
///     that leaves such a loop early skips a barrier the other threads still run.
///
/// ONE EXEMPTION, and only one: a loop whose EXIT CONDITION IS UNIFORM. Every thread of the
/// workgroup then makes the same number of trips, so every thread runs the body's barrier the
/// same number of times and none of them is left waiting. `opt.uniform` proves that, and it
/// refuses to prove it wherever it cannot, so an unprovable trip count keeps the refusal. This
/// admits the tiled-matmul shape, which stages a tile into shared memory, waits, computes from
/// it, and waits again, once per tile.
///
/// The exemption is NOT widened to a plain `if` with a uniform condition, although such a branch
/// sends the whole workgroup the same way and is therefore also safe. The loop is the shape with
/// a demonstrated need. Widening the acceptance any further is a separate decision, and every
/// refusal that stands today keeps standing.
///
/// Every remaining refusal is deliberate. The frontend is not trusted to have predicated the
/// guard: a wrong answer that varies with scheduling is much worse than a compile error. A guard
/// around a barrier must be PREDICATED, not branched, and `encode.barSync` takes a full `Control`
/// so a predicated barrier stays expressible once a lowering builds one.
fn checkBarrierConvergence(allocator: std.mem.Allocator, func: *const Function, conv: *const Convergence) Error!void {
    const n = func.blockCount();

    var any = false;
    for (0..n) |bi| {
        if (blockHasBarrier(func, bi)) {
            any = true;
            break;
        }
    }
    if (!any) return;

    // Only a function that holds a barrier pays for these two analyses.
    var uni = try opt.uniform.analyze(allocator, func);
    defer uni.deinit(allocator);
    var loop_info = try opt.loops.analyze(allocator, func);
    defer loop_info.deinit(allocator);

    for (0..n) |ai| {
        if (divergentIf(func, ai) == null) continue;
        // `computeConvergence` records a join only where it found an immediate post-dominator.
        // Without one the two arms never meet again (each one exits), so nothing reconverges
        // the warp and every barrier the branch can reach runs split.
        const merge: ?usize = if (conv.bar_at_if[ai] != null) @intFromEnum(conv.merge_of_if[ai]) else null;

        for (0..n) |b| {
            if (!blockHasBarrier(func, b)) continue;
            // The branch sits at the END of block A, so A's own instructions run with the warp
            // still whole.
            if (b == ai) continue;
            // The one exemption: A is the exit branch of a loop that holds B, and the exit
            // condition is uniform, so the whole workgroup makes the same trips.
            if (isUniformLoopExit(func, &uni, &loop_info, ai, b)) continue;

            const m = merge orelse {
                if (try reachesFromSuccessors(allocator, func, ai, b, null, null)) return error.Unsupported;
                continue;
            };
            // The join is where the BSYNC reconverges the warp, so a barrier there is safe.
            if (b == m) continue;
            // Outside this region: some other region, or after it. Not this check's business.
            if (!try reachesFromSuccessors(allocator, func, ai, b, m, null)) continue;
            // Inside. If the join is still reachable with B cut out, some thread reaches the
            // join without running the barrier.
            if (try reachesFromSuccessors(allocator, func, ai, m, m, b)) return error.Unsupported;
        }
    }
}

/// How many hardware control barriers `func` needs in its launch descriptor. The NVIDIA QMD has
/// a BARRIER_COUNT field, and a dispatch that leaves it at 0 while the kernel runs a BAR.SYNC is
/// UNDEFINED. NAK sets `info.num_control_barriers = 1` beside its `OpBar`, and this matches: one
/// barrier register serves every BAR.SYNC in the kernel, so the count is 1 or 0.
fn barrierCount(func: *const Function) u32 {
    for (0..func.blockCount()) |bi| {
        if (blockHasBarrier(func, bi)) return 1;
    }
    return 0;
}

/// The shader stage being compiled. Compute kernels source parameters from the
/// constant bank and store via STG. Graphics shaders use the attribute interface
/// (vertex inputs via ALD, fragment inputs via IPA, outputs via AST).
pub const Stage = enum { compute, vertex, fragment };

/// Code-generation choices this backend leaves to the caller. Every field has the default
/// this target ships with, so `.{}` is the shipped compiler.
pub const Options = struct {
    /// Contract `a * b + c` into ONE instruction: FFMA for floats, IMAD for 32-bit
    /// integers. See `FmaFold` for the conditions.
    ///
    /// THIS CHANGES FLOAT RESULTS. `a * b + c` as one FFMA rounds once, and as an FMUL
    /// followed by an FADD it rounds twice. The fused answer is the more accurate of the
    /// two, but it is a DIFFERENT answer, so a program that depends on the rounded
    /// intermediate changes behaviour.
    ///
    /// IT IS ON BY DEFAULT because that is what the platform does. nvcc contracts unless
    /// it is told `-fmad=false`, so a kernel built here matches what the same source
    /// gives under NVIDIA's own compiler. A caller that needs the two-rounding answer,
    /// such as a differential test against a scalar CPU oracle that does not fuse, sets
    /// this to false.
    ///
    /// Integer contraction is exact either way: `a * b + c` in 32-bit wrapping
    /// arithmetic gives the same bits fused or not. The flag still covers it, so that one
    /// switch turns off the whole transform when a codegen question is being isolated.
    contract_fma: bool = true,
};

/// Lower `func` to a SASS compute kernel under the parameter ABI `a`. The caller owns the result.
pub fn compileKernel(allocator: std.mem.Allocator, func: *Function, a: gpu.Abi) Error!Kernel {
    return compileShaderOpts(allocator, func, .compute, a, .{});
}

/// `compileKernel` with explicit code-generation options. See `Options`.
pub fn compileKernelOpts(allocator: std.mem.Allocator, func: *Function, a: gpu.Abi, options: Options) Error!Kernel {
    return compileShaderOpts(allocator, func, .compute, a, options);
}

/// Lower `func` to a SASS shader for `stage` under the parameter ABI `a`. The caller owns the result.
pub fn compileShader(allocator: std.mem.Allocator, func: *Function, stage: Stage, a: gpu.Abi) Error!Kernel {
    return compileShaderOpts(allocator, func, stage, a, .{});
}

/// `compileShader` with explicit code-generation options. See `Options`.
pub fn compileShaderOpts(allocator: std.mem.Allocator, func: *Function, stage: Stage, a: gpu.Abi, options: Options) Error!Kernel {
    // This backend does not lower f16 yet. Reject it cleanly instead of
    // silently treating it as f64. This check covers both this direct entry
    // and compileKernel, which calls this function.
    if (ir.function.functionUsesF16(func)) return error.Unsupported;
    if (ir.function.functionUsesF128(func)) return error.Unsupported;

    const nblocks = func.blockCount();
    if (nblocks == 0) return error.Unsupported;
    // The compute prologue walks the entry parameters positionally against the placed layout,
    // and `layoutParams` reads the same list to decide whether the launch-shape region is
    // present. `mem2reg` adds and removes block parameters on any block that has a
    // predecessor, so an entry a branch reaches has no stable list for the two to share.
    if (stage == .compute and gpu.abi.entryIsBranchTarget(func)) return error.Unsupported;

    // The workgroup shared frame: one byte offset in the CTA's shared window per shared
    // `alloca`. `gpu.abi.layoutSharedFrame` OWNS the placement and cross-checks it against the
    // kernel's declared `vulcan.gpu.shared_bytes`, so a disagreement is an error here and not a
    // window the runtime sizes wrongly at dispatch. See that function for the rule.
    var shared = try gpu.abi.layoutSharedFrame(allocator, func, a);
    defer shared.deinit(allocator);
    // A graphics stage has no workgroup, so it has no workgroup shared memory. This mirrors the
    // refusal of a shared POINTER parameter further down.
    if (stage != .compute and shared.slots.len != 0) return error.Unsupported;

    // Fold constant arith operands into immediates before register allocation.
    // This stops each constant from pinning a GPR for its whole live range.
    // Heavy shaders (the noise and terrain shaders) need this to fit the
    // 251-GPR pool instead of exhausting it.
    foldConstantsToImm(func);

    // Move each constant byte displacement into the load or store that reads it, so the
    // access carries the offset in its own field instead of an IADD3 chain and a register.
    // This runs before `assignLocs`, because it REWRITES each access to name the base
    // pointer and the liveness scan must see that. See `foldAddressDisplacements`.
    var disp = DispFold{};
    defer disp.deinit(allocator);
    try foldAddressDisplacements(allocator, func, &disp);

    // Decide which `a * b + c` pairs become ONE fused instruction. This runs before
    // `assignLocs`, because the allocator has to keep the multiply's operands live as far
    // as the add that now reads them. See `FmaFold`.
    var fma = FmaFold{};
    defer fma.deinit(allocator);
    try scanFma(allocator, func, options, &fma);

    var loc = std.AutoHashMapUnmanaged(Value, Loc){};
    defer loc.deinit(allocator);
    var max_reg: u8 = r_outptr + 1; // the output pointer pair is always live
    try assignLocs(allocator, func, &loc, &max_reg, &fma);

    // Texture-sample lowering. The SPIR-V image-sample op becomes a
    // host-sampler `call_indirect(sampler_fn, {desc, u, v, lod, out_ptr})`
    // that writes an RGBA vec4 into a stack alloca, followed by four reload
    // `load`s of out_ptr+c*4. The GPU has no host stack, so each such alloca
    // instead gets a block of 4 consecutive registers (the TEX result RGBA).
    // The sampler call becomes a TEX into that block, and the reload loads
    // resolve to those registers. This builds the alloca-to-base-register map
    // by allocating one fresh 4-register block, above the watermark, per
    // sampler call. `tex` also records the loads (and the element-pointer
    // arith) that target each tex alloca, so lowerInst maps them to the
    // result registers instead of emitting LDG.
    var tex = TexLowering.init(allocator);
    defer tex.deinit();
    try tex.scan(func, &max_reg, stage);

    // Screen-space-derivative lowering. A varying's dFdx/dFdy was lowered
    // (shared with the software path) to a `grad_buf[index]` load. The GPU has
    // no host gradient buffer. Each such load instead becomes an IPA of the
    // varying, plus a quad SHFL, plus an FSWZADD that differences the quad
    // neighbour (the native 2x2-quad derivative). `deriv` records the
    // grad_buf param (to skip in the prologue), the grad-pointer address
    // arith (a tag carrier), and each grad load's slot, axis, and scratch
    // registers.
    var deriv = DerivLowering.init(allocator);
    defer deriv.deinit();
    try deriv.scan(func, &max_reg);

    // Host-math lowering. A transcendental function (pow, exp, log, sin, or
    // cos) was lowered (shared with the software path) to a
    // `math_fn(op, a, b)` call_indirect through a synthesized function
    // pointer. On the GPU, the special-function unit (MUFU) evaluates these
    // natively. `math` records each call's op code and a scratch register, so
    // lowerInst emits the MUFU sequence and the prologue skips the math_fn
    // param.
    var math = MathLowering.init(allocator);
    defer math.deinit();
    try math.scan(func, &max_reg);

    var code: std.ArrayList(Inst) = .empty;
    defer code.deinit(allocator);
    var fixups: std.ArrayList(Fixup) = .empty;
    defer fixups.deinit(allocator);
    var block_start = try allocator.alloc(usize, nblocks);
    defer allocator.free(block_start);

    const eparams = func.blockParams(@enumFromInt(0));
    // The graphics slice is allocated rather than a `&.{}` literal so `Kernel.deinit` can free
    // it unconditionally, with no special case for a zero-length non-heap slice. A graphics
    // shader sources its inputs from the attribute interface and not from a parameter block,
    // so it never goes through `layoutParams`.
    var layout: gpu.kernel.Layout = if (stage == .compute)
        try gpu.layoutParams(allocator, func, a, returnsValue(func))
    else
        .{ .params = try allocator.alloc(gpu.Param, 0), .bytes = 0, .out_pointer = null, .launch_shape = null };
    errdefer layout.deinit(allocator);

    if (stage == .compute) {
        // A kernel that returns a value reads an output pointer from the front
        // of the constant bank (its `ret` stores the result there). A void
        // compute kernel has no output pointer.
        if (layout.out_pointer) |out| {
            const at: u16 = @intCast(a.param_base + out.offset);
            try emitPointerLdc(allocator, &code, r_outptr, bank0, at);
        }
        var placed: usize = 0;
        for (eparams) |p| {
            if (gpu.attrs.builtinOf(func, p)) |bi| {
                try emitComputeBuiltin(allocator, &code, func, loc, p, bi, a, layout.launch_shape);
                continue;
            }
            const slot = layout.params[placed];
            placed += 1;
            const at: u16 = @intCast(a.param_base + slot.offset);
            const lo = gprOf(loc, p);
            // A 64-bit address occupies a register PAIR, so `emitPointerLdc` fills lo and
            // lo + 1, with one LDC.64 where the offset allows it.
            //
            // A SHARED address does not. It is a 32-bit byte offset into the CTA's
            // shared-memory window, so one 32-bit LDC is the whole load. The parameter
            // block still reserves `pointer_bytes` for it, because `layoutParams` places every
            // pointer at the target's address width, so the runtime writes the offset into the
            // low dword of that slot and leaves the high dword alone.
            switch (slot.kind) {
                .scalar => try code.append(allocator, encode.ldc(lo, bank0, at, .{})),
                .pointer => |space| switch (space) {
                    .global, .constant, .private => try emitPointerLdc(allocator, &code, lo, bank0, at),
                    .shared => try code.append(allocator, encode.ldc(lo, bank0, at, .{})),
                },
            }
        }
    } else {
        // The SMs deliver the hardware-provided inputs (vertex ID, fragment
        // barycentrics, and sysvals) into the low registers asynchronously, a
        // few instructions into warp execution. An attribute fetch or color
        // write issued before that window closes reads zeros, or gets
        // clobbered by them. To avoid this, pad the prologue with throwaway
        // MOVs to a high scratch register before any ALD or IPA. Testing
        // found a clean threshold of 4. This uses 6 for margin.
        var pad: u32 = 0;
        while (pad < graphics_prologue_pad) : (pad += 1) {
            try code.append(allocator, encode.movImm(graphics_pad_reg, pad, .{}));
        }
        // Each parameter loads by kind, in declaration order:
        //   - A buffer or UBO pointer (a `ptr`, such as a uniform-block base)
        //     loads as a 64-bit address pair from constant bank 0. The
        //     dispatch side binds the bound UBO's GPU virtual address into
        //     CB0 at `graphics_ubo_cb_base` + slot*8, in the same
        //     buffer-declaration order the lowering appended the pointer
        //     params. A following `.load` (LDG) then reads the std-layout
        //     members through that pointer pair, reusing the compute load
        //     path.
        //   - An input attribute scalar: a vertex shader fetches it (ALD), a
        //     fragment shader interpolates it (IPA), at the parameter's
        //     `attr` slot.
        // Pointer and attribute loads are variable-latency. The scoreboard
        // pass adds the consumer waits.
        var ubo_slot: u16 = 0;
        // Map each fragment-input attribute byte-slot to the register the
        // prologue IPA'd it into. The screen-space-derivative lowering reuses
        // these registers, instead of a re-IPA in the body, so the quad SHFL
        // reads a value that has long since landed in every lane. A freshly
        // re-IPA'd value is variable-latency, and a cross-lane SHFL cannot
        // wait on the neighbour lane's scoreboard, so shuffling it would read
        // stale garbage. NAK shuffles the existing, single prologue IPA SSA
        // value for exactly this reason. (This records the base attr slot.
        // Per-component grad slots index from it.)
        for (eparams) |p| {
            const rd = gprOf(loc, p);
            // gl_VertexIndex / gl_InstanceIndex: a synthesized i32 builtin
            // param the frontend tagged. On Volta and later, a vertex shader
            // reads it from the attribute interface (ALD
            // a[NAK_ATTR_VERTEX_ID/INSTANCE_ID]), not a special register. The
            // fixed-function Data Assembler writes the per-vertex ID into the
            // attribute RAM. This is what NAK emits for SystemValue VertexId.
            // With SET_VERTEX_ID_BASE = 0 (a non-indexed draw), the delivered
            // value equals Vulkan's gl_VertexIndex. The shader then
            // multiplies it by the array stride, adds it to the UBO base
            // pointer (the dynamic-index OpAccessChain the frontend
            // lowered), and does an LDG load. This pulls its vertices from a
            // UBO array with no vertex buffer. ALD is variable-latency: the
            // scheduler drains it before its use. The pipeline's SPH must
            // also declare the vertex-ID sysval input.
            if (gpu.attrs.builtinOf(func, p)) |bi| {
                switch (bi) {
                    // gl_FragCoord: the window-space fragment
                    // position. Each component is IPA'd (freq Pass) from the
                    // POSITION attribute a[0x70+c*4] (NAK_ATTR_POSITION),
                    // tagged `bicomp` = component. The SPH declares the
                    // position input as SCREEN_LINEAR (readsFragPosition), so
                    // the raster delivers x and y in pixels, z as the
                    // interpolated depth, and w as 1/clip_w.
                    .frag_coord => {
                        const comp: u16 = attrTag(func, p, "bicomp") orelse 0;
                        try code.append(allocator, encode.ipa(rd, encode.ATTR_POSITION + comp * 4, .{}));
                        continue;
                    },
                    // gl_FrontFacing: the raster delivers a flat
                    // per-primitive facing flag at a[0x3fc]
                    // (NAK_ATTR_FRONT_FACE) as an integer mask: all-ones for
                    // a front face, zero for back. The frontend types this as
                    // an f32 param and compares it `!= 0` with a float
                    // set-predicate (FSETP). But the integer all-ones bit
                    // pattern, reinterpreted as f32, is a NaN, and
                    // FSETP.NE(NaN, 0) is false, so a front face would
                    // wrongly read as back. To fix this, convert the
                    // delivered integer to a clean ordered float with I2F
                    // right after the flat IPA: any nonzero mask becomes a
                    // nonzero float (front), and zero stays 0.0 (back), so
                    // the downstream FSETP behaves correctly. The raster
                    // always delivers a[0x3fc], so no extra SPH imap entry is
                    // needed.
                    .front_facing => {
                        try code.append(allocator, encode.ipaConstant(rd, encode.ATTR_FRONT_FACE, .{}));
                        try code.append(allocator, encode.i2f(rd, rd, true, .{}));
                        continue;
                    },
                    // gl_PointCoord: a point sprite's s/t
                    // coordinate, running 0..1 across the sprite quad. Each
                    // component is a normal IPA from the point-sprite
                    // attribute a[0x2e0]+comp*4 (NAK_ATTR_POINT_SPRITE_S/T).
                    // The SPH imap declares these two inputs as
                    // SCREEN_LINEAR (readsPointSprite), so the raster
                    // delivers the perspective-free sprite-local coordinate,
                    // and the draw state enables SET_POINT_SPRITE (done once
                    // at channel init).
                    .point_coord => {
                        const comp: u16 = attrTag(func, p, "bicomp") orelse 0;
                        try code.append(allocator, encode.ipa(rd, encode.ATTR_POINT_SPRITE + comp * 4, .{}));
                        continue;
                    },
                    // gl_VertexIndex / gl_InstanceIndex: a vertex shader reads
                    // them from the DA-delivered attribute interface (ALD),
                    // not IPA.
                    .vertex_index, .instance_index => {
                        const attr: u16 = if (bi == .instance_index) encode.ATTR_INSTANCE_ID else encode.ATTR_VERTEX_ID;
                        try code.append(allocator, encode.ald(rd, attr, 1, .{}));
                        continue;
                    },
                    // A compute builtin has no graphics delivery path. The
                    // hardware gives it to a kernel, not to the attribute
                    // interface, so a graphics shader that asks for one is a
                    // frontend error and not a shape this backend can emit.
                    .thread_id_x,
                    .thread_id_y,
                    .thread_id_z,
                    .block_id_x,
                    .block_id_y,
                    .block_id_z,
                    .block_dim_x,
                    .block_dim_y,
                    .block_dim_z,
                    .grid_dim_x,
                    .grid_dim_y,
                    .grid_dim_z,
                    .global_id_x,
                    .global_id_y,
                    .global_id_z,
                    .lane_id,
                    .warp_id,
                    .subgroup_size,
                    => return error.Unsupported,
                }
            }
            // The host-sampler function pointer the SPIR-V image-sample
            // lowering appends is meaningless on the GPU, since TEX needs no
            // host function. It gets no constant-bank slot, and the sampler
            // `call_indirect` through it lowers to a TEX instead.
            if (isSamplerFn(func, p) or isSamplerVec3Fn(func, p) or isAnyShadowFn(func, p) or isSamplerGatherFn(func, p) or isSamplerFetchFn(func, p) or isSamplerFetch3Fn(func, p)) continue;
            // The synthesized grad_buf pointer (the software path's
            // per-triangle gradient buffer) has no GPU backing. The
            // derivative lowering computes dFdx/dFdy from the live quad
            // through SHFL instead. Source nothing for it: no constant-bank
            // slot, since it is not a bound UBO, and no load.
            if (hasGpuKey(func, p, "grad_buf")) continue;
            // The host-math function pointer the transcendental lowering
            // appends (pow, exp, log, sin, or cos) is meaningless on the
            // GPU. The special-function unit (MUFU) evaluates these
            // natively, so the param gets no constant-bank slot, and the
            // math `call_indirect` through it lowers to MUFU (see the
            // call_indirect arm).
            if (hasGpuKey(func, p, "math_fn")) continue;
            // The discard function pointer (OpKill) is meaningless on the
            // GPU. The discard call lowers to a KIL, so the param gets no
            // constant-bank slot.
            if (hasGpuKey(func, p, "discard_fn")) continue;
            // A combined-image-sampler descriptor param: its constant-bank
            // slot holds the 32-bit bindless texture handle (tic | tsc<<20)
            // the dispatch binds, not a memory address. Load just the low
            // dword (a single LDC) into the value's register. The sampler
            // `call_indirect` feeds it to TEX as the handle. It consumes a
            // UBO or descriptor constant-bank slot in declaration order,
            // exactly like a UBO pointer, so the dispatch writes the handle
            // at the same offset.
            if (isSamplerDesc(func, p)) {
                // Place the descriptor at its Vulkan binding slot, not a
                // per-stage declaration-order slot. The constant bank is
                // shared across the vertex and fragment shaders, and the
                // dispatch side writes each descriptor's handle or address
                // at graphics_ubo_cb_base + binding*8. Using declaration
                // order would collide: for example, a fragment shader
                // sampler at binding 1, whose only-in-stage param is "slot
                // 0", would read the vertex shader UBO's pointer at slot 0.
                // Falls back to ubo_slot when a shader carries no binding
                // decoration, as in the hand-built isel tests.
                const slot = attrTag(func, p, "binding") orelse ubo_slot;
                const off = encode.graphics_ubo_cb_base + slot * 8;
                try code.append(allocator, encode.ldc(rd, encode.graphics_const_bank, off, .{})); // the bindless handle (root table 1)
                ubo_slot += 1;
                continue;
            }
            // A graphics stage has no workgroup, so it has no workgroup shared memory. Refuse
            // a shared pointer here rather than let it reach the attribute path below, which
            // would silently interpolate a varying into the address register.
            if (isSharedPtr(func, p)) return error.Unsupported;
            if (isWidePtr(func, p)) {
                const slot = attrTag(func, p, "binding") orelse ubo_slot;
                const off = encode.graphics_ubo_cb_base + slot * 8;
                try emitPointerLdc(allocator, &code, rd, encode.graphics_const_bank, off); // the 64-bit address (root table 1)
                ubo_slot += 1;
                continue;
            }
            const attr = attrTag(func, p, "attr") orelse encode.ATTR_GENERIC0;
            try code.append(allocator, if (stage == .vertex)
                encode.ald(rd, attr, 1, .{})
            else
                encode.ipa(rd, attr, .{}));
            // Remember which register holds this prologue-interpolated
            // varying scalar, so the derivative lowering can SHFL it
            // directly instead of re-IPA in the body. Each fragment input
            // scalar is its own param, IPA'd at its `attr`, so this adds one
            // (attr, rd) entry.
            if (stage == .fragment)
                try deriv.prologue_reg.put(allocator, attr, rd);
        }
    }

    // Convergence-barrier plan (Volta and later): wrap each divergent `if`
    // region in a BSSY/BSYNC pair, so a TEX or derivative SHFL after the
    // merge runs with the warp reconverged and quad uniformity restored.
    // Without this, divergent branches combined with texture or derivative
    // ops produce per-pixel noise, because the lanes that took the other arm
    // are inactive for the quad op. See computeConvergence and
    // encode.{bssy,bsync}.
    var conv = try computeConvergence(allocator, func);
    defer conv.deinit(allocator);

    // Refuse a barrier the warp can split around, before a single instruction is emitted. See
    // `checkBarrierConvergence` for the hardware fact behind this.
    try checkBarrierConvergence(allocator, func, &conv);

    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        // Reconverge: emit a BSYNC for every divergent region whose join is
        // this block. block_start[bi] points at the BSYNC, so branches into
        // the merge block (the arm BRAs, the BSSY) land on it. The warp then
        // meets on arrival, restoring quad uniformity before this block's
        // code, which may contain a TEX or a derivative op.
        block_start[bi] = code.items.len;
        for (conv.syncs_at[bi]) |bar| {
            try code.append(allocator, encode.bsync(bar, .{ .stall = 1 }));
        }
        var terminated = false;

        for (func.blockInsts(block)) |inst| {
            try lowerInst(allocator, func, &loc, &code, &tex, &deriv, &math, &shared, &disp, &fma, inst);
            if (func.opcode(inst) == .@"if") {
                // Set up the convergence barrier just before the divergent
                // branch. BCLEAR initializes the barrier register, and BSSY
                // records the reconvergence point, the merge block. The
                // fixup pass patches the BSSY's forward offset.
                if (conv.bar_at_if[bi]) |bar| {
                    try code.append(allocator, encode.bclear(bar, .{ .stall = 1 }));
                    const at = code.items.len;
                    try code.append(allocator, encode.bssy(bar, 0, .{ .stall = 1 }));
                    try fixups.append(allocator, .{ .at = at, .target = @intFromEnum(conv.merge_of_if[bi]), .is_bssy = true });
                }
                try emitIf(allocator, func, &loc, &code, &fixups, func.opcode(inst).@"if");
                terminated = true;
            }
        }

        if (!terminated) switch (func.terminator(block) orelse ir.function.Terminator{ .ret = ir.function.Ret.none() }) {
            .ret => |r| {
                switch (r.count) {
                    0 => {},
                    1 => {
                        // The width stays 32 bits on purpose. The out-pointer slot is laid out
                        // by the kernel ABI, not by the isel, so narrowing this store to the
                        // value's own width would leave the rest of the slot holding whatever
                        // the host buffer held before. What the width check DOES buy is the
                        // refusal of a return value wider than one register: `memTypeOf`
                        // rejects a 64-bit scalar, which this store would otherwise truncate
                        // to its low half with no diagnostic.
                        _ = try memTypeOf(func, r.values[0]);
                        const src = gprOf(loc, r.values[0]);
                        try code.append(allocator, encode.stgU32(r_outptr, src, .{}));
                    },
                    else => return error.Unsupported, // multi-value struct return is not yet lowered
                }
                try code.append(allocator, encode.exit(.{ .stall = 1 }));
            },
            .jump => |j| try emitJump(allocator, func, &loc, &code, &fixups, j),
        };
    }

    // Scoreboard scheduling: add write barriers on variable-latency ops (LDG,
    // S2R) and waits on their consumers, so code reads results only once
    // they are ready. The block-start indices let the scheduler drain
    // scoreboards at each basic-block boundary, so its linear walk stays
    // correct across the control flow that inlining introduces.
    schedule.scheduleBlocks(code.items, block_start);

    // Patch each control-flow op's relative displacement. The destination is
    // a block start (resolved through block_start) or a direct instruction
    // index (a local label from emitIf). The offset is computed exactly like
    // NAK's get_rel_offset (sm70_encode.rs): `target_ip - cur_ip - 4`, where
    // `ip` counts in 32-bit words and one 128-bit instruction equals 4
    // words. So the encoded value is in word units:
    // `(dst_inst - cur_inst)*4 - 4 = (dst_inst - next_inst)*4`. An earlier
    // version used a byte-unit `*16` convention, which was 4 times too
    // large. That made every predicated branch land on the wrong
    // instruction, or run the warp off the end.
    for (fixups.items) |f| {
        const cur: i64 = @intCast(f.at);
        const dst_inst: usize = f.target_inst orelse block_start[f.target];
        const off_words: i32 = @intCast((@as(i64, @intCast(dst_inst)) - cur) * 4 - 4);
        if (f.is_bssy) {
            // BSSY uses the same get_rel_offset base: word units, `dst - cur - 1` instructions.
            const bar: u4 = @intCast(code.items[f.at][0] >> 16 & 0xf);
            code.items[f.at] = encode.bssy(bar, off_words, .{ .stall = 1 });
            continue;
        }
        // The branch's taken condition lives at bits 87..89, plus negate at
        // bit 90, that is, word 2 bits 23..25, plus bit 26. This is not the
        // 12..14 guard, which is PT.
        const pred = (code.items[f.at][2] >> 23) & 0x7;
        const neg = ((code.items[f.at][2] >> 26) & 1) == 1;
        code.items[f.at] = encode.bra(off_words, .{ .pred = @intCast(pred), .pred_neg = neg });
    }

    // Flatten to dwords.
    const out = try allocator.alloc(u32, code.items.len * 4);
    errdefer allocator.free(out);
    for (code.items, 0..) |w, i| @memcpy(out[i * 4 ..][0..4], &w);
    // The parameter slice moves from `layout` to the Kernel here, so the Kernel's `deinit`
    // releases it from this point on and the `errdefer layout.deinit` above must not fire.
    const reg_count = regCount(max_reg);
    return .{
        .code = out,
        .reg_count = reg_count,
        .writes_depth = writesFragDepth(func),
        .color_targets = colorTargetCount(func),
        .launch = .{
            .params = layout.params,
            .param_bytes = layout.bytes,
            .launch_shape = layout.launch_shape,
            .block = gpu.attrs.localSize(func),
            .shared_bytes = shared.total,
            .reg_count = reg_count,
            .barrier_count = barrierCount(func),
        },
    };
}

/// Registers per thread for the launch descriptor.
///
/// The hardware RESERVES the top two GPRs of each thread's allocation. A write to one of them
/// is dropped and a read gives zero, with no fault, so a kernel that uses register `max_reg`
/// needs an allocation of `max_reg + 1 + 2`. Rounding to the granularity does not always
/// absorb the two: with `max_reg = 14` the used count is 15, which rounds to 16, and the
/// kernel then silently loses R14. This was measured on Blackwell (GB10) silicon.
fn regCount(max_reg: u8) u32 {
    const used = @as(u32, max_reg) + 1 + hw_reserved_regs;
    return @max(16, (used + 7) & ~@as(u32, 7)); // hardware granularity: multiples of 8, min 16
}

/// The top GPRs of each thread's allocation that the hardware keeps for itself. See `regCount`.
const hw_reserved_regs: u32 = 2;

fn gprOf(loc: std.AutoHashMapUnmanaged(Value, Loc), v: Value) u8 {
    return switch (loc.get(v).?) {
        .gpr => |r| r,
        .pred => unreachable, // a predicate used where a GPR was expected
    };
}

fn predOf(loc: std.AutoHashMapUnmanaged(Value, Loc), v: Value) u8 {
    return switch (loc.get(v).?) {
        .pred => |p| p,
        .gpr => unreachable,
    };
}

const carry_pred: u8 = 6; // predicate reserved for the 64-bit-add carry chain
const Interval = struct { value: Value, start: u32, end: u32 };

fn lessByStart(_: void, a: Interval, b: Interval) bool {
    return a.start < b.start;
}

/// Linear-scan register allocation with reuse. A register frees when its
/// value's last use passes, so short-lived values (for example, the 32
/// compares of a lowered integer division) share a small set of registers
/// instead of each taking a fresh one. Pointers take even-aligned GPR pairs.
/// Booleans take predicates P0..P5 (P6 is the 64-bit-add carry scratch).
/// There is no spilling: a class running out returns `error.Unsupported`,
/// which a real kernel should never hit, since it has 250 or more GPRs.
fn assignLocs(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), max_reg: *u8, fma: *const FmaFold) Error!void {
    const nval = func.valueCount();
    if (nval == 0) return;
    const nblocks = func.blockCount();

    // Live intervals (def to last use) over a block-order linearization,
    // extended by backward liveness so loop-carried values stay live across
    // the loop body.
    const def_pos = try allocator.alloc(u32, nval);
    defer allocator.free(def_pos);
    const last_use = try allocator.alloc(u32, nval);
    defer allocator.free(last_use);
    const block_end = try allocator.alloc(u32, nblocks);
    defer allocator.free(block_end);
    @memset(def_pos, 0);
    for (last_use) |*l| l.* = 0;

    var pos: u32 = 0;
    // The position of the last fragment color-output store, and the set of
    // values that feed a color-output store (see the extension below).
    // `last_color_pos == 0` means there were no color stores.
    var last_color_pos: u32 = 0;
    const feeds_color = try allocator.alloc(bool, nval);
    defer allocator.free(feeds_color);
    @memset(feeds_color, false);
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| {
            def_pos[@intFromEnum(p)] = pos;
            last_use[@intFromEnum(p)] = pos;
        }
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            forEachUse(func, inst, last_use, pos);
            if (func.instResult(inst)) |r| def_pos[@intFromEnum(r)] = pos;
            // A contracted multiply-add reads the MULTIPLY'S OPERANDS here, at the add,
            // because the multiply itself emits nothing. `forEachUse` cannot see that: it
            // reads the IR, where those operands are read by the multiply and nowhere
            // else. Left unextended, a value defined between the two takes an operand's
            // register and the fused instruction multiplies the wrong number, silently.
            // See `FmaFold`.
            if (func.instResult(inst)) |r| {
                if (fma.at.get(r)) |m| {
                    markUse(last_use, m.mul_a, pos);
                    if (m.mul_b) |b| markUse(last_use, b, pos);
                }
            }
            // A fragment color-output store: its value lands in a ROP color
            // register (R0..R3), and the ROP reads all of them together at
            // EXIT. So every color value must stay live until the last color
            // store, not just its own. Otherwise the allocator frees a color
            // value's register after its own, possibly early, store, and
            // reuses it for a later color value's computation. This
            // clobbers the first color in its register before the final
            // color move reads it, causing corruption when dFdx combines
            // with multi-component output. Record which values feed a color
            // store, and the position of the last such store.
            if (func.opcode(inst) == .store) {
                const st = func.opcode(inst).store;
                if (attrTag(func, st.ptr, "color_out") != null) {
                    feeds_color[@intFromEnum(st.value)] = true;
                    last_color_pos = pos;
                }
            }
            pos += 1;
        }
        block_end[bi] = pos;
        if (func.terminator(block)) |term| forEachTermUse(func, term, last_use, pos);
        pos += 1;
    }
    // Extend every color-output value's live range to the last color store,
    // so the four color components occupy four distinct registers that all
    // stay live to EXIT.
    if (last_color_pos != 0) {
        for (0..nval) |v| {
            if (feeds_color[v] and last_use[v] < last_color_pos) last_use[v] = last_color_pos;
        }
    }
    // Screen-space derivatives: the deriv lowering SHFLs the prologue-IPA'd
    // varying register, sourced by register, not as a tracked SSA use. So
    // the linear-scan allocator does not see that use, and would free and
    // reuse the varying register for a later value, such as the shader's
    // `*16` immediate, before the SHFL reads it. The SHFL would then shuffle
    // garbage. This mirrors the color-output fix: find the last grad_buf
    // load, any `.load` whose pointer is the grad_buf param or
    // `add(grad_buf, k)`, and extend every fragment input-attribute entry
    // param's live range to it. This keeps the IPA'd varying registers the
    // SHFL sources live until the last derivative.
    var grad_buf_param: ?Value = null;
    for (func.blockParams(@enumFromInt(0))) |p| {
        if (hasGpuKey(func, p, "grad_buf")) {
            grad_buf_param = p;
            break;
        }
    }
    if (grad_buf_param) |gbp| {
        // Pointers that address the grad buffer: the param itself, or
        // add(param, iconst).
        const is_grad_ptr = try allocator.alloc(bool, nval);
        defer allocator.free(is_grad_ptr);
        @memset(is_grad_ptr, false);
        is_grad_ptr[@intFromEnum(gbp)] = true;
        for (0..nblocks) |bi| {
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (func.opcode(inst) != .arith) continue;
                const a = func.opcode(inst).arith;
                if (a.op != .add or a.lhs != gbp) continue;
                if (func.instResult(inst)) |r| is_grad_ptr[@intFromEnum(r)] = true;
            }
        }
        // The position of the last grad_buf load (re-walk in the same linearization).
        var last_grad_pos: u32 = 0;
        var p2: u32 = 0;
        for (0..nblocks) |bi| {
            p2 += 1; // block-param slot (matches the first walk)
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (func.opcode(inst) == .load) {
                    const l = func.opcode(inst).load;
                    if (is_grad_ptr[@intFromEnum(l.ptr)]) last_grad_pos = p2;
                }
                p2 += 1;
            }
            p2 += 1; // terminator slot
        }
        if (last_grad_pos != 0) {
            for (func.blockParams(@enumFromInt(0))) |p| {
                // A fragment input-attribute varying param, IPA'd in the
                // prologue into the register the SHFL sources. Identified by
                // the `attr` tag the frontend set.
                if (attrTag(func, p, "attr") != null) {
                    const idx = @intFromEnum(p);
                    if (last_use[idx] < last_grad_pos) last_use[idx] = last_grad_pos;
                }
            }
        }
    }
    try extendLiveRanges(allocator, func, last_use, block_end);

    var ivals = try allocator.alloc(Interval, nval);
    defer allocator.free(ivals);
    for (0..nval) |i| ivals[i] = .{ .value = @enumFromInt(i), .start = def_pos[i], .end = last_use[i] };
    std.mem.sort(Interval, ivals, {}, lessByStart);

    // Free pools: GPRs R4..R254 (R0 and R1 are scratch, R2:R3 is the output
    // pointer), and predicates P0..P5.
    var gpr_free = [_]bool{false} ** 256;
    for (value_reg_base..encode.RZ) |r| gpr_free[r] = true;
    gpr_free[graphics_pad_reg] = false; // reserved as the graphics prologue pad scratch
    // gl_FragDepth: reserve the ROP depth-output register so no live value
    // takes it. The frag_depth store moves the depth into it, and it must
    // stay untouched until EXIT. The register sits past all N color targets
    // (fragDepthReg), so MRT and depth do not collide.
    if (writesFragDepth(func)) gpr_free[fragDepthReg(func)] = false;
    // MRT: for N > 1 render targets, reserve R4..R[4N-1] (RT0 uses the
    // always-reserved R0..R3). Each color store moves its component into
    // R[target*4+comp], which the ROP reads at EXIT, so those registers must
    // stay free of other live values. N == 1 is unchanged.
    {
        const nt = colorTargetCount(func);
        if (nt > 1) {
            var r: usize = value_reg_base;
            while (r < @as(usize, nt) * 4) : (r += 1) gpr_free[r] = false;
        }
    }
    var pred_free = [_]bool{true} ** carry_pred;

    const Active = struct { end: u32, loc: Loc, is_ptr: bool };
    var active: std.ArrayList(Active) = .empty;
    defer active.deinit(allocator);

    for (ivals) |iv| {
        // Expire intervals that ended before this one starts, freeing their registers.
        var w: usize = 0;
        for (active.items) |a| {
            if (a.end < iv.start) {
                switch (a.loc) {
                    .gpr => |r| {
                        gpr_free[r] = true;
                        if (a.is_ptr) gpr_free[r + 1] = true;
                    },
                    .pred => |p| pred_free[p] = true,
                }
            } else {
                active.items[w] = a;
                w += 1;
            }
        }
        active.shrinkRetainingCapacity(w);

        const v = iv.value;
        const l: Loc = if (isBool(func, v)) blk: {
            const p = firstFree(pred_free[0..]) orelse return error.Unsupported;
            pred_free[p] = false;
            break :blk .{ .pred = @intCast(p) };
        } else if (isWidePtr(func, v)) blk: {
            // A 64-bit address needs an aligned pair. A shared address is 32 bits, so it
            // falls through to the single-register arm below.
            const r = firstFreePair(gpr_free[0..]) orelse return error.Unsupported;
            gpr_free[r] = false;
            gpr_free[r + 1] = false;
            if (r + 1 > max_reg.*) max_reg.* = @intCast(r + 1);
            break :blk .{ .gpr = @intCast(r) };
        } else blk: {
            const r = firstFreeSingle(gpr_free[0..]) orelse return error.Unsupported;
            gpr_free[r] = false;
            if (r > max_reg.*) max_reg.* = @intCast(r);
            break :blk .{ .gpr = @intCast(r) };
        };
        try loc.put(allocator, v, l);
        try active.append(allocator, .{ .end = iv.end, .loc = l, .is_ptr = isWidePtr(func, v) });
    }
}

fn firstFree(pool: []const bool) ?usize {
    for (pool, 0..) |f, i| if (f) return i;
    return null;
}

fn firstFreeSingle(gpr_free: []const bool) ?usize {
    for (value_reg_base..encode.RZ) |r| if (gpr_free[r]) return r;
    return null;
}

fn firstFreePair(gpr_free: []const bool) ?usize {
    var r: usize = value_reg_base; // R4 is even, so the scan keeps pairs aligned
    while (r + 1 < encode.RZ) : (r += 2) if (gpr_free[r] and gpr_free[r + 1]) return r;
    return null;
}

fn isBool(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .bool;
}

/// The 32-bit value or bit pattern of a scalar constant value: an integer
/// constant's value, or a float constant's IEEE-754 f32 bits. Returns null if
/// `value` is not a constant. The nvidia `arith_imm` lowering writes this
/// straight into the instruction's 32-bit immediate field, so any 32-bit
/// constant, int or float, can be an immediate operand.
fn constBits(func: *const Function, value: Value) ?i64 {
    const inst = func.definingInst(value) orelse return null;
    return switch (func.opcode(inst)) {
        .iconst => |c| c,
        .fconst => |v| @as(i64, @as(u32, @bitCast(@as(f32, @floatCast(v))))),
        else => null,
    };
}

fn isCommutativeBinOp(op: ir.function.BinOp) bool {
    return switch (op) {
        .add, .mul, .bit_and, .bit_or, .bit_xor => true,
        else => false,
    };
}

/// Fold a constant operand of an `arith` into `arith_imm`, so codegen puts the
/// constant in the instruction's own immediate field instead of holding it in
/// an allocated GPR for its whole live range.
/// The simplex-noise and terrain shaders define 100 or more float constants.
/// Without this fold, those constants would each pin a register and exhaust
/// the GPR pool, since the linear-scan allocator has no spilling. After
/// folding, those constants are dead: each is a [def,def] interval that
/// reuses one register, so peak pressure drops to the real computation
/// pressure. This skips div and rem, since those are lowered specially and
/// not through the general `arithImm` path, pointer adds (64-bit carry), and
/// bool ops (predicate combines). Non-commutative ops fold only the right
/// operand, since the arith_imm form computes `lhs op imm`.
fn foldConstantsToImm(func: *Function) void {
    var i: usize = 0;
    while (i < func.instCount()) : (i += 1) {
        const inst: ir.function.Inst = @enumFromInt(i);
        const op = func.opcodeMut(inst);
        const a = switch (op.*) {
            .arith => |a| a,
            else => continue,
        };
        if (a.op == .div or a.op == .rem) continue;
        const result = func.instResult(inst) orelse continue;
        // A 64-bit address add is a carry chain over a register pair, and a bool op is a
        // predicate combine. Neither has an immediate form. A SHARED address add is plain
        // 32-bit integer arithmetic, so it folds like any other integer.
        if (isWidePtr(func, result) or isBool(func, result)) continue;
        if (constBits(func, a.rhs)) |c| {
            op.* = .{ .arith_imm = .{ .op = a.op, .lhs = a.lhs, .imm = c } };
        } else if (isCommutativeBinOp(a.op)) {
            if (constBits(func, a.lhs)) |c| {
                op.* = .{ .arith_imm = .{ .op = a.op, .lhs = a.rhs, .imm = c } };
            }
        }
    }
}

/// What the address-displacement fold decided, per instruction.
///
/// LDG, STG, LDS and STS each carry a 24-BIT SIGNED BYTE DISPLACEMENT that the hardware
/// adds to the address register. A constant array index or a struct field offset belongs
/// there. Without this fold the instruction selector computed every such address with an
/// IADD3, or with an IADD3 plus an IADD3.X for a 64-bit pointer, and gave the sum a
/// register of its own: three instructions and one register for a number the access could
/// have carried itself.
const DispFold = struct {
    /// The displacement each folded access carries, keyed by the load or store.
    at: std.AutoHashMapUnmanaged(ir.function.Inst, i32) = .empty,
    /// The address instructions the fold left with NO USE AT ALL. Emitting one would
    /// compute a register nothing reads.
    dead: std.AutoHashMapUnmanaged(ir.function.Inst, void) = .empty,

    fn deinit(self: *DispFold, allocator: std.mem.Allocator) void {
        self.at.deinit(allocator);
        self.dead.deinit(allocator);
    }

    /// The byte displacement of a load or store. Zero when nothing folded into it.
    fn offsetOf(self: *const DispFold, inst: ir.function.Inst) i32 {
        return self.at.get(inst) orelse 0;
    }

    /// Whether `inst` computes an address that no instruction reads any more.
    fn isDead(self: *const DispFold, inst: ir.function.Inst) bool {
        return self.dead.contains(inst);
    }
};

/// One step up an address chain: the pointer `v` is `base + imm` bytes, and the constant
/// can move into the access that reads `v`. Returns null when it cannot.
///
/// THE PRIVATE ADDRESS SPACE IS REFUSED. The only private pointers this backend supports
/// are the texture-sample result alloca and its element pointers, which `TexLowering`
/// resolves to TEX result REGISTERS and never to memory at all.
///
/// A POINTER THAT CARRIES AN ATTRIBUTE IS REFUSED. A graphics output attribute, a fragment
/// color, gl_FragDepth and the derivative gradient buffer are all identified by a tag on
/// the pointer value, and each lowers to something other than a memory access. Folding
/// would move the access onto a different value and lose the tag.
fn addrChainStep(func: *const Function, v: Value) ?struct { base: Value, imm: i64 } {
    const space = ptrSpace(func, v) orelse return null;
    switch (space) {
        .global, .constant, .shared => {},
        .private => return null,
    }
    if (hasAnyAttribute(func, v)) return null;
    const inst = func.definingInst(v) orelse return null;
    const a = switch (func.opcode(inst)) {
        .arith_imm => |x| x,
        else => return null,
    };
    if (a.op != .add) return null;
    return .{ .base = a.lhs, .imm = a.imm };
}

/// Whether any attribute is attached to `v`. See `addrChainStep` for why the fold cares.
fn hasAnyAttribute(func: *const Function, v: Value) bool {
    var it = func.attributesOf(.{ .value = v });
    return it.next() != null;
}

/// Move every constant byte displacement out of a load's or store's address chain and into
/// the access itself, then record which address instructions that left with nothing to do.
///
/// THE REWRITE CHANGES THE IR, and that is deliberate. The access now names the BASE
/// pointer as its operand, so the liveness scan that follows extends the base's live range
/// to the access on its own. A side table would not: the linear-scan allocator would end
/// the base's range at the address instruction and hand its register to the next value
/// before the access read it.
///
/// This runs BEFORE `assignLocs`, for that reason, and therefore before `TexLowering` and
/// `DerivLowering` scan. `addrChainStep` refuses every pointer either of those owns.
fn foldAddressDisplacements(allocator: std.mem.Allocator, func: *Function, out: *DispFold) Error!void {
    const nblocks = func.blockCount();
    for (0..nblocks) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            const ptr = switch (func.opcode(inst)) {
                .load => |l| l.ptr,
                .store => |s| s.ptr,
                else => continue,
            };
            var base = ptr;
            var total: i64 = 0;
            // Collapse a whole chain, so `(p + 4) + 8` folds as one displacement of 12.
            // The walk stops at the first step that would leave the field's range.
            while (addrChainStep(func, base)) |step| {
                if (!encode.fitsAddrOffset(total + step.imm)) break;
                total += step.imm;
                base = step.base;
            }
            if (base == ptr) continue;
            // The base is what the access will read, so it must keep its own identity for
            // the same reason `addrChainStep` refuses a tagged pointer.
            if (hasAnyAttribute(func, base)) continue;
            const op = func.opcodeMut(inst);
            switch (op.*) {
                .load => |l| {
                    var moved = l;
                    moved.ptr = base;
                    op.* = .{ .load = moved };
                },
                .store => |s| {
                    var moved = s;
                    moved.ptr = base;
                    op.* = .{ .store = moved };
                },
                else => continue,
            }
            try out.at.put(allocator, inst, @intCast(total));
        }
    }

    // An address instruction whose result nothing reads any more. The fold above is the
    // only thing that removes a use, so this finds exactly the addresses it emptied, plus
    // any pointer arithmetic that was already dead.
    //
    // THE SCAN REPEATS UNTIL IT FINDS NOTHING NEW, because a chain dies from the far end
    // inwards. In `(p + 4) + 8` the fold empties the outer add first, and the inner add
    // still looks used until the outer one is known dead. One pass therefore leaves the
    // inner add alive, and it is a 64-bit pointer add, so it costs two instructions and a
    // register pair for an address the load no longer reads.
    const nval = func.valueCount();
    if (nval == 0) return;
    const used = try allocator.alloc(bool, nval);
    defer allocator.free(used);
    var changed = true;
    while (changed) {
        changed = false;
        @memset(used, false);
        for (0..nblocks) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockInsts(block)) |inst| {
                if (out.dead.contains(inst)) continue;
                markUsedBitset(func, inst, used);
            }
            if (func.terminator(block)) |term| markUsedTermBitset(func, term, used);
        }
        for (0..nblocks) |bi| {
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (out.dead.contains(inst)) continue;
                if (func.opcode(inst) != .arith_imm) continue;
                const r = func.instResult(inst) orelse continue;
                if (ptrSpace(func, r) == null) continue; // an address, not a value
                if (used[@intFromEnum(r)]) continue;
                try out.dead.put(allocator, inst, {});
                changed = true;
            }
        }
    }
}

/// One contracted multiply-add: `dst = mul_a * (mul_b or mul_imm) + addend`.
const Fma = struct {
    /// The multiply's first operand. Always a register.
    mul_a: Value,
    /// The multiply's second operand, when it is a register. Null when the multiply
    /// scales by a constant, which `foldConstantsToImm` has already moved into `mul_imm`.
    mul_b: ?Value,
    /// The constant multiplier, valid only when `mul_b` is null. For a float this is the
    /// IEEE-754 binary32 bit pattern, exactly as `arithImm` passes it.
    mul_imm: u32,
    /// The value the add contributed, which becomes the third source.
    addend: Value,
    /// True for FFMA, false for IMAD.
    is_float: bool,
};

/// Fused multiply-add contraction: which multiplies disappear into which adds.
///
/// THIS IS WORTH A FACTOR OF TWO ON COMPUTE-BOUND KERNELS. Without it `a * b + c` is an
/// FMUL and then an FADD: two instructions, and two dependent latencies, where the
/// hardware has a single-instruction fused multiply-add. Measured against ptxas on an RTX
/// 5070 (sm_120), the missing contraction was the largest single part of a 3.09x gap on
/// compute-bound work. The integer half costs nothing at all to add, because `.mul`
/// already emits `IMAD dst, a, b, RZ` and the addend field it wastes on RZ is exactly
/// where the add's operand belongs.
///
/// `scan` decides, and `lowerInst` obeys: a multiply in `folded` emits NOTHING, and the
/// add in `at` emits the fused instruction instead of a plain add.
///
/// THE MULTIPLY'S OPERANDS MUST STILL BE IN THEIR REGISTERS AT THE ADD. The multiply no
/// longer reads them where it stood, so nothing but this would keep them: the linear-scan
/// allocator ends a value's interval at its last use, which for such an operand was the
/// multiply. `assignLocs` therefore extends both operands to the add, and the scan below
/// is what tells it which those are. Without that extension a value defined between the
/// multiply and the add can take an operand's register and the fused instruction
/// multiplies the wrong number, with no diagnostic.
const FmaFold = struct {
    /// Multiplies that emit nothing, keyed by the multiply's RESULT value.
    folded: std.AutoHashMapUnmanaged(Value, void) = .empty,
    /// The fused form of each contracted add, keyed by the add's RESULT value.
    at: std.AutoHashMapUnmanaged(Value, Fma) = .empty,

    fn deinit(self: *FmaFold, allocator: std.mem.Allocator) void {
        self.folded.deinit(allocator);
        self.at.deinit(allocator);
    }

    /// Whether the instruction that defines `v` is a multiply the fused form absorbed.
    fn isFolded(self: *const FmaFold, v: Value) bool {
        return self.folded.contains(v);
    }
};

/// The multiply an `add` can absorb, or null when it cannot absorb `v`.
///
/// EVERY CONDITION HERE IS CHECKED, NOT ASSUMED, because each one is a wrong answer when
/// it does not hold:
///
///   - ONE USE. A multiply read twice must keep its own instruction. Contracting it would
///     put the multiply inside the add and leave the other reader with an unwritten
///     register. `uses` comes from `opt.dce.countUses`, which counts every instruction
///     operand, every `if` edge argument and every terminator operand.
///   - SAME BLOCK. `in_block` holds the values this block defines above the add, so a
///     multiply from another block is refused. SSA already orders a definition before its
///     use inside a block, so membership is the whole test.
///   - THE SAME 32-BIT TYPE. FFMA and IMAD are both 32 bits wide. A wider or narrower
///     type would be truncated silently, so it is refused rather than contracted.
fn fmaOperand(
    func: *const Function,
    uses: []const u32,
    in_block: *const std.AutoHashMapUnmanaged(Value, void),
    v: Value,
    addend: Value,
) ?Fma {
    if (uses[@intFromEnum(v)] != 1) return null;
    if (!in_block.contains(v)) return null;
    // Both sources of the fused instruction must be the SAME 32-bit type. Checking the
    // add's result alone would not say that, so each operand is tested here.
    if (!isFmaWidth(func, v)) return null;
    if (func.valueType(v) != func.valueType(addend)) return null;
    const inst = func.definingInst(v) orelse return null;
    const is_float = isFloat(func, v);
    return switch (func.opcode(inst)) {
        .arith => |m| if (m.op == .mul)
            .{ .mul_a = m.lhs, .mul_b = m.rhs, .mul_imm = 0, .addend = addend, .is_float = is_float }
        else
            null,
        // `foldConstantsToImm` has already turned `x * k` into an `arith_imm`. FFMA and
        // IMAD both read a 32-bit immediate multiplier with a register addend, which is
        // the `base + index * stride` an array walk writes.
        .arith_imm => |m| if (m.op == .mul)
            .{
                .mul_a = m.lhs,
                .mul_b = null,
                .mul_imm = @truncate(@as(u64, @bitCast(m.imm))),
                .addend = addend,
                .is_float = is_float,
            }
        else
            null,
        else => null,
    };
}

/// Whether `v` is a 32-bit float or a 32-bit integer, the two types FFMA and IMAD cover.
/// A bool, a pointer, a vector and every other width are refused.
fn isFmaWidth(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |k| k == .f32,
        .int => |x| x.bits == 32,
        .bool, .ptr, .vector, .array, .slice, .@"struct" => false,
    };
}

/// Find every `a * b + c` this backend can fuse. See `FmaFold`.
///
/// SUBTRACTION IS DELIBERATELY LEFT ALONE. `a * b - c` and `c - a * b` are both fusible on
/// the hardware, but each negates a DIFFERENT source, and this encoder has no tested
/// negate modifier for either FFMA or IMAD. A wrong negate bit is a silent wrong answer,
/// which this backend has been bitten by before, so `sub` keeps its FMUL and FADD pair
/// until the negate bits are read out of a real ptxas encoding and tested.
fn scanFma(allocator: std.mem.Allocator, func: *const Function, options: Options, out: *FmaFold) Error!void {
    if (!options.contract_fma) return;
    const nval = func.valueCount();
    if (nval == 0) return;

    const uses = try allocator.alloc(u32, nval);
    defer allocator.free(uses);
    opt.dce.countUses(func, uses);

    var in_block: std.AutoHashMapUnmanaged(Value, void) = .empty;
    defer in_block.deinit(allocator);

    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        in_block.clearRetainingCapacity();
        for (func.blockParams(block)) |p| try in_block.put(allocator, p, {});
        for (func.blockInsts(block)) |inst| {
            try contractOne(allocator, func, uses, &in_block, inst, out);
            // After the decision, never before it: `in_block` must hold what stands ABOVE
            // this instruction, so an add can never absorb its own result.
            if (func.instResult(inst)) |r| try in_block.put(allocator, r, {});
        }
    }
}

/// Contract `inst` if it is an add that can absorb a multiply. See `scanFma`.
fn contractOne(
    allocator: std.mem.Allocator,
    func: *const Function,
    uses: []const u32,
    in_block: *const std.AutoHashMapUnmanaged(Value, void),
    inst: ir.function.Inst,
    out: *FmaFold,
) Error!void {
    const a = switch (func.opcode(inst)) {
        .arith => |x| x,
        // An `arith_imm` add carries its addend as an IMMEDIATE, and neither FFMA nor IMAD
        // has an immediate third source: the immediate field is the multiplier. So `a * b
        // + k` keeps its two instructions.
        else => return,
    };
    if (a.op != .add) return;
    const r = func.instResult(inst) orelse return;
    // Only the plain 32-bit register-add arm of `lowerInst` becomes a fused instruction. A
    // 64-bit pointer add is an IMAD.WIDE carry chain and a boolean add is a predicate
    // combine, and the width test refuses both.
    if (!isFmaWidth(func, r)) return;
    if (isWidePtr(func, r)) return;
    // `a * b + c`, then the commuted `c + a * b`. Both forms reach here, because nothing
    // orders an add's operands. When BOTH operands are contractible multiplies only one is
    // taken, since the instruction has room for one multiply.
    const fused = fmaOperand(func, uses, in_block, a.rhs, a.lhs) orelse
        fmaOperand(func, uses, in_block, a.lhs, a.rhs) orelse
        return;
    try out.folded.put(allocator, if (fused.addend == a.lhs) a.rhs else a.lhs, {});
    try out.at.put(allocator, r, fused);
}

/// The one instruction a contracted multiply-add becomes.
fn fmaInst(loc: std.AutoHashMapUnmanaged(Value, Loc), rd: u8, m: Fma) Inst {
    const ra = gprOf(loc, m.mul_a);
    const rc = gprOf(loc, m.addend);
    if (m.mul_b) |b| {
        const rb = gprOf(loc, b);
        return if (m.is_float) encode.ffma(rd, ra, rb, rc, .{}) else encode.imad(rd, ra, rb, rc, .{});
    }
    return if (m.is_float) encode.ffmaImm(rd, ra, m.mul_imm, rc, .{}) else encode.imadImm(rd, ra, m.mul_imm, rc, .{});
}

/// The address space a pointer-typed value points into, or null when the value is not a
/// pointer at all.
fn ptrSpace(func: *const Function, v: Value) ?ir.types.AddressSpace {
    return switch (func.types.type_kind(func.valueType(v))) {
        .ptr => |space| space,
        .bool, .int, .float, .vector, .array, .slice, .@"struct" => null,
    };
}

/// True when a value is an address that occupies an ALIGNED 64-BIT GPR PAIR (lo, lo+1).
///
/// This answers the WIDTH question, and the width depends on the address space. A `global` or
/// `constant` address is a full 64-bit device address, so it needs a pair, a carry-chain add,
/// and two constant-bank loads. A `private` address is one too on this backend, because the
/// only local storage it models is reached through the generic 64-bit window.
///
/// A `shared` address is NOT. It is a 32-bit byte offset into the CTA's shared-memory window,
/// which LDS and STS read out of ONE register. Giving it a pair would waste a register, add a
/// meaningless high word to every address computation, and emit a 64-bit carry chain over an
/// offset that cannot carry.
fn isWidePtr(func: *const Function, v: Value) bool {
    const space = ptrSpace(func, v) orelse return false;
    return switch (space) {
        .global, .constant, .private => true,
        .shared => false,
    };
}

/// The memory access width for the value a load produces or a store consumes.
///
/// The IR value type is the only thing that knows how many bytes the access moves, so an
/// encoder that always used B32 read or wrote the wrong number of bytes for every other type,
/// with no diagnostic: a byte store destroyed the three bytes next to it, and a 64-bit load
/// took only the low half.
///
/// A value 32 bits wide or narrower lands in ONE GPR, which is what `assignLocs` gives it, so
/// every such width is emitted here. A narrow load extends into the whole register, signed or
/// unsigned to match the type.
///
/// A 64-BIT SCALAR is refused, and that refusal is deliberate. `assignLocs` gives a pointer an
/// aligned GPR pair, but it gives every other non-boolean value exactly ONE register, and this
/// backend has no 64-bit scalar arithmetic: an `arith` on an i64 lowers to a single 32-bit
/// IADD3 or IMAD over the low half. A B64 load would therefore write a register the allocator
/// has already given to a different live value, and its consumers would still do 32-bit
/// arithmetic on it. Refusing follows `emitComputeBuiltin`: a kernel that looks right and
/// computes garbage is worse than one that does not compile, because no test in this
/// repository can execute SASS to catch it. Full 64-bit scalar support needs a paired register
/// class plus a carry-chain lowering for every integer operation.
///
/// A POINTER is different: `isWidePtr` already reserves the aligned pair for a global,
/// constant or private address, so a B64 access fills or reads exactly the pair that value
/// owns. A shared address is a 32-bit window offset in one register.
fn memTypeOf(func: *const Function, v: Value) Error!encode.MemType {
    switch (func.types.type_kind(func.valueType(v))) {
        // A boolean lives in a PREDICATE, not a GPR, so it has no register for a load to fill
        // or a store to read. `gprOf` would trip its `unreachable` on the way here.
        .bool => return error.Unsupported,
        .int => |i| {
            const is_signed = i.signedness == .signed;
            if (i.bits <= 8) return if (is_signed) .i8 else .u8;
            if (i.bits <= 16) return if (is_signed) .i16 else .u16;
            if (i.bits <= 32) return .b32;
            return error.Unsupported; // see the 64-bit note above
        },
        // f16 and f128 never reach here: `compileKernel` and `compileShader` refuse a function
        // that holds either. f64 needs the register pair the note above describes.
        .float => |f| return switch (f) {
            .f32 => .b32,
            .f16, .f64, .f128 => error.Unsupported,
        },
        .ptr => return if (isWidePtr(func, v)) .b64 else .b32,
        .vector, .@"struct", .array, .slice => return error.Unsupported,
    }
}

/// The atomic access type for `v`. It sets the WIDTH of the read-modify-write and, for `min`
/// and `max`, whether the comparison is signed.
///
/// The hardware field holds one of four values, so only a 32-bit and a 64-bit access exist.
/// There is no byte or half-word atomic, so an i8 or an i16 is REFUSED and never widened: a
/// widened access reads and writes the bytes beside it, which is a data race this backend
/// would have created on its own.
///
/// 64 bits is refused for the reason `memTypeOf` refuses a 64-bit scalar: the old value comes
/// back in a register PAIR, and this backend has no pair allocation for a scalar value. The
/// encoders accept `u64`/`i64` and are tested for them, so the refusal is here and not there.
fn atomTypeOf(func: *const Function, v: Value) Error!encode.AtomType {
    switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| {
            if (i.bits != 32) return error.Unsupported;
            return if (i.signedness == .signed) .i32 else .u32;
        },
        // `verify` rejects a non-integer operand, so none of these reaches a verified
        // function. The refusal keeps an unverified one from picking an integer atomic of
        // the same width and computing integer arithmetic on a float bit pattern.
        .bool, .float, .ptr, .vector, .@"struct", .array, .slice => return error.Unsupported,
    }
}

/// The hardware operation selector for an IR atomic operation.
fn atomOpOf(op: ir.function.AtomicOp) Error!encode.AtomOp {
    return switch (op) {
        .add => .add,
        .min => .min,
        .max => .max,
        .bit_and => .bit_and,
        .bit_or => .bit_or,
        .bit_xor => .bit_xor,
        .exchange => .exch,
        // Compare-exchange is a SEPARATE opcode with a second data operand, so it has no
        // value in this field. `lowerAtomicRmw` routes it to ATOMG.CAS or ATOMS.CAS before
        // it reaches here.
        .compare_exchange => error.Unsupported,
    };
}

/// Lower an atomic read-modify-write. Four instructions, picked by two independent facts.
///
/// The ADDRESS SPACE comes from the pointer type, exactly as it does for `load` and `store`:
/// a shared pointer takes ATOMS, whose address is a 32-bit window offset in ONE register, and
/// anything else takes the global form, whose address is a 64-bit pair at (addr, addr+1).
///
/// WHETHER THE OLD VALUE IS READ picks between the two global forms, and this is the whole
/// reason the IR result is optional. A read old value takes ATOMG, which writes a destination
/// register and so claims one of the SIX scoreboards this GPU has. An old value nobody reads
/// takes RED, which writes no register at all and claims none. A fire-and-forget counter
/// increment therefore costs no scoreboard.
///
/// A `compare_exchange` is its own opcode in both spaces, because it needs a second data
/// operand.
///
/// Nothing here refuses a divergent placement, unlike `checkBarrierConvergence`. An atomic
/// inside a divergent arm is well defined: the threads that take the arm apply it, and the
/// hardware serialises them.
fn lowerAtomicRmw(
    allocator: std.mem.Allocator,
    func: *const Function,
    loc: std.AutoHashMapUnmanaged(Value, Loc),
    code: *std.ArrayList(Inst),
    inst: ir.function.Inst,
    a: ir.function.AtomicRmw,
) Error!void {
    const ty = try atomTypeOf(func, a.value);
    const addr = gprOf(loc, a.ptr);
    const data = gprOf(loc, a.value);
    // RZ discards the value written to it, and the scheduler skips an RZ destination when it
    // hands out scoreboards, so this is how a form that must name a destination register
    // gives back nothing.
    const dst: u8 = if (func.instResult(inst)) |r| gprOf(loc, r) else encode.RZ;

    if (isSharedPtr(func, a.ptr)) {
        // ATOMS is Strong(CTA) in the hardware: NAK asserts it, and the encoder writes no
        // memory-order field at all. Shared memory is private to the CTA, so workgroup scope
        // is the whole of it and a wider scope is a promise ATOMS cannot keep. Refuse rather
        // than emit an instruction that orders less than the program asked for.
        if (a.scope != .workgroup) return error.Unsupported;
        if (a.op == .compare_exchange) {
            try code.append(allocator, encode.atomsCas(dst, addr, gprOf(loc, a.compare.?), data, ty, .{}));
            return;
        }
        try code.append(allocator, encode.atoms(dst, addr, data, try atomOpOf(a.op), ty, .{}));
        return;
    }

    // Every global form the encoders build sets memory order STRONG / SYS, the strongest one
    // there is, so it satisfies every ordering and every scope the IR can ask for. A backend
    // may lower a weak order with a stronger instruction; the reverse is what is forbidden.
    if (a.op == .compare_exchange) {
        try code.append(allocator, encode.atomgCas(dst, addr, gprOf(loc, a.compare.?), data, ty, .{}));
        return;
    }
    if (func.instResult(inst) == null and a.op != .exchange) {
        try code.append(allocator, encode.redg(addr, data, try atomOpOf(a.op), ty, .{}));
        return;
    }
    // ATOMG, either because the old value is read or because the operation is an exchange.
    // RED's operation field is only 3 bits wide, so `exch` (8) does not fit it and
    // `encode.redg` asserts on one. An unread exchange therefore takes ATOMG with an RZ
    // destination, which discards the old value and still claims no scoreboard.
    try code.append(allocator, encode.atomg(dst, addr, data, try atomOpOf(a.op), ty, .{}));
}

/// True when a value is a workgroup-shared address: a 32-bit window offset in ONE register,
/// which a load or a store reaches with LDS or STS instead of LDG or STG.
fn isSharedPtr(func: *const Function, v: Value) bool {
    const space = ptrSpace(func, v) orelse return false;
    return switch (space) {
        .shared => true,
        .global, .constant, .private => false,
    };
}

/// Emit the hardware read for a compute builtin parameter.
///
/// The thread index, the block index and the fused global index lower on all three axes. Each
/// index reads its own special register, whose number comes from NAK (see the `SR_*` block in
/// `encode.zig`). The workgroup size lowers to an immediate, because the kernel declares it.
/// The grid size loads from the launch-shape region of the parameter block, exactly as a
/// 32-bit scalar parameter loads.
///
/// The subgroup builtins have no correct lowering here yet and return `error.Unsupported`.
/// Each refusal says why below. A refusal is deliberate: a builtin that reads the wrong
/// register produces a kernel that looks right and computes garbage, and no test in this
/// repository can execute SASS to catch that.
fn emitComputeBuiltin(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(Inst),
    func: *const Function,
    loc: std.AutoHashMapUnmanaged(Value, Loc),
    p: Value,
    bi: gpu.Builtin,
    a: gpu.Abi,
    shape: ?gpu.kernel.LaunchShape,
) Error!void {
    if (!bi.isCompute()) return error.Unsupported;
    const dst = gprOf(loc, p);
    switch (bi) {
        .thread_id_x => try code.append(allocator, encode.s2r(dst, encode.SR_TID_X, .{})),
        .thread_id_y => try code.append(allocator, encode.s2r(dst, encode.SR_TID_Y, .{})),
        .thread_id_z => try code.append(allocator, encode.s2r(dst, encode.SR_TID_Z, .{})),
        .block_id_x => try code.append(allocator, encode.s2r(dst, encode.SR_CTAID_X, .{})),
        .block_id_y => try code.append(allocator, encode.s2r(dst, encode.SR_CTAID_Y, .{})),
        .block_id_z => try code.append(allocator, encode.s2r(dst, encode.SR_CTAID_Z, .{})),
        .lane_id => try code.append(allocator, encode.s2r(dst, encode.SR_LANEID, .{})),
        .global_id_x => try emitGlobalId(allocator, code, func, dst, 0),
        .global_id_y => try emitGlobalId(allocator, code, func, dst, 1),
        .global_id_z => try emitGlobalId(allocator, code, func, dst, 2),
        // The workgroup size is not a special register on this hardware. It is the size the
        // kernel DECLARES, which `Kernel.launch.block` carries to the runtime and which the
        // launch descriptor must match, so it is a compile-time constant here. The fused
        // global index above folds the same number into the same immediate, so a kernel that
        // reads both stays self-consistent.
        .block_dim_x => try code.append(allocator, encode.movImm(dst, gpu.attrs.localSize(func)[0], .{})),
        .block_dim_y => try code.append(allocator, encode.movImm(dst, gpu.attrs.localSize(func)[1], .{})),
        .block_dim_z => try code.append(allocator, encode.movImm(dst, gpu.attrs.localSize(func)[2], .{})),
        // The grid size is a launch-time value, not a compile-time one, and NVIDIA has no
        // special register for it. It comes from the launch-shape region that `layoutParams`
        // reserves at the end of the parameter block, and it loads exactly as a 32-bit scalar
        // parameter loads: one LDC from constant bank 0 at the ABI's parameter base plus the
        // region's offset for this axis. The CUDA driver puts the same three numbers in its
        // own reserved area of the bank, but that offset is a driver convention this ABI does
        // not define, so the region is where a vulcan runtime writes them.
        //
        // `layoutParams` reserves the region for exactly these three builtins, so `shape` is
        // never null on this path. It stays a checked refusal rather than an assert, because
        // the two decisions live in different modules.
        .grid_dim_x,
        .grid_dim_y,
        .grid_dim_z,
        => {
            const region = shape orelse return error.Unsupported;
            const at: u16 = @intCast(a.param_base + region.axisOffset(bi.axis().?));
            try code.append(allocator, encode.ldc(dst, bank0, at, .{}));
        },
        // The warp index inside a workgroup has no special register of its own. It is
        // derivable from the combined thread index and the warp width, but that derivation
        // needs the full workgroup shape and has no oracle to check it against, because the
        // host loop nest runs one thread at a time and has no subgroup. `subgroup_size` waits
        // with it, so the whole subgroup group lands together and gets tested together.
        .warp_id,
        .subgroup_size,
        => return error.Unsupported,
        .vertex_index,
        .instance_index,
        .frag_coord,
        .point_coord,
        .front_facing,
        => return error.Unsupported,
    }
}

/// Load a 64-bit address from a constant bank into the register pair (dst, dst + 1).
///
/// ONE `LDC.64` DOES BOTH HALVES. Two 32-bit LDCs were emitted before, and every kernel
/// paid an extra instruction per pointer parameter in its prologue.
///
/// A 64-bit constant-bank read needs an 8-ALIGNED offset. `gpu.abi.layoutParams` aligns
/// every pointer slot to the target's pointer width, so the offset inside the block is
/// always 8-aligned, but `Abi.param_base` is DATA that a runtime chooses: the nvidia.zig
/// dispatch uses 0 and the CUDA driver uses 0x160, both 8-aligned, and a base that is only
/// 4-aligned would move every pointer off it. So the alignment is checked and not assumed,
/// and a misaligned offset takes the two-LDC path that always worked.
///
/// `dst` is even, because `assignLocs` gives every 64-bit address an aligned pair.
fn emitPointerLdc(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(Inst),
    dst: u8,
    bank: u5,
    at: u16,
) Error!void {
    if (at % 8 == 0 and dst % 2 == 0) {
        try code.append(allocator, encode.ldcWide(dst, bank, at, .{}));
        return;
    }
    try code.append(allocator, encode.ldc(dst, bank, at, .{})); // address lo
    try code.append(allocator, encode.ldc(dst + 1, bank, at + 4, .{})); // address hi
}

/// Emit `gid.a = ctaid.a * ntid.a + tid.a` into `dst` for axis `a`.
///
/// The workgroup size is a compile-time constant, so it goes in IMAD's own immediate field
/// rather than into a register of its own. The two scratch registers hold the hardware
/// reads: the prologue owns them, and the allocator keeps values out of them.
fn emitGlobalId(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(Inst),
    func: *const Function,
    dst: u8,
    a: u2,
) Error!void {
    const size = gpu.attrs.localSize(func)[a];
    try code.append(allocator, encode.s2r(r_scratch, encode.sr_tid[a], .{}));
    try code.append(allocator, encode.s2r(r_scratch2, encode.sr_ctaid[a], .{}));
    try code.append(allocator, encode.imadImm(dst, r_scratch2, size, r_scratch, .{}));
}

/// Whether `v` carries the named `vulcan.gpu` flag or attribute, in any value form.
fn hasGpuKey(func: *const Function, v: Value, key: []const u8) bool {
    var it = func.attributesOf(.{ .value = v });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, key)) return true,
        else => {},
    };
    return false;
}

/// Whether `v` is the synthesized host-sampler function-pointer entry param
/// the SPIR-V image-sample lowering appends, tagged `vulcan.gpu.sampler_fn`.
/// The NVIDIA backend ignores it: a GPU TEX needs no host function pointer,
/// so the param gets no constant-bank slot, and the sampler `call_indirect`
/// through it becomes a TEX.
fn isSamplerFn(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "sampler_fn");
}

/// Whether `v` is the host vec3-sampler param: a `samplerCube`
/// (sampler_cube_fn), `sampler3D` (sampler_3d_fn), or `sampler2DArray`
/// (sampler_2darray_fn). The GPU emits a TEX with the matching dimension and
/// a 3-register coordinate group.
fn isSamplerVec3Fn(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "sampler_cube_fn") or hasGpuKey(func, v, "sampler_3d_fn") or hasGpuKey(func, v, "sampler_2darray_fn");
}
/// The NAK TEX dim for a vec3 sampler param (cube vs 3D vs 2D-array).
fn samplerVec3Dim(func: *const Function, v: Value) u8 {
    if (hasGpuKey(func, v, "sampler_3d_fn")) return encode.TexDim.dim_3d;
    if (hasGpuKey(func, v, "sampler_2darray_fn")) return encode.TexDim.array_2d;
    return encode.TexDim.cube;
}

/// Whether `v` is the host depth-compare sampler param, tagged
/// `vulcan.gpu.sampler_shadow_fn`: a `sampler2DShadow` sample (SPIR-V
/// OpImageSampleDref). The GPU emits a TEX with z_cmpr (bit 78) that compares
/// the shader dref against the stored depth (R) and returns a scalar pass
/// fraction instead of a host call. The ABI is
/// `f32 sampler_shadow_fn(desc, u, v, lod, dref)`, a direct scalar result
/// with no out pointer. The dref is packed into src1 right after the handle
/// (src1 = [handle, dref], NAK z_cmpr).
fn isSamplerShadowFn(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "sampler_shadow_fn");
}

/// Whether `v` is the host cube depth-compare sampler param
/// (`sampler_cube_shadow_fn`): a `samplerCubeShadow` sample. ABI
/// `f32 sampler_cube_shadow_fn(desc, x, y, z, lod, dref)`. The GPU reuses the
/// samplerCube atlas lowering (major-axis to a 2D atlas u, v), then does a
/// 2D z_cmpr TEX.
fn isSamplerCubeShadowFn(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "sampler_cube_shadow_fn");
}

/// Whether `v` is the host 2D-array depth-compare sampler param
/// (`sampler_2darray_shadow_fn`): a `sampler2DArrayShadow` sample. ABI
/// `f32 sampler_2darray_shadow_fn(desc, u, v, layer, lod, dref)`. The GPU
/// emits a native TWO_D_ARRAY z_cmpr TEX, with coordinate = layer, u, v and
/// the layer index first.
fn isSampler2dArrayShadowFn(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "sampler_2darray_shadow_fn");
}

/// Any depth-compare (shadow) sampler param: 2D, cube, or 2D-array. All three
/// lower to a z_cmpr TEX (encode.texShadow) that returns a scalar compare
/// fraction with the dref in src1, right after the handle. They differ only
/// in the TEX dimension and coordinate assembly.
fn isAnyShadowFn(func: *const Function, v: Value) bool {
    return isSamplerShadowFn(func, v) or isSamplerCubeShadowFn(func, v) or isSampler2dArrayShadowFn(func, v);
}

/// Whether `v` is the host gather param, tagged
/// `vulcan.gpu.sampler_gather_fn`: the `textureGather` idiom. The GPU emits a
/// TLD4 (bindless gather) instead of a host call. The ABI is
/// `sampler_gather_fn(desc, u, v, comp, out)`, where `comp` (0..3) is a
/// compile-time fconst.
fn isSamplerGatherFn(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "sampler_gather_fn");
}

/// Whether `v` is the host fetch param, tagged
/// `vulcan.gpu.sampler_fetch_fn`: the `texelFetch` idiom. The GPU emits a TLD
/// (bindless texel fetch) instead of a host call. The ABI is
/// `sampler_fetch_fn(desc, x:i32, y:i32, lod:i32, out)`: integer
/// coordinates, an explicit LOD, and no filter.
fn isSamplerFetchFn(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "sampler_fetch_fn");
}

/// Whether `v` is a 2D-array or 3D fetch param (`sampler_fetch_array_fn` or
/// `sampler_fetch_3d_fn`): a `texelFetch` on a layered or volume texture. ABI
/// `fn(desc, x:i32, y:i32, z:i32, lod:i32, out)`. The GPU emits a TLD with
/// the matching dimension (Array2D or 3D) and a 3-register integer
/// coordinate.
fn isSamplerFetch3Fn(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "sampler_fetch_array_fn") or hasGpuKey(func, v, "sampler_fetch_3d_fn");
}
/// The NAK TLD dimension for a fetch3 param (2D-array or 3D). The coordinate
/// order also differs. See the emit code.
fn fetch3Dim(func: *const Function, v: Value) u8 {
    return if (hasGpuKey(func, v, "sampler_fetch_array_fn")) encode.TexDim.array_2d else encode.TexDim.dim_3d;
}

/// Whether `v` is a combined-image-sampler descriptor entry param, tagged
/// `vulcan.gpu.sampler_desc`. On the NVIDIA backend it is not a memory
/// pointer: its constant-bank slot holds the bindless texture handle
/// (tic | tsc<<20) the dispatch side binds, loaded with a single LDC and fed
/// to TEX.
fn isSamplerDesc(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "sampler_desc");
}

/// A `vulcan.gpu` integer attribute named `key` attached to value `v` (a
/// graphics attribute slot). Returns null if absent.
fn attrTag(func: *const Function, v: Value, key: []const u8) ?u16 {
    var it = func.attributesOf(.{ .value = v });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, key)) {
            return switch (c.value) {
                .int => |n| @intCast(n),
                else => null,
            };
        },
        else => {},
    };
    return null;
}

fn returnsValue(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        if (func.terminator(@enumFromInt(bi))) |t| switch (t) {
            .ret => |r| if (r.count != 0) return true,
            else => {},
        };
    }
    return false;
}

/// Per-function texture-sample lowering state for the NVIDIA backend. This
/// maps the SPIR-V image-sample idiom (a stack alloca written by a
/// host-sampler call and reloaded component by component) onto a TEX result
/// register block.
const TexLowering = struct {
    allocator: std.mem.Allocator,
    /// Maps an alloca Value (the sampler out-pointer) to the base of its
    /// 4-register RGBA block.
    out_base: std.AutoHashMapUnmanaged(Value, u8) = .empty,
    /// Maps an element-pointer value (the alloca itself or `alloca + c*4`) to
    /// its out alloca and component index. A reload `load` of one of these
    /// resolves to out_base+component.
    elem: std.AutoHashMapUnmanaged(Value, Elem) = .empty,
    /// Maps a `call_indirect` Inst that is a sampler call to its out alloca,
    /// u, v, and handle.
    calls: std.AutoHashMapUnmanaged(u32, Call) = .empty,

    const Elem = struct { alloca: Value, comp: u8 };
    // `coord` is a reserved consecutive register pair (coord, coord+1) the
    // TEX reads u and v from. It is reserved above the allocator's
    // watermark, not the fixed R0/R1 scratch. When a shader has both a
    // texture and other low-register-pressure features (for example,
    // derivatives), the linear-scan allocator can legitimately assign live
    // SSA values to R0/R1, and moving u/v into R0/R1 for the TEX would
    // clobber them mid-shader.
    // `w`/`dim`: a vec3 sampler (cube or 3D) threads a third coordinate (w)
    // and a non-2D TEX dimension. The reserved coordinate group is then a
    // triple (coord, coord+1, coord+2). `w` is undefined for 2D.
    // `scratch`: the base of a reserved 12-register block the cube lowering
    // uses for the branchless major-axis math (direction to face plus face
    // u, v). This is 0 for non-cube calls, which need no scratch.
    // `gather_comp`, when non-null, marks a `textureGather` call: emit a
    // TLD4 that fetches this component (0..3) of the 4-texel footprint
    // instead of a filtered TEX. Null means an ordinary sample.
    // `is_fetch` marks a `texelFetch` call: emit a TLD (integer coordinates
    // plus explicit LOD) into a quad coordinate group (x, y, handle, lod)
    // instead of a filtered TEX.
    // `is_shadow` marks a `sampler2DShadow` depth-compare sample
    // (OpImageSampleDref): emit a TEX with z_cmpr (encode.texShadow) that
    // returns a scalar into the call's SSA result register, not a 4-register
    // out block, since there is no out pointer. `dref` is the compare
    // reference, packed into src1 right after the handle (src1 = [handle,
    // dref]). The coordinate group is a quad (u, v, handle-copy, dref).
    // `explicit_lod` marks a 2D sample whose LOD is explicit: textureLod, or
    // any sample in a vertex shader, since a vertex shader has no
    // derivatives, so its `texture2D` lowered to explicit LOD-0. Emit a
    // TEX.LL (encode.texLod) that reads the LOD from the (handle, lod) pair
    // at coord+2, instead of the auto-LOD TEX. The auto-LOD TEX needs quad
    // derivatives, which are undefined in a vertex shader and give the
    // wrong level for textureLod. Implicit 2D samples carry the LOD
    // sentinel (-1e30) and keep the auto-LOD TEX.
    const Call = struct { out: Value, u: Value, v: Value, w: Value, lod: Value, handle: Value, coord: u8, dim: u8, scratch: u8 = 0, gather_comp: ?u8 = null, is_fetch: bool = false, is_shadow: bool = false, explicit_lod: bool = false, dref: Value = undefined };
    const cube_scratch_regs: u8 = 12;

    fn init(allocator: std.mem.Allocator) TexLowering {
        return .{ .allocator = allocator };
    }
    fn deinit(self: *TexLowering) void {
        self.out_base.deinit(self.allocator);
        self.elem.deinit(self.allocator);
        self.calls.deinit(self.allocator);
    }

    /// Whether any sampler call was found (this function textures).
    fn any(self: *const TexLowering) bool {
        return self.calls.count() > 0;
    }

    /// Scan the function for sampler `call_indirect`s whose target is tagged
    /// `sampler_fn`. Allocate a 4-register RGBA block per out-pointer alloca,
    /// above `max_reg`, and record the element-pointer values and components
    /// so the reload loads resolve.
    fn scan(self: *TexLowering, func: *const Function, max_reg: *u8, stage: Stage) Error!void {
        const nblocks = func.blockCount();
        // Map each iconst result value to its integer, so a texture
        // element-pointer's static byte offset (`alloca + iconst`) can be
        // recovered without a value-to-def index, since the IR exposes no
        // such lookup.
        var iconst_of = std.AutoHashMapUnmanaged(Value, i64){};
        defer iconst_of.deinit(self.allocator);
        // The gather component reaches isel as an `fconst` call argument,
        // since the reader synthesizes `fconst(comp)`. Map fconst results to
        // their value so a gather call can recover its component.
        var fconst_of = std.AutoHashMapUnmanaged(Value, f64){};
        defer fconst_of.deinit(self.allocator);
        for (0..nblocks) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockInsts(block)) |inst| {
                if (func.opcode(inst) == .iconst) {
                    if (func.instResult(inst)) |r| try iconst_of.put(self.allocator, r, func.opcode(inst).iconst);
                } else if (func.opcode(inst) == .fconst) {
                    if (func.instResult(inst)) |r| try fconst_of.put(self.allocator, r, func.opcode(inst).fconst);
                }
            }
        }
        for (0..nblocks) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockInsts(block)) |inst| {
                if (func.opcode(inst) != .call_indirect) continue;
                const c = func.opcode(inst).call_indirect;
                const is_2d = isSamplerFn(func, c.target);
                const is_vec3 = isSamplerVec3Fn(func, c.target);
                const is_gather = isSamplerGatherFn(func, c.target);
                const is_fetch = isSamplerFetchFn(func, c.target);
                const is_fetch3 = isSamplerFetch3Fn(func, c.target);
                const is_cube_shadow = isSamplerCubeShadowFn(func, c.target);
                const is_array_shadow = isSampler2dArrayShadowFn(func, c.target);
                const is_shadow = isSamplerShadowFn(func, c.target) or is_cube_shadow or is_array_shadow;
                if (!is_2d and !is_vec3 and !is_gather and !is_fetch and !is_fetch3 and !is_shadow) continue;
                const args = func.valueList(c.args);
                // Depth-compare samples (OpImageSampleDref) have no out
                // pointer. The call returns a scalar directly into its
                // allocator-assigned result register, and the dref is the
                // last arg, packed into src1 right after the handle (z_cmpr
                // reads it from handle_reg+1). The three variants differ in
                // coordinate assembly:
                //   sampler2DShadow:      {desc, u, v, lod, dref} becomes a
                //                         coordinate quad (u, v, handle, dref).
                //   samplerCubeShadow:    {desc, x, y, z, lod, dref} becomes
                //                         atlas major-axis math (reusing the
                //                         samplerCube lowering) into a
                //                         coordinate quad (u', v, handle,
                //                         dref) plus a 12-register scratch
                //                         block. The emitted TEX is 2D over
                //                         the 6-face atlas.
                //   sampler2DArrayShadow: {desc, u, v, layer, lod, dref}
                //                         becomes a native TWO_D_ARRAY: a
                //                         4-aligned coordinate group (layer,
                //                         u, v) (a 3-register coordinate
                //                         faults Xid 13 if only 2-aligned)
                //                         plus a (handle, dref) pair at
                //                         coord+4/coord+5.
                if (is_shadow) {
                    if (is_cube_shadow) {
                        if (args.len != 6) return error.Unsupported;
                        // The emitted TEX is 2D over the 6-face atlas: it
                        // reads a 2-register coordinate pair (u', v) at
                        // `coord`, and a 2-register [handle, dref] pair at
                        // coord+2. Both are power-of-2-sized vector
                        // operands, so `coord` must be even-aligned. An odd
                        // base faults the SM with Xid 13, "Misaligned
                        // Register". The 2D shadow path only survived this
                        // because its watermark happened to land coord
                        // even. alignForward to 2 keeps coord and coord+2
                        // even.
                        var coord: u8 = @intCast(@as(u32, max_reg.*) + 1);
                        coord = std.mem.alignForward(u8, coord, 2);
                        if (@as(u32, coord) + 4 - 1 >= encode.RZ) return error.Unsupported;
                        max_reg.* = coord + 3;
                        const scratch: u8 = @intCast(@as(u32, max_reg.*) + 1);
                        if (@as(u32, scratch) + cube_scratch_regs - 1 >= encode.RZ) return error.Unsupported;
                        max_reg.* = scratch + cube_scratch_regs - 1;
                        try self.calls.put(self.allocator, @intFromEnum(inst), .{
                            .out = args[0], // unused for shadow (no out block); kept non-undefined
                            .u = args[1], // x
                            .v = args[2], // y
                            .w = args[3], // z
                            .lod = args[4],
                            .handle = args[0],
                            .coord = coord,
                            .dim = encode.TexDim.cube,
                            .scratch = scratch,
                            .is_shadow = true,
                            .dref = args[5],
                        });
                    } else if (is_array_shadow) {
                        if (args.len != 6) return error.Unsupported;
                        var coord: u8 = @intCast(@as(u32, max_reg.*) + 1);
                        coord = std.mem.alignForward(u8, coord, 4);
                        if (@as(u32, coord) + 6 - 1 >= encode.RZ) return error.Unsupported;
                        max_reg.* = coord + 5;
                        try self.calls.put(self.allocator, @intFromEnum(inst), .{
                            .out = args[0],
                            .u = args[1],
                            .v = args[2],
                            .w = args[3], // layer
                            .lod = args[4],
                            .handle = args[0],
                            .coord = coord,
                            .dim = encode.TexDim.array_2d,
                            .is_shadow = true,
                            .dref = args[5],
                        });
                    } else {
                        if (args.len != 5) return error.Unsupported;
                        // sampler2DShadow emits a 2D TEX reading a
                        // 2-register coordinate pair (u, v) at `coord`,
                        // plus a 2-register [handle, dref] pair at coord+2.
                        // Even-align `coord` so both power-of-2 vector
                        // operands are aligned. An odd base faults with
                        // Xid 13, "Misaligned Register". This path
                        // previously survived only when the watermark
                        // happened to leave coord even.
                        var coord: u8 = @intCast(@as(u32, max_reg.*) + 1);
                        coord = std.mem.alignForward(u8, coord, 2);
                        if (@as(u32, coord) + 4 - 1 >= encode.RZ) return error.Unsupported;
                        max_reg.* = coord + 3;
                        try self.calls.put(self.allocator, @intFromEnum(inst), .{
                            .out = args[0],
                            .u = args[1],
                            .v = args[2],
                            .w = args[1],
                            .lod = args[3],
                            .handle = args[0],
                            .coord = coord,
                            .dim = encode.TexDim.dim_2d,
                            .is_shadow = true,
                            .dref = args[4],
                        });
                    }
                    continue;
                }
                // 2D: {desc, u, v, lod, out}. Cube/3D: {desc, u, v, w, lod,
                // out}. Gather: {desc, u, v, comp, out}. Fetch: {desc, x, y,
                // lod, out}. Fetch3 (array/3D): {desc, x, y, z, lod, out}.
                if (is_2d and args.len != 5) return error.Unsupported;
                if (is_vec3 and args.len != 6) return error.Unsupported;
                if (is_gather and args.len != 5) return error.Unsupported;
                if (is_fetch and args.len != 5) return error.Unsupported;
                if (is_fetch3 and args.len != 6) return error.Unsupported;
                const has_w = is_vec3 or is_fetch3; // a 3-register coordinate (u,v,w / x,y,z)
                const out = if (has_w) args[5] else args[4];
                // A gather's comp is the fconst at args[3], rounded to a
                // 0..3 channel index.
                const gather_comp: ?u8 = if (is_gather) blk: {
                    const cf = fconst_of.get(args[3]) orelse return error.Unsupported;
                    const ci: i64 = @intFromFloat(@round(cf));
                    break :blk @intCast(std.math.clamp(ci, 0, 3));
                } else null;
                // A vertex-stage 2D sample with an explicit LOD: the lod arg
                // (args[3]) is a real value, not the implicit sentinel
                // (-1e30). A vertex shader has no derivatives, so it must
                // emit a TEX.LL over a quad coordinate (u, v, handle, lod)
                // rather than the auto-LOD TEX, which is undefined with no
                // quad neighbours. This is scoped to the vertex stage: a
                // fragment shader keeps the auto-LOD TEX for both implicit
                // sampling and textureLod, the pre-existing behavior, so the
                // deferred fragment-textureLod-level bug stays deferred. The
                // extra 2 coordinate registers per sample would otherwise
                // overflow a heavy fragment shader's register budget (the
                // glmark2 desktop blur shader has many taps) and fail the
                // compile.
                const is_explicit_2d = is_2d and !is_gather and stage == .vertex and blk: {
                    const lc = fconst_of.get(args[3]);
                    break :blk (lc == null) or (lc.? > -1.0e29);
                };
                // Allocate the 4-register RGBA result block above the
                // watermark. TEX does not require an even base, but this
                // keeps it tidy. One block per sampler call.
                const base: u8 = @intCast(@as(u32, max_reg.*) + 1);
                if (@as(u32, base) + 3 >= encode.RZ) return error.Unsupported;
                max_reg.* = base + 3;
                // A dedicated, reserved coordinate group for this TEX, never
                // the fixed R0/R1, which the allocator may have given to
                // live values. 2D uses a pair, 3D uses a triple. A cube
                // samples the atlas as 2D with an explicit LOD: it needs
                // coord, coord+1 = u', v, plus a consecutive (handle, lod)
                // pair at coord+2, coord+3 (NAK packs the explicit LOD as
                // src1[1], that is, handle_reg + 1), so it reserves a quad.
                const is_cube_call = is_vec3 and samplerVec3Dim(func, c.target) == encode.TexDim.cube;
                // A 2D fetch reserves a quad like the cube: coord, coord+1 =
                // x, y (integer), plus a consecutive (handle, lod) pair at
                // coord+2, coord+3 (the explicit LOD, Lod mode). A fetch3
                // (array/3D) reserves 6: three integer coordinates at
                // coord..coord+2 (4-aligned), a padding register at
                // coord+3, and the (handle, lod) pair at coord+4/coord+5.
                // That pair must be even-aligned, since it is a 2-register
                // Lod operand. coord is 4-aligned, so coord+4 is even
                // (coord+3 is odd).
                // An explicit-LOD 2D sample reserves a quad like the cube or
                // fetch: u, v at coord/coord+1, and the (handle, lod) pair
                // at coord+2/coord+3 (NAK packs the explicit LOD as src1[1]).
                const ncoord: u8 = if (is_fetch3) 6 else if (is_cube_call or is_fetch or is_explicit_2d) 4 else if (is_vec3) 3 else 2;
                // A genuine 3D TEX (dim_3d) reads a 3-register coordinate.
                // NAK allocates a 3-component tex-coordinate vector
                // 4-register-aligned (alloc_ssa_vec rounds the count up to a
                // power of two), and the hardware faults with Xid 13,
                // "Misaligned Register", if that base is only 2-aligned. 2D
                // (a coordinate pair) and the cube path (its final emitted
                // TEX is 2D) need only even alignment. A 3D or array fetch
                // also has a 3-register coordinate, so 4-align it too.
                const is_3d_sample = is_vec3 and !is_cube_call;
                var coord: u8 = @intCast(@as(u32, max_reg.*) + 1);
                if (is_3d_sample or is_fetch3) coord = std.mem.alignForward(u8, coord, 4);
                // The explicit-2D (handle, lod) pair at coord+2 is a
                // 2-register Lod operand: coord must be even so coord+2 is
                // even. An odd base faults with Xid 13, "Misaligned
                // Register".
                if (is_explicit_2d) coord = std.mem.alignForward(u8, coord, 2);
                if (@as(u32, coord) + ncoord - 1 >= encode.RZ) return error.Unsupported;
                max_reg.* = coord + ncoord - 1;
                const dim = if (is_vec3) samplerVec3Dim(func, c.target) else if (is_fetch3) fetch3Dim(func, c.target) else encode.TexDim.dim_2d;
                // Cube samples lower to major-axis math plus a 3D TEX.
                // Reserve a scratch block for it.
                var scratch: u8 = 0;
                if (dim == encode.TexDim.cube) {
                    scratch = @intCast(@as(u32, max_reg.*) + 1);
                    if (@as(u32, scratch) + cube_scratch_regs - 1 >= encode.RZ) return error.Unsupported;
                    max_reg.* = scratch + cube_scratch_regs - 1;
                }
                try self.out_base.put(self.allocator, out, base);
                try self.calls.put(self.allocator, @intFromEnum(inst), .{
                    .out = out,
                    .u = args[1],
                    .v = args[2],
                    .w = if (has_w) args[3] else args[1],
                    .lod = if (has_w) args[4] else args[3], // {desc,u,v,[w,]lod,out}
                    .handle = args[0],
                    .coord = coord,
                    .dim = dim,
                    .scratch = scratch,
                    .gather_comp = gather_comp,
                    .is_fetch = is_fetch or is_fetch3,
                    .explicit_lod = is_explicit_2d,
                });
                // Component 0 is the alloca itself.
                try self.elem.put(self.allocator, out, .{ .alloca = out, .comp = 0 });
            }
        }
        // A second pass records `alloca + c*4` element pointers as
        // components. The lowering builds these as
        // `arith add(out_ptr, iconst c*4)`. Match that shape.
        for (0..nblocks) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockInsts(block)) |inst| {
                if (func.opcode(inst) != .arith) continue;
                const a = func.opcode(inst).arith;
                if (a.op != .add) continue;
                if (!self.out_base.contains(a.lhs)) continue; // base must be a tex alloca
                const off = iconst_of.get(a.rhs) orelse continue;
                const result = func.instResult(inst) orelse continue;
                try self.elem.put(self.allocator, result, .{ .alloca = a.lhs, .comp = @intCast(@divTrunc(off, 4)) });
            }
        }
    }

    /// The (out alloca, component) a `load`'s pointer resolves to, if it is a
    /// reload of a sampled-texture result. Returns null for an ordinary
    /// memory load.
    fn loadComp(self: *const TexLowering, ptr: Value) ?Elem {
        return self.elem.get(ptr);
    }
};

/// Per-function screen-space-derivative lowering for the NVIDIA backend. The
/// SPIR-V frontend lowers OpDPdx/OpDPdy/OpFwidth of a varying scalar to a
/// load of a synthesized `grad_buf[index]`, a per-(varying-slot, axis)
/// gradient the software rasterizer fills. The GPU has no host gradient
/// buffer. The warp shades 2x2 pixel quads whose four lanes are co-resident,
/// so a derivative is computed natively by shuffling the varying from the
/// quad neighbour and differencing (NAK's nir_op_fddx/fddy, which is
/// SHFL.BFLY plus FSWZADD). This pass recognizes each grad_buf load,
/// recovers which varying attribute slot and axis it is from the ordered
/// `grad_slot` func attrs the frontend emitted, and records it, so lowerInst
/// emits IPA(slot) plus SHFL plus FSWZADD into the load's result register
/// instead of LDG.
const DerivLowering = struct {
    allocator: std.mem.Allocator,
    /// The grad_buf pointer entry param, tagged `vulcan.gpu.grad_buf`, or
    /// null if the function takes no derivatives. It is not a real memory
    /// buffer on the GPU, so the prologue must not load a constant-bank slot
    /// for it.
    grad_buf: ?Value = null,
    /// Per buffer index, the attribute byte slot and axis of the varying
    /// derivative, recovered from the `grad_slot` func attrs in append
    /// (index) order.
    descs: std.ArrayList(Desc) = .empty,
    /// Maps a pointer value that addresses `grad_buf[index]`, the grad_buf
    /// param itself for index 0, or `add(grad_buf, iconst index*4)`, to its
    /// buffer index. The address arithmetic for these is a tag carrier and
    /// is never emitted.
    grad_ptr: std.AutoHashMapUnmanaged(Value, u32) = .empty,
    /// Maps a `load` result value of a grad pointer to its slot and axis,
    /// plus the two scratch registers (the IPA'd varying, and the SHFL'd
    /// neighbour) the derivation uses.
    loads: std.AutoHashMapUnmanaged(Value, Load) = .empty,
    /// Maps a fragment varying attribute byte-slot to the register the
    /// prologue IPA'd it into. The derivative SHFLs this prologue value,
    /// which has long since landed in every lane, rather than doing a body
    /// re-IPA. A cross-lane SHFL cannot scoreboard-wait on a per-lane basis.
    prologue_reg: std.AutoHashMapUnmanaged(u16, u8) = .empty,

    const Desc = struct { slot: u16, axis: u1 }; // axis: 0 = x (dFdx), 1 = y (dFdy)
    const Load = struct { slot: u16, axis: u1, varying_reg: u8, shfl_reg: u8 };

    fn init(allocator: std.mem.Allocator) DerivLowering {
        return .{ .allocator = allocator };
    }
    fn deinit(self: *DerivLowering) void {
        self.descs.deinit(self.allocator);
        self.grad_ptr.deinit(self.allocator);
        self.loads.deinit(self.allocator);
        self.prologue_reg.deinit(self.allocator);
    }

    fn any(self: *const DerivLowering) bool {
        return self.grad_buf != null;
    }

    /// Find the grad_buf param and the slot/axis descriptor table, map each
    /// grad_buf load to its derivative, and reserve two scratch registers
    /// per load.
    fn scan(self: *DerivLowering, func: *const Function, max_reg: *u8) Error!void {
        // The grad_buf entry param (tagged on the entry block's parameters).
        for (func.blockParams(@enumFromInt(0))) |p| {
            if (hasGpuKey(func, p, "grad_buf")) {
                self.grad_buf = p;
                break;
            }
        }
        if (self.grad_buf == null) return; // no derivatives in this function

        // The slot and axis per buffer index, in the order the frontend
        // appended the `grad_slot` func attrs (one per index), packed as
        // (slot << 1 | axis).
        var it = func.attributesOf(.func);
        while (it.next()) |attr| switch (attr) {
            .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "grad_slot")) {
                const packed_val: i64 = switch (c.value) {
                    .int => |n| n,
                    else => return error.Unsupported,
                };
                try self.descs.append(self.allocator, .{
                    .slot = @intCast(@as(u64, @bitCast(packed_val)) >> 1),
                    .axis = @intCast(@as(u64, @bitCast(packed_val)) & 1),
                });
            },
            else => {},
        };

        const nblocks = func.blockCount();
        // Recover iconst values so a grad-pointer
        // `add(grad_buf, iconst index*4)` can be decoded to its buffer
        // index. This mirrors TexLowering's iconst table.
        var iconst_of = std.AutoHashMapUnmanaged(Value, i64){};
        defer iconst_of.deinit(self.allocator);
        for (0..nblocks) |bi| {
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (func.opcode(inst) == .iconst) {
                    if (func.instResult(inst)) |r| try iconst_of.put(self.allocator, r, func.opcode(inst).iconst);
                }
            }
        }
        // The grad_buf param itself addresses index 0.
        try self.grad_ptr.put(self.allocator, self.grad_buf.?, 0);
        // `add(grad_buf, iconst k)` addresses index k/4.
        for (0..nblocks) |bi| {
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (func.opcode(inst) != .arith) continue;
                const a = func.opcode(inst).arith;
                if (a.op != .add or a.lhs != self.grad_buf.?) continue;
                const off = iconst_of.get(a.rhs) orelse continue;
                const result = func.instResult(inst) orelse continue;
                try self.grad_ptr.put(self.allocator, result, @intCast(@divTrunc(off, 4)));
            }
        }
        // Each grad_buf load maps to its slot, axis, and two reserved scratch registers.
        for (0..nblocks) |bi| {
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (func.opcode(inst) != .load) continue;
                const l = func.opcode(inst).load;
                const index = self.grad_ptr.get(l.ptr) orelse continue;
                if (index >= self.descs.items.len) return error.Unsupported;
                const result = func.instResult(inst) orelse continue;
                const varying_reg: u8 = @intCast(@as(u32, max_reg.*) + 1);
                const shfl_reg: u8 = varying_reg + 1;
                if (@as(u32, shfl_reg) >= encode.RZ) return error.Unsupported;
                max_reg.* = shfl_reg;
                const d = self.descs.items[index];
                try self.loads.put(self.allocator, result, .{
                    .slot = d.slot,
                    .axis = d.axis,
                    .varying_reg = varying_reg,
                    .shfl_reg = shfl_reg,
                });
            }
        }
    }
};

/// Whether `v` is the synthesized host-math function-pointer entry param the
/// transcendental lowering appends, tagged `vulcan.gpu.math_fn`. The NVIDIA
/// backend ignores it: the special-function unit (MUFU) evaluates pow, exp,
/// log, sin, and cos natively, so the param gets no constant-bank slot, and
/// the math `call_indirect` through it lowers to MUFU.
fn isMathFn(func: *const Function, v: Value) bool {
    return hasGpuKey(func, v, "math_fn");
}

// The host-math op selector codes the SPIR-V transcendental lowering passes
// as the math_fn call's first argument (mirrors lower.zig MATH_*). The
// backend dispatches these to the MUFU special-function unit.
const MATH_POW: i64 = 0;
const MATH_EXP: i64 = 1;
const MATH_LOG: i64 = 2;
const MATH_EXP2: i64 = 3;
const MATH_LOG2: i64 = 4;
const MATH_SIN: i64 = 5;
const MATH_COS: i64 = 6;

/// Per-function host-math lowering state for the NVIDIA backend. This maps
/// each math_fn `call_indirect(op, a, b)` to its op code and scratch
/// register, so lowerInst emits the native MUFU sequence. pow, exp, and log
/// need a free scratch register for the intermediate value (MUFU.LG2, then
/// FMUL, then MUFU.EX2). The unary ops (exp2, log2, sin, cos) need none.
const MathLowering = struct {
    allocator: std.mem.Allocator,
    /// Maps a math_fn `call_indirect` Inst to the op code it carries plus a
    /// reserved scratch register.
    calls: std.AutoHashMapUnmanaged(u32, Call) = .empty,

    const Call = struct { op: i64, scratch: u8 };

    fn init(allocator: std.mem.Allocator) MathLowering {
        return .{ .allocator = allocator };
    }
    fn deinit(self: *MathLowering) void {
        self.calls.deinit(self.allocator);
    }

    /// Find every math_fn `call_indirect`, decode its op-code constant, the
    /// first arg, and reserve one scratch register per call for the
    /// intermediate value.
    fn scan(self: *MathLowering, func: *const Function, max_reg: *u8) Error!void {
        const nblocks = func.blockCount();
        // Recover iconst values so a call's op-code argument can be decoded.
        var iconst_of = std.AutoHashMapUnmanaged(Value, i64){};
        defer iconst_of.deinit(self.allocator);
        for (0..nblocks) |bi| {
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (func.opcode(inst) == .iconst) {
                    if (func.instResult(inst)) |r| try iconst_of.put(self.allocator, r, func.opcode(inst).iconst);
                }
            }
        }
        for (0..nblocks) |bi| {
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                if (func.opcode(inst) != .call_indirect) continue;
                const c = func.opcode(inst).call_indirect;
                if (!isMathFn(func, c.target)) continue;
                const args = func.valueList(c.args);
                if (args.len != 3) return error.Unsupported; // {op, a, b}
                const op = iconst_of.get(args[0]) orelse return error.Unsupported;
                const scratch: u8 = @intCast(@as(u32, max_reg.*) + 1);
                if (@as(u32, scratch) >= encode.RZ) return error.Unsupported;
                max_reg.* = scratch;
                try self.calls.put(self.allocator, @intFromEnum(inst), .{ .op = op, .scratch = scratch });
            }
        }
    }
};

/// Emit the samplerCube atlas major-axis lowering shared by the non-shadow
/// `samplerCube` sample and the `samplerCubeShadow` depth-compare sample.
/// From the direction (x = call.u, y = call.v, z = call.w), it computes the
/// GL cube (face, within-face u, v) branchlessly: the largest absolute
/// component picks the axis, and its sign picks the face. It writes the
/// 6-face-atlas coordinate u' = (face + u)/6 into `call.coord`, and v into
/// `call.coord + 1`. This is a 2D sample of the 6-face-wide atlas at column
/// [face/6, (face+1)/6). It uses the reserved 12-register scratch block
/// `call.scratch`. The caller then places the src1 operands (handle plus lod
/// for a plain sample, or handle plus dref for the z_cmpr shadow) and emits
/// the 2D TEX.
fn emitCubeAtlasUv(allocator: std.mem.Allocator, code: *std.ArrayList(Inst), loc: *std.AutoHashMapUnmanaged(Value, Loc), call: TexLowering.Call) Error!void {
    const x = gprOf(loc.*, call.u);
    const y = gprOf(loc.*, call.v);
    const z = gprOf(loc.*, call.w);
    const s = call.scratch;
    const F = struct {
        fn base(comptime face: comptime_int) u32 {
            return @bitCast(@as(f32, @as(f32, face) / 6.0));
        }
    }.base;
    const f_half: u32 = @bitCast(@as(f32, 0.5));
    const f_sixth: u32 = @bitCast(@as(f32, 1.0 / 6.0));
    // |x|, |y|, |z| go into s+1, s+3, s+5. Sign predicates P0/P1/P2 = (comp >= 0).
    try code.append(allocator, encode.fsub(s + 0, encode.RZ, x, .{})); // -x
    try code.append(allocator, encode.fsetp(0, x, encode.RZ, .ge, .{})); // P0 = x>=0
    try code.append(allocator, encode.sel(s + 1, x, s + 0, 0, .{})); // |x|
    try code.append(allocator, encode.fsub(s + 2, encode.RZ, y, .{}));
    try code.append(allocator, encode.fsetp(1, y, encode.RZ, .ge, .{})); // P1 = y>=0
    try code.append(allocator, encode.sel(s + 3, y, s + 2, 1, .{})); // |y|
    try code.append(allocator, encode.fsub(s + 4, encode.RZ, z, .{}));
    try code.append(allocator, encode.fsetp(2, z, encode.RZ, .ge, .{})); // P2 = z>=0
    try code.append(allocator, encode.sel(s + 5, z, s + 4, 2, .{})); // |z|
    // P3 = |x|>=|y|. maxxy = P3 ? |x| : |y|. P4 = maxxy >= |z|.
    try code.append(allocator, encode.fsetp(3, s + 1, s + 3, .ge, .{}));
    try code.append(allocator, encode.sel(s + 6, s + 1, s + 3, 3, .{})); // max(|x|,|y|)
    try code.append(allocator, encode.fsetp(4, s + 6, s + 5, .ge, .{})); // P4 = xy wins vs z
    // faceBase = face/6, picked by sign then major axis.
    try code.append(allocator, encode.movImm(s + 7, F(0), .{}));
    try code.append(allocator, encode.movImm(s + 8, F(1), .{}));
    try code.append(allocator, encode.sel(s + 7, s + 7, s + 8, 0, .{})); // baseX
    try code.append(allocator, encode.movImm(s + 8, F(2), .{}));
    try code.append(allocator, encode.movImm(s + 9, F(3), .{}));
    try code.append(allocator, encode.sel(s + 8, s + 8, s + 9, 1, .{})); // baseY
    try code.append(allocator, encode.movImm(s + 9, F(4), .{}));
    try code.append(allocator, encode.movImm(s + 10, F(5), .{}));
    try code.append(allocator, encode.sel(s + 9, s + 9, s + 10, 2, .{})); // baseZ
    try code.append(allocator, encode.sel(s + 7, s + 7, s + 8, 3, .{})); // baseXY
    try code.append(allocator, encode.sel(s + 7, s + 7, s + 9, 4, .{})); // faceBase -> s+7
    // Within-face u, v (GL convention, matching software cubeFaceUv): ma is
    // the winning absolute axis value. sc/tc are the two other coordinates
    // (signed per face). u=(sc/ma+1)/2, v=(tc/ma+1)/2. negx/negy/negz are
    // still live in s+0/s+2/s+4 from the abs step above.
    try code.append(allocator, encode.sel(s + 8, s + 6, s + 5, 4, .{})); // ma = P4? maxxy : |z|
    try code.append(allocator, encode.sel(s + 9, s + 4, z, 0, .{})); // scX = P0? -z : z
    try code.append(allocator, encode.sel(s + 10, x, s + 0, 2, .{})); // scZ = P2? x : -x
    try code.append(allocator, encode.sel(s + 9, s + 9, x, 3, .{})); // scXY = P3? scX : x
    try code.append(allocator, encode.sel(s + 9, s + 9, s + 10, 4, .{})); // sc -> s+9
    try code.append(allocator, encode.sel(s + 10, z, s + 4, 1, .{})); // tcY = P1? z : -z
    try code.append(allocator, encode.sel(s + 10, s + 2, s + 10, 3, .{})); // tcXY = P3? -y : tcY
    try code.append(allocator, encode.sel(s + 10, s + 10, s + 2, 4, .{})); // tc -> s+10
    try code.append(allocator, encode.mufu(s + 11, s + 8, .rcp, .{})); // 1/ma
    try code.append(allocator, encode.movImm(s + 1, f_half, .{})); // 0.5
    try code.append(allocator, encode.movImm(s + 2, f_sixth, .{})); // 1/6
    // u_within = sc/ma*0.5 + 0.5. Then u' = u_within*(1/6) + faceBase.
    try code.append(allocator, encode.fmul(s + 3, s + 9, s + 11, .{})); // sc/ma
    try code.append(allocator, encode.ffma(s + 9, s + 3, s + 1, s + 1, .{})); // u_within
    // Clamp u_within to [half_texel, 1-half_texel], where half_texel =
    // 0.5/face_w, from CB0. This keeps a linear tap near a face edge inside
    // this face's atlas column (a per-face clamp-to-edge), instead of
    // bleeding into the neighbour. Reuses P0/P1, since the sign predicates
    // are dead here.
    try code.append(allocator, encode.ldc(s + 3, encode.graphics_const_bank, encode.cube_halftexel_cb, .{})); // half_texel (root table 1)
    try code.append(allocator, encode.fsetp(0, s + 9, s + 3, .ge, .{})); // u_within >= ht
    try code.append(allocator, encode.sel(s + 9, s + 9, s + 3, 0, .{})); // max(u_within, ht)
    try code.append(allocator, encode.movImm(s + 4, @bitCast(@as(f32, 1.0)), .{}));
    try code.append(allocator, encode.fsub(s + 4, s + 4, s + 3, .{})); // 1 - ht
    try code.append(allocator, encode.fsetp(1, s + 9, s + 4, .le, .{})); // u_within <= 1-ht
    try code.append(allocator, encode.sel(s + 9, s + 9, s + 4, 1, .{})); // min(.., 1-ht)
    try code.append(allocator, encode.ffma(call.coord, s + 9, s + 2, s + 7, .{})); // u'
    // v = tc/ma*0.5 + 0.5.
    try code.append(allocator, encode.fmul(s + 3, s + 10, s + 11, .{})); // tc/ma
    try code.append(allocator, encode.ffma(call.coord + 1, s + 3, s + 1, s + 1, .{})); // v
}

fn lowerInst(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), tex: *const TexLowering, deriv: *const DerivLowering, math: *const MathLowering, shared: *const gpu.abi.SharedFrame, disp: *const DispFold, fma: *const FmaFold, inst: ir.function.Inst) Error!void {
    switch (func.opcode(inst)) {
        .iconst => |c| {
            // A graphics output-attribute store pointer is a tag-carrier
            // iconst (the slot), never a real value the SASS computes. Skip
            // emitting it.
            const result = func.instResult(inst).?;
            if (attrTag(func, result, "out_attr") != null or attrTag(func, result, "color_out") != null or attrTag(func, result, "frag_depth") != null) return;
            const rd = gprOf(loc.*, result);
            try code.append(allocator, encode.movImm(rd, @truncate(@as(u64, @bitCast(c))), .{}));
        },
        .fconst => |val| {
            const rd = gprOf(loc.*, func.instResult(inst).?);
            const bits: u32 = @bitCast(@as(f32, @floatCast(val)));
            try code.append(allocator, encode.movImm(rd, bits, .{}));
        },
        .arith => |a| {
            const result = func.instResult(inst).?;
            // A texture-result element pointer (`tex_alloca + c*4`) is a tag
            // carrier: the reload load resolves straight to a TEX result
            // register, so the address arithmetic is never emitted.
            if (tex.elem.contains(result)) return;
            // A grad_buf element pointer (`grad_buf + index*4`) is likewise
            // a tag carrier: the load is replaced by the SHFL-quad
            // derivative, so its address arith is never emitted.
            if (deriv.grad_ptr.contains(result)) return;
            // A multiply the add below absorbed emits NOTHING here, and the add emits one
            // FFMA or IMAD that multiplies and adds together. See `FmaFold`.
            if (fma.isFolded(result)) return;
            if (fma.at.get(result)) |m| {
                try code.append(allocator, fmaInst(loc.*, gprOf(loc.*, result), m));
                return;
            }
            if (isWidePtr(func, result) and a.op == .add) {
                // 64-bit pointer add: (dst:dst+1) = (base:base+1) + zext(offset).
                //
                // ONE IMAD.WIDE.U32 DOES ALL OF IT. The unsigned wide multiply-add
                // computes `zext(offset) * 1 + (base:base+1)` as one 64-bit sum, which is
                // exactly what the pair of adds computed: the low half, the carry out of
                // it, and the high half. It replaces an IADD3 with a carry-out predicate
                // plus an IADD3.X that reads it, so it also frees the carry predicate for
                // the whole span the two adds used to hold it.
                //
                // The scale is 1 and not the element size on purpose. Folding the element
                // size in would make the product a FULL 64-bit `index * size`, while the
                // IR says the byte offset is computed in 32 bits and then widened, and the
                // two differ whenever that 32-bit offset would wrap. See the report note
                // on what folding the scale would need.
                const dlo = gprOf(loc.*, result);
                const base = gprOf(loc.*, a.lhs); // pointer pair (lo:hi)
                const offset = gprOf(loc.*, a.rhs); // 32-bit, zero-extended
                try code.append(allocator, encode.imadWideImm(dlo, offset, 1, base, false, .{}));
            } else if (isBool(func, result)) {
                // A boolean-valued bitwise op is a logical predicate combine
                // (`a && b`, `a || b`, `a ^^ b`). The shared lowering emits
                // SPIR-V LogicalAnd/Or/NotEqual as a bool-typed `.binary`
                // bit_and/bit_or/bit_xor. The result lives in a predicate
                // register, since the allocator gives bools predicates, so
                // it combines the operand predicates with PLOP3, not GPR
                // LOP3. The glmark2 light-phong fragment shader hits this: a
                // comparison ANDed or ORed into another bool.
                const pd = predOf(loc.*, result);
                const pa = predOf(loc.*, a.lhs);
                const pb = predOf(loc.*, a.rhs);
                try code.append(allocator, encode.plop3(pd, pa, pb, try lutOf(a.op), .{}));
            } else if (a.op == .div and isFloat(func, a.lhs)) {
                // Float divide a/b = a * (1/b): the GPU has no FDIV, so
                // reciprocate b on the multifunction unit (MUFU.RCP), then
                // multiply. `normalize` uses this path for its 1.0/sqrt(dot)
                // reciprocal. The RCP result lands in a scratch register.
                // MUFU has fixed latency, so the default stall covers the
                // FMUL dependency.
                const rd = gprOf(loc.*, result);
                const ra = gprOf(loc.*, a.lhs);
                const rb = gprOf(loc.*, a.rhs);
                try code.append(allocator, encode.mufu(r_scratch, rb, .rcp, .{}));
                try code.append(allocator, encode.fmul(rd, ra, r_scratch, .{}));
            } else {
                const rd = gprOf(loc.*, result);
                const ra = gprOf(loc.*, a.lhs);
                const rb = gprOf(loc.*, a.rhs);
                try code.append(allocator, try arith(func, a.op, rd, ra, rb, a.lhs));
            }
        },
        .unary => |u| {
            // Float transcendentals on the multifunction unit. `sqrt`, used
            // by `length` and `normalize`'s sqrt(dot), maps to MUFU.SQRT.
            // `reinterpret` is a bitcast, so it is a register copy. The
            // rounding ops (floor, ceil, trunc) have no direct F32-to-F32
            // instruction here, so they go through F2I, with the matching
            // round mode, then back through I2F. This is exact for any
            // integer-representable value, which covers the GLSL floor(),
            // ceil(), and trunc() the simplex-noise and terrain shaders
            // need. `nearest` is not modeled.
            const rd = gprOf(loc.*, func.instResult(inst).?);
            const rs = gprOf(loc.*, u.value);
            switch (u.op) {
                .sqrt => try code.append(allocator, encode.mufu(rd, rs, .sqrt, .{})),
                .reinterpret => try code.append(allocator, encode.movReg(rd, rs, .{})),
                .floor, .ceil, .trunc => {
                    const mode: encode.F2IRound = switch (u.op) {
                        .floor => .floor,
                        .ceil => .ceil,
                        else => .zero, // trunc
                    };
                    try code.append(allocator, encode.f2iRound(rd, rs, true, mode, .{})); // rd = round(x) as i32
                    try code.append(allocator, encode.i2f(rd, rd, true, .{})); // rd = (f32) that int
                },
                else => return error.Unsupported,
            }
        },
        .load => |l| {
            const rd = gprOf(loc.*, func.instResult(inst).?);
            // A reload of a sampled-texture result: the four loads of the
            // host-sampler out-pointer resolve to the TEX result block
            // (out_base + component), so the load is a register copy, not
            // an LDG. The TEX's write barrier, set when the sampler call
            // lowered, already gates the read through the scheduler.
            if (tex.loadComp(l.ptr)) |e| {
                const src = tex.out_base.get(e.alloca).? + e.comp;
                try code.append(allocator, encode.movReg(rd, src, .{}));
                return;
            }
            // A grad_buf load: the screen-space derivative of a varying.
            // Interpolate the varying into a scratch register (IPA at its
            // attribute slot), shuffle the varying from the quad neighbour
            // (XOR the lane index with 1 for dFdx/horizontal, 2 for
            // dFdy/vertical), then use FSWZADD to difference them with the
            // correct per-lane sign. The coarse per-quad gradient lands in
            // `rd`. The varying IPA is variable-latency: the scheduler
            // waits for the SHFL, which reads it at srcA, on its scoreboard
            // automatically.
            if (deriv.loads.get(func.instResult(inst).?)) |gl| {
                const scratch = gl.shfl_reg;
                // Source the varying for the quad SHFL with a fresh IPA into
                // a private reserved register. The prologue-reused register
                // is not safe here: the body reuses it for fragment-shader
                // values between the prologue interpolation and this grad
                // load, so by the time the SHFL/FSWZADD read it, `self` is a
                // different, much larger, value than the actual varying. The
                // derivative then comes out saturated. Proven on the GPU: a
                // fresh-IPA SHFL gives the exact 1-pixel-neighbour step and
                // the correct dFdx, while the prologue-reg path gave a
                // 2x-or-more saturated dFdx. A fresh IPA is
                // variable-latency, but the scheduler scoreboards both the
                // IPA and the SHFL (both in isVariableLatency), so the
                // cross-lane read waits until the value has landed in every
                // lane.
                try code.append(allocator, encode.ipa(gl.varying_reg, gl.slot, .{}));
                const vary: u8 = gl.varying_reg;
                const lane_xor: u5 = if (gl.axis == 0) 1 else 2;
                try code.append(allocator, encode.shflBflyQuad(scratch, vary, lane_xor, .{}));
                // dFdx: [SubLeft, SubRight, SubLeft, SubRight], NAK's fddx
                // pattern. dFdy: [SubLeft, SubLeft, SubRight, SubRight],
                // NAK's fddy pattern. This is the raw NAK orientation.
                // prism's nvidia viewport now uses a positive Y scale
                // (setViewport: window_y = (ndc_y+1)/2*h, NDC y=-1 maps to
                // row 0), the same framebuffer Y-origin as the software
                // driver, so the on-screen vertical quad neighbour matches
                // NAK's assumed quad orientation. No sign compensation is
                // needed. The earlier `[SubRight,SubRight,SubLeft,SubLeft]`
                // negation existed only to cancel the old negative-Y-scale
                // flip, and was reverted along with restoring the positive Y
                // scale.
                const ops: [4]encode.SwzOp = if (gl.axis == 0)
                    .{ .sub_left, .sub_right, .sub_left, .sub_right }
                else
                    .{ .sub_left, .sub_left, .sub_right, .sub_right };
                try code.append(allocator, encode.fswzadd(rd, scratch, vary, ops, .{}));
                return;
            }
            // The loaded value's own type decides how many bytes the access
            // moves. Without it every load was a 32-bit one, which over-read a
            // byte or a half word and truncated anything wider.
            const width = try memTypeOf(func, func.instResult(inst).?);
            // A load through a SHARED pointer reads the CTA's shared-memory
            // window, which is a different memory and a different
            // instruction. Its address is a 32-bit offset in ONE register,
            // so there is no pointer pair to read.
            if (isSharedPtr(func, l.ptr)) {
                try code.append(allocator, encode.ldsAt(rd, gprOf(loc.*, l.ptr), disp.offsetOf(inst), width, .{}));
                return;
            }
            // Otherwise this is an ordinary LDG from the 64-bit pointer pair
            // into the result register. This is variable latency: the
            // scoreboard scheduler assigns its write barrier and the wait on
            // each consumer.
            try code.append(allocator, encode.ldgAt(rd, gprOf(loc.*, l.ptr), disp.offsetOf(inst), width, .{}));
        },
        .store => |st| {
            // A store whose pointer is tagged with a graphics output
            // attribute goes to that attribute (AST). A fragment color
            // output is moved into the ROP color register (R0..R3), and
            // otherwise it is an ordinary global store.
            if (attrTag(func, st.ptr, "out_attr")) |attr| {
                try code.append(allocator, encode.ast(attr, gprOf(loc.*, st.value), 1, .{}));
            } else if (attrTag(func, st.ptr, "frag_depth") != null) {
                // gl_FragDepth: move the shader-computed depth into the ROP
                // depth-output register, reserved in assignLocs past all N
                // color targets. The SPH's OMAP_DEPTH makes the ROP read
                // the fragment depth from here instead of the interpolated
                // z value.
                try code.append(allocator, encode.movReg(fragDepthReg(func), gprOf(loc.*, st.value), .{}));
            } else if (attrTag(func, st.ptr, "color_out")) |comp| {
                // The fragment shader's render-target color: the ROP reads
                // target T's RGBA from R[T*4 .. T*4+3] at EXIT, so `comp`
                // (= target*4 + component) moves into R<comp>. R0..R3 (RT0)
                // are always reserved. R4..R[4N-1] (RT1+) are reserved in
                // assignLocs when the shader is MRT. The register allocator
                // extends every color value's live range to the last color
                // store, so the color values occupy distinct registers that
                // all stay live to EXIT. The source register read here is
                // never reused for another value before its move. The
                // prologue pad already covers the async input-delivery
                // window.
                if (comp < 32) try code.append(allocator, encode.movReg(@intCast(comp), gprOf(loc.*, st.value), .{}));
            } else if (isSharedPtr(func, st.ptr)) {
                // A store through a SHARED pointer writes the CTA's
                // shared-memory window. Its address is a 32-bit offset in ONE
                // register, so there is no pointer pair to read. The STORED
                // VALUE's type decides the width: a 32-bit store of a byte
                // value destroys the three bytes beside it.
                const width = try memTypeOf(func, st.value);
                try code.append(allocator, encode.stsAt(gprOf(loc.*, st.ptr), gprOf(loc.*, st.value), disp.offsetOf(inst), width, .{}));
            } else {
                const width = try memTypeOf(func, st.value);
                try code.append(allocator, encode.stgAt(gprOf(loc.*, st.ptr), gprOf(loc.*, st.value), disp.offsetOf(inst), width, .{}));
            }
        },
        .prefetch => {}, // a hint; this GPU target has no CPU-style prefetch, so it is dropped
        .arith_imm => |a| {
            const result = func.instResult(inst).?;
            // An address whose constant moved into the load or store that read it. The
            // sum has no reader left, so computing it would waste an instruction and a
            // register. See `foldAddressDisplacements`.
            if (disp.isDead(inst)) return;
            // A constant-scale multiply that a fused multiply-add absorbed. The add reads
            // the same constant out of the FFMA or IMAD immediate field. See `FmaFold`.
            if (fma.isFolded(result)) return;
            // Logical NOT lowers to `bool ^ -1` (bit_xor against all-ones).
            // A boolean result lives in a predicate, so negate the source
            // predicate with PLOP3 (`p ^ PT` = `!p`, since PT is true). The
            // GPR LOP3 path cannot touch it.
            if (isBool(func, result)) {
                std.debug.assert(a.op == .bit_xor); // the only bool arith_imm the lowering emits
                const pd = predOf(loc.*, result);
                const pa = predOf(loc.*, a.lhs);
                try code.append(allocator, encode.plop3(pd, pa, encode.PT, encode.LUT_XOR, .{}));
            } else if (isWidePtr(func, result) and a.op == .add) {
                // A 64-bit pointer plus a CONSTANT byte offset, the same carry chain the
                // register form uses: (dst:dst+1) = (base:base+1) + zext(imm).
                //
                // Before this arm the constant add fell to the 32-bit path below, which
                // wrote the LOW half of the pair and LEFT THE HIGH HALF UNWRITTEN. The
                // access that read the pair then took whatever the high register held.
                // Every kernel that reached a constant array index through a global
                // pointer had that shape. Most such adds now fold into the access itself,
                // so this arm covers the ones that do not: a displacement too wide for the
                // 24-bit field, or an address a second instruction still reads.
                //
                // THE CONSTANT IS SIGN EXTENDED, not zero extended. A negative byte offset
                // is an ordinary thing to write, and its high word is 0xFFFFFFFF: added as
                // zero it would move the address 4 GiB up instead of a few bytes down. The
                // REGISTER form beside this one zero-extends, and that is right for it,
                // because the offset there is a 32-bit IR value and not a signed constant.
                const dlo = gprOf(loc.*, result);
                const base = gprOf(loc.*, a.lhs);
                const wide: i64 = a.imm;
                const lo_bits: u32 = @truncate(@as(u64, @bitCast(wide)));
                const hi_bits: u32 = @truncate(@as(u64, @bitCast(wide)) >> 32);
                try code.append(allocator, encode.iadd3CarryOutImm(dlo, base, lo_bits, carry_pred, .{}));
                try code.append(allocator, encode.iadd3CarryInImm(dlo + 1, base + 1, hi_bits, carry_pred, .{}));
            } else {
                const rd = gprOf(loc.*, result);
                const ra = gprOf(loc.*, a.lhs);
                const bits: u32 = @truncate(@as(u64, @bitCast(a.imm)));
                try code.append(allocator, try arithImm(func, a.op, rd, ra, bits, a.lhs));
            }
        },
        .icmp => |cmp| {
            const pd = predOf(loc.*, func.instResult(inst).?);
            // A compare of float operands must read them as IEEE floats
            // (FSETP), not as integer bit patterns (ISETP). The shared
            // lowering emits `.icmp` for the GLSL float min, max, and clamp
            // ordered compares too (lower.zig f_max maps to icmp.gt). An
            // integer compare mis-orders negative floats, so for example
            // `max(0.0, dot)` would return 0 for a positive dot. This is
            // what made vkcube's lighting go black. This mirrors the
            // software backend, which already picks a float compare for
            // float operands.
            if (isFloat(func, cmp.lhs)) {
                try code.append(allocator, encode.fsetp(pd, gprOf(loc.*, cmp.lhs), gprOf(loc.*, cmp.rhs), cmpOf(cmp.op), .{}));
            } else {
                try code.append(allocator, encode.isetp(pd, gprOf(loc.*, cmp.lhs), gprOf(loc.*, cmp.rhs), cmpOf(cmp.op), isSigned(func, cmp.lhs), .{}));
            }
        },
        .select => |s| {
            const rd = gprOf(loc.*, func.instResult(inst).?);
            try code.append(allocator, encode.sel(rd, gprOf(loc.*, s.then), gprOf(loc.*, s.@"else"), predOf(loc.*, s.cond), .{}));
        },
        .convert => |cv| {
            const result = func.instResult(inst).?;
            const rd = gprOf(loc.*, result);
            const rs = gprOf(loc.*, cv.value);
            const dst_float = isFloat(func, result);
            const src_float = isFloat(func, cv.value);
            if (src_float and !dst_float) {
                try code.append(allocator, encode.f2i(rd, rs, isSignedRaw(func, result), .{})); // f32 -> i32
            } else if (!src_float and dst_float) {
                try code.append(allocator, encode.i2f(rd, rs, isSignedRaw(func, cv.value), .{})); // i32 -> f32
            } else {
                return error.Unsupported; // int-to-int width change, or f32-to-f64, is not modeled yet
            }
        },
        .alloca => {
            const result = func.instResult(inst).?;
            // A WORKGROUP-SHARED alloca is a slot in the CTA's shared window, and its address
            // is that slot's byte offset. The offset is a 32-bit constant in ONE register,
            // which LDS and STS consume directly, so the whole lowering is a MOV of the
            // immediate the shared frame assigned. There is no pointer pair and no window base
            // to add, because the hardware addresses the window from zero.
            if (shared.offsetOf(result)) |off| {
                try code.append(allocator, encode.movImm(gprOf(loc.*, result), off, .{}));
                return;
            }
            // The only other alloca the NVIDIA backend supports is the
            // host-sampler out-pointer (a vec4 RGBA result slot), which is
            // materialized as a 4-register TEX result block, with no real
            // stack. Any other alloca is unsupported.
            if (!tex.out_base.contains(result)) return error.Unsupported;
        },
        .call_indirect => |c| {
            // The only indirect call the NVIDIA backend supports is the
            // host-sampler call the SPIR-V image-sample lowering emits:
            // lower it to a GPU TEX. The descriptor arg is the bindless
            // handle (tic | tsc<<20) the prologue loaded from the constant
            // bank. The (u, v) coordinate must occupy a consecutive
            // register pair, since TEX reads the pair from one source, so
            // move them into the R0:R1 scratch pair. The RGBA result lands
            // in the alloca's TEX result block.
            // A discard call (OpKill): emit a KIL, which masks the fragment
            // so the ROP does not write it. Execution continues to EXIT,
            // since the surrounding structured control flow already gates a
            // conditional `if (cond) discard`.
            if (hasGpuKey(func, c.target, "discard_fn")) {
                try code.append(allocator, encode.kil(.{}));
                return;
            }
            // A host-math call (pow, exp, log, sin, cos): evaluate it on the
            // MUFU special-function unit instead of a host function. The
            // lowering passes (op:i32, a:f32, b:f32). The unary ops (exp2,
            // log2, sin, cos) ignore b. pow, exp, and log compose two MUFUs
            // around an FMUL through a reserved scratch register.
            if (isMathFn(func, c.target)) {
                const m = math.calls.get(@intFromEnum(inst)) orelse return error.Unsupported;
                const args = func.valueList(c.args);
                const rd = gprOf(loc.*, func.instResult(inst).?);
                const a = gprOf(loc.*, args[1]); // primary operand
                switch (m.op) {
                    MATH_EXP2 => try code.append(allocator, encode.mufu(rd, a, .exp2, .{})),
                    MATH_LOG2 => try code.append(allocator, encode.mufu(rd, a, .log2, .{})),
                    MATH_SIN => try code.append(allocator, encode.mufu(rd, a, .sin, .{})),
                    MATH_COS => try code.append(allocator, encode.mufu(rd, a, .cos, .{})),
                    // pow(a, b) = exp2(b * log2(a)).
                    MATH_POW => {
                        const b = gprOf(loc.*, args[2]); // the exponent
                        try code.append(allocator, encode.mufu(m.scratch, a, .log2, .{}));
                        try code.append(allocator, encode.fmul(m.scratch, m.scratch, b, .{}));
                        try code.append(allocator, encode.mufu(rd, m.scratch, .exp2, .{}));
                    },
                    // exp(a) = exp2(a * log2(e)). log2(e) = 1.4426950408889634.
                    MATH_EXP => {
                        try code.append(allocator, encode.movImm(m.scratch, @as(u32, @bitCast(@as(f32, 1.4426950408889634))), .{}));
                        try code.append(allocator, encode.fmul(m.scratch, a, m.scratch, .{}));
                        try code.append(allocator, encode.mufu(rd, m.scratch, .exp2, .{}));
                    },
                    // log(a) = log2(a) * ln(2). ln(2) = 0.6931471805599453.
                    MATH_LOG => {
                        try code.append(allocator, encode.mufu(m.scratch, a, .log2, .{}));
                        try code.append(allocator, encode.movImm(rd, @as(u32, @bitCast(@as(f32, 0.6931471805599453))), .{}));
                        try code.append(allocator, encode.fmul(rd, m.scratch, rd, .{}));
                    },
                    else => return error.Unsupported,
                }
                return;
            }
            if (!isSamplerFn(func, c.target) and !isSamplerVec3Fn(func, c.target) and !isAnyShadowFn(func, c.target) and !isSamplerGatherFn(func, c.target) and !isSamplerFetchFn(func, c.target) and !isSamplerFetch3Fn(func, c.target)) return error.Unsupported;
            const call = tex.calls.get(@intFromEnum(inst)) orelse return error.Unsupported;
            // sampler2DShadow: a depth-compare TEX (z_cmpr) that returns a
            // scalar into the call's SSA result register, with no out
            // block. Build src0 = (u, v) and src1 = [handle, dref] in the
            // reserved coordinate quad, then emit texShadow (bit 78). The
            // scheduler gives the single-register result a write barrier,
            // from the R-only channel mask span, so consumers wait
            // correctly.
            if (call.is_shadow) {
                const rdst = gprOf(loc.*, func.instResult(inst).?);
                const handle = gprOf(loc.*, call.handle);
                const dref = gprOf(loc.*, call.dref);
                if (call.dim == encode.TexDim.cube) {
                    // samplerCubeShadow: run the samplerCube atlas major-axis
                    // lowering to get (u', v) into coord/coord+1, then do a
                    // 2D z_cmpr TEX over the 6-face ZF32 atlas. src1 =
                    // [handle, dref] at coord+2/coord+3. z_cmpr reads the
                    // dref from handle_reg+1 (NAK: src1 = [tex_h, z_cmpr],
                    // with no explicit lod). LOD is implicit (Auto), giving
                    // a base-level shadow cube.
                    try emitCubeAtlasUv(allocator, code, loc, call);
                    try code.append(allocator, encode.movReg(call.coord + 2, handle, .{})); // src1[0] = handle
                    try code.append(allocator, encode.movReg(call.coord + 3, dref, .{})); // src1[1] = dref
                    try code.append(allocator, encode.texShadow(rdst, call.coord, call.coord + 2, encode.TexDim.dim_2d, .{ .wr_barrier = 0 }));
                    return;
                }
                if (call.dim == encode.TexDim.array_2d) {
                    // sampler2DArrayShadow: native TWO_D_ARRAY z_cmpr TEX.
                    // NAK assembles src0 = [arr_idx, coords...], with the
                    // layer first. The hardware layer index is an integer,
                    // f2u(layer+0.5), like the non-shadow array path. The
                    // 3-register coordinate (layer, u, v) is 4-aligned. The
                    // (handle, dref) pair goes at coord+4/coord+5 (coord+3
                    // is padding), with src1 base = coord+4.
                    const layer = gprOf(loc.*, call.w);
                    try code.append(allocator, encode.movImm(call.coord + 1, @bitCast(@as(f32, 0.5)), .{}));
                    try code.append(allocator, encode.fadd(call.coord, layer, call.coord + 1, .{}));
                    try code.append(allocator, encode.f2iRound(call.coord, call.coord, false, .zero, .{})); // (u32)floor(layer+0.5)
                    try code.append(allocator, encode.movReg(call.coord + 1, gprOf(loc.*, call.u), .{})); // u
                    try code.append(allocator, encode.movReg(call.coord + 2, gprOf(loc.*, call.v), .{})); // v
                    try code.append(allocator, encode.movReg(call.coord + 4, handle, .{})); // src1[0] = handle
                    try code.append(allocator, encode.movReg(call.coord + 5, dref, .{})); // src1[1] = dref
                    try code.append(allocator, encode.texShadow(rdst, call.coord, call.coord + 4, encode.TexDim.array_2d, .{ .wr_barrier = 0 }));
                    return;
                }
                // sampler2DShadow: coordinate pair (u, v) plus
                // src1 = [handle, dref] at coord+2/coord+3.
                try code.append(allocator, encode.movReg(call.coord, gprOf(loc.*, call.u), .{})); // u
                try code.append(allocator, encode.movReg(call.coord + 1, gprOf(loc.*, call.v), .{})); // v
                try code.append(allocator, encode.movReg(call.coord + 2, handle, .{})); // src1[0] = handle
                try code.append(allocator, encode.movReg(call.coord + 3, dref, .{})); // src1[1] = dref
                try code.append(allocator, encode.texShadow(rdst, call.coord, call.coord + 2, encode.TexDim.dim_2d, .{ .wr_barrier = 0 }));
                return;
            }
            const dst = tex.out_base.get(call.out).?;
            const handle = gprOf(loc.*, call.handle);
            const u = gprOf(loc.*, call.u);
            const v = gprOf(loc.*, call.v);
            // Build the coordinate group in the reserved registers
            // (coord = u, coord+1 = v, and for a cube or 3D sample
            // coord+2 = w). Never use the fixed R0/R1, which the allocator
            // may have given to live SSA values a texture-and-derivative
            // shader still needs after the sample.
            var tex_dim = call.dim;
            var cube_lod = false; // cube samples carry an explicit LOD in coord+2 (emit TLD)
            var fetch_src1: u8 = call.coord + 2; // the TLD (handle, lod) pair base (2D fetch)
            if (call.is_fetch) {
                // texelFetch: integer coordinates plus a consecutive
                // (handle, lod) pair, the explicit LOD, Lod mode, read from
                // handle_reg+1. This emits a TLD, an integer fetch with no
                // filter. See encode.tld. 2D = (x, y). 3D = (x, y, z) =
                // (u, v, w). 2D-array = (LAYER, x, y) = (w, u, v), since NAK
                // packs the array index first. The (handle, lod) pair
                // follows the spatial coordinates.
                const w = gprOf(loc.*, call.w);
                const lodr = gprOf(loc.*, call.lod);
                if (call.dim == encode.TexDim.dim_3d) {
                    // The 3-register coordinate is 4-aligned. The
                    // (handle, lod) pair goes at coord+4 (even). coord+3 is
                    // padding. An odd base would fault with "Misaligned
                    // Register".
                    try code.append(allocator, encode.movReg(call.coord, u, .{})); // x
                    try code.append(allocator, encode.movReg(call.coord + 1, v, .{})); // y
                    try code.append(allocator, encode.movReg(call.coord + 2, w, .{})); // z
                    try code.append(allocator, encode.movReg(call.coord + 4, handle, .{}));
                    try code.append(allocator, encode.movReg(call.coord + 5, lodr, .{}));
                    fetch_src1 = call.coord + 4;
                } else if (call.dim == encode.TexDim.array_2d) {
                    try code.append(allocator, encode.movReg(call.coord, w, .{})); // layer FIRST
                    try code.append(allocator, encode.movReg(call.coord + 1, u, .{})); // x
                    try code.append(allocator, encode.movReg(call.coord + 2, v, .{})); // y
                    try code.append(allocator, encode.movReg(call.coord + 4, handle, .{}));
                    try code.append(allocator, encode.movReg(call.coord + 5, lodr, .{}));
                    fetch_src1 = call.coord + 4;
                } else {
                    try code.append(allocator, encode.movReg(call.coord, u, .{})); // x
                    try code.append(allocator, encode.movReg(call.coord + 1, v, .{})); // y
                    try code.append(allocator, encode.movReg(call.coord + 2, handle, .{}));
                    try code.append(allocator, encode.movReg(call.coord + 3, lodr, .{}));
                    fetch_src1 = call.coord + 2;
                }
            } else if (call.dim == encode.TexDim.cube) {
                // Cube is lowered to a 2D sample of a 6-face-wide atlas: the
                // 6 faces stored side by side, with face f in the x-column
                // block [f/6, (f+1)/6). The native cube TEX does not select
                // the face on this Blackwell path, and the 3D
                // within-slice u, v addressing is unreliable. So this
                // computes the GL cube (face, u, v) from (x, y, z) here and
                // samples the proven 2D path at (u' = (face + u)/6, v). The
                // branchless major-axis lowering is shared with
                // samplerCubeShadow through emitCubeAtlasUv, which writes
                // u' to coord and v to coord+1.
                try emitCubeAtlasUv(allocator, code, loc, call);
                // Explicit LOD: NAK packs the sample's src1 as [handle,
                // lod], so the hardware reads the LOD from handle_reg + 1.
                // Build that consecutive pair in coord+2 (handle copy) and
                // coord+3 (lod), and point the TEX's src1 at coord+2.
                // textureCube passes lod 0, and textureCubeLod passes its
                // explicit level. Explicit LOD, not implicit, matches the
                // software path and avoids the atlas face-boundary
                // derivative seam.
                try code.append(allocator, encode.movReg(call.coord + 2, handle, .{})); // handle copy
                try code.append(allocator, encode.movReg(call.coord + 3, gprOf(loc.*, call.lod), .{})); // lod
                tex_dim = encode.TexDim.dim_2d;
                cube_lod = true;
            } else if (call.dim == encode.TexDim.array_2d) {
                // 2D array: NAK packs the array-texture coordinate with the
                // layer first (arr_idx at src0[0], then u, v). Crucially,
                // the hardware array index is an integer, not a float. NAK
                // converts it with f2u(layer + 0.5) (nak_nir_lower_tex.c,
                // around line 244). A raw float layer, for example
                // 1.0 = 0x3F800000, reads as garbage, always layer 0. So
                // this rounds and converts the layer to a u32 in coord[0],
                // using coord[1] as scratch for the 0.5 before u lands
                // there. Then u, v (still floats) go at coord[1], coord[2].
                // The 3-register coordinate is 4-aligned.
                const layer = gprOf(loc.*, call.w);
                try code.append(allocator, encode.movImm(call.coord + 1, @bitCast(@as(f32, 0.5)), .{}));
                try code.append(allocator, encode.fadd(call.coord, layer, call.coord + 1, .{}));
                try code.append(allocator, encode.f2iRound(call.coord, call.coord, false, .zero, .{})); // (u32)floor(layer+0.5)
                try code.append(allocator, encode.movReg(call.coord + 1, u, .{}));
                try code.append(allocator, encode.movReg(call.coord + 2, v, .{}));
            } else if (call.dim != encode.TexDim.dim_2d) {
                // 3D on Blackwell: with a properly 4-aligned coordinate
                // group, the hardware reads the natural NAK/NIR order
                // (u, v, w): u, v in-slice at coord/coord+1, and the
                // slice/depth w at coord+2. The earlier "(w,u,v)
                // slice-first" finding was an artifact of a 2-aligned
                // coordinate that also faulted with Xid 13, "Misaligned
                // Register". Fixing the alignment restores the standard
                // order. See [[prism-3d-textures]].
                const w = gprOf(loc.*, call.w);
                try code.append(allocator, encode.movReg(call.coord, u, .{}));
                try code.append(allocator, encode.movReg(call.coord + 1, v, .{}));
                try code.append(allocator, encode.movReg(call.coord + 2, w, .{}));
            } else if (call.explicit_lod) {
                // Explicit-LOD 2D (textureLod, or any vertex-shader sample,
                // since a vertex shader has no derivatives). Pack (u, v),
                // then a consecutive (handle, lod) pair at coord+2/coord+3,
                // and emit a TEX.LL, like the cube-LOD path: NAK reads the
                // explicit LOD from src1[1] = handle+1.
                try code.append(allocator, encode.movReg(call.coord, u, .{}));
                try code.append(allocator, encode.movReg(call.coord + 1, v, .{}));
                try code.append(allocator, encode.movReg(call.coord + 2, handle, .{})); // handle copy
                try code.append(allocator, encode.movReg(call.coord + 3, gprOf(loc.*, call.lod), .{})); // lod
                cube_lod = true;
            } else {
                try code.append(allocator, encode.movReg(call.coord, u, .{}));
                try code.append(allocator, encode.movReg(call.coord + 1, v, .{}));
            }
            // TEX result goes to dst..dst+3. wr_barrier lets the scheduler
            // gate the reloads, since the result lands a variable number of
            // cycles after issue. `dim` selects the texture target and
            // coordinate count. A gather emits TLD4, fetching one component
            // of the footprint. A cube carries an explicit LOD (coord+2), so
            // it emits TLD. Otherwise this emits an implicit-LOD TEX.
            const tex_inst = if (call.is_fetch)
                // src1 base = fetch_src1, which holds the handle. The
                // hardware reads the explicit LOD from the next register.
                encode.tld(dst, call.coord, fetch_src1, tex_dim, .{ .wr_barrier = 0 })
            else if (call.gather_comp) |comp|
                encode.tld4(dst, call.coord, handle, tex_dim, comp, .{ .wr_barrier = 0 })
            else if (cube_lod)
                // src1 base = coord+2, which holds the handle. The hardware
                // reads the explicit LOD from coord+3.
                encode.texLod(dst, call.coord, call.coord + 2, tex_dim, .{ .wr_barrier = 0 })
            else
                encode.tex(dst, call.coord, handle, tex_dim, .{ .wr_barrier = 0 });
            try code.append(allocator, tex_inst);
        },
        .barrier => |bar| switch (bar.scope) {
            // BAR.SYNC makes every thread of the CTA wait, and the CTA is the workgroup.
            // `checkBarrierConvergence` has already refused a placement the warp can split
            // around, so this emits the plain unpredicated form.
            .workgroup => try code.append(allocator, encode.barSync(.{})),
            // A subgroup (warp) barrier is a different instruction. Emitting BAR.SYNC for it
            // would make every warp of the workgroup wait, not just the asking one, which
            // deadlocks a kernel whose other warps never reach the barrier. Refuse instead.
            .subgroup => return error.Unsupported,
        },
        .atomic_rmw => |a| try lowerAtomicRmw(allocator, func, loc.*, code, inst, a),
        .@"if" => {}, // handled by the caller (it terminates the block)
        else => return error.Unsupported,
    }
}

fn arith(func: *const Function, op: ir.function.BinOp, rd: u8, ra: u8, rb: u8, lhs: Value) Error!Inst {
    const is_float = isFloat(func, lhs);
    return switch (op) {
        .add => if (is_float) encode.fadd(rd, ra, rb, .{}) else encode.iadd3(rd, ra, rb, .{}),
        .sub => if (is_float) encode.fsub(rd, ra, rb, .{}) else encode.isub(rd, ra, rb, .{}),
        .mul => if (is_float) encode.fmul(rd, ra, rb, .{}) else encode.imad(rd, ra, rb, encode.RZ, .{}),
        .bit_and => encode.lop3(rd, ra, rb, encode.LUT_AND, .{}),
        .bit_or => encode.lop3(rd, ra, rb, encode.LUT_OR, .{}),
        .bit_xor => encode.lop3(rd, ra, rb, encode.LUT_XOR, .{}),
        .shl => encode.shf(rd, ra, rb, false, false, .{}),
        .shr => encode.shf(rd, ra, rb, true, isSignedRaw(func, lhs), .{}),
        // Integer divide is a multi-instruction reciprocal sequence, and is
        // deferred. `mulh` is expanded to plain multiplies and shifts
        // (`expandMulh`) before this backend's isel.
        .div, .rem, .mulh => error.Unsupported,
    };
}

/// The same binary operation as `arith`, but with a CONSTANT right operand that
/// goes straight into the instruction's immediate field.
///
/// Every ALU op here has a form that reads a 32-bit immediate in place of its
/// second source register, so the constant costs no MOV and no register. The
/// old lowering emitted `movImm` into a scratch register and then the register
/// form, which is two instructions and an extra write for every constant in the
/// kernel. See the immediate-form section of `encode.zig`.
///
/// SUBTRACTION NEGATES THE VALUE rather than the operand. The register forms
/// subtract by setting the srcB negate bit, which is bit 63, and the immediate
/// forms use bit 63 as a value bit.
///
/// The shift ops read a shift COUNT, not a general operand, so their immediate
/// is the count and the value stays in a register.
fn arithImm(func: *const Function, op: ir.function.BinOp, rd: u8, ra: u8, imm: u32, lhs: Value) Error!Inst {
    const is_float = isFloat(func, lhs);
    return switch (op) {
        .add => if (is_float) encode.faddImm(rd, ra, imm, .{}) else encode.iadd3Imm(rd, ra, imm, .{}),
        .sub => if (is_float) encode.fsubImm(rd, ra, imm, .{}) else encode.iadd3Imm(rd, ra, 0 -% imm, .{}),
        .mul => if (is_float) encode.fmulImm(rd, ra, imm, .{}) else encode.imadImm(rd, ra, imm, encode.RZ, .{}),
        .bit_and => encode.lop3Imm(rd, ra, imm, encode.LUT_AND, .{}),
        .bit_or => encode.lop3Imm(rd, ra, imm, encode.LUT_OR, .{}),
        .bit_xor => encode.lop3Imm(rd, ra, imm, encode.LUT_XOR, .{}),
        .shl => encode.shfImm(rd, ra, imm, false, false, .{}),
        .shr => encode.shfImm(rd, ra, imm, true, isSignedRaw(func, lhs), .{}),
        // The same three `arith` refuses, for the same reasons.
        .div, .rem, .mulh => error.Unsupported,
    };
}

/// The 3-input logic-op LUT (src0=0xF0, src1=0xCC, src2=0xAA truth table)
/// for a two-input bitwise op, shared by LOP3 (integer) and PLOP3
/// (predicate). Only the bitwise ops are valid here, since a logical
/// predicate combine is always one of these.
fn lutOf(op: ir.function.BinOp) error{Unsupported}!u8 {
    return switch (op) {
        .bit_and => encode.LUT_AND,
        .bit_or => encode.LUT_OR,
        .bit_xor => encode.LUT_XOR,
        // Only bitwise ops produce a bool that reaches a predicate combine.
        // Any other op here is an unsupported IR shape. This surfaces as an
        // error, not a panic, matching isel's no-panic policy, since the
        // convention is not compiler-enforced.
        else => error.Unsupported,
    };
}

fn cmpOf(op: ir.function.CmpOp) encode.Cmp {
    return switch (op) {
        .eq => .eq,
        .ne => .ne,
        .lt => .lt,
        .le => .le,
        .gt => .gt,
        .ge => .ge,
    };
}

fn isFloat(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => true,
        else => false,
    };
}

fn isSigned(func: *const Function, v: Value) bool {
    return isSignedRaw(func, v);
}

fn isSignedRaw(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |x| x.signedness == .signed,
        else => true,
    };
}

fn emitIf(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), fixups: *std.ArrayList(Fixup), cf: ir.function.If) Error!void {
    const pred = predOf(loc.*, cf.cond);
    // Each path's phi edge moves must execute only on that path. The old
    // layout emitted the `then` moves unconditionally, before the guarded
    // branch, so when the condition was false they still ran and clobbered
    // registers before the `else` moves. This was benign for a single phi,
    // since the else move overwrote it, but it corrupted state when a then
    // edge move's source was a register a later, else-path, value needed.
    // A shader with several phi-merging branches over live texture or
    // derivative values hits this. The layout is now:
    //     @P BRA L_then          (cond true -> skip the else moves)
    //        <else edge moves>
    //        BRA else_target
    //   L_then:
    //        <then edge moves>
    //        BRA then_target
    const skip_else = code.items.len;
    try code.append(allocator, encode.bra(0, .{ .pred = pred })); // taken if cond -> L_then
    // else path
    try emitMoves(allocator, func, loc, code, cf.@"else");
    const else_bra = code.items.len;
    try code.append(allocator, encode.bra(0, .{}));
    try fixups.append(allocator, .{ .at = else_bra, .target = @intFromEnum(cf.@"else".target) });
    // L_then: patch the guarded branch to here (a local fixup by instruction index).
    try fixups.append(allocator, .{ .at = skip_else, .target_inst = code.items.len });
    try emitMoves(allocator, func, loc, code, cf.then);
    const then_bra = code.items.len;
    try code.append(allocator, encode.bra(0, .{}));
    try fixups.append(allocator, .{ .at = then_bra, .target = @intFromEnum(cf.then.target) });
}

fn emitJump(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), fixups: *std.ArrayList(Fixup), jump: ir.function.Jump) Error!void {
    try emitMoves(allocator, func, loc, code, jump);
    const at = code.items.len;
    try code.append(allocator, encode.bra(0, .{}));
    try fixups.append(allocator, .{ .at = at, .target = @intFromEnum(jump.target) });
}

/// One register-to-register copy on a control-flow edge.
const EdgeMove = struct { dst: u8, src: u8 };

/// Edge moves into the target block's parameters.
///
/// These copies are a PARALLEL assignment: every source is read as it was before the edge,
/// and every destination is written after. Emitting them in argument order does not do
/// that, and `emitParallelCopy` is what makes the emitted order behave as if they happened
/// at once.
///
/// A 64-BIT ADDRESS MOVES AS A PAIR. `assignLocs` gives a global, constant or private
/// pointer an aligned register pair, so a single copy of the low half left the high half of
/// the target parameter holding whatever the allocator last put there, and every access
/// through that parameter addressed another place.
///
/// A BOOLEAN LIVES IN A PREDICATE, not a GPR. There is no spare predicate to break a cycle
/// with: the allocator hands out P0..P5 and P6 is the carry scratch. An edge that passes
/// one is refused, which is what `gprOf` would otherwise reach an `unreachable` on.
fn emitMoves(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), jump: ir.function.Jump) Error!void {
    const args = func.blockArgs(jump);
    const params = func.blockParams(jump.target);
    if (args.len != params.len) return error.Unsupported;

    var moves: std.ArrayList(EdgeMove) = .empty;
    defer moves.deinit(allocator);
    for (args, params) |arg, param| {
        if (isBool(func, param) or isBool(func, arg)) return error.Unsupported;
        const dst = gprOf(loc.*, param);
        const src = gprOf(loc.*, arg);
        const span: u8 = if (isWidePtr(func, param)) 2 else 1;
        var i: u8 = 0;
        while (i < span) : (i += 1) {
            if (dst + i != src + i) try moves.append(allocator, .{ .dst = dst + i, .src = src + i });
        }
    }
    try emitParallelCopy(allocator, code, &moves);
}

/// Whether a move OTHER than the one at `skip` still reads `reg`.
fn readByAnotherMove(moves: []const EdgeMove, skip: usize, reg: u8) bool {
    for (moves, 0..) |m, i| {
        if (i == skip) continue;
        if (m.src == reg) return true;
    }
    return false;
}

/// Emit a set of register copies so that the result is the PARALLEL assignment they
/// describe, whatever order the hardware runs them in.
///
/// Emitted in the order they were built, a copy can overwrite a register a later copy still
/// reads. A CYCLE has no safe order at all. The old code emitted a five-way block-parameter
/// permutation as five plain MOVs:
///
///     MOV R4, R9 ; MOV R5, R7 ; MOV R7, R4 ; MOV R8, R5 ; MOV R9, R8
///
/// The first MOV destroyed R4, so the third copied the NEW R4 into R7 and the value R4 held
/// never arrived. R7 came out zero. Every block argument on an edge whose permutation
/// contains a cycle was silently lost.
///
/// The rule is the standard one. Emit any move whose DESTINATION no other remaining move
/// reads: nothing else needs the register it overwrites, so it is safe now. When no such
/// move is left, only cycles remain. Park one cycle member's source in the scratch register
/// and point every reader of that register at the scratch instead, which opens the cycle
/// into a chain.
///
/// THE SCRATCH IS FREE AGAIN BEFORE THE NEXT PARK. Breaking a cycle makes exactly one of its
/// own moves ready, and no move of any OTHER cycle can be ready, so the broken cycle unwinds
/// to its end before the loop stalls again. The last move of that chain is the one that
/// reads the scratch.
///
/// `r_scratch` is the parking register. The prologue owns it and `assignLocs` gives it to no
/// value, so parking there destroys nothing.
fn emitParallelCopy(allocator: std.mem.Allocator, code: *std.ArrayList(Inst), moves: *std.ArrayList(EdgeMove)) Error!void {
    // Each turn either emits a move or breaks a cycle, and a broken cycle lets at least one
    // move go on the next turn, so the whole set drains.
    while (moves.items.len > 0) {
        var ready: ?usize = null;
        for (moves.items, 0..) |m, i| {
            if (readByAnotherMove(moves.items, i, m.dst)) continue;
            ready = i;
            break;
        }
        if (ready) |i| {
            const m = moves.orderedRemove(i);
            try code.append(allocator, encode.movReg(m.dst, m.src, .{}));
            continue;
        }
        const parked = moves.items[0].src;
        try code.append(allocator, encode.movReg(r_scratch, parked, .{}));
        for (moves.items) |*m| {
            if (m.src == parked) m.src = r_scratch;
        }
    }
}

// Liveness (for the allocator).

fn markUse(last_use: []u32, v: Value, pos: u32) void {
    if (pos > last_use[@intFromEnum(v)]) last_use[@intFromEnum(v)] = pos;
}

fn forEachUse(func: *const Function, inst: ir.function.Inst, last_use: []u32, pos: u32) void {
    switch (func.opcode(inst)) {
        .atomic_rmw => |a| {
            markUse(last_use, a.ptr, pos);
            markUse(last_use, a.value, pos);
            if (a.compare) |c| markUse(last_use, c, pos);
        },
        // A barrier reads no Value operand.
        .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
        .arith => |a| {
            markUse(last_use, a.lhs, pos);
            markUse(last_use, a.rhs, pos);
        },
        .arith_imm => |a| markUse(last_use, a.lhs, pos),
        .icmp => |c| {
            markUse(last_use, c.lhs, pos);
            markUse(last_use, c.rhs, pos);
        },
        .select => |s| {
            markUse(last_use, s.cond, pos);
            markUse(last_use, s.then, pos);
            markUse(last_use, s.@"else", pos);
        },
        .extract => |e| markUse(last_use, e.aggregate, pos),
        .convert => |cv| markUse(last_use, cv.value, pos),
        .unary => |u| markUse(last_use, u.value, pos),
        .load => |ld| markUse(last_use, ld.ptr, pos),
        .store => |st| {
            markUse(last_use, st.value, pos);
            markUse(last_use, st.ptr, pos);
        },
        .prefetch => {}, // dropped at emission, no register read here
        .va_start => |vs| markUse(last_use, vs.list, pos),
        .va_arg => |va| markUse(last_use, va.list, pos),
        .va_end => |ve| markUse(last_use, ve.list, pos),
        .dot => |d| {
            markUse(last_use, d.acc, pos);
            markUse(last_use, d.a, pos);
            markUse(last_use, d.b, pos);
        },
        .matmul => |mmv| {
            markUse(last_use, mmv.a, pos);
            markUse(last_use, mmv.b, pos);
            markUse(last_use, mmv.c, pos);
        },
        .struct_new => |sn| for (func.valueList(sn.fields)) |f| markUse(last_use, f, pos),
        .call => |c| for (func.valueList(c.args)) |a| markUse(last_use, a, pos),
        .call_indirect => |c| {
            markUse(last_use, c.target, pos);
            for (func.valueList(c.args)) |a| markUse(last_use, a, pos);
        },
        .@"if" => |cf| {
            markUse(last_use, cf.cond, pos);
            for (func.blockArgs(cf.then)) |a| markUse(last_use, a, pos);
            for (func.blockArgs(cf.@"else")) |a| markUse(last_use, a, pos);
        },
    }
}

fn forEachTermUse(func: *const Function, term: Terminator, last_use: []u32, pos: u32) void {
    switch (term) {
        .ret => |r| for (r.slice()) |vv| markUse(last_use, vv, pos),
        .jump => |j| for (func.blockArgs(j)) |a| markUse(last_use, a, pos),
    }
}

fn setUsed(row: []bool, v: Value) void {
    row[@intFromEnum(v)] = true;
}

fn markUsedBitset(func: *const Function, inst: ir.function.Inst, row: []bool) void {
    switch (func.opcode(inst)) {
        .atomic_rmw => |a| {
            setUsed(row, a.ptr);
            setUsed(row, a.value);
            if (a.compare) |c| setUsed(row, c);
        },
        // A barrier reads no Value operand.
        .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
        .arith => |a| {
            setUsed(row, a.lhs);
            setUsed(row, a.rhs);
        },
        .arith_imm => |a| setUsed(row, a.lhs),
        .icmp => |c| {
            setUsed(row, c.lhs);
            setUsed(row, c.rhs);
        },
        .select => |s| {
            setUsed(row, s.cond);
            setUsed(row, s.then);
            setUsed(row, s.@"else");
        },
        .extract => |e| setUsed(row, e.aggregate),
        .convert => |cv| setUsed(row, cv.value),
        .unary => |u| setUsed(row, u.value),
        .load => |ld| setUsed(row, ld.ptr),
        .store => |st| {
            setUsed(row, st.value);
            setUsed(row, st.ptr);
        },
        .prefetch => {}, // dropped at emission, no register read here
        .va_start => |vs| setUsed(row, vs.list),
        .va_arg => |va| setUsed(row, va.list),
        .va_end => |ve| setUsed(row, ve.list),
        .dot => |d| {
            setUsed(row, d.acc);
            setUsed(row, d.a);
            setUsed(row, d.b);
        },
        .matmul => |mmv| {
            setUsed(row, mmv.a);
            setUsed(row, mmv.b);
            setUsed(row, mmv.c);
        },
        .struct_new => |sn| for (func.valueList(sn.fields)) |f| setUsed(row, f),
        .call => |c| for (func.valueList(c.args)) |a| setUsed(row, a),
        .call_indirect => |c| {
            setUsed(row, c.target);
            for (func.valueList(c.args)) |a| setUsed(row, a);
        },
        .@"if" => |cf| {
            setUsed(row, cf.cond);
            for (func.blockArgs(cf.then)) |a| setUsed(row, a);
            for (func.blockArgs(cf.@"else")) |a| setUsed(row, a);
        },
    }
}

fn markUsedTermBitset(func: *const Function, term: Terminator, row: []bool) void {
    switch (term) {
        .ret => |r| for (r.slice()) |vv| setUsed(row, vv),
        .jump => |j| for (func.blockArgs(j)) |a| setUsed(row, a),
    }
}

/// Backward liveness dataflow. Extends `last_use[v]` to the end of every block
/// where `v` is live-out, so a value live across a loop keeps its register.
fn extendLiveRanges(allocator: std.mem.Allocator, func: *const Function, last_use: []u32, block_end: []const u32) Error!void {
    const nblocks = func.blockCount();
    const nval = func.valueCount();
    if (nblocks == 0 or nval == 0) return;

    var succ = try allocator.alloc(std.ArrayList(u32), nblocks);
    defer {
        for (succ) |*s| s.deinit(allocator);
        allocator.free(succ);
    }
    for (succ) |*s| s.* = .empty;
    const defined = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(defined);
    const used = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(used);
    @memset(defined, false);
    @memset(used, false);

    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        const row = used[bi * nval ..][0..nval];
        for (func.blockParams(block)) |p| defined[bi * nval + @intFromEnum(p)] = true;
        for (func.blockInsts(block)) |inst| {
            markUsedBitset(func, inst, row);
            if (func.instResult(inst)) |r| defined[bi * nval + @intFromEnum(r)] = true;
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                try succ[bi].append(allocator, @intFromEnum(cf.then.target));
                try succ[bi].append(allocator, @intFromEnum(cf.@"else".target));
            }
        }
        if (func.terminator(block)) |term| {
            markUsedTermBitset(func, term, row);
            if (term == .jump) try succ[bi].append(allocator, @intFromEnum(term.jump.target));
        }
    }

    const live_in = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(live_in);
    const live_out = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(live_out);
    @memset(live_in, false);
    @memset(live_out, false);

    var changed = true;
    while (changed) {
        changed = false;
        var b: usize = nblocks;
        while (b > 0) {
            b -= 1;
            for (succ[b].items) |s| {
                for (0..nval) |v| {
                    if (live_in[@as(usize, s) * nval + v] and !live_out[b * nval + v]) {
                        live_out[b * nval + v] = true;
                        changed = true;
                    }
                }
            }
            for (0..nval) |v| {
                const new_in = (used[b * nval + v] or live_out[b * nval + v]) and !defined[b * nval + v];
                if (new_in and !live_in[b * nval + v]) {
                    live_in[b * nval + v] = true;
                    changed = true;
                }
            }
        }
    }

    for (0..nblocks) |b| {
        for (0..nval) |v| {
            if (live_out[b * nval + v] and block_end[b] > last_use[v]) last_use[v] = block_end[b];
        }
    }
}

const testing = std.testing;

test "compiles a vertex shader: attribute load, compute, attribute store, exit" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    // A vertex input attribute, tagged with its slot, incremented and
    // written to the clip-space position output.
    const in = try func.appendBlockParam(b, f32_t);
    try func.addAttr(.{ .value = in }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = encode.ATTR_GENERIC0 } } });
    const one = try func.appendInst(b, f32_t, .{ .fconst = 1.0 });
    const sum = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = in, .rhs = one } });
    const out_ptr = try func.appendInst(b, i32_t, .{ .iconst = 0 }); // the position output slot
    try func.addAttr(.{ .value = out_ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = encode.ATTR_POSITION } } });
    try func.appendStore(b, sum, out_ptr);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileShader(allocator, &func, .vertex, nvidia_abi);
    defer kernel.deinit(allocator);

    // The sequence is: ALD (attribute fetch), FADD, AST (write position), EXIT.
    //
    // The FADD is the IMMEDIATE form (0x421), not the register form (0x221): the 1.0 is a
    // constant, so `foldConstantsToImm` moves it into the instruction and no MOV
    // materializes it. `encode.faddImm` explains why an immediate FADD carries form 2.
    var has_ald = false;
    var has_fadd = false;
    var has_ast = false;
    var has_exit = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0x321 => has_ald = true,
            0x421 => has_fadd = true,
            0x322 => has_ast = true,
            0x94d => has_exit = true,
            else => {},
        }
    }
    try testing.expect(has_ald and has_fadd and has_ast and has_exit);
}

test "graphics: a UBO pointer param loads its address from constant bank (LDC), then LDG" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();

    // Entry params, in graphics order: a vertex input attribute scalar,
    // then a UBO base pointer (a `ptr`, untagged, exactly what the SPIR-V
    // lowering appends for a uniform block). The body loads a uniform
    // float through the UBO pointer, adds the input, and writes the
    // clip-space position output.
    const in = try func.appendBlockParam(b, f32_t);
    try func.addAttr(.{ .value = in }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = encode.ATTR_GENERIC0 } } });
    const ubo = try func.appendBlockParam(b, ptr_t);
    const uval = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = ubo } });
    const sum = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = in, .rhs = uval } });
    const out_ptr = try func.appendInst(b, i32_t, .{ .iconst = 0 });
    try func.addAttr(.{ .value = out_ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = encode.ATTR_POSITION } } });
    try func.appendStore(b, sum, out_ptr);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileShader(allocator, &func, .vertex, nvidia_abi);
    defer kernel.deinit(allocator);

    // The prologue must source the UBO pointer from the constant bank (ONE LDC.64 for the
    // whole 64-bit address), and the body must LDG through it, plus do an ALD for the
    // input.
    var ldc_count: usize = 0;
    var has_ldg = false;
    var has_ald = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0xb82 => ldc_count += 1, // LDC (constant-bank load)
            0x981 => has_ldg = true, // LDG (global load through the UBO pointer)
            0x321 => has_ald = true, // ALD (the input attribute)
            else => {},
        }
    }
    try testing.expectEqual(@as(usize, 1), ldc_count); // one LDC.64 for the whole address
    try testing.expect(has_ldg);
    try testing.expect(has_ald);
    // The LDC reads the whole address at graphics_ubo_cb_base.
    i = 0;
    var first_ldc_off: ?u16 = null;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff == 0xb82) {
            first_ldc_off = @truncate(kernel.code[i + 1] >> 6);
            break;
        }
    }
    try testing.expectEqual(@as(u16, encode.graphics_ubo_cb_base), first_ldc_off.?);
}

test "graphics: gl_VertexIndex sources from S2R and pulls a vec from a UBO array (no attribute)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();

    // Entry params, in vertex-pulling order with no attribute inputs: the
    // gl_VertexIndex builtin (i32, tagged with the vertex_index builtin), then the
    // UBO base pointer. The body computes
    // &u.pos[gl_VertexIndex] = base + index*stride, loads a float through
    // it, and writes the clip-space position output. This is exactly the
    // IR the SPIR-V lowering produces for `u.pos[gl_VertexIndex]` with a
    // zero-attribute pipeline.
    const vi = try func.appendBlockParam(b, i32_t);
    try gpu.attrs.setBuiltin(&func, vi, .vertex_index);
    const ubo = try func.appendBlockParam(b, ptr_t);
    const stride = try func.appendInst(b, i32_t, .{ .iconst = 16 }); // std140 vec4 stride
    const off = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .mul, .lhs = vi, .rhs = stride } });
    const elem_ptr = try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = ubo, .rhs = off } });
    const uval = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = elem_ptr } });
    const out_ptr = try func.appendInst(b, i32_t, .{ .iconst = 0 });
    try func.addAttr(.{ .value = out_ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = encode.ATTR_POSITION } } });
    try func.appendStore(b, uval, out_ptr);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileShader(allocator, &func, .vertex, nvidia_abi);
    defer kernel.deinit(allocator);

    // This must source gl_VertexIndex through ALD a[ATTR_VERTEX_ID] (the
    // DA-delivered vertex-ID attribute), scale it (IMAD index*stride), load
    // the UBO pointer (one LDC.64), and LDG through base+offset. The only ALD
    // is the vertex-ID read. There is no vertex attribute, since the vertex
    // shader pulls from the UBO, not a vertex buffer.
    var has_ald_vid = false;
    var has_imad = false;
    var ldc_count: usize = 0;
    var has_ldg = false;
    var ald_count: usize = 0;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0x321 => { // ALD
                ald_count += 1;
                if ((kernel.code[i + 1] >> 8) & 0x3ff == encode.ATTR_VERTEX_ID) has_ald_vid = true;
            },
            0x824 => has_imad = true, // IMAD with an immediate scale (base 0x024 | imm form)
            0xb82 => ldc_count += 1,
            0x981 => has_ldg = true,
            else => {},
        }
    }
    try testing.expect(has_ald_vid);
    try testing.expect(has_imad);
    // ONE LDC.64 reads the whole UBO address. It was two 32-bit LDCs before.
    try testing.expectEqual(@as(usize, 1), ldc_count);
    try testing.expect(has_ldg);
    try testing.expectEqual(@as(usize, 1), ald_count); // only the vertex-id ALD, no attribute fetch
}

test "compiles a kernel: load params, multiply-add, store, exit" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    // Prologue: ONE LDC.64 for the output pointer plus two scalar LDCs equals 3
    // instructions, then IMAD, STG, EXIT equals 6 instructions total (24 dwords).
    // The output pointer took two LDCs of its own before.
    //
    // THE ADD IS GONE, and that is the contraction: `x * y + x` is one IMAD, whose addend
    // field held RZ while the sum needed a separate IADD3. This test read 7 instructions
    // with an IADD3 after the IMAD before `FmaFold` existed.
    try testing.expectEqual(@as(usize, 6 * 4), kernel.code.len);
    try testing.expectEqual(@as(u32, 0xb82), kernel.code[0] & 0xfff); // first LDC
    // The first LDC reads the whole output pointer at the ABI's parameter base.
    try testing.expectEqual(
        @as(u32, nvidia_abi.param_base),
        @as(u32, @as(u16, @truncate(kernel.code[1] >> 6)) & 0xffff),
    );

    // The instruction words: LDC.64, LDC, LDC, IMAD, STG, EXIT.
    const op = struct {
        fn at(code: []const u32, i: usize) u32 {
            return code[i * 4] & 0xfff;
        }
    }.at;
    try testing.expectEqual(@as(u32, 0xb82), op(kernel.code, 2)); // last param LDC
    try testing.expectEqual(@as(u32, 0x224), op(kernel.code, 3)); // IMAD (base 0x024 | reg form)
    try testing.expectEqual(@as(u32, 0x986), op(kernel.code, 4)); // STG
    try testing.expectEqual(@as(u32, 0x94d), op(kernel.code, 5)); // EXIT
    // The addend is x, not RZ. A plain multiply leaves RZ there, so this field is the
    // whole difference between a contracted IMAD and the multiply it grew out of.
    try testing.expectEqual(gprOfCompiled(&func, x, .{}), regAt(kernel.code, 3, 64));
}

// The opcodes the contraction tests below read, as `opAt` sees them: the 9-bit NAK base
// ORed with the source form in bits 9..11. Form 1 is a register second source and form 4 a
// 32-bit immediate one. See `encode.alu` and `encode.aluImm`.
const FMUL_REG: u32 = 0x220;
const FMUL_IMM: u32 = 0x820;
const FADD_REG: u32 = 0x221;
const FFMA_REG: u32 = 0x223;
const FFMA_IMM: u32 = 0x823;
const IMAD_REG: u32 = 0x224;
const IMAD_IMM: u32 = 0x824;
const IADD3_REG: u32 = 0x210;

/// A three-parameter scalar kernel `out = f(p0, p1, p2)`, where `body` builds the value to
/// return out of the three parameters. Every contraction test has this shape.
fn buildTernaryKernel(
    func: *Function,
    ty: ir.types.Type,
    body: *const fn (*Function, Block, Value, Value, Value) anyerror!Value,
) !struct { a: Value, b: Value, c: Value } {
    const blk = try func.appendBlock();
    const a = try func.appendBlockParam(blk, ty);
    const b = try func.appendBlockParam(blk, ty);
    const c = try func.appendBlockParam(blk, ty);
    const r = try body(func, blk, a, b, c);
    func.setTerminator(blk, .{ .ret = ir.function.Ret.one(r) });
    return .{ .a = a, .b = b, .c = c };
}

fn mulThenAdd(func: *Function, blk: Block, a: Value, b: Value, c: Value) anyerror!Value {
    const ty = func.valueType(a);
    const p = try func.appendInst(blk, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    return func.appendInst(blk, ty, .{ .arith = .{ .op = .add, .lhs = p, .rhs = c } });
}

fn addThenMul(func: *Function, blk: Block, a: Value, b: Value, c: Value) anyerror!Value {
    const ty = func.valueType(a);
    const p = try func.appendInst(blk, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    return func.appendInst(blk, ty, .{ .arith = .{ .op = .add, .lhs = c, .rhs = p } });
}

test "a float multiply-add contracts into ONE FFMA that names all three operands" {
    // `a * b + c`. The pair it replaces is an FMUL and then an FADD: two instructions and
    // two dependent latencies where the hardware fuses both into one.
    //
    // COUNTING OPCODES IS NOT ENOUGH. An FFMA whose operand fields name the wrong
    // registers is shorter AND wrong, so every field is checked against the register the
    // allocator really gave that value.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const p = try buildTernaryKernel(&func, f32_t, mulThenAdd);

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    try testing.expectEqual(@as(?usize, null), findOp(kernel.code, FMUL_REG));
    try testing.expectEqual(@as(?usize, null), findOp(kernel.code, FADD_REG));
    const at = try onlyOpAt(kernel.code, FFMA_REG);
    try testing.expectEqual(gprOfCompiled(&func, p.a, .{}), regAt(kernel.code, at, 24));
    try testing.expectEqual(gprOfCompiled(&func, p.b, .{}), regAt(kernel.code, at, 32));
    try testing.expectEqual(gprOfCompiled(&func, p.c, .{}), regAt(kernel.code, at, 64));
}

test "the commuted float form c + a * b contracts, with the same three operands" {
    // Nothing orders an add's operands, so a rule that only reads the left one leaves half
    // the multiply-adds in a real kernel uncontracted.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const p = try buildTernaryKernel(&func, f32_t, addThenMul);

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    try testing.expectEqual(@as(?usize, null), findOp(kernel.code, FMUL_REG));
    try testing.expectEqual(@as(?usize, null), findOp(kernel.code, FADD_REG));
    const at = try onlyOpAt(kernel.code, FFMA_REG);
    // The multiply keeps its own operands at srcA and srcB. The ADD's other operand is the
    // addend, wherever it stood in the add.
    try testing.expectEqual(gprOfCompiled(&func, p.a, .{}), regAt(kernel.code, at, 24));
    try testing.expectEqual(gprOfCompiled(&func, p.b, .{}), regAt(kernel.code, at, 32));
    try testing.expectEqual(gprOfCompiled(&func, p.c, .{}), regAt(kernel.code, at, 64));
}

test "an integer multiply-add contracts into IMAD, with the addend where RZ stood" {
    // This costs nothing to encode. A plain integer `.mul` ALREADY emits
    // `IMAD dst, a, b, RZ`, so contraction is the addend field holding the add's other
    // operand instead of the zero register.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const p = try buildTernaryKernel(&func, i32_t, mulThenAdd);

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    try testing.expectEqual(@as(?usize, null), findOp(kernel.code, IADD3_REG));
    const at = try onlyOpAt(kernel.code, IMAD_REG);
    try testing.expectEqual(gprOfCompiled(&func, p.a, .{}), regAt(kernel.code, at, 24));
    try testing.expectEqual(gprOfCompiled(&func, p.b, .{}), regAt(kernel.code, at, 32));
    const addend = regAt(kernel.code, at, 64);
    try testing.expectEqual(gprOfCompiled(&func, p.c, .{}), addend);
    try testing.expect(addend != encode.RZ); // the uncontracted multiply's addend
}

test "a constant-scale multiply-add contracts into the IMMEDIATE FFMA and IMAD forms" {
    // `x * k + y`, where the scale is a compile-time constant. `foldConstantsToImm` has
    // already put the constant in the multiply's immediate field, and FFMA and IMAD both
    // read a 32-bit immediate multiplier beside a register addend. This is the shape
    // `base + index * stride` takes, which every array walk writes.
    const allocator = testing.allocator;
    const Case = struct { float: bool, want: u32, imm: u32 };
    for ([_]Case{
        .{ .float = true, .want = FFMA_IMM, .imm = @bitCast(@as(f32, 0.5)) },
        .{ .float = false, .want = IMAD_IMM, .imm = 4 },
    }) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const ty = if (case.float)
            try func.types.intern(.{ .float = .f32 })
        else
            try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const blk = try func.appendBlock();
        const x = try func.appendBlockParam(blk, ty);
        const y = try func.appendBlockParam(blk, ty);
        const k = if (case.float)
            try func.appendInst(blk, ty, .{ .fconst = 0.5 })
        else
            try func.appendInst(blk, ty, .{ .iconst = 4 });
        const scaled = try func.appendInst(blk, ty, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = k } });
        const sum = try func.appendInst(blk, ty, .{ .arith = .{ .op = .add, .lhs = scaled, .rhs = y } });
        func.setTerminator(blk, .{ .ret = ir.function.Ret.one(sum) });

        var kernel = try compileKernel(allocator, &func, nvidia_abi);
        defer kernel.deinit(allocator);

        try testing.expectEqual(@as(?usize, null), findOp(kernel.code, FMUL_IMM));
        const at = try onlyOpAt(kernel.code, case.want);
        try testing.expectEqual(gprOfCompiled(&func, x, .{}), regAt(kernel.code, at, 24));
        try testing.expectEqual(case.imm, immAt(kernel.code, at)); // the scale, at bits 32..63
        try testing.expectEqual(gprOfCompiled(&func, y, .{}), regAt(kernel.code, at, 64));
    }
}

test "a multiply read TWICE keeps its own FMUL, and neither add contracts" {
    // THE GUARD IN THE OTHER DIRECTION. Contracting a multiply with two readers would put
    // the multiply inside one add and leave the second reader with a register nothing ever
    // writes. A guard that never refuses is not a guard, so this proves it refuses.
    //
    // `p = a * b; s = p + c; out = s + p`. p has two readers.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const a = try func.appendBlockParam(blk, f32_t);
    const b = try func.appendBlockParam(blk, f32_t);
    const c = try func.appendBlockParam(blk, f32_t);
    const prod = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const s = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = c } });
    const out = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = prod } });
    func.setTerminator(blk, .{ .ret = ir.function.Ret.one(out) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    try testing.expectError(error.NotFound, onlyOpAt(kernel.code, FFMA_REG));
    const mul_at = try onlyOpAt(kernel.code, FMUL_REG);
    try testing.expectEqual(gprOfCompiled(&func, a, .{}), regAt(kernel.code, mul_at, 24));
    try testing.expectEqual(gprOfCompiled(&func, b, .{}), regAt(kernel.code, mul_at, 32));
    // Both adds are still there, in their own instructions.
    var adds: usize = 0;
    var i: usize = 0;
    while (i * 4 < kernel.code.len) : (i += 1) {
        if (opAt(kernel.code, i) == FADD_REG) adds += 1;
    }
    try testing.expectEqual(@as(usize, 2), adds);
}

test "a multiply in ANOTHER block is not contracted into the add that reads it" {
    // The operand registers of the multiply have to survive as far as the add, and this
    // backend only extends live ranges inside the block it scans. A multiply reached
    // through a branch can be executed a different number of times from the add, so
    // folding it across the edge would move work onto a path that never ran it.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);
    const b = try func.appendBlockParam(entry, f32_t);
    const c = try func.appendBlockParam(entry, f32_t);
    const tail = try func.appendBlock();
    const prod = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    try func.setJump(entry, tail, &.{});
    const sum = try func.appendInst(tail, f32_t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = c } });
    func.setTerminator(tail, .{ .ret = ir.function.Ret.one(sum) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    try testing.expectError(error.NotFound, onlyOpAt(kernel.code, FFMA_REG));
    _ = try onlyOpAt(kernel.code, FMUL_REG);
    _ = try onlyOpAt(kernel.code, FADD_REG);
}

test "contract_fma = false emits the FMUL and FADD pair, and nothing fuses" {
    // The default is ON, matching nvcc's `-fmad=true`. This proves the switch is real: the
    // SAME function compiled with contraction off keeps both instructions, so a caller
    // that needs the two-rounding answer can have it. See `Options.contract_fma`.
    const allocator = testing.allocator;
    const cases = [_]struct { options: Options, fused: bool }{
        .{ .options = .{}, .fused = true },
        .{ .options = .{ .contract_fma = false }, .fused = false },
    };
    for (cases) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const f32_t = try func.types.intern(.{ .float = .f32 });
        _ = try buildTernaryKernel(&func, f32_t, mulThenAdd);

        var kernel = try compileKernelOpts(allocator, &func, nvidia_abi, case.options);
        defer kernel.deinit(allocator);

        if (case.fused) {
            _ = try onlyOpAt(kernel.code, FFMA_REG);
            try testing.expectError(error.NotFound, onlyOpAt(kernel.code, FMUL_REG));
            try testing.expectError(error.NotFound, onlyOpAt(kernel.code, FADD_REG));
        } else {
            try testing.expectError(error.NotFound, onlyOpAt(kernel.code, FFMA_REG));
            _ = try onlyOpAt(kernel.code, FMUL_REG);
            _ = try onlyOpAt(kernel.code, FADD_REG);
        }
    }
}

test "a SUBTRACT is left alone: no negate bit is guessed at" {
    // `a * b - c` is fusible on the hardware, and so is `c - a * b`, but the two negate
    // DIFFERENT sources and this encoder has no tested negate modifier for FFMA. A wrong
    // negate bit returns a wrong number with no diagnostic, so the pair stands until the
    // bits are read out of a real ptxas encoding. This test states that choice, so a later
    // change that adds the negate has to come here and say so.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const blk = try func.appendBlock();
    const a = try func.appendBlockParam(blk, f32_t);
    const b = try func.appendBlockParam(blk, f32_t);
    const c = try func.appendBlockParam(blk, f32_t);
    const prod = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const diff = try func.appendInst(blk, f32_t, .{ .arith = .{ .op = .sub, .lhs = prod, .rhs = c } });
    func.setTerminator(blk, .{ .ret = ir.function.Ret.one(diff) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    try testing.expectError(error.NotFound, onlyOpAt(kernel.code, FFMA_REG));
    _ = try onlyOpAt(kernel.code, FMUL_REG);
    // FADD with the srcB negate bit (63) set is the subtract. See `encode.fsub`.
    const at = try onlyOpAt(kernel.code, FADD_REG);
    try testing.expectEqual(@as(u32, 1), (kernel.code[at * 4 + 1] >> 31) & 1);
}

test "the emitted LDC offsets match the offsets LaunchInfo reports" {
    // The whole point of LaunchInfo is that a runtime can build a parameter buffer without
    // reading the instruction stream. If codegen and the metadata ever disagree, the runtime
    // writes a parameter where the kernel does not read it, and the failure is silent garbage
    // rather than an error. This decodes the real LDC offsets back out of the SASS and
    // compares them.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const buf = try func.appendBlockParam(b, ptr_t);
    const n = try func.appendBlockParam(b, i32_t);
    const loaded = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = buf } });
    const sum = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = loaded, .rhs = n } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    // Collect the static offset field of every LDC in the stream, in order.
    var offsets: std.ArrayList(u32) = .empty;
    defer offsets.deinit(allocator);
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff != 0xb82) continue;
        try offsets.append(allocator, @as(u16, @truncate(kernel.code[i + 1] >> 6)) & 0xffff);
    }

    // The output pointer is ONE LDC.64, then the buffer pointer is ONE LDC.64, then the
    // scalar. Each pointer took two LDCs before, so this was five.
    try testing.expectEqual(@as(usize, 3), offsets.items.len);
    try testing.expectEqual(@as(usize, 2), kernel.launch.params.len);

    const base = nvidia_abi.param_base;
    try testing.expectEqual(base + kernel.launch.params[0].offset, offsets.items[1]);
    try testing.expectEqual(base + kernel.launch.params[1].offset, offsets.items[2]);

    // The reported block size is the declared default, and the pointer is global in M1.
    try testing.expectEqual([3]u32{ 1, 1, 1 }, kernel.launch.block);
    try testing.expectEqual(@as(u32, 0), kernel.launch.shared_bytes);
    try testing.expectEqual(gpu.AddressSpace.global, kernel.launch.params[0].kind.pointer);
    try testing.expectEqual(@as(u8, 4), kernel.launch.params[1].kind.scalar);
}

test "a graphics builtin on a compute kernel is rejected, not miscompiled" {
    // Regression: isInvocationId fired on ANY builtin attribute and compiled every tagged
    // parameter into the global-id sequence, so a graphics builtin silently became gid.x.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);
    try gpu.attrs.setBuiltin(&func, v, .vertex_index);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

test "the register count covers the two GPRs the hardware reserves" {
    // Regression, measured on Blackwell (GB10) silicon: the hardware keeps the top two GPRs of
    // each thread's allocation, dropping writes and reading zero with NO fault. A launch
    // descriptor built from a count that omits them loses the kernel's highest registers and
    // computes silent garbage.
    //
    // The granularity rounding hides this for most values, so the cases that matter are the
    // ones where the used count already sits on or just under a multiple of 8.
    try testing.expectEqual(@as(u32, 16), regCount(0)); // the 16 floor dominates
    try testing.expectEqual(@as(u32, 16), regCount(13)); // used 14, +2 = 16, exactly fits
    try testing.expectEqual(@as(u32, 24), regCount(14)); // used 15, +2 = 17: the old code gave 16
    try testing.expectEqual(@as(u32, 24), regCount(15)); // used 16, +2 = 18: the old code gave 16
    try testing.expectEqual(@as(u32, 32), regCount(22)); // used 23, +2 = 25: the old code gave 24

    // Every register the allocator may hand out stays inside the reported count, with the two
    // reserved registers still free above it. This is the property the launch depends on.
    var reg: u8 = 0;
    while (reg < 200) : (reg += 1) {
        try testing.expect(regCount(reg) >= @as(u32, reg) + 1 + hw_reserved_regs);
    }
}

test "an f16 function is rejected cleanly, not miscompiled as f64" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f16 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

test "compiles control flow: a max via if and a merge block" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const exit_b = try func.appendBlock();
    const r = try func.appendBlockParam(exit_b, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = exit_b, .args = &.{a} }, .{ .target = exit_b, .args = &.{b} });
    func.setTerminator(exit_b, .{ .ret = ir.function.Ret.one(r) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    // The stream contains an ISETP (compare), at least two BRA instructions,
    // an STG, and an EXIT.
    var saw_isetp = false;
    var bra_count: usize = 0;
    var saw_exit = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0x20c => saw_isetp = true, // ISETP (base 0x00c | reg form)
            0x947 => bra_count += 1,
            0x94d => saw_exit = true,
            else => {},
        }
    }
    try testing.expect(saw_isetp);
    try testing.expect(bra_count >= 2);
    try testing.expect(saw_exit);
}

test "convergence: a DIVERGENT if (distinct then/else blocks) wraps in BCLEAR/BSSY/BSYNC" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    // entry: if c { then_b } else { else_b }. Both go to merge. Merge does
    // a ret. This is a genuinely divergent branch, since then and else are
    // distinct blocks, so the Volta-and-later convergence barrier must wrap
    // it: BCLEAR plus BSSY before the branch, and BSYNC at the merge. A
    // degenerate if whose then and else targets match is not divergent and
    // gets no barrier. See the "max via if" test above.
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const r = try func.appendBlockParam(merge, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    const one = try func.appendInst(then_b, t, .{ .iconst = 1 });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{one}) } });
    const two = try func.appendInst(else_b, t, .{ .iconst = 2 });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{two}) } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(r) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    var saw_bclear = false;
    var saw_bssy = false;
    var saw_bsync = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0x355 => saw_bclear = true, // BCLEAR
            0x945 => saw_bssy = true, // BSSY
            0x941 => saw_bsync = true, // BSYNC
            else => {},
        }
    }
    try testing.expect(saw_bclear);
    try testing.expect(saw_bssy);
    try testing.expect(saw_bsync);
}

test "a FLOAT compare (max/min of floats) lowers to FSETP, not ISETP" {
    // The shared lowering turns GLSL f_max(a,b) into `icmp .gt` of the
    // float operands plus a select. The NVIDIA backend must emit a float
    // set-predicate (FSETP, opcode 0x00b) for float operands, not an
    // integer ISETP (0x00c). An integer compare of the float bit patterns
    // mis-orders values, for example max(0.0, x) would return 0 for a
    // positive x, which is the bug that rendered vkcube's lit faces black.
    // This test asserts the codegen distinction.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    // out[0] = max(a, 0.0) modelled as (a > 0) ? a : 0 over FLOAT operands.
    const a = try func.appendBlockParam(b, f32_t);
    const outp = try func.appendBlockParam(b, ptr_t);
    const zero = try func.appendInst(b, f32_t, .{ .fconst = 0.0 });
    const gt = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = zero } });
    const mx = try func.appendInst(b, f32_t, .{ .select = .{ .cond = gt, .then = a, .@"else" = zero } });
    try func.appendStore(b, mx, outp);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    var saw_fsetp = false;
    var saw_isetp = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0x20b => saw_fsetp = true, // FSETP (base 0x00b | reg form 1<<9)
            0x20c => saw_isetp = true, // ISETP (base 0x00c | reg form)
            else => {},
        }
    }
    try testing.expect(saw_fsetp); // a float compare emits FSETP
    try testing.expect(!saw_isetp); // and not an integer ISETP
}

test "REPRO: derivative + multi-component color outputs stay distinct until their stores" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();

    // A frag_pos.x varying, interpolated. The derivative descriptor table
    // records its slot and axis. The fragment shader computes
    // 0.5 + frag_pos.x*0.5 (RED) and 0.5 + dFdx(x)*32 (GREEN).
    const x = try func.appendBlockParam(b, f32_t);
    try func.addAttr(.{ .value = x }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = encode.ATTR_GENERIC0 } } });
    // The synthesized grad_buf pointer param, lazily appended after the
    // varyings, exactly as the SPIR-V derivative lowering does, plus the
    // one grad_slot descriptor: index 0, slot = ATTR_GENERIC0, axis = x.
    const grad_buf = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = grad_buf }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_buf", .value = .{ .int = 0 } } });
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_slot", .value = .{ .int = @as(i64, encode.ATTR_GENERIC0) << 1 } } });

    // RED = 0.5 + frag_pos.x*0.5.
    const half = try func.appendInst(b, f32_t, .{ .fconst = 0.5 });
    const x_half = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = half } });
    const red = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = half, .rhs = x_half } });
    // GREEN = 0.5 + dFdx(x)*32. dFdx(x) is a grad_buf[0] load (index 0
    // maps to the grad_buf param itself), replaced by the SHFL/FSWZADD
    // quad-derivative in the backend.
    const dfdx = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = grad_buf } });
    const k32 = try func.appendInst(b, f32_t, .{ .fconst = 32.0 });
    const dfdx32 = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = dfdx, .rhs = k32 } });
    const green = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = half, .rhs = dfdx32 } });
    // BLUE = 0.5, ALPHA = 1.0.
    const blue = try func.appendInst(b, f32_t, .{ .fconst = 0.5 });
    const alpha = try func.appendInst(b, f32_t, .{ .fconst = 1.0 });
    const comps = [_]Value{ red, green, blue, alpha };

    // Now the four color-out stores, batched at the end.
    for (comps, 0..) |comp, ci| {
        const color_slot = try func.appendInst(b, i32_t, .{ .iconst = @as(i64, @intCast(ci)) });
        try func.addAttr(.{ .value = color_slot }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = @intCast(ci) } } });
        try func.appendStore(b, comp, color_slot);
    }
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileShader(allocator, &func, .fragment, nvidia_abi);
    defer kernel.deinit(allocator);

    try assertNoColorClobber(&kernel);
}

test "REPRO: derivative FS with interleaved color stores does not clobber a color value" {
    // The exact shape behind the reported trace: a fragment shader that
    // takes a screen-space derivative and writes a multi-component color,
    // where each color component is stored as soon as it is computed, an
    // interleaved store, the natural per-component lowering. The hazard:
    // RED is computed and stored, then GREEN, the derivative path, reuses
    // RED's register, and a later batched color-store move for RED reads
    // the clobbered register. The fix must keep each color value live
    // until its store move.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();

    const x = try func.appendBlockParam(b, f32_t);
    try func.addAttr(.{ .value = x }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = encode.ATTR_GENERIC0 } } });
    const grad_buf = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = grad_buf }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_buf", .value = .{ .int = 0 } } });
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_slot", .value = .{ .int = @as(i64, encode.ATTR_GENERIC0) << 1 } } });

    const half = try func.appendInst(b, f32_t, .{ .fconst = 0.5 });

    // RED = 0.5 + x*0.5, stored immediately.
    const x_half = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = half } });
    const red = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = half, .rhs = x_half } });
    {
        const slot = try func.appendInst(b, i32_t, .{ .iconst = 0 });
        try func.addAttr(.{ .value = slot }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = 0 } } });
        try func.appendStore(b, red, slot);
    }
    // GREEN = 0.5 + dFdx(x)*32, computed after red's store, then stored.
    const dfdx = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = grad_buf } });
    const k32 = try func.appendInst(b, f32_t, .{ .fconst = 32.0 });
    const dfdx32 = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = dfdx, .rhs = k32 } });
    const green = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = half, .rhs = dfdx32 } });
    {
        const slot = try func.appendInst(b, i32_t, .{ .iconst = 1 });
        try func.addAttr(.{ .value = slot }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = 1 } } });
        try func.appendStore(b, green, slot);
    }
    // BLUE, ALPHA stored immediately.
    const blue = try func.appendInst(b, f32_t, .{ .fconst = 0.5 });
    {
        const slot = try func.appendInst(b, i32_t, .{ .iconst = 2 });
        try func.addAttr(.{ .value = slot }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = 2 } } });
        try func.appendStore(b, blue, slot);
    }
    const alpha = try func.appendInst(b, f32_t, .{ .fconst = 1.0 });
    {
        const slot = try func.appendInst(b, i32_t, .{ .iconst = 3 });
        try func.addAttr(.{ .value = slot }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = 3 } } });
        try func.appendStore(b, alpha, slot);
    }
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileShader(allocator, &func, .fragment, nvidia_abi);
    defer kernel.deinit(allocator);
    try assertNoColorClobber(&kernel);
}

test "REPRO: a derivative SHFL's source varying register is not clobbered before the SHFL" {
    // The screen-space-derivative SHFL sources the prologue-IPA'd varying
    // by register number, not as a tracked SSA use, so the linear-scan
    // allocator did not see that use. It freed and reused the varying
    // register for a later value, the shader's `*32` immediate, before the
    // SHFL read it, so the SHFL then shuffled garbage. assignLocs now
    // extends every fragment input-attribute param's live range to the
    // last grad_buf load. This asserts the SHFL's source register is
    // written by an IPA and by nothing else between that IPA and the SHFL.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();

    const x = try func.appendBlockParam(b, f32_t);
    try func.addAttr(.{ .value = x }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = encode.ATTR_GENERIC0 } } });
    const grad_buf = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = grad_buf }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_buf", .value = .{ .int = 0 } } });
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "grad_slot", .value = .{ .int = @as(i64, encode.ATTR_GENERIC0) << 1 } } });

    // o.r = 0.5 + dFdx(x)*32. The multiply materializes a constant into a
    // GPR that the allocator would otherwise place in the IPA'd varying's
    // register, which was the bug.
    const half = try func.appendInst(b, f32_t, .{ .fconst = 0.5 });
    const dfdx = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = grad_buf } });
    const k32 = try func.appendInst(b, f32_t, .{ .fconst = 32.0 });
    const dfdx32 = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .mul, .lhs = dfdx, .rhs = k32 } });
    const red = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = half, .rhs = dfdx32 } });
    const slot = try func.appendInst(b, i32_t, .{ .iconst = 0 });
    try func.addAttr(.{ .value = slot }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = 0 } } });
    try func.appendStore(b, red, slot);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileShader(allocator, &func, .fragment, nvidia_abi);
    defer kernel.deinit(allocator);

    // Find the first SHFL (opcode 0xf89) and its source register (bits 24..31).
    var shfl_idx: ?usize = null;
    var shfl_src: u8 = 0;
    {
        var i: usize = 0;
        var prog: usize = 0;
        while (i < kernel.code.len) : (i += 4) {
            if ((kernel.code[i] & 0xfff) == 0xf89) {
                shfl_idx = prog;
                shfl_src = @intCast((kernel.code[i] >> 24) & 0xff);
                break;
            }
            prog += 1;
        }
    }
    try testing.expect(shfl_idx != null);
    // The SHFL source must be produced by an IPA (opcode 0x326) and not
    // written again between that IPA and the SHFL.
    var last_ipa: ?usize = null;
    {
        var k: usize = 0;
        var p: usize = 0;
        while (k < kernel.code.len) : (k += 4) {
            if (p >= shfl_idx.?) break;
            const op = kernel.code[k] & 0xfff;
            const dst: u8 = @intCast((kernel.code[k] >> 16) & 0xff);
            if (op == 0x326 and dst == shfl_src) last_ipa = p;
            p += 1;
        }
    }
    try testing.expect(last_ipa != null); // the SHFL source is an interpolated varying
    {
        var k: usize = 0;
        var p: usize = 0;
        while (k < kernel.code.len) : (k += 4) {
            defer p += 1;
            if (p <= last_ipa.?) continue;
            if (p >= shfl_idx.?) break;
            const op = kernel.code[k] & 0x1ff;
            const dst: u8 = @intCast((kernel.code[k] >> 16) & 0xff);
            const writes = switch (op) {
                0x086, 0x047, 0x04d => false, // STG, BRA, EXIT
                else => true,
            };
            // A write to the SHFL source between its IPA and the SHFL would shuffle garbage.
            try testing.expect(!(writes and dst == shfl_src));
        }
    }
}

/// A color-store move is `MOV R<comp>, R<src>` (the 9-bit opcode field
/// == 0x002, dst in 0..3). The bug: an instruction writes R<src> between the
/// move's source's last definition and the move, so the move reads a
/// clobbered value. RED ended up reading GREEN's data because the allocator
/// reused the register. This asserts that for each color move, nothing
/// writes its source register in the window from that source's last write
/// before the move, up to but not including the move.
fn assertNoColorClobber(kernel: *const Kernel) !void {
    // Walk the stream once. For every color move, find its source's most
    // recent writer and ensure no later writer clobbers it before the move
    // executes. Also assert the four color sources are mutually distinct
    // registers at their move points.
    var move_src: [4]?u8 = .{ null, null, null, null };
    var move_idx: [4]usize = .{ 0, 0, 0, 0 };
    var prog: usize = 0;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        const op = kernel.code[i] & 0x1ff;
        const dst: u8 = @intCast((kernel.code[i] >> 16) & 0xff);
        if (op == 0x002 and dst < 4) {
            move_src[dst] = @intCast(kernel.code[i + 1] & 0xff);
            move_idx[dst] = prog;
        }
        prog += 1;
    }
    for (move_src) |s| try testing.expect(s != null);
    // For each color move, the source register must hold the value
    // produced for that component. That is, no instruction between the
    // producing write and the move writes the source register, which would
    // be a clobber. The producing write is the last write to the source
    // strictly before the move.
    for (0..4) |ci| {
        const src = move_src[ci].?;
        const mv = move_idx[ci];
        // The last writer of `src` strictly before the move.
        var last_writer: ?usize = null;
        var p: usize = 0;
        var k: usize = 0;
        while (k < kernel.code.len) : (k += 4) {
            if (p >= mv) break;
            const op = kernel.code[k] & 0x1ff;
            const dst: u8 = @intCast((kernel.code[k] >> 16) & 0xff);
            // Instructions that write a GPR dst. This excludes stores,
            // branches, and exit. The color moves themselves are fine to
            // count as writers of R0..R3, not src.
            const writes = switch (op) {
                0x086, 0x047, 0x04d => false, // STG, BRA, EXIT (low 9 bits)
                else => true,
            };
            if (writes and dst == src and dst != encode.RZ) last_writer = p;
            p += 1;
        }
        try testing.expect(last_writer != null); // the value was produced into src
        // No intervening writer of `src` between last_writer and the move.
        p = 0;
        k = 0;
        while (k < kernel.code.len) : (k += 4) {
            defer p += 1;
            if (p <= last_writer.?) continue;
            if (p >= mv) break;
            const op = kernel.code[k] & 0x1ff;
            const dst: u8 = @intCast((kernel.code[k] >> 16) & 0xff);
            const writes = switch (op) {
                0x086, 0x047, 0x04d => false,
                else => true,
            };
            // A write to `src` here would clobber the color value before its move reads it.
            try testing.expect(!(writes and dst == src));
        }
    }
    // Mutually distinct color sources. Overlapping liveness would give the
    // same register, which is the bug.
    for (0..4) |a| for (a + 1..4) |c| {
        try testing.expect(move_src[a].? != move_src[c].?);
    };
}

test "graphics: a texturing fragment shader lowers the host-sampler call to a TEX" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u128_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 128 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();

    // Entry params, in exactly the order the SPIR-V image-sample lowering
    // produces for a fragment shader `o = texture(tex, uv)`: the two
    // interpolated uv components (attribute inputs), then the
    // combined-image-sampler descriptor (tagged sampler_desc), then the
    // host sampler-fn pointer (tagged sampler_fn, appended last and lazily).
    const u = try func.appendBlockParam(b, f32_t);
    try func.addAttr(.{ .value = u }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = encode.ATTR_GENERIC0 } } });
    const v = try func.appendBlockParam(b, f32_t);
    try func.addAttr(.{ .value = v }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = encode.ATTR_GENERIC0 + 4 } } });
    const desc = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = desc }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_desc", .value = .{ .int = 1 } } });
    const sampler_fn = try func.appendBlockParam(b, ptr_t);
    try func.addAttr(.{ .value = sampler_fn }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "sampler_fn", .value = .flag } });

    // out_ptr alloca (vec4 RGBA slot), the sampler call, then 4 reloads stored to color.
    const out_ptr = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = u128_t } });
    const lod0 = try func.appendInst(b, f32_t, .{ .fconst = 0 });
    _ = try func.appendStmtRaw(b, .{ .call_indirect = .{
        .target = sampler_fn,
        .args = try func.internValues(&.{ desc, u, v, lod0, out_ptr }),
    } });
    var c: u8 = 0;
    while (c < 4) : (c += 1) {
        const eptr = if (c == 0) out_ptr else blk: {
            const off = try func.appendInst(b, i32_t, .{ .iconst = @as(i64, c) * 4 });
            break :blk try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = out_ptr, .rhs = off } });
        };
        const comp = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = eptr } });
        const color_slot = try func.appendInst(b, i32_t, .{ .iconst = c });
        try func.addAttr(.{ .value = color_slot }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "color_out", .value = .{ .int = c } } });
        try func.appendStore(b, comp, color_slot);
    }
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileShader(allocator, &func, .fragment, nvidia_abi);
    defer kernel.deinit(allocator);

    // The compiled fragment shader must load the bindless handle from the
    // constant bank (LDC), emit exactly one TEX, and have no LDG, since the
    // four reloads are register copies from the TEX result, not global
    // loads. The two uv inputs are IPA'd (fragment interpolation).
    var has_ldc = false;
    var tex_count: usize = 0;
    var ldg_count: usize = 0;
    var ipa_count: usize = 0;
    var tex_idx: ?usize = null;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0xb82 => has_ldc = true, // LDC (handle from constant bank)
            encode.TEX_OPCODE => {
                tex_count += 1;
                tex_idx = i;
            },
            0x981 => ldg_count += 1, // LDG
            0x326 => ipa_count += 1, // IPA
            else => {},
        }
    }
    try testing.expect(has_ldc);
    try testing.expectEqual(@as(usize, 1), tex_count);
    try testing.expectEqual(@as(usize, 0), ldg_count); // reloads are register copies, no LDG
    try testing.expectEqual(@as(usize, 2), ipa_count); // u and v interpolated
    // The TEX carries the bindless marker (bit 91, word 2 bit 27), 2D dimension, RGBA mask.
    const t = tex_idx.?;
    try testing.expectEqual(@as(u32, 1), (kernel.code[t + 2] >> 27) & 1); // bindless bit 91
    try testing.expectEqual(@as(u32, 1), (kernel.code[t + 1] >> 29) & 0x7); // dim _2D at bit 61
    try testing.expectEqual(@as(u32, 0xf), (kernel.code[t + 2] >> 8) & 0xf); // channel mask at bit 72
}

test "a boolean-valued && (bit_and of two bool compares) lowers to PLOP3, not a GPR LOP3" {
    // The shared SPIR-V lowering emits `LogicalAnd`/`LogicalOr`/`LogicalNot`
    // as a bool-typed `.binary` (bit_and/bit_or/bit_xor). The allocator
    // gives a bool a predicate register, so the result must combine the
    // source predicates with PLOP3, the predicate-logic op, opcode 0x81c,
    // not the integer GPR LOP3. The GPR path would call `gprOf` on a
    // predicate and hit `unreachable`, which was the glmark2 light-phong
    // panic.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f32_t);
    const outp = try func.appendBlockParam(b, ptr_t);
    const zero = try func.appendInst(b, f32_t, .{ .fconst = 0.0 });
    const one = try func.appendInst(b, f32_t, .{ .fconst = 1.0 });
    const c1 = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = zero } });
    const c2 = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = one } });
    // bool both = c1 && c2 (LogicalAnd becomes bit_and of bools). A boolean value consumed by select.
    const both = try func.appendInst(b, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = c1, .rhs = c2 } });
    const sel = try func.appendInst(b, f32_t, .{ .select = .{ .cond = both, .then = one, .@"else" = zero } });
    try func.appendStore(b, sel, outp);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);
    var saw_plop3 = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff == 0x81c) saw_plop3 = true; // PLOP3 warp form
    }
    try testing.expect(saw_plop3);
}

test "a boolean-valued NOT (bit_xor bool, -1) lowers to PLOP3 (predicate negation)" {
    // LogicalNot lowers to `bool ^ -1` (an arith_imm bit_xor). The bool
    // result is a predicate, so it negates through PLOP3 (`p ^ PT`), not
    // the GPR immediate-xor path.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, f32_t);
    const outp = try func.appendBlockParam(b, ptr_t);
    const zero = try func.appendInst(b, f32_t, .{ .fconst = 0.0 });
    const one = try func.appendInst(b, f32_t, .{ .fconst = 1.0 });
    const c = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = zero } });
    const nc = try func.appendInst(b, bool_t, .{ .arith_imm = .{ .op = .bit_xor, .lhs = c, .imm = -1 } });
    const sel = try func.appendInst(b, f32_t, .{ .select = .{ .cond = nc, .then = one, .@"else" = zero } });
    try func.appendStore(b, sel, outp);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);
    var saw_plop3 = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff == 0x81c) saw_plop3 = true;
    }
    try testing.expect(saw_plop3);
}

/// The opcode of the instruction at index `i` in a compiled kernel's dword stream.
fn opAt(code: []const u32, i: usize) u32 {
    return code[i * 4] & 0xfff;
}

/// The 8-bit register field at bit `lo` of the instruction at index `i`.
fn regAt(code: []const u32, i: usize, comptime lo: usize) u8 {
    return @truncate(code[i * 4 + lo / 32] >> (lo % 32));
}

test "a shared pointer parameter is 32 bits: ONE LDC, and its accesses are LDS and STS" {
    // The two things this pins, both of them silent miscompiles if they regress:
    //   - a shared address takes ONE register, not an aligned pair, so the prologue reads
    //     ONE dword out of the parameter block and the allocator gives it one register;
    //   - a load or store through it reaches the CTA's shared window with LDS/STS, not the
    //     global memory with LDG/STG.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    const tile = try func.appendBlockParam(b, shared_t);
    const n = try func.appendBlockParam(b, i32_t);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = tile } });
    const sum = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = n } });
    try func.appendStore(b, sum, tile);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    // LDC.64 outptr, LDC tile, LDC n, LDS, IADD3, STS, STG, EXIT.
    // The tile pointer contributes ONE LDC. A 64-bit pointer would read a PAIR, which is
    // what the outptr LDC.64 does in one instruction.
    try testing.expectEqual(@as(usize, 8 * 4), kernel.code.len);
    try testing.expectEqual(@as(u32, 0xb82), opAt(kernel.code, 0)); // outptr, one LDC.64
    try testing.expectEqual(@as(u32, 0xb82), opAt(kernel.code, 1)); // tile, the ONLY one
    try testing.expectEqual(@as(u32, 0xb82), opAt(kernel.code, 2)); // n
    try testing.expectEqual(@as(u32, 0x984), opAt(kernel.code, 3)); // LDS, not LDG (0x981)
    try testing.expectEqual(@as(u32, 0x210), opAt(kernel.code, 4)); // IADD3
    try testing.expectEqual(@as(u32, 0x988), opAt(kernel.code, 5)); // STS, not STG (0x986)
    try testing.expectEqual(@as(u32, 0x986), opAt(kernel.code, 6)); // STG: the return, still global
    try testing.expectEqual(@as(u32, 0x94d), opAt(kernel.code, 7)); // EXIT

    // The outptr LDC.64 fills a PAIR and the tile LDC fills ONE register, so a b64 width
    // field on the first and a b32 on the second is what keeps the two apart.
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.b64)), (kernel.code[2] >> (73 - 64)) & 0x7);
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.b32)), (kernel.code[6] >> (73 - 64)) & 0x7);

    // The tile parameter occupies exactly ONE register: the next parameter takes the very
    // next register. A pair would have pushed `n` one further along.
    const r_tile = regAt(kernel.code, 1, 16); // LDC dst
    const r_n = regAt(kernel.code, 2, 16);
    try testing.expectEqual(r_tile + 1, r_n);
    try testing.expectEqual(@as(u8, value_reg_base), r_tile);

    // LDS reads the shared window offset out of that one register at bit 24, and writes the
    // loaded value into the register the STG then returns.
    const r_v = regAt(kernel.code, 3, 16); // LDS dst
    try testing.expectEqual(r_tile, regAt(kernel.code, 3, 24)); // LDS address
    try testing.expectEqual(@as(u8, encode.URZ), regAt(kernel.code, 3, 32)); // no uniform base
    // STS addresses the same register and stores the IADD3 result at bit 32.
    try testing.expectEqual(r_tile, regAt(kernel.code, 5, 24)); // STS address
    try testing.expectEqual(regAt(kernel.code, 4, 16), regAt(kernel.code, 5, 32)); // STS data
    try testing.expectEqual(r_v, regAt(kernel.code, 6, 32)); // STG stores the loaded value

    // The runtime is told the space, so it binds shared memory rather than a buffer.
    try testing.expectEqual(@as(usize, 2), kernel.launch.params.len);
    try testing.expectEqual(gpu.AddressSpace.shared, kernel.launch.params[0].kind.pointer);
    try testing.expectEqual(@as(u8, 4), kernel.launch.params[1].kind.scalar);
}

test "shared address arithmetic is one 32-bit IADD3, where a global address is a carry chain" {
    // A shared address cannot carry out of 32 bits, so `tile + i` is a plain integer add.
    // The same shape on a global pointer stays the two-instruction carry chain.
    const allocator = testing.allocator;

    const Case = struct {
        fn compile(alloc: std.mem.Allocator, space: ir.types.AddressSpace) !Kernel {
            var func = Function.init(alloc);
            defer func.deinit();
            const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
            const ptr_t = try func.types.intern(.{ .ptr = space });
            const b = try func.appendBlock();
            const base_ptr = try func.appendBlockParam(b, ptr_t);
            const i = try func.appendBlockParam(b, i32_t);
            const elem = try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base_ptr, .rhs = i } });
            const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = elem } });
            func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });
            return compileKernel(alloc, &func, nvidia_abi);
        }
    };

    var shared_k = try Case.compile(allocator, .shared);
    defer shared_k.deinit(allocator);
    var global_k = try Case.compile(allocator, .global);
    defer global_k.deinit(allocator);

    const count = struct {
        fn of(code: []const u32, opcode: u32) usize {
            var n: usize = 0;
            var i: usize = 0;
            while (i < code.len) : (i += 4) {
                if (code[i] & 0xfff == opcode) n += 1;
            }
            return n;
        }
    }.of;

    // Shared: LDC.64 outptr, LDC tile, LDC i, IADD3, LDS, STG, EXIT.
    try testing.expectEqual(@as(usize, 7 * 4), shared_k.code.len);
    try testing.expectEqual(@as(usize, 1), count(shared_k.code, 0x210)); // ONE IADD3
    try testing.expectEqual(@as(usize, 1), count(shared_k.code, 0x984)); // LDS
    try testing.expectEqual(@as(usize, 0), count(shared_k.code, 0x981)); // no LDG
    // The LDS reads the register the single IADD3 wrote.
    try testing.expectEqual(@as(u32, 0x210), opAt(shared_k.code, 3));
    try testing.expectEqual(@as(u32, 0x984), opAt(shared_k.code, 4));
    try testing.expectEqual(regAt(shared_k.code, 3, 16), regAt(shared_k.code, 4, 24));

    // Global: the pointer is a PAIR, so it is one LDC.64 and one IMAD.WIDE. The wide
    // multiply-add replaced the IADD3 carry-out plus IADD3.X pair, and it keeps the carry
    // inside one instruction, so no plain IADD3 forms the address at all.
    try testing.expectEqual(@as(usize, 0), count(global_k.code, 0x210)); // no 32-bit address add
    try testing.expectEqual(@as(usize, 1), count(global_k.code, encode.IMAD_WIDE_IMM_OPCODE));
    try testing.expectEqual(@as(usize, 1), count(global_k.code, 0x981)); // LDG
    try testing.expectEqual(@as(usize, 0), count(global_k.code, 0x984)); // no LDS
    try testing.expectEqual(@as(usize, 3), count(global_k.code, 0xb82)); // outptr, base, i
}

test "a graphics stage rejects a shared pointer parameter instead of interpolating it" {
    // A graphics stage has no workgroup and therefore no workgroup shared memory. Before the
    // address-space split, `isPtr` sent this parameter down the UBO path and read a 64-bit
    // address out of the graphics constant bank for it.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    const tile = try func.appendBlockParam(b, shared_t);
    const v = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = tile } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    try testing.expectError(
        error.Unsupported,
        compileShader(allocator, &func, .fragment, nvidia_abi),
    );
}

/// The index of the first instruction with `opcode` in a compiled kernel's dword stream, or
/// null when there is none.
fn findOp(code: []const u32, opcode: u32) ?usize {
    var i: usize = 0;
    while (i * 4 < code.len) : (i += 1) if (opAt(code, i) == opcode) return i;
    return null;
}

/// The 3-bit memory type field (bits 73..76) of the instruction at index `i`.
fn memTypeAt(code: []const u32, i: usize) u32 {
    return (code[i * 4 + 2] >> (73 - 64)) & 0x7;
}

test "a byte access uses the 8-bit memory type, not a 32-bit one" {
    // Regression: the load and store arms ignored the IR value type and always emitted the
    // B32 encoders. A byte store then wrote FOUR bytes, destroying the three bytes beside its
    // own, and a byte load read four and kept whatever the neighbours held in the top 24 bits.
    // Neither produced a diagnostic. The signed type picks I8 so the value sign-extends into
    // the whole destination register.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendInst(b, i8_t, .{ .load = .{ .ptr = p } });
    try func.appendStore(b, v, p);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    const ld = findOp(kernel.code, 0x981).?; // LDG
    const st = findOp(kernel.code, 0x986).?; // STG
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.i8)), memTypeAt(kernel.code, ld));
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.i8)), memTypeAt(kernel.code, st));
    // The address is still the 64-bit pointer pair, and the data register is still at bit 32.
    try testing.expectEqual(regAt(kernel.code, ld, 24), regAt(kernel.code, st, 24));
    try testing.expectEqual(regAt(kernel.code, ld, 16), regAt(kernel.code, st, 32));
}

test "an unsigned 16-bit access uses U16 and a 32-bit one is unchanged" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const half = try func.appendInst(b, u16_t, .{ .load = .{ .ptr = p } });
    try func.appendStore(b, half, p);
    const word = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = p } });
    try func.appendStore(b, word, p);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    // Two LDG/STG pairs in source order: the 16-bit one first, then the 32-bit one.
    const first_ld = findOp(kernel.code, 0x981).?;
    const first_st = findOp(kernel.code, 0x986).?;
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.u16)), memTypeAt(kernel.code, first_ld));
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.u16)), memTypeAt(kernel.code, first_st));
    const second_ld = findOp(kernel.code[(first_ld + 1) * 4 ..], 0x981).? + first_ld + 1;
    const second_st = findOp(kernel.code[(first_st + 1) * 4 ..], 0x986).? + first_st + 1;
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.b32)), memTypeAt(kernel.code, second_ld));
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.b32)), memTypeAt(kernel.code, second_st));
}

test "a shared byte access uses the 8-bit memory type on LDS and STS too" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    const tile = try func.appendBlockParam(b, shared_t);
    const v = try func.appendInst(b, u8_t, .{ .load = .{ .ptr = tile } });
    try func.appendStore(b, v, tile);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    const ld = findOp(kernel.code, 0x984).?; // LDS
    const st = findOp(kernel.code, 0x988).?; // STS
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.u8)), memTypeAt(kernel.code, ld));
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.u8)), memTypeAt(kernel.code, st));
    // A shared access still reads ONE address register, and the STS data is at bit 32.
    try testing.expectEqual(regAt(kernel.code, ld, 24), regAt(kernel.code, st, 24));
    try testing.expectEqual(regAt(kernel.code, ld, 16), regAt(kernel.code, st, 32));
}

test "a pointer load and store move BOTH halves of the address pair (B64)" {
    // A pointer value owns an aligned GPR pair, so the 64-bit width fills exactly the pair
    // the allocator reserved. Before the width came from the value type, this loaded only the
    // low dword and left the high dword holding whatever the register had, so the next access
    // through that pointer went to a garbage address.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const pp = try func.appendBlockParam(b, ptr_t);
    const inner = try func.appendInst(b, ptr_t, .{ .load = .{ .ptr = pp } });
    try func.appendStore(b, inner, pp);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    const ld = findOp(kernel.code, 0x981).?;
    const st = findOp(kernel.code, 0x986).?;
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.b64)), memTypeAt(kernel.code, ld));
    try testing.expectEqual(@as(u32, @intFromEnum(encode.MemType.b64)), memTypeAt(kernel.code, st));
    // The loaded pointer lands in an EVEN register, so (dst, dst+1) is an aligned pair.
    try testing.expectEqual(@as(u8, 0), regAt(kernel.code, ld, 16) % 2);
}

test "a 64-bit scalar access is REFUSED, not silently truncated" {
    // The value model gives every non-pointer, non-boolean value exactly ONE 32-bit register,
    // and this backend has no 64-bit scalar arithmetic. A B64 load would write a register the
    // allocator gave to a different live value, and a B32 load would keep only the low half.
    // Refusing is the only answer that does not produce a kernel which looks right and
    // computes garbage.
    const allocator = testing.allocator;
    const cases = [_]ir.types.TypeKind{
        .{ .int = .{ .signedness = .signed, .bits = 64 } },
        .{ .float = .f64 },
    };
    for (cases) |kind| {
        var func = Function.init(allocator);
        defer func.deinit();
        const wide_t = try func.types.intern(kind);
        const ptr_t = try func.types.ptrGlobal();
        const b = try func.appendBlock();
        const p = try func.appendBlockParam(b, ptr_t);
        const v = try func.appendInst(b, wide_t, .{ .load = .{ .ptr = p } });
        try func.appendStore(b, v, p);
        func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

        try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
    }
}

test "a workgroup barrier lowers to BAR.SYNC and declares one control barrier" {
    // The launch descriptor half matters as much as the instruction: the NVIDIA QMD has a
    // BARRIER_COUNT field, and a dispatch that leaves it at 0 while the kernel runs a
    // BAR.SYNC is UNDEFINED.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    const tile = try func.appendBlockParam(b, shared_t);
    const n = try func.appendBlockParam(b, i32_t);
    try func.appendStore(b, n, tile);
    try func.appendBarrier(b, .workgroup);
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = tile } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    var bars: usize = 0;
    var bar_at: usize = 0;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff == 0xb1d) {
            bars += 1;
            bar_at = i / 4;
        }
    }
    try testing.expectEqual(@as(usize, 1), bars);

    // The BAR.SYNC sits between the STS and the LDS, in program order.
    try testing.expectEqual(@as(u32, 0x988), opAt(kernel.code, bar_at - 1)); // STS
    try testing.expectEqual(@as(u32, 0x984), opAt(kernel.code, bar_at + 1)); // LDS

    try testing.expectEqual(@as(u32, 1), kernel.launch.barrier_count);
}

test "a kernel with no barrier declares zero control barriers" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const n = try func.appendBlockParam(b, i32_t);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(n) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);
    try testing.expectEqual(@as(u32, 0), kernel.launch.barrier_count);
}

test "a subgroup barrier is refused, not lowered to a workgroup BAR.SYNC" {
    // BAR.SYNC makes every warp of the CTA wait. Using it for a warp-scope request would
    // deadlock a kernel whose other warps never reach the barrier, so the backend refuses.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const n = try func.appendBlockParam(b, i32_t);
    try func.appendBarrier(b, .subgroup);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(n) });

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

test "a barrier INSIDE a divergent arm is refused" {
    // The Blackwell quirk, measured on the GB10: a divergent branch around a BAR.SYNC
    // corrupts a staged shared-memory tile. The BSSY/BSYNC pair this backend emits does not
    // save it, because BSYNC sits at the JOIN, after the arm's body. The other arm reaches
    // the merge without running the barrier, so this placement is refused rather than
    // trusted to a frontend that may not have predicated the guard.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const r = try func.appendBlockParam(merge, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    try func.appendBarrier(then_b, .workgroup); // only ONE arm runs it
    const one = try func.appendInst(then_b, t, .{ .iconst = 1 });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{one}) } });
    const two = try func.appendInst(else_b, t, .{ .iconst = 2 });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{two}) } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(r) });

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

test "a barrier BEFORE and AFTER a divergent region is accepted" {
    // The same CFG, with the barrier moved out of the arm. Before the branch the warp is
    // still whole, and at the merge the BSYNC has reconverged it, so both placements are
    // safe and both must compile. This is the case vulcan is genuinely better off in than a
    // hand assembler with no reconvergence markers.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const r = try func.appendBlockParam(merge, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try func.appendBarrier(entry, .workgroup); // before the branch
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    const one = try func.appendInst(then_b, t, .{ .iconst = 1 });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{one}) } });
    const two = try func.appendInst(else_b, t, .{ .iconst = 2 });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{two}) } });
    try func.appendBarrier(merge, .workgroup); // at the join, after the BSYNC
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(r) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    var bars: usize = 0;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        if (kernel.code[i] & 0xfff == 0xb1d) bars += 1;
    }
    try testing.expectEqual(@as(usize, 2), bars);
    try testing.expectEqual(@as(u32, 1), kernel.launch.barrier_count);
}

/// Build the tiled-matmul control shape into `func`: a counted loop whose body stages a tile,
/// waits, computes, and waits again. `trips` is the trip-count value the loop compares against,
/// which is what decides whether the loop is uniform. The caller owns `entry` and supplies
/// `trips`, so each test below changes only where the trip count comes from.
fn buildTileLoop(func: *Function, trips: Value, entry: Block) !void {
    const t = func.valueType(trips);
    const bool_t = try func.types.intern(.bool);
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const i = try func.appendBlockParam(head, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    func.setTerminator(entry, .{ .jump = .{ .target = head, .args = try func.internValues(&.{zero}) } });
    const c = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = trips } });
    try func.appendIf(head, c, .{ .target = body, .args = &.{} }, .{ .target = done, .args = &.{} });
    try func.appendBarrier(body, .workgroup); // the tile is staged, wait for every thread
    const acc = try func.appendArithImm(body, t, .mul, i, 3); // stands in for the tile compute
    try func.appendBarrier(body, .workgroup); // the tile is consumed, wait before overwriting it
    const next = try func.appendArithImm(body, t, .add, acc, 1);
    func.setTerminator(body, .{ .jump = .{ .target = head, .args = try func.internValues(&.{next}) } });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(trips) });
}

/// How many BAR.SYNC instructions a compiled kernel holds.
fn countBarSync(code: []const u32) usize {
    var bars: usize = 0;
    var i: usize = 0;
    while (i < code.len) : (i += 4) {
        if (code[i] & 0xfff == 0xb1d) bars += 1;
    }
    return bars;
}

test "a barrier in a LOOP body is accepted when the trip count is a kernel parameter" {
    // The tiled-matmul shape, and the reason the uniformity analysis exists. The trip count is
    // a scalar kernel parameter, so every thread of the workgroup reads the same number of
    // tiles and makes the same number of trips. No thread leaves the loop while another waits,
    // so both barriers in the body are safe.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const tiles = try func.appendBlockParam(entry, t);
    try buildTileLoop(&func, tiles, entry);

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);
    try testing.expectEqual(@as(usize, 2), countBarSync(kernel.code));
    try testing.expectEqual(@as(u32, 1), kernel.launch.barrier_count);
}

test "a barrier in a LOOP body is refused when the trip count comes from thread_id_x" {
    // The negative control for the acceptance above: the SAME shape, with the trip count tagged
    // as the thread index instead of read from the parameter block. Threads then leave the loop
    // at different trips, and a thread that leaves early skips a barrier the others still wait
    // at. A check that accepted every loop would pass the test above and fail this one.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    try buildTileLoop(&func, tid, entry);

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

test "a barrier in a LOOP body is refused when the trip count is LOADED from memory" {
    // The conservative choice in `opt.uniform`, seen from the backend. Nothing proves that no
    // divergent thread wrote to the address, so the loaded tile count is not uniform and the
    // loop keeps the refusal. Passing the count as a scalar parameter is the way through.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const tiles = try func.appendInst(entry, t, .{ .load = .{ .ptr = p } });
    try buildTileLoop(&func, tiles, entry);

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

test "a barrier in a LOOP body is refused when the trip count arrives as a block parameter" {
    // The same refusal as the `thread_id_x` case, reached through an EDGE instead of an operand.
    // The bound is handed to the loop header as a block argument, so nothing inside the header
    // names the thread index. An analysis that only walks operands would read the bound as
    // uniform here and admit a barrier that half the workgroup walks away from.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const i = try func.appendBlockParam(head, t);
    const bound = try func.appendBlockParam(head, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    func.setTerminator(entry, .{ .jump = .{ .target = head, .args = try func.internValues(&.{ zero, tid }) } });
    const c = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = bound } });
    try func.appendIf(head, c, .{ .target = body, .args = &.{} }, .{ .target = done, .args = &.{} });
    try func.appendBarrier(body, .workgroup);
    const next = try func.appendArithImm(body, t, .add, i, 1);
    func.setTerminator(body, .{ .jump = .{ .target = head, .args = try func.internValues(&.{ next, bound }) } });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(bound) });

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

test "a UNIFORM if inside a uniform loop is still refused around a barrier" {
    // The exemption stops at the loop's own exit branch. This `if` sits in the body with both
    // edges inside the loop, and its condition is a pair of kernel parameters, so the whole
    // workgroup does take the same arm and the placement is in fact safe. The backend refuses it
    // all the same, exactly as it refuses the same shape outside a loop. Widening the acceptance
    // to a plain `if` is a separate decision, and this test pins where the line sits today.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const tiles = try func.appendBlockParam(entry, t);
    const limit = try func.appendBlockParam(entry, t);
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const guarded = try func.appendBlock();
    const latch = try func.appendBlock();
    const done = try func.appendBlock();
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    func.setTerminator(entry, .{ .jump = .{ .target = head, .args = try func.internValues(&.{zero}) } });
    const i = try func.appendBlockParam(head, t);
    const c = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = tiles } });
    try func.appendIf(head, c, .{ .target = body, .args = &.{} }, .{ .target = done, .args = &.{} });
    const guard = try func.appendInst(body, bool_t, .{ .icmp = .{ .op = .gt, .lhs = limit, .rhs = i } });
    try func.appendIf(body, guard, .{ .target = guarded, .args = &.{} }, .{ .target = latch, .args = &.{} });
    try func.appendBarrier(guarded, .workgroup);
    try func.setJump(guarded, latch, &.{});
    const next = try func.appendArithImm(latch, t, .add, i, 1);
    func.setTerminator(latch, .{ .jump = .{ .target = head, .args = try func.internValues(&.{next}) } });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(tiles) });

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

test "a uniform loop nested inside a divergent guard is still refused" {
    // The exemption covers the loop's own exit branch and nothing else. The guard around the
    // whole loop is a separate divergent branch, and the threads it turns away reach the join
    // without running any of the loop's barriers.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    const tiles = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const pre = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const four = try func.appendInst(entry, t, .{ .iconst = 4 });
    const guard = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = four } });
    try func.appendIf(entry, guard, .{ .target = pre, .args = &.{} }, .{ .target = done, .args = &.{} });
    const zero = try func.appendInst(pre, t, .{ .iconst = 0 });
    func.setTerminator(pre, .{ .jump = .{ .target = head, .args = try func.internValues(&.{zero}) } });
    const i = try func.appendBlockParam(head, t);
    const c = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = tiles } });
    try func.appendIf(head, c, .{ .target = body, .args = &.{} }, .{ .target = done, .args = &.{} });
    try func.appendBarrier(body, .workgroup);
    const next = try func.appendArithImm(body, t, .add, i, 1);
    func.setTerminator(body, .{ .jump = .{ .target = head, .args = try func.internValues(&.{next}) } });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(tiles) });

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

test "a divergent guard AROUND the barrier inside a uniform loop is still refused" {
    // The loop is uniform, so its own exit branch is exempt. The `if (tid < 4)` inside the body
    // is not, and the threads it turns away reach the latch without running the barrier. Each
    // branch is judged on its own, so the exemption on one does not cover the other.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    const tiles = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const guarded = try func.appendBlock();
    const latch = try func.appendBlock();
    const done = try func.appendBlock();
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    func.setTerminator(entry, .{ .jump = .{ .target = head, .args = try func.internValues(&.{zero}) } });
    const i = try func.appendBlockParam(head, t);
    const c = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = tiles } });
    try func.appendIf(head, c, .{ .target = body, .args = &.{} }, .{ .target = done, .args = &.{} });
    const four = try func.appendInst(body, t, .{ .iconst = 4 });
    const guard = try func.appendInst(body, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = four } });
    try func.appendIf(body, guard, .{ .target = guarded, .args = &.{} }, .{ .target = latch, .args = &.{} });
    try func.appendBarrier(guarded, .workgroup); // only some threads run it
    try func.setJump(guarded, latch, &.{});
    const next = try func.appendArithImm(latch, t, .add, i, 1);
    func.setTerminator(latch, .{ .jump = .{ .target = head, .args = try func.internValues(&.{next}) } });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(tiles) });

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

/// The 4-bit field at bit `lo` of the instruction at index `i` in a compiled kernel's dword
/// stream. The atomic operation selector and the atomic type both live in such a field.
fn nibbleAt(code: []const u32, i: usize, comptime lo: usize) u32 {
    return (code[i * 4 + lo / 32] >> (lo % 32)) & 0xf;
}

/// Find the single instruction with opcode `want` in a compiled kernel, and fail if there is
/// not exactly one. Every atomic test below wants exactly one.
fn onlyOpAt(code: []const u32, want: u32) !usize {
    var found: ?usize = null;
    var i: usize = 0;
    while (i * 4 < code.len) : (i += 1) {
        if (opAt(code, i) == want) {
            try testing.expect(found == null); // exactly one, never two
            found = i;
        }
    }
    return found orelse error.NotFound;
}

test "a global atomic whose result is READ lowers to ATOMG with the right operation" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, i32_t);
    const old = try func.appendAtomicRmw(b, .{ .op = .min, .ptr = p, .value = v, .ordering = .seq_cst, .scope = .device });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(old) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    const at = try onlyOpAt(kernel.code, encode.ATOMG_OPCODE);
    // The operation selector is the 4-bit field at bit 87, and `min` is 1.
    try testing.expectEqual(@intFromEnum(encode.AtomOp.min), nibbleAt(kernel.code, at, 87));
    // The atomic TYPE is the 4-bit field at bit 73, and a signed 32-bit operand is `i32`.
    // Signedness comes from the operand type alone, and it is what makes `min` a signed
    // comparison.
    try testing.expectEqual(@intFromEnum(encode.AtomType.i32), nibbleAt(kernel.code, at, 73));
    // No RED was emitted: the reading form is the ATOMG one.
    try testing.expectError(error.NotFound, onlyOpAt(kernel.code, encode.RED_OPCODE));
}

test "a global atomic whose result is UNREAD lowers to RED, which claims no scoreboard" {
    // This is the whole reason the IR result is optional. RED writes no register, so the
    // scheduler hands it no write barrier, and a fire-and-forget counter increment costs
    // none of the six scoreboards. The test above is the control: the same operation with
    // its result read produces ATOMG instead.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, u32_t);
    try func.appendAtomicRmwStmt(b, .{ .op = .add, .ptr = p, .value = v, .ordering = .relaxed, .scope = .device });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    const at = try onlyOpAt(kernel.code, encode.RED_OPCODE);
    // RED's operation selector is only 3 bits wide, and `add` is 0.
    try testing.expectEqual(@as(u32, @intFromEnum(encode.AtomOp.add)), (kernel.code[at * 4 + 2] >> (87 - 64)) & 0x7);
    // An unsigned operand selects the unsigned atomic type.
    try testing.expectEqual(@intFromEnum(encode.AtomType.u32), nibbleAt(kernel.code, at, 73));
    // The destination field reads RZ, so no consumer can wait on it and the scheduler
    // records no write.
    try testing.expectEqual(encode.RZ, regAt(kernel.code, at, 16));
    // No ATOMG was emitted.
    try testing.expectError(error.NotFound, onlyOpAt(kernel.code, encode.ATOMG_OPCODE));
}

test "an unread global EXCHANGE lowers to ATOMG with an RZ destination, not RED" {
    // RED's operation field is 3 bits, so `exch` (8) does not fit it. An unread exchange
    // takes ATOMG with RZ instead, which discards the old value and still claims no
    // scoreboard, rather than truncating the selector to `add`.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, u32_t);
    try func.appendAtomicRmwStmt(b, .{ .op = .exchange, .ptr = p, .value = v, .ordering = .relaxed, .scope = .device });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    const at = try onlyOpAt(kernel.code, encode.ATOMG_OPCODE);
    try testing.expectEqual(@intFromEnum(encode.AtomOp.exch), nibbleAt(kernel.code, at, 87));
    try testing.expectEqual(encode.RZ, regAt(kernel.code, at, 16));
    try testing.expectError(error.NotFound, onlyOpAt(kernel.code, encode.RED_OPCODE));
}

test "a SHARED-pointer atomic lowers to ATOMS, from the pointer type alone" {
    // The address space rides in the pointer type, exactly as it does for LDS and STS, so
    // no new attribute is needed. The global tests above are the control: the same
    // operation through a global pointer produces ATOMG or RED.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    const tile = try func.appendBlockParam(b, shared_t);
    const v = try func.appendBlockParam(b, i32_t);
    const old = try func.appendAtomicRmw(b, .{ .op = .bit_xor, .ptr = tile, .value = v, .ordering = .relaxed, .scope = .workgroup });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(old) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    const at = try onlyOpAt(kernel.code, encode.ATOMS_OPCODE);
    try testing.expectEqual(@intFromEnum(encode.AtomOp.bit_xor), nibbleAt(kernel.code, at, 87));
    try testing.expectEqual(@intFromEnum(encode.AtomType.i32), nibbleAt(kernel.code, at, 73));
    try testing.expectError(error.NotFound, onlyOpAt(kernel.code, encode.ATOMG_OPCODE));
    try testing.expectError(error.NotFound, onlyOpAt(kernel.code, encode.RED_OPCODE));
}

test "an unread SHARED atomic still uses ATOMS, with RZ as its destination" {
    // The encoders have no shared reduction form, so ATOMS is the only shared instruction.
    // NAK writes RZ into the destination field for a `Dst::None` atomic and this matches:
    // the scheduler skips an RZ destination, so this claims no scoreboard either.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    const tile = try func.appendBlockParam(b, shared_t);
    const v = try func.appendBlockParam(b, i32_t);
    try func.appendAtomicRmwStmt(b, .{ .op = .add, .ptr = tile, .value = v, .ordering = .relaxed, .scope = .workgroup });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    const at = try onlyOpAt(kernel.code, encode.ATOMS_OPCODE);
    try testing.expectEqual(encode.RZ, regAt(kernel.code, at, 16));
}

test "a compare-exchange lowers to the CAS opcode of its address space" {
    const allocator = testing.allocator;
    const spaces = [_]struct { space: ir.types.AddressSpace, want: u32 }{
        .{ .space = .global, .want = encode.ATOMG_CAS_OPCODE },
        .{ .space = .shared, .want = encode.ATOMS_CAS_OPCODE },
    };
    for (spaces) |s| {
        var func = Function.init(allocator);
        defer func.deinit();
        const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const ptr_t = try func.types.intern(.{ .ptr = s.space });
        const b = try func.appendBlock();
        const p = try func.appendBlockParam(b, ptr_t);
        const desired = try func.appendBlockParam(b, i32_t);
        const expected = try func.appendBlockParam(b, i32_t);
        const old = try func.appendAtomicRmw(b, .{
            .op = .compare_exchange,
            .ptr = p,
            .value = desired,
            .compare = expected,
            .ordering = .seq_cst,
            .scope = .workgroup,
        });
        func.setTerminator(b, .{ .ret = ir.function.Ret.one(old) });

        var kernel = try compileKernel(allocator, &func, nvidia_abi);
        defer kernel.deinit(allocator);

        const at = try onlyOpAt(kernel.code, s.want);
        try testing.expectEqual(@intFromEnum(encode.AtomType.i32), nibbleAt(kernel.code, at, 73));
        // The compare operand goes in the bit-32 register field and the swap data in the
        // bit-64 one. Swapping the two silently inverts what the instruction does.
        try testing.expectEqual(gprOfTest(&func, expected), regAt(kernel.code, at, 32));
        try testing.expectEqual(gprOfTest(&func, desired), regAt(kernel.code, at, 64));
        // The plain forms were NOT emitted for a compare-exchange.
        try testing.expectError(error.NotFound, onlyOpAt(kernel.code, encode.ATOMG_OPCODE));
        try testing.expectError(error.NotFound, onlyOpAt(kernel.code, encode.ATOMS_OPCODE));
    }
}

/// The register a kernel parameter lands in, recomputed the way `assignLocs` does, so the CAS
/// test can name the operand registers without reaching into the isel's private map.
fn gprOfTest(func: *const Function, v: Value) u8 {
    var locs: std.AutoHashMapUnmanaged(Value, Loc) = .empty;
    defer locs.deinit(std.testing.allocator);
    var max_reg: u8 = 0;
    // An empty fold: the atomic kernels this helper serves contain no multiply-add, so
    // nothing here changes when contraction is on.
    const fma = FmaFold{};
    assignLocs(std.testing.allocator, func, &locs, &max_reg, &fma) catch unreachable;
    return gprOf(locs, v);
}

/// The register `compileShaderOpts` gives `v`, recomputed by running the SAME pre-passes in
/// the same order it does. A contraction test needs this and not `gprOfTest`: the fused
/// instruction names the MULTIPLY's operand registers, and keeping those operands live to
/// the add is itself part of what `assignLocs` now decides.
///
/// Every pre-pass here is idempotent, so calling this after `compileKernel` on the same
/// function gives the allocation that kernel used.
fn gprOfCompiled(func: *Function, v: Value, options: Options) u8 {
    const allocator = std.testing.allocator;
    var locs: std.AutoHashMapUnmanaged(Value, Loc) = .empty;
    defer locs.deinit(allocator);
    foldConstantsToImm(func);
    var disp = DispFold{};
    defer disp.deinit(allocator);
    foldAddressDisplacements(allocator, func, &disp) catch unreachable;
    var fma = FmaFold{};
    defer fma.deinit(allocator);
    scanFma(allocator, func, options, &fma) catch unreachable;
    var max_reg: u8 = r_outptr + 1;
    assignLocs(allocator, func, &locs, &max_reg, &fma) catch unreachable;
    return gprOf(locs, v);
}

test "a shared atomic wider than workgroup scope is refused, not silently narrowed" {
    // ATOMS is Strong(CTA) in the hardware. Emitting it for a device- or system-scope
    // request would order less than the program asked for, so it is refused. The
    // workgroup control proves the refusal comes from the scope and nothing else.
    const allocator = testing.allocator;
    const scopes = [_]struct { scope: ir.function.AtomicScope, ok: bool }{
        .{ .scope = .workgroup, .ok = true },
        .{ .scope = .device, .ok = false },
        .{ .scope = .system, .ok = false },
    };
    for (scopes) |s| {
        var func = Function.init(allocator);
        defer func.deinit();
        const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const shared_t = try func.types.intern(.{ .ptr = .shared });
        const b = try func.appendBlock();
        const tile = try func.appendBlockParam(b, shared_t);
        const v = try func.appendBlockParam(b, i32_t);
        try func.appendAtomicRmwStmt(b, .{ .op = .add, .ptr = tile, .value = v, .ordering = .relaxed, .scope = s.scope });
        func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

        if (s.ok) {
            var kernel = try compileKernel(allocator, &func, nvidia_abi);
            defer kernel.deinit(allocator);
            _ = try onlyOpAt(kernel.code, encode.ATOMS_OPCODE);
        } else {
            try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
        }
    }
}

test "an atomic on a narrow integer is refused, not widened to 32 bits" {
    // There is no byte or half-word atomic. Widening the access would read and write the
    // three bytes beside it, which is a data race this backend would have created itself.
    // The i32 control proves the refusal comes from the width.
    const allocator = testing.allocator;
    const widths = [_]struct { bits: u16, ok: bool }{ .{ .bits = 32, .ok = true }, .{ .bits = 8, .ok = false }, .{ .bits = 16, .ok = false } };
    for (widths) |w| {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = w.bits } });
        const ptr_t = try func.types.ptrGlobal();
        const b = try func.appendBlock();
        const p = try func.appendBlockParam(b, ptr_t);
        const v = try func.appendBlockParam(b, t);
        try func.appendAtomicRmwStmt(b, .{ .op = .add, .ptr = p, .value = v, .ordering = .relaxed, .scope = .device });
        func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

        if (w.ok) {
            var kernel = try compileKernel(allocator, &func, nvidia_abi);
            defer kernel.deinit(allocator);
            _ = try onlyOpAt(kernel.code, encode.RED_OPCODE);
        } else {
            try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
        }
    }
}

test "an atomic inside a divergent arm compiles, unlike a barrier" {
    // An atomic in a divergent arm is well defined: the threads that take the arm apply it
    // and the hardware serialises them. `checkBarrierConvergence` refuses the same CFG with
    // a barrier in the arm, and this test pins that the atomic is NOT given that treatment.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const a = try func.appendBlockParam(entry, t);
    const bv = try func.appendBlockParam(entry, t);
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const r = try func.appendBlockParam(merge, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = bv } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    try func.appendAtomicRmwStmt(then_b, .{ .op = .add, .ptr = p, .value = a, .ordering = .relaxed, .scope = .device });
    const one = try func.appendInst(then_b, t, .{ .iconst = 1 });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{one}) } });
    const two = try func.appendInst(else_b, t, .{ .iconst = 2 });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{two}) } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(r) });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);
    _ = try onlyOpAt(kernel.code, encode.RED_OPCODE);
}

test "a compute kernel whose entry a branch reaches refuses, and the forward edge compiles" {
    // `layoutParams` reads the entry parameter list to place every slot and to decide whether
    // the launch-shape region is present, and the prologue walks the same list. `mem2reg`
    // adds and removes block parameters on any block with a predecessor, so the two would
    // stop agreeing. The negative control is the same two blocks with the edge forward.
    const allocator = testing.allocator;
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const ptr_t = try func.types.ptrGlobal();
        const b0 = try func.appendBlock();
        const out = try func.appendBlockParam(b0, ptr_t);
        const v = try func.appendBlockParam(b0, t);
        try gpu.attrs.setBuiltin(&func, v, .grid_dim_x);
        try func.appendStore(b0, v, out);
        const back = try func.appendBlock();
        try func.setJump(b0, back, &.{});
        try func.setJump(back, b0, &.{});
        try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
    }
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const ptr_t = try func.types.ptrGlobal();
        const b0 = try func.appendBlock();
        const out = try func.appendBlockParam(b0, ptr_t);
        const v = try func.appendBlockParam(b0, t);
        try gpu.attrs.setBuiltin(&func, v, .grid_dim_x);
        try func.appendStore(b0, v, out);
        const tail = try func.appendBlock();
        try func.setJump(b0, tail, &.{});
        func.setTerminator(tail, .{ .ret = ir.function.Ret.none() });
        var kernel = try compileKernel(allocator, &func, nvidia_abi);
        kernel.deinit(allocator);
    }
}

/// The 32-bit immediate of the instruction at index `i`. It sits in the second dword, which is
/// where `encode.movImm` writes it.
fn immAt(code: []const u32, i: usize) u32 {
    return code[i * 4 + 1];
}

test "a shared alloca lowers to a MOV of its frame offset, and two of them do not overlap" {
    // The whole shared-alloca lowering in one shape. A shared address is a 32-bit byte offset
    // into the CTA's window, so the address of a slot is the offset the frame assigned and
    // nothing else: no pointer pair, no base register, no LDC. The two slots must land at
    // distinct offsets, or a kernel would stage both tiles on top of each other and read one
    // thread's data as another's.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const tile_t = try func.types.intern(.{ .array = .{ .len = 32, .elem = f32_t } });
    const small_t = try func.types.intern(.{ .array = .{ .len = 16, .elem = i32_t } });
    const b = try func.appendBlock();
    const n = try func.appendBlockParam(b, i32_t);
    const tile = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = tile_t } });
    const small = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = small_t } });
    try func.appendStore(b, n, tile);
    try func.appendStore(b, n, small);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    // LDC n, MOV tile, MOV small, STS, STS, EXIT. No LDC for either slot.
    try testing.expectEqual(@as(usize, 6 * 4), kernel.code.len);
    try testing.expectEqual(@as(u32, 0xb82), opAt(kernel.code, 0)); // LDC n
    try testing.expectEqual(@as(u32, 0x802), opAt(kernel.code, 1)); // MOV immediate
    try testing.expectEqual(@as(u32, 0x802), opAt(kernel.code, 2));
    try testing.expectEqual(@as(u32, 0x988), opAt(kernel.code, 3)); // STS, not STG (0x986)
    try testing.expectEqual(@as(u32, 0x988), opAt(kernel.code, 4));

    // The first slot starts the window and the second follows it, 32 floats along.
    try testing.expectEqual(@as(u32, 0), immAt(kernel.code, 1));
    try testing.expectEqual(@as(u32, 128), immAt(kernel.code, 2));
    // Each STS addresses the register its own MOV wrote, so the two stores reach two slots.
    try testing.expectEqual(regAt(kernel.code, 1, 16), regAt(kernel.code, 3, 24));
    try testing.expectEqual(regAt(kernel.code, 2, 16), regAt(kernel.code, 4, 24));
    try testing.expect(regAt(kernel.code, 1, 16) != regAt(kernel.code, 2, 16));

    // The runtime sizes the CTA window from this, and it is the frame the isel placed.
    try testing.expectEqual(@as(u32, 128 + 64), kernel.launch.shared_bytes);
}

test "a shared alloca's element address is ONE 32-bit IADD3, with no carry chain" {
    // The M1.5 address-space rule, checked on an alloca rather than on a parameter:
    // `tile + i` keeps the shared space, so it stays a plain integer add of the frame offset
    // and its accesses stay LDS and STS. A carry chain here would mean the allocator had given
    // the slot a register PAIR and the second register held a meaningless high word.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const tile_t = try func.types.intern(.{ .array = .{ .len = 8, .elem = i32_t } });
    const b = try func.appendBlock();
    const i = try func.appendBlockParam(b, i32_t);
    const tile = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = tile_t } });
    const elem = try func.appendInst(b, shared_t, .{ .arith = .{ .op = .add, .lhs = tile, .rhs = i } });
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = elem } });
    try func.appendStore(b, v, elem);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);

    // LDC i, MOV tile, IADD3, LDS, STS, EXIT. A global pointer would need two IADD3s here.
    try testing.expectEqual(@as(usize, 6 * 4), kernel.code.len);
    try testing.expectEqual(@as(u32, 0x802), opAt(kernel.code, 1)); // MOV frame offset
    try testing.expectEqual(@as(u32, 0x210), opAt(kernel.code, 2)); // the ONE IADD3
    try testing.expectEqual(@as(u32, 0x984), opAt(kernel.code, 3)); // LDS
    try testing.expectEqual(@as(u32, 0x988), opAt(kernel.code, 4)); // STS
    // The add reads the MOV's register, and both accesses address the add's result.
    try testing.expectEqual(regAt(kernel.code, 1, 16), regAt(kernel.code, 2, 24));
    try testing.expectEqual(regAt(kernel.code, 2, 16), regAt(kernel.code, 3, 24));
    try testing.expectEqual(regAt(kernel.code, 2, 16), regAt(kernel.code, 4, 24));
    try testing.expectEqual(@as(u32, 32), kernel.launch.shared_bytes);
}

test "a shared frame larger than the target window is refused, not truncated" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const big_t = try func.types.intern(.{ .array = .{ .len = 64 * 1024, .elem = f32_t } });
    const b = try func.appendBlock();
    _ = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = big_t } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    try testing.expectError(error.SharedMemoryTooLarge, compileKernel(allocator, &func, nvidia_abi));
}

test "a declared shared_bytes that disagrees with the assigned frame is refused" {
    // The frontend says 64 bytes and the frame places 128. Reporting either number would give
    // the runtime a window the kernel does not have, and a shared access past the end of a CTA
    // window is not faulted on this hardware.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const tile_t = try func.types.intern(.{ .array = .{ .len = 32, .elem = f32_t } });
    const b = try func.appendBlock();
    _ = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = tile_t } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    try gpu.attrs.setSharedBytes(&func, 64);

    try testing.expectError(error.SharedFrameMismatch, compileKernel(allocator, &func, nvidia_abi));
}

test "a declared shared_bytes that AGREES with the frame compiles, the negative control" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const tile_t = try func.types.intern(.{ .array = .{ .len = 32, .elem = f32_t } });
    const b = try func.appendBlock();
    _ = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = tile_t } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    try gpu.attrs.setSharedBytes(&func, 128);

    var kernel = try compileKernel(allocator, &func, nvidia_abi);
    defer kernel.deinit(allocator);
    try testing.expectEqual(@as(u32, 128), kernel.launch.shared_bytes);
}

test "a shared alloca beside a shared PARAMETER is refused" {
    // Both claim offset 0 of one window, and nothing in the IR says that they do not overlap.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    const extern_tile = try func.appendBlockParam(b, shared_t);
    const own_tile = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = i32_t } });
    const v = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = extern_tile } });
    try func.appendStore(b, v, own_tile);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    try testing.expectError(error.SharedFrameAliasesParameter, compileKernel(allocator, &func, nvidia_abi));
}

test "a graphics stage rejects a shared alloca instead of placing a workgroup frame" {
    // A graphics stage has no workgroup, so it has no workgroup shared memory. This is the
    // alloca half of the same refusal a shared POINTER parameter already gets.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const b = try func.appendBlock();
    const uv = try func.appendBlockParam(b, f32_t);
    const tile = try func.appendInst(b, shared_t, .{ .alloca = .{ .elem = f32_t } });
    try func.appendStore(b, uv, tile);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    try testing.expectError(error.Unsupported, compileShader(allocator, &func, .fragment, nvidia_abi));
}

test "a PRIVATE alloca is still refused, and takes no room in the shared window" {
    // The negative control for the whole feature. Only a `ptr(shared)` alloca gets a frame
    // slot; an ordinary stack slot has no backing on this target and keeps its refusal.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const private_t = try func.types.intern(.{ .ptr = .private });
    const b = try func.appendBlock();
    const n = try func.appendBlockParam(b, i32_t);
    const slot = try func.appendInst(b, private_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(b, n, slot);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    try testing.expectError(error.Unsupported, compileKernel(allocator, &func, nvidia_abi));
}

/// Run a list of edge moves through `emitParallelCopy` and simulate the MOVs it emits over a
/// register file, so a test can compare the result against the parallel assignment the moves
/// describe. `initial[r]` is what register r holds before the edge.
fn simulateParallelCopy(allocator: std.mem.Allocator, moves: []const EdgeMove, initial: [256]u32) ![256]u32 {
    var list: std.ArrayList(EdgeMove) = .empty;
    defer list.deinit(allocator);
    try list.appendSlice(allocator, moves);
    var code: std.ArrayList(Inst) = .empty;
    defer code.deinit(allocator);
    try emitParallelCopy(allocator, &code, &list);

    var regs = initial;
    for (code.items) |inst| {
        try testing.expectEqual(@as(u32, 0x202), inst[0] & 0xfff); // MOV, register form
        const dst: u8 = @truncate(inst[0] >> 16);
        const src: u8 = @truncate(inst[1]);
        regs[dst] = regs[src];
    }
    return regs;
}

test "a CYCLE of edge moves still delivers every value" {
    // The five-way block-argument permutation that came out wrong. Emitted in order, the
    // first MOV destroys R4 and the value it held never reaches R7, which read back zero.
    // A parallel assignment must deliver all five.
    const allocator = testing.allocator;
    const moves = [_]EdgeMove{
        .{ .dst = 4, .src = 9 },
        .{ .dst = 5, .src = 7 },
        .{ .dst = 7, .src = 4 },
        .{ .dst = 8, .src = 5 },
        .{ .dst = 9, .src = 8 },
    };
    var initial = [_]u32{0} ** 256;
    for (4..10) |r| initial[r] = @intCast(0x100 + r);

    const regs = try simulateParallelCopy(allocator, &moves, initial);
    for (moves) |m| {
        try testing.expectEqual(initial[m.src], regs[m.dst]);
    }
}

test "a SWAP of two edge moves still delivers both values" {
    // The smallest cycle. Two plain MOVs give both registers the same value.
    const allocator = testing.allocator;
    const moves = [_]EdgeMove{
        .{ .dst = 4, .src = 5 },
        .{ .dst = 5, .src = 4 },
    };
    var initial = [_]u32{0} ** 256;
    initial[4] = 0xaaaa;
    initial[5] = 0xbbbb;

    const regs = try simulateParallelCopy(allocator, &moves, initial);
    try testing.expectEqual(@as(u32, 0xbbbb), regs[4]);
    try testing.expectEqual(@as(u32, 0xaaaa), regs[5]);
}

test "a CHAIN of edge moves needs no scratch register" {
    // The negative control. `R4 <- R5 <- R6` has a safe order, so the ordering must find it
    // and emit exactly one MOV per move, with none through the scratch register. A copy
    // that always parked through the scratch would pass the two tests above and double the
    // instruction count of every ordinary edge.
    const allocator = testing.allocator;
    var list: std.ArrayList(EdgeMove) = .empty;
    defer list.deinit(allocator);
    try list.appendSlice(allocator, &.{
        .{ .dst = 4, .src = 5 },
        .{ .dst = 5, .src = 6 },
    });
    var code: std.ArrayList(Inst) = .empty;
    defer code.deinit(allocator);
    try emitParallelCopy(allocator, &code, &list);

    try testing.expectEqual(@as(usize, 2), code.items.len);
    for (code.items) |inst| {
        try testing.expect(@as(u8, @truncate(inst[0] >> 16)) != r_scratch);
    }
}
