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

/// The shader stage being compiled. Compute kernels source parameters from the
/// constant bank and store via STG. Graphics shaders use the attribute interface
/// (vertex inputs via ALD, fragment inputs via IPA, outputs via AST).
pub const Stage = enum { compute, vertex, fragment };

/// Lower `func` to a SASS compute kernel under the parameter ABI `a`. The caller owns the result.
pub fn compileKernel(allocator: std.mem.Allocator, func: *Function, a: gpu.Abi) Error!Kernel {
    return compileShader(allocator, func, .compute, a);
}

/// Lower `func` to a SASS shader for `stage` under the parameter ABI `a`. The caller owns the result.
pub fn compileShader(allocator: std.mem.Allocator, func: *Function, stage: Stage, a: gpu.Abi) Error!Kernel {
    // This backend does not lower f16 yet. Reject it cleanly instead of
    // silently treating it as f64. This check covers both this direct entry
    // and compileKernel, which calls this function.
    if (ir.function.functionUsesF16(func)) return error.Unsupported;
    if (ir.function.functionUsesF128(func)) return error.Unsupported;

    const nblocks = func.blockCount();
    if (nblocks == 0) return error.Unsupported;

    // Fold constant arith operands into immediates before register allocation.
    // This stops each constant from pinning a GPR for its whole live range.
    // Heavy shaders (the noise and terrain shaders) need this to fit the
    // 251-GPR pool instead of exhausting it.
    foldConstantsToImm(func);

    var loc = std.AutoHashMapUnmanaged(Value, Loc){};
    defer loc.deinit(allocator);
    var max_reg: u8 = r_outptr + 1; // the output pointer pair is always live
    try assignLocs(allocator, func, &loc, &max_reg);

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
        .{ .params = try allocator.alloc(gpu.Param, 0), .bytes = 0, .out_pointer = null };
    errdefer layout.deinit(allocator);

    if (stage == .compute) {
        // A kernel that returns a value reads an output pointer from the front
        // of the constant bank (its `ret` stores the result there). A void
        // compute kernel has no output pointer.
        if (layout.out_pointer) |out| {
            const at: u16 = @intCast(a.param_base + out.offset);
            try code.append(allocator, encode.ldc(r_outptr, bank0, at, .{})); // outptr lo
            try code.append(allocator, encode.ldc(r_outptr + 1, bank0, at + 4, .{})); // outptr hi
        }
        var placed: usize = 0;
        for (eparams) |p| {
            if (gpu.attrs.builtinOf(func, p)) |bi| {
                try emitComputeBuiltin(allocator, &code, func, loc, p, bi);
                continue;
            }
            const slot = layout.params[placed];
            placed += 1;
            const at: u16 = @intCast(a.param_base + slot.offset);
            const lo = gprOf(loc, p);
            try code.append(allocator, encode.ldc(lo, bank0, at, .{}));
            // A 64-bit address occupies a register pair, so its high word follows in lo + 1.
            //
            // A SHARED address does not. It is a 32-bit byte offset into the CTA's
            // shared-memory window, so the single LDC above is the whole load. The parameter
            // block still reserves `pointer_bytes` for it, because `layoutParams` places every
            // pointer at the target's address width, so the runtime writes the offset into the
            // low dword of that slot and leaves the high dword alone.
            switch (slot.kind) {
                .scalar => {},
                .pointer => |space| switch (space) {
                    .global, .constant, .private => {
                        try code.append(allocator, encode.ldc(lo + 1, bank0, at + 4, .{}));
                    },
                    .shared => {},
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
                try code.append(allocator, encode.ldc(rd, encode.graphics_const_bank, off, .{})); // address lo (root table 1)
                try code.append(allocator, encode.ldc(rd + 1, encode.graphics_const_bank, off + 4, .{})); // address hi
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
            try lowerInst(allocator, func, &loc, &code, &tex, &deriv, &math, inst);
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
            .block = gpu.attrs.localSize(func),
            .shared_bytes = gpu.attrs.sharedBytes(func),
            .reg_count = reg_count,
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
fn assignLocs(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), max_reg: *u8) Error!void {
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
/// `value` is not a constant. The nvidia `arith_imm` lowering materializes
/// this with `movImm` into a scratch register, so any 32-bit constant, int or
/// float, can be an immediate operand.
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

/// Fold a constant operand of an `arith` into `arith_imm`, so codegen
/// materializes the constant as a scratch-register immediate (movImm) at the
/// use, instead of holding it in an allocated GPR for its whole live range.
/// The simplex-noise and terrain shaders define 100 or more float constants.
/// Without this fold, those constants would each pin a register and exhaust
/// the GPR pool, since the linear-scan allocator has no spilling. After
/// folding, those constants are dead: each is a [def,def] interval that
/// reuses one register, so peak pressure drops to the real computation
/// pressure. This skips div and rem, since those are lowered specially and
/// not through the general arith_imm movImm-plus-arith path, pointer adds
/// (64-bit carry), and bool ops (predicate combines). Non-commutative ops
/// fold only the right operand, since the arith_imm form computes
/// `lhs op imm`.
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
///
/// The grid size and the subgroup builtins have no correct lowering here yet and return
/// `error.Unsupported`. Each refusal says why below. A refusal is deliberate: a builtin that
/// reads the wrong register produces a kernel that looks right and computes garbage, and no
/// test in this repository can execute SASS to catch that.
fn emitComputeBuiltin(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(Inst),
    func: *const Function,
    loc: std.AutoHashMapUnmanaged(Value, Loc),
    p: Value,
    bi: gpu.Builtin,
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
        // special register for it. The CUDA driver puts it in its own reserved area of
        // constant bank 0, but that offset belongs to a driver convention this ABI does not
        // define: `gpu.Abi` describes where the PARAMETER block starts and nothing else, and
        // `layoutParams` reserves no slot for the grid size. Picking an offset here would
        // invent a delivery mechanism the runtime does not implement. This lowers once
        // `vulcan-gpu` gives the grid size a place in the contract.
        .grid_dim_x,
        .grid_dim_y,
        .grid_dim_z,
        => return error.Unsupported,
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

/// Emit `gid.a = ctaid.a * ntid.a + tid.a` into `dst` for axis `a`.
///
/// The workgroup size is a compile-time constant, so it becomes an immediate rather than a
/// second hardware read. The two scratch registers hold the hardware reads: the prologue owns
/// them, and the allocator keeps values out of them.
fn emitGlobalId(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(Inst),
    func: *const Function,
    dst: u8,
    a: u2,
) Error!void {
    const size = gpu.attrs.localSize(func)[a];
    try code.append(allocator, encode.movImm(dst, size, .{}));
    try code.append(allocator, encode.s2r(r_scratch, encode.sr_tid[a], .{}));
    try code.append(allocator, encode.s2r(r_scratch2, encode.sr_ctaid[a], .{}));
    try code.append(allocator, encode.imad(dst, r_scratch2, dst, r_scratch, .{}));
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

fn lowerInst(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), tex: *const TexLowering, deriv: *const DerivLowering, math: *const MathLowering, inst: ir.function.Inst) Error!void {
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
            if (isWidePtr(func, result) and a.op == .add) {
                // 64-bit pointer add: (dst:dst+1) = (base:base+1) +
                // zext(offset). The low add produces a carry that the high
                // add (`.X`) consumes.
                const dlo = gprOf(loc.*, result);
                const base = gprOf(loc.*, a.lhs); // pointer pair (lo:hi)
                const offset = gprOf(loc.*, a.rhs); // 32-bit, zero-extended
                try code.append(allocator, encode.iadd3CarryOut(dlo, base, offset, carry_pred, .{}));
                try code.append(allocator, encode.iadd3CarryIn(dlo + 1, base + 1, encode.RZ, carry_pred, .{}));
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
            // A load through a SHARED pointer reads the CTA's shared-memory
            // window, which is a different memory and a different
            // instruction. Its address is a 32-bit offset in ONE register,
            // so there is no pointer pair to read.
            if (isSharedPtr(func, l.ptr)) {
                try code.append(allocator, encode.ldsU32(rd, gprOf(loc.*, l.ptr), .{}));
                return;
            }
            // Otherwise this is an ordinary LDG from the 64-bit pointer pair
            // into the 32-bit result register. This is variable latency: the
            // scoreboard scheduler assigns its write barrier and the wait on
            // each consumer.
            try code.append(allocator, encode.ldgU32(rd, gprOf(loc.*, l.ptr), .{}));
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
                // register, so there is no pointer pair to read.
                try code.append(allocator, encode.stsU32(gprOf(loc.*, st.ptr), gprOf(loc.*, st.value), .{}));
            } else {
                try code.append(allocator, encode.stgU32(gprOf(loc.*, st.ptr), gprOf(loc.*, st.value), .{}));
            }
        },
        .prefetch => {}, // a hint; this GPU target has no CPU-style prefetch, so it is dropped
        .arith_imm => |a| {
            const result = func.instResult(inst).?;
            // Logical NOT lowers to `bool ^ -1` (bit_xor against all-ones).
            // A boolean result lives in a predicate, so negate the source
            // predicate with PLOP3 (`p ^ PT` = `!p`, since PT is true). The
            // GPR LOP3 path cannot touch it.
            if (isBool(func, result)) {
                std.debug.assert(a.op == .bit_xor); // the only bool arith_imm the lowering emits
                const pd = predOf(loc.*, result);
                const pa = predOf(loc.*, a.lhs);
                try code.append(allocator, encode.plop3(pd, pa, encode.PT, encode.LUT_XOR, .{}));
            } else {
                const rd = gprOf(loc.*, result);
                const ra = gprOf(loc.*, a.lhs);
                try code.append(allocator, encode.movImm(r_scratch, @truncate(@as(u64, @bitCast(a.imm))), .{}));
                try code.append(allocator, try arith(func, a.op, rd, ra, r_scratch, a.lhs));
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
            // The only alloca the NVIDIA backend supports is the
            // host-sampler out-pointer (a vec4 RGBA result slot), which is
            // materialized as a 4-register TEX result block, with no real
            // stack. Any other alloca is unsupported.
            const result = func.instResult(inst).?;
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

/// Edge moves into the target block's parameters (register copies). Distinct
/// registers per value mean the moves are independent, except for genuine
/// swaps. A scratch register breaks any cycle.
fn emitMoves(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), jump: ir.function.Jump) Error!void {
    const args = func.blockArgs(jump);
    const params = func.blockParams(jump.target);
    if (args.len != params.len) return error.Unsupported;
    for (args, params) |arg, param| {
        const dst = gprOf(loc.*, param);
        const src = gprOf(loc.*, arg);
        if (dst != src) try code.append(allocator, encode.movReg(dst, src, .{}));
    }
}

// Liveness (for the allocator).

fn markUse(last_use: []u32, v: Value, pos: u32) void {
    if (pos > last_use[@intFromEnum(v)]) last_use[@intFromEnum(v)] = pos;
}

fn forEachUse(func: *const Function, inst: ir.function.Inst, last_use: []u32, pos: u32) void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .fconst128, .alloca, .global_addr => {},
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
        .iconst, .fconst, .fconst128, .alloca, .global_addr => {},
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
    var has_ald = false;
    var has_fadd = false;
    var has_ast = false;
    var has_exit = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0x321 => has_ald = true,
            0x221 => has_fadd = true,
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

    // The prologue must source the UBO pointer from the constant bank (two
    // LDCs for the 64-bit address pair), and the body must LDG through it,
    // plus do an ALD for the input.
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
    try testing.expectEqual(@as(usize, 2), ldc_count); // address lo + hi
    try testing.expect(has_ldg);
    try testing.expect(has_ald);
    // The LDC offset of the first (address-lo) load is graphics_ubo_cb_base.
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
    // the UBO pointer (LDC x2), and LDG through base+offset. The only ALD
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
            0x224 => has_imad = true, // IMAD (base 0x024 | reg form)
            0xb82 => ldc_count += 1,
            0x981 => has_ldg = true,
            else => {},
        }
    }
    try testing.expect(has_ald_vid);
    try testing.expect(has_imad);
    try testing.expectEqual(@as(usize, 2), ldc_count); // UBO address lo + hi
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

    // Prologue: LDC outptr lo/hi plus two inputs equals 4 instructions,
    // then IMAD, IADD3, STG, EXIT equals 8 instructions total (32 dwords).
    try testing.expectEqual(@as(usize, 8 * 4), kernel.code.len);
    try testing.expectEqual(@as(u32, 0xb82), kernel.code[0] & 0xfff); // first LDC
    // The first LDC reads the output pointer low word at the ABI's parameter base.
    try testing.expectEqual(
        @as(u32, nvidia_abi.param_base),
        @as(u32, @as(u16, @truncate(kernel.code[1] >> 6)) & 0xffff),
    );

    // The instruction words: LDC x4, IMAD, IADD3, STG, EXIT.
    const op = struct {
        fn at(code: []const u32, i: usize) u32 {
            return code[i * 4] & 0xfff;
        }
    }.at;
    try testing.expectEqual(@as(u32, 0xb82), op(kernel.code, 3)); // last param LDC
    try testing.expectEqual(@as(u32, 0x224), op(kernel.code, 4)); // IMAD (base 0x024 | reg form)
    try testing.expectEqual(@as(u32, 0x210), op(kernel.code, 5)); // IADD3
    try testing.expectEqual(@as(u32, 0x986), op(kernel.code, 6)); // STG
    try testing.expectEqual(@as(u32, 0x94d), op(kernel.code, 7)); // EXIT
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

    // The output pointer is a pair, then the buffer pointer is a pair, then the scalar.
    try testing.expectEqual(@as(usize, 5), offsets.items.len);
    try testing.expectEqual(@as(usize, 2), kernel.launch.params.len);

    const base = nvidia_abi.param_base;
    try testing.expectEqual(base + kernel.launch.params[0].offset, offsets.items[2]);
    try testing.expectEqual(base + kernel.launch.params[0].offset + 4, offsets.items[3]);
    try testing.expectEqual(base + kernel.launch.params[1].offset, offsets.items[4]);

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

    // LDC outptr lo, LDC outptr hi, LDC tile, LDC n, LDS, IADD3, STS, STG, EXIT.
    // The tile pointer contributes ONE LDC. A 64-bit pointer would add a tenth.
    try testing.expectEqual(@as(usize, 9 * 4), kernel.code.len);
    try testing.expectEqual(@as(u32, 0xb82), opAt(kernel.code, 0)); // outptr lo
    try testing.expectEqual(@as(u32, 0xb82), opAt(kernel.code, 1)); // outptr hi
    try testing.expectEqual(@as(u32, 0xb82), opAt(kernel.code, 2)); // tile, the ONLY one
    try testing.expectEqual(@as(u32, 0xb82), opAt(kernel.code, 3)); // n
    try testing.expectEqual(@as(u32, 0x984), opAt(kernel.code, 4)); // LDS, not LDG (0x981)
    try testing.expectEqual(@as(u32, 0x210), opAt(kernel.code, 5)); // IADD3
    try testing.expectEqual(@as(u32, 0x988), opAt(kernel.code, 6)); // STS, not STG (0x986)
    try testing.expectEqual(@as(u32, 0x986), opAt(kernel.code, 7)); // STG: the return, still global
    try testing.expectEqual(@as(u32, 0x94d), opAt(kernel.code, 8)); // EXIT

    // The tile parameter occupies exactly ONE register: the next parameter takes the very
    // next register. A pair would have pushed `n` one further along.
    const r_tile = regAt(kernel.code, 2, 16); // LDC dst
    const r_n = regAt(kernel.code, 3, 16);
    try testing.expectEqual(r_tile + 1, r_n);
    try testing.expectEqual(@as(u8, value_reg_base), r_tile);

    // LDS reads the shared window offset out of that one register at bit 24, and writes the
    // loaded value into the register the STG then returns.
    const r_v = regAt(kernel.code, 4, 16); // LDS dst
    try testing.expectEqual(r_tile, regAt(kernel.code, 4, 24)); // LDS address
    try testing.expectEqual(@as(u8, encode.URZ), regAt(kernel.code, 4, 32)); // no uniform base
    // STS addresses the same register and stores the IADD3 result at bit 32.
    try testing.expectEqual(r_tile, regAt(kernel.code, 6, 24)); // STS address
    try testing.expectEqual(regAt(kernel.code, 5, 16), regAt(kernel.code, 6, 32)); // STS data
    try testing.expectEqual(r_v, regAt(kernel.code, 7, 32)); // STG stores the loaded value

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

    // Shared: LDC outptr lo/hi, LDC tile, LDC i, IADD3, LDS, STG, EXIT.
    try testing.expectEqual(@as(usize, 8 * 4), shared_k.code.len);
    try testing.expectEqual(@as(usize, 1), count(shared_k.code, 0x210)); // ONE IADD3
    try testing.expectEqual(@as(usize, 1), count(shared_k.code, 0x984)); // LDS
    try testing.expectEqual(@as(usize, 0), count(shared_k.code, 0x981)); // no LDG
    // The LDS reads the register the single IADD3 wrote.
    try testing.expectEqual(@as(u32, 0x210), opAt(shared_k.code, 4));
    try testing.expectEqual(@as(u32, 0x984), opAt(shared_k.code, 5));
    try testing.expectEqual(regAt(shared_k.code, 4, 16), regAt(shared_k.code, 5, 24));

    // Global: the pointer is a pair, so it is two LDCs and a two-instruction carry chain.
    try testing.expectEqual(@as(usize, 2), count(global_k.code, 0x210)); // carry-out plus carry-in
    try testing.expectEqual(@as(usize, 1), count(global_k.code, 0x981)); // LDG
    try testing.expectEqual(@as(usize, 0), count(global_k.code, 0x984)); // no LDS
    try testing.expectEqual(@as(usize, 5), count(global_k.code, 0xb82)); // outptr pair, base pair, i
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
