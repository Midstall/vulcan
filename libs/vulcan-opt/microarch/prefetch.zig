//! Model-driven prefetch insertion. Finds counted loops with a pointer-typed induction variable
//! that steps by a constant byte stride each iteration, and a load whose address is that IV (or a
//! constant offset from it), and inserts a software prefetch of `iv + distance` where `distance` is
//! a model-derived number of cache lines. Gated on `Model.prefetches()` (aarch64 only today: it is
//! the only backend that actually lowers the hint to a real instruction).
//!
//! A prefetch is a HINT with no observable effect (see `Function.appendPrefetch` and the
//! differential JIT tests in `libs/vulcan-target/tests/prefetch_differential.zig`), so this pass
//! only ever needs to prove a loop/load shape correct enough to compute a sane distance; it never
//! needs to prove the transform preserves results; that is a structural fact about the opcode
//! itself. Coverage is a heuristic. Correctness (never insert into a shape we cannot read) is not:
//! SKIP-IF-UNSURE, every check below returns null/false rather than guess. Inserting zero prefetches
//! is always a legal outcome of this pass.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const loops = @import("../loops.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;

pub const Error = std.mem.Allocator.Error;

/// A counted loop's shape, vetted enough to look for a strided load inside it. Unlike
/// `unroll.zig`'s `Plan`, this pass never clones or rewrites control flow, so there is nothing to
/// snapshot beyond the block handles themselves (appending instructions later does not invalidate
/// them: blocks are never renumbered or removed).
const Candidate = struct {
    header: Block,
    if_inst: Inst,
    /// The loop's single body block. Required to equal the loop's single latch (see `loopShape`),
    /// so the induction variable's step and any load using it are provably in one straight-line
    /// block; a multi-block body is left untouched rather than traced across blocks.
    body_entry: Block,
    exit: Block,
};

/// One place to insert a prefetch: `distance` bytes/elements ahead of `addr`, in `block`.
const Site = struct {
    block: Block,
    addr: Value,
    distance: i64,
};

/// Whether `b` is inside a loop's body bitset. Mirrors `unroll.zig`'s helper of the same name.
fn inLoop(in_loop: []const bool, b: Block) bool {
    const idx = @intFromEnum(b);
    return idx < in_loop.len and in_loop[idx];
}

/// Insert model-derived prefetches for strided loads in `func`'s counted loops. Returns whether
/// anything changed. Never mutates a loop or load it cannot prove eligible.
pub fn run(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) Error!bool {
    // No target lowers the hint to anything real, so inserting would only add dead IR. Bail before
    // doing any analysis at all: returning false here is always correct.
    if (!model.prefetches()) return false;

    var info = try loops.analyze(allocator, func);
    defer info.deinit(allocator);

    // Gather every eligible site before mutating. `run` only ever appends instructions (never moves
    // or deletes any), and appending does not renumber blocks, so the Candidate/Site handles stay
    // valid; gathering first still mirrors unroll.zig's safer pattern and keeps loop analysis (which
    // is read, never re-run, below) unambiguous: every site was found by the same, single snapshot.
    var sites: std.ArrayList(Site) = .empty;
    defer sites.deinit(allocator);

    for (info.loops) |*loop| {
        const cand = loopShape(func, info.loops, loop) orelse continue; // not a shape we can read: skip
        try collectStridedLoads(func, model, cand, &sites, allocator);
    }

    if (sites.items.len == 0) return false;

    for (sites.items) |site| {
        // The prefetch address has the same type as the address it is ahead of (pointer arithmetic
        // stays within one type, matching every other pointer-stepping site in this IR).
        const ty = func.valueType(site.addr);
        const padd = try func.appendArithImm(site.block, ty, .add, site.addr, site.distance);
        try func.appendPrefetch(site.block, padd);
    }
    return true;
}

/// Step A: prove a loop is a counted loop with a single-block body, or return null to skip it.
/// Mirrors `unroll.zig`'s `eligible` header/latch/exit checks (a proven shape, reused rather than
/// re-derived), minus everything specific to cloning: this pass never clones or rewrites control
/// flow, it only ever appends straight-line instructions into an existing block.
fn loopShape(func: *const Function, all_loops: []const loops.Loop, loop: *const loops.Loop) ?Candidate {
    const n = func.blockCount();
    const h_idx = loop.header;
    const header: Block = @enumFromInt(h_idx);
    const in_loop = loop.body;

    // Innermost only: an outer loop's stride is only meaningful once its inner loop has already
    // been reasoned about on its own terms, so a loop containing another loop's header is skipped.
    for (all_loops) |*other| {
        if (other.header == h_idx) continue;
        if (other.header < in_loop.len and in_loop[other.header]) return null;
    }

    // Pure test header: every instruction is a pure value op, save exactly one `if` ending the
    // block. A header with a side effect (or any other shape) is not a counted loop we model.
    const h_insts = func.blockInsts(header);
    if (h_insts.len == 0) return null;
    var if_inst: ?Inst = null;
    for (h_insts, 0..) |inst, idx| {
        switch (func.opcode(inst)) {
            .@"if" => {
                if (idx != h_insts.len - 1) return null; // the `if` must end the block
                if_inst = inst;
            },
            .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .convert, .unary, .extract, .struct_new => {},
            // load/store/prefetch/call/call_indirect/alloca/global_addr are impure or memory ops.
            else => return null,
        }
    }
    const iff = if_inst orelse return null;
    if (func.terminator(header)) |term| switch (term) {
        .ret => |v| if (v != null) return null,
        .jump => return null,
    };

    const cf = func.opcode(iff).@"if";
    const then_in = inLoop(in_loop, cf.then.target);
    const else_in = inLoop(in_loop, cf.@"else".target);
    var body_entry: Block = undefined;
    var exit: Block = undefined;
    if (then_in and !else_in) {
        body_entry = cf.then.target;
        exit = cf.@"else".target;
    } else if (else_in and !then_in) {
        body_entry = cf.@"else".target;
        exit = cf.then.target;
    } else return null; // need exactly one in-loop and one out-of-loop edge
    if (@intFromEnum(body_entry) == h_idx) return null; // need a real body

    // Single latch: exactly one in-loop block whose *terminator* jumps back to the header.
    // Conditional (if-edge) back-edges are not modeled.
    var latch: ?Block = null;
    var bi: usize = 0;
    while (bi < n) : (bi += 1) {
        if (!inLoop(in_loop, @enumFromInt(bi))) continue;
        const b: Block = @enumFromInt(bi);
        for (func.blockInsts(b)) |inst| {
            if (func.opcode(inst) == .@"if") {
                const c2 = func.opcode(inst).@"if";
                if (@intFromEnum(c2.then.target) == h_idx or @intFromEnum(c2.@"else".target) == h_idx) return null;
            }
        }
        if (func.terminator(b)) |term| switch (term) {
            .jump => |j| if (@intFromEnum(j.target) == h_idx) {
                if (bi == h_idx) return null; // header is not its own latch
                if (latch != null) return null; // more than one latch
                latch = b;
            },
            .ret => {},
        };
    }
    const l = latch orelse return null;

    // Single exit: the header's out-of-loop edge is the *only* edge leaving the loop.
    var out_edges: usize = 0;
    bi = 0;
    while (bi < n) : (bi += 1) {
        if (!inLoop(in_loop, @enumFromInt(bi))) continue;
        const b: Block = @enumFromInt(bi);
        for (func.blockInsts(b)) |inst| {
            if (func.opcode(inst) == .@"if") {
                const c2 = func.opcode(inst).@"if";
                if (!inLoop(in_loop, c2.then.target)) {
                    out_edges += 1;
                    if (c2.then.target != exit) return null;
                }
                if (!inLoop(in_loop, c2.@"else".target)) {
                    out_edges += 1;
                    if (c2.@"else".target != exit) return null;
                }
            }
        }
        if (func.terminator(b)) |term| switch (term) {
            .jump => |j| if (!inLoop(in_loop, j.target)) {
                out_edges += 1;
                if (j.target != exit) return null;
            },
            .ret => {},
        };
    }
    if (out_edges != 1) return null;

    // This pass only traces an induction variable through a single-block body: the body entry and
    // the latch must be the same block, so the IV's step and any load using it are both, provably,
    // in one straight-line block. A multi-block body (an internal diamond, say) is left untouched:
    // SKIP-IF-UNSURE, never a correctness issue, only a missed optimization.
    if (body_entry != l) return null;

    const back_args = switch (func.terminator(l).?) {
        .jump => |j| func.blockArgs(j),
        .ret => return null,
    };
    if (back_args.len != func.blockParams(header).len) return null;

    return .{ .header = header, .if_inst = iff, .body_entry = body_entry, .exit = exit };
}

/// Step B: within an eligible loop, find every header-param-derived induction variable that steps
/// by a provable constant stride, and every load in the body whose address is that IV (or a
/// constant offset from it). Appends one `Site` per such load; finds nothing (leaves `sites`
/// untouched) for any shape it cannot prove.
fn collectStridedLoads(
    func: *const Function,
    model: *const mm.Model,
    cand: Candidate,
    sites: *std.ArrayList(Site),
    allocator: std.mem.Allocator,
) Error!void {
    const header_params = func.blockParams(cand.header);
    const body_params = func.blockParams(cand.body_entry);
    if (body_params.len != header_params.len) return; // shape we do not model

    const cf = func.opcode(cand.if_inst).@"if";
    const in_edge = if (cf.then.target == cand.body_entry) cf.then else cf.@"else";
    const in_args = func.blockArgs(in_edge);
    if (in_args.len != header_params.len) return;

    const back_args = switch (func.terminator(cand.body_entry).?) {
        .jump => |j| func.blockArgs(j),
        .ret => return, // loopShape only accepts a jump latch; defensive, never actually hit
    };
    if (back_args.len != header_params.len) return;

    var k: usize = 0;
    while (k < header_params.len) : (k += 1) {
        // A candidate IV must forward unchanged from the header into the body: the in-loop edge's
        // k-th argument must be the header's own k-th parameter, so the body's k-th param reads
        // exactly this iteration's IV value with no renaming to trace through.
        if (in_args[k] != header_params[k]) continue;
        const biv = body_params[k];

        const stride = strideOf(func, cand.body_entry, biv, back_args[k]) orelse continue;
        if (stride == 0) continue; // no motion each iteration, nothing to prefetch ahead of

        try collectLoadsOn(func, model, cand.body_entry, biv, stride, sites, allocator);
    }
}

/// If `target` is defined, in `block`, as `arith_imm(add|sub, biv, imm)`, the signed per-iteration
/// stride (positive for add, negated for sub). Null when `target` is not a provable constant step
/// of `biv` (a block param with no defining instruction, some other opcode, or a different lhs):
/// SKIP-IF-UNSURE.
fn strideOf(func: *const Function, block: Block, biv: Value, target: Value) ?i64 {
    for (func.blockInsts(block)) |inst| {
        const result = func.instResult(inst) orelse continue;
        if (result != target) continue;
        return switch (func.opcode(inst)) {
            .arith_imm => |a| if (a.lhs == biv) switch (a.op) {
                .add => a.imm,
                .sub => -a.imm,
                .mul, .mulh, .div, .rem, .bit_and, .bit_or, .bit_xor, .shl, .shr => null,
            } else null,
            else => null, // any other opcode is not a recognized constant-stride shape
        };
    }
    return null; // target is not defined by an instruction in this block
}

/// Append a `Site` for every load in `block` whose address is `biv` itself or `biv` plus/minus a
/// compile-time constant, i.e. affine in the induction variable with the loop's own per-iteration
/// stride (the constant offset does not change iteration to iteration, so the load's own
/// byte-to-byte delta across iterations is exactly `stride`, whatever `biv`'s type actually is).
fn collectLoadsOn(
    func: *const Function,
    model: *const mm.Model,
    block: Block,
    biv: Value,
    stride: i64,
    sites: *std.ArrayList(Site),
    allocator: std.mem.Allocator,
) Error!void {
    const distance = prefetchDistance(model, stride);
    if (distance == 0) return;
    for (func.blockInsts(block)) |inst| {
        const ptr = switch (func.opcode(inst)) {
            .load => |l| l.ptr,
            else => continue,
        };
        if (!isAffine(func, block, biv, ptr)) continue;
        try sites.append(allocator, .{ .block = block, .addr = ptr, .distance = distance });
    }
}

/// Whether `ptr` is provably `biv` itself, or `biv` plus/minus a compile-time constant computed in
/// `block`. Anything else (a different induction variable, a runtime-computed offset, a value
/// defined outside `block`) is not provably affine on `biv`: SKIP-IF-UNSURE.
fn isAffine(func: *const Function, block: Block, biv: Value, ptr: Value) bool {
    if (ptr == biv) return true;
    for (func.blockInsts(block)) |inst| {
        const result = func.instResult(inst) orelse continue;
        if (result != ptr) continue;
        return switch (func.opcode(inst)) {
            .arith_imm => |a| a.lhs == biv,
            else => false,
        };
    }
    return false;
}

/// The signed byte/element distance to prefetch ahead of an affine-strided address: the number of
/// cache lines needed to cover one load-use latency's worth of travel at `stride` per iteration,
/// rounded up to a whole line and clamped to a small multiple of lines so a fast-striding loop does
/// not prefetch wildly far ahead and evict data it will need sooner. The sign follows `stride`, so
/// the hint always points in the loop's actual direction of travel. This is a heuristic: only the
/// direction (forward for a positive stride) and staying within the loop's own address stream
/// matter for correctness, the exact magnitude does not.
fn prefetchDistance(model: *const mm.Model, stride: i64) i64 {
    if (stride == 0) return 0;
    // Any opcode payload works: every latency table switches on the opcode's tag only.
    const load_latency = model.latency(.{ .load = .{ .ptr = @enumFromInt(0) } });
    const magnitude: u64 = @intCast(@abs(stride));
    const bytes_needed = @as(u64, load_latency) * magnitude;
    const line: u64 = model.cache_line;
    if (line == 0) return 0; // defensive: every validated model has a nonzero cache line
    const lines_needed = (bytes_needed + line - 1) / line; // ceil-divide, line > 0 just checked
    const lines_clamped = std.math.clamp(lines_needed, 1, 8);
    const dist_mag: i64 = @intCast(lines_clamped * line);
    return if (stride < 0) -dist_mag else dist_mag;
}

/// `fn(n: i32, arr: *i64) i64`: `s = 0; p = arr; for (i = 0; i < n; i += 1) { s += *p; p += 8; }
/// return s`. Two loop-carried values step by a constant each iteration: `i` (the bound counter,
/// stride 1) and `p` (a pointer, stride 8, the strided load's address). `s` (the accumulator) is
/// not a constant stride and must not be mistaken for one.
fn buildStridedSum(func: *Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i32_t);
    const arr = try func.appendBlockParam(entry, ptr_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const p = try func.appendBlockParam(loop, ptr_t);
    const s = try func.appendBlockParam(loop, i64_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const bp = try func.appendBlockParam(body, ptr_t);
    const bs = try func.appendBlockParam(body, i64_t);
    const ds = try func.appendBlockParam(done, i64_t);

    const zero32 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const zero64 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero32, arr, zero64 });

    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, p, s } }, .{ .target = done, .args = &.{s} });

    const val = try func.appendInst(body, i64_t, .{ .load = .{ .ptr = bp } });
    const ns = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = val } });
    const np = try func.appendArithImm(body, ptr_t, .add, bp, 8);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, np, ns });

    func.setTerminator(done, .{ .ret = ds });
}

/// The same shape as `buildStridedSum`, minus the load: `p` still walks by a constant stride, but
/// nothing in the body reads through it, so there is nothing to prefetch.
fn buildNoLoad(func: *Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i32_t);
    const arr = try func.appendBlockParam(entry, ptr_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const p = try func.appendBlockParam(loop, ptr_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const bp = try func.appendBlockParam(body, ptr_t);

    const zero32 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero32, arr });

    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, p } }, .{ .target = done });

    const np = try func.appendArithImm(body, ptr_t, .add, bp, 8);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, np });

    func.setTerminator(done, .{ .ret = null });
}

fn countPrefetches(func: *const Function) usize {
    var count: usize = 0;
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .prefetch) count += 1;
        }
    }
    return count;
}

test "run inserts a prefetch for a strided load under ampere-altra, staying verifiable" {
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildStridedSum(&func);

    try std.testing.expectEqual(@as(usize, 0), countPrefetches(&func));
    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(changed);
    try std.testing.expect(countPrefetches(&func) >= 1);

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "run is a no-op under a model that does not prefetch" {
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildStridedSum(&func);
    const before = func.blockCount();

    const changed = try run(allocator, &func, registry.modelFor(.@"et-soc"));
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before, func.blockCount());
    try std.testing.expectEqual(@as(usize, 0), countPrefetches(&func));
}

test "run leaves a loop with no strided load unchanged" {
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildNoLoad(&func);
    const before = func.blockCount();

    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before, func.blockCount());
    try std.testing.expectEqual(@as(usize, 0), countPrefetches(&func));
}
