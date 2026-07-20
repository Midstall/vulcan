//! Block-layout pass: computes a fall-through-friendly linear order of the blocks and permutes the
//! function into it via `reorderBlocks`, so the per-backend fall-through elision fires on adjacent
//! forward edges. The order is a DOMINANCE-RESPECTING greedy trace (each block placed only after its
//! immediate dominator), which keeps the array-order linear-scan liveness valid: a definition's block
//! always precedes every block it dominates, so operands are numbered before their uses and only
//! back-edges cross, which the backends already handle via extendLiveRanges.
//!
//! CORRECTNESS is paramount. The pass either emits a provably valid linearization or keeps the
//! original order (identity, no-op). It is a one-shot layout run after the main pipeline fixpoint, not
//! an iterated transform. Functions that carry block-keyed attributes are skipped, because
//! `reorderBlocks` does not remap block ids encoded in attribute payloads (see its caveat).

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const cfg_mod = @import("cfg.zig");
const dominators = @import("dominators.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;

pub const pass_def = pass.Pass{ .name = "blocklayout", .run = run };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    return layout(allocator, func, null);
}

/// Compute a fall-through-friendly order and permute `func` into it. `profile` is optional per-block
/// execution counts (pgo): when present the hottest successor is chosen to fall through, when null a
/// structural heuristic is used. Returns whether the function was reordered (false = identity/no-op or
/// a guard tripped). Never emits an invalid layout: if the computed order is not a valid dominance
/// linearization it keeps the original order.
pub fn layout(allocator: std.mem.Allocator, func: *Function, profile: ?[]const u64) pass.Error!bool {
    const n = func.blockCount();
    // GUARD: a single-block function has nothing to lay out.
    if (n <= 1) return false;
    if (profile) |p| std.debug.assert(p.len == n);

    // GUARD: skip any function carrying a block-keyed attribute. `reorderBlocks` remaps block
    // references in terminators and `if` edges only, not block ids encoded in attribute payloads, so
    // reordering such a function would leave those references stale. The three executable backends do
    // not use these attributes, this keeps the pass safe on wasm/spirv/glsl-lowered IR.
    for (func.attributeEntries()) |entry| {
        if (entry.target == .block) return false;
    }

    var cfg = try cfg_mod.build(allocator, func);
    defer cfg.deinit(allocator);
    var doms = try dominators.compute(allocator, func);
    defer doms.deinit(allocator);

    // Reverse postorder gives each reachable block an ordering used both for the structural tie-break
    // (prefer the RPO-earliest fall-through) and for picking the next trace seed. Unreachable blocks
    // are not in the RPO, they carry the sentinel rank.
    const rpo = try cfg.reversePostorder(allocator);
    defer allocator.free(rpo);
    const rpo_rank = try allocator.alloc(u32, n);
    defer allocator.free(rpo_rank);
    const rpo_sentinel: u32 = std.math.maxInt(u32);
    @memset(rpo_rank, rpo_sentinel);
    for (rpo, 0..) |b, i| rpo_rank[b] = @intCast(i);

    const order = try greedyTrace(allocator, &cfg, &doms, rpo, rpo_rank, profile);
    defer allocator.free(order);

    // IDENTITY: when the computed order is already the current order the pass is a byte-identical
    // no-op. This is the common case for functions whose input order is already fall-through friendly.
    var identical = true;
    for (order, 0..) |b, i| {
        if (b != @as(u32, @intCast(i))) {
            identical = false;
            break;
        }
    }
    if (identical) return false;

    // VALIDATE (belt-and-suspenders): the order must be a permutation with the entry first, and every
    // reachable non-entry block's immediate dominator must appear at an EARLIER index. The greedy
    // trace guarantees this by construction, but if any check fails we keep the original order rather
    // than emit an invalid layout.
    if (!validLayout(&doms, order)) return false;

    // Convert to the Block-typed permutation `reorderBlocks` expects.
    const block_order = try allocator.alloc(Block, n);
    defer allocator.free(block_order);
    for (order, 0..) |b, i| block_order[i] = @enumFromInt(b);
    try func.reorderBlocks(allocator, block_order);
    return true;
}

/// Build the dominance-respecting greedy trace order. Result is a full permutation of `0..n`: the
/// reachable blocks in trace order followed by the unreachable blocks in id order. The caller owns it.
fn greedyTrace(
    allocator: std.mem.Allocator,
    cfg: *const cfg_mod.Cfg,
    doms: *const dominators.Dominators,
    rpo: []const u32,
    rpo_rank: []const u32,
    profile: ?[]const u64,
) std.mem.Allocator.Error![]u32 {
    const n = cfg.blockCount();
    std.debug.assert(n >= 2);

    const placed = try allocator.alloc(bool, n);
    defer allocator.free(placed);
    @memset(placed, false);

    var order: std.ArrayList(u32) = .empty;
    errdefer order.deinit(allocator);

    var reachable_count: usize = 0;
    for (0..n) |b| {
        if (doms.isReachable(b)) reachable_count += 1;
    }
    std.debug.assert(doms.isReachable(0)); // the entry is always reachable

    // Start the placement at the entry (block 0). A block is PLACEABLE once its immediate dominator is
    // already placed, so placing the entry first unlocks its dominator-tree children.
    placed[0] = true;
    try order.append(allocator, 0);
    var last: u32 = 0;

    // Each iteration places exactly one block, so the loop runs at most `reachable_count - 1` times.
    var guard: usize = 0;
    while (order.items.len < reachable_count) : (guard += 1) {
        std.debug.assert(guard < n); // progress guarantee: bounded by the block count

        // Try to extend the current trace: pick the best unplaced, placeable successor of `last`.
        const extend = bestSuccessor(cfg, doms, rpo_rank, placed, profile, last);
        if (extend) |s| {
            placed[s] = true;
            try order.append(allocator, s);
            last = s;
            continue;
        }

        // No fall-through available. Start a new trace from the RPO-earliest unplaced placeable block.
        // Such a block always exists while any reachable block is unplaced: walking any unplaced
        // block's idom chain toward the entry reaches a placed ancestor, and the lowest unplaced block
        // on that chain has a placed idom.
        var seed: ?u32 = null;
        for (rpo) |b| {
            if (placed[b]) continue;
            if (isPlaceable(doms, placed, b)) {
                seed = b;
                break;
            }
        }
        const s = seed orelse break; // safety: should never happen, keep whatever is placed
        placed[s] = true;
        try order.append(allocator, s);
        last = s;
    }

    // Append the unreachable blocks last, in id order, so the result is a full permutation.
    for (0..n) |b| {
        if (!placed[b]) try order.append(allocator, @intCast(b));
    }
    std.debug.assert(order.items.len == n);
    return order.toOwnedSlice(allocator);
}

/// Whether block `b` is placeable: its immediate dominator is already placed. The entry dominates
/// itself, so it is only ever placed explicitly at the start.
fn isPlaceable(doms: *const dominators.Dominators, placed: []const bool, b: u32) bool {
    if (!doms.isReachable(b)) return false;
    return placed[doms.immediateDominator(b)];
}

/// Choose the best unplaced, placeable successor of `from` to fall through to, or null if none.
/// With a profile the hottest successor wins. Without, the structural bias prefers a forward edge
/// (not a back-edge to a dominator/header, which keeps loop bodies contiguous), then the RPO-earliest
/// successor. RPO rank breaks ties in both modes.
fn bestSuccessor(
    cfg: *const cfg_mod.Cfg,
    doms: *const dominators.Dominators,
    rpo_rank: []const u32,
    placed: []const bool,
    profile: ?[]const u64,
    from: u32,
) ?u32 {
    var best: ?u32 = null;
    var best_back: bool = true; // a back-edge candidate is worse than a forward one
    var best_weight: u64 = 0;
    var best_rank: u32 = std.math.maxInt(u32);
    for (cfg.successors(from)) |s| {
        if (placed[s]) continue;
        if (!isPlaceable(doms, placed, s)) continue;

        // A back-edge successor is one that dominates `from` (a loop header we branch back to).
        const is_back = doms.dominates(s, from);
        const rank = rpo_rank[s];
        const weight: u64 = if (profile) |p| p[s] else 0;

        const better = if (best == null)
            true
        else if (profile != null)
            // Profile mode: highest execution count first, RPO rank as the tie-break.
            (weight > best_weight or (weight == best_weight and rank < best_rank))
        else
            // Structural mode: forward edges before back-edges, then RPO-earliest.
            (@intFromBool(is_back) < @intFromBool(best_back) or
                (is_back == best_back and rank < best_rank));

        if (better) {
            best = s;
            best_back = is_back;
            best_weight = weight;
            best_rank = rank;
        }
    }
    return best;
}

/// Validate that `order` is a valid dominance linearization: a permutation of `0..n` with the entry
/// first, where every reachable non-entry block's immediate dominator appears at an earlier index.
fn validLayout(doms: *const dominators.Dominators, order: []const u32) bool {
    const n = doms.n;
    if (order.len != n) return false;
    if (order[0] != 0) return false;

    // Inverse permutation: pos[block] = its index in the order. Also proves it is a permutation.
    var buf: [4096]u32 = undefined;
    // The block count is bounded well under this in practice, but guard the stack buffer anyway.
    if (n > buf.len) return false;
    const pos = buf[0..n];
    @memset(pos, std.math.maxInt(u32));
    for (order, 0..) |b, i| {
        if (b >= n) return false;
        if (pos[b] != std.math.maxInt(u32)) return false; // duplicate id
        pos[b] = @intCast(i);
    }

    for (0..n) |b| {
        if (b == 0 or !doms.isReachable(b)) continue;
        const id = doms.immediateDominator(b);
        if (pos[id] >= pos[b]) return false; // the idom must precede its dominated block
    }
    return true;
}

const testing = std.testing;

/// Build `fn(c: bool, x: i32, y: i32) i32` whose blocks are appended in an order that puts the merge
/// before the arms: b0 (if c -> then else els), b1 = merge (ret), b2 = then (-> merge), b3 = els
/// (-> merge). The entry falls through to neither arm in input order.
fn buildDiamondMergeFirst(func: *Function) !struct { Block, Block, Block, Block } {
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const c = try func.appendBlockParam(b0, bool_t);
    const merge = try func.appendBlock();
    const mv = try func.appendBlockParam(merge, i32_t);
    const then_b = try func.appendBlock();
    const els_b = try func.appendBlock();
    try func.appendIf(b0, c, .{ .target = then_b }, .{ .target = els_b });
    const x = try func.appendInst(then_b, i32_t, .{ .iconst = 1 });
    try func.setJump(then_b, merge, &.{x});
    const y = try func.appendInst(els_b, i32_t, .{ .iconst = 2 });
    try func.setJump(els_b, merge, &.{y});
    func.setTerminator(merge, .{ .ret = mv });
    return .{ b0, merge, then_b, els_b };
}

/// The `iconst` value of the first constant instruction in `block`, or null. The two diamond arms
/// carry distinct constants (then = 1, els = 2), so this identifies which arm a block became.
fn blockConst(func: *const Function, block: Block) ?i64 {
    for (func.blockInsts(block)) |inst| {
        if (func.opcode(inst) == .iconst) return func.opcode(inst).iconst;
    }
    return null;
}

test "blocklayout: reorders so a forward successor falls through" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    _ = try buildDiamondMergeFirst(&func);

    // Input order is b0, merge(1), then(2), els(3): the entry's successors are not adjacent to it.
    const changed = try layout(allocator, &func, null);
    try testing.expect(changed);

    // After layout an arm falls through: the block at new index 1 is one of the entry's successors,
    // so the backend elision drops that branch. Structurally the RPO-earliest arm (els, const 2) wins.
    var cfg = try cfg_mod.build(allocator, &func);
    defer cfg.deinit(allocator);
    const succs = cfg.successors(0);
    try testing.expect(succs[0] == 1 or succs[1] == 1);
    try testing.expectEqual(@as(i64, 2), blockConst(&func, @enumFromInt(1)).?);

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "blocklayout: a profile picks the hot successor to fall through" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    _ = try buildDiamondMergeFirst(&func);

    // Make the THEN arm (old id 2, const 1) the hot one, which is the opposite of the structural
    // choice, so the reorder is provably profile-driven. The profile is indexed by OLD block id.
    const profile = [_]u64{ 100, 100, 200, 1 };
    const changed = try layout(allocator, &func, &profile);
    try testing.expect(changed);

    // The hot arm (then, const 1) now falls through right after the entry, at new index 1.
    var cfg = try cfg_mod.build(allocator, &func);
    defer cfg.deinit(allocator);
    const succs = cfg.successors(0);
    try testing.expect(succs[0] == 1 or succs[1] == 1);
    try testing.expectEqual(@as(i64, 1), blockConst(&func, @enumFromInt(1)).?);

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "blocklayout: an already-optimal order is unchanged (identity, returns false)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // A straight-line chain b0 -> b1 -> b2 is already the greedy trace order (single successors, no
    // choice), so layout is a byte-identical no-op.
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const v = try func.appendBlockParam(b0, i32_t);
    const b1 = try func.appendBlock();
    const b2 = try func.appendBlock();
    try func.setJump(b0, b1, &.{});
    const w = try func.appendInst(b1, i32_t, .{ .iconst = 3 });
    try func.setJump(b1, b2, &.{});
    _ = v;
    func.setTerminator(b2, .{ .ret = w });

    const changed = try layout(allocator, &func, null);
    try testing.expect(!changed);
}

test "blocklayout: the reordered function verifies and is dominance-respecting (loop + diamond)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // A counted loop wrapping a diamond, blocks appended out of a fall-through order:
    //   b0 entry -> header(1)
    //   header(1): if c -> exit(2) else body_a(3)
    //   exit(2): ret
    //   body_a(3): if d -> tA(4) else tB(5)
    //   tA(4) -> latch(6), tB(5) -> latch(6)
    //   latch(6) -> header(1)   (back-edge)
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const c0 = try func.appendBlockParam(b0, bool_t);
    const d0 = try func.appendBlockParam(b0, bool_t);
    const header = try func.appendBlock();
    const exit = try func.appendBlock();
    const body_a = try func.appendBlock();
    const t_a = try func.appendBlock();
    const t_b = try func.appendBlock();
    const latch = try func.appendBlock();
    try func.setJump(b0, header, &.{});
    try func.appendIf(header, c0, .{ .target = exit }, .{ .target = body_a });
    const r = try func.appendInst(exit, i32_t, .{ .iconst = 7 });
    func.setTerminator(exit, .{ .ret = r });
    try func.appendIf(body_a, d0, .{ .target = t_a }, .{ .target = t_b });
    try func.setJump(t_a, latch, &.{});
    try func.setJump(t_b, latch, &.{});
    try func.setJump(latch, header, &.{});

    const changed = try layout(allocator, &func, null);
    try testing.expect(changed);

    // Every reachable non-entry block's idom precedes it in the new order.
    var doms = try dominators.compute(allocator, &func);
    defer doms.deinit(allocator);
    for (0..func.blockCount()) |b| {
        if (b == 0 or !doms.isReachable(b)) continue;
        try testing.expect(doms.immediateDominator(b) < @as(u32, @intCast(b)));
    }

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "blocklayout: a function with a block-keyed attribute is skipped (no-op)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const blocks = try buildDiamondMergeFirst(&func);
    const merge = blocks[1];
    // Attach a block-keyed attribute. reorderBlocks would leave it stale, so layout must skip.
    try func.addAttr(.{ .block = merge }, .{ .@"align" = 16 });

    const changed = try layout(allocator, &func, null);
    try testing.expect(!changed);
}

test "blocklayout: single-block function is a no-op" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const v = try func.appendBlockParam(entry, i32_t);
    func.setTerminator(entry, .{ .ret = v });

    const changed = try layout(allocator, &func, null);
    try testing.expect(!changed);
}
