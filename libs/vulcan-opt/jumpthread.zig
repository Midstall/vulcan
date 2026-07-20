//! Jump threading. Two transforms, run together:
//!
//! 1. Identity forwarding: an empty block that does nothing but pass its own parameters straight to a
//!    single successor is a pure detour, so every edge into it is redirected to that successor. These
//!    forwarding blocks are what critical-edge splitting and structured lowering leave behind, and
//!    removing them shortens branch chains and helps fallthrough.
//!
//! 2. Implied-condition threading (non-duplicating): a block B whose `@"if"(cond)` outcome is already
//!    KNOWN on an incoming edge P->B is threaded straight to the successor that outcome selects, so
//!    the P path skips B's branch entirely. The outcome is known when P passes a constant for the
//!    param B tests (constant-param), or when P branches on the SAME value B does and B sits on P's
//!    then/else edge (correlated-branch). This is done WITHOUT cloning B, which is only sound when
//!    every value the redirected edge would carry is already available at P and B has no side effect
//!    to lose. When it is not sound the edge is left for tail duplication (a separate, heavier pass).
//!
//! Threaded blocks are left in place as dead code (the reachability-aware analyses ignore them),
//! matching how branchfold leaves an unreachable arm.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const cfg_mod = @import("cfg.zig");
const dom = @import("dominators.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Value = ir.function.Value;
const Jump = ir.function.Jump;

pub const pass_def = pass.Pass{ .name = "jumpthread", .run = run };

/// Which out-edge of a predecessor lands on the threaded block, i.e. which slot the redirect rewrites.
const EdgeRef = enum { jump, if_then, if_else };

/// The largest block (in instruction count) tail duplication will copy. A block above this is left
/// un-threaded, since duplicating a big block trades too much code size for the branch it removes.
const tail_dup_block_cap: usize = 8;

/// The total number of instructions tail duplication may copy across a single `run`. This is the
/// code-growth ceiling and, together with the fact that each duplication strictly removes a P->B
/// edge, one of the termination guarantees (once it is exhausted, no more blocks are cloned).
const tail_dup_total_budget: usize = 64;

/// The bloat budget carried across the one-at-a-time threading loop. It bounds both the size of any
/// single duplicated block and the total instructions duplicated this run.
const Budget = struct {
    /// Instructions still available to duplicate this run.
    remaining: usize,

    /// Whether a block of `inst_count` instructions may be tail-duplicated: it must be small enough
    /// (per-block cap) and fit in what remains of the total budget.
    fn allows(self: *const Budget, inst_count: usize) bool {
        return inst_count <= tail_dup_block_cap and inst_count <= self.remaining;
    }

    /// Charge a completed duplication against the total budget. `allows` must have returned true for
    /// this same count, so the subtraction cannot underflow.
    fn charge(self: *Budget, inst_count: usize) void {
        std.debug.assert(inst_count <= self.remaining); // caller checked `allows` first
        self.remaining -= inst_count;
    }
};

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    if (func.blockCount() == 0) return false;

    var changed = false;

    // Apply implied-condition threadings one at a time. Each application redirects a P->B edge away
    // from B, so re-running finds a different edge. The loop stops when none remain. The step cap is
    // a belt-and-braces termination guarantee (every applied step strictly removes a P->B edge). The
    // extra `tail_dup_total_budget` headroom covers the fresh blocks tail duplication appends, none
    // of which end in an `@"if"` so none becomes a new threading candidate.
    var budget = Budget{ .remaining = tail_dup_total_budget };
    var steps: usize = 0;
    const cap = func.blockCount() * 4 + 4 + tail_dup_total_budget;
    while (steps < cap) : (steps += 1) {
        if (!try threadOne(allocator, func, &budget)) break;
        changed = true;
    }

    // Then the identity forwarder rewrite (the original transform).
    if (try forwardIdentity(allocator, func)) changed = true;
    return changed;
}

/// Redirect every out-edge that lands on a pure identity forwarder to the forwarder's target.
fn forwardIdentity(allocator: std.mem.Allocator, func: *Function) pass.Error!bool {
    const n = func.blockCount();

    // forward_to[b] = the successor to which block b is a pure identity forwarder, else null.
    const forward_to = try allocator.alloc(?Block, n);
    defer allocator.free(forward_to);
    for (0..n) |bi| forward_to[bi] = identityForwardTarget(func, @enumFromInt(bi));

    var changed = false;
    for (0..n) |bi| {
        const block: Block = @enumFromInt(bi);
        // Redirect this block's out-edges that land on a forwarder to the forwarder's target.
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => |cf| {
                    if (retarget(forward_to, cf.then.target)) |dest| {
                        func.opcodeMut(inst).@"if".then.target = dest;
                        changed = true;
                    }
                    if (retarget(forward_to, cf.@"else".target)) |dest| {
                        func.opcodeMut(inst).@"if".@"else".target = dest;
                        changed = true;
                    }
                },
                else => {},
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .jump => |jmp| if (retarget(forward_to, jmp.target)) |dest| {
                func.terminatorPtr(block).*.?.jump.target = dest;
                changed = true;
            },
            .ret => {},
        };
    }
    return changed;
}

/// Find the first predecessor edge P->B (over every block B that ends in an `@"if"`) whose branch
/// outcome is implied and can be threaded non-duplicating, apply it, and return true. Return false
/// when no such edge exists. Recomputes the CFG, dominators, and value->defining-block map from
/// scratch, since the previous application invalidated them.
fn threadOne(allocator: std.mem.Allocator, func: *Function, budget: *Budget) pass.Error!bool {
    const n = func.blockCount();

    var cfg = try cfg_mod.build(allocator, func);
    defer cfg.deinit(allocator);
    var doms = try dom.compute(allocator, func);
    defer doms.deinit(allocator);

    // def_block[value] = index of the block that defines it (its param block or its instruction's
    // block). A value not attached to any block keeps the sentinel `n` (treated as unavailable).
    const def_block = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(def_block);
    @memset(def_block, @intCast(n));
    for (0..n) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| def_block[@intFromEnum(p)] = @intCast(bi);
        for (func.blockInsts(block)) |inst| {
            if (func.instResult(inst)) |r| def_block[@intFromEnum(r)] = @intCast(bi);
        }
    }

    for (0..n) |bi_b| {
        const b: Block = @enumFromInt(bi_b);
        const if_b = endsInIf(func, b) orelse continue;
        const cond = func.opcode(if_b).@"if".cond;
        const then_edge = func.opcode(if_b).@"if".then;
        const else_edge = func.opcode(if_b).@"if".@"else";

        for (cfg.predecessors(bi_b)) |pi| {
            if (pi == bi_b) continue; // P == B guard (a self-edge)
            if (!doms.isReachable(pi)) continue; // an unreachable P has a degenerate dominance relation
            const p: Block = @enumFromInt(pi);

            // Enumerate P's out-edges that land on B. A block ends in an `@"if"` OR a jump/ret
            // terminator, never both, so at most the two `@"if"` sides or the single jump apply.
            if (endsInIf(func, p)) |if_p| {
                const cf = func.opcode(if_p).@"if";
                const both = cf.then.target == b and cf.@"else".target == b;
                if (cf.then.target == b and !both) {
                    const corr: ?bool = if (cf.cond == cond) true else null;
                    if (try tryThreadEdge(allocator, func, &doms, &cfg, def_block, n, budget, p, b, cond, then_edge, else_edge, func.blockArgs(cf.then), corr, .if_then, if_p)) return true;
                }
                if (cf.@"else".target == b and !both) {
                    const corr: ?bool = if (cf.cond == cond) false else null;
                    if (try tryThreadEdge(allocator, func, &doms, &cfg, def_block, n, budget, p, b, cond, then_edge, else_edge, func.blockArgs(cf.@"else"), corr, .if_else, if_p)) return true;
                }
            } else if (func.terminator(p)) |term| switch (term) {
                .jump => |j| if (j.target == b) {
                    if (try tryThreadEdge(allocator, func, &doms, &cfg, def_block, n, budget, p, b, cond, then_edge, else_edge, func.blockArgs(j), null, .jump, null)) return true;
                },
                .ret => {},
            };
        }
    }
    return false;
}

/// Test whether the edge P->B (which passes `p_args` to B's params) has an implied `@"if"` outcome
/// that can be threaded non-duplicating, and if so rewrite it and return true. `corr` is the outcome
/// implied by correlated-branch detection (the caller already matched P's branch value against B's),
/// or null to fall back to constant-param detection. `p_if` is P's `@"if"` instruction for the
/// `if_then`/`if_else` edge kinds, unused for `jump`.
fn tryThreadEdge(
    allocator: std.mem.Allocator,
    func: *Function,
    doms: *const dom.Dominators,
    cfg: *const cfg_mod.Cfg,
    def_block: []const u32,
    n: usize,
    budget: *Budget,
    p: Block,
    b: Block,
    cond: Value,
    then_edge: Jump,
    else_edge: Jump,
    p_args: []const Value,
    corr: ?bool,
    edge: EdgeRef,
    p_if: ?Inst,
) pass.Error!bool {
    const bi_b: u32 = @intFromEnum(b);
    const pi: u32 = @intFromEnum(p);

    // The known truth of `cond` on this edge: correlated-branch outcome, else constant-param.
    var known = corr;
    if (known == null) {
        for (func.blockParams(b), 0..) |param, i| {
            if (param != cond) continue;
            std.debug.assert(i < p_args.len); // valid IR: P passes one arg per B param
            if (func.definingInst(p_args[i])) |di| {
                if (func.opcode(di) == .iconst) known = func.opcode(di).iconst != 0;
            }
            break;
        }
    }
    const truth = known orelse return false;

    const s_edge = if (truth) then_edge else else_edge;
    const s = s_edge.target;
    if (s == b) return false; // threading to B itself would recreate the edge
    if (s == p) return false; // threading P onto itself would forge a self-loop, leave it for dup

    // REQUIRED precondition for threading P->B to S, for BOTH non-dup and future tail-dup: threading
    // must not break a DOMINANCE-USE of a B-defined value. This IR permits dominance-based cross-block
    // uses, so an instruction in S (or a block reachable from S) may reference a value defined directly
    // in B with no edge arg, sound only while B dominates that use. Redirecting P->B to P->S gives S a
    // predecessor bypassing B, so every block reachable from S without re-entering B loses B as a
    // dominator. If any such block uses a B-defined value, the thread would make it use-before-def, so
    // this edge is left unthreaded. If B does not dominate S there is nothing to break (S already has a
    // path not through B, so the verifier would have rejected any dominance-use of a B value below it).
    if (try threadBreaksDominanceUse(allocator, func, doms, cfg, def_block, b, s)) return false;

    // Substitute: each successor-edge arg that is a B-param becomes P's matching incoming arg. Other
    // values (defined outside B) pass through. `from_sub` records which came from P (always available).
    const s_args = func.blockArgs(s_edge);
    const params_b = func.blockParams(b);
    const scratch = try allocator.alloc(Value, s_args.len);
    defer allocator.free(scratch);
    const from_sub = try allocator.alloc(bool, s_args.len);
    defer allocator.free(from_sub);
    for (s_args, 0..) |w, wi| {
        scratch[wi] = w;
        from_sub[wi] = false;
        for (params_b, 0..) |param, j| {
            if (param != w) continue;
            std.debug.assert(j < p_args.len);
            scratch[wi] = p_args[j];
            from_sub[wi] = true;
            break;
        }
    }

    // Non-dup safety (a): every substituted-edge value must be available at P. A P-arg always is. Any
    // other value must be defined by a block that dominates P and is NOT B (a B-defined value is
    // exactly the use-before-def surface, since the threaded path skips B).
    var nondup_safe = true;
    for (scratch, from_sub) |w, fs| {
        if (fs) continue;
        if (!availableAt(doms, def_block, n, w, bi_b, pi)) {
            nondup_safe = false;
            break;
        }
    }

    // Non-dup safety (b): B must have no side effect, since the threaded path no longer runs B.
    if (nondup_safe and hasSideEffect(func, b)) nondup_safe = false;

    if (!nondup_safe) {
        // Non-dup is unsound here: either a successor-edge value is defined in B (so it is not
        // available on the P path) or B has a side effect the threaded path must still run. Tail
        // duplicate B onto a fresh block that recomputes its values, then jumps straight to S, so the
        // duplicated values ARE available and the effect runs. Guarded by the bloat budget.
        return tryTailDup(allocator, func, budget, p, b, s, s_edge, edge, p_if);
    }

    // Safe: redirect P's edge onto S, carrying the substituted args.
    const new_args = try func.internValues(scratch);
    switch (edge) {
        .jump => func.terminatorPtr(p).* = .{ .jump = .{ .target = s, .args = new_args } },
        .if_then => func.opcodeMut(p_if.?).@"if".then = .{ .target = s, .args = new_args },
        .if_else => func.opcodeMut(p_if.?).@"if".@"else" = .{ .target = s, .args = new_args },
    }
    return true;
}

/// Tail-duplicate B for the edge P->B whose `@"if"` outcome selects successor S. Clone B onto a fresh
/// block B', drop B''s copied trailing `@"if"` so it stops re-branching, point B' straight at S with
/// S's edge args remapped through the clone (so B-computed values become B''s fresh copies), then
/// redirect P onto B' carrying the same args P already passed to B. B' is reached only from P, so this
/// does not disturb B's other predecessors. Returns false (leaving the edge un-threaded) when the
/// bloat budget forbids duplicating a block this large. `threadBreaksDominanceUse` has already gated
/// the caller, so no dominance-use outside the edge args can break.
fn tryTailDup(
    allocator: std.mem.Allocator,
    func: *Function,
    budget: *Budget,
    p: Block,
    b: Block,
    s: Block,
    s_edge: Jump,
    edge: EdgeRef,
    p_if: ?Inst,
) pass.Error!bool {
    const b_inst_count = func.blockInsts(b).len;
    std.debug.assert(b_inst_count > 0); // B ends in an `@"if"`, so it has at least that instruction
    if (!budget.allows(b_inst_count)) return false; // too big, or the run's duplication budget is spent

    // Clone B: fresh params mirroring B's, fresh copies of B's instructions (including the trailing
    // `@"if"`, which the clone remaps but we drop next), and `map` records old value -> fresh copy.
    var map: std.AutoHashMapUnmanaged(Value, Value) = .{};
    defer map.deinit(allocator);
    const bprime = try func.cloneBlock(allocator, b, &map);

    // Drop B''s copied trailing `@"if"` so B' falls through to the jump we install instead of
    // re-branching. It is the last instruction (B ends in an `@"if"` and carries no terminator, so the
    // clone reproduces that shape). Shrinking the list orphans the popped `@"if"`: it lives on in the
    // global instruction pool but sits in no block, so the reachability-aware analyses and the verifier
    // (which walk per-block instruction lists) never see it, and DCE reclaims its now-dead operands.
    const bi = func.blockInstsMut(bprime);
    std.debug.assert(bi.items.len > 0);
    std.debug.assert(func.opcode(bi.items[bi.items.len - 1]) == .@"if"); // the shape endsInIf recognized
    bi.items.len -= 1;

    // Point B' at S with S's original edge args remapped through the clone: a B-param becomes B''s
    // fresh param (fed by P's incoming arg), a B-computed value becomes B''s fresh copy, and a value
    // defined outside B passes through unchanged.
    const s_args = func.blockArgs(s_edge);
    const args = try allocator.alloc(Value, s_args.len);
    defer allocator.free(args);
    for (s_args, 0..) |v, i| args[i] = map.get(v) orelse v;
    try func.setJump(bprime, s, args);

    // Redirect P's edge from B to B', passing the SAME args P already passed to B (B' has B's params).
    switch (edge) {
        .jump => func.terminatorPtr(p).*.?.jump.target = bprime,
        .if_then => func.opcodeMut(p_if.?).@"if".then.target = bprime,
        .if_else => func.opcodeMut(p_if.?).@"if".@"else".target = bprime,
    }

    budget.charge(b_inst_count);
    return true;
}

/// Whether value `w` is available at block `p`: defined by a known block that dominates `p` and is
/// not `b`. A value defined in `b` is excluded even if `b` happens to dominate `p` (a loop back
/// edge), since on the threaded path `p` no longer flows through `b` to recompute it.
fn availableAt(doms: *const dom.Dominators, def_block: []const u32, n: usize, w: Value, bi_b: u32, pi: u32) bool {
    const db = def_block[@intFromEnum(w)];
    if (db >= n) return false; // unknown definition
    if (db == bi_b) return false; // defined in B, lost when the P path skips B
    return doms.dominates(db, pi);
}

/// Whether threading a P->B edge onto S would break a dominance-use of a B-defined value. A value
/// defined in B may be used, with no edge arg, by any block B dominates (this IR permits dominance
/// cross-block uses). Redirecting P->B to P->S gives S a predecessor that bypasses B, so every block
/// reachable from S WITHOUT re-entering B loses B as a dominator. If any such block uses a value B
/// defines, that use becomes use-before-def, so the thread is unsound (for both non-dup and future
/// tail duplication). If B does not dominate S there is nothing to break, since S already has a path
/// not through B and the verifier would have rejected any dominance-use of a B value below S.
fn threadBreaksDominanceUse(
    allocator: std.mem.Allocator,
    func: *const Function,
    doms: *const dom.Dominators,
    cfg: *const cfg_mod.Cfg,
    def_block: []const u32,
    b: Block,
    s: Block,
) pass.Error!bool {
    const bi_b: u32 = @intFromEnum(b);
    const si: u32 = @intFromEnum(s);
    if (!doms.dominates(bi_b, si)) return false; // B does not dominate S: no dominance-use to break

    // Walk the region reachable from S without ever re-entering B. Those are exactly the blocks that
    // lose B as a dominator once P->S bypasses B. A B-defined value used in any of them would break.
    const n = cfg.blockCount();
    const visited = try allocator.alloc(bool, n);
    defer allocator.free(visited);
    @memset(visited, false);

    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(allocator);
    visited[si] = true;
    try stack.append(allocator, si);
    while (stack.pop()) |x| {
        if (blockUsesValueFromBlock(func, @enumFromInt(x), bi_b, def_block)) return true;
        for (cfg.successors(x)) |sx| {
            if (sx == bi_b) continue; // re-entering B keeps B's domination downstream, so cut it here
            if (!visited[sx]) {
                visited[sx] = true;
                try stack.append(allocator, sx);
            }
        }
    }
    return false;
}

/// Whether any operand use in `block` refers to a value defined in the block indexed `def_bi`.
/// Exhaustive over `Opcode` (mirroring the verifier's operand walk) so a newly added op with operands
/// cannot silently escape the check.
fn blockUsesValueFromBlock(func: *const Function, block: Block, def_bi: u32, def_block: []const u32) bool {
    const usesB = struct {
        fn f(db: []const u32, v: Value, target: u32) bool {
            return db[@intFromEnum(v)] == target;
        }
    }.f;
    for (func.blockInsts(block)) |inst| {
        switch (func.opcode(inst)) {
            .iconst, .fconst, .alloca, .global_addr => {},
            .arith => |a| if (usesB(def_block, a.lhs, def_bi) or usesB(def_block, a.rhs, def_bi)) return true,
            .arith_imm => |a| if (usesB(def_block, a.lhs, def_bi)) return true,
            .icmp => |c| if (usesB(def_block, c.lhs, def_bi) or usesB(def_block, c.rhs, def_bi)) return true,
            .select => |sel| if (usesB(def_block, sel.cond, def_bi) or usesB(def_block, sel.then, def_bi) or usesB(def_block, sel.@"else", def_bi)) return true,
            .struct_new => |sn| for (func.valueList(sn.fields)) |field| {
                if (usesB(def_block, field, def_bi)) return true;
            },
            .extract => |ex| if (usesB(def_block, ex.aggregate, def_bi)) return true,
            .convert => |cv| if (usesB(def_block, cv.value, def_bi)) return true,
            .unary => |u| if (usesB(def_block, u.value, def_bi)) return true,
            .load => |ld| if (usesB(def_block, ld.ptr, def_bi)) return true,
            .store => |st| if (usesB(def_block, st.value, def_bi) or usesB(def_block, st.ptr, def_bi)) return true,
            .prefetch => |pf| if (usesB(def_block, pf.ptr, def_bi)) return true,
            .call => |c| for (func.valueList(c.args)) |arg| {
                if (usesB(def_block, arg, def_bi)) return true;
            },
            .call_indirect => |c| {
                if (usesB(def_block, c.target, def_bi)) return true;
                for (func.valueList(c.args)) |arg| if (usesB(def_block, arg, def_bi)) return true;
            },
            .dot => |d| if (usesB(def_block, d.acc, def_bi) or usesB(def_block, d.a, def_bi) or usesB(def_block, d.b, def_bi)) return true,
            .matmul => |mm| if (usesB(def_block, mm.a, def_bi) or usesB(def_block, mm.b, def_bi) or usesB(def_block, mm.c, def_bi)) return true,
            .@"if" => |cf| {
                if (usesB(def_block, cf.cond, def_bi)) return true;
                for (func.blockArgs(cf.then)) |arg| if (usesB(def_block, arg, def_bi)) return true;
                for (func.blockArgs(cf.@"else")) |arg| if (usesB(def_block, arg, def_bi)) return true;
            },
        }
    }
    if (func.terminator(block)) |term| switch (term) {
        .ret => |value| if (value) |v| {
            if (usesB(def_block, v, def_bi)) return true;
        },
        .jump => |j| for (func.blockArgs(j)) |arg| {
            if (usesB(def_block, arg, def_bi)) return true;
        },
    };
    return false;
}

/// Whether a block contains any instruction with a side effect that the non-dup thread would drop.
/// Exhaustive over `Opcode` so a newly added effectful op cannot silently slip through as pure.
fn hasSideEffect(func: *const Function, block: Block) bool {
    for (func.blockInsts(block)) |inst| {
        switch (func.opcode(inst)) {
            .store, .call, .call_indirect, .prefetch, .matmul => return true,
            .iconst,
            .fconst,
            .arith,
            .arith_imm,
            .icmp,
            .select,
            .struct_new,
            .extract,
            .convert,
            .unary,
            .alloca,
            .global_addr,
            .load,
            .dot,
            .@"if",
            => {},
        }
    }
    return false;
}

/// The `@"if"` instruction a block ends in (its last instruction, with no terminator set), else null.
/// This is the shape jump threading recognizes as a conditional block: control reaches the `@"if"`
/// and branches out of the block from there.
fn endsInIf(func: *const Function, block: Block) ?Inst {
    if (func.terminator(block) != null) return null;
    const insts = func.blockInsts(block);
    if (insts.len == 0) return null;
    const last = insts[insts.len - 1];
    return switch (func.opcode(last)) {
        .@"if" => last,
        else => null,
    };
}

/// The final destination for an edge into `target`: follows a chain of identity forwarders (bounded
/// by the block count to avoid looping on a forwarder cycle), or null if `target` is not a forwarder.
fn retarget(forward_to: []const ?Block, target: Block) ?Block {
    var dest = forward_to[@intFromEnum(target)] orelse return null;
    var steps: usize = 0;
    while (steps < forward_to.len) : (steps += 1) {
        const next = forward_to[@intFromEnum(dest)] orelse return dest;
        if (next == dest) return dest; // self-forwarder guard (should not arise)
        dest = next;
    }
    return dest;
}

/// If `block` is a pure identity forwarder (not the entry, no instructions, terminated by a jump to a
/// different block whose arguments are exactly this block's parameters in order), return that
/// successor. Otherwise null.
fn identityForwardTarget(func: *const Function, block: Block) ?Block {
    if (@intFromEnum(block) == 0) return null; // never redirect away from the entry
    if (func.blockInsts(block).len != 0) return null; // must do nothing but forward
    const term = func.terminator(block) orelse return null;
    const jmp = switch (term) {
        .jump => |j| j,
        .ret => return null,
    };
    if (jmp.target == block) return null; // a self-loop is not a forwarder
    const params = func.blockParams(block);
    const args = func.blockArgs(jmp);
    if (!std.mem.eql(ir.function.Value, params, args)) return null; // only the identity case
    return jmp.target;
}

const testing = std.testing;

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

fn i32Ty(func: *Function) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}

fn verifyClean(allocator: std.mem.Allocator, func: *const Function) !void {
    var diags = try ir.verify.verify(allocator, func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

/// A tiny structural interpreter over the opcode subset the threading tests use (constants, integer
/// arithmetic and comparisons, select, `@"if"`, jump, ret). It exists to prove EXECUTION-EQUIVALENCE:
/// the threaded and unthreaded functions must return the same value on every sampled input. Entry
/// params are seeded from `inputs` and booleans are carried as 0/1. Bounded by a step cap so a malformed
/// function cannot hang the test.
fn evalFunc(func: *const Function, inputs: []const i64) !i64 {
    const allocator = testing.allocator;
    const vals = try allocator.alloc(i64, func.valueCount());
    defer allocator.free(vals);

    var cur: Block = @enumFromInt(0);
    for (func.blockParams(cur), 0..) |param, i| vals[@intFromEnum(param)] = inputs[i];

    var steps: usize = 0;
    const cap: usize = 100_000;
    while (steps < cap) : (steps += 1) {
        for (func.blockInsts(cur)) |inst| {
            const result = func.instResult(inst) orelse continue; // side-effect statements: skipped
            vals[@intFromEnum(result)] = evalInst(func, vals, inst);
        }

        // Where control leaves the block: an `@"if"` branch, then the terminator.
        var next_edge: ?Jump = null;
        if (endsInIf(func, cur)) |if_inst| {
            const cf = func.opcode(if_inst).@"if";
            next_edge = if (vals[@intFromEnum(cf.cond)] != 0) cf.then else cf.@"else";
        } else if (func.terminator(cur)) |term| switch (term) {
            .ret => |v| return if (v) |vv| vals[@intFromEnum(vv)] else 0,
            .jump => |j| next_edge = j,
        } else return 0; // implicit ret void

        const edge = next_edge.?;
        const args = func.blockArgs(edge);
        const target_params = func.blockParams(edge.target);
        // Read every arg before writing any param, so param<-arg self-references stay correct.
        var buf: [16]i64 = undefined;
        for (args, 0..) |a, i| buf[i] = vals[@intFromEnum(a)];
        for (target_params, 0..) |param, i| vals[@intFromEnum(param)] = buf[i];
        cur = edge.target;
    }
    return error.NonTerminating;
}

fn evalInst(func: *const Function, vals: []const i64, inst: Inst) i64 {
    return switch (func.opcode(inst)) {
        .iconst => |c| c,
        .arith => |a| applyBin(a.op, vals[@intFromEnum(a.lhs)], vals[@intFromEnum(a.rhs)]),
        .arith_imm => |a| applyBin(a.op, vals[@intFromEnum(a.lhs)], a.imm),
        .icmp => |c| applyCmp(c.op, vals[@intFromEnum(c.lhs)], vals[@intFromEnum(c.rhs)]),
        .select => |s| if (vals[@intFromEnum(s.cond)] != 0) vals[@intFromEnum(s.then)] else vals[@intFromEnum(s.@"else")],
        .alloca => 0, // an opaque stack-slot pointer, never loaded through in these tests (store is skipped)
        else => unreachable, // the threading tests only build the subset above
    };
}

fn applyBin(op: ir.function.BinOp, lhs: i64, rhs: i64) i64 {
    return switch (op) {
        .add => lhs +% rhs,
        .sub => lhs -% rhs,
        .mul => lhs *% rhs,
        .div => @divTrunc(lhs, rhs),
        .rem => @rem(lhs, rhs),
        .bit_and => lhs & rhs,
        .bit_or => lhs | rhs,
        .bit_xor => lhs ^ rhs,
        .shl => lhs << @intCast(rhs),
        .shr => lhs >> @intCast(rhs),
        .mulh => @intCast((@as(i128, lhs) * @as(i128, rhs)) >> 64),
    };
}

fn applyCmp(op: ir.function.CmpOp, lhs: i64, rhs: i64) i64 {
    const r = switch (op) {
        .eq => lhs == rhs,
        .ne => lhs != rhs,
        .lt => lhs < rhs,
        .le => lhs <= rhs,
        .gt => lhs > rhs,
        .ge => lhs >= rhs,
    };
    return if (r) 1 else 0;
}

test "a jump through an identity-forwarding block is redirected to its target" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const entry = try func.appendBlock();
    const mid = try func.appendBlock();
    const dest = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const mp = try func.appendBlockParam(mid, t);
    const dp = try func.appendBlockParam(dest, t);
    try func.setJump(entry, mid, &.{x});
    try func.setJump(mid, dest, &.{mp}); // mid forwards its param straight through
    func.setTerminator(dest, .{ .ret = dp });

    try testing.expect(try runOnce(allocator, &func));
    const term = func.terminator(entry).?;
    try testing.expectEqual(dest, term.jump.target); // entry now jumps past mid
    try testing.expectEqual(x, func.blockArgs(term.jump)[0]);
}

test "an if edge into a forwarding block is redirected" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const mid = try func.appendBlock();
    const other = try func.appendBlock();
    const dest = try func.appendBlock();
    const c = try func.appendBlockParam(entry, bool_t);
    const x = try func.appendBlockParam(entry, t);
    const mp = try func.appendBlockParam(mid, t);
    const dp = try func.appendBlockParam(dest, t);
    const op = try func.appendBlockParam(other, t);
    try func.appendIf(entry, c, .{ .target = mid, .args = &.{x} }, .{ .target = other, .args = &.{x} });
    try func.setJump(mid, dest, &.{mp});
    func.setTerminator(other, .{ .ret = op });
    func.setTerminator(dest, .{ .ret = dp });

    try testing.expect(try runOnce(allocator, &func));
    const cf = func.opcode(func.blockInsts(entry)[0]).@"if";
    try testing.expectEqual(dest, cf.then.target); // then edge threaded past mid
    try testing.expectEqual(other, cf.@"else".target); // else edge unchanged
}

test "a block that computes something is not a forwarder" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const entry = try func.appendBlock();
    const mid = try func.appendBlock();
    const dest = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const mp = try func.appendBlockParam(mid, t);
    const dp = try func.appendBlockParam(dest, t);
    try func.setJump(entry, mid, &.{x});
    const doubled = try func.appendArithImm(mid, t, .mul, mp, 2); // mid does real work
    try func.setJump(mid, dest, &.{doubled});
    func.setTerminator(dest, .{ .ret = dp });

    try testing.expect(!try runOnce(allocator, &func)); // mid is not a pure forwarder
    try testing.expectEqual(mid, func.terminator(entry).?.jump.target);
}

test "jumpthread: constant-param implied branch threads non-duplicating" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const b = try func.appendBlock();
    const d = try func.appendBlock();
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const c = try func.appendBlockParam(b, bool_t);
    const xp = try func.appendBlockParam(b, t);
    const dp = try func.appendBlockParam(d, t);
    // entry passes a TRUE constant for B's condition param, so B's `@"if"` outcome is known.
    const c_true = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.setJump(entry, b, &.{ c_true, x });
    try func.appendIf(b, c, .{ .target = d, .args = &.{xp} }, .{ .target = e, .args = &.{} });
    func.setTerminator(d, .{ .ret = dp });
    const em = try func.appendInst(e, t, .{ .iconst = -1 });
    func.setTerminator(e, .{ .ret = em });

    // Execution before threading, over an input sweep.
    var expected: [7]i64 = undefined;
    for (0..7) |i| expected[i] = try evalFunc(&func, &.{@as(i64, @intCast(i)) - 3});

    try testing.expect(try runOnce(allocator, &func));

    // entry now jumps straight to D, carrying x in place of B's forwarded param.
    const term = func.terminator(entry).?;
    try testing.expectEqual(d, term.jump.target);
    try testing.expectEqual(x, func.blockArgs(term.jump)[0]);
    try verifyClean(allocator, &func);

    // Execution after threading matches, value for value.
    for (0..7) |i| try testing.expectEqual(expected[i], try evalFunc(&func, &.{@as(i64, @intCast(i)) - 3}));
}

test "jumpthread: correlated branch threads non-duplicating" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const p = try func.appendBlock();
    const b = try func.appendBlock();
    const d = try func.appendBlock();
    const e = try func.appendBlock();
    const x_block = try func.appendBlock();
    const v = try func.appendBlockParam(p, bool_t);
    const x = try func.appendBlockParam(p, t);
    const dp = try func.appendBlockParam(d, t);
    // P branches on v to B and B branches on the SAME v, so on P's then-edge v is known TRUE.
    try func.appendIf(p, v, .{ .target = b, .args = &.{} }, .{ .target = x_block, .args = &.{} });
    try func.appendIf(b, v, .{ .target = d, .args = &.{x} }, .{ .target = e, .args = &.{} });
    func.setTerminator(d, .{ .ret = dp });
    const em = try func.appendInst(e, t, .{ .iconst = 7 });
    func.setTerminator(e, .{ .ret = em });
    const xm = try func.appendInst(x_block, t, .{ .iconst = 9 });
    func.setTerminator(x_block, .{ .ret = xm });

    var expected: [8]i64 = undefined;
    for (0..8) |i| {
        const vv: i64 = @intCast(i & 1);
        expected[i] = try evalFunc(&func, &.{ vv, @as(i64, @intCast(i)) });
    }

    try testing.expect(try runOnce(allocator, &func));

    // P's then-edge is threaded past B to D, else-edge untouched.
    const cf = func.opcode(func.blockInsts(p)[0]).@"if";
    try testing.expectEqual(d, cf.then.target);
    try testing.expectEqual(x, func.blockArgs(cf.then)[0]);
    try testing.expectEqual(x_block, cf.@"else".target);
    try verifyClean(allocator, &func);

    for (0..8) |i| {
        const vv: i64 = @intCast(i & 1);
        try testing.expectEqual(expected[i], try evalFunc(&func, &.{ vv, @as(i64, @intCast(i)) }));
    }
}

test "jumpthread: correlated branch on the else edge threads to the else successor" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const p = try func.appendBlock();
    const b = try func.appendBlock();
    const d = try func.appendBlock();
    const e = try func.appendBlock();
    const x_block = try func.appendBlock();
    const v = try func.appendBlockParam(p, bool_t);
    const x = try func.appendBlockParam(p, t);
    const ep = try func.appendBlockParam(e, t);
    const dp = try func.appendBlockParam(d, t);
    // B sits on P's ELSE edge, so on P->B v is known FALSE, selecting B's else-successor E.
    try func.appendIf(p, v, .{ .target = x_block, .args = &.{} }, .{ .target = b, .args = &.{} });
    try func.appendIf(b, v, .{ .target = d, .args = &.{x} }, .{ .target = e, .args = &.{x} });
    func.setTerminator(d, .{ .ret = dp });
    func.setTerminator(e, .{ .ret = ep });
    const xm = try func.appendInst(x_block, t, .{ .iconst = 5 });
    func.setTerminator(x_block, .{ .ret = xm });

    var expected: [8]i64 = undefined;
    for (0..8) |i| {
        const vv: i64 = @intCast(i & 1);
        expected[i] = try evalFunc(&func, &.{ vv, @as(i64, @intCast(i)) });
    }

    try testing.expect(try runOnce(allocator, &func));

    const cf = func.opcode(func.blockInsts(p)[0]).@"if";
    try testing.expectEqual(x_block, cf.then.target); // then-edge untouched
    try testing.expectEqual(e, cf.@"else".target); // else-edge threaded past B to E
    try testing.expectEqual(x, func.blockArgs(cf.@"else")[0]);
    try verifyClean(allocator, &func);

    for (0..8) |i| {
        const vv: i64 = @intCast(i & 1);
        try testing.expectEqual(expected[i], try evalFunc(&func, &.{ vv, @as(i64, @intCast(i)) }));
    }
}

/// Whether `block` holds an instruction with the given opcode tag (used by the tail-dup tests to
/// confirm the duplicate recomputes B's work or reruns its side effect).
fn blockHasOpcode(func: *const Function, block: Block, tag: std.meta.Tag(ir.function.Opcode)) bool {
    for (func.blockInsts(block)) |inst| {
        if (std.meta.activeTag(func.opcode(inst)) == tag) return true;
    }
    return false;
}

test "jumpthread: tail-duplicates a block computing a value used in the threaded successor args" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const b = try func.appendBlock();
    const d = try func.appendBlock();
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const c = try func.appendBlockParam(b, bool_t);
    const xp = try func.appendBlockParam(b, t);
    const dp = try func.appendBlockParam(d, t);
    // entry passes a TRUE constant for B's condition, so on entry->B the `@"if"` selects D. But D's
    // arg y is COMPUTED IN B, so it is not available on the entry path: non-dup is unsound, tail-dup
    // must fire (recompute y in a duplicate B', then jump to D).
    const c_true = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.setJump(entry, b, &.{ c_true, x });
    const y = try func.appendArithImm(b, t, .add, xp, 1);
    try func.appendIf(b, c, .{ .target = d, .args = &.{y} }, .{ .target = e, .args = &.{} });
    func.setTerminator(d, .{ .ret = dp });
    const em = try func.appendInst(e, t, .{ .iconst = -1 });
    func.setTerminator(e, .{ .ret = em });

    const before_blocks = func.blockCount();
    var expected: [7]i64 = undefined;
    for (0..7) |i| expected[i] = try evalFunc(&func, &.{@as(i64, @intCast(i)) - 3});

    try testing.expect(try runOnce(allocator, &func));

    // entry now targets a fresh duplicate B', NOT B, and passes it the same args it passed to B.
    const entry_term = func.terminator(entry).?;
    const bprime = entry_term.jump.target;
    try testing.expect(bprime != b);
    try testing.expect(func.blockCount() > before_blocks); // a clone was appended
    try testing.expectEqual(x, func.blockArgs(entry_term.jump)[1]);
    // B' recomputes y (an arith) then jumps straight to D with no `@"if"` left to branch on.
    try testing.expect(blockHasOpcode(&func, bprime, .arith_imm));
    try testing.expect(endsInIf(&func, bprime) == null);
    try testing.expectEqual(d, func.terminator(bprime).?.jump.target);
    // B itself is untouched (still a conditional block) for any other predecessors.
    try testing.expect(endsInIf(&func, b) != null);
    try verifyClean(allocator, &func);

    // Execution after threading matches the original, value for value, across the input sweep.
    for (0..7) |i| try testing.expectEqual(expected[i], try evalFunc(&func, &.{@as(i64, @intCast(i)) - 3}));
}

test "jumpthread: tail-duplicates a side-effecting B so the effect runs on the threaded path" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const b = try func.appendBlock();
    const d = try func.appendBlock();
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const c = try func.appendBlockParam(b, bool_t);
    const xp = try func.appendBlockParam(b, t);
    const dp = try func.appendBlockParam(d, t);
    const slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = t } });
    const c_true = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.setJump(entry, b, &.{ c_true, x });
    // Every value is available at entry, but B stores to memory: skipping B would lose that write, so
    // non-dup is unsound. Tail-dup reruns the store on the threaded path.
    try func.appendStore(b, xp, slot);
    try func.appendIf(b, c, .{ .target = d, .args = &.{xp} }, .{ .target = e, .args = &.{} });
    func.setTerminator(d, .{ .ret = dp });
    const em = try func.appendInst(e, t, .{ .iconst = 0 });
    func.setTerminator(e, .{ .ret = em });

    var expected: [7]i64 = undefined;
    for (0..7) |i| expected[i] = try evalFunc(&func, &.{@as(i64, @intCast(i)) - 3});

    try testing.expect(try runOnce(allocator, &func));

    // entry now targets a duplicate B' that still contains the store (the effect runs on this path).
    const bprime = func.terminator(entry).?.jump.target;
    try testing.expect(bprime != b);
    try testing.expect(blockHasOpcode(&func, bprime, .store)); // the side effect was duplicated, not dropped
    try testing.expect(endsInIf(&func, bprime) == null);
    try testing.expectEqual(d, func.terminator(bprime).?.jump.target);
    try verifyClean(allocator, &func);

    for (0..7) |i| try testing.expectEqual(expected[i], try evalFunc(&func, &.{@as(i64, @intCast(i)) - 3}));
}

test "jumpthread: the bloat budget caps tail duplication of an oversized block" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const b = try func.appendBlock();
    const d = try func.appendBlock();
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const c = try func.appendBlockParam(b, bool_t);
    const xp = try func.appendBlockParam(b, t);
    const dp = try func.appendBlockParam(d, t);
    const c_true = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.setJump(entry, b, &.{ c_true, x });
    // B is far bigger than `tail_dup_block_cap`: a long arithmetic chain, then a B-computed value in
    // the threaded successor args. Non-dup is unsound (y is defined in B), and B is too big to
    // duplicate, so the edge is left un-threaded.
    var acc = xp;
    for (0..tail_dup_block_cap + 2) |_| acc = try func.appendArithImm(b, t, .add, acc, 1);
    try func.appendIf(b, c, .{ .target = d, .args = &.{acc} }, .{ .target = e, .args = &.{} });
    func.setTerminator(d, .{ .ret = dp });
    const em = try func.appendInst(e, t, .{ .iconst = 0 });
    func.setTerminator(e, .{ .ret = em });

    try testing.expect(!try runOnce(allocator, &func)); // over the per-block cap: not duplicated
    try testing.expectEqual(b, func.terminator(entry).?.jump.target); // entry still jumps B
    try verifyClean(allocator, &func);
}

test "jumpthread: an edge is NOT threaded when B dominates the target (dominance-use in S would break)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const b = try func.appendBlock();
    const s = try func.appendBlock();
    const e = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const cp = try func.appendBlockParam(b, bool_t);
    const xp = try func.appendBlockParam(b, t);
    // entry passes TRUE for B's condition, so B's `@"if"` outcome is known and would otherwise thread.
    const c_true = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.setJump(entry, b, &.{ c_true, x });
    // B computes z (pure, no side effect) and branches to S. S has no params and B is its only
    // predecessor, so B DOMINATES S, and S reads z by DOMINANCE (no edge arg carries it).
    const z = try func.appendArithImm(b, t, .mul, xp, 2);
    try func.appendIf(b, cp, .{ .target = s, .args = &.{} }, .{ .target = e, .args = &.{} });
    const w = try func.appendArithImm(s, t, .add, z, 1); // dominance-use of z, defined in B
    func.setTerminator(s, .{ .ret = w });
    const em = try func.appendInst(e, t, .{ .iconst = 0 });
    func.setTerminator(e, .{ .ret = em });

    // The original function is well formed: B dominates S so z's dominance-use verifies.
    try verifyClean(allocator, &func);

    // Threading entry->B to entry->S would bypass B, undefining z on that path, so it must NOT happen.
    try testing.expect(!try runOnce(allocator, &func));
    try testing.expectEqual(b, func.terminator(entry).?.jump.target); // entry still jumps B
    try verifyClean(allocator, &func); // still well formed, no use-before-def introduced
}

test "jumpthread: reaches a fixpoint and terminates on a small loop" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const h = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();
    const i = try func.appendBlockParam(h, t);
    const bi = try func.appendBlockParam(body, t);
    const ep = try func.appendBlockParam(exit, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, h, &.{zero});
    // The header's condition is COMPUTED in the header, not implied on any edge, so nothing threads
    // and the pass must still terminate (no hang) via the step cap and edge-removal progress.
    const ten = try func.appendInst(h, t, .{ .iconst = 10 });
    const cond = try func.appendInst(h, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = ten } });
    try func.appendIf(h, cond, .{ .target = body, .args = &.{i} }, .{ .target = exit, .args = &.{i} });
    const inc = try func.appendArithImm(body, t, .add, bi, 1);
    try func.setJump(body, h, &.{inc});
    func.setTerminator(exit, .{ .ret = ep });

    try testing.expect(!try runOnce(allocator, &func)); // no implied edge, so no change (and no hang)
    try verifyClean(allocator, &func);
    try testing.expectEqual(@as(i64, 10), try evalFunc(&func, &.{})); // loop still counts to 10
}

test "jumpthread: a dispatch shape with tail-dup reaches a fixpoint (terminates, bounded, correct)" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlock();
    const c_blk = try func.appendBlock();
    const b = try func.appendBlock();
    const d = try func.appendBlock();
    const e = try func.appendBlock();
    const sel = try func.appendBlockParam(entry, bool_t);
    const x = try func.appendBlockParam(entry, t);
    const cp = try func.appendBlockParam(b, bool_t);
    const xp = try func.appendBlockParam(b, t);
    const dp = try func.appendBlockParam(d, t);
    // Two predecessors A and C both funnel into the shared conditional block B, each passing a TRUE
    // constant for B's condition. B computes y and forwards it to D, so every A->B and C->B edge is an
    // implied thread whose successor arg is B-computed: each must tail-duplicate B. The pass must
    // duplicate B once per predecessor edge, then reach a fixpoint (no hang, bounded growth).
    const c_true = try func.appendInst(entry, bool_t, .{ .iconst = 1 });
    try func.appendIf(entry, sel, .{ .target = a, .args = &.{} }, .{ .target = c_blk, .args = &.{} });
    try func.setJump(a, b, &.{ c_true, x });
    try func.setJump(c_blk, b, &.{ c_true, x });
    const y = try func.appendArithImm(b, t, .add, xp, 1);
    try func.appendIf(b, cp, .{ .target = d, .args = &.{y} }, .{ .target = e, .args = &.{} });
    func.setTerminator(d, .{ .ret = dp });
    const em = try func.appendInst(e, t, .{ .iconst = -1 });
    func.setTerminator(e, .{ .ret = em });

    const before_blocks = func.blockCount();
    var expected: [2]i64 = undefined;
    for (0..2) |i| expected[i] = try evalFunc(&func, &.{ @as(i64, @intCast(i)), 41 });

    try testing.expect(try runOnce(allocator, &func)); // duplicates B for both A and C, then stops

    // Bounded duplication: at most one fresh block per predecessor edge (here two).
    try testing.expect(func.blockCount() <= before_blocks + 2);
    // Neither A nor C targets the original B any more: both were threaded onto duplicates.
    try testing.expect(func.terminator(a).?.jump.target != b);
    try testing.expect(func.terminator(c_blk).?.jump.target != b);
    try verifyClean(allocator, &func);

    // Execution is equivalent on both dispatch arms, and the interpreter terminates (no hang).
    for (0..2) |i| try testing.expectEqual(expected[i], try evalFunc(&func, &.{ @as(i64, @intCast(i)), 41 }));
}
