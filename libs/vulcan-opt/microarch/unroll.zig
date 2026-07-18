//! Model-driven loop unrolling. Unrolls hot innermost loops by a factor derived from the target's
//! issue width and latency, to expose instruction-level parallelism a wide out-of-order core can
//! use. Conservative: only reducible, innermost, single-latch loops with a pure test header and
//! header-param loop-carried values are transformed, everything else is skipped unchanged. The
//! transform is a guarded partial unroll (K guarded body copies), correct by construction, proven
//! by the differential JIT tests in libs/vulcan-target/tests/unroll_differential.zig.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const loops = @import("../loops.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Opcode = ir.function.Opcode;
const Jump = ir.function.Jump;
const Terminator = ir.function.Terminator;

/// Maps original values to their clones, for a single cloneBlocks call. The
/// caller may pre-seed entries for external value substitutions (e.g. a
/// loop-carried value entering a clone) before calling.
pub const ValueMap = std.AutoHashMapUnmanaged(Value, Value);

/// Maps original blocks to their clones, for a single cloneBlocks call. The
/// caller may pre-seed entries for branch retargeting before calling.
pub const BlockMap = std.AutoHashMapUnmanaged(Block, Block);

pub const Error = std.mem.Allocator.Error;

/// Deep-copies `blocks` into `func`, remapping every Value operand through
/// `value_map` and every Block target through `block_map`. Two passes: the
/// first creates every cloned block and its params (so forward branches and
/// cross-block value uses resolve), the second clones each instruction and
/// terminator with remapped operands. `value_map` and `block_map` may already
/// hold entries (external substitutions), and are grown with the clone's new
/// value/block correspondences. Returns a freshly allocated slice of the
/// cloned blocks, in the same order as `blocks` (caller owns it).
pub fn cloneBlocks(
    allocator: std.mem.Allocator,
    func: *Function,
    blocks: []const Block,
    value_map: *ValueMap,
    block_map: *BlockMap,
) Error![]Block {
    const remapValue = struct {
        fn call(map: *const ValueMap, v: Value) Value {
            return map.get(v) orelse v;
        }
    }.call;
    const remapBlock = struct {
        fn call(map: *const BlockMap, b: Block) Block {
            return map.get(b) orelse b;
        }
    }.call;

    // Pass 1: create the cloned blocks and their params, so any forward branch
    // target or cross-block value use resolves in pass 2.
    const clones = try allocator.alloc(Block, blocks.len);
    errdefer allocator.free(clones);
    for (blocks, 0..) |b, i| {
        const cloned = try func.appendBlock();
        try block_map.put(allocator, b, cloned);
        clones[i] = cloned;
        for (func.blockParams(b)) |p| {
            const cloned_p = try func.appendBlockParam(cloned, func.valueType(p));
            try value_map.put(allocator, p, cloned_p);
        }
    }

    // Pass 2: clone each instruction and the terminator, with every Value
    // operand mapped through value_map and every Block target through
    // block_map.
    var args_buf: std.ArrayList(Value) = .empty;
    defer args_buf.deinit(allocator);

    for (blocks, 0..) |b, i| {
        const cloned = clones[i];
        for (func.blockInsts(b)) |inst| {
            const op = func.opcode(inst);
            const rebuilt: Opcode = switch (op) {
                .iconst => |v| .{ .iconst = v },
                .fconst => |v| .{ .fconst = v },
                .arith => |a| .{ .arith = .{
                    .op = a.op,
                    .lhs = remapValue(value_map, a.lhs),
                    .rhs = remapValue(value_map, a.rhs),
                } },
                .arith_imm => |a| .{ .arith_imm = .{
                    .op = a.op,
                    .lhs = remapValue(value_map, a.lhs),
                    .imm = a.imm,
                } },
                .icmp => |c| .{ .icmp = .{
                    .op = c.op,
                    .lhs = remapValue(value_map, c.lhs),
                    .rhs = remapValue(value_map, c.rhs),
                } },
                .select => |s| .{ .select = .{
                    .cond = remapValue(value_map, s.cond),
                    .then = remapValue(value_map, s.then),
                    .@"else" = remapValue(value_map, s.@"else"),
                } },
                .struct_new => |sn| blk: {
                    args_buf.clearRetainingCapacity();
                    for (func.valueList(sn.fields)) |v| {
                        try args_buf.append(allocator, remapValue(value_map, v));
                    }
                    break :blk .{ .struct_new = .{ .fields = try func.internValues(args_buf.items) } };
                },
                .extract => |e| .{ .extract = .{
                    .aggregate = remapValue(value_map, e.aggregate),
                    .index = e.index,
                } },
                .convert => |cv| .{ .convert = .{ .value = remapValue(value_map, cv.value) } },
                .unary => |u| .{ .unary = .{ .op = u.op, .value = remapValue(value_map, u.value) } },
                .alloca => |a| .{ .alloca = .{ .elem = a.elem } },
                .call => |c| blk: {
                    args_buf.clearRetainingCapacity();
                    for (func.valueList(c.args)) |v| {
                        try args_buf.append(allocator, remapValue(value_map, v));
                    }
                    break :blk .{ .call = .{ .symbol = c.symbol, .args = try func.internValues(args_buf.items) } };
                },
                .call_indirect => |c| blk: {
                    args_buf.clearRetainingCapacity();
                    for (func.valueList(c.args)) |v| {
                        try args_buf.append(allocator, remapValue(value_map, v));
                    }
                    break :blk .{ .call_indirect = .{
                        .target = remapValue(value_map, c.target),
                        .args = try func.internValues(args_buf.items),
                    } };
                },
                .global_addr => |g| .{ .global_addr = .{ .symbol = g.symbol } },
                .load => |l| .{ .load = .{ .ptr = remapValue(value_map, l.ptr) } },
                .store => |st| .{ .store = .{
                    .value = remapValue(value_map, st.value),
                    .ptr = remapValue(value_map, st.ptr),
                } },
                .prefetch => |pf| .{ .prefetch = .{
                    .ptr = remapValue(value_map, pf.ptr),
                } },
                .dot => |d| .{ .dot = .{
                    .acc = remapValue(value_map, d.acc),
                    .a = remapValue(value_map, d.a),
                    .b = remapValue(value_map, d.b),
                } },
                .matmul => |mmv| .{
                    .matmul = .{
                        .a = remapValue(value_map, mmv.a),
                        .b = remapValue(value_map, mmv.b),
                        .c = remapValue(value_map, mmv.c),
                        .m = mmv.m,
                        .n = mmv.n,
                        .k = mmv.k,
                        .dtype = mmv.dtype,
                        .accumulate = mmv.accumulate,
                        // Copied verbatim, including a `per_column` scale's ScaleList handle: unrolling
                        // clones within the SAME function, so the handle stays relative to the same
                        // `scale_pool` and needs no re-interning (unlike inline.zig's cross-function clone).
                        .quant = mmv.quant,
                        // Plain op metadata (not a Value, not a pool handle), so it copies as-is.
                        .input_signs = mmv.input_signs,
                    },
                },
                .@"if" => |cf| blk: {
                    args_buf.clearRetainingCapacity();
                    for (func.valueList(cf.then.args)) |v| {
                        try args_buf.append(allocator, remapValue(value_map, v));
                    }
                    const then_args = try func.internValues(args_buf.items);
                    const then_jump: Jump = .{ .target = remapBlock(block_map, cf.then.target), .args = then_args };

                    args_buf.clearRetainingCapacity();
                    for (func.valueList(cf.@"else".args)) |v| {
                        try args_buf.append(allocator, remapValue(value_map, v));
                    }
                    const else_args = try func.internValues(args_buf.items);
                    const else_jump: Jump = .{ .target = remapBlock(block_map, cf.@"else".target), .args = else_args };

                    break :blk .{ .@"if" = .{
                        .cond = remapValue(value_map, cf.cond),
                        .then = then_jump,
                        .@"else" = else_jump,
                    } };
                },
            };

            switch (rebuilt) {
                .store, .prefetch, .matmul, .@"if" => _ = try func.appendStmtRaw(cloned, rebuilt),
                else => {
                    const result = func.instResult(inst) orelse unreachable;
                    const cloned_result = try func.appendInst(cloned, func.valueType(result), rebuilt);
                    try value_map.put(allocator, result, cloned_result);
                },
            }
        }

        if (func.terminator(b)) |term| {
            const rebuilt_term: Terminator = switch (term) {
                .ret => |maybe_v| .{ .ret = if (maybe_v) |v| remapValue(value_map, v) else null },
                .jump => |j| blk: {
                    args_buf.clearRetainingCapacity();
                    for (func.valueList(j.args)) |v| {
                        try args_buf.append(allocator, remapValue(value_map, v));
                    }
                    break :blk .{ .jump = .{
                        .target = remapBlock(block_map, j.target),
                        .args = try func.internValues(args_buf.items),
                    } };
                },
            };
            func.setTerminator(cloned, rebuilt_term);
        }
    }

    return clones;
}

/// The largest factor we ever unroll by (keeps code growth bounded).
const MAX_FACTOR: u32 = 8;

/// How many times to unroll a loop whose body has `body_ops` instructions for `model`. Returns 1
/// (no unroll) for a single-issue in-order model, since it has no width to fill. For a wider model,
/// enough copies to keep issue_width ports busy across the dominant latency, capped at MAX_FACTOR
/// and never more than makes sense for the body size.
pub fn unrollFactor(model: *const mm.Model, body_ops: u32) u32 {
    if (model.issue_width <= 1) return 1; // in-order single-issue gains nothing
    if (body_ops == 0) return 1;
    // Rough ILP target: cover the issue width across a typical multi-cycle latency (use 3 as a
    // representative arithmetic latency), divided by the work already in one body.
    const target = (@as(u32, model.issue_width) * 3) / body_ops;
    return std.math.clamp(target, 1, MAX_FACTOR);
}

test "unrollFactor is 1 for single-issue in-order models" {
    const registry = @import("registry.zig");
    try std.testing.expectEqual(@as(u32, 1), unrollFactor(registry.modelFor(.@"et-soc"), 2));
    try std.testing.expectEqual(@as(u32, 1), unrollFactor(registry.modelFor(.@"river-rc1.s"), 2));
}

test "unrollFactor grows with issue width for a wide model, bounded" {
    const registry = @import("registry.zig");
    // ampere-altra issue_width 4: (4*3)/2 = 6, clamped to MAX_FACTOR 8 -> 6.
    try std.testing.expectEqual(@as(u32, 6), unrollFactor(registry.modelFor(.@"ampere-altra"), 2));
    // A large body needs no unrolling.
    try std.testing.expectEqual(@as(u32, 1), unrollFactor(registry.modelFor(.@"ampere-altra"), 100));
    // The factor never exceeds MAX_FACTOR.
    try std.testing.expect(unrollFactor(registry.modelFor(.@"ampere-altra"), 1) <= 8);
}

/// A vetted, eligible loop plus everything the transform needs, snapshotted so
/// later mutation (we only ever append blocks/values) cannot invalidate it.
const Plan = struct {
    header: Block,
    if_inst: Inst,
    exit: Block, // E: the single out-of-loop successor
    body_entry: Block, // Be: the in-loop successor of the header's `if`
    latch: Block, // L: the single block whose jump closes the back-edge
    body_blocks: []Block, // the loop minus the header, in block order (owned)
    in_loop: []bool, // owned copy of the loop's body bitset
    factor: u32, // K >= 2
};

/// Whether `b` is inside the loop described by `in_loop`. Blocks added after the
/// bitset was captured (index past its end) are, by construction, outside.
fn inLoop(in_loop: []const bool, b: Block) bool {
    const idx = @intFromEnum(b);
    return idx < in_loop.len and in_loop[idx];
}

/// Remap a value through a value map (identity for values not in the map).
fn rv(map: *const ValueMap, v: Value) Value {
    return map.get(v) orelse v;
}

/// Model-driven guarded partial unroll. Analyzes `func`'s natural loops, unrolls
/// every loop it can prove eligible by `unrollFactor(model, ...)` copies, and
/// leaves everything else untouched. Returns whether anything changed. An
/// ineligible or un-cleanly-transformable loop is always left exactly as it was.
/// Idempotence note: re-running `run` on an already-unrolled function does not
/// re-unroll it, because the body has grown (extra guard blocks and copies),
/// so `unrollFactor` collapses to 1 and `eligible` below rejects it again.
pub fn run(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) Error!bool {
    var info = try loops.analyze(allocator, func);
    defer info.deinit(allocator);

    // Snapshot all eligible loops before mutating (mutation invalidates the
    // analysis; only appends happen, so the snapshot's handles stay valid).
    var plans: std.ArrayList(Plan) = .empty;
    defer {
        for (plans.items) |*p| {
            allocator.free(p.body_blocks);
            allocator.free(p.in_loop);
        }
        plans.deinit(allocator);
    }

    for (info.loops) |*loop| {
        if (try eligible(allocator, func, model, info.loops, loop)) |plan| {
            try plans.append(allocator, plan);
        }
    }

    for (plans.items) |*plan| try apply(allocator, func, plan);
    return plans.items.len != 0;
}

/// Step A: prove a loop eligible and gather a Plan, or return null to skip it.
fn eligible(
    allocator: std.mem.Allocator,
    func: *Function,
    model: *const mm.Model,
    all_loops: []const loops.Loop,
    loop: *const loops.Loop,
) Error!?Plan {
    const n = func.blockCount();
    const h_idx = loop.header;
    const header: Block = @enumFromInt(h_idx);
    const in_loop = loop.body; // borrowed, length == blockCount at analysis time

    // A preheader must exist (a single, clean entry into the loop).
    if (loop.preheader == null) return null;

    // Innermost only: no other loop's header lies inside this loop's body.
    for (all_loops) |*other| {
        if (other.header == h_idx) continue;
        if (other.header < in_loop.len and in_loop[other.header]) return null;
    }

    // Pure test header: every instruction is a pure value op, save exactly one
    // `if` which must be the last instruction. No side-effecting ops allowed.
    const h_insts = func.blockInsts(header);
    if (h_insts.len == 0) return null;
    var if_inst: ?Inst = null;
    for (h_insts, 0..) |inst, idx| {
        switch (func.opcode(inst)) {
            .@"if" => {
                if (idx != h_insts.len - 1) return null; // the `if` must end the block
                if_inst = inst;
            },
            .iconst, .fconst, .arith, .arith_imm, .icmp, .select, .convert, .unary, .extract, .struct_new, .dot => {},
            // load/store/call/call_indirect/alloca/global_addr are impure or memory ops.
            else => return null,
        }
    }
    const iff = if_inst orelse return null;
    // The header's control is the `if`; any explicit terminator other than an
    // implicit/void return means a shape we do not model.
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
    if (@intFromEnum(body_entry) == h_idx) return null; // need a real body to clone

    // Single latch: exactly one in-loop block whose *terminator* jumps back to
    // the header. Conditional (if-edge) back-edges are not modeled.
    var latch: ?Block = null;
    var bi: usize = 0;
    while (bi < n) : (bi += 1) {
        if (!(bi < in_loop.len and in_loop[bi])) continue;
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

    // Single exit: the header's `if` exit edge is the *only* edge leaving the
    // loop. Any other out-of-loop edge is a second exit we do not model.
    var out_edges: usize = 0;
    bi = 0;
    while (bi < n) : (bi += 1) {
        if (!(bi < in_loop.len and in_loop[bi])) continue;
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

    // The back-edge must feed exactly the header's parameters.
    const back_args = switch (func.terminator(l).?) {
        .jump => |j| func.blockArgs(j),
        .ret => return null,
    };
    if (back_args.len != func.blockParams(header).len) return null;

    // The body must not read a value the header computes (a header *instruction*
    // result). We clone the body before recomputing the header test, so such a
    // reference could not be remapped to the current iteration. Header params
    // are fine (they map to the current carried values).
    var header_results: std.AutoHashMapUnmanaged(Value, void) = .empty;
    defer header_results.deinit(allocator);
    for (h_insts) |inst| {
        if (inst == iff) continue;
        if (func.instResult(inst)) |r| try header_results.put(allocator, r, {});
    }
    bi = 0;
    while (bi < n) : (bi += 1) {
        if (!(bi < in_loop.len and in_loop[bi])) continue;
        if (bi == h_idx) continue;
        var uses: std.AutoHashMapUnmanaged(Value, void) = .empty;
        defer uses.deinit(allocator);
        try collectOperands(func, @enumFromInt(bi), &uses, allocator);
        var it = uses.keyIterator();
        while (it.next()) |u| {
            if (header_results.contains(u.*)) return null;
        }
    }

    // Body instruction budget drives the unroll factor.
    var body_ops: u32 = 0;
    bi = 0;
    while (bi < n) : (bi += 1) {
        if (bi < in_loop.len and in_loop[bi]) body_ops += @intCast(func.blockInsts(@enumFromInt(bi)).len);
    }
    const factor = unrollFactor(model, body_ops);
    if (factor < 2) return null;

    // Snapshot the body blocks (loop minus header) and the body bitset.
    var body_list: std.ArrayList(Block) = .empty;
    errdefer body_list.deinit(allocator);
    bi = 0;
    while (bi < n) : (bi += 1) {
        if (bi < in_loop.len and in_loop[bi] and bi != h_idx) try body_list.append(allocator, @enumFromInt(bi));
    }
    const in_loop_copy = try allocator.dupe(bool, in_loop);
    errdefer allocator.free(in_loop_copy);

    return Plan{
        .header = header,
        .if_inst = iff,
        .exit = exit,
        .body_entry = body_entry,
        .latch = l,
        .body_blocks = try body_list.toOwnedSlice(allocator),
        .in_loop = in_loop_copy,
        .factor = factor,
    };
}

/// Steps B and C: rewrite escaping values into loop-closed form, then splice K-1
/// guarded body copies between the header entry and the back-edge.
fn apply(allocator: std.mem.Allocator, func: *Function, plan: *const Plan) Error!void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    // --- Step B: loop-closed SSA for values escaping the loop ---
    // Every value used by a block outside the loop.
    var used_out: std.AutoHashMapUnmanaged(Value, void) = .empty;
    var b: usize = 0;
    while (b < func.blockCount()) : (b += 1) {
        if (inLoop(plan.in_loop, @enumFromInt(b))) continue;
        try collectOperands(func, @enumFromInt(b), &used_out, a);
    }
    // Values defined inside the loop and used outside it, in definition order.
    var escaping: std.ArrayList(Value) = .empty;
    b = 0;
    while (b < func.blockCount()) : (b += 1) {
        const blk: Block = @enumFromInt(b);
        if (!inLoop(plan.in_loop, blk)) continue;
        for (func.blockParams(blk)) |p| {
            if (used_out.contains(p)) try escaping.append(a, p);
        }
        for (func.blockInsts(blk)) |inst| {
            if (func.instResult(inst)) |r| {
                if (used_out.contains(r)) try escaping.append(a, r);
            }
        }
    }
    const esc = try a.dupe(Value, escaping.items);

    // Add an exit param per escaping value and route every *outside* use through
    // it, so the exit block reads its values from its params (each exit edge
    // then supplies the current-iteration values).
    for (esc) |v| {
        const p = try func.appendBlockParam(plan.exit, func.valueType(v));
        var bx: usize = 0;
        while (bx < func.blockCount()) : (bx += 1) {
            const blk: Block = @enumFromInt(bx);
            if (inLoop(plan.in_loop, blk)) continue;
            replaceInBlock(func, blk, v, p);
        }
    }
    // Append the (original) escaping values to the header's exit edge, matching
    // the new exit params. The header is in-loop, so the replace above left it
    // untouched and these still name the original values.
    if (esc.len > 0) {
        const cfp = func.opcodeMut(plan.if_inst);
        const exit_is_then = cfp.@"if".then.target == plan.exit;
        const cur = if (exit_is_then) func.valueList(cfp.@"if".then.args) else func.valueList(cfp.@"if".@"else".args);
        var buf: std.ArrayList(Value) = .empty;
        try buf.appendSlice(a, cur);
        try buf.appendSlice(a, esc);
        const newlist = try func.internValues(buf.items);
        const cfp2 = func.opcodeMut(plan.if_inst);
        if (exit_is_then) cfp2.@"if".then.args = newlist else cfp2.@"if".@"else".args = newlist;
    }

    // --- Step C: guarded unroll by K ---
    // Snapshot the header test's pieces (cloning below mutates the value pool).
    const cf = func.opcode(plan.if_inst).@"if";
    const cond0 = cf.cond;
    const in_is_then = cf.then.target == plan.body_entry;
    const in_args = try a.dupe(Value, func.blockArgs(if (in_is_then) cf.then else cf.@"else"));
    const ex_args = try a.dupe(Value, func.blockArgs(if (in_is_then) cf.@"else" else cf.then));
    const hparams = try a.dupe(Value, func.blockParams(plan.header));

    // The header's pure instructions (everything but the `if`), to re-emit as
    // each guard's test.
    var pure_list: std.ArrayList(Inst) = .empty;
    for (func.blockInsts(plan.header)) |inst| {
        if (inst != plan.if_inst) try pure_list.append(a, inst);
    }
    const h_pure = pure_list.items;

    // One guard per extra body copy.
    const Guard = struct { ib: Block, carried: []Value, body_entry: Block };
    var guards: std.ArrayList(Guard) = .empty;

    // Phase 1: clone the K-1 extra body copies from the pristine originals. The
    // carried values chain iteration to iteration (each clone's back-edge args
    // feed the next). No original block is mutated yet, so every clone is clean.
    var carried = try a.dupe(Value, func.blockArgs(func.terminator(plan.latch).?.jump));
    var ib = plan.latch; // the block whose back-edge the next guard replaces
    var copy: u32 = 1;
    while (copy < plan.factor) : (copy += 1) {
        var vmap: ValueMap = .empty;
        var bmap: BlockMap = .empty;
        for (hparams, carried) |hp, cv| try vmap.put(a, hp, cv);

        const cloned = try cloneBlocks(a, func, plan.body_blocks, &vmap, &bmap);
        _ = cloned;
        const be_i = bmap.get(plan.body_entry).?;
        const l_i = bmap.get(plan.latch).?;
        const u_i = try a.dupe(Value, func.blockArgs(func.terminator(l_i).?.jump));

        try guards.append(a, .{ .ib = ib, .carried = carried, .body_entry = be_i });
        // The cloned latch (still jumping to the header) becomes the next
        // insertion block; the last clone keeps that jump as the real back-edge.
        carried = u_i;
        ib = l_i;
    }

    // Phase 2: wire each guard. Re-emit the header's pure test over the guard's
    // carried values into its insertion block, then replace that block's
    // back-edge with the guarded branch (run the copy, or exit with the current
    // escaping values). Safe now: all body copies already exist.
    for (guards.items) |g| {
        var vmap: ValueMap = .empty;
        for (hparams, g.carried) |hp, cv| try vmap.put(a, hp, cv);
        for (h_pure) |inst| try cloneInstInto(func, g.ib, inst, &vmap, a);

        func.terminatorPtr(g.ib).* = null;
        const then_args_i = try remapArgs(&vmap, in_args, a);
        const else_args_i = try remapArgs(&vmap, ex_args, a);
        try func.appendIf(
            g.ib,
            rv(&vmap, cond0),
            .{ .target = g.body_entry, .args = then_args_i },
            .{ .target = plan.exit, .args = else_args_i },
        );
    }
}

/// Copy `src`, remapping each value through `map`.
fn remapArgs(map: *const ValueMap, src: []const Value, a: std.mem.Allocator) Error![]Value {
    const out = try a.alloc(Value, src.len);
    for (src, 0..) |v, i| out[i] = rv(map, v);
    return out;
}

/// Clone one pure (value-producing) instruction into `dest`, remapping operands
/// through `vmap` and recording result -> clone. Only the pure opcodes a vetted
/// header can hold are handled; anything else is unreachable by eligibility.
fn cloneInstInto(func: *Function, dest: Block, inst: Inst, vmap: *ValueMap, a: std.mem.Allocator) Error!void {
    const rebuilt: Opcode = switch (func.opcode(inst)) {
        .iconst => |v| .{ .iconst = v },
        .fconst => |v| .{ .fconst = v },
        .arith => |x| .{ .arith = .{ .op = x.op, .lhs = rv(vmap, x.lhs), .rhs = rv(vmap, x.rhs) } },
        .arith_imm => |x| .{ .arith_imm = .{ .op = x.op, .lhs = rv(vmap, x.lhs), .imm = x.imm } },
        .icmp => |x| .{ .icmp = .{ .op = x.op, .lhs = rv(vmap, x.lhs), .rhs = rv(vmap, x.rhs) } },
        .select => |x| .{ .select = .{ .cond = rv(vmap, x.cond), .then = rv(vmap, x.then), .@"else" = rv(vmap, x.@"else") } },
        .convert => |x| .{ .convert = .{ .value = rv(vmap, x.value) } },
        .unary => |x| .{ .unary = .{ .op = x.op, .value = rv(vmap, x.value) } },
        .extract => |x| .{ .extract = .{ .aggregate = rv(vmap, x.aggregate), .index = x.index } },
        .struct_new => |x| blk: {
            var fields: std.ArrayList(Value) = .empty;
            for (func.valueList(x.fields)) |v| try fields.append(a, rv(vmap, v));
            break :blk .{ .struct_new = .{ .fields = try func.internValues(fields.items) } };
        },
        .dot => |x| .{ .dot = .{ .acc = rv(vmap, x.acc), .a = rv(vmap, x.a), .b = rv(vmap, x.b) } },
        else => unreachable, // eligibility guarantees a pure test header
    };
    const result = func.instResult(inst).?;
    const cloned = try func.appendInst(dest, func.valueType(result), rebuilt);
    try vmap.put(a, result, cloned);
}

/// Add every value operand used by `block` (instructions, `if` edges, and the
/// terminator) to `set`.
fn collectOperands(
    func: *Function,
    block: Block,
    set: *std.AutoHashMapUnmanaged(Value, void),
    a: std.mem.Allocator,
) Error!void {
    for (func.blockInsts(block)) |inst| {
        switch (func.opcode(inst)) {
            .iconst, .fconst, .alloca, .global_addr => {},
            .arith => |x| {
                try set.put(a, x.lhs, {});
                try set.put(a, x.rhs, {});
            },
            .arith_imm => |x| try set.put(a, x.lhs, {}),
            .icmp => |x| {
                try set.put(a, x.lhs, {});
                try set.put(a, x.rhs, {});
            },
            .select => |x| {
                try set.put(a, x.cond, {});
                try set.put(a, x.then, {});
                try set.put(a, x.@"else", {});
            },
            .extract => |x| try set.put(a, x.aggregate, {}),
            .convert => |x| try set.put(a, x.value, {}),
            .unary => |x| try set.put(a, x.value, {}),
            .load => |x| try set.put(a, x.ptr, {}),
            .store => |x| {
                try set.put(a, x.value, {});
                try set.put(a, x.ptr, {});
            },
            .prefetch => |x| try set.put(a, x.ptr, {}),
            .dot => |x| {
                try set.put(a, x.acc, {});
                try set.put(a, x.a, {});
                try set.put(a, x.b, {});
            },
            .matmul => |x| {
                try set.put(a, x.a, {});
                try set.put(a, x.b, {});
                try set.put(a, x.c, {});
            },
            .struct_new => |x| for (func.valueList(x.fields)) |v| try set.put(a, v, {}),
            .call => |x| for (func.valueList(x.args)) |v| try set.put(a, v, {}),
            .call_indirect => |x| {
                try set.put(a, x.target, {});
                for (func.valueList(x.args)) |v| try set.put(a, v, {});
            },
            .@"if" => |x| {
                try set.put(a, x.cond, {});
                for (func.valueList(x.then.args)) |v| try set.put(a, v, {});
                for (func.valueList(x.@"else".args)) |v| try set.put(a, v, {});
            },
        }
    }
    if (func.terminator(block)) |term| switch (term) {
        .ret => |v| if (v) |vv| try set.put(a, vv, {}),
        .jump => |j| for (func.valueList(j.args)) |v| try set.put(a, v, {}),
    };
}

/// Replace every use of `from` with `to` within a single block (instructions,
/// `if` edges, and the terminator). The block's definitions are untouched.
fn replaceInBlock(func: *Function, block: Block, from: Value, to: Value) void {
    const rep = struct {
        fn f(fr: Value, t: Value, v: Value) Value {
            return if (v == fr) t else v;
        }
    }.f;
    for (func.blockInsts(block)) |inst| {
        const op = func.opcodeMut(inst);
        switch (op.*) {
            .iconst, .fconst, .alloca, .global_addr => {},
            .arith => |*x| {
                x.lhs = rep(from, to, x.lhs);
                x.rhs = rep(from, to, x.rhs);
            },
            .arith_imm => |*x| x.lhs = rep(from, to, x.lhs),
            .icmp => |*x| {
                x.lhs = rep(from, to, x.lhs);
                x.rhs = rep(from, to, x.rhs);
            },
            .select => |*x| {
                x.cond = rep(from, to, x.cond);
                x.then = rep(from, to, x.then);
                x.@"else" = rep(from, to, x.@"else");
            },
            .extract => |*x| x.aggregate = rep(from, to, x.aggregate),
            .convert => |*x| x.value = rep(from, to, x.value),
            .unary => |*x| x.value = rep(from, to, x.value),
            .load => |*x| x.ptr = rep(from, to, x.ptr),
            .store => |*x| {
                x.value = rep(from, to, x.value);
                x.ptr = rep(from, to, x.ptr);
            },
            .prefetch => |*x| x.ptr = rep(from, to, x.ptr),
            .dot => |*x| {
                x.acc = rep(from, to, x.acc);
                x.a = rep(from, to, x.a);
                x.b = rep(from, to, x.b);
            },
            .matmul => |*x| {
                x.a = rep(from, to, x.a);
                x.b = rep(from, to, x.b);
                x.c = rep(from, to, x.c);
            },
            .struct_new => |x| for (func.valueListMut(x.fields)) |*f| {
                f.* = rep(from, to, f.*);
            },
            .call => |x| for (func.valueListMut(x.args)) |*arg| {
                arg.* = rep(from, to, arg.*);
            },
            .call_indirect => |*x| {
                x.target = rep(from, to, x.target);
                for (func.valueListMut(x.args)) |*arg| arg.* = rep(from, to, arg.*);
            },
            .@"if" => |*x| {
                x.cond = rep(from, to, x.cond);
                for (func.valueListMut(x.then.args)) |*arg| arg.* = rep(from, to, arg.*);
                for (func.valueListMut(x.@"else".args)) |*arg| arg.* = rep(from, to, arg.*);
            },
        }
    }
    if (func.terminatorPtr(block).*) |*t| switch (t.*) {
        .ret => |*v| if (v.*) |vv| {
            v.* = rep(from, to, vv);
        },
        .jump => |*j| for (func.valueListMut(j.args)) |*arg| {
            arg.* = rep(from, to, arg.*);
        },
    };
}

/// Build the canonical counted loop `for i in 0..n { } ; ret i` used by the
/// structural tests. Optionally makes the header impure by storing in it.
fn buildCountedLoop(func: *Function, impure_header: bool) Error!void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const slot = if (impure_header) try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } }) else undefined;
    try func.setJump(entry, loop, &.{zero});
    if (impure_header) try func.appendStore(loop, i, slot);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const next = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{next});
    func.setTerminator(done, .{ .ret = i });
}

test "run leaves an ineligible loop (impure header) unchanged" {
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildCountedLoop(&func, true);
    const before = func.blockCount();
    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before, func.blockCount());
}

test "run skips a single-issue in-order model (factor 1)" {
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildCountedLoop(&func, false);
    const before = func.blockCount();
    const changed = try run(allocator, &func, registry.modelFor(.@"et-soc"));
    try std.testing.expect(!changed);
    try std.testing.expectEqual(before, func.blockCount());
}

test "run unrolls an eligible counted loop under a wide model, staying verifiable" {
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildCountedLoop(&func, false);
    const before = func.blockCount();
    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(changed);
    try std.testing.expect(func.blockCount() > before);
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "cloneBlocks duplicates a region with remapped values and independent blocks" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const p = try func.appendBlockParam(b0, i32_t);
    const d = try func.appendArithImm(b0, i32_t, .mul, p, 2);
    func.setTerminator(b0, .{ .ret = d });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    var bmap: BlockMap = .empty;
    defer bmap.deinit(allocator);
    const clones = try cloneBlocks(allocator, &func, &.{b0}, &vmap, &bmap);
    defer allocator.free(clones);

    try std.testing.expectEqual(@as(usize, 1), clones.len);
    try std.testing.expect(clones[0] != b0); // a fresh block
    try std.testing.expectEqual(@as(usize, 1), func.blockParams(clones[0]).len);
    // The clone verifies as part of the whole function.
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}
