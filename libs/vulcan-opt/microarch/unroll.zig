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

/// The clone of `v` under `map`, or `v` itself when it is defined outside the region.
fn remapValue(map: *const ValueMap, v: Value) Value {
    return map.get(v) orelse v;
}

/// The clone of `b` under `map`, or `b` itself when it is outside the region.
fn remapBlock(map: *const BlockMap, b: Block) Block {
    return map.get(b) orelse b;
}

/// Rebuild `op` with every Value operand mapped through `value_map` and every Block target
/// mapped through `block_map`. `args_buf` is scratch for the variadic operands, so one buffer
/// serves a whole clone. The switch is exhaustive over `Opcode` with NO `else`: a new opcode
/// stops the build here instead of reaching a run-time `unreachable`.
fn rebuildOpcode(
    func: *Function,
    op: Opcode,
    value_map: *const ValueMap,
    block_map: *const BlockMap,
    args_buf: *std.ArrayList(Value),
    allocator: std.mem.Allocator,
) Error!Opcode {
    return switch (op) {
        .iconst => |v| .{ .iconst = v },
        .fconst => |v| .{ .fconst = v },
        .fconst128 => |v| .{ .fconst128 = v },
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
        // Only `args` (and `target`/`ret_dest`) are Values. `is_variadic`/`num_fixed` and
        // the `ret_dest`/`ret_regs`/`ret_pieces`/`sret` struct-return description are
        // call-site metadata that each unrolled copy must keep, so start from the source
        // op and overwrite only the Values.
        .call => |c| blk: {
            args_buf.clearRetainingCapacity();
            for (func.valueList(c.args)) |v| {
                try args_buf.append(allocator, remapValue(value_map, v));
            }
            var out = c;
            out.args = try func.internValues(args_buf.items);
            if (c.ret_dest) |rd| out.ret_dest = remapValue(value_map, rd);
            break :blk .{ .call = out };
        },
        .call_indirect => |c| blk: {
            args_buf.clearRetainingCapacity();
            for (func.valueList(c.args)) |v| {
                try args_buf.append(allocator, remapValue(value_map, v));
            }
            var out = c;
            out.target = remapValue(value_map, c.target);
            out.args = try func.internValues(args_buf.items);
            if (c.ret_dest) |rd| out.ret_dest = remapValue(value_map, rd);
            break :blk .{ .call_indirect = out };
        },
        .global_addr => |g| .{ .global_addr = .{ .symbol = g.symbol, .via_got = g.via_got } },
        .load => |l| .{ .load = .{ .ptr = remapValue(value_map, l.ptr), .@"volatile" = l.@"volatile" } },
        .store => |st| .{ .store = .{
            .value = remapValue(value_map, st.value),
            .ptr = remapValue(value_map, st.ptr),
            .@"volatile" = st.@"volatile",
        } },
        .prefetch => |pf| .{ .prefetch = .{
            .ptr = remapValue(value_map, pf.ptr),
        } },
        .va_start => |vs| .{ .va_start = .{ .list = remapValue(value_map, vs.list) } },
        .va_arg => |va| .{ .va_arg = .{ .list = remapValue(value_map, va.list), .ty = va.ty } },
        .va_end => |ve| .{ .va_end = .{ .list = remapValue(value_map, ve.list) } },
        // A barrier has no Value to remap. Each unrolled copy of the body keeps its
        // own barrier, so the number of times a thread meets is unchanged.
        .barrier => |bar| .{ .barrier = bar },
        .dot => |d| .{ .dot = .{
            .acc = remapValue(value_map, d.acc),
            .a = remapValue(value_map, d.a),
            .b = remapValue(value_map, d.b),
        } },
        // Only a/b/c are Values. Everything else (m/n/k, dtype, accumulate, embedded,
        // quant, input_signs) is compile-time metadata, so start from the source op and
        // overwrite only the Values. A named-field rebuild silently defaulted `embedded`
        // back to false, which told the backend a matmul inside a live-value region was a
        // standalone kernel free to clobber registers.
        // The `quant` handle copies verbatim, including a `per_column` scale's ScaleList:
        // unrolling clones within the SAME function, so the handle stays relative to the
        // same `scale_pool` and needs no re-interning (unlike inline.zig's cross-function
        // clone).
        .matmul => |mmv| blk: {
            var out = mmv;
            out.a = remapValue(value_map, mmv.a);
            out.b = remapValue(value_map, mmv.b);
            out.c = remapValue(value_map, mmv.c);
            break :blk .{ .matmul = out };
        },
        // Each unrolled copy of the body keeps its own atomic, so the number of
        // read-modify-writes is unchanged. The caller appends the copy with or without a
        // result by asking `instResult`, so both forms clone correctly.
        .atomic_rmw => |av| blk: {
            var out = av;
            out.ptr = remapValue(value_map, av.ptr);
            out.value = remapValue(value_map, av.value);
            if (av.compare) |c| out.compare = remapValue(value_map, c);
            break :blk .{ .atomic_rmw = out };
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
}

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
    // The `original -> clone` record the attribute copy needs. It holds ONLY what this call
    // creates, never a caller's pre-seeded substitution, so an attribute cannot land on a
    // value the clone does not own.
    var value_pairs: std.ArrayList(ir.function.Function.ValuePair) = .empty;
    defer value_pairs.deinit(allocator);
    var inst_pairs: std.ArrayList(ir.function.Function.InstPair) = .empty;
    defer inst_pairs.deinit(allocator);

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
            try value_pairs.append(allocator, .{ .old = p, .new = cloned_p });
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
            const rebuilt = try rebuildOpcode(func, func.opcode(inst), value_map, block_map, &args_buf, allocator);

            // Ask the INSTRUCTION whether it defines a result, rather than naming the
            // result-less opcodes in a list. A hand-kept list has to be updated for every new
            // opcode, and it also missed a VOID `call` and a void `call_indirect`, which the
            // `else` then sent into an `orelse unreachable`. `instResult` is right for every
            // opcode, now and later.
            if (func.instResult(inst)) |result| {
                const cloned_result = try func.appendInst(cloned, func.valueType(result), rebuilt);
                try value_map.put(allocator, result, cloned_result);
                try value_pairs.append(allocator, .{ .old = result, .new = cloned_result });
                try inst_pairs.append(allocator, .{ .old = inst, .new = func.definingInst(cloned_result).? });
            } else {
                const cloned_inst = try func.appendStmtRaw(cloned, rebuilt);
                try inst_pairs.append(allocator, .{ .old = inst, .new = cloned_inst });
            }
        }

        if (func.terminator(b)) |term| {
            const rebuilt_term: Terminator = switch (term) {
                .ret => |r| blk: {
                    var nr = r;
                    for (nr.values[0..nr.count]) |*vv| vv.* = remapValue(value_map, vv.*);
                    break :blk .{ .ret = nr };
                },
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

    // Carry every attribute of a cloned parameter, instruction or result onto its copy. An
    // unrolled body copy runs the SAME operations as the original, so it must keep the same
    // annotations: an `endian` a copy loses is a byte order the backend stops applying.
    try func.cloneAttrs(value_pairs.items, inst_pairs.items);

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
            .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .select, .convert, .unary, .extract, .struct_new, .dot => {},
            // load/store/call/call_indirect/alloca/global_addr are impure or memory ops.
            else => return null,
        }
    }
    const iff = if_inst orelse return null;
    // The header's control is the `if`; any explicit terminator other than an
    // implicit/void return means a shape we do not model.
    if (func.terminator(header)) |term| switch (term) {
        .ret => |r| if (r.count != 0) return null,
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

/// Clone one instruction of the loop test header into `dest`, remapping operands through `vmap`
/// and recording result -> clone.
///
/// It shares `rebuildOpcode` with `cloneBlocks` instead of listing the pure opcodes again. The
/// old private list ended in `else => unreachable`, so a header opcode eligibility accepted but
/// this list had missed reached undefined behaviour in a release build. `fconst128` was already
/// such an opcode. `rebuildOpcode` is exhaustive over `Opcode` with no `else`, so a new opcode
/// stops the BUILD instead.
///
/// The instruction's attributes travel with the copy: the guard re-runs the same test, so it
/// must keep the same annotations.
fn cloneInstInto(func: *Function, dest: Block, inst: Inst, vmap: *ValueMap, a: std.mem.Allocator) Error!void {
    // Scratch for the variadic operands, released whatever happens. The old `struct_new` arm
    // built this list and never released it, so unrolling a header holding a `struct_new`
    // leaked one allocation per guard.
    var args_buf: std.ArrayList(Value) = .empty;
    defer args_buf.deinit(a);

    // The header holds no branch, so no block target can need remapping. An empty map leaves
    // every target as it is.
    var block_map: BlockMap = .empty;
    defer block_map.deinit(a);

    const rebuilt = try rebuildOpcode(func, func.opcode(inst), vmap, &block_map, &args_buf, a);

    // Ask the INSTRUCTION whether it defines a result, rather than assuming one. The old
    // `func.instResult(inst).?` was an unchecked assumption about the header's shape.
    if (func.instResult(inst)) |result| {
        const cloned = try func.appendInst(dest, func.valueType(result), rebuilt);
        try vmap.put(a, result, cloned);
        try func.cloneAttrs(
            &.{.{ .old = result, .new = cloned }},
            &.{.{ .old = inst, .new = func.definingInst(cloned).? }},
        );
    } else {
        const cloned = try func.appendStmtRaw(dest, rebuilt);
        try func.cloneAttrs(&.{}, &.{.{ .old = inst, .new = cloned }});
    }
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
            .atomic_rmw => |x| {
                try set.put(a, x.ptr, {});
                try set.put(a, x.value, {});
                if (x.compare) |c| try set.put(a, c, {});
            },
            // A barrier reads no Value operand.
            .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
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
            .va_start => |x| try set.put(a, x.list, {}),
            .va_arg => |x| try set.put(a, x.list, {}),
            .va_end => |x| try set.put(a, x.list, {}),
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
        .ret => |r| for (r.slice()) |vv| try set.put(a, vv, {}),
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
            .atomic_rmw => |*x| {
                x.ptr = rep(from, to, x.ptr);
                x.value = rep(from, to, x.value);
                if (x.compare) |*c| c.* = rep(from, to, c.*);
            },
            // A barrier reads no Value operand.
            .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
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
            .va_start => |*x| x.list = rep(from, to, x.list),
            .va_arg => |*x| x.list = rep(from, to, x.list),
            .va_end => |*x| x.list = rep(from, to, x.list),
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
        .ret => |*r| for (r.values[0..r.count]) |*vv| {
            vv.* = rep(from, to, vv.*);
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
    const ptr_t = try func.types.ptrGlobal();
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
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(i) });
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
    func.setTerminator(b0, .{ .ret = ir.function.Ret.one(d) });

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

test "cloneBlocks keeps via_got=true on a cloned global_addr (loop-unroll body copy)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const b0 = try func.appendBlock();
    const g = try func.appendGlobalAddrGot(b0, ptr_t, "G");
    func.setTerminator(b0, .{ .ret = ir.function.Ret.one(g) });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    var bmap: BlockMap = .empty;
    defer bmap.deinit(allocator);
    const clones = try cloneBlocks(allocator, &func, &.{b0}, &vmap, &bmap);
    defer allocator.free(clones);

    // The unrolled body's global_addr copy must still be via_got=true: the
    // per-instruction remap must forward the flag, not rebuild the op from just
    // `.symbol` (which would silently default it false).
    var found = false;
    for (func.blockInsts(clones[0])) |inst| {
        if (func.opcode(inst) == .global_addr) {
            found = true;
            try std.testing.expect(func.opcode(inst).global_addr.via_got);
        }
    }
    try std.testing.expect(found);
}

test "cloneBlocks keeps a call's variadic and struct-return metadata (loop-unroll body copy)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b0 = try func.appendBlock();
    const n = try func.appendInst(b0, i32_t, .{ .iconst = 7 });
    // printf("%d", n): one declared parameter, then one variadic argument.
    const fmt = try func.appendGlobalAddr(b0, ptr_t, "fmt");
    _ = try func.appendCallV(b0, i32_t, "printf", &.{ fmt, n }, 1);
    // A struct-by-value return into a caller slot.
    const slot = try func.appendInst(b0, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendCallStructRet(b0, "mkpair", &.{}, slot, &.{ .{}, .{ .fp = true, .offset = 8, .bytes = 4 } });
    func.setTerminator(b0, .{ .ret = ir.function.Ret.none() });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    var bmap: BlockMap = .empty;
    defer bmap.deinit(allocator);
    const clones = try cloneBlocks(allocator, &func, &.{b0}, &vmap, &bmap);
    defer allocator.free(clones);

    // Every field but `args`/`ret_dest` describes the CALL SITE, not the operands. A rebuild
    // from just `.symbol` and `.args` defaults them, which makes a variadic call look
    // fixed-arity and loses the struct return's destination slot entirely.
    var variadic_calls: usize = 0;
    var sret_calls: usize = 0;
    for (func.blockInsts(clones[0])) |inst| {
        if (func.opcode(inst) != .call) continue;
        const c = func.opcode(inst).call;
        if (std.mem.eql(u8, func.symbolName(c.symbol), "printf")) {
            variadic_calls += 1;
            try std.testing.expect(c.is_variadic);
            try std.testing.expectEqual(@as(u32, 1), c.num_fixed);
        }
        if (std.mem.eql(u8, func.symbolName(c.symbol), "mkpair")) {
            sret_calls += 1;
            try std.testing.expectEqual(@as(u8, 2), c.ret_regs);
            try std.testing.expect(c.ret_dest != null);
            try std.testing.expectEqual(vmap.get(slot).?, c.ret_dest.?); // remapped to the clone's slot
            try std.testing.expect(c.ret_pieces[1].fp);
            try std.testing.expectEqual(@as(u8, 8), c.ret_pieces[1].offset);
            try std.testing.expectEqual(@as(u8, 4), c.ret_pieces[1].bytes);
        }
    }
    try std.testing.expectEqual(@as(usize, 1), variadic_calls);
    try std.testing.expectEqual(@as(usize, 1), sret_calls);
}

test "cloneBlocks keeps embedded=true on a cloned matmul (loop-unroll body copy)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const b0 = try func.appendBlock();
    const a = try func.appendGlobalAddr(b0, ptr_t, "A");
    const b = try func.appendGlobalAddr(b0, ptr_t, "B");
    const c = try func.appendGlobalAddr(b0, ptr_t, "C");
    try func.appendMatmulEmbedded(b0, a, b, c, 4, 4, 4, .fp32, true, null);
    func.setTerminator(b0, .{ .ret = ir.function.Ret.none() });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    var bmap: BlockMap = .empty;
    defer bmap.deinit(allocator);
    const clones = try cloneBlocks(allocator, &func, &.{b0}, &vmap, &bmap);
    defer allocator.free(clones);

    // `embedded` tells the backend the matmul sits inside a live-value region and must save
    // every register it clobbers. A named-field rebuild defaults it to false, which turns the
    // copy into a standalone kernel that is free to clobber the surrounding values.
    var found: usize = 0;
    for (func.blockInsts(clones[0])) |inst| {
        if (func.opcode(inst) != .matmul) continue;
        found += 1;
        const m = func.opcode(inst).matmul;
        try std.testing.expect(m.embedded);
        try std.testing.expect(m.accumulate);
        try std.testing.expectEqual(ir.function.MatMulType.fp32, m.dtype);
    }
    try std.testing.expectEqual(@as(usize, 1), found);
}

test "cloneBlocks keeps volatile on a cloned load and store (loop-unroll body copy)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b0 = try func.appendBlock();
    const reg = try func.appendGlobalAddr(b0, ptr_t, "MMIO");
    const v = try func.appendInst(b0, i32_t, .{ .load = .{ .ptr = reg, .@"volatile" = true } });
    try func.appendStoreVol(b0, v, reg, true);
    // A plain access in the same block proves the flag is carried, not forced on.
    const p = try func.appendGlobalAddr(b0, ptr_t, "RAM");
    const w = try func.appendInst(b0, i32_t, .{ .load = .{ .ptr = p } });
    try func.appendStore(b0, w, p);
    func.setTerminator(b0, .{ .ret = ir.function.Ret.none() });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    var bmap: BlockMap = .empty;
    defer bmap.deinit(allocator);
    const clones = try cloneBlocks(allocator, &func, &.{b0}, &vmap, &bmap);
    defer allocator.free(clones);

    // Unrolling a loop that reads or writes a hardware register must keep each copy volatile.
    // An ordinary copy is one a later pass may move, duplicate or delete.
    var vol_loads: usize = 0;
    var vol_stores: usize = 0;
    var plain_loads: usize = 0;
    var plain_stores: usize = 0;
    for (func.blockInsts(clones[0])) |inst| switch (func.opcode(inst)) {
        .load => |l| if (l.@"volatile") {
            vol_loads += 1;
        } else {
            plain_loads += 1;
        },
        .store => |s| if (s.@"volatile") {
            vol_stores += 1;
        } else {
            plain_stores += 1;
        },
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 1), vol_loads);
    try std.testing.expectEqual(@as(usize, 1), vol_stores);
    try std.testing.expectEqual(@as(usize, 1), plain_loads);
    try std.testing.expectEqual(@as(usize, 1), plain_stores);
}

/// The int payload of the first `namespace.key` attribute on `target`, or null when absent.
fn testCustomInt(func: *const Function, target: ir.function.AttrTarget, namespace: []const u8, key: []const u8) ?i64 {
    var it = func.attributesOf(target);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (!std.mem.eql(u8, c.namespace, namespace)) continue;
            if (!std.mem.eql(u8, c.key, key)) continue;
            return switch (c.value) {
                .int => |n| n,
                .flag, .string => null,
            };
        },
        .@"inline", .noreturn, .cold, .@"align", .endian => {},
    };
    return null;
}

/// How many attributes sit on `target`.
fn testAttrCount(func: *const Function, target: ir.function.AttrTarget) usize {
    var n: usize = 0;
    var it = func.attributesOf(target);
    while (it.next()) |_| n += 1;
    return n;
}

test "cloneBlocks carries a param, result and instruction attribute onto the body copy" {
    // An unrolled body copy runs the same operations as the original, so it must keep the same
    // annotations. A copied load that loses `endian` is one the backend stops byte-swapping.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b0 = try func.appendBlock();
    const p = try func.appendBlockParam(b0, ptr_t);
    const v = try func.appendInst(b0, i32_t, .{ .load = .{ .ptr = p } });
    try func.appendStore(b0, v, p);
    const store_inst = func.blockInsts(b0)[1];
    func.setTerminator(b0, .{ .ret = ir.function.Ret.none() });

    try func.addAttr(.{ .value = p }, .{ .@"align" = 16 });
    try func.addAttr(.{ .value = v }, .{ .endian = .big });
    try func.addAttr(.{ .inst = store_inst }, .{ .custom = .{
        .namespace = "debug",
        .key = "line",
        .value = .{ .int = 9 },
    } });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    var bmap: BlockMap = .empty;
    defer bmap.deinit(allocator);
    const clones = try cloneBlocks(allocator, &func, &.{b0}, &vmap, &bmap);
    defer allocator.free(clones);

    const new_p = func.blockParams(clones[0])[0];
    var align_it = func.attributesOf(.{ .value = new_p });
    try std.testing.expectEqual(ir.function.Attribute{ .@"align" = 16 }, align_it.next().?);

    const new_v = vmap.get(v).?;
    var endian_it = func.attributesOf(.{ .value = new_v });
    try std.testing.expectEqual(ir.function.Attribute{ .endian = .big }, endian_it.next().?);

    const new_store = func.blockInsts(clones[0])[1];
    try std.testing.expectEqual(@as(?i64, 9), testCustomInt(&func, .{ .inst = new_store }, "debug", "line"));
}

test "cloneBlocks does not carry a block attribute onto the copied block" {
    // Suspicious case, the other direction. A `cf` attribute names the merge block of the
    // ORIGINAL region; a body copy claiming the same merge point would misdescribe the CFG.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const exit = try func.appendBlock();
    const b0 = try func.appendBlock();
    const p = try func.appendBlockParam(b0, i32_t);
    _ = try func.appendInst(b0, i32_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = p } });
    func.setTerminator(b0, .{ .ret = ir.function.Ret.none() });
    try func.addAttr(.{ .block = b0 }, .{ .custom = .{
        .namespace = "cf",
        .key = "merge",
        .value = .{ .int = @intFromEnum(exit) },
    } });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    var bmap: BlockMap = .empty;
    defer bmap.deinit(allocator);
    const clones = try cloneBlocks(allocator, &func, &.{b0}, &vmap, &bmap);
    defer allocator.free(clones);

    try std.testing.expectEqual(@as(usize, 0), testAttrCount(&func, .{ .block = clones[0] }));
    try std.testing.expectEqual(@as(usize, 1), testAttrCount(&func, .{ .block = b0 }));
}

test "cloneInstInto rebuilds an f128 constant instead of reaching unreachable" {
    // Regression: `eligible` accepts `fconst128` in a loop test header, but the private opcode
    // list this function used had no `fconst128` arm and ended in `else => unreachable`. A
    // header holding an f128 constant reached undefined behaviour in a release build.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f128_t = try func.types.intern(.{ .float = .f128 });
    const b0 = try func.appendBlock();
    const dest = try func.appendBlock();
    const k = try func.appendInst(b0, f128_t, .{ .fconst128 = 0x3fff_8000_0000_0000_0000_0000_0000_0000 });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    try cloneInstInto(&func, dest, func.definingInst(k).?, &vmap, allocator);

    const cloned = func.blockInsts(dest)[0];
    try std.testing.expectEqual(@as(u128, 0x3fff_8000_0000_0000_0000_0000_0000_0000), func.opcode(cloned).fconst128);
}

test "cloneInstInto carries the header instruction's attribute onto the guard copy" {
    // Each guard re-runs the header test, so its copy must keep the header's annotations.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const dest = try func.appendBlock();
    const p = try func.appendBlockParam(b0, i32_t);
    const sum = try func.appendInst(b0, i32_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = p } });
    const sum_inst = func.definingInst(sum).?;
    try func.addAttr(.{ .value = sum }, .{ .@"align" = 8 });
    try func.addAttr(.{ .inst = sum_inst }, .{ .custom = .{
        .namespace = "debug",
        .key = "line",
        .value = .{ .int = 5 },
    } });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    try cloneInstInto(&func, dest, sum_inst, &vmap, allocator);

    const cloned = func.blockInsts(dest)[0];
    const cloned_result = func.instResult(cloned).?;
    var align_it = func.attributesOf(.{ .value = cloned_result });
    try std.testing.expectEqual(ir.function.Attribute{ .@"align" = 8 }, align_it.next().?);
    try std.testing.expectEqual(@as(?i64, 5), testCustomInt(&func, .{ .inst = cloned }, "debug", "line"));
}

test "cloneInstInto releases the scratch list it builds for a struct_new header op" {
    // Regression: the `struct_new` arm built an ArrayList of remapped fields and never released
    // it, so every guard of an unrolled loop leaked one allocation. `testing.allocator` fails
    // this test if the list is not released.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const vec_t = try func.types.intern(.{ .vector = .{ .len = 2, .elem = i32_t } });
    const b0 = try func.appendBlock();
    const dest = try func.appendBlock();
    const p = try func.appendBlockParam(b0, i32_t);
    const packed_val = try func.appendInst(b0, vec_t, .{
        .struct_new = .{ .fields = try func.internValues(&.{ p, p }) },
    });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    try cloneInstInto(&func, dest, func.definingInst(packed_val).?, &vmap, allocator);

    const cloned = func.blockInsts(dest)[0];
    try std.testing.expectEqualSlices(Value, &.{ p, p }, func.valueList(func.opcode(cloned).struct_new.fields));
}

test "cloneBlocks copies BOTH atomic forms, keeping each copy's result state" {
    // Unrolling duplicates the body once per unrolled iteration, so the number of
    // read-modify-writes is unchanged and an atomic may be cloned. The two forms are the
    // point: the reduction one must come back result-less, and the reading one must come
    // back with a fresh result. A hand-kept result-less opcode list would get one of them
    // wrong, which is why `cloneBlocks` asks `instResult` instead.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b0 = try func.appendBlock();
    const p = try func.appendBlockParam(b0, ptr_t);
    const v = try func.appendBlockParam(b0, i32_t);
    const c = try func.appendBlockParam(b0, i32_t);
    try func.appendAtomicRmwStmt(b0, .{ .op = .add, .ptr = p, .value = v, .ordering = .relaxed, .scope = .device });
    _ = try func.appendAtomicRmw(b0, .{
        .op = .compare_exchange,
        .ptr = p,
        .value = v,
        .compare = c,
        .ordering = .seq_cst,
        .scope = .system,
    });
    func.setTerminator(b0, .{ .ret = ir.function.Ret.none() });

    var vmap: ValueMap = .empty;
    defer vmap.deinit(allocator);
    var bmap: BlockMap = .empty;
    defer bmap.deinit(allocator);
    const clones = try cloneBlocks(allocator, &func, &.{b0}, &vmap, &bmap);
    defer allocator.free(clones);

    const copied = func.blockInsts(clones[0]);
    try std.testing.expectEqual(@as(usize, 2), copied.len);

    const reduction = func.opcode(copied[0]).atomic_rmw;
    try std.testing.expectEqual(ir.function.AtomicOp.add, reduction.op);
    try std.testing.expectEqual(ir.function.AtomicOrdering.relaxed, reduction.ordering);
    try std.testing.expectEqual(ir.function.AtomicScope.device, reduction.scope);
    try std.testing.expectEqual(@as(?Value, null), func.instResult(copied[0]));

    const cas = func.opcode(copied[1]).atomic_rmw;
    try std.testing.expectEqual(ir.function.AtomicOp.compare_exchange, cas.op);
    try std.testing.expectEqual(ir.function.AtomicScope.system, cas.scope);
    // The operands were remapped onto the copied block's own parameters, and the compare
    // operand survived the copy.
    const new_params = func.blockParams(clones[0]);
    try std.testing.expectEqual(new_params[0], cas.ptr);
    try std.testing.expectEqual(new_params[1], cas.value);
    try std.testing.expectEqual(new_params[2], cas.compare.?);
    try std.testing.expect(func.instResult(copied[1]) != null);
}
