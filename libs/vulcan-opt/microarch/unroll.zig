//! Model-driven loop unrolling. Unrolls hot innermost loops by a factor the target model decides,
//! to expose the independent work the hardware needs to hide latency. Conservative: only
//! reducible, innermost, single-latch loops with a pure test header and
//! header-param loop-carried values are transformed, everything else is skipped unchanged. The
//! transform emits one of two shapes, both correct by construction. A CPU model, or a SIMT loop
//! that fails the strict counted checks, gets a guarded partial unroll: K body copies, each closed
//! by its own recomputed guard. A SIMT loop that passes those checks gets a main loop plus a
//! remainder: one guard and K unguarded copies in one block. Load-bearing factor-five loops use
//! four guarded forward tail copies; other factors leave the original loop whole as the tail.
//! The guarded shape is proven by the differential JIT tests in
//! libs/vulcan-target/tests/unroll_differential.zig.
//!
//! Why the second shape exists: the SIMT scheduler (microarch/schedule.zig) hoists loads only
//! inside one block. A guard per copy splits the copies into separate blocks, so a load-bound
//! loop waits on each copy's loads in turn. One guard per K copies keeps the loads in one block,
//! where the scheduler hoists them as a group. That is the shape ptxas gives a counted loop.
//!
//! Two targets, two bounds. A CPU core is bounded by issue width: enough copies to fill the ports
//! across a dependency chain. A streaming multiprocessor has no issue width, and is bounded by the
//! REGISTER BUDGET, because registers per thread decide how many warps stay resident and resident
//! warps are what hides latency. `unrollFactor` therefore splits on `Model.exec`.
//!
//! What the SIMT rule cannot see, stated plainly: this pass runs on SSA IR, before register
//! allocation, so the live-value count of a loop body is not available to it. The register bound is
//! computed from an UPPER BOUND on that count (a body of N instructions defines at most N values),
//! which makes the rule conservative rather than accurate. A body whose values die quickly is
//! under-unrolled. Closing that gap needs a live-range estimate the IR does not carry today, and a
//! guess dressed up as one would be worse than the bound.

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

/// The share of peak occupancy a SIMT unroll refuses to fall below, written as a divisor of
/// `Simt.warps_per_sm`: 2 means "never below half the resident warps the SM can hold".
///
/// This is a POLICY number and not a hardware one, so it lives here and not in the model. Its
/// justification: NVIDIA's own CUDA C++ Best Practices Guide ("Occupancy") states that raising
/// occupancy past roughly half of peak usually stops improving performance, because by then the SM
/// already has enough warps to switch to. Read the other way round, which is the way this rule
/// needs, half of peak is where the SM stops having a spare warp for every one that is waiting, and
/// so it is where trading occupancy for in-flight work stops paying. sm_120 has four warp
/// schedulers per SM, so half of its 48 resident warps leaves six per scheduler.
///
/// It is a rule of thumb, not a measurement. It is the one number in this file that a measured
/// sweep on real silicon should replace first.
const OCCUPANCY_FLOOR_DIVISOR: u32 = 2;

/// The fewest resident warps a SIMT unroll may leave. Never below 1, so the rule still answers for
/// a model with a tiny warp ceiling.
fn occupancyFloor(s: mm.Simt) u32 {
    return @max(1, @as(u32, s.warps_per_sm) / OCCUPANCY_FLOOR_DIVISOR);
}

/// The registers one thread holds after unrolling a `body_ops`-instruction body `k` times.
///
/// THIS IS AN UPPER BOUND AND NOT AN ESTIMATE, and the reason matters. Unrolling runs on SSA IR,
/// before register allocation, so the true live-value count of a body is not available here: it is
/// decided later by the allocator, over an interval graph this pass cannot see. What IS available
/// is the instruction count, and a body of `body_ops` instructions defines at most `body_ops`
/// values, so `k * body_ops` can never understate the demand. Using the bound makes the rule
/// conservative in the safe direction: it under-unrolls a body whose values die quickly, and it
/// never over-unrolls one into a spill. See the note in the module doc comment above.
///
/// Saturating, so a pathological body size cannot wrap into a small number and unlock an unroll.
fn simtLiveRegs(body_ops: u32, k: u32) u32 {
    return body_ops *| k;
}

/// Whether `warps` resident warps, each holding `k` iterations worth of independent work, already
/// put enough operations in flight to cover a memory access of `latency` cycles.
///
/// Little's law: an SM whose load/store unit accepts one request per cycle needs `latency` requests
/// outstanding to keep from idling across that latency, and the requests come from every resident
/// warp at once. Past that point more unrolling only adds registers.
///
/// On sm_120 this clause cannot fire: the SM's own ceiling of 48 resident warps times the cap of 8
/// copies is 384 operations, and a device-memory access costs about 500 cycles. That is the whole
/// reason the register budget, not the latency, is what bounds a GPU unroll here, and it is checked
/// by the comptime block below rather than left as a claim.
fn coversLatency(warps: u32, k: u32, latency: u32) bool {
    return warps *| k >= latency;
}

comptime {
    // Recompute the regime claim in `coversLatency`'s doc comment: on sm_120 the most work this
    // pass can ever put in flight falls short of a device-memory round trip, so that clause is
    // inert and the register budget is what decides. A part where this stops holding gets a
    // different rule, and the build says so here instead of silently changing factors.
    const s = @import("registry.zig").modelFor(.sm_120).simt.?;
    if (coversLatency(s.warps_per_sm, MAX_FACTOR, s.global_latency))
        @compileError("sm_120 can now cover a global access from warp parallelism alone: the unroll rule's register-bound assumption needs re-deriving");
}

/// How many times to unroll a `body_ops`-instruction loop body on a SIMT target.
///
/// A GPU hides latency with warps in flight and with independent work inside each warp. Unrolling
/// buys the second, and pays for it in the first: every extra copy raises the registers a thread
/// holds, and registers are the fixed budget that decides how many warps stay resident. So the rule
/// takes the largest factor that still leaves the SM enough resident warps, and it stops for one of
/// three reasons, each of which the tests below make fire:
///
///   1. The thread would need more registers than the hardware gives one (`max_regs_per_thread`),
///      so the kernel would spill to local memory. A spill is a memory access added to hide a
///      memory access.
///   2. Occupancy would fall below `occupancyFloor`. This is the real bound on this hardware.
///   3. The work in flight already covers the memory latency, so another copy buys nothing. Inert
///      on sm_120, see `coversLatency`.
///
/// Note what is NOT here: issue width, port pressure and a reorder window, which is what the CPU
/// rule is made of. An SM has none of the three.
fn simtUnrollFactor(s: mm.Simt, body_ops: u32) u32 {
    if (body_ops == 0) return 1;
    const floor = occupancyFloor(s);
    var best: u32 = 1;
    var k: u32 = 2;
    while (k <= MAX_FACTOR) : (k += 1) {
        const regs = simtLiveRegs(body_ops, k);
        // Check the raw demand before the rounded one, so an absurd body size cannot reach the
        // rounding arithmetic at all.
        if (regs > s.max_regs_per_thread) break;
        if (s.allocFor(regs) > s.max_regs_per_thread) break; // rounding up can cross the cap on its own
        const warps = s.residentWarps(regs);
        if (warps < floor) break;
        best = k;
        if (coversLatency(warps, k, s.global_latency)) break;
    }
    return best;
}

/// How many times to unroll a loop whose body has `body_ops` instructions on a CPU model. Returns 1
/// (no unroll) for a single-issue in-order model, since it has no width to fill. For a wider model,
/// enough copies to keep issue_width ports busy across the dominant latency, capped at MAX_FACTOR
/// and never more than makes sense for the body size.
fn cpuUnrollFactor(model: *const mm.Model, body_ops: u32) u32 {
    if (model.issue_width <= 1) return 1; // in-order single-issue gains nothing
    if (body_ops == 0) return 1;
    // Rough ILP target: cover the issue width across a typical multi-cycle latency (use 3 as a
    // representative arithmetic latency), divided by the work already in one body.
    const target = (@as(u32, model.issue_width) * 3) / body_ops;
    return std.math.clamp(target, 1, MAX_FACTOR);
}

/// How many times to unroll a loop whose body has `body_ops` instructions for `model`. A CPU core
/// and a streaming multiprocessor are bounded by different things, so they get different rules: see
/// `cpuUnrollFactor` (issue width and port pressure) and `simtUnrollFactor` (the register budget,
/// through occupancy). The split is on `exec` and not on a width, because a SIMT model's
/// `issue_width` is zero and would otherwise land in the CPU rule's single-issue arm.
pub fn unrollFactor(model: *const mm.Model, body_ops: u32) u32 {
    return switch (model.exec) {
        // `validate` ties `.simt` to a non-null `simt` block, so the orelse is unreachable for a
        // validated model. It answers 1 rather than trapping for a hand-built one.
        .simt => if (model.simt) |s| simtUnrollFactor(s, body_ops) else 1,
        .in_order, .out_of_order => cpuUnrollFactor(model, body_ops),
    };
}

test "unrollFactor is 1 for single-issue in-order models" {
    const registry = @import("registry.zig");
    try std.testing.expectEqual(@as(u32, 1), unrollFactor(registry.modelFor(.@"et-soc"), 2));
    try std.testing.expectEqual(@as(u32, 1), unrollFactor(registry.modelFor(.@"river-rc1.s"), 2));
}

/// The CPU unroll rule as it stood before a SIMT branch existed, written out again so the test
/// below compares against an INDEPENDENT statement of it rather than against the function it is
/// meant to be guarding.
fn preSimtCpuFactor(issue_width: u8, body_ops: u32) u32 {
    if (issue_width <= 1) return 1;
    if (body_ops == 0) return 1;
    return std.math.clamp((@as(u32, issue_width) * 3) / body_ops, 1, MAX_FACTOR);
}

test "every CPU model's factor is unchanged by the SIMT branch, over the whole body-size range" {
    // The guard on the whole change: adding a SIMT rule must move no CPU factor at all. Checked for
    // EVERY predefined CPU part against a separate statement of the old formula, over every body
    // size that can produce a factor above 1 plus a long tail that cannot.
    const registry = @import("registry.zig");
    inline for (std.meta.tags(mm.Microarch)) |t| {
        const m = registry.modelFor(t);
        // An `if` block and not a `continue`: the loop is an inline one, so a `continue` under a
        // run-time condition is comptime control flow in a run-time block.
        if (m.exec != .simt) {
            var body: u32 = 0;
            while (body <= 64) : (body += 1) {
                try std.testing.expectEqual(preSimtCpuFactor(m.issue_width, body), unrollFactor(m, body));
            }
            try std.testing.expectEqual(preSimtCpuFactor(m.issue_width, 1000), unrollFactor(m, 1000));
            try std.testing.expectEqual(preSimtCpuFactor(m.issue_width, std.math.maxInt(u32)), unrollFactor(m, std.math.maxInt(u32)));
        }
    }
}

test "sm_120 unrolls by the factor the register budget allows, for several body sizes" {
    // Every expectation below is the largest K whose thread allocation still leaves at least half
    // of sm_120's 48 resident warps. On this part that means an allocation of at most 80 registers
    // (81 rounds to 88, which holds only 23 warps), so the factor is about 80 / body_ops, in the
    // steps the 8-register granularity creates.
    const registry = @import("registry.zig");
    const g = registry.modelFor(.sm_120);
    const cases = [_]struct { body: u32, factor: u32 }{
        .{ .body = 0, .factor = 1 }, // nothing to unroll
        .{ .body = 1, .factor = 8 }, // MAX_FACTOR, not the register budget
        .{ .body = 4, .factor = 8 },
        .{ .body = 10, .factor = 8 }, // 80 registers exactly: the last body size that reaches the cap
        .{ .body = 11, .factor = 7 }, // 88 would drop occupancy below the floor
        .{ .body = 12, .factor = 6 },
        .{ .body = 16, .factor = 5 },
        .{ .body = 20, .factor = 4 },
        .{ .body = 27, .factor = 2 },
        .{ .body = 40, .factor = 2 },
        .{ .body = 41, .factor = 1 }, // two copies already cost the occupancy
        .{ .body = 100, .factor = 1 },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c.factor, unrollFactor(g, c.body));
    }
}

test "the sm_120 factor sits exactly on the occupancy floor: one more copy crosses it" {
    // The boundary the rule is built on, checked from the occupancy side rather than by repeating
    // the expected factors. For every body size that gives a factor below the cap, the chosen K
    // must hold at least the floor and K+1 must not. This is the assertion that fails if the
    // occupancy bound is removed (the factor runs to MAX_FACTOR) or pinned to 1.
    const registry = @import("registry.zig");
    const g = registry.modelFor(.sm_120);
    const s = g.simt.?;
    const floor = occupancyFloor(s);
    try std.testing.expectEqual(@as(u32, 24), floor); // half of 48 resident warps

    var body: u32 = 1;
    while (body <= 80) : (body += 1) {
        const k = unrollFactor(g, body);
        try std.testing.expect(k >= 1 and k <= MAX_FACTOR);
        if (k > 1) try std.testing.expect(s.residentWarps(simtLiveRegs(body, k)) >= floor);
        if (k < MAX_FACTOR) try std.testing.expect(s.residentWarps(simtLiveRegs(body, k + 1)) < floor);
    }
}

/// A hand-built SM whose register file is large enough that occupancy never binds, so the
/// per-thread register CAP is the clause that stops the unroll. `Model` is public and
/// user-constructible for exactly this reason.
const uncapped_occupancy_simt = mm.Simt{
    .warp_size = 32,
    .warps_per_sm = 4, // a tiny ceiling, so the floor is 2 warps and easily met
    .regfile_per_sm = 1 << 24, // far more than 4 warps can spend
    // Deliberately NOT a multiple of the granularity, as sm_120's own 255 is not, so the rounding
    // can carry a request that fits over the cap. The test below proves that case.
    .max_regs_per_thread = 60,
    .reg_alloc_granularity = 8,
    .min_regs_per_thread = 16,
    .global_latency = 500,
    .shared_latency = 30,
};

test "the per-thread register cap stops the unroll when occupancy does not" {
    // On sm_120 the occupancy floor always bites first, so this clause never fires there. It is not
    // dead: a part with a large register file relative to its warp ceiling reaches the per-thread
    // cap instead, and a kernel past that cap spills to local memory, which adds a memory access to
    // hide a memory access.
    const s = uncapped_occupancy_simt;
    try std.testing.expectEqual(@as(u32, 4), simtUnrollFactor(s, 12)); // 48 registers, under the 60 cap
    try std.testing.expectEqual(@as(u32, 3), simtUnrollFactor(s, 15)); // a fourth copy would need 60, which rounds to 64
    try std.testing.expectEqual(@as(u32, 1), simtUnrollFactor(s, 31)); // two copies need 62 outright
    // Rounding crosses the cap on its own: 58 registers is under 60, but it rounds up to 64.
    try std.testing.expectEqual(@as(u32, 1), simtUnrollFactor(s, 29));
}

/// A hand-built SM whose memory latency is small enough that the resident warps already cover it.
/// Same shape as sm_120 otherwise.
const short_latency_simt = mm.Simt{
    .warp_size = 32,
    .warps_per_sm = 48,
    .regfile_per_sm = 65536,
    .max_regs_per_thread = 255,
    .reg_alloc_granularity = 8,
    .min_regs_per_thread = 16,
    .global_latency = 96, // 48 warps * 2 copies covers it exactly
    .shared_latency = 30,
};

test "the latency-cover clause stops an unroll once the work in flight already covers the memory" {
    // Inert on sm_120 (a comptime block above asserts it cannot fire there), so it is proven on a
    // part where it can. With 48 resident warps and a 96-cycle memory, two copies per warp put 96
    // operations in flight and a third buys nothing but registers.
    try std.testing.expectEqual(@as(u32, 2), simtUnrollFactor(short_latency_simt, 4));
    // The same body under sm_120's real 500-cycle memory is bounded by registers instead, and goes
    // all the way to the cap.
    const registry = @import("registry.zig");
    try std.testing.expectEqual(@as(u32, 8), unrollFactor(registry.modelFor(.sm_120), 4));
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

/// The strict counted-loop shape, proven by `countedShape`. A loop that
/// matches it can take the main-loop-plus-remainder emission, which needs one
/// guard for K body copies instead of one guard per copy.
const Counted = struct {
    counter: u32, // the header param position of the loop counter
    bound: Value, // the loop-invariant bound the counter is compared against
    op: ir.function.CmpOp, // the header test's relation, `.lt` or `.le`
};

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
    preheader: Block, // the loop's single clean entry
    counted: ?Counted, // non-null when the strict SIMT counted shape was proven
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

/// Model-driven partial unroll. Analyzes `func`'s natural loops, unrolls
/// every loop it can prove eligible by `unrollFactor(model, ...)` copies, and
/// leaves everything else untouched. Returns whether anything changed. An
/// ineligible or un-cleanly-transformable loop is always left exactly as it was.
/// Two shapes are emitted: K guarded body copies, or, when the model is SIMT
/// and the loop matches the strict counted shape, a main block of K unguarded
/// copies plus a bounded remainder. A load-bearing factor-five loop gets four
/// forward tail copies; other factors retain the original scalar loop. See
/// `countedShape` for what the strict checks prove.
/// Idempotence note: re-running `run` on its own output stays correct. The
/// remainder loop loses its preheader (the main header branches to two
/// blocks), the main loop no longer matches the strict shape, and any nested
/// transform is correct by construction, so nesting is legal rather than
/// forbidden.
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
    var factor = unrollFactor(model, body_ops);
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

    // The strict counted shape decides between the two emissions. Null means
    // the loop keeps the per-copy-guard shape.
    const counted = countedShape(
        func,
        model,
        header,
        iff,
        l,
        body_entry,
        body_list.items,
        func.blockParams(header),
        back_args,
    );

    // The instruction-count register bound is deliberately conservative, but
    // it under-unrolls arithmetic recurrences whose temporary results die in
    // each copy. On sm_120, ptxas carries 64 FFMAs per branch for k2 while the
    // generic factor carries 32. Four copies retain the same four accumulator
    // registers and match that hot-loop shape.
    if (factor == 2 and counted != null and arithmeticRecurrenceCanUseFour(func, in_loop, body_ops))
        factor = 4;

    return Plan{
        .header = header,
        .if_inst = iff,
        .exit = exit,
        .body_entry = body_entry,
        .latch = l,
        .body_blocks = try body_list.toOwnedSlice(allocator),
        .in_loop = in_loop_copy,
        .factor = factor,
        .preheader = @enumFromInt(loop.preheader.?),
        .counted = counted,
    };
}

fn arithmeticRecurrenceCanUseFour(func: *const Function, in_loop: []const bool, body_ops: u32) bool {
    if (body_ops > 64) return false;
    var float_arith: u32 = 0;
    for (in_loop, 0..) |member, block_index| {
        if (!member) continue;
        for (func.blockInsts(@enumFromInt(block_index))) |inst| switch (func.opcode(inst)) {
            .load, .store, .call, .call_indirect, .prefetch, .matmul, .atomic_rmw, .barrier, .va_start, .va_arg, .va_end => return false,
            .arith, .arith_imm => if (func.instResult(inst)) |result| switch (func.types.type_kind(func.valueType(result))) {
                .float => float_arith += 1,
                else => {},
            },
            else => {},
        };
    }
    return float_arith >= 8;
}

/// Whether `v` is a parameter or an instruction result of `block`.
fn definedInBlock(func: *const Function, v: Value, block: Block) bool {
    for (func.blockParams(block)) |p| {
        if (p == v) return true;
    }
    for (func.blockInsts(block)) |inst| {
        if (func.instResult(inst)) |r| {
            if (r == v) return true;
        }
    }
    return false;
}

/// Prove the strict counted shape on a loop `eligible` already accepted, for
/// the SIMT main-loop-plus-remainder emission. Every condition must hold. Any
/// miss returns null, and the loop keeps the per-copy-guard shape:
///
///   1. The model executes as SIMT.
///   2. The loop is one straight-line body block: it is the body entry and the
///      latch at once, and holds no `if`.
///   3. The body receives the carried values either through identity edge
///      args (body param k enters as header param k) or by reading the
///      header params directly with no edge args.
///   4. One header param j is the counter: the back edge feeds it the value
///      `body param j + 1`.
///   5. The header test is `hparams[j] <op> bound` with `.lt` or `.le`, and
///      the bound is defined outside the loop.
///   6. The counter is an integer.
fn countedShape(
    func: *const Function,
    model: *const mm.Model,
    header: Block,
    iff: Inst,
    latch: Block,
    body_entry: Block,
    body_blocks: []const Block,
    hparams: []const Value,
    back_args: []const Value,
) ?Counted {
    if (model.exec != .simt) return null;

    // One body block, which is the body entry and the latch at once.
    if (body_blocks.len != 1) return null;
    const body = body_blocks[0];
    if (body != body_entry or latch != body) return null;

    // A straight-line body: the K copies must fall through each other inside
    // one block, so nothing inside the body may branch.
    for (func.blockInsts(body)) |inst| {
        if (func.opcode(inst) == .@"if") return null;
    }

    // How the body receives the carried values. Two shapes are accepted:
    // either the header passes its params to the body unchanged (body param k
    // enters as header param k), or the body takes no edge args at all and
    // reads the header params directly. Both name the same values, and both
    // map onto the main header's params one to one.
    const cf = func.opcode(iff).@"if";
    const in_jump = if (cf.then.target == body_entry) cf.then else cf.@"else";
    const in_args = func.blockArgs(in_jump);
    const bparams = func.blockParams(body);
    if (in_args.len == 0) {
        if (bparams.len != 0) return null;
    } else {
        if (in_args.len != hparams.len) return null;
        if (bparams.len != hparams.len) return null;
        for (in_args, hparams) |ia, hp| {
            if (ia != hp) return null;
        }
    }

    // The header test: `counter <op> bound`, the counter on the left.
    const cmp = switch (func.opcode(func.definingInst(cf.cond) orelse return null)) {
        .icmp => |c| c,
        else => return null,
    };
    if (cmp.op != .lt and cmp.op != .le) return null;

    // The counter: a back-edge arg that is `counter + 1`, with the test
    // reading header param j. The +1 step is what lets one guard prove K
    // iterations: the counter at copy k is exactly `counter + k`. The step
    // reads either the header param itself (the no-args shape) or the body
    // param bound to it (the pass-through shape).
    var counter: ?u32 = null;
    for (back_args, 0..) |ba, j| {
        const step = switch (func.opcode(func.definingInst(ba) orelse continue)) {
            .arith_imm => |a| a,
            else => continue,
        };
        if (step.op != .add or step.imm != 1) continue;
        if (cmp.lhs != hparams[j]) continue;
        const stepped_on = if (bparams.len == hparams.len and j < bparams.len)
            step.lhs == bparams[j] or step.lhs == hparams[j]
        else
            step.lhs == hparams[j];
        if (!stepped_on) continue;
        counter = @intCast(j);
        break;
    }
    const j = counter orelse return null;

    // The bound must be loop-invariant: the main header reads it, and it is
    // reached from the preheader, so a value defined inside the loop would not
    // dominate the new block.
    if (definedInBlock(func, cmp.rhs, header)) return null;
    if (definedInBlock(func, cmp.rhs, body)) return null;

    // The counter must be an integer: the main guard adds (K-1) to it with an
    // integer add. The guard carries its own wrap check, so every width is
    // safe, and no width rule is needed here.
    switch (func.types.type_kind(func.valueType(hparams[j]))) {
        .int => {},
        else => return null,
    }

    return .{
        .counter = j,
        .bound = cmp.rhs,
        .op = cmp.op,
    };
}

/// Steps B and C: rewrite escaping values into loop-closed form, then unroll by
/// K. The strict counted shape takes the main-loop-plus-remainder emission; any
/// other eligible loop gets K-1 extra guarded body copies spliced between the
/// header entry and the back-edge.
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

    // --- Step C: unroll by K ---
    // The strict SIMT counted shape takes the main-loop-plus-remainder
    // emission. Everything else takes the guarded one below.
    if (plan.counted) |ct| {
        try applyCounted(a, func, plan, ct);
        return;
    }

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

/// Step C, counted shape: a main loop of K unguarded body copies plus the
/// original loop as the remainder.
///
/// WHY ONE GUARD IS ENOUGH. In the guarded shape, copy k tests the counter at
/// `c + k`. The main guard tests only the last of those counters, `c + (K-1)`,
/// and adds one check: that the add did not wrap. With no wrap, the counter
/// rises by exactly one per copy, so the last copy's test proves every earlier
/// copy's test at once, and the K copies need no guards of their own. When the
/// add wraps (the counter sits within K-1 of the top of its range), the main
/// loop is refused, and the remainder runs the copies one by one under their
/// own wrapped tests. The wrap check is `sum > counter` in the counter's own
/// width: a positive addend that wraps lands the sum below its operand, in
/// both signed and unsigned arithmetic. No wider type is needed, so the guard
/// costs one add, two compares, and one and.
///
/// The blocks built (P the preheader, H the original header, B the original
/// body, K the factor):
///
///   P        jumps to H_main instead of H, with the same initial values
///   H_main   one param per H param, same types. The FIRST trip's guard
///            branches to M or H. It runs once per kernel.
///   M        the rotated main body, one param per H param. Each trip computes
///            the NEXT trip's guard at its top, runs K instruction-clones of B
///            chained copy to copy, and closes with the guarded branch at its
///            bottom: back to M, or out to H with the values the trip ended on
///   H and B  untouched: the remainder loop, entered from H_main and from M
///
/// WHY THE BRANCH SITS AT THE BOTTOM. A compare feeding a branch needs about
/// fourteen cycles on sm_120 before the branch can read its predicate. A guard
/// in its own head block puts the two beside each other, and the warp stalls
/// on that gap every trip. The rotated form computes the next trip's guard at
/// the top of the body, so a full body of work travels between the compare
/// and the branch that reads it, and the latency hides behind the copies. The
/// one guard chain left in H_main runs once per kernel.
///
/// The body never reads a header instruction result (`eligible` proves that),
/// so nothing but the guard needs recomputing in H_main or M.
fn applyCounted(a: std.mem.Allocator, func: *Function, plan: *const Plan, ct: Counted) Error!void {
    // Snapshot every borrowed slice first. The appends below can move the
    // pools those slices point into.
    const hparams = try a.dupe(Value, func.blockParams(plan.header));
    const bparams = try a.dupe(Value, func.blockParams(plan.body_entry));
    const body_insts = try a.dupe(Inst, func.blockInsts(plan.body_entry));
    const back_args = try a.dupe(Value, func.blockArgs(func.terminator(plan.latch).?.jump));
    const step_value = back_args[ct.counter];
    const step_inst = func.definingInst(step_value).?;
    const exit_jump = blk: {
        const cf = func.opcode(plan.if_inst).@"if";
        break :blk if (cf.then.target == plan.exit) cf.then else cf.@"else";
    };
    const exit_args = try a.dupe(Value, func.blockArgs(exit_jump));
    const collapsed_counter = inductionStepIsPrivate(
        func,
        body_insts,
        hparams[ct.counter],
        if (bparams.len == hparams.len) bparams[ct.counter] else null,
        step_inst,
        step_value,
        back_args,
        ct.counter,
    );

    // H_main: one param per original header param, same types. It holds the
    // guard for the FIRST main trip only, so its short predicate chain runs
    // once per kernel, not once per trip.
    const h_main = try func.appendBlock();
    const hmparams = try a.alloc(Value, hparams.len);
    for (hparams, 0..) |hp, k| hmparams[k] = try func.appendBlockParam(h_main, func.valueType(hp));

    // The guard: two tests in the counter's own width. The add may wrap, and
    // the first test catches that: a positive addend that wraps lands the sum
    // below its operand. The second test is the last copy's own test, and
    // with no wrap it proves every copy's test at once.
    const counter_t = func.valueType(hmparams[ct.counter]);
    const bool_t = try func.types.intern(.bool);
    const last0 = try func.appendArithImm(h_main, counter_t, .add, hmparams[ct.counter], @intCast(plan.factor - 1));
    const nowrap0 = try func.appendInst(h_main, bool_t, .{ .icmp = .{ .op = .gt, .lhs = last0, .rhs = hmparams[ct.counter] } });
    const fits0 = try func.appendInst(h_main, bool_t, .{ .icmp = .{ .op = ct.op, .lhs = last0, .rhs = ct.bound } });
    const guard0 = try func.appendInst(h_main, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = nowrap0, .rhs = fits0 } });

    // Once the exact first guard admits a signed `<` loop whose counter is
    // private, `counter + K` cannot overflow: it is at most `bound`. The next
    // K iterations then fit exactly when `counter + K <= bound - K`. Hoist
    // that invariant subtraction out of the hot loop and replace its
    // add/wrap-test/bound-test chain with one comparison.
    const threshold_guard = collapsed_counter and ct.op == .lt and switch (func.types.type_kind(counter_t)) {
        .int => |int| int.signedness == .signed,
        else => false,
    };
    const main_limit = if (threshold_guard)
        try func.appendArithImm(h_main, counter_t, .add, ct.bound, -@as(i64, @intCast(plan.factor)))
    else
        undefined;

    // M: the rotated main body, one param per header param. The trip enters
    // through its params, computes the guard for the NEXT trip at the top,
    // runs the K copies, and closes with the guarded branch at the bottom.
    const m = try func.appendBlock();
    const mparams = try a.alloc(Value, hparams.len);
    for (hparams, 0..) |hp, k| mparams[k] = try func.appendBlockParam(m, func.valueType(hp));
    try func.appendIf(h_main, guard0, .{ .target = m, .args = hmparams }, .{ .target = plan.header, .args = hmparams });

    const advanced_counter = if (collapsed_counter)
        try func.appendArithImm(m, counter_t, .add, mparams[ct.counter], @intCast(plan.factor))
    else
        undefined;

    // The general guard compares for the next trip from THIS trip's entry
    // counter. Its wrap test and last-copy test remain separate until the
    // middle of the body. The private signed-`<` case uses the hoisted
    // threshold above and needs only one comparison.
    var nowrap: Value = undefined;
    var fits: Value = undefined;
    var guard: ?Value = null;
    if (threshold_guard) {
        guard = try func.appendInst(m, bool_t, .{ .icmp = .{ .op = .le, .lhs = advanced_counter, .rhs = main_limit } });
    } else {
        const last = try func.appendArithImm(m, counter_t, .add, mparams[ct.counter], @intCast(2 * plan.factor - 1));
        nowrap = try func.appendInst(m, bool_t, .{ .icmp = .{ .op = .gt, .lhs = last, .rhs = mparams[ct.counter] } });
        fits = try func.appendInst(m, bool_t, .{ .icmp = .{ .op = ct.op, .lhs = last, .rhs = ct.bound } });
    }

    // K copies of the body in one block, chained through the map: the entry
    // values of copy n+1 are the back-edge args of copy n, read through the
    // map. Copy 0 reads M's own params, through both the body params (the
    // pass-through shape) and the header params (both shapes).
    var vmap: ValueMap = .empty;
    for (hparams, mparams) |hp, mp| try vmap.put(a, hp, mp);
    if (bparams.len == hparams.len) {
        for (bparams, mparams) |bp, mp| try vmap.put(a, bp, mp);
    }
    var carried = try a.alloc(Value, back_args.len);
    // If the induction value is observed only by its own `+ 1` update, the K
    // cloned updates are indistinguishable from one wrapping `+ K`. Keep that
    // update before the independent body work so scheduling can hide it, and
    // omit the K originals. This is the counted-loop analogue of ptxas's
    // single `UIADD3 counter, counter, K`; in an arithmetic recurrence it
    // removes K-1 instructions from every hot trip.
    // The combine lands mid-body, after half the copies. The predicate path
    // holds two slow hops, compare to combine and combine to branch, so a
    // combine beside its compares stacks both at the top of the block, and a
    // combine beside the branch stacks both at the bottom. Mid-body splits
    // them: each hop gets half the copies to travel.
    const combine_after = (plan.factor + 1) / 2;
    var copy: u32 = 0;
    while (copy < plan.factor) : (copy += 1) {
        for (body_insts) |inst| {
            if (collapsed_counter and inst == step_inst) continue;
            try cloneInstInto(func, m, inst, &vmap, a);
        }

        // Read the whole carried list before any map entry is overwritten. A
        // back-edge arg may name a header or body param directly, and its
        // entry still maps to the copy that just ran.
        for (back_args, 0..) |ba, k| carried[k] = rv(&vmap, ba);
        if (collapsed_counter) carried[ct.counter] = advanced_counter;
        if (copy + 1 < plan.factor) {
            for (hparams, 0..) |hp, k| try vmap.put(a, hp, carried[k]);
            for (bparams, 0..) |bp, k| try vmap.put(a, bp, carried[k]);
        }
        if (guard == null and copy + 1 == combine_after) {
            guard = try func.appendInst(m, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = nowrap, .rhs = fits } });
        }
    }

    // The closing branch. The guard was computed at the top of this trip, so
    // it is ready when the trip ends. Both edges carry the values the trip
    // finished on: the back edge feeds M's params, and the else edge hands
    // the remainder the counter and accumulators as they stand.
    try func.appendIf(m, guard.?, .{ .target = m, .args = carried }, .{ .target = plan.header, .args = carried });

    // Move the preheader onto the main header, keeping the initial values.
    // The loop analysis names a preheader only when its single successor is
    // the header, so a jump is the shape that reaches here. The `if` arm
    // keeps a conditional preheader correct should the analysis ever name
    // one.
    if (func.terminatorPtr(plan.preheader).*) |term| switch (term) {
        .jump => |j| {
            func.terminatorPtr(plan.preheader).* = null;
            try func.setJump(plan.preheader, h_main, func.valueList(j.args));
        },
        // A preheader with a `ret` terminator has no successor, so the loop
        // analysis cannot have named it one.
        .ret => {},
    } else {
        for (func.blockInsts(plan.preheader)) |inst| {
            const op = func.opcodeMut(inst);
            if (op.* != .@"if") continue;
            if (op.@"if".then.target == plan.header) op.@"if".then.target = h_main;
            if (op.@"if".@"else".target == plan.header) op.@"if".@"else".target = h_main;
        }
    }

    if (plan.factor == 5 and ct.op == .lt and bodyContainsLoad(func, body_insts) and tailExitArgsSafe(func, plan.header, exit_args)) {
        try linearizeCountedRemainder(a, func, plan, ct, hparams, bparams, body_insts, back_args, exit_args);
    }
}

fn bodyContainsLoad(func: *const Function, insts: []const Inst) bool {
    for (insts) |inst| if (func.opcode(inst) == .load) return true;
    return false;
}

/// The counted-loop update may be widened only when no body computation or
/// other carried value observes an intermediate counter. `countedShape`
/// already proves `step_inst` is exactly the counter's wrapping `+ 1`.
fn inductionStepIsPrivate(
    func: *const Function,
    body_insts: []const Inst,
    header_counter: Value,
    body_counter: ?Value,
    step_inst: Inst,
    step_value: Value,
    back_args: []const Value,
    counter_index: u32,
) bool {
    for (body_insts) |inst| {
        if (inst == step_inst) continue;
        if (instUsesValue(func, inst, header_counter) or
            (body_counter != null and instUsesValue(func, inst, body_counter.?)) or
            instUsesValue(func, inst, step_value)) return false;
    }
    for (back_args, 0..) |arg, i| {
        if (i == counter_index) continue;
        if (arg == header_counter or arg == body_counter or arg == step_value) return false;
    }
    return true;
}

/// Whether one instruction reads `value`. Exhaustive so a newly added opcode
/// cannot silently make the induction proof unsound.
fn instUsesValue(func: *const Function, inst: Inst, value: Value) bool {
    return switch (func.opcode(inst)) {
        .arith => |x| x.lhs == value or x.rhs == value,
        .arith_imm => |x| x.lhs == value,
        .icmp => |x| x.lhs == value or x.rhs == value,
        .select => |x| x.cond == value or x.then == value or x.@"else" == value,
        .extract => |x| x.aggregate == value,
        .convert => |x| x.value == value,
        .unary => |x| x.value == value,
        .load => |x| x.ptr == value,
        .store => |x| x.value == value or x.ptr == value,
        .atomic_rmw => |x| x.ptr == value or x.value == value or (x.compare != null and x.compare.? == value),
        .prefetch => |x| x.ptr == value,
        .va_start => |x| x.list == value,
        .va_arg => |x| x.list == value,
        .va_end => |x| x.list == value,
        .dot => |x| x.acc == value or x.a == value or x.b == value,
        .matmul => |x| x.a == value or x.b == value or x.c == value,
        .struct_new => |x| blk: {
            for (func.valueList(x.fields)) |field| if (field == value) break :blk true;
            break :blk false;
        },
        .call => |x| blk: {
            for (func.valueList(x.args)) |arg| if (arg == value) break :blk true;
            break :blk false;
        },
        .call_indirect => |x| blk: {
            if (x.target == value) break :blk true;
            for (func.valueList(x.args)) |arg| if (arg == value) break :blk true;
            break :blk false;
        },
        .@"if" => |x| blk: {
            if (x.cond == value) break :blk true;
            for (func.blockArgs(x.then)) |arg| if (arg == value) break :blk true;
            for (func.blockArgs(x.@"else")) |arg| if (arg == value) break :blk true;
            break :blk false;
        },
        .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => false,
    };
}

fn tailExitArgsSafe(func: *const Function, header: Block, args: []const Value) bool {
    for (args) |arg| {
        if (func.definingInst(arg) != null and definedInBlock(func, arg, header)) return false;
    }
    return true;
}

/// Replace the bounded scalar remainder with at most K minus 1 forward body
/// copies. Each copy keeps the original test, but no copy branches backward.
fn linearizeCountedRemainder(
    a: std.mem.Allocator,
    func: *Function,
    plan: *const Plan,
    ct: Counted,
    hparams: []const Value,
    bparams: []const Value,
    body_insts: []const Inst,
    back_args: []const Value,
    exit_args: []const Value,
) Error!void {
    var prev_body = plan.body_entry;
    var carried = try a.dupe(Value, back_args);
    var copy: u32 = 1;
    while (copy < plan.factor - 1) : (copy += 1) {
        const check = try func.appendBlock();
        const check_params = try a.alloc(Value, hparams.len);
        for (hparams, 0..) |hp, i| check_params[i] = try func.appendBlockParam(check, func.valueType(hp));

        func.terminatorPtr(prev_body).* = null;
        try func.setJump(prev_body, check, carried);

        const body = try func.appendBlock();
        const cond = try func.appendInst(check, try func.types.intern(.bool), .{ .icmp = .{
            .op = ct.op,
            .lhs = check_params[ct.counter],
            .rhs = ct.bound,
        } });
        var exit_map: ValueMap = .empty;
        for (hparams, check_params) |hp, cp| try exit_map.put(a, hp, cp);
        const out = try a.alloc(Value, exit_args.len);
        for (exit_args, 0..) |arg, i| out[i] = rv(&exit_map, arg);
        try func.appendIf(check, cond, .{ .target = body, .args = &.{} }, .{ .target = plan.exit, .args = out });

        var vmap: ValueMap = .empty;
        for (hparams, check_params) |hp, cp| try vmap.put(a, hp, cp);
        if (bparams.len == hparams.len) {
            for (bparams, check_params) |bp, cp| try vmap.put(a, bp, cp);
        }
        for (body_insts) |inst| try cloneInstInto(func, body, inst, &vmap, a);
        const next = try a.alloc(Value, back_args.len);
        for (back_args, 0..) |arg, i| next[i] = rv(&vmap, arg);
        carried = next;
        prev_body = body;
    }

    func.terminatorPtr(prev_body).* = null;
    var final_map: ValueMap = .empty;
    for (hparams, carried) |hp, value| try final_map.put(a, hp, value);
    const out = try a.alloc(Value, exit_args.len);
    for (exit_args, 0..) |arg, i| out[i] = rv(&final_map, arg);
    try func.setJump(prev_body, plan.exit, out);
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

/// A loop shaped like a GPU kernel's inner loop: one global load per iteration, `pad` independent
/// index computations beside it, and a counted induction variable. `pad` is how the test varies the
/// body size, which is the only thing the SIMT rule reads.
///
/// The single load is the marker the tests count: guards re-emit only the header's pure test, never
/// a load, so the number of loads in the whole function after unrolling IS the unroll factor.
fn buildGpuLoop(func: *Function, pad: u32) Error!void {
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
    const base = try func.appendGlobalAddr(entry, ptr_t, "A");
    try func.setJump(entry, loop, &.{zero});

    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });

    _ = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = base } });
    var j: u32 = 0;
    while (j < pad) : (j += 1) {
        _ = try func.appendArithImm(body, i32_t, .add, bi, @intCast(j + 2));
    }
    const next = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{next});

    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

/// `buildGpuLoop(0)` with the counter stepping by two. The strict counted check
/// needs `body param + 1` on the back edge, so this loop must keep the
/// per-copy-guard shape.
fn buildGpuLoopStep2(func: *Function) Error!void {
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
    const base = try func.appendGlobalAddr(entry, ptr_t, "A");
    try func.setJump(entry, loop, &.{zero});

    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });

    _ = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = base } });
    const next = try func.appendArithImm(body, i32_t, .add, bi, 2);
    try func.setJump(body, loop, &.{next});

    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

/// `buildGpuLoop(0)` with an i64 counter. Every strict condition holds, and
/// the wrap check makes the guard exact at any width, so the counted emission
/// must take it.
fn buildGpuLoopI64(func: *Function) Error!void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);
    const i = try func.appendBlockParam(loop, i64_t);
    const bi = try func.appendBlockParam(body, i64_t);
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const base = try func.appendGlobalAddr(entry, ptr_t, "A");
    try func.setJump(entry, loop, &.{zero});

    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });

    _ = try func.appendInst(body, i64_t, .{ .load = .{ .ptr = base } });
    const next = try func.appendArithImm(body, i64_t, .add, bi, 1);
    try func.setJump(body, loop, &.{next});

    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

/// How many `load` instructions the whole function holds.
fn countLoads(func: *const Function) u32 {
    var loads: u32 = 0;
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .load) loads += 1;
        }
    }
    return loads;
}

/// How many `load` instructions one block holds.
fn loadsInBlock(func: *const Function, block: Block) u32 {
    var loads: u32 = 0;
    for (func.blockInsts(block)) |inst| {
        if (func.opcode(inst) == .load) loads += 1;
    }
    return loads;
}

test "a GPU loop unrolls by exactly the factor the SIMT rule predicts, and stays verifiable" {
    // The end-to-end check of the rule at the IR level. `buildGpuLoop(pad)` makes a loop whose body
    // budget is `pad + 4` (two header instructions plus the load, the pads and the increment), and
    // the function must come out holding `unrollFactor + 1` loads: one per copy in the main block,
    // plus the single load of the remainder body the split leaves behind.
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    const g = registry.modelFor(.sm_120);

    const cases = [_]struct { pad: u32, factor: u32 }{
        .{ .pad = 0, .factor = 8 }, // body 4: the shared MAX_FACTOR cap, not the register budget
        .{ .pad = 8, .factor = 6 }, // body 12
        .{ .pad = 12, .factor = 5 }, // body 16: four-copy forward remainder
        .{ .pad = 16, .factor = 4 }, // body 20
        .{ .pad = 23, .factor = 2 }, // body 27
        .{ .pad = 37, .factor = 1 }, // body 41: two copies already cost the occupancy
    };
    for (cases) |c| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildGpuLoop(&func, c.pad);

        // The rule's own answer for this body size, so the table below is pinned to the rule and
        // not only to a number typed out by hand.
        try std.testing.expectEqual(c.factor, unrollFactor(g, c.pad + 4));

        const before_blocks = func.blockCount();
        const changed = try run(allocator, &func, g);
        try std.testing.expectEqual(c.factor > 1, changed);
        // Factor five gets four forward tail copies. Other counted loops keep
        // one scalar remainder body.
        if (c.factor > 1) {
            const tail_copies: u32 = if (c.factor == 5) 4 else 1;
            try std.testing.expectEqual(c.factor + tail_copies, countLoads(&func));
        } else {
            try std.testing.expectEqual(@as(u32, 1), countLoads(&func));
        }
        if (c.factor == 1) try std.testing.expectEqual(before_blocks, func.blockCount());

        if (c.factor == 5) {
            var info = try loops.analyze(allocator, &func);
            defer info.deinit(allocator);
            for (info.loops) |l| {
                try std.testing.expect(l.header != 1);
            }
        }

        var diags = try ir.verify.verify(allocator, &func, .low);
        defer diags.deinit();
        try std.testing.expect(diags.ok());
    }
}

test "the same GPU loop under a CPU model unrolls by the CPU factor, not the SIMT one" {
    // The two rules must not be reading each other's numbers. A body of 12 gives 6 copies on
    // sm_120 (register budget) and 1 on ampere-altra ((4*3)/12 = 0, clamped to 1).
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildGpuLoop(&func, 8);

    const changed = try run(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(!changed);
    try std.testing.expectEqual(@as(u32, 1), countLoads(&func));
}

test "a counted sm_120 loop becomes one unguarded main block plus a guarded remainder" {
    // The counted emission, checked end to end at the IR level. `buildGpuLoop(0)`
    // makes a body budget of 4, so sm_120 unrolls by 8. The result must hold: a
    // main body block with exactly 8 loads, all copies unguarded and closed by
    // one rotated guard branch at the block's end (which is what lets the
    // per-block load hoister group them), the original loop left whole as the
    // remainder (still guarding its one load), and the preheader moved onto
    // the new main header.
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildGpuLoop(&func, 0);
    const entry: Block = @enumFromInt(0);
    const loop: Block = @enumFromInt(1);
    const body: Block = @enumFromInt(2);

    const changed = try run(allocator, &func, registry.modelFor(.sm_120));
    try std.testing.expect(changed);

    // The preheader now jumps to the main header, not to the original one.
    const h_main = func.terminator(entry).?.jump.target;
    try std.testing.expect(h_main != loop);

    // The main header holds exactly one `if`, the promoted guard, and it
    // branches to the main body and back to the remainder header.
    var m: Block = undefined;
    var guard_ifs: usize = 0;
    for (func.blockInsts(h_main)) |inst| {
        if (func.opcode(inst) != .@"if") continue;
        guard_ifs += 1;
        const cf = func.opcode(inst).@"if";
        m = cf.then.target;
        try std.testing.expectEqual(loop, cf.@"else".target);
    }
    try std.testing.expectEqual(@as(usize, 1), guard_ifs);

    // The main body: K unguarded copies in ONE block, closed by the rotated
    // guard branch. That one `if` is the ONLY branch in the block, it is the
    // last instruction, and it loops back to the block itself.
    try std.testing.expectEqual(@as(u32, 8), loadsInBlock(&func, m));
    var m_ifs: usize = 0;
    for (func.blockInsts(m)) |inst| {
        if (func.opcode(inst) != .@"if") continue;
        m_ifs += 1;
        const cf = func.opcode(inst).@"if";
        try std.testing.expectEqual(m, cf.then.target);
        try std.testing.expectEqual(loop, cf.@"else".target);
    }
    try std.testing.expectEqual(@as(usize, 1), m_ifs);
    try std.testing.expect(func.terminator(m) == null);

    // The body never observes its counter, so its eight cloned `+ 1`
    // updates collapse to one `+ 8`. The signed threshold guard also hoists
    // its bound adjustment into H_main, leaving no other add in M.
    var main_adds: usize = 0;
    var saw_advance = false;
    var saw_threshold_guard = false;
    for (func.blockInsts(m)) |inst| switch (func.opcode(inst)) {
        .arith_imm => |arith| if (arith.op == .add) {
            main_adds += 1;
            saw_advance = saw_advance or arith.imm == 8;
        },
        .icmp => |cmp| saw_threshold_guard = saw_threshold_guard or cmp.op == .le,
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 1), main_adds);
    try std.testing.expect(saw_advance);
    try std.testing.expect(saw_threshold_guard);

    // The guard must add (K-1) to the counter, and test the sum twice: once
    // against the counter itself (the wrap check) and once against the bound
    // (the last copy's test). The unguarded copies are legal only because the
    // pair proves every per-copy test at once.
    var icmps: usize = 0;
    var gt = false;
    var lt = false;
    for (func.blockInsts(h_main)) |inst| {
        if (func.opcode(inst) != .icmp) continue;
        icmps += 1;
        const cmp = func.opcode(inst).icmp;
        switch (func.opcode(func.definingInst(cmp.lhs).?)) {
            .arith_imm => |a| {
                try std.testing.expectEqual(ir.function.BinOp.add, a.op);
                try std.testing.expectEqual(@as(i64, 7), a.imm);
            },
            else => return error.TestUnexpectedResult,
        }
        switch (cmp.op) {
            .gt => gt = cmp.rhs == func.blockParams(h_main)[0],
            .lt => lt = cmp.rhs == func.blockParams(@enumFromInt(0))[0],
            else => {},
        }
    }
    try std.testing.expectEqual(@as(usize, 2), icmps);
    try std.testing.expect(gt);
    try std.testing.expect(lt);
    var ands: usize = 0;
    for (func.blockInsts(h_main)) |inst| {
        if (func.opcode(inst) != .arith) continue;
        const ar = func.opcode(inst).arith;
        if (ar.op == .bit_and) ands += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), ands);

    // The remainder: the original header still guards the original body, and
    // the body keeps its single load.
    for (func.blockInsts(loop)) |inst| {
        if (func.opcode(inst) != .@"if") continue;
        try std.testing.expectEqual(body, func.opcode(inst).@"if".then.target);
    }
    try std.testing.expectEqual(@as(u32, 1), loadsInBlock(&func, body));

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "the main header's guard works in the counter's own width, with no converts" {
    // The guard must not widen the counter. The wrap check makes the add
    // exact in the counter's own width, so the guard holds no convert at all:
    // the backend for a target without wide scalars still lowers it.
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildGpuLoop(&func, 0);

    const changed = try run(allocator, &func, registry.modelFor(.sm_120));
    try std.testing.expect(changed);

    const h_main = func.terminator(@enumFromInt(0)).?.jump.target;
    var converts: usize = 0;
    var add: ?Inst = null;
    for (func.blockInsts(h_main)) |inst| {
        if (func.opcode(inst) == .convert) converts += 1;
        if (func.opcode(inst) == .arith_imm) add = inst;
    }
    try std.testing.expectEqual(@as(usize, 0), converts);

    // The add runs in the counter's own type, an i32.
    const bits = switch (func.types.type_kind(func.valueType(func.instResult(add.?).?))) {
        .int => |i| i.bits,
        else => return error.TestUnexpectedResult,
    };
    try std.testing.expectEqual(@as(u16, 32), bits);
}

test "a counted loop with a runtime trip count still verifies, and keeps both loop headers" {
    // The tail path. The trip count is a runtime value, so the guard decides at
    // run time how many main trips run before the remainder takes over. After
    // the split, the original header must still head a loop (the remainder) and
    // the new main header must head one too, or the two halves cannot close.
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildGpuLoop(&func, 0);
    const loop: Block = @enumFromInt(1);

    const changed = try run(allocator, &func, registry.modelFor(.sm_120));
    try std.testing.expect(changed);

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    var loop_is_header = false;
    for (info.loops) |l| {
        if (@as(Block, @enumFromInt(l.header)) == loop) loop_is_header = true;
    }
    try std.testing.expect(loop_is_header);
    try std.testing.expectEqual(@as(usize, 2), info.loops.len);

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "a simt counter that steps by two keeps the per-copy guards" {
    // The strict checks need `body param + 1` on the back edge, because the
    // promoted guard proves only a counter that grows by one per copy. A step
    // of two must fall back to the guarded shape, which holds exactly
    // `factor` loads: one per copy, and no remainder copy.
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildGpuLoopStep2(&func);

    const changed = try run(allocator, &func, registry.modelFor(.sm_120));
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(u32, 8), countLoads(&func));

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "an i64 counter takes the counted shape on simt" {
    // The wrap check makes the guard exact in any width, so a 64-bit counter
    // is no longer a reason to refuse the counted shape. The loop must come
    // out with the main block plus the one remainder body.
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildGpuLoopI64(&func);

    const changed = try run(allocator, &func, registry.modelFor(.sm_120));
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(u32, 9), countLoads(&func));

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
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

/// `buildGpuLoop(0)` with no edge args: the body reads the header param
/// directly. This is the shape the GPU frontends build, so the counted
/// emission must take it exactly like the pass-through shape.
fn buildGpuLoopNoArgs(func: *Function) Error!void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const base = try func.appendGlobalAddr(entry, ptr_t, "A");
    try func.setJump(entry, loop, &.{zero});

    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body }, .{ .target = done });

    _ = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = base } });
    const next = try func.appendArithImm(body, i32_t, .add, i, 1);
    try func.setJump(body, loop, &.{next});

    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

test "a counted loop whose body reads the header params directly takes the counted shape" {
    // The no-args shape: the body holds no params of its own and names the
    // header params instead. The counted emission must still fire, with the
    // main block holding K copies plus the one remainder body.
    const registry = @import("registry.zig");
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildGpuLoopNoArgs(&func);

    const changed = try run(allocator, &func, registry.modelFor(.sm_120));
    try std.testing.expect(changed);
    try std.testing.expectEqual(@as(u32, 9), countLoads(&func));

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}
