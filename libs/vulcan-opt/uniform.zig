//! Uniformity (divergence) analysis for a GPU kernel.
//!
//! A value is UNIFORM when every thread of one workgroup holds the same value in it, and
//! DIVERGENT when two threads of one workgroup can hold different values. The scope is the
//! workgroup, because the first consumer is the workgroup-barrier legality check: a barrier
//! synchronizes the threads of one workgroup, so only a disagreement inside one workgroup can
//! desynchronize it. `gpu.Builtin.isWorkgroupUniform` owns the per-builtin half of that rule.
//!
//! The analysis is the least fixpoint of a monotone transfer over the function. It starts with
//! every value uniform, seeds the divergent sources, and only ever ADDS divergence. Divergence
//! cannot appear out of nothing, so a value stays uniform exactly when no source reaches it.
//!
//! ## What seeds divergence
//!
//!   - A parameter tagged with a per-thread builtin: `thread_id_*`, `global_id_*`, `lane_id`,
//!     `warp_id`, and every graphics builtin. See `gpu.Builtin.isWorkgroupUniform`.
//!   - A `load`, ALWAYS. See the conservative choices below.
//!   - A `call`, a `call_indirect`, an `atomic_rmw`, a `va_arg`, and an `alloca`.
//!
//! An untagged entry-block parameter is UNIFORM: it is a slot of the kernel parameter block,
//! and every thread reads the same parameter block. This analysis therefore assumes `func` is
//! a kernel entry. Running it on an ordinary function reads that function's parameters as
//! uniform, which is meaningless rather than unsafe, because such a function has no barrier.
//!
//! ## How divergence spreads
//!
//! Through OPERANDS: an operation with a divergent operand gives a divergent result.
//!
//! Through CONTROL, the classic sync dependence: a branch on a divergent condition splits the
//! threads, so a block that runs only on one side of it runs in some threads and not in others.
//! Every value defined in such a block is divergent, and so is every parameter of the block
//! where the two sides meet again, because the threads arrive there from different sides.
//!
//! ## Where this is deliberately conservative
//!
//! A LOAD IS ALWAYS DIVERGENT. A load is uniform only when its address is uniform AND no
//! divergent thread wrote to that address. The second half needs alias information this IR does
//! not carry, and shared memory exists precisely so that divergent threads can write where
//! another thread reads. Proving it is out of reach, so the analysis refuses to try. The cost:
//! a loop whose trip count is loaded from memory, for example a tile count read out of a
//! descriptor struct, reads as divergent. Passing that count as a scalar kernel parameter keeps
//! it uniform.
//!
//! A VALUE DEFINED UNDER A DIVERGENT BRANCH IS DIVERGENT, even when all of its operands are
//! uniform. The value itself would agree across the threads that compute it, but not every
//! thread computes it. This is the safe direction and it keeps the rule short. The cost is
//! precision only: a uniform expression sunk into a divergent arm reads as divergent.
//!
//! Both choices refuse where they cannot prove. An analysis that wrongly reports uniform admits
//! a barrier the hardware corrupts a shared-memory tile around, with no fault to show for it.

const std = @import("std");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");
const cfg_mod = @import("cfg.zig");
const dominators = @import("dominators.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;

pub const Error = std.mem.Allocator.Error;

/// The result of the analysis. Indexed by `Value` and by block index.
pub const Uniformity = struct {
    /// `divergent[v]` is true when value `v` can differ between two threads of one workgroup.
    divergent: []bool,
    /// `split[b]` is true when block `b` runs under a divergent branch, so the threads of the
    /// workgroup can be split while its instructions run.
    split: []bool,

    pub fn deinit(self: *Uniformity, allocator: std.mem.Allocator) void {
        allocator.free(self.divergent);
        allocator.free(self.split);
    }

    /// Whether every thread of one workgroup holds the same value in `value`.
    pub fn isUniform(self: *const Uniformity, value: Value) bool {
        return !self.isDivergent(value);
    }

    /// Whether two threads of one workgroup can hold different values in `value`.
    pub fn isDivergent(self: *const Uniformity, value: Value) bool {
        return self.divergent[@intFromEnum(value)];
    }

    /// Whether the workgroup can be split while block `block` runs.
    pub fn blockIsSplit(self: *const Uniformity, block: usize) bool {
        return self.split[block];
    }
};

/// The `if` of block `bi` when it branches two ways, or null. A one-way `if` (both edges to the
/// same block) never splits anything, so it is not a branch for this analysis.
pub fn twoWayIf(func: *const Function, bi: usize) ?ir.function.If {
    const block: Block = @enumFromInt(bi);
    for (func.blockInsts(block)) |inst| {
        switch (func.opcode(inst)) {
            .@"if" => |cf| {
                if (cf.then.target == cf.@"else".target) return null;
                return cf;
            },
            .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .select => {},
            .struct_new, .extract, .convert, .unary, .alloca => {},
            .call, .call_indirect, .global_addr, .load, .store, .prefetch => {},
            .va_start, .va_arg, .va_end, .dot, .matmul, .barrier, .atomic_rmw => {},
        }
    }
    return null;
}

/// Whether the result of `inst` is divergent because of its OPERANDS alone. The control-flow
/// half is added by the caller.
///
/// The switch is exhaustive on purpose. A new opcode must land here with a deliberate answer,
/// and an `else` arm would silently give it "uniform", which is the unsafe answer.
fn operandsDiverge(func: *const Function, divergent: []const bool, op: ir.function.Opcode) bool {
    const d = struct {
        fn f(set: []const bool, v: Value) bool {
            return set[@intFromEnum(v)];
        }
    }.f;
    return switch (op) {
        // Constants and a link-time address are the same in every thread.
        .iconst, .fconst, .fconst128, .global_addr => false,
        // Pure arithmetic: divergent exactly when an input is.
        .arith => |a| d(divergent, a.lhs) or d(divergent, a.rhs),
        .arith_imm => |a| d(divergent, a.lhs),
        .icmp => |c| d(divergent, c.lhs) or d(divergent, c.rhs),
        .select => |s| d(divergent, s.cond) or d(divergent, s.then) or d(divergent, s.@"else"),
        .convert => |c| d(divergent, c.value),
        .unary => |u| d(divergent, u.value),
        .extract => |e| d(divergent, e.aggregate),
        .dot => |x| d(divergent, x.acc) or d(divergent, x.a) or d(divergent, x.b),
        .struct_new => |s| blk: {
            for (func.valueList(s.fields)) |f| {
                if (d(divergent, f)) break :blk true;
            }
            break :blk false;
        },
        // Every thread owns a private stack frame, so the address differs per thread.
        .alloca => true,
        // See the module header: an address this analysis cannot prove nobody divergent wrote to.
        .load => true,
        // The callee is opaque, so its result can come from anywhere.
        .call, .call_indirect => true,
        // The OLD value an atomic returns depends on the order the threads arrive in.
        .atomic_rmw => true,
        // The next variadic argument comes out of memory, like a load.
        .va_arg => true,
        // These define no value, so nothing reads this answer for them.
        .store, .prefetch, .va_start, .va_end, .matmul, .barrier, .@"if" => false,
    };
}

/// Analyze the uniformity of every value in `func`. The caller owns the result (`deinit`).
pub fn analyze(allocator: std.mem.Allocator, func: *const Function) Error!Uniformity {
    const n = func.blockCount();
    const nv = func.valueCount();

    const divergent = try allocator.alloc(bool, nv);
    errdefer allocator.free(divergent);
    @memset(divergent, false);
    const split = try allocator.alloc(bool, n);
    errdefer allocator.free(split);
    @memset(split, false);

    // A block whose parameters are divergent because the threads reach it from more than one
    // side of a divergent branch. Kept apart from `split`, because the block itself runs with
    // the workgroup back together: it is the join, not the region.
    const join_split = try allocator.alloc(bool, n);
    defer allocator.free(join_split);
    @memset(join_split, false);

    // Seed: a parameter tagged with a per-thread builtin. Every other value starts uniform.
    for (0..nv) |vi| {
        const v: Value = @enumFromInt(vi);
        const b = gpu.attrs.builtinOf(func, v) orelse continue;
        if (!b.isWorkgroupUniform()) divergent[vi] = true;
    }

    var cfg = try cfg_mod.build(allocator, func);
    defer cfg.deinit(allocator);
    var pdoms = try dominators.computePost(allocator, func);
    defer pdoms.deinit(allocator);

    const seen = try allocator.alloc(bool, n);
    defer allocator.free(seen);

    var changed = true;
    while (changed) {
        changed = false;

        // Control: taint the region of every branch whose condition is now divergent.
        for (0..n) |ai| {
            const cf = twoWayIf(func, ai) orelse continue;
            if (!divergent[@intFromEnum(cf.cond)]) continue;
            // The join is where the two sides meet again. Null means they never do, and then
            // the region runs to the end of the function.
            const join = pdoms.immediatePostDominator(ai);
            try markRegion(allocator, &cfg, ai, join, split, seen, &changed);
            if (join) |m| {
                if (!join_split[m]) {
                    join_split[m] = true;
                    changed = true;
                }
            }
        }

        // Data: the transfer over each block's parameters and instructions.
        for (0..n) |bi| {
            const block: Block = @enumFromInt(bi);
            if (split[bi] or join_split[bi]) {
                for (func.blockParams(block)) |p| {
                    if (divergent[@intFromEnum(p)]) continue;
                    divergent[@intFromEnum(p)] = true;
                    changed = true;
                }
            }
            for (func.blockInsts(block)) |inst| {
                const res = func.instResult(inst) orelse continue;
                if (divergent[@intFromEnum(res)]) continue;
                if (!split[bi] and !operandsDiverge(func, divergent, func.opcode(inst))) continue;
                divergent[@intFromEnum(res)] = true;
                changed = true;
            }
        }

        // Data across edges: a divergent argument makes the block parameter it feeds divergent.
        for (0..n) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockInsts(block)) |inst| {
                switch (func.opcode(inst)) {
                    .@"if" => |cf| {
                        propagateEdge(func, divergent, cf.then, &changed);
                        propagateEdge(func, divergent, cf.@"else", &changed);
                    },
                    .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .select => {},
                    .struct_new, .extract, .convert, .unary, .alloca => {},
                    .call, .call_indirect, .global_addr, .load, .store, .prefetch => {},
                    .va_start, .va_arg, .va_end, .dot, .matmul, .barrier, .atomic_rmw => {},
                }
            }
            if (func.terminator(block)) |term| switch (term) {
                .jump => |j| propagateEdge(func, divergent, j, &changed),
                .ret => {},
            };
        }
    }

    return .{ .divergent = divergent, .split = split };
}

/// Carry divergence from a jump's arguments into the target block's parameters.
fn propagateEdge(func: *const Function, divergent: []bool, jump: ir.function.Jump, changed: *bool) void {
    const args = func.blockArgs(jump);
    const params = func.blockParams(jump.target);
    const count = @min(args.len, params.len);
    for (args[0..count], params[0..count]) |a, p| {
        if (!divergent[@intFromEnum(a)]) continue;
        if (divergent[@intFromEnum(p)]) continue;
        divergent[@intFromEnum(p)] = true;
        changed.* = true;
    }
}

/// Mark every block that runs between the branch at `from` and its join at `join`. These are the
/// blocks a thread can run while another thread of the same workgroup runs a different one. The
/// join itself is NOT in the region: the threads are back together there. A null `join` means
/// the two sides never meet again, so everything the branch can reach is in the region.
fn markRegion(
    allocator: std.mem.Allocator,
    cfg: *const cfg_mod.Cfg,
    from: usize,
    join: ?u32,
    split: []bool,
    seen: []bool,
    changed: *bool,
) Error!void {
    @memset(seen, false);
    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(allocator);

    for (cfg.successors(from)) |s| {
        if (join != null and s == join.?) continue;
        if (seen[s]) continue;
        seen[s] = true;
        try stack.append(allocator, s);
    }
    while (stack.pop()) |b| {
        if (!split[b]) {
            split[b] = true;
            changed.* = true;
        }
        for (cfg.successors(b)) |s| {
            if (join != null and s == join.?) continue;
            if (seen[s]) continue;
            seen[s] = true;
            try stack.append(allocator, s);
        }
    }
}

const testing = std.testing;

/// The signed 32-bit type every test below builds its values from.
fn i32Type(func: *Function) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}

test "a kernel parameter is uniform and thread_id_x is divergent" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const entry = try func.appendBlock();
    const param = try func.appendBlockParam(entry, t);
    const tid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(param) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isUniform(param));
    try testing.expect(uni.isDivergent(tid));
}

test "block_id, block_dim and grid_dim are uniform inside a workgroup" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const entry = try func.appendBlock();
    const bid = try func.appendBlockParam(entry, t);
    const bdim = try func.appendBlockParam(entry, t);
    const gdim = try func.appendBlockParam(entry, t);
    const gid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, bid, .block_id_x);
    try gpu.attrs.setBuiltin(&func, bdim, .block_dim_x);
    try gpu.attrs.setBuiltin(&func, gdim, .grid_dim_y);
    try gpu.attrs.setBuiltin(&func, gid, .global_id_x);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(bid) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isUniform(bid));
    try testing.expect(uni.isUniform(bdim));
    try testing.expect(uni.isUniform(gdim));
    // The negative control on the same shape: the fused global index is NOT uniform, so this
    // test cannot pass by reporting everything uniform.
    try testing.expect(uni.isDivergent(gid));
}

test "arithmetic keeps a uniform input uniform and carries a divergent input through" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, t);
    const tid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const k = try func.appendInst(entry, t, .{ .iconst = 7 });
    const uni_sum = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = k } });
    const div_sum = try func.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = tid } });
    const div_imm = try func.appendArithImm(entry, t, .mul, div_sum, 4);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(uni_sum) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isUniform(k));
    try testing.expect(uni.isUniform(uni_sum));
    try testing.expect(uni.isDivergent(div_sum));
    try testing.expect(uni.isDivergent(div_imm));
}

test "a load is divergent even from a uniform address" {
    // The deliberate conservative choice. Nothing here proves that no divergent thread wrote to
    // the address, so the analysis refuses to call the result uniform.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendInst(entry, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(v) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isUniform(p)); // the ADDRESS is uniform
    try testing.expect(uni.isDivergent(v)); // the loaded value is not
}

test "a value becomes divergent through CONTROL alone, with every operand uniform" {
    // The sync-dependence case. Both arms compute a constant, and a constant is uniform, so
    // operand propagation alone would call the merge parameter uniform. Only some threads run
    // each arm, so it is not.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const r = try func.appendBlockParam(merge, t);
    const four = try func.appendInst(entry, t, .{ .iconst = 4 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = four } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    const one = try func.appendInst(then_b, t, .{ .iconst = 1 });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{one}) } });
    const two = try func.appendInst(else_b, t, .{ .iconst = 2 });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{two}) } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(r) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isDivergent(c));
    try testing.expect(uni.blockIsSplit(@intFromEnum(then_b)));
    try testing.expect(uni.blockIsSplit(@intFromEnum(else_b)));
    try testing.expect(!uni.blockIsSplit(@intFromEnum(merge))); // the threads are back together
    try testing.expect(uni.isDivergent(one)); // defined under a divergent branch
    try testing.expect(uni.isDivergent(r)); // and so is the merge parameter
    // The negative control: the entry block runs with the workgroup whole, and the constant it
    // defines stays uniform. Reporting everything divergent fails here.
    try testing.expect(!uni.blockIsSplit(@intFromEnum(entry)));
    try testing.expect(uni.isUniform(four));
}

test "a divergent jump argument makes the block parameter it feeds divergent" {
    // The data half across an edge, on its own. One block jumps to the next and hands it the
    // thread index. Nothing is split here, and the target defines nothing, so only the argument
    // itself carries the divergence into the parameter.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const entry = try func.appendBlock();
    const next = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    const keep = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const carried = try func.appendBlockParam(next, t);
    const plain = try func.appendBlockParam(next, t);
    func.setTerminator(entry, .{ .jump = .{ .target = next, .args = try func.internValues(&.{ tid, keep }) } });
    func.setTerminator(next, .{ .ret = ir.function.Ret.one(carried) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isDivergent(carried));
    // The negative control on the same edge: the other argument is a kernel parameter, so the
    // parameter it feeds stays uniform. An edge that taints everything fails here.
    try testing.expect(uni.isUniform(plain));
}

test "a merge parameter fed by two HOISTED uniform constants is still divergent" {
    // The join half of the sync dependence, on its own. Both constants sit in the entry block,
    // which runs with the workgroup whole, so neither of them is divergent and neither arm
    // defines anything. Only the arrival at the merge from two different sides makes the merge
    // parameter divergent. Without the join taint every value here reads as uniform.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const r = try func.appendBlockParam(merge, t);
    const four = try func.appendInst(entry, t, .{ .iconst = 4 });
    const one = try func.appendInst(entry, t, .{ .iconst = 1 }); // hoisted out of the arms
    const two = try func.appendInst(entry, t, .{ .iconst = 2 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = four } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{one}) } });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{two}) } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(r) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isUniform(one)); // both arguments really are uniform
    try testing.expect(uni.isUniform(two));
    try testing.expect(uni.isDivergent(r)); // and the merge parameter still is not
}

test "a uniform branch taints nothing" {
    // The negative control for the control-flow half. The same diamond with the condition built
    // from kernel parameters instead of a thread index. Nothing is split, and the merge
    // parameter stays uniform.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
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
    const one = try func.appendInst(then_b, t, .{ .iconst = 1 });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{one}) } });
    const two = try func.appendInst(else_b, t, .{ .iconst = 2 });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValues(&.{two}) } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(r) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isUniform(c));
    try testing.expect(!uni.blockIsSplit(@intFromEnum(then_b)));
    try testing.expect(uni.isUniform(r));
}

test "a loop counter against a kernel parameter stays uniform" {
    // The tiled-matmul trip count. The counter is a loop-header parameter fed by a constant on
    // the way in and by its own increment on the back edge, so the fixpoint has to settle on
    // uniform instead of chasing itself into divergent.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const tiles = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(head, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    func.setTerminator(entry, .{ .jump = .{ .target = head, .args = try func.internValues(&.{zero}) } });
    const c = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = tiles } });
    try func.appendIf(head, c, .{ .target = body, .args = &.{} }, .{ .target = done, .args = &.{} });
    const next = try func.appendArithImm(body, t, .add, i, 1);
    func.setTerminator(body, .{ .jump = .{ .target = head, .args = try func.internValues(&.{next}) } });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(tiles) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isUniform(i));
    try testing.expect(uni.isUniform(c));
    try testing.expect(!uni.blockIsSplit(@intFromEnum(body)));
}

test "a loop counter against thread_id_x is divergent, and so is the loop body" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const i = try func.appendBlockParam(head, t);
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    func.setTerminator(entry, .{ .jump = .{ .target = head, .args = try func.internValues(&.{zero}) } });
    const c = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = tid } });
    try func.appendIf(head, c, .{ .target = body, .args = &.{} }, .{ .target = done, .args = &.{} });
    const next = try func.appendArithImm(body, t, .add, i, 1);
    func.setTerminator(body, .{ .jump = .{ .target = head, .args = try func.internValues(&.{next}) } });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(tid) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isDivergent(c));
    try testing.expect(uni.blockIsSplit(@intFromEnum(body)));
    try testing.expect(uni.isDivergent(i)); // the header parameter, through the region
    try testing.expect(uni.isDivergent(next));
}

test "a nested uniform branch inside a divergent arm is still split" {
    // Transitivity. The inner condition is a pair of kernel parameters, so the inner branch is
    // uniform by itself. The threads that reach it are already split by the outer branch, so
    // the inner arms run in some threads and not in others all the same.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const outer = try func.appendBlock();
    const inner = try func.appendBlock();
    const inner_join = try func.appendBlock();
    const merge = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    const a = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const four = try func.appendInst(entry, t, .{ .iconst = 4 });
    const oc = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = four } });
    try func.appendIf(entry, oc, .{ .target = outer, .args = &.{} }, .{ .target = merge, .args = &.{} });
    const ic = try func.appendInst(outer, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = four } });
    try func.appendIf(outer, ic, .{ .target = inner, .args = &.{} }, .{ .target = inner_join, .args = &.{} });
    const one = try func.appendInst(inner, t, .{ .iconst = 1 });
    try func.setJump(inner, inner_join, &.{});
    try func.setJump(inner_join, merge, &.{});
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(a) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.blockIsSplit(@intFromEnum(outer)));
    try testing.expect(uni.blockIsSplit(@intFromEnum(inner)));
    try testing.expect(uni.blockIsSplit(@intFromEnum(inner_join)));
    try testing.expect(uni.isDivergent(one));
    // The negative control: the entry runs whole and `a` is a kernel parameter.
    try testing.expect(!uni.blockIsSplit(@intFromEnum(entry)));
    try testing.expect(uni.isUniform(a));
}

test "a branch whose arms never meet again taints everything it reaches" {
    // No join, so `immediatePostDominator` reports null. The region has to run to the end of the
    // function rather than being cut at block 0.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const left = try func.appendBlock();
    const right = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const four = try func.appendInst(entry, t, .{ .iconst = 4 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = four } });
    try func.appendIf(entry, c, .{ .target = left, .args = &.{} }, .{ .target = right, .args = &.{} });
    const l = try func.appendInst(left, t, .{ .iconst = 1 });
    func.setTerminator(left, .{ .ret = ir.function.Ret.one(l) });
    const r = try func.appendInst(right, t, .{ .iconst = 2 });
    func.setTerminator(right, .{ .ret = ir.function.Ret.one(r) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.blockIsSplit(@intFromEnum(left)));
    try testing.expect(uni.blockIsSplit(@intFromEnum(right)));
    try testing.expect(uni.isDivergent(l));
    try testing.expect(uni.isDivergent(r));
}

test "a one-way if splits nothing" {
    // Both edges land on the same block, so no thread can take a different way, whatever the
    // condition holds.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const after = try func.appendBlock();
    const tid = try func.appendBlockParam(entry, t);
    try gpu.attrs.setBuiltin(&func, tid, .thread_id_x);
    const four = try func.appendInst(entry, t, .{ .iconst = 4 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = four } });
    try func.appendIf(entry, c, .{ .target = after, .args = &.{} }, .{ .target = after, .args = &.{} });
    const one = try func.appendInst(after, t, .{ .iconst = 1 });
    func.setTerminator(after, .{ .ret = ir.function.Ret.one(one) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expectEqual(@as(?ir.function.If, null), twoWayIf(&func, @intFromEnum(entry)));
    try testing.expect(!uni.blockIsSplit(@intFromEnum(after)));
    try testing.expect(uni.isUniform(one));
}

test "an alloca address and a call result are divergent" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Type(&func);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, t);
    const slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = t } });
    const ret = try func.appendCall(entry, t, "f", &.{p});
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(ret) });

    var uni = try analyze(allocator, &func);
    defer uni.deinit(allocator);
    try testing.expect(uni.isDivergent(slot));
    try testing.expect(uni.isDivergent(ret));
    try testing.expect(uni.isUniform(p)); // the negative control on the same function
}
