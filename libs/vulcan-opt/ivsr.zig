//! Induction-variable strength reduction.
//!
//! A BASIC induction variable is a loop header block parameter that the back edge advances by a
//! loop-invariant amount: `k` in `for (k = 0; k < n; k += 1)`. A DERIVED value is any value the
//! loop computes as `k*a + b`, where `a` and `b` are loop-invariant. This pass gives each derived
//! value its own induction variable, so the loop advances it by `a*step` on the back edge instead
//! of rebuilding it from `k` on every trip.
//!
//! The motivating shape is a naive matmul inner loop. `acc += a[row*k + p] * b[p*n + col]`
//! recomputes `p*n` and both addresses on every trip. After this pass the loop carries one pointer
//! per operand and advances each by a constant, which is what a good GPU compiler emits.
//!
//! INTEGER SEMANTICS. Every integer operation in this IR wraps at its declared width: there is no
//! trap, no poison, and no undefined overflow (`constfold` and `expand.evalConst` both model an
//! `arith` with `+%`, `-%` and `*%` and then truncate to the width). So the arithmetic is the ring
//! Z/2^W, in which `k -> a*k + b` is a well-defined function. The recurrence
//! `d[0] = a*k[0] + b`, `d[i+1] = d[i] + a*step` gives `d[i] = a*(k[0] + i*step) + b = a*k[i] + b`
//! in that same ring, because `k[i+1] = k[i] + step` is the same wrapping add. The rewrite is
//! therefore EXACT at every iteration, wrapping included, and needs no no-overflow precondition.
//! This holds only while the derived value's width equals the basic variable's width, which
//! `affineArith` and `affineArithImm` check directly (`la.scale_ty != ty`) rather than assume.
//!
//! `mul` and `shl` distribute over the affine form for the same reason: `(a*k + b) * m` is
//! `(a*m)*k + (b*m)` in Z/2^W, and `x << s` is `x * 2^s` in Z/2^W for `0 <= s < W`.
//!
//! POINTERS. `arith add ptr, int` is address arithmetic whose integer operand may be narrower than
//! the pointer (`verify.pointerArith` allows it), and the IR states no extension rule for that
//! operand. A pointer induction variable advances in pointer width, so it agrees with the original
//! address only while the integer offset stays inside its own type. This pass does not try to prove
//! that. Instead it reduces a pointer ONLY when every use of it is an address the loop
//! dereferences, so an offset that did wrap would have made the original program read or write an
//! address outside the object it indexes. See `pointerUsesAreAddresses`.
//!
//! The address space rides along with the type: every emitted pointer instruction takes its result
//! type from the value it replaces, and the invariant base pointer keeps its own type, so
//! `verify.pointerArithChangesSpace` sees the same space on both sides.
//!
//! This pass adds no block and moves no block, so the dominance-respecting block order the machine
//! backends need is exactly the order it was given (see `blocklayout`). Each new value is either a
//! header block parameter, which dominates the whole loop, or an instruction in the preheader or
//! the latch, which is where the value it feeds is read.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const loops_mod = @import("loops.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;
const BinOp = ir.function.BinOp;
const Type = ir.types.Type;

pub const pass_def = pass.Pass{ .name = "ivsr", .run = run };

/// An invariant integer expression, built during analysis and emitted into the preheader only for
/// the reductions this pass keeps. `binary` operands index the pool that holds the node.
const Node = struct {
    ty: Type,
    kind: union(enum) {
        konst: i64,
        value: Value,
        binary: struct { op: BinOp, lhs: u32, rhs: u32 },
    },
};

/// The affine form of a value: `value == base + (scale*biv + offset)`, with `base` absent for an
/// integer value. `scale` and `offset` are pool indices of invariant integer expressions, both at
/// `scale_ty`, which is also the basic variable's type.
const Affine = struct {
    /// Which header block parameter is the basic induction variable, by index.
    biv: u32,
    scale: u32,
    offset: u32,
    scale_ty: Type,
    /// The loop-invariant pointer this value walks, for a pointer-typed value.
    base: ?Value,
};

/// One value this pass turns into its own induction variable.
const Reduction = struct {
    /// The value the loop recomputes today.
    old: Value,
    ty: Type,
    /// The pool index of `scale*biv_init + offset`, this variable's value on entry.
    init: u32,
    /// The pool index of `scale*biv_step`, the amount the back edge adds.
    step: u32,
    /// The invariant pointer base, for a pointer-typed reduction.
    base: ?Value,
    /// The header parameter that replaces `old`. Filled in when the rewrite runs.
    param: Value = undefined,
};

/// The expression pool of one loop's analysis.
const Pool = struct {
    nodes: std.ArrayList(Node),
    /// The Value each already-emitted node produced, so a shared subexpression is emitted once.
    emitted: std.ArrayList(?Value),

    fn deinit(self: *Pool, allocator: std.mem.Allocator) void {
        self.nodes.deinit(allocator);
        self.emitted.deinit(allocator);
    }

    fn add(self: *Pool, allocator: std.mem.Allocator, node: Node) std.mem.Allocator.Error!u32 {
        try self.nodes.append(allocator, node);
        try self.emitted.append(allocator, null);
        return @intCast(self.nodes.items.len - 1);
    }

    fn konst(self: *Pool, allocator: std.mem.Allocator, ty: Type, c: i64) std.mem.Allocator.Error!u32 {
        return self.add(allocator, .{ .ty = ty, .kind = .{ .konst = c } });
    }

    fn value(self: *Pool, allocator: std.mem.Allocator, ty: Type, v: Value) std.mem.Allocator.Error!u32 {
        return self.add(allocator, .{ .ty = ty, .kind = .{ .value = v } });
    }

    /// The constant a node holds, or null when it is not a folded constant.
    fn constOf(self: *const Pool, index: u32) ?i64 {
        return switch (self.nodes.items[index].kind) {
            .konst => |c| c,
            .value, .binary => null,
        };
    }
};

/// Build `lhs <op> rhs`, folding when both sides are constants and dropping the identity cases.
/// Both operands must already share `ty`.
fn binary(allocator: std.mem.Allocator, func: *const Function, pool: *Pool, ty: Type, op: BinOp, lhs: u32, rhs: u32) std.mem.Allocator.Error!?u32 {
    const info = intInfo(func, ty) orelse return null;
    if (pool.constOf(lhs)) |l| {
        if (pool.constOf(rhs)) |r| {
            const folded = foldBin(op, l, r, info) orelse return null;
            return try pool.konst(allocator, ty, folded);
        }
    }
    // Identities that keep the pool (and the emitted preheader code) small.
    if (pool.constOf(rhs)) |r| switch (op) {
        .add, .sub, .shl => if (r == 0) return lhs,
        .mul => {
            if (r == 1) return lhs;
            if (r == 0) return try pool.konst(allocator, ty, 0);
        },
        .div, .rem, .bit_and, .bit_or, .bit_xor, .shr, .mulh => {},
    };
    if (pool.constOf(lhs)) |l| switch (op) {
        .add => if (l == 0) return rhs,
        .mul => {
            if (l == 1) return rhs;
            if (l == 0) return try pool.konst(allocator, ty, 0);
        },
        .sub, .div, .rem, .bit_and, .bit_or, .bit_xor, .shl, .shr, .mulh => {},
    };
    return try pool.add(allocator, .{ .ty = ty, .kind = .{ .binary = .{ .op = op, .lhs = lhs, .rhs = rhs } } });
}

const IntInfo = struct { bits: u16, signedness: std.builtin.Signedness };

fn intInfo(func: *const Function, ty: Type) ?IntInfo {
    return switch (func.types.type_kind(ty)) {
        .int => |i| .{ .bits = i.bits, .signedness = i.signedness },
        else => null,
    };
}

/// Re-read a W-bit result as a canonical i64, the way a W-bit register value reads back. Mirrors
/// `expand.wrapTo`, which is the same model `constfold` uses.
fn wrapTo(v: i64, info: IntInfo) i64 {
    if (info.bits >= 64) return v;
    const mask: u64 = (@as(u64, 1) << @intCast(info.bits)) - 1;
    const low: u64 = @as(u64, @bitCast(v)) & mask;
    return switch (info.signedness) {
        .unsigned => @bitCast(low),
        .signed => blk: {
            const sign = @as(u64, 1) << @intCast(info.bits - 1);
            break :blk @bitCast(if (low & sign != 0) low | ~mask else low);
        },
    };
}

/// Fold `l <op> r` at `info`'s width, or null when this pass does not model the operation.
fn foldBin(op: BinOp, l: i64, r: i64, info: IntInfo) ?i64 {
    const raw: i64 = switch (op) {
        .add => l +% r,
        .sub => l -% r,
        .mul => l *% r,
        .shl => blk: {
            if (r < 0 or r >= info.bits) return null; // out of range: not modelled, refuse
            break :blk l << @intCast(r);
        },
        .div, .rem, .bit_and, .bit_or, .bit_xor, .shr, .mulh => return null,
    };
    return wrapTo(raw, info);
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const info = try analyses.loops();
    if (info.loops.len == 0) return false;

    var changed = false;
    for (info.loops) |*loop| {
        if (try reduceLoop(allocator, func, loop)) changed = true;
    }
    return changed;
}

/// Everything one loop's analysis needs to answer questions about its values.
const LoopCtx = struct {
    func: *const Function,
    loop: *const loops_mod.Loop,
    header: Block,
    latch: Block,
    preheader: Block,
    /// The block that defines each value.
    def_block: []u32,
    /// The header parameter each in-loop value carries, when it carries one unchanged.
    alias: []?Value,
    /// The affine form of each in-loop value, when it has one.
    affine: []?Affine,
    /// How many uses of each value this pass cannot rewrite away by reducing its consumer.
    demanding_uses: []u32,
    /// How many uses of each value are an address a load, store or prefetch reads.
    address_uses: []u32,
    /// Total uses of each value, over the whole function.
    total_uses: []u32,
    /// The preheader argument of each header parameter the loop carries around unchanged. Such a
    /// parameter holds one value for the whole loop, so it is loop-invariant, but it is not
    /// READABLE outside the loop. The preheader argument holds the same value and is, so every
    /// expression this pass builds names that instead. See `fillSubstitutions`.
    substitute: []?Value,

    /// Whether `v` holds one value for the whole loop.
    fn invariant(self: *const LoopCtx, v: Value) bool {
        if (self.substitute[@intFromEnum(v)] != null) return true;
        return !self.loop.contains(self.def_block[@intFromEnum(v)]);
    }

    /// The value to name in the preheader when an expression reads `v`.
    fn outsideName(self: *const LoopCtx, v: Value) Value {
        return self.substitute[@intFromEnum(v)] orelse v;
    }
};

/// Reduce one loop. Returns whether it changed the function.
fn reduceLoop(allocator: std.mem.Allocator, func: *Function, loop: *const loops_mod.Loop) pass.Error!bool {
    const preheader_index = loop.preheader orelse return false;
    const latch_index = findLatch(func, loop) orelse return false;

    const value_count = func.valueCount();
    const def_block = try allocator.alloc(u32, value_count);
    defer allocator.free(def_block);
    const alias = try allocator.alloc(?Value, value_count);
    defer allocator.free(alias);
    const affine = try allocator.alloc(?Affine, value_count);
    defer allocator.free(affine);
    const demanding_uses = try allocator.alloc(u32, value_count);
    defer allocator.free(demanding_uses);
    const address_uses = try allocator.alloc(u32, value_count);
    defer allocator.free(address_uses);
    const total_uses = try allocator.alloc(u32, value_count);
    defer allocator.free(total_uses);
    const substitute = try allocator.alloc(?Value, value_count);
    defer allocator.free(substitute);
    @memset(substitute, null);

    var ctx = LoopCtx{
        .func = func,
        .loop = loop,
        .header = @enumFromInt(loop.header),
        .latch = @enumFromInt(latch_index),
        .preheader = @enumFromInt(preheader_index),
        .def_block = def_block,
        .alias = alias,
        .affine = affine,
        .demanding_uses = demanding_uses,
        .address_uses = address_uses,
        .total_uses = total_uses,
        .substitute = substitute,
    };
    fillDefBlocks(func, def_block);

    var pool = Pool{ .nodes = .empty, .emitted = .empty };
    defer pool.deinit(allocator);
    var plan: std.ArrayList(Reduction) = .empty;
    defer plan.deinit(allocator);

    try analyzeLoop(allocator, &ctx, &pool, &plan);
    if (plan.items.len == 0) return false;

    try applyPlan(allocator, func, &ctx, &pool, plan.items);
    return true;
}

/// The one block inside the loop whose terminator jumps back to the header, or null when the loop
/// does not have exactly one such block. A loop whose back edge comes from an `if` is refused: the
/// increment this pass appends goes at the end of the latch's instruction list, which is past the
/// point an `if` leaves the block.
fn findLatch(func: *const Function, loop: *const loops_mod.Loop) ?u32 {
    var latch: ?u32 = null;
    for (0..func.blockCount()) |bi| {
        if (!loop.contains(bi)) continue;
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .@"if" => |cf| {
                if (@intFromEnum(cf.then.target) == loop.header) return null;
                if (@intFromEnum(cf.@"else".target) == loop.header) return null;
            },
            else => {},
        };
        const term = func.terminator(block) orelse continue;
        switch (term) {
            .jump => |j| if (@intFromEnum(j.target) == loop.header) {
                if (latch != null) return null; // more than one back edge
                latch = @intCast(bi);
            },
            .ret => {},
        }
    }
    const found = latch orelse return null;
    // The increment goes at the end of the latch, so the latch must not leave early.
    for (func.blockInsts(@enumFromInt(found))) |inst| {
        if (func.opcode(inst) == .@"if") return null;
    }
    return found;
}

fn fillDefBlocks(func: *const Function, def_block: []u32) void {
    // A value with no live definition (one an earlier rewrite orphaned) reads as entry-defined,
    // which makes it invariant, never an induction variable. Same default `licm` uses.
    @memset(def_block, 0);
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| def_block[@intFromEnum(p)] = @intCast(bi);
        for (func.blockInsts(block)) |inst| {
            if (func.instResult(inst)) |r| def_block[@intFromEnum(r)] = @intCast(bi);
        }
    }
}

/// The pool node for a loop-invariant value read at `ty`. A value an `iconst` defines becomes a
/// folded constant, which is what collapses `0 * n` in a variable's initial value and `scale * 1`
/// in its step, so a unit-step counter costs no preheader arithmetic at all.
fn termOf(allocator: std.mem.Allocator, ctx: *const LoopCtx, pool: *Pool, ty: Type, v: Value) pass.Error!u32 {
    const named = ctx.outsideName(v);
    if (ctx.func.valueType(named) == ty) {
        if (ctx.func.definingInst(named)) |inst| {
            if (ctx.func.opcode(inst) == .iconst) return pool.konst(allocator, ty, ctx.func.opcode(inst).iconst);
        }
    }
    return pool.value(allocator, ty, named);
}

/// Record every header parameter the loop carries around unchanged, against the preheader argument
/// that holds the same value.
///
/// An outer index is exactly this from the inner loop's side: the j-loop of a matmul nest hands `i`
/// to the p-loop's header, and the p-loop's back edge hands the same `i` straight back. Its value
/// never changes inside the p-loop, so `i*k` is invariant there, but `i` itself is defined by the
/// p-header and cannot be read from the preheader. The preheader argument can, and it is the same
/// value, because a natural loop has one entry and this parameter is unchanged on the one path from
/// the header back to itself.
fn fillSubstitutions(ctx: *LoopCtx, header_params: []const Value, latch_args: []const Value, pre_args: []const Value) void {
    for (header_params, 0..) |hp, i| {
        const carried = ctx.alias[@intFromEnum(latch_args[i])] orelse continue;
        if (carried != hp) continue;
        ctx.substitute[@intFromEnum(hp)] = pre_args[i];
    }
    // Every value that carries such a parameter unchanged holds the same one value too, so it gets
    // the same substitute. A loop body that re-declares the header's parameters, which is what the
    // `if` at the end of a header produces, reads the ALIAS and never the parameter itself, so
    // without this the whole rule would reach nothing in the shape it exists for.
    for (0..ctx.alias.len) |vi| {
        const carried = ctx.alias[vi] orelse continue;
        if (ctx.substitute[@intFromEnum(carried)]) |outside| ctx.substitute[vi] = outside;
    }
}

/// Find the reducible values of one loop and append a `Reduction` for each.
fn analyzeLoop(allocator: std.mem.Allocator, ctx: *LoopCtx, pool: *Pool, plan: *std.ArrayList(Reduction)) pass.Error!void {
    const func = ctx.func;
    try fillAliases(allocator, ctx);

    @memset(ctx.affine, null);
    const header_params = func.blockParams(ctx.header);
    const latch_args = func.blockArgs(func.terminator(ctx.latch).?.jump);
    const pre_args = func.blockArgs(func.terminator(ctx.preheader).?.jump);
    if (latch_args.len != header_params.len or pre_args.len != header_params.len) return;
    fillSubstitutions(ctx, header_params, latch_args, pre_args);

    // Seed the basic induction variables, and the values that carry one unchanged.
    var any_biv = false;
    for (header_params, 0..) |hp, i| {
        const ty = func.valueType(hp);
        if (intInfo(func, ty) == null) continue; // an integer counter only
        if (stepOf(ctx, hp, latch_args[i]) == null) continue;
        const one = try pool.konst(allocator, ty, 1);
        const zero = try pool.konst(allocator, ty, 0);
        const rep = Affine{ .biv = @intCast(i), .scale = one, .offset = zero, .scale_ty = ty, .base = null };
        for (0..ctx.alias.len) |vi| {
            if (ctx.alias[vi]) |a| {
                if (a == hp) ctx.affine[vi] = rep;
            }
        }
        any_biv = true;
    }
    if (!any_biv) return;

    // Propagate the affine form forward through the loop body. One pass in block order reaches
    // every straight-line chain; the loop repeats until nothing new appears, so a body whose
    // blocks are not in dependency order converges too.
    var again = true;
    while (again) {
        again = false;
        for (0..func.blockCount()) |bi| {
            if (!ctx.loop.contains(bi)) continue;
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                const result = func.instResult(inst) orelse continue;
                if (ctx.affine[@intFromEnum(result)] != null) continue;
                const rep = try affineOf(allocator, ctx, pool, inst, result) orelse continue;
                ctx.affine[@intFromEnum(result)] = rep;
                again = true;
            }
        }
    }

    try countUses(allocator, ctx);

    for (0..func.blockCount()) |bi| {
        if (!ctx.loop.contains(bi)) continue;
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            const result = func.instResult(inst) orelse continue;
            const rep = ctx.affine[@intFromEnum(result)] orelse continue;
            if (!worthReducing(ctx, pool, result, rep)) continue;

            const ty = func.valueType(result);
            const biv = func.blockParams(ctx.header)[rep.biv];
            const step = stepOf(ctx, biv, latch_args[rep.biv]).?;
            const step_expr = switch (step) {
                .konst => |c| try pool.konst(allocator, rep.scale_ty, c),
                .value => |v| try termOf(allocator, ctx, pool, rep.scale_ty, v),
            };
            const init_index = try termOf(allocator, ctx, pool, rep.scale_ty, pre_args[rep.biv]);
            const scaled_init = try binary(allocator, func, pool, rep.scale_ty, .mul, rep.scale, init_index) orelse continue;
            const init_expr = try binary(allocator, func, pool, rep.scale_ty, .add, scaled_init, rep.offset) orelse continue;
            const delta = try binary(allocator, func, pool, rep.scale_ty, .mul, rep.scale, step_expr) orelse continue;

            try plan.append(allocator, .{
                .old = result,
                .ty = ty,
                .init = init_expr,
                .step = delta,
                .base = rep.base,
            });
        }
    }
}

/// Map each in-loop value that carries a header parameter unchanged to that parameter. Header
/// parameters map to themselves. A natural loop has one entry, so every other loop block takes its
/// parameters only from in-loop predecessors, which is what makes this well defined.
fn fillAliases(allocator: std.mem.Allocator, ctx: *LoopCtx) pass.Error!void {
    const func = ctx.func;
    @memset(ctx.alias, null);
    for (func.blockParams(ctx.header)) |hp| ctx.alias[@intFromEnum(hp)] = hp;

    var edges: std.ArrayList(Edge) = .empty;
    defer edges.deinit(allocator);

    var again = true;
    while (again) {
        again = false;
        for (0..func.blockCount()) |bi| {
            if (!ctx.loop.contains(bi) or bi == ctx.loop.header) continue;
            const block: Block = @enumFromInt(bi);
            const params = func.blockParams(block);
            if (params.len == 0) continue;
            edges.clearRetainingCapacity();
            try collectEdgesInto(allocator, func, block, &edges);

            for (params, 0..) |p, pi| {
                if (ctx.alias[@intFromEnum(p)] != null) continue;
                var merged: ?Value = null;
                var ok = edges.items.len > 0;
                for (edges.items) |edge| {
                    const args = edgeArgs(func, edge);
                    if (pi >= args.len) {
                        ok = false;
                        break;
                    }
                    const cand = ctx.alias[@intFromEnum(args[pi])] orelse {
                        ok = false;
                        break;
                    };
                    if (merged) |m| {
                        if (m != cand) {
                            ok = false;
                            break;
                        }
                    } else merged = cand;
                }
                if (!ok) continue;
                ctx.alias[@intFromEnum(p)] = merged.?;
                again = true;
            }
        }
    }
}

/// Where one control-flow edge is written down, so its arguments can be read and replaced.
const Edge = struct {
    from: Block,
    site: union(enum) {
        /// The block's own terminator jump.
        term,
        /// The `then` edge of an `if` instruction.
        if_then: Inst,
        /// The `else` edge of an `if` instruction.
        if_else: Inst,
    },
};

fn collectEdgesInto(allocator: std.mem.Allocator, func: *const Function, target: Block, out: *std.ArrayList(Edge)) pass.Error!void {
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .@"if" => |cf| {
                if (cf.then.target == target) try out.append(allocator, .{ .from = block, .site = .{ .if_then = inst } });
                if (cf.@"else".target == target) try out.append(allocator, .{ .from = block, .site = .{ .if_else = inst } });
            },
            else => {},
        };
        if (func.terminator(block)) |term| switch (term) {
            .jump => |j| if (j.target == target) try out.append(allocator, .{ .from = block, .site = .term }),
            .ret => {},
        };
    }
}

fn edgeArgs(func: *const Function, edge: Edge) []const Value {
    return switch (edge.site) {
        .term => func.blockArgs(func.terminator(edge.from).?.jump),
        .if_then => |inst| func.blockArgs(func.opcode(inst).@"if".then),
        .if_else => |inst| func.blockArgs(func.opcode(inst).@"if".@"else"),
    };
}

/// The amount the back edge adds to a basic induction variable.
const Step = union(enum) { konst: i64, value: Value };

/// The step of header parameter `hp`, given the value the back edge passes for it, or null when
/// that value is not `hp` plus a loop-invariant amount.
fn stepOf(ctx: *const LoopCtx, hp: Value, next: Value) ?Step {
    const func = ctx.func;
    if (ctx.alias[@intFromEnum(next)] != null) return null; // carried through, never advanced
    const inst = func.definingInst(next) orelse return null;
    if (!ctx.loop.contains(ctx.def_block[@intFromEnum(next)])) return null;
    const carries = struct {
        fn f(c: *const LoopCtx, target: Value, v: Value) bool {
            const a = c.alias[@intFromEnum(v)] orelse return false;
            return a == target;
        }
    }.f;
    return switch (func.opcode(inst)) {
        .arith_imm => |a| switch (a.op) {
            .add => if (carries(ctx, hp, a.lhs)) Step{ .konst = a.imm } else null,
            .sub => if (carries(ctx, hp, a.lhs)) Step{ .konst = -%a.imm } else null,
            .mul, .div, .rem, .bit_and, .bit_or, .bit_xor, .shl, .shr, .mulh => null,
        },
        .arith => |a| switch (a.op) {
            .add => blk: {
                if (carries(ctx, hp, a.lhs) and ctx.invariant(a.rhs)) break :blk Step{ .value = a.rhs };
                if (carries(ctx, hp, a.rhs) and ctx.invariant(a.lhs)) break :blk Step{ .value = a.lhs };
                break :blk null;
            },
            .sub, .mul, .div, .rem, .bit_and, .bit_or, .bit_xor, .shl, .shr, .mulh => null,
        },
        .iconst,
        .fconst,
        .fconst128,
        .icmp,
        .select,
        .struct_new,
        .extract,
        .convert,
        .unary,
        .alloca,
        .call,
        .call_indirect,
        .global_addr,
        .load,
        .store,
        .prefetch,
        .va_start,
        .va_arg,
        .va_end,
        .dot,
        .matmul,
        .barrier,
        .atomic_rmw,
        .@"if",
        => null,
    };
}

/// The affine form of `inst`'s result, or null when the instruction is not an affine step from a
/// value that already has one.
fn affineOf(allocator: std.mem.Allocator, ctx: *LoopCtx, pool: *Pool, inst: Inst, result: Value) pass.Error!?Affine {
    const func = ctx.func;
    const ty = func.valueType(result);
    return switch (func.opcode(inst)) {
        .arith => |a| try affineArith(allocator, ctx, pool, a.op, a.lhs, a.rhs, ty),
        .arith_imm => |a| try affineArithImm(allocator, ctx, pool, a.op, a.lhs, a.imm, ty),
        // Everything else either reads memory, calls out, or is not linear in an induction
        // variable. No `else` prong: a new opcode must be classified deliberately, because an
        // `else` would silently answer "not affine" and hide the decision.
        .iconst,
        .fconst,
        .fconst128,
        .icmp,
        .select,
        .struct_new,
        .extract,
        .convert,
        .unary,
        .alloca,
        .call,
        .call_indirect,
        .global_addr,
        .load,
        .store,
        .prefetch,
        .va_start,
        .va_arg,
        .va_end,
        .dot,
        .matmul,
        .barrier,
        .atomic_rmw,
        .@"if",
        => null,
    };
}

fn affineArith(allocator: std.mem.Allocator, ctx: *LoopCtx, pool: *Pool, op: BinOp, lhs: Value, rhs: Value, ty: Type) pass.Error!?Affine {
    const func = ctx.func;
    const l = ctx.affine[@intFromEnum(lhs)];
    const r = ctx.affine[@intFromEnum(rhs)];
    const l_inv = ctx.invariant(lhs);
    const r_inv = ctx.invariant(rhs);
    const is_ptr = func.types.type_kind(ty) == .ptr;

    if (is_ptr) {
        // A pointer's affine form is born at `invariant_ptr + affine_int` and grows by adding or
        // subtracting an invariant integer. The result type is the value's own pointer type, so
        // the address space rides along untouched.
        if (op != .add and op != .sub) return null;
        if (l) |la| {
            if (la.base != null) {
                // An existing pointer variable stepped by an invariant integer.
                if (!r_inv or func.valueType(rhs) != la.scale_ty) return null;
                const term = try termOf(allocator, ctx, pool, la.scale_ty, rhs);
                const off = try binary(allocator, func, pool, la.scale_ty, op, la.offset, term) orelse return null;
                return Affine{ .biv = la.biv, .scale = la.scale, .offset = off, .scale_ty = la.scale_ty, .base = la.base };
            }
            // `affine_int + invariant_ptr`: the pointer is on the right.
            if (op != .add or !r_inv or func.valueType(rhs) != ty) return null;
            return Affine{ .biv = la.biv, .scale = la.scale, .offset = la.offset, .scale_ty = la.scale_ty, .base = ctx.outsideName(rhs) };
        }
        if (r) |ra| {
            if (ra.base != null) return null; // pointer minus pointer variable: not address arithmetic
            if (op != .add or !l_inv or func.valueType(lhs) != ty) return null;
            return Affine{ .biv = ra.biv, .scale = ra.scale, .offset = ra.offset, .scale_ty = ra.scale_ty, .base = ctx.outsideName(lhs) };
        }
        return null;
    }

    // Integer forms. `verify` requires both operands to share the result type here, so every
    // expression in the chain is computed at one width, which is what makes the rewrite exact.
    if (l) |la| {
        if (la.base != null or la.scale_ty != ty) return null;
        if (r) |ra| {
            if (ra.base != null or ra.scale_ty != ty or ra.biv != la.biv) return null;
            if (op != .add and op != .sub) return null;
            const scale = try binary(allocator, func, pool, ty, op, la.scale, ra.scale) orelse return null;
            const off = try binary(allocator, func, pool, ty, op, la.offset, ra.offset) orelse return null;
            return Affine{ .biv = la.biv, .scale = scale, .offset = off, .scale_ty = ty, .base = null };
        }
        if (!r_inv) return null;
        const term = try termOf(allocator, ctx, pool, ty, rhs);
        switch (op) {
            .add, .sub => {
                const off = try binary(allocator, func, pool, ty, op, la.offset, term) orelse return null;
                return Affine{ .biv = la.biv, .scale = la.scale, .offset = off, .scale_ty = ty, .base = null };
            },
            // Both distribute over the affine form in Z/2^W: see the file comment.
            .mul, .shl => {
                const scale = try binary(allocator, func, pool, ty, op, la.scale, term) orelse return null;
                const off = try binary(allocator, func, pool, ty, op, la.offset, term) orelse return null;
                return Affine{ .biv = la.biv, .scale = scale, .offset = off, .scale_ty = ty, .base = null };
            },
            .div, .rem, .bit_and, .bit_or, .bit_xor, .shr, .mulh => return null,
        }
    }
    if (r) |ra| {
        if (ra.base != null or ra.scale_ty != ty or !l_inv) return null;
        const term = try termOf(allocator, ctx, pool, ty, lhs);
        switch (op) {
            .add => {
                const off = try binary(allocator, func, pool, ty, .add, ra.offset, term) orelse return null;
                return Affine{ .biv = ra.biv, .scale = ra.scale, .offset = off, .scale_ty = ty, .base = null };
            },
            .mul => {
                const scale = try binary(allocator, func, pool, ty, .mul, ra.scale, term) orelse return null;
                const off = try binary(allocator, func, pool, ty, .mul, ra.offset, term) orelse return null;
                return Affine{ .biv = ra.biv, .scale = scale, .offset = off, .scale_ty = ty, .base = null };
            },
            // `inv - affine` negates the scale. `inv << affine` and the rest are not linear.
            .sub => {
                const zero = try pool.konst(allocator, ty, 0);
                const scale = try binary(allocator, func, pool, ty, .sub, zero, ra.scale) orelse return null;
                const off = try binary(allocator, func, pool, ty, .sub, term, ra.offset) orelse return null;
                return Affine{ .biv = ra.biv, .scale = scale, .offset = off, .scale_ty = ty, .base = null };
            },
            .div, .rem, .bit_and, .bit_or, .bit_xor, .shl, .shr, .mulh => return null,
        }
    }
    return null;
}

fn affineArithImm(allocator: std.mem.Allocator, ctx: *LoopCtx, pool: *Pool, op: BinOp, lhs: Value, imm: i64, ty: Type) pass.Error!?Affine {
    const func = ctx.func;
    const la = ctx.affine[@intFromEnum(lhs)] orelse return null;
    const is_ptr = func.types.type_kind(ty) == .ptr;
    if (is_ptr) {
        if (la.base == null) return null; // an integer cannot become a pointer by adding a constant
        if (op != .add and op != .sub) return null;
        const term = try pool.konst(allocator, la.scale_ty, imm);
        const off = try binary(allocator, func, pool, la.scale_ty, op, la.offset, term) orelse return null;
        return Affine{ .biv = la.biv, .scale = la.scale, .offset = off, .scale_ty = la.scale_ty, .base = la.base };
    }
    if (la.base != null or la.scale_ty != ty) return null;
    const term = try pool.konst(allocator, ty, imm);
    switch (op) {
        .add, .sub => {
            const off = try binary(allocator, func, pool, ty, op, la.offset, term) orelse return null;
            return Affine{ .biv = la.biv, .scale = la.scale, .offset = off, .scale_ty = ty, .base = null };
        },
        .mul, .shl => {
            const scale = try binary(allocator, func, pool, ty, op, la.scale, term) orelse return null;
            const off = try binary(allocator, func, pool, ty, op, la.offset, term) orelse return null;
            return Affine{ .biv = la.biv, .scale = scale, .offset = off, .scale_ty = ty, .base = null };
        },
        .div, .rem, .bit_and, .bit_or, .bit_xor, .shr, .mulh => return null,
    }
}

/// Count, for every value, the uses this pass cannot make disappear by reducing the consumer, the
/// uses that are a dereferenced address, and the uses in total.
fn countUses(allocator: std.mem.Allocator, ctx: *LoopCtx) pass.Error!void {
    const func = ctx.func;
    @memset(ctx.demanding_uses, 0);
    @memset(ctx.address_uses, 0);
    @memset(ctx.total_uses, 0);

    var operands: std.ArrayList(Value) = .empty;
    defer operands.deinit(allocator);

    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        const in_loop = ctx.loop.contains(bi);
        for (func.blockInsts(block)) |inst| {
            // A use inside a reducible consumer disappears with that consumer. Any other use has
            // to keep reading a real value, which is what makes this value worth its own variable.
            const consumer_is_affine = in_loop and blk: {
                const r = func.instResult(inst) orelse break :blk false;
                break :blk ctx.affine[@intFromEnum(r)] != null;
            };
            operands.clearRetainingCapacity();
            try appendOperands(allocator, func, inst, &operands);
            for (operands.items) |v| {
                ctx.total_uses[@intFromEnum(v)] += 1;
                if (!consumer_is_affine) ctx.demanding_uses[@intFromEnum(v)] += 1;
            }
            // The three ops that DEREFERENCE an address. `atomic_rmw` also reads and writes its
            // pointer, and is deliberately absent: counting it here would let a pointer an atomic
            // touches become an induction variable, and this pass has no test that runs one.
            // Exhaustive with no `else` prong, like `appendOperands`: a new dereferencing opcode
            // left out only makes `pointerUsesAreAddresses` refuse, which is the safe direction,
            // but the decision belongs in this list rather than in a silent default.
            if (in_loop) switch (func.opcode(inst)) {
                .load => |l| ctx.address_uses[@intFromEnum(l.ptr)] += 1,
                .store => |s| ctx.address_uses[@intFromEnum(s.ptr)] += 1,
                .prefetch => |p| ctx.address_uses[@intFromEnum(p.ptr)] += 1,
                .iconst,
                .fconst,
                .fconst128,
                .arith,
                .arith_imm,
                .icmp,
                .select,
                .struct_new,
                .extract,
                .convert,
                .unary,
                .alloca,
                .call,
                .call_indirect,
                .global_addr,
                .va_start,
                .va_arg,
                .va_end,
                .dot,
                .matmul,
                .barrier,
                .atomic_rmw,
                .@"if",
                => {},
            };
        }
        if (func.terminator(block)) |term| {
            const args: []const Value = switch (term) {
                .jump => |j| func.blockArgs(j),
                .ret => |r| r.slice(),
            };
            for (args) |v| {
                ctx.total_uses[@intFromEnum(v)] += 1;
                ctx.demanding_uses[@intFromEnum(v)] += 1;
            }
        }
    }
}

/// Every value operand of `inst`, in no particular order. Exhaustive with no `else` prong: a new
/// opcode whose operands this misses would leave a use uncounted, and an uncounted use is a value
/// this pass may replace without rewriting its reader.
fn appendOperands(allocator: std.mem.Allocator, func: *const Function, inst: Inst, out: *std.ArrayList(Value)) pass.Error!void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .fconst128, .global_addr, .alloca, .barrier => {},
        .arith => |a| try out.appendSlice(allocator, &.{ a.lhs, a.rhs }),
        .arith_imm => |a| try out.append(allocator, a.lhs),
        .icmp => |c| try out.appendSlice(allocator, &.{ c.lhs, c.rhs }),
        .select => |s| try out.appendSlice(allocator, &.{ s.cond, s.then, s.@"else" }),
        .struct_new => |s| try out.appendSlice(allocator, func.valueList(s.fields)),
        .extract => |e| try out.append(allocator, e.aggregate),
        .convert => |c| try out.append(allocator, c.value),
        .unary => |u| try out.append(allocator, u.value),
        .call => |c| {
            try out.appendSlice(allocator, func.valueList(c.args));
            if (c.ret_dest) |rd| try out.append(allocator, rd);
        },
        .call_indirect => |c| {
            try out.append(allocator, c.target);
            try out.appendSlice(allocator, func.valueList(c.args));
            if (c.ret_dest) |rd| try out.append(allocator, rd);
        },
        .load => |l| try out.append(allocator, l.ptr),
        .store => |s| try out.appendSlice(allocator, &.{ s.value, s.ptr }),
        .prefetch => |p| try out.append(allocator, p.ptr),
        .va_start => |v| try out.append(allocator, v.list),
        .va_arg => |v| try out.append(allocator, v.list),
        .va_end => |v| try out.append(allocator, v.list),
        .dot => |d| try out.appendSlice(allocator, &.{ d.acc, d.a, d.b }),
        .matmul => |m| try out.appendSlice(allocator, &.{ m.a, m.b, m.c }),
        .atomic_rmw => |a| {
            try out.appendSlice(allocator, &.{ a.ptr, a.value });
            if (a.compare) |c| try out.append(allocator, c);
        },
        .@"if" => |cf| {
            try out.append(allocator, cf.cond);
            try out.appendSlice(allocator, func.blockArgs(cf.then));
            try out.appendSlice(allocator, func.blockArgs(cf.@"else"));
        },
    }
}

/// Whether `v` earns its own induction variable.
fn worthReducing(ctx: *const LoopCtx, pool: *const Pool, v: Value, rep: Affine) bool {
    // A zero scale is a loop-invariant value. Hoisting it is `licm`'s job, not this pass's.
    if (pool.constOf(rep.scale)) |s| {
        if (s == 0) return false;
    }
    // An INTEGER value with a unit scale is the basic variable plus an invariant amount, which
    // already costs one add per trip. An induction variable for it costs the same add and adds a
    // live value, so there is nothing to win. This is a COST rule, not a correctness one: reducing
    // such a value computes the same numbers.
    //
    // A POINTER with a unit scale is a different trade. Its address chain is the integer add plus
    // the base add, so one advance per trip replaces two operations, and it is kept.
    if (rep.base == null and pool.constOf(rep.scale) == @as(?i64, 1)) return false;
    // Nothing left that has to read it: reducing it would only add a dead variable.
    if (ctx.demanding_uses[@intFromEnum(v)] == 0) return false;
    // A value the back edge already carries to the header is already a loop recurrence, so a
    // second one only renames it and feeds the old parameter from the new one.
    //
    // This is also what keeps the pass from RUNNING AWAY. The basic variable's own update `k+step`
    // is a linear function of `k`, and the back edge is precisely where it is read: reduce it and
    // the loop gains a second counter feeding the first, then on the pass manager's next sweep a
    // third feeding the second, once per sweep until the iteration cap.
    if (isCarriedBack(ctx, v)) return false;
    if (rep.base != null and !pointerUsesAreAddresses(ctx, v)) return false;
    return true;
}

/// Whether the back edge passes `v` to the header as a block argument.
fn isCarriedBack(ctx: *const LoopCtx, v: Value) bool {
    const args = ctx.func.blockArgs(ctx.func.terminator(ctx.latch).?.jump);
    for (args) |a| {
        if (a == v) return true;
    }
    return false;
}

/// Whether every use of pointer `v` is an address the loop dereferences, or another in-loop address
/// computation that is itself affine.
///
/// This is the one precondition a pointer induction variable needs. The new variable advances in
/// POINTER width, while the address it replaces was built by adding an integer offset that may be
/// narrower (see the file comment). The two agree while that offset stays inside its own type. This
/// does not prove that; it requires instead that the loop only ever DEREFERENCES the address, so an
/// offset that wrapped would already have made the original program touch memory outside the object
/// it indexes. A pointer that escapes the loop, is compared, or is stored somewhere is refused,
/// because for those the wrapped value is an observable result rather than a bad access.
fn pointerUsesAreAddresses(ctx: *const LoopCtx, v: Value) bool {
    const i = @intFromEnum(v);
    // EVERY use, not just the demanding ones: a use inside another address computation would
    // carry a wrapped offset onward, and that computation may itself be refused.
    return ctx.address_uses[i] > 0 and ctx.total_uses[i] == ctx.address_uses[i];
}

/// Emit the plan: the initial values and the steps into the preheader, one new header parameter per
/// reduction, and one advance per reduction into the latch.
fn applyPlan(allocator: std.mem.Allocator, func: *Function, ctx: *LoopCtx, pool: *Pool, plan: []Reduction) pass.Error!void {
    // Preheader first: the initial value and the step of every reduction are loop-invariant, and
    // the preheader dominates the header, so this is where they can be read from.
    var inits = try allocator.alloc(Value, plan.len);
    defer allocator.free(inits);
    var steps = try allocator.alloc(?Value, plan.len);
    defer allocator.free(steps);
    var step_consts = try allocator.alloc(?i64, plan.len);
    defer allocator.free(step_consts);

    for (plan, 0..) |*r, i| {
        const offset = try emit(allocator, func, ctx.preheader, pool, r.init);
        inits[i] = if (r.base) |base| try addToPointer(func, ctx.preheader, r.ty, base, offset, pool.constOf(r.init)) else offset;
        step_consts[i] = pool.constOf(r.step);
        steps[i] = if (step_consts[i] == null) try emit(allocator, func, ctx.preheader, pool, r.step) else null;
    }

    // One new header parameter per reduction, then the old value's readers move to it. The header
    // dominates every block of the loop, so the parameter is in scope wherever the old value was.
    for (plan) |*r| r.param = try func.appendBlockParam(ctx.header, r.ty);
    for (plan) |*r| func.replaceAllUses(r.old, r.param);

    // The advance, at the end of the latch, and the two edges into the header.
    var next_values = try allocator.alloc(Value, plan.len);
    defer allocator.free(next_values);
    for (plan, 0..) |*r, i| {
        next_values[i] = if (step_consts[i]) |c|
            try func.appendArithImm(ctx.latch, r.ty, .add, r.param, c)
        else
            try func.appendInst(ctx.latch, r.ty, .{ .arith = .{ .op = .add, .lhs = r.param, .rhs = steps[i].? } });
    }

    try appendEdgeArgs(allocator, func, ctx.preheader, inits);
    try appendEdgeArgs(allocator, func, ctx.latch, next_values);
}

/// `base + offset` at `ty`, using the immediate form when the offset is a known constant. The
/// result takes `ty` from the value being replaced, so the pointer keeps its address space.
fn addToPointer(func: *Function, block: Block, ty: Type, base: Value, offset: Value, offset_const: ?i64) pass.Error!Value {
    if (offset_const) |c| {
        if (c == 0) return base;
        return func.appendArithImm(block, ty, .add, base, c);
    }
    return func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
}

/// Materialize an invariant expression at the end of `block`, reusing what is already emitted.
fn emit(allocator: std.mem.Allocator, func: *Function, block: Block, pool: *Pool, index: u32) pass.Error!Value {
    if (pool.emitted.items[index]) |v| return v;
    const node = pool.nodes.items[index];
    const result: Value = switch (node.kind) {
        .konst => |c| try func.appendInst(block, node.ty, .{ .iconst = c }),
        .value => |v| v,
        .binary => |b| blk: {
            const lhs = try emit(allocator, func, block, pool, b.lhs);
            if (pool.constOf(b.rhs)) |c| break :blk try func.appendArithImm(block, node.ty, b.op, lhs, c);
            const rhs = try emit(allocator, func, block, pool, b.rhs);
            break :blk try func.appendInst(block, node.ty, .{ .arith = .{ .op = b.op, .lhs = lhs, .rhs = rhs } });
        },
    };
    pool.emitted.items[index] = result;
    return result;
}

/// Append `extra` to the arguments the terminator of `block` passes.
fn appendEdgeArgs(allocator: std.mem.Allocator, func: *Function, block: Block, extra: []const Value) pass.Error!void {
    const jump = func.terminator(block).?.jump;
    var args: std.ArrayList(Value) = .empty;
    defer args.deinit(allocator);
    try args.appendSlice(allocator, func.blockArgs(jump));
    try args.appendSlice(allocator, extra);
    const list = try func.internValues(args.items);
    func.terminatorPtr(block).*.?.jump.args = list;
}

const testing = std.testing;

/// The blocks of the canonical counted loop the tests below build:
/// `entry -> head; head: if k < bound { body } else { done }; body: ... -> head; done: ret`.
const Counted = struct {
    entry: Block,
    head: Block,
    body: Block,
    done: Block,
    /// The header's induction parameter.
    k: Value,
    /// The header's accumulator parameter.
    acc: Value,
    /// The body's alias of `k`.
    bk: Value,
    /// The body's alias of `acc`.
    bacc: Value,
    i32_t: Type,
};

/// Build `entry(extra...) -> for (k = 0; k < bound; k += 1) { <caller fills body> }`, with an i32
/// accumulator carried alongside so the loop has a result. The caller finishes the body by calling
/// `closeCounted`.
fn buildCounted(func: *Function, bound: Value, i32_t: Type, entry: Block) !Counted {
    const bool_t = try func.types.intern(.bool);
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{ zero, zero });

    const k = try func.appendBlockParam(head, i32_t);
    const acc = try func.appendBlockParam(head, i32_t);
    const lt = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = k, .rhs = bound } });
    try func.appendIf(head, lt, .{ .target = body, .args = &.{ k, acc } }, .{ .target = done, .args = &.{acc} });

    const bk = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const fin = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(fin) });

    return .{ .entry = entry, .head = head, .body = body, .done = done, .k = k, .acc = acc, .bk = bk, .bacc = bacc, .i32_t = i32_t };
}

/// Close the body: `acc' = acc + contribution; k' = k + 1; jump head(k', acc')`.
fn closeCounted(func: *Function, c: Counted, contribution: Value) !void {
    const next_acc = try func.appendInst(c.body, c.i32_t, .{ .arith = .{ .op = .add, .lhs = c.bacc, .rhs = contribution } });
    const next_k = try func.appendArithImm(c.body, c.i32_t, .add, c.bk, 1);
    try func.setJump(c.body, c.head, &.{ next_k, next_acc });
}

fn runPass(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

/// How many `mul` operations, in either form, the block still holds.
fn countMul(func: *const Function, block: Block) usize {
    var n: usize = 0;
    for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
        .arith => |a| if (a.op == .mul) {
            n += 1;
        },
        .arith_imm => |a| if (a.op == .mul) {
            n += 1;
        },
        else => {},
    };
    return n;
}

test "reduces k*n into an induction variable that steps by n" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const c = try buildCounted(&func, n, i32_t, entry);
    // acc += k * n. `n` comes from outside the loop, so the product is a derived variable.
    const prod = try func.appendInst(c.body, i32_t, .{ .arith = .{ .op = .mul, .lhs = c.bk, .rhs = n } });
    try closeCounted(&func, c, prod);

    try testing.expect(try runPass(allocator, &func));

    // The header carries a third value now, and the body's multiply has no reader left.
    try testing.expectEqual(@as(usize, 3), func.blockParams(c.head).len);
    const derived = func.blockParams(c.head)[2];
    for (func.blockInsts(c.body)) |inst| switch (func.opcode(inst)) {
        .arith => |a| try testing.expect(a.lhs != prod and a.rhs != prod),
        else => {},
    };

    // The back edge advances it by `n` itself: one add per trip in place of the multiply.
    const args = func.blockArgs(func.terminator(c.body).?.jump);
    try testing.expectEqual(@as(usize, 3), args.len);
    const step = func.opcode(func.definingInst(args[2]).?).arith;
    try testing.expectEqual(ir.function.BinOp.add, step.op);
    try testing.expectEqual(derived, step.lhs);
    try testing.expectEqual(n, step.rhs);

    // It starts at `0 * n`, which folds to a constant, so the preheader materializes no multiply.
    const init_args = func.blockArgs(func.terminator(c.entry).?.jump);
    try testing.expectEqual(@as(i64, 0), func.opcode(func.definingInst(init_args[2]).?).iconst);
    try testing.expectEqual(@as(usize, 0), countMul(&func, c.entry));

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "reduces an indexed address into a pointer induction variable that steps by the element size" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const c = try buildCounted(&func, n, i32_t, entry);
    // acc += p[k], written the way a frontend writes it: scale the index, then add the base.
    const off = try func.appendArithImm(c.body, i32_t, .mul, c.bk, 4);
    const addr = try func.appendInst(c.body, ptr_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = off } });
    const loaded = try func.appendInst(c.body, i32_t, .{ .load = .{ .ptr = addr } });
    try closeCounted(&func, c, loaded);

    try testing.expect(try runPass(allocator, &func));

    try testing.expectEqual(@as(usize, 3), func.blockParams(c.head).len);
    const walker = func.blockParams(c.head)[2];
    try testing.expectEqual(ptr_t, func.valueType(walker)); // the address space rode along

    // The load reads the walker directly: no scale and no base add left on the path.
    var loads: usize = 0;
    for (func.blockInsts(c.body)) |inst| switch (func.opcode(inst)) {
        .load => |l| {
            loads += 1;
            try testing.expectEqual(walker, l.ptr);
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, 1), loads);

    // One immediate add of the element size per trip, at the pointer's own type.
    const args = func.blockArgs(func.terminator(c.body).?.jump);
    const step_inst = func.definingInst(args[2]).?;
    const step = func.opcode(step_inst).arith_imm;
    try testing.expectEqual(ir.function.BinOp.add, step.op);
    try testing.expectEqual(@as(i64, 4), step.imm);
    try testing.expectEqual(walker, step.lhs);
    try testing.expectEqual(ptr_t, func.valueType(func.instResult(step_inst).?));

    // It starts at `p` itself, since `0*4` is zero: no add in the preheader either.
    const init_args = func.blockArgs(func.terminator(c.entry).?.jump);
    try testing.expectEqual(p, init_args[2]);

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "a shared address space stays shared through the reduction" {
    // `verify.pointerArithChangesSpace` rejects a pointer add whose result names a different space,
    // so a reduction that took its type from anywhere but the value it replaces fails loudly here.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const shared_t = try func.types.intern(.{ .ptr = .shared });
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, shared_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const c = try buildCounted(&func, n, i32_t, entry);
    const off = try func.appendArithImm(c.body, i32_t, .mul, c.bk, 4);
    const addr = try func.appendInst(c.body, shared_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = off } });
    const loaded = try func.appendInst(c.body, i32_t, .{ .load = .{ .ptr = addr } });
    try closeCounted(&func, c, loaded);

    try testing.expect(try runPass(allocator, &func));
    try testing.expectEqual(shared_t, func.valueType(func.blockParams(c.head)[2]));

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "leaves a loop alone when the other multiplier is not loop-invariant" {
    // `k * acc` changes with the accumulator, so it is not `k` times an invariant and there is no
    // induction variable to give it.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const c = try buildCounted(&func, n, i32_t, entry);
    const prod = try func.appendInst(c.body, i32_t, .{ .arith = .{ .op = .mul, .lhs = c.bk, .rhs = c.bacc } });
    try closeCounted(&func, c, prod);

    try testing.expect(!try runPass(allocator, &func));
    try testing.expectEqual(@as(usize, 2), func.blockParams(c.head).len);
    try testing.expectEqual(@as(usize, 1), countMul(&func, c.body));
}

test "leaves a loop with no induction variable alone" {
    // The header parameter is squared on the back edge rather than advanced by a step, so nothing
    // in the loop is a linear function of a counter.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const one = try func.appendInst(entry, i32_t, .{ .iconst = 1 });
    try func.setJump(entry, head, &.{one});
    const k = try func.appendBlockParam(head, i32_t);
    const lt = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = k, .rhs = n } });
    try func.appendIf(head, lt, .{ .target = body, .args = &.{k} }, .{ .target = done, .args = &.{k} });
    const bk = try func.appendBlockParam(body, i32_t);
    const squared = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = bk, .rhs = bk } });
    const scaled = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = squared, .rhs = n } });
    try func.setJump(body, head, &.{scaled});
    const fin = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(fin) });

    try testing.expect(!try runPass(allocator, &func));
    try testing.expectEqual(@as(usize, 1), func.blockParams(head).len);
}

test "leaves a pointer alone when the loop does more with it than dereference it" {
    // The address is also STORED, so its value is an observable result rather than only an access.
    // A pointer variable advances in pointer width, which matches the original address only while
    // the integer offset stays inside its own type, so this shape is refused. See
    // `pointerUsesAreAddresses`.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const sink = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const c = try buildCounted(&func, n, i32_t, entry);
    const off = try func.appendArithImm(c.body, i32_t, .mul, c.bk, 4);
    const addr = try func.appendInst(c.body, ptr_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = off } });
    const loaded = try func.appendInst(c.body, i32_t, .{ .load = .{ .ptr = addr } });
    try func.appendStore(c.body, addr, sink); // the address escapes
    try closeCounted(&func, c, loaded);

    try testing.expect(!try runPass(allocator, &func));
    try testing.expectEqual(@as(usize, 2), func.blockParams(c.head).len);
}

test "does not turn the counter's own update into a second counter" {
    // `k + 1` is a linear function of `k`, and the back edge is where it is read, so reducing it
    // would hand the loop a second counter feeding the first and the next run of the pass would add
    // a third. `isCarriedBack` is the guard that stops that. Running to a fixpoint must settle after
    // the first run, with the header no wider than the one variable the loop earns.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const c = try buildCounted(&func, n, i32_t, entry);
    const prod = try func.appendInst(c.body, i32_t, .{ .arith = .{ .op = .mul, .lhs = c.bk, .rhs = n } });
    try closeCounted(&func, c, prod);

    try testing.expect(try runPass(allocator, &func));
    try testing.expect(!try runPass(allocator, &func)); // a fixpoint after one run
    try testing.expectEqual(@as(usize, 3), func.blockParams(c.head).len);
}

test "reduces the inner loop of a nest whose outer variable feeds the inner bound" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    // entry, outer head, outer body (the inner preheader), inner head, inner body, inner exit
    // (the outer latch), done. Laid out so every block follows the block that dominates it.
    const entry = try func.appendBlock();
    const outer_head = try func.appendBlock();
    const outer_body = try func.appendBlock();
    const inner_head = try func.appendBlock();
    const inner_body = try func.appendBlock();
    const outer_latch = try func.appendBlock();
    const done = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, outer_head, &.{ zero, zero });

    const i = try func.appendBlockParam(outer_head, i32_t);
    const outer_acc = try func.appendBlockParam(outer_head, i32_t);
    const i_lt = try func.appendInst(outer_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(outer_head, i_lt, .{ .target = outer_body, .args = &.{ i, outer_acc } }, .{ .target = done, .args = &.{outer_acc} });

    const oi = try func.appendBlockParam(outer_body, i32_t);
    const oacc = try func.appendBlockParam(outer_body, i32_t);
    const inner_zero = try func.appendInst(outer_body, i32_t, .{ .iconst = 0 });
    try func.setJump(outer_body, inner_head, &.{ inner_zero, oacc });

    const j = try func.appendBlockParam(inner_head, i32_t);
    const inner_acc = try func.appendBlockParam(inner_head, i32_t);
    // The inner bound is the OUTER variable, which is invariant inside the inner loop.
    const j_lt = try func.appendInst(inner_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = j, .rhs = oi } });
    try func.appendIf(inner_head, j_lt, .{ .target = inner_body, .args = &.{ j, inner_acc } }, .{ .target = outer_latch, .args = &.{inner_acc} });

    const bj = try func.appendBlockParam(inner_body, i32_t);
    const bacc = try func.appendBlockParam(inner_body, i32_t);
    const prod = try func.appendInst(inner_body, i32_t, .{ .arith = .{ .op = .mul, .lhs = bj, .rhs = n } });
    const next_acc = try func.appendInst(inner_body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = prod } });
    const next_j = try func.appendArithImm(inner_body, i32_t, .add, bj, 1);
    try func.setJump(inner_body, inner_head, &.{ next_j, next_acc });

    const carried = try func.appendBlockParam(outer_latch, i32_t);
    const next_i = try func.appendArithImm(outer_latch, i32_t, .add, oi, 1);
    try func.setJump(outer_latch, outer_head, &.{ next_i, carried });

    const fin = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(fin) });

    try testing.expect(try runPass(allocator, &func));

    // Only the inner loop earned a variable. The outer loop's own body holds no product to reduce.
    try testing.expectEqual(@as(usize, 3), func.blockParams(inner_head).len);
    try testing.expectEqual(@as(usize, 2), func.blockParams(outer_head).len);
    // The inner variable starts at zero and steps by `n`, once per inner trip.
    const back = func.blockArgs(func.terminator(inner_body).?.jump);
    const step = func.opcode(func.definingInst(back[2]).?).arith;
    try testing.expectEqual(n, step.rhs);
    try testing.expectEqual(func.blockParams(inner_head)[2], step.lhs);

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "gives a shifted index its own variable, since strength reduction turns the scale into a shift" {
    // `strength.zig` runs before this pass and rewrites `k * 4` into `k << 2`, so the shift form is
    // the one this pass actually meets on the matmul path. It is a multiply by 2^s in Z/2^W, so it
    // distributes over the affine form exactly like the multiply does.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const row = try func.appendBlockParam(entry, i32_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const c = try buildCounted(&func, n, i32_t, entry);
    // `p[row + k]`, addressed as `p + ((row + k) << 2)`.
    const idx = try func.appendInst(c.body, i32_t, .{ .arith = .{ .op = .add, .lhs = row, .rhs = c.bk } });
    const off = try func.appendArithImm(c.body, i32_t, .shl, idx, 2);
    const addr = try func.appendInst(c.body, ptr_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = off } });
    const loaded = try func.appendInst(c.body, i32_t, .{ .load = .{ .ptr = addr } });
    try closeCounted(&func, c, loaded);

    try testing.expect(try runPass(allocator, &func));
    const walker = func.blockParams(c.head)[2];
    try testing.expectEqual(ptr_t, func.valueType(walker));
    const back = func.blockArgs(func.terminator(c.body).?.jump);
    const step = func.opcode(func.definingInst(back[2]).?).arith_imm;
    try testing.expectEqual(@as(i64, 4), step.imm); // 1 << 2 bytes per trip

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "leaves a loop with no preheader alone" {
    // Two entries into the header mean there is no single block that runs once before the loop, so
    // there is nowhere to put an initial value.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const other = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const cond = try func.appendBlockParam(entry, bool_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const one = try func.appendInst(entry, i32_t, .{ .iconst = 1 });
    try func.appendIf(entry, cond, .{ .target = head, .args = &.{zero} }, .{ .target = other, .args = &.{} });
    try func.setJump(other, head, &.{one}); // a second entry into the header

    const k = try func.appendBlockParam(head, i32_t);
    const lt = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = k, .rhs = n } });
    try func.appendIf(head, lt, .{ .target = body, .args = &.{k} }, .{ .target = done, .args = &.{k} });
    const bk = try func.appendBlockParam(body, i32_t);
    _ = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = bk, .rhs = n } });
    const next_k = try func.appendArithImm(body, i32_t, .add, bk, 1);
    try func.setJump(body, head, &.{next_k});
    const fin = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(fin) });

    try testing.expect(!try runPass(allocator, &func));
    try testing.expectEqual(@as(usize, 1), func.blockParams(head).len);
}

test "leaves a product the back edge already carries alone" {
    // `k*n` is passed straight to the header as the next value of a second parameter, so the loop
    // already carries it as a recurrence. A variable for it would only feed the old parameter from
    // a new one. This is a COST guard, not a correctness one: reducing this shape computes the same
    // numbers, it just adds a live value and saves nothing.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{ zero, zero });
    const k = try func.appendBlockParam(head, i32_t);
    const previous = try func.appendBlockParam(head, i32_t);
    const lt = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = k, .rhs = n } });
    try func.appendIf(head, lt, .{ .target = body, .args = &.{ k, previous } }, .{ .target = done, .args = &.{previous} });
    const bk = try func.appendBlockParam(body, i32_t);
    _ = try func.appendBlockParam(body, i32_t);
    const prod = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = bk, .rhs = n } });
    const next_k = try func.appendArithImm(body, i32_t, .add, bk, 1);
    try func.setJump(body, head, &.{ next_k, prod }); // the product IS the recurrence
    const fin = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(fin) });

    try testing.expect(!try runPass(allocator, &func));
    try testing.expectEqual(@as(usize, 2), func.blockParams(head).len);
    try testing.expectEqual(@as(usize, 1), countMul(&func, body));
}

test "an outer index the inner loop carries unchanged is treated as invariant" {
    // The inner loop's header carries `row` around untouched, so `row` holds one value for the
    // whole loop even though the header defines it. `fillSubstitutions` is what lets the pass see
    // that and name the preheader's copy instead. Without it, `row + k` is not a linear function of
    // anything and the address stays in the body.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const p = try func.appendBlockParam(entry, ptr_t);
    const row = try func.appendBlockParam(entry, i32_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{ row, zero, zero });

    // The header carries `row` (never changed), the counter, and the accumulator.
    const hrow = try func.appendBlockParam(head, i32_t);
    const k = try func.appendBlockParam(head, i32_t);
    const acc = try func.appendBlockParam(head, i32_t);
    const lt = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = k, .rhs = n } });
    try func.appendIf(head, lt, .{ .target = body, .args = &.{ hrow, k, acc } }, .{ .target = done, .args = &.{acc} });

    const brow = try func.appendBlockParam(body, i32_t);
    const bk = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const idx = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = brow, .rhs = bk } });
    const off = try func.appendArithImm(body, i32_t, .shl, idx, 2);
    const addr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = off } });
    const loaded = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = addr } });
    const next_acc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = loaded } });
    const next_k = try func.appendArithImm(body, i32_t, .add, bk, 1);
    try func.setJump(body, head, &.{ brow, next_k, next_acc });
    const fin = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(fin) });

    try testing.expect(try runPass(allocator, &func));

    // A fourth header parameter walks the row, and it starts from the PREHEADER's copy of `row`,
    // which is the only name for that value outside the loop.
    try testing.expectEqual(@as(usize, 4), func.blockParams(head).len);
    const walker = func.blockParams(head)[3];
    try testing.expectEqual(ptr_t, func.valueType(walker));
    var loads: usize = 0;
    for (func.blockInsts(body)) |inst| switch (func.opcode(inst)) {
        .load => |l| {
            loads += 1;
            try testing.expectEqual(walker, l.ptr);
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, 1), loads);
    const back = func.blockArgs(func.terminator(body).?.jump);
    try testing.expectEqual(@as(i64, 4), func.opcode(func.definingInst(back[3]).?).arith_imm.imm);

    // The preheader forms `p + (row << 2)` from the outer `row`, never from the header parameter.
    var reads_outer = false;
    for (func.blockInsts(entry)) |inst| switch (func.opcode(inst)) {
        .arith_imm => |a| if (a.op == .shl and a.lhs == row) {
            reads_outer = true;
        },
        else => {},
    };
    try testing.expect(reads_outer);

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "the default pipeline leaves the loop shape the microarch recognizers read" {
    // PHASE ORDER, pinned. `loopvec.recognize` refuses a header with more than one parameter,
    // `loopvec.recognizeReduction` and `splitunroll` refuse anything past the induction and the
    // reduction accumulators, and `dotprod` refuses anything but four. This pass adds parameters,
    // so it must not run inside `default_pipeline`, where it would reach a loop before them.
    //
    // The loop below is a map: one header parameter, one indexed load, one indexed store. After
    // the default pipeline the header still carries exactly that one parameter. After the late
    // pipeline it carries the two walking pointers as well.
    const allocator = testing.allocator;
    const root = @import("../vulcan-opt.zig");

    for (root.default_pipeline) |p| try testing.expect(!std.mem.eql(u8, p.name, pass_def.name));
    try testing.expectEqualStrings(pass_def.name, root.late_pipeline[0].name);

    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const src = try func.appendBlockParam(entry, ptr_t);
    const dst = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{zero});
    const k = try func.appendBlockParam(head, i32_t);
    const lt = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = k, .rhs = n } });
    try func.appendIf(head, lt, .{ .target = body, .args = &.{k} }, .{ .target = done });
    const bk = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bk, 4);
    const from = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = src, .rhs = off } });
    const value = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = from } });
    const tripled = try func.appendArithImm(body, i32_t, .mul, value, 3);
    const to = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = dst, .rhs = off } });
    try func.appendStore(body, tripled, to);
    const next_k = try func.appendArithImm(body, i32_t, .add, bk, 1);
    try func.setJump(body, head, &.{next_k});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    _ = try root.optimizeEarly(allocator, &func);
    // The loop still carries one value, so `loopvec.recognize` can still read it.
    try testing.expectEqual(@as(usize, 1), try loopHeaderWidth(allocator, &func));

    _ = try root.optimizeLate(allocator, &func);
    // And the late pipeline does reach the loop: the header carries the walking pointers now.
    try testing.expect(try loopHeaderWidth(allocator, &func) > 1);
}

/// How many values the only loop of `func` carries. The count `loopvec`, `splitunroll` and
/// `dotprod` each match on exactly.
fn loopHeaderWidth(allocator: std.mem.Allocator, func: *const Function) !usize {
    var info = try loops_mod.analyze(allocator, func);
    defer info.deinit(allocator);
    try testing.expectEqual(@as(usize, 1), info.loops.len);
    return func.blockParams(@enumFromInt(info.loops[0].header)).len;
}

test "leaves an integer offset of the counter alone, and still walks a pointer by one" {
    // COST rule, both halves of it. `k + n` at integer type already costs one add per trip, so a
    // variable for it wins nothing and is refused. The pointer `p + k` in the same loop is the
    // other side: its address takes an integer add AND a base add, so one advance per trip does
    // replace two operations, and it is reduced even though the scale is also one.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const c = try buildCounted(&func, n, i32_t, entry);
    const shifted = try func.appendInst(c.body, i32_t, .{ .arith = .{ .op = .add, .lhs = c.bk, .rhs = n } });
    const addr = try func.appendInst(c.body, ptr_t, .{ .arith = .{ .op = .add, .lhs = p, .rhs = c.bk } });
    const loaded = try func.appendInst(c.body, i32_t, .{ .load = .{ .ptr = addr } });
    const sum = try func.appendInst(c.body, i32_t, .{ .arith = .{ .op = .mul, .lhs = shifted, .rhs = loaded } });
    try closeCounted(&func, c, sum);

    try testing.expect(try runPass(allocator, &func));

    // Exactly one new value: the walking pointer. `k + n` is still computed in the body.
    try testing.expectEqual(@as(usize, 3), func.blockParams(c.head).len);
    try testing.expectEqual(ptr_t, func.valueType(func.blockParams(c.head)[2]));
    var keeps_offset = false;
    for (func.blockInsts(c.body)) |inst| switch (func.opcode(inst)) {
        .arith => |a| if (a.op == .add and a.lhs == c.bk and a.rhs == n) {
            keeps_offset = true;
        },
        else => {},
    };
    try testing.expect(keeps_offset);

    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}
