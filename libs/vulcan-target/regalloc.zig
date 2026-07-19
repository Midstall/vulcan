//! Shared register-allocation support: live-interval computation over a Vulcan IR
//! function. The analysis is target-independent (it reads the IR, not any encoder),
//! so backends share it. Each backend keeps its own register assignment (the pool,
//! ABI pins, and any fixed-register constraints are target-specific).
//!
//! An interval is `[def_pos, last_use]` over a block-order linearization, extended
//! by a backward liveness pass so a value live across a loop keeps its register for
//! the whole loop body. A linear scan over the returned (start-sorted) intervals,
//! freeing a register when its interval ends, gives reuse.

const std = @import("std");
const ir = @import("vulcan-ir");
const addrfold = @import("addrfold.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;

pub const Error = std.mem.Allocator.Error;

pub const Interval = struct { value: Value, start: u32, end: u32 };

pub fn lessByStart(_: void, a: Interval, b: Interval) bool {
    return a.start < b.start;
}

/// Compute one live interval per value, sorted by start position. Caller owns the
/// returned slice.
///
/// `fold` is an optional address-mode-fold analysis. When null (every legacy caller) the output
/// is byte-identical to before folding existed: no use is rerouted and no value is excluded. When
/// non-null, a folded load/store attributes its POINTER use to the fold base (so the base stays
/// live to the mem op, even across a block), and the dead-add result values (their only uses folded
/// away) are EXCLUDED from the returned intervals. A dead-add result would otherwise be an
/// end-before-start interval (its uses rerouted off it), so dropping it is both correct and
/// necessary. Both consumers iterate the returned array and map `interval.value -> reg`, so a
/// shorter (filtered) array allocates the right registers and never resurrects a dead add.
pub fn computeLiveIntervals(allocator: std.mem.Allocator, func: *const Function, fold: ?*const addrfold.Analysis) Error![]Interval {
    const nval = func.valueCount();
    const nblocks = func.blockCount();

    const def_pos = try allocator.alloc(u32, nval);
    defer allocator.free(def_pos);
    const last_use = try allocator.alloc(u32, nval);
    defer allocator.free(last_use);
    const block_end = try allocator.alloc(u32, nblocks);
    defer allocator.free(block_end);
    @memset(def_pos, 0);
    @memset(last_use, 0);

    var pos: u32 = 0;
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| {
            def_pos[@intFromEnum(p)] = pos;
            last_use[@intFromEnum(p)] = pos;
        }
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            forEachUse(func, inst, last_use, pos, fold);
            if (func.instResult(inst)) |r| def_pos[@intFromEnum(r)] = pos;
            pos += 1;
        }
        block_end[bi] = pos;
        if (func.terminator(block)) |term| forEachTermUse(func, term, last_use, pos);
        pos += 1;
    }
    try extendLiveRanges(allocator, func, last_use, block_end, fold);

    // Count the surviving values first so the returned slice is exactly its own length (the caller
    // frees the returned slice, so it must not be a subslice of a larger allocation). With fold null
    // every value survives, so `nlive == nval` and the fill below is byte-identical index order.
    var nlive: usize = nval;
    if (fold) |fa| {
        nlive = 0;
        for (0..nval) |i| {
            if (isDeadAddResult(func, fa, @enumFromInt(i))) continue;
            nlive += 1;
        }
    }
    const ivals = try allocator.alloc(Interval, nlive);
    errdefer allocator.free(ivals);
    var w: usize = 0;
    for (0..nval) |i| {
        if (fold) |fa| if (isDeadAddResult(func, fa, @enumFromInt(i))) continue;
        ivals[w] = .{ .value = @enumFromInt(i), .start = def_pos[i], .end = last_use[i] };
        w += 1;
    }
    std.debug.assert(w == nlive);
    std.mem.sort(Interval, ivals, {}, lessByStart);
    return ivals;
}

/// Whether `v` is the result of an `arith_imm.add` that the fold marked dead (its only uses folded
/// away). A block param has no defining inst, so it never qualifies.
fn isDeadAddResult(func: *const Function, fold: *const addrfold.Analysis, v: Value) bool {
    const def = func.definingInst(v) orelse return false;
    return fold.isDeadAdd(def);
}

fn markUse(last_use: []u32, v: Value, pos: u32) void {
    if (pos > last_use[@intFromEnum(v)]) last_use[@intFromEnum(v)] = pos;
}

fn forEachUse(func: *const Function, inst: ir.function.Inst, last_use: []u32, pos: u32, fold: ?*const addrfold.Analysis) void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .alloca, .global_addr => {},
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
        // A folded load/store attributes its pointer use to the fold base, so the base's live range
        // reaches the mem op (even cross-block). `baseOf` returns the raw ptr when unfolded, so the
        // fold-null path (no fold call at all) is byte-identical.
        .load => |l| markUse(last_use, if (fold) |fa| fa.baseOf(func, inst) else l.ptr, pos),
        .store => |st| {
            markUse(last_use, st.value, pos);
            markUse(last_use, if (fold) |fa| fa.baseOf(func, inst) else st.ptr, pos);
        },
        .prefetch => |pf| markUse(last_use, pf.ptr, pos),
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

fn forEachTermUse(func: *const Function, term: ir.function.Terminator, last_use: []u32, pos: u32) void {
    switch (term) {
        .ret => |v| if (v) |vv| markUse(last_use, vv, pos),
        .jump => |j| for (func.blockArgs(j)) |a| markUse(last_use, a, pos),
    }
}

fn setUsed(row: []bool, v: Value) void {
    row[@intFromEnum(v)] = true;
}

fn markUsedBitset(func: *const Function, inst: ir.function.Inst, row: []bool, fold: ?*const addrfold.Analysis) void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .alloca, .global_addr => {},
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
        // Same fold reroute as `forEachUse`, in the backward liveness fixpoint (`extendLiveRanges`):
        // a folded mem op's pointer use is the fold base, so the base's cross-block liveness reaches
        // the mem op. `baseOf` is the raw ptr when unfolded, so the fold-null path is byte-identical.
        .load => |l| setUsed(row, if (fold) |fa| fa.baseOf(func, inst) else l.ptr),
        .store => |st| {
            setUsed(row, st.value);
            setUsed(row, if (fold) |fa| fa.baseOf(func, inst) else st.ptr);
        },
        .prefetch => |pf| setUsed(row, pf.ptr),
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

fn markUsedTermBitset(func: *const Function, term: ir.function.Terminator, row: []bool) void {
    switch (term) {
        .ret => |v| if (v) |vv| setUsed(row, vv),
        .jump => |j| for (func.blockArgs(j)) |a| setUsed(row, a),
    }
}

/// Backward liveness dataflow. Extends `last_use[v]` to the end of every block
/// where `v` is live-out, so a value live across a loop keeps its register.
fn extendLiveRanges(allocator: std.mem.Allocator, func: *const Function, last_use: []u32, block_end: []const u32, fold: ?*const addrfold.Analysis) Error!void {
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
            markUsedBitset(func, inst, row, fold);
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
