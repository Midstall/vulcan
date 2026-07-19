//! Shared, target-independent address-mode-folding analysis. Recognizes when a load or
//! store's pointer is a foldable `arith_imm.add(base, imm)` (a base register plus a constant
//! byte offset) and computes which of those adds become dead once every one of their uses is
//! folded away. Pure analysis: no IR is rewritten here. Each backend later consumes this
//! analysis to emit a base+offset addressing mode directly and to drop the now-dead add.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;

/// A folded address: a base value plus a constant byte displacement.
pub const Addr = struct { base: Value, off: i64 };

/// The result of folding one function: which mem ops folded, and which adds died as a result.
pub const Analysis = struct {
    /// The load/store instructions that folded, and the base+offset they fold to.
    folds: std.AutoHashMapUnmanaged(Inst, Addr),
    /// `arith_imm.add` instructions whose result is used only by folded mem ops (dead once
    /// the fold is applied and the backend stops reading the add's result).
    dead_adds: std.AutoHashMapUnmanaged(Inst, void),

    /// A no-fold analysis: nothing folds, no add is dead. `baseOf` returns the raw ptr, `offOf`
    /// returns 0, and `isDeadAdd` is false for every instruction. Backends thread this through the
    /// fold-agnostic paths (a Wimmer differential compile, a liveness debug hook) so those paths are
    /// byte-identical to before folding existed. Holds no allocation, so it never needs `deinit`.
    pub const empty: Analysis = .{ .folds = .empty, .dead_adds = .empty };

    pub fn deinit(self: *Analysis, allocator: std.mem.Allocator) void {
        self.folds.deinit(allocator);
        self.dead_adds.deinit(allocator);
    }

    /// The pointer VALUE the mem op addresses after folding: the fold base if folded, else the
    /// raw ptr operand.
    pub fn baseOf(self: *const Analysis, func: *const Function, mem_inst: Inst) Value {
        if (self.folds.get(mem_inst)) |addr| return addr.base;
        return rawPtr(func, mem_inst);
    }

    /// The byte displacement: the fold offset if folded, else 0.
    pub fn offOf(self: *const Analysis, mem_inst: Inst) i64 {
        if (self.folds.get(mem_inst)) |addr| return addr.off;
        return 0;
    }

    /// Whether `inst` (an `arith_imm.add`) is dead because every one of its uses folded away.
    pub fn isDeadAdd(self: *const Analysis, inst: Inst) bool {
        return self.dead_adds.contains(inst);
    }
};

/// The raw, unfolded pointer operand of a load or store. Asserts on any other opcode: callers
/// only ever pass a mem_inst that came from `folds`/a load/store scan.
fn rawPtr(func: *const Function, mem_inst: Inst) Value {
    return switch (func.opcode(mem_inst)) {
        .load => |l| l.ptr,
        .store => |st| st.ptr,
        else => unreachable, // mem_inst is always a load or store: the only ops that can fold
    };
}

/// Build the address-fold analysis for `func`. `foldOffset` is the target predicate: given a
/// load/store instruction whose pointer is defined by an `arith_imm.add`, it returns the byte
/// offset to fold (always equal to the add's imm) if that imm is within the target's addressing
/// range for the op's access size, else null. This function does the "ptr defined by
/// arith_imm.add" recognition and base extraction; `foldOffset` only judges size and range.
pub fn analyze(
    allocator: std.mem.Allocator,
    func: *const Function,
    ctx: anytype,
    comptime foldOffset: fn (@TypeOf(ctx), *const Function, Inst) ?i64,
) std.mem.Allocator.Error!Analysis {
    var folds: std.AutoHashMapUnmanaged(Inst, Addr) = .empty;
    errdefer folds.deinit(allocator);

    const block_count = func.blockCount();
    var bi: usize = 0;
    while (bi < block_count) : (bi += 1) {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            const ptr: Value = switch (func.opcode(inst)) {
                .load => |l| l.ptr,
                .store => |st| st.ptr,
                else => continue,
            };
            const def = func.definingInst(ptr) orelse continue;
            const add = switch (func.opcode(def)) {
                .arith_imm => |a| a,
                else => continue,
            };
            if (add.op != .add) continue;
            const off = foldOffset(ctx, func, inst) orelse continue;
            try folds.put(allocator, inst, .{ .base = add.lhs, .off = off });
        }
    }

    const value_count = func.valueCount();
    const total_uses = try allocator.alloc(u32, value_count);
    defer allocator.free(total_uses);
    const folded_ptr_uses = try allocator.alloc(u32, value_count);
    defer allocator.free(folded_ptr_uses);
    @memset(total_uses, 0);
    @memset(folded_ptr_uses, 0);

    countUses(func, total_uses);

    var fold_it = folds.iterator();
    while (fold_it.next()) |entry| {
        const mem_inst = entry.key_ptr.*;
        const p = rawPtr(func, mem_inst);
        folded_ptr_uses[@intFromEnum(p)] += 1;
    }

    var dead_adds: std.AutoHashMapUnmanaged(Inst, void) = .empty;
    errdefer dead_adds.deinit(allocator);

    bi = 0;
    while (bi < block_count) : (bi += 1) {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            const add = switch (func.opcode(inst)) {
                .arith_imm => |a| a,
                else => continue,
            };
            if (add.op != .add) continue;
            // An arith_imm always defines a result, so a missing one is a programmer error.
            const result = func.instResult(inst) orelse unreachable;
            const rv = @intFromEnum(result);
            if (total_uses[rv] > 0 and total_uses[rv] == folded_ptr_uses[rv]) {
                try dead_adds.put(allocator, inst, {});
            }
        }
    }

    return .{ .folds = folds, .dead_adds = dead_adds };
}

/// Count every use of every value across the whole function: every Value-carrying operand of
/// every instruction (exhaustive over every `Opcode` tag) plus every terminator edge. Mirrors
/// `libs/vulcan-opt/dce.zig`'s `countUses`, the reference exhaustive whole-function use walk.
fn countUses(func: *const Function, uses: []u32) void {
    const block_count = func.blockCount();
    var bi: usize = 0;
    while (bi < block_count) : (bi += 1) {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .iconst, .fconst, .alloca, .global_addr => {},
                .arith => |a| {
                    uses[@intFromEnum(a.lhs)] += 1;
                    uses[@intFromEnum(a.rhs)] += 1;
                },
                .arith_imm => |a| uses[@intFromEnum(a.lhs)] += 1,
                .icmp => |c| {
                    uses[@intFromEnum(c.lhs)] += 1;
                    uses[@intFromEnum(c.rhs)] += 1;
                },
                .select => |s| {
                    uses[@intFromEnum(s.cond)] += 1;
                    uses[@intFromEnum(s.then)] += 1;
                    uses[@intFromEnum(s.@"else")] += 1;
                },
                .extract => |e| uses[@intFromEnum(e.aggregate)] += 1,
                .convert => |cv| uses[@intFromEnum(cv.value)] += 1,
                .unary => |u| uses[@intFromEnum(u.value)] += 1,
                .load => |l| uses[@intFromEnum(l.ptr)] += 1,
                .store => |st| {
                    uses[@intFromEnum(st.value)] += 1;
                    uses[@intFromEnum(st.ptr)] += 1;
                },
                .prefetch => |pf| uses[@intFromEnum(pf.ptr)] += 1,
                .dot => |d| {
                    uses[@intFromEnum(d.acc)] += 1;
                    uses[@intFromEnum(d.a)] += 1;
                    uses[@intFromEnum(d.b)] += 1;
                },
                .matmul => |mm| {
                    uses[@intFromEnum(mm.a)] += 1;
                    uses[@intFromEnum(mm.b)] += 1;
                    uses[@intFromEnum(mm.c)] += 1;
                },
                .struct_new => |sn| for (func.valueList(sn.fields)) |f| {
                    uses[@intFromEnum(f)] += 1;
                },
                .call => |c| for (func.valueList(c.args)) |arg| {
                    uses[@intFromEnum(arg)] += 1;
                },
                .call_indirect => |c| {
                    uses[@intFromEnum(c.target)] += 1;
                    for (func.valueList(c.args)) |arg| uses[@intFromEnum(arg)] += 1;
                },
                .@"if" => |cf| {
                    uses[@intFromEnum(cf.cond)] += 1;
                    for (func.blockArgs(cf.then)) |arg| uses[@intFromEnum(arg)] += 1;
                    for (func.blockArgs(cf.@"else")) |arg| uses[@intFromEnum(arg)] += 1;
                },
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .ret => |v| if (v) |vv| {
                uses[@intFromEnum(vv)] += 1;
            },
            .jump => |j| for (func.blockArgs(j)) |arg| {
                uses[@intFromEnum(arg)] += 1;
            },
        };
    }
}

// Test scaffolding: a trivial ctx and a foldOffset predicate that folds any imm in
// `[0, 100]` that is a multiple of 4, so tests can exercise both in-range and out-of-range imms.

const TestCtx = struct {};

fn testFoldOffset(ctx: TestCtx, func: *const Function, mem_inst: Inst) ?i64 {
    _ = ctx;
    const ptr = rawPtr(func, mem_inst);
    const def = func.definingInst(ptr) orelse unreachable; // caller already checked
    const add = switch (func.opcode(def)) {
        .arith_imm => |a| a,
        else => unreachable,
    };
    if (@rem(add.imm, 4) != 0) return null;
    if (add.imm < 0 or add.imm > 100) return null;
    return add.imm;
}

test "recognizes a load whose ptr is arith_imm.add and records base+off" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const p = try func.appendArithImm(b, ptr_t, .add, base, 8);
    const load = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = load });
    const load_inst = func.definingInst(load).?;

    var analysis = try analyze(allocator, &func, TestCtx{}, testFoldOffset);
    defer analysis.deinit(allocator);

    const addr = analysis.folds.get(load_inst).?;
    try std.testing.expectEqual(base, addr.base);
    try std.testing.expectEqual(@as(i64, 8), addr.off);
    try std.testing.expectEqual(base, analysis.baseOf(&func, load_inst));
    try std.testing.expectEqual(@as(i64, 8), analysis.offOf(load_inst));
}

test "a load whose ptr is a block param does not fold" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const load = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = base } });
    func.setTerminator(b, .{ .ret = load });
    const load_inst = func.definingInst(load).?;

    var analysis = try analyze(allocator, &func, TestCtx{}, testFoldOffset);
    defer analysis.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), analysis.folds.count());
    try std.testing.expectEqual(base, analysis.baseOf(&func, load_inst));
    try std.testing.expectEqual(@as(i64, 0), analysis.offOf(load_inst));
}

test "a load whose ptr is a reg+reg arith (not arith_imm) does not fold" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const idx = try func.appendBlockParam(b, ptr_t);
    const p = try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = idx } });
    const load = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = load });

    var analysis = try analyze(allocator, &func, TestCtx{}, testFoldOffset);
    defer analysis.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), analysis.folds.count());
}

test "a load whose ptr is arith_imm.sub does not fold" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const p = try func.appendArithImm(b, ptr_t, .sub, base, 8);
    const load = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = load });

    var analysis = try analyze(allocator, &func, TestCtx{}, testFoldOffset);
    defer analysis.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), analysis.folds.count());
}

test "foldOffset returning null (out of range imm) leaves the op unfolded" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    // 200 is out of the test predicate's [0, 100] range.
    const p = try func.appendArithImm(b, ptr_t, .add, base, 200);
    const load = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = load });
    const load_inst = func.definingInst(load).?;

    var analysis = try analyze(allocator, &func, TestCtx{}, testFoldOffset);
    defer analysis.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), analysis.folds.count());
    // Unfolded: baseOf returns the raw (unfolded) ptr operand, the add's own result `p`, not
    // the add's lhs `base`.
    try std.testing.expectEqual(p, analysis.baseOf(&func, load_inst));
    try std.testing.expectEqual(@as(i64, 0), analysis.offOf(load_inst));
}

test "baseOf and offOf return the raw ptr and 0 for an unfolded op" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, i32_t);
    // A store whose ptr is a raw block param, never even reaching arith_imm recognition.
    try func.appendStore(b, v, base);
    func.setTerminator(b, .{ .ret = null });
    const store_inst = func.blockInsts(b)[0];

    var analysis = try analyze(allocator, &func, TestCtx{}, testFoldOffset);
    defer analysis.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), analysis.folds.count());
    try std.testing.expectEqual(base, analysis.baseOf(&func, store_inst));
    try std.testing.expectEqual(@as(i64, 0), analysis.offOf(store_inst));
}

test "an add used only by a folded load is dead" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const p = try func.appendArithImm(b, ptr_t, .add, base, 8);
    const load = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = load });
    const add_inst = func.definingInst(p).?;

    var analysis = try analyze(allocator, &func, TestCtx{}, testFoldOffset);
    defer analysis.deinit(allocator);

    try std.testing.expect(analysis.isDeadAdd(add_inst));
}

test "an add used by a folded load AND a ret is not dead" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const p = try func.appendArithImm(b, ptr_t, .add, base, 8);
    const load = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = p } });
    _ = load;
    // p is also returned directly, so a use of it survives the fold and it stays live.
    func.setTerminator(b, .{ .ret = p });
    const add_inst = func.definingInst(p).?;

    var analysis = try analyze(allocator, &func, TestCtx{}, testFoldOffset);
    defer analysis.deinit(allocator);

    try std.testing.expect(!analysis.isDeadAdd(add_inst));
}

test "cross-block: an add in the entry block feeding a load in a successor folds and the add is dead" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.intern(.ptr);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const b_block = try func.appendBlock();
    const q = try func.appendBlockParam(entry, ptr_t);
    // p is computed in entry and used directly in b_block: entry strictly dominates b_block, so
    // this is a legal direct (non-block-param) cross-block SSA use in this IR.
    const p = try func.appendArithImm(entry, ptr_t, .add, q, 8);
    try func.setJump(entry, b_block, &.{});
    const load = try func.appendInst(b_block, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b_block, .{ .ret = load });
    const load_inst = func.definingInst(load).?;
    const add_inst = func.definingInst(p).?;

    var analysis = try analyze(allocator, &func, TestCtx{}, testFoldOffset);
    defer analysis.deinit(allocator);

    try std.testing.expect(analysis.folds.contains(load_inst));
    try std.testing.expect(analysis.isDeadAdd(add_inst));
}
