//! Immediate-form canonicalization: an `arith` with one integer-constant operand is rewritten into
//! the `arith_imm` form (the constant carried in the instruction, not in a separate `iconst`). This
//! lets the backend fold a constant address offset straight into a load or store (`ldr [base, #off]`
//! instead of materializing the offset and adding it), and lets it pick an immediate-form arithmetic
//! instruction. The rewrite is a pure representation change: `arith_imm{op, x, c}` computes exactly
//! what `arith{op, x, (iconst c)}` did. The now-orphaned `iconst` is left for DCE.
//!
//! Only add/sub and the bitwise and shift ops are canonicalized. mul/div/rem are left to the strength
//! pass, which already reads both forms. A commutative op (add, and, or, xor) accepts the constant on
//! either side; a non-commutative op (sub, shl, shr) only when the constant is the RIGHT operand,
//! since the immediate form fixes it there. An instruction with two constant operands is skipped, so
//! constant folding collapses it to a single `iconst` instead.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const dce = @import("dce.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const BinOp = ir.function.BinOp;

pub const pass_def = pass.Pass{ .name = "immform", .run = run };

/// Whether `op` has an `arith_imm` form this pass emits, and whether it is commutative (so the
/// constant may sit on the left). Only `add` is canonicalized: it is the op the backend folds into a
/// load or store address offset, the whole point of this pass. The other ops (sub, bitwise, shift)
/// gain nothing from the `arith_imm` form here (the backend materializes their immediate either way)
/// and canonicalizing them only disturbs folds that read the `arith` form, so they are left alone.
fn immOp(op: BinOp) ?struct { commutative: bool } {
    return switch (op) {
        .add => .{ .commutative = true },
        else => null,
    };
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;

    // Map each value defined by an `iconst` to its constant. Iterate LIVE block instructions, not the
    // pool, so a dropped instruction is not re-processed (matching the other passes).
    var consts = try allocator.alloc(?i64, func.valueCount());
    defer allocator.free(consts);
    @memset(consts, null);
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .iconst) {
                if (func.instResult(inst)) |r| consts[@intFromEnum(r)] = func.opcode(inst).iconst;
            }
        }
    }

    // Only a SINGLE-use constant is folded (a constant shared by several `arith` would be
    // re-materialized inside each `arith_imm`, a loss over the one shared register the `iconst` gives).
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    @memset(uses, 0);
    dce.countUses(func, uses);

    // Only an `add` whose result is used as a load or store ADDRESS is folded. There the backend
    // folds the offset straight into the memory access (`ldr [base, #off]`), a clear win. A plain
    // `x + c` gains nothing from `arith_imm` (the backend materializes the immediate either way) and
    // canonicalizing it only disturbs other folds that read the `arith` form, so it is left alone.
    var is_addr = try allocator.alloc(bool, func.valueCount());
    defer allocator.free(is_addr);
    @memset(is_addr, false);
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .load => |l| is_addr[@intFromEnum(l.ptr)] = true,
                .store => |st| is_addr[@intFromEnum(st.ptr)] = true,
                else => {},
            }
        }
    }

    var changed = false;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            const a = switch (func.opcode(inst)) {
                .arith => |a| a,
                else => continue,
            };
            const info = immOp(a.op) orelse continue;
            const result = func.instResult(inst) orelse continue;
            if (!is_addr[@intFromEnum(result)]) continue; // not an address base: leave it as `arith`
            const lc = consts[@intFromEnum(a.lhs)];
            const rc = consts[@intFromEnum(a.rhs)];
            if (lc != null and rc != null) continue; // both constant: leave it for constant folding
            if (rc) |c| {
                if (uses[@intFromEnum(a.rhs)] != 1) continue;
                func.opcodeMut(inst).* = .{ .arith_imm = .{ .op = a.op, .lhs = a.lhs, .imm = c } };
                changed = true;
            } else if (info.commutative) if (lc) |c| {
                if (uses[@intFromEnum(a.lhs)] != 1) continue;
                func.opcodeMut(inst).* = .{ .arith_imm = .{ .op = a.op, .lhs = a.rhs, .imm = c } };
                changed = true;
            };
        }
    }
    return changed;
}

test "an address add of a single-use constant becomes arith_imm; a plain add is left alone" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const x = try func.appendBlockParam(b, i32_t);
    // An ADDRESS add: base + 40, its result used as a load pointer -> canonicalized.
    const off = try func.appendInst(b, i32_t, .{ .iconst = 40 });
    const addr = try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = off } });
    const loaded = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr } });
    // A PLAIN add: x + 7, not an address -> left as `arith`.
    const seven = try func.appendInst(b, i32_t, .{ .iconst = 7 });
    const plain = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = seven } });
    const sum = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = loaded, .rhs = plain } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));

    // The address add carries its offset inline now.
    const a2 = func.opcode(func.definingInst(addr).?).arith_imm;
    try std.testing.expectEqual(@as(i64, 40), a2.imm);
    try std.testing.expectEqual(base, a2.lhs);
    // The plain add is untouched (not an address base).
    try std.testing.expect(func.opcode(func.definingInst(plain).?) == .arith);
}

test {
    std.testing.refAllDecls(@This());
}
