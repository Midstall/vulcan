//! Lower integer `div`/`rem` to a sequence of supported operations, for targets
//! with no hardware integer divide (NVIDIA GPUs). Target-independent IR-to-IR pass
//! run before isel. Backends with a divide instruction (RISC-V, AArch64) skip it.
//!
//! The core is a 32-step restoring division on unsigned 32-bit values, unrolled
//! inline with no control flow (each step uses `select` instead of a branch). A
//! signed `div`/`rem` divides the magnitudes and applies the sign afterward. The
//! expansion replaces the instruction in place. Later uses of its result are
//! rewritten to the computed quotient or remainder.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Type = ir.types.Type;
const BinOp = ir.function.BinOp;

pub const Error = std.mem.Allocator.Error;

/// Lower every integer `div`/`rem` in `func`. Returns true if anything changed.
pub fn run(allocator: std.mem.Allocator, func: *Function) Error!bool {
    if (!hasDivision(func)) return false;

    const u32t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    // The substitution from each lowered div/rem result to its replacement value.
    var subst = std.AutoHashMapUnmanaged(Value, Value){};
    defer subst.deinit(allocator);

    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        if (!blockHasDivision(func, block)) continue;

        // Rebuild the block's instruction list, expanding div/rem in place. Existing
        // instructions keep their handles. Appended expansion instructions land in
        // order because `appendInst` pushes onto the (now-cleared) list.
        var original = std.ArrayList(Inst).empty;
        defer original.deinit(allocator);
        try original.appendSlice(allocator, func.blockInstsMut(block).items);
        func.blockInstsMut(block).clearRetainingCapacity();

        for (original.items) |inst| {
            const op = func.opcode(inst);
            const is_div = op == .arith and (op.arith.op == .div or op.arith.op == .rem);
            if (!is_div) {
                try func.blockInstsMut(block).append(allocator, inst);
                continue;
            }
            const a = op.arith;
            const result = func.instResult(inst).?;
            const want_rem = a.op == .rem;
            const signed = isSigned(func, a.lhs);
            const ctx = Ctx{ .func = func, .block = block, .u32t = u32t, .bool_t = bool_t, .result_ty = func.valueType(result) };
            const replacement = if (signed)
                try ctx.signedDivRem(allocator, a.lhs, a.rhs, want_rem)
            else
                try ctx.unsignedDivRem(allocator, try ctx.retype(allocator, a.lhs, u32t), try ctx.retype(allocator, a.rhs, u32t), want_rem);
            try subst.put(allocator, result, replacement);
        }
    }

    rewriteUses(func, subst);
    return true;
}

const Ctx = struct {
    func: *Function,
    block: Block,
    u32t: Type,
    bool_t: Type,
    result_ty: Type,

    fn ai(self: Ctx, allocator: std.mem.Allocator, ty: Type, op: BinOp, lhs: Value, imm: i64) Error!Value {
        _ = allocator;
        return self.func.appendArithImm(self.block, ty, op, lhs, imm);
    }
    fn bin(self: Ctx, allocator: std.mem.Allocator, ty: Type, op: BinOp, lhs: Value, rhs: Value) Error!Value {
        _ = allocator;
        return self.func.appendInst(self.block, ty, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    }
    fn konst(self: Ctx, ty: Type, v: i64) Error!Value {
        return self.func.appendInst(self.block, ty, .{ .iconst = v });
    }
    /// Reinterpret `v` as type `ty` without changing its bits (a no-op `| 0`).
    fn retype(self: Ctx, allocator: std.mem.Allocator, v: Value, ty: Type) Error!Value {
        return self.ai(allocator, ty, .bit_or, v, 0);
    }

    /// The unsigned restoring-division core: returns the quotient, or the remainder
    /// when `want_rem`, retyped to the original result type.
    fn unsignedDivRem(self: Ctx, allocator: std.mem.Allocator, n: Value, d: Value, want_rem: bool) Error!Value {
        const u = self.u32t;
        var q = try self.konst(u, 0);
        var r = try self.konst(u, 0);
        var bit: i64 = 31;
        while (bit >= 0) : (bit -= 1) {
            // r = (r << 1) | ((n >> bit) & 1)
            const nb = try self.ai(allocator, u, .shr, n, bit);
            const nbit = try self.ai(allocator, u, .bit_and, nb, 1);
            const rsh = try self.ai(allocator, u, .shl, r, 1);
            r = try self.bin(allocator, u, .bit_or, rsh, nbit);
            // ge = r >= d (unsigned), then r -= d and set the quotient bit when ge.
            const ge = try self.func.appendInst(self.block, self.bool_t, .{ .icmp = .{ .op = .ge, .lhs = r, .rhs = d } });
            const rsub = try self.bin(allocator, u, .sub, r, d);
            r = try self.func.appendInst(self.block, u, .{ .select = .{ .cond = ge, .then = rsub, .@"else" = r } });
            const qset = try self.ai(allocator, u, .bit_or, q, @as(i64, 1) << @intCast(bit));
            q = try self.func.appendInst(self.block, u, .{ .select = .{ .cond = ge, .then = qset, .@"else" = q } });
        }
        return self.retype(allocator, if (want_rem) r else q, self.result_ty);
    }

    /// Signed div/rem: divide magnitudes, then apply the sign. The quotient's sign
    /// is `sign(n) xor sign(d)`. The remainder takes the sign of `n`.
    fn signedDivRem(self: Ctx, allocator: std.mem.Allocator, n: Value, d: Value, want_rem: bool) Error!Value {
        const s = self.result_ty; // the signed result type
        const u = self.u32t;
        const nmask = try self.ai(allocator, s, .shr, n, 31); // 0 or -1
        const dmask = try self.ai(allocator, s, .shr, d, 31);
        const na = try self.absVia(allocator, n, nmask, s);
        const da = try self.absVia(allocator, d, dmask, s);
        const core = try self.unsignedDivRem(allocator, try self.retype(allocator, na, u), try self.retype(allocator, da, u), want_rem);
        const mag = try self.retype(allocator, core, s);
        // Negate when the result should be negative: (mag ^ sign) - sign.
        const sign = if (want_rem) nmask else try self.bin(allocator, s, .bit_xor, nmask, dmask);
        const xored = try self.bin(allocator, s, .bit_xor, mag, sign);
        return self.bin(allocator, s, .sub, xored, sign);
    }

    /// |x| via `(x + mask) ^ mask`, where `mask` is `x >> 31` (all ones if negative).
    fn absVia(self: Ctx, allocator: std.mem.Allocator, x: Value, mask: Value, ty: Type) Error!Value {
        const add = try self.bin(allocator, ty, .add, x, mask);
        return self.bin(allocator, ty, .bit_xor, add, mask);
    }
};

fn isSigned(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |x| x.signedness == .signed,
        else => false,
    };
}

fn hasDivision(func: *const Function) bool {
    for (0..func.blockCount()) |bi| if (blockHasDivision(func, @enumFromInt(bi))) return true;
    return false;
}

fn blockHasDivision(func: *const Function, block: Block) bool {
    for (func.blockInsts(block)) |inst| {
        const op = func.opcode(inst);
        if (op == .arith and (op.arith.op == .div or op.arith.op == .rem)) return true;
    }
    return false;
}

/// Replace every use of a lowered div/rem result with its replacement value.
fn rewriteUses(func: *Function, subst: std.AutoHashMapUnmanaged(Value, Value)) void {
    const sub = struct {
        fn f(m: std.AutoHashMapUnmanaged(Value, Value), v: Value) Value {
            return m.get(v) orelse v;
        }
    }.f;
    for (0..func.instCount()) |i| {
        switch (func.opcodeMut(@enumFromInt(i)).*) {
            .iconst, .fconst, .alloca, .global_addr => {},
            .arith => |*a| {
                a.lhs = sub(subst, a.lhs);
                a.rhs = sub(subst, a.rhs);
            },
            .arith_imm => |*a| a.lhs = sub(subst, a.lhs),
            .icmp => |*c| {
                c.lhs = sub(subst, c.lhs);
                c.rhs = sub(subst, c.rhs);
            },
            .select => |*sl| {
                sl.cond = sub(subst, sl.cond);
                sl.then = sub(subst, sl.then);
                sl.@"else" = sub(subst, sl.@"else");
            },
            .extract => |*e| e.aggregate = sub(subst, e.aggregate),
            .convert => |*cv| cv.value = sub(subst, cv.value),
            .unary => |*u| u.value = sub(subst, u.value),
            .load => |*l| l.ptr = sub(subst, l.ptr),
            .store => |*st| {
                st.value = sub(subst, st.value);
                st.ptr = sub(subst, st.ptr);
            },
            .struct_new => |sn| for (func.valueListMut(sn.fields)) |*f| {
                f.* = sub(subst, f.*);
            },
            .call => |c| for (func.valueListMut(c.args)) |*arg| {
                arg.* = sub(subst, arg.*);
            },
            .call_indirect => |*c| {
                c.target = sub(subst, c.target);
                for (func.valueListMut(c.args)) |*arg| arg.* = sub(subst, arg.*);
            },
            .@"if" => |*cf| {
                cf.cond = sub(subst, cf.cond);
                for (func.valueListMut(cf.then.args)) |*arg| arg.* = sub(subst, arg.*);
                for (func.valueListMut(cf.@"else".args)) |*arg| arg.* = sub(subst, arg.*);
            },
        }
    }
    for (0..func.blockCount()) |bi| {
        const term = func.terminatorPtr(@enumFromInt(bi));
        if (term.*) |*t| switch (t.*) {
            .ret => |*v| if (v.*) |vv| {
                v.* = sub(subst, vv);
            },
            .jump => |*j| for (func.valueListMut(j.args)) |*arg| {
                arg.* = sub(subst, arg.*);
            },
        };
    }
}

test "lowers an unsigned division to division-free IR" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, u32t);
    const y = try func.appendBlockParam(b, u32t);
    const q = try func.appendInst(b, u32t, .{ .arith = .{ .op = .div, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = q });

    try std.testing.expect(try run(allocator, &func));
    // No div/rem instruction survives.
    for (func.blockInsts(b)) |inst| {
        const op = func.opcode(inst);
        if (op == .arith) try std.testing.expect(op.arith.op != .div and op.arith.op != .rem);
    }
}
