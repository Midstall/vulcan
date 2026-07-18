//! x86 (32-bit, cdecl) instruction selection. Integer arguments pass on the stack
//! (`[esp+4]`, `[esp+8]`, ...) and the result returns in EAX. Covers multiple blocks
//! (high-IR if/jump with edge moves), comparison, division, shifts, immediate-operand
//! arithmetic, and register spilling.
//!
//! The prologue loads each argument into its assigned register. The body mirrors the
//! x86-64 selector. EBX/ESI are the spill reload scratches and EDI the parallel-move
//! scratch. EBX is low-byte-addressable so a spilled boolean result can be setcc'd
//! (32-bit setcc only targets EAX/ECX/EDX/EBX). EAX/EDX are reserved when dividing, ECX
//! when shifting. Entry parameters are not spilled (more parameters than pool registers
//! is Unsupported). Spill slots are 4-byte, living below the arguments at `[esp + slot*4]`.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const regalloc = @import("../regalloc.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Reg = encode.Reg;

pub const Error = std.mem.Allocator.Error || error{Unsupported};

const ret_reg: Reg = .eax;
const scratch1: Reg = .ebx; // low-4: holds a left operand / spilled destination (setcc-able)
const scratch2: Reg = .edi; // right operand reload scratch (= move scratch, never overlaps)
const move_scratch: Reg = .edi;

const Loc = union(enum) { reg: Reg, spill: u32 };

const Fixup = struct { at: usize, target: u32 };

/// A `call`'s rel32 displacement (at byte `offset`) targets symbol `symbol`.
pub const Reloc = struct { offset: usize, symbol: []const u8 };

/// A compiled function: machine code plus its unresolved call relocations.
pub const Compiled = struct {
    code: []u8,
    relocs: []Reloc,

    pub fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.relocs);
    }
};

const Ctx = struct {
    func: *const Function,
    loc_of: std.AutoHashMapUnmanaged(Value, Loc) = .{},
    code: std.ArrayList(u8) = .empty,
    fixups: std.ArrayList(Fixup) = .empty,
    relocs: std.ArrayList(Reloc) = .empty,

    fn loc(self: *const Ctx, v: Value) Loc {
        return self.loc_of.get(v).?;
    }
    fn put(self: *Ctx, allocator: std.mem.Allocator, inst: encode.Inst) Error!void {
        try self.code.appendSlice(allocator, inst.slice());
    }
    fn use(self: *Ctx, allocator: std.mem.Allocator, v: Value, scratch: Reg) Error!Reg {
        return switch (self.loc(v)) {
            .reg => |r| r,
            .spill => |slot| {
                try self.put(allocator, encode.stackLoad(scratch, slotDisp(slot)));
                return scratch;
            },
        };
    }
    fn dst(self: *const Ctx, v: Value, scratch: Reg) Reg {
        return switch (self.loc(v)) {
            .reg => |r| r,
            .spill => scratch,
        };
    }
    fn store(self: *Ctx, allocator: std.mem.Allocator, v: Value, reg: Reg) Error!void {
        switch (self.loc(v)) {
            .reg => {},
            .spill => |slot| try self.put(allocator, encode.stackStore(slotDisp(slot), reg)),
        }
    }
};

fn slotDisp(slot: u32) i32 {
    return @intCast(slot * 4);
}

/// Select i386 machine code for `func` (code only, call relocations are dropped).
pub fn selectFunction(allocator: std.mem.Allocator, func: *const Function) Error![]u8 {
    const compiled = try compile(allocator, func);
    allocator.free(compiled.relocs);
    return compiled.code;
}

/// Compile `func` to machine code plus its call relocations. The caller owns it.
pub fn compile(allocator: std.mem.Allocator, func: *const Function) Error!Compiled {
    // f16 not yet lowered on this backend (f16 roadmap Pn); reject cleanly rather than
    // silently treat as f64.
    if (ir.function.functionUsesF16(func)) return error.Unsupported;

    const nblocks = func.blockCount();
    if (nblocks == 0) return error.Unsupported;

    var ctx = Ctx{ .func = func };
    defer ctx.loc_of.deinit(allocator);
    defer ctx.code.deinit(allocator);
    defer ctx.fixups.deinit(allocator);
    defer ctx.relocs.deinit(allocator);
    var num_slots: u32 = 0;
    try assignRegs(allocator, func, &ctx.loc_of, &num_slots);
    const frame: i32 = @intCast((@as(u64, num_slots) * 4 + 15) & ~@as(u64, 15));

    const block_start = try allocator.alloc(usize, nblocks);
    defer allocator.free(block_start);

    // Prologue: reserve the spill frame, then load each cdecl argument. After the ESP
    // adjustment the arguments sit `frame` bytes higher (above the return address). A
    // register parameter goes to its register, a spilled one (live across a call) is copied
    // through a scratch into its slot.
    if (frame > 0) try ctx.put(allocator, encode.aluImm(5, .esp, frame)); // sub esp, frame
    const eparams = func.blockParams(@enumFromInt(0));
    for (eparams, 0..) |p, i| {
        const src: i32 = frame + @as(i32, @intCast(4 + 4 * i));
        switch (ctx.loc(p)) {
            .reg => |r| try ctx.put(allocator, encode.stackLoad(r, src)),
            .spill => |slot| {
                try ctx.put(allocator, encode.stackLoad(scratch1, src));
                try ctx.put(allocator, encode.stackStore(slotDisp(slot), scratch1));
            },
        }
    }

    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        block_start[bi] = ctx.code.items.len;
        var terminated = false;
        for (func.blockInsts(block)) |inst| {
            if (func.opcode(inst) == .@"if") {
                try emitIf(allocator, &ctx, func.opcode(inst).@"if");
                terminated = true;
            } else {
                try lowerInst(allocator, &ctx, inst);
            }
        }
        if (!terminated) switch (func.terminator(block) orelse ir.function.Terminator{ .ret = null }) {
            .ret => |v| {
                if (v) |value| {
                    const src = try ctx.use(allocator, value, ret_reg);
                    if (src != ret_reg) try ctx.put(allocator, encode.movReg(ret_reg, src));
                }
                if (frame > 0) try ctx.put(allocator, encode.aluImm(0, .esp, frame)); // add esp, frame
                try ctx.put(allocator, encode.ret());
            },
            .jump => |j| try emitJump(allocator, &ctx, j),
        };
    }

    for (ctx.fixups.items) |f| {
        const rel: i32 = @intCast(@as(i64, @intCast(block_start[f.target])) - @as(i64, @intCast(f.at + 4)));
        std.mem.writeInt(u32, ctx.code.items[f.at..][0..4], @bitCast(rel), .little);
    }
    return .{ .code = try ctx.code.toOwnedSlice(allocator), .relocs = try ctx.relocs.toOwnedSlice(allocator) };
}

fn lowerInst(allocator: std.mem.Allocator, ctx: *Ctx, inst: ir.function.Inst) Error!void {
    const func = ctx.func;
    const result = func.instResult(inst).?;
    switch (func.opcode(inst)) {
        .iconst => |c| {
            const rd = ctx.dst(result, scratch1);
            try ctx.put(allocator, encode.movImm(rd, @intCast(c)));
            try ctx.store(allocator, result, rd);
        },
        .arith => |a| {
            const signed = isSigned(func, a.lhs);
            switch (a.op) {
                .div, .rem => {
                    try ctx.put(allocator, encode.movReg(.eax, try ctx.use(allocator, a.lhs, scratch1)));
                    try ctx.put(allocator, if (signed) encode.cdq() else encode.xorr(.edx, .edx));
                    try ctx.put(allocator, if (signed) encode.idiv(try ctx.use(allocator, a.rhs, scratch2)) else encode.divu(try ctx.use(allocator, a.rhs, scratch2)));
                    const rd = ctx.dst(result, scratch1);
                    const res: Reg = if (a.op == .div) .eax else .edx;
                    if (rd != res) try ctx.put(allocator, encode.movReg(rd, res));
                    try ctx.store(allocator, result, rd);
                },
                .shl, .shr => {
                    const rl = try ctx.use(allocator, a.lhs, scratch1);
                    try ctx.put(allocator, encode.movReg(.ecx, try ctx.use(allocator, a.rhs, scratch2)));
                    const rd = ctx.dst(result, scratch1);
                    if (rd != rl) try ctx.put(allocator, encode.movReg(rd, rl));
                    try ctx.put(allocator, if (a.op == .shl) encode.shlCl(rd) else if (signed) encode.sarCl(rd) else encode.shrCl(rd));
                    try ctx.store(allocator, result, rd);
                },
                else => {
                    const rl = try ctx.use(allocator, a.lhs, scratch1);
                    const rr = try ctx.use(allocator, a.rhs, scratch2);
                    const rd = ctx.dst(result, scratch1);
                    if (rd != rl) try ctx.put(allocator, encode.movReg(rd, rl));
                    try ctx.put(allocator, try binary(a.op, rd, rr));
                    try ctx.store(allocator, result, rd);
                },
            }
        },
        .arith_imm => |a| {
            const imm: i32 = @intCast(a.imm);
            switch (a.op) {
                .mul => {
                    const rd = ctx.dst(result, scratch1);
                    try ctx.put(allocator, encode.imulImm(rd, try ctx.use(allocator, a.lhs, scratch1), imm));
                    try ctx.store(allocator, result, rd);
                },
                .add, .sub, .bit_and, .bit_or, .bit_xor => {
                    const rl = try ctx.use(allocator, a.lhs, scratch1);
                    const rd = ctx.dst(result, scratch1);
                    if (rd != rl) try ctx.put(allocator, encode.movReg(rd, rl));
                    try ctx.put(allocator, encode.aluImm(aluDigit(a.op), rd, imm));
                    try ctx.store(allocator, result, rd);
                },
                .shl, .shr => {
                    const rl = try ctx.use(allocator, a.lhs, scratch1);
                    const rd = ctx.dst(result, scratch1);
                    if (rd != rl) try ctx.put(allocator, encode.movReg(rd, rl));
                    try ctx.put(allocator, encode.shiftImm(shiftDigit(a.op, isSigned(func, a.lhs)), rd, @truncate(@as(u32, @bitCast(imm)))));
                    try ctx.store(allocator, result, rd);
                },
                .div, .rem => {
                    const signed = isSigned(func, a.lhs);
                    try ctx.put(allocator, encode.movReg(.eax, try ctx.use(allocator, a.lhs, scratch1)));
                    try ctx.put(allocator, if (signed) encode.cdq() else encode.xorr(.edx, .edx));
                    try ctx.put(allocator, encode.movImm(scratch2, imm));
                    try ctx.put(allocator, if (signed) encode.idiv(scratch2) else encode.divu(scratch2));
                    const rd = ctx.dst(result, scratch1);
                    const res: Reg = if (a.op == .div) .eax else .edx;
                    if (rd != res) try ctx.put(allocator, encode.movReg(rd, res));
                    try ctx.store(allocator, result, rd);
                },
            }
        },
        .icmp => |cmp| {
            const rl = try ctx.use(allocator, cmp.lhs, scratch1);
            const rr = try ctx.use(allocator, cmp.rhs, scratch2);
            const rd = ctx.dst(result, scratch1); // scratch1 (EBX) is low-byte-addressable
            try ctx.put(allocator, encode.cmp(rl, rr));
            try ctx.put(allocator, encode.setcc(rd, condOf(cmp.op, isSigned(func, cmp.lhs))));
            try ctx.put(allocator, encode.movzxByte(rd, rd));
            try ctx.store(allocator, result, rd);
        },
        .call => |c| {
            // cdecl: push arguments right-to-left, `call` (relocated), clean the stack,
            // result in EAX. Each spilled argument's slot offset accounts for the pushes
            // already done (ESP has moved). Caller-saved registers are clobbered, values live
            // across the call are force-spilled by assignRegs.
            const args = func.valueList(c.args);
            var pushed: usize = 0;
            var j = args.len;
            while (j > 0) {
                j -= 1;
                switch (ctx.loc(args[j])) {
                    .reg => |r| try ctx.put(allocator, encode.pushReg(r)),
                    .spill => |slot| {
                        try ctx.put(allocator, encode.stackLoad(scratch1, slotDisp(slot) + @as(i32, @intCast(pushed)) * 4));
                        try ctx.put(allocator, encode.pushReg(scratch1));
                    },
                }
                pushed += 1;
            }
            try ctx.put(allocator, encode.callRel(0));
            try ctx.relocs.append(allocator, .{ .offset = ctx.code.items.len - 4, .symbol = func.symbolName(c.symbol) });
            if (args.len > 0) try ctx.put(allocator, encode.aluImm(0, .esp, @intCast(args.len * 4))); // add esp, n*4
            const rd = ctx.dst(result, scratch1);
            if (rd != .eax) try ctx.put(allocator, encode.movReg(rd, .eax));
            try ctx.store(allocator, result, rd);
        },
        else => return error.Unsupported,
    }
}

fn emitIf(allocator: std.mem.Allocator, ctx: *Ctx, cf: ir.function.If) Error!void {
    const cond = try ctx.use(allocator, cf.cond, scratch1);
    try ctx.put(allocator, encode.testReg(cond, cond));
    const jnz = try emitBranch(allocator, ctx, encode.jcc(.ne, 0));
    try emitMoves(allocator, ctx, cf.@"else");
    try emitBranchTo(allocator, ctx, encode.jmp(0), @intFromEnum(cf.@"else".target));
    const then_start = ctx.code.items.len;
    const rel: i32 = @intCast(@as(i64, @intCast(then_start)) - @as(i64, @intCast(jnz + 4)));
    std.mem.writeInt(u32, ctx.code.items[jnz..][0..4], @bitCast(rel), .little);
    try emitMoves(allocator, ctx, cf.then);
    try emitBranchTo(allocator, ctx, encode.jmp(0), @intFromEnum(cf.then.target));
}

fn emitJump(allocator: std.mem.Allocator, ctx: *Ctx, jump: ir.function.Jump) Error!void {
    try emitMoves(allocator, ctx, jump);
    try emitBranchTo(allocator, ctx, encode.jmp(0), @intFromEnum(jump.target));
}

fn emitBranch(allocator: std.mem.Allocator, ctx: *Ctx, inst: encode.Inst) Error!usize {
    const at = ctx.code.items.len + inst.len - 4;
    try ctx.put(allocator, inst);
    return at;
}

fn emitBranchTo(allocator: std.mem.Allocator, ctx: *Ctx, inst: encode.Inst, target: u32) Error!void {
    const at = try emitBranch(allocator, ctx, inst);
    try ctx.fixups.append(allocator, .{ .at = at, .target = target });
}

const Move = struct { src: Reg, dst: Reg };

fn emitMoves(allocator: std.mem.Allocator, ctx: *Ctx, jump: ir.function.Jump) Error!void {
    const args = ctx.func.blockArgs(jump);
    const params = ctx.func.blockParams(jump.target);
    if (args.len != params.len) return error.Unsupported;

    var moves: std.ArrayList(Move) = .empty;
    defer moves.deinit(allocator);
    for (args, params) |arg, param| {
        switch (ctx.loc(param)) {
            .spill => |slot| {
                const src = try ctx.use(allocator, arg, scratch1);
                try ctx.put(allocator, encode.stackStore(slotDisp(slot), src));
            },
            .reg => |dst| switch (ctx.loc(arg)) {
                .reg => |src| if (src != dst) try moves.append(allocator, .{ .src = src, .dst = dst }),
                .spill => {},
            },
        }
    }
    try parallelMove(allocator, ctx, &moves);
    for (args, params) |arg, param| {
        if (ctx.loc(param) == .reg and ctx.loc(arg) == .spill) {
            try ctx.put(allocator, encode.stackLoad(ctx.loc(param).reg, slotDisp(ctx.loc(arg).spill)));
        }
    }
}

fn parallelMove(allocator: std.mem.Allocator, ctx: *Ctx, moves: *std.ArrayList(Move)) Error!void {
    while (moves.items.len > 0) {
        var emitted = false;
        for (moves.items, 0..) |m, i| {
            var blocked = false;
            for (moves.items, 0..) |o, j| if (i != j and o.src == m.dst) {
                blocked = true;
            };
            if (!blocked) {
                try ctx.put(allocator, encode.movReg(m.dst, m.src));
                _ = moves.swapRemove(i);
                emitted = true;
                break;
            }
        }
        if (!emitted) {
            const dst0 = moves.items[0].dst;
            try ctx.put(allocator, encode.movReg(move_scratch, dst0));
            for (moves.items) |*m| if (m.src == dst0) {
                m.src = move_scratch;
            };
        }
    }
}

fn binary(op: ir.function.BinOp, dst: Reg, src: Reg) Error!encode.Inst {
    return switch (op) {
        .add => encode.add(dst, src),
        .sub => encode.sub(dst, src),
        .mul => encode.imul(dst, src),
        .bit_and => encode.andr(dst, src),
        .bit_or => encode.orr(dst, src),
        .bit_xor => encode.xorr(dst, src),
        .div, .rem, .shl, .shr => error.Unsupported,
    };
}

fn aluDigit(op: ir.function.BinOp) u3 {
    return switch (op) {
        .add => 0,
        .bit_or => 1,
        .bit_and => 4,
        .sub => 5,
        .bit_xor => 6,
        else => unreachable,
    };
}

fn shiftDigit(op: ir.function.BinOp, signed: bool) u3 {
    return switch (op) {
        .shl => 4,
        .shr => if (signed) 7 else 5,
        else => unreachable,
    };
}

fn condOf(op: ir.function.CmpOp, signed: bool) encode.Cond {
    return switch (op) {
        .eq => .e,
        .ne => .ne,
        .lt => if (signed) .l else .b,
        .le => if (signed) .le else .be,
        .gt => if (signed) .g else .a,
        .ge => if (signed) .ge else .ae,
    };
}

fn isSigned(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.signedness == .signed,
        else => true,
    };
}

fn fixedRegNeeds(func: *const Function) struct { div: bool, shift: bool } {
    var div = false;
    var shift = false;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .arith => |a| switch (a.op) {
                    .div, .rem => div = true,
                    .shl, .shr => shift = true,
                    else => {},
                },
                .arith_imm => |a| if (a.op == .div or a.op == .rem) {
                    div = true;
                },
                else => {},
            }
        }
    }
    return .{ .div = div, .shift = shift };
}

/// The instruction positions of `call`s, in `regalloc`'s linearization.
fn callPositions(allocator: std.mem.Allocator, func: *const Function) Error![]u32 {
    var positions: std.ArrayList(u32) = .empty;
    errdefer positions.deinit(allocator);
    var pos: u32 = 0;
    for (0..func.blockCount()) |bi| {
        pos += 1;
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .call) try positions.append(allocator, pos);
            pos += 1;
        }
        pos += 1;
    }
    return positions.toOwnedSlice(allocator);
}

/// Linear-scan allocation with reuse and spilling over the pool {EAX,ECX,EDX,ESI} (EBX/EDI
/// are scratch). The pool registers are byte-addressable, so any boolean lands in a
/// setcc-able register. Values live across a call are force-spilled (caller-saved clobber).
fn assignRegs(allocator: std.mem.Allocator, func: *const Function, loc_of: *std.AutoHashMapUnmanaged(Value, Loc), num_slots: *u32) Error!void {
    const needs = fixedRegNeeds(func);
    var pool: std.ArrayList(Reg) = .empty;
    defer pool.deinit(allocator);
    const candidates = [_]Reg{ .eax, .ecx, .edx, .esi };
    for (candidates) |r| {
        if (needs.div and (r == .eax or r == .edx)) continue;
        if (needs.shift and r == .ecx) continue;
        try pool.append(allocator, r);
    }

    const ivals = try regalloc.computeLiveIntervals(allocator, func);
    defer allocator.free(ivals);
    const calls = try callPositions(allocator, func);
    defer allocator.free(calls);

    const Active = struct { end: u32, value: Value, reg: Reg };
    var active: std.ArrayList(Active) = .empty;
    defer active.deinit(allocator);
    var free = try pool.clone(allocator);
    defer free.deinit(allocator);

    for (ivals) |iv| {
        var w: usize = 0;
        for (active.items) |act| {
            if (act.end < iv.start) {
                try free.append(allocator, act.reg);
            } else {
                active.items[w] = act;
                w += 1;
            }
        }
        active.shrinkRetainingCapacity(w);

        var across = false;
        for (calls) |cp| if (iv.start < cp and cp < iv.end) {
            across = true;
        };
        if (across) {
            try loc_of.put(allocator, iv.value, .{ .spill = num_slots.* });
            num_slots.* += 1;
        } else if (free.pop()) |r| {
            try loc_of.put(allocator, iv.value, .{ .reg = r });
            try active.append(allocator, .{ .end = iv.end, .value = iv.value, .reg = r });
        } else {
            var victim: usize = 0;
            for (active.items, 0..) |act, i| if (act.end > active.items[victim].end) {
                victim = i;
            };
            if (active.items.len > 0 and active.items[victim].end > iv.end) {
                const v = active.items[victim];
                try loc_of.put(allocator, v.value, .{ .spill = num_slots.* });
                num_slots.* += 1;
                try loc_of.put(allocator, iv.value, .{ .reg = v.reg });
                active.items[victim] = .{ .end = iv.end, .value = iv.value, .reg = v.reg };
            } else {
                try loc_of.put(allocator, iv.value, .{ .spill = num_slots.* });
                num_slots.* += 1;
            }
        }
    }
}

test "selects a straight-line arithmetic function" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = prod });
    const code = try selectFunction(allocator, &func);
    defer allocator.free(code);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]); // ret
}

test "an f16 function is rejected cleanly, not miscompiled as f64" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f16 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = s });

    try std.testing.expectError(error.Unsupported, selectFunction(allocator, &func));
}
