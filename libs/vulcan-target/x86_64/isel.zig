//! x86-64 instruction selection. System V AMD64 ABI: integer args in RDI, RSI, RDX,
//! RCX, R8, R9, result in RAX. Covers multiple blocks (high-IR if/jump with
//! block-parameter edge moves), comparison, division, shifts, immediate-operand
//! arithmetic, and register spilling.
//!
//! x86 arithmetic is two-operand, so `c = a op b` becomes `mov c, a` then `op c, b`.
//! Registers are linear-scanned with reuse and spilling over the shared regalloc
//! intervals. A spilled value lives in a stack slot, reloaded into a scratch register
//! (R10/R11) at each use, computed in a scratch, and stored back. R11 doubles as the
//! parallel-move scratch (non-overlapping in time). RAX/RDX are reserved when dividing,
//! RCX when shifting.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const regalloc = @import("../regalloc.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Reg = encode.Reg;

pub const Error = std.mem.Allocator.Error || error{Unsupported};

const arg_regs = [_]Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 };
const ret_reg: Reg = .rax;
const scratch1: Reg = .r10; // reload scratch for a left operand / destination
const scratch2: Reg = .r11; // reload scratch for a right operand
const move_scratch: Reg = .r11; // parallel-move cycle scratch (non-overlapping with spills)

/// Where a value lives: a general register, an SSE (xmm) register, a general-register stack
/// spill slot, or an xmm stack spill slot (16-byte, holds a scalar float or a whole vector).
const Loc = union(enum) { reg: Reg, xmm: encode.Xmm, spill: u32, xmm_spill: u32 };

const Xmm = encode.Xmm;
const xmm_arg_regs = [_]Xmm{ .xmm0, .xmm1, .xmm2, .xmm3, .xmm4, .xmm5, .xmm6, .xmm7 };
const xmm_ret: Xmm = .xmm0;
// xmm13/14/15 are reserved scratch (operand reloads + the move/aliasing temp). assignXmm
// allocates xmm0..xmm12. A reloaded left operand goes to op0, a right operand to op1.
const xmm_op0: Xmm = .xmm13;
const xmm_op1: Xmm = .xmm14;
const xmm_scratch: Xmm = .xmm15;

/// Whether `v` is a floating-point value (lives in an xmm register).
fn isFloat(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .float;
}
/// Whether `v` is a SIMD vector (<N x f32>, also an xmm register).
fn isVector(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .vector;
}
/// Whether `v` is a 256-bit (AVX/YMM) vector: more than four f32 lanes, needing the
/// VEX-encoded ops rather than the 128-bit SSE ones. Exactly 8 lanes today.
fn isWide(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .vector => |vec| vec.len > 4,
        else => false,
    };
}
/// Whether `v` is a double-precision (f64) scalar float.
fn isDouble(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f64,
        else => false,
    };
}
/// Whether `v` lives in an xmm register (a scalar float or a SIMD vector).
fn isXmm(func: *const Function, v: Value) bool {
    return isFloat(func, v) or isVector(func, v);
}
/// The bit width of an integer value, or 64 for non-integers (the safe 64-bit default).
fn intBits(func: *const Function, v: Value) u16 {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.bits,
        else => 64,
    };
}

const Fixup = struct { at: usize, target: u32 };

/// A `call`'s rel32 displacement (at byte `offset`) targets `symbol`, resolved by the
/// linker (PC-relative, addend -4 for the E8 displacement).
pub const Reloc = struct { offset: usize, symbol: []const u8 };

/// A compiled function: machine code plus its unresolved call relocations.
/// A source-line row: the byte offset where a source line's code begins (from `debug.line` attrs).
pub const LineEntry = struct { offset: u32, line: u32 };

pub const Compiled = struct {
    code: []u8,
    relocs: []Reloc,
    lines: []LineEntry = &.{},

    pub fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.relocs);
        allocator.free(self.lines);
    }
};

const Ctx = struct {
    func: *const Function,
    loc_of: std.AutoHashMapUnmanaged(Value, Loc) = .{},
    code: std.ArrayList(u8) = .empty,
    fixups: std.ArrayList(Fixup) = .empty,
    relocs: std.ArrayList(Reloc) = .empty,
    lines: std.ArrayList(LineEntry) = .empty,
    last_line: u32 = 0,
    xmm_base: i32 = 0, // rsp offset of the xmm spill area (16-byte slots)
    alloca_base: i32 = 0, // rsp offset of the alloca region (sits above the spill areas)
    alloca_off: std.AutoHashMapUnmanaged(Value, u32) = .{}, // each alloca result -> its byte offset in that region

    fn loc(self: *const Ctx, v: Value) Loc {
        return self.loc_of.get(v).?;
    }
    fn put(self: *Ctx, allocator: std.mem.Allocator, inst: encode.Inst) Error!void {
        try self.code.appendSlice(allocator, inst.slice());
    }
    fn xmmDisp(self: *const Ctx, slot: u32) i32 {
        return self.xmm_base + @as(i32, @intCast(slot)) * 32;
    }
    /// The register holding `v`, reloading it from its spill slot into `scratch`.
    fn use(self: *Ctx, allocator: std.mem.Allocator, v: Value, scratch: Reg) Error!Reg {
        return switch (self.loc(v)) {
            .reg => |r| r,
            .spill => |slot| {
                try self.put(allocator, encode.movFromStack(scratch, slotDisp(slot)));
                return scratch;
            },
            .xmm, .xmm_spill => unreachable, // a gpr consumer never sees an xmm value
        };
    }
    /// A register to compute `v` into (its own register, or `scratch` if spilled).
    fn dst(self: *const Ctx, v: Value, scratch: Reg) Reg {
        return switch (self.loc(v)) {
            .reg => |r| r,
            .spill => scratch,
            .xmm, .xmm_spill => unreachable,
        };
    }
    /// Store a freshly-computed `v` (in `reg`) back to its spill slot, if any.
    fn store(self: *Ctx, allocator: std.mem.Allocator, v: Value, reg: Reg) Error!void {
        switch (self.loc(v)) {
            .reg => {},
            .spill => |slot| try self.put(allocator, encode.movToStack(slotDisp(slot), reg)),
            .xmm, .xmm_spill => unreachable,
        }
    }

    /// The xmm register holding `v`, reloading a spilled value into `scratch` (movups for a
    /// vector, movss for a scalar float).
    fn useXmm(self: *Ctx, allocator: std.mem.Allocator, v: Value, scratch: Xmm) Error!Xmm {
        return switch (self.loc(v)) {
            .xmm => |x| x,
            .xmm_spill => |slot| {
                // Each slot is 32 bytes. A 256-bit vmovups reloads a whole <8 x f32>. A 128-bit
                // movups a scalar float / double / <4 x f32> (low 16 bytes). Neither truncates.
                if (isWide(self.func, v)) {
                    try self.put(allocator, encode.vmovupsLoad(scratch, self.xmmDisp(slot)));
                } else {
                    try self.put(allocator, encode.movupsLoad(scratch, self.xmmDisp(slot)));
                }
                return scratch;
            },
            else => error.Unsupported,
        };
    }
    /// The xmm register to compute `v` into (its own register, or `scratch` if spilled).
    fn dstXmm(self: *const Ctx, v: Value, scratch: Xmm) Error!Xmm {
        return switch (self.loc(v)) {
            .xmm => |x| x,
            .xmm_spill => scratch,
            else => error.Unsupported,
        };
    }
    /// Store a freshly-computed xmm `v` (in `reg`) back to its spill slot, if any.
    fn storeXmm(self: *Ctx, allocator: std.mem.Allocator, v: Value, reg: Xmm) Error!void {
        switch (self.loc(v)) {
            .xmm_spill => |slot| if (isWide(self.func, v)) {
                try self.put(allocator, encode.vmovupsStore(self.xmmDisp(slot), reg));
            } else {
                try self.put(allocator, encode.movupsStore(self.xmmDisp(slot), reg));
            },
            else => {},
        }
    }
};

fn slotDisp(slot: u32) i32 {
    return @intCast(slot * 8);
}

/// Select x86-64 machine code for `func` (code only, call relocations dropped). Caller
/// owns the slice.
pub fn selectFunction(allocator: std.mem.Allocator, func: *const Function) Error![]u8 {
    const compiled = try compile(allocator, func);
    allocator.free(compiled.relocs);
    return compiled.code;
}

/// Compile `func` to machine code plus its call relocations. Caller owns it.
pub fn compile(allocator: std.mem.Allocator, func: *const Function) Error!Compiled {
    const nblocks = func.blockCount();
    if (nblocks == 0) return error.Unsupported;

    var ctx = Ctx{ .func = func };
    defer ctx.loc_of.deinit(allocator);
    defer ctx.code.deinit(allocator);
    defer ctx.fixups.deinit(allocator);
    defer ctx.relocs.deinit(allocator);
    defer ctx.lines.deinit(allocator);
    defer ctx.alloca_off.deinit(allocator);
    var num_slots: u32 = 0;
    try assignRegs(allocator, func, &ctx.loc_of, &num_slots);
    var xmm_slots: u32 = 0;
    try assignXmm(allocator, func, &ctx.loc_of, &xmm_slots);
    // Frame layout: general spills (8 bytes each), then the xmm spill area (32-byte slots
    // at a 16-aligned base, sized for a whole 256-bit ymm. A scalar/128-bit value uses the
    // low half), then the alloca region (each alloca offset relative to its 16-aligned base).
    const xmm_base: u64 = (@as(u64, num_slots) * 8 + 15) & ~@as(u64, 15);
    ctx.xmm_base = @intCast(xmm_base);
    const alloca_base: u64 = (xmm_base + @as(u64, xmm_slots) * 32 + 15) & ~@as(u64, 15);
    ctx.alloca_base = @intCast(alloca_base);
    const alloca_bytes = try computeAllocaSlots(allocator, func, &ctx.alloca_off);
    const frame: i32 = @intCast((alloca_base + alloca_bytes + 15) & ~@as(u64, 15));

    const block_start = try allocator.alloc(usize, nblocks);
    defer allocator.free(block_start);

    // Prologue: reserve the spill frame, then move each argument from its ABI register to
    // the entry parameter's location (a register parallel move, or a store for a spilled
    // parameter).
    if (frame > 0) try ctx.put(allocator, encode.aluImm(5, .rsp, frame)); // sub rsp, frame
    // System V passes general args in rdi,rsi,... and fp args in xmm0,xmm1,... (separate
    // sequences), so each class has its own incoming-register index.
    const eparams = func.blockParams(@enumFromInt(0));
    var arg_moves: std.ArrayList(Move) = .empty;
    defer arg_moves.deinit(allocator);
    var gpr_i: usize = 0;
    var xmm_i: usize = 0;
    for (eparams) |p| {
        if (isXmm(func, p)) {
            // A vector param also lives in an xmm register, so classify by isXmm (float or
            // vector), matching the call-site arg handling. A scalar float moves/stores as
            // 128-bit (movups, the extra lanes are harmless); a 128-bit vector as movups; a
            // 256-bit vector as vmovups so no lanes are dropped.
            if (xmm_i >= xmm_arg_regs.len) return error.Unsupported; // fp stack args not handled
            const incoming = xmm_arg_regs[xmm_i];
            xmm_i += 1;
            switch (ctx.loc(p)) {
                .xmm => |x| if (x != incoming) try ctx.put(allocator, if (isWide(func, p)) encode.vmovupsRR(x, incoming) else encode.movupsRR(x, incoming)), // no fp move cycles for a single arg
                .xmm_spill => |slot| try ctx.put(allocator, if (isWide(func, p)) encode.vmovupsStore(ctx.xmmDisp(slot), incoming) else if (isVector(func, p)) encode.movupsStore(ctx.xmmDisp(slot), incoming) else encode.movssStore(ctx.xmmDisp(slot), incoming)),
                else => unreachable,
            }
        } else {
            if (gpr_i >= arg_regs.len) return error.Unsupported;
            const incoming = arg_regs[gpr_i];
            gpr_i += 1;
            switch (ctx.loc(p)) {
                .spill => |slot| try ctx.put(allocator, encode.movToStack(slotDisp(slot), incoming)),
                .reg => |r| if (r != incoming) try arg_moves.append(allocator, .{ .src = incoming, .dst = r }),
                .xmm, .xmm_spill => unreachable,
            }
        }
    }
    try parallelMove(allocator, &ctx, &arg_moves);

    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        block_start[bi] = ctx.code.items.len;
        var terminated = false;
        for (func.blockInsts(block)) |inst| {
            // Record a source-line row when this instruction starts a new line (byte offset = the
            // current code length, since x86 code is already a byte stream).
            if (lineOf(func, inst)) |line| {
                if (line != ctx.last_line) {
                    try ctx.lines.append(allocator, .{ .offset = @intCast(ctx.code.items.len), .line = line });
                    ctx.last_line = line;
                }
            }
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
                    if (isFloat(func, value)) {
                        const src = try ctx.useXmm(allocator, value, xmm_scratch);
                        if (src != xmm_ret) try ctx.put(allocator, encode.movupsRR(xmm_ret, src));
                    } else {
                        const src = try ctx.use(allocator, value, ret_reg);
                        if (src != ret_reg) try ctx.put(allocator, encode.movReg(ret_reg, src));
                    }
                }
                if (frame > 0) try ctx.put(allocator, encode.aluImm(0, .rsp, frame)); // add rsp, frame
                try ctx.put(allocator, encode.ret());
            },
            .jump => |j| try emitJump(allocator, &ctx, j),
        };
    }

    for (ctx.fixups.items) |f| {
        const rel: i32 = @intCast(@as(i64, @intCast(block_start[f.target])) - @as(i64, @intCast(f.at + 4)));
        std.mem.writeInt(u32, ctx.code.items[f.at..][0..4], @bitCast(rel), .little);
    }
    return .{ .code = try ctx.code.toOwnedSlice(allocator), .relocs = try ctx.relocs.toOwnedSlice(allocator), .lines = try ctx.lines.toOwnedSlice(allocator) };
}

/// The `debug.line` source line attached to an IR instruction, if any.
fn lineOf(func: *const Function, inst: ir.function.Inst) ?u32 {
    var it = func.attributesOf(.{ .inst = inst });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "debug") and std.mem.eql(u8, c.key, "line")) {
            if (c.value == .int) return @intCast(c.value.int);
        },
        else => {},
    };
    return null;
}

fn lowerInst(allocator: std.mem.Allocator, ctx: *Ctx, inst: ir.function.Inst) Error!void {
    const func = ctx.func;
    if (func.opcode(inst) == .store) {
        // `store` produces no result, so handle it before the result unwrap below.
        const st = func.opcode(inst).store;
        const base = try ctx.use(allocator, st.ptr, scratch2);
        if (isXmm(func, st.value)) {
            const val = try ctx.useXmm(allocator, st.value, xmm_op0);
            try ctx.put(allocator, if (isVector(func, st.value)) encode.movupsStoreMem(base, 0, val) else if (isDouble(func, st.value)) encode.movsdStoreMem(base, 0, val) else encode.movssStoreMem(base, 0, val));
        } else {
            const val = try ctx.use(allocator, st.value, scratch1);
            // Store the value's own width: a 32-bit int writes 4 bytes, not 8 (an 8-byte store
            // would clobber the next element of a tightly-packed i32 array).
            try ctx.put(allocator, if (intBits(func, st.value) <= 32) encode.movToMem32(base, 0, val) else encode.movToMem(base, 0, val));
        }
        return;
    }
    if (func.opcode(inst) == .call_indirect) {
        // Indirect call through a function pointer (e.g. the software sampler helper, which
        // returns nothing and writes its result through a pointer arg, so this has no result).
        // Stage the target into r10 (survives the arg moves and the call, never an arg reg),
        // move the arguments into the System V arg registers, then `call r10`.
        const c = func.opcode(inst).call_indirect;
        const tgt = try ctx.use(allocator, c.target, scratch1);
        if (tgt != scratch1) try ctx.put(allocator, encode.movReg(scratch1, tgt));
        const args = func.valueList(c.args);
        var moves: std.ArrayList(Move) = .empty;
        defer moves.deinit(allocator);
        var xmm_moves: std.ArrayList(XmmMove) = .empty;
        defer xmm_moves.deinit(allocator);
        var gi: usize = 0;
        var xi: usize = 0;
        for (args) |arg| {
            if (isXmm(func, arg)) {
                if (xi >= xmm_arg_regs.len) return error.Unsupported; // fp stack args not handled
                const dst = xmm_arg_regs[xi];
                xi += 1;
                switch (ctx.loc(arg)) {
                    .xmm => |src| if (src != dst) try xmm_moves.append(allocator, .{ .src = src, .dst = dst, .wide = isWide(func, arg) }),
                    .xmm_spill => {}, // reloaded after the parallel move
                    else => unreachable,
                }
            } else {
                if (gi >= arg_regs.len) return error.Unsupported;
                const dst = arg_regs[gi];
                gi += 1;
                switch (ctx.loc(arg)) {
                    .reg => |src| if (src != dst) try moves.append(allocator, .{ .src = src, .dst = dst }),
                    .spill => {},
                    else => unreachable,
                }
            }
        }
        try parallelMove(allocator, ctx, &moves);
        try parallelMoveXmm(allocator, ctx, &xmm_moves);
        gi = 0;
        xi = 0;
        for (args) |arg| {
            if (isXmm(func, arg)) {
                const dst = xmm_arg_regs[xi];
                xi += 1;
                if (ctx.loc(arg) == .xmm_spill) {
                    const disp = ctx.xmmDisp(ctx.loc(arg).xmm_spill);
                    try ctx.put(allocator, if (isWide(func, arg)) encode.vmovupsLoad(dst, disp) else if (isVector(func, arg)) encode.movupsLoad(dst, disp) else encode.movssLoad(dst, disp));
                }
            } else {
                const dst = arg_regs[gi];
                gi += 1;
                if (ctx.loc(arg) == .spill) try ctx.put(allocator, encode.movFromStack(dst, slotDisp(ctx.loc(arg).spill)));
            }
        }
        try ctx.put(allocator, encode.callReg(scratch1)); // call r10
        if (func.instResult(inst)) |res| {
            if (isXmm(func, res)) {
                const rd = try ctx.dstXmm(res, xmm_scratch); // fp result comes back in xmm0
                if (rd != xmm_ret) try ctx.put(allocator, if (isWide(func, res)) encode.vmovupsRR(rd, xmm_ret) else encode.movupsRR(rd, xmm_ret));
                try ctx.storeXmm(allocator, res, rd);
            } else {
                const rd = ctx.dst(res, scratch1);
                if (rd != .rax) try ctx.put(allocator, encode.movReg(rd, .rax));
                try ctx.store(allocator, res, rd);
            }
        }
        return;
    }
    const result = func.instResult(inst).?;
    switch (func.opcode(inst)) {
        .iconst => |c| {
            if (isXmm(func, result)) {
                // A float-typed integer constant (e.g. a zero-init). The result lives in
                // an xmm register, so materialize the bits in a gpr and move them across,
                // never a plain integer store into an xmm slot.
                const rd = try ctx.dstXmm(result, xmm_scratch);
                if (isDouble(func, result)) {
                    try ctx.put(allocator, encode.movImm64(scratch1, @bitCast(c)));
                    try ctx.put(allocator, encode.movqToXmm(rd, scratch1));
                } else {
                    const bits: u32 = @truncate(@as(u64, @bitCast(c)));
                    try ctx.put(allocator, encode.movImm(scratch1, @bitCast(bits)));
                    try ctx.put(allocator, encode.movdToXmm(rd, scratch1));
                }
                try ctx.storeXmm(allocator, result, rd);
            } else {
                const rd = ctx.dst(result, scratch1);
                try ctx.put(allocator, encode.movImm(rd, @intCast(c)));
                try ctx.store(allocator, result, rd);
            }
        },
        .fconst => |val| {
            // Materialize the float bits in a scratch gpr, then move into xmm: 32 bits via
            // movd for f32, 64 bits via movq for f64.
            const rd = try ctx.dstXmm(result, xmm_scratch);
            if (isDouble(func, result)) {
                try ctx.put(allocator, encode.movImm64(scratch1, @bitCast(val)));
                try ctx.put(allocator, encode.movqToXmm(rd, scratch1));
            } else {
                const bits: u32 = @bitCast(@as(f32, @floatCast(val)));
                try ctx.put(allocator, encode.movImm(scratch1, @bitCast(bits)));
                try ctx.put(allocator, encode.movdToXmm(rd, scratch1));
            }
            try ctx.storeXmm(allocator, result, rd);
        },
        .arith => |a| {
            if (isWide(func, result)) {
                // AVX 256-bit is three-operand and non-destructive: dst = v<op>ps src1, src2
                // directly, so no copy and no dst==src alias handling.
                const rl = try ctx.useXmm(allocator, a.lhs, xmm_op0);
                const rr = try ctx.useXmm(allocator, a.rhs, xmm_op1);
                const work = try ctx.dstXmm(result, xmm_scratch);
                try ctx.put(allocator, switch (a.op) {
                    .add => encode.vaddps(work, rl, rr),
                    .sub => encode.vsubps(work, rl, rr),
                    .mul => encode.vmulps(work, rl, rr),
                    .div => encode.vdivps(work, rl, rr),
                    else => return error.Unsupported,
                });
                try ctx.storeXmm(allocator, result, work);
                return;
            }
            if (isXmm(func, result)) {
                // SSE two-operand (dst op= src): packed-single (...ps), scalar-double (...sd),
                // or scalar-single (...ss) by type. Copies use a 128-bit movups so a double or
                // vector is not truncated. Spilled operands reload into op0/op1, a spilled
                // result computes into xmm_scratch, which also breaks the rd==rr alias.
                const vec = isVector(func, result);
                const dbl = isDouble(func, result);
                const rl = try ctx.useXmm(allocator, a.lhs, xmm_op0);
                const rr = try ctx.useXmm(allocator, a.rhs, xmm_op1);
                const work = try ctx.dstXmm(result, xmm_scratch);
                const op = struct {
                    fn e(o: ir.function.BinOp, v: bool, d: bool, dst: Xmm, src: Xmm) Error!encode.Inst {
                        return switch (o) {
                            .add => if (v) encode.addps(dst, src) else if (d) encode.addsd(dst, src) else encode.addss(dst, src),
                            .sub => if (v) encode.subps(dst, src) else if (d) encode.subsd(dst, src) else encode.subss(dst, src),
                            .mul => if (v) encode.mulps(dst, src) else if (d) encode.mulsd(dst, src) else encode.mulss(dst, src),
                            .div => if (v) encode.divps(dst, src) else if (d) encode.divsd(dst, src) else encode.divss(dst, src),
                            else => error.Unsupported,
                        };
                    }
                }.e;
                if (work == rl) {
                    try ctx.put(allocator, try op(a.op, vec, dbl, work, rr));
                } else if (work == rr) {
                    try ctx.put(allocator, encode.movupsRR(xmm_scratch, rl));
                    try ctx.put(allocator, try op(a.op, vec, dbl, xmm_scratch, rr));
                    try ctx.put(allocator, encode.movupsRR(work, xmm_scratch));
                } else {
                    try ctx.put(allocator, encode.movupsRR(work, rl));
                    try ctx.put(allocator, try op(a.op, vec, dbl, work, rr));
                }
                try ctx.storeXmm(allocator, result, work);
                return;
            }
            const signed = isSigned(func, a.lhs);
            switch (a.op) {
                .div, .rem => {
                    try ctx.put(allocator, encode.movReg(.rax, try ctx.use(allocator, a.lhs, scratch1)));
                    try ctx.put(allocator, if (signed) encode.cqo() else encode.xorr(.rdx, .rdx));
                    try ctx.put(allocator, if (signed) encode.idiv(try ctx.use(allocator, a.rhs, scratch2)) else encode.divu(try ctx.use(allocator, a.rhs, scratch2)));
                    const rd = ctx.dst(result, scratch1);
                    const res: Reg = if (a.op == .div) .rax else .rdx;
                    if (rd != res) try ctx.put(allocator, encode.movReg(rd, res));
                    try ctx.store(allocator, result, rd);
                },
                .shl, .shr => {
                    const rl = try ctx.use(allocator, a.lhs, scratch1);
                    try ctx.put(allocator, encode.movReg(.rcx, try ctx.use(allocator, a.rhs, scratch2)));
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
                    try ctx.put(allocator, encode.movReg(.rax, try ctx.use(allocator, a.lhs, scratch1)));
                    try ctx.put(allocator, if (signed) encode.cqo() else encode.xorr(.rdx, .rdx));
                    try ctx.put(allocator, encode.movImm(scratch2, imm));
                    try ctx.put(allocator, if (signed) encode.idiv(scratch2) else encode.divu(scratch2));
                    const rd = ctx.dst(result, scratch1);
                    const res: Reg = if (a.op == .div) .rax else .rdx;
                    if (rd != res) try ctx.put(allocator, encode.movReg(rd, res));
                    try ctx.store(allocator, result, rd);
                },
            }
        },
        .icmp => |cmp| {
            if (isVector(func, cmp.lhs)) {
                // Per-lane float compare via cmpps, producing an all-ones/all-zero lane mask that
                // a later vector select reads. gt/ge have no ordered SSE predicate, so swap the
                // operands and use lt/le. Compare in xmm_scratch (cmpps overwrites its first
                // operand) then move to the result register.
                const rl = try ctx.useXmm(allocator, cmp.lhs, xmm_op0);
                const rr = try ctx.useXmm(allocator, cmp.rhs, xmm_op1);
                const swap = cmp.op == .gt or cmp.op == .ge;
                const first = if (swap) rr else rl;
                const second = if (swap) rl else rr;
                const pred: u8 = switch (cmp.op) {
                    .eq => 0,
                    .ne => 4,
                    .lt, .gt => 1, // LT (gt swapped to lt)
                    .le, .ge => 2, // LE (ge swapped to le)
                };
                try ctx.put(allocator, encode.movupsRR(xmm_scratch, first));
                try ctx.put(allocator, encode.cmpps(xmm_scratch, second, pred));
                const rd = try ctx.dstXmm(result, xmm_scratch);
                if (rd != xmm_scratch) try ctx.put(allocator, encode.movupsRR(rd, xmm_scratch));
                try ctx.storeXmm(allocator, result, rd);
                return;
            }
            if (isFloat(func, cmp.lhs)) {
                // Float compare via ucomiss + setcc. The bool result lives in a gpr.
                const rd = ctx.dst(result, scratch1);
                if (cmp.op == .eq or cmp.op == .ne) {
                    // ucomiss sets PF on unordered (NaN). Ordered-equal is ZF=1 and PF=0, and
                    // != its inverse, so combine two setcc results.
                    const ra = try ctx.useXmm(allocator, cmp.lhs, xmm_op0);
                    const rb = try ctx.useXmm(allocator, cmp.rhs, xmm_op1);
                    try ctx.put(allocator, if (isDouble(func, cmp.lhs)) encode.ucomisd(ra, rb) else encode.ucomiss(ra, rb));
                    try ctx.put(allocator, encode.setcc(rd, if (cmp.op == .eq) .e else .ne));
                    try ctx.put(allocator, encode.movzxByte(rd, rd));
                    try ctx.put(allocator, encode.setcc(scratch2, if (cmp.op == .eq) .np else .p));
                    try ctx.put(allocator, encode.movzxByte(scratch2, scratch2));
                    try ctx.put(allocator, if (cmp.op == .eq) encode.andr(rd, scratch2) else encode.orr(rd, scratch2));
                    try ctx.store(allocator, result, rd);
                    return;
                }
                // lt/le swap the operands so seta/setae (CF=0 and ZF=0) excludes the unordered case.
                const swap = cmp.op == .lt or cmp.op == .le;
                const ra = try ctx.useXmm(allocator, if (swap) cmp.rhs else cmp.lhs, xmm_op0);
                const rb = try ctx.useXmm(allocator, if (swap) cmp.lhs else cmp.rhs, xmm_op1);
                try ctx.put(allocator, if (isDouble(func, cmp.lhs)) encode.ucomisd(ra, rb) else encode.ucomiss(ra, rb));
                const cc: encode.Cond = if (cmp.op == .lt or cmp.op == .gt) .a else .ae;
                try ctx.put(allocator, encode.setcc(rd, cc));
                try ctx.put(allocator, encode.movzxByte(rd, rd));
                try ctx.store(allocator, result, rd);
                return;
            }
            const rl = try ctx.use(allocator, cmp.lhs, scratch1);
            const rr = try ctx.use(allocator, cmp.rhs, scratch2);
            const rd = ctx.dst(result, scratch1);
            try ctx.put(allocator, encode.cmp(rl, rr));
            try ctx.put(allocator, encode.setcc(rd, condOf(cmp.op, isSigned(func, cmp.lhs))));
            try ctx.put(allocator, encode.movzxByte(rd, rd));
            try ctx.store(allocator, result, rd);
        },
        .select => |s| {
            if (isVector(func, result)) {
                // Per-lane vector select from a cmpps mask: result = (then & mask) | (else & ~mask).
                // SSE1 and/andn/or only (no SSE4.1 blendv on the baseline). Build both halves in the
                // reserved scratch xmms so the operand registers are never clobbered.
                const mask = try ctx.useXmm(allocator, s.cond, xmm_op0);
                const tr = try ctx.useXmm(allocator, s.then, xmm_op1);
                const el = try ctx.useXmm(allocator, s.@"else", xmm_scratch);
                try ctx.put(allocator, encode.movupsRR(xmm_op0, mask)); // op0 = mask, for the two ands
                try ctx.put(allocator, encode.movupsRR(xmm_scratch, tr));
                try ctx.put(allocator, encode.andps(xmm_scratch, xmm_op0)); // then & mask
                try ctx.put(allocator, encode.andnps(xmm_op0, el)); // op0 = (~mask) & else
                try ctx.put(allocator, encode.orps(xmm_scratch, xmm_op0)); // combine both halves
                const rd = try ctx.dstXmm(result, xmm_scratch);
                if (rd != xmm_scratch) try ctx.put(allocator, encode.movupsRR(rd, xmm_scratch));
                try ctx.storeXmm(allocator, result, rd);
                return;
            }
            // x86 has no SSE conditional move, and the result register may alias an operand,
            // so lower select as a two-armed branch: each arm writes exactly one operand into
            // the result, touching nothing else. Works for both gpr and xmm results.
            const c = try ctx.use(allocator, s.cond, scratch1);
            try ctx.put(allocator, encode.testReg(c, c));
            try ctx.put(allocator, encode.jcc(.e, 0)); // je -> else arm (cond == 0), patched
            const je_at = ctx.code.items.len - 4;
            try selectInto(allocator, ctx, result, s.then);
            try ctx.put(allocator, encode.jmp(0)); // jmp -> end, patched
            const jmp_at = ctx.code.items.len - 4;
            const else_rel: i32 = @intCast(@as(i64, @intCast(ctx.code.items.len)) - @as(i64, @intCast(je_at + 4)));
            std.mem.writeInt(u32, ctx.code.items[je_at..][0..4], @bitCast(else_rel), .little);
            try selectInto(allocator, ctx, result, s.@"else");
            const end_rel: i32 = @intCast(@as(i64, @intCast(ctx.code.items.len)) - @as(i64, @intCast(jmp_at + 4)));
            std.mem.writeInt(u32, ctx.code.items[jmp_at..][0..4], @bitCast(end_rel), .little);
        },
        .convert => |cv| {
            // Numeric conversions: int <-> float (32-bit int, f32 or f64), int <-> int (low
            // bits), and f32 <-> f64.
            const src_float = isFloat(func, cv.value);
            const dst_float = isFloat(func, result);
            if (!src_float and dst_float) {
                const src = try ctx.use(allocator, cv.value, scratch1); // i32 in a gpr
                const rd = try ctx.dstXmm(result, xmm_scratch);
                try ctx.put(allocator, if (isDouble(func, result)) encode.cvtsi2sd(rd, src) else encode.cvtsi2ss(rd, src));
                try ctx.storeXmm(allocator, result, rd);
            } else if (src_float and !dst_float) {
                const src = try ctx.useXmm(allocator, cv.value, xmm_op0);
                const rd = ctx.dst(result, scratch1); // i32 result in a gpr
                try ctx.put(allocator, if (isDouble(func, cv.value)) encode.cvttsd2si(rd, src) else encode.cvttss2si(rd, src));
                try ctx.store(allocator, result, rd);
            } else if (src_float and dst_float) {
                // f32 <-> f64: widen or narrow (a same-width float convert is just a copy).
                const src = try ctx.useXmm(allocator, cv.value, xmm_op0);
                const rd = try ctx.dstXmm(result, xmm_scratch);
                const sd = isDouble(func, cv.value);
                const dd = isDouble(func, result);
                if (sd == dd) {
                    if (rd != src) try ctx.put(allocator, encode.movupsRR(rd, src));
                } else {
                    try ctx.put(allocator, if (dd) encode.cvtss2sd(rd, src) else encode.cvtsd2ss(rd, src));
                }
                try ctx.storeXmm(allocator, result, rd);
            } else {
                // int <-> int: low bits carry over (any width change is the consumer's view).
                const src = try ctx.use(allocator, cv.value, scratch2);
                const rd = ctx.dst(result, scratch1);
                if (rd != src) try ctx.put(allocator, encode.movReg(rd, src));
                try ctx.store(allocator, result, rd);
            }
        },
        .unary => |u| {
            const result_val = func.instResult(inst);
            const src_float = isFloat(func, u.value);
            if (u.op == .reinterpret) {
                // int <-> float reinterpret: same width, movd/movq between gpr and xmm.
                if (src_float) {
                    // float -> int: xmm -> gpr.
                    const src = try ctx.useXmm(allocator, u.value, xmm_op0);
                    if (result_val) |res| {
                        const rd = ctx.dst(res, scratch1);
                        try ctx.put(allocator, if (isDouble(func, u.value)) encode.movqFromXmm(rd, src) else encode.movdFromXmm(rd, src));
                        try ctx.store(allocator, res, rd);
                    }
                    // void result: src already loaded, nothing to do.
                } else {
                    // int -> float: gpr -> xmm.
                    const src = try ctx.use(allocator, u.value, scratch1);
                    if (result_val) |res| {
                        const rd = try ctx.dstXmm(res, xmm_scratch);
                        try ctx.put(allocator, if (isDouble(func, res)) encode.movqToXmm(rd, src) else encode.movdToXmm(rd, src));
                        try ctx.storeXmm(allocator, res, rd);
                    }
                    // void result: src already loaded, nothing to do.
                }
                return;
            }
            // Vector math unary: the packed forms so every lane is computed, not just lane 0.
            if (isVector(func, result_val.?)) {
                const src = try ctx.useXmm(allocator, u.value, xmm_op0);
                const rd = try ctx.dstXmm(result_val.?, xmm_scratch);
                try ctx.put(allocator, switch (u.op) {
                    .sqrt => encode.sqrtps(rd, src),
                    .ceil => encode.roundps(rd, src, 2),
                    .floor => encode.roundps(rd, src, 1),
                    .trunc => encode.roundps(rd, src, 3),
                    .nearest => encode.roundps(rd, src, 0),
                    .reinterpret => unreachable,
                });
                try ctx.storeXmm(allocator, result_val.?, rd);
                return;
            }
            // Float math unary ops: sqrt, ceil, floor, trunc, nearest. All live in xmm.
            const src = try ctx.useXmm(allocator, u.value, xmm_op0);
            const rd = try ctx.dstXmm(result_val.?, xmm_scratch);
            const dbl = isDouble(func, u.value);
            try ctx.put(allocator, switch (u.op) {
                .sqrt => if (dbl) encode.sqrtsd(rd, src) else encode.sqrtss(rd, src),
                .ceil => if (dbl) encode.roundsd(rd, src, 2) else encode.roundss(rd, src, 2),
                .floor => if (dbl) encode.roundsd(rd, src, 1) else encode.roundss(rd, src, 1),
                .trunc => if (dbl) encode.roundsd(rd, src, 3) else encode.roundss(rd, src, 3),
                .nearest => if (dbl) encode.roundsd(rd, src, 0) else encode.roundss(rd, src, 0),
                .reinterpret => unreachable,
            });
            try ctx.storeXmm(allocator, result_val.?, rd);
        },
        .call => |c| {
            // Move arguments into the System V argument registers (register sources through a
            // parallel move, spilled sources reloaded straight into the arg register), then
            // `call` the symbol via a relocation. The result is in RAX. Caller-saved registers
            // are clobbered. Values must not be live across the call other than the result.
            // General args go in rdi,rsi,... and fp args in xmm0,xmm1,..., each class has its
            // own incoming-register index.
            const args = func.valueList(c.args);
            var moves: std.ArrayList(Move) = .empty;
            defer moves.deinit(allocator);
            var xmm_moves: std.ArrayList(XmmMove) = .empty;
            defer xmm_moves.deinit(allocator);
            var gi: usize = 0;
            var xi: usize = 0;
            for (args) |arg| {
                if (isXmm(func, arg)) {
                    if (xi >= xmm_arg_regs.len) return error.Unsupported; // fp stack args not handled
                    const dst = xmm_arg_regs[xi];
                    xi += 1;
                    switch (ctx.loc(arg)) {
                        .xmm => |src| if (src != dst) try xmm_moves.append(allocator, .{ .src = src, .dst = dst, .wide = isWide(func, arg) }),
                        .xmm_spill => {}, // reloaded after the parallel move
                        else => unreachable,
                    }
                } else {
                    if (gi >= arg_regs.len) return error.Unsupported;
                    const dst = arg_regs[gi];
                    gi += 1;
                    switch (ctx.loc(arg)) {
                        .reg => |src| if (src != dst) try moves.append(allocator, .{ .src = src, .dst = dst }),
                        .spill => {},
                        else => unreachable,
                    }
                }
            }
            try parallelMove(allocator, ctx, &moves);
            try parallelMoveXmm(allocator, ctx, &xmm_moves);
            gi = 0;
            xi = 0;
            for (args) |arg| {
                if (isXmm(func, arg)) {
                    const dst = xmm_arg_regs[xi];
                    xi += 1;
                    if (ctx.loc(arg) == .xmm_spill) {
                        const disp = ctx.xmmDisp(ctx.loc(arg).xmm_spill);
                        try ctx.put(allocator, if (isWide(func, arg)) encode.vmovupsLoad(dst, disp) else if (isVector(func, arg)) encode.movupsLoad(dst, disp) else encode.movssLoad(dst, disp));
                    }
                } else {
                    const dst = arg_regs[gi];
                    gi += 1;
                    if (ctx.loc(arg) == .spill) try ctx.put(allocator, encode.movFromStack(dst, slotDisp(ctx.loc(arg).spill)));
                }
            }
            try ctx.put(allocator, encode.callRel(0));
            try ctx.relocs.append(allocator, .{ .offset = ctx.code.items.len - 4, .symbol = func.symbolName(c.symbol) });
            if (isXmm(func, result)) {
                const rd = try ctx.dstXmm(result, xmm_scratch); // fp result comes back in xmm0
                if (rd != xmm_ret) try ctx.put(allocator, if (isWide(func, result)) encode.vmovupsRR(rd, xmm_ret) else encode.movupsRR(rd, xmm_ret));
                try ctx.storeXmm(allocator, result, rd);
            } else {
                const rd = ctx.dst(result, scratch1);
                if (rd != .rax) try ctx.put(allocator, encode.movReg(rd, .rax));
                try ctx.store(allocator, result, rd);
            }
        },
        .struct_new => |sn| {
            // Build a SIMD vector from scalar lanes (the vectorizer's pack): one insertps per
            // lane, lane 0 last so a field the allocator placed in the result register keeps
            // its value (in lane 0) until its own insert reads it.
            if (!isVector(func, result)) return error.Unsupported;
            const fields = func.valueList(sn.fields);
            const rd = try ctx.dstXmm(result, xmm_scratch); // vectors do not spill (rd is a register)
            if (isWide(func, result)) {
                // AVX <8 x f32>: build the two 128-bit halves with insertps (low in op0, high
                // in op1), then join them into the 256-bit result with vinsertf128.
                if (fields.len != 8) return error.Unsupported;
                for (0..4) |lane| {
                    const fr = try ctx.useXmm(allocator, fields[lane], xmm_scratch);
                    try ctx.put(allocator, encode.insertps(xmm_op0, fr, @as(u8, @intCast(lane)) << 4));
                }
                for (0..4) |lane| {
                    const fr = try ctx.useXmm(allocator, fields[4 + lane], xmm_scratch);
                    try ctx.put(allocator, encode.insertps(xmm_op1, fr, @as(u8, @intCast(lane)) << 4));
                }
                try ctx.put(allocator, encode.vinsertf128(rd, xmm_op0, xmm_op1, 1)); // rd = [hi:lo]
                try ctx.storeXmm(allocator, result, rd);
                return;
            }
            if (fields.len != 4) return error.Unsupported; // <4 x f32>
            for ([_]u8{ 1, 2, 3, 0 }) |lane| {
                const fr = try ctx.useXmm(allocator, fields[lane], xmm_op0); // a spilled field reloads to op0
                try ctx.put(allocator, encode.insertps(rd, fr, lane << 4)); // src lane 0 -> dst lane
            }
            try ctx.storeXmm(allocator, result, rd);
        },
        .extract => |e| {
            // Extract a lane of a SIMD vector to a scalar (the vectorizer's unpack): a single
            // pshufd moving that lane to lane 0, so it is pure and dead extracts fall to DCE.
            if (!isVector(func, e.aggregate)) return error.Unsupported;
            const src = try ctx.useXmm(allocator, e.aggregate, xmm_op0);
            const rd = try ctx.dstXmm(result, xmm_scratch);
            if (isWide(func, e.aggregate) and e.index >= 4) {
                // The lane is in the high 128 bits of the ymm: bring that half down to an xmm
                // first, then shuffle the lane (relative to the half) into lane 0.
                try ctx.put(allocator, encode.vextractf128(xmm_op1, src, 1));
                try ctx.put(allocator, encode.pshufd(rd, xmm_op1, @intCast(e.index - 4)));
            } else {
                // Lane 0..3 lives in the low 128 bits, which a 128-bit pshufd reads directly.
                try ctx.put(allocator, encode.pshufd(rd, src, @intCast(e.index)));
            }
            try ctx.storeXmm(allocator, result, rd);
        },
        .alloca => {
            // The alloca's result is the address of its stack slot: lea it from rsp. The slot
            // offset was assigned in `computeAllocaSlots`. The result is a pointer (gpr).
            const off = ctx.alloca_base + @as(i32, @intCast(ctx.alloca_off.get(result).?));
            const rd = ctx.dst(result, scratch1);
            try ctx.put(allocator, encode.leaFromStack(rd, off));
            try ctx.store(allocator, result, rd);
        },
        .load => |l| {
            // The pointer operand is a general-register value, load through `[base + 0]`. An
            // xmm result uses movups (vector) / movsd (f64) / movss (f32), else a general mov.
            const base = try ctx.use(allocator, l.ptr, scratch2);
            if (isXmm(func, result)) {
                const rd = try ctx.dstXmm(result, xmm_scratch);
                try ctx.put(allocator, if (isVector(func, result)) encode.movupsLoadMem(rd, base, 0) else if (isDouble(func, result)) encode.movsdLoadMem(rd, base, 0) else encode.movssLoadMem(rd, base, 0));
                try ctx.storeXmm(allocator, result, rd);
            } else {
                const rd = ctx.dst(result, scratch1);
                // Load the value's own width: a 32-bit int reads 4 bytes (an 8-byte load would
                // pull garbage from the next array element into the upper half, which then
                // breaks a 64-bit compare). Sign-extend a signed i32, zero-extend otherwise.
                if (intBits(func, result) <= 32) {
                    try ctx.put(allocator, if (isSigned(func, result)) encode.movsxdFromMem(rd, base, 0) else encode.movFromMem32(rd, base, 0));
                } else {
                    try ctx.put(allocator, encode.movFromMem(rd, base, 0));
                }
                try ctx.store(allocator, result, rd);
            }
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
const XmmMove = struct { src: Xmm, dst: Xmm, wide: bool };

/// Edge moves into the target block's parameter locations, for both gpr and xmm parameters.
/// Register-to-register moves go through a parallel move (per class), spilled args/params
/// are reloaded/stored via the scratch register around the parallel move, so register
/// sources are read before they are overwritten.
fn emitMoves(allocator: std.mem.Allocator, ctx: *Ctx, jump: ir.function.Jump) Error!void {
    const func = ctx.func;
    const args = func.blockArgs(jump);
    const params = func.blockParams(jump.target);
    if (args.len != params.len) return error.Unsupported;

    var moves: std.ArrayList(Move) = .empty;
    defer moves.deinit(allocator);
    var xmm_moves: std.ArrayList(XmmMove) = .empty;
    defer xmm_moves.deinit(allocator);
    // First: stores into spilled parameters read their register sources now.
    for (args, params) |arg, param| {
        switch (ctx.loc(param)) {
            .spill => |slot| {
                const src = try ctx.use(allocator, arg, scratch1);
                try ctx.put(allocator, encode.movToStack(slotDisp(slot), src));
            },
            .xmm_spill => |slot| {
                const src = try ctx.useXmm(allocator, arg, xmm_scratch);
                const disp = ctx.xmmDisp(slot);
                try ctx.put(allocator, if (isWide(func, param)) encode.vmovupsStore(disp, src) else if (isVector(func, param)) encode.movupsStore(disp, src) else encode.movssStore(disp, src));
            },
            .reg => |dst| switch (ctx.loc(arg)) {
                .reg => |src| if (src != dst) try moves.append(allocator, .{ .src = src, .dst = dst }),
                .spill => {}, // reloaded after the parallel move
                .xmm, .xmm_spill => unreachable,
            },
            .xmm => |dst| switch (ctx.loc(arg)) {
                .xmm => |src| if (src != dst) try xmm_moves.append(allocator, .{ .src = src, .dst = dst, .wide = isWide(func, param) }),
                .xmm_spill => {}, // reloaded after the parallel move
                .reg, .spill => unreachable,
            },
        }
    }
    try parallelMove(allocator, ctx, &moves);
    try parallelMoveXmm(allocator, ctx, &xmm_moves);
    // Then: reloads of spilled arguments into register parameters.
    for (args, params) |arg, param| {
        switch (ctx.loc(param)) {
            .reg => |dst| if (ctx.loc(arg) == .spill) try ctx.put(allocator, encode.movFromStack(dst, slotDisp(ctx.loc(arg).spill))),
            .xmm => |dst| if (ctx.loc(arg) == .xmm_spill) {
                const disp = ctx.xmmDisp(ctx.loc(arg).xmm_spill);
                try ctx.put(allocator, if (isWide(func, param)) encode.vmovupsLoad(dst, disp) else if (isVector(func, param)) encode.movupsLoad(dst, disp) else encode.movssLoad(dst, disp));
            },
            else => {},
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

/// The xmm counterpart of parallelMove. A 256-bit (AVX) move copies the whole ymm with
/// vmovups. A 128-bit move uses movups (which also covers a scalar float / double / <4 x f32>
/// without truncation). The cycle break saves through xmm_scratch at 256 bits whenever any
/// move in the batch is wide, so a wide register parked in the scratch keeps all its lanes.
fn parallelMoveXmm(allocator: std.mem.Allocator, ctx: *Ctx, moves: *std.ArrayList(XmmMove)) Error!void {
    var any_wide = false;
    for (moves.items) |m| {
        if (m.wide) any_wide = true;
    }
    while (moves.items.len > 0) {
        var emitted = false;
        for (moves.items, 0..) |m, i| {
            var blocked = false;
            for (moves.items, 0..) |o, j| if (i != j and o.src == m.dst) {
                blocked = true;
            };
            if (!blocked) {
                try ctx.put(allocator, if (m.wide) encode.vmovupsRR(m.dst, m.src) else encode.movupsRR(m.dst, m.src));
                _ = moves.swapRemove(i);
                emitted = true;
                break;
            }
        }
        if (!emitted) {
            const dst0 = moves.items[0].dst;
            try ctx.put(allocator, if (any_wide) encode.vmovupsRR(xmm_scratch, dst0) else encode.movupsRR(xmm_scratch, dst0));
            for (moves.items) |*m| if (m.src == dst0) {
                m.src = xmm_scratch;
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

/// Emit `result = value` (a register move plus a spill store if needed): one arm of a
/// select. Touches only `result` and `value`, so it stays correct when those share a
/// register.
fn selectInto(allocator: std.mem.Allocator, ctx: *Ctx, result: Value, value: Value) Error!void {
    const func = ctx.func;
    if (isXmm(func, result)) {
        const rv = try ctx.useXmm(allocator, value, xmm_op0);
        const rd = try ctx.dstXmm(result, xmm_scratch);
        if (rd != rv) try ctx.put(allocator, encode.movupsRR(rd, rv));
        try ctx.storeXmm(allocator, result, rd);
    } else {
        const rv = try ctx.use(allocator, value, scratch2);
        const rd = ctx.dst(result, scratch1);
        if (rd != rv) try ctx.put(allocator, encode.movReg(rd, rv));
        try ctx.store(allocator, result, rd);
    }
}

fn isSigned(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.signedness == .signed,
        else => true,
    };
}

/// Which fixed registers the function needs (kept out of the general pool).
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

/// The instruction positions of `call`s, in the same linearization regalloc uses for
/// intervals: block params share the block's start position. Each instruction and each
/// terminator advance the position by one.
fn callPositions(allocator: std.mem.Allocator, func: *const Function) Error![]u32 {
    var positions: std.ArrayList(u32) = .empty;
    errdefer positions.deinit(allocator);
    var pos: u32 = 0;
    for (0..func.blockCount()) |bi| {
        pos += 1; // block parameters
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .call or func.opcode(inst) == .call_indirect) try positions.append(allocator, pos);
            pos += 1;
        }
        pos += 1; // terminator
    }
    return positions.toOwnedSlice(allocator);
}

/// Linear-scan register allocation with reuse and spilling. Entry parameters are not pinned,
/// the prologue moves arguments into their assigned locations. R10/R11 are reserved as
/// spill/move scratch. RAX/RDX/RCX are reserved for division/shifts.
fn assignRegs(allocator: std.mem.Allocator, func: *const Function, loc_of: *std.AutoHashMapUnmanaged(Value, Loc), num_slots: *u32) Error!void {
    { // each class of entry parameter must fit in its ABI argument registers (no stack args)
        var gpr_params: usize = 0;
        var xmm_params: usize = 0;
        for (func.blockParams(@enumFromInt(0))) |p| {
            if (isXmm(func, p)) xmm_params += 1 else gpr_params += 1;
        }
        if (gpr_params > arg_regs.len or xmm_params > xmm_arg_regs.len) return error.Unsupported;
    }

    const needs = fixedRegNeeds(func);
    var pool: std.ArrayList(Reg) = .empty;
    defer pool.deinit(allocator);
    const candidates = [_]Reg{ .rax, .rcx, .rdx, .rsi, .rdi, .r8, .r9 }; // R10/R11 are scratch
    for (candidates) |r| {
        if (needs.div and (r == .rax or r == .rdx)) continue;
        if (needs.shift and r == .rcx) continue;
        try pool.append(allocator, r);
    }

    const ivals = try regalloc.computeLiveIntervals(allocator, func);
    defer allocator.free(ivals);

    // A `call` clobbers every caller-saved register, so a value live across a call (defined
    // before it, used after it) cannot stay in a register, force it to a spill slot. Its call
    // arguments and the call result are not "across".
    const calls = try callPositions(allocator, func);
    defer allocator.free(calls);

    const Active = struct { end: u32, value: Value, reg: Reg };
    var active: std.ArrayList(Active) = .empty;
    defer active.deinit(allocator);
    var free = try pool.clone(allocator);
    defer free.deinit(allocator);

    for (ivals) |iv| {
        if (isXmm(func, iv.value)) continue; // fp/vector values are allocated by `assignXmm`
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
            // Spill the longest-living of {this interval, the active intervals}.
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

/// Linear-scan allocation of the fp/vector (xmm) values, parallel to assignRegs. xmm0..xmm12
/// are allocatable (xmm13/14/15 are reserved scratch). A value that does not fit a register
/// (pressure, or live across a caller-clobbering call) spills to a 16-byte slot (movss for a
/// scalar, movups for a whole vector). `xmm_slots` receives the slots used.
fn assignXmm(allocator: std.mem.Allocator, func: *const Function, loc_of: *std.AutoHashMapUnmanaged(Value, Loc), xmm_slots: *u32) Error!void {
    const ivals = try regalloc.computeLiveIntervals(allocator, func);
    defer allocator.free(ivals);
    const calls = try callPositions(allocator, func);
    defer allocator.free(calls);

    const Active = struct { end: u32, value: Value, reg: Xmm };
    var active: std.ArrayList(Active) = .empty;
    defer active.deinit(allocator);
    var free: std.ArrayList(Xmm) = .empty;
    defer free.deinit(allocator);
    var k: usize = 13;
    while (k > 0) : (k -= 1) try free.append(allocator, @enumFromInt(@as(u4, @intCast(k - 1)))); // pop gives xmm0 first

    for (ivals) |iv| {
        if (!isXmm(func, iv.value)) continue;
        var w: usize = 0;
        for (active.items) |act| {
            if (act.end < iv.start) try free.append(allocator, act.reg) else {
                active.items[w] = act;
                w += 1;
            }
        }
        active.shrinkRetainingCapacity(w);

        var across = false;
        for (calls) |cp| if (iv.start < cp and cp < iv.end) {
            across = true;
        };
        const spill = across or free.items.len == 0;
        if (spill) {
            // A scalar uses an 8-byte movss slot's worth, a vector uses the whole 16-byte
            // slot. Both share the 16-byte-slot xmm spill area (movss vs movups by type).
            try loc_of.put(allocator, iv.value, .{ .xmm_spill = xmm_slots.* });
            xmm_slots.* += 1;
        } else {
            const r = free.pop().?;
            try loc_of.put(allocator, iv.value, .{ .xmm = r });
            try active.append(allocator, .{ .end = iv.end, .value = iv.value, .reg = r });
        }
    }
}

/// Lay out the alloca region: each `alloca` result gets a naturally-aligned byte offset
/// (relative to the region base), recorded in `map`. Returns the region's size.
fn computeAllocaSlots(allocator: std.mem.Allocator, func: *const Function, map: *std.AutoHashMapUnmanaged(Value, u32)) Error!u32 {
    var cur: u32 = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .alloca => |al| {
                    cur = alignUp(cur, typeAlign(func, al.elem));
                    try map.put(allocator, func.instResult(inst).?, cur);
                    cur += typeSize(func, al.elem);
                },
                else => {},
            }
        }
    }
    return cur;
}

fn alignUp(v: u32, a: u32) u32 {
    return (v + a - 1) & ~(a - 1);
}

/// The storage size of a type in bytes (for sizing an alloca slot).
fn typeSize(func: *const Function, ty: ir.types.Type) u32 {
    return switch (func.types.type_kind(ty)) {
        .bool => 1,
        .int => |i| (@as(u32, i.bits) + 7) / 8,
        .ptr => 8,
        .float => |f| if (f == .f32) 4 else 8,
        .array => |a| @as(u32, @intCast(a.len)) * typeSize(func, a.elem),
        .vector => |v| @as(u32, v.len) * typeSize(func, v.elem),
        else => 8,
    };
}

/// The natural alignment of a type's storage (for alloca slot placement).
fn typeAlign(func: *const Function, ty: ir.types.Type) u32 {
    const sz = switch (func.types.type_kind(ty)) {
        .array => |a| typeSize(func, a.elem), // align an array to its element
        else => typeSize(func, ty),
    };
    return if (sz <= 1) 1 else if (sz <= 2) 2 else if (sz <= 4) 4 else if (sz <= 8) 8 else 16;
}

test "selects a scalar float function (SSE, xmm allocation)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    const p = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = s, .rhs = x } });
    func.setTerminator(b, .{ .ret = p });
    const code = try selectFunction(allocator, &func);
    defer allocator.free(code);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]); // ends in ret
    var addss = false;
    var mulss = false;
    for (0..code.len - 1) |i| {
        if (code[i] == 0x0F and code[i + 1] == 0x58) addss = true; // addss opcode (any prefix)
        if (code[i] == 0x0F and code[i + 1] == 0x59) mulss = true; // mulss opcode
    }
    try std.testing.expect(addss and mulss); // the float add/mul lowered to SSE
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
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });
    const code = try selectFunction(allocator, &func);
    defer allocator.free(code);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]); // ends in ret
}
