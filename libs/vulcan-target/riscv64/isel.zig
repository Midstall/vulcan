//! RISC-V instruction selection and register allocation. Lowers a low-profile
//! Vulcan function to machine words: integer/float/RVV-vector arithmetic,
//! control flow, calls, memory, with a liveness-based linear-scan allocator and
//! stack spilling.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const emit = @import("emit.zig");
const schedule = @import("schedule.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const BinOp = ir.function.BinOp;
const Reg = encode.Reg;
const FReg = encode.FReg;

/// Register-file assignment. A value lives in an int, float, or vector register,
/// or (when the file is exhausted) spills to a numbered stack slot.
const Allocation = struct {
    int: std.AutoHashMapUnmanaged(Value, Reg),
    float: std.AutoHashMapUnmanaged(Value, FReg),
    /// SIMD vector values, mapped to an RVV vector register (v1..v27).
    vector: std.AutoHashMapUnmanaged(Value, VReg),
    /// Spilled vector values, mapped to a 16-byte spill-slot index (0-based).
    vector_spill: std.AutoHashMapUnmanaged(Value, u32),
    vector_spill_count: u32,
    /// Spilled integer values, mapped to their spill-slot index (0-based). The
    /// frame layout turns the index into an `sp` offset.
    int_spill: std.AutoHashMapUnmanaged(Value, u32),
    spill_count: u32,
    /// Entry integer parameters beyond the 8 argument registers: each maps to its
    /// incoming stack-argument index (0 = the 9th arg). The selector loads it from
    /// the caller's frame at function entry.
    incoming_stack: std.AutoHashMapUnmanaged(Value, u32),

    fn deinit(self: *Allocation, allocator: std.mem.Allocator) void {
        self.int.deinit(allocator);
        self.float.deinit(allocator);
        self.vector.deinit(allocator);
        self.vector_spill.deinit(allocator);
        self.int_spill.deinit(allocator);
        self.incoming_stack.deinit(allocator);
    }
};

const VReg = encode.VReg;
// v0 is the RVV mask register. The top four vector registers are reserved scratch:
// v28/v29 reload spilled left/right operands, v30 holds a spilled result, v31 is
// the slide-based pack/extract scratch. v1..v27 is the allocatable pool.
const vec_op0: VReg = .v28;
const vec_op1: VReg = .v29;
const vec_work: VReg = .v30;
const vector_scratch: VReg = .v31;

/// Scratch registers for reloading/storing spilled integer values. x6 is the
/// general scratch. x8 (fp, which Vulcan does not use) is the second so a binary
/// op with two spilled operands can reload both.
const spill_scratch0: Reg = .x6;
const spill_scratch1: Reg = .x8;

/// Caller-saved float temporaries (ft0-ft7, ft8-ft11).
const float_temp_regs = [_]FReg{ .f0, .f1, .f2, .f3, .f4, .f5, .f6, .f7, .f28, .f29, .f30, .f31 };

/// Callee-saved float registers (fs0-fs11). Drawn after the caller-saved float
/// temporaries. Each used is preserved in the frame with fsd/fld.
const float_saved_regs = [_]FReg{ .f8, .f9, .f18, .f19, .f20, .f21, .f22, .f23, .f24, .f25, .f26, .f27 };

/// Allocatable RVV vector registers (v1..v27, where v0 is the mask register and v28..v31 are reserved
/// scratch). All vector registers are caller-saved.
const vector_regs = [_]VReg{
    .v1,  .v2,  .v3,  .v4,  .v5,  .v6,  .v7,  .v8,  .v9,  .v10,
    .v11, .v12, .v13, .v14, .v15, .v16, .v17, .v18, .v19, .v20,
    .v21, .v22, .v23, .v24, .v25, .v26, .v27,
};

fn isFloatSavedReg(reg: FReg) bool {
    for (float_saved_regs) |s| {
        if (s == reg) return true;
    }
    return false;
}

/// Float argument register `i`: fa0 = f10, fa1 = f11, ...
fn fargReg(i: usize) FReg {
    return @enumFromInt(@as(u5, @intCast(10 + i)));
}

fn isFloat(func: *const Function, ty: ir.types.Type) bool {
    return func.types.type_kind(ty) == .float;
}

fn isVector(func: *const Function, ty: ir.types.Type) bool {
    return func.types.type_kind(ty) == .vector;
}

fn is64Float(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .float => |f| f == .f64,
        else => false,
    };
}

fn isFloatTempReg(reg: FReg) bool {
    for (float_temp_regs) |t| {
        if (t == reg) return true;
    }
    return false;
}

pub const Error = std.mem.Allocator.Error || error{Unsupported};

/// A branch/jump whose target offset is patched once block positions are known.
const Fixup = struct {
    index: usize,
    target: Block,
    kind: union(enum) { branch: Reg, jal },
};

/// The temporary registers used for instruction results. x6 is reserved as a
/// scratch register for helper sequences (e.g. materializing float constants).
const temp_regs = [_]Reg{ .x5, .x7, .x28, .x29, .x30, .x31 };
const scratch_reg: Reg = .x6;

/// Callee-saved integer registers (s1, s2-s11). Drawn only after the caller-saved
/// temporaries are exhausted. Each one actually used is saved/restored in the
/// frame. x8 (s0/fp) is left reserved.
const saved_regs = [_]Reg{ .x9, .x18, .x19, .x20, .x21, .x22, .x23, .x24, .x25, .x26, .x27 };

fn isSavedReg(reg: Reg) bool {
    for (saved_regs) |s| {
        if (s == reg) return true;
    }
    return false;
}

/// Resolve an integer operand to a register: if `v` lives in a register, return
/// it. If it was spilled, reload it from its stack slot into `scratch` and return
/// `scratch`. Spilled values occupy a full 8-byte slot.
fn reloadInt(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, spill_base: u32, v: Value, scratch: Reg) std.mem.Allocator.Error!Reg {
    if (alloc.int.get(v)) |r| return r;
    const idx = alloc.int_spill.get(v).?;
    const off: i12 = @intCast(spill_base + idx * 8);
    try code.append(allocator, encode.ld(scratch, .x2, off));
    return scratch;
}

/// Reload a vector `v` into `scratch` if it was spilled (vle32 from its 16-byte slot, whose
/// address is computed into `addr`), else return its assigned vector register.
fn reloadVector(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, vspill_base: u32, v: Value, scratch: VReg, addr: Reg) std.mem.Allocator.Error!VReg {
    if (alloc.vector.get(v)) |vr| return vr;
    const off: i12 = @intCast(vspill_base + alloc.vector_spill.get(v).? * 16);
    try code.append(allocator, encode.addi(addr, .x2, off));
    try code.append(allocator, encode.vle32(scratch, addr));
    return scratch;
}
/// The vector register to compute `v` into: its assigned register, or `scratch` if spilled.
fn dstVector(alloc: *const Allocation, v: Value, scratch: VReg) VReg {
    return alloc.vector.get(v) orelse scratch;
}
/// Store a freshly-computed vector `v` (in `vr`) back to its spill slot, if it was spilled.
fn storeVector(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, vspill_base: u32, v: Value, vr: VReg, addr: Reg) std.mem.Allocator.Error!void {
    if (alloc.vector.get(v) != null) return;
    const off: i12 = @intCast(vspill_base + alloc.vector_spill.get(v).? * 16);
    try code.append(allocator, encode.addi(addr, .x2, off));
    try code.append(allocator, encode.vse32(vr, addr));
}

/// Materialize a 32-bit value into integer register `rd`.
fn loadImm32(allocator: std.mem.Allocator, code: *std.ArrayList(u32), rd: Reg, bits: u32) std.mem.Allocator.Error!void {
    const signed: i32 = @bitCast(bits);
    if (signed >= -2048 and signed <= 2047) {
        try code.append(allocator, encode.addi(rd, .x0, @intCast(signed)));
    } else {
        const hi: u20 = @truncate((bits +% 0x800) >> 12);
        const lo: i12 = @bitCast(@as(u12, @truncate(bits)));
        try code.append(allocator, encode.lui(rd, hi));
        try code.append(allocator, encode.addi(rd, rd, lo));
    }
}

/// Round `x` up to a multiple of `a` (a power of two).
fn alignUp(x: u32, a: u32) u32 {
    return (x + a - 1) & ~(a - 1);
}

/// Size in bytes of a stack slot for `ty`. Aggregates are not yet supported.
fn typeSize(func: *const Function, ty: ir.types.Type) Error!u32 {
    return switch (func.types.type_kind(ty)) {
        .bool => 1,
        .int => |i| (@as(u32, i.bits) + 7) / 8,
        .float => |f| switch (f) {
            .f32 => 4,
            .f64 => 8,
        },
        .ptr => 8,
        .vector => |v| v.len * try typeSize(func, v.elem),
        else => error.Unsupported,
    };
}

/// A register-to-register move, used for shuffling call arguments into place.
const RegMove = struct { src: Reg, dst: Reg };

/// Emit integer register moves in an order that respects conflicts (one move's
/// destination being another's source), breaking cycles with `scratch`. Needed
/// so e.g. swapping two argument registers is correct.
fn parallelMoveInt(allocator: std.mem.Allocator, code: *std.ArrayList(u32), moves_in: []const RegMove, scratch: Reg) std.mem.Allocator.Error!void {
    var moves: std.ArrayList(RegMove) = .empty;
    defer moves.deinit(allocator);
    for (moves_in) |m| if (m.src != m.dst) try moves.append(allocator, m);

    while (moves.items.len > 0) {
        var emitted = false;
        var i: usize = 0;
        while (i < moves.items.len) {
            const m = moves.items[i];
            var is_src = false;
            for (moves.items, 0..) |o, j| {
                if (j != i and o.src == m.dst) {
                    is_src = true;
                    break;
                }
            }
            if (is_src) {
                i += 1;
            } else {
                try code.append(allocator, encode.addi(m.dst, m.src, 0)); // mv dst, src
                _ = moves.swapRemove(i);
                emitted = true;
            }
        }
        if (!emitted) {
            // Remaining moves form one or more cycles. Break by saving a
            // destination into the scratch and redirecting reads of it.
            const m = moves.items[0];
            try code.append(allocator, encode.addi(scratch, m.dst, 0)); // save dst
            for (moves.items) |*o| {
                if (o.src == m.dst) o.src = scratch;
            }
            try code.append(allocator, encode.addi(m.dst, m.src, 0)); // mv dst, src
            _ = moves.orderedRemove(0);
        }
    }
}

/// Integer argument register `i`: a0 = x10, a1 = x11, ...
fn argReg(i: usize) Reg {
    return @enumFromInt(@as(u5, @intCast(10 + i)));
}

/// Split critical edges: for each `if` edge that carries arguments, insert a
/// landing block that jumps to the original target with those arguments, and
/// point the `if` edge at the (now arg-free) landing block. This reuses the
/// jump-edge block-argument lowering instead of needing per-edge moves.
pub fn splitCriticalEdges(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!void {
    const original = func.blockCount();
    for (0..original) |bi| {
        const block: Block = @enumFromInt(bi);
        const insts = try allocator.dupe(ir.function.Inst, func.blockInsts(block));
        defer allocator.free(insts);

        for (insts) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => {
                    try splitEdge(allocator, func, inst, .then);
                    try splitEdge(allocator, func, inst, .@"else");
                },
                else => {},
            }
        }
    }
}

const Side = enum { then, @"else" };

fn splitEdge(allocator: std.mem.Allocator, func: *Function, inst: ir.function.Inst, side: Side) std.mem.Allocator.Error!void {
    const edge = switch (side) {
        .then => func.opcode(inst).@"if".then,
        .@"else" => func.opcode(inst).@"if".@"else",
    };
    const edge_args = func.blockArgs(edge);
    if (edge_args.len == 0) return;

    const args = try allocator.dupe(Value, edge_args);
    defer allocator.free(args);

    const landing = try func.appendBlock();
    try func.setJump(landing, edge.target, args);

    const empty = try func.internValueList(&.{});
    const op = func.opcodeMut(inst);
    switch (side) {
        .then => op.@"if".then = .{ .target = landing, .args = empty },
        .@"else" => op.@"if".@"else" = .{ .target = landing, .args = empty },
    }
}

/// Whether a type is an unsigned integer (so comparisons use `sltu`).
fn isUnsignedInt(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .int => |i| i.signedness == .unsigned,
        else => false,
    };
}

/// Whether a type loads/stores in 32 bits or fewer (vs a 64-bit doubleword).
fn isWord(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .int => |i| i.bits <= 32,
        .bool => true,
        else => false,
    };
}

fn isSignedInt(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .int => |i| i.signedness == .signed,
        else => false,
    };
}

/// Bit width of an integer-like type for choosing a load/store width.
fn intBits(func: *const Function, ty: ir.types.Type) u16 {
    return switch (func.types.type_kind(ty)) {
        .int => |i| i.bits,
        .bool => 1,
        else => 64, // ptr and the like
    };
}

/// The load instruction for an integer-like value of `ty`: byte/halfword loads
/// sign- or zero-extend by signedness. Word and doubleword as usual.
fn intLoadInsn(func: *const Function, ty: ir.types.Type, rd: Reg, base: Reg, off: i12) u32 {
    const signed = isSignedInt(func, ty);
    const bits = intBits(func, ty);
    if (bits <= 8) return if (signed) encode.lb(rd, base, off) else encode.lbu(rd, base, off);
    if (bits <= 16) return if (signed) encode.lh(rd, base, off) else encode.lhu(rd, base, off);
    if (bits <= 32) return if (signed) encode.lw(rd, base, off) else encode.lwu(rd, base, off);
    return encode.ld(rd, base, off);
}

/// The store instruction for an integer-like value of `ty` (width only).
fn intStoreInsn(func: *const Function, ty: ir.types.Type, vr: Reg, base: Reg, off: i12) u32 {
    const bits = intBits(func, ty);
    if (bits <= 8) return encode.sb(vr, base, off);
    if (bits <= 16) return encode.sh(vr, base, off);
    if (bits <= 32) return encode.sw(vr, base, off);
    return encode.sd(vr, base, off);
}

/// True when `target` carries an `endian(big)` attribute. RISC-V is little-endian,
/// so a big-endian access needs a byte-swap to reach native order.
fn endianBig(func: *const Function, target: ir.function.AttrTarget) bool {
    var it = func.attributesOf(target);
    while (it.next()) |attr| switch (attr) {
        .endian => |e| return e == .big,
        else => {},
    };
    return false;
}

fn arithWord(op: BinOp, rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return switch (op) {
        .add => encode.add(rd, rs1, rs2),
        .sub => encode.sub(rd, rs1, rs2),
        .mul => encode.mul(rd, rs1, rs2),
        .div => encode.div(rd, rs1, rs2),
        .rem => encode.rem(rd, rs1, rs2),
        .bit_and => encode.and_(rd, rs1, rs2),
        .bit_or => encode.or_(rd, rs1, rs2),
        .bit_xor => encode.xor_(rd, rs1, rs2),
        .shl => encode.sll(rd, rs1, rs2),
        .shr => encode.sra(rd, rs1, rs2), // arithmetic (signed) by default
    };
}

fn isTempReg(reg: Reg) bool {
    for (temp_regs) |t| {
        if (t == reg) return true;
    }
    return false;
}

fn mark(allocator: std.mem.Allocator, last_use: *std.AutoHashMapUnmanaged(Value, usize), v: Value, pos: usize) std.mem.Allocator.Error!void {
    try last_use.put(allocator, v, pos);
}

/// Record the position `pos` as a use of each of an instruction's operands.
fn recordUses(allocator: std.mem.Allocator, func: *const Function, inst: ir.function.Inst, pos: usize, last_use: *std.AutoHashMapUnmanaged(Value, usize)) std.mem.Allocator.Error!void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .alloca, .global_addr => {},
        .arith => |a| {
            try mark(allocator, last_use, a.lhs, pos);
            try mark(allocator, last_use, a.rhs, pos);
        },
        .arith_imm => |a| try mark(allocator, last_use, a.lhs, pos),
        .icmp => |c| {
            try mark(allocator, last_use, c.lhs, pos);
            try mark(allocator, last_use, c.rhs, pos);
        },
        .select => |s| {
            try mark(allocator, last_use, s.cond, pos);
            try mark(allocator, last_use, s.then, pos);
            try mark(allocator, last_use, s.@"else", pos);
        },
        .load => |l| try mark(allocator, last_use, l.ptr, pos),
        .store => |st| {
            try mark(allocator, last_use, st.value, pos);
            try mark(allocator, last_use, st.ptr, pos);
        },
        .struct_new => |sn| for (func.valueList(sn.fields)) |v| try mark(allocator, last_use, v, pos),
        .call => |c| for (func.valueList(c.args)) |v| try mark(allocator, last_use, v, pos),
        .call_indirect => |c| {
            try mark(allocator, last_use, c.target, pos);
            for (func.valueList(c.args)) |v| try mark(allocator, last_use, v, pos);
        },
        .extract => |ex| try mark(allocator, last_use, ex.aggregate, pos),
        .convert => |cv| try mark(allocator, last_use, cv.value, pos),
        .unary => |u| try mark(allocator, last_use, u.value, pos),
        .@"if" => |cf| {
            try mark(allocator, last_use, cf.cond, pos);
            for (func.blockArgs(cf.then)) |v| try mark(allocator, last_use, v, pos);
            for (func.blockArgs(cf.@"else")) |v| try mark(allocator, last_use, v, pos);
        },
    }
}

fn recordTermUses(allocator: std.mem.Allocator, func: *const Function, block: Block, pos: usize, last_use: *std.AutoHashMapUnmanaged(Value, usize)) std.mem.Allocator.Error!void {
    if (func.terminator(block)) |term| switch (term) {
        .ret => |v| if (v) |vv| try mark(allocator, last_use, vv, pos),
        .jump => |j| for (func.blockArgs(j)) |v| try mark(allocator, last_use, v, pos),
    };
}

/// Return temp registers of dying values to the appropriate free list.
fn freeDying(allocator: std.mem.Allocator, func: *const Function, dying: []const Value, alloc: *const Allocation, int_free: *std.ArrayList(Reg), float_free: *std.ArrayList(FReg), vector_free: *std.ArrayList(VReg)) std.mem.Allocator.Error!void {
    for (dying) |v| {
        if (isVector(func, func.valueType(v))) {
            if (alloc.vector.get(v)) |r| try vector_free.append(allocator, r);
        } else if (isFloat(func, func.valueType(v))) {
            const r = alloc.float.get(v).?;
            if (isFloatTempReg(r) or isFloatSavedReg(r)) try float_free.append(allocator, r);
        } else {
            // A spilled value holds no register. Only registered values free one.
            if (alloc.int.get(v)) |r| {
                if (isTempReg(r) or isSavedReg(r)) try int_free.append(allocator, r);
            }
        }
    }
}

/// Draw a callee-saved register from a free list (highest index first, matching
/// the order in which they would normally be drawn). Used for values live across
/// a call, which the callee preserves. Returns null if none remain.
fn popSaved(free: *std.ArrayList(Reg)) ?Reg {
    var i = free.items.len;
    while (i > 0) {
        i -= 1;
        if (isSavedReg(free.items[i])) return free.orderedRemove(i);
    }
    return null;
}

fn popFloatSaved(free: *std.ArrayList(FReg)) ?FReg {
    var i = free.items.len;
    while (i > 0) {
        i -= 1;
        if (isFloatSavedReg(free.items[i])) return free.orderedRemove(i);
    }
    return null;
}

/// Assign a register to every value via a liveness-based linear scan over the
/// register files. Entry parameters are pre-colored to argument registers. Other
/// values draw from a temporary free list, reusing a register once its value
/// dies. A value live across a call is placed in a callee-saved register so the
/// call cannot clobber it.
fn allocateRegisters(allocator: std.mem.Allocator, func: *const Function) Error!Allocation {
    var alloc: Allocation = .{ .int = .empty, .float = .empty, .vector = .empty, .vector_spill = .empty, .vector_spill_count = 0, .int_spill = .empty, .spill_count = 0, .incoming_stack = .empty };
    errdefer alloc.deinit(allocator);

    // Liveness: the last position at which each value is used (walking forward,
    // a later use overwrites an earlier one).
    var last_use: std.AutoHashMapUnmanaged(Value, usize) = .empty;
    defer last_use.deinit(allocator);

    // Mark which positions hold a call, to identify values live across one.
    var is_call: std.ArrayList(bool) = .empty;
    defer is_call.deinit(allocator);

    var total: usize = 0;
    var pos: usize = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| try last_use.put(allocator, p, pos);
        try is_call.append(allocator, false);
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            try recordUses(allocator, func, inst, pos, &last_use);
            if (func.instResult(inst)) |r| {
                if (!last_use.contains(r)) try last_use.put(allocator, r, pos);
            }
            try is_call.append(allocator, func.opcode(inst) == .call);
            pos += 1;
        }
        try recordTermUses(allocator, func, block, pos, &last_use);
        try is_call.append(allocator, false);
        pos += 1;
    }
    total = pos;

    // call_prefix[p] = number of call positions strictly before p.
    const call_prefix = try allocator.alloc(usize, total + 1);
    defer allocator.free(call_prefix);
    call_prefix[0] = 0;
    for (0..total) |p| call_prefix[p + 1] = call_prefix[p] + @intFromBool(is_call.items[p]);
    // A value defined at `d` and last used at `l` crosses a call when a call
    // position lands strictly between them.
    const crossesCall = struct {
        fn f(prefix: []const usize, d: usize, l: usize) bool {
            if (l <= d + 1) return false;
            return prefix[l] - prefix[d + 1] > 0;
        }
    }.f;

    // dying[pos] = the values whose last use is at pos.
    const dying = try allocator.alloc(std.ArrayList(Value), total);
    defer {
        for (dying) |*d| d.deinit(allocator);
        allocator.free(dying);
    }
    for (dying) |*d| d.* = .empty;
    {
        var it = last_use.iterator();
        while (it.next()) |e| try dying[e.value_ptr.*].append(allocator, e.key_ptr.*);
    }

    // Temp-register free lists (pop yields t0/ft0 first).
    var int_free: std.ArrayList(Reg) = .empty;
    defer int_free.deinit(allocator);
    // Callee-saved registers sit at the bottom of the stack (drawn last), the
    // caller-saved temporaries on top (drawn first).
    var sk: usize = saved_regs.len;
    while (sk > 0) {
        sk -= 1;
        try int_free.append(allocator, saved_regs[sk]);
    }
    var ik: usize = temp_regs.len;
    while (ik > 0) {
        ik -= 1;
        try int_free.append(allocator, temp_regs[ik]);
    }
    var float_free: std.ArrayList(FReg) = .empty;
    defer float_free.deinit(allocator);
    var fs: usize = float_saved_regs.len;
    while (fs > 0) {
        fs -= 1;
        try float_free.append(allocator, float_saved_regs[fs]);
    }
    var fk: usize = float_temp_regs.len;
    while (fk > 0) {
        fk -= 1;
        try float_free.append(allocator, float_temp_regs[fk]);
    }
    var vector_free: std.ArrayList(VReg) = .empty;
    defer vector_free.deinit(allocator);
    var vk: usize = vector_regs.len;
    while (vk > 0) {
        vk -= 1;
        try vector_free.append(allocator, vector_regs[vk]); // pop yields v1 first
    }

    pos = 0;
    var int_arg: usize = 0;
    var float_arg: usize = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| {
            const cross = crossesCall(call_prefix, pos, last_use.get(p).?);
            if (isVector(func, func.valueType(p))) {
                if (bi == 0) return error.Unsupported; // an entry vector parameter has no ABI register
                if (cross) return error.Unsupported; // vector live across a call
                if (vector_free.pop()) |vr| {
                    try alloc.vector.put(allocator, p, vr);
                } else {
                    try alloc.vector_spill.put(allocator, p, alloc.vector_spill_count);
                    alloc.vector_spill_count += 1;
                }
                continue;
            }
            if (isFloat(func, func.valueType(p))) {
                if (bi == 0) {
                    // Entry params arrive in arg registers. One that outlives a call
                    // is homed to a callee-saved register (the selector moves it
                    // there at entry). The ABI arg slot is still consumed.
                    const reg = if (cross) (popFloatSaved(&float_free) orelse return error.Unsupported) else fargReg(float_arg);
                    try alloc.float.put(allocator, p, reg);
                    float_arg += 1;
                } else {
                    const reg = if (cross) popFloatSaved(&float_free) else float_free.pop();
                    try alloc.float.put(allocator, p, reg orelse return error.Unsupported);
                }
            } else {
                if (bi == 0) {
                    if (int_arg < 8) {
                        const reg = if (cross) (popSaved(&int_free) orelse return error.Unsupported) else argReg(int_arg);
                        try alloc.int.put(allocator, p, reg);
                    } else {
                        // 9th+ integer parameter arrives on the stack. Give it a
                        // register (loaded at entry) and record the stack index.
                        const reg = if (cross) popSaved(&int_free) else int_free.pop();
                        try alloc.int.put(allocator, p, reg orelse return error.Unsupported);
                        try alloc.incoming_stack.put(allocator, p, @intCast(int_arg - 8));
                    }
                    int_arg += 1;
                } else {
                    const reg = if (cross) popSaved(&int_free) else int_free.pop();
                    try alloc.int.put(allocator, p, reg orelse return error.Unsupported);
                }
            }
        }
        try freeDying(allocator, func, dying[pos].items, &alloc, &int_free, &float_free, &vector_free);
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            if (func.instResult(inst)) |r| {
                const cross = crossesCall(call_prefix, pos, last_use.get(r).?);
                if (isVector(func, func.valueType(r))) {
                    if (cross) return error.Unsupported; // a vector live across a call (all vregs are caller-saved)
                    if (vector_free.pop()) |vr| {
                        try alloc.vector.put(allocator, r, vr);
                    } else {
                        try alloc.vector_spill.put(allocator, r, alloc.vector_spill_count); // pressure: spill to a 16-byte slot
                        alloc.vector_spill_count += 1;
                    }
                } else if (isFloat(func, func.valueType(r))) {
                    const reg = if (cross) popFloatSaved(&float_free) else float_free.pop();
                    try alloc.float.put(allocator, r, reg orelse return error.Unsupported);
                } else {
                    const reg = if (cross) popSaved(&int_free) else int_free.pop();
                    if (reg) |rr| {
                        try alloc.int.put(allocator, r, rr);
                    } else {
                        // The integer file is exhausted: spill to a stack slot.
                        try alloc.int_spill.put(allocator, r, alloc.spill_count);
                        alloc.spill_count += 1;
                    }
                }
            }
            try freeDying(allocator, func, dying[pos].items, &alloc, &int_free, &float_free, &vector_free);
            pos += 1;
        }
        try freeDying(allocator, func, dying[pos].items, &alloc, &int_free, &float_free, &vector_free);
        pos += 1;
    }

    return alloc;
}

/// How a relocation patches its instruction word.
pub const RelocKind = enum {
    /// A `jal`'s call target (the default).
    call,
    /// An `auipc`'s high 20 bits of a PC-relative symbol address.
    pcrel_hi20,
    /// An `addi`'s low 12 bits, paired with the `auipc` at `pair`.
    pcrel_lo12,
};

pub const Reloc = struct {
    /// Word index of the instruction to patch in the emitted code.
    offset: usize,
    /// Target symbol name, borrowed from the function (valid while it lives).
    /// Empty for `pcrel_lo12` (its target is the local `auipc` at `pair`).
    symbol: []const u8,
    kind: RelocKind = .call,
    /// For `pcrel_lo12`: the word index of the paired `auipc`/`pcrel_hi20`.
    pair: usize = 0,
};

/// A compiled function: machine words plus the relocations its calls need.
/// One source-line-table row: the byte offset where a new source line's code begins.
pub const LineEntry = struct { offset: u32, line: u32 };

pub const Compiled = struct {
    code: []u32,
    relocs: []Reloc,
    lines: []LineEntry = &.{},

    pub fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
        allocator.free(self.relocs);
        allocator.free(self.lines);
    }
};

/// Select RISC-V machine words for a function (wrapper over `compileFunction`
/// that drops the relocations). The caller owns the returned slice.
pub fn selectFunction(allocator: std.mem.Allocator, func: *const Function) Error![]u32 {
    const compiled = try compileFunction(allocator, func);
    allocator.free(compiled.relocs);
    allocator.free(compiled.lines);
    return compiled.code;
}

/// Compiled code plus its source-line table (from the `debug.line` IR attributes), for DWARF.
pub const CodeWithLines = struct { code: []u32, lines: []LineEntry };

/// Like `selectFunction`, but also returns the source-line table. Caller owns both slices.
pub fn selectFunctionWithLines(allocator: std.mem.Allocator, func: *const Function) Error!CodeWithLines {
    const compiled = try compileFunction(allocator, func);
    allocator.free(compiled.relocs);
    return .{ .code = compiled.code, .lines = compiled.lines };
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

/// Compile a function to machine words and call relocations.
pub fn compileFunction(allocator: std.mem.Allocator, func: *const Function) Error!Compiled {
    var alloc = try allocateRegisters(allocator, func);
    defer alloc.deinit(allocator);

    var code: std.ArrayList(u32) = .empty;
    errdefer code.deinit(allocator);
    var relocs: std.ArrayList(Reloc) = .empty;
    errdefer relocs.deinit(allocator);
    var lines: std.ArrayList(LineEntry) = .empty;
    errdefer lines.deinit(allocator);
    var last_line: u32 = 0;

    // Stack frame: lay out an offset for every `alloca` slot. The whole frame is
    // rounded to the 16-byte ABI alignment below.
    var slot_offset: std.AutoHashMapUnmanaged(Value, i12) = .empty;
    defer slot_offset.deinit(allocator);
    var frame: u32 = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .alloca => |al| {
                    const size = try typeSize(func, al.elem);
                    frame = alignUp(frame, size); // natural alignment
                    if (frame > 2047) return error.Unsupported; // large frames: later
                    try slot_offset.put(allocator, func.instResult(inst).?, @intCast(frame));
                    frame += size;
                },
                else => {},
            }
        }
    }
    // Reserve a frame slot for each callee-saved register the allocation used.
    var used_saved: std.ArrayList(struct { reg: Reg, off: i12 }) = .empty;
    defer used_saved.deinit(allocator);
    for (saved_regs) |s| {
        var used = false;
        var it = alloc.int.valueIterator();
        while (it.next()) |r| if (r.* == s) {
            used = true;
            break;
        };
        if (used) {
            frame = alignUp(frame, 8);
            if (frame > 2047) return error.Unsupported;
            try used_saved.append(allocator, .{ .reg = s, .off = @intCast(frame) });
            frame += 8;
        }
    }

    // ...and a slot for each callee-saved float register used.
    var used_float_saved: std.ArrayList(struct { reg: FReg, off: i12 }) = .empty;
    defer used_float_saved.deinit(allocator);
    for (float_saved_regs) |s| {
        var used = false;
        var it = alloc.float.valueIterator();
        while (it.next()) |r| if (r.* == s) {
            used = true;
            break;
        };
        if (used) {
            frame = alignUp(frame, 8);
            if (frame > 2047) return error.Unsupported;
            try used_float_saved.append(allocator, .{ .reg = s, .off = @intCast(frame) });
            frame += 8;
        }
    }

    // A non-leaf function (one that makes a call) must preserve ra across the
    // call, which clobbers it: reserve a frame slot for ra.
    var non_leaf = false;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .call) {
                non_leaf = true;
                break;
            }
        }
        if (non_leaf) break;
    }
    var ra_off: i12 = 0;
    if (non_leaf) {
        frame = alignUp(frame, 8);
        if (frame > 2047) return error.Unsupported;
        ra_off = @intCast(frame);
        frame += 8;
    }

    // Spill slots: one 8-byte doubleword per spilled integer value.
    frame = alignUp(frame, 8);
    const spill_base: u32 = frame;
    frame += alloc.spill_count * 8;
    // Vector spill slots: one 16-byte (a <4 x f32>) slot per spilled vector, 16-aligned.
    frame = alignUp(frame, 16);
    const vspill_base: u32 = frame;
    frame += alloc.vector_spill_count * 16;
    if (frame > 2047) return error.Unsupported;

    const frame_size: i12 = @intCast(alignUp(frame, 16));
    // Prologue: open the frame, then preserve ra and the callee-saved registers.
    if (frame_size != 0) try code.append(allocator, encode.addi(.x2, .x2, -frame_size));
    if (non_leaf) try code.append(allocator, encode.sd(.x1, .x2, ra_off)); // save ra
    for (used_saved.items) |sv| try code.append(allocator, encode.sd(sv.reg, .x2, sv.off));
    for (used_float_saved.items) |sv| try code.append(allocator, encode.fsd(sv.reg, .x2, sv.off));

    // Move any entry parameter homed to a non-argument register (because it
    // outlives a call) out of its incoming argument register. Load stack
    // parameters from the caller's outgoing-argument area.
    if (func.blockCount() != 0) {
        var ia: usize = 0;
        var fa: usize = 0;
        for (func.blockParams(@enumFromInt(0))) |p| {
            if (isFloat(func, func.valueType(p))) {
                const home = alloc.float.get(p).?;
                const arg = fargReg(fa);
                if (home != arg) try code.append(allocator, if (is64Float(func, func.valueType(p))) encode.fmv_d(home, arg) else encode.fmv_s(home, arg));
                fa += 1;
            } else {
                const home = alloc.int.get(p).?;
                if (alloc.incoming_stack.get(p)) |idx| {
                    // Above this frame, in the caller's outgoing argument area.
                    const off: i12 = @intCast(@as(i32, frame_size) + @as(i32, @intCast(idx)) * 8);
                    try code.append(allocator, encode.ld(home, .x2, off));
                } else {
                    const arg = argReg(ia);
                    if (home != arg) try code.append(allocator, encode.addi(home, arg, 0));
                }
                ia += 1;
            }
        }
    }

    // Configure the RVV unit once for the fixed 4-lane f32 group. VL and SEW
    // persist as CPU state and nothing in a vector function changes them (a value
    // live across a call is rejected by the allocator), so one vsetivli suffices.
    if (alloc.vector.count() != 0) try code.append(allocator, encode.vsetivli(.x0, 4, 0xD0));

    const block_start = try allocator.alloc(usize, func.blockCount());
    defer allocator.free(block_start);
    var fixups: std.ArrayList(Fixup) = .empty;
    defer fixups.deinit(allocator);

    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        block_start[bi] = code.items.len;

        // A structured `if` is the block's exit. The trailing terminator is dead.
        var exited = false;
        for (func.blockInsts(block)) |inst| {
            // Record a source-line row when this instruction starts a new line.
            if (lineOf(func, inst)) |line| {
                if (line != last_line) {
                    try lines.append(allocator, .{ .offset = @intCast(code.items.len * 4), .line = line });
                    last_line = line;
                }
            }
            switch (func.opcode(inst)) {
                .arith => |a| {
                    if (isVector(func, func.valueType(a.lhs))) {
                        // RVV vector arithmetic. Spilled operands reload into
                        // vec_op0/op1, a spilled result computes in vec_work.
                        // spill_scratch1 holds the slot address.
                        const result = func.instResult(inst).?;
                        const lhs = try reloadVector(allocator, &code, &alloc, vspill_base, a.lhs, vec_op0, spill_scratch1);
                        const rhs = try reloadVector(allocator, &code, &alloc, vspill_base, a.rhs, vec_op1, spill_scratch1);
                        const rd = dstVector(&alloc, result, vec_work);
                        try code.append(allocator, switch (a.op) {
                            .add => encode.vfadd_vv(rd, lhs, rhs),
                            .sub => encode.vfsub_vv(rd, lhs, rhs),
                            .mul => encode.vfmul_vv(rd, lhs, rhs),
                            .div => encode.vfdiv_vv(rd, lhs, rhs),
                            else => return error.Unsupported,
                        });
                        try storeVector(allocator, &code, &alloc, vspill_base, result, rd, spill_scratch1);
                    } else if (isFloat(func, func.valueType(a.lhs))) {
                        const rd = alloc.float.get(func.instResult(inst).?).?;
                        const rs1 = alloc.float.get(a.lhs).?;
                        const rs2 = alloc.float.get(a.rhs).?;
                        const d = is64Float(func, func.valueType(a.lhs));
                        const word = switch (a.op) {
                            .add => if (d) encode.fadd_d(rd, rs1, rs2) else encode.fadd_s(rd, rs1, rs2),
                            .sub => if (d) encode.fsub_d(rd, rs1, rs2) else encode.fsub_s(rd, rs1, rs2),
                            .mul => if (d) encode.fmul_d(rd, rs1, rs2) else encode.fmul_s(rd, rs1, rs2),
                            .div => if (d) encode.fdiv_d(rd, rs1, rs2) else encode.fdiv_s(rd, rs1, rs2),
                            else => return error.Unsupported, // bitwise/shift/rem on floats
                        };
                        try code.append(allocator, word);
                    } else {
                        const result = func.instResult(inst).?;
                        const rs1 = try reloadInt(allocator, &code, &alloc, spill_base, a.lhs, spill_scratch0);
                        const rs2 = try reloadInt(allocator, &code, &alloc, spill_base, a.rhs, spill_scratch1);
                        const spill_idx = alloc.int_spill.get(result);
                        const rd = if (spill_idx == null) alloc.int.get(result).? else spill_scratch0;
                        const unsigned = isUnsignedInt(func, func.valueType(a.lhs));
                        const word = if (unsigned and a.op == .div)
                            encode.divu(rd, rs1, rs2)
                        else if (unsigned and a.op == .rem)
                            encode.remu(rd, rs1, rs2)
                        else if (unsigned and a.op == .shr)
                            encode.srl(rd, rs1, rs2) // logical right shift
                        else
                            arithWord(a.op, rd, rs1, rs2);
                        try code.append(allocator, word);
                        if (spill_idx) |idx| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + idx * 8)));
                    }
                },
                .arith_imm => |a| {
                    if (isFloat(func, func.valueType(a.lhs))) return error.Unsupported;
                    const result = func.instResult(inst).?;
                    const rs1 = try reloadInt(allocator, &code, &alloc, spill_base, a.lhs, spill_scratch1);
                    const ai_spill = alloc.int_spill.get(result);
                    const rd = if (ai_spill == null) alloc.int.get(result).? else spill_scratch0;
                    const unsigned = isUnsignedInt(func, func.valueType(a.lhs));
                    const fits12 = a.imm >= -2048 and a.imm <= 2047;
                    const word = switch (a.op) {
                        .add => if (fits12) encode.addi(rd, rs1, @intCast(a.imm)) else return error.Unsupported,
                        .sub => if (a.imm >= -2047 and a.imm <= 2048) encode.addi(rd, rs1, @intCast(-a.imm)) else return error.Unsupported,
                        .bit_and => if (fits12) encode.andi(rd, rs1, @intCast(a.imm)) else return error.Unsupported,
                        .bit_or => if (fits12) encode.ori(rd, rs1, @intCast(a.imm)) else return error.Unsupported,
                        .bit_xor => if (fits12) encode.xori(rd, rs1, @intCast(a.imm)) else return error.Unsupported,
                        .shl => if (a.imm >= 0 and a.imm <= 63) encode.slli(rd, rs1, @intCast(a.imm)) else return error.Unsupported,
                        .shr => if (a.imm >= 0 and a.imm <= 63)
                            (if (unsigned) encode.srli(rd, rs1, @intCast(a.imm)) else encode.srai(rd, rs1, @intCast(a.imm)))
                        else
                            return error.Unsupported,
                        .mul, .div, .rem => return error.Unsupported, // no immediate form
                    };
                    try code.append(allocator, word);
                    if (ai_spill) |idx| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + idx * 8)));
                },
                .iconst => |c| {
                    const res = func.instResult(inst).?;
                    if (isFloat(func, func.valueType(res))) {
                        // A float-typed integer constant (a zero-init). Materialize the
                        // bits and move them into the float register, never an integer
                        // register the value was never assigned.
                        if (is64Float(func, func.valueType(res))) return error.Unsupported;
                        const fr = alloc.float.get(res).?;
                        const bits: u32 = @truncate(@as(u64, @bitCast(c)));
                        try loadImm32(allocator, &code, scratch_reg, bits);
                        try code.append(allocator, encode.fmv_w_x(fr, scratch_reg));
                        continue;
                    }
                    const rd = alloc.int.get(res).?;

                    if (c >= -2048 and c <= 2047) {
                        // Fits a 12-bit immediate: `addi rd, zero, imm`.
                        try code.append(allocator, encode.addi(rd, .x0, @intCast(c)));
                    } else if (c >= std.math.minInt(i32) and c <= std.math.maxInt(i32)) {
                        // 32-bit constant: `lui rd, hi` then `addi rd, rd, lo`. The +0x800
                        // pre-rounds `hi` for the sign-extended `addi`.
                        const bits: u32 = @bitCast(@as(i32, @intCast(c)));
                        const hi: u20 = @truncate((bits +% 0x800) >> 12);
                        const lo: i12 = @bitCast(@as(u12, @truncate(bits)));
                        try code.append(allocator, encode.lui(rd, hi));
                        try code.append(allocator, encode.addi(rd, rd, lo));
                    } else {
                        return error.Unsupported; // 64-bit constant: needs a longer sequence
                    }
                },
                .fconst => |val| {
                    const result = func.instResult(inst).?;
                    const fr = alloc.float.get(result).?;
                    if (is64Float(func, func.valueType(result))) return error.Unsupported; // f64 const: later
                    // Load the 32-bit pattern, then move it into the float register.
                    const bits: u32 = @bitCast(@as(f32, @floatCast(val)));
                    try loadImm32(allocator, &code, scratch_reg, bits);
                    try code.append(allocator, encode.fmv_w_x(fr, scratch_reg));
                },
                .alloca => {
                    // The slot address is `sp + offset` into the frame.
                    const result = func.instResult(inst).?;
                    const rd = alloc.int.get(result).?;
                    const off = slot_offset.get(result).?;
                    try code.append(allocator, encode.addi(rd, .x2, off));
                },
                .global_addr => |ga| {
                    // PC-relative symbol address: `auipc rd, %pcrel_hi(sym)` then
                    // `addi rd, rd, %pcrel_lo(.Lhi)`. The two relocations resolve together.
                    const rd = alloc.int.get(func.instResult(inst).?).?;
                    const name = func.symbolName(ga.symbol);
                    const hi = code.items.len;
                    try relocs.append(allocator, .{ .offset = hi, .symbol = name, .kind = .pcrel_hi20 });
                    try code.append(allocator, encode.auipc(rd, 0));
                    try relocs.append(allocator, .{ .offset = code.items.len, .symbol = "", .kind = .pcrel_lo12, .pair = hi });
                    try code.append(allocator, encode.addi(rd, rd, 0));
                },
                .select => |sel| {
                    // `cond ? then : else`, lowered to a short forward branch. The
                    // result register is distinct from the operands (drawn while
                    // they are still live), so there is no aliasing hazard.
                    if (isFloat(func, func.valueType(sel.then))) return error.Unsupported; // float select: later
                    const rd = alloc.int.get(func.instResult(inst).?).?;
                    const cond = alloc.int.get(sel.cond).?;
                    const then_r = alloc.int.get(sel.then).?;
                    const else_r = alloc.int.get(sel.@"else").?;
                    try code.append(allocator, encode.addi(rd, then_r, 0)); // rd = then
                    try code.append(allocator, encode.bne(cond, .x0, 8)); // if cond != 0, keep then
                    try code.append(allocator, encode.addi(rd, else_r, 0)); // else rd = else
                },
                .convert => |cv| {
                    const result = func.instResult(inst).?;
                    const src_ty = func.valueType(cv.value);
                    const dst_ty = func.valueType(result);
                    const src_float = isFloat(func, src_ty);
                    const dst_float = isFloat(func, dst_ty);
                    if (src_float == dst_float) return error.Unsupported; // int<->int / float<->float later
                    if (dst_float) {
                        // integer -> float, only a 32-bit signed source for now.
                        if (!isWord(func, src_ty)) return error.Unsupported;
                        const rs = alloc.int.get(cv.value).?;
                        const rd = alloc.float.get(result).?;
                        try code.append(allocator, if (is64Float(func, dst_ty)) encode.fcvt_d_w(rd, rs) else encode.fcvt_s_w(rd, rs));
                    } else {
                        // float -> integer, only a 32-bit signed destination for now.
                        if (!isWord(func, dst_ty)) return error.Unsupported;
                        const rs = alloc.float.get(cv.value).?;
                        const rd = alloc.int.get(result).?;
                        try code.append(allocator, if (is64Float(func, src_ty)) encode.fcvt_w_d(rd, rs) else encode.fcvt_w_s(rd, rs));
                    }
                },
                .load => |l| {
                    const result = func.instResult(inst).?;
                    const base = alloc.int.get(l.ptr).?;
                    if (isVector(func, func.valueType(result))) {
                        const rd = dstVector(&alloc, result, vec_work);
                        try code.append(allocator, encode.vle32(rd, base));
                        try storeVector(allocator, &code, &alloc, vspill_base, result, rd, spill_scratch1);
                    } else if (isFloat(func, func.valueType(result))) {
                        const rd = alloc.float.get(result).?;
                        try code.append(allocator, if (is64Float(func, func.valueType(result))) encode.fld(rd, base, 0) else encode.flw(rd, base, 0));
                    } else {
                        const rd = alloc.int.get(result).?;
                        const ty = func.valueType(result);
                        const word = isWord(func, ty);
                        try code.append(allocator, intLoadInsn(func, ty, rd, base, 0));
                        // Swap a big-endian 64-bit load to native order.
                        if (!word and endianBig(func, .{ .value = result })) {
                            try code.append(allocator, encode.rev8(rd, rd));
                        } else if (word and endianBig(func, .{ .value = result })) {
                            return error.Unsupported; // sub-word byte-swap not yet handled
                        }
                    }
                },
                .store => |st| {
                    const base = alloc.int.get(st.ptr).?;
                    if (isVector(func, func.valueType(st.value))) {
                        const vr = try reloadVector(allocator, &code, &alloc, vspill_base, st.value, vec_op0, spill_scratch1);
                        try code.append(allocator, encode.vse32(vr, base));
                    } else if (isFloat(func, func.valueType(st.value))) {
                        const vr = alloc.float.get(st.value).?;
                        try code.append(allocator, if (is64Float(func, func.valueType(st.value))) encode.fsd(vr, base, 0) else encode.fsw(vr, base, 0));
                    } else {
                        const vr = alloc.int.get(st.value).?;
                        const ty = func.valueType(st.value);
                        const word = isWord(func, ty);
                        // Reverse a big-endian 64-bit value before storing.
                        if (!word and endianBig(func, .{ .inst = inst })) {
                            try code.append(allocator, encode.rev8(scratch_reg, vr));
                            try code.append(allocator, encode.sd(scratch_reg, base, 0));
                        } else if (word and endianBig(func, .{ .inst = inst })) {
                            return error.Unsupported; // sub-word byte-swap not yet handled
                        } else {
                            try code.append(allocator, intStoreInsn(func, ty, vr, base, 0));
                        }
                    }
                },
                .icmp => |cmp| if (isFloat(func, func.valueType(cmp.lhs))) {
                    // Float comparison: float operands, integer (bool) result.
                    const rd = alloc.int.get(func.instResult(inst).?).?;
                    const rs1 = alloc.float.get(cmp.lhs).?;
                    const rs2 = alloc.float.get(cmp.rhs).?;
                    const d = is64Float(func, func.valueType(cmp.lhs));
                    const feq = if (d) &encode.feq_d else &encode.feq_s;
                    const flt = if (d) &encode.flt_d else &encode.flt_s;
                    const fle = if (d) &encode.fle_d else &encode.fle_s;
                    switch (cmp.op) {
                        .eq => try code.append(allocator, feq(rd, rs1, rs2)),
                        .lt => try code.append(allocator, flt(rd, rs1, rs2)),
                        .gt => try code.append(allocator, flt(rd, rs2, rs1)),
                        .le => try code.append(allocator, fle(rd, rs1, rs2)),
                        .ge => try code.append(allocator, fle(rd, rs2, rs1)),
                        .ne => {
                            try code.append(allocator, feq(rd, rs1, rs2));
                            try code.append(allocator, encode.xori(rd, rd, 1));
                        },
                    }
                } else {
                    const result = func.instResult(inst).?;
                    const rs1 = try reloadInt(allocator, &code, &alloc, spill_base, cmp.lhs, spill_scratch0);
                    const rs2 = try reloadInt(allocator, &code, &alloc, spill_base, cmp.rhs, spill_scratch1);
                    const ic_spill = alloc.int_spill.get(result);
                    const rd = if (ic_spill == null) alloc.int.get(result).? else spill_scratch0;
                    // Pick signed `slt` or unsigned `sltu` from the operand type.
                    const setLt = if (isUnsignedInt(func, func.valueType(cmp.lhs))) &encode.sltu else &encode.slt;
                    switch (cmp.op) {
                        .lt => try code.append(allocator, setLt(rd, rs1, rs2)),
                        .gt => try code.append(allocator, setLt(rd, rs2, rs1)),
                        .ge => {
                            try code.append(allocator, setLt(rd, rs1, rs2));
                            try code.append(allocator, encode.xori(rd, rd, 1));
                        },
                        .le => {
                            try code.append(allocator, setLt(rd, rs2, rs1));
                            try code.append(allocator, encode.xori(rd, rd, 1));
                        },
                        .eq => {
                            try code.append(allocator, encode.sub(rd, rs1, rs2));
                            try code.append(allocator, encode.sltiu(rd, rd, 1));
                        },
                        .ne => {
                            try code.append(allocator, encode.sub(rd, rs1, rs2));
                            try code.append(allocator, encode.sltu(rd, .x0, rd));
                        },
                    }
                    if (ic_spill) |idx| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + idx * 8)));
                },
                .@"if" => |cf| {
                    // Edge arguments require prior critical-edge splitting.
                    if (func.blockArgs(cf.then).len != 0 or func.blockArgs(cf.@"else").len != 0) return error.Unsupported;
                    const cond_reg = alloc.int.get(cf.cond).?;
                    // bne cond, x0, then  /  jal x0, else  (offsets patched later)
                    try fixups.append(allocator, .{ .index = code.items.len, .target = cf.then.target, .kind = .{ .branch = cond_reg } });
                    try code.append(allocator, encode.bne(cond_reg, .x0, 0));
                    try fixups.append(allocator, .{ .index = code.items.len, .target = cf.@"else".target, .kind = .jal });
                    try code.append(allocator, encode.jal(.x0, 0));
                    exited = true;
                    break;
                },
                .call => |c| {
                    // Place arguments. Integer register args (0-7) go through a
                    // parallel move (so a conflicting permutation of a0-a7 is
                    // correct). Spilled integer args load from their slot. Integer
                    // args 9+ store to an outgoing area below sp. Float args move
                    // directly. More than eight float args, and mixing stack args
                    // with spilled args, are unsupported.
                    var int_moves: std.ArrayList(RegMove) = .empty;
                    defer int_moves.deinit(allocator);
                    var int_spilled: std.ArrayList(struct { dst: Reg, off: i12 }) = .empty;
                    defer int_spilled.deinit(allocator);
                    var int_stack: std.ArrayList(Reg) = .empty; // register sources for args 9+
                    defer int_stack.deinit(allocator);
                    var int_i: usize = 0;
                    var float_i: usize = 0;
                    for (func.valueList(c.args)) |arg| {
                        if (isFloat(func, func.valueType(arg))) {
                            if (float_i >= 8) return error.Unsupported;
                            const dst = fargReg(float_i);
                            const src = alloc.float.get(arg).?;
                            if (src != dst) try code.append(allocator, if (is64Float(func, func.valueType(arg))) encode.fmv_d(dst, src) else encode.fmv_s(dst, src));
                            float_i += 1;
                        } else if (int_i < 8) {
                            const dst = argReg(int_i);
                            if (alloc.int.get(arg)) |src| {
                                try int_moves.append(allocator, .{ .src = src, .dst = dst });
                            } else {
                                const idx = alloc.int_spill.get(arg).?;
                                try int_spilled.append(allocator, .{ .dst = dst, .off = @intCast(spill_base + idx * 8) });
                            }
                            int_i += 1;
                        } else {
                            // Stack argument: must be register-resident for now.
                            try int_stack.append(allocator, alloc.int.get(arg) orelse return error.Unsupported);
                            int_i += 1;
                        }
                    }
                    if (int_stack.items.len != 0 and int_spilled.items.len != 0) return error.Unsupported;

                    // Reserve the outgoing stack-argument area (16-byte aligned) and
                    // store the args into it, before shuffling the register args.
                    const stack_area: i12 = @intCast(alignUp(@intCast(int_stack.items.len * 8), 16));
                    if (stack_area != 0) {
                        try code.append(allocator, encode.addi(.x2, .x2, -stack_area));
                        for (int_stack.items, 0..) |src, j| try code.append(allocator, encode.sd(src, .x2, @intCast(j * 8)));
                    }
                    try parallelMoveInt(allocator, &code, int_moves.items, spill_scratch0);
                    for (int_spilled.items) |s| try code.append(allocator, encode.ld(s.dst, .x2, s.off));

                    // jal ra, <callee>. The target is a relocation.
                    try relocs.append(allocator, .{ .offset = code.items.len, .symbol = func.symbolName(c.symbol) });
                    try code.append(allocator, encode.jal(.x1, 0));
                    if (stack_area != 0) try code.append(allocator, encode.addi(.x2, .x2, stack_area));

                    // A result returns in a0 / fa0. Route it to its register or slot.
                    if (func.instResult(inst)) |result| {
                        if (isFloat(func, func.valueType(result))) {
                            const rd = alloc.float.get(result).?;
                            if (rd != .f10) try code.append(allocator, if (is64Float(func, func.valueType(result))) encode.fmv_d(rd, .f10) else encode.fmv_s(rd, .f10));
                        } else if (alloc.int.get(result)) |rd| {
                            if (rd != .x10) try code.append(allocator, encode.addi(rd, .x10, 0));
                        } else {
                            const idx = alloc.int_spill.get(result).?;
                            try code.append(allocator, encode.sd(.x10, .x2, @intCast(spill_base + idx * 8)));
                        }
                    }
                },
                .struct_new => |sn| {
                    // Pack four scalar floats into a <4 x f32>. Seed lane 0 with the
                    // last field, then slide up inserting the earlier ones. The
                    // slide's vd must not overlap vs2, so alternate result and scratch.
                    const result = func.instResult(inst).?;
                    if (!isVector(func, func.valueType(result))) return error.Unsupported;
                    const fields = func.valueList(sn.fields);
                    if (fields.len != 4) return error.Unsupported;
                    const rd = dstVector(&alloc, result, vec_work); // vec_work if the result is spilled
                    const f0 = alloc.float.get(fields[0]).?;
                    const f1 = alloc.float.get(fields[1]).?;
                    const f2 = alloc.float.get(fields[2]).?;
                    const f3 = alloc.float.get(fields[3]).?;
                    try code.append(allocator, encode.vfmv_s_f(vector_scratch, f3)); // [f3]
                    try code.append(allocator, encode.vfslide1up_vf(rd, vector_scratch, f2)); // [f2,f3]
                    try code.append(allocator, encode.vfslide1up_vf(vector_scratch, rd, f1)); // [f1,f2,f3]
                    try code.append(allocator, encode.vfslide1up_vf(rd, vector_scratch, f0)); // [f0,f1,f2,f3]
                    try storeVector(allocator, &code, &alloc, vspill_base, result, rd, spill_scratch1);
                },
                .extract => |ex| {
                    // Extract a lane to a scalar float. Lane 0 is a direct vfmv.f.s.
                    // a higher lane slides down to lane 0 first.
                    const rd = alloc.float.get(func.instResult(inst).?).?;
                    const vs = try reloadVector(allocator, &code, &alloc, vspill_base, ex.aggregate, vec_op0, spill_scratch1);
                    if (ex.index == 0) {
                        try code.append(allocator, encode.vfmv_f_s(rd, vs));
                    } else {
                        try code.append(allocator, encode.vslidedown_vi(vector_scratch, vs, @intCast(ex.index)));
                        try code.append(allocator, encode.vfmv_f_s(rd, vector_scratch));
                    }
                },
                else => return error.Unsupported,
            }
        }

        if (!exited) {
            switch (func.terminator(block) orelse return error.Unsupported) {
                .ret => |value| {
                    if (value) |v| {
                        if (isFloat(func, func.valueType(v))) {
                            // fmv fa0, freg  (skipped when already in fa0)
                            const fr = alloc.float.get(v).?;
                            if (fr != .f10) try code.append(allocator, if (is64Float(func, func.valueType(v))) encode.fmv_d(.f10, fr) else encode.fmv_s(.f10, fr));
                        } else {
                            // mv a0, reg  (skipped when already in a0)
                            const r = try reloadInt(allocator, &code, &alloc, spill_base, v, spill_scratch0);
                            if (r != .x10) try code.append(allocator, encode.addi(.x10, r, 0));
                        }
                    }
                    // Epilogue: restore ra and the callee-saved registers, close the frame.
                    if (non_leaf) try code.append(allocator, encode.ld(.x1, .x2, ra_off)); // restore ra
                    for (used_saved.items) |sv| try code.append(allocator, encode.ld(sv.reg, .x2, sv.off));
                    for (used_float_saved.items) |sv| try code.append(allocator, encode.fld(sv.reg, .x2, sv.off));
                    if (frame_size != 0) try code.append(allocator, encode.addi(.x2, .x2, frame_size));
                    try code.append(allocator, encode.jalr(.x0, .x1, 0)); // ret
                },
                .jump => |j| {
                    // Move each argument into its block parameter's register before
                    // the jump.
                    const args = func.blockArgs(j);
                    const params = func.blockParams(j.target);
                    for (args, params) |arg, param| {
                        if (isVector(func, func.valueType(arg))) {
                            // Move a vector into its block parameter (vmv.v.v),
                            // reloading a spilled arg and storing into a spilled
                            // parameter as needed.
                            const ar = try reloadVector(allocator, &code, &alloc, vspill_base, arg, vec_op0, spill_scratch1);
                            if (alloc.vector.get(param)) |pr| {
                                if (ar != pr) try code.append(allocator, encode.vmv_v_v(pr, ar));
                            } else {
                                const off: i12 = @intCast(vspill_base + alloc.vector_spill.get(param).? * 16);
                                try code.append(allocator, encode.addi(spill_scratch1, .x2, off));
                                try code.append(allocator, encode.vse32(ar, spill_scratch1));
                            }
                        } else if (isFloat(func, func.valueType(arg))) {
                            const ar = alloc.float.get(arg).?;
                            const pr = alloc.float.get(param).?;
                            if (ar != pr) try code.append(allocator, if (is64Float(func, func.valueType(arg))) encode.fmv_d(pr, ar) else encode.fmv_s(pr, ar));
                        } else {
                            const ar = alloc.int.get(arg).?;
                            const pr = alloc.int.get(param).?;
                            if (ar != pr) try code.append(allocator, encode.addi(pr, ar, 0)); // mv pr, ar
                        }
                    }
                    try fixups.append(allocator, .{ .index = code.items.len, .target = j.target, .kind = .jal });
                    try code.append(allocator, encode.jal(.x0, 0));
                },
            }
        }
    }

    // Patch each branch/jump now that every block's position is known.
    for (fixups.items) |fx| {
        const target_idx: i64 = @intCast(block_start[@intFromEnum(fx.target)]);
        const from_idx: i64 = @intCast(fx.index);
        const off: i32 = @intCast((target_idx - from_idx) * 4);
        switch (fx.kind) {
            .branch => |rs1| code.items[fx.index] = encode.bne(rs1, .x0, @intCast(off)),
            .jal => code.items[fx.index] = encode.jal(.x0, @intCast(off)),
        }
    }

    return .{
        .code = try code.toOwnedSlice(allocator),
        .relocs = try relocs.toOwnedSlice(allocator),
        .lines = try lines.toOwnedSlice(allocator),
    };
}

test "a big-endian load byte-swaps after the load" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendInst(entry, i64_t, .{ .load = .{ .ptr = p } });
    try func.addAttr(.{ .value = v }, .{ .endian = .big });
    func.setTerminator(entry, .{ .ret = v });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.ld(.x5, .x10, 0), // ld t0, 0(a0)
        encode.rev8(.x5, .x5), // rev8 t0, t0  (big-endian -> native)
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "a big-endian store byte-swaps before the store" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendBlockParam(entry, i64_t);
    try func.appendStore(entry, v, p);
    const insts = func.blockInsts(entry);
    try func.addAttr(.{ .inst = insts[insts.len - 1] }, .{ .endian = .big });
    func.setTerminator(entry, .{ .ret = null });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.rev8(.x6, .x11), // rev8 scratch, v  (reverse without clobbering v)
        encode.sd(.x6, .x10, 0), // sd scratch, 0(p)
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "callee-saved float registers are preserved in the frame" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const p0 = try func.appendBlockParam(entry, f32_t);
    const p1 = try func.appendBlockParam(entry, f32_t);

    // Thirteen independent float values exceed the twelve caller-saved float
    // temporaries, so callee-saved float registers (fs0 = f8, ...) are drawn.
    var vals: [13]Value = undefined;
    for (&vals) |*v| v.* = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = p0, .rhs = p1 } });
    var acc = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = vals[0], .rhs = vals[1] } });
    for (vals[2..]) |v| acc = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(entry, .{ .ret = acc });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // The frame preserves two callee-saved float registers (fs0=f8, fs1=f9) with
    // fsd/fld, restoring them before the return.
    try std.testing.expectEqual(encode.addi(.x2, .x2, -16), code[0]);
    try std.testing.expectEqual(encode.fsd(.f8, .x2, 0), code[1]);
    try std.testing.expectEqual(encode.fsd(.f9, .x2, 8), code[2]);
    try std.testing.expectEqual(encode.fld(.f8, .x2, 0), code[code.len - 4]);
    try std.testing.expectEqual(encode.fld(.f9, .x2, 8), code[code.len - 3]);
    try std.testing.expectEqual(encode.addi(.x2, .x2, 16), code[code.len - 2]);
    try std.testing.expectEqual(encode.jalr(.x0, .x1, 0), code[code.len - 1]);
}

test "a callee-saved register is saved in the prologue and restored before ret" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const p0 = try func.appendBlockParam(entry, i32_t);
    const p1 = try func.appendBlockParam(entry, i32_t);

    // Seven independent values, all live at once, exceed the six caller-saved
    // temporaries, so the seventh draws a callee-saved register (s1 = x9).
    var vals: [7]Value = undefined;
    for (&vals) |*v| v.* = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = p0, .rhs = p1 } });
    var acc = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = vals[0], .rhs = vals[1] } });
    for (vals[2..]) |v| acc = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(entry, .{ .ret = acc });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // Two callee-saved registers (s1=x9, s2=x18) are drawn under the pressure, so
    // the prologue preserves both in a 16-byte frame and the epilogue restores them.
    try std.testing.expectEqual(encode.addi(.x2, .x2, -16), code[0]);
    try std.testing.expectEqual(encode.sd(.x9, .x2, 0), code[1]);
    try std.testing.expectEqual(encode.sd(.x18, .x2, 8), code[2]);
    try std.testing.expectEqual(encode.ld(.x9, .x2, 0), code[code.len - 4]);
    try std.testing.expectEqual(encode.ld(.x18, .x2, 8), code[code.len - 3]);
    try std.testing.expectEqual(encode.addi(.x2, .x2, 16), code[code.len - 2]);
    try std.testing.expectEqual(encode.jalr(.x0, .x1, 0), code[code.len - 1]);
}

test "an entry parameter that outlives a call is homed to a callee-saved register" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const y = try func.appendBlockParam(entry, i32_t);
    const a = try func.appendCall(entry, i32_t, "f", &.{y}); // x is live across this call
    const r = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = a } });
    func.setTerminator(entry, .{ .ret = r });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // Non-leaf (it calls): the prologue saves ra (slot 8) and callee-saved x9
    // (slot 0). x arrives in a0 but outlives the call, so it moves into x9.
    try std.testing.expectEqual(encode.addi(.x2, .x2, -16), code[0]);
    try std.testing.expectEqual(encode.sd(.x1, .x2, 8), code[1]); // save ra
    try std.testing.expectEqual(encode.sd(.x9, .x2, 0), code[2]); // save x9
    try std.testing.expectEqual(encode.addi(.x9, .x10, 0), code[3]); // mv x9, a0
}

test "a value live across a call is placed in a callee-saved register" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const y = try func.appendBlockParam(entry, i32_t);
    const s = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    const a = try func.appendCall(entry, i32_t, "f", &.{y}); // s is live across this call
    const r = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = a } });
    func.setTerminator(entry, .{ .ret = r });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // Non-leaf: prologue saves ra (slot 8) and callee-saved x9 (slot 0). `s`
    // outlives the call so it is computed into x9 (`add x9, x10, x11`).
    try std.testing.expectEqual(encode.addi(.x2, .x2, -16), code[0]);
    try std.testing.expectEqual(encode.sd(.x1, .x2, 8), code[1]); // save ra
    try std.testing.expectEqual(encode.sd(.x9, .x2, 0), code[2]); // save x9
    try std.testing.expectEqual(encode.add(.x9, .x10, .x11), code[3]);
}

test "a void call discards its result" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    try func.appendVoidCall(entry, "sink", &.{x});
    func.setTerminator(entry, .{ .ret = null });

    var compiled = try compileFunction(std.testing.allocator, &func);
    defer compiled.deinit(std.testing.allocator);

    // Non-leaf: prologue/epilogue save and restore ra around the call.
    try std.testing.expectEqualSlices(u32, &.{
        encode.addi(.x2, .x2, -16), // open frame
        encode.sd(.x1, .x2, 0), // save ra
        encode.jal(.x1, 0), // call sink  (no result routing)
        encode.ld(.x1, .x2, 0), // restore ra
        encode.addi(.x2, .x2, 16), // close frame
        encode.jalr(.x0, .x1, 0), // ret
    }, compiled.code);
    try std.testing.expectEqualStrings("sink", compiled.relocs[0].symbol);
}

test "calls an external symbol and routes its result" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const v0 = try func.appendBlockParam(entry, i32_t);
    const v1 = try func.appendBlockParam(entry, i32_t);
    const r = try func.appendCall(entry, i32_t, "add", &.{ v0, v1 });
    func.setTerminator(entry, .{ .ret = r });

    var compiled = try compileFunction(std.testing.allocator, &func);
    defer compiled.deinit(std.testing.allocator);

    // Non-leaf: ra is saved/restored around the call (the args are already in
    // a0/a1, so the call itself is just `jal ra`). The result moves out of a0.
    try std.testing.expectEqualSlices(u32, &.{
        encode.addi(.x2, .x2, -16), // open frame
        encode.sd(.x1, .x2, 0), // save ra
        encode.jal(.x1, 0), // call add  (target is a relocation)
        encode.addi(.x5, .x10, 0), // r = a0
        encode.addi(.x10, .x5, 0), // mv a0, r
        encode.ld(.x1, .x2, 0), // restore ra
        encode.addi(.x2, .x2, 16), // close frame
        encode.jalr(.x0, .x1, 0), // ret
    }, compiled.code);
    try std.testing.expectEqual(@as(usize, 1), compiled.relocs.len);
    try std.testing.expectEqual(@as(usize, 2), compiled.relocs[0].offset); // jal at word 2
    try std.testing.expectEqualStrings("add", compiled.relocs[0].symbol);
}

test "an alloca opens a stack frame and addresses its slot" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const p = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(entry, x, p);
    const v = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(entry, .{ .ret = v });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.addi(.x2, .x2, -16), // prologue: sp -= 16 (4-byte slot, 16-aligned)
        encode.addi(.x5, .x2, 0), // p = sp + 0
        encode.sw(.x10, .x5, 0), // store x, [p]
        encode.lw(.x7, .x5, 0), // v = load [p]
        encode.addi(.x10, .x7, 0), // mv a0, v
        encode.addi(.x2, .x2, 16), // epilogue: sp += 16
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "sub-word integer loads and stores pick the right width" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, i8_t);
    try func.appendStore(entry, b, p);
    const v = try func.appendInst(entry, i8_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(entry, .{ .ret = v });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.sb(.x11, .x10, 0), // store i8: sb b, [p]
        encode.lb(.x5, .x10, 0), // load i8 (signed): lb v, [p]
        encode.addi(.x10, .x5, 0), // mv a0, v
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "an unsigned halfword load zero-extends" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendInst(entry, u16_t, .{ .load = .{ .ptr = p } });
    func.setTerminator(entry, .{ .ret = v });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.lhu(.x5, .x10, 0), // load u16 (zero-extended): lhu v, [p]
        encode.addi(.x10, .x5, 0), // mv a0, v
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects loads and stores" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = p } });
    try func.appendStore(entry, v, p);
    func.setTerminator(entry, .{ .ret = null });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.lw(.x5, .x10, 0), // lw t0, 0(a0)
        encode.sw(.x5, .x10, 0), // sw t0, 0(a0)
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "splits critical edges so the canonical max lowers" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const block0 = try func.appendBlock();
    const c = try func.appendBlockParam(block0, bool_t);
    const a = try func.appendBlockParam(block0, i32_t);
    const b = try func.appendBlockParam(block0, i32_t);
    const block1 = try func.appendBlock();
    const r = try func.appendBlockParam(block1, i32_t);

    try func.appendIf(block0, c, .{ .target = block1, .args = &.{a} }, .{ .target = block1, .args = &.{b} });
    func.setTerminator(block1, .{ .ret = r });

    try splitCriticalEdges(std.testing.allocator, &func);

    // Two landing blocks were inserted, one per arg-carrying edge.
    try std.testing.expectEqual(@as(usize, 4), func.blockCount());

    // And the whole thing now lowers to RISC-V without error.
    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "selects float block arguments on a jump edge" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block0 = try func.appendBlock();
    const v0 = try func.appendBlockParam(block0, f32_t);
    const block1 = try func.appendBlock();
    const v1 = try func.appendBlockParam(block1, f32_t);

    try func.setJump(block0, block1, &.{v0});
    func.setTerminator(block1, .{ .ret = v1 });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.fmv_s(.f0, .f10), // fmv.s ft0, fa0  (pass v0 into v1's float reg)
        encode.jal(.x0, 4), // jal block1
        encode.fmv_s(.f10, .f0), // fmv.s fa0, ft0  (return v1)
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects block arguments on a jump edge" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block0 = try func.appendBlock();
    const v0 = try func.appendBlockParam(block0, i32_t);
    const block1 = try func.appendBlock();
    const v1 = try func.appendBlockParam(block1, i32_t);

    try func.setJump(block0, block1, &.{v0});
    func.setTerminator(block1, .{ .ret = v1 });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // block0: mv t0, a0  (pass v0 into v1's register) then jal block1.
    // block1: mv a0, t0  (return v1) then ret.
    try std.testing.expectEqualSlices(u32, &.{
        encode.addi(.x5, .x10, 0), // mv t0, a0
        encode.jal(.x0, 4), // jal block1
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects unsigned division with divu" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, u32_t);
    const b = try func.appendBlockParam(entry, u32_t);
    const q = try func.appendInst(entry, u32_t, .{ .arith = .{ .op = .div, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = q });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.divu(.x5, .x10, .x11), // unsigned: divu t0, a0, a1
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects an unsigned comparison with sltu" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, u32_t);
    const b = try func.appendBlockParam(entry, u32_t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = c });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.sltu(.x5, .x10, .x11), // unsigned: sltu t0, a0, a1
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects an integer comparison" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = c });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.slt(.x5, .x10, .x11), // slt t0, a0, a1
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects a conditional branch" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);
    const block0 = try func.appendBlock();
    const c = try func.appendBlockParam(block0, bool_t);
    const block1 = try func.appendBlock();
    const block2 = try func.appendBlock();

    try func.appendIf(block0, c, .{ .target = block1 }, .{ .target = block2 });
    func.setTerminator(block1, .{ .ret = null });
    func.setTerminator(block2, .{ .ret = null });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // Layout: block0 [bne, jal], block1 [ret], block2 [ret].
    try std.testing.expectEqualSlices(u32, &.{
        encode.bne(.x10, .x0, 8), // bne c, x0, block1
        encode.jal(.x0, 8), // jal x0, block2
        encode.jalr(.x0, .x1, 0), // block1: ret
        encode.jalr(.x0, .x1, 0), // block2: ret
    }, code);
}

test "selects a wide constant with lui+addi" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const v = try func.appendInst(entry, i32_t, .{ .iconst = 0x12345 });
    func.setTerminator(entry, .{ .ret = v });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.lui(.x5, 0x12), // lui t0, 0x12
        encode.addi(.x5, .x5, 0x345), // addi t0, t0, 0x345
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects a small constant" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const v = try func.appendInst(entry, i32_t, .{ .iconst = 42 });
    func.setTerminator(entry, .{ .ret = v });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.addi(.x5, .x0, 42), // li t0, 42  ==  addi t0, zero, 42
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "register allocation reuses registers across a long value chain" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);

    // A 12-deep chain: each value is used only by the next, so at most two are
    // live at once. The naive counter would overflow the 7-register pool. The
    // allocator reuses registers and fits.
    var last = a;
    for (0..12) |_| {
        last = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = last, .rhs = last } });
    }
    func.setTerminator(entry, .{ .ret = last });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "full pipeline: high-profile struct IR lowers to machine bytes" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const st = try func.types.intern(.{ .@"struct" = &.{ i32_t, i32_t } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);

    // s = { a, b }, then return s.#0 + b   (uses a high-profile aggregate)
    const s = try func.appendStructNew(entry, st, &.{ a, b });
    const f0 = try func.appendInst(entry, i32_t, .{ .extract = .{ .aggregate = s, .index = 0 } });
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = f0, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    // Legalize the aggregate away, then select and emit.
    try ir.legalize.legalize(std.testing.allocator, &func);

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // The struct collapsed: `s.#0` forwarded to `a`, so this is just `a + b`.
    try std.testing.expectEqualSlices(u32, &.{
        encode.add(.x5, .x10, .x11), // add t0, a0, a1
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);

    const bytes = try emit.emitBytes(std.testing.allocator, code);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqual(@as(usize, 12), bytes.len);
}

test "end-to-end: parse text, schedule, then compile to machine code" {
    const text =
        \\fn {
        \\  block0(v0: i32, v1: i32):
        \\    let v2 = v0 * v1
        \\    let v3 = v2 + v0
        \\    ret v3
        \\}
    ;
    var func = try ir.parser.parse(std.testing.allocator, text);
    defer func.deinit();

    try schedule.scheduleFunction(std.testing.allocator, &func);
    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // The dependent chain keeps its order through scheduling and lowers cleanly.
    try std.testing.expectEqualSlices(u32, &.{
        encode.mul(.x5, .x10, .x11), // v2 = v0 * v1
        encode.add(.x7, .x5, .x10), // v3 = v2 + v0
        encode.addi(.x10, .x7, 0), // mv a0, v3
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects a float comparison" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);
    const b = try func.appendBlockParam(entry, f32_t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = c });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.flt_s(.x5, .f10, .f11), // flt.s t0, fa0, fa1
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects float loads and stores" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendInst(entry, f32_t, .{ .load = .{ .ptr = p } });
    try func.appendStore(entry, v, p);
    func.setTerminator(entry, .{ .ret = null });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.flw(.f0, .x10, 0), // flw ft0, 0(a0)
        encode.fsw(.f0, .x10, 0), // fsw ft0, 0(a0)
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects a float constant" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const v = try func.appendInst(entry, f32_t, .{ .fconst = 1.5 });
    func.setTerminator(entry, .{ .ret = v });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // 1.5f == 0x3FC00000.  lui x6, 0x3FC00, then addi x6, x6, 0, then fmv.w.x ft0, x6.
    try std.testing.expectEqualSlices(u32, &.{
        encode.lui(.x6, 0x3FC00),
        encode.addi(.x6, .x6, 0),
        encode.fmv_w_x(.f0, .x6),
        encode.fmv_s(.f10, .f0), // fmv.s fa0, ft0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "immediate arithmetic lowers without materializing the constant" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const a = try func.appendArithImm(entry, i32_t, .add, x, 5); // x + 5  -> addi
    const b = try func.appendArithImm(entry, i32_t, .shl, a, 2); // a << 2 -> slli
    const c = try func.appendArithImm(entry, i32_t, .bit_and, b, 255); // b & 255 -> andi
    func.setTerminator(entry, .{ .ret = c });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.addi(.x5, .x10, 5), // x + 5
        encode.slli(.x7, .x5, 2), // a << 2
        encode.andi(.x5, .x7, 255), // b & 255
        encode.addi(.x10, .x5, 0), // mv a0, c
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects an int-to-float conversion" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, i32_t);
    const f = try func.appendInst(entry, f32_t, .{ .convert = .{ .value = x } });
    func.setTerminator(entry, .{ .ret = f });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.fcvt_s_w(.f0, .x10), // fcvt.s.w ft0, a0
        encode.fmv_s(.f10, .f0), // fmv.s fa0, ft0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects a float-to-int conversion" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const x = try func.appendBlockParam(entry, f32_t);
    const i = try func.appendInst(entry, i32_t, .{ .convert = .{ .value = x } });
    func.setTerminator(entry, .{ .ret = i });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.fcvt_w_s(.x5, .f10), // fcvt.w.s t0, fa0 (round-toward-zero)
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects a float add function" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f32_t);
    const b = try func.appendBlockParam(entry, f32_t);
    const sum = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.fadd_s(.f0, .f10, .f11), // fadd.s ft0, fa0, fa1
        encode.fmv_s(.f10, .f0), // fmv.s fa0, ft0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

test "selects a simple add function to RISC-V" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    try std.testing.expectEqualSlices(u32, &.{
        encode.add(.x5, .x10, .x11), // add t0, a0, a1
        encode.addi(.x10, .x5, 0), // mv a0, t0
        encode.jalr(.x0, .x1, 0), // ret
    }, code);
}

/// Run a vector-float function under qemu-riscv64 with the V extension. An entry
/// stub loads the f32 args into fa0.., calls the function, and exits with the f32
/// result's bits, returning the low byte. Skips when qemu-riscv64 is not on PATH.
fn runRvvFloat(allocator: std.mem.Allocator, func: *Function, fargs: []const f32) !u8 {
    const code = try selectFunction(allocator, func);
    defer allocator.free(code);
    var program: std.ArrayList(u32) = .empty;
    defer program.deinit(allocator);
    for (fargs, 0..) |fa, i| {
        const bits: u32 = @bitCast(fa);
        const hi: u20 = @truncate((bits +% 0x800) >> 12);
        const lo: i12 = @bitCast(@as(u12, @truncate(bits)));
        try program.append(allocator, encode.lui(.x5, hi));
        try program.append(allocator, encode.addi(.x5, .x5, lo));
        try program.append(allocator, encode.fmv_w_x(@enumFromInt(@as(u5, @intCast(10 + i))), .x5)); // fa_i
    }
    try program.append(allocator, encode.jal(.x1, 16)); // jal ra, function (skip the 3-word epilogue)
    try program.append(allocator, encode.fmv_x_w(.x10, .f10)); // fmv.x.w a0, fa0
    try program.append(allocator, encode.addi(.x17, .x0, 93)); // li a7, 93 (exit)
    try program.append(allocator, encode.ecall());
    try program.appendSlice(allocator, code);

    const bytes = std.mem.sliceAsBytes(program.items);
    const ld = @import("ld.zig");
    const elf = try ld.writeElfExec(allocator, bytes, bytes.len, 0x10000, 0x10000);
    defer allocator.free(elf);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.elf", .data = elf, .flags = .{ .permissions = .executable_file } });
    const run = std.process.run(allocator, std.testing.io, .{
        .argv = &.{ "qemu-riscv64", "-cpu", "rv64,v=true,vlen=128", "a.elf" },
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest, // qemu-riscv64 not on PATH
        else => return e,
    };
    defer allocator.free(run.stdout);
    defer allocator.free(run.stderr);
    return switch (run.term) {
        .exited => |ec| ec,
        else => error.BackendFailed,
    };
}

test "qemu-riscv-V: a packed <4 x f32> add runs on RVV and reduces to the right sum" {
    const allocator = std.testing.allocator;
    const V = ir.function.Value;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const b = try func.appendBlock();
    var ap: [4]V = undefined;
    var bp: [4]V = undefined;
    for (0..4) |i| ap[i] = try func.appendBlockParam(b, t);
    for (0..4) |i| bp[i] = try func.appendBlockParam(b, t);
    const va = try func.appendInst(b, v4, .{ .struct_new = .{ .fields = try func.internValueList(&ap) } });
    const vb = try func.appendInst(b, v4, .{ .struct_new = .{ .fields = try func.internValueList(&bp) } });
    const vc = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    var c: [4]V = undefined;
    for (0..4) |i| c[i] = try func.appendInst(b, t, .{ .extract = .{ .aggregate = vc, .index = @intCast(i) } });
    const s01 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c[0], .rhs = c[1] } });
    const s012 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = c[2] } });
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = c[3] } });
    func.setTerminator(b, .{ .ret = s });

    const av = [4]f32{ 1.1, 2.2, 3.3, 4.4 };
    const bv = [4]f32{ 5.5, 6.6, 7.7, 8.8 };
    const ec = try runRvvFloat(allocator, &func, &(av ++ bv));
    var cc: [4]f32 = undefined;
    for (0..4) |i| cc[i] = av[i] + bv[i];
    const expected = ((cc[0] + cc[1]) + cc[2]) + cc[3];
    try std.testing.expectEqual(@as(u8, @truncate(@as(u32, @bitCast(expected)))), ec);
}

test "qemu-riscv-V: a chained (a+b)*a keeps the intermediate in a vector register" {
    const allocator = std.testing.allocator;
    const V = ir.function.Value;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const b = try func.appendBlock();
    var ap: [4]V = undefined;
    var bp: [4]V = undefined;
    for (0..4) |i| ap[i] = try func.appendBlockParam(b, t);
    for (0..4) |i| bp[i] = try func.appendBlockParam(b, t);
    const va = try func.appendInst(b, v4, .{ .struct_new = .{ .fields = try func.internValueList(&ap) } });
    const vb = try func.appendInst(b, v4, .{ .struct_new = .{ .fields = try func.internValueList(&bp) } });
    const vc = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    const vp = try func.appendInst(b, v4, .{ .arith = .{ .op = .mul, .lhs = vc, .rhs = va } }); // vc and va both live
    var c: [4]V = undefined;
    for (0..4) |i| c[i] = try func.appendInst(b, t, .{ .extract = .{ .aggregate = vp, .index = @intCast(i) } });
    const s01 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c[0], .rhs = c[1] } });
    const s012 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = c[2] } });
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = c[3] } });
    func.setTerminator(b, .{ .ret = s });

    const av = [4]f32{ 1.1, 2.2, 3.3, 4.4 };
    const bv = [4]f32{ 5.5, 6.6, 7.7, 8.8 };
    const ec = try runRvvFloat(allocator, &func, &(av ++ bv));
    var pp: [4]f32 = undefined;
    for (0..4) |i| pp[i] = (av[i] + bv[i]) * av[i];
    const expected = ((pp[0] + pp[1]) + pp[2]) + pp[3];
    try std.testing.expectEqual(@as(u8, @truncate(@as(u32, @bitCast(expected)))), ec);
}

test "qemu-riscv-V: a <4 x f32> round-trips through an alloca slot (vse32 then vle32)" {
    const allocator = std.testing.allocator;
    const V = ir.function.Value;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    var ap: [4]V = undefined;
    var bp: [4]V = undefined;
    for (0..4) |i| ap[i] = try func.appendBlockParam(b, t);
    for (0..4) |i| bp[i] = try func.appendBlockParam(b, t);
    const va = try func.appendInst(b, v4, .{ .struct_new = .{ .fields = try func.internValueList(&ap) } });
    const vb = try func.appendInst(b, v4, .{ .struct_new = .{ .fields = try func.internValueList(&bp) } });
    const vc = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = v4 } });
    try func.appendStore(b, vc, slot); // vse32 the vector to the slot
    const vd = try func.appendInst(b, v4, .{ .load = .{ .ptr = slot } }); // vle32 it back
    var c: [4]V = undefined;
    for (0..4) |i| c[i] = try func.appendInst(b, t, .{ .extract = .{ .aggregate = vd, .index = @intCast(i) } });
    const s01 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c[0], .rhs = c[1] } });
    const s012 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = c[2] } });
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = c[3] } });
    func.setTerminator(b, .{ .ret = s });

    const av = [4]f32{ 1.1, 2.2, 3.3, 4.4 };
    const bv = [4]f32{ 5.5, 6.6, 7.7, 8.8 };
    const ec = try runRvvFloat(allocator, &func, &(av ++ bv));
    var cc: [4]f32 = undefined;
    for (0..4) |i| cc[i] = av[i] + bv[i];
    const expected = ((cc[0] + cc[1]) + cc[2]) + cc[3];
    try std.testing.expectEqual(@as(u8, @truncate(@as(u32, @bitCast(expected)))), ec);
}

test "qemu-riscv-V: high vector pressure spills whole vectors to 16-byte slots and reloads them" {
    const allocator = std.testing.allocator;
    const V = ir.function.Value;
    const N = 30; // > 27 allocatable vector registers, so several vectors spill (vse32/vle32)
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const b = try func.appendBlock();
    var vs: [N]V = undefined;
    for (0..N) |i| {
        const c = try func.appendInst(b, t, .{ .fconst = @as(f64, @floatFromInt(i)) + 0.1 });
        vs[i] = try func.appendInst(b, v4, .{ .struct_new = .{ .fields = try func.internValueList(&.{ c, c, c, c }) } });
    }
    var acc = vs[0];
    for (1..N) |i| acc = try func.appendInst(b, v4, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vs[i] } });
    var c: [4]V = undefined;
    for (0..4) |i| c[i] = try func.appendInst(b, t, .{ .extract = .{ .aggregate = acc, .index = @intCast(i) } });
    const s01 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = c[0], .rhs = c[1] } });
    const s012 = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = c[2] } });
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = c[3] } });
    func.setTerminator(b, .{ .ret = s });

    const ec = try runRvvFloat(allocator, &func, &.{}); // no float args, vectors built from fconsts
    var lane: f32 = @floatCast(@as(f64, 0.1));
    for (1..N) |i| lane += @as(f32, @floatCast(@as(f64, @floatFromInt(i)) + 0.1));
    const expected = ((lane + lane) + lane) + lane; // four equal lanes, same reduction order
    try std.testing.expectEqual(@as(u8, @truncate(@as(u32, @bitCast(expected)))), ec);
}

test "qemu-riscv-V: a vector crosses a block edge via a merge-block vector parameter" {
    const allocator = std.testing.allocator;
    const V = ir.function.Value;
    // (a0 < b0) ? sum(a) : sum(b), where the chosen vector reaches the merge block as a
    // <4 x f32> parameter, so a vmv.v.v carries it across each edge.
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = t } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    var ap: [4]V = undefined;
    var bp: [4]V = undefined;
    for (0..4) |i| ap[i] = try func.appendBlockParam(entry, t);
    for (0..4) |i| bp[i] = try func.appendBlockParam(entry, t);
    const va = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&ap) } });
    const vb = try func.appendInst(entry, v4, .{ .struct_new = .{ .fields = try func.internValueList(&bp) } });
    const lt = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = ap[0], .rhs = bp[0] } });
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const m = try func.appendBlockParam(merge, v4);
    try func.appendIf(entry, lt, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });
    func.setTerminator(then_b, .{ .jump = .{ .target = merge, .args = try func.internValueList(&.{va}) } });
    func.setTerminator(else_b, .{ .jump = .{ .target = merge, .args = try func.internValueList(&.{vb}) } });
    var c: [4]V = undefined;
    for (0..4) |i| c[i] = try func.appendInst(merge, t, .{ .extract = .{ .aggregate = m, .index = @intCast(i) } });
    const s01 = try func.appendInst(merge, t, .{ .arith = .{ .op = .add, .lhs = c[0], .rhs = c[1] } });
    const s012 = try func.appendInst(merge, t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = c[2] } });
    const s = try func.appendInst(merge, t, .{ .arith = .{ .op = .add, .lhs = s012, .rhs = c[3] } });
    func.setTerminator(merge, .{ .ret = s });

    const a1 = [4]f32{ 1.1, 2.2, 3.3, 4.5 };
    const b1 = [4]f32{ 9.9, 1.0, 1.0, 1.0 };
    const ec1 = try runRvvFloat(allocator, &func, &(a1 ++ b1)); // a0 < b0 -> then -> sum(a)
    try std.testing.expectEqual(@as(u8, @truncate(@as(u32, @bitCast(((a1[0] + a1[1]) + a1[2]) + a1[3])))), ec1);

    const a2 = [4]f32{ 9.9, 2.2, 3.3, 4.5 };
    const b2 = [4]f32{ 5.5, 6.6, 7.7, 8.5 };
    const ec2 = try runRvvFloat(allocator, &func, &(a2 ++ b2)); // a0 >= b0 -> else -> sum(b)
    try std.testing.expectEqual(@as(u8, @truncate(@as(u32, @bitCast(((b2[0] + b2[1]) + b2[2]) + b2[3])))), ec2);
}
