//! NVIDIA SASS instruction selection: lowers a Vulcan IR function to a compute
//! kernel or graphics shader.
//!
//! Kernels are leaf (no call stack, inline before isel). ~255 GPRs, so allocation
//! is naive: a pointer takes an even-aligned register pair, a boolean a predicate
//! (P0..P5, P6 is the 64-bit-add carry). Kernel ABI: parameters arrive in constant
//! bank 0 at `param_base`. A value-returning kernel reads a 64-bit output pointer
//! first (its `ret` stores there), a void compute kernel has none. Each parameter
//! is then sourced in order: the tagged invocation id from the hardware thread id
//! (S2R), a pointer as a 64-bit constant-bank pair load, a scalar as a single load.
//! Memory load/store are LDG/STG through a 64-bit pointer pair. Pointer arithmetic
//! is a 64-bit IADD3 carry chain (low add carries out, high `.X` add carries in).
//! Control flow is BRA with block-parameter edge moves. schedule.zig then assigns
//! write barriers to the variable-latency ops (LDG/S2R) and waits to consumers.
//!
//! Validation is structural (the emitted instruction stream). Live execution is
//! deferred to prism's compute dispatch. Unsupported IR (calls, aggregates, integer
//! divide) returns `error.Unsupported`.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const schedule = @import("schedule.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Terminator = ir.function.Terminator;
const Inst = encode.Inst;

pub const Error = std.mem.Allocator.Error || error{Unsupported};

/// The constant-bank byte offset where kernel parameters begin. 0x160 is the
/// Volta..Ampere kernel-param base. The dispatch side (prism's QMD) must match.
pub const param_base: u16 = 0x160;
const bank0: u5 = 0;

/// Graphics prologue padding: throwaway instructions emitted before the first
/// attribute fetch / color write, to wait out the asynchronous hardware delivery
/// of sysvals/barycentrics into the low registers (clean threshold 4, 6 for
/// margin). Written to a dedicated high scratch register, not RZ: a write to RZ
/// can retire instantly and not consume the cycles the delivery window needs.
/// The register allocator excludes it from its pool (see assignLocs).
const graphics_prologue_pad: u32 = 6;
const graphics_pad_reg: u8 = 40;

/// Reserved registers: R0/R1 scratch, R2:R3 the 64-bit output pointer. Values are
/// assigned GPRs from R4 up.
const r_scratch: u8 = 0;
const r_scratch2: u8 = 1; // second prologue scratch (invocation-id computation)
const r_outptr: u8 = 2; // pair R2:R3
const value_reg_base: u8 = 4;

/// A compiled kernel: the SASS instruction stream and the register count the
/// launch descriptor needs.
pub const Kernel = struct {
    code: []u32,
    reg_count: u32,

    pub fn deinit(self: *Kernel, allocator: std.mem.Allocator) void {
        allocator.free(self.code);
    }
};

/// Where each IR value lives: a general register, or a predicate register (for
/// booleans produced by a compare).
const Loc = union(enum) { gpr: u8, pred: u8 };

const Fixup = struct { at: usize, target: u32 }; // BRA word index -> block

/// The shader stage being compiled. Compute kernels source parameters from the
/// constant bank and store via STG. Graphics shaders use the attribute interface
/// (vertex inputs via ALD, fragment inputs via IPA, outputs via AST).
pub const Stage = enum { compute, vertex, fragment };

/// Lower `func` to a SASS compute kernel. The caller owns the result.
pub fn compileKernel(allocator: std.mem.Allocator, func: *const Function) Error!Kernel {
    return compileShader(allocator, func, .compute);
}

/// Lower `func` to a SASS shader for `stage`. The caller owns the result.
pub fn compileShader(allocator: std.mem.Allocator, func: *const Function, stage: Stage) Error!Kernel {
    const nblocks = func.blockCount();
    if (nblocks == 0) return error.Unsupported;

    var loc = std.AutoHashMapUnmanaged(Value, Loc){};
    defer loc.deinit(allocator);
    var max_reg: u8 = r_outptr + 1; // the output pointer pair is always live
    try assignLocs(allocator, func, &loc, &max_reg);

    var code: std.ArrayList(Inst) = .empty;
    defer code.deinit(allocator);
    var fixups: std.ArrayList(Fixup) = .empty;
    defer fixups.deinit(allocator);
    var block_start = try allocator.alloc(usize, nblocks);
    defer allocator.free(block_start);

    const eparams = func.blockParams(@enumFromInt(0));
    if (stage == .compute) {
        // A value-returning kernel reads an output pointer from the front of the
        // constant bank (its `ret` stores there), a void compute kernel has none.
        var cursor: u16 = param_base;
        if (returnsValue(func)) {
            try code.append(allocator, encode.ldc(r_outptr, bank0, cursor, .{})); // outptr lo
            try code.append(allocator, encode.ldc(r_outptr + 1, bank0, cursor + 4, .{})); // outptr hi
            cursor += 8;
        }
        for (eparams) |p| {
            if (isInvocationId(func, p)) {
                // gid.x = blockIdx.x * local_size_x + threadIdx.x (workgroup size is a
                // compile-time constant, thread/block ids come from S2R).
                const gid = gprOf(loc, p);
                try code.append(allocator, encode.movImm(gid, localSizeX(func), .{}));
                try code.append(allocator, encode.s2r(r_scratch, encode.SR_TID_X, .{})); // threadIdx.x
                try code.append(allocator, encode.s2r(r_scratch2, encode.SR_CTAID_X, .{})); // blockIdx.x
                try code.append(allocator, encode.imad(gid, r_scratch2, gid, r_scratch, .{}));
            } else if (isPtr(func, p)) {
                const lo = gprOf(loc, p);
                try code.append(allocator, encode.ldc(lo, bank0, cursor, .{}));
                try code.append(allocator, encode.ldc(lo + 1, bank0, cursor + 4, .{}));
                cursor += 8;
            } else {
                try code.append(allocator, encode.ldc(gprOf(loc, p), bank0, cursor, .{}));
                cursor += 4;
            }
        }
    } else {
        // The SMs deliver the hardware-provided inputs (vertex-id / fragment
        // barycentrics + sysvals) into the low registers asynchronously a few
        // instructions into warp execution. An attribute fetch or color write
        // issued before that window closes reads or is clobbered by zeros, so pad
        // the prologue with throwaway MOVs (verified: clean threshold 4, 6 for
        // margin) to a high scratch register before any ALD/IPA.
        var pad: u32 = 0;
        while (pad < graphics_prologue_pad) : (pad += 1) {
            try code.append(allocator, encode.movImm(graphics_pad_reg, pad, .{}));
        }
        // Each input parameter loads from its attribute slot: a vertex shader
        // fetches the attribute (ALD), a fragment shader interpolates it (IPA). The
        // slot is the parameter's `attr` tag (default ATTR_GENERIC0). These are
        // variable-latency. The scoreboard pass adds the consumer waits.
        for (eparams) |p| {
            const attr = attrTag(func, p, "attr") orelse encode.ATTR_GENERIC0;
            const rd = gprOf(loc, p);
            try code.append(allocator, if (stage == .vertex)
                encode.ald(rd, attr, 1, .{})
            else
                encode.ipa(rd, attr, .{}));
        }
    }

    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        block_start[bi] = code.items.len;
        var terminated = false;

        for (func.blockInsts(block)) |inst| {
            try lowerInst(allocator, func, &loc, &code, inst);
            if (func.opcode(inst) == .@"if") {
                try emitIf(allocator, func, &loc, &code, &fixups, func.opcode(inst).@"if");
                terminated = true;
            }
        }

        if (!terminated) switch (func.terminator(block) orelse ir.function.Terminator{ .ret = null }) {
            .ret => |v| {
                if (v) |value| {
                    const src = gprOf(loc, value);
                    try code.append(allocator, encode.stgU32(r_outptr, src, .{}));
                }
                try code.append(allocator, encode.exit(.{ .stall = 1 }));
            },
            .jump => |j| try emitJump(allocator, func, &loc, &code, &fixups, j),
        };
    }

    // Scoreboard scheduling: write barriers on variable-latency ops (LDG/S2R) and
    // waits on their consumers, so results are read only once ready.
    schedule.schedule(code.items);

    // Patch each BRA's relative displacement (bytes from the next instruction).
    for (fixups.items) |f| {
        const next = f.at + 1;
        const delta: i32 = @intCast((@as(i64, @intCast(block_start[f.target])) - @as(i64, @intCast(next))) * 16);
        const pred = (code.items[f.at][0] >> 12) & 0x7;
        const neg = ((code.items[f.at][0] >> 15) & 1) == 1;
        code.items[f.at] = encode.bra(delta, .{ .pred = @intCast(pred), .pred_neg = neg });
    }

    // Flatten to dwords.
    const out = try allocator.alloc(u32, code.items.len * 4);
    errdefer allocator.free(out);
    for (code.items, 0..) |w, i| @memcpy(out[i * 4 ..][0..4], &w);
    return .{ .code = out, .reg_count = regCount(max_reg) };
}

fn regCount(max_reg: u8) u32 {
    const used = @as(u32, max_reg) + 1;
    return @max(16, (used + 7) & ~@as(u32, 7)); // hardware granularity: multiples of 8, min 16
}

fn gprOf(loc: std.AutoHashMapUnmanaged(Value, Loc), v: Value) u8 {
    return switch (loc.get(v).?) {
        .gpr => |r| r,
        .pred => unreachable, // a predicate used where a GPR was expected
    };
}

fn predOf(loc: std.AutoHashMapUnmanaged(Value, Loc), v: Value) u8 {
    return switch (loc.get(v).?) {
        .pred => |p| p,
        .gpr => unreachable,
    };
}

const carry_pred: u8 = 6; // predicate reserved for the 64-bit-add carry chain
const Interval = struct { value: Value, start: u32, end: u32 };

fn lessByStart(_: void, a: Interval, b: Interval) bool {
    return a.start < b.start;
}

/// Linear-scan register allocation with reuse: a register frees when its value's
/// last use passes, so short-lived values (e.g. the 32 compares of a lowered
/// integer division) share a small set of registers instead of each taking a fresh
/// one. Pointers take even-aligned GPR pairs, booleans take predicates P0..P5 (P6
/// is the 64-bit-add carry scratch). No spilling: a class running out is
/// `error.Unsupported`, which a real kernel should never hit (250+ GPRs).
fn assignLocs(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), max_reg: *u8) Error!void {
    const nval = func.valueCount();
    if (nval == 0) return;
    const nblocks = func.blockCount();

    // Live intervals (def..last-use) over a block-order linearization, extended by
    // backward liveness so loop-carried values stay live across the loop body.
    const def_pos = try allocator.alloc(u32, nval);
    defer allocator.free(def_pos);
    const last_use = try allocator.alloc(u32, nval);
    defer allocator.free(last_use);
    const block_end = try allocator.alloc(u32, nblocks);
    defer allocator.free(block_end);
    @memset(def_pos, 0);
    for (last_use) |*l| l.* = 0;

    var pos: u32 = 0;
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| {
            def_pos[@intFromEnum(p)] = pos;
            last_use[@intFromEnum(p)] = pos;
        }
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            forEachUse(func, inst, last_use, pos);
            if (func.instResult(inst)) |r| def_pos[@intFromEnum(r)] = pos;
            pos += 1;
        }
        block_end[bi] = pos;
        if (func.terminator(block)) |term| forEachTermUse(func, term, last_use, pos);
        pos += 1;
    }
    try extendLiveRanges(allocator, func, last_use, block_end);

    var ivals = try allocator.alloc(Interval, nval);
    defer allocator.free(ivals);
    for (0..nval) |i| ivals[i] = .{ .value = @enumFromInt(i), .start = def_pos[i], .end = last_use[i] };
    std.mem.sort(Interval, ivals, {}, lessByStart);

    // Free pools: GPRs R4..R254 (R0/R1 scratch, R2:R3 the output pointer), and
    // predicates P0..P5.
    var gpr_free = [_]bool{false} ** 256;
    for (value_reg_base..encode.RZ) |r| gpr_free[r] = true;
    gpr_free[graphics_pad_reg] = false; // reserved as the graphics prologue pad scratch
    var pred_free = [_]bool{true} ** carry_pred;

    const Active = struct { end: u32, loc: Loc, is_ptr: bool };
    var active: std.ArrayList(Active) = .empty;
    defer active.deinit(allocator);

    for (ivals) |iv| {
        // Expire intervals that ended before this one starts, freeing their regs.
        var w: usize = 0;
        for (active.items) |a| {
            if (a.end < iv.start) {
                switch (a.loc) {
                    .gpr => |r| {
                        gpr_free[r] = true;
                        if (a.is_ptr) gpr_free[r + 1] = true;
                    },
                    .pred => |p| pred_free[p] = true,
                }
            } else {
                active.items[w] = a;
                w += 1;
            }
        }
        active.shrinkRetainingCapacity(w);

        const v = iv.value;
        const l: Loc = if (isBool(func, v)) blk: {
            const p = firstFree(pred_free[0..]) orelse return error.Unsupported;
            pred_free[p] = false;
            break :blk .{ .pred = @intCast(p) };
        } else if (isPtr(func, v)) blk: {
            const r = firstFreePair(gpr_free[0..]) orelse return error.Unsupported;
            gpr_free[r] = false;
            gpr_free[r + 1] = false;
            if (r + 1 > max_reg.*) max_reg.* = @intCast(r + 1);
            break :blk .{ .gpr = @intCast(r) };
        } else blk: {
            const r = firstFreeSingle(gpr_free[0..]) orelse return error.Unsupported;
            gpr_free[r] = false;
            if (r > max_reg.*) max_reg.* = @intCast(r);
            break :blk .{ .gpr = @intCast(r) };
        };
        try loc.put(allocator, v, l);
        try active.append(allocator, .{ .end = iv.end, .loc = l, .is_ptr = isPtr(func, v) });
    }
}

fn firstFree(pool: []const bool) ?usize {
    for (pool, 0..) |f, i| if (f) return i;
    return null;
}

fn firstFreeSingle(gpr_free: []const bool) ?usize {
    for (value_reg_base..encode.RZ) |r| if (gpr_free[r]) return r;
    return null;
}

fn firstFreePair(gpr_free: []const bool) ?usize {
    var r: usize = value_reg_base; // R4 is even, so the scan keeps pairs aligned
    while (r + 1 < encode.RZ) : (r += 2) if (gpr_free[r] and gpr_free[r + 1]) return r;
    return null;
}

fn isBool(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .bool;
}

fn isPtr(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .ptr;
}

/// Whether `v` is the invocation-id parameter the frontend tagged (sourced from
/// the hardware thread id, not a uniform kernel argument).
fn isInvocationId(func: *const Function, v: Value) bool {
    var it = func.attributesOf(.{ .value = v });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "builtin")) return true,
        else => {},
    };
    return false;
}

/// A `vulcan.gpu` integer attribute named `key` attached to value `v` (a graphics
/// attribute slot), or null if absent.
fn attrTag(func: *const Function, v: Value, key: []const u8) ?u16 {
    var it = func.attributesOf(.{ .value = v });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, key)) {
            return switch (c.value) {
                .int => |n| @intCast(n),
                else => null,
            };
        },
        else => {},
    };
    return null;
}

/// The workgroup x dimension the frontend recorded (the LocalSize execution mode),
/// used to fold the block offset into the invocation id. Defaults to 1.
fn localSizeX(func: *const Function) u32 {
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan.gpu") and std.mem.eql(u8, c.key, "local_size_x")) {
            return switch (c.value) {
                .int => |n| @intCast(n),
                else => 1,
            };
        },
        else => {},
    };
    return 1;
}

fn returnsValue(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        if (func.terminator(@enumFromInt(bi))) |t| switch (t) {
            .ret => |v| if (v != null) return true,
            else => {},
        };
    }
    return false;
}

fn lowerInst(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), inst: ir.function.Inst) Error!void {
    switch (func.opcode(inst)) {
        .iconst => |c| {
            // A graphics output-attribute store pointer is a tag-carrier iconst
            // (the slot), never a real value the SASS computes. Skip emitting it.
            const result = func.instResult(inst).?;
            if (attrTag(func, result, "out_attr") != null or attrTag(func, result, "color_out") != null) return;
            const rd = gprOf(loc.*, result);
            try code.append(allocator, encode.movImm(rd, @truncate(@as(u64, @bitCast(c))), .{}));
        },
        .fconst => |val| {
            const rd = gprOf(loc.*, func.instResult(inst).?);
            const bits: u32 = @bitCast(@as(f32, @floatCast(val)));
            try code.append(allocator, encode.movImm(rd, bits, .{}));
        },
        .arith => |a| {
            const result = func.instResult(inst).?;
            if (isPtr(func, result) and a.op == .add) {
                // 64-bit pointer add: (dst:dst+1) = (base:base+1) + zext(offset). The
                // low add produces a carry that the high add (`.X`) consumes.
                const dlo = gprOf(loc.*, result);
                const base = gprOf(loc.*, a.lhs); // pointer pair (lo:hi)
                const offset = gprOf(loc.*, a.rhs); // 32-bit, zero-extended
                try code.append(allocator, encode.iadd3CarryOut(dlo, base, offset, carry_pred, .{}));
                try code.append(allocator, encode.iadd3CarryIn(dlo + 1, base + 1, encode.RZ, carry_pred, .{}));
            } else {
                const rd = gprOf(loc.*, result);
                const ra = gprOf(loc.*, a.lhs);
                const rb = gprOf(loc.*, a.rhs);
                try code.append(allocator, try arith(func, a.op, rd, ra, rb, a.lhs));
            }
        },
        .load => |l| {
            // LDG from the 64-bit pointer pair into the 32-bit result register. The
            // load is variable latency. The scoreboard scheduler assigns its write
            // barrier and the wait on each consumer.
            const rd = gprOf(loc.*, func.instResult(inst).?);
            try code.append(allocator, encode.ldgU32(rd, gprOf(loc.*, l.ptr), .{}));
        },
        .store => |st| {
            // A store whose pointer is tagged with a graphics output attribute goes
            // to that attribute (AST). A fragment color output is moved into the ROP
            // color register (R0..R3), otherwise it is an ordinary global store.
            if (attrTag(func, st.ptr, "out_attr")) |attr| {
                try code.append(allocator, encode.ast(attr, gprOf(loc.*, st.value), 1, .{}));
            } else if (attrTag(func, st.ptr, "color_out")) |comp| {
                // The fragment shader's render-target color: the ROP reads color0
                // from R0..R3 at EXIT, so component `comp` moves into R<comp>.
                // Emitted inline (the IR liveness that drives register allocation
                // models the move at this store, deferring it past the value's last
                // use would let the allocator reuse the source register). The
                // prologue pad already covers the async input-delivery window.
                if (comp < 4) try code.append(allocator, encode.movReg(@intCast(comp), gprOf(loc.*, st.value), .{}));
            } else {
                try code.append(allocator, encode.stgU32(gprOf(loc.*, st.ptr), gprOf(loc.*, st.value), .{}));
            }
        },
        .arith_imm => |a| {
            const rd = gprOf(loc.*, func.instResult(inst).?);
            const ra = gprOf(loc.*, a.lhs);
            try code.append(allocator, encode.movImm(r_scratch, @truncate(@as(u64, @bitCast(a.imm))), .{}));
            try code.append(allocator, try arith(func, a.op, rd, ra, r_scratch, a.lhs));
        },
        .icmp => |cmp| {
            const pd = predOf(loc.*, func.instResult(inst).?);
            try code.append(allocator, encode.isetp(pd, gprOf(loc.*, cmp.lhs), gprOf(loc.*, cmp.rhs), cmpOf(cmp.op), isSigned(func, cmp.lhs), .{}));
        },
        .select => |s| {
            const rd = gprOf(loc.*, func.instResult(inst).?);
            try code.append(allocator, encode.sel(rd, gprOf(loc.*, s.then), gprOf(loc.*, s.@"else"), predOf(loc.*, s.cond), .{}));
        },
        .convert => |cv| {
            const result = func.instResult(inst).?;
            const rd = gprOf(loc.*, result);
            const rs = gprOf(loc.*, cv.value);
            const dst_float = isFloat(func, result);
            const src_float = isFloat(func, cv.value);
            if (src_float and !dst_float) {
                try code.append(allocator, encode.f2i(rd, rs, isSignedRaw(func, result), .{})); // f32 -> i32
            } else if (!src_float and dst_float) {
                try code.append(allocator, encode.i2f(rd, rs, isSignedRaw(func, cv.value), .{})); // i32 -> f32
            } else {
                return error.Unsupported; // int<->int width change / f32<->f64 not modeled yet
            }
        },
        .@"if" => {}, // handled by the caller (it terminates the block)
        else => return error.Unsupported,
    }
}

fn arith(func: *const Function, op: ir.function.BinOp, rd: u8, ra: u8, rb: u8, lhs: Value) Error!Inst {
    const is_float = isFloat(func, lhs);
    return switch (op) {
        .add => if (is_float) encode.fadd(rd, ra, rb, .{}) else encode.iadd3(rd, ra, rb, .{}),
        .sub => if (is_float) encode.fsub(rd, ra, rb, .{}) else encode.isub(rd, ra, rb, .{}),
        .mul => if (is_float) encode.fmul(rd, ra, rb, .{}) else encode.imad(rd, ra, rb, encode.RZ, .{}),
        .bit_and => encode.lop3(rd, ra, rb, encode.LUT_AND, .{}),
        .bit_or => encode.lop3(rd, ra, rb, encode.LUT_OR, .{}),
        .bit_xor => encode.lop3(rd, ra, rb, encode.LUT_XOR, .{}),
        .shl => encode.shf(rd, ra, rb, false, false, .{}),
        .shr => encode.shf(rd, ra, rb, true, isSignedRaw(func, lhs), .{}),
        // Integer divide is a multi-instruction reciprocal sequence, deferred.
        .div, .rem => error.Unsupported,
    };
}

fn cmpOf(op: ir.function.CmpOp) encode.Cmp {
    return switch (op) {
        .eq => .eq,
        .ne => .ne,
        .lt => .lt,
        .le => .le,
        .gt => .gt,
        .ge => .ge,
    };
}

fn isFloat(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => true,
        else => false,
    };
}

fn isSigned(func: *const Function, v: Value) bool {
    return isSignedRaw(func, v);
}

fn isSignedRaw(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |x| x.signedness == .signed,
        else => true,
    };
}

fn emitIf(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), fixups: *std.ArrayList(Fixup), cf: ir.function.If) Error!void {
    const pred = predOf(loc.*, cf.cond);
    // If the condition is true, branch to the `then` target. Else fall through to a
    // branch to the `else` target. Edge moves precede each branch.
    try emitMoves(allocator, func, loc, code, cf.then);
    const then_bra = code.items.len;
    try code.append(allocator, encode.bra(0, .{ .pred = pred })); // guarded: taken if cond
    try fixups.append(allocator, .{ .at = then_bra, .target = @intFromEnum(cf.then.target) });

    try emitMoves(allocator, func, loc, code, cf.@"else");
    const else_bra = code.items.len;
    try code.append(allocator, encode.bra(0, .{})); // unconditional
    try fixups.append(allocator, .{ .at = else_bra, .target = @intFromEnum(cf.@"else".target) });
}

fn emitJump(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), fixups: *std.ArrayList(Fixup), jump: ir.function.Jump) Error!void {
    try emitMoves(allocator, func, loc, code, jump);
    const at = code.items.len;
    try code.append(allocator, encode.bra(0, .{}));
    try fixups.append(allocator, .{ .at = at, .target = @intFromEnum(jump.target) });
}

/// Edge moves into the target block's parameters (register copies). Distinct
/// registers per value mean the moves are independent except for genuine swaps. A
/// scratch register breaks any cycle.
fn emitMoves(allocator: std.mem.Allocator, func: *const Function, loc: *std.AutoHashMapUnmanaged(Value, Loc), code: *std.ArrayList(Inst), jump: ir.function.Jump) Error!void {
    const args = func.blockArgs(jump);
    const params = func.blockParams(jump.target);
    if (args.len != params.len) return error.Unsupported;
    for (args, params) |arg, param| {
        const dst = gprOf(loc.*, param);
        const src = gprOf(loc.*, arg);
        if (dst != src) try code.append(allocator, encode.movReg(dst, src, .{}));
    }
}

// Liveness (for the allocator).

fn markUse(last_use: []u32, v: Value, pos: u32) void {
    if (pos > last_use[@intFromEnum(v)]) last_use[@intFromEnum(v)] = pos;
}

fn forEachUse(func: *const Function, inst: ir.function.Inst, last_use: []u32, pos: u32) void {
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
        .load => |ld| markUse(last_use, ld.ptr, pos),
        .store => |st| {
            markUse(last_use, st.value, pos);
            markUse(last_use, st.ptr, pos);
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

fn forEachTermUse(func: *const Function, term: Terminator, last_use: []u32, pos: u32) void {
    switch (term) {
        .ret => |v| if (v) |vv| markUse(last_use, vv, pos),
        .jump => |j| for (func.blockArgs(j)) |a| markUse(last_use, a, pos),
    }
}

fn setUsed(row: []bool, v: Value) void {
    row[@intFromEnum(v)] = true;
}

fn markUsedBitset(func: *const Function, inst: ir.function.Inst, row: []bool) void {
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
        .load => |ld| setUsed(row, ld.ptr),
        .store => |st| {
            setUsed(row, st.value);
            setUsed(row, st.ptr);
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

fn markUsedTermBitset(func: *const Function, term: Terminator, row: []bool) void {
    switch (term) {
        .ret => |v| if (v) |vv| setUsed(row, vv),
        .jump => |j| for (func.blockArgs(j)) |a| setUsed(row, a),
    }
}

/// Backward liveness dataflow. Extends `last_use[v]` to the end of every block
/// where `v` is live-out, so a value live across a loop keeps its register.
fn extendLiveRanges(allocator: std.mem.Allocator, func: *const Function, last_use: []u32, block_end: []const u32) Error!void {
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
            markUsedBitset(func, inst, row);
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

const testing = std.testing;

test "compiles a vertex shader: attribute load, compute, attribute store, exit" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    // A vertex input attribute (tagged with its slot), incremented and written to
    // the clip-space position output.
    const in = try func.appendBlockParam(b, f32_t);
    try func.addAttr(.{ .value = in }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "attr", .value = .{ .int = encode.ATTR_GENERIC0 } } });
    const one = try func.appendInst(b, f32_t, .{ .fconst = 1.0 });
    const sum = try func.appendInst(b, f32_t, .{ .arith = .{ .op = .add, .lhs = in, .rhs = one } });
    const out_ptr = try func.appendInst(b, i32_t, .{ .iconst = 0 }); // the position output slot
    try func.addAttr(.{ .value = out_ptr }, .{ .custom = .{ .namespace = "vulcan.gpu", .key = "out_attr", .value = .{ .int = encode.ATTR_POSITION } } });
    try func.appendStore(b, sum, out_ptr);
    func.setTerminator(b, .{ .ret = null });

    var kernel = try compileShader(allocator, &func, .vertex);
    defer kernel.deinit(allocator);

    // ALD (attribute fetch) -> FADD -> AST (write position) -> EXIT.
    var has_ald = false;
    var has_fadd = false;
    var has_ast = false;
    var has_exit = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0x321 => has_ald = true,
            0x221 => has_fadd = true,
            0x322 => has_ast = true,
            0x94d => has_exit = true,
            else => {},
        }
    }
    try testing.expect(has_ald and has_fadd and has_ast and has_exit);
}

test "compiles a kernel: load params, multiply-add, store, exit" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = y } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    var kernel = try compileKernel(allocator, &func);
    defer kernel.deinit(allocator);

    // Prologue: LDC outptr lo/hi + two inputs = 4 instructions, then IMAD, IADD3,
    // STG, EXIT = 8 instructions total (32 dwords).
    try testing.expectEqual(@as(usize, 8 * 4), kernel.code.len);
    try testing.expectEqual(@as(u32, 0xb82), kernel.code[0] & 0xfff); // first LDC
    // The first LDC reads the output pointer low word from the param base.
    try testing.expectEqual(@as(u32, param_base), @as(u16, @truncate(kernel.code[1] >> 6)) & 0xffff);

    // The instruction words: LDC x4, IMAD, IADD3, STG, EXIT.
    const op = struct {
        fn at(code: []const u32, i: usize) u32 {
            return code[i * 4] & 0xfff;
        }
    }.at;
    try testing.expectEqual(@as(u32, 0xb82), op(kernel.code, 3)); // last param LDC
    try testing.expectEqual(@as(u32, 0x224), op(kernel.code, 4)); // IMAD (base 0x024 | reg form)
    try testing.expectEqual(@as(u32, 0x210), op(kernel.code, 5)); // IADD3
    try testing.expectEqual(@as(u32, 0x986), op(kernel.code, 6)); // STG
    try testing.expectEqual(@as(u32, 0x94d), op(kernel.code, 7)); // EXIT
}

test "compiles control flow: a max via if and a merge block" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, t);
    const b = try func.appendBlockParam(entry, t);
    const exit_b = try func.appendBlock();
    const r = try func.appendBlockParam(exit_b, t);
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = a, .rhs = b } });
    try func.appendIf(entry, c, .{ .target = exit_b, .args = &.{a} }, .{ .target = exit_b, .args = &.{b} });
    func.setTerminator(exit_b, .{ .ret = r });

    var kernel = try compileKernel(allocator, &func);
    defer kernel.deinit(allocator);

    // The stream contains an ISETP (compare), at least two BRA, an STG, and EXIT.
    var saw_isetp = false;
    var bra_count: usize = 0;
    var saw_exit = false;
    var i: usize = 0;
    while (i < kernel.code.len) : (i += 4) {
        switch (kernel.code[i] & 0xfff) {
            0x20c => saw_isetp = true, // ISETP (base 0x00c | reg form)
            0x947 => bra_count += 1,
            0x94d => saw_exit = true,
            else => {},
        }
    }
    try testing.expect(saw_isetp);
    try testing.expect(bra_count >= 2);
    try testing.expect(saw_exit);
}
