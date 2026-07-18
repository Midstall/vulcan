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

/// One piece of a split GPR value's life: the value lives in `loc` from position `from` until the
/// next segment (or, for the last one, to the end of its range). `segments[0].from` is the value's
/// def position, so a lookup at any position at or after the def resolves to some segment. Task 7c
/// fills the segment map. While it is empty, `loc` falls back to the whole-life `loc_of` lookup and
/// emission is byte-identical to before splitting.
const Segment = struct { from: u32, loc: Loc };

/// A store the emitter must insert at a split boundary. `at` is the instruction position the
/// store lands before (the position at which the GPR pool was exhausted for a tail split). The
/// store writes the victim's register to its new slot BEFORE the taker (the value defined at
/// `at`) overwrites that register, so the victim's tail uses reload the correct bits. Task 7c
/// only produces `.store`. The `.reload` kind is reserved for the second-chance reload task (7d).
const SplitAction = struct {
    at: u32,
    kind: enum { store, reload },
    value: Value,
    reg: Reg,
    slot: u32,
};

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
/// Whether `v` is an f16 (half). f16 is emulated: it lives in an xmm register as its f32
/// widening (so `isDouble(f16)` is false and every in-register op uses the scalar-single SSE
/// form naturally), and the boundaries widen/narrow with the F16C conversions. `isHalf` marks
/// the sites that must add that widening/narrowing: memory load/store, narrowing converts, the
/// int->f16 and f32/f64->f16 converts, arithmetic results, and the f16 constant.
fn isHalf(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f16,
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
/// Whether the function makes any call (direct or indirect), so its frame needs 16-byte
/// call-site alignment.
fn hasCall(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| switch (func.opcode(inst)) {
            .call, .call_indirect => return true,
            else => {},
        };
    }
    return false;
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
    // Split GPR values only: value -> ascending-by-`from` segment list. Empty means no value was
    // split, so `loc` falls back to `loc_of` and emission is byte-identical (Task 7c fills it).
    segments: std.AutoHashMapUnmanaged(Value, []Segment) = .{},
    // Stores to emit at split boundaries, in ascending-`at` order after the sort in `compile`.
    // Empty means no value was split, so emission is byte-identical to before splitting.
    actions: std.ArrayList(SplitAction) = .empty,
    def_pos: []u32 = &.{}, // per value: its def position (duped from local liveness, and the emission assert reads it)
    pos: u32 = 0, // current emission position, threaded per instruction so `loc` can pick the active segment

    fn loc(self: *const Ctx, v: Value) Loc {
        if (self.segments.get(v)) |segs| {
            var chosen = segs[0]; // non-empty, ascending by `from`
            for (segs) |s| {
                if (s.from <= self.pos) chosen = s else break;
            }
            return chosen.loc;
        }
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
    allocator.free(compiled.lines);
    return compiled.code;
}

/// Test hook: run GPR allocation for `func` and report how many values were tail-split (their
/// life spans a register prefix plus a spill tail). Zero means no split occurred. The execution
/// tests call this to assert a case actually exercises the splitter before checking its results.
pub fn splitCountForTest(allocator: std.mem.Allocator, func: *const Function) Error!usize {
    var loc_of: std.AutoHashMapUnmanaged(Value, Loc) = .{};
    defer loc_of.deinit(allocator);
    var segments: std.AutoHashMapUnmanaged(Value, []Segment) = .{};
    defer {
        var it = segments.valueIterator();
        while (it.next()) |s| allocator.free(s.*);
        segments.deinit(allocator);
    }
    var actions: std.ArrayList(SplitAction) = .empty;
    defer actions.deinit(allocator);
    var num_slots: u32 = 0;
    var def_pos: []u32 = &.{};
    defer allocator.free(def_pos);
    try assignRegs(allocator, func, &loc_of, &num_slots, &def_pos, &segments, &actions);
    return segments.count();
}

/// Test hook: run GPR allocation for `func` and report how many values were SECOND-CHANCE
/// RE-HOMED, a value whose segment list holds a `.reg` segment after a `.spill` segment (spilled,
/// then brought back into a register for its remaining tail uses). Exists so an execution test can
/// prove second-chance reload actually fired, not merely that a tail split happened (which a plain
/// per-use reload would also satisfy).
pub fn reHomeCountForTest(allocator: std.mem.Allocator, func: *const Function) Error!usize {
    var loc_of: std.AutoHashMapUnmanaged(Value, Loc) = .{};
    defer loc_of.deinit(allocator);
    var segments: std.AutoHashMapUnmanaged(Value, []Segment) = .{};
    defer {
        var it = segments.valueIterator();
        while (it.next()) |s| allocator.free(s.*);
        segments.deinit(allocator);
    }
    var actions: std.ArrayList(SplitAction) = .empty;
    defer actions.deinit(allocator);
    var num_slots: u32 = 0;
    var def_pos: []u32 = &.{};
    defer allocator.free(def_pos);
    try assignRegs(allocator, func, &loc_of, &num_slots, &def_pos, &segments, &actions);
    var count: usize = 0;
    var it = segments.valueIterator();
    while (it.next()) |segs| {
        var saw_slot = false;
        for (segs.*) |s| switch (s.loc) {
            .spill => saw_slot = true,
            .reg => if (saw_slot) {
                count += 1;
            },
            .xmm, .xmm_spill => {},
        };
    }
    return count;
}

/// Compile `func` to machine code plus its call relocations. Caller owns it.
pub fn compile(allocator: std.mem.Allocator, func: *const Function) Error!Compiled {
    // f16 is now lowered here via F16C (held as its f32 widening in an xmm register, all
    // arithmetic in scalar-single SSE, hardware vcvtph2ps/vcvtps2ph conversion at the
    // boundaries). The other backends still reject f16 via `functionUsesF16`; x86_64 no longer
    // does. Only SCALAR f16 is handled, so f16 nested in a vector/aggregate (which would fall
    // through to the raw-vector path and miscompile the half lanes) is still rejected cleanly.
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;
    const nblocks = func.blockCount();
    if (nblocks == 0) return error.Unsupported;

    var ctx = Ctx{ .func = func };
    defer ctx.loc_of.deinit(allocator);
    defer ctx.code.deinit(allocator);
    defer ctx.fixups.deinit(allocator);
    defer ctx.relocs.deinit(allocator);
    defer ctx.lines.deinit(allocator);
    defer ctx.alloca_off.deinit(allocator);
    defer {
        var seg_it = ctx.segments.valueIterator();
        while (seg_it.next()) |s| allocator.free(s.*);
        ctx.segments.deinit(allocator);
    }
    defer ctx.actions.deinit(allocator);
    // `def_pos` is always a heap-owned dupe (the `&.{}` sentinel is a zero-length slice with no
    // backing allocation, so freeing it is a no-op), so an unconditional free is safe.
    defer allocator.free(ctx.def_pos);
    var num_slots: u32 = 0;
    try assignRegs(allocator, func, &ctx.loc_of, &num_slots, &ctx.def_pos, &ctx.segments, &ctx.actions);
    var xmm_slots: u32 = 0;
    try assignXmm(allocator, func, &ctx.loc_of, &xmm_slots);
    // Split-boundary actions are appended in monotonic `at` order already. Sort defensively so the
    // per-instruction drain below can advance a single cursor. At the SAME position a `.reload` must
    // precede a `.store`: a value can be reloaded slot->reg and then immediately re-spilled reg->slot
    // at one use position, and the reload has to run first or the store would save a stale register.
    // `std.mem.sort` is not stable, so the comparator breaks `at` ties on kind (reload before store).
    std.mem.sort(SplitAction, ctx.actions.items, {}, struct {
        fn order(k: @TypeOf(@as(SplitAction, undefined).kind)) u8 {
            return switch (k) {
                .reload => 0,
                .store => 1,
            };
        }
        fn f(_: void, a: SplitAction, b: SplitAction) bool {
            if (a.at != b.at) return a.at < b.at;
            return order(a.kind) < order(b.kind);
        }
    }.f);
    // Frame layout: general spills (8 bytes each), then the xmm spill area (32-byte slots
    // at a 16-aligned base, sized for a whole 256-bit ymm. A scalar/128-bit value uses the
    // low half), then the alloca region (each alloca offset relative to its 16-aligned base).
    const xmm_base: u64 = (@as(u64, num_slots) * 8 + 15) & ~@as(u64, 15);
    ctx.xmm_base = @intCast(xmm_base);
    const alloca_base: u64 = (xmm_base + @as(u64, xmm_slots) * 32 + 15) & ~@as(u64, 15);
    ctx.alloca_base = @intCast(alloca_base);
    const alloca_bytes = try computeAllocaSlots(allocator, func, &ctx.alloca_off);
    // A function that makes calls must keep RSP 16-aligned at the call site: System V requires
    // RSP+8 aligned at a callee's entry, so RSP here is 8 (mod 16), and a 16-multiple frame would
    // leave calls 8 off, faulting a callee that reads its stack with movaps on real hardware
    // (qemu-user does not enforce this). Pad the frame by 8 so RSP at a call site lands on a 16
    // boundary. A leaf function needs no such padding.
    const frame_base = (alloca_base + alloca_bytes + 15) & ~@as(u64, 15);
    const frame: i32 = @intCast(if (hasCall(func)) frame_base + 8 else frame_base);

    const block_start = try allocator.alloc(usize, nblocks);
    defer allocator.free(block_start);

    // Prologue: reserve the spill frame, then move each argument from its ABI register to
    // the entry parameter's location (a register parallel move, or a store for a spilled
    // parameter).
    if (frame > 0) try ctx.put(allocator, encode.aluImm(5, .rsp, frame, true)); // sub rsp, frame (64-bit stack ptr)
    // System V passes general args in rdi,rsi,... and fp args in xmm0,xmm1,... (separate
    // sequences), so each class has its own incoming-register index.
    const eparams = func.blockParams(@enumFromInt(0));
    var arg_moves: std.ArrayList(Move) = .empty;
    defer arg_moves.deinit(allocator);
    var gpr_i: usize = 0;
    var xmm_i: usize = 0;
    // Args beyond the ABI registers arrive on the stack, in declaration order. At entry [rsp+frame]
    // holds the return address (the prologue already did `sub rsp, frame`), so the first stack arg is
    // at [rsp+frame+8], the next at [rsp+frame+16], and so on.
    var stack_arg: u32 = 0;
    for (eparams) |p| {
        if (isXmm(func, p)) {
            // A vector param also lives in an xmm register, so classify by isXmm (float or
            // vector), matching the call-site arg handling. A scalar float moves/stores as
            // 128-bit (movups, the extra lanes are harmless); a 128-bit vector as movups; a
            // 256-bit vector as vmovups so no lanes are dropped.
            if (xmm_i >= xmm_arg_regs.len) {
                // Fp stack arg. System V lays them out in 8-byte slots above the return address, so
                // slot k is at [rsp + frame + 8 + k*8]. Only a scalar float is handled; a vector on the
                // stack would span several slots (rare, unsupported). Load it (scalar movss) into the
                // param's home, via the xmm scratch when the home is a spill slot.
                // An f64 stack arg would need a 64-bit `movsd` load (movss reads only its low 4 bytes
                // and would truncate the double); reject it fail-closed until a movsd path exists.
                if (isVector(func, p) or isWide(func, p) or isDouble(func, p)) return error.Unsupported;
                const off: i32 = frame + 8 + @as(i32, @intCast(stack_arg)) * 8;
                stack_arg += 1;
                switch (ctx.loc(p)) {
                    .xmm => |x| try ctx.put(allocator, encode.movssLoad(x, off)),
                    .xmm_spill => |slot| {
                        try ctx.put(allocator, encode.movssLoad(xmm_scratch, off));
                        try ctx.put(allocator, encode.movssStore(ctx.xmmDisp(slot), xmm_scratch));
                    },
                    else => unreachable,
                }
                continue;
            }
            const incoming = xmm_arg_regs[xmm_i];
            xmm_i += 1;
            switch (ctx.loc(p)) {
                .xmm => |x| if (x != incoming) try ctx.put(allocator, if (isWide(func, p)) encode.vmovupsRR(x, incoming) else encode.movupsRR(x, incoming)), // no fp move cycles for a single arg
                .xmm_spill => |slot| try ctx.put(allocator, if (isWide(func, p)) encode.vmovupsStore(ctx.xmmDisp(slot), incoming) else if (isVector(func, p)) encode.movupsStore(ctx.xmmDisp(slot), incoming) else encode.movssStore(ctx.xmmDisp(slot), incoming)),
                else => unreachable,
            }
        } else {
            if (gpr_i >= arg_regs.len) return error.Unsupported; // gpr stack args not handled yet
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

    // `pos_base` is the current block's param-row position. It mirrors computeLocalLiveness's
    // numbering EXACTLY (param row, then one slot per instruction, then one terminator slot), so
    // `ctx.pos` set per instruction below equals the position each value's def was numbered at. With
    // `segments` empty the pos is otherwise unobservable, so the pos-coupling assert is how a
    // threading bug is caught in this task.
    var pos_base: u32 = 0;
    var action_cursor: usize = 0;
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        block_start[bi] = ctx.code.items.len;
        var terminated = false;
        const insts = func.blockInsts(block);
        for (insts, 0..) |inst, inst_idx| {
            // Continue-safe: derive the position from the block base plus the instruction index so any
            // early exit cannot desync it. This equals computeLocalLiveness's numbering for this inst
            // (param row at `pos_base`, then one position per instruction).
            ctx.pos = pos_base + 1 + @as(u32, @intCast(inst_idx));
            // Record a source-line row when this instruction starts a new line (byte offset = the
            // current code length, since x86 code is already a byte stream).
            if (lineOf(func, inst)) |line| {
                if (line != ctx.last_line) {
                    try ctx.lines.append(allocator, .{ .offset = @intCast(ctx.code.items.len), .line = line });
                    ctx.last_line = line;
                }
            }
            // An instruction with a result must be emitted at exactly that result's def position. This
            // pins the emission numbering to computeLocalLiveness's numbering and must never trip.
            if (func.instResult(inst)) |r| std.debug.assert(ctx.pos == ctx.def_pos[@intFromEnum(r)]);
            // Drain split-boundary stores landing at this position BEFORE emitting the instruction.
            // A tail-split store writes the victim's register to its slot before the taker (the
            // instruction defined at `p`) computes its result into that same register. The victim's
            // value still occupies the register here (its last prefix use is before `p`, and nothing
            // reused the register before `p`), so the store captures the correct bits.
            while (action_cursor < ctx.actions.items.len and ctx.actions.items[action_cursor].at <= ctx.pos) {
                const act = ctx.actions.items[action_cursor];
                std.debug.assert(act.at == ctx.pos); // stores land on instruction positions only
                switch (act.kind) {
                    .store => try ctx.put(allocator, encode.movToStack(slotDisp(act.slot), act.reg)),
                    .reload => {
                        // Second-chance reload: bring `act.value` back from its slot into `act.reg`
                        // just before its next use, mirroring `use`'s slot path. From here on its
                        // `.reg` re-home segment makes `loc` read the register directly.
                        try ctx.put(allocator, encode.movFromStack(act.reg, slotDisp(act.slot)));
                    },
                }
                action_cursor += 1;
            }
            if (func.opcode(inst) == .@"if") {
                try emitIf(allocator, &ctx, func.opcode(inst).@"if");
                terminated = true;
            } else {
                try lowerInst(allocator, &ctx, inst);
            }
        }
        // The terminator shares the block-end position. An `.@"if"` terminator is one of the
        // instructions above (it set `terminated`), so this only positions a ret/jump, but the
        // numbering still reserves a terminator slot whether or not one is emitted here.
        ctx.pos = pos_base + 1 + @as(u32, @intCast(insts.len));
        // Drain any split-boundary actions recorded AT the terminator position BEFORE emitting the
        // terminator. `secondChance` can re-home a value used only by `ret` (a non-edge-arg operand,
        // hence `is_intra`) at its next use, which is the terminator position (`block_end`), recording
        // a `.reload` there. The per-instruction drain above only reaches `term_pos - 1`, so without
        // this drain the reload is never emitted and `ret` reads a register that was never loaded. When
        // no action lands on a terminator (the normal case) this drains nothing and is byte-identical.
        while (action_cursor < ctx.actions.items.len and ctx.actions.items[action_cursor].at <= ctx.pos) {
            const act = ctx.actions.items[action_cursor];
            std.debug.assert(act.at == ctx.pos); // only terminator-position actions remain here
            switch (act.kind) {
                .store => try ctx.put(allocator, encode.movToStack(slotDisp(act.slot), act.reg)),
                .reload => try ctx.put(allocator, encode.movFromStack(act.reg, slotDisp(act.slot))),
            }
            action_cursor += 1;
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
                if (frame > 0) try ctx.put(allocator, encode.aluImm(0, .rsp, frame, true)); // add rsp, frame (64-bit stack ptr)
                try ctx.put(allocator, encode.ret());
            },
            .jump => |j| try emitJump(allocator, &ctx, j),
        };
        // Advance to the next block's param row: param row (1) + one slot per instruction + one
        // terminator slot (reserved unconditionally, matching computeLocalLiveness's per-block
        // final increment).
        pos_base = pos_base + 2 + @as(u32, @intCast(insts.len));
    }
    // Every store must have drained. A tail-split store lands at the taker's def position, which is
    // always an instruction position visited by the loop above, so no store can outlive it.
    std.debug.assert(action_cursor == ctx.actions.items.len);

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

/// Round the f16 value held in the low f32 lane of `reg` to nearest-even half and re-widen it,
/// in place. This is the per-op IEEE rounding of the f16 emulation: an f16 arithmetic result,
/// an int/f32/f64 -> f16 convert, or an f16 constant is first computed in f32, then this
/// narrows it to a half (vcvtps2ph, RNE) and widens it back (vcvtph2ps) so the register again
/// holds an exact half value. Skipping it would keep f32 precision, which is WRONG for f16
/// semantics (each op must round to nearest-even half). Both F16C ops are 128-bit and operate
/// in place, so no scratch register is needed.
fn roundToHalf(allocator: std.mem.Allocator, ctx: *Ctx, reg: Xmm) Error!void {
    try ctx.put(allocator, encode.vcvtps2ph(reg, reg, 0)); // imm8 = 0 -> round-to-nearest-even
    try ctx.put(allocator, encode.vcvtph2ps(reg, reg));
}

fn lowerInst(allocator: std.mem.Allocator, ctx: *Ctx, inst: ir.function.Inst) Error!void {
    const func = ctx.func;
    if (func.opcode(inst) == .store) {
        // `store` produces no result, so handle it before the result unwrap below.
        const st = func.opcode(inst).store;
        const base = try ctx.use(allocator, st.ptr, scratch2);
        if (isXmm(func, st.value)) {
            const val = try ctx.useXmm(allocator, st.value, xmm_op0);
            if (isHalf(func, st.value)) {
                // Store a 16-bit IEEE half: narrow the held f32 (lane 0) to a half with
                // vcvtps2ph (RNE), move it to a gpr, and write exactly 2 bytes. The held value
                // is already an exact half, so the narrow is lossless. Narrow into xmm_scratch
                // so `val` (which useXmm never reloads into the scratch) is not clobbered.
                try ctx.put(allocator, encode.vcvtps2ph(xmm_scratch, val, 0));
                try ctx.put(allocator, encode.movdFromXmm(scratch1, xmm_scratch));
                try ctx.put(allocator, encode.movToMem16(base, 0, scratch1));
            } else {
                try ctx.put(allocator, if (isVector(func, st.value)) encode.movupsStoreMem(base, 0, val) else if (isDouble(func, st.value)) encode.movsdStoreMem(base, 0, val) else encode.movssStoreMem(base, 0, val));
            }
        } else {
            const val = try ctx.use(allocator, st.value, scratch1);
            // Store the value's own width so exactly the object's bytes are written, never more:
            // an 8-bit store writes 1 byte, a 16-bit 2, a 32-bit 4, a 64-bit/pointer 8. A wider
            // store would clobber the next element of a tightly-packed array (e.g. an i8 store8).
            try ctx.put(allocator, switch (intBits(func, st.value)) {
                0...8 => encode.movToMem8(base, 0, val),
                9...16 => encode.movToMem16(base, 0, val),
                17...32 => encode.movToMem32(base, 0, val),
                else => encode.movToMem(base, 0, val),
            });
        }
        return;
    }
    if (func.opcode(inst) == .prefetch) {
        // A software prefetch hint: no result, and x86-64 codegen here has no need for
        // one, so it is simply dropped (a valid no-op lowering of a hint).
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
                    try ctx.put(allocator, encode.movImm(scratch1, @bitCast(bits), false)); // 32-bit float bits into a gpr
                    try ctx.put(allocator, encode.movdToXmm(rd, scratch1));
                }
                try ctx.storeXmm(allocator, result, rd);
            } else {
                const rd = ctx.dst(result, scratch1);
                // A 64-bit constant needs the full imm64 mov; `movImm` only carries a
                // sign-extended imm32, and `@intCast(c)` would panic on any value outside i32
                // (e.g. the 64-bit SWAR popcount masks, or a 0x80000000 sign mask). For a
                // <=32-bit result, take the low 32 bits as a bit pattern (movImm sign-extends
                // them into the 64-bit register, which a 32-bit use reads back correctly).
                if (intBits(func, result) > 32) {
                    try ctx.put(allocator, encode.movImm64(rd, @bitCast(c)));
                } else {
                    // A <=32-bit constant: the zero-extending `mov r32, imm32` puts the exact
                    // low-32 bit pattern in the register and clears the upper 32, keeping the
                    // clean-upper-bits invariant that the width-aware ops rely on.
                    const bits: u32 = @truncate(@as(u64, @bitCast(c)));
                    try ctx.put(allocator, encode.movImm(rd, @bitCast(bits), false));
                }
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
                // An f16 constant is materialized as its f32 widening (the value rounded to
                // half first, `@as(f32, @as(f16, val))`), keeping the invariant that an f16 in
                // a register is its exact-half f32 form. f32 keeps its full value.
                const bits: u32 = @bitCast(if (isHalf(func, result)) @as(f32, @as(f16, @floatCast(val))) else @as(f32, @floatCast(val)));
                try ctx.put(allocator, encode.movImm(scratch1, @bitCast(bits), false)); // 32-bit float bits into a gpr
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
                // An f16 op is done in the scalar-single (f32) form, then its result is rounded
                // to nearest-even half so the register again holds an exact half value (correct
                // per-op IEEE f16 semantics; the operands were already exact halves).
                if (isHalf(func, result)) try roundToHalf(allocator, ctx, work);
                try ctx.storeXmm(allocator, result, work);
                return;
            }
            const signed = isSigned(func, a.lhs);
            // 64-bit operand size only for i64 (and any wider); i32/i16/i8 use 32-bit ops, whose
            // result auto-zeroes the upper 32 bits so the value stays clean.
            const w64 = intBits(func, result) > 32;
            switch (a.op) {
                .div, .rem => {
                    // The dividend width follows the operands: a 32-bit divide sign-extends with
                    // cdq (not cqo) and uses the 32-bit idiv/div, reading only E(D)X:EAX so a
                    // dirty upper half of RAX is ignored (e.g. u32 divu(-1, 2) = 0x7FFFFFFF).
                    const dw = intBits(func, a.lhs) > 32;
                    try ctx.put(allocator, encode.movReg(.rax, try ctx.use(allocator, a.lhs, scratch1)));
                    try ctx.put(allocator, if (signed) (if (dw) encode.cqo() else encode.cdq()) else encode.xorr(.rdx, .rdx, dw));
                    try ctx.put(allocator, if (signed) encode.idiv(try ctx.use(allocator, a.rhs, scratch2), dw) else encode.divu(try ctx.use(allocator, a.rhs, scratch2), dw));
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
                    // A 32-bit shr/sar shifts within 32 bits (filling from bit 31), which an i32
                    // needs; a 64-bit shift would cross bit 31/32 and corrupt the result.
                    try ctx.put(allocator, if (a.op == .shl) encode.shlCl(rd, w64) else if (signed) encode.sarCl(rd, w64) else encode.shrCl(rd, w64));
                    try ctx.store(allocator, result, rd);
                },
                else => {
                    const rl = try ctx.use(allocator, a.lhs, scratch1);
                    const rr = try ctx.use(allocator, a.rhs, scratch2);
                    const rd = ctx.dst(result, scratch1);
                    if (rd != rl) try ctx.put(allocator, encode.movReg(rd, rl));
                    try ctx.put(allocator, try binary(a.op, rd, rr, w64));
                    try ctx.store(allocator, result, rd);
                },
            }
        },
        .arith_imm => |a| {
            const imm: i32 = @intCast(a.imm);
            // 64-bit operand size only for i64 (and wider); a <=32-bit result uses 32-bit ops.
            const w64 = intBits(func, result) > 32;
            switch (a.op) {
                .mul => {
                    const rd = ctx.dst(result, scratch1);
                    try ctx.put(allocator, encode.imulImm(rd, try ctx.use(allocator, a.lhs, scratch1), imm, w64));
                    try ctx.store(allocator, result, rd);
                },
                .mulh => return error.Unsupported, // no immediate form; expanded before isel
                .add, .sub, .bit_and, .bit_or, .bit_xor => {
                    const rl = try ctx.use(allocator, a.lhs, scratch1);
                    const rd = ctx.dst(result, scratch1);
                    if (rd != rl) try ctx.put(allocator, encode.movReg(rd, rl));
                    try ctx.put(allocator, encode.aluImm(aluDigit(a.op), rd, imm, w64));
                    try ctx.store(allocator, result, rd);
                },
                .shl, .shr => {
                    const rl = try ctx.use(allocator, a.lhs, scratch1);
                    const rd = ctx.dst(result, scratch1);
                    if (rd != rl) try ctx.put(allocator, encode.movReg(rd, rl));
                    // A 32-bit shr/sar fills from bit 31, which is what the signExt lowering of
                    // i32.extend8_s/16_s ((x << 24) >> 24) and the clz/ctz smears rely on; a
                    // 64-bit shift would sign-extend from bit 39/47 instead.
                    try ctx.put(allocator, encode.shiftImm(shiftDigit(a.op, isSigned(func, a.lhs)), rd, @truncate(@as(u32, @bitCast(imm))), w64));
                    try ctx.store(allocator, result, rd);
                },
                .div, .rem => {
                    const signed = isSigned(func, a.lhs);
                    const dw = intBits(func, a.lhs) > 32;
                    try ctx.put(allocator, encode.movReg(.rax, try ctx.use(allocator, a.lhs, scratch1)));
                    try ctx.put(allocator, if (signed) (if (dw) encode.cqo() else encode.cdq()) else encode.xorr(.rdx, .rdx, dw));
                    try ctx.put(allocator, encode.movImm(scratch2, imm, dw)); // divisor at the operand width
                    try ctx.put(allocator, if (signed) encode.idiv(scratch2, dw) else encode.divu(scratch2, dw));
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
                    // Combine the two zero-extended 0/1 bytes. The bool result is <=32 bits, so a
                    // 32-bit and/or is used (correct for the clean 0/1 operands either way).
                    try ctx.put(allocator, if (cmp.op == .eq) encode.andr(rd, scratch2, intBits(func, result) > 32) else encode.orr(rd, scratch2, intBits(func, result) > 32));
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
            // Compare at the operand width: an i32 compare sets flags from the low 32 bits, so a
            // dirty upper half (e.g. a sign-extended incoming i32 arg) does not skew the result.
            try ctx.put(allocator, encode.cmp(rl, rr, intBits(func, cmp.lhs) > 32));
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
            // Test at the condition's width so a raw i32 cond with a dirty upper half is not
            // read as nonzero on its garbage bits.
            try ctx.put(allocator, encode.testReg(c, c, intBits(func, s.cond) > 32));
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
                // int -> float: cvtsi2ss/cvtsi2sd. isDouble(f16) is false, so an int->f16 lands
                // in the scalar-single form first, then rounds to nearest-even half.
                try ctx.put(allocator, if (isDouble(func, result)) encode.cvtsi2sd(rd, src) else encode.cvtsi2ss(rd, src));
                if (isHalf(func, result)) try roundToHalf(allocator, ctx, rd);
                try ctx.storeXmm(allocator, result, rd);
            } else if (src_float and !dst_float) {
                // float -> int (truncate toward zero). An f16 source is held as its f32
                // widening, and isDouble(f16) is false, so cvttss2si reads the right value.
                const src = try ctx.useXmm(allocator, cv.value, xmm_op0);
                const rd = ctx.dst(result, scratch1); // i32 result in a gpr
                try ctx.put(allocator, if (isDouble(func, cv.value)) encode.cvttsd2si(rd, src) else encode.cvttss2si(rd, src));
                try ctx.store(allocator, result, rd);
            } else if (src_float and dst_float) {
                // float -> float. An f16 is always held as its f32 widening (isDouble(f16) is
                // false), so f16->f32 falls out as a same-width copy and f16->f64 as the plain
                // cvtss2sd widen. Only a narrowing TO f16 needs special handling: it must round
                // to nearest-even half rather than leave f32 precision in place.
                const src = try ctx.useXmm(allocator, cv.value, xmm_op0);
                const rd = try ctx.dstXmm(result, xmm_scratch);
                const sd = isDouble(func, cv.value);
                const dd = isDouble(func, result);
                if (isHalf(func, result)) {
                    // -> f16: bring the source to f32 (an f64 source narrows with cvtsd2ss),
                    // then round to nearest-even half and widen back to the held f32 form.
                    if (sd) try ctx.put(allocator, encode.cvtsd2ss(rd, src)) else if (rd != src) try ctx.put(allocator, encode.movupsRR(rd, src));
                    try roundToHalf(allocator, ctx, rd);
                } else if (sd == dd) {
                    // Same width, dest not half: a plain copy (f32->f32, f64->f64, f16->f32).
                    if (rd != src) try ctx.put(allocator, encode.movupsRR(rd, src));
                } else {
                    // Different widths, dest not half: the base single<->double convert,
                    // byte-identical to the pre-f16 behavior, and also the exact f16->f64 widen.
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
            // f16 is held as its f32 widening; a sqrtss/roundss would leave an un-narrowed f32 (no
            // round-to-half), so an f16 unary math op is not lowered - reject cleanly rather than
            // silently produce a value that is not a valid half (mirrors the wasm/aarch64 backends).
            if (isHalf(func, result_val.?)) return error.Unsupported;
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
                if (isHalf(func, result)) {
                    // Load a 16-bit IEEE half and widen to the held f32 form: movzx word into a
                    // gpr, movd into the xmm low lane, vcvtph2ps. NOT movss, which would read 4
                    // bytes from a 2-byte object (pulling in the next element).
                    try ctx.put(allocator, encode.movzxWordFromMem(scratch1, base, 0));
                    try ctx.put(allocator, encode.movdToXmm(rd, scratch1));
                    try ctx.put(allocator, encode.vcvtph2ps(rd, rd));
                } else {
                    try ctx.put(allocator, if (isVector(func, result)) encode.movupsLoadMem(rd, base, 0) else if (isDouble(func, result)) encode.movsdLoadMem(rd, base, 0) else encode.movssLoadMem(rd, base, 0));
                }
                try ctx.storeXmm(allocator, result, rd);
            } else {
                const rd = ctx.dst(result, scratch1);
                // Load exactly the value's own width so no bytes beyond the object are read (a
                // wider load would pull garbage from the next array element into the register).
                // A narrow load sign-extends a signed value and zero-extends an unsigned one, and
                // the extend targets a 32-bit register so the upper 32 bits end up clean.
                const signed = isSigned(func, result);
                try ctx.put(allocator, switch (intBits(func, result)) {
                    0...8 => if (signed) encode.movsxByteFromMem(rd, base, 0) else encode.movzxByteFromMem(rd, base, 0),
                    9...16 => if (signed) encode.movsxWordFromMem(rd, base, 0) else encode.movzxWordFromMem(rd, base, 0),
                    17...32 => if (signed) encode.movsxdFromMem(rd, base, 0) else encode.movFromMem32(rd, base, 0),
                    else => encode.movFromMem(rd, base, 0),
                });
                try ctx.store(allocator, result, rd);
            }
        },
        else => return error.Unsupported,
    }
}

fn emitIf(allocator: std.mem.Allocator, ctx: *Ctx, cf: ir.function.If) Error!void {
    const func = ctx.func;
    const cond = try ctx.use(allocator, cf.cond, scratch1);
    // Test at the condition's width so a dirty upper half of a raw i32 cond is ignored.
    try ctx.put(allocator, encode.testReg(cond, cond, intBits(func, cf.cond) > 32));
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

fn binary(op: ir.function.BinOp, dst: Reg, src: Reg, w: bool) Error!encode.Inst {
    return switch (op) {
        .add => encode.add(dst, src, w),
        .sub => encode.sub(dst, src, w),
        .mul => encode.imul(dst, src, w),
        .bit_and => encode.andr(dst, src, w),
        .bit_or => encode.orr(dst, src, w),
        .bit_xor => encode.xorr(dst, src, w),
        .div, .rem, .shl, .shr, .mulh => error.Unsupported,
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

/// Visit every operand VALUE read by `inst`, calling `f(ctx, value, is_edge_arg)`. The block
/// arguments of an `if` are edge args (they move along a control edge), every other operand is an
/// ordinary use. Mirrors the shared `regalloc.forEachUse` operand set, adding the edge-arg flag the
/// split-liveness `is_intra` predicate needs. Kept local so `regalloc.zig` stays untouched.
fn forEachOperand(func: *const Function, inst: ir.function.Inst, ctx: anytype, comptime f: fn (@TypeOf(ctx), Value, bool) void) void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .alloca, .global_addr => {},
        .arith => |a| {
            f(ctx, a.lhs, false);
            f(ctx, a.rhs, false);
        },
        .arith_imm => |a| f(ctx, a.lhs, false),
        .icmp => |c| {
            f(ctx, c.lhs, false);
            f(ctx, c.rhs, false);
        },
        .select => |s| {
            f(ctx, s.cond, false);
            f(ctx, s.then, false);
            f(ctx, s.@"else", false);
        },
        .extract => |e| f(ctx, e.aggregate, false),
        .convert => |cv| f(ctx, cv.value, false),
        .unary => |u| f(ctx, u.value, false),
        .load => |l| f(ctx, l.ptr, false),
        .store => |st| {
            f(ctx, st.value, false);
            f(ctx, st.ptr, false);
        },
        .prefetch => |pf| f(ctx, pf.ptr, false),
        .dot => |d| {
            f(ctx, d.acc, false);
            f(ctx, d.a, false);
            f(ctx, d.b, false);
        },
        .matmul => |mmv| {
            f(ctx, mmv.a, false);
            f(ctx, mmv.b, false);
            f(ctx, mmv.c, false);
        },
        .struct_new => |sn| for (func.valueList(sn.fields)) |fld| f(ctx, fld, false),
        .call => |c| for (func.valueList(c.args)) |a| f(ctx, a, false),
        .call_indirect => |c| {
            f(ctx, c.target, false);
            for (func.valueList(c.args)) |a| f(ctx, a, false);
        },
        .@"if" => |cf| {
            f(ctx, cf.cond, false);
            for (func.blockArgs(cf.then)) |a| f(ctx, a, true);
            for (func.blockArgs(cf.@"else")) |a| f(ctx, a, true);
        },
    }
}

/// Terminator analogue of `forEachOperand`. The `jump` arguments are edge args, the `ret` value is
/// an ordinary operand.
fn forEachTermOperand(func: *const Function, term: ir.function.Terminator, ctx: anytype, comptime f: fn (@TypeOf(ctx), Value, bool) void) void {
    switch (term) {
        .ret => |v| if (v) |vv| f(ctx, vv, false),
        .jump => |j| for (func.blockArgs(j)) |a| f(ctx, a, true),
    }
}

/// Per-value split-liveness data for the eviction heuristic, built locally so `regalloc.zig` stays
/// the sole owner of the start/end intervals. `use_positions` and `is_intra`/`def_pos` are computed
/// together (the later live-range-splitting tasks consume all three), but for now only the victim
/// selection reads `use_positions`. The position numbering is exactly the one
/// `regalloc.computeLiveIntervals` uses (block params share the block-start position, then one per
/// instruction and one per terminator), so a position from here indexes the same timeline as an
/// interval's start/end.
const LocalLiveness = struct {
    def_pos: []u32, // per value: position of its definition
    is_intra: []bool, // per value: true iff NOT a param, NEVER an edge arg, and every use is in its def block
    use_positions: [][]u32, // per value: ascending positions where the value is an OPERAND use (edge args at the terminator position)

    fn deinit(self: *LocalLiveness, allocator: std.mem.Allocator) void {
        allocator.free(self.def_pos);
        allocator.free(self.is_intra);
        for (self.use_positions) |u| allocator.free(u);
        allocator.free(self.use_positions);
    }
};

/// Build the local split-liveness data over `func`, mirroring `regalloc.computeLiveIntervals`'s
/// position numbering. Caller owns the returned `LocalLiveness`.
fn computeLocalLiveness(allocator: std.mem.Allocator, func: *const Function) Error!LocalLiveness {
    const nval = func.valueCount();

    const def_pos = try allocator.alloc(u32, nval);
    errdefer allocator.free(def_pos);
    const is_intra = try allocator.alloc(bool, nval);
    errdefer allocator.free(is_intra);

    // def_block is a scratch row (which block defined each value), used only to decide is_intra.
    const def_block = try allocator.alloc(u32, nval);
    defer allocator.free(def_block);

    @memset(def_pos, 0);
    @memset(def_block, 0);
    // A value starts intra-splittable; the walk clears it for params, any edge-argument use, and any
    // use outside its def block.
    @memset(is_intra, true);

    // Per-value operand-use positions are gathered into temporary lists, then transferred into the
    // owned `use_positions` slices below.
    const use_lists = try allocator.alloc(std.ArrayList(u32), nval);
    defer allocator.free(use_lists);
    for (use_lists) |*u| u.* = .empty;
    errdefer for (use_lists) |*u| u.deinit(allocator);

    const Collector = struct {
        is_intra: []bool,
        def_block: []const u32,
        use_lists: []std.ArrayList(u32),
        allocator: std.mem.Allocator,
        pos: u32,
        bi: u32,
        err: ?Error = null,

        fn visit(self: *@This(), v: Value, is_edge_arg: bool) void {
            const vi = @intFromEnum(v);
            if (is_edge_arg) {
                self.is_intra[vi] = false;
            } else if (self.def_block[vi] != self.bi) {
                self.is_intra[vi] = false;
            }
            self.use_lists[vi].append(self.allocator, self.pos) catch |e| {
                self.err = e;
            };
        }
    };

    var pos: u32 = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| {
            def_pos[@intFromEnum(p)] = pos;
            is_intra[@intFromEnum(p)] = false;
            def_block[@intFromEnum(p)] = @intCast(bi);
        }
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            var col = Collector{
                .is_intra = is_intra,
                .def_block = def_block,
                .use_lists = use_lists,
                .allocator = allocator,
                .pos = pos,
                .bi = @intCast(bi),
            };
            forEachOperand(func, inst, &col, Collector.visit);
            if (col.err) |e| return e;
            if (func.instResult(inst)) |r| {
                def_pos[@intFromEnum(r)] = pos;
                def_block[@intFromEnum(r)] = @intCast(bi);
            }
            pos += 1;
        }
        if (func.terminator(block)) |term| {
            var col = Collector{
                .is_intra = is_intra,
                .def_block = def_block,
                .use_lists = use_lists,
                .allocator = allocator,
                .pos = pos,
                .bi = @intCast(bi),
            };
            forEachTermOperand(func, term, &col, Collector.visit);
            if (col.err) |e| return e;
        }
        pos += 1;
    }

    // Transfer the temporary use lists into owned slices. Blocks were walked in order with
    // increasing positions, so each list is already ascending.
    const use_positions = try allocator.alloc([]u32, nval);
    errdefer allocator.free(use_positions);
    var converted: usize = 0;
    errdefer for (use_positions[0..converted]) |u| allocator.free(u);
    for (use_lists, 0..) |*u, idx| {
        use_positions[idx] = try u.toOwnedSlice(allocator);
        converted = idx + 1;
    }

    return .{ .def_pos = def_pos, .is_intra = is_intra, .use_positions = use_positions };
}

/// The position of `value`'s next use strictly after `p`, or `end` (its interval end) if it has no
/// further use. This is the Belady/MIN spill key: prefer to spill the value that will not be needed
/// for the longest.
fn nextUseOrEnd(use_positions: []const []const u32, value: Value, end: u32, p: u32) u32 {
    return nextUseAfter(use_positions[@intFromEnum(value)], p) orelse end;
}

/// Return the first element of ascending `uses` that is strictly greater than `p`, else null.
fn nextUseAfter(uses: []const u32, p: u32) ?u32 {
    var lo: usize = 0;
    var hi: usize = uses.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (uses[mid] <= p) lo = mid + 1 else hi = mid;
    }
    return if (lo < uses.len) uses[lo] else null;
}

/// A GPR value currently resident in a register during linear scan, with the position its live
/// range ends. Hoisted to file scope (from a local in `assignRegs`) so `secondChance` can name the
/// type when it re-adds a re-homed value to the active set. (`assignXmm` keeps its own local
/// `Active` over `Xmm`, which this does not touch.)
const GprActive = struct { end: u32, value: Value, reg: Reg };

/// Append `seg` to `value`'s segment list (growing the owned slice). Segments must be appended in
/// ascending `from` order by the caller, so a re-spill after a re-home lands at or after the re-home
/// position (asserted). Creates a single-element list if the value has none yet.
fn appendSegment(allocator: std.mem.Allocator, segments: *std.AutoHashMapUnmanaged(Value, []Segment), value: Value, seg: Segment) Error!void {
    const gop = try segments.getOrPut(allocator, value);
    if (!gop.found_existing) {
        const s = try allocator.alloc(Segment, 1);
        s[0] = seg;
        gop.value_ptr.* = s;
        return;
    }
    const old = gop.value_ptr.*;
    // Ascending-`from` is the representation invariant `loc` relies on (it scans until the first
    // segment past `pos`). A caller that appends out of order is a programmer error and would
    // silently miscompile, so assert it rather than trust it.
    std.debug.assert(seg.from >= old[old.len - 1].from);
    const grown = try allocator.alloc(Segment, old.len + 1);
    @memcpy(grown[0..old.len], old);
    grown[old.len] = seg;
    allocator.free(old);
    gop.value_ptr.* = grown;
}

/// Undo a still-PENDING second-chance re-home of `value`: drop its trailing `.reg` segment and the
/// matching reload action, so the value stays in its previous slot and the re-home register frees.
/// Used when pressure reclaims that register before its reload fires (the value never actually
/// re-entered a register). Leaving the reload in place would clobber the taker's register at the old
/// re-home position, and leaving the segment would break the ascending-`from` order `loc` relies on,
/// so both must go.
fn cancelReHome(allocator: std.mem.Allocator, segments: *std.AutoHashMapUnmanaged(Value, []Segment), actions: *std.ArrayList(SplitAction), value: Value) Error!void {
    const old = segments.get(value).?;
    std.debug.assert(old.len >= 2 and old[old.len - 1].loc == .reg);
    const rehome = old[old.len - 1];
    const rehome_reg = rehome.loc.reg;
    // Remove the pending reload action for this re-home. The (value, position, register) triple is
    // unique (each re-home pops a distinct register), so exactly one action matches. Actions are not
    // sorted until emission, so a `swapRemove` is safe here.
    var found = false;
    var i: usize = 0;
    while (i < actions.items.len) : (i += 1) {
        const act = actions.items[i];
        if (act.kind == .reload and act.value == value and act.at == rehome.from and act.reg == rehome_reg) {
            _ = actions.swapRemove(i);
            found = true;
            break;
        }
    }
    std.debug.assert(found);
    // Shrink the owned segment slice by one (drop the trailing `.reg`). The key already exists, so the
    // `getPtr` update never allocates and cannot fail after the new slice is built.
    const shrunk = try allocator.alloc(Segment, old.len - 1);
    @memcpy(shrunk, old[0 .. old.len - 1]);
    allocator.free(old);
    segments.getPtr(value).?.* = shrunk;
}

/// After the current interval is placed at `pos`, re-home split GPR values that presently live in a
/// slot and still have an upcoming use into any LEFTOVER free register, so their remaining tail uses
/// read a register instead of reloading from the slot on every use. The reload lands at the value's
/// next use position. Most-urgent (nearest next use) first, so a scarce free register goes to the
/// value that reloads soonest. The re-homed value is added back to `active` so its register is
/// tracked and it can be spilled AGAIN (via the append-aware split path) if pressure returns.
///
/// Runs AFTER the current interval was placed, so that interval already claimed whatever register it
/// needed and `secondChance` only pops registers that are genuinely free. It therefore can never
/// dispossess a live value. A split value provably never crosses a call (across-call values are
/// force-whole-spilled and never enter `active`, so they are never a split victim), so its whole
/// tail is call-free and re-homing into a caller-saved register is call-safe with no extra gate.
fn secondChance(
    allocator: std.mem.Allocator,
    lin: *const LocalLiveness,
    free: *std.ArrayList(Reg),
    active: *std.ArrayList(GprActive),
    segments: *std.AutoHashMapUnmanaged(Value, []Segment),
    actions: *std.ArrayList(SplitAction),
    pos: u32,
) Error!void {
    const Cand = struct { value: Value, next: u32, slot: u32, end: u32 };
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(allocator);
    var it = segments.iterator();
    while (it.next()) |e| {
        const segs = e.value_ptr.*;
        const last = segs[segs.len - 1];
        switch (last.loc) {
            .spill => |slot| {
                const v = e.key_ptr.*;
                const uses = lin.use_positions[@intFromEnum(v)];
                const nu = nextUseAfter(uses, pos) orelse continue;
                // A split value is `is_intra` (only intra values are ever tail-split), so its live
                // range ends at its last use. That is the register's expiry point in `active`.
                try cands.append(allocator, .{ .value = v, .next = nu, .slot = slot, .end = uses[uses.len - 1] });
            },
            .reg => {}, // already in a register, nothing pending
            .xmm, .xmm_spill => unreachable, // GPR segments only
        }
    }
    std.mem.sort(Cand, cands.items, {}, struct {
        fn f(_: void, a: Cand, b: Cand) bool {
            return a.next < b.next;
        }
    }.f);
    for (cands.items) |c| {
        const r2 = free.pop() orelse continue;
        try appendSegment(allocator, segments, c.value, .{ .from = c.next, .loc = .{ .reg = r2 } });
        try actions.append(allocator, .{ .at = c.next, .kind = .reload, .value = c.value, .reg = r2, .slot = c.slot });
        try active.append(allocator, .{ .end = c.end, .value = c.value, .reg = r2 });
    }
}

/// Linear-scan register allocation with reuse and spilling. Entry parameters are not pinned,
/// the prologue moves arguments into their assigned locations. R10/R11 are reserved as
/// spill/move scratch. RAX/RDX/RCX are reserved for division/shifts.
fn assignRegs(allocator: std.mem.Allocator, func: *const Function, loc_of: *std.AutoHashMapUnmanaged(Value, Loc), num_slots: *u32, def_pos_out: *[]u32, segments: *std.AutoHashMapUnmanaged(Value, []Segment), actions: *std.ArrayList(SplitAction)) Error!void {
    { // general args must fit the gpr ABI registers; fp args beyond xmm0..7 are loaded from the stack
        // by the prologue (System V callee-side stack args). Gpr stack args are not handled yet.
        var gpr_params: usize = 0;
        for (func.blockParams(@enumFromInt(0))) |p| {
            if (!isXmm(func, p)) gpr_params += 1;
        }
        if (gpr_params > arg_regs.len) return error.Unsupported;
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

    // Local split-liveness (use positions, is_intra, def positions) over the same position timeline
    // as the intervals. The victim heuristic below reads `use_positions` for the Belady/MIN key; the
    // rest is kept for the later live-range-splitting tasks.
    var lin = try computeLocalLiveness(allocator, func);
    defer lin.deinit(allocator);
    // Surface the per-value def positions to the caller (compile stores them on Ctx for the
    // pos-coupling assert). Dupe because `lin` is freed when assignRegs returns. Ownership passes to
    // the caller immediately via the out-param, so a later failure here is freed by the caller.
    def_pos_out.* = try allocator.dupe(u32, lin.def_pos);

    // A `call` clobbers every caller-saved register, so a value live across a call (defined
    // before it, used after it) cannot stay in a register, force it to a spill slot. Its call
    // arguments and the call result are not "across".
    const calls = try callPositions(allocator, func);
    defer allocator.free(calls);

    var active: std.ArrayList(GprActive) = .empty;
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
            // Out of registers: pick the active value whose NEXT USE after this point is furthest
            // ahead (Belady/MIN), ties broken by the larger interval end. `p` is where the pool is
            // exhausted, this interval's definition position. Reaching here means `free` was empty,
            // so the whole (non-empty) GPR pool is resident in `active`.
            std.debug.assert(active.items.len > 0);
            const p = iv.start;
            var victim: usize = 0;
            for (active.items, 0..) |act, i| {
                const act_nu = nextUseOrEnd(lin.use_positions, act.value, act.end, p);
                const best = active.items[victim];
                const best_nu = nextUseOrEnd(lin.use_positions, best.value, best.end, p);
                if (act_nu > best_nu or (act_nu == best_nu and act.end > best.end)) victim = i;
            }
            // Evict the victim (whole-spill) and give its register to `iv` only when the victim's
            // next use is strictly further than `iv`'s own, else whole-spill `iv` instead.
            const vic_nu = nextUseOrEnd(lin.use_positions, active.items[victim].value, active.items[victim].end, p);
            const iv_nu = nextUseOrEnd(lin.use_positions, iv.value, iv.end, p);
            if (vic_nu > iv_nu) {
                const v = active.items[victim];
                const vidx = @intFromEnum(v.value);
                // TAIL-SPLIT the victim when it is intra with a register prefix: keep its register
                // for the hot prefix `[def, p)`, move to a fresh slot for the cold tail `[p, end)`,
                // and record a store at `p`. Its old whole-life `.reg` entry (if any) is shadowed by
                // the segments (loc() checks segments first), so removing it is cosmetic but cleaner.
                // The `across` branch force-whole-spills before we ever reach here, so an active
                // victim is never a cross-call value, and every GPR use is reloadable, so splitting
                // is fully general. Cross-block and param victims (`!is_intra`) whole-spill.
                if (lin.is_intra[vidx] and lin.def_pos[vidx] < p) {
                    if (segments.get(v.value)) |segs| {
                        // Already tail-split then SECOND-CHANCE RE-HOMED into `v.reg`, so its last
                        // segment is a `.reg`. Two sub-cases by whether that re-home's reload has fired
                        // yet at `p`:
                        const last = segs[segs.len - 1];
                        std.debug.assert(last.loc == .reg);
                        if (last.from > p) {
                            // PENDING re-home: pressure reclaims `v.reg` BEFORE the reload at
                            // `last.from` runs, so the value is still physically in its previous slot
                            // here and never re-enters a register. Cancel the re-home (drop the trailing
                            // `.reg` segment and its reload action) rather than append an out-of-order
                            // slot. The value stays in that prior slot (no store, its bits are there)
                            // and `v.reg` frees for the taker.
                            try cancelReHome(allocator, segments, actions, v.value);
                        } else {
                            // ACTIVE re-home (`last.from <= p`): the value truly lives in `v.reg` at
                            // `p`. Spill that register's live part from `p` onward. The append lands at
                            // or after the re-home position, so segment order is preserved.
                            const slot = num_slots.*;
                            num_slots.* += 1;
                            try appendSegment(allocator, segments, v.value, .{ .from = p, .loc = .{ .spill = slot } });
                            try actions.append(allocator, .{ .at = p, .kind = .store, .value = v.value, .reg = v.reg, .slot = slot });
                        }
                    } else {
                        // FIRST split: keep its register for the hot prefix `[def, p)`, move to a fresh
                        // slot for the cold tail `[p, end)`, and record a store at `p`.
                        const slot = num_slots.*;
                        num_slots.* += 1;
                        const segs = try allocator.alloc(Segment, 2);
                        segs[0] = .{ .from = lin.def_pos[vidx], .loc = .{ .reg = v.reg } };
                        segs[1] = .{ .from = p, .loc = .{ .spill = slot } };
                        segments.put(allocator, v.value, segs) catch |e| {
                            allocator.free(segs);
                            return e;
                        };
                        try actions.append(allocator, .{ .at = p, .kind = .store, .value = v.value, .reg = v.reg, .slot = slot });
                        _ = loc_of.remove(v.value);
                    }
                } else {
                    try loc_of.put(allocator, v.value, .{ .spill = num_slots.* });
                    num_slots.* += 1;
                }
                try loc_of.put(allocator, iv.value, .{ .reg = v.reg });
                active.items[victim] = .{ .end = iv.end, .value = iv.value, .reg = v.reg };
            } else {
                try loc_of.put(allocator, iv.value, .{ .spill = num_slots.* });
                num_slots.* += 1;
            }
        }

        // `iv` is now placed. Offer any leftover free registers to slot-resident split values whose
        // tail is still ahead, re-homing them so their remaining uses read a register. Runs after
        // placement so it can only ever hand out registers `iv` did not need.
        try secondChance(allocator, &lin, &free, &active, segments, actions, iv.start);
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
        .float => |f| switch (f) {
            .f16 => 2, // a 2-byte IEEE half in memory (its in-register form is the f32 widening)
            .f32 => 4,
            .f64 => 8,
        },
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

test "an f16 function now compiles on x86_64 (F16C, no reject gate)" {
    // The f16 rejection gate was replaced with real F16C lowering. A function that adds two f16
    // values must now compile: it emits the scalar-single addss followed by the round-to-half
    // pair (vcvtps2ph then vcvtph2ps). Byte-scan the code for both F16C opcode maps.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f16 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = s });

    const code = try selectFunction(allocator, &func);
    defer allocator.free(code);
    var saw_narrow = false; // vcvtps2ph: VEX C4, ..., 0F3A map, opcode 1D
    var saw_widen = false; // vcvtph2ps: VEX C4, ..., 0F38 map, opcode 13
    for (0..code.len) |i| {
        if (code[i] != 0xC4 or i + 4 >= code.len) continue;
        const map = code[i + 1] & 0x1F; // low 5 bits of VEX byte2 are mmmmm
        const op = code[i + 3];
        if (map == 0x03 and op == 0x1D) saw_narrow = true;
        if (map == 0x02 and op == 0x13) saw_widen = true;
    }
    try std.testing.expect(saw_narrow and saw_widen);
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

test "x86-64 eviction spills the furthest-next-use value, not the furthest-end one" {
    // Belady/MIN spill selection: when the GPR pool is exhausted the allocator must spill the
    // active value whose NEXT USE lies furthest ahead, not the one whose interval merely ENDS
    // furthest. The function below is crafted so the two heuristics disagree on exactly one
    // forced spill. Seven i32 constants fill the seven-register GPR pool, then an eighth
    // definition (`trig`) forces one spill:
    //   far_end is used once soon (its next use is close) but again very late, so its interval
    //           END is the furthest of all candidates. The old furthest-end rule spills it.
    //   far_use is used exactly once, further ahead than far_end's soon use, so its NEXT USE is
    //           the furthest of all candidates. The new Belady rule spills it.
    // The assertions hold only for the next-use rule and both fail for the furthest-end rule.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();

    const far_use = try func.appendInst(b, t, .{ .iconst = 300 });
    const far_end = try func.appendInst(b, t, .{ .iconst = 400 });
    var fillers: [5]Value = undefined;
    for (&fillers, 0..) |*fv, i| fv.* = try func.appendInst(b, t, .{ .iconst = @intCast(i + 1) });
    // The eighth live value: exhausts the seven-register pool and forces the single spill. It
    // consumes two fillers so their intervals end exactly here (they are the smallest next-use
    // candidates and are never chosen as the victim).
    const trig = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = fillers[0], .rhs = fillers[1] } });

    // Tail accumulator: each add sets a candidate's next-use / interval-end. far_end is used
    // once soon (here) and once at the very end; far_use exactly once, further ahead than
    // far_end's soon use; the remaining fillers drain between them so pressure only falls.
    var acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = trig, .rhs = far_end } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = fillers[2] } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = fillers[3] } });
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = fillers[4] } });
    // far_use's sole use, the furthest NEXT use of any candidate at the spill point.
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = far_use } });
    // far_end's final use, the furthest interval END of any candidate.
    acc = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = far_end } });
    func.setTerminator(b, .{ .ret = acc });

    var loc_of: std.AutoHashMapUnmanaged(Value, Loc) = .{};
    defer loc_of.deinit(allocator);
    var num_slots: u32 = 0;
    var def_pos: []u32 = &.{};
    defer allocator.free(def_pos);
    var segments: std.AutoHashMapUnmanaged(Value, []Segment) = .{};
    defer {
        var it = segments.valueIterator();
        while (it.next()) |s| allocator.free(s.*);
        segments.deinit(allocator);
    }
    var actions: std.ArrayList(SplitAction) = .empty;
    defer actions.deinit(allocator);
    try assignRegs(allocator, &func, &loc_of, &num_slots, &def_pos, &segments, &actions);

    // Exactly the next-use victim is evicted: far_use (furthest next use), not far_end (furthest
    // end). far_use is intra with a register prefix, so eviction TAIL-SPLITS it (register prefix,
    // spill tail) rather than whole-spilling, and its whole-life `loc_of` entry is removed. far_end
    // stays whole in a register. The old furthest-end heuristic would evict far_end instead, so
    // neither the split-of-far_use nor the register-for-far_end assertion would hold. After the split
    // the fillers drain and pressure falls, so SECOND-CHANCE (Task 7d) re-homes far_use into a freed
    // register at its sole tail use, appending a trailing `.reg` segment: the split shape is
    // `.reg` (hot prefix) -> `.spill` (cold tail) -> `.reg` (re-home).
    try std.testing.expect(loc_of.get(far_use) == null);
    const far_use_segs = segments.get(far_use).?;
    try std.testing.expectEqual(@as(usize, 3), far_use_segs.len);
    try std.testing.expect(far_use_segs[0].loc == .reg);
    try std.testing.expect(far_use_segs[1].loc == .spill);
    try std.testing.expect(far_use_segs[2].loc == .reg);
    try std.testing.expect(segments.get(far_end) == null);
    try std.testing.expect(loc_of.get(far_end).? == .reg);
}

test "x86-64 loc returns the whole-life location for an unsplit value" {
    // A tiny no-spill int function: every value stays in one register for its whole life, so
    // `segments` is empty. The segment-aware `loc` must return exactly the `loc_of` location at
    // every position, proving the fallback path is byte-identical while nothing is split (Task 7c
    // fills `segments`).
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, t);
    const bp = try func.appendBlockParam(b, t);
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    const prod = try func.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = sum, .rhs = bp } });
    func.setTerminator(b, .{ .ret = prod });

    var ctx = Ctx{ .func = &func };
    defer ctx.loc_of.deinit(allocator);
    defer {
        var seg_it = ctx.segments.valueIterator();
        while (seg_it.next()) |s| allocator.free(s.*);
        ctx.segments.deinit(allocator);
    }
    defer allocator.free(ctx.def_pos);
    defer ctx.actions.deinit(allocator);
    var num_slots: u32 = 0;
    try assignRegs(allocator, &func, &ctx.loc_of, &num_slots, &ctx.def_pos, &ctx.segments, &ctx.actions);

    // Nothing was split.
    try std.testing.expectEqual(@as(usize, 0), ctx.segments.count());

    // At several positions the segment-aware loc equals the whole-life loc_of location.
    const values = [_]Value{ a, bp, sum, prod };
    const positions = [_]u32{ 0, 1, 2, 3, 4, 5 };
    for (positions) |p| {
        ctx.pos = p;
        for (values) |v| {
            try std.testing.expectEqual(ctx.loc_of.get(v).?, ctx.loc(v));
        }
    }
}

test "nextUseAfter finds the first strictly-greater use" {
    try std.testing.expectEqual(@as(?u32, 9), nextUseAfter(&.{ 2, 5, 9 }, 5));
    try std.testing.expectEqual(@as(?u32, 5), nextUseAfter(&.{ 2, 5, 9 }, 2));
    try std.testing.expectEqual(@as(?u32, 2), nextUseAfter(&.{ 2, 5, 9 }, 0));
    try std.testing.expectEqual(@as(?u32, null), nextUseAfter(&.{ 2, 5, 9 }, 9));
    try std.testing.expectEqual(@as(?u32, null), nextUseAfter(&.{}, 0));
}

test "x86-64 local liveness records intra predicate and ascending use positions" {
    // V is defined and used ONLY in its def block (intra-splittable). W is defined in the entry
    // block and passed as an edge argument to a successor (not intra). X is defined in the entry
    // block but used by a normal instruction in a DIFFERENT block (not intra). Params are never
    // intra. This exercises the local split-liveness the eviction heuristic and later splitting
    // tasks read.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const b0 = try func.appendBlock();
    const b1 = try func.appendBlock();
    const a = try func.appendBlockParam(b0, t);
    const bp = try func.appendBlockParam(b0, t);
    const pparam = try func.appendBlockParam(b1, t);

    const v = try func.appendInst(b0, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    _ = try func.appendInst(b0, t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = bp } }); // V's sole use, in its def block
    const w = try func.appendInst(b0, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    const x = try func.appendInst(b0, t, .{ .arith = .{ .op = .mul, .lhs = bp, .rhs = bp } });
    const cond = try func.appendInst(b0, bool_t, .{ .icmp = .{ .op = .le, .lhs = a, .rhs = bp } });
    try func.appendIf(b0, cond, .{ .target = b1, .args = &.{w} }, .{ .target = b1, .args = &.{w} }); // W flows as an edge arg

    const xuse = try func.appendInst(b1, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = pparam } }); // X used in b1, a different block
    func.setTerminator(b1, .{ .ret = xuse });

    var lin = try computeLocalLiveness(allocator, &func);
    defer lin.deinit(allocator);

    try std.testing.expect(lin.is_intra[@intFromEnum(v)]);
    try std.testing.expect(!lin.is_intra[@intFromEnum(w)]);
    try std.testing.expect(!lin.is_intra[@intFromEnum(x)]);
    try std.testing.expect(!lin.is_intra[@intFromEnum(a)]);
    try std.testing.expect(!lin.is_intra[@intFromEnum(bp)]);
    try std.testing.expect(!lin.is_intra[@intFromEnum(pparam)]);

    // V's use positions are non-empty and strictly ascending, and def_pos precedes the first use.
    const vuses = lin.use_positions[@intFromEnum(v)];
    try std.testing.expect(vuses.len > 0);
    try std.testing.expect(lin.def_pos[@intFromEnum(v)] < vuses[0]);
    var i: usize = 1;
    while (i < vuses.len) : (i += 1) try std.testing.expect(vuses[i] > vuses[i - 1]);
}

test "a call keeps RSP 16-aligned at the call site (movaps-safe host calls)" {
    // A callee reading its stack with movaps faults on real hardware when the caller's RSP is
    // misaligned (qemu-user does not enforce this, so only a codegen check catches it). Entry RSP
    // is 8 (mod 16), so the prologue frame must be 8 (mod 16) to land calls on a 16 boundary.
    const allocator = std.testing.allocator;
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(.{ .float = .f32 });
        const b = try callee.appendBlock();
        const a = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = a });
    }
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(.{ .float = .f32 });
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "helper", &.{x});
        caller.setTerminator(b, .{ .ret = r });
    }
    const code = try selectFunction(allocator, &caller);
    defer allocator.free(code);
    const text = try @import("disasm.zig").format(allocator, code);
    defer allocator.free(text);
    const marker = "sub rsp, ";
    const at = std.mem.indexOf(u8, text, marker) orelse return error.NoPrologue;
    const rest = text[at + marker.len ..];
    const end = std.mem.indexOfScalar(u8, rest, '\n') orelse rest.len;
    const frame = try std.fmt.parseInt(u32, rest[0..end], 10);
    try std.testing.expectEqual(@as(u32, 8), frame % 16);
}
