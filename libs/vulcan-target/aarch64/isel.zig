//! AArch64 instruction selection: lowers an IR function to A64 machine words.
//! Covers int/float arith (incl. div/rem/shifts), comparisons, select, the high-IR
//! `if`, jumps with block-param edge moves, ret/call, NEON vectors, and stack memory.
//!
//! Linear-scan register allocation with spilling. Live intervals (def..last-use)
//! are scanned in start order from a register pool (caller-saved x9..x12 plus unused
//! arg regs for a leaf, callee-saved x19..x28 for a non-leaf). On pool exhaustion a
//! result spills to a stack slot. Block parameters are never spilled, keeping edge
//! moves simple. A non-leaf opens a frame saving lr and its callee-saved registers.
//! alloca/spill slots live above them.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const loops = @import("vulcan-opt").loops;
const mm = @import("vulcan-opt").microarch;

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Reg = encode.Reg;
const RegMap = std.AutoHashMapUnmanaged(Value, Reg);

pub const Error = std.mem.Allocator.Error || error{Unsupported};

const sp: Reg = .zr; // register 31 is the stack pointer in load/store/frame context

/// Emit `rd = rn -/+ amount` via the add/sub-immediate form, splitting amounts
/// wider than the 12-bit immediate across the unshifted and `LSL #12` forms (up to
/// ~16MB in two instructions). The immediate form is required for SP operands
/// (Rn/Rd 31 means SP), so the register form can not be used to adjust the frame.
fn emitFrameImm(allocator: std.mem.Allocator, code: *std.ArrayList(u32), is_sub: bool, rd: Reg, rn: Reg, amount: usize) !void {
    const lo: u12 = @intCast(amount & 0xFFF);
    const hi: u12 = @intCast(amount >> 12); // shader frames are far below the 16MB ceiling
    var src = rn;
    if (hi != 0) {
        try code.append(allocator, if (is_sub) encode.subImm64Shift(rd, src, hi) else encode.addImm64Shift(rd, src, hi));
        src = rd;
    }
    if (lo != 0 or hi == 0) {
        try code.append(allocator, if (is_sub) encode.subImm64(rd, src, lo) else encode.addImm64(rd, src, lo));
    }
}
const spill_op = [_]Reg{ .x13, .x14 }; // GPR scratch for reloading operands
const spill_res: Reg = .x15; // GPR scratch for a spilled result
const scratch_imm: Reg = .x16; // arith-immediate constant / remainder quotient / fp bits
const scratch_move: Reg = .x17; // GPR parallel-move cycle breaking
// FP scratch (same index space names v-registers, v24..v27 are caller-saved and
// outside every FP pool).
const fp_spill_op = [_]Reg{ @as(Reg, @enumFromInt(24)), @as(Reg, @enumFromInt(25)) };
const fp_spill_res: Reg = @enumFromInt(26);
const fp_move: Reg = @enumFromInt(27);

/// A value lives in either the general-register file (gpr) or the FP/SIMD file
/// (fpr), decided by its type. The same numeric register index names x_n or v_n.
const Class = enum { gpr, fpr };

fn regClass(func: *const Function, v: Value) Class {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float, .vector => .fpr, // a SIMD vector lives in the v registers (Q view)
        else => .gpr,
    };
}

/// Whether `v` is a SIMD vector (lowered with NEON `.4S`-style instructions rather
/// than the scalar single/double forms).
fn isVector(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .vector;
}

fn isDouble(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f64,
        else => false,
    };
}

/// Whether `v` is an f16 (half). f16 is emulated: it lives in an S register as its f32
/// widening (so `isDouble(f16)` is false and every in-register op uses the S-form naturally),
/// and the boundaries widen/narrow with `fcvt`. `isHalf` marks the sites that must add that
/// widening/narrowing: memory load/store, narrowing converts, and arithmetic results.
fn isHalf(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f16,
        else => false,
    };
}

/// The scalar-FP `ftype` selector for `v`'s register view. Under `fp16` (native FEAT_FP16), an
/// f16 lives in an H register and selects `.half`. Otherwise (the emulation, where an f16 lives
/// as its exact f32 widening in an S register) an f16 selects `.single` exactly as f32 does, so
/// the encoding is byte-identical to the pre-FEAT_FP16 output. f64 always selects `.double`.
fn fkindOf(func: *const Function, v: Value, fp16: bool) encode.FKind {
    if (fp16 and isHalf(func, v)) return .half;
    return if (isDouble(func, v)) .double else .single;
}

/// Round the f16 value held in the S view of `reg` to nearest-even half and re-widen it, in
/// place. This is the per-op IEEE rounding of the f16 emulation: an f16 arithmetic result or
/// f32->f16 convert is first computed in f32, then this narrows it to half and widens back so
/// the S register again holds an exact half value. Skipping it would keep f32 precision, which
/// is WRONG for f16 semantics (each op must round to nearest-even half).
fn roundToHalf(allocator: std.mem.Allocator, code: *std.ArrayList(u32), reg: Reg) Error!void {
    try code.append(allocator, encode.fcvtHfromS(reg, reg));
    try code.append(allocator, encode.fcvtSfromH(reg, reg));
}

/// Where an argument/parameter is passed: which file, and its index within that
/// file (registers x0..x7 / v0..v7, or a stack slot when the index is >= 8).
const ArgLoc = struct { class: Class, idx: usize };

fn computeArgLocs(func: *const Function, values: []const Value, out: []ArgLoc) void {
    var gi: usize = 0;
    var fi: usize = 0;
    for (values, 0..) |v, k| {
        if (regClass(func, v) == .fpr) {
            out[k] = .{ .class = .fpr, .idx = fi };
            fi += 1;
        } else {
            out[k] = .{ .class = .gpr, .idx = gi };
            gi += 1;
        }
    }
}

const Move = struct { src: Reg, dst: Reg };
const Fixup = struct { at: usize, target: u32 };

/// A relocation: the word index of a `bl` whose target is the named symbol.
pub const Reloc = struct { offset: usize, symbol: []const u8 };

/// Machine words plus the call relocations for the linker to resolve.
/// One row of the source-line table: the byte offset (from the function start) where a new
/// source line's code begins, and that line. Built from the `debug.line` IR attributes.
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

/// The result of register allocation.
const Allocation = struct {
    reg: RegMap = .empty, // value -> register (index, class implied by the value type)
    spill: std.AutoHashMapUnmanaged(Value, u32) = .empty, // value -> spill slot index
    saved_gpr: std.ArrayList(Reg) = .empty, // callee-saved x-registers used (non-leaf)
    saved_fpr: std.ArrayList(Reg) = .empty, // callee-saved v-registers used (non-leaf)
    spill_count: u32 = 0,

    fn deinit(self: *Allocation, allocator: std.mem.Allocator) void {
        self.reg.deinit(allocator);
        self.spill.deinit(allocator);
        self.saved_gpr.deinit(allocator);
        self.saved_fpr.deinit(allocator);
    }
};

/// Capabilities a model-aware call site threads into `compileFunction`. Grouped into one struct
/// (rather than growing `compileFunction`'s parameter list one flag per model feature) so adding
/// the next capability never touches every existing call site. Every field defaults off, so `.{}`
/// is exactly today's behavior for every non-model caller (`selectFunction`,
/// `selectFunctionWithLines`, and the direct `compileFunction` callers in `link.zig`/`object.zig`):
/// no loop-header alignment padding, and the base-ISA f16 EMULATION (an f16 held as its f32
/// widening in an S register, rounded per-op with `fcvt`).
pub const ModelCaps = struct {
    /// Loop-header alignment in bytes (0 disables it). See `compileFunction`'s doc comment.
    fetch_align: u16 = 0,
    /// Use NATIVE half-precision arithmetic (H-form ops, an f16 held in an H register, single-
    /// rounded, no per-op widen/narrow) instead of the emulation. Set only when the target model's
    /// `features.aarch64.fp16` (FEAT_FP16) is true (see `selectFunctionForModel`). When false the
    /// f16 lowering is byte-identical to the pre-FEAT_FP16 emulation.
    fp16: bool = false,
};

/// Select A64 words for `func`, discarding relocations. The caller owns the slice.
pub fn selectFunction(allocator: std.mem.Allocator, func: *const Function) Error![]u32 {
    const compiled = try compileFunction(allocator, func, .{});
    allocator.free(compiled.relocs);
    allocator.free(compiled.lines);
    return compiled.code;
}

/// Like `selectFunction`, but pads loop-header blocks with nops so they land on a
/// `fetch_align`-byte boundary (a performance hint from the microarch model; 0
/// disables it). Never changes the function's result, only where headers fall.
pub fn selectFunctionAligned(allocator: std.mem.Allocator, func: *const Function, fetch_align: u16) Error![]u32 {
    const compiled = try compileFunction(allocator, func, .{ .fetch_align = fetch_align });
    allocator.free(compiled.relocs);
    allocator.free(compiled.lines);
    return compiled.code;
}

/// Compile `func` tuned to `model`: the machine-level hooks read the model's `fetch_align`
/// (loop-header alignment) and `features.aarch64.fp16` (whether to use native FEAT_FP16 half
/// arithmetic instead of the emulation). Fusion is already unconditional, so these are the
/// model-aware seams a caller needs. An inert model (fetch_align 0, fp16 false) makes this
/// byte-identical to `selectFunction`. Builds the full `ModelCaps` and calls `compileFunction`
/// directly rather than through `selectFunctionAligned`, since that narrower entry point only
/// ever carries `fetch_align`.
pub fn selectFunctionForModel(allocator: std.mem.Allocator, func: *const Function, model: *const mm.Model) Error![]u32 {
    // Passing a foreign-arch model here is a caller bug, not a runtime fault.
    std.debug.assert(model.arch == .aarch64);
    const caps: ModelCaps = .{
        .fetch_align = model.fetch_align,
        .fp16 = model.arch == .aarch64 and model.features.aarch64.fp16,
    };
    const compiled = try compileFunction(allocator, func, caps);
    allocator.free(compiled.relocs);
    allocator.free(compiled.lines);
    return compiled.code;
}

/// Compiled code plus its source-line table (from the `debug.line` IR attributes).
pub const CodeWithLines = struct { code: []u32, lines: []LineEntry };

/// Like `selectFunction`, but also returns the source-line table for DWARF `.debug_line`.
/// Caller owns both slices.
pub fn selectFunctionWithLines(allocator: std.mem.Allocator, func: *const Function) Error!CodeWithLines {
    const compiled = try compileFunction(allocator, func, .{});
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

fn isLeaf(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .call or func.opcode(inst) == .call_indirect) return false;
        }
    }
    return true;
}

/// The largest argument count of any call (so the frame can reserve outgoing
/// stack-argument space). Zero if the function makes no calls.
fn maxCallArgs(func: *const Function) usize {
    var max: usize = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .call => |c| max = @max(max, func.valueList(c.args).len),
                .call_indirect => |c| max = @max(max, func.valueList(c.args).len),
                else => {},
            }
        }
    }
    return max;
}

fn alignUp(v: usize, a: usize) usize {
    return (v + a - 1) & ~(a - 1);
}

/// Number of nop words to insert so a block starting at `words` (current code length in
/// 4-byte words) lands on a `fetch_align`-byte boundary. Zero when fetch_align is at most
/// one word (already aligned) or the block is already on a boundary.
fn alignPadWords(words: usize, fetch_align: u16) usize {
    if (fetch_align <= 4) return 0;
    const per: usize = fetch_align / 4; // words per alignment boundary
    const rem = words % per;
    return if (rem == 0) 0 else per - rem;
}

/// Compile `func` to A64 words and call relocations. The caller owns the result.
/// `fetch_align` is the microarch model's fetch granularity in bytes (0 disables loop-header
/// alignment, the behavior of every existing caller). When greater than one instruction word,
/// each loop-header block is padded with nops up to a `fetch_align` boundary before its code is
/// emitted, so a hot loop's fetch groups pack efficiently. This is purely a placement hint: the
/// padding falls straight through into the header and every branch fixup is patched from
/// `block_start` (recorded after padding), so it can never change what the function computes.
pub fn compileFunction(allocator: std.mem.Allocator, func: *const Function, caps: ModelCaps) Error!Compiled {
    const fetch_align = caps.fetch_align;
    // Native half arithmetic (FEAT_FP16) vs the base-ISA emulation, gated on the model feature.
    // `fp16 == false` (every non-model caller) keeps the emulation, byte-identical to before this
    // capability existed; only `selectFunctionForModel` under a FEAT_FP16 model sets it true.
    const fp16 = caps.fp16;
    // aarch64 is the reference f16 backend (f16 roadmap Task 3). By default f16 is EMULATED: an
    // f16 value lives in an S register as its f32 WIDENING (a value exactly representable in
    // half). The boundaries do the rounding via base-ISA `fcvt` (no FEAT_FP16): a memory
    // load is `ldr h; fcvt s,h`, a store is `fcvt h,s; str h`, every arithmetic result and
    // narrowing convert rounds to nearest-even half with `fcvt h,s; fcvt s,h`. Under `fp16` the
    // Native path holds the f16 in an H register and uses the single-rounded H-form ops directly
    // (no widen/narrow). The other backends still reject f16 via `functionUsesF16` (kept for
    // them); aarch64 no longer does.
    // Only SCALAR f16 is handled; f16 nested in a vector/aggregate would fall through to the
    // raw-vector path and miscompile the half lanes, so reject that composite case cleanly.
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;
    const nblocks = func.blockCount();
    if (nblocks == 0) return error.Unsupported;
    const leaf = isLeaf(func);

    var alloc = try allocate(allocator, func, leaf);
    defer alloc.deinit(allocator);

    // Frame, from sp upward: [outgoing stack args][saved lr][callee-saved x-regs]
    // [callee-saved v-regs][alloca slots][spill slots]. The outgoing-arg area sits at
    // the bottom, exactly where a callee reads its incoming stack arguments. lr is
    // saved like a callee-saved register (x29 is never used).
    var alloca_off: std.AutoHashMapUnmanaged(Value, u32) = .empty;
    defer alloca_off.deinit(allocator);
    const alloca_bytes = try computeAllocaSlots(allocator, func, &alloca_off);
    const outgoing_bytes = alignUp(8 * (maxCallArgs(func) -| 8), 16);
    const lr_bytes: usize = if (leaf) 0 else 8;
    const gpr_saved_base = outgoing_bytes + lr_bytes;
    const fpr_saved_base = gpr_saved_base + 8 * alloc.saved_gpr.items.len;
    const alloca_base = fpr_saved_base + 8 * alloc.saved_fpr.items.len;
    // Spill slots are a uniform 16 bytes so a 128-bit SIMD vector spills/reloads
    // whole (a scalar wastes the upper 8 bytes). The base is 16-aligned for `ldr q`.
    const spill_base = alignUp(alloca_base + alloca_bytes, 16);
    const spill_bytes = alloc.spill_count * 16;
    const frame: usize = if (!leaf)
        alignUp(@max(spill_base + spill_bytes, lr_bytes), 16)
    else if (spill_base + spill_bytes > 0)
        alignUp(spill_base + spill_bytes, 16)
    else
        0;
    const lr_off = outgoing_bytes;

    var code: std.ArrayList(u32) = .empty;
    errdefer code.deinit(allocator);
    var relocs: std.ArrayList(Reloc) = .empty;
    errdefer relocs.deinit(allocator);
    var lines: std.ArrayList(LineEntry) = .empty;
    errdefer lines.deinit(allocator);
    var last_line: u32 = 0; // suppress duplicate rows for consecutive same-line instructions
    var fixups: std.ArrayList(Fixup) = .empty;
    defer fixups.deinit(allocator);
    var block_start = try allocator.alloc(usize, nblocks);
    defer allocator.free(block_start);

    // Loop-header alignment (a placement hint only, see the doc comment above): computed once,
    // up front, so the per-block loop below just checks a bit per block.
    var is_loop_header = try allocator.alloc(bool, nblocks);
    defer allocator.free(is_loop_header);
    @memset(is_loop_header, false);
    if (fetch_align > 4) {
        var li = try loops.analyze(allocator, func);
        defer li.deinit(allocator);
        for (li.loops) |l| is_loop_header[l.header] = true;
    }

    if (frame > 0) try emitFrameImm(allocator, &code, true, sp, sp, frame);
    if (!leaf) {
        try code.append(allocator, encode.strOff(.x30, sp, @intCast(lr_off)));
        for (alloc.saved_gpr.items, 0..) |r, i| try code.append(allocator, encode.strOff(r, sp, @intCast(gpr_saved_base + 8 * i)));
        for (alloc.saved_fpr.items, 0..) |r, i| try code.append(allocator, encode.strFp(r, sp, @intCast(fpr_saved_base + 8 * i), true));
    }
    // Move register arguments into their parameter registers (a leaf keeps them
    // pinned, so no move), and load stack parameters (the 9th onward) from the
    // caller's outgoing-argument area.
    const eparams = func.blockParams(@enumFromInt(0));
    const plocs = try allocator.alloc(ArgLoc, eparams.len);
    defer allocator.free(plocs);
    computeArgLocs(func, eparams, plocs);
    for (eparams, plocs) |p, l| {
        // A spilled parameter (e.g. a vector input that crosses a call - the callee-saved FP
        // registers only preserve the low 64 bits, so a lane vector must live on the stack):
        // store its incoming argument (register or caller stack slot) into its spill slot.
        if (alloc.reg.get(p) == null) {
            const slot_off: u15 = @intCast(spill_base + alloc.spill.get(p).? * 16);
            if (l.idx < 8) {
                const incoming: Reg = @enumFromInt(@as(u5, @intCast(l.idx)));
                if (l.class == .fpr) {
                    try code.append(allocator, if (isVector(func, p)) encode.strQ(incoming, sp, slot_off) else encode.strFp(incoming, sp, slot_off, true));
                } else {
                    try code.append(allocator, encode.strOff(incoming, sp, slot_off));
                }
            } else {
                // An incoming stack parameter that is also spilled: load from the caller's
                // outgoing area into a scratch, then store to our spill slot.
                if (l.class == .fpr) {
                    try code.append(allocator, encode.ldrFp(fp_spill_op[0], sp, @intCast(frame + 8 * (l.idx - 8)), false));
                    try code.append(allocator, encode.strFp(fp_spill_op[0], sp, slot_off, true));
                } else {
                    try code.append(allocator, encode.ldrOff(spill_op[0], sp, @intCast(frame + 8 * (l.idx - 8))));
                    try code.append(allocator, encode.strOff(spill_op[0], sp, slot_off));
                }
            }
            continue;
        }
        const pr = alloc.reg.get(p).?;
        if (l.idx < 8) {
            if (!leaf) {
                const incoming: Reg = @enumFromInt(@as(u5, @intCast(l.idx)));
                try code.append(allocator, if (l.class == .fpr) encode.fmovReg(pr, incoming) else encode.mov(pr, incoming));
            }
        } else if (l.class == .fpr) {
            // A floating-point stack parameter (the 9th+ FP arg): the caller placed it in
            // its outgoing-argument area, now just below our frame. Load it into the
            // parameter's FP register. (A graphics shader with many scalarized varyings +
            // synthesized derivative gradient inputs can exceed the 8 FP arg registers.)
            try code.append(allocator, encode.ldrFp(pr, sp, @intCast(frame + 8 * (l.idx - 8)), false));
        } else {
            try code.append(allocator, encode.ldrOff(pr, sp, @intCast(frame + 8 * (l.idx - 8))));
        }
    }

    const ctx = Ctx{ .func = func, .alloc = &alloc, .spill_base = spill_base, .alloca_base = alloca_base, .alloca_off = &alloca_off, .fp16 = fp16 };

    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        if (fetch_align > 4 and is_loop_header[bi]) {
            var pad = alignPadWords(code.items.len, fetch_align);
            while (pad > 0) : (pad -= 1) try code.append(allocator, encode.nop());
        }
        block_start[bi] = code.items.len;
        var terminated = false;

        const insts = func.blockInsts(block);
        for (insts, 0..) |inst, inst_idx| {
            // Record a source-line row when this instruction begins a new line (its
            // `debug.line` attribute differs from the previous instruction's).
            if (lineOf(func, inst)) |line| {
                if (line != last_line) {
                    try lines.append(allocator, .{ .offset = @intCast(code.items.len * 4), .line = line });
                    last_line = line;
                }
            }
            switch (func.opcode(inst)) {
                .iconst => |c| {
                    const result = func.instResult(inst).?;
                    const rd = resultReg(&alloc, func, result);
                    if (regClass(func, result) == .fpr) {
                        // A float-typed integer constant (the frontend zero-inits float
                        // locals this way). The result lives in an fp register, so the
                        // bits must go there via a GPR scratch, never a plain integer
                        // move into the fp register's number.
                        const d = isDouble(func, result);
                        if (d) {
                            try loadConst64(allocator, &code, scratch_imm, @bitCast(c));
                        } else {
                            const bits: u32 = @truncate(@as(u64, @bitCast(c)));
                            try loadConst(allocator, &code, scratch_imm, @intCast(bits));
                        }
                        try code.append(allocator, encode.fmovFromGpr(rd, scratch_imm, d));
                    } else if (isWide(func, result)) {
                        try loadConst64(allocator, &code, rd, @bitCast(c)); // full 64 bits (i64/ptr)
                    } else {
                        try loadConst(allocator, &code, rd, c);
                    }
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .fconst => |val| {
                    const result = func.instResult(inst).?;
                    const rd = resultReg(&alloc, func, result);
                    const d = isDouble(func, result);
                    if (fp16 and isHalf(func, result)) {
                        // Native (fp16): materialize the raw 16-bit IEEE-half pattern of the
                        // half-rounded value in a GPR, then `fmov h, w` into the H register. The
                        // half lives natively (no f32 widening), so no `fcvt` is needed.
                        const h: f16 = @floatCast(val);
                        const bits: u16 = @bitCast(h);
                        try loadConst(allocator, &code, scratch_imm, @intCast(bits));
                        try code.append(allocator, encode.fmovHfromGpr(rd, scratch_imm));
                    } else {
                        if (d) {
                            try loadConst64(allocator, &code, scratch_imm, @bitCast(val));
                        } else {
                            // In the emulation path an f16 constant is materialized as its f32 widening (round
                            // the value to half first, `@as(f32, @as(f16, val))`), keeping the
                            // invariant that an f16 in a register is its exact-half f32 form. f32
                            // keeps full value.
                            const rounded: f32 = if (isHalf(func, result)) @as(f16, @floatCast(val)) else @floatCast(val);
                            const bits: u32 = @bitCast(rounded);
                            try loadConst(allocator, &code, scratch_imm, @intCast(bits));
                        }
                        try code.append(allocator, encode.fmovFromGpr(rd, scratch_imm, d));
                    }
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .arith => |a| {
                    // Fused multiply-add/sub: when this is a float `mul` that is the
                    // single-use, immediately-preceding operand of the next add/sub, skip
                    // its materialization entirely. `emitFusedArith` re-checks the SAME
                    // predicate and emits the fused fmadd/fmsub/fnmsub on these operands,
                    // so the multiply is emitted exactly once (mirrors the icmp/if fusion
                    // above).
                    if (a.op == .mul and fusesIntoNextArith(func, insts, inst_idx)) continue;
                    const result = func.instResult(inst).?;
                    if ((a.op == .add or a.op == .sub) and inst_idx >= 1 and fusesIntoNextArith(func, insts, inst_idx - 1)) {
                        const mul = func.opcode(insts[inst_idx - 1]).arith;
                        const mul_result = func.instResult(insts[inst_idx - 1]).?;
                        try ctx.emitFusedArith(allocator, &code, result, a.op, a.lhs, a.rhs, mul, mul_result);
                        continue;
                    }
                    try ctx.binary(allocator, &code, result, a.op, a.lhs, a.rhs);
                },
                .arith_imm => |a| {
                    const result = func.instResult(inst).?;
                    const rl = try ctx.loadOp(allocator, &code, a.lhs, spill_op[0]);
                    try loadConst(allocator, &code, spill_op[1], a.imm); // imm in x14, x16 stays free for rem
                    const rd = resultReg(&alloc, func, result);
                    try emitBinary(allocator, &code, a.op, rd, rl, spill_op[1], isSignedInt(func, a.lhs), isWide(func, result));
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .icmp => |cmp| {
                    // Fused compare-and-branch: when this integer icmp is the single-use
                    // condition of the immediately-following if, skip its `cmp; cset`
                    // materialization entirely. `emitIf` re-checks the SAME predicate and
                    // emits the fused `cmp; b.cc` on these operands, so the compare is
                    // emitted exactly once.
                    if (fusesIntoNextIf(func, insts, inst_idx)) continue;
                    const result = func.instResult(inst).?;
                    if (isVector(func, cmp.lhs)) {
                        // Vectorized compare (widened FS): produce a per-lane MASK
                        // (all-ones where the relation holds, else all-zeros) in a
                        // <4 x f32>-typed result. <, <= are >, >= with operands swapped.
                        // != is the bitwise complement of ==.
                        const rl = try ctx.loadOp(allocator, &code, cmp.lhs, fp_spill_op[0]);
                        const rr = try ctx.loadOp(allocator, &code, cmp.rhs, fp_spill_op[1]);
                        const rd = resultReg(&alloc, func, result);
                        switch (cmp.op) {
                            .eq => try code.append(allocator, encode.fcmeqVec(rd, rl, rr)),
                            .ne => {
                                try code.append(allocator, encode.fcmeqVec(rd, rl, rr));
                                try code.append(allocator, encode.mvnVec(rd, rd));
                            },
                            .gt => try code.append(allocator, encode.fcmgtVec(rd, rl, rr)),
                            .ge => try code.append(allocator, encode.fcmgeVec(rd, rl, rr)),
                            .lt => try code.append(allocator, encode.fcmgtVec(rd, rr, rl)), // rl < rr  ==  rr > rl
                            .le => try code.append(allocator, encode.fcmgeVec(rd, rr, rl)), // rl <= rr ==  rr >= rl
                        }
                        try storeResult(allocator, &code, ctx, result, rd);
                        continue;
                    }
                    if (regClass(func, cmp.lhs) == .fpr) {
                        const rl = try ctx.loadOp(allocator, &code, cmp.lhs, fp_spill_op[0]);
                        const rr = try ctx.loadOp(allocator, &code, cmp.rhs, fp_spill_op[1]);
                        const rd = resultReg(&alloc, func, result);
                        // Native f16 (fp16) compares in the H form; emulation compares the S-held
                        // f32 widening (fkindOf yields `.single`, byte-identical to before).
                        try code.append(allocator, encode.fcmp(rl, rr, fkindOf(func, cmp.lhs, fp16)));
                        try code.append(allocator, encode.cset(rd, condForFloat(cmp.op)));
                        try storeResult(allocator, &code, ctx, result, rd);
                    } else {
                        const rl = try ctx.loadOp(allocator, &code, cmp.lhs, spill_op[0]);
                        const rr = try ctx.loadOp(allocator, &code, cmp.rhs, spill_op[1]);
                        const rd = resultReg(&alloc, func, result);
                        try code.append(allocator, encode.cmp(rl, rr));
                        try code.append(allocator, encode.cset(rd, condFor(cmp.op, isSignedInt(func, cmp.lhs))));
                        try storeResult(allocator, &code, ctx, result, rd);
                    }
                },
                .select => |s| {
                    const result = func.instResult(inst).?;
                    if (isVector(func, result)) {
                        // Vectorized masked blend (widened FS): cond is a per-lane mask
                        // (<4 x f32>, all-ones/all-zeros), then/else are <4 x f32>. NEON
                        // `bsl` selects then's lane where the mask is set, else's where
                        // clear. bsl reads+writes its destination (the mask), so copy the
                        // mask into the result register first, then bsl with then/else.
                        const cm = try ctx.loadOp(allocator, &code, s.cond, fp_spill_op[0]);
                        const tr = try ctx.loadOp(allocator, &code, s.then, fp_spill_op[1]);
                        const el = try ctx.loadOp(allocator, &code, s.@"else", fp_spill_res);
                        // Accumulate the blend into a fixed scratch (fp_move = v27, outside
                        // every allocation pool) so it never aliases then/else/the result.
                        try code.append(allocator, encode.movVec(fp_move, cm));
                        try code.append(allocator, encode.bslVec(fp_move, tr, el));
                        const rd = resultReg(&alloc, func, result);
                        if (@intFromEnum(rd) != @intFromEnum(fp_move)) try code.append(allocator, encode.movVec(rd, fp_move));
                        try storeResult(allocator, &code, ctx, result, rd);
                        continue;
                    }
                    const c = try ctx.loadOp(allocator, &code, s.cond, spill_op[0]); // cond is a gpr bool
                    if (regClass(func, result) == .fpr) {
                        const tr = try ctx.loadOp(allocator, &code, s.then, fp_spill_op[0]);
                        const el = try ctx.loadOp(allocator, &code, s.@"else", fp_spill_op[1]);
                        const rd = resultReg(&alloc, func, result);
                        try code.append(allocator, encode.cmp(c, .zr));
                        // Native f16 (fp16) selects in the H form; emulation selects the S-held
                        // f32 widening (fkindOf yields `.single`, byte-identical to before).
                        try code.append(allocator, encode.fcsel(rd, tr, el, .ne, fkindOf(func, result, fp16)));
                        try storeResult(allocator, &code, ctx, result, rd);
                    } else {
                        const tr = try ctx.loadOp(allocator, &code, s.then, spill_op[1]);
                        const el = try ctx.loadOp(allocator, &code, s.@"else", scratch_imm);
                        const rd = resultReg(&alloc, func, result);
                        try code.append(allocator, encode.cmp(c, .zr));
                        try code.append(allocator, encode.csel(rd, tr, el, .ne));
                        try storeResult(allocator, &code, ctx, result, rd);
                    }
                },
                .convert => |cv| {
                    const result = func.instResult(inst).?;
                    const sc = regClass(func, cv.value);
                    const dc = regClass(func, result);
                    const src = try ctx.loadOp(allocator, &code, cv.value, if (sc == .fpr) fp_spill_op[0] else spill_op[0]);
                    const rd = resultReg(&alloc, func, result);
                    if (sc == .gpr and dc == .fpr) {
                        // int -> float: scvtf/ucvtf. NATIVE (fp16) converts straight into the H
                        // view (single-rounded, no fixup). EMULATION lands an int->f16 in the S
                        // view first (isDouble(f16) is false), then rounds to nearest-even half so
                        // the S reg holds an exact half; fkindOf yields `.single` there, keeping
                        // the emulation encoding byte-identical.
                        try code.append(allocator, encode.cvtIntToFloat(rd, src, fkindOf(func, result, fp16), isSignedInt(func, cv.value)));
                        if (isHalf(func, result) and !fp16) try roundToHalf(allocator, &code, rd);
                    } else if (sc == .fpr and dc == .gpr) {
                        // float -> int: fcvtzs/fcvtzu (round toward zero). NATIVE (fp16) reads the
                        // f16 straight from the H view. EMULATION reads the exact f32 widening in
                        // the S view (fkindOf yields `.single`, byte-identical to before).
                        try code.append(allocator, encode.cvtFloatToInt(rd, src, fkindOf(func, cv.value, fp16), isSignedInt(func, result)));
                    } else if (sc == .fpr and dc == .fpr) {
                        const src_half = isHalf(func, cv.value);
                        const dst_half = isHalf(func, result);
                        const src_d = isDouble(func, cv.value);
                        const dst_d = isDouble(func, result);
                        if (fp16 and (src_half or dst_half)) {
                            // In the native path an f16 lives in an H register (not an S-held f32 widening),
                            // so convert directly between the H view and s/d with a SINGLE fcvt
                            // (or a plain copy for f16 -> f16), never re-materializing an S form.
                            if (dst_half and src_half) {
                                try code.append(allocator, encode.fmovReg(rd, src)); // f16 -> f16: copy the H view
                            } else if (dst_half) {
                                // f32/f64 -> f16: one round to native half (fcvt h,d rounds once
                                // directly from double, avoiding a double-rounding d->s->h path).
                                try code.append(allocator, if (src_d) encode.fcvtHfromD(rd, src) else encode.fcvtHfromS(rd, src));
                            } else if (dst_d) {
                                try code.append(allocator, encode.fcvtDfromH(rd, src)); // f16 -> f64 (exact widen)
                            } else {
                                try code.append(allocator, encode.fcvtSfromH(rd, src)); // f16 -> f32 (exact widen)
                            }
                        } else if (isHalf(func, result)) {
                            // Emulation -> f16. The old 2-way isDouble split (`sd == dd -> fmov`)
                            // is WRONG for f16: f32->f16 has isDouble false on both sides and would
                            // emit a bare `fmov` with no rounding. Narrow to nearest-even half (a
                            // single round from the source width) then widen back to the S-held
                            // representation. fcvt h,d rounds once directly from double.
                            try code.append(allocator, if (src_d) encode.fcvtHfromD(rd, src) else encode.fcvtHfromS(rd, src));
                            try code.append(allocator, encode.fcvtSfromH(rd, rd));
                        } else if (src_d == dst_d) {
                            // Same register view, dest not half: a plain copy. Covers f32->f32,
                            // f64->f64, and (emulation) f16->f32 (the S reg already holds the exact
                            // half value as its f32 widening, so widening to f32 is the identity).
                            try code.append(allocator, encode.fmovReg(rd, src));
                        } else {
                            // Different views, dest not half: the base single<->double convert,
                            // byte-identical to the pre-f16 behavior for f32<->f64, and also the
                            // exact (emulation) f16->f64 widen (the S-held half widens to double).
                            // dst_d picks the direction.
                            try code.append(allocator, encode.fcvt(rd, src, dst_d));
                        }
                    } else {
                        // int <-> int. Widening sign/zero-extends by the SOURCE signedness (sbfm for a
                        // signed source, ubfm for unsigned); same-width or narrowing keeps the low
                        // bits, byte-identical to the previous unconditional mov for those cases.
                        const src_bits = intBitsOf(func, cv.value);
                        const dst_bits = intBitsOf(func, result);
                        if (dst_bits > src_bits and src_bits < 64) {
                            const imms: u6 = @intCast(src_bits - 1);
                            try code.append(allocator, if (isSignedInt(func, cv.value))
                                encode.sbfm(rd, src, 0, imms)
                            else
                                encode.ubfm(rd, src, 0, imms));
                        } else {
                            try code.append(allocator, encode.mov(rd, src)); // same width / narrowing
                        }
                    }
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .unary => |u| if (isVector(func, func.instResult(inst).?)) {
                    // Vectorized unary (a widened FS lane-op over <4 x f32>): NEON sqrt.
                    const result = func.instResult(inst).?;
                    const src = try ctx.loadOp(allocator, &code, u.value, fp_spill_op[0]);
                    const rd = resultReg(&alloc, func, result);
                    try code.append(allocator, switch (u.op) {
                        .sqrt => encode.fsqrtVec(rd, src),
                        else => return error.Unsupported,
                    });
                    try storeResult(allocator, &code, ctx, result, rd);
                } else {
                    const result = func.instResult(inst).?;
                    const sc = regClass(func, u.value);
                    const dc = regClass(func, result);
                    const src = try ctx.loadOp(allocator, &code, u.value, if (sc == .fpr) fp_spill_op[0] else spill_op[0]);
                    const rd = resultReg(&alloc, func, result);
                    if (u.op == .reinterpret) {
                        if (sc == .gpr and dc == .fpr) {
                            try code.append(allocator, encode.fmovFromGpr(rd, src, isDouble(func, result)));
                        } else if (sc == .fpr and dc == .gpr) {
                            try code.append(allocator, encode.fmovToGpr(rd, src, isDouble(func, u.value)));
                        } else {
                            try code.append(allocator, encode.mov(rd, src));
                        }
                    } else {
                        // f16 float-math unary (sqrt/ceil/floor/trunc/nearest) is not lowered on
                        // either f16 path: the emulation holds f16 as an f32 (an fsqrt.s would leave
                        // an un-narrowed f32), and the native path holds it in an H register (an
                        // fsqrt.s would mis-read the low 32 bits). No front-end emits it today; reject
                        // cleanly rather than silently miscompile (mirrors the wasm backend).
                        if (isHalf(func, result)) return error.Unsupported;
                        const d = isDouble(func, result);
                        try code.append(allocator, switch (u.op) {
                            .sqrt => encode.fsqrt(rd, src, d),
                            .ceil => encode.frintp(rd, src, d),
                            .floor => encode.frintm(rd, src, d),
                            .trunc => encode.frintz(rd, src, d),
                            .nearest => encode.frintn(rd, src, d),
                            .reinterpret => unreachable,
                        });
                    }
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .alloca => {
                    const result = func.instResult(inst).?;
                    const off = alloca_base + alloca_off.get(result).?;
                    const rd = resultReg(&alloc, func, result);
                    try emitFrameImm(allocator, &code, false, rd, sp, off);
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .load => |l| {
                    const result = func.instResult(inst).?;
                    const base = try ctx.loadOp(allocator, &code, l.ptr, spill_op[0]); // ptr is a gpr
                    const rd = resultReg(&alloc, func, result);
                    try emitLoad(allocator, &code, func, result, rd, base, fp16);
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .store => |st| {
                    const val = try ctx.loadOp(allocator, &code, st.value, if (regClass(func, st.value) == .fpr) fp_spill_op[0] else spill_op[0]);
                    const base = try ctx.loadOp(allocator, &code, st.ptr, spill_op[1]);
                    try emitStore(allocator, &code, func, st.value, val, base, fp16);
                },
                .prefetch => |pf| {
                    // A software prefetch hint: bring [ptr] into L1, no result, no
                    // observable effect on the function.
                    const base = try ctx.loadOp(allocator, &code, pf.ptr, spill_op[0]);
                    try code.append(allocator, encode.prfm(base));
                },
                .extract => |e| {
                    // Extract a SIMD lane to a scalar (the vectorizer's unpack): one NEON
                    // `dup`, pure, so dead extracts fall to DCE.
                    if (!isVector(func, e.aggregate)) return error.Unsupported;
                    const result = func.instResult(inst).?;
                    const src = try ctx.loadOp(allocator, &code, e.aggregate, fp_spill_op[0]);
                    const rd = resultReg(&alloc, func, result);
                    try code.append(allocator, encode.dupLane(rd, src, @intCast(e.index)));
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .struct_new => |sn| {
                    // Build a SIMD vector from scalar lanes (the vectorizer's pack): one NEON
                    // `ins` per lane, lane 0 last so a field the allocator placed in the result
                    // register keeps its value (in lane 0) until its own `ins` reads it.
                    const result = func.instResult(inst).?;
                    if (!isVector(func, result)) return error.Unsupported;
                    const fields = func.valueList(sn.fields);
                    if (fields.len != 4) return error.Unsupported; // <4 x f32> only for now
                    const rd = resultReg(&alloc, func, result);
                    for ([_]u2{ 1, 2, 3, 0 }) |lane| {
                        const fr = try ctx.loadOp(allocator, &code, fields[lane], fp_spill_op[0]);
                        try code.append(allocator, encode.insLane(rd, lane, fr));
                    }
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .call => |c| {
                    const args = func.valueList(c.args);
                    const locs = try allocator.alloc(ArgLoc, args.len);
                    defer allocator.free(locs);
                    computeArgLocs(func, args, locs);
                    for (args, locs) |arg, loc| {
                        const target: Reg = @enumFromInt(@as(u5, @intCast(loc.idx)));
                        if (loc.class == .fpr) {
                            const src = try ctx.loadOp(allocator, &code, arg, fp_spill_op[0]);
                            if (loc.idx >= 8) return error.Unsupported; // fp stack args not handled
                            try code.append(allocator, encode.fmovReg(target, src));
                        } else {
                            const src = try ctx.loadOp(allocator, &code, arg, spill_op[0]);
                            if (loc.idx < 8) {
                                try code.append(allocator, encode.mov(target, src));
                            } else {
                                try code.append(allocator, encode.strOff(src, sp, @intCast(8 * (loc.idx - 8))));
                            }
                        }
                    }
                    try relocs.append(allocator, .{ .offset = code.items.len, .symbol = func.symbolName(c.symbol) });
                    try code.append(allocator, encode.bl(0));
                    if (func.instResult(inst)) |result| {
                        const rd = resultReg(&alloc, func, result);
                        try code.append(allocator, if (regClass(func, result) == .fpr) encode.fmovReg(rd, @enumFromInt(0)) else encode.mov(rd, .x0));
                        try storeResult(allocator, &code, ctx, result, rd);
                    }
                },
                .call_indirect => |c| {
                    // Stage the target address into a scratch register that survives the
                    // argument moves (x0..x7), then `blr`.
                    const tsrc = try ctx.loadOp(allocator, &code, c.target, spill_op[0]);
                    try code.append(allocator, encode.mov(scratch_imm, tsrc));
                    const args = func.valueList(c.args);
                    const locs = try allocator.alloc(ArgLoc, args.len);
                    defer allocator.free(locs);
                    computeArgLocs(func, args, locs);
                    for (args, locs) |arg, loc| {
                        const target: Reg = @enumFromInt(@as(u5, @intCast(loc.idx)));
                        if (loc.class == .fpr) {
                            const src = try ctx.loadOp(allocator, &code, arg, fp_spill_op[0]);
                            if (loc.idx >= 8) return error.Unsupported;
                            try code.append(allocator, encode.fmovReg(target, src));
                        } else {
                            const src = try ctx.loadOp(allocator, &code, arg, spill_op[0]);
                            if (loc.idx < 8) {
                                try code.append(allocator, encode.mov(target, src));
                            } else {
                                try code.append(allocator, encode.strOff(src, sp, @intCast(8 * (loc.idx - 8))));
                            }
                        }
                    }
                    try code.append(allocator, encode.blr(scratch_imm));
                    if (func.instResult(inst)) |result| {
                        const rd = resultReg(&alloc, func, result);
                        try code.append(allocator, if (regClass(func, result) == .fpr) encode.fmovReg(rd, @enumFromInt(0)) else encode.mov(rd, .x0));
                        try storeResult(allocator, &code, ctx, result, rd);
                    }
                },
                .@"if" => |cf| {
                    try ctx.emitIf(allocator, &code, &fixups, cf, insts, inst_idx);
                    terminated = true;
                },
                .dot => |d| {
                    // SDOT/UDOT ACCUMULATE into Vd, so `acc` must be resident in the
                    // destination before the dot executes. Mirrors the vectorized
                    // select's blend below: accumulate into the fixed scratch `fp_move`
                    // (v27, outside every allocation pool) first, so it never aliases
                    // the acc/a/b operand registers (a naive `movVec(rd, acc)` would
                    // clobber `a` or `b` if the allocator reused either's register as
                    // `rd`), then move into the result register only if they differ.
                    const result = func.instResult(inst).?;
                    const racc = try ctx.loadOp(allocator, &code, d.acc, fp_spill_op[0]);
                    const ra = try ctx.loadOp(allocator, &code, d.a, fp_spill_op[1]);
                    const rb = try ctx.loadOp(allocator, &code, d.b, fp_spill_res);
                    try code.append(allocator, encode.movVec(fp_move, racc));
                    try code.append(allocator, if (dotSigned(func, d.a)) encode.sdot(fp_move, ra, rb) else encode.udot(fp_move, ra, rb));
                    const rd = resultReg(&alloc, func, result);
                    if (@intFromEnum(rd) != @intFromEnum(fp_move)) try code.append(allocator, encode.movVec(rd, fp_move));
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                else => return error.Unsupported,
            }
        }

        if (!terminated) {
            switch (func.terminator(block) orelse Terminator{ .ret = null }) {
                .ret => |v| {
                    if (v) |value| {
                        if (regClass(func, value) == .fpr) {
                            const vec = isVector(func, value);
                            if (alloc.spill.get(value)) |slot| {
                                const off: u15 = @intCast(spill_base + slot * 16);
                                try code.append(allocator, if (vec) encode.ldrQ(@enumFromInt(0), sp, off) else encode.ldrFp(@enumFromInt(0), sp, off, true));
                            } else {
                                const src = alloc.reg.get(value).?;
                                if (@intFromEnum(src) != 0) try code.append(allocator, if (vec) encode.movVec(@enumFromInt(0), src) else encode.fmovReg(@enumFromInt(0), src));
                            }
                        } else if (alloc.spill.get(value)) |slot| {
                            try code.append(allocator, encode.ldrOff(.x0, sp, @intCast(spill_base + slot * 16)));
                        } else {
                            const src = alloc.reg.get(value).?;
                            if (src != .x0) try code.append(allocator, encode.mov(.x0, src));
                        }
                    }
                    if (!leaf) {
                        for (alloc.saved_gpr.items, 0..) |r, i| try code.append(allocator, encode.ldrOff(r, sp, @intCast(gpr_saved_base + 8 * i)));
                        for (alloc.saved_fpr.items, 0..) |r, i| try code.append(allocator, encode.ldrFp(r, sp, @intCast(fpr_saved_base + 8 * i), true));
                        try code.append(allocator, encode.ldrOff(.x30, sp, @intCast(lr_off)));
                    }
                    if (frame > 0) try emitFrameImm(allocator, &code, false, sp, sp, frame);
                    try code.append(allocator, encode.ret());
                },
                .jump => |j| try ctx.emitJump(allocator, &code, &fixups, j),
            }
        }
    }

    for (fixups.items) |f| {
        const off: i28 = @intCast((@as(i64, @intCast(block_start[f.target])) - @as(i64, @intCast(f.at))) * 4);
        code.items[f.at] = encode.b(off);
    }

    return .{ .code = try code.toOwnedSlice(allocator), .relocs = try relocs.toOwnedSlice(allocator), .lines = try lines.toOwnedSlice(allocator) };
}

const Terminator = ir.function.Terminator;

/// Per-function lowering context. Gives the spill-aware operand/result helpers
/// access to the allocation and frame offsets.
const Ctx = struct {
    func: *const Function,
    alloc: *Allocation,
    spill_base: usize,
    alloca_base: usize,
    alloca_off: *std.AutoHashMapUnmanaged(Value, u32),
    /// Whether to lower f16 with NATIVE FEAT_FP16 H-form ops instead of the emulation. Threaded
    /// from `compileFunction`'s `caps.fp16`; false for every non-model caller (byte-identical).
    fp16: bool = false,

    /// The register holding `v`: its assigned register, or reload a spilled value into
    /// `scratch` (a register of `v`'s class). A SIMD vector reloads all 128 bits.
    fn loadOp(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), v: Value, scratch: Reg) Error!Reg {
        if (self.alloc.reg.get(v)) |r| return r;
        const off: u15 = @intCast(self.spill_base + self.alloc.spill.get(v).? * 16);
        if (isVector(self.func, v)) {
            try code.append(allocator, encode.ldrQ(scratch, sp, off));
        } else if (regClass(self.func, v) == .fpr) {
            try code.append(allocator, encode.ldrFp(scratch, sp, off, true));
        } else {
            try code.append(allocator, encode.ldrOff(scratch, sp, off));
        }
        return scratch;
    }

    /// Emit the fused multiply-add/sub for a float `add`/`sub` (`op`, `lhs`, `rhs`) whose
    /// single-use, immediately-preceding operand is the float `mul` (`mul`, defining
    /// `mul_result`) - see `fusesIntoNextArith` for the shared eligibility check. Loads the
    /// mul's own operands directly (its materialization was skipped by the caller) plus the
    /// add/sub's OTHER operand (the accumulator).
    ///
    /// Scalar picks the A64 3-source variant whose hardware semantics matches the IR shape
    /// (confirmed against @mulAdd and by executing each variant on this aarch64 host, not
    /// assumed from the ARM ARM alone):
    ///   add(mul(a,b), c) = a*b+c  -> fmadd:  Rd = Ra + Rn*Rm
    ///   sub(mul(a,b), c) = a*b-c  -> fnmsub: Rd = Rn*Rm - Ra
    ///   sub(c, mul(a,b)) = c-a*b  -> fmsub:  Rd = Ra - Rn*Rm
    ///
    /// Vector NEON FMLA/FMLS ACCUMULATE into their destination register (Vd = Vd (+/-)
    /// Vn*Vm), so `c` must be resident in the destination before either executes. Mirrors
    /// the `dot` lowering: move `c` into the fixed scratch `fp_move` (v27, outside every
    /// allocation pool) first, so it never aliases `a`/`b` (a naive `movVec(rd, c)` would
    /// clobber `a` or `b` if the allocator reused either's register as `rd`), then move
    /// the scratch into the result register only if they differ.
    ///   add(mul(a,b), c) = a*b+c -> fmla: Vd = Vd + Vn*Vm, Vd preloaded with c
    ///   sub(c, mul(a,b)) = c-a*b -> fmls: Vd = Vd - Vn*Vm, Vd preloaded with c
    /// (`fusesIntoNextArith` never fuses the third vector shape, sub(mul,c) = a*b-c, since
    /// no single NEON instruction expresses it - that shape stays on the separate
    /// fmul+fsub path in `binary`.)
    ///
    /// Nothing runs between the (skipped) mul and this add/sub, so the mul's operand
    /// registers still hold their values here exactly as `fusesIntoNextIf` relies on for
    /// the fused compare-and-branch.
    fn emitFusedArith(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), result: Value, op: ir.function.BinOp, lhs: Value, rhs: Value, mul: ir.function.Arith, mul_result: Value) Error!void {
        const ra_val = if (lhs == mul_result) rhs else lhs; // the add/sub's non-mul operand
        const a = try self.loadOp(allocator, code, mul.lhs, fp_spill_op[0]);
        const b = try self.loadOp(allocator, code, mul.rhs, fp_spill_op[1]);
        const c = try self.loadOp(allocator, code, ra_val, fp_spill_res);
        const rd = resultReg(self.alloc, self.func, result);
        if (isVector(self.func, result)) {
            // `op == .sub` here only ever means `sub(c, mul)` (`fusesIntoNextArith` rejects
            // `sub(mul, c)` for a vector), so fmls's `Vd - Vn*Vm` is always the right shape.
            try code.append(allocator, encode.movVec(fp_move, c));
            try code.append(allocator, switch (op) {
                .add => encode.fmlaVec(fp_move, a, b),
                .sub => encode.fmlsVec(fp_move, a, b),
                else => unreachable, // the caller only routes .add/.sub here (see the switch above)
            });
            if (@intFromEnum(rd) != @intFromEnum(fp_move)) try code.append(allocator, encode.movVec(rd, fp_move));
            try storeResult(allocator, code, self, result, rd);
            return;
        }
        const d = isDouble(self.func, result);
        try code.append(allocator, switch (op) {
            .add => encode.fmadd(rd, a, b, c, d),
            .sub => if (lhs == mul_result) encode.fnmsub(rd, a, b, c, d) else encode.fmsub(rd, a, b, c, d),
            else => unreachable, // the caller only routes .add/.sub here (see the switch above)
        });
        try storeResult(allocator, code, self, result, rd);
    }

    fn binary(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), result: Value, op: ir.function.BinOp, lhs: Value, rhs: Value) Error!void {
        if (isVector(self.func, result)) {
            // NEON lane-wise arithmetic over a packed vector (currently <4 x f32> only).
            const rl = try self.loadOp(allocator, code, lhs, fp_spill_op[0]);
            const rr = try self.loadOp(allocator, code, rhs, fp_spill_op[1]);
            const rd = resultReg(self.alloc, self.func, result);
            try code.append(allocator, switch (op) {
                .add => encode.faddVec(rd, rl, rr),
                .sub => encode.fsubVec(rd, rl, rr),
                .mul => encode.fmulVec(rd, rl, rr),
                .div => encode.fdivVec(rd, rl, rr),
                else => return error.Unsupported,
            });
            try storeResult(allocator, code, self, result, rd);
            return;
        }
        if (regClass(self.func, result) == .fpr) {
            const rl = try self.loadOp(allocator, code, lhs, fp_spill_op[0]);
            const rr = try self.loadOp(allocator, code, rhs, fp_spill_op[1]);
            const rd = resultReg(self.alloc, self.func, result);
            const kind = fkindOf(self.func, result, self.fp16);
            try code.append(allocator, switch (op) {
                .add => encode.fadd(rd, rl, rr, kind),
                .sub => encode.fsub(rd, rl, rr, kind),
                .mul => encode.fmul(rd, rl, rr, kind),
                .div => encode.fdiv(rd, rl, rr, kind),
                else => return error.Unsupported,
            });
            // In the emulation path an f16 op is done in the S (f32) form, then its result is rounded to
            // nearest-even half so the S register again holds an exact half value (correct per-op
            // IEEE f16 semantics; the operands were already exact halves). fp mul/add fusion is
            // disabled for f16 in `fusesIntoNextArith`, so this rounds every op individually.
            // Native (fp16): the H-form op is already single-rounded to half, so no re-rounding.
            if (isHalf(self.func, result) and !self.fp16) try roundToHalf(allocator, code, rd);
            try storeResult(allocator, code, self, result, rd);
        } else {
            const rl = try self.loadOp(allocator, code, lhs, spill_op[0]);
            const rr = try self.loadOp(allocator, code, rhs, spill_op[1]);
            const rd = resultReg(self.alloc, self.func, result);
            try emitBinary(allocator, code, op, rd, rl, rr, isSignedInt(self.func, lhs), isWide(self.func, result));
            try storeResult(allocator, code, self, result, rd);
        }
    }

    fn emitIf(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), fixups: *std.ArrayList(Fixup), cf: ir.function.If, insts: []const ir.function.Inst, if_idx: usize) Error!void {
        // Fused compare-and-branch: when the immediately-preceding instruction is a
        // single-use integer icmp that is exactly this if's condition (the SAME predicate
        // the icmp case used to skip its materialization), load the icmp's operands, set
        // the flags with `cmp`, and branch to the then-edge with `b.cc` on the icmp's
        // condition instead of materializing a boolean and re-testing it with `cbnz`. The
        // edge-move / fixup structure is identical to the plain path, only the branch and
        // its operand load differ. condFor here mirrors the icmp lowering's cset condition,
        // so the fused branch takes the then-edge under exactly the same condition.
        if (if_idx >= 1 and fusesIntoNextIf(self.func, insts, if_idx - 1)) {
            const cmp = self.func.opcode(insts[if_idx - 1]).icmp;
            const rl = try self.loadOp(allocator, code, cmp.lhs, spill_op[0]);
            const rr = try self.loadOp(allocator, code, cmp.rhs, spill_op[1]);
            try code.append(allocator, encode.cmp(rl, rr));
            const cond = condFor(cmp.op, isSignedInt(self.func, cmp.lhs));
            const bcc_at = code.items.len;
            try code.append(allocator, encode.bcc(cond, 0));
            try self.emitMoves(allocator, code, cf.@"else");
            try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(cf.@"else".target) });
            try code.append(allocator, encode.b(0));
            const then_at = code.items.len;
            code.items[bcc_at] = encode.bcc(cond, @intCast((@as(i64, @intCast(then_at)) - @as(i64, @intCast(bcc_at))) * 4));
            try self.emitMoves(allocator, code, cf.then);
            try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(cf.then.target) });
            try code.append(allocator, encode.b(0));
            return;
        }
        const cond = try self.loadOp(allocator, code, cf.cond, spill_op[0]);
        const cbnz_at = code.items.len;
        try code.append(allocator, encode.cbnz(cond, 0));
        try self.emitMoves(allocator, code, cf.@"else");
        try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(cf.@"else".target) });
        try code.append(allocator, encode.b(0));
        const then_at = code.items.len;
        code.items[cbnz_at] = encode.cbnz(cond, @intCast((@as(i64, @intCast(then_at)) - @as(i64, @intCast(cbnz_at))) * 4));
        try self.emitMoves(allocator, code, cf.then);
        try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(cf.then.target) });
        try code.append(allocator, encode.b(0));
    }

    fn emitJump(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), fixups: *std.ArrayList(Fixup), jump: ir.function.Jump) Error!void {
        try self.emitMoves(allocator, code, jump);
        try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(jump.target) });
        try code.append(allocator, encode.b(0));
    }

    /// Edge moves into the target's parameters (which are always in registers).
    /// GPR and FP register-resident arguments go through separate parallel moves.
    /// spilled arguments are reloaded straight into their parameter register.
    fn emitMoves(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), jump: ir.function.Jump) Error!void {
        const args = self.func.blockArgs(jump);
        const params = self.func.blockParams(jump.target);
        if (args.len != params.len) return error.Unsupported;

        var gpr_moves: std.ArrayList(Move) = .empty;
        defer gpr_moves.deinit(allocator);
        var fpr_moves: std.ArrayList(Move) = .empty;
        defer fpr_moves.deinit(allocator);
        for (args, params) |arg, param| {
            const dst = self.alloc.reg.get(param).?; // params are never spilled
            if (self.alloc.reg.get(arg)) |src| {
                const m = Move{ .src = src, .dst = dst };
                if (regClass(self.func, param) == .fpr) try fpr_moves.append(allocator, m) else try gpr_moves.append(allocator, m);
            }
        }
        try parallelMove(allocator, code, gpr_moves.items, encode.mov, scratch_move);
        // `movVec` copies the whole 128-bit register: correct for vector block
        // parameters, harmless (a few extra bits) for scalar floats.
        try parallelMove(allocator, code, fpr_moves.items, encode.movVec, fp_move);
        for (args, params) |arg, param| {
            if (self.alloc.spill.get(arg)) |slot| {
                const pr = self.alloc.reg.get(param).?;
                const off: u15 = @intCast(self.spill_base + slot * 16);
                if (isVector(self.func, param)) {
                    try code.append(allocator, encode.ldrQ(pr, sp, off));
                } else if (regClass(self.func, param) == .fpr) {
                    try code.append(allocator, encode.ldrFp(pr, sp, off, true));
                } else {
                    try code.append(allocator, encode.ldrOff(pr, sp, off));
                }
            }
        }
    }
};

/// The register to compute `result` into: its assigned register, or the spilled
/// result scratch for its class (the caller then stores it with `storeResult`).
fn resultReg(alloc: *const Allocation, func: *const Function, result: Value) Reg {
    return alloc.reg.get(result) orelse (if (regClass(func, result) == .fpr) fp_spill_res else spill_res);
}

fn storeResult(allocator: std.mem.Allocator, code: *std.ArrayList(u32), ctx: Ctx, result: Value, reg: Reg) Error!void {
    if (ctx.alloc.spill.get(result)) |slot| {
        const off: u15 = @intCast(ctx.spill_base + slot * 16);
        if (isVector(ctx.func, result)) {
            try code.append(allocator, encode.strQ(reg, sp, off));
        } else if (regClass(ctx.func, result) == .fpr) {
            try code.append(allocator, encode.strFp(reg, sp, off, true));
        } else {
            try code.append(allocator, encode.strOff(reg, sp, off));
        }
    }
}

const Interval = struct { value: Value, start: u32, end: u32, is_param: bool };

/// Linear-scan register allocation with spilling. Block parameters are pinned to
/// registers. Instruction results spill when the pool is exhausted.
fn allocate(allocator: std.mem.Allocator, func: *const Function, leaf: bool) Error!Allocation {
    const nval = func.valueCount();
    var alloc = Allocation{};
    errdefer alloc.deinit(allocator);

    // Linearize: assign each instruction a position, record def positions, last
    // uses, params, and each block's terminator position.
    const nblocks = func.blockCount();
    const def_pos = try allocator.alloc(u32, nval);
    defer allocator.free(def_pos);
    const last_use = try allocator.alloc(u32, nval);
    defer allocator.free(last_use);
    const is_param = try allocator.alloc(bool, nval);
    defer allocator.free(is_param);
    const block_end = try allocator.alloc(u32, nblocks);
    defer allocator.free(block_end);
    @memset(def_pos, 0);
    @memset(is_param, false);
    for (last_use) |*l| l.* = 0;

    // Positions at which a call clobbers caller-saved registers. AAPCS only preserves the
    // LOW 64 bits of the callee-saved FP registers v8..v15 across a call, so a 128-bit SIMD
    // `<4 x f32>` held in one of them would lose its upper two lanes over a `bl`/`blr`. We
    // record every call position and force-spill any VECTOR interval that spans a call to the
    // stack (16-byte slots are fully preserved), keeping the widened (quad) FS correct when it
    // gathers through a sampler_fn / math_fn call.
    var call_positions = std.ArrayList(u32).empty;
    defer call_positions.deinit(allocator);

    var pos: u32 = 0;
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| {
            def_pos[@intFromEnum(p)] = pos;
            last_use[@intFromEnum(p)] = pos;
            is_param[@intFromEnum(p)] = true;
        }
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            forEachUse(func, inst, last_use, pos);
            if (func.instResult(inst)) |r| def_pos[@intFromEnum(r)] = pos;
            switch (func.opcode(inst)) {
                .call, .call_indirect => try call_positions.append(allocator, pos),
                else => {},
            }
            pos += 1;
        }
        block_end[bi] = pos;
        if (func.terminator(block)) |term| forEachTermUse(func, term, last_use, pos);
        pos += 1;
    }

    // Liveness: a value live-out of a block stays live to that block's end. A loop
    // back-edge makes the header's live-in flow into the body's live-out, extending
    // loop-carried values across the body so their registers are not reused inside it.
    try extendLiveRanges(allocator, func, last_use, block_end);

    // Per-class register pools. GPR: caller-saved temporaries + unused integer arg
    // registers (leaf) or callee-saved x19..x28 (non-leaf). FPR: caller-saved
    // v16..v23 + unused FP arg registers (leaf) or callee-saved v8..v15 (non-leaf).
    const eparams = func.blockParams(@enumFromInt(0));
    const plocs = try allocator.alloc(ArgLoc, eparams.len);
    defer allocator.free(plocs);
    computeArgLocs(func, eparams, plocs);
    var n_gpr: usize = 0;
    var n_fpr: usize = 0;
    for (plocs) |l| {
        if (l.class == .gpr) n_gpr += 1 else n_fpr += 1;
    }

    var pools = [_]std.ArrayList(Reg){ .empty, .empty };
    defer for (&pools) |*p| p.deinit(allocator);
    if (leaf) {
        for (9..13) |r| try pools[0].append(allocator, @enumFromInt(@as(u5, @intCast(r))));
        for (@min(n_gpr, 8)..8) |r| try pools[0].append(allocator, @enumFromInt(@as(u5, @intCast(r))));
        for (16..24) |r| try pools[1].append(allocator, @enumFromInt(@as(u5, @intCast(r))));
        for (@min(n_fpr, 8)..8) |r| try pools[1].append(allocator, @enumFromInt(@as(u5, @intCast(r))));
    } else {
        for (19..29) |r| try pools[0].append(allocator, @enumFromInt(@as(u5, @intCast(r))));
        for (8..16) |r| try pools[1].append(allocator, @enumFromInt(@as(u5, @intCast(r))));
    }

    var ivals = try allocator.alloc(Interval, nval);
    defer allocator.free(ivals);
    for (0..nval) |i| ivals[i] = .{ .value = @enumFromInt(i), .start = def_pos[i], .end = last_use[i], .is_param = is_param[i] };
    std.mem.sort(Interval, ivals, {}, lessByStart);

    // A leaf keeps its register parameters in the incoming argument registers
    // (x0..x7 / v0..v7). A non-leaf lets them flow through the scan into
    // callee-saved registers (prologue moves them in). Stack parameters (index >= 8)
    // always flow through the scan and are loaded in the prologue.
    if (leaf) for (eparams, plocs) |p, l| {
        if (l.idx < 8) try alloc.reg.put(allocator, p, @enumFromInt(@as(u5, @intCast(l.idx))));
    };

    var frees = [_]std.ArrayList(Reg){ .empty, .empty };
    defer for (&frees) |*f| f.deinit(allocator);
    for (pools[0].items) |r| try frees[0].append(allocator, r);
    for (pools[1].items) |r| try frees[1].append(allocator, r);

    const Active = struct { end: u32, value: Value, reg: Reg, is_param: bool };
    var actives = [_]std.ArrayList(Active){ .empty, .empty };
    defer for (&actives) |*a| a.deinit(allocator);

    for (ivals) |iv| {
        const cls: usize = @intFromEnum(regClass(func, iv.value));
        const free = &frees[cls];
        const active = &actives[cls];

        // An entry parameter pinned by a leaf just occupies its register.
        if (alloc.reg.get(iv.value)) |r| {
            try active.append(allocator, .{ .end = iv.end, .value = iv.value, .reg = r, .is_param = iv.is_param });
            continue;
        }

        // Expire intervals that ended before this one starts.
        var w: usize = 0;
        for (active.items) |a| {
            if (a.end < iv.start) {
                try free.append(allocator, a.reg);
            } else {
                active.items[w] = a;
                w += 1;
            }
        }
        active.shrinkRetainingCapacity(w);

        // A non-parameter VECTOR value that is live across a call must live on the stack: the
        // callee-saved FP registers only preserve their low 64 bits, so a 128-bit lane vector
        // in v8..v15 would be corrupted by the call. (Params are pinned/handled separately.
        // A vector param crossing a call is spilled too, below.)
        if (isVector(func, iv.value) and spansCall(call_positions.items, iv.start, iv.end)) {
            try spillValue(allocator, &alloc, iv.value);
            continue;
        }

        if (free.pop()) |r| {
            try alloc.reg.put(allocator, iv.value, r);
            try active.append(allocator, .{ .end = iv.end, .value = iv.value, .reg = r, .is_param = iv.is_param });
            continue;
        }

        // Out of registers: pick the spillable (non-parameter) active interval with
        // the furthest end in this class.
        var victim: ?usize = null;
        for (active.items, 0..) |a, i| {
            if (a.is_param) continue;
            if (victim == null or a.end > active.items[victim.?].end) victim = i;
        }
        if (iv.is_param) {
            const vi = victim orelse return error.Unsupported; // too many live params
            try spillValue(allocator, &alloc, active.items[vi].value);
            try alloc.reg.put(allocator, iv.value, active.items[vi].reg);
            active.items[vi] = .{ .end = iv.end, .value = iv.value, .reg = active.items[vi].reg, .is_param = true };
        } else if (victim) |vi| {
            if (active.items[vi].end > iv.end) {
                try spillValue(allocator, &alloc, active.items[vi].value);
                try alloc.reg.put(allocator, iv.value, active.items[vi].reg);
                active.items[vi] = .{ .end = iv.end, .value = iv.value, .reg = active.items[vi].reg, .is_param = false };
            } else {
                try spillValue(allocator, &alloc, iv.value);
            }
        } else {
            try spillValue(allocator, &alloc, iv.value);
        }
    }

    // The callee-saved registers actually used (non-leaf), per class, ascending.
    // The GPR (x19..x28) and FPR (v8..v15) callee-saved ranges do not overlap, so
    // a register index identifies its class unambiguously.
    if (!leaf) {
        try collectSaved(&alloc, 19, 29, &alloc.saved_gpr, allocator);
        try collectSaved(&alloc, 8, 16, &alloc.saved_fpr, allocator);
    }

    return alloc;
}

fn collectSaved(alloc: *Allocation, lo: u6, hi: u6, out: *std.ArrayList(Reg), allocator: std.mem.Allocator) Error!void {
    var r: u6 = lo;
    while (r < hi) : (r += 1) {
        const reg: Reg = @enumFromInt(@as(u5, @intCast(r)));
        var used = false;
        var it = alloc.reg.valueIterator();
        while (it.next()) |v| if (v.* == reg) {
            used = true;
        };
        if (used) try out.append(allocator, reg);
    }
}

fn spillValue(allocator: std.mem.Allocator, alloc: *Allocation, v: Value) Error!void {
    if (alloc.spill.contains(v)) return;
    _ = alloc.reg.remove(v);
    try alloc.spill.put(allocator, v, alloc.spill_count);
    alloc.spill_count += 1;
}

fn lessByStart(_: void, a: Interval, b: Interval) bool {
    return a.start < b.start;
}

/// Whether a value live over `[start, end)` is live ACROSS a call (a call position strictly
/// between its definition and its last use). Such a vector value must be stack-resident.
fn spansCall(call_positions: []const u32, start: u32, end: u32) bool {
    for (call_positions) |p| {
        if (p > start and p < end) return true;
    }
    return false;
}

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
        .load => |l| markUse(last_use, l.ptr, pos),
        .store => |st| {
            markUse(last_use, st.value, pos);
            markUse(last_use, st.ptr, pos);
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

fn forEachTermUse(func: *const Function, term: Terminator, last_use: []u32, pos: u32) void {
    switch (term) {
        .ret => |v| if (v) |vv| markUse(last_use, vv, pos),
        .jump => |j| for (func.blockArgs(j)) |a| markUse(last_use, a, pos),
    }
}

/// Whether the integer `icmp` at `insts[idx]` fuses into an immediately-following
/// `@"if"` whose condition it is and whose only use it is. When it fuses, the icmp
/// materialization (`cmp; cset`) is skipped and the if emits a fused `cmp; b.cc` on
/// the icmp's operands (see `emitIf`). This is the ONE eligibility predicate shared by
/// the icmp-skip and the fused emitIf, so they never disagree (no dangling or doubled
/// compare). It is gated to integer operands (the gpr compare path); float and vector
/// compares keep the current materialize-then-test path. The immediately-preceding +
/// single-use conditions make skipping the boolean register-safe: nothing runs between
/// the icmp and the if, so the icmp's operand registers still hold their values at the
/// if, and no other reader needs the boolean.
fn fusesIntoNextIf(func: *const Function, insts: []const ir.function.Inst, idx: usize) bool {
    const cmp = switch (func.opcode(insts[idx])) {
        .icmp => |c| c,
        else => return false,
    };
    // Integer operands only (the else branch of the icmp lowering). isVector and the
    // fpr class both route float/vector compares elsewhere, so require the gpr path.
    if (isVector(func, cmp.lhs) or regClass(func, cmp.lhs) != .gpr) return false;
    if (idx + 1 >= insts.len) return false; // must be immediately followed by the if
    const cf = switch (func.opcode(insts[idx + 1])) {
        .@"if" => |c| c,
        else => return false,
    };
    const result = func.instResult(insts[idx]) orelse return false;
    if (cf.cond != result) return false; // the if must test exactly this icmp's result
    // Single-use: the boolean is read only by this if's condition. Since the icmp
    // immediately precedes the if and equals cf.cond, a total use-count of exactly 1
    // means the if's cond is the sole use, so skipping the boolean harms nothing.
    return countUses(func, result) == 1;
}

/// Whether `v`'s type is a vector over a float element (the only vector shape arith
/// fusion ever sees today - `<4 x f32>` per `binary`'s vector path; `dot` is a separate
/// IR op over int8 vectors and never reaches here).
fn isFloatVector(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .vector => |vec| func.types.type_kind(vec.elem) == .float,
        else => false,
    };
}

/// Whether the float `mul` at `insts[idx]` fuses into an immediately-following `add`/`sub`
/// that consumes its result as a fused multiply-add/subtract (one rounding instead of
/// two - legal because Vulcan permits fp-contraction). When it fuses, the mul's
/// materialization is skipped and the add/sub emits fmadd/fmsub/fnmsub (scalar) or
/// fmla/fmls (vector) on the mul's own operands (see `Ctx.emitFusedArith`). This is the ONE
/// eligibility predicate shared by the mul-skip and the fused add/sub emission, so they
/// never disagree (no dangling or doubled multiply) - mirrors `fusesIntoNextIf`. Gated to
/// float operands (scalar or vector): integer `add(mul,c)` has no rounding to fuse away and
/// is never an fma. For a vector mul, only the two shapes a single NEON instruction can
/// express are allowed: `add(mul,c)` = a*b+c -> FMLA, and `sub(c,mul)` = c-a*b -> FMLS.
/// `sub(mul,c)` = a*b-c has no matching NEON op (FMLA/FMLS only ever add or subtract the
/// product, never negate the whole result), so that shape is rejected here and falls
/// through to the separate fmul+fsub path in both the mul-skip and the add/sub emission,
/// since both call this same function. The immediately-preceding + single-use conditions
/// make skipping the product register-safe: nothing runs between the mul and the add/sub,
/// so the mul's operand registers still hold their values there, and no other reader needs
/// the standalone product.
fn fusesIntoNextArith(func: *const Function, insts: []const ir.function.Inst, idx: usize) bool {
    const mul = switch (func.opcode(insts[idx])) {
        .arith => |a| a,
        else => return false,
    };
    if (mul.op != .mul) return false;
    const vector = isVector(func, mul.lhs);
    // Float operands only: regClass != .fpr means an integer scalar mul (no rounding to
    // fuse away); a vector mul must have a float element (see `isFloatVector`).
    if (vector) {
        if (!isFloatVector(func, mul.lhs)) return false;
    } else if (regClass(func, mul.lhs) != .fpr) {
        return false;
    } else if (isHalf(func, mul.lhs)) {
        // f16 arithmetic must round to nearest-even half after EACH op (the emulation holds a
        // half as its f32 widening). A fused fmadd rounds the product-sum only once, at f32
        // precision, skipping the intermediate half-rounding of the multiply, so it is not
        // valid per-op f16 semantics. Both fusion call sites share this predicate, so they
        // agree and fall back to a rounded fmul followed by a rounded fadd/fsub.
        return false;
    }
    if (idx + 1 >= insts.len) return false; // must be immediately followed by the add/sub
    const addsub = switch (func.opcode(insts[idx + 1])) {
        .arith => |a| a,
        else => return false,
    };
    if (addsub.op != .add and addsub.op != .sub) return false;
    const result = func.instResult(insts[idx]) orelse return false;
    if (addsub.lhs != result and addsub.rhs != result) return false; // must consume this mul's result
    // Vector sub(mul, c) = a*b-c has no single-instruction NEON form: reject it here so
    // both call sites (the mul-skip and the fused emit) agree and fall back to fmul+fsub.
    if (vector and addsub.op == .sub and addsub.lhs == result) return false;
    // Single-use: the product is read only by this add/sub. Since the mul immediately
    // precedes it and is one of its operands, a total use-count of exactly 1 means this is
    // the sole use, so skipping the materialization harms nothing.
    return countUses(func, result) == 1;
}

/// Total operand uses of `v` across the whole function (instruction operands, if/jump
/// edge args, and terminators). Used by the fusion eligibility's single-use check.
fn countUses(func: *const Function, v: Value) usize {
    var count: usize = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| count += usesOfInInst(func, inst, v);
        if (func.terminator(block)) |term| count += usesOfInTerm(func, term, v);
    }
    return count;
}

fn usesOfInInst(func: *const Function, inst: ir.function.Inst, v: Value) usize {
    var c: usize = 0;
    switch (func.opcode(inst)) {
        .iconst, .fconst, .alloca, .global_addr => {},
        .arith => |a| {
            if (a.lhs == v) c += 1;
            if (a.rhs == v) c += 1;
        },
        .arith_imm => |a| {
            if (a.lhs == v) c += 1;
        },
        .icmp => |cc| {
            if (cc.lhs == v) c += 1;
            if (cc.rhs == v) c += 1;
        },
        .select => |s| {
            if (s.cond == v) c += 1;
            if (s.then == v) c += 1;
            if (s.@"else" == v) c += 1;
        },
        .extract => |e| {
            if (e.aggregate == v) c += 1;
        },
        .convert => |cv| {
            if (cv.value == v) c += 1;
        },
        .unary => |u| {
            if (u.value == v) c += 1;
        },
        .load => |l| {
            if (l.ptr == v) c += 1;
        },
        .store => |st| {
            if (st.value == v) c += 1;
            if (st.ptr == v) c += 1;
        },
        .prefetch => |pf| {
            if (pf.ptr == v) c += 1;
        },
        .dot => |d| {
            if (d.acc == v) c += 1;
            if (d.a == v) c += 1;
            if (d.b == v) c += 1;
        },
        .matmul => |mmv| {
            if (mmv.a == v) c += 1;
            if (mmv.b == v) c += 1;
            if (mmv.c == v) c += 1;
        },
        .struct_new => |sn| for (func.valueList(sn.fields)) |f| {
            if (f == v) c += 1;
        },
        .call => |cl| for (func.valueList(cl.args)) |a| {
            if (a == v) c += 1;
        },
        .call_indirect => |cl| {
            if (cl.target == v) c += 1;
            for (func.valueList(cl.args)) |a| {
                if (a == v) c += 1;
            }
        },
        .@"if" => |cf| {
            if (cf.cond == v) c += 1;
            for (func.blockArgs(cf.then)) |a| {
                if (a == v) c += 1;
            }
            for (func.blockArgs(cf.@"else")) |a| {
                if (a == v) c += 1;
            }
        },
    }
    return c;
}

fn usesOfInTerm(func: *const Function, term: Terminator, v: Value) usize {
    var c: usize = 0;
    switch (term) {
        .ret => |x| if (x) |xx| {
            if (xx == v) c += 1;
        },
        .jump => |j| for (func.blockArgs(j)) |a| {
            if (a == v) c += 1;
        },
    }
    return c;
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
        .load => |l| setUsed(row, l.ptr),
        .store => |st| {
            setUsed(row, st.value);
            setUsed(row, st.ptr);
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

fn markUsedTermBitset(func: *const Function, term: Terminator, row: []bool) void {
    switch (term) {
        .ret => |v| if (v) |vv| setUsed(row, vv),
        .jump => |j| for (func.blockArgs(j)) |a| setUsed(row, a),
    }
}

/// Backward liveness dataflow. Extends `last_use[v]` to the end position of every
/// block where `v` is live-out, so a value live across a loop keeps its register.
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

/// Sequence a set of parallel register moves (within one register class), using
/// `movFn` to emit a move and `scratch` to break cycles.
fn parallelMove(allocator: std.mem.Allocator, code: *std.ArrayList(u32), moves_in: []const Move, movFn: *const fn (Reg, Reg) u32, scratch: Reg) Error!void {
    var list: std.ArrayList(Move) = .empty;
    defer list.deinit(allocator);
    for (moves_in) |m| if (m.src != m.dst) try list.append(allocator, m);

    var guard: usize = moves_in.len * 2 + 4;
    while (list.items.len > 0 and guard > 0) : (guard -= 1) {
        var emitted = false;
        for (list.items, 0..) |m, i| {
            var blocked = false;
            for (list.items) |o| if (o.src == m.dst) {
                blocked = true;
                break;
            };
            if (!blocked) {
                try code.append(allocator, movFn(m.dst, m.src));
                _ = list.orderedRemove(i);
                emitted = true;
                break;
            }
        }
        if (!emitted) {
            const c = list.items[0];
            try code.append(allocator, movFn(scratch, c.src));
            for (list.items) |*m| if (m.src == c.src) {
                m.src = scratch;
            };
        }
    }
}

fn typeSize(func: *const Function, ty: ir.types.Type) usize {
    return switch (func.types.type_kind(ty)) {
        .bool => 1,
        .int => |i| (@as(usize, i.bits) + 7) / 8,
        .ptr => 8,
        // The MEMORY size of a float. f16 is a 2-byte IEEE half in memory (its in-register
        // spill form is separate: an f16 spills as its f32 widening via the uniform 16-byte
        // scalar-fpr spill slot, so this 2 never sizes a spill, only alloca/struct layout).
        .float => |f| switch (f) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
        },
        .array => |a| @as(usize, @intCast(a.len)) * typeSize(func, a.elem),
        .vector => |v| @as(usize, v.len) * typeSize(func, v.elem),
        else => 8,
    };
}

/// The natural alignment of a type's storage (for stack-slot placement).
fn typeAlign(func: *const Function, ty: ir.types.Type) usize {
    const sz = switch (func.types.type_kind(ty)) {
        .array => |a| typeSize(func, a.elem), // align an array to its element
        else => typeSize(func, ty),
    };
    return if (sz <= 1) 1 else if (sz <= 2) 2 else if (sz <= 4) 4 else 8;
}

fn computeAllocaSlots(allocator: std.mem.Allocator, func: *const Function, map: *std.AutoHashMapUnmanaged(Value, u32)) Error!usize {
    var cur: usize = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .alloca => |al| {
                    const sz = typeSize(func, al.elem);
                    cur = alignUp(cur, typeAlign(func, al.elem));
                    try map.put(allocator, func.instResult(inst).?, @intCast(cur));
                    cur += sz;
                },
                else => {},
            }
        }
    }
    return alignUp(cur, 16);
}

fn emitLoad(allocator: std.mem.Allocator, code: *std.ArrayList(u32), func: *const Function, result: Value, rd: Reg, base: Reg, fp16: bool) Error!void {
    if (isVector(func, result)) {
        try code.append(allocator, encode.ldrQ(rd, base, 0)); // 128-bit NEON load
        return;
    }
    if (regClass(func, result) == .fpr) {
        if (isHalf(func, result)) {
            // Load a 16-bit IEEE-half memory object. NOT `ldr s`, which would read 32 bits from a
            // 2-byte object. NATIVE (fp16): `ldr h` leaves the value in the H view ready to use.
            // In the emulation path, also widen it to the S-held f32 form with `fcvt s,h`.
            try code.append(allocator, encode.ldrHfp(rd, base, 0));
            if (!fp16) try code.append(allocator, encode.fcvtSfromH(rd, rd));
            return;
        }
        try code.append(allocator, encode.ldrFp(rd, base, 0, isDouble(func, result)));
        return;
    }
    const sz = typeSize(func, func.valueType(result));
    const signed = isSignedInt(func, result);
    if (sz <= 1) {
        try code.append(allocator, if (signed) encode.ldrsb(rd, base) else encode.ldrb(rd, base));
    } else if (sz <= 2) {
        try code.append(allocator, if (signed) encode.ldrsh(rd, base) else encode.ldrh(rd, base));
    } else if (sz <= 4) {
        try code.append(allocator, encode.ldrW(rd, base, 0));
    } else {
        try code.append(allocator, encode.ldrOff(rd, base, 0));
    }
}

fn emitStore(allocator: std.mem.Allocator, code: *std.ArrayList(u32), func: *const Function, value: Value, val: Reg, base: Reg, fp16: bool) Error!void {
    if (isVector(func, value)) {
        try code.append(allocator, encode.strQ(val, base, 0)); // 128-bit NEON store
        return;
    }
    if (regClass(func, value) == .fpr) {
        if (isHalf(func, value)) {
            // Store a 16-bit IEEE-half memory object. NATIVE (fp16): the value is already a native
            // half in the H view, so `str h` writes it directly. EMULATION: the value is an S-held
            // f32 widening, so narrow it first (`fcvt h,s`) into a fixed scratch (fp_move, v27,
            // outside every allocation pool) so `val`, which may still be live, is never clobbered
            // (fcvt h zeroes the upper bits of its destination register); the narrow is lossless
            // since the S value is already an exact half.
            if (fp16) {
                try code.append(allocator, encode.strHfp(val, base, 0));
            } else {
                try code.append(allocator, encode.fcvtHfromS(fp_move, val));
                try code.append(allocator, encode.strHfp(fp_move, base, 0));
            }
            return;
        }
        try code.append(allocator, encode.strFp(val, base, 0, isDouble(func, value)));
        return;
    }
    const sz = typeSize(func, func.valueType(value));
    if (sz <= 1) {
        try code.append(allocator, encode.strb(val, base));
    } else if (sz <= 2) {
        try code.append(allocator, encode.strh(val, base));
    } else if (sz <= 4) {
        try code.append(allocator, encode.strW(val, base, 0));
    } else {
        try code.append(allocator, encode.strOff(val, base, 0));
    }
}

fn isSignedInt(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |x| x.signedness == .signed,
        else => true,
    };
}

/// Element signedness of a `dot` data operand (`a`/`b`, always `<16 x i8>` or
/// `<16 x u8>` per verify.zig): signed picks SDOT, unsigned picks UDOT.
fn dotSigned(func: *const Function, v: Value) bool {
    const elem = func.types.type_kind(func.valueType(v)).vector.elem;
    return switch (func.types.type_kind(elem)) {
        .int => |x| x.signedness == .signed,
        else => unreachable, // verify.zig requires an int8 element
    };
}

fn condFor(op: ir.function.CmpOp, signed: bool) encode.Cond {
    return switch (op) {
        .eq => .eq,
        .ne => .ne,
        .lt => if (signed) .lt else .lo,
        .le => if (signed) .le else .ls,
        .gt => if (signed) .gt else .hi,
        .ge => if (signed) .ge else .hs,
    };
}

/// Condition codes for an ordered floating-point compare (after `fcmp`). Less-than
/// uses `mi` and less-or-equal uses `ls`, which read the flags `fcmp` sets.
fn condForFloat(op: ir.function.CmpOp) encode.Cond {
    return switch (op) {
        .eq => .eq,
        .ne => .ne,
        .lt => .mi,
        .le => .ls,
        .gt => .gt,
        .ge => .ge,
    };
}

/// Emit a binary op into `rd` from `rn`, `rm`. Division/remainder/shift-right pick
/// signed or unsigned forms from `signed`. A remainder is a divide plus `msub`
/// through a quotient scratch. `wide` selects the 64-bit (x-register) form for
/// pointer/address arithmetic. The `sf` bit (31) is uniform across these A64
/// data-processing encodings, so it composes with the 32-bit base.
fn emitBinary(allocator: std.mem.Allocator, code: *std.ArrayList(u32), op: ir.function.BinOp, rd: Reg, rn: Reg, rm: Reg, signed: bool, wide: bool) Error!void {
    const sf: u32 = if (wide) @as(u32, 1) << 31 else 0;
    const word = switch (op) {
        .add => encode.add(rd, rn, rm),
        .sub => encode.sub(rd, rn, rm),
        .mul => encode.mul(rd, rn, rm),
        // High half of a 64x64 product. Only the magic-number divide emits this, always on 64-bit
        // operands, so the x-register smulh/umulh (which read the full 64 bits) is exactly right.
        .mulh => if (signed) encode.smulh(rd, rn, rm) else encode.umulh(rd, rn, rm),
        .bit_and => encode.andr(rd, rn, rm),
        .bit_or => encode.orr(rd, rn, rm),
        .bit_xor => encode.eor(rd, rn, rm),
        .div => if (signed) encode.sdiv(rd, rn, rm) else encode.udiv(rd, rn, rm),
        .shl => encode.lslv(rd, rn, rm),
        .shr => if (signed) encode.asrv(rd, rn, rm) else encode.lsrv(rd, rn, rm),
        .rem => {
            // rd = rn - (rn / rm) * rm.
            try code.append(allocator, sf | (if (signed) encode.sdiv(scratch_imm, rn, rm) else encode.udiv(scratch_imm, rn, rm)));
            try code.append(allocator, sf | encode.msub(rd, scratch_imm, rm, rn));
            return;
        },
    };
    try code.append(allocator, sf | word);
}

/// The bit width of an integer value (its type is assumed to be an int, as at an int<->int convert).
fn intBitsOf(func: *const Function, v: Value) u16 {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |x| x.bits,
        else => 64,
    };
}

/// Whether a value occupies a full 64-bit register (a pointer or a 64-bit int),
/// so arithmetic on it must use the 64-bit ALU form.
fn isWide(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .ptr => true,
        .int => |x| x.bits > 32,
        else => false,
    };
}

fn loadConst(allocator: std.mem.Allocator, code: *std.ArrayList(u32), rd: Reg, c: i64) Error!void {
    const bits: u32 = @truncate(@as(u64, @bitCast(c)));
    const lo: u16 = @truncate(bits);
    const hi: u16 = @truncate(bits >> 16);
    try code.append(allocator, encode.movz(rd, lo, 0));
    if (hi != 0) try code.append(allocator, encode.movk(rd, hi, 1));
}

/// Materialize a full 64-bit constant (for f64 bit patterns) with movz + movk.
fn loadConst64(allocator: std.mem.Allocator, code: *std.ArrayList(u32), rd: Reg, bits: u64) Error!void {
    try code.append(allocator, encode.movz64(rd, @truncate(bits), 0));
    inline for (1..4) |i| {
        const part: u16 = @truncate(bits >> (16 * i));
        if (part != 0) try code.append(allocator, encode.movk64(rd, part, i));
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
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = prod, .rhs = x } });
    func.setTerminator(b, .{ .ret = sum });

    const code = try selectFunction(allocator, &func);
    defer allocator.free(code);
    // The first instruction multiplies the two parameters (x0, x1) into an
    // allocator-chosen register. The function ends in ret.
    const rd_mask = ~@as(u32, 0x1f);
    try std.testing.expectEqual(encode.mul(.x0, .x0, .x1) & rd_mask, code[0] & rd_mask);
    try std.testing.expectEqual(encode.ret(), code[code.len - 1]);
}

test "an f16 function now compiles on aarch64 (the reference f16 backend, no reject gate)" {
    // Task 3 replaced the Task-2 rejection gate with real f16 lowering. A function that adds
    // two f16 values must now compile (it emits the S-form fadd plus the round-to-half
    // narrow/widen); the executable differentials in tests/native.zig prove correctness.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f16_t = try func.types.intern(.{ .float = .f16 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, f16_t);
    const y = try func.appendBlockParam(b, f16_t);
    const sum = try func.appendInst(b, f16_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = sum });

    const code = try selectFunction(allocator, &func);
    defer allocator.free(code);
    // The S-form fadd is followed by the round-to-half pair (fcvt h,s; fcvt s,h): the emitted
    // stream must contain both narrow and widen opcodes (register fields masked off).
    const rd_mask = ~@as(u32, 0x1f);
    var saw_narrow = false;
    var saw_widen = false;
    for (code) |w| {
        if (w & 0xFFFFFC00 == encode.fcvtHfromS(.x0, .x0) & rd_mask) saw_narrow = true;
        if (w & 0xFFFFFC00 == encode.fcvtSfromH(.x0, .x0) & rd_mask) saw_widen = true;
    }
    try std.testing.expect(saw_narrow);
    try std.testing.expect(saw_widen);
}

test "alignPadWords computes the nop count to reach a fetch-align boundary" {
    // 3 words in, 32-byte (8-word) alignment: 8 - 3 = 5 words of padding.
    try std.testing.expectEqual(@as(usize, 5), alignPadWords(3, 32));
    // Already on an 8-word boundary: no padding needed.
    try std.testing.expectEqual(@as(usize, 0), alignPadWords(8, 32));
    // fetch_align <= 4 (one word or less, or disabled): always a no-op.
    try std.testing.expectEqual(@as(usize, 0), alignPadWords(3, 4));
    try std.testing.expectEqual(@as(usize, 0), alignPadWords(3, 0));
}
