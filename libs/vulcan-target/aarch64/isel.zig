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
const peephole = @import("peephole.zig");
const addrfold = @import("../addrfold.zig");
const wimmer = @import("../wimmer.zig");
const loops = @import("vulcan-opt").loops;
const mm = @import("vulcan-opt").microarch;

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Reg = encode.Reg;
const RegMap = std.AutoHashMapUnmanaged(Value, Reg);

pub const Error = std.mem.Allocator.Error || error{Unsupported};

const sp: Reg = .zr; // register 31 is the stack pointer in load/store/frame context

/// A shared no-fold analysis for the paths that must stay fold-agnostic (a Wimmer differential
/// compile, the debug/liveness hooks): its `baseOf`/`offOf`/`isDeadAdd` behave as if nothing folded,
/// so those paths emit byte-identical code to before address folding existed.
const empty_fold: addrfold.Analysis = addrfold.Analysis.empty;

/// The byte-scaled access granule of `v`'s memory form, matching EXACTLY the encoder bucket that
/// `emitLoad`/`emitStore` select for it. A folded displacement must be a whole multiple of this
/// granule (the base+offset encoders scale the immediate by it), so `aarch64FoldOffset` uses it for
/// both the divisibility and the range check. Vector is a 16-byte Q access, an fp half a 2-byte H,
/// fp single/double a 4/8-byte S/D, and an integer its 1/2/4/8-byte bucket.
fn aarch64AccessScale(func: *const Function, v: Value) usize {
    if (isVector(func, v)) return 16;
    if (regClass(func, v) == .fpr) {
        if (isHalf(func, v)) return 2;
        return if (isDouble(func, v)) 8 else 4;
    }
    const sz = typeSize(func, func.valueType(v));
    if (sz <= 1) return 1;
    if (sz <= 2) return 2;
    if (sz <= 4) return 4;
    return 8;
}

/// The aarch64 fold predicate for `addrfold.analyze`: fold a load/store whose pointer is an
/// `arith_imm.add(base, imm)` when `imm` fits the u12 scaled unsigned-offset addressing form for the
/// op's access granule (imm >= 0, imm a multiple of the granule, imm / granule <= 4095). Returns the
/// byte offset (equal to the add's imm) when in range, else null. `analyze` calls this only after
/// confirming the pointer is an `arith_imm.add`, so the unwraps below are guaranteed, still asserted.
fn aarch64FoldOffset(_: void, func: *const Function, mem_inst: ir.function.Inst) ?i64 {
    const val = switch (func.opcode(mem_inst)) {
        .load => func.instResult(mem_inst).?, // the loaded value decides the access size
        .store => |st| st.value, // the stored value decides the access size
        else => unreachable, // analyze only hands foldOffset a load or store
    };
    const ptr = switch (func.opcode(mem_inst)) {
        .load => |l| l.ptr,
        .store => |st| st.ptr,
        else => unreachable,
    };
    const def = func.definingInst(ptr).?; // analyze confirmed ptr is defined by an arith_imm.add
    const add = switch (func.opcode(def)) {
        .arith_imm => |a| a,
        else => unreachable,
    };
    std.debug.assert(add.op == .add);
    const scale: i64 = @intCast(aarch64AccessScale(func, val));
    if (add.imm < 0) return null;
    if (@rem(add.imm, scale) != 0) return null;
    if (@divExact(add.imm, scale) > 4095) return null;
    return add.imm;
}

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
pub const Fixup = struct { at: usize, target: u32 };

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

/// Where a value lives at a given program point: a register or a stack spill slot. A whole-life
/// value has one location for its entire range (today's `reg`/`spill`); a split value has several,
/// selected by position through `segments`.
const Location = union(enum) { reg: Reg, slot: u32 };

/// One piece of a split value's life: the value lives in `loc` from position `from` until the next
/// segment (or, for the last one, to the end of its range). `segments[0].from` is the value's def
/// position, so a lookup at any position at or after the def resolves to some segment.
const Segment = struct { from: u32, loc: Location };

/// One precomputed control-flow-edge move (Task 8), translated from a `wimmer.Move`: shuffle `src`
/// into `dst`, where `class` (0 gpr, 1 fpr) selects the move/load/store width. The shared allocator
/// already ORDERED these into a valid parallel-move sequence (every source read before it is
/// overwritten, cycles broken and slot->slot shuffles routed through the class scratch), so the
/// emitter replays them op-by-op with no reordering.
const EdgeMove = struct { src: Location, dst: Location, class: u16 };

/// The ordered move list on one control-flow edge `pred -> succ` (Task 7's `wimmer.EdgeMoves`,
/// translated into this backend's `Location` space). Keyed by the block pair so emission can find
/// the moves for the edge it is currently lowering (the block being emitted plus the jump target,
/// including the forwarding blocks `splitCriticalEdges` inserted).
const EdgeMoveSet = struct { pred: Block, succ: Block, moves: []EdgeMove };

/// A store or reload the emitter must insert at a split boundary. `at` is the instruction position
/// the action lands before (the position at which the pool was exhausted for a tail split). `store`
/// writes the victim's register to its new slot before the taker overwrites the register at `at`.
/// `reload` (Task 5) brings a value back into a register. `value` selects the ldr/str form
/// (vector vs fp vs gpr). `slot_to_slot` (Wimmer bridge gap #7) re-homes a spilled value from one
/// slot to another WITHOUT ever holding it in a value register: `emitSplitAction` expands it into a
/// reload-then-store pair through the class scratch (`reg`/`move_from` stay at their defaults and
/// are never read for this kind, since the scratch register is implied by `value`'s class, not
/// carried in the action).
const SplitAction = struct {
    at: u32,
    kind: enum { store, reload, move, slot_to_slot },
    value: Value,
    reg: Reg,
    slot: u32 = 0,
    /// The SOURCE register of a `.move` (a register-to-register re-home at a split point); `reg` is
    /// the destination. Only the shared Wimmer translation produces `.move`; the native `allocate`
    /// never does, so it leaves this at its default and the native drain never reads it.
    move_from: Reg = .zr,
    /// The SOURCE slot of a `.slot_to_slot` re-home; `slot` is the destination. Only the shared
    /// Wimmer translation produces `.slot_to_slot`; the native `allocate` never does.
    move_from_slot: u32 = 0,
};

/// The result of register allocation.
const Allocation = struct {
    reg: RegMap = .empty, // value -> register (index, class implied by the value type)
    spill: std.AutoHashMapUnmanaged(Value, u32) = .empty, // value -> spill slot index
    // Split values only: value -> ascending-by-`from` segment list. Empty means no value was split,
    // so `locationAt` falls back to `reg`/`spill` and emission is byte-identical to before splitting.
    segments: std.AutoHashMapUnmanaged(Value, []Segment) = .empty,
    // Stores/reloads to emit at split boundaries, appended in monotonically-ascending `at` order.
    // Empty means no value was split, so emission is byte-identical to before splitting.
    actions: std.ArrayList(SplitAction) = .empty,
    // Precomputed, ordered control-flow-edge moves (the shared Wimmer path only). `edge_move_driven`
    // marks this as an edge-move-authoritative allocation: emission then replays these per edge and
    // never derives block-param moves itself. Empty + false is the DEFAULT `compileFunction` path,
    // whose edge lowering stays byte-identical (it keeps deriving block-param moves).
    edge_moves: []EdgeMoveSet = &.{},
    edge_move_driven: bool = false,
    def_pos: []u32 = &.{}, // per value: its definition position (copied from Liveness, and the emission assert reads it)
    saved_gpr: std.ArrayList(Reg) = .empty, // callee-saved x-registers used (non-leaf)
    saved_fpr: std.ArrayList(Reg) = .empty, // callee-saved v-registers used (non-leaf)
    spill_count: u32 = 0,

    fn deinit(self: *Allocation, allocator: std.mem.Allocator) void {
        self.reg.deinit(allocator);
        self.spill.deinit(allocator);
        var seg_it = self.segments.valueIterator();
        while (seg_it.next()) |segs| allocator.free(segs.*);
        self.segments.deinit(allocator);
        self.actions.deinit(allocator);
        for (self.edge_moves) |em| allocator.free(em.moves);
        allocator.free(self.edge_moves);
        // `def_pos` is always a heap-owned dupe (the `&.{}` sentinel is a zero-length slice with no
        // backing allocation, and freeing that is a no-op), so an unconditional free is safe.
        allocator.free(self.def_pos);
        self.saved_gpr.deinit(allocator);
        self.saved_fpr.deinit(allocator);
    }
};

/// Capabilities a model-aware call site threads into `compileFunction`. Grouped into one struct
/// (rather than growing `compileFunction`'s parameter list one flag per model feature) so adding
/// the next capability never touches every existing call site. `.{}` is exactly today's behavior
/// for every non-model caller (`selectFunction`, `selectFunctionWithLines`, and the direct
/// `compileFunction` callers in `link.zig`/`object.zig`): no loop-header alignment padding, the
/// base-ISA f16 EMULATION (an f16 held as its f32 widening in an S register, rounded per-op with
/// `fcvt`), and the `fuse_*` flags at their base-ISA-available defaults. `fuse_cmp_branch` gates
/// the compare-into-branch fold (see `fusesIntoNextIf`); the rest are foundation only (no fold
/// reads them yet, so they are inert either way).
pub const ModelCaps = struct {
    /// Loop-header alignment in bytes (0 disables it). See `compileFunction`'s doc comment.
    fetch_align: u16 = 0,
    /// Use NATIVE half-precision arithmetic (H-form ops, an f16 held in an H register, single-
    /// rounded, no per-op widen/narrow) instead of the emulation. Set only when the target model's
    /// `features.aarch64.fp16` (FEAT_FP16) is true (see `selectFunctionForModel`). When false the
    /// f16 lowering is byte-identical to the pre-FEAT_FP16 emulation.
    fp16: bool = false,
    /// Fuse a compare into its consumer branch (CMP+Bcc -> a single conditional branch on flags).
    /// Base-ISA on every aarch64 core, so true by default. Gates `fusesIntoNextIf` (see its doc
    /// comment); false falls back to materializing the boolean with `cset` then testing it with
    /// `cbnz`, exactly as if the icmp/if pair were never eligible to fuse.
    fuse_cmp_branch: bool = true,
    /// Fuse an arithmetic op's flag-setting form into its consumer branch (e.g. ADDS+Bcc). Base-
    /// ISA on every aarch64 core, so true by default. Gates `Ctx.fuse_arith_branch`, which the
    /// arith-branch fold (`fusesArithIntoBranch`) reads at both the arith-skip site and the S-form
    /// emit site; false falls back to materializing the arith plainly and keeping the separate
    /// `cmp` in the fused compare-and-branch path, exactly as if the arith/icmp/if triple were
    /// never eligible to fold.
    fuse_arith_branch: bool = true,
    /// Fuse a shift into a following add (ADD Xd, Xn, Xm, LSL #imm). Base-ISA on every aarch64
    /// core, so true by default. No fold reads this yet.
    fuse_shift_add: bool = true,
    /// Fuse a high/low address-pair computation (ADRP+ADD) into one microarch-recognized macro-op.
    /// Microarch-specific (not every core's front end recognizes the pair), so off unless the
    /// model declares it. No fold reads this yet.
    fuse_addr_hi_lo: bool = false,
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
/// (loop-header alignment), `features.aarch64.fp16` (whether to use native FEAT_FP16 half
/// arithmetic instead of the emulation), and `fuse_cmp_branch` (whether the compare-into-branch
/// fold runs). An inert model (fetch_align 0, fp16 false, fuse_cmp_branch true - the base-ISA
/// default) makes this byte-identical to `selectFunction`. Builds the full `ModelCaps` and calls
/// `compileFunction` directly rather than through `selectFunctionAligned`, since that narrower
/// entry point only ever carries `fetch_align`.
pub fn selectFunctionForModel(allocator: std.mem.Allocator, func: *const Function, model: *const mm.Model) Error![]u32 {
    // Passing a foreign-arch model here is a caller bug, not a runtime fault.
    std.debug.assert(model.arch == .aarch64);
    const compiled = try compileFunction(allocator, func, capsForModel(model));
    allocator.free(compiled.relocs);
    allocator.free(compiled.lines);
    return compiled.code;
}

/// The `ModelCaps` `selectFunctionForModel` builds for `model`. Split out so the model-to-caps
/// mapping is unit-testable without compiling a whole function. Asserts `model.arch == .aarch64`,
/// same as the caller above.
pub fn capsForModel(model: *const mm.Model) ModelCaps {
    std.debug.assert(model.arch == .aarch64);
    return .{
        .fetch_align = model.fetch_align,
        .fp16 = model.arch == .aarch64 and model.features.aarch64.fp16,
        .fuse_cmp_branch = model.fuses(.cmp_branch),
        .fuse_arith_branch = model.fuses(.arith_branch),
        .fuse_shift_add = model.fuses(.shift_add),
        .fuse_addr_hi_lo = model.fuses(.addr_hi_lo),
    };
}

test "capsForModel reads ampere-altra's fusion table: cmp/arith/shift on, addr_hi_lo off" {
    const caps = capsForModel(mm.modelFor(.@"ampere-altra"));
    try std.testing.expect(caps.fuse_cmp_branch);
    try std.testing.expect(caps.fuse_arith_branch);
    try std.testing.expect(caps.fuse_shift_add);
    try std.testing.expect(!caps.fuse_addr_hi_lo);
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

/// Rewrite `work` in place so address folding is SOUND under the fold-agnostic Wimmer allocator.
/// The shared allocator sees only the actual IR operands, so for a foldable
/// `p = arith_imm.add(base, imm); load(p)` it keeps `p` live to the load and lets `base` die at the
/// add, then reuses base's register - and emitting the fold (`[base, #imm]`, add dropped) would read a
/// stale register. This rewrite makes the fold visible to the allocator INSTEAD of hiding it:
///   1. Repoint every folded load/store's `ptr` operand directly to its fold BASE, so the shared
///      `buildIntervals` sees `base` used AT the load/store position and keeps its live range correct.
///   2. Drop every now-dead address-add (its result had no use left once the folded ptr uses moved to
///      the base), so the allocator wastes no register on it.
/// `fold` stays consistent for emission: `folds` is keyed by the SURVIVING mem inst and holds the
/// base+off, so `baseOf` returns the (now raw) ptr = base and `offOf` the displacement. Only dead
/// adds are removed, never a mem inst, so the offsets survive. Runs on the CLONE, after critical-edge
/// splitting and BEFORE `wimmer.allocate`. Sound cross-block: base dominated the add and the add
/// dominated the load, so base dominates the load.
fn applyFoldRewrite(func: *Function, fold: *const addrfold.Analysis) void {
    var it = fold.folds.iterator();
    while (it.next()) |entry| {
        const mem_inst = entry.key_ptr.*;
        const base = entry.value_ptr.base;
        const op = func.opcodeMut(mem_inst);
        switch (op.*) {
            .load => |*l| l.ptr = base,
            .store => |*st| st.ptr = base,
            else => unreachable, // folds only ever holds a load or store
        }
    }
    // Drop the dead adds. A dead add's every use was a folded ptr use now repointed to the base, so its
    // result is unused. Assert that (a surviving use would mean dropping a live def = a miscompile)
    // before removing it. Removal order is irrelevant: no dead add's result feeds another instruction.
    for (0..func.blockCount()) |bi| {
        const list = func.blockInstsMut(@enumFromInt(bi));
        var i: usize = 0;
        while (i < list.items.len) {
            const inst = list.items[i];
            if (!fold.isDeadAdd(inst)) {
                i += 1;
                continue;
            }
            const result = func.instResult(inst).?; // an arith_imm always defines a result
            std.debug.assert(countUses(func, result) == 0);
            _ = list.orderedRemove(i); // the next inst slides into i, so do not advance
        }
    }
}

/// Compile `func` to A64 words and call relocations. The caller owns the result.
/// `fetch_align` is the microarch model's fetch granularity in bytes (0 disables loop-header
/// alignment, the behavior of every existing caller). When greater than one instruction word,
/// each loop-header block is padded with nops up to a `fetch_align` boundary before its code is
/// emitted, so a hot loop's fetch groups pack efficiently. This is purely a placement hint: the
/// padding falls straight through into the header and every branch fixup is patched from
/// `block_start` (recorded after padding), so it can never change what the function computes.
///
/// The register allocator is the SHARED Wimmer-Franz linear-scan-on-SSA allocator (Wimmer cutover
/// SP1, the production flip): the scan runs on the SPLIT CFG, its target-independent `Allocation`
/// is translated into this backend's own representation by `translateAllocation`, and the
/// battle-tested `emitFromAllocation` turns it into machine code. There is NO fallback to the
/// retired native linear scan.
pub fn compileFunction(allocator: std.mem.Allocator, func: *const Function, caps: ModelCaps) Error!Compiled {
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
    if (func.blockCount() == 0) return error.Unsupported;

    // The Wimmer resolver needs a block on every critical edge to place its shuffle moves, so split
    // them FIRST, before any numbering is built. `splitCriticalEdges` MUTATES the function (it appends
    // forwarding blocks and retargets `if` edges), but the public entry points are `*const Function`
    // and a caller may reuse or SHARE its function across backends (the wasm-vs-aarch64 differential
    // does exactly this), so we must not touch the input. We split on an independently-owned deep
    // `clone` and compile that; the caller's function is left byte-for-byte pristine. This keeps every
    // public *const signature (`selectFunction`, `selectFunctionForModel`, link/object) unchanged.
    var work = try func.clone(allocator);
    defer work.deinit();
    try ir.critical_edge.splitCriticalEdges(allocator, &work);

    // Address-mode folding is a PRE-ALLOCATION IR REWRITE so it is SOUND under the fold-agnostic
    // shared Wimmer allocator (which sees only the actual IR operands and keeps a frozen `wimmer.zig`).
    // `analyze` recognizes each foldable `p = arith_imm.add(base, imm); load/store(p)` and which adds
    // die once folded; `applyFoldRewrite` then repoints each folded mem op's `ptr` to `base` and drops
    // the dead adds IN THE CLONE. After that the allocator sees `base` used at the load/store and keeps
    // its live range correct, so emitting `[base, #off]` reads the right register. The SAME analysis is
    // threaded to `emitFromAllocation`: `folds` is keyed by the surviving mem inst, so `baseOf` returns
    // the (now raw) ptr = base and `offOf` the displacement, staying consistent with the rewritten IR.
    var fold = try addrfold.analyze(allocator, &work, {}, aarch64FoldOffset);
    defer fold.deinit(allocator);
    applyFoldRewrite(&work, &fold);

    var desc = try aarch64RegDescription(allocator, &work);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, &work, &desc);
    defer walloc.deinit(allocator);

    var alloc = try translateAllocation(allocator, &work, &walloc);
    defer alloc.deinit(allocator);

    // The MODEL caps (loop-header alignment, macro-op fusion, native f16) are allocation-agnostic
    // emission seams and ARE threaded: the differential corpus proved the fusions execution-equivalent
    // under the Wimmer allocation, and alignment/f16 only change placement/encoding of an already-
    // correct allocation.
    var compiled = try emitFromAllocation(allocator, &work, caps, &alloc, &fold);
    errdefer compiled.deinit(allocator);

    // Every `Reloc.symbol` is a BORROWED slice into the emitting function's symbol storage; the
    // contract is that those names outlive `Compiled` (the caller's function does). Emission borrowed
    // them from the CLONE, whose storage `work.deinit` frees on return, so re-point each name to the
    // original `func`'s identical, longer-lived symbol string (the clone re-interned symbols 1:1, so
    // the same name exists there). This keeps the borrowed-name contract intact without making the
    // whole `Compiled` own its names.
    for (compiled.relocs) |*r| r.symbol = rebindSymbolName(func, r.symbol);
    return compiled;
}

/// The `func`-owned symbol string equal to `name`. Every emitted relocation names a callee the
/// function interned (a `call`'s `symbol` indexes `func.symbols`), so a match always exists; a miss
/// would be a codegen bug, not a runtime condition.
fn rebindSymbolName(func: *const Function, name: []const u8) []const u8 {
    var i: u32 = 0;
    while (i < func.symbolCount()) : (i += 1) {
        const s = func.symbolName(i);
        if (std.mem.eql(u8, s, name)) return s;
    }
    unreachable;
}

/// Emit one split-boundary action's machine code: a `store` writes `reg` to its slot, a `reload`
/// brings a slot back into `reg`, a `move` copies `move_from` into `reg` (register re-home), and a
/// `slot_to_slot` re-homes a spilled value from `move_from_slot` to `slot` without ever giving it a
/// value register (Wimmer bridge gap #7). The store/reload width follows the value type (a SIMD
/// vector round-trips all 128 bits, a scalar FP the low 64, a GPR the low 64), exactly as
/// `loadOp`/`storeResult` do, so a stored value always reloads whole. Shared by the per-instruction
/// and terminator drains in `emitFromAllocation`. The native `allocate` only ever produces
/// `store`/`reload`, so those two arms are byte-identical to before this extraction; `move` and
/// `slot_to_slot` are reachable only through the shared Wimmer translation.
fn emitSplitAction(allocator: std.mem.Allocator, code: *std.ArrayList(u32), func: *const Function, spill_base: usize, act: SplitAction) Error!void {
    switch (act.kind) {
        .store => {
            const off: u15 = @intCast(spill_base + act.slot * 16);
            if (isVector(func, act.value)) {
                try code.append(allocator, encode.strQ(act.reg, sp, off));
            } else if (regClass(func, act.value) == .fpr) {
                try code.append(allocator, encode.strFp(act.reg, sp, off, true));
            } else {
                try code.append(allocator, encode.strOff(act.reg, sp, off));
            }
        },
        .reload => {
            const off: u15 = @intCast(spill_base + act.slot * 16);
            if (isVector(func, act.value)) {
                try code.append(allocator, encode.ldrQ(act.reg, sp, off));
            } else if (regClass(func, act.value) == .fpr) {
                try code.append(allocator, encode.ldrFp(act.reg, sp, off, true));
            } else {
                try code.append(allocator, encode.ldrOff(act.reg, sp, off));
            }
        },
        .move => {
            // A register-to-register re-home (Wimmer path only): copy `move_from` into `reg`. Same
            // value, so the copy width follows the value type. An identity move emits nothing.
            if (act.move_from == act.reg) return;
            if (isVector(func, act.value)) {
                try code.append(allocator, encode.movVec(act.reg, act.move_from));
            } else if (regClass(func, act.value) == .fpr) {
                try code.append(allocator, encode.fmovReg(act.reg, act.move_from));
            } else {
                try code.append(allocator, encode.mov(act.reg, act.move_from));
            }
        },
        .slot_to_slot => {
            // A spill-to-spill re-home: reload `move_from_slot` into the class scratch (the SAME
            // reserved register `aarch64RegDescription` keeps out of every pool: `scratch_move`/x17
            // for gpr, `fp_move`/v27 for fpr, so it never collides with a live value), then store the
            // scratch straight back out to `slot`. The shared allocator now expands slot->slot moves
            // through the class scratch itself (`orderMoves`), so the Wimmer path no longer produces
            // this kind; the arm stays for defensive completeness (the native `allocate` never emits
            // it either). The scratch is reserved, so this touch can never conflict with a live value.
            const off_src: u15 = @intCast(spill_base + act.move_from_slot * 16);
            const off_dst: u15 = @intCast(spill_base + act.slot * 16);
            if (isVector(func, act.value)) {
                try code.append(allocator, encode.ldrQ(fp_move, sp, off_src));
                try code.append(allocator, encode.strQ(fp_move, sp, off_dst));
            } else if (regClass(func, act.value) == .fpr) {
                try code.append(allocator, encode.ldrFp(fp_move, sp, off_src, true));
                try code.append(allocator, encode.strFp(fp_move, sp, off_dst, true));
            } else {
                try code.append(allocator, encode.ldrOff(scratch_move, sp, off_src));
                try code.append(allocator, encode.strOff(scratch_move, sp, off_dst));
            }
        },
    }
}

/// Emit A64 machine code from a finished `Allocation` (the second half of `compileFunction`, split
/// out so the shared Wimmer allocator can drive the SAME battle-tested emission through
/// `compileFunctionWimmer`). This is a PURE extraction of everything after `allocate`: prologue and
/// frame, the per-block/instruction loop reading each value's location through `Ctx.locationAt`, the
/// split-boundary action drain, block-edge moves, and the epilogue. `caps` carries the model seams
/// (`fetch_align`, `fp16`); an inert `.{}` reproduces `selectFunction`'s output exactly. `nblocks`
/// and `leaf` are recomputed from `func` (identical to the values `compileFunction` passed to
/// `allocate`), so the extraction is byte-identical for every existing caller.
fn emitFromAllocation(allocator: std.mem.Allocator, func: *const Function, caps: ModelCaps, alloc: *Allocation, fold: *const addrfold.Analysis) Error!Compiled {
    const fetch_align = caps.fetch_align;
    // Native half arithmetic (FEAT_FP16) vs the base-ISA emulation, gated on the model feature.
    // `fp16 == false` (every non-model caller) keeps the emulation, byte-identical to before this
    // capability existed; only `selectFunctionForModel` under a FEAT_FP16 model sets it true.
    const fp16 = caps.fp16;
    const nblocks = func.blockCount();
    const leaf = isLeaf(func);

    // Split-boundary actions must drain in ascending `at` with each same-position cluster in a
    // hazard-free order. The edge-move-driven (shared Wimmer) allocation ALREADY delivers exactly
    // that: `wimmer.allocate` orders every same-position cluster as a parallel move and the bridge
    // appends the clusters in ascending `at`, so re-sorting here (an unstable sort) would only risk
    // scrambling that resolution. Leave it untouched. The native `allocate` appends in monotonic `at`
    // order too but does NOT parallel-move-order a cluster, so it keeps the defensive sort whose
    // comparator breaks `at` ties on kind (a `.reload` before a `.store`: a value reloaded slot->reg
    // and immediately re-spilled reg->slot at one use position must reload first, or the store saves a
    // stale register).
    if (!alloc.edge_move_driven) {
        std.mem.sort(SplitAction, alloc.actions.items, {}, struct {
            fn order(k: @TypeOf(@as(SplitAction, undefined).kind)) u8 {
                return switch (k) {
                    .reload => 0,
                    .move => 1,
                    .store => 2,
                    // Touches only the class scratch (never a value register or a slot any other action
                    // targets, since every spilled interval owns a distinct slot index), so its position
                    // relative to reload/move/store at the same `at` is a don't-care for correctness.
                    .slot_to_slot => 3,
                };
            }
            fn f(_: void, a: SplitAction, b: SplitAction) bool {
                if (a.at != b.at) return a.at < b.at;
                return order(a.kind) < order(b.kind);
            }
        }.f);
    }

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
        // A SPLIT entry param (Wimmer bridge gap #4): under pressure the shared allocator can
        // re-home a param's interval one or more times, leaving its first home in `segments`
        // instead of `alloc.reg`/`alloc.spill` (see `translateAllocation`, which builds the
        // per-transition drain actions for every segment boundary from `segments[0]` onward,
        // exactly like any other split value). Nothing establishes `segments[0]` itself though:
        // unlike a computed value (whose defining instruction writes straight into its first
        // segment), a param's "definition" is the ABI calling convention, so establish it here
        // from the incoming argument into its first home: a register (mov/fmov) or a stack slot
        // (str). This runs regardless of `leaf`: a genuinely split param is not the "hint always
        // honored" shape the whole-life LEAF check below assumes, so it needs the same move a
        // non-leaf whole-life re-home gets. Now that the shared Wimmer allocator is the production
        // allocator, this branch IS reachable from the default `compileFunction` path (a param whose
        // interval is evicted under pressure), not just the differential path.
        if (alloc.segments.get(p)) |segs| {
            // `regClass` maps BOTH scalar float and 128-bit vectors to `.fpr`, so every fpr sub-branch
            // must pick its op width by `isVector`: a vector needs the 128-bit form (`movVec`/`ldrQ`/
            // `strQ`) or lanes 2-3 are dropped, a scalar float the 64-bit form (`fmovReg`/`ldrFp`/
            // `strFp`). This mirrors the whole-life-spilled-param branch below and `emitSplitAction`'s
            // `.move` arm exactly. The gpr paths are width-agnostic (`mov`/`ldrOff`/`strOff`).
            const vec = isVector(func, p);
            switch (segs[0].loc) {
                .reg => |first_reg| {
                    if (l.idx < 8) {
                        const incoming: Reg = @enumFromInt(@as(u5, @intCast(l.idx)));
                        if (first_reg != incoming) {
                            try code.append(allocator, if (l.class == .fpr)
                                (if (vec) encode.movVec(first_reg, incoming) else encode.fmovReg(first_reg, incoming))
                            else
                                encode.mov(first_reg, incoming));
                        }
                    } else if (l.class == .fpr) {
                        try code.append(allocator, if (vec) encode.ldrQ(first_reg, sp, @intCast(frame + 8 * (l.idx - 8))) else encode.ldrFp(first_reg, sp, @intCast(frame + 8 * (l.idx - 8)), false));
                    } else {
                        try code.append(allocator, encode.ldrOff(first_reg, sp, @intCast(frame + 8 * (l.idx - 8))));
                    }
                },
                .slot => |slot| {
                    const slot_off: u15 = @intCast(spill_base + slot * 16);
                    if (l.idx < 8) {
                        const incoming: Reg = @enumFromInt(@as(u5, @intCast(l.idx)));
                        if (l.class == .fpr) {
                            try code.append(allocator, if (vec) encode.strQ(incoming, sp, slot_off) else encode.strFp(incoming, sp, slot_off, true));
                        } else {
                            try code.append(allocator, encode.strOff(incoming, sp, slot_off));
                        }
                    } else if (l.class == .fpr) {
                        if (vec) {
                            try code.append(allocator, encode.ldrQ(fp_spill_op[0], sp, @intCast(frame + 8 * (l.idx - 8))));
                            try code.append(allocator, encode.strQ(fp_spill_op[0], sp, slot_off));
                        } else {
                            try code.append(allocator, encode.ldrFp(fp_spill_op[0], sp, @intCast(frame + 8 * (l.idx - 8)), false));
                            try code.append(allocator, encode.strFp(fp_spill_op[0], sp, slot_off, true));
                        }
                    } else {
                        try code.append(allocator, encode.ldrOff(spill_op[0], sp, @intCast(frame + 8 * (l.idx - 8))));
                        try code.append(allocator, encode.strOff(spill_op[0], sp, slot_off));
                    }
                },
            }
            continue;
        }
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

    var ctx = Ctx{
        .func = func,
        .alloc = alloc,
        .spill_base = spill_base,
        .alloca_base = alloca_base,
        .alloca_off = &alloca_off,
        .fp16 = fp16,
        .fuse_cmp_branch = caps.fuse_cmp_branch,
        .fuse_arith_branch = caps.fuse_arith_branch,
        .pos = 0,
    };

    // `pos` mirrors `linearize`'s position counter EXACTLY so `ctx.pos` (set per instruction below)
    // matches the position each value's location was computed for. Per block: the block-parameter row
    // occupies one position, then each instruction one position, and the terminator shares the
    // block-end position (`block_end[bi]`), after which one final increment lands on the next block.
    var pos: u32 = 0;
    var action_cursor: usize = 0;
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        ctx.block = block; // the predecessor for any edge this block terminates in (edge-move keying)
        if (fetch_align > 4 and is_loop_header[bi]) {
            var pad = alignPadWords(code.items.len, fetch_align);
            while (pad > 0) : (pad -= 1) try code.append(allocator, encode.nop());
        }
        block_start[bi] = code.items.len;
        var terminated = false;

        pos += 1; // the block-parameter row. `pos` is now the first instruction's position
        const insts = func.blockInsts(block);
        for (insts, 0..) |inst, inst_idx| {
            // Set the current position from the block base plus the instruction index (robust to the
            // `continue`s below, which a trailing increment would skip). This equals what `linearize`
            // numbered this instruction, so the assert below pins the two numberings together.
            ctx.pos = pos + @as(u32, @intCast(inst_idx));
            // Record a source-line row when this instruction begins a new line (its
            // `debug.line` attribute differs from the previous instruction's).
            if (lineOf(func, inst)) |line| {
                if (line != last_line) {
                    try lines.append(allocator, .{ .offset = @intCast(code.items.len * 4), .line = line });
                    last_line = line;
                }
            }
            // The pos coupling is otherwise unobservable while `segments` is empty, so assert it now:
            // an instruction with a result must be emitted at exactly that result's def position.
            if (func.instResult(inst)) |r| std.debug.assert(ctx.pos == alloc.def_pos[@intFromEnum(r)]);
            // Drain split-boundary actions landing at this position BEFORE emitting the instruction.
            // A tail-split store writes the victim's register to its slot before the taker `iv` (the
            // instruction defined at `p`) computes its result into that same register. The victim's
            // value is still in the register here (its last prefix use is before `p`, and nothing
            // reused the register before `p`), so the store captures the correct bits.
            while (action_cursor < alloc.actions.items.len and alloc.actions.items[action_cursor].at <= ctx.pos) {
                const act = alloc.actions.items[action_cursor];
                std.debug.assert(act.at == ctx.pos); // actions land on instruction positions only
                try emitSplitAction(allocator, &code, func, spill_base, act);
                action_cursor += 1;
            }
            switch (func.opcode(inst)) {
                .iconst => |c| {
                    const result = func.instResult(inst).?;
                    const rd = ctx.resultReg(result);
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
                    const rd = ctx.resultReg(result);
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
                    // Shift-add fold: when the previous inst is the single-use, immediately-
                    // preceding constant shift this add/sub consumes, emit ONE shifted-register
                    // add/sub (`add xd, xn, xm, lsl #k`) on the shift's operand instead of a
                    // separate shift + add. The shl's own materialization was skipped in the
                    // `.arith_imm` arm under the SAME `fusesIntoNextShiftAdd` gate, so the shift
                    // is emitted exactly once (mirrors the fma/icmp fusions). An add is at most
                    // one of fma / shift-add / plain: the previous inst is a mul, a shl, or
                    // neither, so this is checked after the fma case and before the plain form.
                    if ((a.op == .add or a.op == .sub) and inst_idx >= 1 and fusesIntoNextShiftAdd(func, insts, inst_idx - 1, caps.fuse_shift_add)) {
                        const shl = func.opcode(insts[inst_idx - 1]).arith_imm;
                        const shl_result = func.instResult(insts[inst_idx - 1]).?;
                        try ctx.emitShiftAdd(allocator, &code, result, a.op, a.lhs, a.rhs, shl, shl_result);
                        continue;
                    }
                    // Arith-branch fold: this add/sub/and is the single-use operand of an
                    // immediately-following icmp eq/ne 0 that feeds the next if. Skip its
                    // materialization here; `emitIf` re-checks the SAME `fusesArithIntoBranch`
                    // predicate and emits the flag-setting S-form (adds/subs/ands) whose Z flag is
                    // (result == 0) plus the `b.cc`, so the arith is emitted exactly once. Checked
                    // after the fma/shift consumer folds (an integer add is never an fma consumer,
                    // and a shift_add consumer is rejected by the predicate, so this never doubles).
                    if (isArithBranchProducer(func, insts, inst_idx, caps.fuse_arith_branch and caps.fuse_cmp_branch)) continue;
                    try ctx.binary(allocator, &code, result, a.op, a.lhs, a.rhs);
                },
                .arith_imm => |a| {
                    // A folded address-add is dead: every use of its result was rerouted to the base
                    // by the fold, so it claims no register and emits nothing (mirrors the mul/icmp
                    // fusion skips). `ctx.pos` still advances from `inst_idx`, so numbering holds.
                    if (fold.isDeadAdd(inst)) continue;
                    // Skip a constant shift that folds into the next add/sub as a shifted-
                    // register operand (the fused `add xd, xn, xm, lsl #k` is emitted at the
                    // consumer in the `.arith` arm). Same `fusesIntoNextShiftAdd` gate as that
                    // emit site, so the shift is never both skipped-and-unfused nor fused-and-
                    // kept.
                    if (a.op == .shl and fusesIntoNextShiftAdd(func, insts, inst_idx, caps.fuse_shift_add)) continue;
                    // Arith-branch fold (immediate form): this add/sub of a u12 immediate is the
                    // single-use operand of an icmp eq/ne 0 feeding the next if. Skip it; `emitIf`
                    // emits the flag-setting `addsImm`/`subsImm` + `b.cc` under the SAME predicate.
                    if (isArithBranchProducer(func, insts, inst_idx, caps.fuse_arith_branch and caps.fuse_cmp_branch)) continue;
                    const result = func.instResult(inst).?;
                    const rl = try ctx.loadOp(allocator, &code, a.lhs, spill_op[0]);
                    try loadConst(allocator, &code, spill_op[1], a.imm); // imm in x14, x16 stays free for rem
                    const rd = ctx.resultReg(result);
                    try emitBinary(allocator, &code, a.op, rd, rl, spill_op[1], isSignedInt(func, a.lhs), isWide(func, result));
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .icmp => |cmp| {
                    // Fused compare-and-branch: when this integer icmp is the single-use
                    // condition of the immediately-following if, skip its `cmp; cset`
                    // materialization entirely. `emitIf` re-checks the SAME predicate and
                    // emits the fused `cmp; b.cc` on these operands, so the compare is
                    // emitted exactly once.
                    if (caps.fuse_cmp_branch and fusesIntoNextIf(func, insts, inst_idx)) continue;
                    const result = func.instResult(inst).?;
                    if (isVector(func, cmp.lhs)) {
                        // Vectorized compare (widened FS): produce a per-lane MASK
                        // (all-ones where the relation holds, else all-zeros) in a
                        // <4 x f32>-typed result. <, <= are >, >= with operands swapped.
                        // != is the bitwise complement of ==.
                        const rl = try ctx.loadOp(allocator, &code, cmp.lhs, fp_spill_op[0]);
                        const rr = try ctx.loadOp(allocator, &code, cmp.rhs, fp_spill_op[1]);
                        const rd = ctx.resultReg(result);
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
                        const rd = ctx.resultReg(result);
                        // Native f16 (fp16) compares in the H form; emulation compares the S-held
                        // f32 widening (fkindOf yields `.single`, byte-identical to before).
                        try code.append(allocator, encode.fcmp(rl, rr, fkindOf(func, cmp.lhs, fp16)));
                        try code.append(allocator, encode.cset(rd, condForFloat(cmp.op)));
                        try storeResult(allocator, &code, ctx, result, rd);
                    } else {
                        const rl = try ctx.loadOp(allocator, &code, cmp.lhs, spill_op[0]);
                        const rr = try ctx.loadOp(allocator, &code, cmp.rhs, spill_op[1]);
                        const rd = ctx.resultReg(result);
                        // Compare at the operand width: a 32-bit cmp on an i64/ptr operand would
                        // only test the low 32 bits, mismeasuring values whose high bits differ.
                        const wide = isWide(func, cmp.lhs);
                        try code.append(allocator, if (wide) encode.cmp64(rl, rr) else encode.cmp(rl, rr));
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
                        const rd = ctx.resultReg(result);
                        if (@intFromEnum(rd) != @intFromEnum(fp_move)) try code.append(allocator, encode.movVec(rd, fp_move));
                        try storeResult(allocator, &code, ctx, result, rd);
                        continue;
                    }
                    const c = try ctx.loadOp(allocator, &code, s.cond, spill_op[0]); // cond is a gpr bool
                    if (regClass(func, result) == .fpr) {
                        const tr = try ctx.loadOp(allocator, &code, s.then, fp_spill_op[0]);
                        const el = try ctx.loadOp(allocator, &code, s.@"else", fp_spill_op[1]);
                        const rd = ctx.resultReg(result);
                        try code.append(allocator, encode.cmp(c, .zr));
                        // Native f16 (fp16) selects in the H form; emulation selects the S-held
                        // f32 widening (fkindOf yields `.single`, byte-identical to before).
                        try code.append(allocator, encode.fcsel(rd, tr, el, .ne, fkindOf(func, result, fp16)));
                        try storeResult(allocator, &code, ctx, result, rd);
                    } else {
                        const tr = try ctx.loadOp(allocator, &code, s.then, spill_op[1]);
                        const el = try ctx.loadOp(allocator, &code, s.@"else", scratch_imm);
                        const rd = ctx.resultReg(result);
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
                    const rd = ctx.resultReg(result);
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
                    const rd = ctx.resultReg(result);
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
                    const rd = ctx.resultReg(result);
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
                    const rd = ctx.resultReg(result);
                    try emitFrameImm(allocator, &code, false, rd, sp, off);
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .load => {
                    // A folded load addresses `[base, #off]` directly: `baseOf` yields the fold base
                    // (the add's lhs) and `offOf` the displacement; both are the raw ptr and 0 when
                    // unfolded, so the non-folding case is byte-identical.
                    const result = func.instResult(inst).?;
                    const base = try ctx.loadOp(allocator, &code, fold.baseOf(func, inst), spill_op[0]); // ptr is a gpr
                    const rd = ctx.resultReg(result);
                    try emitLoad(allocator, &code, func, result, rd, base, @intCast(fold.offOf(inst)), fp16);
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .store => |st| {
                    const val = try ctx.loadOp(allocator, &code, st.value, if (regClass(func, st.value) == .fpr) fp_spill_op[0] else spill_op[0]);
                    const base = try ctx.loadOp(allocator, &code, fold.baseOf(func, inst), spill_op[1]);
                    try emitStore(allocator, &code, func, st.value, val, base, @intCast(fold.offOf(inst)), fp16);
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
                    const rd = ctx.resultReg(result);
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
                    const rd = ctx.resultReg(result);
                    // A splat (every lane the same scalar) is one `dup` from that scalar's lane 0,
                    // not four inserts. This is what makes the vectorizer's invariant operand (e.g.
                    // the SAXPY multiplier) cheap enough to fuse on a wide core.
                    const splat = for (fields[1..]) |f| {
                        if (f != fields[0]) break false;
                    } else true;
                    if (splat) {
                        const fr = try ctx.loadOp(allocator, &code, fields[0], fp_spill_op[0]);
                        try code.append(allocator, encode.dupVecLane(rd, fr, 0));
                    } else {
                        for ([_]u2{ 1, 2, 3, 0 }) |lane| {
                            const fr = try ctx.loadOp(allocator, &code, fields[lane], fp_spill_op[0]);
                            try code.append(allocator, encode.insLane(rd, lane, fr));
                        }
                    }
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                .call => |c| {
                    try ctx.emitCallArgs(allocator, &code, func.valueList(c.args));
                    try relocs.append(allocator, .{ .offset = code.items.len, .symbol = func.symbolName(c.symbol) });
                    try code.append(allocator, encode.bl(0));
                    if (func.instResult(inst)) |result| {
                        const rd = ctx.resultReg(result);
                        try code.append(allocator, if (regClass(func, result) == .fpr) encode.fmovReg(rd, @enumFromInt(0)) else encode.mov(rd, .x0));
                        try storeResult(allocator, &code, ctx, result, rd);
                    }
                },
                .call_indirect => |c| {
                    // Stage the target address into a scratch register that survives the
                    // argument moves (x0..x7), then `blr`.
                    const tsrc = try ctx.loadOp(allocator, &code, c.target, spill_op[0]);
                    try code.append(allocator, encode.mov(scratch_imm, tsrc));
                    try ctx.emitCallArgs(allocator, &code, func.valueList(c.args));
                    try code.append(allocator, encode.blr(scratch_imm));
                    if (func.instResult(inst)) |result| {
                        const rd = ctx.resultReg(result);
                        try code.append(allocator, if (regClass(func, result) == .fpr) encode.fmovReg(rd, @enumFromInt(0)) else encode.mov(rd, .x0));
                        try storeResult(allocator, &code, ctx, result, rd);
                    }
                },
                .@"if" => |cf| {
                    // The block emitted immediately after this one, in array order. An `if` edge to it
                    // can fall through (elide its branch) instead of jumping around one instruction.
                    const next_block: ?Block = if (bi + 1 < nblocks) @enumFromInt(bi + 1) else null;
                    try ctx.emitIf(allocator, &code, &fixups, cf, insts, inst_idx, next_block);
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
                    const rd = ctx.resultReg(result);
                    if (@intFromEnum(rd) != @intFromEnum(fp_move)) try code.append(allocator, encode.movVec(rd, fp_move));
                    try storeResult(allocator, &code, ctx, result, rd);
                },
                else => return error.Unsupported,
            }
        }

        // After the instruction loop `pos` is the block-end (terminator) position, exactly where
        // `linearize` numbers a jump/ret terminator's operands. An `if` terminator is one of the
        // instructions above (it set `terminated`), so this only positions ret/jump.
        pos += @intCast(insts.len);
        ctx.pos = pos;
        // Drain any split-boundary actions recorded AT the terminator position BEFORE emitting the
        // terminator. The shared allocator can re-home a value used only by `ret` (a non-edge-arg
        // operand) at its next use, which is the terminator position (`block_end`), recording
        // a `.reload` there. The per-instruction drain above only reaches `term_pos - 1`, so without
        // this drain the reload is never emitted and `ret` reads a register that was never loaded. When
        // no action lands on a terminator (the normal case) this drains nothing and is byte-identical.
        while (action_cursor < alloc.actions.items.len and alloc.actions.items[action_cursor].at <= ctx.pos) {
            const act = alloc.actions.items[action_cursor];
            std.debug.assert(act.at == ctx.pos); // only terminator-position actions remain here
            try emitSplitAction(allocator, &code, func, spill_base, act);
            action_cursor += 1;
        }
        if (!terminated) {
            switch (func.terminator(block) orelse Terminator{ .ret = null }) {
                .ret => |v| {
                    if (v) |value| {
                        // The return value is a non-param value, so its location is read through
                        // `locationAt` at the terminator position (ctx.pos == block_end here). With no
                        // splits this is byte-identical to today's direct `reg`/`spill` reads.
                        if (regClass(func, value) == .fpr) {
                            const vec = isVector(func, value);
                            switch (ctx.locationAt(value)) {
                                .slot => |slot| {
                                    const off: u15 = @intCast(spill_base + slot * 16);
                                    try code.append(allocator, if (vec) encode.ldrQ(@enumFromInt(0), sp, off) else encode.ldrFp(@enumFromInt(0), sp, off, true));
                                },
                                .reg => |src| if (@intFromEnum(src) != 0) try code.append(allocator, if (vec) encode.movVec(@enumFromInt(0), src) else encode.fmovReg(@enumFromInt(0), src)),
                            }
                        } else switch (ctx.locationAt(value)) {
                            .slot => |slot| try code.append(allocator, encode.ldrOff(.x0, sp, @intCast(spill_base + slot * 16))),
                            .reg => |src| if (src != .x0) try code.append(allocator, encode.mov(.x0, src)),
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
                .jump => |j| {
                    // The block emitted immediately after this one, in array order. A jump to it needs
                    // no branch (fall through), which `emitJump` elides.
                    const next_block: ?Block = if (bi + 1 < nblocks) @enumFromInt(bi + 1) else null;
                    try ctx.emitJump(allocator, &code, &fixups, j, next_block);
                },
            }
        }
        pos += 1; // the terminator's own position slot (mirrors linearize's per-block final increment)
    }

    // Every recorded split-boundary action must have been drained by a per-instruction or the
    // terminator-position drain above. A leftover action means an `at` no emission position reached: a
    // numbering bug that would silently drop a store/reload. Fail loudly instead of miscompiling.
    std.debug.assert(action_cursor == alloc.actions.items.len);

    // Fuse adjacent ldr/str pairs into ldp/stp and remap the side tables to the shrunk layout
    // BEFORE resolving fixups, so the branch-displacement loop below computes correct offsets
    // against the paired `code` and remapped `block_start`/`fixups`. A function with no fusable
    // pair is left byte-identical.
    try peephole.pairMemory(allocator, &code, block_start, fixups.items, relocs.items, lines.items);

    for (fixups.items) |f| {
        const off: i28 = @intCast((@as(i64, @intCast(block_start[f.target])) - @as(i64, @intCast(f.at))) * 4);
        code.items[f.at] = encode.b(off);
    }

    return .{
        .code = try code.toOwnedSlice(allocator),
        .relocs = try relocs.toOwnedSlice(allocator),
        .lines = try lines.toOwnedSlice(allocator),
    };
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
    /// Whether to fuse a compare into its consumer branch (`cmp; b.cc` instead of `cmp; cset` then
    /// a separate test-and-branch). Threaded from `compileFunction`'s `caps.fuse_cmp_branch`; true
    /// for every non-model caller (byte-identical to before this capability existed). Must agree
    /// with the icmp-skip site's `caps.fuse_cmp_branch` check in `emitFromAllocation` (the dual-check
    /// pair around `fusesIntoNextIf`), so the two never disagree on whether a given icmp/if fuses.
    fuse_cmp_branch: bool = true,
    /// Whether to fold a flag-setting arithmetic op into its consumer branch (the arith-branch
    /// fold: `arith; icmp ==/!= 0; if` -> one S-form `adds`/`subs`/`ands` plus `b.cc`). Threaded
    /// from `compileFunction`'s `caps.fuse_arith_branch`; true for every non-model caller. This
    /// fold lives INSIDE the fused compare-and-branch path, so it also requires `fuse_cmp_branch`.
    /// Must agree with the arith-skip site's `caps.fuse_arith_branch and caps.fuse_cmp_branch`
    /// check in `emitFromAllocation` (both gate `fusesArithIntoBranch`), so the two never disagree
    /// on whether a given arith/icmp/if triple folds (else a dangling or doubled arith).
    fuse_arith_branch: bool = true,
    /// The current instruction position, mirrored from the allocator's `linearize` numbering and
    /// advanced by the emission loop. `locationAt` reads it to pick a split value's active segment.
    pos: u32 = 0,
    /// The block currently being emitted (the PREDECESSOR of any edge this block terminates in).
    /// `emitMoves` reads it with the jump target to key into the precomputed `edge_moves`. Defaults
    /// to the entry block; the emission loop sets it per block.
    block: Block = @enumFromInt(0),

    /// The location of `v` at the current instruction position (`self.pos`): its active segment if
    /// `v` was split, otherwise its whole-life register or spill slot. With no splits (segments
    /// empty) this is exactly today's `reg`/`spill` lookup.
    fn locationAt(self: Ctx, v: Value) Location {
        if (self.alloc.segments.get(v)) |segs| {
            // `segs` is non-empty and ascending by `from`, with `segs[0].from` == v's def position,
            // so the last segment whose `from` is at or before `pos` is the active one.
            var chosen = segs[0];
            for (segs) |s| {
                if (s.from <= self.pos) chosen = s else break;
            }
            return chosen.loc;
        }
        if (self.alloc.reg.get(v)) |r| return .{ .reg = r };
        return .{ .slot = self.alloc.spill.get(v).? };
    }

    /// The register holding `v` at the current position: its assigned register, or reload a
    /// spilled value into `scratch` (a register of `v`'s class). A SIMD vector reloads all 128 bits.
    fn loadOp(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), v: Value, scratch: Reg) Error!Reg {
        switch (self.locationAt(v)) {
            .reg => |r| return r,
            .slot => |slot| {
                const off: u15 = @intCast(self.spill_base + slot * 16);
                if (isVector(self.func, v)) {
                    try code.append(allocator, encode.ldrQ(scratch, sp, off));
                } else if (regClass(self.func, v) == .fpr) {
                    try code.append(allocator, encode.ldrFp(scratch, sp, off, true));
                } else {
                    try code.append(allocator, encode.ldrOff(scratch, sp, off));
                }
                return scratch;
            },
        }
    }

    /// The register to compute `result` into at its def position: its assigned register, or the
    /// class spill scratch when it is (wholly) spilled there (the caller then stores it with
    /// `storeResult`). At a def position a split value's first segment is always `.reg`, so `.slot`
    /// here means a wholly-spilled value, identical to today's behavior.
    fn resultReg(self: Ctx, result: Value) Reg {
        return switch (self.locationAt(result)) {
            .reg => |r| r,
            .slot => if (regClass(self.func, result) == .fpr) fp_spill_res else spill_res,
        };
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
    fn emitFusedArith(
        self: Ctx,
        allocator: std.mem.Allocator,
        code: *std.ArrayList(u32),
        result: Value,
        op: ir.function.BinOp,
        lhs: Value,
        rhs: Value,
        mul: ir.function.Arith,
        mul_result: Value,
    ) Error!void {
        const ra_val = if (lhs == mul_result) rhs else lhs; // the add/sub's non-mul operand
        const a = try self.loadOp(allocator, code, mul.lhs, fp_spill_op[0]);
        const b = try self.loadOp(allocator, code, mul.rhs, fp_spill_op[1]);
        const c = try self.loadOp(allocator, code, ra_val, fp_spill_res);
        const rd = self.resultReg(result);
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

    /// Emit the fused shifted-register add/sub for an integer `add`/`sub` (`op`, `lhs`,
    /// `rhs`) whose single-use, immediately-preceding operand is the constant left shift
    /// `arith_imm{.shl, b, k}` (`shl`, defining `shl_result`) - see `fusesIntoNextShiftAdd`
    /// for the shared eligibility check. Folds `a + (b << k)` / `a - (b << k)` into one
    /// `add`/`sub xd, xn, xm, lsl #k`. Loads the add/sub's OTHER operand (`rn`, the addend
    /// or minuend) and the shift's own operand `b` (`rm`) directly, since the shl's
    /// materialization was skipped by the caller.
    ///
    /// `rn` is the add/sub's non-shl operand. For `.sub` the shl result is always the RHS
    /// (subtrahend, guaranteed by the predicate), so `subShifted` computes `rn - (rm << k)`,
    /// matching `a - (b << k)`. `.add` is commutative, so `rn` is whichever operand is not
    /// the shift.
    ///
    /// Nothing runs between the (skipped) shl and this add/sub, so `b` still holds its value
    /// here exactly as `emitFusedArith` relies on for the skipped product's operands.
    fn emitShiftAdd(
        self: Ctx,
        allocator: std.mem.Allocator,
        code: *std.ArrayList(u32),
        result: Value,
        op: ir.function.BinOp,
        lhs: Value,
        rhs: Value,
        shl: ir.function.ArithImm,
        shl_result: Value,
    ) Error!void {
        const rn_val = if (lhs == shl_result) rhs else lhs; // the add/sub's non-shl operand
        const rn = try self.loadOp(allocator, code, rn_val, spill_op[0]);
        const rm = try self.loadOp(allocator, code, shl.lhs, spill_op[1]);
        const rd = self.resultReg(result);
        // The predicate range-checked `shl.imm` to 0..63 (64-bit) / 0..31 (32-bit), so it
        // always fits the imm6 field.
        const k: u6 = @intCast(shl.imm);
        const wide = isWide(self.func, result);
        try code.append(allocator, switch (op) {
            .add => if (wide) encode.addShifted64(rd, rn, rm, k) else encode.addShifted(rd, rn, rm, k),
            .sub => if (wide) encode.subShifted64(rd, rn, rm, k) else encode.subShifted(rd, rn, rm, k),
            else => unreachable, // the caller only routes .add/.sub here (see the switch above)
        });
        try storeResult(allocator, code, self, result, rd);
    }

    fn binary(
        self: Ctx,
        allocator: std.mem.Allocator,
        code: *std.ArrayList(u32),
        result: Value,
        op: ir.function.BinOp,
        lhs: Value,
        rhs: Value,
    ) Error!void {
        if (isVector(self.func, result)) {
            // NEON lane-wise arithmetic over a packed vector (currently <4 x f32> only).
            const rl = try self.loadOp(allocator, code, lhs, fp_spill_op[0]);
            const rr = try self.loadOp(allocator, code, rhs, fp_spill_op[1]);
            const rd = self.resultReg(result);
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
            const rd = self.resultReg(result);
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
            const rd = self.resultReg(result);
            try emitBinary(allocator, code, op, rd, rl, rr, isSignedInt(self.func, lhs), isWide(self.func, result));
            try storeResult(allocator, code, self, result, rd);
        }
    }

    /// Emit the flag-setting S-form of the arith at `if_idx-2` for the fused arith-branch (the
    /// caller has confirmed `fusesArithIntoBranch` at `if_idx`). Writes the arith's own result
    /// register (`resultReg`), so the Z flag it sets equals (result == 0), which the eq/ne `b.cc`
    /// the caller emits next tests. The register form emits `adds`/`subs`/`ands` (or their 64-bit
    /// `adds64`/`subs64`/`ands64` counterparts), the immediate form `addsImm`/`subsImm` (or
    /// `addsImm64`/`subsImm64`), imm gated to u12 by the predicate. The S-form width is selected by
    /// `isWide(self.func, result)`: a wide result emits the 64-bit S-form, whose Z flag is (result
    /// == 0) over all 64 bits, matching the surrounding icmp lowering's width-selected
    /// `encode.cmp`/`encode.cmp64`, so the two stay in agreement at every width. The operands are
    /// loaded fresh here: the arith's own materialization was skipped and nothing runs between it
    /// and this if, so they still hold their values, exactly as the fused compare-and-branch relies
    /// on. The S-form reads both operands before writing Rd, so Rd aliasing an operand (the natural
    /// in-place update) is correct.
    fn emitArithBranch(
        self: Ctx,
        allocator: std.mem.Allocator,
        code: *std.ArrayList(u32),
        insts: []const ir.function.Inst,
        if_idx: usize,
    ) Error!void {
        const arith_inst = insts[if_idx - 2];
        const result = self.func.instResult(arith_inst).?;
        const rd = self.resultReg(result);
        const wide = isWide(self.func, result);
        switch (self.func.opcode(arith_inst)) {
            .arith => |a| {
                const rl = try self.loadOp(allocator, code, a.lhs, spill_op[0]);
                const rr = try self.loadOp(allocator, code, a.rhs, spill_op[1]);
                try code.append(allocator, switch (a.op) {
                    .add => if (wide) encode.adds64(rd, rl, rr) else encode.adds(rd, rl, rr),
                    .sub => if (wide) encode.subs64(rd, rl, rr) else encode.subs(rd, rl, rr),
                    .bit_and => if (wide) encode.ands64(rd, rl, rr) else encode.ands(rd, rl, rr),
                    // fusesArithIntoBranch admits only add/sub/bit_and in the register form.
                    else => unreachable,
                });
            },
            .arith_imm => |a| {
                const rl = try self.loadOp(allocator, code, a.lhs, spill_op[0]);
                const imm: u12 = @intCast(a.imm); // gated to [0, 0xFFF] by fusesArithIntoBranch
                try code.append(allocator, switch (a.op) {
                    .add => if (wide) encode.addsImm64(rd, rl, imm) else encode.addsImm(rd, rl, imm),
                    .sub => if (wide) encode.subsImm64(rd, rl, imm) else encode.subsImm(rd, rl, imm),
                    // fusesArithIntoBranch admits only add/sub in the immediate form.
                    else => unreachable,
                });
            },
            // fusesArithIntoBranch requires an arith / arith_imm at if_idx-2.
            else => unreachable,
        }
    }

    /// The conditional branch an `if` reduces to after its SETUP has emitted the flag-setting compare
    /// (or arith S-form) or materialized the boolean. `cc` branches on a `Cond` (the fused
    /// compare-and-branch path), `reg` branches on a boolean register being non-zero (the plain path).
    /// The two layouts below emit either the not-inverted form (branch to THEN) or the inverted form
    /// (branch to ELSE) from this single descriptor.
    const IfBranch = union(enum) {
        cc: encode.Cond,
        reg: Reg,
    };

    fn emitIf(
        self: Ctx,
        allocator: std.mem.Allocator,
        code: *std.ArrayList(u32),
        fixups: *std.ArrayList(Fixup),
        cf: ir.function.If,
        insts: []const ir.function.Inst,
        if_idx: usize,
        next_block: ?Block,
    ) Error!void {
        // SETUP (shared by both layouts): emit the test that decides the branch and yield the branch
        // descriptor. Fused compare-and-branch: when the immediately-preceding instruction is a
        // single-use integer icmp that is exactly this if's condition (the SAME predicate the icmp case
        // used to skip its materialization), load the icmp's operands, set the flags with `cmp`, and
        // branch on the icmp's condition instead of materializing a boolean and re-testing it with
        // `cbnz`. condFor here mirrors the icmp lowering's cset condition, so the branch takes the
        // then-edge under exactly the same condition.
        var branch: IfBranch = undefined;
        if (if_idx >= 1 and self.fuse_cmp_branch and fusesIntoNextIf(self.func, insts, if_idx - 1)) {
            const cmp = self.func.opcode(insts[if_idx - 1]).icmp;
            const cond = condFor(cmp.op, isSignedInt(self.func, cmp.lhs));
            // Arith-branch fold: when the arith at if_idx-2 is a single-use add/sub/and whose
            // result the icmp compares eq/ne against 0, emit its flag-setting S-form (which sets
            // Z = (result == 0)) instead of loading the icmp operands and doing a `cmp #0`. The
            // eq/ne `b.cc` below then branches on that Z flag. The SAME predicate skipped the
            // arith's own materialization, so it is emitted exactly once. Otherwise fall back to
            // the plain compare-and-branch (load the icmp operands, `cmp`).
            if (fusesArithIntoBranch(self.func, insts, if_idx, self.fuse_arith_branch and self.fuse_cmp_branch)) {
                try self.emitArithBranch(allocator, code, insts, if_idx);
            } else {
                const rl = try self.loadOp(allocator, code, cmp.lhs, spill_op[0]);
                const rr = try self.loadOp(allocator, code, cmp.rhs, spill_op[1]);
                // Compare at the operand width: a 32-bit cmp on an i64/ptr operand would only
                // test the low 32 bits, mismeasuring values whose high bits differ.
                const wide = isWide(self.func, cmp.lhs);
                try code.append(allocator, if (wide) encode.cmp64(rl, rr) else encode.cmp(rl, rr));
            }
            branch = .{ .cc = cond };
        } else {
            const cond = try self.loadOp(allocator, code, cf.cond, spill_op[0]);
            branch = .{ .reg = cond };
        }

        // Layout selection. THEN and ELSE are the two successor blocks; NEXT is the block emitted right
        // after this one (or null at the last block). Whichever edge targets NEXT can fall through, so
        // its branch is elided. THEN is checked first, so a degenerate `if` with THEN == ELSE == NEXT
        // takes the THEN-fall-through layout (elides one branch, both edges still reach the same block).
        const then_target = cf.then.target;
        const else_target = cf.@"else".target;
        const then_next = next_block != null and then_target == next_block.?;
        const else_next = next_block != null and else_target == next_block.?;

        if (then_next) {
            // THEN falls through. Keep the not-inverted branch to the then-label, emit the ELSE-moves +
            // `b ELSE` inline, then the THEN-moves last and FALL THROUGH to THEN (elide the trailing
            // `b THEN`). The conditional's forward displacement targets the then-label as before.
            const bcc_at = code.items.len;
            try code.append(allocator, switch (branch) {
                .cc => |cond| encode.bcc(cond, 0),
                .reg => |reg| encode.cbnz(reg, 0),
            });
            try self.emitMoves(allocator, code, cf.@"else"); // ELSE-moves on the else path
            try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(else_target) });
            try code.append(allocator, encode.b(0));
            const then_at = code.items.len;
            const disp: i21 = @intCast((@as(i64, @intCast(then_at)) - @as(i64, @intCast(bcc_at))) * 4);
            code.items[bcc_at] = switch (branch) {
                .cc => |cond| encode.bcc(cond, disp),
                .reg => |reg| encode.cbnz(reg, disp),
            };
            try self.emitMoves(allocator, code, cf.then); // THEN-moves, then fall through to THEN
            return;
        }

        if (else_next) {
            // ELSE falls through. INVERT the branch so it targets the else-label when the ELSE edge is
            // taken (bcc on invertCond, or `cbz` for the boolean path), emit the THEN-moves + `b THEN`
            // inline (the not-taken path), then the ELSE-moves last and FALL THROUGH to ELSE. The
            // conditional's forward displacement targets the else-label.
            const bcc_at = code.items.len;
            try code.append(allocator, switch (branch) {
                .cc => |cond| encode.bcc(encode.invertCond(cond), 0),
                .reg => |reg| encode.cbz(reg, 0),
            });
            try self.emitMoves(allocator, code, cf.then); // THEN-moves on the then (not-taken) path
            try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(then_target) });
            try code.append(allocator, encode.b(0));
            const else_at = code.items.len;
            const disp: i21 = @intCast((@as(i64, @intCast(else_at)) - @as(i64, @intCast(bcc_at))) * 4);
            code.items[bcc_at] = switch (branch) {
                .cc => |cond| encode.bcc(encode.invertCond(cond), disp),
                .reg => |reg| encode.cbz(reg, disp),
            };
            try self.emitMoves(allocator, code, cf.@"else"); // ELSE-moves, then fall through to ELSE
            return;
        }

        // Neither edge is NEXT (or this is the last block): emit both branches as before.
        const bcc_at = code.items.len;
        try code.append(allocator, switch (branch) {
            .cc => |cond| encode.bcc(cond, 0),
            .reg => |reg| encode.cbnz(reg, 0),
        });
        try self.emitMoves(allocator, code, cf.@"else");
        try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(else_target) });
        try code.append(allocator, encode.b(0));
        const then_at = code.items.len;
        const disp: i21 = @intCast((@as(i64, @intCast(then_at)) - @as(i64, @intCast(bcc_at))) * 4);
        code.items[bcc_at] = switch (branch) {
            .cc => |cond| encode.bcc(cond, disp),
            .reg => |reg| encode.cbnz(reg, disp),
        };
        try self.emitMoves(allocator, code, cf.then);
        try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(then_target) });
        try code.append(allocator, encode.b(0));
    }

    fn emitJump(
        self: Ctx,
        allocator: std.mem.Allocator,
        code: *std.ArrayList(u32),
        fixups: *std.ArrayList(Fixup),
        jump: ir.function.Jump,
        next_block: ?Block,
    ) Error!void {
        try self.emitMoves(allocator, code, jump);
        // A jump to the block emitted immediately after this one falls through: the edge-moves above
        // still run, but the `b` (and its fixup) is elided.
        if (next_block != null and jump.target == next_block.?) return;
        try fixups.append(allocator, .{ .at = code.items.len, .target = @intFromEnum(jump.target) });
        try code.append(allocator, encode.b(0));
    }

    /// Edge moves into the target's parameters (which are always in registers).
    /// GPR and FP register-resident arguments go through separate parallel moves.
    /// spilled arguments are reloaded straight into their parameter register.
    ///
    /// Two paths: an edge-move-driven allocation (the shared Wimmer path) replays the precomputed,
    /// already-ordered `edge_moves` for the current edge `self.block -> jump.target` op-by-op and
    /// derives nothing (the shared allocator resolved parameters AND live-through values, spills, and
    /// cycles). The DEFAULT path (`edge_move_driven == false`) keeps deriving block-parameter moves
    /// exactly as before, so its output is byte-identical when no allocation sets `edge_moves`.
    fn emitMoves(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), jump: ir.function.Jump) Error!void {
        if (self.alloc.edge_move_driven) {
            try self.emitEdgeMoves(allocator, code, self.block, jump.target);
            return;
        }
        const args = self.func.blockArgs(jump);
        const params = self.func.blockParams(jump.target);
        if (args.len != params.len) return error.Unsupported;

        var gpr_moves: std.ArrayList(Move) = .empty;
        defer gpr_moves.deinit(allocator);
        var fpr_moves: std.ArrayList(Move) = .empty;
        defer fpr_moves.deinit(allocator);
        for (args, params) |arg, param| {
            const dst = self.alloc.reg.get(param).?; // params are never split, so a whole-life register
            // A register-resident edge argument moves; a spilled one is reloaded below. The argument's
            // location is read at the current (terminator) position so a split argument uses its active
            // segment. With no splits this is exactly today's `reg`/`spill` reads.
            switch (self.locationAt(arg)) {
                .reg => |src| {
                    const m = Move{ .src = src, .dst = dst };
                    if (regClass(self.func, param) == .fpr) try fpr_moves.append(allocator, m) else try gpr_moves.append(allocator, m);
                },
                .slot => {}, // reloaded straight into its parameter register below
            }
        }
        try parallelMove(allocator, code, gpr_moves.items, encode.mov, scratch_move);
        // `movVec` copies the whole 128-bit register: correct for vector block
        // parameters, harmless (a few extra bits) for scalar floats.
        try parallelMove(allocator, code, fpr_moves.items, encode.movVec, fp_move);
        for (args, params) |arg, param| {
            switch (self.locationAt(arg)) {
                .slot => |slot| {
                    const pr = self.alloc.reg.get(param).?;
                    const off: u15 = @intCast(self.spill_base + slot * 16);
                    if (isVector(self.func, param)) {
                        try code.append(allocator, encode.ldrQ(pr, sp, off));
                    } else if (regClass(self.func, param) == .fpr) {
                        try code.append(allocator, encode.ldrFp(pr, sp, off, true));
                    } else {
                        try code.append(allocator, encode.ldrOff(pr, sp, off));
                    }
                },
                .reg => {}, // already handled by the parallel move above
            }
        }
    }

    /// Replay the precomputed, already-ordered edge moves for `pred -> succ` op-by-op. The shared
    /// allocator resolved the parallel move (sources read before overwrite, cycles broken and
    /// slot->slot shuffles routed through the class scratch), so each op is a primitive reg->reg
    /// (move), reg->slot (store), or slot->reg (load). An edge with no shuffle has no entry, so
    /// nothing is emitted for it. No reordering here would break that resolution.
    fn emitEdgeMoves(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), pred: Block, succ: Block) Error!void {
        const set = self.findEdgeMoves(pred, succ) orelse return;
        for (set.moves) |m| try self.emitEdgeMove(allocator, code, m);
    }

    /// The precomputed move set for the edge `pred -> succ`, or null when the edge needs no shuffle.
    fn findEdgeMoves(self: Ctx, pred: Block, succ: Block) ?*const EdgeMoveSet {
        for (self.alloc.edge_moves) |*set| {
            if (set.pred == pred and set.succ == succ) return set;
        }
        return null;
    }

    /// Emit one ordered edge move as A64. The move width follows its class: a gpr move is a 64-bit
    /// `mov`/`ldr`/`str`, an fpr move copies the whole 128-bit register (`movVec`) and loads/stores
    /// the whole 16-byte slot (`ldrQ`/`strQ`) so both scalar floats and lane vectors round-trip
    /// intact. Slot offsets mirror the action/reload math (`spill_base + slot * 16`). A slot->slot op
    /// never appears (the shared ordering already expanded it through the class scratch), so it is a
    /// programmer error here.
    fn emitEdgeMove(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), m: EdgeMove) Error!void {
        std.debug.assert(m.class == 0 or m.class == 1);
        const fpr = m.class == 1;
        switch (m.src) {
            .reg => |sr| switch (m.dst) {
                .reg => |dr| {
                    if (@intFromEnum(sr) == @intFromEnum(dr)) return; // identity move
                    try code.append(allocator, if (fpr) encode.movVec(dr, sr) else encode.mov(dr, sr));
                },
                .slot => |ds| {
                    const off: u15 = @intCast(self.spill_base + ds * 16);
                    try code.append(allocator, if (fpr) encode.strQ(sr, sp, off) else encode.strOff(sr, sp, off));
                },
            },
            .slot => |ss| switch (m.dst) {
                .reg => |dr| {
                    const off: u15 = @intCast(self.spill_base + ss * 16);
                    try code.append(allocator, if (fpr) encode.ldrQ(dr, sp, off) else encode.ldrOff(dr, sp, off));
                },
                .slot => unreachable, // the shared ordering expands slot->slot through the class scratch
            },
        }
    }

    /// Move a call's arguments into their ABI locations (x0..x7 / v0..v7, or the outgoing stack area
    /// for the 9th+). Shared by `.call` and `.call_indirect`.
    ///
    /// Two paths. The DEFAULT (`edge_move_driven == false`, the native `allocate`) keeps the exact
    /// sequential per-argument emission, byte-identical to before this extraction: the native
    /// allocator never places one argument's source in another argument's destination ABI register, so
    /// a straight `mov x_idx, src` in argument order is safe. The edge-move-driven (shared Wimmer)
    /// path CANNOT assume that: the shared allocator's register choices can make an earlier argument's
    /// destination ABI register alias a later argument's still-unread source (a real permutation, e.g.
    /// a 3-cycle rotation), so the sequential moves would clobber. It emits the setup as a parallel
    /// move instead: (1) stack-passed stores first, while every source register is still intact;
    /// (2) the register-resident register arguments as a `parallelMove` into their ABI registers,
    /// breaking cycles through the reserved scratch (`scratch_move`/`fp_move`); (3) the slot-resident
    /// register arguments reloaded straight into their ABI register, after the permutation has read
    /// every source. A slot-sourced argument holds its value in memory, so it is never clobbered by an
    /// earlier move and is safe to reload last.
    fn emitCallArgs(self: Ctx, allocator: std.mem.Allocator, code: *std.ArrayList(u32), args: []const Value) Error!void {
        const locs = try allocator.alloc(ArgLoc, args.len);
        defer allocator.free(locs);
        computeArgLocs(self.func, args, locs);

        if (!self.alloc.edge_move_driven) {
            for (args, locs) |arg, loc| {
                const target: Reg = @enumFromInt(@as(u5, @intCast(loc.idx)));
                if (loc.class == .fpr) {
                    const src = try self.loadOp(allocator, code, arg, fp_spill_op[0]);
                    if (loc.idx >= 8) return error.Unsupported; // fp stack args not handled
                    try code.append(allocator, encode.fmovReg(target, src));
                } else {
                    const src = try self.loadOp(allocator, code, arg, spill_op[0]);
                    if (loc.idx < 8) {
                        try code.append(allocator, encode.mov(target, src));
                    } else {
                        try code.append(allocator, encode.strOff(src, sp, @intCast(8 * (loc.idx - 8))));
                    }
                }
            }
            return;
        }

        // Phase 1: stack-passed arguments (idx >= 8) store first, before the register permutation
        // overwrites any ABI register a store still needs to read.
        for (args, locs) |arg, loc| {
            if (loc.idx < 8) continue;
            if (loc.class == .fpr) return error.Unsupported; // fp stack args not handled (matches native)
            const src = try self.loadOp(allocator, code, arg, spill_op[0]);
            try code.append(allocator, encode.strOff(src, sp, @intCast(8 * (loc.idx - 8))));
        }

        // Phase 2a: register-resident register arguments as a parallel move into their ABI registers.
        var gpr_moves: std.ArrayList(Move) = .empty;
        defer gpr_moves.deinit(allocator);
        var fpr_moves: std.ArrayList(Move) = .empty;
        defer fpr_moves.deinit(allocator);
        for (args, locs) |arg, loc| {
            if (loc.idx >= 8) continue;
            const target: Reg = @enumFromInt(@as(u5, @intCast(loc.idx)));
            switch (self.locationAt(arg)) {
                .reg => |src| {
                    const m = Move{ .src = src, .dst = target };
                    if (loc.class == .fpr) try fpr_moves.append(allocator, m) else try gpr_moves.append(allocator, m);
                },
                .slot => {}, // reloaded in phase 2b
            }
        }
        try parallelMove(allocator, code, gpr_moves.items, encode.mov, scratch_move);
        // `fmovReg` copies the low 64 bits, matching the native sequential path's `fmovReg` exactly, so
        // a float argument round-trips identically whichever allocator placed it.
        try parallelMove(allocator, code, fpr_moves.items, encode.fmovReg, fp_move);

        // Phase 2b: slot-resident register arguments reload into their ABI register (the permutation
        // above has already read every source register, so overwriting one now is safe).
        for (args, locs) |arg, loc| {
            if (loc.idx >= 8) continue;
            const target: Reg = @enumFromInt(@as(u5, @intCast(loc.idx)));
            switch (self.locationAt(arg)) {
                .slot => |slot| {
                    const off: u15 = @intCast(self.spill_base + slot * 16);
                    if (isVector(self.func, arg)) {
                        try code.append(allocator, encode.ldrQ(target, sp, off));
                    } else if (loc.class == .fpr) {
                        try code.append(allocator, encode.ldrFp(target, sp, off, true));
                    } else {
                        try code.append(allocator, encode.ldrOff(target, sp, off));
                    }
                },
                .reg => {}, // handled by the parallel move above
            }
        }
    }
};

/// Store `result` (held in `reg`) to its spill slot iff its location at its def position is a
/// slot. A value that stays in a register at its def emits nothing (a later split, if any, would
/// store at the split point). With no splits this is exactly today's spill-if-spilled behavior.
fn storeResult(allocator: std.mem.Allocator, code: *std.ArrayList(u32), ctx: Ctx, result: Value, reg: Reg) Error!void {
    switch (ctx.locationAt(result)) {
        .reg => {},
        .slot => |slot| {
            const off: u15 = @intCast(ctx.spill_base + slot * 16);
            if (isVector(ctx.func, result)) {
                try code.append(allocator, encode.strQ(reg, sp, off));
            } else if (regClass(ctx.func, result) == .fpr) {
                try code.append(allocator, encode.strFp(reg, sp, off, true));
            } else {
                try code.append(allocator, encode.strOff(reg, sp, off));
            }
        },
    }
}

/// Append `seg` to `value`'s segment list (growing the owned slice). Segments must be appended in
/// ascending `from` order by the caller, so a re-spill after a re-home lands at or after the re-home
/// position (asserted). Creates a single-element list if the value has none yet.
fn appendSegment(allocator: std.mem.Allocator, alloc: *Allocation, value: Value, seg: Segment) Error!void {
    const gop = try alloc.segments.getOrPut(allocator, value);
    if (!gop.found_existing) {
        const s = try allocator.alloc(Segment, 1);
        s[0] = seg;
        gop.value_ptr.* = s;
        return;
    }
    const old = gop.value_ptr.*;
    // Ascending-`from` is the representation invariant `locationAt` relies on (it scans until the
    // first segment past `pos`). A caller that appends out of order is a programmer error and would
    // silently miscompile, so assert it rather than trust it.
    std.debug.assert(seg.from >= old[old.len - 1].from);
    const grown = try allocator.alloc(Segment, old.len + 1);
    @memcpy(grown[0..old.len], old);
    grown[old.len] = seg;
    allocator.free(old);
    gop.value_ptr.* = grown;
}

/// The linearization of a function: positions, per-value liveness, and the extra split-liveness
/// data a live-range splitter needs. `use_positions` and `is_intra` are computed here but do not
/// yet influence any allocation decision (they are consumed by later splitting work).
const Liveness = struct {
    def_pos: []u32, // per value: position of its definition
    last_use: []u32, // per value: position of its last use (after extendLiveRanges)
    is_param: []bool, // per value: is a block parameter
    block_end: []u32, // per block: terminator position
    call_positions: std.ArrayList(u32),
    use_positions: [][]u32, // per value: ascending positions where the value is an OPERAND use
    //   (includes edge-argument uses, at the terminator position)
    is_intra: []bool, // per value: true iff the value can be split within one block:
    //   NOT a param, NEVER used as an edge argument, and every use is in its def block

    fn deinit(self: *Liveness, allocator: std.mem.Allocator) void {
        allocator.free(self.def_pos);
        allocator.free(self.last_use);
        allocator.free(self.is_param);
        allocator.free(self.block_end);
        self.call_positions.deinit(allocator);
        for (self.use_positions) |u| allocator.free(u);
        allocator.free(self.use_positions);
        allocator.free(self.is_intra);
    }
};

/// Assign each parameter/instruction/terminator a monotonically increasing position and record,
/// per value: its def position, its last use, whether it is a block parameter, the ascending list
/// of positions where it is used as an operand (including edge arguments at the terminator
/// position), and whether it is intra-block-splittable. Also records each block's terminator
/// position and every call position, then extends live ranges. This is exactly the linearization
/// `allocate` used to do inline, plus the split-liveness data (`use_positions`, `is_intra`).
fn linearize(allocator: std.mem.Allocator, func: *const Function, fold: *const addrfold.Analysis) Error!Liveness {
    const nval = func.valueCount();
    const nblocks = func.blockCount();

    const def_pos = try allocator.alloc(u32, nval);
    errdefer allocator.free(def_pos);
    const last_use = try allocator.alloc(u32, nval);
    errdefer allocator.free(last_use);
    const is_param = try allocator.alloc(bool, nval);
    errdefer allocator.free(is_param);
    const block_end = try allocator.alloc(u32, nblocks);
    errdefer allocator.free(block_end);
    const is_intra = try allocator.alloc(bool, nval);
    errdefer allocator.free(is_intra);

    // def_block is a scratch row (which block defined each value), used only to decide is_intra.
    const def_block = try allocator.alloc(u32, nval);
    defer allocator.free(def_block);

    @memset(def_pos, 0);
    @memset(is_param, false);
    for (last_use) |*l| l.* = 0;
    @memset(def_block, 0);
    // A value starts intra-splittable iff it is not a parameter; the walk below clears it on any
    // edge-argument use or any use outside its def block.
    for (is_intra, 0..) |*flag, i| flag.* = !is_param[i];

    // Positions at which a call clobbers caller-saved registers. AAPCS only preserves the
    // LOW 64 bits of the callee-saved FP registers v8..v15 across a call, so a 128-bit SIMD
    // `<4 x f32>` held in one of them would lose its upper two lanes over a `bl`/`blr`. We
    // record every call position and force-spill any VECTOR interval that spans a call to the
    // stack (16-byte slots are fully preserved), keeping the widened (quad) FS correct when it
    // gathers through a sampler_fn / math_fn call.
    var call_positions = std.ArrayList(u32).empty;
    errdefer call_positions.deinit(allocator);

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
        last_use: []u32,
        pos: u32,
        bi: u32,
        err: ?Error = null,

        fn visit(self: *@This(), v: Value, is_edge_arg: bool) void {
            const vi = @intFromEnum(v);
            markUse(self.last_use, v, self.pos);
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
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| {
            def_pos[@intFromEnum(p)] = pos;
            last_use[@intFromEnum(p)] = pos;
            is_param[@intFromEnum(p)] = true;
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
                .last_use = last_use,
                .pos = pos,
                .bi = @intCast(bi),
            };
            forEachOperand(func, inst, fold, &col, Collector.visit);
            if (col.err) |e| return e;
            if (func.instResult(inst)) |r| {
                def_pos[@intFromEnum(r)] = pos;
                def_block[@intFromEnum(r)] = @intCast(bi);
            }
            switch (func.opcode(inst)) {
                .call, .call_indirect => try call_positions.append(allocator, pos),
                else => {},
            }
            pos += 1;
        }
        block_end[bi] = pos;
        if (func.terminator(block)) |term| {
            var col = Collector{
                .is_intra = is_intra,
                .def_block = def_block,
                .use_lists = use_lists,
                .allocator = allocator,
                .last_use = last_use,
                .pos = pos,
                .bi = @intCast(bi),
            };
            forEachTermOperand(func, term, &col, Collector.visit);
            if (col.err) |e| return e;
        }
        pos += 1;
    }

    // Liveness: a value live-out of a block stays live to that block's end. A loop back-edge makes
    // the header's live-in flow into the body's live-out, extending loop-carried values across the
    // body so their registers are not reused inside it.
    try extendLiveRanges(allocator, func, last_use, block_end, fold);

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

    return .{
        .def_pos = def_pos,
        .last_use = last_use,
        .is_param = is_param,
        .block_end = block_end,
        .call_positions = call_positions,
        .use_positions = use_positions,
        .is_intra = is_intra,
    };
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

/// Linear-scan register allocation with spilling. Block parameters are pinned to
/// registers. Instruction results spill when the pool is exhausted.
// The backend context the shared Wimmer-Franz allocator threads through `classOf`/`useKind`. aarch64
// needs no extra state (its class/use decisions read only the function, passed separately), so this
// is a zero-field singleton whose address is a stable, non-owned `ctx` pointer.
const AArch64RegCtx = struct {};
const aarch64_reg_ctx: AArch64RegCtx = .{};

/// `RegDescription.classOf` for aarch64: a value lives in the gpr class (0) or the fpr class (1),
/// exactly as `regClass` decides (float/vector -> fpr, everything else -> gpr).
fn aarch64ClassOf(ctx: *const anyopaque, func: *const Function, v: Value) u16 {
    _ = ctx;
    return @intFromEnum(regClass(func, v));
}

/// `RegDescription.useKind` for aarch64: every operand needs a register (aarch64 cannot fold a spill
/// slot into an arithmetic operand). This is the conservative default; a backend that can read memory
/// operands relaxes specific opcodes. Unused parameters are the generic hook shape.
fn aarch64UseKind(ctx: *const anyopaque, func: *const Function, inst: ir.function.Inst, operand: Value) wimmer.UseKind {
    _ = ctx;
    _ = func;
    _ = inst;
    _ = operand;
    return .must_have_register;
}

/// Build the per-function aarch64 `RegDescription` the shared Wimmer-Franz allocator consumes. The
/// physical-register INDEX numbering is the register's own enum integer value: gpr class index n
/// names x_n (x0..x30), fpr class index n names v_n (v0..v31), the two classes disambiguating the
/// shared 0..31 index space. This MIRRORS the existing `allocate`: the leaf/non-leaf pools match its
/// pool-building loop, entry params pin the ABI arg registers via `computeArgLocs`, and call sites
/// use `linearize`'s call positions so the numbering matches emission. Task 1 builds only the
/// description (no allocation runs). The caller owns the result and must `deinit` it.
///
/// Call-clobber approximation: class 1's per-call clobber lists ALL fp registers v0..v31, i.e. the
/// caller-saved fp regs PLUS the callee-saved v8..v15. The extra v8..v15 encodes the vector quirk
/// (a 128-bit vector in a callee-saved fp reg loses its upper half across a call, since AAPCS only
/// preserves the low 64 bits) without type info at description-build time. It over-clobbers scalar
/// floats slightly (a scalar float is also forced out of v8..v15 across a call); Task 2's interval
/// builder may refine it to vector-only. Entry stack params (arg index >= 8) are NOT pre-colored
/// here (they arrive on the stack); the Task 2 interval builder handles them.
pub fn aarch64RegDescription(allocator: std.mem.Allocator, func: *const Function) Error!wimmer.RegDescription {
    const leaf = isLeaf(func);

    const eparams = func.blockParams(@enumFromInt(0));
    const plocs = try allocator.alloc(ArgLoc, eparams.len);
    defer allocator.free(plocs);
    computeArgLocs(func, eparams, plocs);
    var n_gpr: usize = 0;
    var n_fpr: usize = 0;
    for (plocs) |l| {
        if (l.class == .gpr) n_gpr += 1 else n_fpr += 1;
    }

    // Allocatable pools, mirroring `allocate`'s pool-building loop EXACTLY. GPR: caller-saved
    // temporaries x9..x12 + unused integer arg registers (leaf) or callee-saved x19..x28 (non-leaf).
    // FPR: caller-saved v16..v23 + unused FP arg registers (leaf) or callee-saved v8..v15 (non-leaf).
    var gpr_alloc: std.ArrayList(u16) = .empty;
    errdefer gpr_alloc.deinit(allocator);
    var fpr_alloc: std.ArrayList(u16) = .empty;
    errdefer fpr_alloc.deinit(allocator);
    if (leaf) {
        for (9..13) |r| try gpr_alloc.append(allocator, @intCast(r));
        for (@min(n_gpr, 8)..8) |r| try gpr_alloc.append(allocator, @intCast(r));
        for (16..24) |r| try fpr_alloc.append(allocator, @intCast(r));
        for (@min(n_fpr, 8)..8) |r| try fpr_alloc.append(allocator, @intCast(r));
    } else {
        for (19..29) |r| try gpr_alloc.append(allocator, @intCast(r));
        for (8..16) |r| try fpr_alloc.append(allocator, @intCast(r));
    }

    // Callee-saved sets: x19..x28 (gpr), v8..v15 (fpr).
    const gpr_cs = try allocator.alloc(u16, 10);
    errdefer allocator.free(gpr_cs);
    for (0..10) |i| gpr_cs[i] = @intCast(19 + i);
    const fpr_cs = try allocator.alloc(u16, 8);
    errdefer allocator.free(fpr_cs);
    for (0..8) |i| fpr_cs[i] = @intCast(8 + i);

    const gpr_alloc_owned = try gpr_alloc.toOwnedSlice(allocator);
    errdefer allocator.free(gpr_alloc_owned);
    const fpr_alloc_owned = try fpr_alloc.toOwnedSlice(allocator);
    errdefer allocator.free(fpr_alloc_owned);

    const classes = try allocator.alloc(wimmer.RegClass, 2);
    errdefer allocator.free(classes);
    // aarch64 uses uniform 16-byte spill slots for both classes (see the existing spill/frame code).
    classes[0] = .{ .name = "gpr", .allocatable = gpr_alloc_owned, .callee_saved = gpr_cs, .slot_bytes = 16 };
    classes[1] = .{ .name = "fpr", .allocatable = fpr_alloc_owned, .callee_saved = fpr_cs, .slot_bytes = 16 };

    // Entry params: the first 8 gpr params pin x0..x7, the first 8 fp params pin v0..v7. Params with
    // arg index >= 8 arrive on the stack and are left for the Task 2 interval builder.
    var ef: std.ArrayList(wimmer.FixedAssign) = .empty;
    errdefer ef.deinit(allocator);
    for (eparams, plocs) |p, l| {
        if (l.idx < 8) {
            const class: u16 = if (l.class == .fpr) 1 else 0;
            try ef.append(allocator, .{ .value = p, .class = class, .reg = @intCast(l.idx) });
        }
    }
    const entry_fixed = try ef.toOwnedSlice(allocator);
    errdefer allocator.free(entry_fixed);

    // Call sites: one per call position, using `linearize`'s numbering so it matches emission. The
    // Wimmer path is fold-agnostic, so linearize under the empty analysis (no rerouting).
    var lin = try linearize(allocator, func, &empty_fold);
    defer lin.deinit(allocator);
    const call_sites = try allocator.alloc(wimmer.CallSite, lin.call_positions.items.len);
    var built: usize = 0;
    errdefer {
        for (call_sites[0..built]) |cs| {
            for (cs.clobbered) |cr| allocator.free(cr.regs);
            allocator.free(cs.clobbered);
        }
        allocator.free(call_sites);
    }
    for (lin.call_positions.items, 0..) |cpos, i| {
        // Class 0: the caller-saved gpr set x0..x17 (x18 is the reserved platform register, excluded).
        const gpr_clob = try allocator.alloc(u16, 18);
        errdefer allocator.free(gpr_clob);
        for (0..18) |j| gpr_clob[j] = @intCast(j);
        // Class 1: all fp registers v0..v31 (caller-saved fp regs plus the v8..v15 vector quirk).
        const fpr_clob = try allocator.alloc(u16, 32);
        errdefer allocator.free(fpr_clob);
        for (0..32) |j| fpr_clob[j] = @intCast(j);
        const clob = try allocator.alloc(wimmer.ClassRegs, 2);
        clob[0] = .{ .class = 0, .regs = gpr_clob };
        clob[1] = .{ .class = 1, .regs = fpr_clob };
        call_sites[i] = .{ .pos = cpos, .clobbered = clob };
        built = i + 1;
    }

    // Scratch registers reserved for parallel-move cycle breaking, indexed by class: gpr uses
    // `scratch_move` (x17), fpr uses `fp_move` (v27). These are the same regs the backend already
    // reserves for parallel moves, kept out of every pool.
    const scratch = try allocator.alloc(u16, 2);
    errdefer allocator.free(scratch);
    scratch[0] = @intCast(@intFromEnum(scratch_move));
    scratch[1] = @intCast(@intFromEnum(fp_move));

    return .{
        .classes = classes,
        .classOf = aarch64ClassOf,
        .useKind = aarch64UseKind,
        .entry_fixed = entry_fixed,
        .call_sites = call_sites,
        .scratch = scratch,
        .ctx = &aarch64_reg_ctx,
    };
}

/// TEST-ONLY: the differential-harness entry into the SHARED Wimmer-Franz allocator. It runs the
/// same pipeline as the production `compileFunction` (split critical edges, `wimmer.allocate`,
/// `translateAllocation`, `emitFromAllocation`), but takes `func` by MUTABLE pointer and splits its
/// edges IN PLACE (no clone) with INERT `.{}` caps and no address folding, so a test can build two
/// identical functions and compare this against the production path. `compileFunction` is now the
/// SAME allocator (the Wimmer cutover flip), so there is no longer a separate native `allocate` to
/// differ from; this entry survives as the mutable-in-place, inert-caps compile the harness drives.
///
/// Scope (Wimmer cutover Task 3): LEAF functions, including CROSS-BLOCK ones (loops, diamonds,
/// lifetime holes, critical edges), AND NOW NON-LEAF functions too (the leaf-only gate is open):
/// `aarch64RegDescription`/`wimmer.allocate` are already non-leaf-capable (per-call clobber lists,
/// callee-saved pools), and `translateAllocation`'s remaining bridge gaps still bail
/// `error.Unsupported` for the non-leaf shapes they cannot yet express FAITHFULLY (never a silent
/// miscompile): a same-position intra-block action register hazard (gap #6), and the sequential
/// call-argument clobber (no parallel-move resolution at a call site yet). A whole-life entry
/// parameter placed off its ABI arg register (gap #5, the callee-saved param-across-a-call case)
/// IS now handled: the existing non-leaf prologue already emits `mov allocated_reg,
/// incoming_abi_reg`, so the bridge only needs to record the allocated register. A split or
/// slot-resident entry param (gap #4) IS now handled too: a whole-life spilled param is recorded
/// in `alloc.spill` like any other whole-spilled value, and a genuinely split param's first
/// segment is established from the incoming ABI argument by the emission param-setup loop.
///
/// Critical edges are split up front so the shared allocator's edge-move resolution has a block to
/// place its shuffle on, and the translated `edge_moves` drive emission.
///
/// Takes `func` by mutable pointer because `splitCriticalEdges` inserts forwarding blocks in place;
/// all downstream numbering (the RegDescription, the scan, and emission) runs on the SPLIT CFG. A
/// differential caller that must keep an unmutated reference builds two identical functions and
/// compiles one each way (see the cross-block tests).
pub fn compileFunctionWimmer(allocator: std.mem.Allocator, func: *Function) Error!Compiled {
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;
    if (func.blockCount() == 0) return error.Unsupported;

    // Split critical edges FIRST (mutating `func`), before any numbering is built, so the resolver's
    // no-critical-edge precondition holds and the RegDescription/scan/emission all see one CFG.
    try ir.critical_edge.splitCriticalEdges(allocator, func);

    var desc = try aarch64RegDescription(allocator, func);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, func, &desc);
    defer walloc.deinit(allocator);

    var alloc = try translateAllocation(allocator, func, &walloc);
    defer alloc.deinit(allocator);
    // The Wimmer differential path never folds addresses, so it emits through the empty analysis and
    // stays byte-identical to the pre-fold emission.
    return emitFromAllocation(allocator, func, .{}, &alloc, &empty_fold);
}

/// The aarch64 (uniform 16-byte) spill-slot index for a Wimmer per-class slot: GPR (class 0) slots
/// take `[0, gpr_slots)`, FPR (class 1) slots take `[gpr_slots, gpr_slots + fpr_slots)`. This keeps
/// every spilled value on a distinct slot, which is all the frame offset math needs.
fn aarch64Slot(class: u16, wimmer_slot: u32, gpr_slots: u32) u32 {
    return switch (class) {
        0 => wimmer_slot,
        1 => gpr_slots + wimmer_slot,
        else => unreachable,
    };
}

/// Translate a finished shared `wimmer.Allocation` into this backend's own `Allocation` so the
/// existing emission can consume it. A whole-life value (one segment) lands in the `reg`/`spill`
/// maps (so the prologue's direct `reg` reads and the callee-saved collection from
/// `walloc.used_callee_saved` see it directly); a genuinely split value lands in `segments`, which
/// `locationAt` resolves per position. Each intra-block segment transition becomes a drain action
/// (store / reload / register move). Entry params are required to be whole-life: a LEAF param must
/// stay in its ABI arg register (the leaf hint, since a leaf's prologue emits no param move); a
/// NON-LEAF param may land in any register (the non-leaf prologue always emits
/// `mov allocated_reg, incoming_abi_reg`, so an off-ABI whole-life placement - e.g. a value live
/// across a call parked in a callee-saved register - is realized by that move, Wimmer bridge gap
/// #5). A split or slot-resident entry param (gap #4) lands in `alloc.spill` (whole-life,
/// immediately spilled: the existing spilled-param prologue code already stores the incoming
/// argument straight into the slot, no param-specific handling needed) or `alloc.segments`
/// (genuinely split: the emission param-setup loop establishes `segments[0]` from the incoming
/// argument, then this function's normal per-transition drain actions realize every later re-home).
fn translateAllocation(allocator: std.mem.Allocator, func: *const Function, walloc: *const wimmer.Allocation) Error!Allocation {
    var alloc = Allocation{};
    errdefer alloc.deinit(allocator);
    const leaf = isLeaf(func);

    // The emission's pos-coupling assert reads `alloc.def_pos[value]`. Reuse the backend's own
    // `linearize` numbering, which is identical to the shared allocator's, so every def position
    // matches the `ctx.pos` the emission advances through. This path is fold-agnostic (empty analysis).
    var lin = try linearize(allocator, func, &empty_fold);
    defer lin.deinit(allocator);
    alloc.def_pos = try allocator.dupe(u32, lin.def_pos);

    std.debug.assert(walloc.slot_count_per_class.len == 2);
    const gpr_slots = walloc.slot_count_per_class[0];
    const fpr_slots = walloc.slot_count_per_class[1];
    alloc.spill_count = gpr_slots + fpr_slots;

    // Entry params of a leaf: each is pinned by the hint to its ABI arg register, so the prologue
    // needs no move. Collect them (and their ABI arg locations) to require exactly that shape below.
    const eparams = func.blockParams(@enumFromInt(0));
    const plocs = try allocator.alloc(ArgLoc, eparams.len);
    defer allocator.free(plocs);
    computeArgLocs(func, eparams, plocs);

    var it = walloc.segments.iterator();
    while (it.next()) |e| {
        const value = e.key_ptr.*;
        const wsegs = e.value_ptr.*;
        std.debug.assert(wsegs.len > 0);
        const class: u16 = @intFromEnum(regClass(func, value)); // 0 gpr, 1 fpr, matching desc classes
        const param_loc = entryParamArgLoc(eparams, plocs, value);

        if (wsegs.len == 1) {
            switch (wsegs[0].loc) {
                .reg => |ri| {
                    // A LEAF entry param passed in an ABI arg register (idx < 8) gets NO prologue
                    // move: the incoming value already sits in its ABI register. The Wimmer register
                    // hint is only a PREFERENCE, so if the allocator placed a leaf's param anywhere
                    // other than its ABI arg register the prologue would leave the wrong value there.
                    // Guard it (this is a JIT path with release safety off, so bail rather than
                    // assert). A NON-LEAF param has no such requirement: `emitFromAllocation`'s
                    // entry-param loop always emits `mov allocated_reg, incoming_abi_reg` for a
                    // non-leaf (isel.zig ~676-681), so an off-ABI whole-life placement (gap #5, e.g.
                    // a value live across a call parked in a callee-saved register) is realized by
                    // that move; just record the allocated register below.
                    if (leaf) {
                        if (param_loc) |pl| {
                            if (pl.idx < 8 and @as(usize, ri) != pl.idx) return error.Unsupported;
                        }
                    }
                    try alloc.reg.put(allocator, value, @enumFromInt(@as(u5, @intCast(ri))));
                },
                // A whole-life, immediately-spilled entry param (gap #4, e.g. a param whose whole
                // range the allocator decided to keep in memory under pressure): the emission
                // param-setup loop's spilled-param branch already stores the incoming argument
                // (register or caller stack slot) straight into this slot for ANY value recorded
                // in `alloc.spill`, param or not, so recording it here is all this needs.
                .slot => |s| try alloc.spill.put(allocator, value, aarch64Slot(class, s, gpr_slots)),
            }
            continue;
        }

        // A genuinely split value: build the aarch64 segment list AND the per-transition actions.
        // Applies uniformly whether `value` is a param or not (gap #4): a split entry param's
        // `segments[0]` still needs establishing from the incoming ABI argument, but that is
        // emission's job (the per-instruction position it runs at does not exist yet here), so
        // `emitFromAllocation`'s param-setup loop does it by consulting `alloc.segments` directly.
        // Nothing between this `alloc` and the `put` can error (the fill loop and `translateLoc` are
        // infallible), so there is NO errdefer freeing `segs` here: on a `put` failure the explicit
        // catch below frees it, and after `put` succeeds `alloc.segments` OWNS it, so any later error
        // (an `actions.append` OOM; `translateTransition` itself is infallible) is freed exactly once
        // by the function-level `errdefer alloc.deinit`. An errdefer here would double-free that
        // owned slice.
        const segs = try allocator.alloc(Segment, wsegs.len);
        for (wsegs, 0..) |ws, i| {
            segs[i] = .{ .from = ws.from, .loc = translateLoc(ws.loc, class, gpr_slots) };
        }
        alloc.segments.put(allocator, value, segs) catch |err| {
            allocator.free(segs);
            return err;
        };
        // The intra-block re-home actions for these transitions are NOT derived here: the shared
        // allocator already emitted them into `walloc.actions`, ORDERED per same-position cluster into
        // a hazard-free parallel-move sequence (Task 5). Consuming that list below (rather than
        // re-deriving raw per-transition actions and policing them) is what makes the same-position
        // drain always safe and lets the ad-hoc hazard detector retire.
    }

    // Consume the shared allocator's already-ordered intra-block actions. Each is one primitive
    // transfer at its position (`src -> dst` in the shared per-class `Location` space); map both sides
    // into this backend's uniform-slot `Location` and turn it into the matching `SplitAction`. The
    // list is ascending by `at` with each same-position cluster in hazard-free order, so appending it
    // verbatim and draining in order never clobbers a live value (the retired `hasSamePosRegHazard`).
    // A cross-block location change is NOT here (it is an edge move, translated below); only genuine
    // mid-block re-homes appear.
    for (walloc.actions) |wa| {
        std.debug.assert(wa.class == 0 or wa.class == 1);
        const src = translateLoc(wa.src, wa.class, gpr_slots);
        const dst = translateLoc(wa.dst, wa.class, gpr_slots);
        try alloc.actions.append(allocator, try translateTransition(wa.value, src, dst, wa.at));
    }

    // The same-position drain hazard (gap #6) and the call-argument clobber are both handled properly
    // now, so neither ad-hoc detector remains: the intra-block actions consumed above are already
    // parallel-move ordered by the shared allocator, and `emitFromAllocation`'s edge-move-driven call
    // lowering emits the argument setup as a parallel move (stack stores first, then a register
    // permutation through the reserved scratch, then slot reloads).

    // Callee-saved registers the shared allocation used, so the prologue saves them (empty for a
    // leaf, whose pools are caller-saved, but populated for completeness).
    for (walloc.used_callee_saved) |us| {
        const reg: Reg = @enumFromInt(@as(u5, @intCast(us.reg)));
        if (us.class == 0) try alloc.saved_gpr.append(allocator, reg) else try alloc.saved_fpr.append(allocator, reg);
    }
    std.mem.sort(Reg, alloc.saved_gpr.items, {}, regLessThan);
    std.mem.sort(Reg, alloc.saved_fpr.items, {}, regLessThan);

    // Control-flow-edge moves (Task 7 -> Task 8): the shared allocator already ordered each edge's
    // moves into a valid parallel-move sequence, so translate each `wimmer.Location` into this
    // backend's `Location` (per-class slot -> uniform 16-byte slot) keyed by the (pred, succ) block
    // pair and hand the ordered list to emission verbatim. `edge_move_driven` makes emission replay
    // these instead of deriving block-param moves, so a Wimmer function whose values change location
    // across a block (or whose params spill) no longer bails.
    var edge_sets: std.ArrayList(EdgeMoveSet) = .empty;
    errdefer {
        for (edge_sets.items) |es| allocator.free(es.moves);
        edge_sets.deinit(allocator);
    }
    for (walloc.edge_moves) |wem| {
        const moves = try allocator.alloc(EdgeMove, wem.moves.len);
        errdefer allocator.free(moves);
        for (wem.moves, 0..) |wm, i| {
            std.debug.assert(wm.class == 0 or wm.class == 1);
            // `translateLoc` maps a per-class slot through `aarch64Slot` (gpr base 0, fpr base
            // `gpr_slots`), matching the segment/action translation, so both sides key the same frame.
            moves[i] = .{
                .src = translateLoc(wm.src, wm.class, gpr_slots),
                .dst = translateLoc(wm.dst, wm.class, gpr_slots),
                .class = wm.class,
            };
        }
        try edge_sets.append(allocator, .{ .pred = wem.pred, .succ = wem.succ, .moves = moves });
    }
    alloc.edge_moves = try edge_sets.toOwnedSlice(allocator);
    alloc.edge_move_driven = true;

    return alloc;
}

fn regLessThan(_: void, a: Reg, b: Reg) bool {
    return @intFromEnum(a) < @intFromEnum(b);
}

/// The ABI argument location of `v` if it is one of the entry block's parameters, else null.
/// `plocs` is `computeArgLocs(eparams)`, so `plocs[k]` describes `eparams[k]`.
fn entryParamArgLoc(eparams: []const Value, plocs: []const ArgLoc, v: Value) ?ArgLoc {
    for (eparams, plocs) |p, l| if (p == v) return l;
    return null;
}

/// Map a shared `wimmer.Location` to this backend's `Location`: a register index `n` names `x_n`
/// (gpr) or `v_n` (fpr) via the shared enum, and a per-class slot maps through `aarch64Slot`.
fn translateLoc(loc: wimmer.Location, class: u16, gpr_slots: u32) Location {
    return switch (loc) {
        .reg => |ri| .{ .reg = @enumFromInt(@as(u5, @intCast(ri))) },
        .slot => |s| .{ .slot = aarch64Slot(class, s, gpr_slots) },
    };
}

/// Build the drain action realizing `src -> dst` for `value` at `at`: register->slot is a `store`,
/// slot->register a `reload`, register->register a `move`, and slot->slot a `slot_to_slot` (Wimmer
/// bridge gap #7: `emitSplitAction` expands it into a reload-then-store pair through the class
/// scratch, so it never needs a value register of its own). Infallible, but kept `Error!` for symmetry
/// with the rest of the translation pipeline (every arm here always succeeds).
fn translateTransition(value: Value, src: Location, dst: Location, at: u32) Error!SplitAction {
    return switch (src) {
        .reg => |sr| switch (dst) {
            .reg => |dr| SplitAction{ .at = at, .kind = .move, .value = value, .reg = dr, .move_from = sr },
            .slot => |ds| SplitAction{ .at = at, .kind = .store, .value = value, .reg = sr, .slot = ds },
        },
        .slot => |ss| switch (dst) {
            .reg => |dr| SplitAction{ .at = at, .kind = .reload, .value = value, .reg = dr, .slot = ss },
            .slot => |ds| SplitAction{ .at = at, .kind = .slot_to_slot, .value = value, .reg = .zr, .slot = ds, .move_from_slot = ss },
        },
    };
}

fn markUse(last_use: []u32, v: Value, pos: u32) void {
    if (pos > last_use[@intFromEnum(v)]) last_use[@intFromEnum(v)] = pos;
}

/// Visit every operand VALUE used by `inst`, calling `f(ctx, value, is_edge_arg)`. `is_edge_arg` is
/// true for values passed to a successor block (the `.@"if"` branch args), false for ordinary
/// operands. This is the single source of truth for "what does this instruction read".
fn forEachOperand(
    func: *const Function,
    inst: ir.function.Inst,
    fold: *const addrfold.Analysis,
    ctx: anytype,
    comptime f: fn (@TypeOf(ctx), Value, bool) void,
) void {
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
        // A folded load/store attributes its POINTER use to the fold BASE (the add's lhs), not the
        // raw ptr, so the base's live range reaches the mem op (including cross-block) and the dead
        // add's own result gets no use. `baseOf` returns the raw ptr when unfolded, so the
        // non-folding case is byte-identical.
        .load => f(ctx, fold.baseOf(func, inst), false),
        .store => |st| {
            f(ctx, st.value, false);
            f(ctx, fold.baseOf(func, inst), false);
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

/// Visit every operand VALUE used by a terminator, calling `f(ctx, value, is_edge_arg)`. The
/// `.jump` arguments are edge args; the `.ret` value is an ordinary operand. Terminator analogue of
/// `forEachOperand`.
fn forEachTermOperand(
    func: *const Function,
    term: Terminator,
    ctx: anytype,
    comptime f: fn (@TypeOf(ctx), Value, bool) void,
) void {
    switch (term) {
        .ret => |v| if (v) |vv| f(ctx, vv, false),
        .jump => |j| for (func.blockArgs(j)) |a| f(ctx, a, true),
    }
}

const MarkCtx = struct { last_use: []u32, pos: u32 };

fn markOperand(ctx: MarkCtx, v: Value, is_edge_arg: bool) void {
    _ = is_edge_arg;
    markUse(ctx.last_use, v, ctx.pos);
}

/// Thin wrapper over `forEachOperand` that only extends `last_use`. Kept so callers that need
/// nothing but the last-use marking read clearly.
fn forEachUse(func: *const Function, inst: ir.function.Inst, fold: *const addrfold.Analysis, last_use: []u32, pos: u32) void {
    forEachOperand(func, inst, fold, MarkCtx{ .last_use = last_use, .pos = pos }, markOperand);
}

fn forEachTermUse(func: *const Function, term: Terminator, last_use: []u32, pos: u32) void {
    forEachTermOperand(func, term, MarkCtx{ .last_use = last_use, .pos = pos }, markOperand);
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
///
/// This predicate itself carries no model gate; both call sites (the icmp-skip in
/// `emitFromAllocation` and the fused branch in `Ctx.emitIf`) additionally require
/// `caps.fuse_cmp_branch` (threaded onto `Ctx` as `fuse_cmp_branch`) before honoring it, so a
/// model without the fusion falls back to the materialize-then-test path unchanged.
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

/// Whether the `if` at `insts[if_idx]` folds a flag-setting arithmetic op into its branch: the
/// arith at `if_idx-2` is a single-use `add`/`sub`/`bit_and` (register form) or `add`/`sub`
/// (u12-immediate form) whose result is compared eq/ne against a literal `0` by the icmp at
/// `if_idx-1`, which is itself the single-use condition of this if (i.e. the compare-and-branch
/// fold already applies). When it holds, the arith emits its flag-setting S-form
/// (`adds`/`subs`/`ands`, or `addsImm`/`subsImm`) whose Z flag is exactly (result == 0), and the
/// fused `b.cc` branches on eq/ne, eliding the separate `cmp #0`. This is the ONE eligibility
/// predicate shared by BOTH the arith-skip (in the `.arith`/`.arith_imm` arms, via
/// `isArithBranchProducer`) and the S-form emit (in `Ctx.emitIf`), so the two never disagree (no
/// dangling or doubled arith, no stale-flags branch) - mirrors `fusesIntoNextIf`/`fusesIntoNextArith`.
///
/// `enabled` carries `caps.fuse_arith_branch and caps.fuse_cmp_branch` (the fold requires both:
/// it lives inside the compare-and-branch path, so a model without either falls back to the plain
/// arith + cmp + branch at both sites and stays byte-identical).
///
/// Scope (bounded for correctness): eq/ne ONLY (the S-form's Z flag is exactly (result == 0), the
/// eq/ne-against-0 relation; lt/le/gt/ge need N/V reasoning and stay on the plain cmp path). The
/// icmp RHS must be a literal `iconst 0`. Register `add`/`sub`/`bit_and`, or `add`/`sub` with an
/// immediate that fits u12 (a `bit_and` immediate would need a bitmask-encoded `ands` immediate,
/// too complex, so it falls back). Integer / GPR only, all three adjacent in one block, and the
/// arith result single-use (only the icmp reads it, so overwriting its register via the S-form
/// harms nothing).
fn fusesArithIntoBranch(func: *const Function, insts: []const ir.function.Inst, if_idx: usize, enabled: bool) bool {
    if (!enabled) return false;
    if (if_idx < 2) return false; // need the arith at if_idx-2 and the icmp at if_idx-1
    // The compare-and-branch fold must already apply: the icmp at if_idx-1 is a single-use,
    // integer/GPR icmp that is exactly this if's condition (see `fusesIntoNextIf`).
    if (!fusesIntoNextIf(func, insts, if_idx - 1)) return false;
    const cmp = func.opcode(insts[if_idx - 1]).icmp; // an icmp, per fusesIntoNextIf
    // eq/ne only: the S-form's Z flag equals (result == 0), exactly these two relations.
    if (cmp.op != .eq and cmp.op != .ne) return false;
    // The icmp RHS must be a literal 0 (its defining instruction is `iconst 0`).
    const rhs_def = func.definingInst(cmp.rhs) orelse return false;
    switch (func.opcode(rhs_def)) {
        .iconst => |c| if (c != 0) return false,
        else => return false,
    }
    // The icmp LHS must be the result of the arith at if_idx-2.
    const arith_inst = insts[if_idx - 2];
    const arith_result = func.instResult(arith_inst) orelse return false;
    if (cmp.lhs != arith_result) return false;
    // Integer / GPR only (the S-forms are GPR ALU ops; a float/vector arith routes elsewhere).
    if (isVector(func, arith_result) or regClass(func, arith_result) != .gpr) return false;
    // Wide (64-bit or ptr) results are admitted too: `Ctx.emitArithBranch` width-selects the
    // S-form (64-bit `subs64`/`adds64`/`ands64` when the result is wide, the 32-bit forms
    // otherwise), and the surrounding icmp lowering now width-selects `encode.cmp`/`encode.cmp64`
    // by operand width as well, so the S-form's Z flag (result == 0 over the result's full width)
    // agrees with the compare path at every width. No width gate is needed here.
    // Single-use: the arith result is read only by the icmp. Since the arith immediately precedes
    // the icmp and is its LHS, a total use-count of exactly 1 means the icmp is the sole reader, so
    // the S-form overwriting the arith's result register (nothing else reads it) is safe.
    if (countUses(func, arith_result) != 1) return false;
    switch (func.opcode(arith_inst)) {
        .arith => |a| {
            // A shift that fuses into this add/sub (shift_add) is skipped and folded into a
            // shifted-register add, so its shifted operand is never materialized in a register.
            // Reject the arith-branch fold there so the S-form never reads an unmaterialized
            // operand (the two folds are mutually exclusive; fall back to plain arith + cmp +
            // branch). Checked with shift_add enabled unconditionally since it is on by default:
            // when it is off the shl IS materialized and this is merely conservative, and both
            // fold sites share this predicate so they still agree.
            if (if_idx >= 3 and fusesIntoNextShiftAdd(func, insts, if_idx - 3, true)) return false;
            return a.op == .add or a.op == .sub or a.op == .bit_and;
        },
        .arith_imm => |a| {
            // Only add/sub in the immediate form: `bit_and` immediate would need a bitmask-
            // encoded `ands` immediate (complex), so it falls back to the plain path.
            if (a.op != .add and a.op != .sub) return false;
            // The immediate must fit the 12-bit arith-immediate field (else addsImm/subsImm
            // cannot express it). A negative immediate has no unsigned-12 form here either.
            return a.imm >= 0 and a.imm <= 0xFFF;
        },
        else => return false,
    }
}

/// Whether the arith at `insts[inst_idx]` is the producer of an arith-branch fold, i.e. the `if`
/// two instructions later folds it (with the icmp between). The arith-skip sites in the
/// `.arith`/`.arith_imm` arms call this to `continue` past the arith's materialization; the S-form
/// is then emitted by `Ctx.emitIf` under the SAME `fusesArithIntoBranch` predicate, so the arith
/// is emitted exactly once. `enabled` carries `caps.fuse_arith_branch and caps.fuse_cmp_branch`.
fn isArithBranchProducer(func: *const Function, insts: []const ir.function.Inst, inst_idx: usize, enabled: bool) bool {
    if (inst_idx + 2 >= insts.len) return false; // need the icmp at +1 and the if at +2
    return fusesArithIntoBranch(func, insts, inst_idx + 2, enabled);
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

/// Whether the constant left shift `arith_imm{.shl, b, k}` at `insts[idx]` fuses into an
/// immediately-following integer `add`/`sub` that consumes its result, folding
/// `a + (b << k)` / `a - (b << k)` into ONE shifted-register add/sub
/// (`add xd, xn, xm, lsl #k`, see `Ctx.emitShiftAdd`). This is the ONE eligibility predicate
/// shared by the shl-skip (in the `.arith_imm` arm) and the fused emit (in the `.arith`
/// arm), so they never disagree (no dangling or doubled shift) - mirrors `fusesIntoNextArith`.
/// `enabled` carries `caps.fuse_shift_add`, so a model without the fusion (or the flag off)
/// falls back to the plain shift-then-add path at BOTH sites and stays byte-identical.
///
/// Conditions: integer / GPR operands only (a shifted-register add is a GPR-only ALU form,
/// never fpr or vector). The shl result is SINGLE-USE (its only reader is this add/sub, so
/// skipping the standalone shifted value is safe). The shift amount `k` fits the add's
/// result width (0..63 for a 64-bit result, 0..31 for 32-bit, so the imm6 is
/// architecturally valid) and is at least 1 (k == 0 is a degenerate shift left to the plain
/// path). For `.sub` the shl result must be the RHS, the subtrahend: `subShifted` computes
/// `rn - (rm << k)`, so `a - (b << k)` folds but `(b << k) - a` cannot and stays on the plain
/// path. `.add` is commutative, so the shl result may be either operand.
///
/// The immediately-preceding + single-use conditions make skipping the shift register-safe:
/// nothing runs between the shl and the add/sub, so the shl's operand `b` still holds its
/// value there (loaded fresh at the add), exactly as `fusesIntoNextArith` relies on for the
/// fused product's operands.
fn fusesIntoNextShiftAdd(func: *const Function, insts: []const ir.function.Inst, idx: usize, enabled: bool) bool {
    if (!enabled) return false;
    const shl = switch (func.opcode(insts[idx])) {
        .arith_imm => |a| a,
        else => return false,
    };
    if (shl.op != .shl) return false;
    // Integer / GPR operands only: a shifted-register add/sub is a GPR ALU form. A vector or
    // non-gpr class routes elsewhere.
    if (isVector(func, shl.lhs) or regClass(func, shl.lhs) != .gpr) return false;
    if (idx + 1 >= insts.len) return false; // must be immediately followed by the add/sub
    const addsub = switch (func.opcode(insts[idx + 1])) {
        .arith => |a| a,
        else => return false,
    };
    if (addsub.op != .add and addsub.op != .sub) return false;
    const result = func.instResult(insts[idx]) orelse return false;
    if (addsub.lhs != result and addsub.rhs != result) return false; // must consume the shl's result
    // `.sub` folds only when the shl result is the subtrahend (rhs): subShifted computes
    // rn - (rm << k), so `(b << k) - a` has no shifted form and stays on the plain path.
    if (addsub.op == .sub and addsub.lhs == result) return false;
    // `k` must fit the result width's imm6 range and be a real (at least 1) shift.
    const max_shift: i64 = if (isWide(func, result)) 63 else 31;
    if (shl.imm < 1 or shl.imm > max_shift) return false;
    // Single-use: the shifted value is read only by this add/sub. Since the shl immediately
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

fn markUsedBitset(func: *const Function, inst: ir.function.Inst, fold: *const addrfold.Analysis, row: []bool) void {
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
        // Same fold reroute as `forEachOperand`: a folded mem op's pointer use is the fold base, so
        // the base's cross-block liveness reaches the mem op. `baseOf` is the raw ptr when unfolded.
        .load => setUsed(row, fold.baseOf(func, inst)),
        .store => |st| {
            setUsed(row, st.value);
            setUsed(row, fold.baseOf(func, inst));
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
fn extendLiveRanges(allocator: std.mem.Allocator, func: *const Function, last_use: []u32, block_end: []const u32, fold: *const addrfold.Analysis) Error!void {
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
            markUsedBitset(func, inst, fold, row);
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
fn parallelMove(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    moves_in: []const Move,
    movFn: *const fn (Reg, Reg) u32,
    scratch: Reg,
) Error!void {
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
    // The guard bound (2n+4) is provably sufficient to drain every move (each iteration either emits
    // one move or breaks one cycle), so an empty list on exit is an invariant. Assert it: this routine
    // is now load-bearing for call-arg setup, and a silently-undrained move would be a lost register
    // write, not a visible failure.
    std.debug.assert(list.items.len == 0);
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

fn emitLoad(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    func: *const Function,
    result: Value,
    rd: Reg,
    base: Reg,
    off: u32,
    fp16: bool,
) Error!void {
    // A folded load addresses `[base, #off]`; a non-folded load passes off = 0, which every encoder
    // below reproduces byte-identically to its old zero-displacement form. `aarch64FoldOffset`
    // guarantees off fits the per-size scaled range, so the `@intCast`es cannot truncate.
    if (isVector(func, result)) {
        try code.append(allocator, encode.ldrQ(rd, base, @intCast(off))); // 128-bit NEON load
        return;
    }
    if (regClass(func, result) == .fpr) {
        if (isHalf(func, result)) {
            // Load a 16-bit IEEE-half memory object. NOT `ldr s`, which would read 32 bits from a
            // 2-byte object. NATIVE (fp16): `ldr h` leaves the value in the H view ready to use.
            // In the emulation path, also widen it to the S-held f32 form with `fcvt s,h`.
            try code.append(allocator, encode.ldrHfp(rd, base, @intCast(off)));
            if (!fp16) try code.append(allocator, encode.fcvtSfromH(rd, rd));
            return;
        }
        try code.append(allocator, encode.ldrFp(rd, base, @intCast(off), isDouble(func, result)));
        return;
    }
    const sz = typeSize(func, func.valueType(result));
    const signed = isSignedInt(func, result);
    if (sz <= 1) {
        try code.append(allocator, if (signed) encode.ldrsbOff(rd, base, @intCast(off)) else encode.ldrbOff(rd, base, @intCast(off)));
    } else if (sz <= 2) {
        try code.append(allocator, if (signed) encode.ldrshOff(rd, base, @intCast(off)) else encode.ldrhOff(rd, base, @intCast(off)));
    } else if (sz <= 4) {
        try code.append(allocator, encode.ldrW(rd, base, @intCast(off)));
    } else {
        try code.append(allocator, encode.ldrOff(rd, base, @intCast(off)));
    }
}

fn emitStore(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    func: *const Function,
    value: Value,
    val: Reg,
    base: Reg,
    off: u32,
    fp16: bool,
) Error!void {
    // A folded store addresses `[base, #off]`; a non-folded store passes off = 0, which every encoder
    // below reproduces byte-identically to its old zero-displacement form. `aarch64FoldOffset`
    // guarantees off fits the per-size scaled range, so the `@intCast`es cannot truncate.
    if (isVector(func, value)) {
        try code.append(allocator, encode.strQ(val, base, @intCast(off))); // 128-bit NEON store
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
                try code.append(allocator, encode.strHfp(val, base, @intCast(off)));
            } else {
                try code.append(allocator, encode.fcvtHfromS(fp_move, val));
                try code.append(allocator, encode.strHfp(fp_move, base, @intCast(off)));
            }
            return;
        }
        try code.append(allocator, encode.strFp(val, base, @intCast(off), isDouble(func, value)));
        return;
    }
    const sz = typeSize(func, func.valueType(value));
    if (sz <= 1) {
        try code.append(allocator, encode.strbOff(val, base, @intCast(off)));
    } else if (sz <= 2) {
        try code.append(allocator, encode.strhOff(val, base, @intCast(off)));
    } else if (sz <= 4) {
        try code.append(allocator, encode.strW(val, base, @intCast(off)));
    } else {
        try code.append(allocator, encode.strOff(val, base, @intCast(off)));
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
fn emitBinary(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    op: ir.function.BinOp,
    rd: Reg,
    rn: Reg,
    rm: Reg,
    signed: bool,
    wide: bool,
) Error!void {
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

/// Test/measurement hook only: runs the linearization on `func` and returns its `Liveness` so a
/// test can inspect the computed split-liveness data (`use_positions`, `is_intra`). The caller owns
/// the returned `Liveness` and must `deinit` it. Exists so tests can assert the liveness data
/// without threading it through the allocator's public API.
pub fn debugLiveness(allocator: std.mem.Allocator, func: *const Function) Error!Liveness {
    return linearize(allocator, func, &empty_fold);
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

test "intra-block predicate and use positions" {
    // V is defined and used ONLY in its def block (intra-splittable). W is defined in the entry
    // block and passed as an edge argument to a successor (not intra). X is defined in the entry
    // block but used by a normal instruction in a DIFFERENT block (not intra). This exercises the
    // liveness data the splitter needs, without changing any allocation decision.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    const b0 = try func.appendBlock();
    const b1 = try func.appendBlock();
    const a = try func.appendBlockParam(b0, t);
    const bp = try func.appendBlockParam(b0, t);
    const p = try func.appendBlockParam(b1, t);

    const v = try func.appendInst(b0, t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = bp } });
    // V's sole use, in its own def block.
    _ = try func.appendInst(b0, t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = bp } });
    const w = try func.appendInst(b0, t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = bp } });
    const x = try func.appendInst(b0, t, .{ .arith = .{ .op = .mul, .lhs = bp, .rhs = bp } });
    const cond = try func.appendInst(b0, bool_t, .{ .icmp = .{ .op = .le, .lhs = a, .rhs = bp } });
    // W flows to the successor as an edge argument on both branches.
    try func.appendIf(b0, cond, .{ .target = b1, .args = &.{w} }, .{ .target = b1, .args = &.{w} });

    // X is used by a normal instruction in b1, a different block than its def.
    const xuse = try func.appendInst(b1, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = p } });
    func.setTerminator(b1, .{ .ret = xuse });

    var lin = try debugLiveness(allocator, &func);
    defer lin.deinit(allocator);

    try std.testing.expect(lin.is_intra[@intFromEnum(v)]);
    try std.testing.expect(!lin.is_intra[@intFromEnum(w)]);
    try std.testing.expect(!lin.is_intra[@intFromEnum(x)]);
    // Params are never intra-splittable.
    try std.testing.expect(!lin.is_intra[@intFromEnum(a)]);
    try std.testing.expect(!lin.is_intra[@intFromEnum(bp)]);
    try std.testing.expect(!lin.is_intra[@intFromEnum(p)]);

    // V's use positions are non-empty and strictly ascending.
    const vuses = lin.use_positions[@intFromEnum(v)];
    try std.testing.expect(vuses.len > 0);
    var i: usize = 1;
    while (i < vuses.len) : (i += 1) try std.testing.expect(vuses[i] > vuses[i - 1]);
}

test "nextUseAfter finds the first strictly-greater use" {
    try std.testing.expectEqual(@as(?u32, 9), nextUseAfter(&.{ 2, 5, 9 }, 5));
    try std.testing.expectEqual(@as(?u32, 5), nextUseAfter(&.{ 2, 5, 9 }, 2));
    try std.testing.expectEqual(@as(?u32, 2), nextUseAfter(&.{ 2, 5, 9 }, 0));
    try std.testing.expectEqual(@as(?u32, null), nextUseAfter(&.{ 2, 5, 9 }, 9));
    try std.testing.expectEqual(@as(?u32, null), nextUseAfter(&.{}, 0));
}

// ===========================================================================
// Wimmer bridge gap #7: a live-range split whose two adjacent segments are BOTH spill slots.
// `translateTransition` now builds a `.slot_to_slot` action instead of bailing `error.Unsupported`,
// and `emitSplitAction` expands it into reload-then-store through the class scratch.
//
// A NATURAL Wimmer allocation can never reach this shape on aarch64: `aarch64UseKind` makes EVERY
// operand use (including edge arguments) `must_have_register`, and `spillCurrent`'s only two call
// sites in `wimmer.zig`'s `allocateBlockedReg` either (a) split at a value's OWN next must-have use,
// which its next pop is then PROVEN to resolve into a register rather than a second self-spill
// (`current.firstUseAfter(current.start()) == current.start()` can never exceed any candidate's
// `next_use`, which is always `>= current.start()` too, so the "current is cheaper to spill" branch
// can never fire there), or (b) the "no evictable register" branch, which is only safe when
// `current.start()` is NOT itself a use (otherwise `spillCurrent`'s own
// `assert(u > current.start())` fires first, a separate PRE-EXISTING wimmer.zig limitation on
// same-position over-subscription, unrelated to this bridge gap). Empirically confirmed against two
// differential candidates (an asymmetric-use-gap pressure kernel, and a many-block-param loop
// header): the former always resolves reg -> slot -> reg, never slot -> slot, and the latter crashes
// the pre-existing over-subscription assert before ever reaching `translateAllocation`. So these tests
// exercise the exact mechanism (`emitSplitAction`'s `.slot_to_slot` arm) directly with a hand-built
// `SplitAction`, the same way the file's other low-level tests check exact encoded words rather than
// routing through a full differential compile.
// ===========================================================================

test "emitSplitAction .slot_to_slot expands to reload+store through the gpr scratch" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);
    func.setTerminator(b, .{ .ret = v });

    var code: std.ArrayList(u32) = .empty;
    defer code.deinit(allocator);
    const act = SplitAction{ .at = 0, .kind = .slot_to_slot, .value = v, .reg = .zr, .slot = 5, .move_from_slot = 2 };
    try emitSplitAction(allocator, &code, &func, 0, act);

    try std.testing.expectEqual(@as(usize, 2), code.items.len);
    try std.testing.expectEqual(encode.ldrOff(scratch_move, sp, 2 * 16), code.items[0]);
    try std.testing.expectEqual(encode.strOff(scratch_move, sp, 5 * 16), code.items[1]);
}

test "emitSplitAction .slot_to_slot expands to reload+store through the fpr scratch for a scalar float" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f64 });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);
    func.setTerminator(b, .{ .ret = v });

    var code: std.ArrayList(u32) = .empty;
    defer code.deinit(allocator);
    // A non-zero `spill_base` (as a real frame would have) proves the offset math threads through.
    const act = SplitAction{ .at = 0, .kind = .slot_to_slot, .value = v, .reg = .zr, .slot = 1, .move_from_slot = 3 };
    try emitSplitAction(allocator, &code, &func, 4, act);

    try std.testing.expectEqual(@as(usize, 2), code.items.len);
    try std.testing.expectEqual(encode.ldrFp(fp_move, sp, 4 + 3 * 16, true), code.items[0]);
    try std.testing.expectEqual(encode.strFp(fp_move, sp, 4 + 1 * 16, true), code.items[1]);
}

test "emitSplitAction .slot_to_slot round-trips a full 128-bit vector through the fpr scratch" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, v4);
    func.setTerminator(b, .{ .ret = v });

    var code: std.ArrayList(u32) = .empty;
    defer code.deinit(allocator);
    const act = SplitAction{ .at = 0, .kind = .slot_to_slot, .value = v, .reg = .zr, .slot = 7, .move_from_slot = 0 };
    try emitSplitAction(allocator, &code, &func, 0, act);

    try std.testing.expectEqual(@as(usize, 2), code.items.len);
    try std.testing.expectEqual(encode.ldrQ(fp_move, sp, 0 * 16), code.items[0]);
    try std.testing.expectEqual(encode.strQ(fp_move, sp, 7 * 16), code.items[1]);
}
