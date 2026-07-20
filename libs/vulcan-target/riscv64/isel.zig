//! RISC-V instruction selection and register allocation. Lowers a low-profile
//! Vulcan function to machine words: integer/float/RVV-vector arithmetic,
//! control flow, calls, memory, with a liveness-based linear-scan allocator and
//! stack spilling.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const emit = @import("emit.zig");
const schedule = @import("schedule.zig");
const loops = @import("vulcan-opt").loops;
const dominators = @import("vulcan-opt").dominators;
const mm = @import("vulcan-opt").microarch;
const wimmer = @import("../wimmer.zig");
const addrfold = @import("../addrfold.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const BinOp = ir.function.BinOp;
const Reg = encode.Reg;
const FReg = encode.FReg;

/// Where an integer value lives at a given program point: a register or a stack spill slot. A
/// whole-life value has one location for its entire range (today's `int`/`int_spill`). A split
/// value has several, selected by position through `segments`.
const IntLoc = union(enum) { reg: Reg, slot: u32 };

/// Where a scalar-float value lives at a given program point: a float register or a stack spill slot.
/// The analogue of `IntLoc` for the float class. The DEFAULT riscv64 allocator never splits a float
/// (only int splitting exists), so `floatLocationAt` always falls back to `float`/`float_spill` and
/// this is unused on the default path; the shared Wimmer translation is the only producer of a split
/// float (a `float_segments` list).
const FloatLoc = union(enum) { reg: FReg, slot: u32 };

/// Where an RVV vector value lives at a given program point: a vector register or a 16-byte stack
/// spill slot. The vector-class analogue of `IntLoc`/`FloatLoc`. The DEFAULT riscv64 allocator never
/// splits a vector, so `vectorLocationAt` always falls back to `vector`/`vector_spill` and this is
/// unused on the default path; the shared Wimmer translation is the only producer of a split vector
/// (a `vector_segments` list), which is exactly how a vector live across a call is spilled/reloaded.
const VectorLoc = union(enum) { reg: VReg, slot: u32 };

/// Where an et-soc VPU vector value lives at a given program point: a VPU vector register (an FReg in
/// the f16..f27 partition) or a 32-byte stack spill slot. The VPU-class analogue of `VectorLoc`. The
/// DEFAULT riscv64 allocator never splits a VPU value, so `vpuLocationAt` always falls back to
/// `vpu_vector`/`vpu_vector_spill` and this is unused on the default path; the shared Wimmer
/// translation is the only producer of a split VPU value (a `vpu_segments` list), which is exactly how
/// a VPU value live across a call is spilled/reloaded (every f16..f27 is caller-saved).
const VpuLoc = union(enum) { reg: FReg, slot: u32 };

/// One piece of a split integer value's life: the value lives in `loc` from position `from` until
/// the next segment (or, for the last one, to the end of its range). `segments[0].from` is the
/// value's def position, so a lookup at any position at or after the def resolves to some segment.
const Segment = struct { from: usize, loc: IntLoc };

/// One piece of a split scalar-float value's life. The float-class analogue of `Segment`, held in
/// `float_segments` and resolved by `floatLocationAt`. Only the shared Wimmer translation produces
/// these (the native allocator never splits a float).
const FloatSegment = struct { from: usize, loc: FloatLoc };

/// One piece of a split RVV vector value's life. The vector-class analogue of `Segment`, held in
/// `vector_segments` and resolved by `vectorLocationAt`. Only the shared Wimmer translation produces
/// these (the native allocator never splits a vector). A vector live across a call is expressed as a
/// register segment, then a spill-slot segment over the call, then a register segment again.
const VectorSegment = struct { from: usize, loc: VectorLoc };

/// One piece of a split et-soc VPU vector value's life. The VPU-class analogue of `VectorSegment`,
/// held in `vpu_segments` and resolved by `vpuLocationAt`. Only the shared Wimmer translation produces
/// these (the native allocator never splits a VPU value). A VPU value live across a call is expressed
/// as a register segment, then a spill-slot segment over the call, then a register segment again.
const VpuSegment = struct { from: usize, loc: VpuLoc };

/// A store, reload, or register move to emit at position `at`, produced when a value is live-range
/// split. `class` selects the register file: 0 = integer (sd/ld/mv on `spill_base`), 1 = scalar
/// float (fsd|fsw / fld|flw / fmv on `float_spill_base`), 2 = RVV vector (vse32/vle32/vmv.v.v on
/// `vspill_base`, the address computed into `spill_scratch1`), 3 = et-soc VPU vector (fsw.ps/flw.ps on
/// `vpu_vspill_base`, a reg->reg move routed through the `vpu_pack_base` scratch since the VPU has no
/// packed register move). A `.store` writes the value's register to
/// its slot at a split boundary; a `.reload` brings the slot back into a register (the second-chance
/// re-home, Task 6d); a `.move` copies one register into another (a register-to-register re-home,
/// produced only by the shared Wimmer translation). The native `allocateRegisters` only ever
/// produces class-0 `.store`/`.reload`, so it leaves every added field at its default and emission is
/// byte-identical. Drained in the emission loop in `at` order (see `emitSplitAction`).
const SplitAction = struct {
    at: usize,
    kind: enum { store, reload, move },
    /// 0 = integer file (`reg`/`from_reg`), 1 = scalar-float file (`freg`/`from_freg`), 2 = RVV vector
    /// (`vreg`/`from_vreg`), 3 = et-soc VPU vector (`freg`/`from_freg`, the FReg partition f16..f27).
    class: u8 = 0,
    value: Value,
    /// Integer store source / reload or move destination (class 0).
    reg: Reg = .x0,
    /// Float store source / reload or move destination (class 1 scalar float, and class 3 VPU vector:
    /// both ride the FReg file, disambiguated by `class`, never by index).
    freg: FReg = .f0,
    /// Vector store source / reload or move destination (class 2).
    vreg: VReg = .v0,
    /// Move source register; `reg`/`freg`/`vreg` is the destination. Class picks which is read.
    from_reg: Reg = .x0,
    from_freg: FReg = .f0,
    from_vreg: VReg = .v0,
    slot: u32 = 0,
};

/// One precomputed control-flow-edge move (the shared Wimmer path only), translated from a
/// `wimmer.Move`: shuffle `src` into `dst` within `class` (0 int, 1 scalar float, 2 RVV vector, 3
/// et-soc VPU vector). The register index is stored raw and decoded to a `Reg`/`FReg`/`VReg` per class
/// at emission. The shared allocator already
/// ORDERED these into a valid parallel-move sequence (every source read before it is overwritten,
/// cycles broken and any slot<->slot shuffle routed through the class scratch), so the emitter
/// replays them op-by-op with no reordering.
const EdgeLoc = union(enum) { reg: u16, slot: u32 };
const EdgeMove = struct { class: u8, src: EdgeLoc, dst: EdgeLoc };

/// The ordered move list on one control-flow edge `pred -> succ` (translated from `wimmer.EdgeMoves`).
/// Keyed by the block pair so the `.jump` emission can find the moves for the edge it lowers.
const EdgeMoveSet = struct { pred: Block, succ: Block, moves: []EdgeMove };

/// Register-file assignment. A value lives in an int, float, or vector register,
/// or (when the file is exhausted) spills to a numbered stack slot.
const Allocation = struct {
    int: std.AutoHashMapUnmanaged(Value, Reg),
    float: std.AutoHashMapUnmanaged(Value, FReg),
    /// SIMD vector values, mapped to an RVV vector register (v1..v27). Empty in vpu mode.
    vector: std.AutoHashMapUnmanaged(Value, VReg),
    /// Spilled vector values, mapped to a 16-byte spill-slot index (0-based). Empty in vpu mode.
    vector_spill: std.AutoHashMapUnmanaged(Value, u32),
    vector_spill_count: u32,
    /// et-soc VPU vector values, mapped to an FReg in the disjoint f16..f27 pool. Empty unless
    /// this function was allocated in vpu mode.
    vpu_vector: std.AutoHashMapUnmanaged(Value, FReg),
    /// Spilled VPU vector values, mapped to a 32-byte spill-slot index (0-based). Empty unless
    /// this function was allocated in vpu mode.
    vpu_vector_spill: std.AutoHashMapUnmanaged(Value, u32),
    vpu_vector_spill_count: u32,
    /// Spilled integer values, mapped to their spill-slot index (0-based). The
    /// frame layout turns the index into an `sp` offset.
    int_spill: std.AutoHashMapUnmanaged(Value, u32),
    spill_count: u32,
    /// Spilled scalar-float values, mapped to their spill-slot index (0-based). Mirrors
    /// `int_spill`: the frame layout turns the index into an `sp` offset (8-byte slots, an f32
    /// occupies the low 4 bytes). Populated when the scalar float file is exhausted.
    float_spill: std.AutoHashMapUnmanaged(Value, u32),
    float_spill_count: u32,
    /// Entry integer parameters beyond the 8 argument registers: each maps to its
    /// incoming stack-argument index (0 = the 9th arg). The selector loads it from
    /// the caller's frame at function entry.
    incoming_stack: std.AutoHashMapUnmanaged(Value, u32),
    /// Split integer values only: value -> ascending-by-`from` segment list. Empty means no value
    /// was split, so `intLocationAt` falls back to `int`/`int_spill` and emission is byte-identical
    /// to before splitting.
    segments: std.AutoHashMapUnmanaged(Value, []Segment) = .empty,
    /// Split scalar-float values only (the shared Wimmer path): value -> ascending-by-`from` float
    /// segment list. Empty on the default path (the native allocator never splits a float), so
    /// `floatLocationAt` falls back to `float`/`float_spill` and emission is byte-identical.
    float_segments: std.AutoHashMapUnmanaged(Value, []FloatSegment) = .empty,
    /// Split RVV vector values only (the shared Wimmer path): value -> ascending-by-`from` vector
    /// segment list. Empty on the default path (the native allocator never splits a vector), so
    /// `vectorLocationAt` falls back to `vector`/`vector_spill` and emission is byte-identical. A
    /// vector live across a call lands here (register, then slot over the call, then register).
    vector_segments: std.AutoHashMapUnmanaged(Value, []VectorSegment) = .empty,
    /// Split et-soc VPU vector values only (the shared Wimmer path): value -> ascending-by-`from` VPU
    /// segment list. Empty on the default path (the native allocator never splits a VPU value), so
    /// `vpuLocationAt` falls back to `vpu_vector`/`vpu_vector_spill` and emission is byte-identical. A
    /// VPU value live across a call lands here (register, then 32-byte slot over the call, then register).
    vpu_segments: std.AutoHashMapUnmanaged(Value, []VpuSegment) = .empty,
    /// Precomputed, ordered control-flow-edge moves (the shared Wimmer path only). When
    /// `edge_move_driven` is set the `.jump` emission replays these per edge and derives no block-param
    /// moves itself. Empty + false is the DEFAULT path, whose edge lowering stays byte-identical.
    edge_moves: []EdgeMoveSet = &.{},
    edge_move_driven: bool = false,
    /// Per value: its definition position, in the same linear numbering as the liveness pass. The
    /// emission pos-coupling assert reads it. Always a heap-owned dupe (see `deinit`), so the `&.{}`
    /// sentinel is a zero-length slice with no backing allocation and freeing it is a no-op.
    def_pos: []usize = &.{},
    /// Store/reload actions to drain during emission, one per split boundary, appended in ascending
    /// `at` order. Empty when no value was split, so emission is byte-identical to before splitting.
    actions: std.ArrayList(SplitAction) = .empty,
    /// Test/measurement only: how many times `secondChance` found a slot-resident split value with an
    /// upcoming use but NO free integer register to re-home it (the decline path). Never read by
    /// codegen, so it does not affect emission. A single counter, so it costs nothing at runtime.
    rehome_declines: u32 = 0,

    fn deinit(self: *Allocation, allocator: std.mem.Allocator) void {
        self.int.deinit(allocator);
        self.float.deinit(allocator);
        self.vector.deinit(allocator);
        self.vector_spill.deinit(allocator);
        self.vpu_vector.deinit(allocator);
        self.vpu_vector_spill.deinit(allocator);
        self.int_spill.deinit(allocator);
        self.float_spill.deinit(allocator);
        self.incoming_stack.deinit(allocator);
        var seg_it = self.segments.valueIterator();
        while (seg_it.next()) |segs| allocator.free(segs.*);
        self.segments.deinit(allocator);
        var fseg_it = self.float_segments.valueIterator();
        while (fseg_it.next()) |segs| allocator.free(segs.*);
        self.float_segments.deinit(allocator);
        var vseg_it = self.vector_segments.valueIterator();
        while (vseg_it.next()) |segs| allocator.free(segs.*);
        self.vector_segments.deinit(allocator);
        var pseg_it = self.vpu_segments.valueIterator();
        while (pseg_it.next()) |segs| allocator.free(segs.*);
        self.vpu_segments.deinit(allocator);
        for (self.edge_moves) |em| allocator.free(em.moves);
        allocator.free(self.edge_moves);
        allocator.free(self.def_pos);
        self.actions.deinit(allocator);
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

/// Caller-saved float temporaries (ft0-ft7, ft8-ft9). ft10 (f30) and ft11 (f31) are reserved as
/// the two float spill scratch registers (`float_spill_scratch0`/`1` below) rather than allocated,
/// so they are kept out of this pool.
const float_temp_regs = [_]FReg{ .f0, .f1, .f2, .f3, .f4, .f5, .f6, .f7, .f28, .f29 };

/// Scratch registers for reloading/storing spilled scalar-float values. Two are needed so a binary
/// float op with both operands spilled can reload both. In non-vpu mode these are f30/f31, both
/// caller-saved (ft10/ft11, so no callee-save slot is needed) and both kept out of the allocatable
/// float pool. f31 doubles as the parallel-move `float_scratch`: safe because operand-spill reloads
/// happen mid-block during instruction emit while float edge moves happen only at a block
/// terminator, so the two uses of f31 are never live at the same time. In vpu mode the scratches
/// are f8/f9 instead (see `float_spill_scratch0_vpu`), since f30/f31 sit inside the vpu vector
/// partition (f16..f31).
const float_spill_scratch0: FReg = .f30;
const float_spill_scratch1: FReg = .f31;
/// vpu-mode float spill scratch: f8/f9 (fs0/fs1), reserved out of the vpu scalar pool (which is then
/// just f0..f7) and disjoint from the vpu vector partition (f16..f31). They are callee-saved, so a
/// vpu function that actually spills a scalar float preserves them in its frame (see the frame
/// layout). f31 (the parallel-move `float_scratch`) stays valid in vpu mode too: it is reserved
/// headroom in the vpu vector partition that nothing in this lowering draws on, so it never aliases
/// a real value during a scalar-float edge move.
const float_spill_scratch0_vpu: FReg = .f8;
const float_spill_scratch1_vpu: FReg = .f9;

/// Reserved float scratch for cycle-breaking parallel moves across a jump edge (f31, the RVV/float
/// analogue of `vector_scratch`). It is kept out of `float_temp_regs` (so it is never an allocatable
/// float register in non-vpu mode) and is already reserved headroom in vpu mode (outside every vpu
/// pool), so it is safe as a scratch in both modes: never a move source or destination.
const float_scratch: FReg = .f31;

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

// --- et-soc VPU (CORE-ET Erbium packed-single) mode ---
//
// The VPU has no separate vector register file: its 8-lane f32 registers ARE f0..f31, the same
// file scalar floats use. With zero execution feedback (no emulator decodes these custom opcodes,
// see encode.zig), unifying the scalar and vector allocators is too risky to prove correct here. So
// `vpu` mode instead partitions the file in half at comptime: scalar floats only ever draw from
// f0..f9, VPU vectors only ever draw from f16..f31. The two halves can never alias, by construction,
// with zero runtime check needed. fa0..fa5 (f10..f15) still carry the first six ABI float
// arguments directly; a 7th+ float argument would land in fa6/fa7 (f16/f17), inside the vector
// half, so `allocateRegisters` rejects that case instead (see the vpu bound check there).

/// VPU vector pool: f16..f27 (12 registers), disjoint from the vpu-mode scalar pool below.
/// f28..f31 are reserved VPU scratch, mirroring the RVV vec_op0/op1/work/vector_scratch scheme.
const vpu_vector_regs = [_]FReg{
    .f16, .f17, .f18, .f19, .f20, .f21, .f22, .f23, .f24, .f25, .f26, .f27,
};
const vpu_vec_op0: FReg = .f28;
const vpu_vec_op1: FReg = .f29;
const vpu_vec_work: FReg = .f30;
// f31 is reserved headroom in the VPU vector partition (kept out of `vpu_vector_regs` and every
// allocatable pool above), mirroring the RVV vec_op0/op1/work/vector_scratch scheme, but nothing
// in this file's lowering (struct_new/extract included) currently draws on it: those use
// vpu_vec_work/vpu_vec_op0 above.

/// vpu-mode scalar float temporaries: f0..f7 (the subset of `float_temp_regs` that stays clear of
/// the vpu vector partition, f16..f31).
const float_temp_regs_vpu = [_]FReg{ .f0, .f1, .f2, .f3, .f4, .f5, .f6, .f7 };
/// vpu-mode scalar float callee-saved registers: empty. f8/f9 (fs0/fs1), the only callee-saved
/// float pair clear of the vpu vector partition, are reserved as `float_spill_scratch0/1_vpu`
/// rather than allocated, so the vpu scalar float pool is exactly f0..f7.
const float_saved_regs_vpu = [_]FReg{};

fn isFloatSavedReg(reg: FReg) bool {
    for (float_saved_regs) |s| {
        if (s == reg) return true;
    }
    return false;
}

fn isFloatSavedRegVpu(reg: FReg) bool {
    for (float_saved_regs_vpu) |s| {
        if (s == reg) return true;
    }
    return false;
}

/// Whether `ty` is an 8-lane VPU vector type (the only width the VPU path lowers). A vector of any
/// other width in vpu mode is a shape this path cannot serve.
fn isVpuWidth(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .vector => |v| v.len == 8,
        else => false,
    };
}

/// Whether `ty` is a 4-lane RVV vector type (the only width the RVV path below lowers: it hardcodes
/// VL=4 in the `vsetivli` preamble).
fn isRvvWidth(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .vector => |v| v.len == 4,
        else => false,
    };
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

/// Whether `ty` is a vector whose scalar element is an integer. Under the VPU (`vpu`) this selects
/// the CORE-ET packed-integer (`pi`) lowering; a float-element vector selects the packed-single
/// (`ps`) lowering instead. The lane scalars of a `<N x i32>` are plain i32 that live in (and spill
/// from) the INT register file, so packing/unpacking them costs no scalar-float-pool pressure.
fn isIntVector(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .vector => |v| func.types.type_kind(v.elem) == .int,
        else => false,
    };
}

/// Whether the scalar element of the vector type `ty` is an unsigned integer. Programmer error if
/// `ty` is not an integer-element vector (callers gate with `isIntVector`); it drives the logical
/// vs. arithmetic choice for a packed-integer right shift.
fn isUnsignedIntVector(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .vector => |v| switch (func.types.type_kind(v.elem)) {
            .int => |i| i.signedness == .unsigned,
            else => unreachable, // callers gate with isIntVector
        },
        else => unreachable, // callers gate with isIntVector
    };
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

/// A shared no-fold analysis for the paths that must stay fold-agnostic (the shared Wimmer
/// differential compile). Its `baseOf`/`offOf`/`isDeadAdd` behave as if nothing folded, so those
/// paths emit byte-identical code to before address folding existed.
const empty_fold: addrfold.Analysis = addrfold.Analysis.empty;

/// The riscv64 fold predicate for `addrfold.analyze`: fold a load/store whose pointer is an
/// `arith_imm.add(base, imm)` when `imm` fits the signed 12-bit displacement of the base+offset
/// load/store forms (`ld`/`lw`/`flw`/`fld`/... and their stores), i.e. imm in [-2048, 2047]. Returns
/// the byte offset (equal to the add's imm) when in range, else null. `analyze` calls this only after
/// confirming the pointer is an `arith_imm.add`, so the unwraps below are guaranteed, still asserted.
///
/// NEVER folds a vector load/store, RVV OR VPU. RVV `vle32`/`vse32` have NO immediate operand, so a
/// folded displacement would be impossible to encode; refusing the VPU `flw.ps`/`fsw.ps` too (they do
/// carry an imm12) keeps the RVV constraint impossible to violate and costs only a marginal, here-
/// unexecuted VPU win. So only SCALAR int and SCALAR float ever fold.
fn riscv64FoldOffset(_: void, func: *const Function, mem_inst: ir.function.Inst) ?i64 {
    const val = switch (func.opcode(mem_inst)) {
        .load => func.instResult(mem_inst).?, // the loaded value decides the access class
        .store => |st| st.value, // the stored value decides the access class
        else => unreachable, // analyze only hands foldOffset a load or store
    };
    // Refuse every vector load/store unconditionally (see the doc comment).
    if (isVector(func, func.valueType(val))) return null;
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
    if (add.imm < -2048 or add.imm > 2047) return null;
    return add.imm;
}

fn isFloatTempRegVpu(reg: FReg) bool {
    for (float_temp_regs_vpu) |t| {
        if (t == reg) return true;
    }
    return false;
}

pub const Error = std.mem.Allocator.Error || error{Unsupported};

/// Which B-type branch a fused compare-and-branch uses. Chosen from the icmp's
/// CmpOp and operand signedness (see `branchFor`); `emit` re-encodes it at patch
/// time once the offset is known.
const BranchKind = enum {
    beq,
    bne,
    blt,
    bge,
    bltu,
    bgeu,

    fn emit(self: BranchKind, rs1: Reg, rs2: Reg, off: i13) u32 {
        return switch (self) {
            .beq => encode.beq(rs1, rs2, off),
            .bne => encode.bne(rs1, rs2, off),
            .blt => encode.blt(rs1, rs2, off),
            .bge => encode.bge(rs1, rs2, off),
            .bltu => encode.bltu(rs1, rs2, off),
            .bgeu => encode.bgeu(rs1, rs2, off),
        };
    }

    /// The logically-negated branch: taken exactly when `self` is not-taken. Used by
    /// branch relaxation to build an inverted short branch that skips over a `jal`
    /// carrying the far target (beq<->bne, blt<->bge, bltu<->bgeu).
    fn invert(self: BranchKind) BranchKind {
        return switch (self) {
            .beq => .bne,
            .bne => .beq,
            .blt => .bge,
            .bge => .blt,
            .bltu => .bgeu,
            .bgeu => .bltu,
        };
    }
};

/// A branch/jump whose target offset is patched once block positions are known.
const Fixup = struct {
    index: usize,
    target: Block,
    kind: union(enum) {
        /// The plain materialize-then-test path: `bne cond, x0, off`.
        branch: Reg,
        /// A fused compare-and-branch on two real operands (the boolean is skipped):
        /// re-encodes `b<cc> rs1, rs2, off` with the chosen B-type branch.
        cbranch: struct { kind: BranchKind, rs1: Reg, rs2: Reg },
        jal,
    },
};

/// The signed 13-bit B-type branch reach, in bytes: offsets outside `[-4096, 4094]`
/// (bit 0 is always 0, so 4095 is unrepresentable) cannot be encoded and force
/// relaxation to the long form.
const b_type_min: i64 = -4096;
const b_type_max: i64 = 4094;
/// The signed 21-bit J-type `jal` reach, in bytes: ±1MiB. A jump farther than this
/// (a genuinely huge function) is rejected cleanly rather than wrapped/panicked.
const j_type_min: i64 = -1048576;
const j_type_max: i64 = 1048574;

/// Count how many conditional-branch fixups marked `long` sit strictly before word
/// index `idx` in the ORIGINAL layout. Each long branch expands from one word (the
/// short branch) to two (inverted short branch + far `jal`), so this is exactly the
/// number of EXTRA words relaxation inserts before `idx`. `long[i]` is only ever set
/// for `.branch`/`.cbranch` fixups, so no other kind is counted.
fn extraBeforeWord(fixups: []const Fixup, long: []const bool, idx: usize) usize {
    var n: usize = 0;
    for (fixups, 0..) |fx, i| {
        if (long[i] and fx.index < idx) n += 1;
    }
    return n;
}

/// The fused branch (encoder + operand order) that takes the then-edge under exactly
/// the condition the icmp's boolean would be true. Mirrors the slt/sltu selection in
/// the icmp lowering: gt/le swap operands to reuse the lt/ge forms, unsigned operands
/// use the u-forms. `unsigned` comes from the icmp operand type.
fn branchFor(op: ir.function.CmpOp, unsigned: bool) struct { kind: BranchKind, swap: bool } {
    return switch (op) {
        .eq => .{ .kind = .beq, .swap = false },
        .ne => .{ .kind = .bne, .swap = false },
        .lt => .{ .kind = if (unsigned) .bltu else .blt, .swap = false },
        .gt => .{ .kind = if (unsigned) .bltu else .blt, .swap = true },
        .ge => .{ .kind = if (unsigned) .bgeu else .bge, .swap = false },
        .le => .{ .kind = if (unsigned) .bgeu else .bge, .swap = true },
    };
}

/// The temporary registers used for instruction results. x6 is reserved as a
/// scratch register for helper sequences (e.g. materializing float constants).
const temp_regs = [_]Reg{ .x5, .x7, .x28, .x29, .x30, .x31 };
const scratch_reg: Reg = .x6;

/// The allocatable integer temp pool when the function uses f16. riscv64 has no hardware f16
/// (no Zfh), so every f16 boundary emits an inline software convert (see `emitHalfToFloat` /
/// `emitFloatToHalf`) that needs several dedicated scratch GPRs. x28..x31 (t3..t6) are reserved
/// out of the allocatable pool for exactly that when f16 is present, leaving x5/x7 as the only
/// caller-saved temps; the eleven callee-saved registers still back the rest. A non-f16 function
/// keeps the full `temp_regs` and is byte-identical to before, so nothing else regresses.
const temp_regs_f16 = [_]Reg{ .x5, .x7 };

/// The four dedicated f16 software-convert scratch GPRs (t3..t6), reserved out of the allocatable
/// pool whenever the function uses f16. Together with `scratch_reg` (x6) and `spill_scratch1` (x8)
/// - both already reserved out of every pool - they give the convert routines six free GPRs, which
/// is exactly what the round-to-nearest-even float->half truncate needs. None can ever alias a
/// value-carrying register (a base pointer, an operand), so the sequences never clobber live state.
const f16_scratch_a: Reg = .x28;
const f16_scratch_b: Reg = .x29;
const f16_scratch_c: Reg = .x30;
const f16_scratch_d: Reg = .x31;

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
/// The location of integer value `v` at position `pos`: its active segment if `v` was split,
/// otherwise its whole-life register or spill slot. With no splits (segments empty) this is exactly
/// today's int/int_spill lookup.
fn intLocationAt(alloc: *const Allocation, v: Value, pos: usize) IntLoc {
    if (alloc.segments.get(v)) |segs| {
        var chosen = segs[0]; // non-empty, ascending by `from`
        for (segs) |s| {
            if (s.from <= pos) chosen = s else break;
        }
        return chosen.loc;
    }
    if (alloc.int.get(v)) |r| return .{ .reg = r };
    return .{ .slot = alloc.int_spill.get(v).? };
}

fn reloadInt(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, spill_base: u32, v: Value, pos: usize, scratch: Reg) std.mem.Allocator.Error!Reg {
    switch (intLocationAt(alloc, v, pos)) {
        .reg => |r| return r,
        .slot => |slot| {
            const off: i12 = @intCast(spill_base + slot * 8);
            try code.append(allocator, encode.ld(scratch, .x2, off));
            return scratch;
        },
    }
}

/// The location of RVV vector value `v` at position `pos`: its active segment if `v` was split,
/// otherwise its whole-life register or spill slot. The vector-class analogue of `intLocationAt`/
/// `floatLocationAt`. With no splits (`vector_segments` empty, every default-path function) this is
/// exactly the old `vector`/`vector_spill` lookup, so callers threaded through it stay byte-identical.
fn vectorLocationAt(alloc: *const Allocation, v: Value, pos: usize) VectorLoc {
    if (alloc.vector_segments.get(v)) |segs| {
        var chosen = segs[0]; // non-empty, ascending by `from`
        for (segs) |s| {
            if (s.from <= pos) chosen = s else break;
        }
        return chosen.loc;
    }
    if (alloc.vector.get(v)) |r| return .{ .reg = r };
    return .{ .slot = alloc.vector_spill.get(v).? };
}

/// Reload a vector `v` into `scratch` if it lives in a slot at `pos` (vle32 from its 16-byte slot,
/// whose address is computed into `addr`), else return the vector register it lives in there. `pos`
/// selects a split value's active segment; with no splits it is unobservable (whole-life fallback),
/// so the default path is byte-identical.
fn reloadVector(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, vspill_base: u32, v: Value, pos: usize, scratch: VReg, addr: Reg) std.mem.Allocator.Error!VReg {
    switch (vectorLocationAt(alloc, v, pos)) {
        .reg => |vr| return vr,
        .slot => |slot| {
            const off: i12 = @intCast(vspill_base + slot * 16);
            try code.append(allocator, encode.addi(addr, .x2, off));
            try code.append(allocator, encode.vle32(scratch, addr));
            return scratch;
        },
    }
}
/// The vector register to compute `v` into at `pos`: its assigned register, or `scratch` if it lives
/// in a slot there. At a def position a split value's first segment is `.reg`, so `.slot` means a
/// wholly-spilled value, identical to the old `vector.get orelse scratch`.
fn dstVector(alloc: *const Allocation, v: Value, pos: usize, scratch: VReg) VReg {
    return switch (vectorLocationAt(alloc, v, pos)) {
        .reg => |r| r,
        .slot => scratch,
    };
}
/// Store a freshly-computed vector `v` (in `vr`) back to its spill slot, if it lives in a slot at
/// `pos` (vse32 to its 16-byte slot, whose address is computed into `addr`).
fn storeVector(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, vspill_base: u32, v: Value, pos: usize, vr: VReg, addr: Reg) std.mem.Allocator.Error!void {
    switch (vectorLocationAt(alloc, v, pos)) {
        .reg => {},
        .slot => |slot| {
            const off: i12 = @intCast(vspill_base + slot * 16);
            try code.append(allocator, encode.addi(addr, .x2, off));
            try code.append(allocator, encode.vse32(vr, addr));
        },
    }
}

/// The location of et-soc VPU vector value `v` at position `pos`: its active segment if `v` was
/// split, otherwise its whole-life FReg or spill slot. The VPU-class analogue of `vectorLocationAt`.
/// With no splits (`vpu_segments` empty, every default-path vpu function) this is exactly the old
/// `vpu_vector`/`vpu_vector_spill` lookup, so callers threaded through it stay byte-identical.
fn vpuLocationAt(alloc: *const Allocation, v: Value, pos: usize) VpuLoc {
    if (alloc.vpu_segments.get(v)) |segs| {
        var chosen = segs[0]; // non-empty, ascending by `from`
        for (segs) |s| {
            if (s.from <= pos) chosen = s else break;
        }
        return chosen.loc;
    }
    if (alloc.vpu_vector.get(v)) |fr| return .{ .reg = fr };
    return .{ .slot = alloc.vpu_vector_spill.get(v).? };
}

/// Reload a vpu-mode vector `v` into `scratch` if it lives in a slot at `pos` (`flw.ps` from its
/// 32-byte slot on `sp`), else return the FReg it lives in there. Unlike `reloadVector`, no address
/// register is needed: `flw.ps` (like scalar `flw`) carries its own 12-bit displacement. `pos` selects
/// a split value's active segment; with no splits it is unobservable (whole-life fallback), so the
/// default path is byte-identical.
fn reloadVpuVector(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, vpu_vspill_base: u32, v: Value, pos: usize, scratch: FReg) std.mem.Allocator.Error!FReg {
    switch (vpuLocationAt(alloc, v, pos)) {
        .reg => |fr| return fr,
        .slot => |slot| {
            const off: i12 = @intCast(vpu_vspill_base + slot * 32);
            try code.append(allocator, encode.flw_ps(scratch, .x2, off));
            return scratch;
        },
    }
}
/// The FReg to compute vpu-mode vector `v` into at `pos`: its assigned register, or `scratch` if it
/// lives in a slot there. At a def position a split value's first segment is `.reg`, so `.slot` means a
/// wholly-spilled value, identical to the old `vpu_vector.get orelse scratch`.
fn dstVpuVector(alloc: *const Allocation, v: Value, pos: usize, scratch: FReg) FReg {
    return switch (vpuLocationAt(alloc, v, pos)) {
        .reg => |fr| fr,
        .slot => scratch,
    };
}
/// Store a freshly-computed vpu-mode vector `v` (in `fr`) back to its spill slot, if it lives in a
/// slot at `pos` (`fsw.ps` to its 32-byte slot on `sp`).
fn storeVpuVector(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, vpu_vspill_base: u32, v: Value, pos: usize, fr: FReg) std.mem.Allocator.Error!void {
    switch (vpuLocationAt(alloc, v, pos)) {
        .reg => {},
        .slot => |slot| {
            const off: i12 = @intCast(vpu_vspill_base + slot * 32);
            try code.append(allocator, encode.fsw_ps(fr, .x2, off));
        },
    }
}

/// The location of scalar-float value `v` at position `pos`: its active segment if `v` was split,
/// otherwise its whole-life register or spill slot. The float-class analogue of `intLocationAt`. With
/// no splits (`float_segments` empty, every default-path function) this is exactly the old
/// `float`/`float_spill` lookup, so callers threaded through it stay byte-identical.
fn floatLocationAt(alloc: *const Allocation, v: Value, pos: usize) FloatLoc {
    if (alloc.float_segments.get(v)) |segs| {
        var chosen = segs[0]; // non-empty, ascending by `from`
        for (segs) |s| {
            if (s.from <= pos) chosen = s else break;
        }
        return chosen.loc;
    }
    if (alloc.float.get(v)) |r| return .{ .reg = r };
    return .{ .slot = alloc.float_spill.get(v).? };
}

/// Resolve a scalar-float operand to a register: if `v` lives in a float register at `pos`, return
/// it. If it lives in a slot, reload it from there into `scratch` and return `scratch`. `d64` picks
/// the load width (fld for f64, flw for f32); spilled values occupy a full 8-byte slot regardless (an
/// f32 uses the low 4 bytes), mirroring `reloadInt`. `pos` selects a split value's active segment;
/// with no splits it is unobservable (whole-life fallback), so the default path is byte-identical.
fn reloadFloat(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, float_spill_base: u32, v: Value, pos: usize, d64: bool, scratch: FReg) std.mem.Allocator.Error!FReg {
    switch (floatLocationAt(alloc, v, pos)) {
        .reg => |r| return r,
        .slot => |slot| {
            const off: i12 = @intCast(float_spill_base + slot * 8);
            try code.append(allocator, if (d64) encode.fld(scratch, .x2, off) else encode.flw(scratch, .x2, off));
            return scratch;
        },
    }
}
/// The float register to compute `v` into at `pos`: its assigned register, or `scratch` if it lives
/// in a slot there. At a def position a split value's first segment is `.reg`, so `.slot` means a
/// wholly-spilled value, identical to the old `float.get orelse scratch`.
fn dstFloat(alloc: *const Allocation, v: Value, pos: usize, scratch: FReg) FReg {
    return switch (floatLocationAt(alloc, v, pos)) {
        .reg => |r| r,
        .slot => scratch,
    };
}
/// Store a freshly-computed scalar-float `v` (in `fr`) back to its spill slot, if it lives in a slot
/// at `pos`. `d64` picks the store width (fsd for f64, fsw for f32).
fn storeFloat(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, float_spill_base: u32, v: Value, pos: usize, d64: bool, fr: FReg) std.mem.Allocator.Error!void {
    switch (floatLocationAt(alloc, v, pos)) {
        .reg => {},
        .slot => |slot| {
            const off: i12 = @intCast(float_spill_base + slot * 8);
            try code.append(allocator, if (d64) encode.fsd(fr, .x2, off) else encode.fsw(fr, .x2, off));
        },
    }
}

/// Emit one split-boundary action (see `SplitAction`). Class 0 is the integer file (sd/ld to
/// `spill_base`, `mv` for a re-home), class 1 the scalar-float file (fsd|fsw / fld|flw to
/// `float_spill_base`, `fmv` for a re-home, the width taken from the value's type). The native
/// `allocateRegisters` only ever produces class-0 `.store`/`.reload`, so those arms are byte-identical
/// to the inline drain they replace; `.move`, class 1, class 2 and class 3 are reachable only through
/// the Wimmer path. Class 3 (et-soc VPU) stores/reloads a 32-byte packed slot with `fsw.ps`/`flw.ps`
/// on `vpu_vspill_base`; its reg->reg re-home has no packed move instruction, so it round-trips
/// through the reserved 32-byte `vpu_pack_base` scratch slot (`fsw.ps` then `flw.ps`).
fn emitSplitAction(allocator: std.mem.Allocator, code: *std.ArrayList(u32), func: *const Function, spill_base: u32, float_spill_base: u32, vspill_base: u32, vpu_vspill_base: u32, vpu_pack_base: u32, act: SplitAction) std.mem.Allocator.Error!void {
    switch (act.class) {
        0 => switch (act.kind) {
            .store => try code.append(allocator, encode.sd(act.reg, .x2, @intCast(spill_base + act.slot * 8))),
            .reload => try code.append(allocator, encode.ld(act.reg, .x2, @intCast(spill_base + act.slot * 8))),
            .move => if (act.reg != act.from_reg) try code.append(allocator, encode.addi(act.reg, act.from_reg, 0)),
        },
        1 => {
            const off: i12 = @intCast(float_spill_base + act.slot * 8);
            const d64 = is64Float(func, func.valueType(act.value));
            switch (act.kind) {
                .store => try code.append(allocator, if (d64) encode.fsd(act.freg, .x2, off) else encode.fsw(act.freg, .x2, off)),
                .reload => try code.append(allocator, if (d64) encode.fld(act.freg, .x2, off) else encode.flw(act.freg, .x2, off)),
                .move => if (act.freg != act.from_freg) try code.append(allocator, if (d64) encode.fmv_d(act.freg, act.from_freg) else encode.fmv_s(act.freg, act.from_freg)),
            }
        },
        2 => {
            // RVV vector: a 16-byte <4 x f32> slot. vse32/vle32 need the slot address in a GPR, so
            // compute it into `spill_scratch1` (x8, reserved out of every pool). A reg->reg re-home is
            // a whole-register `vmv.v.v`. This is the class a vector live across a call spills through.
            const off: i12 = @intCast(vspill_base + act.slot * 16);
            switch (act.kind) {
                .store => {
                    try code.append(allocator, encode.addi(spill_scratch1, .x2, off));
                    try code.append(allocator, encode.vse32(act.vreg, spill_scratch1));
                },
                .reload => {
                    try code.append(allocator, encode.addi(spill_scratch1, .x2, off));
                    try code.append(allocator, encode.vle32(act.vreg, spill_scratch1));
                },
                .move => if (act.vreg != act.from_vreg) try code.append(allocator, encode.vmv_v_v(act.vreg, act.from_vreg)),
            }
        },
        3 => {
            // et-soc VPU vector: a 32-byte (8 x f32) slot addressed by `fsw.ps`/`flw.ps`'s own 12-bit
            // displacement off `sp` (no address register needed). A reg->reg re-home has no packed VPU
            // move instruction, so it round-trips through the reserved 32-byte `vpu_pack_base` scratch
            // slot: `fsw.ps` the source, then `flw.ps` into the destination (each move is atomic, so a
            // shared scratch slot is safe even inside a parallel-move cycle). This is the class a VPU
            // value live across a call spills through (every f16..f27 is caller-saved).
            const off: i12 = @intCast(vpu_vspill_base + act.slot * 32);
            switch (act.kind) {
                .store => try code.append(allocator, encode.fsw_ps(act.freg, .x2, off)),
                .reload => try code.append(allocator, encode.flw_ps(act.freg, .x2, off)),
                .move => if (act.freg != act.from_freg) {
                    try code.append(allocator, encode.fsw_ps(act.from_freg, .x2, @intCast(vpu_pack_base)));
                    try code.append(allocator, encode.flw_ps(act.freg, .x2, @intCast(vpu_pack_base)));
                },
            }
        },
        else => unreachable,
    }
}

/// Materialize a 32-bit value into integer register `rd`. When `bits` does not fit a 12-bit
/// signed immediate this emits `lui rd, hi` immediately followed by `addi rd, rd, lo`, adjacent
/// by construction (the same hi/lo address-pair shape `caps.fuse_addr_hi_lo` guards for
/// `.global_addr`), but this helper is a free function called from ~25 sites with no `ModelCaps`
/// in scope, so it carries no adjacency assert of its own: threading `caps` through every call
/// site just for an assert is not worth it, and the invariant holds unconditionally regardless.
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

/// Materialize a full 64-bit value into integer register `rd`. Small values fold to a single
/// `addi`, signed-32-bit ones to the `lui`+`addi` pair (via `loadImm32`, whose sign extension
/// fills the high 32 bits correctly). Anything wider is built MSB-first in 11-bit chunks: the top
/// 9 bits seed `rd`, then each remaining 11-bit chunk is shifted in with `slli 11; ori chunk`.
/// 9 + 5*11 = 64 exactly, and every `ori` immediate is a positive `u11` (bit 11 clear), so its
/// sign extension is all zeros and the OR only sets the freshly-shifted low 11 bits. Needed for the
/// et-soc tensor descriptors (`.matmul`), which are packed 64-bit CSR words with high fields set.
fn loadImm64(allocator: std.mem.Allocator, code: *std.ArrayList(u32), rd: Reg, value: u64) std.mem.Allocator.Error!void {
    const signed: i64 = @bitCast(value);
    if (signed >= -2048 and signed <= 2047) {
        try code.append(allocator, encode.addi(rd, .x0, @intCast(signed)));
        return;
    }
    if (signed >= std.math.minInt(i32) and signed <= std.math.maxInt(i32)) {
        try loadImm32(allocator, code, rd, @truncate(value));
        return;
    }
    // General case: emit the top 9 bits, then five 11-bit chunks from high to low.
    try code.append(allocator, encode.addi(rd, .x0, @intCast(value >> 55))); // bits [63:55], a positive u9
    const shifts = [_]u6{ 44, 33, 22, 11, 0 };
    for (shifts) |sh| {
        try code.append(allocator, encode.slli(rd, rd, 11));
        try code.append(allocator, encode.ori(rd, rd, @intCast((value >> sh) & 0x7ff)));
    }
}

/// Round `x` up to a multiple of `a` (a power of two).
fn alignUp(x: u32, a: u32) u32 {
    return (x + a - 1) & ~(a - 1);
}

/// True if `func` contains any `matmul` op (used to reserve the et-soc tensor staging scratch in
/// the frame). Cheap: matmul is rare and functions are small, so a full scan is fine.
fn functionHasMatmul(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .matmul) return true;
        }
    }
    return false;
}

/// True if `func` contains any `embedded` matmul, i.e. one that is NOT the whole reachable function
/// and so needs the self-contained save/restore lowering (which draws a stack save-area from the
/// frame). Reserving that area is gated on this so a function with only whole-function/standalone
/// matmuls (every matmul built today) keeps its frame, and therefore its emitted bytes, unchanged.
fn functionHasEmbeddedMatmul(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .matmul => |mmv| if (mmv.embedded) return true,
                else => {},
            }
        }
    }
    return false;
}

/// True when the function holds any value the embedded matmul's per-float 32-bit save cannot preserve
/// across it: an f64 (64-bit), or a vector (RVV/VPU, wider than 32 bits) that lives in a float register.
/// The embedded save/restore stores each clobbered float register as a 32-bit `fsw` (the et-soc fp32
/// case, the only float width sw-sysemu implements). A live f64 or vector crossing an embedded matmul
/// would lose its high bits, so reject cleanly rather than silently miscompile it. fp32-scalar
/// surroundings (the recognizer's target) are unaffected.
fn functionHasWideFloatValue(func: *const Function) bool {
    var i: usize = 0;
    while (i < func.valueCount()) : (i += 1) {
        const v: Value = @enumFromInt(@as(u32, @intCast(i)));
        switch (func.types.type_kind(func.valueType(v))) {
            .float => |f| if (f == .f64) return true,
            .vector => return true,
            else => {},
        }
    }
    return false;
}

/// Registers holding a/b/c for the duration of an EMBEDDED matmul lowering, so the base pointers are
/// stable regardless of where the allocator placed a/b/c (they may land in the op's scratch set in an
/// embedded context). x29/x30 are caller-saved temps the matmul body never otherwise uses; x9 is a
/// callee-saved temp used as the third holder. All three are saved on entry and restored on exit, so
/// any live-across value the allocator put in them survives.
const matmul_holder_a: Reg = .x29;
const matmul_holder_b: Reg = .x30;
const matmul_holder_c: Reg = .x9;

/// The embedded-matmul stack save-area layout. Ten int slots (8 bytes each): the 4 clobbered scratch
/// temps, the 3 holder registers' incoming values, and 3 a/b/c pointer transfer slots. Then up to 32
/// float slots (4 bytes each) for the worst-case TenC clobber f0..f31. See `matmul_save_base`. The
/// float saves are 32-bit `fsw`/`flw` (not 64-bit `fsd`/`fld`): the et-soc tensor unit holds fp32
/// accumulators and every et-soc scalar float is fp32, so the low 32 bits are the whole value, and
/// the sw-sysemu oracle implements the 32-bit scalar float load/store but not the 64-bit doubleword
/// forms. (A live f64 or 256-bit VPU vector across an embedded matmul would need a wider save, so
/// `functionHasWideFloatValue` rejects that combination up front rather than truncating it.)
const matmul_save_int_slots: u32 = 10;
const matmul_save_int_bytes: u32 = matmul_save_int_slots * 8; // 80
const matmul_save_float_bytes: u32 = 32 * 4; // 128, worst case f0..f31 saved as fp32 each

/// Materialize `base_reg + offset` (a compile-time byte offset) into `dst`, or return `base_reg`
/// unchanged (emitting nothing) when `offset` is zero. Used by the matmul lowering to form
/// sub-tile pointers from a runtime a/b/c pointer plus a compile-time tile offset.
fn emitPtrPlusOffset(allocator: std.mem.Allocator, code: *std.ArrayList(u32), dst: Reg, base_reg: Reg, offset: u64) std.mem.Allocator.Error!Reg {
    if (offset == 0) return base_reg;
    try loadImm64(allocator, code, dst, offset);
    try code.append(allocator, encode.add(dst, base_reg, dst));
    return dst;
}

/// Load a `num_rows` x `width` (element size `elem_bytes`, 1/2/4) sub-tile of a REAL row-major
/// matrix into consecutive L1 scratchpad lines `dst_scp .. dst_scp+num_rows-1`, then
/// `tensor_wait` on `id`. `tensor_fma` reads one ROW-MAJOR matrix row per SCP line (the low
/// `width` elements of each 64-byte line are the valid data), so every row must land on its own
/// line. This is the A-operand layout for every dtype, AND the B-operand layout for fp32 only
/// (fp16/int8 B needs the K-interleaved transpose-pack `emitMatmulLoadBPacked`, since the tensor
/// unit reads those B lines as `factor` consecutive-K elements per column, not one row per line).
///
/// The et-soc `tensor_load` can only address 64-byte-aligned lines with a 64-byte-granular stride:
/// hardware masks BOTH the descriptor address and the x31 stride with `~0x3f` (sw-sysemu
/// tensors.cpp `tensor_load_start`: `addr = control & 0xFFFFFFFFFFC0`, `stride = X31 &
/// 0xFFFFFFFFFFC0`). A real row-major sub-tile has rows `row_pitch` bytes apart (k*4 for A, n*4 for
/// B); that pitch is a multiple of 64 only when the matrix's inner dimension is a multiple of 16.
/// So:
///   - `row_pitch % 64 == 0`: load the rows directly with one strided `tensor_load`.
///   - otherwise: stage the rows into `stage_ptr` (a 64-byte-aligned scratch with 64-byte row
///     pitch) via scalar word copies, then one stride-64 `tensor_load` from the staging buffer.
/// The sub-tile base (`base_reg + tile_off`) is always 64-byte aligned (both `tile_off`, a
/// multiple of 64, and the caller's 64-aligned base), so the descriptor's addr mask drops nothing.
/// Emit a scalar load of `elem_bytes` (1/2/4) from `imm(rs1)` into `rd`. Widths select
/// `lb`/`lh`/`lw`; the matmul staging only ever copies (never interprets) the bytes, so the
/// sign of the load is irrelevant (the paired store writes back the same width). Programmer
/// error for any other width.
fn emitScalarLoad(allocator: std.mem.Allocator, code: *std.ArrayList(u32), elem_bytes: u32, rd: Reg, rs1: Reg, imm: i12) std.mem.Allocator.Error!void {
    try code.append(allocator, switch (elem_bytes) {
        1 => encode.lb(rd, rs1, imm),
        2 => encode.lh(rd, rs1, imm),
        4 => encode.lw(rd, rs1, imm),
        else => unreachable, // matmul dtypes are int8/fp16/fp32 -> 1/2/4 bytes only
    });
}

/// Emit a scalar store of `elem_bytes` (1/2/4) of `rs2` to `imm(rs1)`. Sibling of
/// `emitScalarLoad`; `sb`/`sh`/`sw` write only the low `elem_bytes` of `rs2`. Programmer error
/// for any other width.
fn emitScalarStore(allocator: std.mem.Allocator, code: *std.ArrayList(u32), elem_bytes: u32, rs2: Reg, rs1: Reg, imm: i12) std.mem.Allocator.Error!void {
    try code.append(allocator, switch (elem_bytes) {
        1 => encode.sb(rs2, rs1, imm),
        2 => encode.sh(rs2, rs1, imm),
        4 => encode.sw(rs2, rs1, imm),
        else => unreachable, // matmul dtypes are int8/fp16/fp32 -> 1/2/4 bytes only
    });
}

fn emitMatmulLoadSubtile(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    base_reg: Reg,
    tile_off: u64,
    row_pitch: u64,
    num_rows: u16,
    width: u16,
    elem_bytes: u32,
    dst_scp: u6,
    id: u1,
    stage_ptr: Reg,
    addr_scratch: Reg,
    copy_tmp: Reg,
    desc: Reg,
    stride_reg: Reg,
) std.mem.Allocator.Error!void {
    // Static descriptor bits: dst_start (SCP line) in [58:53], (num_rows-1) in [3:0], addr 0. The
    // 64-aligned base is OR'd in below (its low 6 bits are zero, so num_rows-1 never collides).
    const load_static = encode.packTensorLoad(dst_scp, @intCast(num_rows), 0, false, false, false);
    if (row_pitch % 64 == 0) {
        // Direct strided load: x31 = stride|id, descriptor = static | (base + tile_off).
        try loadImm64(allocator, code, stride_reg, encode.tensorLoadX31(row_pitch, id));
        const addr_reg = try emitPtrPlusOffset(allocator, code, addr_scratch, base_reg, tile_off);
        try loadImm64(allocator, code, desc, load_static);
        try code.append(allocator, encode.or_(desc, desc, addr_reg));
        try code.append(allocator, encode.csrw(encode.CSR_TENSOR_LOAD, desc));
    } else {
        // Stage each row's `width` elements (of `elem_bytes` each) into stage_ptr[i*64 ..] with
        // scalar copies (element granular, so no 64-byte source alignment is needed), then load
        // stride-64 from the stage. The staged line is row-major packed (element c at byte
        // c*elem_bytes), exactly how tensor_fma reads an A row from an SCP line (A[i][k] =
        // SCP[astart+i].{u8/f16/f32}[k], tensors.cpp fma execute functions). Offsets stay tiny:
        // i*64 (i<16) + c*elem_bytes (c*4 <= 60 worst case) <= 1020, all fit the imm12.
        var i: u16 = 0;
        while (i < num_rows) : (i += 1) {
            const row_off = tile_off + @as(u64, i) * row_pitch;
            const src = try emitPtrPlusOffset(allocator, code, addr_scratch, base_reg, row_off);
            var c: u16 = 0;
            while (c < width) : (c += 1) {
                try emitScalarLoad(allocator, code, elem_bytes, copy_tmp, src, @intCast(@as(u32, c) * elem_bytes));
                try emitScalarStore(allocator, code, elem_bytes, copy_tmp, stage_ptr, @intCast(@as(u32, i) * 64 + @as(u32, c) * elem_bytes));
            }
        }
        try loadImm64(allocator, code, stride_reg, encode.tensorLoadX31(64, id));
        try loadImm64(allocator, code, desc, load_static);
        try code.append(allocator, encode.or_(desc, desc, stage_ptr));
        try code.append(allocator, encode.csrw(encode.CSR_TENSOR_LOAD, desc));
    }
    // tensor_wait on this load's event id (0 or 1). The load already executed synchronously under
    // sw-sysemu (non-coop), but the wait is required on real hardware before the fma reads the SCP.
    try loadImm32(allocator, code, desc, id);
    try code.append(allocator, encode.csrw(encode.CSR_TENSOR_WAIT, desc));
}

/// Load and K-interleave-transpose-pack a `kslice` x `cols` sub-tile of a REAL row-major
/// fp16/int8 B matrix into `kslice/factor` L1 scratchpad lines starting at `dst_scp`, then
/// `tensor_wait` on `id`. Only for `factor > 1` (fp16 factor=2, int8 factor=4); fp32 B uses the
/// plain row-per-line `emitMatmulLoadSubtile`.
///
/// WHY the transpose-pack: for the multi-element-per-K dtypes the tensor unit reads B NOT as one
/// row per SCP line, but as `factor` consecutive-K elements packed into a fixed 4-byte slot per
/// output column. From sw-sysemu `tensor_fma16a32_execute` (tensors.cpp:1322) and
/// `tensor_ima8a32_execute` (:1424): for contraction index `k`, B lives in line `bstart + k/factor`
/// and element B[k+x][j] is read from that line as `f16[2*j + x]` (fp16) / `u8[j*4 + x]` (int8),
/// i.e. byte `j*4 + x*elem_bytes` (x in 0..factor). So SCP line p holds, for each column j, the
/// `factor` elements B[factor*p + 0..factor-1][j] in a 4-byte group (`factor * elem_bytes == 4`).
/// The source B is plain row-major (`B[kk][j]` at byte `kk*n_pitch + j*elem_bytes`), so this is a
/// (factor x cols) -> (cols x factor) transpose done with scalar copies. `kslice` is a multiple of
/// `factor` (the caller rejects `k % factor != 0`), so every line is fully populated.
fn emitMatmulLoadBPacked(
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    base_reg: Reg,
    tile_off: u64,
    n_pitch: u64,
    elem_bytes: u32,
    factor: u32,
    kslice: u16,
    cols: u16,
    dst_scp: u6,
    id: u1,
    stage_ptr: Reg,
    addr_scratch: Reg,
    copy_tmp: Reg,
    desc: Reg,
    stride_reg: Reg,
) std.mem.Allocator.Error!void {
    std.debug.assert(@as(u32, factor) * elem_bytes == 4); // fp16: 2*2, int8: 4*1 (one 4-byte column slot)
    std.debug.assert(kslice % factor == 0); // caller guarantees k (and thus every slice) is factor-aligned
    const lines: u16 = @intCast(kslice / factor);
    // Stage into `stage_ptr`: line p (SCP line dst_scp+p), column j, sub-K x -> byte
    // p*64 + j*4 + x*elem_bytes, sourced from B[factor*p + x][j]. Offsets: p*64 (p<16) + j*4 (<=60)
    // + x*elem_bytes (<=3) <= 1023, and the source column offset j*elem_bytes (<=30) both fit imm12.
    var p: u16 = 0;
    while (p < lines) : (p += 1) {
        var x: u32 = 0;
        while (x < factor) : (x += 1) {
            const row_k = @as(u64, p) * factor + x; // K index of this staged sub-row
            const src = try emitPtrPlusOffset(allocator, code, addr_scratch, base_reg, tile_off + row_k * n_pitch);
            var j: u16 = 0;
            while (j < cols) : (j += 1) {
                try emitScalarLoad(allocator, code, elem_bytes, copy_tmp, src, @intCast(@as(u32, j) * elem_bytes));
                try emitScalarStore(allocator, code, elem_bytes, copy_tmp, stage_ptr, @intCast(@as(u32, p) * 64 + @as(u32, j) * 4 + x * elem_bytes));
            }
        }
    }
    // One stride-64 tensor_load of `lines` lines from the staging buffer into SCP dst_scp..
    try loadImm64(allocator, code, stride_reg, encode.tensorLoadX31(64, id));
    const load_static = encode.packTensorLoad(dst_scp, @intCast(lines), 0, false, false, false);
    try loadImm64(allocator, code, desc, load_static);
    try code.append(allocator, encode.or_(desc, desc, stage_ptr));
    try code.append(allocator, encode.csrw(encode.CSR_TENSOR_LOAD, desc));
    // tensor_wait (required on hardware before the fma reads the SCP; a no-op under sw-sysemu).
    try loadImm32(allocator, code, desc, id);
    try code.append(allocator, encode.csrw(encode.CSR_TENSOR_WAIT, desc));
}

/// Load the 64-byte-aligned staging line at `stage_ptr` (already populated by the caller with this
/// line's 16 or `cols` fp32/int32 words) into SCP line `scp_line`, then tensor_wait it. Shared by
/// every matmul-quant SCP-line load (bias, scale, zero-point): each is a distinct 4-byte-per-column
/// vector that lands on its own consecutive SCP line, staged and loaded the same way. Mirrors the
/// direct-load sequence the non-quant A/B path uses (`emitMatmulLoadSubtile`), just without the
/// per-row stage loop since the caller already wrote the whole line.
fn emitQuantScpLineLoad(allocator: std.mem.Allocator, code: *std.ArrayList(u32), scp_line: u6, stage_ptr: Reg, desc: Reg, stride_reg: Reg) std.mem.Allocator.Error!void {
    const static = encode.packTensorLoad(scp_line, 1, 0, false, false, false);
    try loadImm64(allocator, code, stride_reg, encode.tensorLoadX31(64, 0));
    try loadImm64(allocator, code, desc, static);
    try code.append(allocator, encode.or_(desc, desc, stage_ptr));
    try code.append(allocator, encode.csrw(encode.CSR_TENSOR_LOAD, desc));
    try loadImm32(allocator, code, desc, @intCast(encode.TENSOR_WAIT_LOAD_0));
    try code.append(allocator, encode.csrw(encode.CSR_TENSOR_WAIT, desc));
}

/// Size in bytes of a stack slot for `ty`. Aggregates are not yet supported.
fn typeSize(func: *const Function, ty: ir.types.Type) Error!u32 {
    return switch (func.types.type_kind(ty)) {
        .bool => 1,
        .int => |i| (@as(u32, i.bits) + 7) / 8,
        .float => |f| switch (f) {
            .f32 => 4,
            .f64 => 8,
            // Size only, not lowering: riscv64 has no f16 codegen yet.
            .f16 => 2,
        },
        .ptr => 8,
        .vector => |v| v.len * try typeSize(func, v.elem),
        else => error.Unsupported,
    };
}

test "typeSize reports 2 bytes for f16, unlike 4 for f32 and 8 for f64" {
    // f16 has no riscv64 codegen yet (a later task), but the size switch itself
    // must already know f16 is a 2-byte scalar so it stays exhaustive.
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const f64_t = try func.types.intern(.{ .float = .f64 });

    try std.testing.expectEqual(@as(u32, 2), try typeSize(&func, f16_t));
    try std.testing.expectEqual(@as(u32, 4), try typeSize(&func, f32_t));
    try std.testing.expectEqual(@as(u32, 8), try typeSize(&func, f64_t));
}

/// A register-to-register move within one register class.
fn Move(comptime R: type) type {
    return struct { src: R, dst: R };
}

/// A register-to-register integer move, used for shuffling call arguments and integer block-edge
/// arguments into place.
const RegMove = Move(Reg);

/// Emit `moves_in` (dst<-src copies within ONE register class) as a *parallel* copy: every dst ends
/// holding its src's ORIGINAL value. Non-conflicting moves (a dst that is no other move's source) go
/// first; the remainder form permutation cycles (e.g. swapping two registers), broken by staging one
/// value through `scratch`. `emit(allocator, code, dst, src)` appends the class's move instruction.
///
/// `scratch` must be a register reserved out of the allocatable pool for that class, so it is never
/// itself a move source or destination. A single scratch suffices even with several disjoint cycles:
/// after a cycle is broken, its redirected (scratch-reading) move is emittable in the very next inner
/// pass, so the scratch is always consumed before a later cycle-break reuses it.
fn parallelMove(
    comptime R: type,
    allocator: std.mem.Allocator,
    code: *std.ArrayList(u32),
    moves_in: []const Move(R),
    scratch: R,
    comptime emitMove: fn (std.mem.Allocator, *std.ArrayList(u32), R, R) std.mem.Allocator.Error!void,
) std.mem.Allocator.Error!void {
    var moves: std.ArrayList(Move(R)) = .empty;
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
                try emitMove(allocator, code, m.dst, m.src); // mv dst, src
                _ = moves.swapRemove(i);
                emitted = true;
            }
        }
        if (!emitted) {
            // Remaining moves form one or more cycles. Break by saving a
            // destination into the scratch and redirecting reads of it.
            const m = moves.items[0];
            try emitMove(allocator, code, scratch, m.dst); // save dst
            for (moves.items) |*o| {
                if (o.src == m.dst) o.src = scratch;
            }
            try emitMove(allocator, code, m.dst, m.src); // mv dst, src
            _ = moves.orderedRemove(0);
        }
    }
}

/// Append `mv dst, src` (an `addi dst, src, 0`).
fn emitIntMove(allocator: std.mem.Allocator, code: *std.ArrayList(u32), dst: Reg, src: Reg) std.mem.Allocator.Error!void {
    try code.append(allocator, encode.addi(dst, src, 0));
}

/// Append `fmv.d dst, src`. A full 64-bit register copy: it relocates the exact bits (an f32 value's
/// NaN-boxed low half included), so it correctly moves f32 and f64 values alike across an edge.
fn emitFloatMove(allocator: std.mem.Allocator, code: *std.ArrayList(u32), dst: FReg, src: FReg) std.mem.Allocator.Error!void {
    try code.append(allocator, encode.fmv_d(dst, src));
}

/// Append `vmv.v.v dst, src` (whole-vector register copy).
fn emitVectorMove(allocator: std.mem.Allocator, code: *std.ArrayList(u32), dst: VReg, src: VReg) std.mem.Allocator.Error!void {
    try code.append(allocator, encode.vmv_v_v(dst, src));
}

/// Emit integer register moves as a parallel copy, breaking cycles with `scratch`. Needed so e.g.
/// swapping two argument registers is correct.
fn parallelMoveInt(allocator: std.mem.Allocator, code: *std.ArrayList(u32), moves_in: []const RegMove, scratch: Reg) std.mem.Allocator.Error!void {
    try parallelMove(Reg, allocator, code, moves_in, scratch, emitIntMove);
}

/// Emit float register moves as a parallel copy, breaking cycles with `scratch` (a reserved float
/// register, `float_scratch`). Needed so a block edge that permutes its float loop-carried values
/// (e.g. a swap) is correct.
fn parallelMoveFloat(allocator: std.mem.Allocator, code: *std.ArrayList(u32), moves_in: []const Move(FReg), scratch: FReg) std.mem.Allocator.Error!void {
    try parallelMove(FReg, allocator, code, moves_in, scratch, emitFloatMove);
}

/// Emit vector register moves as a parallel copy, breaking cycles with `scratch` (a reserved vector
/// register, `vector_scratch`). Needed so a block edge that permutes its vector loop-carried values
/// is correct.
fn parallelMoveVector(allocator: std.mem.Allocator, code: *std.ArrayList(u32), moves_in: []const Move(VReg), scratch: VReg) std.mem.Allocator.Error!void {
    try parallelMove(VReg, allocator, code, moves_in, scratch, emitVectorMove);
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

/// Whether `ty` is the half-precision float `f16`. riscv64 has no hardware f16 (no Zfh), so an f16
/// SSA value lives in a float register as its f32 WIDENING (a value exactly representable in half),
/// mirroring the aarch64 emulation model. `is64Float(f16)` is false, so every in-register op picks
/// the single-precision (f32) form naturally; the emulation work is only at the boundaries (load,
/// store, arithmetic rounding, and cross-type convert), which round via the software routines below.
fn isHalf(func: *const Function, ty: ir.types.Type) bool {
    return switch (func.types.type_kind(ty)) {
        .float => |f| f == .f16,
        else => false,
    };
}

/// Software EXTEND f16 -> f32 (exact, no rounding). Reads the 16-bit half pattern in `in` (zero-
/// extended, e.g. straight from an `lhu`) and writes the f32 bit pattern of the same value into
/// `out`. Branchless via the Fabian Giesen magic-multiply half->float algorithm: shift the
/// exponent+mantissa into an f32 whose exponent is biased low, multiply by a pure power of two
/// (2^112, exact regardless of rounding mode) to rebias, then patch the inf/NaN exponent and OR in
/// the sign. Every finite AND subnormal half comes out exact because the multiply renormalizes
/// subnormals for free. `in` is left untouched; `out`, `f16_scratch_a`, `f16_scratch_b`, and the
/// two float scratch registers `f0`/`f1` are clobbered. `out` must differ from `in`.
fn emitHalfToFloat(allocator: std.mem.Allocator, code: *std.ArrayList(u32), out: Reg, in: Reg, f0: FReg, f1: FReg) std.mem.Allocator.Error!void {
    const s0 = f16_scratch_a;
    const s1 = f16_scratch_b;
    // out = (h & 0x7fff) << 13: place the 15 exponent+mantissa bits at f32 bit 13. slli 49 drops
    // the sign bit (bit 15) off the top of the 64-bit register, srli 36 brings the rest back down.
    try code.append(allocator, encode.slli(out, in, 49));
    try code.append(allocator, encode.srli(out, out, 36));
    try code.append(allocator, encode.fmv_w_x(f0, out)); // f0 = o.f (exponent biased low)
    // Multiply by the magic 2^112 (0x77800000) to rebias the exponent into f32 range. A pure power
    // of two, so the product is exact and independent of the rounding mode; it also renormalizes a
    // subnormal half into a normal f32.
    try loadImm32(allocator, code, s0, 0x77800000);
    try code.append(allocator, encode.fmv_w_x(f1, s0));
    try code.append(allocator, encode.fmul_s(f0, f0, f1));
    // inf/NaN fix: a half with exponent 31 lands at exactly (or above) 2^16 = 0x47800000 after the
    // multiply, so set the f32 all-ones exponent (0x7f800000) whenever o.f >= that boundary. The
    // shifted value is never itself a NaN/inf f32 (its exponent maxes at 31), so `fle.s` is exact.
    try loadImm32(allocator, code, s0, 0x47800000);
    try code.append(allocator, encode.fmv_w_x(f1, s0));
    try code.append(allocator, encode.fle_s(s0, f1, f0)); // s0 = (0x47800000 <= o.f) ? 1 : 0
    try code.append(allocator, encode.fmv_x_w(out, f0)); // o.u = bits(o.f)
    try code.append(allocator, encode.sub(s0, .x0, s0)); // s0 = 0 or ~0 (mask)
    try loadImm32(allocator, code, s1, 0x7F800000); // f32 all-ones exponent
    try code.append(allocator, encode.and_(s0, s0, s1));
    try code.append(allocator, encode.or_(out, out, s0));
    // sign: bit 15 of the half -> bit 31 of the f32.
    try code.append(allocator, encode.srli(s1, in, 15));
    try code.append(allocator, encode.slli(s1, s1, 31));
    try code.append(allocator, encode.or_(out, out, s1));
}

/// Software TRUNCATE f32 -> f16 with round-to-nearest-EVEN. Reads the f32 bit pattern in `in` and
/// writes the 16-bit half pattern into the LOW 16 bits of `out`. NOTE: the upper bits of `out` are
/// NOT guaranteed clear (a negative input, sign-extended by the caller's `fmv.x.w`, leaves bits
/// 16..47 set after the final sign OR); the only consumers are `sh` (takes the low 16) and
/// `emitHalfToFloat` (re-masks bit 15 down), so this is fine, but a consumer that reads the whole
/// register (e.g. `sw` or a full-width compare) must mask to 16 bits first. Branchless: it
/// computes the normal, subnormal, and inf/NaN candidate results and blends them with masks derived
/// from the input's exponent range, mirroring Fabian Giesen's `float_to_half_fast3_rtne` but with
/// the three branches turned into masked selects so no basic block is split mid-emit. Handles RNE
/// ties in both directions (the mant-odd bias), overflow to inf, gradual underflow into f16
/// subnormals or signed zero, and NaN (mapped to a quiet NaN with nonzero mantissa). `in` is left
/// untouched; `out` and the four `f16_scratch_*` registers plus float scratch `f0`/`f1` are
/// clobbered. `in` and `out` must differ from each of the four scratch GPRs.
fn emitFloatToHalf(allocator: std.mem.Allocator, code: *std.ArrayList(u32), out: Reg, in: Reg, f0: FReg, f1: FReg) std.mem.Allocator.Error!void {
    const abs = f16_scratch_a; // |f| bits (kept live for the whole routine)
    const s1 = f16_scratch_b;
    const s2 = f16_scratch_c;
    const s3 = f16_scratch_d;
    // abs = in & 0x7fffffff (strip the sign; slli 33 / srli 33 keeps the low 31 bits).
    try code.append(allocator, encode.slli(abs, in, 33));
    try code.append(allocator, encode.srli(abs, abs, 33));

    // --- NORMAL candidate into `out` ---
    // mant_odd = (abs >> 13) & 1: the low bit of the retained 10-bit mantissa, for the RNE bias.
    try code.append(allocator, encode.srli(s1, abs, 13));
    try code.append(allocator, encode.andi(s1, s1, 1));
    // out = abs + ((15 - 127) << 23) + 0xfff  (exponent rebias + rounding bias part 1), then + mant_odd.
    try loadImm32(allocator, code, s2, 0xC8000FFF);
    try code.append(allocator, encode.add(out, abs, s2));
    try code.append(allocator, encode.add(out, out, s1));
    // o_normal = (low 32 bits of the sum) >> 13. slli 32 clears the sign-extension carried in from
    // the negative addend above, srli 45 (= 32 + 13) both re-aligns and applies the >> 13.
    try code.append(allocator, encode.slli(out, out, 32));
    try code.append(allocator, encode.srli(out, out, 45));

    // --- SUBNORMAL candidate, selected when abs < (113 << 23) ---
    // o_sub = bits(abs_as_f32 + 0.5) - 0x3f000000. Adding the magic 0.5 aligns the 10 mantissa bits
    // at the bottom of the float under RNE; the integer subtract of the bias yields the half.
    try code.append(allocator, encode.fmv_w_x(f0, abs));
    try loadImm32(allocator, code, s1, 0x3F000000); // 0.5f
    try code.append(allocator, encode.fmv_w_x(f1, s1));
    try code.append(allocator, encode.fadd_s(f0, f0, f1));
    try code.append(allocator, encode.fmv_x_w(s2, f0));
    try code.append(allocator, encode.sub(s2, s2, s1)); // s2 = o_sub
    try loadImm32(allocator, code, s1, 0x38800000); // 113 << 23
    try code.append(allocator, encode.sltu(s1, abs, s1)); // s1 = (abs < 0x38800000) ? 1 : 0
    // out = flag_sub ? o_sub : o_normal, via xor-select (mask = -flag).
    try code.append(allocator, encode.sub(s1, .x0, s1));
    try code.append(allocator, encode.xor_(s3, out, s2));
    try code.append(allocator, encode.and_(s3, s3, s1));
    try code.append(allocator, encode.xor_(out, out, s3));

    // --- INF/NaN candidate, selected when abs >= (143 << 23) = f16max ---
    // o_inf = 0x7c00 | (abs > 0x7f800000 ? 0x200 : 0): Inf stays Inf, any NaN becomes a quiet NaN.
    try loadImm32(allocator, code, s1, 0x7F800000);
    try code.append(allocator, encode.sltu(s1, s1, abs)); // s1 = (0x7f800000 < abs) ? 1 : 0  (NaN)
    try code.append(allocator, encode.slli(s2, s1, 9)); // 0x200 or 0
    try code.append(allocator, encode.addi(s1, .x0, 0x1F));
    try code.append(allocator, encode.slli(s1, s1, 10)); // 0x7c00
    try code.append(allocator, encode.or_(s2, s2, s1)); // s2 = o_inf
    try loadImm32(allocator, code, s1, 0x47800000); // f16max = 143 << 23
    try code.append(allocator, encode.sltu(s1, abs, s1)); // s1 = (abs < f16max) ? 1 : 0
    try code.append(allocator, encode.xori(s1, s1, 1)); // flag_inf = !(abs < f16max)
    try code.append(allocator, encode.sub(s1, .x0, s1)); // mask
    try code.append(allocator, encode.xor_(s3, out, s2));
    try code.append(allocator, encode.and_(s3, s3, s1));
    try code.append(allocator, encode.xor_(out, out, s3));

    // Mask to 16 bits, then OR in the sign (bit 31 of the input -> bit 15 of the half).
    try code.append(allocator, encode.slli(out, out, 48));
    try code.append(allocator, encode.srli(out, out, 48));
    try code.append(allocator, encode.srli(s1, in, 31));
    try code.append(allocator, encode.slli(s1, s1, 15));
    try code.append(allocator, encode.or_(out, out, s1));
}

/// Round the f32-widening f16 value held in float register `fr` to nearest-even half and re-widen
/// it back into `fr`, preserving the held-as-f32 invariant. This is the per-op rounding an f16
/// arithmetic result (or an f32/f64 -> f16 convert) needs: truncate to half then extend back, both
/// in software. Uses the always-reserved `scratch_reg` (x6) and `spill_scratch1` (x8) to shuttle
/// the bit pattern between the float register and the convert routines, whose own scratch is the
/// four `f16_scratch_*` GPRs and the two float spill scratches `fspill0`/`fspill1`.
fn emitRoundToHalf(allocator: std.mem.Allocator, code: *std.ArrayList(u32), fr: FReg, fspill0: FReg, fspill1: FReg) std.mem.Allocator.Error!void {
    try code.append(allocator, encode.fmv_x_w(scratch_reg, fr)); // f32 bits -> x6
    try emitFloatToHalf(allocator, code, spill_scratch1, scratch_reg, fspill0, fspill1); // half -> x8
    try emitHalfToFloat(allocator, code, scratch_reg, spill_scratch1, fspill0, fspill1); // f32 -> x6
    try code.append(allocator, encode.fmv_w_x(fr, scratch_reg)); // back into the float register
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
        .mulh => encode.mulh(rd, rs1, rs2), // signed high multiply; unsigned takes mulhu in the caller
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
fn recordUses(allocator: std.mem.Allocator, func: *const Function, inst: ir.function.Inst, pos: usize, last_use: *std.AutoHashMapUnmanaged(Value, usize), fold: *const addrfold.Analysis) std.mem.Allocator.Error!void {
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
        // A folded load/store addresses `off(base)`, so its pointer USE is attributed to the fold
        // base (the add's lhs), not the raw `ptr` operand (the now-dead add's result). `baseOf` is
        // the raw ptr when nothing folds, so the non-folding case is byte-identical. This reroute
        // is what keeps `base` live all the way to the mem op, cross-block included.
        .load => try mark(allocator, last_use, fold.baseOf(func, inst), pos),
        .store => |st| {
            try mark(allocator, last_use, st.value, pos);
            try mark(allocator, last_use, fold.baseOf(func, inst), pos);
        },
        .prefetch => |pf| try mark(allocator, last_use, pf.ptr, pos),
        .dot => |d| {
            try mark(allocator, last_use, d.acc, pos);
            try mark(allocator, last_use, d.a, pos);
            try mark(allocator, last_use, d.b, pos);
        },
        .matmul => |mmv| {
            try mark(allocator, last_use, mmv.a, pos);
            try mark(allocator, last_use, mmv.b, pos);
            try mark(allocator, last_use, mmv.c, pos);
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

/// Append `pos` to each operand's ascending use-position list. The exact mirror of `recordUses`, but
/// recording every use position (not just the last one) so the eviction path can ask, at any point,
/// where a value is next used. Because the numbering loop calls this with strictly increasing `pos`,
/// each list stays ascending, so `nextUseAfter` can binary-search it.
fn recordUsePositions(allocator: std.mem.Allocator, func: *const Function, inst: ir.function.Inst, pos: usize, use_lists: []std.ArrayList(usize), fold: *const addrfold.Analysis) std.mem.Allocator.Error!void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .alloca, .global_addr => {},
        .arith => |a| {
            try appendUse(allocator, use_lists, a.lhs, pos);
            try appendUse(allocator, use_lists, a.rhs, pos);
        },
        .arith_imm => |a| try appendUse(allocator, use_lists, a.lhs, pos),
        .icmp => |c| {
            try appendUse(allocator, use_lists, c.lhs, pos);
            try appendUse(allocator, use_lists, c.rhs, pos);
        },
        .select => |s| {
            try appendUse(allocator, use_lists, s.cond, pos);
            try appendUse(allocator, use_lists, s.then, pos);
            try appendUse(allocator, use_lists, s.@"else", pos);
        },
        // Same fold reroute as `recordUses`: the pointer use is attributed to the fold base so the
        // eviction path sees the base's next use at the mem op (not the dead add's).
        .load => try appendUse(allocator, use_lists, fold.baseOf(func, inst), pos),
        .store => |st| {
            try appendUse(allocator, use_lists, st.value, pos);
            try appendUse(allocator, use_lists, fold.baseOf(func, inst), pos);
        },
        .prefetch => |pf| try appendUse(allocator, use_lists, pf.ptr, pos),
        .dot => |d| {
            try appendUse(allocator, use_lists, d.acc, pos);
            try appendUse(allocator, use_lists, d.a, pos);
            try appendUse(allocator, use_lists, d.b, pos);
        },
        .matmul => |mmv| {
            try appendUse(allocator, use_lists, mmv.a, pos);
            try appendUse(allocator, use_lists, mmv.b, pos);
            try appendUse(allocator, use_lists, mmv.c, pos);
        },
        .struct_new => |sn| for (func.valueList(sn.fields)) |v| try appendUse(allocator, use_lists, v, pos),
        .call => |c| for (func.valueList(c.args)) |v| try appendUse(allocator, use_lists, v, pos),
        .call_indirect => |c| {
            try appendUse(allocator, use_lists, c.target, pos);
            for (func.valueList(c.args)) |v| try appendUse(allocator, use_lists, v, pos);
        },
        .extract => |ex| try appendUse(allocator, use_lists, ex.aggregate, pos),
        .convert => |cv| try appendUse(allocator, use_lists, cv.value, pos),
        .unary => |u| try appendUse(allocator, use_lists, u.value, pos),
        .@"if" => |cf| {
            try appendUse(allocator, use_lists, cf.cond, pos);
            for (func.blockArgs(cf.then)) |v| try appendUse(allocator, use_lists, v, pos);
            for (func.blockArgs(cf.@"else")) |v| try appendUse(allocator, use_lists, v, pos);
        },
    }
}

/// The terminator analogue of `recordUsePositions`, mirroring `recordTermUses`.
fn recordUsePositionsTerm(allocator: std.mem.Allocator, func: *const Function, block: Block, pos: usize, use_lists: []std.ArrayList(usize)) std.mem.Allocator.Error!void {
    if (func.terminator(block)) |term| switch (term) {
        .ret => |v| if (v) |vv| try appendUse(allocator, use_lists, vv, pos),
        .jump => |j| for (func.blockArgs(j)) |v| try appendUse(allocator, use_lists, v, pos),
    };
}

fn appendUse(allocator: std.mem.Allocator, use_lists: []std.ArrayList(usize), v: Value, pos: usize) std.mem.Allocator.Error!void {
    try use_lists[@intFromEnum(v)].append(allocator, pos);
}

/// Clear `is_intra[v]` for a value used at block `bi`: a plain use in a block other than `v`'s def
/// block, or (regardless of block) an edge argument, disqualifies `v` from intra-block splitting.
fn markIntraUse(def_block: []const usize, is_intra: []bool, v: Value, bi: usize, is_edge_arg: bool) void {
    const vi = @intFromEnum(v);
    if (is_edge_arg or def_block[vi] != bi) is_intra[vi] = false;
}

/// The `is_intra` analogue of `recordUses`: walk an instruction's operands and disqualify any value
/// used outside its def block or passed as an edge argument (an `if`'s successor block arguments).
fn markIntraUses(func: *const Function, inst: ir.function.Inst, bi: usize, def_block: []const usize, is_intra: []bool, fold: *const addrfold.Analysis) void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .alloca, .global_addr => {},
        .arith => |a| {
            markIntraUse(def_block, is_intra, a.lhs, bi, false);
            markIntraUse(def_block, is_intra, a.rhs, bi, false);
        },
        .arith_imm => |a| markIntraUse(def_block, is_intra, a.lhs, bi, false),
        .icmp => |c| {
            markIntraUse(def_block, is_intra, c.lhs, bi, false);
            markIntraUse(def_block, is_intra, c.rhs, bi, false);
        },
        .select => |s| {
            markIntraUse(def_block, is_intra, s.cond, bi, false);
            markIntraUse(def_block, is_intra, s.then, bi, false);
            markIntraUse(def_block, is_intra, s.@"else", bi, false);
        },
        // Same fold reroute: attribute the pointer use to the fold base. A base used cross-block by a
        // folded mem op is thereby correctly disqualified from intra-block splitting.
        .load => markIntraUse(def_block, is_intra, fold.baseOf(func, inst), bi, false),
        .store => |st| {
            markIntraUse(def_block, is_intra, st.value, bi, false);
            markIntraUse(def_block, is_intra, fold.baseOf(func, inst), bi, false);
        },
        .prefetch => |pf| markIntraUse(def_block, is_intra, pf.ptr, bi, false),
        .dot => |d| {
            markIntraUse(def_block, is_intra, d.acc, bi, false);
            markIntraUse(def_block, is_intra, d.a, bi, false);
            markIntraUse(def_block, is_intra, d.b, bi, false);
        },
        .matmul => |mmv| {
            markIntraUse(def_block, is_intra, mmv.a, bi, false);
            markIntraUse(def_block, is_intra, mmv.b, bi, false);
            markIntraUse(def_block, is_intra, mmv.c, bi, false);
        },
        .struct_new => |sn| for (func.valueList(sn.fields)) |v| markIntraUse(def_block, is_intra, v, bi, false),
        .call => |c| for (func.valueList(c.args)) |v| markIntraUse(def_block, is_intra, v, bi, false),
        .call_indirect => |c| {
            markIntraUse(def_block, is_intra, c.target, bi, false);
            for (func.valueList(c.args)) |v| markIntraUse(def_block, is_intra, v, bi, false);
        },
        .extract => |ex| markIntraUse(def_block, is_intra, ex.aggregate, bi, false),
        .convert => |cv| markIntraUse(def_block, is_intra, cv.value, bi, false),
        .unary => |u| markIntraUse(def_block, is_intra, u.value, bi, false),
        .@"if" => |cf| {
            markIntraUse(def_block, is_intra, cf.cond, bi, false);
            for (func.blockArgs(cf.then)) |v| markIntraUse(def_block, is_intra, v, bi, true);
            for (func.blockArgs(cf.@"else")) |v| markIntraUse(def_block, is_intra, v, bi, true);
        },
    }
}

/// The `is_intra` analogue of `recordTermUses`. A `jump`'s block arguments are edge arguments (the
/// value flows across the edge into a successor block param), so they disqualify intra splitting.
fn markIntraTermUses(func: *const Function, block: Block, bi: usize, def_block: []const usize, is_intra: []bool) void {
    if (func.terminator(block)) |term| switch (term) {
        .ret => |v| if (v) |vv| markIntraUse(def_block, is_intra, vv, bi, false),
        .jump => |j| for (func.blockArgs(j)) |v| markIntraUse(def_block, is_intra, v, bi, true),
    };
}

/// Return the first element of ascending `uses` strictly greater than `p`, else null. The next
/// position at which a value is used after `p`. Mirrors the aarch64 helper but over `usize`.
fn nextUseAfter(uses: []const usize, p: usize) ?usize {
    var lo: usize = 0;
    var hi: usize = uses.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (uses[mid] <= p) lo = mid + 1 else hi = mid;
    }
    return if (lo < uses.len) uses[lo] else null;
}

/// Append `seg` to `value`'s segment list (growing the owned slice). Segments must be appended in
/// ascending `from` order by the caller, so a re-spill after a re-home lands at or after the re-home
/// position (asserted). Creates a single-element list if the value has none yet. Used for both the
/// second-chance re-home (`.reg`) and the append-aware re-spill (`.slot`).
fn appendSegment(allocator: std.mem.Allocator, alloc: *Allocation, value: Value, seg: Segment) Error!void {
    const gop = try alloc.segments.getOrPut(allocator, value);
    if (!gop.found_existing) {
        const s = try allocator.alloc(Segment, 1);
        s[0] = seg;
        gop.value_ptr.* = s;
        return;
    }
    const old = gop.value_ptr.*;
    // Ascending-`from` is the representation invariant `intLocationAt` relies on (it scans until the
    // first segment past `pos`). A caller that appends out of order is a programmer error and would
    // silently miscompile, so assert it rather than trust it. This assert is the guard that the
    // append-aware/cancelReHome branch in `placeIntUnderPressure` chose the right sub-case.
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
/// re-home position, and leaving the segment would break the ascending-`from` order `intLocationAt`
/// relies on, so both must go. The caller removes `value` from `alloc.int` (the riscv64 register
/// accounting), exactly as the initial split does.
fn cancelReHome(allocator: std.mem.Allocator, alloc: *Allocation, value: Value) Error!void {
    const old = alloc.segments.get(value).?;
    std.debug.assert(old.len >= 2 and old[old.len - 1].loc == .reg);
    const rehome = old[old.len - 1];
    const rehome_reg = rehome.loc.reg;
    // Remove the pending reload action for this re-home. The (value, position, register) triple is
    // unique (each re-home pops a distinct register), so exactly one action matches. Actions are not
    // sorted until emission, so a `swapRemove` is safe here.
    var found = false;
    var i: usize = 0;
    while (i < alloc.actions.items.len) : (i += 1) {
        const act = alloc.actions.items[i];
        if (act.kind == .reload and act.value == value and act.at == rehome.from and act.reg == rehome_reg) {
            _ = alloc.actions.swapRemove(i);
            found = true;
            break;
        }
    }
    std.debug.assert(found);
    // Shrink the owned segment slice by one (drop the trailing `.reg`). `getPtr` update never
    // allocates (the key already exists), so this cannot fail after the new slice is built.
    const shrunk = try allocator.alloc(Segment, old.len - 1);
    @memcpy(shrunk, old[0 .. old.len - 1]);
    allocator.free(old);
    alloc.segments.getPtr(value).?.* = shrunk;
}

/// Whether a value defined at `d` and last used at `l` is live ACROSS a call (a call position lands
/// strictly between them), from the prefix-sum `call_prefix`. Hoisted to file scope so `secondChance`
/// can decide whether a re-homed tail needs a callee-saved register.
fn crossesCall(call_prefix: []const usize, d: usize, l: usize) bool {
    if (l <= d + 1) return false;
    return call_prefix[l] - call_prefix[d + 1] > 0;
}

const IntVictim = struct { value: Value, next: usize, reg: Reg };

/// The active integer value (in `alloc.int`, still live after `pos`) whose next use after `pos` is
/// furthest away. When `need_saved` is true (the current value crosses a call and needs a
/// callee-saved register), only values holding a callee-saved register are eligible. Returns null if
/// there is no eligible active value.
fn pickIntVictim(alloc: *const Allocation, use_lists: []const std.ArrayList(usize), last_use: *const std.AutoHashMapUnmanaged(Value, usize), no_evict: []const bool, pos: usize, need_saved: bool) ?IntVictim {
    var best: ?IntVictim = null;
    var it = alloc.int.iterator();
    while (it.next()) |e| {
        const v = e.key_ptr.*;
        const reg = e.value_ptr.*;
        if (no_evict[@intFromEnum(v)]) continue; // entry params must stay in their register (prologue assumes it)
        if (last_use.get(v).? <= pos) continue; // not live past this position (dying/dead): not a useful victim
        if (need_saved and !isSavedReg(reg)) continue;
        const nu = nextUseAfter(use_lists[@intFromEnum(v)].items, pos) orelse last_use.get(v).?;
        if (best == null or nu > best.?.next) best = .{ .value = v, .next = nu, .reg = reg };
    }
    return best;
}

/// Give an integer register to `value` (defined/bound at `pos`, `cross` = it lives across a call) when
/// the int file is exhausted: evict the Belady victim (furthest next use) if its next use is further
/// than `value`'s own, else whole-spill `value`. The evicted victim gives up its register from `pos`
/// onward. When the victim is an INTRA-block value with a non-empty register prefix (`def_pos < pos`),
/// TAIL-SPLIT it: keep its register for `[def, pos)` and move to a fresh slot for `[pos, end)`,
/// recording a store at `pos` (prefix uses read the register, tail uses reload from the slot). A
/// cross-block victim (or one defined exactly at `pos`) whole-spills instead, moving from `alloc.int`
/// to `alloc.int_spill`. `value` owns the freed register from its def. The self-spill fallback always
/// whole-spills `value` (it is defined at `pos`, so it has no register prefix to keep).
fn placeIntUnderPressure(allocator: std.mem.Allocator, alloc: *Allocation, use_lists: []const std.ArrayList(usize), last_use: *const std.AutoHashMapUnmanaged(Value, usize), no_evict: []const bool, is_intra: []const bool, def_pos: []const usize, value: Value, pos: usize, cross: bool) Error!void {
    const self_next = nextUseAfter(use_lists[@intFromEnum(value)].items, pos) orelse last_use.get(value).?;
    if (pickIntVictim(alloc, use_lists, last_use, no_evict, pos, cross)) |vic| {
        if (vic.next > self_next) {
            const vidx = @intFromEnum(vic.value);
            // `alloc.def_pos` is only duped in at the end of allocation, so read the local `def_pos`.
            if (is_intra[vidx] and def_pos[vidx] < pos) {
                if (alloc.segments.get(vic.value)) |segs| {
                    // The victim was already tail-split then SECOND-CHANCE RE-HOMED into `vic.reg`, so
                    // its last segment is a `.reg`. Two sub-cases by whether that re-home's reload has
                    // fired yet at `pos` (the ascending-`from` assert in `appendSegment` is the guard
                    // that this branch chose right).
                    const last = segs[segs.len - 1];
                    std.debug.assert(last.loc == .reg);
                    if (last.from > pos) {
                        // PENDING re-home: pressure reclaimed `vic.reg` BEFORE the reload at
                        // `last.from` runs, so the value is still physically in its previous slot here
                        // and never actually re-enters a register. Cancel the re-home (drop the
                        // trailing `.reg` segment and its reload action) rather than append an
                        // out-of-order slot. The value stays in that prior slot (no store needed, its
                        // bits are already there) and `vic.reg` frees for the taker.
                        try cancelReHome(allocator, alloc, vic.value);
                        _ = alloc.int.remove(vic.value); // it no longer holds a register
                    } else {
                        // ACTIVE re-home (`last.from <= pos`): the value truly lives in `vic.reg` at
                        // `pos`. Spill that register's live part from `pos` onward. The append lands at
                        // or after the re-home position, so segment order is preserved.
                        const slot = alloc.spill_count;
                        alloc.spill_count += 1;
                        try appendSegment(allocator, alloc, vic.value, .{ .from = pos, .loc = .{ .slot = slot } });
                        try alloc.actions.append(allocator, .{ .at = pos, .kind = .store, .value = vic.value, .reg = vic.reg, .slot = slot });
                        _ = alloc.int.remove(vic.value); // moved to a slot: it no longer holds a register
                    }
                    try alloc.int.put(allocator, value, vic.reg);
                    return;
                }
                const slot = alloc.spill_count;
                alloc.spill_count += 1;
                _ = alloc.int.remove(vic.value);
                const segs = try allocator.alloc(Segment, 2);
                segs[0] = .{ .from = def_pos[vidx], .loc = .{ .reg = vic.reg } };
                segs[1] = .{ .from = pos, .loc = .{ .slot = slot } };
                // On a `put` failure `segments` never takes ownership, so free `segs` here to avoid a
                // leak. Once `put` succeeds, `deinit` frees it exactly once, so the later `append`
                // failure must NOT also free it (that would double-free under the errdefer deinit).
                alloc.segments.put(allocator, vic.value, segs) catch |e| {
                    allocator.free(segs);
                    return e;
                };
                try alloc.actions.append(allocator, .{ .at = pos, .kind = .store, .value = vic.value, .reg = vic.reg, .slot = slot });
            } else {
                _ = alloc.int.remove(vic.value);
                try alloc.int_spill.put(allocator, vic.value, alloc.spill_count);
                alloc.spill_count += 1;
            }
            try alloc.int.put(allocator, value, vic.reg);
            return;
        }
    }
    try alloc.int_spill.put(allocator, value, alloc.spill_count);
    alloc.spill_count += 1;
}

/// After the current position is placed and dying registers are freed, re-home split integer values
/// that presently live in a slot and still have an upcoming use into any LEFTOVER free integer
/// register, so their remaining tail uses read a register instead of reloading from the slot on every
/// use. The reload lands at the value's next use position. Most-urgent (nearest next use) first, so a
/// scarce free register goes to the value that reloads soonest.
///
/// riscv64 has no `actives` list: `alloc.int` is the "who holds this register now" map. A re-homed
/// value is therefore put BACK into `alloc.int` (register accounting), so `freeDying` returns its
/// register when it dies and `pickIntVictim` can evict it AGAIN (via `placeIntUnderPressure`'s
/// append-aware path). `intLocationAt` still reads `segments` first, so emission reads the value from
/// its `.reg` re-home segment, not from `alloc.int`.
///
/// Runs AFTER the current position was placed and after `freeDying`, so the placed value already
/// claimed whatever register it needed and `secondChance` only ever hands out genuinely free
/// registers. It can never dispossess a live value. A tail that crosses a call needs a callee-saved
/// register (`popSaved`), else any free register (`int_free.pop`). If none of the needed kind is free
/// the value is left in its slot (correct, it just reloads per use).
fn secondChance(
    allocator: std.mem.Allocator,
    alloc: *Allocation,
    use_lists: []const std.ArrayList(usize),
    last_use: *const std.AutoHashMapUnmanaged(Value, usize),
    int_free: *std.ArrayList(Reg),
    call_prefix: []const usize,
    pos: usize,
) Error!void {
    const Cand = struct { value: Value, next: usize, slot: u32, end: usize };
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(allocator);
    var it = alloc.segments.iterator();
    while (it.next()) |e| {
        const segs = e.value_ptr.*;
        const last = segs[segs.len - 1];
        switch (last.loc) {
            .reg => {}, // already in a register, nothing pending
            .slot => |slot| {
                const v = e.key_ptr.*;
                const nu = nextUseAfter(use_lists[@intFromEnum(v)].items, pos) orelse continue;
                try cands.append(allocator, .{ .value = v, .next = nu, .slot = slot, .end = last_use.get(v).? });
            },
        }
    }
    std.mem.sort(Cand, cands.items, {}, struct {
        fn f(_: void, a: Cand, b: Cand) bool {
            return a.next < b.next;
        }
    }.f);
    for (cands.items) |c| {
        // The re-homed tail is `[c.next, c.end)`: if a call lands in it, the tail needs a register the
        // callee preserves, so draw a callee-saved one. Otherwise any free register will do.
        const cross = crossesCall(call_prefix, c.next, c.end);
        const r2 = if (cross) popSaved(int_free) else int_free.pop();
        if (r2) |reg| {
            try appendSegment(allocator, alloc, c.value, .{ .from = c.next, .loc = .{ .reg = reg } });
            try alloc.actions.append(allocator, .{ .at = c.next, .kind = .reload, .value = c.value, .reg = reg, .slot = c.slot });
            try alloc.int.put(allocator, c.value, reg); // riscv64 register accounting (see the doc comment)
        } else {
            // No register of the needed kind is free: DECLINE. The value stays in its slot and its
            // remaining uses reload per use (correct, just not re-homed). Counted for tests only.
            alloc.rehome_declines += 1;
        }
    }
}

fn setUsed(row: []bool, v: Value) void {
    row[@intFromEnum(v)] = true;
}

/// Mark every value an instruction READS into `row`, a per-block "used" bitset for the liveness
/// fixpoint in `extendLiveRanges`. The mirror of `recordUses`, but recording a bitset instead of
/// last-use positions (so the two never disagree about what an instruction reads).
fn markUsedBitset(func: *const Function, inst: ir.function.Inst, row: []bool, fold: *const addrfold.Analysis) void {
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
        // Same fold reroute, in the backward liveness fixpoint (`extendLiveRanges`): a folded mem op
        // marks the fold base used in its block, so a base whose only in-block use is a folded load
        // is correctly carried live across a back-edge. Missing this reroute silently miscompiles a
        // loop-invariant folded base.
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
        .struct_new => |sn| for (func.valueList(sn.fields)) |v| setUsed(row, v),
        .call => |c| for (func.valueList(c.args)) |v| setUsed(row, v),
        .call_indirect => |c| {
            setUsed(row, c.target);
            for (func.valueList(c.args)) |v| setUsed(row, v);
        },
        .extract => |ex| setUsed(row, ex.aggregate),
        .convert => |cv| setUsed(row, cv.value),
        .unary => |u| setUsed(row, u.value),
        .@"if" => |cf| {
            setUsed(row, cf.cond);
            for (func.blockArgs(cf.then)) |v| setUsed(row, v);
            for (func.blockArgs(cf.@"else")) |v| setUsed(row, v);
        },
    }
}

/// Mark every value a block's terminator READS into `row` (the terminator half of
/// `markUsedBitset`). The mirror of `recordTermUses`.
fn markUsedTermBitset(func: *const Function, block: Block, row: []bool) void {
    if (func.terminator(block)) |term| switch (term) {
        .ret => |v| if (v) |vv| setUsed(row, vv),
        .jump => |j| for (func.blockArgs(j)) |v| setUsed(row, v),
    };
}

/// Backward liveness dataflow (a live-in/live-out fixpoint over the CFG's successor edges,
/// INCLUDING loop back-edges). The forward `last_use` pass records each value's last TEXTUAL use,
/// which under-covers a value live across a back-edge: e.g. a value defined in the entry block and
/// read inside a loop body has its textual last use in the body, but the back-edge re-enters the
/// body, so it is still live there on the next iteration. Freeing its register at the textual use
/// lets a later body temp reuse it, and the next iteration then reads garbage (a silent miscompile).
/// This pass raises `last_use[v]` to the terminator position of every block where `v` is LIVE-OUT,
/// so a loop-carried value keeps its register across the whole body. It ONLY ever raises a
/// last_use, never lowers one, so for forward-dominated code (straight-line, or a loop whose carried
/// values are threaded as block params) - where the forward scan already holds the maximal use
/// position - it is a no-op and allocation is byte-identical. Mirrors aarch64/isel.zig
/// `extendLiveRanges`. `block_end[bi]` is the terminator position of reachable block `bi` (0 for
/// unreachable blocks, which carry no live interval and are skipped throughout).
fn extendLiveRanges(
    allocator: std.mem.Allocator,
    func: *const Function,
    last_use: *std.AutoHashMapUnmanaged(Value, usize),
    block_end: []const usize,
    reachable: []const bool,
    fold: *const addrfold.Analysis,
) std.mem.Allocator.Error!void {
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
        // Unreachable blocks are skipped in the numbering/allocation passes, so here too they get no
        // successor edges and no def/use rows: their live_in/live_out stay all-false and never
        // propagate. Matches the reachable-skipping the two allocation walks already do.
        if (!reachable[bi]) continue;
        const block: Block = @enumFromInt(bi);
        const row = used[bi * nval ..][0..nval];
        for (func.blockParams(block)) |p| defined[bi * nval + @intFromEnum(p)] = true;
        for (func.blockInsts(block)) |inst| {
            markUsedBitset(func, inst, row, fold);
            if (func.instResult(inst)) |r| defined[bi * nval + @intFromEnum(r)] = true;
            if (func.opcode(inst) == .@"if") {
                const cf = func.opcode(inst).@"if";
                try succ[bi].append(allocator, @intFromEnum(cf.then.target));
                try succ[bi].append(allocator, @intFromEnum(cf.@"else".target));
            }
        }
        markUsedTermBitset(func, block, row);
        if (func.terminator(block)) |term| {
            if (term == .jump) try succ[bi].append(allocator, @intFromEnum(term.jump.target));
        }
    }

    const live_in = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(live_in);
    const live_out = try allocator.alloc(bool, nblocks * nval);
    defer allocator.free(live_out);
    @memset(live_in, false);
    @memset(live_out, false);

    // Standard iterative dataflow: live_out[b] = union of live_in over b's successors; live_in[b] =
    // used[b] union (live_out[b] minus defined[b]). Iterate to a fixpoint. Back-edges make a
    // header's live-in flow into the body's live-out, which is exactly the extension we need.
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
        if (!reachable[b]) continue;
        for (0..nval) |v| {
            if (!live_out[b * nval + v]) continue;
            // A live-out value is always defined (a param or an instruction result), so it has a
            // last_use entry; raise it to this block's end if the block outlives its textual use.
            const val: Value = @enumFromInt(v);
            if (last_use.getPtr(val)) |lp| {
                if (block_end[b] > lp.*) lp.* = block_end[b];
            }
        }
    }
}

/// Whether the integer `icmp` at `insts[idx]` fuses into an immediately-following
/// `@"if"` whose condition it is and whose only use it is. When it fuses, the icmp's
/// slt/sltu materialization is skipped and the if emits a native compare-and-branch on
/// the icmp's two operands (see the `.icmp` and `.@"if"` cases). This is the ONE
/// eligibility predicate shared by the icmp-skip and the fused if, so they never
/// disagree (no dangling or doubled compare). Gated to integer operands (the slt/sltu
/// path); float compares keep the materialize-then-test path. Immediately-preceding +
/// single-use make skipping the boolean register-safe: nothing runs between the icmp and
/// the if, so the operand registers still hold their values at the if, and no other
/// reader needs the boolean.
///
/// This predicate itself carries no model gate; both call sites in `emitFromAllocation` (the
/// icmp-skip and the fused `.@"if"`) additionally require the local `fuse_cmp_branch` (threaded
/// from `caps.fuse_cmp_branch`) before honoring it, so a model without the fusion falls back to
/// the materialize-then-test path unchanged.
fn fusesIntoNextIf(func: *const Function, insts: []const ir.function.Inst, idx: usize) bool {
    const cmp = switch (func.opcode(insts[idx])) {
        .icmp => |c| c,
        else => return false,
    };
    // Integer operands only. Float compares route through the flt/feq path, which
    // produces the boolean in an integer register a fused GPR branch cannot consume.
    if (isFloat(func, func.valueType(cmp.lhs))) return false;
    if (isVector(func, func.valueType(cmp.lhs))) return false;
    if (idx + 1 >= insts.len) return false; // must be immediately followed by the if
    const cf = switch (func.opcode(insts[idx + 1])) {
        .@"if" => |c| c,
        else => return false,
    };
    // Fused edges must be arg-free, the same restriction the plain if path enforces.
    if (func.blockArgs(cf.then).len != 0 or func.blockArgs(cf.@"else").len != 0) return false;
    const result = func.instResult(insts[idx]) orelse return false;
    if (cf.cond != result) return false; // the if must test exactly this icmp's result
    // Single-use: the boolean is read only by this if's condition. Since the icmp
    // immediately precedes the if and equals cf.cond, a total use-count of exactly 1
    // means the if's cond is the sole use, so skipping the boolean harms nothing.
    return countUses(func, result) == 1;
}

/// Whether the float `mul` at `insts[idx]` (scalar or RVV vector) fuses into an immediately-
/// following float `add`/`sub` that consumes its result as a fused multiply-add/subtract (one
/// rounding instead of two - legal because Vulcan permits fp-contraction). When it fuses, the
/// mul's materialization is skipped and the add/sub emits the matching fused instruction on the
/// mul's own operands (see the `.arith` case below): `fmadd`/`fmsub`/`fnmsub` for a scalar,
/// `vfmacc`/`vfmsac`/`vfnmsac` for an RVV vector. This is the ONE eligibility predicate shared
/// by the mul-skip and the fused add/sub emission, so they never disagree (no dangling or
/// doubled multiply) - mirrors `fusesIntoNextIf`. Gated to float operands, scalar or vector:
/// integer `add(mul,c)` has no rounding to fuse away and is never an fma. Unlike aarch64's NEON
/// FMLA/FMLS (which can only ever add or subtract the product, never negate the whole result,
/// so `sub(mul,c) = a*b-c` has no matching instruction there), RVV's OPFVV fused family covers
/// all three shapes - `vfmacc`/`vfmsac`/`vfnmsac` - so no shape needs rejecting here for a
/// vector mul. et-soc VPU (`vpu`, the same flag `selectFunction` threads through the whole
/// lowering pass) fuses only the FLOAT add shape: the CORE-ET ISA has just `fmadd.ps` (a*b+c),
/// with no packed subtract-fma (`fmsub.ps`/`fnmsub.ps`) and no packed-integer fma sibling of the
/// `pi` ops. So a `vpu` INTEGER vector mul is rejected outright here, and a `vpu` float vector
/// mul is accepted but only into an `add` (the `addsub.op != .add` guard below drops the sub
/// shapes). The VPU add emission (see `.arith` below) re-checks this SAME predicate, so the two
/// sites never disagree: without agreement a skipped-but-unfused product would be reloaded but
/// never materialized. The immediately-preceding + single-use conditions make skipping the product
/// register-safe: nothing runs between the mul and the add/sub, so the mul's operand registers
/// still hold their values there (floats never spill in this allocator - see `alloc.float` - and
/// a vector operand not yet spilled is unaffected by anything emitted in between, since nothing
/// is), and no other reader needs the standalone product.
fn fusesIntoNextArith(func: *const Function, insts: []const ir.function.Inst, idx: usize, vpu: bool) bool {
    const mul = switch (func.opcode(insts[idx])) {
        .arith => |a| a,
        else => return false,
    };
    if (mul.op != .mul) return false;
    const lhs_ty = func.valueType(mul.lhs);
    // A vector mul is assumed float (this backend's RVV arithmetic path - the `isVector` case
    // in `.arith` below - only ever lowers float lanes; there is no integer RVV path to guard
    // against). A scalar mul must be float too: !isFloat means an integer mul, no rounding to
    // fuse away. A vector mul under `vpu` fuses only when float: there is no packed-integer fma,
    // so a `vpu` INTEGER vector mul is rejected (see the doc comment above); a float one falls
    // through to the add-shape check below.
    const vector = isVector(func, lhs_ty);
    if (vector) {
        if (vpu and isIntVector(func, lhs_ty)) return false;
    } else if (!isFloat(func, lhs_ty)) {
        return false;
    } else if (isHalf(func, lhs_ty)) {
        // f16 is emulated as f32 with per-op rounding to half. A fused fmadd would round the
        // product-sum only once at f32 precision, skipping the multiply's intermediate half
        // rounding, which is not valid f16 semantics. Fall back to a rounded fmul then a rounded
        // fadd/fsub. Both the mul-skip and the fma-emit sites gate on this predicate, so they agree.
        return false;
    }
    if (idx + 1 >= insts.len) return false; // must be immediately followed by the add/sub
    const addsub = switch (func.opcode(insts[idx + 1])) {
        .arith => |a| a,
        else => return false,
    };
    if (addsub.op != .add and addsub.op != .sub) return false;
    // et-soc VPU has only `fmadd.ps` (a*b+c). There is no `fmsub.ps`/`fnmsub.ps`, so a `vpu`
    // float vector mul fuses only into an `add`; the sub shapes keep their separate fmul.ps +
    // fsub.ps. The RVV vector path (vpu == false) still fuses both add and sub via its OPFVV
    // family, and the scalar path is unaffected.
    if (vector and vpu and addsub.op != .add) return false;
    const result = func.instResult(insts[idx]) orelse return false;
    if (addsub.lhs != result and addsub.rhs != result) return false; // must consume this mul's result
    // Single-use: the product is read only by this add/sub. Since the mul immediately
    // precedes it and is one of its operands, a total use-count of exactly 1 means this is
    // the sole use, so skipping the materialization harms nothing.
    return countUses(func, result) == 1;
}

/// Whether the SCALAR-FLOAT fused multiply-add at mul index `idx` may actually be emitted as a
/// single R4-type `fmadd`/`fmsub`/`fnmsub`. Requires the shared `fusesIntoNextArith` eligibility
/// AND that all four float values it involves (the mul's two operands, the accumulator, and the
/// add/sub result) are register-resident: the R4 form reads three source registers at once, one
/// more than the two float spill scratch registers can reload, so under float register pressure
/// the pass must fall back to a separate mul + add (each within the two-scratch budget). Both the
/// mul-skip and the fused-emission sites gate on this same predicate, so they never disagree. With
/// no float spill (the common case) every operand is resident, so this is always true and the
/// fused instruction is emitted exactly as before this spill support existed: byte-identical.
/// Only meaningful for a scalar-float mul; the RVV vector fused path has three vector scratch
/// registers and keeps its own spill handling, so it does not consult this.
fn fusesScalarFloatArith(func: *const Function, alloc: *const Allocation, insts: []const ir.function.Inst, idx: usize, vpu: bool) bool {
    if (!fusesIntoNextArith(func, insts, idx, vpu)) return false;
    const mul = func.opcode(insts[idx]).arith;
    if (isVector(func, func.valueType(mul.lhs))) return false; // scalar-float only
    const mul_result = func.instResult(insts[idx]).?;
    const addsub = func.opcode(insts[idx + 1]).arith;
    const acc = if (addsub.lhs == mul_result) addsub.rhs else addsub.lhs; // the accumulator, c
    const res = func.instResult(insts[idx + 1]).?;
    return alloc.float.get(mul.lhs) != null and alloc.float.get(mul.rhs) != null and
        alloc.float.get(acc) != null and alloc.float.get(res) != null;
}

/// Whether the `arith_imm{.shl, b, k}` at index `idx` fuses with the next `arith{.add}` into one
/// Zba `sh{k}add rd, b, x` (rd = x + (b << k), a 64-bit result). This is the ONE eligibility
/// predicate shared by the shl-skip (in the `.arith_imm` arm) and the fused emit (in the `.arith`
/// add arm), so they never disagree (no dangling or doubled shift). `enabled` carries
/// `caps.fuse_shift_add`, which is FALSE by default and TRUE only for a Zba model, so without it
/// both sites fall back to the plain `slli`+`add` path and stay byte-identical.
///
/// Conditions: `sh{k}add` exists only for k in {1, 2, 3}, so the shift amount must be one of those.
/// The result is 64-bit (`sh{k}add` produces a full 64-bit sum: a 32-bit add would need `sh{k}add.uw`,
/// deferred). Only `.add` folds (there is no sh-sub form), and it is commutative, so the shl result
/// may be either add operand. Integer / GPR operands only (a float or vector shl routes elsewhere).
/// The shl result is SINGLE-USE (its only reader is this add, so skipping the standalone shifted value
/// is safe). The shl must immediately precede the add so nothing runs between them and `b` still holds
/// its value at the add (loaded fresh there), exactly as `fusesIntoNextArith` relies on for the fused
/// product's operands.
fn fusesIntoNextShiftAdd(func: *const Function, insts: []const ir.function.Inst, idx: usize, enabled: bool) bool {
    if (!enabled) return false;
    const shl = switch (func.opcode(insts[idx])) {
        .arith_imm => |a| a,
        else => return false,
    };
    if (shl.op != .shl) return false;
    // sh{k}add supports only k in {1, 2, 3}. Any other shift amount stays on the plain path.
    if (shl.imm < 1 or shl.imm > 3) return false;
    // Integer / GPR operands only: sh-add is a GPR ALU form. A float or vector shl is served elsewhere.
    if (isVector(func, func.valueType(shl.lhs)) or isFloat(func, func.valueType(shl.lhs))) return false;
    if (idx + 1 >= insts.len) return false; // must be immediately followed by the add
    const add = switch (func.opcode(insts[idx + 1])) {
        .arith => |a| a,
        else => return false,
    };
    if (add.op != .add) return false; // only `.add` folds (no sh-sub form)
    const result = func.instResult(insts[idx]) orelse return false;
    if (add.lhs != result and add.rhs != result) return false; // must consume the shl's result
    // 64-bit result only: sh{k}add is a full 64-bit add. A 32-bit result would need sh{k}add.uw.
    if (intBits(func, func.valueType(func.instResult(insts[idx + 1]) orelse return false)) != 64) return false;
    // Single-use: the shifted value is read only by this add. Since the shl immediately precedes it
    // and is one of its operands, a total use-count of exactly 1 means this is the sole use, so
    // skipping the materialization harms nothing.
    return countUses(func, result) == 1;
}

/// Total operand uses of `v` across the whole function (instruction operands, if/jump
/// edge args, and terminators). Backs the fusion eligibility's single-use check.
fn countUses(func: *const Function, v: Value) usize {
    var count: usize = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| count += usesInInst(func, inst, v);
        count += usesInTerm(func, block, v);
    }
    return count;
}

fn usesInInst(func: *const Function, inst: ir.function.Inst, v: Value) usize {
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
        .extract => |e| {
            if (e.aggregate == v) c += 1;
        },
        .convert => |cv| {
            if (cv.value == v) c += 1;
        },
        .unary => |u| {
            if (u.value == v) c += 1;
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

fn usesInTerm(func: *const Function, block: Block, v: Value) usize {
    var c: usize = 0;
    if (func.terminator(block)) |term| switch (term) {
        .ret => |x| if (x) |xx| {
            if (xx == v) c += 1;
        },
        .jump => |j| for (func.blockArgs(j)) |a| {
            if (a == v) c += 1;
        },
    };
    return c;
}

/// Return temp registers of dying values to the appropriate free list. `vpu` selects which
/// disjoint half of the vector/float partition a dying value's register returns to (see the vpu
/// mode note above `vpu_vector_regs`): the RVV vector map/pool when false, the vpu FReg
/// vector map/pool when true. A value's register always belongs to exactly one of the two, since a
/// single `allocateRegisters` call runs in one mode for the whole function.
fn freeDying(allocator: std.mem.Allocator, func: *const Function, dying: []const Value, alloc: *const Allocation, int_free: *std.ArrayList(Reg), float_free: *std.ArrayList(FReg), vector_free: *std.ArrayList(VReg), vpu_vector_free: *std.ArrayList(FReg), vpu: bool) std.mem.Allocator.Error!void {
    for (dying) |v| {
        if (isVector(func, func.valueType(v))) {
            if (vpu) {
                if (alloc.vpu_vector.get(v)) |r| try vpu_vector_free.append(allocator, r);
            } else {
                if (alloc.vector.get(v)) |r| try vector_free.append(allocator, r);
            }
        } else if (isFloat(func, func.valueType(v))) {
            // A spilled float holds no register. Only a registered value frees one.
            if (alloc.float.get(v)) |r| {
                const temp = if (vpu) isFloatTempRegVpu(r) else isFloatTempReg(r);
                const saved = if (vpu) isFloatSavedRegVpu(r) else isFloatSavedReg(r);
                if (temp or saved) try float_free.append(allocator, r);
            }
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

fn popFloatSavedVpu(free: *std.ArrayList(FReg)) ?FReg {
    var i = free.items.len;
    while (i > 0) {
        i -= 1;
        if (isFloatSavedRegVpu(free.items[i])) return free.orderedRemove(i);
    }
    return null;
}

// ===========================================================================
// riscv64 RegDescription for the shared Wimmer-Franz allocator (wimmer.zig).
//
// riscv64 has FOUR register classes, versus aarch64's two:
//   class 0 "int"        (Reg,  8-byte slot)  index = @intFromEnum(Reg)  x0..x31
//   class 1 "float"      (FReg, 8-byte slot)  index = @intFromEnum(FReg) f0..f31
//   class 2 "vector"     (VReg, 16-byte slot) index = @intFromEnum(VReg) v0..v31  (RVV)
//   class 3 "vpu_vector" (FReg, 32-byte slot) index = @intFromEnum(FReg) f0..f31  (et-soc VPU)
//
// A per-function `vpu` bool (from caps.vpu) picks ONE of the two vector classes: RVV (class 2) when
// false, et-soc VPU (class 3) when true, so exactly one of class 2 / class 3 has a non-empty pool.
// vpu mode also NARROWS the scalar-float pool (class 1) to f0..f7, because the VPU has no separate
// vector file and instead partitions the shared FReg file (see the vpu note above `vpu_vector_regs`).
//
// INDEX-SPACE OVERLAP: class 1 and class 3 BOTH index the FReg enum, so their register indices
// collide numerically (e.g. index 16 is f16 for either). They stay disjoint by POOL, never by index:
// class 1 draws from f0..f7 (vpu) or f0..f9/f18..f29 (non-vpu), class 3 draws from f16..f27, and the
// two are never both active in one function. The (class, index) pair is what disambiguates them, and
// the eventual translation (Task 3) maps a (class, index) back to the right FReg/VReg. This is only
// the DESCRIPTION; no allocation runs here.
//
// This mirrors `aarch64RegDescription`'s shape and builds its content from the pools/ABI/call logic
// in `allocateRegisters` above.

/// Backend context threaded through `classOf`/`useKind`. Unlike aarch64 (whose class decision needs
/// no state), riscv64's class for a VECTOR value depends on the per-function `vpu` mode, so the ctx
/// carries it. Two file-scope singletons give a stable, non-owned `ctx` pointer per mode with no
/// per-call allocation (the shared `RegDescription.deinit` does not free `ctx`).
const Riscv64RegCtx = struct { vpu: bool };
const riscv64_reg_ctx_scalar: Riscv64RegCtx = .{ .vpu = false };
const riscv64_reg_ctx_vpu: Riscv64RegCtx = .{ .vpu = true };

/// `RegDescription.classOf` for riscv64: an integer value is class 0, a scalar float is class 1, and
/// a vector is class 2 (RVV) or class 3 (VPU) depending on the ctx's `vpu` mode.
fn riscv64ClassOf(ctx: *const anyopaque, func: *const Function, v: Value) u16 {
    const rc: *const Riscv64RegCtx = @ptrCast(@alignCast(ctx));
    const ty = func.valueType(v);
    if (isVector(func, ty)) return if (rc.vpu) 3 else 2;
    if (isFloat(func, ty)) return 1;
    return 0;
}

/// `RegDescription.useKind` for riscv64: every operand needs a register. riscv64 has no memory
/// operands, and some sites (e.g. the fused compare-and-branch) cannot reload a spilled operand, so
/// `must_have_register` is both conservative and correct. Unused params are the generic hook shape.
fn riscv64UseKind(ctx: *const anyopaque, func: *const Function, inst: ir.function.Inst, operand: Value) wimmer.UseKind {
    _ = ctx;
    _ = func;
    _ = inst;
    _ = operand;
    return .must_have_register;
}

/// Allocate a `[]u16` of the class-relative indices (`@intFromEnum`) of `regs`. The caller owns it.
fn regIndexSlice(allocator: std.mem.Allocator, comptime RegT: type, regs: []const RegT) std.mem.Allocator.Error![]u16 {
    const out = try allocator.alloc(u16, regs.len);
    for (regs, 0..) |r, i| out[i] = @intFromEnum(r);
    return out;
}

/// Build the per-function riscv64 `RegDescription` the shared Wimmer-Franz allocator consumes,
/// mirroring `aarch64RegDescription`. The four classes, entry-param pinning, per-call clobbers, and
/// scratch registers come from `allocateRegisters`'s pools/ABI/call logic. `vpu` selects the et-soc
/// VPU register model (class 3 active, narrowed float pool) over the RVV one (class 2 active). Task 1
/// builds only the description (no allocation runs). The caller owns the result and must `deinit` it.
///
/// Call-clobber mechanism: every call site clobbers, per class, that class's CALLER-SAVED registers,
/// AND ALL of the active vector class's registers (v1..v27 for RVV, f16..f27 for VPU), since every
/// vector register is caller-saved. A vector value therefore cannot survive a call in a register, so
/// the shared allocator SPILLS/splits it across the call. This GENERALIZES the old riscv64 path,
/// which bailed `error.Unsupported` on a vector live across a call.
///
/// Scope note: the f16-scratch shrink (`temp_regs_f16`, when the function uses f16 without Zfh) is
/// NOT modeled here. This signature has no capability input, and it is the byte-identical common
/// case that class 0 uses the full `temp_regs`. A later task that wires the shared allocator into the
/// riscv64 emission path can thread the f16 shrink through when needed.
pub fn riscv64RegDescription(allocator: std.mem.Allocator, func: *const Function, vpu: bool) std.mem.Allocator.Error!wimmer.RegDescription {
    // --- Class 0 (int): caller-saved temps x5/x7/x28..x31 + callee-saved x9/x18..x27. ---
    const int_alloc = try allocator.alloc(u16, temp_regs.len + saved_regs.len);
    errdefer allocator.free(int_alloc);
    for (temp_regs, 0..) |r, i| int_alloc[i] = @intFromEnum(r);
    for (saved_regs, 0..) |r, i| int_alloc[temp_regs.len + i] = @intFromEnum(r);
    const int_cs = try regIndexSlice(allocator, Reg, &saved_regs);
    errdefer allocator.free(int_cs);

    // --- Class 1 (float): vpu narrows to f0..f7 (no callee-saved); non-vpu is the full temp +
    // callee-saved float pool. ---
    const float_alloc = if (vpu)
        try regIndexSlice(allocator, FReg, &float_temp_regs_vpu)
    else blk: {
        const out = try allocator.alloc(u16, float_temp_regs.len + float_saved_regs.len);
        for (float_temp_regs, 0..) |r, i| out[i] = @intFromEnum(r);
        for (float_saved_regs, 0..) |r, i| out[float_temp_regs.len + i] = @intFromEnum(r);
        break :blk out;
    };
    errdefer allocator.free(float_alloc);
    const float_cs = if (vpu)
        try allocator.alloc(u16, 0)
    else
        try regIndexSlice(allocator, FReg, &float_saved_regs);
    errdefer allocator.free(float_cs);

    // --- Class 2 (RVV vector): v1..v27, all caller-saved. Empty under vpu (no RVV). ---
    const vec_alloc = if (vpu)
        try allocator.alloc(u16, 0)
    else
        try regIndexSlice(allocator, VReg, &vector_regs);
    errdefer allocator.free(vec_alloc);
    const vec_cs = try allocator.alloc(u16, 0);
    errdefer allocator.free(vec_cs);

    // --- Class 3 (VPU vector): f16..f27, all caller-saved. Empty in non-vpu. ---
    const vpu_alloc = if (vpu)
        try regIndexSlice(allocator, FReg, &vpu_vector_regs)
    else
        try allocator.alloc(u16, 0);
    errdefer allocator.free(vpu_alloc);
    const vpu_cs = try allocator.alloc(u16, 0);
    errdefer allocator.free(vpu_cs);

    const classes = try allocator.alloc(wimmer.RegClass, 4);
    errdefer allocator.free(classes);
    classes[0] = .{ .name = "int", .allocatable = int_alloc, .callee_saved = int_cs, .slot_bytes = 8 };
    classes[1] = .{ .name = "float", .allocatable = float_alloc, .callee_saved = float_cs, .slot_bytes = 8 };
    classes[2] = .{ .name = "vector", .allocatable = vec_alloc, .callee_saved = vec_cs, .slot_bytes = 16 };
    classes[3] = .{ .name = "vpu_vector", .allocatable = vpu_alloc, .callee_saved = vpu_cs, .slot_bytes = 32 };

    // --- Entry params: the first 8 int params pin a0..a7 (x10..x17), the first 8 float params pin
    // fa0..fa7 (f10..f17). A vector entry param has no ABI register (riscv64 rejects it downstream),
    // so it is NOT pre-colored. Params past the first 8 of a class arrive on the stack and are left
    // to the translation. int/float use SEPARATE ABI counters, matching `allocateRegisters`. ---
    var ef: std.ArrayList(wimmer.FixedAssign) = .empty;
    errdefer ef.deinit(allocator);
    if (func.blockCount() != 0) {
        var int_idx: usize = 0;
        var float_idx: usize = 0;
        for (func.blockParams(@enumFromInt(0))) |p| {
            const ty = func.valueType(p);
            if (isVector(func, ty)) continue; // no ABI vector register; not pre-colored
            if (isFloat(func, ty)) {
                if (float_idx < 8) try ef.append(allocator, .{ .value = p, .class = 1, .reg = @intFromEnum(fargReg(float_idx)) });
                float_idx += 1;
            } else {
                if (int_idx < 8) try ef.append(allocator, .{ .value = p, .class = 0, .reg = @intFromEnum(argReg(int_idx)) });
                int_idx += 1;
            }
        }
    }
    const entry_fixed = try ef.toOwnedSlice(allocator);
    errdefer allocator.free(entry_fixed);

    // --- Call sites: one per `.call` position, in the SAME single-step numbering `buildIntervals`
    // uses (block-param row, one position per instruction, one terminator slot, over every block), so
    // the positions line up with the intervals. ---
    var call_positions: std.ArrayList(u32) = .empty;
    defer call_positions.deinit(allocator);
    {
        var p: u32 = 0;
        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(bi);
            p += 1; // block-parameter row
            for (func.blockInsts(block)) |inst| {
                switch (func.opcode(inst)) {
                    .call, .call_indirect => try call_positions.append(allocator, p),
                    else => {},
                }
                p += 1;
            }
            p += 1; // terminator slot
        }
    }

    const call_sites = try allocator.alloc(wimmer.CallSite, call_positions.items.len);
    var built: usize = 0;
    errdefer {
        for (call_sites[0..built]) |cs| {
            for (cs.clobbered) |cr| allocator.free(cr.regs);
            allocator.free(cs.clobbered);
        }
        allocator.free(call_sites);
    }
    for (call_positions.items, 0..) |cpos, i| {
        // Class 0: the caller-saved int temps (callee-saved x9/x18..x27 survive a call).
        const int_clob = try regIndexSlice(allocator, Reg, &temp_regs);
        errdefer allocator.free(int_clob);
        // Class 1: the caller-saved float temps (vpu: f0..f7; non-vpu: f0..f7/f28/f29).
        const float_clob = if (vpu)
            try regIndexSlice(allocator, FReg, &float_temp_regs_vpu)
        else
            try regIndexSlice(allocator, FReg, &float_temp_regs);
        errdefer allocator.free(float_clob);
        // Class 2: ALL of v1..v27 (every RVV register is caller-saved). Empty under vpu.
        const vec_clob = if (vpu)
            try allocator.alloc(u16, 0)
        else
            try regIndexSlice(allocator, VReg, &vector_regs);
        errdefer allocator.free(vec_clob);
        // Class 3: ALL of f16..f27 (every VPU vector register is caller-saved). Empty in non-vpu.
        const vpu_clob = if (vpu)
            try regIndexSlice(allocator, FReg, &vpu_vector_regs)
        else
            try allocator.alloc(u16, 0);
        errdefer allocator.free(vpu_clob);
        const clob = try allocator.alloc(wimmer.ClassRegs, 4);
        clob[0] = .{ .class = 0, .regs = int_clob };
        clob[1] = .{ .class = 1, .regs = float_clob };
        clob[2] = .{ .class = 2, .regs = vec_clob };
        clob[3] = .{ .class = 3, .regs = vpu_clob };
        call_sites[i] = .{ .pos = cpos, .clobbered = clob };
        built = i + 1;
    }

    // --- Scratch, indexed by class: the reserved registers the backend already keeps out of every
    // pool for parallel-move cycle breaking / spill reload. Int x6, float f31 (f8 in vpu, where f31
    // sits inside the VPU partition), RVV v31, VPU f31 (reserved partition headroom). ---
    const scratch = try allocator.alloc(u16, 4);
    errdefer allocator.free(scratch);
    scratch[0] = @intFromEnum(spill_scratch0);
    scratch[1] = if (vpu) @intFromEnum(float_spill_scratch0_vpu) else @intFromEnum(float_scratch);
    scratch[2] = @intFromEnum(vector_scratch);
    scratch[3] = @intFromEnum(float_scratch);

    return .{
        .classes = classes,
        .classOf = riscv64ClassOf,
        .useKind = riscv64UseKind,
        .entry_fixed = entry_fixed,
        .call_sites = call_sites,
        .scratch = scratch,
        .ctx = if (vpu) &riscv64_reg_ctx_vpu else &riscv64_reg_ctx_scalar,
    };
}

// ===========================================================================
// Shared Wimmer-Franz integration (riscv64 adoption Tasks 2/3/4): translate a finished
// `wimmer.Allocation` into this backend's own `Allocation` and drive the existing
// `emitFromAllocation`. Covers ALL FOUR classes: INT, scalar-FLOAT, RVV VECTOR (0/1/2) and the et-soc
// VPU VECTOR class (3, RV-T4); anything not faithfully translatable (an f16 function, a wrong-width
// vector, a split/spilled entry param, a same-position action hazard, an if-edge move) bails
// `error.Unsupported`, never a silent miscompile. The default `compileFunction` path is untouched.
// ===========================================================================

/// The register class of `v` for the shared allocator: 0 int, 1 scalar float, 2 RVV vector, 3 VPU
/// vector (the last two selected by `vpu`). Mirrors `riscv64ClassOf` without the type-erased ctx.
fn wimmerClassOf(func: *const Function, v: Value, vpu: bool) u16 {
    const ty = func.valueType(v);
    if (isVector(func, ty)) return if (vpu) 3 else 2;
    if (isFloat(func, ty)) return 1;
    return 0;
}

fn intLocFromWimmer(loc: wimmer.Location) IntLoc {
    return switch (loc) {
        .reg => |ri| .{ .reg = @enumFromInt(@as(u5, @intCast(ri))) },
        .slot => |s| .{ .slot = s },
    };
}

fn floatLocFromWimmer(loc: wimmer.Location) FloatLoc {
    return switch (loc) {
        .reg => |ri| .{ .reg = @enumFromInt(@as(u5, @intCast(ri))) },
        .slot => |s| .{ .slot = s },
    };
}

fn vectorLocFromWimmer(loc: wimmer.Location) VectorLoc {
    return switch (loc) {
        .reg => |ri| .{ .reg = @enumFromInt(@as(u5, @intCast(ri))) },
        .slot => |s| .{ .slot = s },
    };
}

fn vpuLocFromWimmer(loc: wimmer.Location) VpuLoc {
    return switch (loc) {
        .reg => |ri| .{ .reg = @enumFromInt(@as(u5, @intCast(ri))) },
        .slot => |s| .{ .slot = s },
    };
}

fn edgeLocFromWimmer(loc: wimmer.Location) EdgeLoc {
    return switch (loc) {
        .reg => |ri| .{ .reg = ri },
        .slot => |s| .{ .slot = s },
    };
}

/// Build the drain action realizing `src -> dst` for an INT `value` at `at`: register->slot a
/// `.store`, slot->register a `.reload`, register->register a `.move`. A slot->slot transition needs
/// a scratch this translation does not model, so it bails (never a silent miscompile).
fn transitionIntAction(value: Value, src: IntLoc, dst: IntLoc, at: usize) Error!SplitAction {
    return switch (src) {
        .reg => |sr| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .move, .class = 0, .value = value, .reg = dr, .from_reg = sr },
            .slot => |ds| .{ .at = at, .kind = .store, .class = 0, .value = value, .reg = sr, .slot = ds },
        },
        .slot => |ss| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .reload, .class = 0, .value = value, .reg = dr, .slot = ss },
            .slot => error.Unsupported,
        },
    };
}

/// The scalar-float analogue of `transitionIntAction` (class 1, `freg`/`from_freg`).
fn transitionFloatAction(value: Value, src: FloatLoc, dst: FloatLoc, at: usize) Error!SplitAction {
    return switch (src) {
        .reg => |sr| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .move, .class = 1, .value = value, .freg = dr, .from_freg = sr },
            .slot => |ds| .{ .at = at, .kind = .store, .class = 1, .value = value, .freg = sr, .slot = ds },
        },
        .slot => |ss| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .reload, .class = 1, .value = value, .freg = dr, .slot = ss },
            .slot => error.Unsupported,
        },
    };
}

/// The RVV vector analogue of `transitionIntAction`/`transitionFloatAction` (class 2, `vreg`/
/// `from_vreg`): register->slot a `.store` (vse32), slot->register a `.reload` (vle32), register->
/// register a `.move` (vmv.v.v). A slot->slot transition needs a scratch this translation does not
/// model, so it bails (never a silent miscompile) - the shared resolver never emits one anyway.
fn transitionVectorAction(value: Value, src: VectorLoc, dst: VectorLoc, at: usize) Error!SplitAction {
    return switch (src) {
        .reg => |sr| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .move, .class = 2, .value = value, .vreg = dr, .from_vreg = sr },
            .slot => |ds| .{ .at = at, .kind = .store, .class = 2, .value = value, .vreg = sr, .slot = ds },
        },
        .slot => |ss| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .reload, .class = 2, .value = value, .vreg = dr, .slot = ss },
            .slot => error.Unsupported,
        },
    };
}

/// The et-soc VPU vector analogue of `transitionVectorAction` (class 3, `freg`/`from_freg` on the
/// f16..f27 partition): register->slot a `.store` (fsw.ps), slot->register a `.reload` (flw.ps),
/// register->register a `.move` (a `vpu_pack_base` round trip, see the class-3 `emitSplitAction`). A
/// slot->slot transition needs a scratch this translation does not model, so it bails (never a silent
/// miscompile) - the shared resolver never emits one anyway.
fn transitionVpuAction(value: Value, src: VpuLoc, dst: VpuLoc, at: usize) Error!SplitAction {
    return switch (src) {
        .reg => |sr| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .move, .class = 3, .value = value, .freg = dr, .from_freg = sr },
            .slot => |ds| .{ .at = at, .kind = .store, .class = 3, .value = value, .freg = sr, .slot = ds },
        },
        .slot => |ss| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .reload, .class = 3, .value = value, .freg = dr, .slot = ss },
            .slot => error.Unsupported,
        },
    };
}

/// A class-tagged register key: the int file is 0x000+, the scalar-float file 0x100+, the RVV vector
/// file 0x200+, the VPU vector file 0x300+, so no two classes ever collide numerically (class 1 and
/// class 3 both ride the FReg file but never coexist in one function; the tag keeps them apart anyway).
fn actionDstKey(a: SplitAction) u32 {
    return switch (a.class) {
        0 => @intFromEnum(a.reg),
        1 => 0x100 | @as(u32, @intFromEnum(a.freg)),
        2 => 0x200 | @as(u32, @intFromEnum(a.vreg)),
        3 => 0x300 | @as(u32, @intFromEnum(a.freg)),
        else => unreachable,
    };
}
fn actionSrcKey(a: SplitAction) u32 {
    return switch (a.class) {
        0 => @intFromEnum(a.reg),
        1 => 0x100 | @as(u32, @intFromEnum(a.freg)),
        2 => 0x200 | @as(u32, @intFromEnum(a.vreg)),
        3 => 0x300 | @as(u32, @intFromEnum(a.freg)),
        else => unreachable,
    };
}
fn actionFromKey(a: SplitAction) u32 {
    return switch (a.class) {
        0 => @intFromEnum(a.from_reg),
        1 => 0x100 | @as(u32, @intFromEnum(a.from_freg)),
        2 => 0x200 | @as(u32, @intFromEnum(a.from_vreg)),
        3 => 0x300 | @as(u32, @intFromEnum(a.from_freg)),
        else => unreachable,
    };
}
/// The register an action writes (reload/move destination), or null if it writes only memory.
fn actionWrites(a: SplitAction) ?u32 {
    return switch (a.kind) {
        .store => null,
        .reload, .move => actionDstKey(a),
    };
}
/// The register an action reads (store source / move source), or null if it reads only memory.
fn actionReads(a: SplitAction) ?u32 {
    return switch (a.kind) {
        .store => actionSrcKey(a),
        .move => actionFromKey(a),
        .reload => null,
    };
}
/// Whether any two actions at the SAME position conflict on a register: one writes a register the
/// other reads or writes. The emission drains a position's actions in a fixed order with no parallel-
/// move resolution, so such a set could clobber a live value. This task does not model that, so the
/// translation bails rather than risk a miscompile. O(n^2) over a short list.
fn hasSamePosRegHazard(actions: []const SplitAction) bool {
    for (actions, 0..) |a, i| {
        for (actions[i + 1 ..]) |b| {
            if (a.at != b.at) continue;
            if (actionWrites(a)) |w| {
                if (actionWrites(b)) |wb| {
                    if (wb == w) return true;
                }
                if (actionReads(b)) |r| {
                    if (r == w) return true;
                }
            }
            if (actionWrites(b)) |w| {
                if (actionReads(a)) |r| {
                    if (r == w) return true;
                }
            }
        }
    }
    return false;
}

/// Whether block `block` ends in an `if` (a multi-successor terminator the riscv64 emission lowers as
/// a bare branch, with no place to realize an edge move). Used to reject an edge move on an if-edge.
fn blockHasIf(func: *const Function, block: Block) bool {
    for (func.blockInsts(block)) |inst| {
        if (func.opcode(inst) == .@"if") return true;
    }
    return false;
}

/// Whether any translated edge move sits on an edge whose predecessor ends in an `if`. The riscv64
/// `.@"if"` emission only branches (it never realizes a move), and `splitCriticalEdges` splits only
/// CRITICAL edges, so a surviving non-critical if-edge move would be silently dropped. Bail on it.
fn edgeMoveOnIfEdge(func: *const Function, alloc: *const Allocation) bool {
    for (alloc.edge_moves) |set| {
        if (set.moves.len != 0 and blockHasIf(func, set.pred)) return true;
    }
    return false;
}

/// Translate a finished shared `wimmer.Allocation` into this backend's `Allocation` so the existing
/// `emitFromAllocation` can consume it. A whole-life value (one segment) lands in the class `int`/
/// `float` (register) or `int_spill`/`float_spill` (slot) maps exactly as the native allocator would
/// leave it; a genuinely split value lands in `segments`/`float_segments` plus per-transition drain
/// actions. Entry params are required whole-life in their ABI register (the shared hint), so the
/// prologue's move is a no-op; anything else bails. An RVV vector (class 2) lands in `vector`/
/// `vector_spill`/`vector_segments`; an et-soc VPU value (class 3) lands in `vpu_vector`/
/// `vpu_vector_spill`/`vpu_segments`, a VPU value live across a call spilling to a 32-byte slot.
fn translateAllocation(allocator: std.mem.Allocator, func: *const Function, vpu: bool, walloc: *const wimmer.Allocation) Error!Allocation {
    var alloc: Allocation = .{
        .int = .empty,
        .float = .empty,
        .vector = .empty,
        .vector_spill = .empty,
        .vector_spill_count = 0,
        .vpu_vector = .empty,
        .vpu_vector_spill = .empty,
        .vpu_vector_spill_count = 0,
        .int_spill = .empty,
        .spill_count = 0,
        .float_spill = .empty,
        .float_spill_count = 0,
        .incoming_stack = .empty,
    };
    errdefer alloc.deinit(allocator);

    std.debug.assert(walloc.slot_count_per_class.len == 4);
    alloc.spill_count = walloc.slot_count_per_class[0];
    alloc.float_spill_count = walloc.slot_count_per_class[1];
    // Class 2 (RVV vector) 16-byte slots and class 3 (et-soc VPU vector) 32-byte slots. Exactly one of
    // the two is ever non-zero (RVV xor VPU, per the per-function `vpu` mode).
    alloc.vector_spill_count = walloc.slot_count_per_class[2];
    alloc.vpu_vector_spill_count = walloc.slot_count_per_class[3];

    // def_pos in the SAME single-step numbering the shared allocator and `emitFromAllocation` use
    // (block-param row, one position per instruction, one terminator slot, over every block). The
    // caller guarantees every block is reachable, so this all-block walk matches the emission exactly.
    const nval = func.valueCount();
    const def_pos = try allocator.alloc(usize, nval);
    errdefer allocator.free(def_pos);
    @memset(def_pos, 0);
    {
        var pos: usize = 0;
        for (0..func.blockCount()) |bi| {
            const block: Block = @enumFromInt(bi);
            for (func.blockParams(block)) |p| def_pos[@intFromEnum(p)] = pos;
            pos += 1;
            for (func.blockInsts(block)) |inst| {
                if (func.instResult(inst)) |r| def_pos[@intFromEnum(r)] = pos;
                pos += 1;
            }
            pos += 1; // terminator slot
        }
    }

    // Entry-param ABI registers (a0..a7 int, fa0..fa7 float), used to require each param whole-life in
    // its ABI register below. Separate int/float counters, matching `riscv64RegDescription`.
    var eparam_class: std.AutoHashMapUnmanaged(Value, u8) = .empty; // 0 int, 1 float
    defer eparam_class.deinit(allocator);
    var eparam_reg: std.AutoHashMapUnmanaged(Value, u16) = .empty; // ABI register index
    defer eparam_reg.deinit(allocator);
    if (func.blockCount() != 0) {
        var int_idx: usize = 0;
        var float_idx: usize = 0;
        for (func.blockParams(@enumFromInt(0))) |p| {
            const ty = func.valueType(p);
            if (isVector(func, ty)) return error.Unsupported; // no ABI vector register (riscv64 rejects it too)
            if (isFloat(func, ty)) {
                if (float_idx >= 8) return error.Unsupported; // fp stack params not modeled here
                try eparam_class.put(allocator, p, 1);
                try eparam_reg.put(allocator, p, @intFromEnum(fargReg(float_idx)));
                float_idx += 1;
            } else {
                if (int_idx >= 8) return error.Unsupported; // int stack params not modeled here
                try eparam_class.put(allocator, p, 0);
                try eparam_reg.put(allocator, p, @intFromEnum(argReg(int_idx)));
                int_idx += 1;
            }
        }
    }

    var it = walloc.segments.iterator();
    while (it.next()) |e| {
        const value = e.key_ptr.*;
        const wsegs = e.value_ptr.*;
        std.debug.assert(wsegs.len > 0);
        const class = wimmerClassOf(func, value, vpu);
        // The RVV lowering hardcodes VL=4 in its `vsetivli` preamble, so a vector of any other width
        // would be miscompiled. Reject it (mirrors the native path's `isRvvWidth` gate).
        if (class == 2 and !isRvvWidth(func, func.valueType(value))) return error.Unsupported;
        // The et-soc VPU is a fixed 8-lane machine; any other width would be miscompiled. Reject it
        // (mirrors the native path's `isVpuWidth` gate).
        if (class == 3 and !isVpuWidth(func, func.valueType(value))) return error.Unsupported;

        const is_eparam = eparam_class.get(value) != null;
        // A split or slot-resident entry param would need prologue handling this task does not model.
        if (is_eparam and (wsegs.len != 1 or wsegs[0].loc != .reg)) return error.Unsupported;

        if (class == 0) {
            if (wsegs.len == 1) {
                switch (wsegs[0].loc) {
                    .reg => |ri| {
                        const r: Reg = @enumFromInt(@as(u5, @intCast(ri)));
                        // An entry param must sit in its ABI register (the hint is a preference); if the
                        // allocator placed it elsewhere the prologue would leave the wrong value there.
                        if (eparam_reg.get(value)) |ar| {
                            if (@as(u16, @intFromEnum(r)) != ar) return error.Unsupported;
                        }
                        try alloc.int.put(allocator, value, r);
                    },
                    .slot => |s| try alloc.int_spill.put(allocator, value, s),
                }
            } else {
                const segs = try allocator.alloc(Segment, wsegs.len);
                for (wsegs, 0..) |ws, i| segs[i] = .{ .from = ws.from, .loc = intLocFromWimmer(ws.loc) };
                alloc.segments.put(allocator, value, segs) catch |err| {
                    allocator.free(segs);
                    return err;
                };
                var i: usize = 0;
                while (i + 1 < wsegs.len) : (i += 1) {
                    try alloc.actions.append(allocator, try transitionIntAction(value, segs[i].loc, segs[i + 1].loc, segs[i + 1].from));
                }
            }
        } else if (class == 1) {
            // class 1: scalar float.
            if (wsegs.len == 1) {
                switch (wsegs[0].loc) {
                    .reg => |ri| {
                        const fr: FReg = @enumFromInt(@as(u5, @intCast(ri)));
                        if (eparam_reg.get(value)) |ar| {
                            if (@as(u16, @intFromEnum(fr)) != ar) return error.Unsupported;
                        }
                        try alloc.float.put(allocator, value, fr);
                    },
                    .slot => |s| try alloc.float_spill.put(allocator, value, s),
                }
            } else {
                const segs = try allocator.alloc(FloatSegment, wsegs.len);
                for (wsegs, 0..) |ws, i| segs[i] = .{ .from = ws.from, .loc = floatLocFromWimmer(ws.loc) };
                alloc.float_segments.put(allocator, value, segs) catch |err| {
                    allocator.free(segs);
                    return err;
                };
                var i: usize = 0;
                while (i + 1 < wsegs.len) : (i += 1) {
                    try alloc.actions.append(allocator, try transitionFloatAction(value, segs[i].loc, segs[i + 1].loc, segs[i + 1].from));
                }
            }
        } else if (class == 2) {
            // class 2: RVV vector. A vector entry param is rejected above (no ABI vector register), so
            // no ABI-register check is needed here. A whole-life vector lands in `vector` (register) or
            // `vector_spill` (16-byte slot); a split vector - the shape a vector live across a call
            // takes, since every vector register is caller-saved - lands in `vector_segments` plus one
            // store/reload/move action per boundary, drained by the class-2 arm of `emitSplitAction`.
            if (wsegs.len == 1) {
                switch (wsegs[0].loc) {
                    .reg => |ri| try alloc.vector.put(allocator, value, @enumFromInt(@as(u5, @intCast(ri)))),
                    .slot => |s| try alloc.vector_spill.put(allocator, value, s),
                }
            } else {
                const segs = try allocator.alloc(VectorSegment, wsegs.len);
                for (wsegs, 0..) |ws, i| segs[i] = .{ .from = ws.from, .loc = vectorLocFromWimmer(ws.loc) };
                alloc.vector_segments.put(allocator, value, segs) catch |err| {
                    allocator.free(segs);
                    return err;
                };
                var i: usize = 0;
                while (i + 1 < wsegs.len) : (i += 1) {
                    try alloc.actions.append(allocator, try transitionVectorAction(value, segs[i].loc, segs[i + 1].loc, segs[i + 1].from));
                }
            }
        } else {
            // class 3: et-soc VPU vector (FReg partition f16..f27, 32-byte slots). A VPU vector entry
            // param is rejected above (a vector entry param has no ABI register), so no ABI check is
            // needed. A whole-life value lands in `vpu_vector` (register) or `vpu_vector_spill` (32-byte
            // slot); a split value - the shape a VPU value live across a call takes, since every f16..f27
            // is caller-saved - lands in `vpu_segments` plus one store/reload/move action per boundary,
            // drained by the class-3 arm of `emitSplitAction`. This UN-BAILS the old RV-T4 stub.
            std.debug.assert(class == 3);
            if (wsegs.len == 1) {
                switch (wsegs[0].loc) {
                    .reg => |ri| try alloc.vpu_vector.put(allocator, value, @enumFromInt(@as(u5, @intCast(ri)))),
                    .slot => |s| try alloc.vpu_vector_spill.put(allocator, value, s),
                }
            } else {
                const segs = try allocator.alloc(VpuSegment, wsegs.len);
                for (wsegs, 0..) |ws, i| segs[i] = .{ .from = ws.from, .loc = vpuLocFromWimmer(ws.loc) };
                alloc.vpu_segments.put(allocator, value, segs) catch |err| {
                    allocator.free(segs);
                    return err;
                };
                var i: usize = 0;
                while (i + 1 < wsegs.len) : (i += 1) {
                    try alloc.actions.append(allocator, try transitionVpuAction(value, segs[i].loc, segs[i + 1].loc, segs[i + 1].from));
                }
            }
        }
    }

    // The emission drains a position's actions in a fixed order with no parallel-move resolution, so a
    // same-position register hazard could clobber a live value. Reject rather than risk a miscompile.
    if (hasSamePosRegHazard(alloc.actions.items)) return error.Unsupported;

    // Control-flow-edge moves: translate each ordered `wimmer.Move` into this backend's `EdgeMove`,
    // keyed by (pred, succ). A scalar-float edge move that touches a SLOT is rejected: the width the
    // slot was written with is unknown here (the `Move` carries no value), so an 8-byte round trip
    // could mismatch a 4-byte f32 store and read a non-NaN-boxed value. Register-to-register float
    // moves are safe (fmv.d copies the whole 64-bit register). A class-2 (RVV vector) or class-3
    // (et-soc VPU vector) edge move IS supported, slots included: every such vector this path allows is
    // a fixed width (16-byte <4 x f32> gated by `isRvvWidth`, 32-byte <8 x f32> gated by `isVpuWidth`),
    // so a load/store round trip through its slot always moves exactly the whole vector, no ambiguity.
    var edge_sets: std.ArrayList(EdgeMoveSet) = .empty;
    errdefer {
        for (edge_sets.items) |es| allocator.free(es.moves);
        edge_sets.deinit(allocator);
    }
    for (walloc.edge_moves) |wem| {
        const moves = try allocator.alloc(EdgeMove, wem.moves.len);
        errdefer allocator.free(moves);
        for (wem.moves, 0..) |wm, i| {
            if (wm.class == 1 and (wm.src == .slot or wm.dst == .slot)) return error.Unsupported;
            moves[i] = .{ .class = @intCast(wm.class), .src = edgeLocFromWimmer(wm.src), .dst = edgeLocFromWimmer(wm.dst) };
        }
        try edge_sets.append(allocator, .{ .pred = wem.pred, .succ = wem.succ, .moves = moves });
    }
    alloc.edge_moves = try edge_sets.toOwnedSlice(allocator);
    alloc.edge_move_driven = true;

    alloc.def_pos = def_pos;
    return alloc;
}

/// The precomputed edge-move set for `pred -> succ`, or null when the edge needs no shuffle.
fn findEdgeMoves(alloc: *const Allocation, pred: Block, succ: Block) ?*const EdgeMoveSet {
    for (alloc.edge_moves) |*set| {
        if (set.pred == pred and set.succ == succ) return set;
    }
    return null;
}

/// Replay the precomputed, already-ordered edge moves for `pred -> succ` op-by-op (the Wimmer path).
/// The shared allocator resolved the parallel move (sources read before overwrite, cycles broken and
/// any slot<->slot shuffle routed through the class scratch), so each move is a primitive reg/slot op.
fn emitEdgeMoves(allocator: std.mem.Allocator, code: *std.ArrayList(u32), alloc: *const Allocation, spill_base: u32, float_spill_base: u32, vspill_base: u32, vpu_vspill_base: u32, vpu_pack_base: u32, pred: Block, succ: Block) std.mem.Allocator.Error!void {
    _ = float_spill_base; // float slot edge moves are rejected in translation, so only int slots exist
    const set = findEdgeMoves(alloc, pred, succ) orelse return;
    for (set.moves) |m| try emitOneEdgeMove(allocator, code, spill_base, vspill_base, vpu_vspill_base, vpu_pack_base, m);
}

/// Emit one ordered edge move. Class 0 (int): reg->reg `mv`, reg->slot `sd`, slot->reg `ld` (8-byte
/// slots). Class 1 (float): reg->reg `fmv.d` (a whole 64-bit copy, correct for both f32 NaN-box and
/// f64); a slot-resident float move never reaches here (the translation rejects it). Class 2 (RVV
/// vector): reg->reg `vmv.v.v`, reg->slot `vse32`, slot->reg `vle32` (16-byte slots, the fixed
/// <4 x f32> width), the slot address computed into `spill_scratch1`. Class 3 (et-soc VPU vector):
/// reg->reg a `vpu_pack_base` round trip (no packed move op), reg->slot `fsw.ps`, slot->reg `flw.ps`
/// (32-byte slots, the fixed <8 x f32> width, addressed by the op's own displacement off `sp`). A
/// slot->slot op never appears (the shared ordering expanded it through the class scratch), so it is
/// unreachable.
fn emitOneEdgeMove(allocator: std.mem.Allocator, code: *std.ArrayList(u32), spill_base: u32, vspill_base: u32, vpu_vspill_base: u32, vpu_pack_base: u32, m: EdgeMove) std.mem.Allocator.Error!void {
    switch (m.class) {
        0 => switch (m.src) {
            .reg => |si| {
                const sr: Reg = @enumFromInt(@as(u5, @intCast(si)));
                switch (m.dst) {
                    .reg => |di| {
                        const dr: Reg = @enumFromInt(@as(u5, @intCast(di)));
                        if (sr != dr) try code.append(allocator, encode.addi(dr, sr, 0));
                    },
                    .slot => |ds| try code.append(allocator, encode.sd(sr, .x2, @intCast(spill_base + ds * 8))),
                }
            },
            .slot => |ss| switch (m.dst) {
                .reg => |di| {
                    const dr: Reg = @enumFromInt(@as(u5, @intCast(di)));
                    try code.append(allocator, encode.ld(dr, .x2, @intCast(spill_base + ss * 8)));
                },
                .slot => unreachable, // slot->slot was expanded through the class scratch
            },
        },
        1 => switch (m.src) {
            .reg => |si| {
                const sr: FReg = @enumFromInt(@as(u5, @intCast(si)));
                switch (m.dst) {
                    .reg => |di| {
                        const dr: FReg = @enumFromInt(@as(u5, @intCast(di)));
                        if (sr != dr) try code.append(allocator, encode.fmv_d(dr, sr));
                    },
                    .slot => unreachable, // a slot-resident float edge move is rejected in translation
                }
            },
            .slot => unreachable, // ditto
        },
        2 => switch (m.src) {
            .reg => |si| {
                const sr: VReg = @enumFromInt(@as(u5, @intCast(si)));
                switch (m.dst) {
                    .reg => |di| {
                        const dr: VReg = @enumFromInt(@as(u5, @intCast(di)));
                        if (sr != dr) try code.append(allocator, encode.vmv_v_v(dr, sr));
                    },
                    .slot => |ds| {
                        try code.append(allocator, encode.addi(spill_scratch1, .x2, @intCast(vspill_base + ds * 16)));
                        try code.append(allocator, encode.vse32(sr, spill_scratch1));
                    },
                }
            },
            .slot => |ss| switch (m.dst) {
                .reg => |di| {
                    const dr: VReg = @enumFromInt(@as(u5, @intCast(di)));
                    try code.append(allocator, encode.addi(spill_scratch1, .x2, @intCast(vspill_base + ss * 16)));
                    try code.append(allocator, encode.vle32(dr, spill_scratch1));
                },
                .slot => unreachable, // slot->slot was expanded through the class scratch
            },
        },
        3 => switch (m.src) {
            .reg => |si| {
                const sr: FReg = @enumFromInt(@as(u5, @intCast(si)));
                switch (m.dst) {
                    .reg => |di| {
                        const dr: FReg = @enumFromInt(@as(u5, @intCast(di)));
                        // No packed VPU register move: round-trip through the reserved 32-byte pack slot.
                        if (sr != dr) {
                            try code.append(allocator, encode.fsw_ps(sr, .x2, @intCast(vpu_pack_base)));
                            try code.append(allocator, encode.flw_ps(dr, .x2, @intCast(vpu_pack_base)));
                        }
                    },
                    .slot => |ds| try code.append(allocator, encode.fsw_ps(sr, .x2, @intCast(vpu_vspill_base + ds * 32))),
                }
            },
            .slot => |ss| switch (m.dst) {
                .reg => |di| {
                    const dr: FReg = @enumFromInt(@as(u5, @intCast(di)));
                    try code.append(allocator, encode.flw_ps(dr, .x2, @intCast(vpu_vspill_base + ss * 32)));
                },
                .slot => unreachable, // slot->slot was expanded through the class scratch
            },
        },
        else => unreachable,
    }
}

/// TEST-ONLY: compile `func` through the SHARED Wimmer-Franz allocator instead of the backend's own
/// `allocateRegisters`, then emit through the SAME battle-tested `emitFromAllocation`. Covers all four
/// classes: INT, scalar-FLOAT, RVV VECTOR, and (when `vpu`) the et-soc VPU VECTOR class. A vector live
/// across a call now spills/splits (every RVV register, and every f16..f27 VPU register, is
/// caller-saved) instead of bailing. An f16 function, a vector of a width other than the fixed 4-lane
/// RVV group or 8-lane VPU width, a (VPU or RVV) vector ENTRY param (no ABI vector register), a
/// split/spilled entry param, a same-position action hazard, an unreachable block, or an if-edge move
/// all bail `error.Unsupported` (never a silent miscompile). `vpu` selects the et-soc VPU register
/// model (class 3, narrowed scalar-float pool) over the RVV one (class 2). Splits critical edges up
/// front (mutating `func`, so a differential caller keeps a separate reference), then runs the shared
/// scan and translates its target-independent `Allocation` into this backend's. The default
/// `compileFunction` is untouched.
pub fn compileFunctionWimmerRiscv(allocator: std.mem.Allocator, func: *Function, vpu: bool) Error!Compiled {
    if (func.blockCount() == 0) return error.Unsupported;
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;
    // The software f16 convert path reserves integer scratch (x28..x31) the shared pools do not model
    // (see `riscv64RegDescription`'s scope note), so bail rather than miscompile a half function.
    if (ir.function.functionUsesF16(func)) return error.Unsupported;

    // Split edges FIRST (mutating `func`), before any numbering is built, so the resolver's
    // no-critical-edge precondition holds and the description/scan/emission all see one CFG. TWO
    // passes: `ir.critical_edge` splits every genuinely critical edge (giving the resolver a block for
    // its shuffle), then this backend's own `splitCriticalEdges` splits every remaining if-edge that
    // still carries ARGS into a jump landing block. The riscv64 `.@"if"` emission only branches (it
    // cannot host an edge move, unlike aarch64's `emitIf`), so every block-param move must land on a
    // JUMP edge; the second pass guarantees that, exactly as the native riscv64 path already does.
    try ir.critical_edge.splitCriticalEdges(allocator, func);
    try splitCriticalEdges(allocator, func);

    // The shared numbering covers EVERY block, but `emitFromAllocation` skips unreachable ones without
    // advancing its position counter. To keep the two numberings in lockstep, require all-reachable.
    var doms = try dominators.compute(allocator, func);
    defer doms.deinit(allocator);
    for (doms.reachable) |r| {
        if (!r) return error.Unsupported;
    }

    var desc = try riscv64RegDescription(allocator, func, vpu);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, func, &desc);
    defer walloc.deinit(allocator);

    var alloc = try translateAllocation(allocator, func, vpu, &walloc);
    defer alloc.deinit(allocator);

    // An edge move on an if-edge cannot be realized by the `.@"if"` emission (it only branches), so
    // reject rather than drop it. Register phis land on jump edges, which the `.jump` path replays.
    if (edgeMoveOnIfEdge(func, &alloc)) return error.Unsupported;

    const caps: ModelCaps = .{ .vpu = vpu };
    // The Wimmer path does NOT fold addresses: the shared `wimmer.zig` liveness is fold-unaware, so
    // feeding a real analysis would desync its intervals from emission. Pass the no-fold analysis so
    // `baseOf`/`offOf`/`isDeadAdd` behave exactly as before folding existed (byte-identical). Note
    // `compileFunctionWimmerRiscv` never calls `allocateRegisters` (it uses the shared allocator via
    // `wimmer.allocate` + `translateAllocation`), so no fold reaches allocation here either.
    return emitFromAllocation(allocator, func, caps, false, doms.reachable, &alloc, &empty_fold);
}

/// Assign a register to every value via a liveness-based linear scan over the
/// register files. Entry parameters are pre-colored to argument registers. Other
/// values draw from a temporary free list, reusing a register once its value
/// dies. A value live across a call is placed in a callee-saved register so the
/// call cannot clobber it.
/// `reachable[bi]` is whether block `bi` is reachable from the entry (block 0). Unreachable blocks
/// contribute NO live intervals and draw NO registers, so a dead block (e.g. an orphaned loop nest
/// left by an optimization) cannot inflate register pressure. When every block is reachable (the
/// common case, and every case before reachability-aware isel existed) every guard below is a no-op
/// and the linear position numbering is exactly what it was, so allocation is byte-identical.
fn allocateRegisters(allocator: std.mem.Allocator, func: *const Function, vpu: bool, uses_f16: bool, reachable: []const bool, fold: *const addrfold.Analysis) Error!Allocation {
    var alloc: Allocation = .{
        .int = .empty,
        .float = .empty,
        .vector = .empty,
        .vector_spill = .empty,
        .vector_spill_count = 0,
        .vpu_vector = .empty,
        .vpu_vector_spill = .empty,
        .vpu_vector_spill_count = 0,
        .int_spill = .empty,
        .spill_count = 0,
        .float_spill = .empty,
        .float_spill_count = 0,
        .incoming_stack = .empty,
    };
    errdefer alloc.deinit(allocator);

    // Liveness: the last position at which each value is used (walking forward,
    // a later use overwrites an earlier one).
    var last_use: std.AutoHashMapUnmanaged(Value, usize) = .empty;
    defer last_use.deinit(allocator);

    // Each value's ascending operand-use positions, indexed by `@intFromEnum(value)`. The
    // integer-eviction path asks, at an exhaustion point, which active value's NEXT use is furthest
    // away (Belady/MIN). Built alongside `last_use` in the same numbering loop, with the same `pos`,
    // so the lists stay ascending and `nextUseAfter` can binary-search them.
    const nval = func.valueCount();
    const use_lists = try allocator.alloc(std.ArrayList(usize), nval);
    defer allocator.free(use_lists);
    for (use_lists) |*u| u.* = .empty;
    defer for (use_lists) |*u| u.deinit(allocator);

    // Each value's definition position, indexed by `@intFromEnum(value)`, in the same linear
    // numbering built below. A block param is defined at its block's param-row position, an
    // instruction result at its instruction position. Duped into `alloc.def_pos` after the pass so
    // it outlives this function-local scratch; the emission pos-coupling assert reads it.
    const def_pos_local = try allocator.alloc(usize, nval);
    defer allocator.free(def_pos_local);
    @memset(def_pos_local, 0);

    // Which block defines each value, and whether the value is intra-block-splittable. `def_block` is
    // scratch used only to decide `is_intra`. A value starts splittable, then loses it (below) if it
    // is a block param, is used as an edge argument, or is used outside its def block. Only intra
    // values ever tail-split under pressure (cross-block values still whole-spill), so `is_intra`
    // gates `placeIntUnderPressure`.
    const def_block = try allocator.alloc(usize, nval);
    defer allocator.free(def_block);
    @memset(def_block, 0);
    const is_intra = try allocator.alloc(bool, nval);
    defer allocator.free(is_intra);
    @memset(is_intra, true);

    // Values that eviction must never displace from their register. Entry integer parameters home to
    // ABI argument registers (or callee-saved registers / stack slots for the call-crossing and 9th+
    // cases) and the prologue in `compileFunction` assumes each one resides in `alloc.int`. Evicting
    // one to a spill slot would break that invariant, so they are pinned out of the Belady victim pool.
    // Only entry params are pinned: non-entry block params and instruction results already have a
    // whole-interval spill path emission handles, so eviction may freely choose them.
    const no_evict = try allocator.alloc(bool, nval);
    defer allocator.free(no_evict);
    @memset(no_evict, false);
    if (func.blockCount() != 0) {
        for (func.blockParams(@enumFromInt(0))) |p| no_evict[@intFromEnum(p)] = true;
    }

    // Mark which positions hold a call, to identify values live across one.
    var is_call: std.ArrayList(bool) = .empty;
    defer is_call.deinit(allocator);

    // Each reachable block's terminator position, in the same linear numbering as `last_use`, so
    // `extendLiveRanges` can raise a live-out value's last use to its containing block's end.
    // Unreachable blocks are skipped in the numbering below, so their entry stays 0 (they carry no
    // live interval, and `extendLiveRanges` skips them too).
    const block_end = try allocator.alloc(usize, func.blockCount());
    defer allocator.free(block_end);
    @memset(block_end, 0);

    var total: usize = 0;
    var pos: usize = 0;
    for (0..func.blockCount()) |bi| {
        // Number and record uses over ONLY the reachable instruction stream, so an unreachable
        // block's params/values get no live interval (and later no register). Skipping here keeps
        // `pos` contiguous over exactly the positions the allocation pass below re-walks, so the
        // two numberings stay in lockstep. When all blocks are reachable this never fires.
        if (!reachable[bi]) continue;
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| {
            try last_use.put(allocator, p, pos);
            def_pos_local[@intFromEnum(p)] = pos; // a block param is defined at its param-row position
            def_block[@intFromEnum(p)] = bi;
            is_intra[@intFromEnum(p)] = false; // a block param is never intra-splittable
        }
        try is_call.append(allocator, false);
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            try recordUses(allocator, func, inst, pos, &last_use, fold);
            try recordUsePositions(allocator, func, inst, pos, use_lists, fold);
            markIntraUses(func, inst, bi, def_block, is_intra, fold);
            if (func.instResult(inst)) |r| {
                def_pos_local[@intFromEnum(r)] = pos; // an instruction result is defined at its position
                def_block[@intFromEnum(r)] = bi;
                if (!last_use.contains(r)) try last_use.put(allocator, r, pos);
            }
            try is_call.append(allocator, func.opcode(inst) == .call);
            pos += 1;
        }
        block_end[bi] = pos; // the terminator's position, in the linear `pos` numbering
        try recordTermUses(allocator, func, block, pos, &last_use);
        try recordUsePositionsTerm(allocator, func, block, pos, use_lists);
        markIntraTermUses(func, block, bi, def_block, is_intra);
        try is_call.append(allocator, false);
        pos += 1;
    }
    total = pos;

    // Extend live ranges across loop back-edges. The forward pass above records each value's last
    // TEXTUAL use, which under-covers a value still live across a back-edge (its textual last use
    // precedes the edge that re-enters the body). This raises `last_use` to the end of every block
    // where the value is live-out, so a loop-carried value keeps its register across the whole body.
    // It ONLY ever raises a last_use, so for forward-dominated code (where the forward scan already
    // holds the maximal use) it is a no-op and allocation stays byte-identical. See the function.
    try extendLiveRanges(allocator, func, &last_use, block_end, reachable, fold);

    // call_prefix[p] = number of call positions strictly before p.
    const call_prefix = try allocator.alloc(usize, total + 1);
    defer allocator.free(call_prefix);
    call_prefix[0] = 0;
    for (0..total) |p| call_prefix[p + 1] = call_prefix[p] + @intFromBool(is_call.items[p]);
    // dying[pos] = the values whose last use is at pos. A value crosses a call via the file-scope
    // `crossesCall(call_prefix, def, last_use)`.
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
    // When the function uses f16, x28..x31 are reserved as software-convert scratch, so the
    // allocatable caller-saved temps shrink to x5/x7 (see `temp_regs_f16`). Otherwise the full pool
    // is used and allocation is byte-identical to before f16 support.
    const int_temps: []const Reg = if (uses_f16) &temp_regs_f16 else &temp_regs;
    var ik: usize = int_temps.len;
    while (ik > 0) {
        ik -= 1;
        try int_free.append(allocator, int_temps[ik]);
    }
    var float_free: std.ArrayList(FReg) = .empty;
    defer float_free.deinit(allocator);
    if (vpu) {
        // vpu mode: scalar floats only ever draw from f0..f7, the disjoint slice of the file that
        // never overlaps the VPU vector partition (f16..f31) nor the vpu float spill scratches
        // (f8/f9). `float_saved_regs_vpu` is empty, so there is no callee-saved slice to push.
        var fkv: usize = float_temp_regs_vpu.len;
        while (fkv > 0) {
            fkv -= 1;
            try float_free.append(allocator, float_temp_regs_vpu[fkv]);
        }
    } else {
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
    }
    var vector_free: std.ArrayList(VReg) = .empty;
    defer vector_free.deinit(allocator);
    var vk: usize = vector_regs.len;
    while (vk > 0) {
        vk -= 1;
        try vector_free.append(allocator, vector_regs[vk]); // pop yields v1 first
    }
    var vpu_vector_free: std.ArrayList(FReg) = .empty;
    defer vpu_vector_free.deinit(allocator);
    var pvk: usize = vpu_vector_regs.len;
    while (pvk > 0) {
        pvk -= 1;
        try vpu_vector_free.append(allocator, vpu_vector_regs[pvk]); // pop yields f16 first
    }

    pos = 0;
    var int_arg: usize = 0;
    var float_arg: usize = 0;
    for (0..func.blockCount()) |bi| {
        // Skip unreachable blocks in lockstep with the numbering pass above: `pos` advances only
        // over the same reachable positions, so every `last_use`/`is_call`/`dying` lookup lines up.
        // An unreachable block therefore consumes no argument slots and draws no registers.
        if (!reachable[bi]) continue;
        const block: Block = @enumFromInt(bi);
        for (func.blockParams(block)) |p| {
            const cross = crossesCall(call_prefix, pos, last_use.get(p).?);
            if (isVector(func, func.valueType(p))) {
                if (bi == 0) return error.Unsupported; // an entry vector parameter has no ABI register
                if (cross) return error.Unsupported; // vector live across a call
                if (vpu) {
                    if (!isVpuWidth(func, func.valueType(p))) return error.Unsupported; // VPU is fixed 8-lane
                    if (vpu_vector_free.pop()) |fr| {
                        try alloc.vpu_vector.put(allocator, p, fr);
                    } else {
                        try alloc.vpu_vector_spill.put(allocator, p, alloc.vpu_vector_spill_count);
                        alloc.vpu_vector_spill_count += 1;
                    }
                } else {
                    if (!isRvvWidth(func, func.valueType(p))) return error.Unsupported; // RVV here is a fixed 4-lane group
                    if (vector_free.pop()) |vr| {
                        try alloc.vector.put(allocator, p, vr);
                    } else {
                        try alloc.vector_spill.put(allocator, p, alloc.vector_spill_count);
                        alloc.vector_spill_count += 1;
                    }
                }
                continue;
            }
            if (isFloat(func, func.valueType(p))) {
                if (bi == 0) {
                    // Entry params arrive in arg registers. One that outlives a call is homed to a
                    // callee-saved register (the selector moves it there at entry), or - when the
                    // callee-saved float file is exhausted - spilled to a stack slot (the selector
                    // stores the incoming arg register into it at entry). The ABI arg slot is still
                    // consumed either way.
                    if (vpu and float_arg >= 6) return error.Unsupported; // fa6/fa7 (f16/f17) sit inside the vpu vector partition
                    if (cross) {
                        if (if (vpu) popFloatSavedVpu(&float_free) else popFloatSaved(&float_free)) |reg| {
                            try alloc.float.put(allocator, p, reg);
                        } else {
                            try alloc.float_spill.put(allocator, p, alloc.float_spill_count);
                            alloc.float_spill_count += 1;
                        }
                    } else {
                        try alloc.float.put(allocator, p, fargReg(float_arg));
                    }
                    float_arg += 1;
                } else {
                    const reg = if (cross) (if (vpu) popFloatSavedVpu(&float_free) else popFloatSaved(&float_free)) else float_free.pop();
                    if (reg) |rr| {
                        try alloc.float.put(allocator, p, rr);
                    } else {
                        // The scalar float file is exhausted: spill to a stack slot.
                        try alloc.float_spill.put(allocator, p, alloc.float_spill_count);
                        alloc.float_spill_count += 1;
                    }
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
                    // Non-entry integer block param. When the integer file is exhausted, evict the
                    // Belady victim (the active int value whose next use is furthest) and give its
                    // register to this param, unless the param's own next use is furthest (then it
                    // spills to a stack slot as before). Whichever loses is whole-spilled: the
                    // block-edge parallel move at the jump stores the incoming arg into its slot, and
                    // body reads reload from it via `reloadInt`. Entry int params (above) are NOT
                    // spilled: they arrive in ABI arg registers and the prologue assumes their
                    // residency, so their exhaustion keeps returning error.Unsupported.
                    const reg = if (cross) popSaved(&int_free) else int_free.pop();
                    if (reg) |rr| {
                        try alloc.int.put(allocator, p, rr);
                    } else {
                        try placeIntUnderPressure(allocator, &alloc, use_lists, &last_use, no_evict, is_intra, def_pos_local, p, pos, cross);
                    }
                }
            }
        }
        try freeDying(allocator, func, dying[pos].items, &alloc, &int_free, &float_free, &vector_free, &vpu_vector_free, vpu);
        try secondChance(allocator, &alloc, use_lists, &last_use, &int_free, call_prefix, pos);
        pos += 1;
        for (func.blockInsts(block)) |inst| {
            // A folded dead address-add claims NO register: every use of its result was rerouted to the
            // fold base in the four liveness scans, so it has no real interval. Skipping the assignment
            // (it still dies at its own def position, so `freeDying` below is a no-op for it) keeps a
            // dead add from drawing a register or, under pressure, evicting a live value for a range it
            // never really has. `pos` still advances, in lockstep with emission's dead-add skip.
            if (fold.isDeadAdd(inst)) {
                try freeDying(allocator, func, dying[pos].items, &alloc, &int_free, &float_free, &vector_free, &vpu_vector_free, vpu);
                try secondChance(allocator, &alloc, use_lists, &last_use, &int_free, call_prefix, pos);
                pos += 1;
                continue;
            }
            if (func.instResult(inst)) |r| {
                const cross = crossesCall(call_prefix, pos, last_use.get(r).?);
                if (isVector(func, func.valueType(r))) {
                    if (cross) return error.Unsupported; // a vector live across a call (all vregs are caller-saved)
                    if (vpu) {
                        if (!isVpuWidth(func, func.valueType(r))) return error.Unsupported; // VPU is fixed 8-lane
                        if (vpu_vector_free.pop()) |fr| {
                            try alloc.vpu_vector.put(allocator, r, fr);
                        } else {
                            try alloc.vpu_vector_spill.put(allocator, r, alloc.vpu_vector_spill_count); // pressure: spill to a 32-byte slot
                            alloc.vpu_vector_spill_count += 1;
                        }
                    } else {
                        if (!isRvvWidth(func, func.valueType(r))) return error.Unsupported; // RVV here is a fixed 4-lane group
                        if (vector_free.pop()) |vr| {
                            try alloc.vector.put(allocator, r, vr);
                        } else {
                            try alloc.vector_spill.put(allocator, r, alloc.vector_spill_count); // pressure: spill to a 16-byte slot
                            alloc.vector_spill_count += 1;
                        }
                    }
                } else if (isFloat(func, func.valueType(r))) {
                    const reg = if (cross) (if (vpu) popFloatSavedVpu(&float_free) else popFloatSaved(&float_free)) else float_free.pop();
                    if (reg) |rr| {
                        try alloc.float.put(allocator, r, rr);
                    } else {
                        // The scalar float file is exhausted: spill to a stack slot.
                        try alloc.float_spill.put(allocator, r, alloc.float_spill_count);
                        alloc.float_spill_count += 1;
                    }
                } else {
                    const reg = if (cross) popSaved(&int_free) else int_free.pop();
                    if (reg) |rr| {
                        try alloc.int.put(allocator, r, rr);
                    } else {
                        // The integer file is exhausted: evict the Belady victim (furthest next use)
                        // and give its register to this result, unless the result's own next use is
                        // furthest, in which case it spills to a stack slot as before.
                        try placeIntUnderPressure(allocator, &alloc, use_lists, &last_use, no_evict, is_intra, def_pos_local, r, pos, cross);
                    }
                }
            }
            try freeDying(allocator, func, dying[pos].items, &alloc, &int_free, &float_free, &vector_free, &vpu_vector_free, vpu);
            try secondChance(allocator, &alloc, use_lists, &last_use, &int_free, call_prefix, pos);
            pos += 1;
        }
        try freeDying(allocator, func, dying[pos].items, &alloc, &int_free, &float_free, &vector_free, &vpu_vector_free, vpu);
        try secondChance(allocator, &alloc, use_lists, &last_use, &int_free, call_prefix, pos);
        pos += 1;
    }

    // Hand the per-value definition positions to the emission pass (heap-owned, so it outlives the
    // function-local scratch). `segments` stays empty here: this pass produces no splits, so
    // `intLocationAt` falls back to `int`/`int_spill` and emission is byte-identical to before.
    alloc.def_pos = try allocator.dupe(usize, def_pos_local);

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

/// Capabilities a model-aware call site threads into `compileFunction`. Grouped into one struct
/// (rather than growing `compileFunction`'s parameter list one flag per model feature) so adding
/// the next capability never touches every existing call site. Every field defaults off (except
/// `fuse_cmp_branch`, always available with no extension), so `.{}` is exactly today's behavior
/// for every non-model caller (`selectFunction`, `selectFunctionWithLines`, and the direct
/// `compileFunction` callers in `link.zig`/`object.zig`): no loop-header alignment padding, RVV
/// (not VPU) vector lowering, a dropped `.prefetch` hint, and the `fuse_*` flags at their
/// no-extension-required defaults. `fuse_cmp_branch` gates the compare-into-branch fold (see
/// `fusesIntoNextIf`); the rest are foundation only (no fold reads them yet, so they are inert
/// either way).
pub const ModelCaps = struct {
    /// Loop-header alignment in bytes (0 disables it). See `compileFunction`'s doc comment.
    fetch_align: u16 = 0,
    /// Lower vectorized f32 arithmetic to the et-soc CORE-ET VPU instead of RVV.
    vpu: bool = false,
    /// Lower the IR `.prefetch` hint to a real Zicbop `prefetch.r` instead of dropping it.
    /// Set only when the target model's `features.riscv64.zicbop` is true (see
    /// `selectFunctionForModel`); `Model.prefetches()` gates whether the insertion pass ever
    /// produces a hint to lower in the first place.
    zicbop: bool = false,
    /// Lower f16 NATIVELY via the Zfh half-precision instructions (an f16 held natively in a float
    /// register) instead of the default software emulation (an f16 held as its f32 widening with a
    /// per-boundary inline convert). Set only when the target model's `features.riscv64.zfh` is
    /// true (see `selectFunctionForModel`). Every non-model caller passes `.{}` (zfh = false), so
    /// the emulation path is unchanged and byte-identical.
    zfh: bool = false,
    /// Fuse a compare into its consumer branch. riscv64's `beq`/`bne`/`blt`/... already compare-
    /// and-branch in one instruction with no extension, so true by default. Gates
    /// `fusesIntoNextIf` (see its doc comment); false falls back to materializing the boolean
    /// with `slt`/`sltu` then testing it with `bne`, exactly as if the icmp/if pair were never
    /// eligible to fuse.
    fuse_cmp_branch: bool = true,
    /// Fuse an arithmetic op's result-setting form into its consumer branch. riscv64 has no
    /// flags register (unlike aarch64), so this fold has no riscv64 instruction to target: off
    /// unconditionally, even if a model mistakenly declared it.
    fuse_arith_branch: bool = false,
    /// Fuse a shift into a following add (sh1add/sh2add/sh3add). Needs the Zba extension, so off
    /// unless the model both declares the fusion and sets `features.riscv64.zba` (see
    /// `selectFunctionForModel`). Gates `fusesIntoNextShiftAdd`; false leaves the plain slli-then-add
    /// path, so a non-Zba compile is byte-identical.
    fuse_shift_add: bool = false,
    /// Fuse a high/low address-pair computation (auipc+addi) into one microarch-recognized
    /// macro-op. Microarch-specific, so off unless the model declares it. The `.global_addr` arm
    /// already emits the `auipc`/`addi` pair back-to-back by construction (there is no separate
    /// transform to gate), so this flag's only reader is an adjacency `std.debug.assert` in that
    /// arm: a forward regression guard proving the invariant the macro-op fusion depends on holds,
    /// not a byte-changing fold. False (or true) never changes emission.
    fuse_addr_hi_lo: bool = false,
};

/// Test-only: allocate `func` and report how many integer values were live-range split (their tail
/// moved to a stack slot). Zero unless intra-block tail-splitting fired. The int-spill differential
/// tests call this to assert a real split occurred before checking execution equivalence on qemu.
pub fn splitCountForTest(allocator: std.mem.Allocator, func: *const Function) Error!usize {
    var doms = try dominators.compute(allocator, func);
    defer doms.deinit(allocator);
    var fold = try addrfold.analyze(allocator, func, {}, riscv64FoldOffset);
    defer fold.deinit(allocator);
    var alloc = try allocateRegisters(allocator, func, false, false, doms.reachable, &fold);
    defer alloc.deinit(allocator);
    return alloc.segments.count();
}

/// Test hook: count the values that were SECOND-CHANCE RE-HOMED (a `.reg` segment following a
/// `.slot` segment). A tail-split alone produces `.reg` then `.slot`, so only a re-home puts a `.reg`
/// after a `.slot`. Zero before Task 6d's `secondChance` runs, positive once it re-homes a value.
pub fn debugReHomeCount(allocator: std.mem.Allocator, func: *const Function) Error!u32 {
    var doms = try dominators.compute(allocator, func);
    defer doms.deinit(allocator);
    var fold = try addrfold.analyze(allocator, func, {}, riscv64FoldOffset);
    defer fold.deinit(allocator);
    var alloc = try allocateRegisters(allocator, func, false, false, doms.reachable, &fold);
    defer alloc.deinit(allocator);
    var count: u32 = 0;
    var it = alloc.segments.valueIterator();
    while (it.next()) |segs| {
        var saw_slot = false;
        for (segs.*) |s| switch (s.loc) {
            .slot => saw_slot = true,
            .reg => if (saw_slot) {
                count += 1;
            },
        };
    }
    return count;
}

/// Test hook: how many times `secondChance` DECLINED to re-home a slot-resident value because no
/// integer register was free at that point (the value reloads per use instead). Positive whenever the
/// allocation drove the register file hard enough that a re-home candidate had to wait.
pub fn debugDeclineCount(allocator: std.mem.Allocator, func: *const Function) Error!u32 {
    var doms = try dominators.compute(allocator, func);
    defer doms.deinit(allocator);
    var fold = try addrfold.analyze(allocator, func, {}, riscv64FoldOffset);
    defer fold.deinit(allocator);
    var alloc = try allocateRegisters(allocator, func, false, false, doms.reachable, &fold);
    defer alloc.deinit(allocator);
    return alloc.rehome_declines;
}

/// Select RISC-V machine words for a function (wrapper over `compileFunction`
/// that drops the relocations). The caller owns the returned slice.
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
/// (loop-header alignment), `vpu()` (whether to lower vectorized f32 arithmetic to the CORE-ET
/// VPU packed-single unit instead of RVV; only et-soc sets this), and `features.riscv64.zicbop`
/// (whether to lower the IR `.prefetch` hint to a real Zicbop `prefetch.r` instead of dropping
/// it; only river-rc1.f/.ma set this, see registry.zig), and `fuse_cmp_branch` (whether the
/// compare-into-branch fold runs). An inert model (fetch_align 0, vpu false, zicbop false,
/// fuse_cmp_branch true - the no-extension-required default) makes this byte-identical to
/// `selectFunction`. Builds the full `ModelCaps` and calls `compileFunction` directly rather than
/// through `selectFunctionAligned`, since that narrower entry point only ever carries
/// `fetch_align`.
pub fn selectFunctionForModel(allocator: std.mem.Allocator, func: *const Function, model: *const mm.Model) Error![]u32 {
    // Passing a foreign-arch model here is a caller bug, not a runtime fault.
    std.debug.assert(model.arch == .riscv64);
    const compiled = try compileFunction(allocator, func, capsForModel(model));
    allocator.free(compiled.relocs);
    allocator.free(compiled.lines);
    return compiled.code;
}

/// The `ModelCaps` `selectFunctionForModel` builds for `model`. Split out so the model-to-caps
/// mapping is unit-testable without compiling a whole function. Asserts `model.arch == .riscv64`,
/// same as the caller above.
pub fn capsForModel(model: *const mm.Model) ModelCaps {
    std.debug.assert(model.arch == .riscv64);
    return .{
        .fetch_align = model.fetch_align,
        .vpu = model.vpu(),
        .zicbop = model.arch == .riscv64 and model.features.riscv64.zicbop,
        .zfh = model.arch == .riscv64 and model.features.riscv64.zfh,
        .fuse_cmp_branch = model.fuses(.cmp_branch),
        // riscv64 has no flags register, so no fold will ever target a fused arith-branch here:
        // keep this false unconditionally even if a model wrongly declared the fusion.
        .fuse_arith_branch = false,
        // Belt-and-suspenders with model.zig's validate rule: shift_add needs Zba.
        .fuse_shift_add = model.fuses(.shift_add) and model.features.riscv64.zba,
        .fuse_addr_hi_lo = model.fuses(.addr_hi_lo),
    };
}

test "capsForModel reads river-rc1.ma's fusion table: cmp/shift/addr_hi_lo on, arith off" {
    const caps = capsForModel(mm.modelFor(.@"river-rc1.ma"));
    try std.testing.expect(caps.fuse_cmp_branch);
    try std.testing.expect(!caps.fuse_arith_branch);
    try std.testing.expect(caps.fuse_shift_add);
    try std.testing.expect(caps.fuse_addr_hi_lo);
}

test "capsForModel withholds shift_add for et-soc: cmp on, shift/addr off (no Zba)" {
    const caps = capsForModel(mm.modelFor(.@"et-soc"));
    try std.testing.expect(caps.fuse_cmp_branch);
    try std.testing.expect(!caps.fuse_arith_branch);
    try std.testing.expect(!caps.fuse_shift_add);
    try std.testing.expect(!caps.fuse_addr_hi_lo);
}

/// Compiled code plus its source-line table (from the `debug.line` IR attributes), for DWARF.
pub const CodeWithLines = struct { code: []u32, lines: []LineEntry };

/// Like `selectFunction`, but also returns the source-line table. Caller owns both slices.
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

/// Number of nop words to insert so a block starting at `words` (current code length in
/// 4-byte words) lands on a `fetch_align`-byte boundary. Zero when fetch_align is at most
/// one word (already aligned) or the block is already on a boundary.
fn alignPadWords(words: usize, fetch_align: u16) usize {
    if (fetch_align <= 4) return 0;
    const per: usize = fetch_align / 4; // words per alignment boundary
    const rem = words % per;
    return if (rem == 0) 0 else per - rem;
}

/// Compile a function to machine words and call relocations. `fetch_align` is the microarch
/// model's fetch granularity in bytes (0 disables loop-header alignment, the behavior of every
/// existing caller). When greater than one instruction word, each loop-header block is padded
/// with nops up to a `fetch_align` boundary before its code is emitted, so a hot loop's fetch
/// groups pack efficiently. This is purely a placement hint: the padding falls straight through
/// into the header and every branch fixup is patched from `block_start` (recorded after padding),
/// so it can never change what the function computes. Note: the riscv64 pipeline may later
/// compress instructions (RVC), which shifts byte offsets and makes this alignment approximate
/// rather than exact, but still never incorrect.
///
/// `caps.vpu` selects the et-soc CORE-ET packed-single VPU lowering for vectorized f32 arithmetic
/// (8-lane, `f16..f31` disjoint from the vpu-mode scalar float pool) instead of the default RVV
/// lowering (4-lane, `v1..v27`). False (the RVV path, the behavior of every existing caller) is
/// byte-identical to before this parameter existed; only a caller that explicitly asks for `vpu`
/// (today, only `selectFunctionForModel` under an et-soc model) reaches the new path. The VPU
/// path is encoding-validated against the CORE-ET RTL masks (see encode.zig) and IR-verified, but
/// unlike RVV it is never executed here: no emulator decodes these custom opcodes.
///
/// `caps.zicbop` lowers the IR `.prefetch` hint to a real Zicbop `prefetch.r` instead of dropping
/// it (see the `.prefetch` case below). False (drop the hint, the behavior of every existing
/// caller) is byte-identical to before this capability existed; only `selectFunctionForModel`
/// under a model with `features.riscv64.zicbop` set reaches the new path. `prefetch.r` is
/// ORI-shaped (see encode.zig), so unlike the VPU path this one IS execution-validated: it
/// decodes as a harmless no-op on any qemu-riscv64 host, Zicbop or not.
pub fn compileFunction(allocator: std.mem.Allocator, func: *const Function, caps: ModelCaps) Error!Compiled {
    // f16 lowering has two modes (see `ModelCaps.zfh`). SOFTWARE EMULATION (no Zfh, the default):
    // an f16 is held as its f32 widening in a float register and every boundary rounds via the
    // inline convert routines (`emitHalfToFloat`/`emitFloatToHalf`); those routines need dedicated
    // scratch GPRs, so the allocator reserves x28..x31 out of the integer temp pool (see
    // `temp_regs_f16`). NATIVE (Zfh): an f16 is held natively in a float register and every op is a
    // real half instruction, so no integer scratch is needed and the reservation is skipped, keeping
    // native allocation closer to a normal function. Only the software path drives the reservation,
    // so `reserve_f16_scratch` gates on `!zfh`. A non-f16 function (or a native one) keeps the full
    // integer pool, byte-identical to before f16 support.
    const uses_f16 = ir.function.functionUsesF16(func);
    const reserve_f16_scratch = uses_f16 and !caps.zfh;
    // Only SCALAR f16 is handled; f16 nested in a vector/aggregate would fall through to the
    // raw-vector path and miscompile the half lanes, so reject that composite case cleanly.
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;

    // Reachability from the entry (block 0), computed ONCE and threaded through every block-
    // processing loop below (allocation, frame layout, emission, branch relaxation). Unreachable
    // blocks (e.g. a loop nest an optimization orphaned but left in the IR, since the IR has no
    // block-deletion primitive) must contribute nothing: no register pressure, no frame slots, no
    // emitted code. Valid SSA guarantees a reachable block only uses values from, and only branches
    // to, blocks that are themselves reachable, so skipping the unreachable ones changes nothing a
    // reachable block observes. When every block is reachable (every case before this) `reachable`
    // is all-true and each guard is a literal no-op, so the output is byte-identical.
    var doms = try dominators.compute(allocator, func);
    defer doms.deinit(allocator);
    const reachable = doms.reachable;

    // Address-mode fold analysis: a load/store whose pointer is a foldable `arith_imm.add(base, imm)`
    // addresses `imm(base)` directly, and the now-dead address-add is dropped. Threaded through
    // allocation (liveness attributes the pointer use to the base, the dead add claims no register)
    // and emission (the base+offset form is emitted, the dead add is skipped). A function with nothing
    // foldable yields an empty analysis, keeping its output byte-identical. riscv64 has no load-pair,
    // so the win is add-elision plus a folded displacement (no ldp/stp).
    var fold = try addrfold.analyze(allocator, func, {}, riscv64FoldOffset);
    defer fold.deinit(allocator);

    var alloc = try allocateRegisters(allocator, func, caps.vpu, reserve_f16_scratch, reachable, &fold);
    defer alloc.deinit(allocator);

    return emitFromAllocation(allocator, func, caps, uses_f16, reachable, &alloc, &fold);
}

/// Emit machine code from a finished `Allocation` (the second half of `compileFunction`, split out
/// so the shared Wimmer allocator can drive the SAME battle-tested emission through
/// `compileFunctionWimmerRiscv`). This is a PURE extraction of everything after `allocateRegisters`:
/// the frame layout, prologue, the per-block/instruction loop reading each value's location through
/// `intLocationAt`/`floatLocationAt`, the split-boundary action drain, block-edge moves, branch
/// relaxation, and the epilogue. `caps` carries the model seams (`fetch_align`/`vpu`/`zicbop`/`zfh`);
/// `uses_f16` and `reachable` are what `compileFunction` computed. `alloc` is consumed read-only
/// except for the defensive sort of its action list. Byte-identical for every existing caller (the
/// full riscv64 suite proves it).
fn emitFromAllocation(allocator: std.mem.Allocator, func: *const Function, caps: ModelCaps, uses_f16: bool, reachable: []const bool, alloc: *Allocation, fold: *const addrfold.Analysis) Error!Compiled {
    const fetch_align = caps.fetch_align;
    const vpu = caps.vpu;
    const zicbop = caps.zicbop;
    const zfh = caps.zfh;
    // Whether the compare-into-branch fold runs (see `fusesIntoNextIf`). Threaded to both the
    // icmp-skip site and the fused `.@"if"` site below so they agree (the dual-check pair): a
    // model without the fusion falls back to the slt/sltu-then-branch path unchanged.
    const fuse_cmp_branch = caps.fuse_cmp_branch;
    // Whether the Zba sh-add fold runs (see `fusesIntoNextShiftAdd`). Threaded to both the shl-skip
    // site (the `.arith_imm` arm) and the fused emit site (the `.arith` add arm) below so they agree:
    // FALSE by default (no Zba), so both fall back to the plain slli-then-add path unchanged.
    const fuse_shift_add = caps.fuse_shift_add;
    // Whether the `.global_addr` arm's adjacency assert runs (see its site below). The auipc+addi
    // pair is emitted back-to-back unconditionally, so this never changes emission either way, it
    // only gates whether the invariant is asserted.
    const fuse_addr_hi_lo = caps.fuse_addr_hi_lo;

    // Split-boundary actions are appended in monotonic `at` order already; sort defensively so the
    // per-instruction drain below can advance a single cursor. At the SAME position a `.reload` must
    // precede a `.store` (Task 6d), so the comparator breaks `at` ties on kind (reload before store).
    std.mem.sort(SplitAction, alloc.actions.items, {}, struct {
        fn order(k: @TypeOf(@as(SplitAction, undefined).kind)) u8 {
            return switch (k) {
                .reload => 0,
                .move => 1,
                .store => 2,
            };
        }
        fn f(_: void, a: SplitAction, b: SplitAction) bool {
            if (a.at != b.at) return a.at < b.at;
            return order(a.kind) < order(b.kind);
        }
    }.f);

    // The two scalar-float spill scratch registers, chosen per mode (see `float_spill_scratch0`).
    const fspill0: FReg = if (vpu) float_spill_scratch0_vpu else float_spill_scratch0;
    const fspill1: FReg = if (vpu) float_spill_scratch1_vpu else float_spill_scratch1;

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
        // An `alloca` reachable only from dead code reserves no frame slot (it is never emitted).
        if (!reachable[bi]) continue;
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
        // A split value is removed from `alloc.int`, its register handed to the taker (usually still
        // seen above). But a callee-saved register held ONLY by a split prefix segment must still be
        // preserved, so scan the segments too when the direct scan came up empty.
        if (!used) {
            var sit = alloc.segments.valueIterator();
            outer: while (sit.next()) |segs| {
                for (segs.*) |seg| switch (seg.loc) {
                    .reg => |rr| if (rr == s) {
                        used = true;
                        break :outer;
                    },
                    .slot => {},
                };
            }
        }
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
        // A split float value is not in `alloc.float`; a callee-saved float register held only by a
        // split prefix segment must still be preserved, so scan `float_segments` too (empty on the
        // default path, so this loop finds nothing and the frame is byte-identical there).
        if (!used) {
            var sit = alloc.float_segments.valueIterator();
            outer: while (sit.next()) |segs| {
                for (segs.*) |seg| switch (seg.loc) {
                    .reg => |rr| if (rr == s) {
                        used = true;
                        break :outer;
                    },
                    .slot => {},
                };
            }
        }
        if (used) {
            frame = alignUp(frame, 8);
            if (frame > 2047) return error.Unsupported;
            try used_float_saved.append(allocator, .{ .reg = s, .off = @intCast(frame) });
            frame += 8;
        }
    }
    // vpu mode borrows f8/f9 (fs0/fs1, callee-saved) as the float spill scratch registers. A vpu
    // function that actually spills a scalar float clobbers them, so preserve the pair in the frame
    // exactly like any other used callee-saved float. Non-vpu mode uses caller-saved f30/f31 as
    // scratch, so it needs no such slot.
    if (vpu and alloc.float_spill_count != 0) {
        for ([_]FReg{ float_spill_scratch0_vpu, float_spill_scratch1_vpu }) |s| {
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
        // A `call` that lives only in dead code is never emitted, so it never clobbers ra and does
        // not force a save slot. (With all blocks reachable this is exactly the old scan.)
        if (!reachable[bi]) continue;
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
    // Float spill slots: one 8-byte doubleword per spilled scalar float (an f32 uses the low 4
    // bytes, an f64 the whole slot), 8-aligned. `alloc.float_spill_count` is 0 for any function that
    // never runs out of float registers, so this reserves nothing (and adds no alignment padding)
    // in the common case: byte-identical to before this field existed.
    frame = alignUp(frame, 8);
    const float_spill_base: u32 = frame;
    frame += alloc.float_spill_count * 8;
    // Vector spill slots: one 16-byte (a <4 x f32>) slot per spilled vector, 16-aligned.
    frame = alignUp(frame, 16);
    const vspill_base: u32 = frame;
    frame += alloc.vector_spill_count * 16;
    // VPU vector spill slots: one 32-byte (8 x f32) slot per spilled VPU vector, 32-aligned.
    // `alloc.vpu_vector_spill_count` is always 0 outside vpu mode, so this reserves nothing
    // (and touches no alignment padding) for every non-vpu caller: byte-identical to before
    // this field existed.
    const vpu_vspill_base: u32 = blk: {
        if (alloc.vpu_vector_spill_count == 0) break :blk frame;
        frame = alignUp(frame, 32);
        const base = frame;
        frame += alloc.vpu_vector_spill_count * 32;
        break :blk base;
    };
    // et-soc VPU pack scratch: `struct_new` has no lane-insert VPU instruction (`fbcx.ps`
    // broadcasts one scalar to every lane, it does not insert into one), so packing a vector
    // from distinct scalars goes through this reserved 32-byte slot: 8 scalar `fsw`s at
    // consecutive 4-byte offsets, then one `flw.ps` loads all 8 lanes at once. Reserved
    // unconditionally in vpu mode (a few bytes in a vpu function that happens not to pack is a
    // fair price for a fixed, easy-to-verify offset); never reserved outside vpu mode.
    var vpu_pack_base: u32 = 0;
    if (vpu) {
        frame = alignUp(frame, 32);
        vpu_pack_base = frame;
        frame += 32;
    }
    // et-soc matmul staging scratch: a real row-major sub-tile whose row pitch (k*4 for A, n*4 for
    // B) is not a multiple of 64 cannot be `tensor_load`ed directly (the load addr/stride are
    // 64-byte granular), so the matmul lowering stages such rows through a 64-byte-aligned buffer
    // with a 64-byte row pitch. Reserve one full 16-line sub-tile (16*64) plus 63 bytes of slack,
    // since sp is only 16-aligned and the stage base is rounded up to 64 at runtime. Reserved only
    // for vpu functions that actually contain a matmul (nothing for every other caller).
    var matmul_stage_base: u32 = 0;
    if (vpu and functionHasMatmul(func)) {
        matmul_stage_base = frame;
        frame += 16 * 64 + 63;
    }
    // et-soc EMBEDDED matmul save-area: an embedded matmul is lowered self-contained (it saves every
    // register it clobbers on entry and restores on exit), so it needs a fixed stack area to save
    // into. Layout (all 8-byte slots, see the `.matmul` lowering): the 4 clobbered int scratch temps
    // (x5/x7/x28/x31), the 3 a/b/c holder registers' incoming values (x29/x30/x9), 3 a/b/c pointer
    // transfer slots, then 32 float slots for the worst-case TenC clobber (f0..f31, one f64 slot
    // each). Base is 8-aligned so every `sd`/`fsd` offset is naturally aligned. Reserved only for a
    // function that actually contains an embedded matmul, so every other function is byte-identical.
    var matmul_save_base: u32 = 0;
    if (vpu and functionHasEmbeddedMatmul(func)) {
        matmul_save_base = alignUp(frame, 8);
        frame = matmul_save_base + matmul_save_int_bytes + matmul_save_float_bytes;
    }
    if (frame > 2047) return error.Unsupported;

    const frame_size: i12 = @intCast(alignUp(frame, 16));
    // Prologue: open the frame, then preserve ra and the callee-saved registers.
    if (frame_size != 0) try code.append(allocator, encode.addi(.x2, .x2, -frame_size));
    if (non_leaf) try code.append(allocator, encode.sd(.x1, .x2, ra_off)); // save ra
    for (used_saved.items) |sv| try code.append(allocator, encode.sd(sv.reg, .x2, sv.off));
    // In vpu (et-soc) mode every scalar float is fp32 and the sw-sysemu oracle implements only the
    // 32-bit fsw/flw scalar forms (not the 64-bit fsd/fld doubleword forms, which trap as illegal),
    // so save the callee-saved float pair (f8/f9, the vpu float spill scratch) with fsw. Outside vpu
    // mode a callee-saved float may hold an f64, so the 64-bit fsd is required.
    for (used_float_saved.items) |sv| try code.append(allocator, if (vpu) encode.fsw(sv.reg, .x2, sv.off) else encode.fsd(sv.reg, .x2, sv.off));

    // Move any entry parameter homed to a non-argument register (because it
    // outlives a call) out of its incoming argument register. Load stack
    // parameters from the caller's outgoing-argument area.
    if (func.blockCount() != 0) {
        var ia: usize = 0;
        var fa: usize = 0;
        for (func.blockParams(@enumFromInt(0))) |p| {
            if (isFloat(func, func.valueType(p))) {
                const arg = fargReg(fa);
                if (alloc.float.get(p)) |home| {
                    if (home != arg) try code.append(allocator, if (is64Float(func, func.valueType(p))) encode.fmv_d(home, arg) else encode.fmv_s(home, arg));
                } else {
                    // Entry float param spilled (it outlives a call but no callee-saved float reg
                    // was free): store the incoming argument register into its stack slot.
                    try storeFloat(allocator, &code, alloc, float_spill_base, p, 0, is64Float(func, func.valueType(p)), arg);
                }
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

    // Configure the RVV unit once for the fixed 4-lane f32 group. VL and SEW persist as CPU state and
    // nothing in a vector function changes them, so one vsetivli suffices for every vle32/vse32/arith.
    // The default path fires this exactly as before (`alloc.vector.count() != 0`, byte-identical). The
    // shared Wimmer path additionally spills/splits vectors (a vector live across a call), so it can
    // hold vectors only in slots or split segments with `vector.count()` zero, yet still emit vle32/
    // vse32 that need VL set - so fire the preamble on those signals too, gated on `edge_move_driven`
    // (the Wimmer-only flag, always false on the default path) to keep default emission byte-identical.
    if (!vpu and (alloc.vector.count() != 0 or
        (alloc.edge_move_driven and (alloc.vector_spill_count != 0 or alloc.vector_segments.count() != 0))))
        try code.append(allocator, encode.vsetivli(.x0, 4, 0xD0));
    // et-soc VPU mask preamble: VPU arithmetic is predicated by an M0..M7 mask register bank
    // rather than a vector length (there is no vtype/VL to configure), so a full-width 8-lane
    // op needs every mask bit set. Write M0 = 0xFF once, up front, analogous to the RVV
    // vsetivli hint above (both persist as CPU/register state for the rest of the function).
    if (vpu and (alloc.vpu_vector.count() != 0 or alloc.vpu_vector_spill_count != 0 or
        (alloc.edge_move_driven and alloc.vpu_segments.count() != 0)))
        try code.append(allocator, encode.mov_m_x(0, .x0, 0xFF));

    const block_start = try allocator.alloc(usize, func.blockCount());
    defer allocator.free(block_start);
    var fixups: std.ArrayList(Fixup) = .empty;
    defer fixups.deinit(allocator);

    // Loop-header alignment (a placement hint only, see the doc comment above): computed once,
    // up front, so the per-block loop below just checks a bit per block.
    var is_loop_header = try allocator.alloc(bool, func.blockCount());
    defer allocator.free(is_loop_header);
    @memset(is_loop_header, false);
    if (fetch_align > 4) {
        var li = try loops.analyze(allocator, func);
        defer li.deinit(allocator);
        for (li.loops) |l| is_loop_header[l.header] = true;
    }

    // `pos` mirrors the allocator's liveness numbering EXACTLY so the location each integer read is
    // resolved at matches the position its allocation was computed for. Per reachable block: the
    // block-parameter row occupies one position, then each instruction one position, then the
    // terminator one position. An unreachable block is skipped WITHOUT advancing `pos`, matching the
    // allocator. With no splits (segments empty) `pos` is otherwise unobservable, so the per-result
    // assert below is how a threading bug is caught now instead of in a later splitting task.
    var pos: usize = 0;
    var action_cursor: usize = 0;
    for (0..func.blockCount()) |bi| {
        // Emit ONLY reachable blocks, so a dead block produces no code (and no header padding). Its
        // `block_start[bi]` entry is left as-is and is never read: no reachable branch fixup targets
        // an unreachable block (valid SSA), and the relaxation loops below skip it in lockstep.
        if (!reachable[bi]) continue;
        const block: Block = @enumFromInt(bi);
        if (fetch_align > 4 and is_loop_header[bi]) {
            var pad = alignPadWords(code.items.len, fetch_align);
            while (pad > 0) : (pad -= 1) try code.append(allocator, encode.nop());
        }
        block_start[bi] = code.items.len;

        // A structured `if` is the block's exit. The trailing terminator is dead.
        var exited = false;
        const block_insts = func.blockInsts(block);
        // The block-parameter row occupies `pos`; the first instruction is the next position. Compute
        // each instruction's position from this base plus its index (robust to the `continue`/`break`
        // paths in the switch, which a trailing increment would desync).
        const first_inst_pos = pos + 1;
        for (block_insts, 0..) |inst, inst_idx| {
            const inst_pos = first_inst_pos + inst_idx;
            // Record a source-line row when this instruction starts a new line.
            if (lineOf(func, inst)) |line| {
                if (line != last_line) {
                    try lines.append(allocator, .{ .offset = @intCast(code.items.len * 4), .line = line });
                    last_line = line;
                }
            }
            // The pos coupling is otherwise unobservable while `segments` is empty, so assert it now:
            // an instruction with a result must be emitted at exactly that result's def position.
            if (func.instResult(inst)) |r| std.debug.assert(inst_pos == alloc.def_pos[@intFromEnum(r)]);
            // Drain split-boundary actions landing at this position BEFORE emitting the instruction. A
            // tail-split store writes the victim's register to its slot before the taker (the value
            // defined here) computes its result into that same register. The victim's value is still
            // in the register at this point (its last prefix use precedes this position and nothing
            // reused the register before it), so the store captures the correct bits.
            while (action_cursor < alloc.actions.items.len and alloc.actions.items[action_cursor].at <= inst_pos) {
                const act = alloc.actions.items[action_cursor];
                std.debug.assert(act.at == inst_pos); // actions land on instruction positions only
                // A `.store` writes the victim's register to its slot; a `.reload` is the second-chance
                // re-home (Task 6d) that loads the slot back into `act.reg` just before its next use,
                // after which the value's `.reg` re-home segment makes `intLocationAt` read the register
                // directly. A reload MUST drain before a store at the same position (the sort tiebreak
                // guarantees it), so a reload-then-respill at one use loads live bits before re-saving.
                // The Wimmer path also produces class-1 (float) and `.move` actions; `emitSplitAction`
                // dispatches on class/kind and is byte-identical for the native class-0 store/reload.
                try emitSplitAction(allocator, &code, func, spill_base, float_spill_base, vspill_base, vpu_vspill_base, vpu_pack_base, act);
                action_cursor += 1;
            }
            switch (func.opcode(inst)) {
                .arith => |a| {
                    // Fused multiply-add/sub: when this is a scalar float `mul` that is the
                    // single-use, immediately-preceding operand of the next add/sub, skip its
                    // materialization entirely. The `.arith` add/sub branch below re-checks the
                    // SAME predicate and emits the fused fmadd/fmsub/fnmsub on these operands,
                    // so the multiply is emitted exactly once (mirrors the icmp/if fusion above).
                    if (a.op == .mul and fusesIntoNextArith(func, block_insts, inst_idx, vpu)) {
                        // A vector mul always fuses (the RVV fused path below has three vector
                        // scratch registers). A scalar-float mul fuses only when every operand and
                        // result is register-resident (the R4-type fma needs three live source
                        // registers, one more than the two float spill scratches): otherwise it
                        // falls through and materializes as a standalone mul, and the add/sub below
                        // gates on the SAME predicate so it likewise does not fuse.
                        if (isVector(func, func.valueType(a.lhs)) or fusesScalarFloatArith(func, alloc, block_insts, inst_idx, vpu)) continue;
                    }
                    if (isVector(func, func.valueType(a.lhs))) {
                        const result = func.instResult(inst).?;
                        if (vpu and a.op == .add and inst_idx >= 1 and
                            fusesIntoNextArith(func, block_insts, inst_idx - 1, vpu))
                        {
                            // Fused et-soc VPU multiply-add (only the float add shape a*b+c fuses:
                            // the CORE-ET ISA has just `fmadd.ps`, no fmsub.ps/fnmsub.ps and no
                            // packed-integer fma, so the sub shapes and `<8 x i32>` mul+add keep
                            // their separate ps/pi lowering below). The mul at inst_idx-1 already
                            // had its materialization skipped by the `.mul` branch above via the
                            // SAME `fusesIntoNextArith` gate, so emit `fmadd.ps rd = a*b + c` here.
                            // fmadd.ps is a 3-source form (fd separate from all of fs1/fs2/fs3), so
                            // unlike the RVV accumulate-into-vd path no copy of c is needed and no
                            // source aliasing can corrupt a live c. Register safety: the vpu scratch
                            // f28/f29/f30 (op0/op1/work) are DISJOINT from the allocatable vpu pool
                            // f16..f27, so a register-resident rd never aliases va/vb/vc. When
                            // `result` spills, rd == vpu_vec_work == vc's reg (f30); `fmadd.ps
                            // f30,f28,f29,f30` reads f30 as fs3 before writing fd = f30, which is
                            // correct (all sources are read before fd is written), and f28/f29
                            // (va/vb) are always distinct from f30 so a*b is read intact.
                            const mul = func.opcode(block_insts[inst_idx - 1]).arith;
                            const mul_result = func.instResult(block_insts[inst_idx - 1]).?;
                            const c_val = if (a.lhs == mul_result) a.rhs else a.lhs; // the accumulator addend, c
                            const va = try reloadVpuVector(allocator, &code, alloc, vpu_vspill_base, mul.lhs, inst_pos, vpu_vec_op0);
                            const vb = try reloadVpuVector(allocator, &code, alloc, vpu_vspill_base, mul.rhs, inst_pos, vpu_vec_op1);
                            const vc = try reloadVpuVector(allocator, &code, alloc, vpu_vspill_base, c_val, inst_pos, vpu_vec_work);
                            const rd = dstVpuVector(alloc, result, inst_pos, vpu_vec_work);
                            try code.append(allocator, encode.fmadd_ps(rd, va, vb, vc));
                            try storeVpuVector(allocator, &code, alloc, vpu_vspill_base, result, inst_pos, rd);
                        } else if (vpu) {
                            // et-soc VPU packed-single arithmetic (8-lane f32, the disjoint
                            // f16..f31 partition). Spilled operands reload into
                            // vpu_vec_op0/op1, a spilled result computes in vpu_vec_work.
                            // Execution-validated against the CORE-ET RTL masks (see encode.zig)
                            // via the sw-sysemu ETSOC-1 emulator, not just encoding-checked: see
                            // riscv64/tests/etsoc_sysemu.zig, which runs this path's compiled
                            // output on sw-sysemu and checks the result bit-for-bit against a
                            // scalar reference. That emulator is not present in CI, so those
                            // tests skip there rather than fail; they run wherever sw-sysemu is
                            // on PATH.
                            const lhs = try reloadVpuVector(allocator, &code, alloc, vpu_vspill_base, a.lhs, inst_pos, vpu_vec_op0);
                            const rhs = try reloadVpuVector(allocator, &code, alloc, vpu_vspill_base, a.rhs, inst_pos, vpu_vec_op1);
                            const rd = dstVpuVector(alloc, result, inst_pos, vpu_vec_work);
                            // The vector partition holds both `<8 x f32>` and `<8 x i32>` (isVector
                            // routes either here). The element type selects the op family: an
                            // integer element lowers to the packed-integer `pi` ops (the sibling of
                            // the packed-single `ps` ops), operating on the SAME vpu vector
                            // registers. The reload/dst/store above are identical for both.
                            const word = if (isIntVector(func, func.valueType(a.lhs))) blk: {
                                // Packed-integer arithmetic (8-lane i32). A right shift picks
                                // logical (`fsrl.pi`) for an unsigned element and arithmetic
                                // (`fsra.pi`) for a signed one, matching the scalar srl/sra split.
                                // `div`/`rem` have no `pi` op, so they cannot be served here.
                                const unsigned = isUnsignedIntVector(func, func.valueType(a.lhs));
                                break :blk switch (a.op) {
                                    .add => encode.fadd_pi(rd, lhs, rhs),
                                    .sub => encode.fsub_pi(rd, lhs, rhs),
                                    .mul => encode.fmul_pi(rd, lhs, rhs),
                                    .bit_and => encode.fand_pi(rd, lhs, rhs),
                                    .bit_or => encode.for_pi(rd, lhs, rhs),
                                    .bit_xor => encode.fxor_pi(rd, lhs, rhs),
                                    .shl => encode.fsll_pi(rd, lhs, rhs),
                                    .shr => if (unsigned) encode.fsrl_pi(rd, lhs, rhs) else encode.fsra_pi(rd, lhs, rhs),
                                    .div, .rem, .mulh => return error.Unsupported, // no packed-integer divide/remainder/high-multiply op
                                };
                            } else switch (a.op) {
                                .add => encode.fadd_ps(rd, lhs, rhs),
                                .sub => encode.fsub_ps(rd, lhs, rhs),
                                .mul => encode.fmul_ps(rd, lhs, rhs),
                                .div => encode.fdiv_ps(rd, lhs, rhs),
                                else => return error.Unsupported, // bitwise/shift/rem on float vectors
                            };
                            try code.append(allocator, word);
                            try storeVpuVector(allocator, &code, alloc, vpu_vspill_base, result, inst_pos, rd);
                        } else if ((a.op == .add or a.op == .sub) and inst_idx >= 1 and
                            fusesIntoNextArith(func, block_insts, inst_idx - 1, vpu))
                        {
                            // Fused RVV multiply-add/sub, the add/sub side (mirrors the scalar
                            // fused case below, and aarch64's vector FMLA/FMLS): the mul at
                            // inst_idx-1 already had its materialization skipped by the `.mul`
                            // branch above via the SAME `fusesIntoNextArith` check. Resolve its
                            // own operands (a, b) plus this add/sub's other operand (the
                            // accumulator, c). vfmacc/vfmsac/vfnmsac ACCUMULATE into vd (vd is
                            // also a source), so c must be resident in vd before the op runs.
                            // Move it into the fixed scratch `vector_scratch` (v31, outside
                            // every allocation pool, so it never aliases a/b) first - a naive
                            // "vfmacc straight into c's own register" would corrupt c for any
                            // other reader if c's register differs from the result register but
                            // is read again later - then move the scratch into the result
                            // register only if they differ:
                            //   add(mul(a,b), c) = a*b+c -> vfmacc: vd = vs1*vs2 + vd, vd preloaded with c
                            //   sub(mul(a,b), c) = a*b-c -> vfmsac: vd = vs1*vs2 - vd, vd preloaded with c
                            //   sub(c, mul(a,b)) = c-a*b -> vfnmsac: vd = vd - vs1*vs2, vd preloaded with c
                            const mul = func.opcode(block_insts[inst_idx - 1]).arith;
                            const mul_result = func.instResult(block_insts[inst_idx - 1]).?;
                            const ra_val = if (a.lhs == mul_result) a.rhs else a.lhs; // the accumulator, c
                            const vm1 = try reloadVector(allocator, &code, alloc, vspill_base, mul.lhs, inst_pos, vec_op0, spill_scratch1);
                            const vm2 = try reloadVector(allocator, &code, alloc, vspill_base, mul.rhs, inst_pos, vec_op1, spill_scratch1);
                            const vc = try reloadVector(allocator, &code, alloc, vspill_base, ra_val, inst_pos, vector_scratch, spill_scratch1);
                            if (vc != vector_scratch) try code.append(allocator, encode.vmv_v_v(vector_scratch, vc));
                            try code.append(allocator, switch (a.op) {
                                .add => encode.vfmacc_vv(vector_scratch, vm1, vm2),
                                .sub => if (a.lhs == mul_result)
                                    encode.vfmsac_vv(vector_scratch, vm1, vm2) // a*b - c
                                else
                                    encode.vfnmsac_vv(vector_scratch, vm1, vm2), // c - a*b
                                else => unreachable, // fusesIntoNextArith only accepts .add/.sub
                            });
                            const rd = dstVector(alloc, result, inst_pos, vec_work);
                            if (rd != vector_scratch) try code.append(allocator, encode.vmv_v_v(rd, vector_scratch));
                            try storeVector(allocator, &code, alloc, vspill_base, result, inst_pos, rd, spill_scratch1);
                        } else {
                            // RVV vector arithmetic. Spilled operands reload into
                            // vec_op0/op1, a spilled result computes in vec_work.
                            // spill_scratch1 holds the slot address.
                            const lhs = try reloadVector(allocator, &code, alloc, vspill_base, a.lhs, inst_pos, vec_op0, spill_scratch1);
                            const rhs = try reloadVector(allocator, &code, alloc, vspill_base, a.rhs, inst_pos, vec_op1, spill_scratch1);
                            const rd = dstVector(alloc, result, inst_pos, vec_work);
                            try code.append(allocator, switch (a.op) {
                                .add => encode.vfadd_vv(rd, lhs, rhs),
                                .sub => encode.vfsub_vv(rd, lhs, rhs),
                                .mul => encode.vfmul_vv(rd, lhs, rhs),
                                .div => encode.vfdiv_vv(rd, lhs, rhs),
                                else => return error.Unsupported,
                            });
                            try storeVector(allocator, &code, alloc, vspill_base, result, inst_pos, rd, spill_scratch1);
                        }
                    } else if (isFloat(func, func.valueType(a.lhs))) {
                        // Fused multiply-add/sub, the add/sub side: when the immediately-
                        // preceding instruction is a single-use float mul that is exactly one
                        // of this add/sub's operands (the SAME predicate the mul case above used
                        // to skip its materialization), load the mul's own operands (never
                        // materialized) plus the add/sub's other operand (the accumulator, `c`),
                        // and emit the one R4-type instruction whose hardware semantics matches
                        // the IR shape. RISC-V's variant mapping differs from aarch64's (RISC-V
                        // FMSUB is rs1*rs2-rs3, not aarch64's fnmsub-shaped subtract):
                        //   add(mul(a,b), c) = a*b+c -> fmadd:  rd = rs1*rs2 + rs3
                        //   sub(mul(a,b), c) = a*b-c -> fmsub:  rd = rs1*rs2 - rs3
                        //   sub(c, mul(a,b)) = c-a*b -> fnmsub: rd = rs3 - rs1*rs2
                        if ((a.op == .add or a.op == .sub) and inst_idx >= 1 and fusesScalarFloatArith(func, alloc, block_insts, inst_idx - 1, vpu)) {
                            // fusesScalarFloatArith guarantees every operand and the result is
                            // register-resident here, so the R4-type fma reads/writes real registers
                            // with no reload or spill store (byte-identical to the pre-spill path).
                            const mul = func.opcode(block_insts[inst_idx - 1]).arith;
                            const mul_result = func.instResult(block_insts[inst_idx - 1]).?;
                            const ra_val = if (a.lhs == mul_result) a.rhs else a.lhs; // the accumulator, `c`
                            const rm1 = alloc.float.get(mul.lhs).?;
                            const rm2 = alloc.float.get(mul.rhs).?;
                            const ra = alloc.float.get(ra_val).?;
                            const rd = alloc.float.get(func.instResult(inst).?).?;
                            const d = is64Float(func, func.valueType(mul.lhs));
                            const word = switch (a.op) {
                                .add => if (d) encode.fmadd_d(rd, rm1, rm2, ra) else encode.fmadd_s(rd, rm1, rm2, ra),
                                .sub => if (a.lhs == mul_result)
                                    (if (d) encode.fmsub_d(rd, rm1, rm2, ra) else encode.fmsub_s(rd, rm1, rm2, ra)) // a*b - c
                                else
                                    (if (d) encode.fnmsub_d(rd, rm1, rm2, ra) else encode.fnmsub_s(rd, rm1, rm2, ra)), // c - a*b
                                else => unreachable, // fusesIntoNextArith only accepts .add/.sub
                            };
                            try code.append(allocator, word);
                            continue;
                        }
                        const result = func.instResult(inst).?;
                        const d = is64Float(func, func.valueType(a.lhs));
                        // Reload spilled operands into the two float spill scratches (distinct, so a
                        // both-operands-spilled binary op keeps both live), compute into the result's
                        // register or `fspill0` if it too spilled, then store it back.
                        const rs1 = try reloadFloat(allocator, &code, alloc, float_spill_base, a.lhs, inst_pos, d, fspill0);
                        const rs2 = try reloadFloat(allocator, &code, alloc, float_spill_base, a.rhs, inst_pos, d, fspill1);
                        const rd = dstFloat(alloc, result, inst_pos, fspill0);
                        // Native f16 (Zfh): the operands are held as native halves (`is64Float(f16)`
                        // is false, so `d` is false and the reload used `flw`, which preserves the
                        // 32-bit NaN-boxed half), so emit the half op directly. It rounds once to
                        // nearest half, so no software re-round follows.
                        const half_native = zfh and isHalf(func, func.valueType(a.lhs));
                        const word = if (half_native) switch (a.op) {
                            .add => encode.fadd_h(rd, rs1, rs2),
                            .sub => encode.fsub_h(rd, rs1, rs2),
                            .mul => encode.fmul_h(rd, rs1, rs2),
                            .div => encode.fdiv_h(rd, rs1, rs2),
                            else => return error.Unsupported, // bitwise/shift/rem on floats
                        } else switch (a.op) {
                            .add => if (d) encode.fadd_d(rd, rs1, rs2) else encode.fadd_s(rd, rs1, rs2),
                            .sub => if (d) encode.fsub_d(rd, rs1, rs2) else encode.fsub_s(rd, rs1, rs2),
                            .mul => if (d) encode.fmul_d(rd, rs1, rs2) else encode.fmul_s(rd, rs1, rs2),
                            .div => if (d) encode.fdiv_d(rd, rs1, rs2) else encode.fdiv_s(rd, rs1, rs2),
                            else => return error.Unsupported, // bitwise/shift/rem on floats
                        };
                        try code.append(allocator, word);
                        // Software f16: arithmetic is done in f32 (the held-as-f32 widening) and then
                        // rounded back to half per op (fusion into an fma is disabled for f16 above,
                        // so this is the only place an f16 result is produced). After `fmv.x.w rd`
                        // the value lives in x6, so both float spill scratches are free for the round
                        // routine. The native path already rounded in-instruction, so it skips this.
                        if (isHalf(func, func.valueType(result)) and !zfh) try emitRoundToHalf(allocator, &code, rd, fspill0, fspill1);
                        try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, d, rd);
                    } else {
                        const result = func.instResult(inst).?;
                        // Zba sh-add fold, the add side: when the immediately-preceding instruction is
                        // a single-use `shl` by 1/2/3 whose result is one of this 64-bit add's operands
                        // (the SAME predicate the `.arith_imm` shl-skip below uses), emit one
                        // `sh{k}add rd, b, x` (rd = x + (b << k)) instead of a separate slli then add.
                        // The shl was skipped, so load `b` (the shifted operand) and `x` (the addend)
                        // directly here (both still resident, nothing ran between them and this add).
                        if (a.op == .add and inst_idx >= 1 and fusesIntoNextShiftAdd(func, block_insts, inst_idx - 1, fuse_shift_add)) {
                            const shl = func.opcode(block_insts[inst_idx - 1]).arith_imm;
                            const shl_result = func.instResult(block_insts[inst_idx - 1]).?;
                            const x_val = if (a.lhs == shl_result) a.rhs else a.lhs; // the add's non-shl operand, x
                            const rs1 = try reloadInt(allocator, &code, alloc, spill_base, shl.lhs, inst_pos, spill_scratch0); // b, the shifted operand
                            const rs2 = try reloadInt(allocator, &code, alloc, spill_base, x_val, inst_pos, spill_scratch1); // x, the addend
                            const rd_loc = intLocationAt(alloc, result, inst_pos);
                            const rd = switch (rd_loc) {
                                .reg => |r| r,
                                .slot => spill_scratch0,
                            };
                            const word = switch (shl.imm) {
                                1 => encode.sh1add(rd, rs1, rs2),
                                2 => encode.sh2add(rd, rs1, rs2),
                                3 => encode.sh3add(rd, rs1, rs2),
                                else => unreachable, // fusesIntoNextShiftAdd only accepts k in 1..3
                            };
                            try code.append(allocator, word);
                            switch (rd_loc) {
                                .reg => {},
                                .slot => |slot| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + slot * 8))),
                            }
                            continue;
                        }
                        const rs1 = try reloadInt(allocator, &code, alloc, spill_base, a.lhs, inst_pos, spill_scratch0);
                        const rs2 = try reloadInt(allocator, &code, alloc, spill_base, a.rhs, inst_pos, spill_scratch1);
                        const rd_loc = intLocationAt(alloc, result, inst_pos);
                        const rd = switch (rd_loc) {
                            .reg => |r| r,
                            .slot => spill_scratch0,
                        };
                        const unsigned = isUnsignedInt(func, func.valueType(a.lhs));
                        const word = if (unsigned and a.op == .div)
                            encode.divu(rd, rs1, rs2)
                        else if (unsigned and a.op == .rem)
                            encode.remu(rd, rs1, rs2)
                        else if (unsigned and a.op == .shr)
                            encode.srl(rd, rs1, rs2) // logical right shift
                        else if (unsigned and a.op == .mulh)
                            encode.mulhu(rd, rs1, rs2) // unsigned high multiply
                        else
                            arithWord(a.op, rd, rs1, rs2);
                        try code.append(allocator, word);
                        switch (rd_loc) {
                            .reg => {},
                            .slot => |slot| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + slot * 8))),
                        }
                    }
                },
                .arith_imm => |a| {
                    // A folded address-add is dead: every use of its result was rerouted to the fold
                    // base, so it claims no register and emits nothing (mirrors the mul/icmp fusion
                    // skips above). `inst_pos` still advanced from `inst_idx`, so numbering holds.
                    if (fold.isDeadAdd(inst)) continue;
                    // Zba sh-add fold, the shl side: when this `shl` by 1/2/3 is the single-use,
                    // immediately-preceding operand of the next 64-bit add, skip its materialization.
                    // The `.arith` add arm re-checks the SAME `fusesIntoNextShiftAdd` gate and emits
                    // the fused `sh{k}add`, so the shift is emitted exactly once (mirrors the mul/fma
                    // skip above). `inst_pos` still advanced from `inst_idx`, so numbering holds.
                    if (a.op == .shl and fusesIntoNextShiftAdd(func, block_insts, inst_idx, fuse_shift_add)) continue;
                    if (isFloat(func, func.valueType(a.lhs))) return error.Unsupported;
                    const result = func.instResult(inst).?;
                    const rs1 = try reloadInt(allocator, &code, alloc, spill_base, a.lhs, inst_pos, spill_scratch1);
                    const rd_loc = intLocationAt(alloc, result, inst_pos);
                    const rd = switch (rd_loc) {
                        .reg => |r| r,
                        .slot => spill_scratch0,
                    };
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
                        .mul, .mulh, .div, .rem => return error.Unsupported, // no immediate form
                    };
                    try code.append(allocator, word);
                    switch (rd_loc) {
                        .reg => {},
                        .slot => |slot| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + slot * 8))),
                    }
                },
                .iconst => |c| {
                    const res = func.instResult(inst).?;
                    if (isFloat(func, func.valueType(res))) {
                        // A float-typed integer constant (a zero-init). Materialize the
                        // bits and move them into the float register, never an integer
                        // register the value was never assigned.
                        if (is64Float(func, func.valueType(res))) return error.Unsupported;
                        const fr = dstFloat(alloc, res, inst_pos, fspill0);
                        // Native f16 (Zfh): a float-typed integer constant is a zero-init, so move
                        // its low 16 bits into the float register NaN-boxed (`fmv.h.x`); moving the
                        // f32-widening bits with `fmv.w.x` would leave an invalid NaN-box that the
                        // half ops read as NaN. SOFTWARE f16 / f32: the f32-widening bits go in via
                        // `fmv.w.x` unchanged.
                        if (zfh and isHalf(func, func.valueType(res))) {
                            const half_bits: u32 = @as(u16, @truncate(@as(u64, @bitCast(c))));
                            try loadImm32(allocator, &code, scratch_reg, half_bits);
                            try code.append(allocator, encode.fmv_h_x(fr, scratch_reg));
                        } else {
                            const bits: u32 = @truncate(@as(u64, @bitCast(c)));
                            try loadImm32(allocator, &code, scratch_reg, bits);
                            try code.append(allocator, encode.fmv_w_x(fr, scratch_reg));
                        }
                        try storeFloat(allocator, &code, alloc, float_spill_base, res, inst_pos, false, fr);
                        continue;
                    }
                    // A spilled integer constant materializes into the scratch, then stores to its
                    // slot (mirrors the `arith`/`arith_imm` result-spill tail). Resident in every
                    // currently-compiling case, so this is byte-identical there.
                    const rd_loc = intLocationAt(alloc, res, inst_pos);
                    const rd = switch (rd_loc) {
                        .reg => |r| r,
                        .slot => spill_scratch0,
                    };

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
                        // Full 64-bit constant (e.g. a division magic number): built MSB-first.
                        try loadImm64(allocator, &code, rd, @bitCast(c));
                    }
                    switch (rd_loc) {
                        .reg => {},
                        .slot => |slot| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + slot * 8))),
                    }
                },
                .fconst => |val| {
                    const result = func.instResult(inst).?;
                    if (is64Float(func, func.valueType(result))) return error.Unsupported; // f64 const: later
                    const fr = dstFloat(alloc, result, inst_pos, fspill0);
                    // Native f16 (Zfh): materialize the 16-bit half pattern and move it into the
                    // float register NaN-boxed with `fmv.h.x`, giving a native half.
                    if (zfh and isHalf(func, func.valueType(result))) {
                        const half_bits: u32 = @as(u16, @bitCast(@as(f16, @floatCast(val))));
                        try loadImm32(allocator, &code, scratch_reg, half_bits);
                        try code.append(allocator, encode.fmv_h_x(fr, scratch_reg));
                        try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, false, fr);
                        continue;
                    }
                    // Software f16 / f32: load the 32-bit pattern, then move it into the float
                    // register. An f16 constant is pre-rounded to half (`@as(f16, val)`) before
                    // widening back to f32, so the materialized value already satisfies the
                    // held-as-f32-widening invariant.
                    const bits: u32 = if (isHalf(func, func.valueType(result)))
                        @bitCast(@as(f32, @as(f16, @floatCast(val))))
                    else
                        @bitCast(@as(f32, @floatCast(val)));
                    try loadImm32(allocator, &code, scratch_reg, bits);
                    try code.append(allocator, encode.fmv_w_x(fr, scratch_reg));
                    try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, false, fr);
                },
                .alloca => {
                    // The slot address is `sp + offset` into the frame.
                    const result = func.instResult(inst).?;
                    const rd = switch (intLocationAt(alloc, result, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const off = slot_offset.get(result).?;
                    try code.append(allocator, encode.addi(rd, .x2, off));
                },
                .global_addr => |ga| {
                    // PC-relative symbol address: `auipc rd, %pcrel_hi(sym)` then
                    // `addi rd, rd, %pcrel_lo(.Lhi)`. The two relocations resolve together.
                    const rd = switch (intLocationAt(alloc, func.instResult(inst).?, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const name = func.symbolName(ga.symbol);
                    const hi = code.items.len;
                    try relocs.append(allocator, .{ .offset = hi, .symbol = name, .kind = .pcrel_hi20 });
                    try code.append(allocator, encode.auipc(rd, 0));
                    try relocs.append(allocator, .{ .offset = code.items.len, .symbol = "", .kind = .pcrel_lo12, .pair = hi });
                    try code.append(allocator, encode.addi(rd, rd, 0));
                    // The addr_hi_lo macro-op fusion (`caps.fuse_addr_hi_lo`) relies on the addi
                    // landing exactly one word after its auipc; the two `code.append`s above
                    // guarantee that unconditionally, so this only asserts the invariant a fusing
                    // microarch depends on rather than changing anything.
                    if (fuse_addr_hi_lo) std.debug.assert(code.items.len == hi + 2);
                },
                .select => |sel| {
                    // `cond ? then : else`, lowered to a short forward branch. The
                    // result register is distinct from the operands (drawn while
                    // they are still live), so there is no aliasing hazard.
                    if (isFloat(func, func.valueType(sel.then))) return error.Unsupported; // float select: later
                    const rd = switch (intLocationAt(alloc, func.instResult(inst).?, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    // The three operands each need a live register through the branch sequence, more
                    // than the two int spill scratches can reload; a spilled operand (e.g. a spilled
                    // int block param) is rejected cleanly rather than panicking on the unwrap.
                    // Resident in every currently-compiling case (byte-identical there).
                    const cond = switch (intLocationAt(alloc, sel.cond, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const then_r = switch (intLocationAt(alloc, sel.then, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const else_r = switch (intLocationAt(alloc, sel.@"else", inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
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
                    const src_half = isHalf(func, src_ty);
                    const dst_half = isHalf(func, dst_ty);
                    if (src_float and dst_float) {
                        // Float -> float. Only conversions involving f16 are lowered here; a plain
                        // f32<->f64 convert is still deferred (falls through to Unsupported below).
                        // An f16 is held as its f32 widening, so the single-precision view is shared.
                        if (src_half and !dst_half) {
                            // f16 -> f32 / f16 -> f64.
                            const rs = try reloadFloat(allocator, &code, alloc, float_spill_base, cv.value, inst_pos, false, fspill0);
                            const rd = dstFloat(alloc, result, inst_pos, fspill1);
                            if (zfh) {
                                // The native path does one exact widen from the native half (`fcvt.s.h` / `fcvt.d.h`).
                                try code.append(allocator, if (is64Float(func, dst_ty)) encode.fcvt_d_h(rd, rs) else encode.fcvt_s_h(rd, rs));
                            } else if (is64Float(func, dst_ty)) {
                                // In the software path the held f32 is already the exact value, so widen it to f64.
                                try code.append(allocator, encode.fcvt_d_s(rd, rs));
                            } else if (rd != rs) {
                                try code.append(allocator, encode.fmv_s(rd, rs)); // identity move (f16 -> f32)
                            }
                            try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, is64Float(func, dst_ty), rd);
                        } else if (dst_half and !src_half) {
                            // f32 -> f16 / f64 -> f16.
                            const rs = try reloadFloat(allocator, &code, alloc, float_spill_base, cv.value, inst_pos, is64Float(func, src_ty), fspill0);
                            const rd = dstFloat(alloc, result, inst_pos, fspill0);
                            if (zfh) {
                                // The native path does one single-rounded narrow to a native half (`fcvt.h.s` /
                                // `fcvt.h.d`); the double-round through f32 is unnecessary.
                                try code.append(allocator, if (is64Float(func, src_ty)) encode.fcvt_h_d(rd, rs) else encode.fcvt_h_s(rd, rs));
                            } else {
                                // In the software path, reduce to f32 (exact for f32, one round for f64) then round
                                // to nearest-even half via the software routine.
                                if (is64Float(func, src_ty)) {
                                    try code.append(allocator, encode.fcvt_s_d(rd, rs));
                                } else if (rd != rs) {
                                    try code.append(allocator, encode.fmv_s(rd, rs));
                                }
                                try emitRoundToHalf(allocator, &code, rd, fspill0, fspill1);
                            }
                            try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, false, rd);
                        } else {
                            return error.Unsupported; // f32<->f64 (no f16) not yet lowered
                        }
                    } else if (src_float == dst_float) {
                        // int <-> int width change. Widening extends by the SOURCE signedness
                        // (sign-extend a signed source, zero-extend an unsigned one); same-width or
                        // narrowing keeps the low bits. Values live in 64-bit registers, so a widen
                        // shifts the source up to its top bit then arithmetic/logically back down,
                        // which also discards any dirty high bits above the source width.
                        const src_bits = intBits(func, src_ty);
                        const dst_bits = intBits(func, dst_ty);
                        const rs = try reloadInt(allocator, &code, alloc, spill_base, cv.value, inst_pos, spill_scratch1);
                        const rd_loc = intLocationAt(alloc, result, inst_pos);
                        const rd = switch (rd_loc) {
                            .reg => |r| r,
                            .slot => spill_scratch0,
                        };
                        if (dst_bits > src_bits and src_bits < 64) {
                            const sh: u6 = @intCast(64 - src_bits);
                            try code.append(allocator, encode.slli(rd, rs, sh));
                            try code.append(allocator, if (isUnsignedInt(func, src_ty))
                                encode.srli(rd, rd, sh)
                            else
                                encode.srai(rd, rd, sh));
                        } else if (rd != rs) {
                            try code.append(allocator, encode.addi(rd, rs, 0)); // mv: same width / narrowing
                        }
                        switch (rd_loc) {
                            .reg => {},
                            .slot => |slot| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + slot * 8))),
                        }
                    } else if (dst_float) {
                        // integer -> float, only a 32-bit signed source for now.
                        if (!isWord(func, src_ty)) return error.Unsupported;
                        // A spilled integer source has no reload path here yet: reject cleanly rather
                        // than panic. Resident in every currently-compiling case (byte-identical).
                        const rs = switch (intLocationAt(alloc, cv.value, inst_pos)) {
                            .reg => |r| r,
                            .slot => return error.Unsupported,
                        };
                        const rd = dstFloat(alloc, result, inst_pos, fspill0);
                        if (zfh and dst_half) {
                            // Native int -> f16: one single-rounded convert straight to a native half
                            // (`fcvt.h.w`), no detour through f32 and no software re-round.
                            try code.append(allocator, encode.fcvt_h_w(rd, rs));
                        } else {
                            // Software int -> f16 goes through f32 (fcvt.s.w) then rounds to half;
                            // int -> f32/f64 is the direct fcvt. `is64Float(f16)` is false, so f16
                            // picks the s-form.
                            try code.append(allocator, if (is64Float(func, dst_ty)) encode.fcvt_d_w(rd, rs) else encode.fcvt_s_w(rd, rs));
                            if (dst_half) try emitRoundToHalf(allocator, &code, rd, fspill0, fspill1);
                        }
                        try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, is64Float(func, dst_ty), rd);
                    } else {
                        // float -> integer, only a 32-bit signed destination for now.
                        if (!isWord(func, dst_ty)) return error.Unsupported;
                        const rs = try reloadFloat(allocator, &code, alloc, float_spill_base, cv.value, inst_pos, is64Float(func, src_ty), fspill0);
                        const rd = switch (intLocationAt(alloc, result, inst_pos)) {
                            .reg => |r| r,
                            .slot => return error.Unsupported,
                        };
                        if (zfh and src_half) {
                            // Native f16 -> int: truncate the native half directly (`fcvt.w.h`, rtz).
                            try code.append(allocator, encode.fcvt_w_h(rd, rs));
                        } else {
                            // Software f16 -> int truncates the held f32 directly (the s-form
                            // fcvt.w.s), exact for the half.
                            try code.append(allocator, if (is64Float(func, src_ty)) encode.fcvt_w_d(rd, rs) else encode.fcvt_w_s(rd, rs));
                        }
                    }
                },
                .load => {
                    const result = func.instResult(inst).?;
                    // A folded load addresses `disp(base)` directly: `baseOf` yields the fold base (the
                    // add's lhs) and `offOf` the displacement; both are the raw ptr and 0 when unfolded,
                    // so the non-folding case is byte-identical. foldOffset never folds a vector, so a
                    // vector load's `offOf` is 0 and its `vle32`/`flw.ps` addressing stays base-only.
                    const base_val = fold.baseOf(func, inst);
                    // A spilled base (a spilled int value used as a load address) has no reload path
                    // here yet, so reject cleanly rather than panic on the unwrap. Resident in every
                    // currently-compiling case, so this is byte-identical there.
                    const base = switch (intLocationAt(alloc, base_val, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const disp: i12 = @intCast(fold.offOf(inst)); // foldOffset guarantees the i12 range
                    if (isVector(func, func.valueType(result))) {
                        if (vpu) {
                            // `flw.ps`, like scalar `flw`, carries its own displacement, so
                            // (unlike RVV's vle32) no separate address register is needed. `disp` is 0
                            // here (vectors never fold), so the addressing is byte-identical.
                            const rd = dstVpuVector(alloc, result, inst_pos, vpu_vec_work);
                            try code.append(allocator, encode.flw_ps(rd, base, disp));
                            try storeVpuVector(allocator, &code, alloc, vpu_vspill_base, result, inst_pos, rd);
                        } else {
                            const rd = dstVector(alloc, result, inst_pos, vec_work);
                            try code.append(allocator, encode.vle32(rd, base));
                            try storeVector(allocator, &code, alloc, vspill_base, result, inst_pos, rd, spill_scratch1);
                        }
                    } else if (isHalf(func, func.valueType(result))) {
                        const rd = dstFloat(alloc, result, inst_pos, fspill0);
                        if (zfh) {
                            // In the native path, load the 2-byte IEEE half straight into the float register
                            // (`flh`, NaN-boxed), no software widen.
                            try code.append(allocator, encode.flh(rd, base, disp));
                        } else {
                            // In the software path, f16 memory is a 2-byte IEEE half, so zero-extend it with `lhu`,
                            // widen to f32 in software (exact), then move into the float register as
                            // the held-as-f32 value. `base` is disjoint from the convert scratch.
                            try code.append(allocator, encode.lhu(scratch_reg, base, disp));
                            try emitHalfToFloat(allocator, &code, spill_scratch1, scratch_reg, fspill0, fspill1);
                            try code.append(allocator, encode.fmv_w_x(rd, spill_scratch1));
                        }
                        try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, false, rd);
                    } else if (isFloat(func, func.valueType(result))) {
                        const d = is64Float(func, func.valueType(result));
                        const rd = dstFloat(alloc, result, inst_pos, fspill0);
                        try code.append(allocator, if (d) encode.fld(rd, base, disp) else encode.flw(rd, base, disp));
                        try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, d, rd);
                    } else {
                        const rd = switch (intLocationAt(alloc, result, inst_pos)) {
                            .reg => |r| r,
                            .slot => return error.Unsupported,
                        };
                        const ty = func.valueType(result);
                        const word = isWord(func, ty);
                        try code.append(allocator, intLoadInsn(func, ty, rd, base, disp));
                        // Swap a big-endian 64-bit load to native order.
                        if (!word and endianBig(func, .{ .value = result })) {
                            try code.append(allocator, encode.rev8(rd, rd));
                        } else if (word and endianBig(func, .{ .value = result })) {
                            return error.Unsupported; // sub-word byte-swap not yet handled
                        }
                    }
                },
                .store => |st| {
                    // A folded store addresses `disp(base)` directly (see the `.load` arm). `baseOf`/
                    // `offOf` are the raw ptr and 0 when unfolded, and never fold a vector, so both the
                    // non-folding case and every vector store are byte-identical.
                    const base_val = fold.baseOf(func, inst);
                    // A spilled store address has no reload path here yet: reject cleanly rather
                    // than panic. Resident in every currently-compiling case (byte-identical there).
                    const base = switch (intLocationAt(alloc, base_val, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const disp: i12 = @intCast(fold.offOf(inst)); // foldOffset guarantees the i12 range
                    if (isVector(func, func.valueType(st.value))) {
                        if (vpu) {
                            const vr = try reloadVpuVector(allocator, &code, alloc, vpu_vspill_base, st.value, inst_pos, vpu_vec_op0);
                            try code.append(allocator, encode.fsw_ps(vr, base, disp)); // disp is 0 (vectors never fold)
                        } else {
                            const vr = try reloadVector(allocator, &code, alloc, vspill_base, st.value, inst_pos, vec_op0, spill_scratch1);
                            try code.append(allocator, encode.vse32(vr, base));
                        }
                    } else if (isHalf(func, func.valueType(st.value))) {
                        const vr = try reloadFloat(allocator, &code, alloc, float_spill_base, st.value, inst_pos, false, fspill0);
                        if (zfh) {
                            // In the native path, store the native half's 2 bytes straight to memory (`fsh`).
                            try code.append(allocator, encode.fsh(vr, base, disp));
                        } else {
                            // Software f16 store: move the held-as-f32 value into a GPR, truncate to
                            // the 2-byte IEEE half in software (exact, since the value is already a
                            // half), then store the low halfword with `sh`. `base` is disjoint from
                            // the scratch.
                            try code.append(allocator, encode.fmv_x_w(scratch_reg, vr));
                            try emitFloatToHalf(allocator, &code, spill_scratch1, scratch_reg, fspill0, fspill1);
                            try code.append(allocator, encode.sh(spill_scratch1, base, disp));
                        }
                    } else if (isFloat(func, func.valueType(st.value))) {
                        const d = is64Float(func, func.valueType(st.value));
                        const vr = try reloadFloat(allocator, &code, alloc, float_spill_base, st.value, inst_pos, d, fspill0);
                        try code.append(allocator, if (d) encode.fsd(vr, base, disp) else encode.fsw(vr, base, disp));
                    } else {
                        // A spilled int store value has no reload path here yet: reject cleanly
                        // rather than panic. Resident in every currently-compiling case.
                        const vr = switch (intLocationAt(alloc, st.value, inst_pos)) {
                            .reg => |r| r,
                            .slot => return error.Unsupported,
                        };
                        const ty = func.valueType(st.value);
                        const word = isWord(func, ty);
                        // Reverse a big-endian 64-bit value before storing.
                        if (!word and endianBig(func, .{ .inst = inst })) {
                            try code.append(allocator, encode.rev8(scratch_reg, vr));
                            try code.append(allocator, encode.sd(scratch_reg, base, disp));
                        } else if (word and endianBig(func, .{ .inst = inst })) {
                            return error.Unsupported; // sub-word byte-swap not yet handled
                        } else {
                            try code.append(allocator, intStoreInsn(func, ty, vr, base, disp));
                        }
                    }
                },
                .prefetch => |pf| if (zicbop) {
                    // `pf.ptr` is already the exact address to prefetch: any model-derived
                    // distance was baked into it as address arithmetic upstream, in the
                    // insertion pass (vulcan-opt/microarch/prefetch.zig), so this lowers
                    // straight to `prefetch.r rs1, 0` with no further arithmetic here. `pf.ptr`
                    // is always recorded as a use (recordUses/usesInInst above), so its defining
                    // instruction already materialized it into an int register.
                    // A spilled prefetch address has no reload path here yet: reject cleanly rather
                    // than panic. Resident in every currently-compiling case (byte-identical).
                    const base = switch (intLocationAt(alloc, pf.ptr, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    try code.append(allocator, encode.prefetch_r(base, 0));
                },
                // Without Zicbop, this hint has nothing to lower to: dropping it here is a
                // correct (if suboptimal) no-op.
                .icmp => |cmp| if (fuse_cmp_branch and fusesIntoNextIf(func, block_insts, inst_idx)) {
                    // Fused compare-and-branch: this integer icmp is the single-use
                    // condition of the immediately-following if, so skip its slt/sltu
                    // materialization entirely. The `.@"if"` case re-checks the SAME
                    // predicate and emits the native compare-and-branch on these operands,
                    // so the comparison is emitted exactly once.
                } else if (isFloat(func, func.valueType(cmp.lhs))) {
                    // Float comparison: float operands, integer (bool) result.
                    const rd = switch (intLocationAt(alloc, func.instResult(inst).?, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const d = is64Float(func, func.valueType(cmp.lhs));
                    const rs1 = try reloadFloat(allocator, &code, alloc, float_spill_base, cmp.lhs, inst_pos, d, fspill0);
                    const rs2 = try reloadFloat(allocator, &code, alloc, float_spill_base, cmp.rhs, inst_pos, d, fspill1);
                    // Native f16 (Zfh): compare the native halves directly (`feq.h`/`flt.h`/`fle.h`);
                    // the s-form compare would misread the NaN-boxed half's upper bits. SOFTWARE f16
                    // compares its held-as-f32 widening with the s-form, which is exact.
                    const half_native = zfh and isHalf(func, func.valueType(cmp.lhs));
                    const feq = if (half_native) &encode.feq_h else if (d) &encode.feq_d else &encode.feq_s;
                    const flt = if (half_native) &encode.flt_h else if (d) &encode.flt_d else &encode.flt_s;
                    const fle = if (half_native) &encode.fle_h else if (d) &encode.fle_d else &encode.fle_s;
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
                    const rs1 = try reloadInt(allocator, &code, alloc, spill_base, cmp.lhs, inst_pos, spill_scratch0);
                    const rs2 = try reloadInt(allocator, &code, alloc, spill_base, cmp.rhs, inst_pos, spill_scratch1);
                    const rd_loc = intLocationAt(alloc, result, inst_pos);
                    const rd = switch (rd_loc) {
                        .reg => |r| r,
                        .slot => spill_scratch0,
                    };
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
                    switch (rd_loc) {
                        .reg => {},
                        .slot => |slot| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + slot * 8))),
                    }
                },
                .@"if" => |cf| {
                    // Edge arguments require prior critical-edge splitting.
                    if (func.blockArgs(cf.then).len != 0 or func.blockArgs(cf.@"else").len != 0) return error.Unsupported;
                    // Fused compare-and-branch: when the immediately-preceding instruction
                    // is a single-use integer icmp that is exactly this if's condition (the
                    // SAME predicate the icmp case used to skip its materialization), branch
                    // on the icmp's two operands directly instead of re-testing a boolean.
                    // The else edge falls through to the jal as usual, and the fixup carries
                    // the chosen branch encoder + both source registers so the offset patch
                    // re-encodes the right instruction.
                    if (inst_idx >= 1 and fuse_cmp_branch and fusesIntoNextIf(func, block_insts, inst_idx - 1)) {
                        const cmp = func.opcode(block_insts[inst_idx - 1]).icmp;
                        const rl = try reloadInt(allocator, &code, alloc, spill_base, cmp.lhs, inst_pos, spill_scratch0);
                        const rr = try reloadInt(allocator, &code, alloc, spill_base, cmp.rhs, inst_pos, spill_scratch1);
                        const sel = branchFor(cmp.op, isUnsignedInt(func, func.valueType(cmp.lhs)));
                        const rs1 = if (sel.swap) rr else rl;
                        const rs2 = if (sel.swap) rl else rr;
                        try fixups.append(allocator, .{ .index = code.items.len, .target = cf.then.target, .kind = .{ .cbranch = .{ .kind = sel.kind, .rs1 = rs1, .rs2 = rs2 } } });
                        try code.append(allocator, sel.kind.emit(rs1, rs2, 0));
                        try fixups.append(allocator, .{ .index = code.items.len, .target = cf.@"else".target, .kind = .jal });
                        try code.append(allocator, encode.jal(.x0, 0));
                        exited = true;
                        break;
                    }
                    // A spilled condition (e.g. an i1 block param used directly as the branch test)
                    // has no reload path here yet: reject cleanly rather than panic. Resident in
                    // every currently-compiling case (byte-identical there).
                    const cond_reg = switch (intLocationAt(alloc, cf.cond, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
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
                            const d = is64Float(func, func.valueType(arg));
                            const dst = fargReg(float_i);
                            // A spilled float arg reloads into the scratch, then moves to the arg
                            // register (the scratch is never an arg register, so it cannot clobber a
                            // not-yet-placed arg).
                            const src = try reloadFloat(allocator, &code, alloc, float_spill_base, arg, inst_pos, d, fspill0);
                            if (src != dst) try code.append(allocator, if (d) encode.fmv_d(dst, src) else encode.fmv_s(dst, src));
                            float_i += 1;
                        } else if (int_i < 8) {
                            const dst = argReg(int_i);
                            switch (intLocationAt(alloc, arg, inst_pos)) {
                                .reg => |src| try int_moves.append(allocator, .{ .src = src, .dst = dst }),
                                .slot => |slot| try int_spilled.append(allocator, .{ .dst = dst, .off = @intCast(spill_base + slot * 8) }),
                            }
                            int_i += 1;
                        } else {
                            // Stack argument: must be register-resident for now.
                            try int_stack.append(allocator, switch (intLocationAt(alloc, arg, inst_pos)) {
                                .reg => |r| r,
                                .slot => return error.Unsupported,
                            });
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
                            const d = is64Float(func, func.valueType(result));
                            if (alloc.float.get(result)) |rd| {
                                if (rd != .f10) try code.append(allocator, if (d) encode.fmv_d(rd, .f10) else encode.fmv_s(rd, .f10));
                            } else {
                                // Spilled float result: store the incoming fa0 directly to its slot.
                                try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, d, .f10);
                            }
                        } else switch (intLocationAt(alloc, result, inst_pos)) {
                            .reg => |rd| if (rd != .x10) try code.append(allocator, encode.addi(rd, .x10, 0)),
                            .slot => |slot| try code.append(allocator, encode.sd(.x10, .x2, @intCast(spill_base + slot * 8))),
                        }
                    }
                },
                .struct_new => |sn| {
                    const result = func.instResult(inst).?;
                    if (!isVector(func, func.valueType(result))) return error.Unsupported;
                    const fields = func.valueList(sn.fields);
                    if (vpu) {
                        // No VPU lane-insert instruction exists (`fbcx.ps` broadcasts one
                        // scalar to every lane, it does not insert into one), so the only
                        // pack this pass can prove correct is a memory round trip: a plain
                        // scalar `fsw` per field at consecutive 4-byte offsets in the
                        // reserved vpu_pack_base scratch slot, then one `flw.ps` loads all
                        // 8 lanes at once.
                        if (fields.len != 8) return error.Unsupported;
                        if (isIntVector(func, func.valueType(result))) {
                            // Packed-integer pack: the 8 lane scalars are i32 living in the INT
                            // register file (which spills freely via int_spill), so each is stored
                            // with a 32-bit `sw` into the pack scratch. No scalar-float-pool
                            // pressure, unlike the packed-single pack below. A single int scratch
                            // (reused per field) suffices because each field is stored immediately
                            // after it is reloaded.
                            for (fields, 0..) |field, k| {
                                const r = try reloadInt(allocator, &code, alloc, spill_base, field, inst_pos, spill_scratch0);
                                try code.append(allocator, encode.sw(r, .x2, @intCast(vpu_pack_base + k * 4)));
                            }
                        } else {
                            for (fields, 0..) |field, k| {
                                // Each field is stored immediately after it is read, so a single float
                                // spill scratch (reused per field) suffices even when several fields are
                                // spilled at once - the exact case the et-soc VPU SLP path hits.
                                const fr = try reloadFloat(allocator, &code, alloc, float_spill_base, field, inst_pos, false, fspill0);
                                try code.append(allocator, encode.fsw(fr, .x2, @intCast(vpu_pack_base + k * 4)));
                            }
                        }
                        const rd = dstVpuVector(alloc, result, inst_pos, vpu_vec_work);
                        try code.append(allocator, encode.flw_ps(rd, .x2, @intCast(vpu_pack_base)));
                        try storeVpuVector(allocator, &code, alloc, vpu_vspill_base, result, inst_pos, rd);
                    } else {
                        // Pack four scalar floats into a <4 x f32>. Seed lane 0 with the
                        // last field, then slide up inserting the earlier ones. The
                        // slide's vd must not overlap vs2, so alternate result and scratch.
                        if (fields.len != 4) return error.Unsupported;
                        const rd = dstVector(alloc, result, inst_pos, vec_work); // vec_work if the result is spilled
                        // Each field feeds exactly one slide instruction and the four uses are
                        // sequential (never simultaneously live), so a spilled field reloads into a
                        // single float scratch right before its slide. Byte-identical with no spill.
                        const f3 = try reloadFloat(allocator, &code, alloc, float_spill_base, fields[3], inst_pos, false, fspill0);
                        try code.append(allocator, encode.vfmv_s_f(vector_scratch, f3)); // [f3]
                        const f2 = try reloadFloat(allocator, &code, alloc, float_spill_base, fields[2], inst_pos, false, fspill0);
                        try code.append(allocator, encode.vfslide1up_vf(rd, vector_scratch, f2)); // [f2,f3]
                        const f1 = try reloadFloat(allocator, &code, alloc, float_spill_base, fields[1], inst_pos, false, fspill0);
                        try code.append(allocator, encode.vfslide1up_vf(vector_scratch, rd, f1)); // [f1,f2,f3]
                        const f0 = try reloadFloat(allocator, &code, alloc, float_spill_base, fields[0], inst_pos, false, fspill0);
                        try code.append(allocator, encode.vfslide1up_vf(rd, vector_scratch, f0)); // [f0,f1,f2,f3]
                        try storeVector(allocator, &code, alloc, vspill_base, result, inst_pos, rd, spill_scratch1);
                    }
                },
                .extract => |ex| {
                    if (vpu) {
                        if (ex.index >= 8) return error.Unsupported; // fmvs.x.ps takes a u3 lane index
                        const result = func.instResult(inst).?;
                        const vs = try reloadVpuVector(allocator, &code, alloc, vpu_vspill_base, ex.aggregate, inst_pos, vpu_vec_op0);
                        if (isIntVector(func, func.valueType(ex.aggregate))) {
                            // Packed-integer lane extract: `fmvs.x.ps` lands the i32 lane straight
                            // in a GPR, and for an integer vector that GPR IS the result - no
                            // `fmv.w.x` float move. A spilled result stores from the int scratch
                            // (mirrors the int-arith spill store).
                            const rd_loc = intLocationAt(alloc, result, inst_pos);
                            const rd = switch (rd_loc) {
                                .reg => |r| r,
                                .slot => spill_scratch0,
                            };
                            try code.append(allocator, encode.fmvs_x_ps(rd, vs, @intCast(ex.index)));
                            switch (rd_loc) {
                                .reg => {},
                                .slot => |slot| try code.append(allocator, encode.sd(rd, .x2, @intCast(spill_base + slot * 8))),
                            }
                        } else {
                            // Packed-single lane extract: extract to a GPR (`fmvs.x.ps`), then move
                            // its bits into the destination float register (`fmv.w.x`). No direct
                            // VPU-lane-to-FPR move instruction exists, so this two-step,
                            // register-only sequence is the smallest change that stays bit-exact
                            // (fmv.w.x only reads the low 32 bits, so fmvs.x.ps's sign-fill of
                            // the upper 32 is harmless).
                            const d = is64Float(func, func.valueType(result));
                            const rd = dstFloat(alloc, result, inst_pos, fspill0);
                            try code.append(allocator, encode.fmvs_x_ps(spill_scratch0, vs, @intCast(ex.index)));
                            try code.append(allocator, encode.fmv_w_x(rd, spill_scratch0));
                            try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, d, rd);
                        }
                    } else {
                        // Extract a lane to a scalar float. Lane 0 is a direct vfmv.f.s.
                        // a higher lane slides down to lane 0 first.
                        const result = func.instResult(inst).?;
                        const d = is64Float(func, func.valueType(result));
                        const rd = dstFloat(alloc, result, inst_pos, fspill0);
                        const vs = try reloadVector(allocator, &code, alloc, vspill_base, ex.aggregate, inst_pos, vec_op0, spill_scratch1);
                        if (ex.index == 0) {
                            try code.append(allocator, encode.vfmv_f_s(rd, vs));
                        } else {
                            try code.append(allocator, encode.vslidedown_vi(vector_scratch, vs, @intCast(ex.index)));
                            try code.append(allocator, encode.vfmv_f_s(rd, vector_scratch));
                        }
                        try storeFloat(allocator, &code, alloc, float_spill_base, result, inst_pos, d, rd);
                    }
                },
                .matmul => |mmv| {
                    // et-soc tensor matmul: `C(m x n) = A(m x k) @ B(k x n)` over ARBITRARY
                    // compile-time m/n/k, emitted as the pure CSR-write protocol proven correct on
                    // sw-sysemu (the single-tile reference kernel is /tmp/etsoc-build/matmul/mm.s; the
                    // encoders + descriptor packers are in encode.zig). The native output tile is 16
                    // rows x up to 16 cols; the contraction per fma pass (K) is up to 16 fp32 / 32 fp16
                    // / 64 int8 (one 64-byte SCP line holds 16 f32 = 32 f16 = 64 int8). Larger shapes
                    // are handled by a fully compile-time-unrolled tile grid (all straight-line, no
                    // runtime branches, since m/n/k are compile-time constants). A/B element dtype comes
                    // from `mmv.dtype`; C is ALWAYS 32-bit (fp32 accumulators for fp32/fp16, int32 for
                    // int8/uint8), so the fsw.ps readback is dtype-independent. Only the et-soc VPU
                    // model reaches this path; every other model (and non-riscv backend) rejects matmul.
                    if (!vpu) return error.Unsupported; // matmul only lowers under the et-soc tensor unit
                    // An EMBEDDED matmul saves the int temps x28..x31 as part of its clobber set, but
                    // when the function uses software f16 those same registers are reserved as the
                    // f16 convert scratch (see `temp_regs_f16` and `f16_scratch_*`), so a live f16
                    // conversion around the matmul would collide. Reject this combination cleanly; a
                    // non-embedded matmul (standalone/whole-function) is unaffected. See the brief.
                    if (mmv.embedded and uses_f16) return error.Unsupported;
                    // An EMBEDDED matmul saves each clobbered float register with a 32-bit `fsw` (the
                    // et-soc fp32 case, the only width sw-sysemu implements), so a live f64 or 256-bit
                    // VPU vector crossing the op would lose its high bits. Reject cleanly rather than
                    // silently truncate it; fp32-scalar surroundings (the recognizer's target) pass.
                    if (mmv.embedded and functionHasWideFloatValue(func)) return error.Unsupported;
                    const m = mmv.m;
                    const n = mmv.n;
                    const k = mmv.k;
                    // Per-dtype layout. `factor` = elements packed per 4-byte column slot (= K per SCP
                    // line / 16), `elem` = A/B element size in bytes, `tt` = the fma `type` field, `uns`
                    // = tena/tenb_unsigned (uint8 only). Citations: tensors.h `tensor_fma` type field
                    // (fp32=0, fp16->fp32=1, int8->int32=3); sw-sysemu tensors.cpp acols scaling per
                    // dtype (`acols=(field+1)*1|2|4` in tensor_fma32/16a32/ima8a32_execute) and the
                    // signed/unsigned int8 element reads (`ua`/`ub` -> sext8 vs zero-extend, :1499/:1507).
                    const DInfo = struct { tt: encode.TensorType, factor: u32, elem: u32, uns: bool };
                    const di: DInfo = switch (mmv.dtype) {
                        .fp32 => .{ .tt = .fp32, .factor = 1, .elem = 4, .uns = false },
                        .fp16 => .{ .tt = .fp16, .factor = 2, .elem = 2, .uns = false },
                        .int8 => .{ .tt = .int8, .factor = 4, .elem = 1, .uns = false }, // signed: sext8
                        .uint8 => .{ .tt = .int8, .factor = 4, .elem = 1, .uns = true }, // unsigned: zero-extend
                    };
                    // The tensor_quant epilogue only requantizes int32 TenC (fp32/fp16 write fp32
                    // TenC, which the quant transform chain cannot consume). `function.verify`
                    // already rejects a non-int8 dtype paired with a quant, but that is upstream of
                    // this backend; check it again here so a malformed IR that skipped verify fails
                    // cleanly instead of mis-lowering.
                    const has_quant = mmv.quant != null;
                    if (has_quant and di.tt != .int8) return error.Unsupported;
                    // Defensively re-check the per-column scale length here (like the dtype check
                    // above): the per-tile materialization below indexes `scales[ni*TILE+g]` up to
                    // column n-1, so a scale list shorter than n would read out of bounds. verify
                    // already enforces len == n, but isel must fail cleanly (not panic) on malformed
                    // IR that reaches lowering without verify.
                    if (has_quant) switch (mmv.quant.?.scale) {
                        .scalar => {},
                        .per_column => |h| if (func.scaleList(h).len != n) return error.Unsupported,
                    };
                    // Same defensive re-check for the optional per-column bias: the per-tile
                    // materialization below indexes `bias[ni*TILE+g]` up to column n-1, so a bias
                    // list shorter than n would read out of bounds. verify already enforces
                    // len == n when bias is present; this is the isel-side backstop.
                    if (has_quant) if (mmv.quant.?.bias) |bh| if (func.biasList(bh).len != n) return error.Unsupported;
                    // accumulate=true means real `C += A*B`: the tile grid below PRELOADS the existing
                    // fp32 C tile into TenC (f0..) before the fma passes so the first_pass=0 fma computes
                    // `C_initial + A*B`. That preload only makes sense for the fp32-accumulator dtypes
                    // (fp32, and fp16 which also accumulates into an fp32 TenC). Two combinations are out
                    // of scope for this slice and rejected here cleanly instead of mis-lowering:
                    //   - accumulate + quant: the requant epilogue consumes an int32 TenC and writes
                    //     packed bytes, so preloading an fp32 C into it is meaningless. verify.zig also
                    //     forbids this pairing, but isel must fail cleanly on IR that skipped verify.
                    //   - accumulate + int8/uint8 (di.tt == .int8 covers both, since uint8 uses tt int8):
                    //     the int8 path routes TenC through the tenc2rf copy-to-regfile step (bit 23 on
                    //     the last K-tile), which a C preload would have to interleave with. Not done here.
                    if (mmv.accumulate and has_quant) return error.Unsupported;
                    if (mmv.accumulate and di.tt == .int8) return error.Unsupported;
                    // Bounds: nonzero dims, N a multiple of 4 (the fma b_cols field is `cols/4 - 1`), and
                    // K a multiple of `factor` so every fma pass reads a whole packed column group (the
                    // acols field encodes K/factor; a partial group has no representation). M is free.
                    if (m == 0 or n == 0 or k == 0) return error.Unsupported;
                    if (n % 4 != 0) return error.Unsupported;
                    if (k % di.factor != 0) return error.Unsupported; // int8 needs K%4==0, fp16 K%2==0
                    const TILE: u16 = 16; // output rows/cols per tile (arows<=16, bcols in {4,8,12,16})
                    const K_TILE: u16 = @intCast(16 * di.factor); // contraction per fma pass: 16/32/64
                    // u32 tile counts: `m + TILE - 1` in u16 would overflow for m >= 65521, panicking
                    // before the cap below could reject it. Widen so absurd dims fail cleanly.
                    const m_tiles: u32 = (@as(u32, m) + TILE - 1) / TILE;
                    const n_tiles: u32 = (@as(u32, n) + TILE - 1) / TILE;
                    const k_tiles: u32 = (@as(u32, k) + K_TILE - 1) / K_TILE;
                    // CODE-SIZE CAP: compile-time unrolling emits O(m_tiles*n_tiles*k_tiles) tensor
                    // passes (plus per-row staging copies for unaligned dims), so a huge matrix would
                    // blow up the instruction stream. Cap the total tile-pass count; beyond it,
                    // runtime-loop tiling (a deferred follow-up) is needed, so reject cleanly. The
                    // product is computed in u64 so it cannot overflow before the cap rejects it.
                    if (@as(u64, m_tiles) * n_tiles * k_tiles > 64) return error.Unsupported;

                    // The lowering clobbers a fixed set of scratch registers: x6 (the reserved
                    // descriptor scratch), x31 (load stride|id + address/lane temp), x5 (sub-tile
                    // pointer), x7 (staging word copy), x28 (64-aligned staging base). The a/b/c
                    // pointers must be register-resident (the raw `alloc.int.get`); how they must
                    // relate to the scratch set differs by embedded-ness and is handled just below.
                    const a_reg = switch (intLocationAt(alloc, mmv.a, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const b_reg = switch (intLocationAt(alloc, mmv.b, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const c_reg = switch (intLocationAt(alloc, mmv.c, inst_pos)) {
                        .reg => |r| r,
                        .slot => return error.Unsupported,
                    };
                    const desc = scratch_reg; // x6, the descriptor-build scratch
                    const addr_scratch: Reg = .x5; // sub-tile / source-row pointer
                    const copy_tmp: Reg = .x7; // staging word-copy temp
                    const stage_ptr: Reg = .x28; // 64-byte-aligned staging base
                    const stride_reg: Reg = .x31; // load stride|id, and store address/lane temp
                    // NON-EMBEDDED (standalone/whole-function) matmul: a/b/c must already live outside
                    // the scratch set. They arrive in arg registers a0..a2 and never move, so this
                    // holds in practice; a conflicting allocation is rejected rather than clobbered.
                    // Embedded matmul drops this gate: it copies a/b/c into the dedicated holder
                    // registers below (saved/restored around the op), so their starting placement,
                    // even inside the scratch set, cannot be clobbered.
                    if (!mmv.embedded) {
                        for ([_]Reg{ a_reg, b_reg, c_reg }) |r| {
                            if (r == desc or r == addr_scratch or r == copy_tmp or r == stage_ptr or r == stride_reg) return error.Unsupported;
                        }
                    }
                    // The A/B/C base pointers used throughout the tile grid below: the holder registers
                    // for an embedded matmul (stable across every clobber), the raw allocated registers
                    // otherwise. For a non-embedded matmul these are exactly a_reg/b_reg/c_reg, so the
                    // emitted bytes are byte-identical to before this field existed.
                    const base_a = if (mmv.embedded) matmul_holder_a else a_reg;
                    const base_b = if (mmv.embedded) matmul_holder_b else b_reg;
                    const base_c = if (mmv.embedded) matmul_holder_c else c_reg;

                    // Staging (a stack scratch is needed) when: A's memory row pitch k*elem is not a
                    // multiple of 64 (`emitMatmulLoadSubtile` can't do a direct strided load); OR B
                    // needs it - fp32 B stages iff n*4 is not 64-aligned (n%16!=0), and fp16/int8 B
                    // ALWAYS stages because its SCP layout is the K-interleaved transpose-pack
                    // (`emitMatmulLoadBPacked`), never a direct copy of row-major memory.
                    const needs_stage = ((@as(u64, k) * di.elem) % 64 != 0) or (di.factor > 1) or (n % 16 != 0);

                    // The float registers this op clobbers: TenC row i lives in f(2i)/f(2i+1) and the
                    // widest tile has min(16, m) rows, so f0..f(2*min(16,m)-1) are written by the fma
                    // readback. An embedded matmul saves exactly these on entry and restores them on
                    // exit with 32-bit `fsw`/`flw` (see `matmul_save_float_bytes`): the low 32 bits
                    // are the whole value for the fp32 TenC and every et-soc scalar float (the
                    // validated case), and they are what the sw-sysemu oracle supports (it lacks the
                    // 64-bit `fld`/`fsd` doubleword forms). A live f64 or 256-bit VPU vector across an
                    // embedded matmul would need a wider save, so `functionHasWideFloatValue` rejects
                    // that combination above rather than truncating the high bits.
                    // (@min narrows to a 0..16 type, so widen to u16 BEFORE the *2 or it overflows.)
                    const fsave_cnt: u16 = 2 * @as(u16, @min(@as(u16, 16), m));

                    // Embedded save prologue: preserve every register this op clobbers, then relocate
                    // a/b/c into the holder registers via the stack (a memory round-trip is immune to
                    // aliasing between the a/b/c source registers and the holder destinations). Slot
                    // offsets follow `matmul_save_base`'s documented layout.
                    if (mmv.embedded) {
                        const sb: i12 = @intCast(matmul_save_base);
                        // 1. clobbered int scratch temps (x6 is reserved-never-allocated, so no value
                        // is ever live in it and it needs no save).
                        try code.append(allocator, encode.sd(addr_scratch, .x2, sb + 0)); // x5
                        try code.append(allocator, encode.sd(copy_tmp, .x2, sb + 8)); // x7
                        try code.append(allocator, encode.sd(stage_ptr, .x2, sb + 16)); // x28
                        try code.append(allocator, encode.sd(stride_reg, .x2, sb + 24)); // x31
                        // 2. the holder registers' incoming (possibly live-across) values.
                        try code.append(allocator, encode.sd(matmul_holder_a, .x2, sb + 32)); // x29
                        try code.append(allocator, encode.sd(matmul_holder_b, .x2, sb + 40)); // x30
                        try code.append(allocator, encode.sd(matmul_holder_c, .x2, sb + 48)); // x9
                        // 3. capture the a/b/c pointers into transfer slots (sources still intact:
                        // nothing above overwrote an allocatable register).
                        try code.append(allocator, encode.sd(a_reg, .x2, sb + 56));
                        try code.append(allocator, encode.sd(b_reg, .x2, sb + 64));
                        try code.append(allocator, encode.sd(c_reg, .x2, sb + 72));
                        // 4. clobbered float TenC registers f0..f(fsave_cnt-1), 32-bit fsw each.
                        const fbase: i12 = @intCast(matmul_save_base + matmul_save_int_bytes);
                        var fi: u16 = 0;
                        while (fi < fsave_cnt) : (fi += 1) {
                            try code.append(allocator, encode.fsw(@enumFromInt(@as(u5, @intCast(fi))), .x2, fbase + @as(i12, @intCast(fi * 4))));
                        }
                        // 5. load a/b/c into the holders (via memory, so no aliasing hazard).
                        try code.append(allocator, encode.ld(matmul_holder_a, .x2, sb + 56));
                        try code.append(allocator, encode.ld(matmul_holder_b, .x2, sb + 64));
                        try code.append(allocator, encode.ld(matmul_holder_c, .x2, sb + 72));
                    }

                    // Enable all 8 packed lanes: M0 = 0xff, needed for the fsw.ps readback below
                    // (mm.s line 17). The general vpu mask preamble only fires when the function has
                    // vpu vector values, which a matmul-only function does not, so emit it here. A
                    // duplicate write in a mixed function is a harmless idempotent re-set of M0.
                    try code.append(allocator, encode.mov_m_x(0, .x0, 0xFF));

                    // Enable the L1 scratchpad: mcache_control 0 -> 1 -> 3 (mm.s lines 10-14). Writing
                    // 3 directly is a silent no-op, so the two-step sequence is mandatory. Done once.
                    try loadImm32(allocator, &code, desc, 1);
                    try code.append(allocator, encode.csrw(encode.CSR_MCACHE_CONTROL, desc));
                    try loadImm32(allocator, &code, desc, 3);
                    try code.append(allocator, encode.csrw(encode.CSR_MCACHE_CONTROL, desc));

                    // Compute the 64-byte-aligned staging base once (if any load stages, or quant
                    // needs it for the replicated-scale staging line below, or an accumulate preload
                    // needs it to stage a 4-col C remainder): round sp + matmul_stage_base up to 64.
                    // The reserved region has 63 bytes of slack. `mmv.accumulate and !has_quant` is the
                    // exact gate the C-tile preload uses below, so stage_ptr is valid whenever that
                    // preload's staged-remainder path can fire (accumulate+quant was rejected above, so
                    // this reduces to plain accumulate here, but the explicit form keeps the two gates
                    // textually identical).
                    if (needs_stage or has_quant or (mmv.accumulate and !has_quant)) {
                        try code.append(allocator, encode.addi(stage_ptr, .x2, @intCast(matmul_stage_base)));
                        try code.append(allocator, encode.addi(stage_ptr, stage_ptr, 63));
                        try code.append(allocator, encode.andi(stage_ptr, stage_ptr, -64));
                    }

                    // BASE SCP line for the matmul-quant epilogue's inputs. L1 SCP has 48 lines
                    // (0..47); A/B tiles use at most lines 0..31 (A occupies 0..rows-1, B occupies
                    // rows..rows+15), so line 40 never collides with a live A/B tile. Up to 3
                    // consecutive lines starting here (40, 41, 42) hold, in READ ORDER, the
                    // optional per-column bias, the scale (scalar-replicated or per-column), and
                    // the optional per-tensor zero-point (also replicated): `packTensorQuant`'s
                    // `scp_loc` field names only this base line, and the tensor unit auto-advances
                    // it by one after each SCP-reading transform in the chain, so the chain order
                    // below must match the load order here exactly.
                    const QUANT_SCP: u6 = 40;

                    // Per-operand signedness: `mmv.input_signs`, when present, overrides di.uns
                    // independently for A and B (mixed uint8-A x int8-B and vice versa). verify.zig
                    // only allows input_signs paired with dtype == .int8, so di.tt is always .int8
                    // here and every other dtype-derived field above (factor/elem/K-tiling/B
                    // transpose-pack/tenc2rf) is unaffected; only the two `ua`/`ub` bits packed into
                    // the fma descriptor change. Constant for the whole matmul, hoisted above the tile grid.
                    const a_uns = if (mmv.input_signs) |s| s.a_unsigned else di.uns;
                    const b_uns = if (mmv.input_signs) |s| s.b_unsigned else di.uns;

                    // Compile-time-unrolled tile grid: for each output tile (mi, ni), accumulate the K
                    // slices (ki) into TenC (f0..f31, row i in f(2i)/f(2i+1)), then store the tile.
                    var mi: u16 = 0;
                    while (mi < m_tiles) : (mi += 1) {
                        const rows = @min(TILE, m - mi * TILE); // output rows in this tile
                        var ni: u16 = 0;
                        while (ni < n_tiles) : (ni += 1) {
                            const cols = @min(TILE, n - ni * TILE); // output cols (a multiple of 4)
                            std.debug.assert(cols % 4 == 0);

                            // C-TILE PRELOAD for accumulate=true (fp32/fp16 only, non-quant). TenC row
                            // i lives in f(2i)/f(2i+1). The (ki==0) fma below runs with first_pass=0
                            // whenever accumulate is set, so it computes `TenC += A*B` onto WHATEVER
                            // TenC holds. Loading the existing C tile into those same FREGS first is
                            // what turns the op into real `C += A*B`. This is the exact REVERSE of the
                            // non-quant C-store below (full 8-col groups <-> fsw.ps, 4-col remainder <->
                            // scalar lane stores), and it is fully gated so accumulate=false emits ZERO
                            // preload bytes (byte-identical to before this field had real semantics).
                            // fp16 needs no special case: a fp16-INPUT matmul writes an fp32 C tile (the
                            // accumulator is fp32 in FREGS), so its C in memory is fp32 and this exact
                            // fp32 flw.ps path preloads it.
                            if (mmv.accumulate and !has_quant) {
                                const full_groups = cols / 8;
                                const has_rem = (cols % 8) == 4;
                                var i: u16 = 0;
                                while (i < rows) : (i += 1) {
                                    // Same C row address the store builds: R = mi*TILE+i, col base
                                    // ni*TILE, byte stride n*4. desc (x6) = base_c + byte offset.
                                    const c_off = (@as(u64, mi) * TILE + i) * n * 4 + @as(u64, ni) * TILE * 4;
                                    try loadImm64(allocator, &code, stride_reg, c_off);
                                    try code.append(allocator, encode.add(desc, base_c, stride_reg));
                                    // Full 8-col groups: one flw.ps each (256-bit, 8 valid f32 = 32
                                    // bytes, no overhang since a full group is 8 valid columns), into
                                    // f(2i+g), the SAME reg the store's fsw.ps reads back.
                                    var g: u16 = 0;
                                    while (g < full_groups) : (g += 1) {
                                        const freg: encode.FReg = @enumFromInt(@as(u5, @intCast(i * 2 + g)));
                                        try code.append(allocator, encode.flw_ps(freg, desc, @intCast(g * 32)));
                                    }
                                    if (has_rem) {
                                        // 4-col remainder: a direct 8-lane flw.ps here would read 4
                                        // valid C words PLUS 4 words past this C row (into the next row,
                                        // or past C's end at the last row) = a potential page fault. So
                                        // STAGE the 4 valid words into the 64-aligned scratch (scalar
                                        // lw/sw, in-bounds) and flw.ps all 8 lanes from the scratch. The
                                        // upper 4 lanes read staging leftovers, which is harmless: the
                                        // 4-col fma only computes lanes 0..3 and the 4-col store only
                                        // writes lanes 0..3, so the leftover upper lanes are never used.
                                        const rem_freg: encode.FReg = @enumFromInt(@as(u5, @intCast(i * 2 + full_groups)));
                                        const base_col = full_groups * 8;
                                        var lane: u16 = 0;
                                        while (lane < 4) : (lane += 1) {
                                            try code.append(allocator, encode.lw(copy_tmp, desc, @intCast((base_col + lane) * 4)));
                                            try code.append(allocator, encode.sw(copy_tmp, stage_ptr, @intCast(lane * 4)));
                                        }
                                        try code.append(allocator, encode.flw_ps(rem_freg, stage_ptr, 0));
                                    }
                                }
                            }

                            var ki: u16 = 0;
                            while (ki < k_tiles) : (ki += 1) {
                                const kslice = @min(K_TILE, k - ki * K_TILE); // contracted dim this pass (multiple of factor)

                                // Load A sub-tile: `rows` rows x `kslice` elements, real A row pitch
                                // k*elem, base = a + (mi*TILE)*(k*elem) + (ki*K_TILE)*elem, row-major
                                // into SCP lines 0..rows-1 (A is row-major for every dtype).
                                const a_off = @as(u64, mi) * TILE * k * di.elem + @as(u64, ki) * K_TILE * di.elem;
                                try emitMatmulLoadSubtile(allocator, &code, base_a, a_off, @as(u64, k) * di.elem, rows, kslice, di.elem, 0, 0, stage_ptr, addr_scratch, copy_tmp, desc, stride_reg);

                                // Load B sub-tile into SCP lines rows.. . fp32: one row per line
                                // (row-major, `kslice` lines). fp16/int8: the K-interleaved
                                // transpose-pack (`kslice/factor` lines). B memory base = b +
                                // (ki*K_TILE)*(n*elem) + (ni*TILE)*elem, pitch n*elem.
                                const b_off = @as(u64, ki) * K_TILE * n * di.elem + @as(u64, ni) * TILE * di.elem;
                                if (di.factor == 1) {
                                    try emitMatmulLoadSubtile(allocator, &code, base_b, b_off, @as(u64, n) * di.elem, kslice, cols, di.elem, @intCast(rows), 1, stage_ptr, addr_scratch, copy_tmp, desc, stride_reg);
                                } else {
                                    try emitMatmulLoadBPacked(allocator, &code, base_b, b_off, @as(u64, n) * di.elem, di.elem, di.factor, kslice, cols, @intCast(rows), 1, stage_ptr, addr_scratch, copy_tmp, desc, stride_reg);
                                }

                                // tensor_fma: reads A from SCP line 0, B from SCP line `rows`. The
                                // a_cols field is K/factor (tensors.cpp scales it back by `factor`);
                                // type + tena/tenb_unsigned come from the dtype. first_pass is set on the
                                // first K slice (fresh TenC = A*B); later slices accumulate (TenC += A*B)
                                // into the same registers. An accumulate op forces the first slice to
                                // accumulate onto whatever TenC held (true C-memory accumulation is unused).
                                const first_pass = (ki == 0) and !mmv.accumulate;
                                // packTensorFma's a_cols param is the pre-decrement count (it packs
                                // a_cols-1 into the field). Passing kslice/factor makes the field
                                // kslice/factor-1, which tensors.cpp scales as (field+1)*factor = kslice.
                                const a_cols_arg: u5 = @intCast(kslice / di.factor);
                                // bit 23 (packTensorFma's `tenc_in_mem`) is REINTERPRETED by the int8
                                // path: tensors.cpp `tensor_ima8a32_execute` accumulates into the
                                // internal TenC and copies it to the vector regfile (FREGS, which the
                                // fsw.ps readback reads) only when this bit ("tenc2rf") is set on the
                                // LAST internal K iteration. fp32/fp16 write FREGS directly and ignore
                                // the bit. So set it for int8/uint8 on the final K-tile only (0 on
                                // intermediate tiles so they keep accumulating in TenC); fp32/fp16 leave
                                // it 0 (fp32 thus stays byte-identical).
                                const tenc_to_rf = (di.tt == .int8) and (ki == k_tiles - 1);
                                const fma_desc = encode.packTensorFma(di.tt, @intCast(rows), a_cols_arg, @intCast(cols), 0, 0, @intCast(rows), tenc_to_rf, a_uns, b_uns, first_pass);
                                try loadImm64(allocator, &code, desc, fma_desc);
                                try code.append(allocator, encode.csrw(encode.CSR_TENSOR_FMA, desc));
                                try loadImm32(allocator, &code, desc, @intCast(encode.TENSOR_WAIT_FMA));
                                try code.append(allocator, encode.csrw(encode.CSR_TENSOR_WAIT, desc));
                            }

                            if (has_quant) {
                                // Materialize + load this tile's quant inputs into consecutive SCP
                                // lines starting at QUANT_SCP, in READ ORDER (bias, then scale, then
                                // zero-point): the tensor unit auto-advances its internal scp_loc by
                                // one after every SCP-reading transform in the chain built below, so
                                // whichever transform reads a given line must be the Nth one in the
                                // chain if this line is loaded Nth here. Uniform per-tile loading for
                                // BOTH scalar and per_column scale (the scalar case now redundantly
                                // re-replicates the same bits every tile - a tiny, harmless waste of
                                // instructions, not correctness) keeps this block a single, simple
                                // shape instead of a pre-loop/per-tile split.
                                const q = mmv.quant.?;
                                var scp_line: u6 = QUANT_SCP;
                                // 1. bias (per-column int32), if present: column (ni*TILE + g)'s bias
                                // in slot g, one line, read by i32_add_row before the scale.
                                if (q.bias) |bh| {
                                    const bias = func.biasList(bh);
                                    var g: u16 = 0;
                                    while (g < cols) : (g += 1) {
                                        try loadImm32(allocator, &code, desc, @bitCast(bias[@as(usize, ni) * TILE + g]));
                                        try code.append(allocator, encode.sw(desc, stage_ptr, @intCast(g * 4)));
                                    }
                                    try emitQuantScpLineLoad(allocator, &code, scp_line, stage_ptr, desc, stride_reg);
                                    scp_line += 1;
                                }
                                // 2. scale: scalar broadcasts one fp32 bit pattern to all 16 slots;
                                // per_column loads this tile's `cols` scales, column-indexed like bias.
                                switch (q.scale) {
                                    .scalar => |scale_bits| {
                                        var w: u16 = 0;
                                        while (w < 16) : (w += 1) {
                                            try loadImm32(allocator, &code, desc, scale_bits);
                                            try code.append(allocator, encode.sw(desc, stage_ptr, @intCast(w * 4)));
                                        }
                                    },
                                    .per_column => |h| {
                                        const scales = func.scaleList(h); // n words, one fp32-bit scale per output column
                                        var g: u16 = 0;
                                        while (g < cols) : (g += 1) {
                                            try loadImm32(allocator, &code, desc, scales[@as(usize, ni) * TILE + g]);
                                            try code.append(allocator, encode.sw(desc, stage_ptr, @intCast(g * 4)));
                                        }
                                    },
                                }
                                try emitQuantScpLineLoad(allocator, &code, scp_line, stage_ptr, desc, stride_reg);
                                scp_line += 1;
                                // 3. zero-point (per-tensor int32), if nonzero: replicated across all
                                // 16 slots (it is the same value for every column), read by the second
                                // i32_add_row after the int32 requantize.
                                if (q.zero_point != 0) {
                                    var w: u16 = 0;
                                    while (w < 16) : (w += 1) {
                                        try loadImm32(allocator, &code, desc, @bitCast(q.zero_point));
                                        try code.append(allocator, encode.sw(desc, stage_ptr, @intCast(w * 4)));
                                    }
                                    try emitQuantScpLineLoad(allocator, &code, scp_line, stage_ptr, desc, stride_reg);
                                    scp_line += 1;
                                }
                            }

                            if (has_quant) {
                                // Requantize this tile's int32 TenC in place to a packed byte:
                                // (bias?) -> (relu?) -> *scale -> round -> (zero_point?) ->
                                // sat[u]int8 -> pack. start_reg is 0 because every tile's TenC is
                                // f0-based (the fma above always writes f0..). The col/row fields
                                // encode this tile's cols/rows; scp_loc is QUANT_SCP, the first of
                                // the (up to 3) lines loaded just above, matched in order.
                                const q = mmv.quant.?;
                                var chain = [_]encode.QuantTransform{.last} ** 10;
                                var ci: usize = 0;
                                if (q.bias != null) { // reads SCP[QUANT_SCP]
                                    chain[ci] = .i32_add_row;
                                    ci += 1;
                                }
                                if (q.relu) {
                                    chain[ci] = .i32_relu;
                                    ci += 1;
                                }
                                chain[ci] = .i32_to_f32;
                                ci += 1;
                                chain[ci] = .fp32_mul_row; // reads the next SCP line (the scale)
                                ci += 1;
                                chain[ci] = .f32_to_i32;
                                ci += 1;
                                if (q.zero_point != 0) { // reads the next SCP line (the zero-point)
                                    chain[ci] = .i32_add_row;
                                    ci += 1;
                                }
                                // Signed int8 or unsigned uint8 output: same pack step either way,
                                // only the saturating clamp range differs.
                                chain[ci] = switch (q.out) {
                                    .i8 => .satint8,
                                    .u8 => .satuint8,
                                };
                                ci += 1;
                                chain[ci] = .pack_128b;
                                ci += 1;
                                const col_field: u2 = @intCast(cols / 4 - 1);
                                const row_field: u4 = @intCast(rows - 1);
                                const qdesc = encode.packTensorQuant(0, col_field, row_field, QUANT_SCP, chain);
                                try loadImm64(allocator, &code, desc, qdesc);
                                try code.append(allocator, encode.csrw(encode.CSR_TENSOR_QUANT, desc));
                                try loadImm32(allocator, &code, desc, @intCast(encode.TENSOR_WAIT_QUANT));
                                try code.append(allocator, encode.csrw(encode.CSR_TENSOR_WAIT, desc));
                            }

                            if (has_quant) {
                                // 8-bit output (signed int8 or unsigned uint8, per quant.out): C is
                                // row-major bytes (stride n). After pack_128b, row i's `cols` results
                                // are packed in the low `cols` bytes of the EVEN reg f(2i) = cols/4
                                // words (lanes 0..cols/4-1). The store is byte-identical for either
                                // signedness (bytes are bytes). Extract each word and store it. cols
                                // is always a multiple of 4, so there is no sub-word remainder.
                                var i: u16 = 0;
                                while (i < rows) : (i += 1) {
                                    const c_off = (@as(u64, mi) * TILE + i) * n + @as(u64, ni) * TILE; // BYTES, int8 stride n
                                    try loadImm64(allocator, &code, stride_reg, c_off);
                                    try code.append(allocator, encode.add(desc, base_c, stride_reg));
                                    const freg: encode.FReg = @enumFromInt(@as(u5, @intCast(i * 2))); // even reg holds the packed row
                                    var g: u16 = 0;
                                    while (g < cols / 4) : (g += 1) {
                                        try code.append(allocator, encode.fmvs_x_ps(stride_reg, freg, @intCast(g)));
                                        try code.append(allocator, encode.sw(stride_reg, desc, @intCast(g * 4)));
                                    }
                                }
                            } else {
                                // Store the completed output tile from TenC to real row-major C. TenC row i
                                // is f(2i) [cols 0..7] and f(2i+1) [cols 8..15]. Each full 8-column group is
                                // one fsw.ps (exactly 8 f32 = 32 bytes, no overhang since all 8 are valid);
                                // a trailing 4-column remainder (cols % 8 == 4) is written with 4 scalar
                                // lane stores, because fsw.ps always writes 8 lanes and would clobber the
                                // next row / past the end of C. c row R = mi*TILE+i, col base = ni*TILE.
                                const full_groups = cols / 8;
                                const has_rem = (cols % 8) == 4;
                                var i: u16 = 0;
                                while (i < rows) : (i += 1) {
                                    const c_off = (@as(u64, mi) * TILE + i) * n * 4 + @as(u64, ni) * TILE * 4;
                                    // Build the row's C address into `desc` (x6). x31 holds the offset first.
                                    try loadImm64(allocator, &code, stride_reg, c_off);
                                    try code.append(allocator, encode.add(desc, base_c, stride_reg));
                                    var g: u16 = 0;
                                    while (g < full_groups) : (g += 1) {
                                        const freg: encode.FReg = @enumFromInt(@as(u5, @intCast(i * 2 + g)));
                                        try code.append(allocator, encode.fsw_ps(freg, desc, @intCast(g * 32)));
                                    }
                                    if (has_rem) {
                                        const rem_freg: encode.FReg = @enumFromInt(@as(u5, @intCast(i * 2 + full_groups)));
                                        const base_col = full_groups * 8;
                                        var lane: u16 = 0;
                                        while (lane < 4) : (lane += 1) {
                                            try code.append(allocator, encode.fmvs_x_ps(stride_reg, rem_freg, @intCast(lane)));
                                            try code.append(allocator, encode.sw(stride_reg, desc, @intCast((base_col + lane) * 4)));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Embedded restore epilogue: the tile grid has finished writing C, so restore
                    // every register saved above, leaving the whole register file exactly as the op
                    // found it (a/b/c holders last, back to their incoming values). Mirrors the save
                    // prologue's layout. M0 is not restored: it is a function-wide 0xFF invariant (the
                    // vpu preamble sets it) that this op only ever re-sets to the same 0xFF.
                    if (mmv.embedded) {
                        const sb: i12 = @intCast(matmul_save_base);
                        const fbase: i12 = @intCast(matmul_save_base + matmul_save_int_bytes);
                        var fi: u16 = 0;
                        while (fi < fsave_cnt) : (fi += 1) {
                            try code.append(allocator, encode.flw(@enumFromInt(@as(u5, @intCast(fi))), .x2, fbase + @as(i12, @intCast(fi * 4))));
                        }
                        try code.append(allocator, encode.ld(addr_scratch, .x2, sb + 0)); // x5
                        try code.append(allocator, encode.ld(copy_tmp, .x2, sb + 8)); // x7
                        try code.append(allocator, encode.ld(stage_ptr, .x2, sb + 16)); // x28
                        try code.append(allocator, encode.ld(stride_reg, .x2, sb + 24)); // x31
                        try code.append(allocator, encode.ld(matmul_holder_a, .x2, sb + 32)); // x29
                        try code.append(allocator, encode.ld(matmul_holder_b, .x2, sb + 40)); // x30
                        try code.append(allocator, encode.ld(matmul_holder_c, .x2, sb + 48)); // x9
                    }
                },
                else => return error.Unsupported,
            }
        }

        // After the instruction loop `term_pos` is the terminator position, exactly where the
        // allocator numbers a jump/ret terminator's operands. An `if` terminator is one of the
        // instructions above (it set `exited`), so this only positions ret/jump.
        const term_pos = first_inst_pos + block_insts.len;
        // Drain any split-boundary actions recorded AT the terminator position BEFORE emitting the
        // terminator. `secondChance` can re-home a value used only by `ret` (a non-edge-arg operand,
        // hence `is_intra`) at its next use, which is the terminator position (`block_end`), recording
        // a `.reload` there. The per-instruction drain above only reaches `term_pos - 1`, so without
        // this drain the reload is never emitted and `ret` reads a register that was never loaded. When
        // no action lands on a terminator (the normal case) this drains nothing and is byte-identical.
        while (action_cursor < alloc.actions.items.len and alloc.actions.items[action_cursor].at <= term_pos) {
            const act = alloc.actions.items[action_cursor];
            std.debug.assert(act.at == term_pos); // only terminator-position actions remain here
            try emitSplitAction(allocator, &code, func, spill_base, float_spill_base, vspill_base, vpu_vspill_base, vpu_pack_base, act);
            action_cursor += 1;
        }
        if (!exited) {
            switch (func.terminator(block) orelse return error.Unsupported) {
                .ret => |value| {
                    if (value) |v| {
                        if (isFloat(func, func.valueType(v))) {
                            // fmv fa0, freg  (skipped when already in fa0). A spilled return value
                            // reloads from its slot into the scratch first.
                            const d = is64Float(func, func.valueType(v));
                            const fr = try reloadFloat(allocator, &code, alloc, float_spill_base, v, term_pos, d, fspill0);
                            if (fr != .f10) try code.append(allocator, if (d) encode.fmv_d(.f10, fr) else encode.fmv_s(.f10, fr));
                        } else {
                            // mv a0, reg  (skipped when already in a0)
                            const r = try reloadInt(allocator, &code, alloc, spill_base, v, term_pos, spill_scratch0);
                            if (r != .x10) try code.append(allocator, encode.addi(.x10, r, 0));
                        }
                    }
                    // Epilogue: restore ra and the callee-saved registers, close the frame.
                    if (non_leaf) try code.append(allocator, encode.ld(.x1, .x2, ra_off)); // restore ra
                    for (used_saved.items) |sv| try code.append(allocator, encode.ld(sv.reg, .x2, sv.off));
                    // Restore the callee-saved floats with the width they were saved with: flw in vpu
                    // mode (fp32 only, sw-sysemu has no 64-bit fld), fld otherwise (an f64 may live there).
                    for (used_float_saved.items) |sv| try code.append(allocator, if (vpu) encode.flw(sv.reg, .x2, sv.off) else encode.fld(sv.reg, .x2, sv.off));
                    if (frame_size != 0) try code.append(allocator, encode.addi(.x2, .x2, frame_size));
                    try code.append(allocator, encode.jalr(.x0, .x1, 0)); // ret
                },
                .jump => |j| jump_blk: {
                    // Shared Wimmer path: the allocator already RESOLVED this edge into an ordered
                    // parallel-move sequence (params, live-through values, spills, and cycles), so
                    // replay it op-by-op and derive nothing, then the jal. `edge_move_driven` is false
                    // for every default caller, so the derivation below runs unchanged there.
                    if (alloc.edge_move_driven) {
                        try emitEdgeMoves(allocator, &code, alloc, spill_base, float_spill_base, vspill_base, vpu_vspill_base, vpu_pack_base, block, j.target);
                        try fixups.append(allocator, .{ .index = code.items.len, .target = j.target, .kind = .jal });
                        try code.append(allocator, encode.jal(.x0, 0));
                        break :jump_blk;
                    }
                    // Move each argument into its block parameter's register before the jump. The
                    // edge's (arg_reg -> param_reg) moves can form a permutation CYCLE (a loop
                    // header whose back-edge permutes its carried values - e.g. a swap x10<->x11 -
                    // has the header's fixed param registers feeding back into themselves reordered).
                    // A naive in-order emit would clobber a value mid-cycle, so the moves are
                    // gathered per register class and realized by `parallelMove*` (which breaks
                    // cycles through a reserved scratch). The classes use disjoint register files, so
                    // cycles never cross a class boundary and each class is shuffled independently.
                    const args = func.blockArgs(j);
                    const params = func.blockParams(j.target);

                    // Integers, like floats and vectors, may now spill on either side of the edge
                    // (a non-entry int block param that could not get a register - see the param
                    // allocation above). So the int class is split the same way: reg->reg moves go
                    // through `parallelMoveInt`, while a spilled arg feeding a register param reloads
                    // and any arg feeding a spilled param stores, ordered around the parallel move.
                    var int_moves: std.ArrayList(RegMove) = .empty;
                    defer int_moves.deinit(allocator);
                    // A spilled int arg feeding a register param reloads AFTER the reg->reg move;
                    // any arg feeding a spilled int param stores BEFORE it (while arg registers still
                    // hold their edge values). Mirrors the float/vector split below.
                    var int_reloads: std.ArrayList(struct { arg: Value, dst: Reg }) = .empty;
                    defer int_reloads.deinit(allocator);
                    var int_stores: std.ArrayList(struct { arg: Value, off: i12 }) = .empty;
                    defer int_stores.deinit(allocator);
                    var float_moves: std.ArrayList(Move(FReg)) = .empty;
                    defer float_moves.deinit(allocator);
                    var vec_moves: std.ArrayList(Move(VReg)) = .empty;
                    defer vec_moves.deinit(allocator);
                    // Spilled-float edges, ordered exactly like the spilled-vector edges below: a
                    // spilled arg feeding a register param reloads; any arg feeding a spilled param
                    // stores. Stores read arg registers first (before the reg->reg parallel move
                    // clobbers them), then the reg->reg move, then reloads write the final param regs.
                    var float_reloads: std.ArrayList(struct { arg: Value, dst: FReg, d64: bool }) = .empty;
                    defer float_reloads.deinit(allocator);
                    var float_stores: std.ArrayList(struct { arg: Value, off: i12, d64: bool }) = .empty;
                    defer float_stores.deinit(allocator);
                    // Spilled-vector edges (rare: the block-local vectorizer keeps vectors in-block).
                    // A spilled arg feeding a register param reloads; any arg feeding a spilled param
                    // stores. Neither forms a cycle (distinct arg/param slots), so they are ordered
                    // safely around the reg->reg move: stores read arg registers first (before the
                    // reg->reg moves clobber them), then the reg->reg parallel move, then reloads
                    // write the now-final param registers.
                    var vec_reloads: std.ArrayList(struct { arg: Value, dst: VReg }) = .empty;
                    defer vec_reloads.deinit(allocator);
                    var vec_stores: std.ArrayList(struct { arg: Value, off: i12 }) = .empty;
                    defer vec_stores.deinit(allocator);

                    for (args, params) |arg, param| {
                        if (isVector(func, func.valueType(arg))) {
                            if (vpu) {
                                // A VPU vector carried across a block edge needs a
                                // register-move/spill-store sequence this pass does not
                                // implement yet (unlike RVV's vmv.v.v, there is no single
                                // whole-vector move instruction in the VPU set, and every
                                // real vectorizer output today stays within one block
                                // anyway). Reject cleanly rather than guess at a move
                                // sequence with zero execution feedback.
                                return error.Unsupported;
                            }
                            const arg_reg = alloc.vector.get(arg);
                            if (alloc.vector.get(param)) |pr| {
                                if (arg_reg) |ar| {
                                    try vec_moves.append(allocator, .{ .src = ar, .dst = pr });
                                } else {
                                    try vec_reloads.append(allocator, .{ .arg = arg, .dst = pr });
                                }
                            } else {
                                const off: i12 = @intCast(vspill_base + alloc.vector_spill.get(param).? * 16);
                                try vec_stores.append(allocator, .{ .arg = arg, .off = off });
                            }
                        } else if (isFloat(func, func.valueType(arg))) {
                            const d64 = is64Float(func, func.valueType(arg));
                            if (alloc.float.get(param)) |pr| {
                                if (alloc.float.get(arg)) |ar| {
                                    try float_moves.append(allocator, .{ .src = ar, .dst = pr });
                                } else {
                                    try float_reloads.append(allocator, .{ .arg = arg, .dst = pr, .d64 = d64 });
                                }
                            } else {
                                const off: i12 = @intCast(float_spill_base + alloc.float_spill.get(param).? * 8);
                                try float_stores.append(allocator, .{ .arg = arg, .off = off, .d64 = d64 });
                            }
                        } else {
                            // The destination param reads stay direct (a param is never split). The
                            // ARG source is read through `intLocationAt` at the terminator position.
                            if (alloc.int.get(param)) |pr| {
                                switch (intLocationAt(alloc, arg, term_pos)) {
                                    .reg => |ar| try int_moves.append(allocator, .{ .src = ar, .dst = pr }),
                                    .slot => try int_reloads.append(allocator, .{ .arg = arg, .dst = pr }),
                                }
                            } else {
                                const off: i12 = @intCast(spill_base + alloc.int_spill.get(param).? * 8);
                                try int_stores.append(allocator, .{ .arg = arg, .off = off });
                            }
                        }
                    }

                    // Spilled-float stores read arg registers while they still hold their edge
                    // values, so they must precede the reg->reg parallel move (mirrors vectors). A
                    // spilled arg reloads into `fspill0` (disjoint from `float_scratch`, the move's
                    // cycle-breaking scratch) before being stored.
                    for (float_stores.items) |s| {
                        const ar = try reloadFloat(allocator, &code, alloc, float_spill_base, s.arg, term_pos, s.d64, fspill0);
                        try code.append(allocator, if (s.d64) encode.fsd(ar, .x2, s.off) else encode.fsw(ar, .x2, s.off));
                    }

                    // Spilled-int stores read arg registers while they still hold their edge values,
                    // so they must precede the reg->reg parallel move (mirrors the float/vector
                    // stores above). A spilled arg reloads into `spill_scratch1` (x8) - disjoint from
                    // `spill_scratch0` (x6), the move's cycle-breaking scratch, which the parallel
                    // move that follows may clobber - before being stored into the param's slot.
                    for (int_stores.items) |s| {
                        const ar = try reloadInt(allocator, &code, alloc, spill_base, s.arg, term_pos, spill_scratch1);
                        try code.append(allocator, encode.sd(ar, .x2, s.off));
                    }

                    try parallelMoveInt(allocator, &code, int_moves.items, spill_scratch0);

                    // Spilled-int reloads write the now-final param registers, after the reg->reg
                    // move (mirrors the float/vector reloads). The arg is spilled by construction
                    // (a register arg took the reg->reg path), so `reloadInt` loads it into `r.dst`.
                    for (int_reloads.items) |r| {
                        _ = try reloadInt(allocator, &code, alloc, spill_base, r.arg, term_pos, r.dst);
                    }

                    try parallelMoveFloat(allocator, &code, float_moves.items, float_scratch);

                    // Spilled-float reloads write the now-final param registers, after the reg->reg
                    // move (mirrors the vector reloads below).
                    for (float_reloads.items) |r| {
                        _ = try reloadFloat(allocator, &code, alloc, float_spill_base, r.arg, term_pos, r.d64, r.dst);
                    }

                    // Vector spill/reg handling (see the ordering note above). Stores read arg
                    // registers while they still hold their edge values, so they must precede the
                    // reg->reg moves.
                    for (vec_stores.items) |s| {
                        const ar = try reloadVector(allocator, &code, alloc, vspill_base, s.arg, term_pos, vec_op0, spill_scratch1);
                        try code.append(allocator, encode.addi(spill_scratch1, .x2, s.off));
                        try code.append(allocator, encode.vse32(ar, spill_scratch1));
                    }
                    try parallelMoveVector(allocator, &code, vec_moves.items, vector_scratch);
                    for (vec_reloads.items) |r| {
                        const ar = try reloadVector(allocator, &code, alloc, vspill_base, r.arg, term_pos, r.dst, spill_scratch1);
                        if (ar != r.dst) try code.append(allocator, encode.vmv_v_v(r.dst, ar));
                    }

                    try fixups.append(allocator, .{ .index = code.items.len, .target = j.target, .kind = .jal });
                    try code.append(allocator, encode.jal(.x0, 0));
                },
            }
        }

        // Advance past the terminator to the next reachable block's parameter row, keeping `pos` in
        // lockstep with the allocator numbering (param + insts + terminator = block_insts.len + 2).
        pos = term_pos + 1;
    }

    // Every recorded split-boundary action must have been drained by a per-instruction or the
    // terminator-position drain above. A leftover action means an `at` no emission position reached: a
    // numbering bug that would silently drop a store/reload. Fail loudly instead of miscompiling.
    std.debug.assert(action_cursor == alloc.actions.items.len);

    // Branch relaxation. RISC-V B-type conditional branches reach only ±4KiB (i13,
    // even); a branch whose target lies farther would wrap (release) or panic (safe)
    // in the patch below. Decide per conditional-branch fixup whether it must take the
    // long form (an inverted short branch that skips a `jal` reaching ±1MiB), then
    // rebuild the code so every other branch/jump still lands correctly.
    //
    // `long[i]` is set only for `.branch`/`.cbranch` fixups. `.jal` already reaches
    // ±1MiB, so it is never relaxed here (only range-checked at patch time).
    const long = try allocator.alloc(bool, fixups.items.len);
    defer allocator.free(long);
    @memset(long, false);

    // Marking one branch long inserts a word and can push a later branch out of range,
    // so iterate to a fixpoint. Marks are monotonic (a long branch never reverts), and
    // each pass either flips at least one flag or stops, so this converges in at most
    // (#conditional branches) passes. At the fixpoint every branch is classified under
    // the FINAL layout: shorts are provably in i13 range, longs are exactly those that
    // are not.
    var any_long = false;
    var changed = true;
    while (changed) {
        changed = false;
        for (fixups.items, 0..) |fx, i| {
            switch (fx.kind) {
                .branch, .cbranch => {},
                .jal => continue,
            }
            if (long[i]) continue; // already long; monotonic
            const target_word = block_start[@intFromEnum(fx.target)];
            const adj_target = target_word + extraBeforeWord(fixups.items, long, target_word);
            const adj_branch = fx.index + extraBeforeWord(fixups.items, long, fx.index);
            const off = (@as(i64, @intCast(adj_target)) - @as(i64, @intCast(adj_branch))) * 4;
            if (off < b_type_min or off > b_type_max) {
                long[i] = true;
                any_long = true;
                changed = true;
            }
        }
    }

    // Rebuild `code`, `block_start`, and the fixup indices for the relaxed layout.
    // Skipped entirely when nothing relaxed, so near-branch functions stay byte-
    // identical (and allocation-free) with respect to the pre-relaxation code path.
    if (any_long) {
        // Map each original code word index to the fixup starting there (if any). Fixup
        // indices are unique (one branch/jump per word), appended in increasing order.
        const fixup_at = try allocator.alloc(?usize, code.items.len);
        defer allocator.free(fixup_at);
        @memset(fixup_at, null);
        for (fixups.items, 0..) |fx, i| fixup_at[fx.index] = i;

        var new_code: std.ArrayList(u32) = .empty;
        errdefer new_code.deinit(allocator);
        var new_fixups: std.ArrayList(Fixup) = .empty;
        errdefer new_fixups.deinit(allocator);

        for (code.items, 0..) |word, oi| {
            if (fixup_at[oi]) |fi| {
                const fx = fixups.items[fi];
                if (long[fi]) {
                    // Long form: inverted short branch skipping the far `jal` (+8 bytes =
                    // the instruction after the jal), then `jal x0, far_target`. The far
                    // target is patched below; the skip is fully determined now.
                    const skip = switch (fx.kind) {
                        .branch => |rs1| BranchKind.bne.invert().emit(rs1, .x0, 8),
                        .cbranch => |cb| cb.kind.invert().emit(cb.rs1, cb.rs2, 8),
                        .jal => unreachable, // a jal is never marked long
                    };
                    try new_code.append(allocator, skip);
                    // The fixup now targets the SECOND word (the jal) and re-encodes as a
                    // plain far jump at patch time.
                    try new_fixups.append(allocator, .{ .index = new_code.items.len, .target = fx.target, .kind = .jal });
                    try new_code.append(allocator, encode.jal(.x0, 0));
                } else {
                    try new_fixups.append(allocator, .{ .index = new_code.items.len, .target = fx.target, .kind = fx.kind });
                    try new_code.append(allocator, word);
                }
            } else {
                try new_code.append(allocator, word);
            }
        }

        // Shift every block start by the extra words inserted before it. Computed from
        // the ORIGINAL positions, so read-before-write in place is safe.
        for (0..func.blockCount()) |bi| {
            // Only reachable blocks have a valid `block_start` (the emission loop set theirs);
            // an unreachable block's entry is untouched and unread, so skip it to avoid reading
            // and rewriting an undefined slot. With all blocks reachable this shifts every entry
            // exactly as before.
            if (!reachable[bi]) continue;
            block_start[bi] = block_start[bi] + extraBeforeWord(fixups.items, long, block_start[bi]);
        }

        // Move the relaxed image into `code`/`fixups`; neutralize the temporaries' error
        // cleanup so ownership is not double-freed.
        code.deinit(allocator);
        code = new_code;
        new_code = .empty;
        fixups.deinit(allocator);
        fixups = new_fixups;
        new_fixups = .empty;
    }

    // Patch each branch/jump now that every block's position is known. Post-relaxation,
    // short conditional branches are guaranteed in i13 range (asserted as a programmer-
    // error invariant); a `jal` beyond ±1MiB is a clean failure, not a wrap.
    for (fixups.items) |fx| {
        const target_idx: i64 = @intCast(block_start[@intFromEnum(fx.target)]);
        const from_idx: i64 = @intCast(fx.index);
        const off: i64 = (target_idx - from_idx) * 4;
        switch (fx.kind) {
            .branch => |rs1| {
                std.debug.assert(off >= b_type_min and off <= b_type_max);
                code.items[fx.index] = encode.bne(rs1, .x0, @intCast(off));
            },
            .cbranch => |cb| {
                std.debug.assert(off >= b_type_min and off <= b_type_max);
                code.items[fx.index] = cb.kind.emit(cb.rs1, cb.rs2, @intCast(off));
            },
            .jal => {
                if (off < j_type_min or off > j_type_max) return error.Unsupported;
                code.items[fx.index] = encode.jal(.x0, @intCast(off));
            },
        }
    }

    return .{
        .code = try code.toOwnedSlice(allocator),
        .relocs = try relocs.toOwnedSlice(allocator),
        .lines = try lines.toOwnedSlice(allocator),
    };
}

test "an unreachable block does not change the compiled output (byte-identical)" {
    // Reachability-aware isel: a block unreachable from the entry must contribute NOTHING. Compile a
    // normal function, then append a dead block (nothing branches to it) carrying enough live
    // block-params and arithmetic that, if isel processed it, it WOULD allocate registers and emit
    // code and thus shift the output. The compiled bytes must be identical before and after.
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();
    const p0 = try func.appendBlockParam(entry, i64_t);
    const p1 = try func.appendBlockParam(entry, i64_t);
    const s = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = p0, .rhs = p1 } });
    func.setTerminator(entry, .{ .ret = s });

    const before = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(before);

    // Append the UNREACHABLE block. Its eight params plus the add chain over them are exactly the
    // kind of live values that would draw registers and emit `add`s if they were ever processed.
    const dead = try func.appendBlock();
    var dp: [8]Value = undefined;
    for (&dp) |*d| d.* = try func.appendBlockParam(dead, i64_t);
    var accd = try func.appendInst(dead, i64_t, .{ .arith = .{ .op = .add, .lhs = dp[0], .rhs = dp[1] } });
    for (dp[2..]) |d| accd = try func.appendInst(dead, i64_t, .{ .arith = .{ .op = .add, .lhs = accd, .rhs = d } });
    func.setTerminator(dead, .{ .ret = accd });

    const after = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(after);

    // Dead block emitted nothing and pressured nothing: identical machine code.
    try std.testing.expectEqualSlices(u32, before, after);
}

test "an unreachable register-pressure block is skipped so the function still compiles" {
    // The plan-17 enabler in miniature: the ONLY reachable content is trivial (a ptr param and a
    // void return), but an unreachable block carries far more simultaneously-live integer
    // block-params than the 17 allocatable integer registers. Block params have no spill path, so if
    // isel walked the dead block `allocateRegisters` would return `error.Unsupported`. Because the
    // block is unreachable it is skipped entirely, and the function compiles.
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    _ = try func.appendBlockParam(entry, ptr_t);
    func.setTerminator(entry, .{ .ret = null });

    const dead = try func.appendBlock();
    var dp: [40]Value = undefined; // 40 > 17 allocatable integer registers, no block-param spill path
    for (&dp) |*d| d.* = try func.appendBlockParam(dead, i64_t);
    // Chain them so every param is live simultaneously at block entry (the last param is used last).
    var accd = try func.appendInst(dead, i64_t, .{ .arith = .{ .op = .add, .lhs = dp[0], .rhs = dp[1] } });
    for (dp[2..]) |d| accd = try func.appendInst(dead, i64_t, .{ .arith = .{ .op = .add, .lhs = accd, .rhs = d } });
    func.setTerminator(dead, .{ .ret = accd });

    // Before reachability-aware isel this returned error.Unsupported; now it succeeds.
    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);
    try std.testing.expect(code.len > 0);
}

test "int eviction spills the value whose next use is furthest, not the current value" {
    // White-box on `allocateRegisters`: at an integer-exhaustion point the allocator must EVICT the
    // active value whose next use is furthest away (Belady) and hand its register to the value being
    // defined, not simply spill the value being defined. Seventeen live iconsts fill the 17
    // allocatable integer registers, then an eighteenth value `w` is defined and exhausts the file.
    // The seventeen are then consumed in reverse order (c[16] first, c[0] last), so c[0] has the
    // FURTHEST next use and `w`'s own next use (the very next instruction) is the nearest. Eviction
    // therefore takes c[0]'s register and keeps `w` in it. Because c[0] is an intra-block value with a
    // live register prefix, eviction TAIL-SPLITS it (register prefix then stack-slot tail) rather than
    // whole-spilling, so it lands in `segments`, not `int_spill`.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const entry = try func.appendBlock();

    // Seventeen simultaneously-live integer values (one per allocatable int register).
    var c: [17]Value = undefined;
    for (&c) |*ci| ci.* = try func.appendInst(entry, i64_t, .{ .iconst = 1 });
    // The eighteenth value: its definition exhausts the integer file.
    const w = try func.appendInst(entry, i64_t, .{ .iconst = 2 });

    // Consume the values so their next-use ordering is deterministic: `w` and c[16] are used first
    // (nearest), c[0] last (furthest). The reduction keeps every c[k] live until its single tail use.
    var acc = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = w, .rhs = c[16] } });
    var k: usize = 16;
    while (k > 0) {
        k -= 1;
        acc = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = c[k] } });
    }
    func.setTerminator(entry, .{ .ret = acc });

    const reachable = try allocator.alloc(bool, func.blockCount());
    defer allocator.free(reachable);
    @memset(reachable, true);

    var alloc = try allocateRegisters(allocator, &func, false, false, reachable, &empty_fold);
    defer alloc.deinit(allocator);

    // Belady victim: c[0] (furthest next use) gives up its register to `w`. It is tail-split (register
    // prefix then stack-slot tail), so it does NOT land in the whole-interval `int_spill`, and instead
    // gets a segment list. Then, as the reduction consumes c[16], c[15], ... the register pressure
    // drops, so by c[0]'s single (furthest) use a register is free again and SECOND-CHANCE RE-HOMES
    // c[0] into it: a `.reg` segment is appended after the `.slot`, and c[0] is put back into
    // `alloc.int` for register accounting (its tail use reads that register, not a per-use slot
    // reload). So the segment list is reg -> slot -> reg and c[0] is present in `alloc.int` again.
    try std.testing.expect(!alloc.int_spill.contains(c[0]));
    const segs = alloc.segments.get(c[0]).?;
    try std.testing.expect(segs.len == 3);
    try std.testing.expect(segs[0].loc == .reg);
    try std.testing.expect(segs[1].loc == .slot);
    try std.testing.expect(segs[2].loc == .reg);
    try std.testing.expect(alloc.int.contains(c[0]));
    // `w` keeps a whole-interval register (never spilled, never split).
    try std.testing.expect(alloc.int.contains(w));
    try std.testing.expect(!alloc.int_spill.contains(w));
    try std.testing.expect(!alloc.segments.contains(w));
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

    // Thirteen independent float values, computed while the two entry params are still
    // live, exceed the ten caller-saved float temporaries (ft10/f30 and ft11/f31 are
    // reserved as the two float spill scratch registers, out of the allocatable pool), so
    // callee-saved float registers (fs0=f8, fs1=f9, fs2=f18, fs3=f19) are drawn. No value
    // spills to a stack slot here: the pressure lands entirely in registers.
    var vals: [13]Value = undefined;
    for (&vals) |*v| v.* = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = p0, .rhs = p1 } });
    var acc = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = vals[0], .rhs = vals[1] } });
    for (vals[2..]) |v| acc = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    func.setTerminator(entry, .{ .ret = acc });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);

    // The frame preserves four callee-saved float registers (fs0=f8, fs1=f9, fs2=f18,
    // fs3=f19) with fsd/fld, restoring them before the return.
    try std.testing.expectEqual(encode.addi(.x2, .x2, -32), code[0]);
    try std.testing.expectEqual(encode.fsd(.f8, .x2, 0), code[1]);
    try std.testing.expectEqual(encode.fsd(.f9, .x2, 8), code[2]);
    try std.testing.expectEqual(encode.fsd(.f18, .x2, 16), code[3]);
    try std.testing.expectEqual(encode.fsd(.f19, .x2, 24), code[4]);
    try std.testing.expectEqual(encode.fld(.f8, .x2, 0), code[code.len - 6]);
    try std.testing.expectEqual(encode.fld(.f9, .x2, 8), code[code.len - 5]);
    try std.testing.expectEqual(encode.fld(.f18, .x2, 16), code[code.len - 4]);
    try std.testing.expectEqual(encode.fld(.f19, .x2, 24), code[code.len - 3]);
    try std.testing.expectEqual(encode.addi(.x2, .x2, 32), code[code.len - 2]);
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

    var compiled = try compileFunction(std.testing.allocator, &func, .{});
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

    var compiled = try compileFunction(std.testing.allocator, &func, .{});
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
        encode.fmv_d(.f0, .f10), // fmv.d ft0, fa0  (parallel-move copies the whole float reg;
        // a full 64-bit copy carries the f32 value's exact bits, NaN-box included, across the edge)
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

test "an f16 function now compiles (software-emulated, no Zfh) instead of being rejected" {
    // f16 was previously rejected on riscv64 (no hardware half). It is now emulated: held as its
    // f32 widening, arithmetic in f32 with a per-op software round to half. This just proves the
    // gate is gone and codegen succeeds; the qemu differentials in tests/f16.zig prove correctness.
    var func = Function.init(std.testing.allocator);
    defer func.deinit();

    const f16_t = try func.types.intern(.{ .float = .f16 });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, f16_t);
    const b = try func.appendBlockParam(entry, f16_t);
    const sum = try func.appendInst(entry, f16_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    const code = try selectFunction(std.testing.allocator, &func);
    defer std.testing.allocator.free(code);
    try std.testing.expect(code.len != 0);
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

test "intLocationAt returns the whole-life register for an unsplit int value" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, i32_t);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(entry, .{ .ret = sum });

    var doms = try dominators.compute(allocator, &func);
    defer doms.deinit(allocator);

    var alloc = try allocateRegisters(allocator, &func, false, false, doms.reachable, &empty_fold);
    defer alloc.deinit(allocator);

    // This tiny function spills or splits nothing: `segments` stays empty, so `intLocationAt` must
    // fall back to the whole-life `int` register for every value, at every position.
    try std.testing.expectEqual(@as(usize, 0), alloc.segments.count());
    for ([_]Value{ a, b, sum }) |v| {
        const r = alloc.int.get(v).?;
        try std.testing.expectEqual(IntLoc{ .reg = r }, intLocationAt(&alloc, v, 0));
        try std.testing.expectEqual(IntLoc{ .reg = r }, intLocationAt(&alloc, v, 999));
    }
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

/// True if `code` contains a VPU packed-single arithmetic word (opcode 0x7B, funct3 0b111: the
/// dynamic-rounding form `fadd.ps`/`fsub.ps`/`fmul.ps`/`fdiv.ps` share this shape, distinguished
/// only by funct7). Masks the opcode field per instructions.vh's `casex`, like encode.zig's tests.
/// True if `code` contains a real VPU packed-single arithmetic word (`fadd.ps`/`fsub.ps`/
/// `fmul.ps`/`fdiv.ps`, opcode 0x7B, `encode.vpuPsRType`). Checking the opcode and funct3 alone is
/// NOT enough: `mov.m.x md, xs, imm8` (the M0 mask preamble every VPU kernel starts with) is ALSO
/// opcode 0x7B, and `mov_m_x(0, .x0, 0xFF)` happens to place 0b111 in the funct3 field too (imm8's
/// low 3 bits, 0xFF & 0x7 == 0b111, sit at bits [14:12]), so an opcode+funct3-only check matches
/// the preamble even when no arithmetic op was ever emitted. Two more conditions rule the preamble
/// out: its funct7 [31:25] is 0b0101011 (not one of the four PS-arith funct7 codes below), and its
/// rd/md field [11:7] is a mask-register index (0..7), never inside the VPU vector pool (f16..f27,
/// see `vpu_vector_regs`) that a real arith destination is always allocated from.
fn hasVpuArithWord(code: []const u32) bool {
    const funct7_fadd = 0b0000000;
    const funct7_fsub = 0b0000100;
    const funct7_fmul = 0b0001000;
    const funct7_fdiv = 0b0001100;
    for (code) |w| {
        if ((w & 0x7F) != 0x7B) continue;
        if (((w >> 12) & 0x7) != 0b111) continue;
        const funct7 = (w >> 25) & 0x7F;
        if (funct7 != funct7_fadd and funct7 != funct7_fsub and funct7 != funct7_fmul and funct7 != funct7_fdiv) continue;
        const rd = (w >> 7) & 0x1F;
        if (rd >= 16 and rd <= 27) return true;
    }
    return false;
}

/// True if `code` contains a VPU `flw.ps` (opcode 0x0B, funct3 010) or `fsw.ps` (opcode 0x0B,
/// funct3 110) word.
fn hasVpuLoadStoreWord(code: []const u32) bool {
    for (code) |w| {
        if ((w & 0x7F) != 0x0B) continue;
        const funct3 = (w >> 12) & 0x7;
        if (funct3 == 0b010 or funct3 == 0b110) return true;
    }
    return false;
}

test "et-soc VPU: an 8-lane elementwise f32 add compiles to VPU words with an M0 preamble" {
    const allocator = std.testing.allocator;
    const V = ir.function.Value;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v8 = try func.types.intern(.{ .vector = .{ .len = 8, .elem = f32_t } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(b, ptr_t);
    const ptr_b = try func.appendBlockParam(b, ptr_t);
    const ptr_out = try func.appendBlockParam(b, ptr_t);

    // out[i] = a[i] + b[i] for i in 0..8: loads and stores stay scalar (the SLP vectorizer never
    // fuses those), only the 8 parallel adds fuse into one <8 x f32> arith op. This is exactly the
    // shape vectorize.runModel produces under the et-soc model (pack via struct_new, one vector
    // arith, unpack via extract).
    // Pack `va` immediately after loading all 8 `a` scalars (before loading any `b` scalar), and
    // likewise for `vb`: this keeps peak scalar-float pressure at 8 simultaneously live values
    // (exactly the vpu-mode scalar pool, f0..f7), not 16. Interleaving the two loops would need
    // 16 live scalars at once, more than the disjoint partition provides, and the point of the
    // partition is to prove correctness without needing a general spill-happy allocator here.
    var av: [8]V = undefined;
    for (0..8) |i| {
        const addr_a = try func.appendArithImm(b, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_a } });
    }
    const va = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&av) } });
    var bv: [8]V = undefined;
    for (0..8) |i| {
        const addr_b = try func.appendArithImm(b, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(b, f32_t, .{ .load = .{ .ptr = addr_b } });
    }
    const vb = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&bv) } });
    const vc = try func.appendInst(b, v8, .{ .arith = .{ .op = .add, .lhs = va, .rhs = vb } });
    for (0..8) |i| {
        const c = try func.appendInst(b, f32_t, .{ .extract = .{ .aggregate = vc, .index = @intCast(i) } });
        const addr_out = try func.appendArithImm(b, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(b, c, addr_out);
    }
    func.setTerminator(b, .{ .ret = null });

    // No emulator decodes et-soc's custom VPU opcodes (see encode.zig), so this is the
    // structural oracle: IR verification plus encoding checks against the RTL match masks.
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());
    const code = try selectFunctionForModel(allocator, &func, model);
    defer allocator.free(code);

    // Compiles cleanly (no error.Unsupported for this simple shape) and contains both a VPU
    // arithmetic word (the fused add) and VPU load/store words (the scalar element accesses use
    // plain flw/fsw, not these; the vector pack/spill path is what would use flw.ps/fsw.ps -- this
    // kernel has no vector spill, so absence would also be acceptable, but the M0 preamble and the
    // fused vector add are load-bearing).
    try std.testing.expect(hasVpuArithWord(code));

    // hasVpuArithWord only proves *some* arith-shaped word exists; pin the exact word too, so a
    // regression that emits, say, fsub.ps by mistake (or drops the add and leaves only the M0
    // preamble, which also decodes to opcode 0x7B/funct3 0b111, see hasVpuArithWord's doc comment)
    // cannot slip through. The register allocator's vpu_vector pool is drawn f16-first (see
    // `vpu_vector_free` in allocateRegisters), and `va`, `vb`, `vc` are the only three vector
    // values live across this kernel, so they land at f16, f17, f18 respectively: `va` is built
    // first (f16), `vb` second (f17), and the `add` computes straight into a fresh register (f18)
    // since neither operand register is free to reuse as the destination.
    const expected_add_word = encode.fadd_ps(.f18, .f16, .f17);
    try std.testing.expect(std.mem.indexOfScalar(u32, code, expected_add_word) != null);

    // The M0 mask preamble (mov_m_x md=0, xs=x0, imm8=0xFF) is the exact known encoding, and must
    // appear near the very start of the function (right after the prologue's frame-open, before
    // any VPU op executes), not merely somewhere in the body.
    const m0_word = encode.mov_m_x(0, .x0, 0xFF);
    var m0_index: ?usize = null;
    for (code, 0..) |w, idx| {
        if (w == m0_word) {
            m0_index = idx;
            break;
        }
    }
    try std.testing.expect(m0_index != null);
    try std.testing.expect(m0_index.? <= 2); // at most: frame-open, then the mask write
}

test "et-soc VPU: a 4-lane vector (RVV's fixed width, not the VPU's fixed 8) is rejected cleanly" {
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
    const c0 = try func.appendInst(b, t, .{ .extract = .{ .aggregate = vc, .index = 0 } });
    func.setTerminator(b, .{ .ret = c0 });

    // The VPU is a fixed 8-lane unit: a 4-lane vector (the RVV width) is a shape this path
    // cannot serve, so `allocateRegisters`'s isVpuWidth check rejects it up front. A clean
    // error, not a crash or (worse) a silent partial-width miscompile.
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expectError(error.Unsupported, selectFunctionForModel(allocator, &func, model));
}

/// True if `code` contains a packed-integer (`pi`) word (opcode 0x7B) with the given
/// funct7/funct3 whose destination is a real VPU vector register (f16..f27, `vpu_vector_regs`).
/// The rd-range check rules out the M0 mask preamble (`mov.m.x`, also opcode 0x7B) the same way
/// `hasVpuArithWord` does: a mask write targets a mask register (0..7), never the vector pool.
fn hasPiWord(code: []const u32, funct7: u32, funct3: u32) bool {
    for (code) |w| {
        if ((w & 0x7F) != 0x7B) continue;
        if (((w >> 12) & 0x7) != funct3) continue;
        if (((w >> 25) & 0x7F) != funct7) continue;
        const rd = (w >> 7) & 0x1F;
        if (rd >= 16 and rd <= 27) return true;
    }
    return false;
}

/// True if `code` contains a scalar 32-bit integer store `sw rs2, imm(x2)` (opcode 0x23, funct3
/// 0b010, base x2): the per-lane store the packed-integer `struct_new` pack emits into the pack
/// scratch on `sp`. The packed-single pack uses `fsw` (opcode 0x27) instead, so this distinguishes
/// an int pack from a float pack.
fn hasIntPackStore(code: []const u32) bool {
    for (code) |w| {
        if ((w & 0x7F) != 0x23) continue; // STORE major opcode
        if (((w >> 12) & 0x7) != 0b010) continue; // funct3 = SW (32-bit)
        if (((w >> 15) & 0x1F) != 2) continue; // rs1 = x2 (sp)
        return true;
    }
    return false;
}

/// True if `code` contains an `fmv.w.x` word (opcode 0x53, funct7 0b1111000, rs2 0, funct3 0): the
/// GPR-to-FPR move the packed-SINGLE lane extract emits after `fmvs.x.ps`. The packed-INTEGER lane
/// extract keeps the extracted lane in the GPR (it IS the i32 result), so it emits no such move.
fn hasFmvWX(code: []const u32) bool {
    for (code) |w| {
        if ((w & 0x7F) != 0x53) continue;
        if (((w >> 25) & 0x7F) != 0b1111000) continue;
        if (((w >> 20) & 0x1F) != 0) continue; // rs2 field = 0
        if (((w >> 12) & 0x7) != 0) continue; // funct3 = 0
        return true;
    }
    return false;
}

/// Build the 8-lane `<8 x i32>` kernel `out[i] = op(a[i], b[i])` in the exact SLP shape the et-soc
/// vectorizer produces (8 scalar int loads, one `struct_new` pack, one vector `arith`, 8 `extract`s,
/// 8 scalar int stores), over an element of the given `signedness`. Mirrors `buildAddKernel` but the
/// lanes are integers, so packing/unpacking rides the INT register file, not the scalar float pool.
fn buildIntVecKernel(func: *Function, op: ir.function.BinOp, signedness: std.builtin.Signedness) !void {
    const V = ir.function.Value;
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = signedness, .bits = 32 } });
    const v8 = try func.types.intern(.{ .vector = .{ .len = 8, .elem = i32_t } });
    const ptr_t = try func.types.intern(.ptr);
    const b = try func.appendBlock();
    const ptr_a = try func.appendBlockParam(b, ptr_t);
    const ptr_b = try func.appendBlockParam(b, ptr_t);
    const ptr_out = try func.appendBlockParam(b, ptr_t);

    var av: [8]V = undefined;
    for (0..8) |i| {
        const addr_a = try func.appendArithImm(b, ptr_t, .add, ptr_a, @intCast(i * 4));
        av[i] = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr_a } });
    }
    const va = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&av) } });
    var bv: [8]V = undefined;
    for (0..8) |i| {
        const addr_b = try func.appendArithImm(b, ptr_t, .add, ptr_b, @intCast(i * 4));
        bv[i] = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr_b } });
    }
    const vb = try func.appendInst(b, v8, .{ .struct_new = .{ .fields = try func.internValueList(&bv) } });
    const vc = try func.appendInst(b, v8, .{ .arith = .{ .op = op, .lhs = va, .rhs = vb } });
    for (0..8) |i| {
        const c = try func.appendInst(b, i32_t, .{ .extract = .{ .aggregate = vc, .index = @intCast(i) } });
        const addr_out = try func.appendArithImm(b, ptr_t, .add, ptr_out, @intCast(i * 4));
        try func.appendStore(b, c, addr_out);
    }
    func.setTerminator(b, .{ .ret = null });
}

test "et-soc VPU: an 8-lane <8 x i32> op lowers to packed-integer (pi) words, int-store pack, and GPR-kept extract" {
    const allocator = std.testing.allocator;
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expect(model.vpu());

    // Each entry pins the exact pi funct7/funct3 (from esperanto-opc.h, see encode.zig) the op must
    // lower to, plus the exact destination register: `va`, `vb`, `vc` are the only three vector
    // values live, and the vpu vector pool is drawn f16-first, so va=f16, vb=f17, and the arith
    // computes into a fresh f18 (neither operand register is free to reuse as the destination).
    const Case = struct { op: ir.function.BinOp, sign: std.builtin.Signedness, word: u32 };
    const cases = [_]Case{
        .{ .op = .add, .sign = .signed, .word = encode.fadd_pi(.f18, .f16, .f17) },
        .{ .op = .sub, .sign = .signed, .word = encode.fsub_pi(.f18, .f16, .f17) },
        .{ .op = .mul, .sign = .signed, .word = encode.fmul_pi(.f18, .f16, .f17) },
        .{ .op = .bit_and, .sign = .signed, .word = encode.fand_pi(.f18, .f16, .f17) },
        .{ .op = .bit_or, .sign = .signed, .word = encode.for_pi(.f18, .f16, .f17) },
        .{ .op = .bit_xor, .sign = .unsigned, .word = encode.fxor_pi(.f18, .f16, .f17) },
        .{ .op = .shl, .sign = .signed, .word = encode.fsll_pi(.f18, .f16, .f17) },
        // Right shift picks the op on the element signedness: arithmetic for signed, logical for
        // unsigned - the single element-type-driven divergence in the pi arith lowering.
        .{ .op = .shr, .sign = .signed, .word = encode.fsra_pi(.f18, .f16, .f17) },
        .{ .op = .shr, .sign = .unsigned, .word = encode.fsrl_pi(.f18, .f16, .f17) },
    };

    for (cases) |c| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildIntVecKernel(&func, c.op, c.sign);

        var diags = try ir.verify.verify(allocator, &func, .low);
        defer diags.deinit();
        try std.testing.expect(diags.ok());

        const code = try selectFunctionForModel(allocator, &func, model);
        defer allocator.free(code);

        // The exact pi word is present, and it decodes as a genuine pi arith into the vector pool
        // (opcode/funct7/funct3/rd all checked), not the M0 preamble.
        try std.testing.expect(std.mem.indexOfScalar(u32, code, c.word) != null);
        try std.testing.expect(hasPiWord(code, (c.word >> 25) & 0x7F, (c.word >> 12) & 0x7));
        // The pack is via 32-bit int stores (`sw`), not float `fsw`, and the extract keeps each
        // lane in a GPR (no `fmv.w.x` move, unlike the packed-single extract).
        try std.testing.expect(hasIntPackStore(code));
        try std.testing.expect(!hasFmvWX(code));
    }
}

test "et-soc VPU: a <8 x i32> vector op with no packed-integer equivalent (div) is rejected cleanly" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // Integer divide has no `pi` op, so the vector arith lowering must reject it rather than emit a
    // wrong word. (`rem` is likewise unsupported; `div` stands in for both.)
    try buildIntVecKernel(&func, .div, .signed);
    const model = mm.modelFor(.@"et-soc");
    try std.testing.expectError(error.Unsupported, selectFunctionForModel(allocator, &func, model));
}

test "riscv64: a <8 x i32> vector under a non-vpu model is rejected cleanly (no integer RVV here)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildIntVecKernel(&func, .add, .signed);
    // Without the VPU (`selectFunction` builds the inert, vpu-false caps), an 8-lane vector is not a
    // shape the RVV path serves (it is fixed 4-lane, and there is no integer-RVV lowering at all
    // here), so `allocateRegisters`'s isRvvWidth check rejects it up front. A clean error, not a
    // crash or a silent miscompile.
    try std.testing.expectError(error.Unsupported, selectFunction(allocator, &func));
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

test "selectFunctionAligned pads a loop header with nops but never changes fetch_align 0 output" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const acc = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bacc = try func.appendBlockParam(body, t);
    const racc = try func.appendBlockParam(done, t);

    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    const nacc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
    try func.setJump(body, loop, &.{ ni, nacc });
    func.setTerminator(done, .{ .ret = racc });

    // The `if` at `loop` carries edge arguments, which this backend requires split
    // into arg-free landing blocks first (the same pipeline `tests/harness.zig`
    // runs before selecting). `loop` keeps its block index, so it is still the loop
    // header `loops.analyze` finds via the `body -> loop` back edge.
    try ir.legalize.legalize(allocator, &func);
    try splitCriticalEdges(allocator, &func);
    try schedule.scheduleFunction(allocator, &func);

    const unaligned = try selectFunction(allocator, &func);
    defer allocator.free(unaligned);
    const aligned = try selectFunctionAligned(allocator, &func, 32);
    defer allocator.free(aligned);

    // The loop header (`loop`, with a real back-edge from `body`) gets padded to a
    // 32-byte boundary, so the aligned build is strictly longer and contains at least
    // one nop word (the padding); the fetch_align-0 path stays untouched.
    try std.testing.expect(aligned.len > unaligned.len);
    var found_nop = false;
    for (aligned) |w| {
        if (w == encode.nop()) found_nop = true;
    }
    try std.testing.expect(found_nop);
}

test "selectFunctionForModel fires the alignment hook from river-rc1.ma, matches selectFunctionAligned" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, t);
    const i = try func.appendBlockParam(loop, t);
    const acc = try func.appendBlockParam(loop, t);
    const bi = try func.appendBlockParam(body, t);
    const bacc = try func.appendBlockParam(body, t);
    const racc = try func.appendBlockParam(done, t);

    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });
    const ni = try func.appendArithImm(body, t, .add, bi, 1);
    const nacc = try func.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = bi } });
    try func.setJump(body, loop, &.{ ni, nacc });
    func.setTerminator(done, .{ .ret = racc });

    try ir.legalize.legalize(allocator, &func);
    try splitCriticalEdges(allocator, &func);
    try schedule.scheduleFunction(allocator, &func);

    const plain = try selectFunction(allocator, &func);
    defer allocator.free(plain);
    const model = mm.modelFor(.@"river-rc1.ma");
    const tuned = try selectFunctionForModel(allocator, &func, model);
    defer allocator.free(tuned);

    // river-rc1.ma's fetch_align is 8 (above the fetch_align<=4 no-op threshold), so the
    // model seam pads the loop header, same as calling selectFunctionAligned(.., 8)
    // directly: the model-compiled build is strictly longer than the plain one and
    // contains at least one nop word (the padding).
    try std.testing.expectEqual(@as(u16, 8), model.fetch_align);
    const via_aligned = try selectFunctionAligned(allocator, &func, model.fetch_align);
    defer allocator.free(via_aligned);
    try std.testing.expectEqualSlices(u32, via_aligned, tuned);
    try std.testing.expect(tuned.len > plain.len);
    var found_nop = false;
    for (tuned) |w| {
        if (w == encode.nop()) found_nop = true;
    }
    try std.testing.expect(found_nop);
}
