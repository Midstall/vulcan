//! This module selects x86-64 instructions. The System V AMD64 ABI puts integer
//! arguments in RDI, RSI, RDX, RCX, R8, and R9, and puts the result in RAX. The
//! module handles multiple blocks, including high-IR if and jump instructions with
//! block-parameter edge moves. It also handles comparison, division, shifts,
//! immediate-operand arithmetic, and register spilling.
//!
//! x86 arithmetic uses two operands, so the module turns `c = a op b` into
//! `mov c, a` then `op c, b`. The shared Wimmer-Franz linear-scan-on-SSA allocator
//! (`wimmer.zig`) assigns registers. A spilled value lives in a stack slot. At each
//! use, the module reloads the value into a scratch register (R10 or R11), computes
//! in the scratch register, and stores the result back. R11 also serves as the
//! parallel-move scratch register, since the two uses do not overlap in time. The
//! module reserves RAX and RDX for division, and RCX for shifts.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const wimmer = @import("../wimmer.zig");
const addrfold = @import("../addrfold.zig");
const mm = @import("vulcan-opt").microarch;

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Reg = encode.Reg;

/// A shared no-fold analysis for paths that must ignore address folding, such as the
/// Wimmer differential compile and the allocator test hooks. Its `baseOf`, `offOf`,
/// and `isDeadAdd` behave as if nothing folded, so those paths emit the same bytes as
/// before address folding existed. `Ctx.fold` defaults to this analysis. Only `compile`
/// (which builds a real analysis) overrides it.
const empty_fold: addrfold.Analysis = addrfold.Analysis.empty;

/// The x86-64 fold predicate for `addrfold.analyze`. It folds a load or store whose
/// pointer is an `arith_imm.add(base, imm)` into a `[base + disp32]` addressing mode,
/// for any access size, since x86 mem operands carry a 32-bit signed displacement
/// regardless of width. The fold applies exactly when the add's imm fits a signed
/// 32-bit displacement. The isel already assumes an `arith_imm` imm fits an i32
/// (`@intCast(a.imm)` when it emits the add), so this matches an existing invariant.
/// The function returns the byte offset (equal to the add's imm) when in range, else
/// null. `analyze` calls this function only after it confirms the pointer is an
/// `arith_imm.add`, so the unwraps below are guaranteed to succeed. The code still
/// asserts them.
fn x86_64FoldOffset(_: void, func: *const Function, mem_inst: ir.function.Inst) ?i64 {
    const ptr = switch (func.opcode(mem_inst)) {
        .load => |l| l.ptr,
        .store => |st| st.ptr,
        else => unreachable, // analyze hands foldOffset only a load or store
    };
    const def = func.definingInst(ptr).?; // analyze confirmed ptr is defined by an arith_imm.add
    const add = switch (func.opcode(def)) {
        .arith_imm => |a| a,
        else => unreachable,
    };
    std.debug.assert(add.op == .add);
    if (std.math.cast(i32, add.imm) == null) return null;
    return add.imm;
}

pub const Error = std.mem.Allocator.Error || error{Unsupported};

const arg_regs = [_]Reg{ .rdi, .rsi, .rdx, .rcx, .r8, .r9 };
const ret_reg: Reg = .rax;
const scratch1: Reg = .r10; // reload scratch for a left operand or a destination
const scratch2: Reg = .r11; // reload scratch for a right operand
const move_scratch: Reg = .r11; // parallel-move cycle scratch, does not overlap with spills

/// Where a value lives: a general register, an SSE (xmm) register, a general-register stack
/// spill slot, or an xmm stack spill slot (16-byte, holds a scalar float or a whole vector).
const Loc = union(enum) { reg: Reg, xmm: encode.Xmm, spill: u32, xmm_spill: u32 };

/// One piece of a split GPR value's life. The value lives in `loc` from position
/// `from` until the next segment, or, for the last segment, to the end of its range.
/// `segments[0].from` is the value's def position, so a lookup at any position at or
/// after the def resolves to some segment. The register allocator fills the segment
/// map when it splits a value's life. While the map is empty, `loc` falls back to the
/// whole-life `loc_of` lookup, and emission produces the same bytes as before
/// splitting existed.
const Segment = struct { from: u32, loc: Loc };

/// A store the emitter must insert at a split boundary. `at` is the instruction
/// position the store lands before, the position at which the register pool ran out
/// for a tail split. The store writes the victim's register to its new slot before
/// the taker (the value defined at `at`) overwrites that register, so the victim's
/// tail uses reload the correct bits. The shared Wimmer translation
/// (`translateAllocationX86`, consuming `walloc.actions`) produces every kind: `.store`
/// and `.reload`, the `.move` re-home, `.slot_to_slot`, and the xmm variants that set
/// `is_xmm` and read `xreg` and `xmove_from`. This is the only allocator in use.
/// `.slot_to_slot` (mirrors aarch64 isel's Wimmer bridge gap #7) re-homes a spilled
/// value from one slot to another without ever holding it in a value register.
/// `emitSplitActionX86` expands it into a reload-then-store pair through the class
/// scratch register (gpr `move_scratch`/r11, xmm `xmm_scratch`/xmm15), so `reg` and
/// `xreg` stay at their defaults and are never read for this kind.
const SplitAction = struct {
    at: u32,
    kind: enum { store, reload, move, slot_to_slot },
    value: Value,
    slot: u32 = 0,
    // GPR class (Wimmer gpr splits).
    reg: Reg = .rax,
    move_from: Reg = .rax, // `.move` source (reg -> reg re-home)
    // XMM class (Wimmer scalar-float / vector splits). `is_xmm` selects which register set the drain reads.
    is_xmm: bool = false,
    xreg: Xmm = .xmm0,
    xmove_from: Xmm = .xmm0,
    /// The source slot of a `.slot_to_slot` re-home. `slot` is the destination. Only
    /// the shared Wimmer translation produces this field.
    move_from_slot: u32 = 0,
};

/// One ordered control-flow-edge move, from the shared Wimmer path. `class` is 0 for
/// gpr or 1 for xmm. A location is a class-relative register index or a per-class
/// spill slot. The shared allocator already ordered these into a valid parallel-move
/// sequence (it reads sources before it overwrites them, and it breaks cycles through
/// the class scratch register), so the emitter replays each one as a primitive op.
const EdgeLoc = union(enum) { reg: u16, slot: u32 };
/// `wide` marks a class-1 (xmm) move of a 256-bit ymm value, emitted with vmovups (32
/// bytes) rather than movups (16 bytes). The moved value's IR type in
/// `translateAllocationX86` sets this field. It is ignored for class 0 (gpr).
const EdgeMove = struct { class: u8, src: EdgeLoc, dst: EdgeLoc, wide: bool = false };
const EdgeMoveSet = struct { pred: Block, succ: Block, moves: []EdgeMove };

const Xmm = encode.Xmm;
const xmm_arg_regs = [_]Xmm{ .xmm0, .xmm1, .xmm2, .xmm3, .xmm4, .xmm5, .xmm6, .xmm7 };
const xmm_ret: Xmm = .xmm0;
// xmm13, xmm14, and xmm15 are reserved scratch registers for operand reloads and the
// move/aliasing temp. The Wimmer xmm pool allocates xmm0 through xmm12. A reloaded
// left operand goes to op0, a right operand to op1.
const xmm_op0: Xmm = .xmm13;
const xmm_op1: Xmm = .xmm14;
const xmm_scratch: Xmm = .xmm15;

/// Whether `v` is a floating-point value. It lives in an xmm register.
fn isFloat(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .float;
}
/// Whether `v` is a SIMD vector, `<N x f32>`. It also lives in an xmm register.
fn isVector(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .vector;
}
/// Whether `v` is a 256-bit (AVX/YMM) vector, with more than four f32 lanes. Such a
/// vector needs the VEX-encoded ops rather than the 128-bit SSE ones. Today the vector
/// has exactly 8 lanes.
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
/// Whether `v` is an f16 (half). The module emulates f16: the value lives in an xmm
/// register as its f32 widening, so `isDouble(f16)` is false and every in-register op
/// naturally uses the scalar-single SSE form. The boundaries widen and narrow the
/// value with the F16C conversions. `isHalf` marks the sites that must add that
/// widening or narrowing: memory load and store, narrowing converts, the int-to-f16
/// and f32/f64-to-f16 converts, arithmetic results, and the f16 constant.
fn isHalf(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f16,
        else => false,
    };
}
/// Whether `v` lives in an xmm register, as a scalar float or a SIMD vector.
fn isXmm(func: *const Function, v: Value) bool {
    return isFloat(func, v) or isVector(func, v);
}
/// The bit width of an integer value. Returns 64 for a non-integer, as the safe
/// 64-bit default.
fn intBits(func: *const Function, v: Value) u16 {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.bits,
        else => 64,
    };
}
/// Whether the function makes any call, direct or indirect. If it does, its frame
/// needs 16-byte call-site alignment.
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

/// Which relocation shape a `Reloc` records. `.call` is a `call rel32`'s displacement,
/// with addend -4 from the E8 opcode, resolved intra-module by the linker.
/// `.pcrel_lea` is a `global_addr`'s `lea rd, [rip+disp32]` displacement, with addend
/// -4 from the end of the disp32 field, patched once the symbol's runtime address is
/// known (see `applyGlobalReloc` in `link.zig`). `.got_pcrel` is a GOT-indirect
/// `global_addr`'s (`via_got`) `mov rd, qword [rip+disp32]` displacement, with addend
/// -4 and the same field math, and it emits `R_X86_64_GOTPCREL`. Its disp32 addresses
/// the symbol's `.got` slot, which a real `ld.so` fills via `R_X86_64_GLOB_DAT`. So the
/// `mov` loads the symbol's address rather than the `lea` computing it. The module uses
/// `.got_pcrel` for a data import of a `.so`'s global.
pub const Kind = enum { call, pcrel_lea, got_pcrel };

/// A relocation `kind` targeting `symbol`, whose disp32 field starts at byte `offset`
/// within the compiled function's code. All three kinds share the same
/// PC-relative-from-end-of-field math, `target - (offset + 4)`. Only the target and
/// the resolution point differ: `.call` resolves intra-module at link time,
/// `.pcrel_lea` resolves the symbol at map time, and `.got_pcrel` resolves the
/// symbol's GOT slot, which the dynamic linker synthesizes.
pub const Reloc = struct { offset: usize, symbol: []const u8, kind: Kind = .call };

/// A compiled function: machine code plus its unresolved call relocations.
/// A source-line row: the byte offset where a source line's code begins, from
/// `debug.line` attributes.
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
    // System V variadic define side, meaningful only when `func.is_variadic`. `frame` is
    // the total `sub rsp` amount, threaded here so `va_start` can address the first
    // incoming stack arg ([rsp+frame+8]). `rsa_off` is the rsp offset of the 176-byte
    // register-save area the prologue reserves and fills. Both stay 0 for a
    // non-variadic function, where the module never emits a `va_*` op.
    frame: i32 = 0,
    rsa_off: i32 = 0,
    // Number of callee-saved GPRs the prologue pushed before `sub rsp` (the `saved`
    // slice length). Each push sits above the frame, between it and the return
    // address, so the first incoming stack argument sits 8 bytes higher per pushed
    // register. `va_start`'s overflow_arg_area must add this.
    saved_count: u32 = 0,
    alloca_off: std.AutoHashMapUnmanaged(Value, u32) = .{}, // each alloca result -> its byte offset in that region
    // Split GPR values only: maps a value to its ascending-by-`from` segment list. An
    // empty list means no value was split, so `loc` falls back to `loc_of` and
    // emission produces the same bytes as before splitting existed.
    segments: std.AutoHashMapUnmanaged(Value, []Segment) = .{},
    // Stores to emit at split boundaries, in ascending-`at` order after the sort in
    // `compile`. An empty list means no value was split, so emission produces the
    // same bytes as before splitting existed.
    actions: std.ArrayList(SplitAction) = .empty,
    def_pos: []u32 = &.{}, // per value: its def position (duped from local liveness, and the emission assert reads it)
    pos: u32 = 0, // current emission position, threaded per instruction so `loc` can pick the active segment
    // Precomputed, already-ordered control-flow-edge moves, from the shared Wimmer
    // path only. When `edge_move_driven` is set, `emitMoves` replays the set for the
    // current edge instead of deriving block-parameter moves. Both stay empty or
    // false for the default path, so emission produces the same bytes as before this
    // path existed.
    edge_moves: []EdgeMoveSet = &.{},
    edge_move_driven: bool = false,
    // Address-mode-fold analysis, consulted by the load/store emit arms and the
    // dead-add skip. Defaults to the empty analysis, where nothing folds, so the
    // Wimmer path and the allocator test hooks produce the same bytes as before
    // folding existed. `compile` overrides it with a real analysis of `func`.
    fold: *const addrfold.Analysis = &empty_fold,
    // Model-tuned capability flags (see `ModelCaps`). Every field defaults to inert
    // (false), so any caller that does not thread a model (`compile`, the Wimmer
    // differential compile, the allocator test hooks) produces the same bytes as
    // before `ModelCaps` existed. Only `compileWithCaps` (via `selectFunctionForModel`)
    // ever sets a field true.
    caps: ModelCaps = .{},
    // The current block's instruction slice and the index of the instruction being
    // emitted. `emitFromAllocation`'s inner loop refreshes both each iteration, before
    // it dispatches to `lowerInst` or `emitIf`. These back `fusesIntoNextIf`: the icmp
    // arm consults `cur_insts[cur_idx]` (itself) to decide whether to skip its
    // materialization, and `emitIf` consults `cur_insts[cur_idx - 1]` (the
    // immediately preceding instruction) to decide whether to fuse. They default to
    // an empty slice and 0, which the code never reads, since no caller reaches
    // either predicate without the loop setting real values first.
    cur_insts: []const ir.function.Inst = &.{},
    cur_idx: usize = 0,

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
                // Each slot is 32 bytes. A 256-bit vmovups reloads a whole `<8 x f32>`.
                // A 128-bit movups reloads a scalar float, a double, or a `<4 x f32>`
                // (the low 16 bytes). Neither instruction truncates.
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

/// Capabilities a model-aware call site threads into `compileWithCaps`. These fields
/// are grouped into one struct, rather than growing `compileWithCaps`'s parameter list
/// one flag per model feature, so adding the next capability never touches every
/// existing call site. `.{}`, where every field is false, is exactly today's behavior
/// for every non-model caller (`compile`/`selectFunction`, the Wimmer differential
/// compile, and the allocator test hooks). Both flags are inert (false) for those
/// callers, so nothing they gate (the `fusesIntoNextIf` and `fusesArithIntoBranch`
/// call sites in `emitIf`/`lowerInst`) ever fires for them.
pub const ModelCaps = struct {
    /// Fuse a compare into its consumer branch: CMP+Jcc becomes a single
    /// flags-setting compare directly followed by the conditional jump, skipping the
    /// separate `setcc`/`test` materialization. This is not base-ISA on x86-64, since
    /// every model still executes a `cmp`, and this flag is about whether the front
    /// end recognizes the macro-op pair. So the flag defaults to false. It gates
    /// `fusesIntoNextIf`, which `fusesArithIntoBranch` also requires.
    fuse_cmp_branch: bool = false,
    /// Fuse an arithmetic op's flag-setting form into its consumer branch, for
    /// example an `add` whose flags are consumed directly by the next `jcc`, without
    /// a separate `cmp`/`test`. Defaults to false. It gates `fusesArithIntoBranch`,
    /// together with `fuse_cmp_branch`, since the fold lives inside the
    /// compare-and-branch path, so both must be on.
    fuse_arith_branch: bool = false,
};

/// Select x86-64 machine code for `func` (code only, call relocations dropped). Caller
/// owns the slice.
pub fn selectFunction(allocator: std.mem.Allocator, func: *const Function) Error![]u8 {
    const compiled = try compile(allocator, func);
    allocator.free(compiled.relocs);
    allocator.free(compiled.lines);
    return compiled.code;
}

/// Compile `func` tuned to `model`. The machine-level hooks read the model's fusion
/// table (see `capsForModel`). An inert model, with an empty `.fusion` and no
/// `cmp_branch`/`arith_branch` rule, makes this produce the same bytes as
/// `selectFunction`.
pub fn selectFunctionForModel(allocator: std.mem.Allocator, func: *const Function, model: *const mm.Model) Error![]u8 {
    // Passing a foreign-arch model here is a caller bug, not a runtime fault.
    std.debug.assert(model.arch == .x86_64);
    const compiled = try compileWithCaps(allocator, func, capsForModel(model));
    allocator.free(compiled.relocs);
    allocator.free(compiled.lines);
    return compiled.code;
}

/// The `ModelCaps` that `selectFunctionForModel` builds for `model`. This is split
/// out so the model-to-caps mapping is unit-testable without compiling a whole
/// function. It asserts `model.arch == .x86_64`, same as the caller above.
pub fn capsForModel(model: *const mm.Model) ModelCaps {
    std.debug.assert(model.arch == .x86_64);
    return .{
        .fuse_cmp_branch = model.fuses(.cmp_branch),
        .fuse_arith_branch = model.fuses(.arith_branch),
    };
}

test "x86_64 capsForModel reads cascadelake-sp fusion: cmp and arith on" {
    const caps = capsForModel(mm.modelFor(.@"cascadelake-sp"));
    try std.testing.expect(caps.fuse_cmp_branch);
    try std.testing.expect(caps.fuse_arith_branch);
}

/// Run the shared Wimmer allocation for `func`, on a throwaway clone so the caller's
/// function stays untouched, and hand its per-value segment lists to `f`. The Wimmer
/// allocation is the production allocator, so the split/re-home test gates measure
/// the same allocator that `compile` uses. This helper skips the address-fold
/// rewrite that `compile` runs first, since that rewrite only matters for functions
/// with foldable addresses, and the split/re-home test inputs are fold-free
/// arithmetic. A value's `[]wimmer.Segment` is ascending by `from`, and each
/// segment's `loc` is a register (`.reg`) or a spill slot (`.slot`), uniformly
/// across the gpr and xmm classes.
fn forEachWimmerSegments(allocator: std.mem.Allocator, func: *const Function, comptime f: fn ([]const wimmer.Segment, *usize) void) Error!usize {
    var work = try func.clone(allocator);
    defer work.deinit();
    try ir.critical_edge.splitCriticalEdges(allocator, &work);
    var desc = try x86_64RegDescription(allocator, &work);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, &work, &desc);
    defer walloc.deinit(allocator);
    var count: usize = 0;
    var it = walloc.segments.iterator();
    while (it.next()) |e| f(e.value_ptr.*, &count);
    return count;
}

/// Test hook: report how many values the shared Wimmer allocator split, where the
/// value's life spans more than one segment, for example a register prefix plus a
/// spill tail. Zero means no split occurred. The execution tests call this to assert
/// a case actually exercises the splitter before they check its results, since the
/// production allocator is Wimmer, so this measures the real path.
pub fn splitCountForTest(allocator: std.mem.Allocator, func: *const Function) Error!usize {
    const count = struct {
        fn f(segs: []const wimmer.Segment, c: *usize) void {
            if (segs.len > 1) c.* += 1;
        }
    };
    return forEachWimmerSegments(allocator, func, count.f);
}

/// Test hook: report how many values the shared Wimmer allocator re-homed, meaning a
/// value whose segment list holds a register (`.reg`) segment after a spill (`.slot`)
/// segment. Such a value was spilled, then brought back into a register for its
/// remaining tail uses. This hook exists so an execution test can prove a
/// reload-into-register actually fired, not merely that a value was spilled, which a
/// plain per-use slot read would also satisfy.
pub fn reHomeCountForTest(allocator: std.mem.Allocator, func: *const Function) Error!usize {
    const count = struct {
        fn f(segs: []const wimmer.Segment, c: *usize) void {
            var saw_slot = false;
            for (segs) |s| switch (s.loc) {
                .slot => saw_slot = true,
                .reg => if (saw_slot) {
                    c.* += 1;
                },
            };
        }
    };
    return forEachWimmerSegments(allocator, func, count.f);
}

/// Compile `func` to machine code plus its call relocations. The caller owns the
/// result. This function delegates to `compileWithCaps` with the inert (all-false)
/// `ModelCaps`, so it produces the same bytes no matter what `compileWithCaps` grows
/// to support.
pub fn compile(allocator: std.mem.Allocator, func: *const Function) Error!Compiled {
    return compileWithCaps(allocator, func, .{});
}

/// Like `compile`, but tuned by `caps` (see `ModelCaps`). `compile` is exactly this
/// function with `.{}`, where every flag is inert. So a caller that passes the
/// default caps gets the same allocation, with the model-tuned emission seams
/// (fusion) turned off.
///
/// This is the production register allocator: the shared Wimmer-Franz
/// linear-scan-on-SSA allocator. There is no fallback to the retired native linear
/// scan (`assignRegs`/`assignXmm`, now removed). The pipeline is exactly
/// `compileFunctionWimmerX86Fold`'s pipeline (splitCriticalEdges, then
/// addrfold.analyze, then applyFoldRewriteX86, then x86_64RegDescription, then
/// wimmer.allocate, then translateAllocationX86, then emitFromAllocation), but it
/// runs on an independently owned deep `clone`. This way the public `*const
/// Function` entry points (`selectFunction`, `selectFunctionForModel`, `compile`)
/// never touch a caller's function, since a caller may reuse or share its function
/// across backends. The real `caps` and the real `fold` analysis are threaded to
/// emission, unlike the differential compile's empty-fold, inert-caps compile.
pub fn compileWithCaps(allocator: std.mem.Allocator, func: *const Function, caps: ModelCaps) Error!Compiled {
    // This backend now lowers f16 here via F16C. The value is held as its f32
    // widening in an xmm register, all arithmetic uses scalar-single SSE, and
    // hardware vcvtph2ps/vcvtps2ph conversions run at the boundaries. The other
    // backends still reject f16 via `functionUsesF16`, but x86_64 no longer does.
    // This backend handles only scalar f16, so it still cleanly rejects f16 nested
    // in a vector or aggregate, which would otherwise fall through to the
    // raw-vector path and miscompile the half lanes.
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;
    if (func.blockCount() == 0) return error.Unsupported;

    // The Wimmer pipeline mutates the function: splitCriticalEdges appends
    // forwarding blocks, and applyFoldRewriteX86 repoints folded pointers and
    // removes dead adds. But the public entry points take `*const Function`, so the
    // code works on an independently owned deep `clone` and leaves the caller's
    // function byte-for-byte pristine. This keeps every public `*const` signature
    // unchanged.
    var work = try func.clone(allocator);
    defer work.deinit();

    // Split critical edges first, before the code builds any numbering. The shared
    // resolver needs a block on every critical edge to place its shuffle moves.
    try ir.critical_edge.splitCriticalEdges(allocator, &work);

    // Neutralize every block unreachable from the entry, before the code builds any
    // numbering. A pass that raises a loop nest into one op could orphan its
    // blocks, since the IR has no block-delete. That would leave instructions that
    // use values defined in reachable blocks. The shared `buildIntervals` walks all
    // blocks, so such a value would get a live range built entirely from the dead
    // region, and would trip the allocator's SSA def-in-range assert. Emptying
    // every unreachable block's params, instructions, and terminator removes those
    // spurious uses. This backend emits every block, so it does not consume the
    // returned reachable set. `emitFromAllocation` detects a neutralized block by
    // its emptied shape (no instructions and a null terminator) and emits nothing
    // for it. This is a no-op for a function whose blocks are all reachable, so
    // output stays the same for every existing caller.
    const reachable = try ir.reachable.neutralizeUnreachable(allocator, &work);
    allocator.free(reachable);

    // Address-mode folding is a pre-allocation IR rewrite, so it stays sound under
    // the fold-agnostic shared Wimmer allocator, which reads only the actual IR
    // operands. `analyze` recognizes each foldable `p = arith_imm.add(base, imm);
    // load/store(p)`. `applyFoldRewriteX86` then repoints each folded mem op's
    // `ptr` to `base` and drops the dead adds in the clone, so the allocator keeps
    // `base` live to the load/store. The same analysis threads into emission via
    // `ctx.fold`. `folds` is keyed by the surviving mem inst, so `baseOf`/`offOf`
    // stay consistent with the rewritten IR.
    var fold = try addrfold.analyze(allocator, &work, {}, x86_64FoldOffset);
    defer fold.deinit(allocator);
    applyFoldRewriteX86(&work, &fold);

    var desc = try x86_64RegDescription(allocator, &work);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, &work, &desc);
    defer walloc.deinit(allocator);

    var ctx = Ctx{ .func = &work, .fold = &fold, .caps = caps };
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
    // `def_pos` is always a heap-owned dupe. The `&.{}` sentinel is a zero-length
    // slice with no backing allocation, so freeing it is a no-op. So an
    // unconditional free is safe.
    defer allocator.free(ctx.def_pos);
    defer {
        for (ctx.edge_moves) |es| allocator.free(es.moves);
        allocator.free(ctx.edge_moves);
    }

    var saved: std.ArrayList(Reg) = .empty;
    defer saved.deinit(allocator);
    var num_slots: u32 = 0;
    var xmm_slots: u32 = 0;
    try translateAllocationX86(allocator, &work, &walloc, &ctx, &num_slots, &xmm_slots, &saved);
    // The code does not re-sort `ctx.actions`. It came straight from
    // `walloc.actions`, already ordered ascending by `at`, with each same-position
    // cluster hazard-free through the shared `orderIntraActions`.
    const frame = try frameLayout(allocator, &ctx, &work, num_slots, xmm_slots, saved.items.len);
    var compiled = try emitFromAllocation(allocator, &ctx, &work, frame, saved.items);
    errdefer compiled.deinit(allocator);

    // Every `Reloc.symbol` is a borrowed slice into the emitting function's symbol
    // storage. The contract is that those names outlive `Compiled`, since the
    // caller's function does. Emission borrowed them from the clone, whose storage
    // `work.deinit` frees on return. So the code re-points each name to the
    // original `func`'s identical, longer-lived symbol string. The clone
    // re-interned symbols one to one, so the same name exists there. This keeps
    // the borrowed-name contract intact.
    for (compiled.relocs) |*r| r.symbol = rebindSymbolName(func, r.symbol);
    return compiled;
}

/// The `func`-owned symbol string equal to `name`. Every emitted relocation names a
/// callee that the function interned, since a `call`'s `symbol` indexes `func`'s
/// symbol table. So a match always exists. A miss would be a codegen bug, not a
/// runtime condition.
fn rebindSymbolName(func: *const Function, name: []const u8) []const u8 {
    var i: u32 = 0;
    while (i < func.symbolCount()) : (i += 1) {
        const s = func.symbolName(i);
        if (std.mem.eql(u8, s, name)) return s;
    }
    unreachable;
}

/// Compute the stack frame and fill the xmm and alloca bases on `ctx`. Frame
/// layout: general spills (8 bytes each), then the xmm spill area (32-byte slots
/// at a 16-aligned base, sized for a whole 256-bit ymm. A scalar or 128-bit value
/// uses the low half), then the alloca region. `num_pushed` is how many
/// callee-saved GPRs the prologue pushes before the `sub rsp`. A function that
/// makes calls must keep RSP 16-aligned at the call site. Entry RSP is 8 (mod 16),
/// and each push subtracts 8, so the `sub` amount must restore 16-alignment. That
/// means frame is congruent to 8 - 8*num_pushed (mod 16): +8 when an even count
/// was pushed (including 0, the default), +0 when odd. A leaf function, with no
/// call, needs no padding.
fn frameLayout(allocator: std.mem.Allocator, ctx: *Ctx, func: *const Function, num_slots: u32, xmm_slots: u32, num_pushed: usize) Error!i32 {
    const xmm_base: u64 = (@as(u64, num_slots) * 8 + 15) & ~@as(u64, 15);
    ctx.xmm_base = @intCast(xmm_base);
    const alloca_base: u64 = (xmm_base + @as(u64, xmm_slots) * 32 + 15) & ~@as(u64, 15);
    ctx.alloca_base = @intCast(alloca_base);
    const alloca_bytes = try computeAllocaSlots(allocator, func, &ctx.alloca_off);
    var frame_base = (alloca_base + alloca_bytes + 15) & ~@as(u64, 15);
    // A variadic function reserves the 176-byte System V register-save area at a
    // 16-byte-aligned frame offset, since the movaps xmm block needs 16-byte
    // alignment. 176 is a multiple of 16, so the frame top stays aligned. This
    // applies only when `is_variadic`, so a non-variadic frame is unchanged.
    if (func.is_variadic) {
        ctx.rsa_off = @intCast(frame_base);
        frame_base += 176;
    }
    // A leaf, non-variadic function needs no call-site alignment padding. The code
    // treats a variadic function like a function that makes calls, for alignment,
    // since its save area must be 16-byte-aligned in absolute terms for movaps.
    // The same padding rule delivers that: entry RSP is 8 mod 16, an even push
    // count keeps that, so a +8 pad makes RSP 16-aligned after the `sub`.
    if (!hasCall(func) and !func.is_variadic) return @intCast(frame_base);
    const pad: u64 = if (num_pushed % 2 == 0) 8 else 0;
    return @intCast(frame_base + pad);
}

/// Emit machine code from a finished, filled `ctx` (allocations, segments,
/// actions, edge moves), plus the computed `frame` and the callee-saved GPRs
/// `saved` to preserve. This is the emission half of `compile`, split out so the
/// shared Wimmer allocator (`compileFunctionWimmerX86`) can drive the same proven
/// emission. `saved` is empty for the default path, since its pool is
/// caller-saved, so the push/pop prologue is inert and the output stays the same.
/// The code pushes `saved` in the given order at the prologue, before `sub rsp`,
/// and pops it in reverse order at each epilogue, after `add rsp`.
fn emitFromAllocation(allocator: std.mem.Allocator, ctx: *Ctx, func: *const Function, frame: i32, saved: []const Reg) Error!Compiled {
    const nblocks = func.blockCount();
    const block_start = try allocator.alloc(usize, nblocks);
    defer allocator.free(block_start);

    // Prologue: push the used callee-saved GPRs, a no-op for the default path.
    // Then reserve the spill frame, then move each argument from its ABI register
    // to the entry parameter's location, either a register parallel move or a
    // store for a spilled parameter. The pushes sit above the frame, at higher
    // addresses than the `sub rsp` region, so they do not affect spill-slot
    // offsets.
    for (saved) |r| try ctx.put(allocator, encode.pushReg(r));
    if (frame > 0) try ctx.put(allocator, encode.aluImm(5, .rsp, frame, true)); // sub rsp, frame (64-bit stack ptr)
    ctx.frame = frame; // threaded so `va_start` can address the first incoming stack arg
    ctx.saved_count = @intCast(saved.len); // pushed regs shift the incoming stack args up
    // System V variadic register-save area: before the argument-homing move
    // below, which may overwrite the incoming arg registers, the code spills
    // every integer arg register into the save area, then conditionally spills
    // the xmm arg registers. The FP save is gated on AL, since the caller passes
    // the count of vector registers used, so a non-variadic caller with AL=0
    // skips the movaps block. The module emits this only for a variadic
    // definition, so a normal function's prologue stays the same.
    if (func.is_variadic) try emitVariadicSave(allocator, ctx);
    // System V passes general args in rdi, rsi, and so on, and fp args in xmm0,
    // xmm1, and so on, as separate sequences. So each class has its own
    // incoming-register index.
    const eparams = func.blockParams(@enumFromInt(0));
    var arg_moves: std.ArrayList(Move) = .empty;
    defer arg_moves.deinit(allocator);
    var gpr_i: usize = 0;
    var xmm_i: usize = 0;
    // Args beyond the ABI registers arrive on the stack, in declaration order. At
    // entry, [rsp+frame] holds the return address, since the prologue already did
    // `sub rsp, frame`. So the first stack arg is at [rsp+frame+8], the next at
    // [rsp+frame+16], and so on.
    var stack_arg: u32 = 0;
    for (eparams) |p| {
        if (isXmm(func, p)) {
            // A vector param also lives in an xmm register, so the code classifies
            // it by isXmm (float or vector), matching the call-site arg handling.
            // A scalar float moves or stores as 128-bit (movups). The extra
            // lanes are harmless. A 128-bit vector moves as movups. A 256-bit vector
            // moves as vmovups, so no lanes are dropped.
            if (xmm_i >= xmm_arg_regs.len) {
                // Fp stack arg. System V lays these out in 8-byte slots above the
                // return address, so slot k is at [rsp + frame + 8 + k*8]. The
                // code handles only a scalar float. A vector on the stack would
                // span several slots, which is rare and unsupported. It loads the
                // arg (scalar movss) into the param's home, via the xmm scratch
                // when the home is a spill slot.
                // An f64 stack arg would need a 64-bit `movsd` load, since movss
                // reads only its low 4 bytes and would truncate the double. So
                // the code rejects it fail-closed until a movsd path exists.
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
            if (gpr_i >= arg_regs.len) {
                // Gp stack arg. It arrives in the same incoming-stack sequence as
                // an fp stack arg, since System V numbers overflow args together
                // in declaration order, so it uses the shared `stack_arg` counter.
                // Above the frame, the prologue also pushed the used callee-saved
                // GPRs, so the return address sits at [rsp + frame +
                // saved_count*8], and stack-arg slot k is at [rsp + frame + 8 +
                // saved_count*8 + k*8]. This is the same formula `va_start` uses
                // for `overflow_arg_area` (see `emitVaStart`). The fp stack-home
                // arm above omits the `saved_count` term, but that arm is
                // unreachable, since an fp arg past the 8 xmm registers fails
                // closed at the call site. So only this gp arm ever homes an
                // incoming stack arg. It loads the arg into its home register, or
                // via a scratch register when the home is a spill slot.
                const off: i32 = frame + 8 + (@as(i32, @intCast(ctx.saved_count)) + @as(i32, @intCast(stack_arg))) * 8;
                stack_arg += 1;
                switch (ctx.loc(p)) {
                    .reg => |r| try ctx.put(allocator, encode.movFromStack(r, off)),
                    .spill => |slot| {
                        try ctx.put(allocator, encode.movFromStack(scratch1, off));
                        try ctx.put(allocator, encode.movToStack(slotDisp(slot), scratch1));
                    },
                    .xmm, .xmm_spill => unreachable,
                }
                continue;
            }
            const incoming = arg_regs[gpr_i];
            gpr_i += 1;
            switch (ctx.loc(p)) {
                .spill => |slot| try ctx.put(allocator, encode.movToStack(slotDisp(slot), incoming)),
                .reg => |r| if (r != incoming) try arg_moves.append(allocator, .{ .src = incoming, .dst = r }),
                .xmm, .xmm_spill => unreachable,
            }
        }
    }
    try parallelMove(allocator, ctx, &arg_moves);

    // `pos_base` is the current block's param-row position. It mirrors
    // translateAllocationX86's numbering exactly: param row, then one slot per
    // instruction, then one terminator slot. So the `ctx.pos` set per instruction
    // below equals the position each value's def was numbered at. When
    // `segments` is empty, the pos is otherwise unobservable, so the
    // pos-coupling assert is how the code catches a threading bug.
    var pos_base: u32 = 0;
    var action_cursor: usize = 0;
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        block_start[bi] = ctx.code.items.len;
        // A neutralized, unreachable block: `neutralizeUnreachable` emptied its
        // params, instructions, and terminator, so downstream passes see only the
        // reachable CFG. It carries no code, and no reachable branch targets it
        // (this is valid SSA), so the code emits nothing for it. The shared
        // Wimmer allocator still numbers it (param row plus 0 instructions plus
        // terminator slot equals 2 positions), so the code advances `pos_base`
        // over that span and discards any split actions the allocator recorded
        // inside a block that is never entered. When every block is reachable,
        // this branch is never taken, so output stays the same for every normal
        // function. The detector, no instructions and a null terminator, is
        // unambiguous: a block ending in a structured `if` has instructions, and
        // a forwarding block has a non-null jump terminator. So only a
        // neutralized block matches both.
        if (func.blockInsts(block).len == 0 and func.terminator(block) == null) {
            const dead_end = pos_base + 1; // the last position this empty block owns: param row, terminator slot
            while (action_cursor < ctx.actions.items.len and ctx.actions.items[action_cursor].at <= dead_end) {
                action_cursor += 1;
            }
            pos_base += 2;
            continue;
        }
        var terminated = false;
        const insts = func.blockInsts(block);
        for (insts, 0..) |inst, inst_idx| {
            // This position stays correct across a `continue`: the code derives
            // it from the block base plus the instruction index, so an early
            // exit cannot desync it. This equals translateAllocationX86's
            // numbering for this instruction: param row at `pos_base`, then one
            // position per instruction.
            ctx.pos = pos_base + 1 + @as(u32, @intCast(inst_idx));
            // Record a source-line row when this instruction starts a new line.
            // The byte offset equals the current code length, since x86 code is
            // already a byte stream.
            if (lineOf(func, inst)) |line| {
                if (line != ctx.last_line) {
                    try ctx.lines.append(allocator, .{ .offset = @intCast(ctx.code.items.len), .line = line });
                    ctx.last_line = line;
                }
            }
            // An instruction with a result must be emitted at exactly that
            // result's def position. This pins the emission numbering to
            // translateAllocationX86's numbering, and the assert must never fire.
            if (func.instResult(inst)) |r| std.debug.assert(ctx.pos == ctx.def_pos[@intFromEnum(r)]);
            // Drain split-boundary stores landing at this position before the
            // code emits the instruction. A tail-split store writes the
            // victim's register to its slot before the taker (the instruction
            // defined at `p`) computes its result into that same register. The
            // victim's value still occupies the register here, since its last
            // prefix use is before `p`, and nothing reused the register before
            // `p`. So the store captures the correct bits.
            while (action_cursor < ctx.actions.items.len and ctx.actions.items[action_cursor].at <= ctx.pos) {
                const act = ctx.actions.items[action_cursor];
                std.debug.assert(act.at == ctx.pos); // stores land on instruction positions only
                try emitSplitActionX86(allocator, ctx, act);
                action_cursor += 1;
            }
            // Thread the current block's instructions and this instruction's
            // index onto `ctx`, so both the `.icmp` arm (in `lowerInst`) and
            // `emitIf` can consult `fusesIntoNextIf` without growing every call
            // site's parameter list.
            ctx.cur_insts = insts;
            ctx.cur_idx = inst_idx;
            if (func.opcode(inst) == .@"if") {
                // `next_block` is the block emitted immediately after this one.
                // x86_64 emits all blocks in order, so `bi+1` is always the next
                // one emitted, or null at the last block. Whichever successor
                // edge targets `next_block` can fall through, so `emitIf` elides
                // that branch.
                const next_block: ?Block = if (bi + 1 < nblocks) @enumFromInt(bi + 1) else null;
                try emitIf(allocator, ctx, func.opcode(inst).@"if", block, next_block);
                terminated = true;
            } else {
                try lowerInst(allocator, ctx, inst);
            }
        }
        // The terminator shares the block-end position. An `.@"if"` terminator
        // is one of the instructions above, and it set `terminated`, so this
        // only positions a ret or jump. But the numbering still reserves a
        // terminator slot, whether or not the code emits one here.
        ctx.pos = pos_base + 1 + @as(u32, @intCast(insts.len));
        // Drain any split-boundary actions recorded at the terminator position
        // before the code emits the terminator. The Wimmer allocation can
        // re-home a value used only by `ret` (a non-edge-arg operand, hence
        // `is_intra`) at its next use, which is the terminator position
        // (`block_end`), and record a `.reload` there. The per-instruction drain
        // above only reaches `term_pos - 1`, so without this drain the reload
        // would never be emitted, and `ret` would read a register that was never
        // loaded. When no action lands on a terminator, the normal case, this
        // drains nothing and produces the same bytes.
        while (action_cursor < ctx.actions.items.len and ctx.actions.items[action_cursor].at <= ctx.pos) {
            const act = ctx.actions.items[action_cursor];
            std.debug.assert(act.at == ctx.pos); // only terminator-position actions remain here
            try emitSplitActionX86(allocator, ctx, act);
            action_cursor += 1;
        }
        if (!terminated) switch (func.terminator(block) orelse ir.function.Terminator{ .ret = ir.function.Ret.none() }) {
            .ret => |r| {
                switch (r.count) {
                    0 => {},
                    1 => {
                        const value = r.values[0];
                        if (isFloat(func, value)) {
                            const src = try ctx.useXmm(allocator, value, xmm_scratch);
                            if (src != xmm_ret) try ctx.put(allocator, encode.movupsRR(xmm_ret, src));
                        } else {
                            const src = try ctx.use(allocator, value, ret_reg);
                            if (src != ret_reg) try ctx.put(allocator, encode.movReg(ret_reg, src));
                        }
                    },
                    else => {
                        // A small struct returned by value across two register
                        // banks. Each value is one eightbyte, routed by its own
                        // type to the next return register of its bank: an
                        // integer eightbyte goes into rax or rdx, an SSE
                        // eightbyte goes into xmm0 or xmm1. The two banks count
                        // independently. x86_64 returns at most 2 eightbytes in
                        // registers. A bigger struct uses `.sret` instead.
                        if (r.count > 2) return error.Unsupported;
                        // Stage each eightbyte into a dedicated scratch register
                        // of its bank first (GPR r10 or r11, XMM xmm13 or xmm14,
                        // none of them a return register), so placing one
                        // eightbyte into a return register cannot clobber
                        // another eightbyte that still lives there. The sources
                        // can overlap the return set in any permutation.
                        const gp_stage = [_]Reg{ scratch1, scratch2 };
                        const fp_stage = [_]Xmm{ xmm_op0, xmm_op1 };
                        const gp_ret = [_]Reg{ .rax, .rdx };
                        const fp_ret = [_]Xmm{ .xmm0, .xmm1 };
                        var gp_staged: [2]Reg = undefined;
                        var fp_staged: [2]Xmm = undefined;
                        var gp_dst: [2]Reg = undefined;
                        var fp_dst: [2]Xmm = undefined;
                        var is_fp: [2]bool = undefined;
                        var gp_i: usize = 0;
                        var fp_i: usize = 0;
                        for (r.slice(), 0..) |value, i| {
                            if (isFloat(func, value)) {
                                const src = try ctx.useXmm(allocator, value, fp_stage[fp_i]);
                                if (src != fp_stage[fp_i]) try ctx.put(allocator, encode.movupsRR(fp_stage[fp_i], src));
                                fp_staged[fp_i] = fp_stage[fp_i];
                                fp_dst[fp_i] = fp_ret[fp_i];
                                is_fp[i] = true;
                                fp_i += 1;
                            } else {
                                const src = try ctx.use(allocator, value, gp_stage[gp_i]);
                                if (src != gp_stage[gp_i]) try ctx.put(allocator, encode.movReg(gp_stage[gp_i], src));
                                gp_staged[gp_i] = gp_stage[gp_i];
                                gp_dst[gp_i] = gp_ret[gp_i];
                                is_fp[i] = false;
                                gp_i += 1;
                            }
                        }
                        var gp_p: usize = 0;
                        var fp_p: usize = 0;
                        for (0..r.count) |i| {
                            if (is_fp[i]) {
                                try ctx.put(allocator, encode.movupsRR(fp_dst[fp_p], fp_staged[fp_p]));
                                fp_p += 1;
                            } else {
                                try ctx.put(allocator, encode.movReg(gp_dst[gp_p], gp_staged[gp_p]));
                                gp_p += 1;
                            }
                        }
                    },
                }
                if (frame > 0) try ctx.put(allocator, encode.aluImm(0, .rsp, frame, true)); // add rsp, frame (64-bit stack ptr)
                // Epilogue: restore the callee-saved GPRs in reverse push order,
                // empty for the default path, so the output stays the same, then
                // return. The `add rsp` already closed the frame, so RSP now
                // points at the topmost pushed register.
                var si: usize = saved.len;
                while (si > 0) {
                    si -= 1;
                    try ctx.put(allocator, encode.popReg(saved[si]));
                }
                try ctx.put(allocator, encode.ret());
            },
            .jump => |j| {
                // `next_block` is the next block emitted (`bi+1`), or null at
                // the last block. A jump to it falls through, so `emitJump`
                // elides the `jmp`.
                const next_block: ?Block = if (bi + 1 < nblocks) @enumFromInt(bi + 1) else null;
                try emitJump(allocator, ctx, j, block, next_block);
            },
        };
        // Advance to the next block's param row: param row (1), plus one slot
        // per instruction, plus one terminator slot, reserved unconditionally,
        // matching translateAllocationX86's per-block final increment.
        pos_base = pos_base + 2 + @as(u32, @intCast(insts.len));
    }
    // Every store must have drained. A tail-split store lands at the taker's def
    // position, which is always an instruction position the loop above visits,
    // so no store can outlive it.
    std.debug.assert(action_cursor == ctx.actions.items.len);

    for (ctx.fixups.items) |f| {
        const rel: i32 = @intCast(@as(i64, @intCast(block_start[f.target])) - @as(i64, @intCast(f.at + 4)));
        std.mem.writeInt(u32, ctx.code.items[f.at..][0..4], @bitCast(rel), .little);
    }
    return .{ .code = try ctx.code.toOwnedSlice(allocator), .relocs = try ctx.relocs.toOwnedSlice(allocator), .lines = try ctx.lines.toOwnedSlice(allocator) };
}

/// Emit one split-boundary drain action. A GPR `store` writes `reg` to its slot.
/// A `reload` brings a slot back into `reg`. A `move` copies `move_from` into
/// `reg`, as a register re-home. The XMM variants (`is_xmm`) mirror these through
/// movups (128-bit, lossless for the scalar floats the Wimmer path allows, or a
/// 32-byte slot's low 16 bytes) or vmovups for a 256-bit ymm. `slot_to_slot`
/// (Wimmer bridge gap #7, mirrors aarch64 isel) re-homes a spilled value from
/// `move_from_slot` to `slot` without ever giving it a value register. It
/// reloads the source slot into the class scratch register (gpr
/// `move_scratch`/r11, xmm `xmm_scratch`/xmm15, both already reserved out of
/// every pool, so this touch can never conflict with a live value), then stores
/// the scratch register straight back out to the destination slot, at the same
/// width the store/reload arms use (movups/vmovups, or the plain gpr mov). The
/// plain gpr split actions are `store` and `reload`, so those two arms produce
/// the same bytes as the pre-extraction inline drain. `move`, `slot_to_slot`,
/// and the xmm arms are reachable only through the shared Wimmer translation.
/// An identity `move` emits nothing.
fn emitSplitActionX86(allocator: std.mem.Allocator, ctx: *Ctx, act: SplitAction) Error!void {
    if (act.is_xmm) {
        const wide = isWide(ctx.func, act.value);
        const disp = ctx.xmmDisp(act.slot);
        switch (act.kind) {
            .store => try ctx.put(allocator, if (wide) encode.vmovupsStore(disp, act.xreg) else encode.movupsStore(disp, act.xreg)),
            .reload => try ctx.put(allocator, if (wide) encode.vmovupsLoad(act.xreg, disp) else encode.movupsLoad(act.xreg, disp)),
            .move => if (act.xreg != act.xmove_from) try ctx.put(allocator, if (wide) encode.vmovupsRR(act.xreg, act.xmove_from) else encode.movupsRR(act.xreg, act.xmove_from)),
            .slot_to_slot => {
                const disp_src = ctx.xmmDisp(act.move_from_slot);
                try ctx.put(allocator, if (wide) encode.vmovupsLoad(xmm_scratch, disp_src) else encode.movupsLoad(xmm_scratch, disp_src));
                try ctx.put(allocator, if (wide) encode.vmovupsStore(disp, xmm_scratch) else encode.movupsStore(disp, xmm_scratch));
            },
        }
        return;
    }
    switch (act.kind) {
        .store => try ctx.put(allocator, encode.movToStack(slotDisp(act.slot), act.reg)),
        .reload => try ctx.put(allocator, encode.movFromStack(act.reg, slotDisp(act.slot))),
        .move => if (act.reg != act.move_from) try ctx.put(allocator, encode.movReg(act.reg, act.move_from)),
        .slot_to_slot => {
            try ctx.put(allocator, encode.movFromStack(move_scratch, slotDisp(act.move_from_slot)));
            try ctx.put(allocator, encode.movToStack(slotDisp(act.slot), move_scratch));
        },
    }
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

/// Round the f16 value held in the low f32 lane of `reg` to nearest-even half,
/// and re-widen it, in place. This is the per-op IEEE rounding of the f16
/// emulation. An f16 arithmetic result, an int/f32/f64-to-f16 convert, or an f16
/// constant is first computed in f32. This function then narrows it to a half
/// (vcvtps2ph, round-to-nearest-even) and widens it back (vcvtph2ps), so the
/// register again holds an exact half value. Skipping this step would keep f32
/// precision, which is wrong for f16 semantics, since each op must round to
/// nearest-even half. Both F16C ops are 128-bit and operate in place, so the
/// function needs no scratch register.
fn roundToHalf(allocator: std.mem.Allocator, ctx: *Ctx, reg: Xmm) Error!void {
    try ctx.put(allocator, encode.vcvtps2ph(reg, reg, 0)); // imm8 = 0 -> round-to-nearest-even
    try ctx.put(allocator, encode.vcvtph2ps(reg, reg));
}

/// Patch a 4-byte PC-relative field at `field` (a `jcc`/`jmp` rel32) to reach
/// `target`. Both are byte offsets into the current code buffer. The
/// displacement is measured from the end of the field, the standard x86
/// relative-branch rule. The code uses this for local branches inside a single
/// instruction's expansion (the variadic prologue AL gate and `va_arg`), which
/// never cross a block boundary and so need no `ctx.fixups` entry.
fn patchRel32(ctx: *Ctx, field: usize, target: usize) void {
    const rel: i32 = @intCast(@as(i64, @intCast(target)) - @as(i64, @intCast(field + 4)));
    std.mem.writeInt(u32, ctx.code.items[field..][0..4], @bitCast(rel), .little);
}

/// Emit the System V variadic register-save-area fill from the prologue, only
/// when `func.is_variadic`. The save area is 176 bytes at `ctx.rsa_off`: a
/// 48-byte GP block (rdi, rsi, rdx, rcx, r8, r9 at +0, +8, ..., +40), then a
/// 128-byte XMM block (xmm0 through xmm7 at +48, +64, ..., +176, 16-byte
/// stride). The GP spills run unconditionally. The XMM spills are gated on AL,
/// which the caller sets to the count of vector registers it used to pass
/// arguments, so an int-only or fixed-arity caller (AL=0) skips the movaps
/// block: `test al,al; je .skip; movaps ...; .skip:`.
fn emitVariadicSave(allocator: std.mem.Allocator, ctx: *Ctx) Error!void {
    for (arg_regs, 0..) |r, i| {
        try ctx.put(allocator, encode.movToStack(ctx.rsa_off + @as(i32, @intCast(i)) * 8, r));
    }
    try ctx.put(allocator, encode.testAlAl());
    try ctx.put(allocator, encode.jcc(.e, 0)); // je .skip (ZF set when the caller used no xmm regs)
    const je_field = ctx.code.items.len - 4;
    for (xmm_arg_regs, 0..) |x, i| {
        try ctx.put(allocator, encode.movapsStore(ctx.rsa_off + 48 + @as(i32, @intCast(i)) * 16, x));
    }
    patchRel32(ctx, je_field, ctx.code.items.len); // .skip lands right after the movaps block
}

/// The System V count of this variadic function's fixed parameters that landed
/// in argument registers. Integer and pointer params consume GP registers,
/// capped at 6. Float and double params consume XMM registers, capped at 8.
/// `va_start` seeds `gp_offset` and `fp_offset` from these counts, so the first
/// `va_arg` reads the first unnamed argument. The classification (`isXmm`) is
/// exactly the prologue's param-homing split.
const NamedRegCounts = struct { gp: u32, fp: u32 };
fn namedRegCounts(func: *const Function) NamedRegCounts {
    const eparams = func.blockParams(@enumFromInt(0));
    const n_fixed = @min(func.num_fixed_params, @as(u32, @intCast(eparams.len)));
    var gp: u32 = 0;
    var fp: u32 = 0;
    for (eparams[0..n_fixed]) |p| {
        if (isXmm(func, p)) fp += 1 else gp += 1;
    }
    return .{ .gp = @min(gp, 6), .fp = @min(fp, 8) };
}

/// Expand `va_start(list)` for System V. This writes the four `va_list` fields
/// at `[list]`, where `list` is the object's address: u32 gp_offset @0, u32
/// fp_offset @4, ptr overflow_arg_area @8, ptr reg_save_area @16. `gp_offset`
/// and `fp_offset` start past the named register args. `reg_save_area` points
/// at the prologue's save area. `overflow_arg_area` points at the first
/// incoming stack argument ([rsp+frame+8], above the return address). Only r10
/// is used as a value scratch, since the list base stays in its own register or
/// r11 through `use`, and is read-only. So the code clobbers no allocated
/// value.
fn emitVaStart(allocator: std.mem.Allocator, ctx: *Ctx, list: Value) Error!void {
    const counts = namedRegCounts(ctx.func);
    const lp = try ctx.use(allocator, list, scratch2); // list base (own reg or r11), read-only
    try ctx.put(allocator, encode.movImm(scratch1, @intCast(counts.gp * 8), false)); // gp_offset
    try ctx.put(allocator, encode.movToMem32(lp, 0, scratch1));
    try ctx.put(allocator, encode.movImm(scratch1, @intCast(48 + counts.fp * 16), false)); // fp_offset
    try ctx.put(allocator, encode.movToMem32(lp, 4, scratch1));
    // overflow_arg_area equals the address of the first incoming stack arg: rsp
    // + frame + 8 + 8*num_pushed. The `+8` steps over the return address. The
    // `8*num_pushed` steps over the callee-saved registers the prologue pushed
    // above the frame, since each one shifts the incoming stack args one slot
    // higher.
    try ctx.put(allocator, encode.leaFromStack(scratch1, ctx.frame + 8 + 8 * @as(i32, @intCast(ctx.saved_count))));
    try ctx.put(allocator, encode.movToMem(lp, 8, scratch1));
    try ctx.put(allocator, encode.leaFromStack(scratch1, ctx.rsa_off)); // reg_save_area
    try ctx.put(allocator, encode.movToMem(lp, 16, scratch1));
}

/// Expand `va_arg(list, result-type)` for System V. This fetches the next
/// variadic argument and advances the `va_list` at `[list]`. There are two
/// shapes, by result class:
///   Int or pointer: if gp_offset < 48, p = reg_save_area + gp_offset, and
///     gp_offset += 8. Otherwise p = overflow_arg_area, and overflow_arg_area
///     += 8. result = *p.
///   Double or float: the same, with fp_offset < 176, a save-area stride of
///     16, and an overflow stride of 8.
/// The list base stays in its own register, or r11 via `use`. r10 holds the
/// running offset, then the argument pointer. The offset-field bumps happen
/// directly in memory (`aluMemImm8`), so only two GPRs are ever live (the base
/// and r10), plus the result register. The compare, and the je/jmp, are local
/// branches backpatched within this instruction, mirroring the prologue AL
/// gate, and they never cross a block.
fn emitVaArg(allocator: std.mem.Allocator, ctx: *Ctx, list: Value, result: Value) Error!void {
    const func = ctx.func;
    const lp = try ctx.use(allocator, list, scratch2); // base (own reg or r11), read-only
    const is_fp = isXmm(func, result);
    const off_field: i32 = if (is_fp) 4 else 0; // fp_offset @4, gp_offset @0
    const off_limit: i32 = if (is_fp) 176 else 48;
    const save_stride: i8 = if (is_fp) 16 else 8;
    try ctx.put(allocator, encode.movFromMem32(scratch1, lp, off_field)); // r10 = offset (zero-extended)
    try ctx.put(allocator, encode.aluImm(7, scratch1, off_limit, false)); // cmp off, limit
    try ctx.put(allocator, encode.jcc(.ae, 0)); // jae .overflow (off is unsigned, 0..limit)
    const jae_field = ctx.code.items.len - 4;
    // Register path: p = reg_save_area + off. Bump the offset field in memory.
    try ctx.put(allocator, encode.addRegMem(scratch1, lp, 16)); // r10 += reg_save_area -> p
    try ctx.put(allocator, encode.aluMemImm8(0, lp, off_field, save_stride, false)); // offset += stride
    try ctx.put(allocator, encode.jmp(0)); // jmp .load
    const jmp_field = ctx.code.items.len - 4;
    // Overflow path: p = overflow_arg_area. overflow_arg_area += 8.
    const overflow_target = ctx.code.items.len;
    try ctx.put(allocator, encode.movFromMem(scratch1, lp, 8)); // r10 = overflow_arg_area -> p
    try ctx.put(allocator, encode.aluMemImm8(0, lp, 8, 8, true)); // overflow_arg_area += 8 (64-bit)
    // .load: result = *p (p in r10).
    const load_target = ctx.code.items.len;
    if (is_fp) {
        const rd = try ctx.dstXmm(result, xmm_scratch);
        try ctx.put(allocator, if (isDouble(func, result)) encode.movsdLoadMem(rd, scratch1, 0) else encode.movssLoadMem(rd, scratch1, 0));
        try ctx.storeXmm(allocator, result, rd);
    } else {
        const rd = ctx.dst(result, scratch1);
        const signed = isSigned(func, result);
        try ctx.put(allocator, switch (intBits(func, result)) {
            0...8 => if (signed) encode.movsxByteFromMem(rd, scratch1, 0) else encode.movzxByteFromMem(rd, scratch1, 0),
            9...16 => if (signed) encode.movsxWordFromMem(rd, scratch1, 0) else encode.movzxWordFromMem(rd, scratch1, 0),
            17...32 => if (signed) encode.movsxdFromMem(rd, scratch1, 0) else encode.movFromMem32(rd, scratch1, 0),
            else => encode.movFromMem(rd, scratch1, 0),
        });
        try ctx.store(allocator, result, rd);
    }
    patchRel32(ctx, jae_field, overflow_target);
    patchRel32(ctx, jmp_field, load_target);
}

/// The call this store serves is a direct call. `result_opt` (on that call) is
/// the call's scalar result, or null for a void call and for a struct-by-value
/// return. This function stores a struct-by-value `.registers` return's
/// register set into the dest slot at rsp-relative displacement `doff`. Each
/// `pieces[0..count]` entry is one return eightbyte. An integer piece stores
/// the next GPR (rax, rdx) with a full-register `mov`. A float piece stores the
/// next XMM (xmm0, xmm1) at its own width (`movsd` for 8 bytes, `movss` for 4).
/// The two banks count independently, matching the ret-placement order. A
/// pure-integer return keeps the pre-existing behavior byte-for-byte (`mov
/// [rsp + i*8], {rax|rdx}`).
fn emitStructRetStoreX86(allocator: std.mem.Allocator, ctx: *Ctx, pieces: [4]ir.function.RetPiece, count: u8, doff: i32) Error!void {
    const gp_ret = [_]Reg{ .rax, .rdx };
    const fp_ret = [_]Xmm{ .xmm0, .xmm1 };
    var gp_i: usize = 0;
    var fp_i: usize = 0;
    for (0..count) |i| {
        const p = pieces[i];
        const off = doff + @as(i32, p.offset);
        if (p.fp) {
            try ctx.put(allocator, if (p.bytes == 8) encode.movsdStore(off, fp_ret[fp_i]) else encode.movssStore(off, fp_ret[fp_i]));
            fp_i += 1;
        } else {
            try ctx.put(allocator, encode.movToStack(off, gp_ret[gp_i]));
            gp_i += 1;
        }
    }
}

/// Handles a struct-returning call (`c.ret_dest` set). A scalar result comes
/// back in rax or xmm0. A struct returned in registers leaves each eightbyte in
/// the next return register of its bank (integer rax/rdx, SSE xmm0/xmm1), and
/// this function stores that eightbyte set into the frontend destination slot
/// `c.ret_dest`, a frame-relative alloca. Everything before the result
/// handling is the argument placement and the call itself, unchanged, so a
/// plain scalar or void call emits exactly the same bytes as before
/// struct-by-value returns were added.
fn lowerDirectCall(allocator: std.mem.Allocator, ctx: *Ctx, c: ir.function.Call, result_opt: ?Value) Error!void {
    const func = ctx.func;
    // Move arguments into the System V argument registers: register sources
    // move through a parallel move, and spilled sources reload straight into
    // the arg register. Then `call` the symbol via a relocation. The result is
    // in RAX. Caller-saved registers are clobbered. Values must not be live
    // across the call, other than the result. General args go in rdi, rsi, and
    // so on, and fp args go in xmm0, xmm1, and so on. Each class has its own
    // incoming-register index.
    const args = func.valueList(c.args);
    var moves: std.ArrayList(Move) = .empty;
    defer moves.deinit(allocator);
    var xmm_moves: std.ArrayList(XmmMove) = .empty;
    defer xmm_moves.deinit(allocator);
    var gi: usize = 0;
    var xi: usize = 0;
    // Count the gp stack args: each non-xmm arg past the 6 gp registers. An fp
    // arg past the 8 xmm registers stays fail-closed, since integer structs
    // never produce one. Only a call with stack args opens a local rsp window,
    // so a call that fits inside the 6 gp or 8 xmm registers emits exactly the
    // same bytes as before stack args were handled.
    var n_stack: usize = 0;
    {
        var g: usize = 0;
        var x: usize = 0;
        for (args) |arg| {
            if (isXmm(func, arg)) {
                if (x >= xmm_arg_regs.len) return error.Unsupported; // fp stack args not handled
                x += 1;
            } else {
                if (g >= arg_regs.len) n_stack += 1;
                g += 1;
            }
        }
    }
    // Round the stack-arg block up to a multiple of 16, so rsp stays 16-aligned
    // at the `call`, per System V. This frame keeps rsp 16-aligned, so
    // subtracting a 16-multiple preserves that. `stack_bytes` is 0 when there
    // are no stack args, so every rsp-relative reload below adds 0 and stays
    // byte-identical for a call that fits inside the argument registers.
    const stack_bytes: i32 = @intCast((n_stack * 8 + 15) & ~@as(usize, 15));
    // Open the local rsp window before any argument placement. This lets the
    // stack-arg stores run while every reg arg still holds its value, so no
    // reg-arg parallel move can clobber a stack arg that the allocator happened
    // to place in an argument register.
    if (n_stack > 0) try ctx.put(allocator, encode.aluImm(5, .rsp, stack_bytes, true)); // sub rsp, stack_bytes
    // Place the gp stack args at [rsp + k*8], with k in declaration order,
    // first, while the reg args are untouched. A reg source stores straight to
    // the slot. A spilled source reloads through scratch1 (r10, reserved from
    // allocation, so it holds no live value), then stores. A spill slot moved
    // up by `stack_bytes` when rsp dropped, so its reload adds that bias.
    if (n_stack > 0) {
        var gk: usize = 0;
        var k: usize = 0;
        for (args) |arg| {
            if (isXmm(func, arg)) continue;
            defer gk += 1;
            if (gk < arg_regs.len) continue;
            switch (ctx.loc(arg)) {
                .reg => |src| try ctx.put(allocator, encode.movToStack(@as(i32, @intCast(k)) * 8, src)),
                .spill => |slot| {
                    try ctx.put(allocator, encode.movFromStack(scratch1, slotDisp(slot) + stack_bytes));
                    try ctx.put(allocator, encode.movToStack(@as(i32, @intCast(k)) * 8, scratch1));
                },
                else => unreachable,
            }
            k += 1;
        }
    }
    for (args) |arg| {
        if (isXmm(func, arg)) {
            const dst = xmm_arg_regs[xi];
            xi += 1;
            switch (ctx.loc(arg)) {
                .xmm => |src| if (src != dst) try xmm_moves.append(allocator, .{ .src = src, .dst = dst, .wide = isWide(func, arg) }),
                .xmm_spill => {}, // reloaded after the parallel move
                else => unreachable,
            }
        } else {
            if (gi >= arg_regs.len) {
                gi += 1; // a gp stack arg, already placed above
                continue;
            }
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
                const disp = ctx.xmmDisp(ctx.loc(arg).xmm_spill) + stack_bytes;
                try ctx.put(allocator, if (isWide(func, arg)) encode.vmovupsLoad(dst, disp) else if (isVector(func, arg)) encode.movupsLoad(dst, disp) else encode.movssLoad(dst, disp));
            }
        } else {
            if (gi >= arg_regs.len) {
                gi += 1; // a gp stack arg, already placed above
                continue;
            }
            const dst = arg_regs[gi];
            gi += 1;
            if (ctx.loc(arg) == .spill) try ctx.put(allocator, encode.movFromStack(dst, slotDisp(ctx.loc(arg).spill) + stack_bytes));
        }
    }
    // System V variadic rule: AL holds the number of xmm registers used to
    // pass arguments (`xi` after the placement loop). A fixed-arity call emits
    // nothing here, so it stays byte-identical. AL is the low byte of RAX,
    // which is caller-saved and holds no argument, so this write clobbers
    // nothing live, and it sits right before the call. The rsp window does not
    // touch RAX, so setting AL here, before the call, is unaffected by it.
    if (c.is_variadic) try ctx.put(allocator, encode.movAlImm(@intCast(xi)));
    try ctx.put(allocator, encode.callRel(0));
    try ctx.relocs.append(allocator, .{ .offset = ctx.code.items.len - 4, .symbol = func.symbolName(c.symbol) });
    // Close the local rsp window. System V is caller-cleaned. The result is
    // still in rax.
    if (n_stack > 0) try ctx.put(allocator, encode.aluImm(0, .rsp, stack_bytes, true)); // add rsp, stack_bytes
    if (result_opt) |result| {
        if (isXmm(func, result)) {
            const rd = try ctx.dstXmm(result, xmm_scratch); // fp result comes back in xmm0
            if (rd != xmm_ret) try ctx.put(allocator, if (isWide(func, result)) encode.vmovupsRR(rd, xmm_ret) else encode.movupsRR(rd, xmm_ret));
            try ctx.storeXmm(allocator, result, rd);
        } else {
            const rd = ctx.dst(result, scratch1);
            if (rd != .rax) try ctx.put(allocator, encode.movReg(rd, .rax));
            try ctx.store(allocator, result, rd);
        }
    }
    if (c.ret_dest) |dest| {
        // A small struct returned in registers. The callee left each eightbyte
        // in the next return register of its bank (integer rax/rdx, SSE
        // xmm0/xmm1). Store them into the frontend destination slot, a
        // frame-relative alloca, addressed rsp-relative like the `.alloca`
        // arm. The rsp window, if any, is already closed above, so rsp is back
        // at the frame base, and the return registers still hold the values,
        // so the code needs no scratch register.
        const doff = ctx.alloca_base + @as(i32, @intCast(ctx.alloca_off.get(dest).?));
        try emitStructRetStoreX86(allocator, ctx, c.ret_pieces, c.ret_regs, doff);
    }
}

fn lowerInst(allocator: std.mem.Allocator, ctx: *Ctx, inst: ir.function.Inst) Error!void {
    const func = ctx.func;
    // A folded address-add is dead. Every use of its result was rerouted to the
    // base by the fold, so it claims no register, excluded from the intervals,
    // and emits nothing. The code skips it before the result unwrap below,
    // since an arith_imm has a result, mirroring the `.prefetch` no-op drop.
    // With the empty analysis (the Wimmer path, the test hooks), `isDeadAdd` is
    // always false, so this stays byte-identical.
    if (ctx.fold.isDeadAdd(inst)) return;
    if (func.opcode(inst) == .store) {
        // `store` produces no result, so the code handles it before the result
        // unwrap below.
        const st = func.opcode(inst).store;
        // A folded store addresses `[base + disp32]`. `baseOf` yields the fold
        // base, the add's lhs, and `offOf` yields the displacement. Both are
        // the raw ptr and 0 when unfolded, so the non-folding case stays
        // byte-identical.
        const base = try ctx.use(allocator, ctx.fold.baseOf(func, inst), scratch2);
        const disp: i32 = @intCast(ctx.fold.offOf(inst));
        if (isXmm(func, st.value)) {
            const val = try ctx.useXmm(allocator, st.value, xmm_op0);
            if (isHalf(func, st.value)) {
                // Store a 16-bit IEEE half. Narrow the held f32 (lane 0) to a
                // half with vcvtps2ph (round-to-nearest-even), move it to a
                // gpr, and write exactly 2 bytes. The held value is already an
                // exact half, so the narrow is lossless. The code narrows into
                // xmm_scratch, so it does not clobber `val`, which useXmm never
                // reloads into the scratch.
                try ctx.put(allocator, encode.vcvtps2ph(xmm_scratch, val, 0));
                try ctx.put(allocator, encode.movdFromXmm(scratch1, xmm_scratch));
                try ctx.put(allocator, encode.movToMem16(base, disp, scratch1));
            } else {
                try ctx.put(allocator, if (isVector(func, st.value)) encode.movupsStoreMem(base, disp, val) else if (isDouble(func, st.value)) encode.movsdStoreMem(base, disp, val) else encode.movssStoreMem(base, disp, val));
            }
        } else {
            const val = try ctx.use(allocator, st.value, scratch1);
            // Store the value's own width, so the code writes exactly the
            // object's bytes, never more: an 8-bit store writes 1 byte, a
            // 16-bit store writes 2, a 32-bit store writes 4, and a 64-bit or
            // pointer store writes 8. A wider store would clobber the next
            // element of a tightly packed array, for example an i8 store.
            try ctx.put(allocator, switch (intBits(func, st.value)) {
                0...8 => encode.movToMem8(base, disp, val),
                9...16 => encode.movToMem16(base, disp, val),
                17...32 => encode.movToMem32(base, disp, val),
                else => encode.movToMem(base, disp, val),
            });
        }
        return;
    }
    if (func.opcode(inst) == .prefetch) {
        // A software prefetch hint. It has no result, and x86-64 codegen has no
        // need for one here, so the code simply drops it, a valid no-op
        // lowering of a hint.
        return;
    }
    if (func.opcode(inst) == .va_start) {
        // Initialize the `va_list` object. It has no result, so, like `store`
        // and `prefetch` above, the code handles it before the result unwrap
        // below.
        return emitVaStart(allocator, ctx, func.opcode(inst).va_start.list);
    }
    if (func.opcode(inst) == .va_end) {
        // `va_end` is a no-op on System V, since the `va_list` holds no owned
        // resource.
        return;
    }
    if (func.opcode(inst) == .call_indirect) {
        // Indirect call through a function pointer, for example the software
        // sampler helper, which returns nothing and writes its result through
        // a pointer arg, so this has no result. Stage the target into r10,
        // which survives the arg moves and the call and is never an arg
        // register. Then move the arguments into the System V arg registers,
        // then `call r10`.
        const c = func.opcode(inst).call_indirect;
        // A variadic call through a function pointer is not supported yet. The
        // code fails closed instead of emitting a call with no AL
        // vector-count, which would miscompile silently.
        if (c.is_variadic) return error.Unsupported;
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
        if (c.ret_dest) |dest| {
            // Store the register-return eightbytes into the dest slot,
            // rsp-relative. See `lowerDirectCall`. This indirect call opens no
            // rsp window, so rsp is at the frame base, and the return
            // registers still hold the values.
            const doff = ctx.alloca_base + @as(i32, @intCast(ctx.alloca_off.get(dest).?));
            try emitStructRetStoreX86(allocator, ctx, c.ret_pieces, c.ret_regs, doff);
        }
        return;
    }
    if (func.opcode(inst) == .call and func.instResult(inst) == null) {
        // A void or struct-returning direct call has no scalar result, so it
        // cannot go through the `.?` result unwrap below, mirroring the
        // `call_indirect` early handling above. A struct return still stores
        // its register-return eightbytes into `ret_dest`, handled by the
        // shared `lowerDirectCall`.
        return lowerDirectCall(allocator, ctx, func.opcode(inst).call, null);
    }
    const result = func.instResult(inst).?;
    switch (func.opcode(inst)) {
        .iconst => |c| {
            if (isXmm(func, result)) {
                // A float-typed integer constant, for example a zero-init. The
                // result lives in an xmm register, so the code materializes
                // the bits in a gpr and moves them across, never a plain
                // integer store into an xmm slot.
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
                // A 64-bit constant needs the full imm64 mov. `movImm` only
                // carries a sign-extended imm32, and `@intCast(c)` would
                // panic on any value outside i32, for example the 64-bit SWAR
                // popcount masks, or a 0x80000000 sign mask. For a 32-bit or
                // smaller result, the code takes the low 32 bits as a bit
                // pattern. `movImm` sign-extends them into the 64-bit
                // register, which a 32-bit use reads back correctly.
                if (intBits(func, result) > 32) {
                    try ctx.put(allocator, encode.movImm64(rd, @bitCast(c)));
                } else {
                    // A 32-bit or smaller constant: the zero-extending `mov
                    // r32, imm32` puts the exact low-32 bit pattern in the
                    // register and clears the upper 32 bits. This keeps the
                    // clean-upper-bits invariant that the width-aware ops
                    // rely on.
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
                // An f16 constant is materialized as its f32 widening. The
                // value is rounded to half first, `@as(f32, @as(f16, val))`,
                // keeping the invariant that an f16 in a register is its
                // exact-half f32 form. f32 keeps its full value.
                const bits: u32 = @bitCast(if (isHalf(func, result)) @as(f32, @as(f16, @floatCast(val))) else @as(f32, @floatCast(val)));
                try ctx.put(allocator, encode.movImm(scratch1, @bitCast(bits), false)); // 32-bit float bits into a gpr
                try ctx.put(allocator, encode.movdToXmm(rd, scratch1));
            }
            try ctx.storeXmm(allocator, result, rd);
        },
        .arith => |a| {
            if (isWide(func, result)) {
                // AVX 256-bit is three-operand and non-destructive: `dst =
                // v<op>ps src1, src2` directly. So the code needs no copy and
                // no dst==src alias handling.
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
                // SSE two-operand (`dst op= src`): packed-single (...ps),
                // scalar-double (...sd), or scalar-single (...ss), by type.
                // Copies use a 128-bit movups, so the code does not truncate
                // a double or vector. Spilled operands reload into op0 or
                // op1. A spilled result computes into xmm_scratch, which also
                // breaks the rd==rr alias.
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
                // An f16 op runs in the scalar-single (f32) form. Then its
                // result is rounded to nearest-even half, so the register
                // again holds an exact half value. This gives correct per-op
                // IEEE f16 semantics, since the operands were already exact
                // halves.
                if (isHalf(func, result)) try roundToHalf(allocator, ctx, work);
                try ctx.storeXmm(allocator, result, work);
                return;
            }
            const signed = isSigned(func, a.lhs);
            // The code uses 64-bit operand size only for i64 and wider types.
            // i32, i16, and i8 use 32-bit ops, whose result auto-zeroes the
            // upper 32 bits, so the value stays clean.
            const w64 = intBits(func, result) > 32;
            switch (a.op) {
                .div, .rem => {
                    // The dividend width follows the operands. A 32-bit
                    // divide sign-extends with cdq, not cqo, and uses the
                    // 32-bit idiv/div, reading only E(D)X:EAX, so a dirty
                    // upper half of RAX is ignored. For example, `u32
                    // divu(-1, 2) = 0x7FFFFFFF`.
                    const dw = intBits(func, a.lhs) > 32;
                    // The idiv/div instruction destroys RDX, for sign
                    // extension or the remainder, so a divisor allocated
                    // there must be copied out before the cdq/xor writes
                    // RDX. This only happens on the Wimmer path, since the
                    // default pool excludes RAX/RDX when dividing. So the
                    // guard is false, and emission stays byte-identical for
                    // the default path. `ctx.loc` reads the location without
                    // emitting, so the reload order below is unchanged when
                    // the guard is false.
                    const rhs_in_clobber = switch (ctx.loc(a.rhs)) {
                        .reg => |r| r == .rax or r == .rdx,
                        else => false,
                    };
                    var divisor: ?Reg = null;
                    if (rhs_in_clobber) {
                        const rr = try ctx.use(allocator, a.rhs, scratch2);
                        try ctx.put(allocator, encode.movReg(scratch2, rr));
                        divisor = scratch2;
                    }
                    try ctx.put(allocator, encode.movReg(.rax, try ctx.use(allocator, a.lhs, scratch1)));
                    try ctx.put(allocator, if (signed) (if (dw) encode.cqo() else encode.cdq()) else encode.xorr(.rdx, .rdx, dw));
                    const rr = divisor orelse try ctx.use(allocator, a.rhs, scratch2);
                    try ctx.put(allocator, if (signed) encode.idiv(rr, dw) else encode.divu(rr, dw));
                    const rd = ctx.dst(result, scratch1);
                    const res: Reg = if (a.op == .div) .rax else .rdx;
                    if (rd != res) try ctx.put(allocator, encode.movReg(rd, res));
                    try ctx.store(allocator, result, rd);
                },
                .shl, .shr => {
                    // The shift count goes in RCX, so `lhs` allocated to RCX
                    // must be copied out before RCX is overwritten with the
                    // count. Only the Wimmer path can put an operand in RCX,
                    // since the default pool excludes it when shifting. So
                    // the guard is false, and emission stays byte-identical
                    // for the default path.
                    const lhs_in_rcx = switch (ctx.loc(a.lhs)) {
                        .reg => |r| r == .rcx,
                        else => false,
                    };
                    var rl = try ctx.use(allocator, a.lhs, scratch1);
                    if (lhs_in_rcx) {
                        try ctx.put(allocator, encode.movReg(scratch1, rl));
                        rl = scratch1;
                    }
                    try ctx.put(allocator, encode.movReg(.rcx, try ctx.use(allocator, a.rhs, scratch2)));
                    const rd = ctx.dst(result, scratch1);
                    if (rd != rl) try ctx.put(allocator, encode.movReg(rd, rl));
                    // A 32-bit shr/sar shifts within 32 bits, filling from
                    // bit 31, which an i32 needs. A 64-bit shift would cross
                    // bit 31 or 32 and corrupt the result.
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
            // The code uses 64-bit operand size only for i64 and wider types.
            // A 32-bit or smaller result uses 32-bit ops.
            const w64 = intBits(func, result) > 32;
            switch (a.op) {
                .mul => {
                    const rd = ctx.dst(result, scratch1);
                    try ctx.put(allocator, encode.imulImm(rd, try ctx.use(allocator, a.lhs, scratch1), imm, w64));
                    try ctx.store(allocator, result, rd);
                },
                .mulh => return error.Unsupported, // mulh has no immediate form. It is expanded before isel.
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
                    // A 32-bit shr/sar fills from bit 31. The signExt
                    // lowering of i32.extend8_s/16_s (`(x << 24) >> 24`) and
                    // the clz/ctz smears rely on this. A 64-bit shift would
                    // sign-extend from bit 39 or 47 instead.
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
                // Per-lane float compare via cmpps, producing an all-ones or
                // all-zero lane mask that a later vector select reads. gt and
                // ge have no ordered SSE predicate, so the code swaps the
                // operands and uses lt or le. It compares in xmm_scratch,
                // since cmpps overwrites its first operand, then moves the
                // result to the result register.
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
                // Float compare via ucomiss and setcc. The bool result lives
                // in a gpr.
                const rd = ctx.dst(result, scratch1);
                if (cmp.op == .eq or cmp.op == .ne) {
                    // ucomiss sets PF on unordered (NaN). Ordered-equal is
                    // ZF=1 and PF=0, and not-equal is its inverse. So the
                    // code combines two setcc results.
                    const ra = try ctx.useXmm(allocator, cmp.lhs, xmm_op0);
                    const rb = try ctx.useXmm(allocator, cmp.rhs, xmm_op1);
                    try ctx.put(allocator, if (isDouble(func, cmp.lhs)) encode.ucomisd(ra, rb) else encode.ucomiss(ra, rb));
                    try ctx.put(allocator, encode.setcc(rd, if (cmp.op == .eq) .e else .ne));
                    try ctx.put(allocator, encode.movzxByte(rd, rd));
                    try ctx.put(allocator, encode.setcc(scratch2, if (cmp.op == .eq) .np else .p));
                    try ctx.put(allocator, encode.movzxByte(scratch2, scratch2));
                    // Combine the two zero-extended 0/1 bytes. The bool
                    // result is 32 bits or smaller, so the code uses a
                    // 32-bit and/or. This is correct for the clean 0/1
                    // operands either way.
                    try ctx.put(allocator, if (cmp.op == .eq) encode.andr(rd, scratch2, intBits(func, result) > 32) else encode.orr(rd, scratch2, intBits(func, result) > 32));
                    try ctx.store(allocator, result, rd);
                    return;
                }
                // lt and le swap the operands, so seta/setae (CF=0 and ZF=0)
                // excludes the unordered case.
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
            // Compare-into-branch fold (cmp_branch): when the model enables
            // it, and this icmp fuses into the immediately following `if`
            // (`fusesIntoNextIf`, the same predicate `emitIf` checks below),
            // the code skips the materialize-then-test path entirely. The
            // if's fused setup re-derives `cmp.lhs`/`cmp.rhs` and emits the
            // `cmp`/`jcc` itself. So this icmp's result is never read, since
            // its sole use, the if's cond, is folded away, and its
            // destination register can stay unwritten.
            if (ctx.caps.fuse_cmp_branch and fusesIntoNextIf(func, ctx.cur_insts, ctx.cur_idx)) return;
            const rl = try ctx.use(allocator, cmp.lhs, scratch1);
            const rr = try ctx.use(allocator, cmp.rhs, scratch2);
            const rd = ctx.dst(result, scratch1);
            // Compare at the operand width. An i32 compare sets flags from
            // the low 32 bits, so a dirty upper half, for example a
            // sign-extended incoming i32 arg, does not skew the result.
            try ctx.put(allocator, encode.cmp(rl, rr, intBits(func, cmp.lhs) > 32));
            try ctx.put(allocator, encode.setcc(rd, condOf(cmp.op, isSigned(func, cmp.lhs))));
            try ctx.put(allocator, encode.movzxByte(rd, rd));
            try ctx.store(allocator, result, rd);
        },
        .select => |s| {
            if (isVector(func, result)) {
                // Per-lane vector select from a cmpps mask: `result = (then &
                // mask) | (else & ~mask)`. This uses SSE1 and/andn/or only,
                // since there is no SSE4.1 blendv on the baseline. The code
                // builds both halves in the reserved scratch xmm registers,
                // so it never clobbers the operand registers.
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
            // x86 has no SSE conditional move, and the result register may
            // alias an operand. So the code lowers select as a two-armed
            // branch: each arm writes exactly one operand into the result,
            // touching nothing else. This works for both gpr and xmm
            // results.
            const c = try ctx.use(allocator, s.cond, scratch1);
            // Test at the condition's width, so a raw i32 cond with a dirty
            // upper half is not read as nonzero on its garbage bits.
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
            // Numeric conversions: int to float or back (32-bit int, f32, or
            // f64), int to int (low bits), and f32 to f64 or back.
            const src_float = isFloat(func, cv.value);
            const dst_float = isFloat(func, result);
            if (!src_float and dst_float) {
                const src = try ctx.use(allocator, cv.value, scratch1); // i32 in a gpr
                const rd = try ctx.dstXmm(result, xmm_scratch);
                // int to float: cvtsi2ss/cvtsi2sd. isDouble(f16) is false, so
                // an int-to-f16 conversion lands in the scalar-single form
                // first, then rounds to nearest-even half.
                try ctx.put(allocator, if (isDouble(func, result)) encode.cvtsi2sd(rd, src) else encode.cvtsi2ss(rd, src));
                if (isHalf(func, result)) try roundToHalf(allocator, ctx, rd);
                try ctx.storeXmm(allocator, result, rd);
            } else if (src_float and !dst_float) {
                // float to int, truncating toward zero. An f16 source is
                // held as its f32 widening, and isDouble(f16) is false, so
                // cvttss2si reads the right value.
                const src = try ctx.useXmm(allocator, cv.value, xmm_op0);
                const rd = ctx.dst(result, scratch1); // i32 result in a gpr
                try ctx.put(allocator, if (isDouble(func, cv.value)) encode.cvttsd2si(rd, src) else encode.cvttss2si(rd, src));
                try ctx.store(allocator, result, rd);
            } else if (src_float and dst_float) {
                // float to float. An f16 is always held as its f32 widening,
                // since isDouble(f16) is false. So f16-to-f32 falls out as a
                // same-width copy, and f16-to-f64 falls out as the plain
                // cvtss2sd widen. Only a narrowing to f16 needs special
                // handling: it must round to nearest-even half rather than
                // leave f32 precision in place.
                const src = try ctx.useXmm(allocator, cv.value, xmm_op0);
                const rd = try ctx.dstXmm(result, xmm_scratch);
                const sd = isDouble(func, cv.value);
                const dd = isDouble(func, result);
                if (isHalf(func, result)) {
                    // To f16: bring the source to f32, narrowing an f64
                    // source with cvtsd2ss, then round to nearest-even half
                    // and widen back to the held f32 form.
                    if (sd) try ctx.put(allocator, encode.cvtsd2ss(rd, src)) else if (rd != src) try ctx.put(allocator, encode.movupsRR(rd, src));
                    try roundToHalf(allocator, ctx, rd);
                } else if (sd == dd) {
                    // Same width, and the dest is not half: a plain copy
                    // (f32 to f32, f64 to f64, f16 to f32).
                    if (rd != src) try ctx.put(allocator, encode.movupsRR(rd, src));
                } else {
                    // Different widths, and the dest is not half: the base
                    // single-to-double convert, byte-identical to the
                    // pre-f16 behavior, and also the exact f16-to-f64 widen.
                    try ctx.put(allocator, if (dd) encode.cvtss2sd(rd, src) else encode.cvtsd2ss(rd, src));
                }
                try ctx.storeXmm(allocator, result, rd);
            } else {
                // int to int. Widening must sign-extend or zero-extend by
                // the source's signedness. A 32-bit x86 ALU result
                // auto-zero-extends into its 64-bit register
                // architecturally, so a negative i32 sits as, for example,
                // 0x00000000_FFFFFFFE. A bare move would carry that
                // zero-extended bit pattern into a wider (for example, i64)
                // destination unchanged, silently discarding the sign. This
                // mirrors aarch64 (sbfm/ubfm) and riscv64 (slli+srai/srli):
                // the code extends by the source's own signedness.
                // Same-width or narrowing keeps the low bits, a plain move,
                // byte-identical to the previous unconditional behavior for
                // those cases.
                const src_bits = intBits(func, cv.value);
                const dst_bits = intBits(func, result);
                const src = try ctx.use(allocator, cv.value, scratch2);
                const rd = ctx.dst(result, scratch1);
                if (dst_bits > src_bits and src_bits < 64) {
                    const signed = isSigned(func, cv.value);
                    try ctx.put(allocator, switch (src_bits) {
                        8 => if (signed) encode.movsxByte(rd, src) else encode.movzxByte(rd, src),
                        16 => if (signed) encode.movsxWord(rd, src) else encode.movzxWord(rd, src),
                        32 => if (signed) encode.movsxdReg(rd, src) else encode.movReg32(rd, src),
                        else => unreachable, // int widths are 8/16/32/64
                    });
                } else if (rd != src) {
                    try ctx.put(allocator, encode.movReg(rd, src)); // same width / narrowing
                }
                try ctx.store(allocator, result, rd);
            }
        },
        .unary => |u| {
            const result_val = func.instResult(inst);
            const src_float = isFloat(func, u.value);
            if (u.op == .reinterpret) {
                // int to float reinterpret: same width, movd/movq between
                // gpr and xmm.
                if (src_float) {
                    // float to int: xmm to gpr.
                    const src = try ctx.useXmm(allocator, u.value, xmm_op0);
                    if (result_val) |res| {
                        const rd = ctx.dst(res, scratch1);
                        try ctx.put(allocator, if (isDouble(func, u.value)) encode.movqFromXmm(rd, src) else encode.movdFromXmm(rd, src));
                        try ctx.store(allocator, res, rd);
                    }
                    // void result: src already loaded, nothing to do.
                } else {
                    // int to float: gpr to xmm.
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
            // Vector math unary: use the packed forms, so every lane is
            // computed, not just lane 0.
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
            // Float math unary ops: sqrt, ceil, floor, trunc, nearest. All
            // live in xmm. f16 is held as its f32 widening. A sqrtss/roundss
            // would leave an un-narrowed f32, with no round-to-half, so the
            // code does not lower an f16 unary math op. It rejects cleanly
            // rather than silently produce a value that is not a valid
            // half, mirroring the wasm and aarch64 backends.
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
        // A direct call. A struct-returning call (`ret_dest` set) and a void
        // call have no scalar result, and the code dispatches them to
        // `lowerDirectCall` before the result unwrap above. So the `result`
        // handed here is always the real scalar result.
        .call => |c| try lowerDirectCall(allocator, ctx, c, result),
        .struct_new => |sn| {
            // Build a SIMD vector from scalar lanes, the vectorizer's pack:
            // one insertps per lane, lane 0 last, so a field the allocator
            // placed in the result register keeps its value (in lane 0)
            // until its own insert reads it.
            if (!isVector(func, result)) return error.Unsupported;
            const fields = func.valueList(sn.fields);
            const rd = try ctx.dstXmm(result, xmm_scratch); // vectors do not spill (rd is a register)
            if (isWide(func, result)) {
                // AVX `<8 x f32>`: build the two 128-bit halves with
                // insertps, low in op0, high in op1, then join them into the
                // 256-bit result with vinsertf128.
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
            // Extract a lane of a SIMD vector to a scalar, the vectorizer's
            // unpack: a single pshufd moves that lane to lane 0. This is
            // pure, so dead extracts fall to DCE.
            if (!isVector(func, e.aggregate)) return error.Unsupported;
            const src = try ctx.useXmm(allocator, e.aggregate, xmm_op0);
            const rd = try ctx.dstXmm(result, xmm_scratch);
            if (isWide(func, e.aggregate) and e.index >= 4) {
                // The lane is in the high 128 bits of the ymm. Bring that
                // half down to an xmm first, then shuffle the lane, relative
                // to the half, into lane 0.
                try ctx.put(allocator, encode.vextractf128(xmm_op1, src, 1));
                try ctx.put(allocator, encode.pshufd(rd, xmm_op1, @intCast(e.index - 4)));
            } else {
                // Lane 0 through 3 lives in the low 128 bits, which a
                // 128-bit pshufd reads directly.
                try ctx.put(allocator, encode.pshufd(rd, src, @intCast(e.index)));
            }
            try ctx.storeXmm(allocator, result, rd);
        },
        .alloca => {
            // The alloca's result is the address of its stack slot. lea it
            // from rsp. The slot offset was assigned in
            // `computeAllocaSlots`. The result is a pointer (gpr).
            const off = ctx.alloca_base + @as(i32, @intCast(ctx.alloca_off.get(result).?));
            const rd = ctx.dst(result, scratch1);
            try ctx.put(allocator, encode.leaFromStack(rd, off));
            try ctx.store(allocator, result, rd);
        },
        .global_addr => |ga| {
            const rd = ctx.dst(result, scratch1);
            if (ga.via_got) {
                // GOT-indirect symbol address, a data import: `mov rd, qword
                // [rip + disp32]`, loading the symbol's address from its
                // `.got` slot, which the real `ld.so` fills via
                // `R_X86_64_GLOB_DAT`. disp32 = 0 is a placeholder. The
                // `.got_pcrel` reloc, emitting `R_X86_64_GOTPCREL`, records
                // the disp32 field's byte offset, patched to
                // `got_slot_vaddr - (site + 4)` once the GOT slot is placed.
                // The result register then holds the same runtime address
                // the direct `lea` path would compute, so a downstream
                // `load` through it produces the same bytes.
                try ctx.put(allocator, encode.movRipRel(rd, 0));
                try ctx.relocs.append(allocator, .{ .offset = ctx.code.items.len - 4, .symbol = func.symbolName(ga.symbol), .kind = .got_pcrel });
            } else {
                // PC-relative symbol address: `lea rd, [rip + disp32]`, with
                // disp32 = 0 here. A reloc records the disp32 field's byte
                // offset, patched once the symbol's runtime address is
                // known. See `applyGlobalReloc` in link.zig.
                try ctx.put(allocator, encode.leaRipRel(rd, 0));
                try ctx.relocs.append(allocator, .{ .offset = ctx.code.items.len - 4, .symbol = func.symbolName(ga.symbol), .kind = .pcrel_lea });
            }
            try ctx.store(allocator, result, rd);
        },
        .load => {
            // A folded load addresses `[base + disp32]`. `baseOf` yields the
            // fold base, the add's lhs, and `offOf` yields the displacement.
            // Both are the raw ptr and 0 when unfolded, so the non-folding
            // case stays byte-identical. An xmm result uses movups (vector),
            // movsd (f64), or movss (f32). Otherwise it uses a general mov.
            const base = try ctx.use(allocator, ctx.fold.baseOf(func, inst), scratch2);
            const disp: i32 = @intCast(ctx.fold.offOf(inst));
            if (isXmm(func, result)) {
                const rd = try ctx.dstXmm(result, xmm_scratch);
                if (isHalf(func, result)) {
                    // Load a 16-bit IEEE half and widen to the held f32
                    // form: movzx word into a gpr, movd into the xmm low
                    // lane, vcvtph2ps. Not movss, which would read 4 bytes
                    // from a 2-byte object, pulling in the next element.
                    try ctx.put(allocator, encode.movzxWordFromMem(scratch1, base, disp));
                    try ctx.put(allocator, encode.movdToXmm(rd, scratch1));
                    try ctx.put(allocator, encode.vcvtph2ps(rd, rd));
                } else {
                    try ctx.put(allocator, if (isVector(func, result)) encode.movupsLoadMem(rd, base, disp) else if (isDouble(func, result)) encode.movsdLoadMem(rd, base, disp) else encode.movssLoadMem(rd, base, disp));
                }
                try ctx.storeXmm(allocator, result, rd);
            } else {
                const rd = ctx.dst(result, scratch1);
                // Load exactly the value's own width, so the code reads no
                // bytes beyond the object. A wider load would pull garbage
                // from the next array element into the register. A narrow
                // load sign-extends a signed value and zero-extends an
                // unsigned one, and the extend targets a 32-bit register, so
                // the upper 32 bits end up clean.
                const signed = isSigned(func, result);
                try ctx.put(allocator, switch (intBits(func, result)) {
                    0...8 => if (signed) encode.movsxByteFromMem(rd, base, disp) else encode.movzxByteFromMem(rd, base, disp),
                    9...16 => if (signed) encode.movsxWordFromMem(rd, base, disp) else encode.movzxWordFromMem(rd, base, disp),
                    17...32 => if (signed) encode.movsxdFromMem(rd, base, disp) else encode.movFromMem32(rd, base, disp),
                    else => encode.movFromMem(rd, base, disp),
                });
                try ctx.store(allocator, result, rd);
            }
        },
        // Fetch the next variadic argument from the `va_list`. `va_start`
        // and `va_end` are handled earlier in this function, before the
        // result unwrap above, since they have no result.
        .va_arg => |va| try emitVaArg(allocator, ctx, va.list, result),
        else => return error.Unsupported,
    }
}

fn emitIf(allocator: std.mem.Allocator, ctx: *Ctx, cf: ir.function.If, pred: Block, next_block: ?Block) Error!void {
    const func = ctx.func;
    // Setup: emit the test that decides the branch, and yield its condition
    // code `cc`. Fused compare-and-branch (the cmp_branch fold): when
    // `caps.fuse_cmp_branch` is on, and the immediately preceding
    // instruction is a single-use integer icmp that is exactly this if's
    // condition (`fusesIntoNextIf`, the same predicate the icmp arm used to
    // skip its own materialization), the code loads the icmp's operands
    // fresh (mov/lea only, so EFLAGS stays untouched between the load and
    // the compare), sets the flags with `cmp`, and branches on the icmp's
    // own condition. Otherwise it materializes the boolean and tests it,
    // branching on `.ne` (nonzero) as before. The three layouts below then
    // all branch on `cc`, inverting it via `encode.invertCond` instead of
    // hardcoding `.ne`, so this is the only place either path diverges.
    //
    // Nested inside the cmp_branch arm: the arith_branch fold. When
    // `fusesArithIntoBranch` also holds, the arith at cur_idx-2, immediately
    // before the icmp, which is immediately before this if, already left ZF
    // = (its result == 0) as a side effect of computing that result. See
    // `fusesArithIntoBranch`'s doc comment: x86's add/sub/and, unlike
    // aarch64's, sets flags in their plain form. So the code emits nothing
    // here at all, neither loading the icmp's operands nor a `cmp`, and
    // branches on eq/ne directly off those flags. The arith itself was
    // lowered normally by `lowerInst`'s `.arith`/`.arith_imm` arm just
    // before this if, since this function never skips it, and the icmp
    // between them was already skipped by the cmp_branch fold above (its
    // sole use, this if's cond, is folded away). So no instruction sits
    // between the arith and the `jcc` this setup yields to, except the
    // arith's own possible flag-neutral spill-store `mov` (see
    // `Ctx.store`), leaving ZF intact at the branch.
    var cc: encode.Cond = .ne;
    if (ctx.cur_idx >= 1 and ctx.caps.fuse_cmp_branch and fusesIntoNextIf(func, ctx.cur_insts, ctx.cur_idx - 1)) {
        const cmp = func.opcode(ctx.cur_insts[ctx.cur_idx - 1]).icmp;
        if (fusesArithIntoBranch(func, ctx.cur_insts, ctx.cur_idx, ctx.caps.fuse_arith_branch and ctx.caps.fuse_cmp_branch)) {
            // Emit nothing here: the arith at cur_idx-2 already left ZF =
            // (result == 0) set. `cc` below is set from `cmp.op` (eq/ne
            // only, per `fusesArithIntoBranch`), so the `jcc` this setup
            // yields to branches directly on those flags.
        } else {
            const rl = try ctx.use(allocator, cmp.lhs, scratch1);
            const rr = try ctx.use(allocator, cmp.rhs, scratch2);
            // Compare at the operand width, exactly like the unfused icmp
            // lowering. An i32 compare sets flags from the low 32 bits, so
            // a dirty upper half does not skew the result.
            try ctx.put(allocator, encode.cmp(rl, rr, intBits(func, cmp.lhs) > 32));
        }
        cc = condOf(cmp.op, isSigned(func, cmp.lhs));
    } else {
        const cond = try ctx.use(allocator, cf.cond, scratch1);
        // Test at the condition's width, so a dirty upper half of a raw i32
        // cond is ignored.
        try ctx.put(allocator, encode.testReg(cond, cond, intBits(func, cf.cond) > 32));
    }

    // Layout selection. `then` and `else` are the two successor blocks.
    // `next` is the block emitted right after this one, or null at the last
    // block. Whichever edge targets `next` can fall through, so the code
    // elides its branch. The code checks `then` first, so a degenerate `if`
    // with then == else == next takes the then-fall-through layout. This
    // elides one branch, and both edges still reach the same block.
    const then_next = next_block != null and cf.then.target == next_block.?;
    const else_next = next_block != null and cf.@"else".target == next_block.?;

    if (then_next) {
        // `then` falls through. Keep `jcc -> then_start`, emit the
        // else-moves and `jmp else` inline, then the then-moves last, and
        // fall through to `then`, eliding the trailing `jmp then`. The jcc
        // still jumps forward over the else section to then_start, so its
        // displacement is unchanged.
        const jcc_at = try emitBranch(allocator, ctx, encode.jcc(cc, 0));
        try emitMoves(allocator, ctx, cf.@"else", pred); // ELSE-moves on the else path
        try emitBranchTo(allocator, ctx, encode.jmp(0), @intFromEnum(cf.@"else".target));
        const then_start = ctx.code.items.len;
        const rel: i32 = @intCast(@as(i64, @intCast(then_start)) - @as(i64, @intCast(jcc_at + 4)));
        std.mem.writeInt(u32, ctx.code.items[jcc_at..][0..4], @bitCast(rel), .little);
        try emitMoves(allocator, ctx, cf.then, pred); // THEN-moves, then fall through to THEN
        return;
    }

    if (else_next) {
        // `else` falls through. The code inverts the branch, so it jumps to
        // else_start when the cond is false (the else edge), and falls
        // through to the then-moves when the cond is true. It emits the
        // then-moves and `jmp then` inline, then the else-moves last, and
        // falls through to `else`. The inverted jcc's forward displacement
        // targets else_start.
        const jcc_at = try emitBranch(allocator, ctx, encode.jcc(encode.invertCond(cc), 0));
        try emitMoves(allocator, ctx, cf.then, pred); // THEN-moves on the then (not-taken) path
        try emitBranchTo(allocator, ctx, encode.jmp(0), @intFromEnum(cf.then.target));
        const else_start = ctx.code.items.len;
        const rel: i32 = @intCast(@as(i64, @intCast(else_start)) - @as(i64, @intCast(jcc_at + 4)));
        std.mem.writeInt(u32, ctx.code.items[jcc_at..][0..4], @bitCast(rel), .little);
        try emitMoves(allocator, ctx, cf.@"else", pred); // ELSE-moves, then fall through to ELSE
        return;
    }

    // Neither edge is `next`, or this is the last block: emit both branches
    // as before.
    const jcc_at = try emitBranch(allocator, ctx, encode.jcc(cc, 0));
    try emitMoves(allocator, ctx, cf.@"else", pred);
    try emitBranchTo(allocator, ctx, encode.jmp(0), @intFromEnum(cf.@"else".target));
    const then_start = ctx.code.items.len;
    const rel: i32 = @intCast(@as(i64, @intCast(then_start)) - @as(i64, @intCast(jcc_at + 4)));
    std.mem.writeInt(u32, ctx.code.items[jcc_at..][0..4], @bitCast(rel), .little);
    try emitMoves(allocator, ctx, cf.then, pred);
    try emitBranchTo(allocator, ctx, encode.jmp(0), @intFromEnum(cf.then.target));
}

fn emitJump(allocator: std.mem.Allocator, ctx: *Ctx, jump: ir.function.Jump, pred: Block, next_block: ?Block) Error!void {
    // The block-param edge-moves always run. A jump to the block emitted
    // immediately after this one falls through, so the code elides the
    // `jmp` and its fixup.
    try emitMoves(allocator, ctx, jump, pred);
    if (next_block != null and jump.target == next_block.?) return;
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

/// Edge moves into the target block's parameter locations, for both gpr and
/// xmm parameters. Register-to-register moves go through a parallel move,
/// per class. Spilled args and params are reloaded or stored via the
/// scratch register around the parallel move, so register sources are read
/// before they are overwritten.
fn emitMoves(allocator: std.mem.Allocator, ctx: *Ctx, jump: ir.function.Jump, pred: Block) Error!void {
    // Shared Wimmer path: the allocator already resolved this edge into an
    // ordered parallel-move sequence (params, live-through values, spills,
    // and cycles broken through the class scratch). So the code replays it
    // op by op and derives nothing. `edge_move_driven` is false for the
    // default path, so the derivation below runs unchanged there.
    if (ctx.edge_move_driven) {
        try emitEdgeMovesX86(allocator, ctx, pred, jump.target);
        return;
    }
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

/// The precomputed edge-move set for `pred -> succ`, or null when the edge needs no shuffle.
fn findEdgeMovesX86(ctx: *const Ctx, pred: Block, succ: Block) ?*const EdgeMoveSet {
    for (ctx.edge_moves) |*set| {
        if (set.pred == pred and set.succ == succ) return set;
    }
    return null;
}

/// Replay the precomputed, already-ordered edge moves for `pred -> succ`,
/// op by op, on the Wimmer path. The shared allocator resolved the parallel
/// move (it reads sources before it overwrites them, breaks cycles, and
/// routes any slot-to-slot shuffle through the class scratch), so each move
/// is a primitive reg or slot op.
fn emitEdgeMovesX86(allocator: std.mem.Allocator, ctx: *Ctx, pred: Block, succ: Block) Error!void {
    const set = findEdgeMovesX86(ctx, pred, succ) orelse return;
    for (set.moves) |m| try emitOneEdgeMoveX86(allocator, ctx, m);
}

/// Emit one ordered edge move. Class 0 (gpr): reg-to-reg `mov`, skipped when
/// equal, reg-to-slot store, slot-to-reg reload (8-byte slots at
/// `slotDisp`). Class 1 (xmm): the analogues through movups (128-bit,
/// lossless for a scalar float or a 128-bit vector, or a 32-byte slot's low
/// 16 bytes) or vmovups (256-bit ymm, selected by `m.wide`) at `xmmDisp`.
/// Both xmm forms are unaligned moves, so no aligned spill slot is needed. A
/// slot-to-slot op never appears, since the shared ordering expanded it
/// through the class scratch, so it is unreachable.
fn emitOneEdgeMoveX86(allocator: std.mem.Allocator, ctx: *Ctx, m: EdgeMove) Error!void {
    switch (m.class) {
        0 => switch (m.src) {
            .reg => |si| {
                const sr: Reg = @enumFromInt(@as(u4, @intCast(si)));
                switch (m.dst) {
                    .reg => |di| {
                        const dr: Reg = @enumFromInt(@as(u4, @intCast(di)));
                        if (sr != dr) try ctx.put(allocator, encode.movReg(dr, sr));
                    },
                    .slot => |ds| try ctx.put(allocator, encode.movToStack(slotDisp(ds), sr)),
                }
            },
            .slot => |ss| switch (m.dst) {
                .reg => |di| try ctx.put(allocator, encode.movFromStack(@enumFromInt(@as(u4, @intCast(di))), slotDisp(ss))),
                .slot => unreachable, // slot->slot was expanded through the class scratch
            },
        },
        1 => switch (m.src) {
            .reg => |si| {
                const sx: Xmm = @enumFromInt(@as(u4, @intCast(si)));
                switch (m.dst) {
                    .reg => |di| {
                        const dx: Xmm = @enumFromInt(@as(u4, @intCast(di)));
                        if (sx != dx) try ctx.put(allocator, if (m.wide) encode.vmovupsRR(dx, sx) else encode.movupsRR(dx, sx));
                    },
                    .slot => |ds| try ctx.put(allocator, if (m.wide) encode.vmovupsStore(ctx.xmmDisp(ds), sx) else encode.movupsStore(ctx.xmmDisp(ds), sx)),
                }
            },
            .slot => |ss| switch (m.dst) {
                .reg => |di| try ctx.put(allocator, if (m.wide) encode.vmovupsLoad(@enumFromInt(@as(u4, @intCast(di))), ctx.xmmDisp(ss)) else encode.movupsLoad(@enumFromInt(@as(u4, @intCast(di))), ctx.xmmDisp(ss))),
                .slot => unreachable,
            },
        },
        else => unreachable, // only gpr (0) and xmm (1) classes exist
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

/// The xmm counterpart of parallelMove. A 256-bit (AVX) move copies the
/// whole ymm with vmovups. A 128-bit move uses movups, which also covers a
/// scalar float, double, or `<4 x f32>` without truncation. The cycle break
/// saves through xmm_scratch at 256 bits whenever any move in the batch is
/// wide, so a wide register parked in the scratch keeps all its lanes.
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

/// Emit `result = value`, a register move plus a spill store if needed.
/// This is one arm of a select. It touches only `result` and `value`, so it
/// stays correct when those share a register.
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

// ===========================================================================
// Shared Wimmer-Franz register model: describe the x86_64 register file to
// the shared allocator. This is the description only. No allocation runs
// through it yet. There are two classes: gpr (0) and xmm (1). Unlike the
// retired native allocator, the gpr class includes the callee-saved
// registers, so a value live across a call can occupy one. The div/shift
// fixed-register needs are modeled as per-position clobber sites (fixed
// intervals) rather than a whole-function pool exclusion. Index equals the
// register's own enum value (`@intFromEnum`), and the class disambiguates
// the shared gpr/xmm index space.
// ===========================================================================

/// The caller-saved gpr set: rax, rcx, rdx, rsi, rdi, r8, r9. A call
/// clobbers exactly these, and the callee-saved set survives. R10 and R11
/// are scratch and never enter the pool.
const caller_saved_gpr = [_]Reg{ .rax, .rcx, .rdx, .rsi, .rdi, .r8, .r9 };
/// The callee-saved gpr set: rbx, r12, r13, r14, r15. This set is
/// allocatable, so cross-call values can use it, but it survives a call, so
/// no call clobbers it. Rbp and rsp are the frame and stack pointers, and
/// are excluded.
const callee_saved_gpr = [_]Reg{ .rbx, .r12, .r13, .r14, .r15 };
/// The number of allocatable xmm registers: xmm0 through xmm12. xmm13,
/// xmm14, and xmm15 are reserved scratch.
const xmm_allocatable_count: u16 = 13;

// The backend context the shared allocator threads through
// `classOf`/`useKind`. x86_64 needs no extra state, since its decisions
// read only the function, passed separately. So this is a zero-field
// singleton whose address is a stable, non-owned `ctx` pointer.
const X86_64RegCtx = struct {};
const x86_64_reg_ctx: X86_64RegCtx = .{};

/// `RegDescription.classOf` for x86_64. A value lives in the gpr class (0)
/// or the xmm class (1), as `isXmm` decides. A float or a SIMD vector is
/// xmm, everything else is gpr.
fn x86_64ClassOf(ctx: *const anyopaque, func: *const Function, v: Value) u16 {
    _ = ctx;
    return if (isXmm(func, v)) 1 else 0;
}

/// `RegDescription.useKind` for x86_64. An xmm (float or vector) operand is
/// `should_have_register`. It may read from its spill slot, since the
/// emitter's `useXmm` reloads a spilled xmm operand into a scratch xmm
/// before the instruction. So the allocator is free to leave it in a slot
/// when no register can hold it. This is required on x86 System V, which
/// has no callee-saved xmm register: an xmm value that is a call argument
/// and also live across the call cannot escape the call clobber into a
/// register, and it must live in a slot across it. The retired native
/// allocator force-spilled exactly these. A gpr operand stays the stricter
/// `must_have_register`. The gpr class does have callee-saved registers, so
/// a cross-call value escapes into one rather than needing a slot. Keeping
/// the stricter kind preserves the register-reload-at-use (re-home)
/// behavior the spill tests assert. `inst` is unused, since the kind
/// depends only on the operand's class. Both are the generic hook shape.
fn x86_64UseKind(ctx: *const anyopaque, func: *const Function, inst: ir.function.Inst, operand: Value) wimmer.UseKind {
    _ = ctx;
    _ = inst;
    return if (isXmm(func, operand)) .should_have_register else .must_have_register;
}

/// `RegDescription.copySource` for x86_64: report the source value when `v`'s defining
/// instruction is a PURE register copy of it, one the backend lowers to a bare `mov`
/// (or `movups`) that changes no bits. The shared allocator uses this to place the copy
/// destination on its source's register, which turns the copy into a self-move the
/// emission then elides (the `rd != src` guards on those branches). Only the EXACT
/// plain-copy convert cases are safe:
///   - int -> int convert that is same width or narrowing. Its `mov` copies the whole
///     64-bit register, so the destination keeps the source bits. A WIDENING int convert
///     emits movsx/movzx/movsxd (or a 32-bit mov that zero-extends), which changes bits,
///     so it is NOT a copy.
///   - float -> float convert between the SAME scalar view (f32 -> f32, f64 -> f64, or
///     the f16 -> f32 widen, since an f16 is held AS its f32 widening so that move is a
///     no-op copy) with neither a narrowing to f16 nor a single/double view change. Those
///     emit a bare `movups`. A narrowing to f16 rounds (roundToHalf), and an f32<->f64
///     view change emits cvtss2sd/cvtsd2ss, both real conversions that change bits.
/// Every other case (a cross-class int <-> float convert via cvtsi2ss/cvttss2si, a vector
/// operand, or any non-convert) returns null. The unused `ctx` is the generic hook shape.
fn x86_64CopySource(ctx: *const anyopaque, func: *const Function, v: Value) ?Value {
    _ = ctx;
    const inst = func.definingInst(v) orelse return null;
    const cv = switch (func.opcode(inst)) {
        .convert => |c| c,
        else => return null,
    };
    const src = cv.value;
    const src_x = isXmm(func, src);
    const dst_x = isXmm(func, v);
    if (!src_x and !dst_x) {
        // int -> int. Same width or narrowing lowers to a full-register `mov`. Only a
        // widening (a wider result from a source under 64 bits) sign/zero extends, so
        // exclude it, matching the emission's extend branch.
        const src_bits = intBits(func, src);
        const dst_bits = intBits(func, v);
        if (dst_bits > src_bits and src_bits < 64) return null;
        return src;
    }
    if (src_x and dst_x) {
        // float -> float. A SIMD vector reports xmm too, but a vector convert never emits
        // a bare `movups` whole-value copy here, so exclude it. A narrowing to f16 rounds,
        // and a single/double view change is a real convert. Everything else (same-view
        // scalar float, or the f16 -> f32 widen) is a bare `movups` copy.
        if (isVector(func, src) or isVector(func, v)) return null;
        if (isHalf(func, v)) return null;
        if (isDouble(func, src) != isDouble(func, v)) return null;
        return src;
    }
    // Cross-class int <-> float convert (cvtsi2ss/cvttss2si). Never a copy.
    return null;
}

/// Append a `u16` index for every register in `regs` to `list` (via `@intFromEnum`).
fn appendRegIndices(allocator: std.mem.Allocator, list: *std.ArrayList(u16), comptime R: type, regs: []const R) Error!void {
    for (regs) |r| try list.append(allocator, @intFromEnum(r));
}

/// The kind of fixed-register clobber a single instruction contributes at
/// its position. A call clobbers all caller-saved registers. A div or rem
/// needs rax and rdx. A shift needs rcx. Everything else clobbers nothing.
const ClobberKind = enum { none, call, div, shift };

/// Which fixed-register clobber `inst` contributes. A `div` or `rem` (arith
/// or arith_imm) uses rax and rdx. A `shl` or `shr` `arith` uses rcx. An
/// `arith_imm` shift has an immediate count, so it needs no rcx, unlike the
/// div/rem rax+rdx and shift rcx needs. A call clobbers every caller-saved
/// register.
fn clobberKindOf(func: *const Function, inst: ir.function.Inst) ClobberKind {
    return switch (func.opcode(inst)) {
        .call, .call_indirect => .call,
        .arith => |a| switch (a.op) {
            .div, .rem => .div,
            .shl, .shr => .shift,
            else => .none,
        },
        .arith_imm => |a| switch (a.op) {
            .div, .rem => .div,
            else => .none,
        },
        else => .none,
    };
}

/// Build the per-function x86_64 `RegDescription` the shared Wimmer-Franz
/// allocator consumes. There are two classes, and the physical-register
/// index equals the register's own enum value. Class 0 (gpr) is allocatable
/// over the caller-saved set plus the callee-saved set, so a cross-call
/// value can live in a callee-saved register instead of always spilling.
/// Class 1 (xmm) is xmm0 through xmm12. Entry params are pre-colored to
/// their System V ABI argument registers as hints. Each call, div/rem, and
/// shl/shr instruction becomes a per-position clobber site (a fixed
/// interval): a call clobbers the caller-saved gpr set and all xmm, a div
/// clobbers {rax, rdx}, and a shift clobbers {rcx}. The caller owns the
/// result and must `deinit` it. This function builds only the description.
/// No allocation runs here.
pub fn x86_64RegDescription(allocator: std.mem.Allocator, func: *const Function) Error!wimmer.RegDescription {
    // --- Class 0 (gpr): caller-saved + callee-saved, 8-byte slots. ---
    var gpr_alloc: std.ArrayList(u16) = .empty;
    errdefer gpr_alloc.deinit(allocator);
    try appendRegIndices(allocator, &gpr_alloc, Reg, &caller_saved_gpr);
    try appendRegIndices(allocator, &gpr_alloc, Reg, &callee_saved_gpr);
    const gpr_alloc_owned = try gpr_alloc.toOwnedSlice(allocator);
    errdefer allocator.free(gpr_alloc_owned);

    var gpr_cs: std.ArrayList(u16) = .empty;
    errdefer gpr_cs.deinit(allocator);
    try appendRegIndices(allocator, &gpr_cs, Reg, &callee_saved_gpr);
    const gpr_cs_owned = try gpr_cs.toOwnedSlice(allocator);
    errdefer allocator.free(gpr_cs_owned);

    // --- Class 1 (xmm): xmm0 through xmm12, no callee-saved, since System
    // V has no callee-saved xmm, 16-byte slots for a scalar float or a
    // whole vector. ---
    const xmm_alloc = try allocator.alloc(u16, xmm_allocatable_count);
    errdefer allocator.free(xmm_alloc);
    for (0..xmm_allocatable_count) |i| xmm_alloc[i] = @intCast(i);
    const xmm_cs = try allocator.alloc(u16, 0);
    errdefer allocator.free(xmm_cs);

    const classes = try allocator.alloc(wimmer.RegClass, 2);
    errdefer allocator.free(classes);
    classes[0] = .{ .name = "gpr", .allocatable = gpr_alloc_owned, .callee_saved = gpr_cs_owned, .slot_bytes = 8 };
    classes[1] = .{ .name = "xmm", .allocatable = xmm_alloc, .callee_saved = xmm_cs, .slot_bytes = 16 };

    // --- Entry params: the first 6 gpr params pin the ABI arg registers
    // rdi, rsi, rdx, rcx, r8, r9. The first 8 xmm params pin xmm0 through
    // xmm7. These are hints, since the prologue moves each param to its
    // assigned register anyway, so the hint just reduces moves. Params past
    // the first 6 gpr or 8 xmm arrive on the stack and are left to the
    // translation. Int and xmm use separate ABI counters. ---
    var ef: std.ArrayList(wimmer.FixedAssign) = .empty;
    errdefer ef.deinit(allocator);
    if (func.blockCount() != 0) {
        var gpr_idx: usize = 0;
        var xmm_idx: usize = 0;
        for (func.blockParams(@enumFromInt(0))) |p| {
            if (isXmm(func, p)) {
                if (xmm_idx < xmm_arg_regs.len) try ef.append(allocator, .{ .value = p, .class = 1, .reg = @intFromEnum(xmm_arg_regs[xmm_idx]) });
                xmm_idx += 1;
            } else {
                if (gpr_idx < arg_regs.len) try ef.append(allocator, .{ .value = p, .class = 0, .reg = @intFromEnum(arg_regs[gpr_idx]) });
                gpr_idx += 1;
            }
        }
    }
    const entry_fixed = try ef.toOwnedSlice(allocator);
    errdefer allocator.free(entry_fixed);

    // --- Clobber sites: one per call, div/rem, and shl/shr position, in
    // the same single-step numbering `buildIntervals` uses (block-param
    // row, one position per instruction, one terminator slot, over every
    // block), so the positions line up with the intervals. A call clobbers
    // the caller-saved gpr set and all xmm. A div clobbers {rax, rdx}. A
    // shift clobbers {rcx}. ---
    var sites: std.ArrayList(wimmer.CallSite) = .empty;
    var built: usize = 0;
    errdefer {
        for (sites.items[0..built]) |cs| {
            for (cs.clobbered) |cr| allocator.free(cr.regs);
            allocator.free(cs.clobbered);
        }
        sites.deinit(allocator);
    }
    {
        var pos: u32 = 0;
        for (0..func.blockCount()) |bi| {
            pos += 1; // block-parameter row
            for (func.blockInsts(@enumFromInt(bi))) |inst| {
                const clob = try buildClobber(allocator, clobberKindOf(func, inst));
                if (clob) |cr| {
                    errdefer freeClassRegs(allocator, cr);
                    try sites.append(allocator, .{ .pos = pos, .clobbered = cr });
                    built = sites.items.len;
                }
                pos += 1;
            }
            pos += 1; // terminator slot
        }
    }
    const call_sites = try sites.toOwnedSlice(allocator);
    errdefer {
        for (call_sites) |cs| {
            for (cs.clobbered) |cr| allocator.free(cr.regs);
            allocator.free(cs.clobbered);
        }
        allocator.free(call_sites);
    }

    // --- Scratch, indexed by class: the reserved registers the backend
    // already keeps out of every pool. Class 0 uses the parallel-move
    // scratch r11 (index 11). Class 1 uses xmm15 (index 15). ---
    const scratch = try allocator.alloc(u16, 2);
    errdefer allocator.free(scratch);
    scratch[0] = @intFromEnum(move_scratch);
    scratch[1] = @intFromEnum(xmm_scratch);

    return .{
        .classes = classes,
        .classOf = x86_64ClassOf,
        .useKind = x86_64UseKind,
        .copySource = x86_64CopySource,
        .coalesce_block_params = true,
        .coalesce_spill_slots = true,
        .entry_fixed = entry_fixed,
        .call_sites = call_sites,
        .scratch = scratch,
        .ctx = &x86_64_reg_ctx,
    };
}

/// Build the per-class clobber list for a clobber `kind`, or null when the
/// instruction clobbers nothing, so no site is recorded. A call clobbers
/// class 0 (the caller-saved gpr set) and class 1 (all allocatable xmm). A
/// div clobbers class 0 ({rax, rdx}). A shift clobbers class 0 ({rcx}). The
/// caller owns the returned slices and frees them via
/// `RegDescription.deinit`.
fn freeClassRegs(allocator: std.mem.Allocator, cr: []wimmer.ClassRegs) void {
    for (cr) |c| allocator.free(c.regs);
    allocator.free(cr);
}

fn buildClobber(allocator: std.mem.Allocator, kind: ClobberKind) Error!?[]wimmer.ClassRegs {
    switch (kind) {
        .none => return null,
        .call => {
            const gpr_clob = try allocator.alloc(u16, caller_saved_gpr.len);
            errdefer allocator.free(gpr_clob);
            for (caller_saved_gpr, 0..) |r, i| gpr_clob[i] = @intFromEnum(r);
            const xmm_clob = try allocator.alloc(u16, xmm_allocatable_count);
            errdefer allocator.free(xmm_clob);
            for (0..xmm_allocatable_count) |i| xmm_clob[i] = @intCast(i);
            const clob = try allocator.alloc(wimmer.ClassRegs, 2);
            clob[0] = .{ .class = 0, .regs = gpr_clob };
            clob[1] = .{ .class = 1, .regs = xmm_clob };
            return clob;
        },
        .div => {
            const gpr_clob = try allocator.alloc(u16, 2);
            errdefer allocator.free(gpr_clob);
            gpr_clob[0] = @intFromEnum(Reg.rax);
            gpr_clob[1] = @intFromEnum(Reg.rdx);
            const clob = try allocator.alloc(wimmer.ClassRegs, 1);
            clob[0] = .{ .class = 0, .regs = gpr_clob };
            return clob;
        },
        .shift => {
            const gpr_clob = try allocator.alloc(u16, 1);
            errdefer allocator.free(gpr_clob);
            gpr_clob[0] = @intFromEnum(Reg.rcx);
            const clob = try allocator.alloc(wimmer.ClassRegs, 1);
            clob[0] = .{ .class = 0, .regs = gpr_clob };
            return clob;
        },
    }
}

// ===========================================================================
// Run the shared Wimmer-Franz allocator and emit executable code through
// `emitFromAllocation`. This path is test-only, additional to the default
// `compile`/`selectFunction`, which stay untouched. The headline capability
// is that a value live across a call can occupy a callee-saved GPR (rbx,
// r12 through r15) via the push/pop prologue, instead of always spilling.
// SIMD vectors (128-bit xmm and 256-bit ymm) also flow through the class-1
// (xmm) maps. The shared `Move` carries the moved value, so every
// spill/reload, reg-move, and edge move picks its width (movups vs
// vmovups) from the value's IR type. All are unaligned moves, so no extra
// spill-slot alignment is needed.
// ===========================================================================

/// Map a shared gpr `wimmer.Location` to this backend's `Loc`. A register
/// index maps to the enum, and a per-class slot maps to a gpr spill slot.
fn wimmerGprLoc(loc: wimmer.Location) Loc {
    return switch (loc) {
        .reg => |ri| .{ .reg = @enumFromInt(@as(u4, @intCast(ri))) },
        .slot => |s| .{ .spill = s },
    };
}

/// Map a shared xmm `wimmer.Location` to this backend's `Loc`. A register
/// index maps to the Xmm enum, and a per-class slot maps to an xmm spill
/// slot.
fn wimmerXmmLoc(loc: wimmer.Location) Loc {
    return switch (loc) {
        .reg => |ri| .{ .xmm = @enumFromInt(@as(u4, @intCast(ri))) },
        .slot => |s| .{ .xmm_spill = s },
    };
}

/// Build the gpr drain action realizing `src -> dst` for `value` at `at`,
/// already translated into this backend's `Loc` space via `wimmerGprLoc`.
/// reg-to-slot becomes a `store`, slot-to-reg becomes a `reload`, reg-to-reg
/// becomes a `move`, and slot-to-slot becomes a `slot_to_slot` (Wimmer
/// bridge gap #7). `emitSplitActionX86` expands the slot-to-slot case into
/// a reload-then-store pair through the gpr class scratch, so it never
/// needs a value register of its own. This function is infallible: every
/// arm always succeeds, matching aarch64's `translateTransition`.
fn wimmerGprTransition(value: Value, src: Loc, dst: Loc, at: u32) SplitAction {
    return switch (src) {
        .reg => |sr| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .move, .value = value, .reg = dr, .move_from = sr },
            .spill => |ds| .{ .at = at, .kind = .store, .value = value, .reg = sr, .slot = ds },
            .xmm, .xmm_spill => unreachable, // a gpr-class transition never carries an xmm Loc
        },
        .spill => |ss| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .reload, .value = value, .reg = dr, .slot = ss },
            .spill => |ds| .{ .at = at, .kind = .slot_to_slot, .value = value, .slot = ds, .move_from_slot = ss },
            .xmm, .xmm_spill => unreachable,
        },
        .xmm, .xmm_spill => unreachable,
    };
}

/// The xmm analogue of `wimmerGprTransition`. `is_xmm` is set, and
/// `xreg`/`xmove_from` carry the Xmm. `slot_to_slot` expands through the
/// xmm class scratch, xmm15.
fn wimmerXmmTransition(value: Value, src: Loc, dst: Loc, at: u32) SplitAction {
    return switch (src) {
        .xmm => |sr| switch (dst) {
            .xmm => |dr| .{ .at = at, .kind = .move, .value = value, .is_xmm = true, .xreg = dr, .xmove_from = sr },
            .xmm_spill => |ds| .{ .at = at, .kind = .store, .value = value, .is_xmm = true, .xreg = sr, .slot = ds },
            .reg, .spill => unreachable, // an xmm-class transition never carries a gpr Loc
        },
        .xmm_spill => |ss| switch (dst) {
            .xmm => |dr| .{ .at = at, .kind = .reload, .value = value, .is_xmm = true, .xreg = dr, .slot = ss },
            .xmm_spill => |ds| .{ .at = at, .kind = .slot_to_slot, .value = value, .is_xmm = true, .slot = ds, .move_from_slot = ss },
            .reg, .spill => unreachable,
        },
        .reg, .spill => unreachable,
    };
}

/// Map a shared edge-move `wimmer.Location` to an `EdgeLoc`: a
/// class-relative register index or a slot.
fn edgeLocX86(loc: wimmer.Location) EdgeLoc {
    return switch (loc) {
        .reg => |ri| .{ .reg = ri },
        .slot => |s| .{ .slot = s },
    };
}

fn regLessThanX86(_: void, a: Reg, b: Reg) bool {
    return @intFromEnum(a) < @intFromEnum(b);
}

/// Translate a finished shared `wimmer.Allocation` into a filled `ctx`
/// (loc_of, segments, actions, edge_moves, def_pos), plus the per-class slot
/// counts and the callee-saved GPR push set. A whole-life value (one
/// segment) lands in `loc_of` exactly as the native allocate would leave it,
/// so the prologue's direct reads and the epilogue behave identically. A
/// genuinely split value lands in `segments`, and the code consumes the
/// shared allocator's already-ordered `walloc.actions` verbatim into
/// `ctx.actions` (one store, reload, move, or slot_to_slot per intra-block
/// boundary, in hazard-free order, see the loop below). The entry-param
/// moves are handled by the same prologue as the default path, since it
/// moves each ABI arg register to the param's location, whatever the
/// allocator chose, so this function imposes no ABI-register requirement.
/// Vectors (128-bit xmm, 256-bit ymm) ride the class-1 maps, with width
/// picked from each value's IR type. This function bails with
/// `error.Unsupported` only on a callee-saved xmm, which would be a model
/// bug, since System V has no callee-saved xmm register.
fn translateAllocationX86(
    allocator: std.mem.Allocator,
    func: *const Function,
    walloc: *const wimmer.Allocation,
    ctx: *Ctx,
    num_slots_out: *u32,
    xmm_slots_out: *u32,
    saved: *std.ArrayList(Reg),
) Error!void {
    // def_pos uses the same single-step numbering the shared allocator and
    // `emitFromAllocation` use (block-param row, one position per
    // instruction, one terminator slot, over every block). `ctx` owns it
    // immediately, so the caller's `defer` frees it on any later failure.
    const nval = func.valueCount();
    const def_pos = try allocator.alloc(u32, nval);
    ctx.def_pos = def_pos;
    @memset(def_pos, 0);
    {
        var pos: u32 = 0;
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

    std.debug.assert(walloc.slot_count_per_class.len == 2);
    num_slots_out.* = walloc.slot_count_per_class[0];
    xmm_slots_out.* = walloc.slot_count_per_class[1];

    var it = walloc.segments.iterator();
    while (it.next()) |e| {
        const value = e.key_ptr.*;
        const wsegs = e.value_ptr.*;
        std.debug.assert(wsegs.len > 0);
        // A vector value, including a 256-bit ymm, rides the same class-1
        // (xmm) maps as a scalar float. Its spill store/reload and
        // reg-to-reg move pick the width from the value's IR type in
        // `emitSplitActionX86` (movups for 128-bit, vmovups for 256-bit),
        // and the edge moves do the same via the `wide` flag set below.
        // Both use unaligned moves, so the existing frame alignment
        // suffices, and no 16- or 32-byte aligned spill slot is required.
        const is_x = isXmm(func, value);
        if (wsegs.len == 1) {
            try ctx.loc_of.put(allocator, value, if (is_x) wimmerXmmLoc(wsegs[0].loc) else wimmerGprLoc(wsegs[0].loc));
            continue;
        }
        const segs = try allocator.alloc(Segment, wsegs.len);
        for (wsegs, 0..) |ws, i| segs[i] = .{ .from = ws.from, .loc = if (is_x) wimmerXmmLoc(ws.loc) else wimmerGprLoc(ws.loc) };
        ctx.segments.put(allocator, value, segs) catch |err| {
            allocator.free(segs);
            return err;
        };
        // The code does not derive the intra-block re-home actions for
        // these transitions here. The shared allocator already emitted them
        // into `walloc.actions`, ordered per same-position cluster into a
        // hazard-free parallel-move sequence (`orderIntraActions`).
        // Consuming that list below, rather than re-deriving raw
        // per-transition actions and policing them with an ad-hoc hazard
        // detector, is what makes the same-position drain always safe.
    }

    // Consume the shared allocator's already-ordered intra-block actions.
    // Each is one primitive transfer at its position (`src -> dst` in the
    // shared per-class `Location` space). The code maps both sides into
    // this backend's `Loc` and turns each one into the matching
    // `SplitAction`. The list is ascending by `at`, with each same-position
    // cluster in hazard-free order, so appending it verbatim and draining
    // in order never clobbers a live value (the retired
    // `wimmerHasSamePosRegHazard`). A cross-block location change does not
    // appear here, since that is an edge move, translated below. Only
    // genuine mid-block re-homes appear.
    for (walloc.actions) |wa| {
        std.debug.assert(wa.class == 0 or wa.class == 1);
        const is_x = wa.class == 1;
        const src = if (is_x) wimmerXmmLoc(wa.src) else wimmerGprLoc(wa.src);
        const dst = if (is_x) wimmerXmmLoc(wa.dst) else wimmerGprLoc(wa.dst);
        const act = if (is_x) wimmerXmmTransition(wa.value, src, dst, wa.at) else wimmerGprTransition(wa.value, src, dst, wa.at);
        try ctx.actions.append(allocator, act);
    }

    // Map the callee-saved GPRs the allocation used to the prologue push
    // set. Class 0 only, since System V has no callee-saved xmm, so a
    // class-1 used-saved register would be a model bug.
    for (walloc.used_callee_saved) |us| {
        if (us.class != 0) return error.Unsupported;
        try saved.append(allocator, @enumFromInt(@as(u4, @intCast(us.reg))));
    }
    std.mem.sort(Reg, saved.items, {}, regLessThanX86);

    // Control-flow-edge moves: translate each ordered `wimmer.Move` into an
    // `EdgeMove`, keyed by (pred, succ). `emitMoves` replays them when
    // `edge_move_driven` is set.
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
            // A class-1 move of a 256-bit ymm needs vmovups. The width comes
            // from the moved value's IR type. The shared ordering routes
            // every step, including a scratch save, with the value whose
            // bits it transfers, so `wm.value` names the correct width even
            // for a cycle break.
            moves[i] = .{ .class = @intCast(wm.class), .src = edgeLocX86(wm.src), .dst = edgeLocX86(wm.dst), .wide = isWide(func, wm.value) };
        }
        try edge_sets.append(allocator, .{ .pred = wem.pred, .succ = wem.succ, .moves = moves });
    }
    ctx.edge_moves = try edge_sets.toOwnedSlice(allocator);
    ctx.edge_move_driven = true;
}

/// Rewrite `func` in place, so address folding stays sound under the
/// fold-agnostic shared Wimmer allocator. The shared allocator
/// (`wimmer.zig`) reads only the raw IR operands, so for a foldable `p =
/// arith_imm.add(base, imm); load(p)`, it sees `base` used only at the add,
/// lets `base` die there, and reuses its register after it. Emitting the
/// fold (`[base + disp32]`, with the add dropped) would then read a stale
/// register. This rewrite makes the fold visible to the allocator instead
/// of hiding it:
///   1. Repoint every folded load/store's `ptr` operand directly to its
///      fold base, so `wimmer`'s interval build sees `base` used at the
///      load/store position and keeps its live range correct.
///   2. Drop every now-dead address-add, since its result had no use left
///      once the folded ptr uses moved to the base, so the allocator
///      wastes no register on it.
/// `fold` stays consistent for emission: `folds` is keyed by the surviving
/// mem inst and holds the base and offset, so `baseOf` returns the (now
/// raw) ptr, equal to base, and `offOf` returns the displacement. The code
/// removes only dead adds, never a mem inst, so the offsets survive. This
/// runs on the caller's function (a Wimmer caller passes a throwaway
/// copy), after critical-edge splitting and before `wimmer.allocate`. It
/// stays sound across blocks: base dominated the add, and the add
/// dominated the load, so base dominates the load.
fn applyFoldRewriteX86(func: *Function, fold: *const addrfold.Analysis) void {
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
    // Drop the dead adds. Every use of a dead add was a folded ptr use, now
    // repointed to the base, so its result is unused. Assert that before
    // removing it, since a surviving use would mean dropping a live def, a
    // miscompile. Removal order does not matter, since no dead add's
    // result feeds another instruction.
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

/// Compile `func` through the shared Wimmer-Franz allocator, then emit
/// through the same proven `emitFromAllocation`. This path is test-only,
/// additional to the default `compile`. It runs the shared scan,
/// translates its target-independent `Allocation` into a filled `Ctx`, and
/// reuses the existing emission verbatim. It bails with `error.Unsupported`
/// on anything not faithfully translatable (see `translateAllocationX86`),
/// never a silent miscompile. This function takes `func` by mutable
/// pointer, since `splitCriticalEdges` inserts forwarding blocks in place,
/// and a differential caller builds two identical functions and compiles
/// one each way.
pub fn compileFunctionWimmerX86(allocator: std.mem.Allocator, func: *Function) Error!Compiled {
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;
    if (func.blockCount() == 0) return error.Unsupported;

    // Split critical edges first, mutating `func`, so the shared resolver's
    // no-critical-edge precondition holds, and the RegDescription, scan,
    // and emission all see one CFG. x86 emits edge moves inline, so a
    // forwarding block just carries the shuffle, same as the other
    // backends.
    try ir.critical_edge.splitCriticalEdges(allocator, func);

    var desc = try x86_64RegDescription(allocator, func);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, func, &desc);
    defer walloc.deinit(allocator);

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
    defer allocator.free(ctx.def_pos);
    defer {
        for (ctx.edge_moves) |es| allocator.free(es.moves);
        allocator.free(ctx.edge_moves);
    }

    var saved: std.ArrayList(Reg) = .empty;
    defer saved.deinit(allocator);
    var num_slots: u32 = 0;
    var xmm_slots: u32 = 0;
    try translateAllocationX86(allocator, func, &walloc, &ctx, &num_slots, &xmm_slots, &saved);
    // The code does not re-sort `ctx.actions`. It came straight from
    // `walloc.actions`, which the shared allocator already orders ascending
    // by `at`, with each same-position cluster in hazard-free order
    // (`orderIntraActions`). An unstable re-sort here would risk scrambling
    // that resolution.
    const frame = try frameLayout(allocator, &ctx, func, num_slots, xmm_slots, saved.items.len);
    return emitFromAllocation(allocator, &ctx, func, frame, saved.items);
}

/// Like `compileFunctionWimmerX86`, but with address-mode folding on: this
/// is the exact pipeline the production entry uses. It analyzes the folds,
/// then `applyFoldRewriteX86` repoints each folded mem op's `ptr` to its
/// base and removes the dead adds in place, so the fold is visible to the
/// fold-blind shared allocator, which reads only raw operands, and `base`
/// stays live to the load/store. The same analysis threads into emission
/// via `ctx.fold`: `folds` is keyed by the surviving mem inst, so `baseOf`
/// returns the (now raw) ptr, equal to base, and `offOf` returns the
/// displacement, consistent with the rewritten IR. The mem inst survives,
/// and only the add is removed, so the offset side-table stays valid. This
/// function is test-only here, since the differential exercises the
/// rewrite. It takes `func` by mutable pointer: `splitCriticalEdges` and
/// `applyFoldRewriteX86` mutate it in place, so a differential caller
/// builds two identical functions and compiles one each way.
pub fn compileFunctionWimmerX86Fold(allocator: std.mem.Allocator, func: *Function) Error!Compiled {
    if (ir.function.functionUsesCompositeF16(func)) return error.Unsupported;
    if (func.blockCount() == 0) return error.Unsupported;

    // Split critical edges first, mutating `func`, matching
    // `compileFunctionWimmerX86`.
    try ir.critical_edge.splitCriticalEdges(allocator, func);

    // Analyze before the rewrite, since it reads the `arith_imm.add` each
    // fold rests on, then rewrite the IR so the fold is visible to the
    // fold-blind shared allocator. `analyze` yields an empty analysis when
    // nothing folds, so this path degrades to
    // `compileFunctionWimmerX86`'s behavior on such a function.
    var fold = try addrfold.analyze(allocator, func, {}, x86_64FoldOffset);
    defer fold.deinit(allocator);
    applyFoldRewriteX86(func, &fold);

    var desc = try x86_64RegDescription(allocator, func);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, func, &desc);
    defer walloc.deinit(allocator);

    var ctx = Ctx{ .func = func, .fold = &fold };
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
    defer allocator.free(ctx.def_pos);
    defer {
        for (ctx.edge_moves) |es| allocator.free(es.moves);
        allocator.free(ctx.edge_moves);
    }

    var saved: std.ArrayList(Reg) = .empty;
    defer saved.deinit(allocator);
    var num_slots: u32 = 0;
    var xmm_slots: u32 = 0;
    try translateAllocationX86(allocator, func, &walloc, &ctx, &num_slots, &xmm_slots, &saved);
    const frame = try frameLayout(allocator, &ctx, func, num_slots, xmm_slots, saved.items.len);
    return emitFromAllocation(allocator, &ctx, func, frame, saved.items);
}

/// Visit every operand value read by `inst`, calling `f(ctx, value,
/// is_edge_arg)`. The block arguments of an `if` are edge args, since they
/// move along a control edge. Every other operand is an ordinary use. This
/// local walk defines the exact operand set the split-liveness `is_intra`
/// predicate scores, carrying the edge-arg flag that predicate needs.
fn forEachOperand(func: *const Function, inst: ir.function.Inst, fold: *const addrfold.Analysis, ctx: anytype, comptime f: fn (@TypeOf(ctx), Value, bool) void) void {
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
        // A folded load/store attributes its pointer use to the fold base,
        // the add's lhs, not the add's own result, so the base stays live
        // to the mem op, and the dead add's result gets no use. `baseOf`
        // returns the raw ptr when unfolded, the empty analysis, so the
        // non-folding case stays byte-identical. Keeping this local walk in
        // lockstep with the emission's operand reads is what keeps the
        // fold's liveness sound.
        .load => f(ctx, fold.baseOf(func, inst), false),
        .store => |st| {
            f(ctx, st.value, false);
            f(ctx, fold.baseOf(func, inst), false);
        },
        .prefetch => |pf| f(ctx, pf.ptr, false),
        // This backend does not yet expand `va_start` here, but the code
        // still visits it here, so regalloc sees `list` as a use ahead of
        // that.
        .va_start => |vs| f(ctx, vs.list, false),
        .va_arg => |va| f(ctx, va.list, false),
        .va_end => |ve| f(ctx, ve.list, false),
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
        .call => |c| {
            for (func.valueList(c.args)) |a| f(ctx, a, false);
            if (c.ret_dest) |rd| f(ctx, rd, false); // the register-return dest, read post-call
        },
        .call_indirect => |c| {
            f(ctx, c.target, false);
            for (func.valueList(c.args)) |a| f(ctx, a, false);
            if (c.ret_dest) |rd| f(ctx, rd, false); // the register-return dest, read post-call
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
        .ret => |r| for (r.slice()) |vv| f(ctx, vv, false),
        .jump => |j| for (func.blockArgs(j)) |a| f(ctx, a, true),
    }
}

const CountCtx = struct { target: Value, count: *usize };
fn countOperand(ctx: CountCtx, operand: Value, is_edge_arg: bool) void {
    _ = is_edge_arg;
    if (operand == ctx.target) ctx.count.* += 1;
}

/// Total operand uses of `v` across the whole function (instruction
/// operands, if/jump edge args, and terminators). This backs
/// `fusesIntoNextIf`'s single-use check. It is built on the shared
/// `forEachOperand`/`forEachTermOperand` walkers, the same operand
/// enumeration the local liveness computation uses, so it never drifts
/// from what the allocator considers a use. The code deliberately uses
/// `empty_fold` here rather than a real fold analysis: `fusesIntoNextIf`
/// only ever counts uses of an icmp's boolean result, never a pointer, so
/// address-fold operand rerouting, which only touches load/store pointer
/// operands, is irrelevant to this count either way.
fn countUses(func: *const Function, v: Value) usize {
    var count: usize = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| forEachOperand(func, inst, &empty_fold, CountCtx{ .target = v, .count = &count }, countOperand);
        if (func.terminator(block)) |term| forEachTermOperand(func, term, CountCtx{ .target = v, .count = &count }, countOperand);
    }
    return count;
}

/// Whether the integer `icmp` at `insts[idx]` fuses into an immediately
/// following `@"if"` whose condition it is, and whose only use it is. When
/// it fuses, the code skips the icmp materialization (`cmp`, `setcc`,
/// `movzx`), and the if emits a fused `cmp`, `jcc` on the icmp's operands
/// directly (see `emitIf`). This is the one eligibility predicate shared by
/// the icmp-skip (in `lowerInst`'s `.icmp` arm) and the fused `emitIf`, so
/// the two never disagree: no dangling or doubled compare.
///
/// This predicate is gated to integer/gpr operands, the plain icmp path. A
/// vector icmp (`cmpps` mask) or a float icmp (`ucomiss`/`ucomisd` plus
/// `setcc`) lowers through a different arm entirely and must keep its
/// current materialize-then-test path untouched.
///
/// This predicate itself carries no model gate. Both call sites
/// additionally require `ctx.caps.fuse_cmp_branch` before honoring it, so a
/// model without the fusion falls back to the materialize-then-test path
/// unchanged, byte-identical to before this fold existed.
fn fusesIntoNextIf(func: *const Function, insts: []const ir.function.Inst, idx: usize) bool {
    const cmp = switch (func.opcode(insts[idx])) {
        .icmp => |c| c,
        else => return false,
    };
    // Integer and gpr operands only: isVector and isFloat both route to a
    // different lowering (cmpps mask, or ucomiss/sd plus setcc) that this
    // fold must not touch.
    if (isVector(func, cmp.lhs) or isFloat(func, cmp.lhs)) return false;
    if (idx + 1 >= insts.len) return false; // must be immediately followed by the if
    const cf = switch (func.opcode(insts[idx + 1])) {
        .@"if" => |c| c,
        else => return false,
    };
    const result = func.instResult(insts[idx]) orelse return false;
    if (cf.cond != result) return false; // the if must test exactly this icmp's result
    // Single-use: the boolean is read only by this if's condition. Since the icmp immediately
    // precedes the if and equals cf.cond, a total use-count of exactly 1 means the if's cond is
    // the sole use, so skipping the boolean harms nothing.
    return countUses(func, result) == 1;
}

/// Whether the `if` at `insts[if_idx]` folds a flag-setting arithmetic op
/// into its branch. The arith at `if_idx-2` is a single-use `add`, `sub`,
/// or `bit_and` (register form), or `add`/`sub` (immediate `arith_imm`
/// form), whose result is compared eq/ne against a literal `0` by the icmp
/// at `if_idx-1`. That icmp is itself the single-use condition of this if,
/// meaning the compare-and-branch fold already applies (`fusesIntoNextIf`).
///
/// Unlike aarch64, a plain x86 `add`/`sub`/`and` already sets ZF, and the
/// rest of the flags, as a side effect of computing its result, lowering
/// through `binary`'s flag-setting `encode.add`/`sub`/`andr`, never a
/// flag-silent `lea`. So this predicate does not gate a skip-and-reemit at
/// the arith's own site the way aarch64's does. The arith runs through its
/// normal `.arith`/`.arith_imm` lowering completely unchanged, since
/// neither arm reads this predicate, still materializing its result and
/// leaving ZF set. It is `emitIf`'s fused setup alone that reads this
/// predicate, to emit no compare at all, neither `cmp` nor `test`, and
/// branch directly on the flags the arith already left behind. This is the
/// one eligibility predicate the fold uses, mirroring `fusesIntoNextIf` and
/// aarch64's `fusesArithIntoBranch`.
///
/// `enabled` carries `caps.fuse_arith_branch and caps.fuse_cmp_branch`,
/// since the fold lives inside the compare-and-branch path. A model
/// without either falls back to the plain arith followed by the cmp/test
/// and branch of the unfused `emitIf` path, byte-identical to before this
/// fold existed.
///
/// Scope, bounded for correctness: eq/ne only. ZF equals (result == 0) for
/// the eq/ne relation only. lt/le/gt/ge need SF/OF/CF reasoning tied to the
/// actual compare, which an arith's flags do not reproduce, so they stay on
/// the plain cmp path. The icmp RHS must be a literal `iconst 0`. The op
/// must be register `add`, `sub`, or `bit_and`, or `add`/`sub` in the
/// immediate (`arith_imm`) form. A `bit_and` immediate is excluded,
/// mirroring aarch64's bitmask-immediate exclusion, and is simply
/// unneeded, since the plain path already handles it. Integer and gpr
/// only, since float/vector arith routes through entirely different
/// lowering and never sets integer ZF this way. The arith result must be
/// single-use, so only the icmp reads it, and nothing else depends on its
/// materialization happening at any particular point relative to the
/// icmp.
fn fusesArithIntoBranch(func: *const Function, insts: []const ir.function.Inst, if_idx: usize, enabled: bool) bool {
    if (!enabled) return false;
    if (if_idx < 2) return false; // need the arith at if_idx-2 and the icmp at if_idx-1
    // The compare-and-branch fold must already apply: the icmp at if_idx-1 is a single-use,
    // integer/gpr icmp that is exactly this if's condition (see `fusesIntoNextIf`).
    if (!fusesIntoNextIf(func, insts, if_idx - 1)) return false;
    const cmp = func.opcode(insts[if_idx - 1]).icmp; // an icmp, per fusesIntoNextIf
    // eq/ne only: ZF equals (result == 0), exactly these two relations.
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
    // Integer and gpr only. A float/vector arith routes through a
    // different lowering entirely and never leaves ZF meaningfully set for
    // this fold.
    if (isVector(func, arith_result) or isFloat(func, arith_result)) return false;
    // Single-use: the arith result is read only by the icmp. Since the arith immediately precedes
    // the icmp and is its LHS, a total use-count of exactly 1 means the icmp is the sole reader.
    if (countUses(func, arith_result) != 1) return false;
    return switch (func.opcode(arith_inst)) {
        .arith => |a| a.op == .add or a.op == .sub or a.op == .bit_and,
        // Only add/sub in the immediate form. A bit_and immediate stays
        // on the plain path, mirroring aarch64's bitmask-immediate
        // exclusion. x86's `aluImm` could encode it, but narrowing the
        // fold's surface keeps this port a direct match to the reference.
        .arith_imm => |a| a.op == .add or a.op == .sub,
        else => false,
    };
}

/// Lay out the alloca region. Each `alloca` result gets a naturally
/// aligned byte offset, relative to the region base, recorded in `map`.
/// This function returns the region's size.
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
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(p) });
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
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });

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
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });
    const code = try selectFunction(allocator, &func);
    defer allocator.free(code);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]); // ends in ret
}

test "global_addr emits lea [rip+disp32] and a .pcrel_lea reloc at the disp32 field" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const b = try func.appendBlock();
    const g = try func.appendGlobalAddr(b, ptr_t, "g");
    const v = try func.appendInst(b, i8_t, .{ .load = .{ .ptr = g } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var compiled = try compile(allocator, &func);
    defer compiled.deinit(allocator);

    // Exactly one reloc, naming "g", kind `.pcrel_lea` (not the default `.call`).
    try std.testing.expectEqual(@as(usize, 1), compiled.relocs.len);
    const r = compiled.relocs[0];
    try std.testing.expectEqualStrings("g", r.symbol);
    try std.testing.expectEqual(Kind.pcrel_lea, r.kind);

    // The reloc offset lands on a real `lea` (REX.W 8D) whose modrm selects the
    // RIP-relative form (mod=00, rm=101), 3 bytes before the recorded disp32 field.
    try std.testing.expect(r.offset >= 3);
    const rex = compiled.code[r.offset - 3];
    const opcode = compiled.code[r.offset - 2];
    const modrm = compiled.code[r.offset - 1];
    try std.testing.expectEqual(@as(u8, 0x8D), opcode);
    try std.testing.expectEqual(@as(u8, 0x48), rex & 0x48); // REX.W set
    try std.testing.expectEqual(@as(u8, 0x05), modrm & 0xC7); // mod=00, rm=101 (RIP-relative)
    // The placeholder disp32 is zero (patched later by the linker).
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, compiled.code[r.offset..][0..4], .little));
}

test "a via_got global_addr emits mov [rip+disp32] and a .got_pcrel reloc (GOT-indirect data import)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.intern(.ptr);
    const i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } });
    const b = try func.appendBlock();
    const g = try func.appendGlobalAddrGot(b, ptr_t, "g");
    const v = try func.appendInst(b, i8_t, .{ .load = .{ .ptr = g } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var compiled = try compile(allocator, &func);
    defer compiled.deinit(allocator);

    // Exactly one reloc, naming "g", kind `.got_pcrel` (the GOT-indirect data import).
    try std.testing.expectEqual(@as(usize, 1), compiled.relocs.len);
    const r = compiled.relocs[0];
    try std.testing.expectEqualStrings("g", r.symbol);
    try std.testing.expectEqual(Kind.got_pcrel, r.kind);

    // The reloc offset lands on a real `mov r64, [rip+disp32]` (REX.W 8B)
    // whose modrm selects the RIP-relative form (mod=00, rm=101), 3 bytes
    // before the recorded disp32 field. The opcode is `0x8B` (mov, which
    // loads from the GOT slot), not `0x8D` (lea, which would compute the
    // address of the GOT slot itself).
    try std.testing.expect(r.offset >= 3);
    const rex = compiled.code[r.offset - 3];
    const opcode = compiled.code[r.offset - 2];
    const modrm = compiled.code[r.offset - 1];
    try std.testing.expectEqual(@as(u8, 0x8B), opcode);
    try std.testing.expectEqual(@as(u8, 0x48), rex & 0x48); // REX.W set
    try std.testing.expectEqual(@as(u8, 0x05), modrm & 0xC7); // mod=00, rm=101 (RIP-relative)
    // The placeholder disp32 is zero (patched later by the linker to the GOT slot).
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, compiled.code[r.offset..][0..4], .little));
}

test "a call's reloc still defaults to .call (existing call sites unaffected by the Reloc.kind field)" {
    const allocator = std.testing.allocator;
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try callee.appendBlock();
        const x = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });
    }
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "callee", &.{x});
        caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    }
    var compiled = try compile(allocator, &caller);
    defer compiled.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), compiled.relocs.len);
    try std.testing.expectEqual(Kind.call, compiled.relocs[0].kind);
    try std.testing.expectEqualStrings("callee", compiled.relocs[0].symbol);
}

test "a call keeps RSP 16-aligned at the call site (movaps-safe host calls)" {
    // A callee that reads its stack with movaps faults on real hardware
    // when the caller's RSP is misaligned. qemu-user does not enforce this,
    // so only a codegen check catches it. Entry RSP is 8 (mod 16), so the
    // prologue frame must be 8 (mod 16) to land calls on a 16 boundary.
    const allocator = std.testing.allocator;
    var callee = Function.init(allocator);
    defer callee.deinit();
    {
        const t = try callee.types.intern(.{ .float = .f32 });
        const b = try callee.appendBlock();
        const a = try callee.appendBlockParam(b, t);
        callee.setTerminator(b, .{ .ret = ir.function.Ret.one(a) });
    }
    var caller = Function.init(allocator);
    defer caller.deinit();
    {
        const t = try caller.types.intern(.{ .float = .f32 });
        const b = try caller.appendBlock();
        const x = try caller.appendBlockParam(b, t);
        const r = try caller.appendCall(b, t, "helper", &.{x});
        caller.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
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

// ===========================================================================
// Wimmer bridge gap #7: a live-range split whose two adjacent segments are
// both spill slots. `wimmerGprTransition`/`wimmerXmmTransition` now build a
// `.slot_to_slot` action instead of bailing with `error.Unsupported`, and
// `emitSplitActionX86` expands it into a reload-then-store pair through the
// class scratch (gpr `move_scratch`/r11, xmm `xmm_scratch`/xmm15).
//
// A natural Wimmer allocation can never reach this shape on x86, for the
// same structural reason aarch64's isel documents at its own
// `.slot_to_slot` tests. `x86_64UseKind` makes every operand use
// `must_have_register` (see its doc comment above), and `spillCurrent`'s
// only two call sites in the shared `wimmer.zig` (backend-independent
// code, not aarch64- or x86-specific) either split a value at its own next
// must-have use, which its next pop is then proven to resolve into a
// register rather than a second self-spill, or fire only when
// `current.start()` is not itself a use. So a placed value alternates
// register, slot, register. Two adjacent slot segments never arise.
// This was confirmed empirically against several x86 differential
// candidates (see wimmer_diff.zig's own same-position cluster shape):
// none ever produced a `walloc.actions` entry with both `src` and `dst` as
// `.slot`. So these tests exercise the exact mechanism
// (`emitSplitActionX86`'s `.slot_to_slot` arm) directly with a hand-built
// `SplitAction`, the same way aarch64's isel tests do, and the same way
// this file's other low-level tests check exact encoded bytes rather than
// routing through a full differential compile.
// ===========================================================================

test "emitSplitActionX86 .slot_to_slot expands to reload+store through the gpr scratch" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var ctx = Ctx{ .func = &func };
    defer ctx.code.deinit(allocator);
    const act = SplitAction{ .at = 0, .kind = .slot_to_slot, .value = v, .slot = 5, .move_from_slot = 2 };
    try emitSplitActionX86(allocator, &ctx, act);

    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(allocator);
    try expected.appendSlice(allocator, encode.movFromStack(move_scratch, slotDisp(2)).slice());
    try expected.appendSlice(allocator, encode.movToStack(slotDisp(5), move_scratch).slice());
    try std.testing.expectEqualSlices(u8, expected.items, ctx.code.items);
}

test "emitSplitActionX86 .slot_to_slot expands to reload+store through the xmm scratch for a scalar float" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f64 });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var ctx = Ctx{ .func = &func };
    defer ctx.code.deinit(allocator);
    // A non-zero `xmm_base` (as a real frame would have) proves the offset math threads through.
    ctx.xmm_base = 64;
    const act = SplitAction{ .at = 0, .kind = .slot_to_slot, .value = v, .is_xmm = true, .slot = 1, .move_from_slot = 3 };
    try emitSplitActionX86(allocator, &ctx, act);

    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(allocator);
    try expected.appendSlice(allocator, encode.movupsLoad(xmm_scratch, ctx.xmmDisp(3)).slice());
    try expected.appendSlice(allocator, encode.movupsStore(ctx.xmmDisp(1), xmm_scratch).slice());
    try std.testing.expectEqualSlices(u8, expected.items, ctx.code.items);
}

test "emitSplitActionX86 .slot_to_slot round-trips a full 128-bit vector through the xmm scratch" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v4 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = f32_t } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, v4);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var ctx = Ctx{ .func = &func };
    defer ctx.code.deinit(allocator);
    const act = SplitAction{ .at = 0, .kind = .slot_to_slot, .value = v, .is_xmm = true, .slot = 7, .move_from_slot = 0 };
    try emitSplitActionX86(allocator, &ctx, act);

    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(allocator);
    try expected.appendSlice(allocator, encode.movupsLoad(xmm_scratch, ctx.xmmDisp(0)).slice());
    try expected.appendSlice(allocator, encode.movupsStore(ctx.xmmDisp(7), xmm_scratch).slice());
    try std.testing.expectEqualSlices(u8, expected.items, ctx.code.items);
}

test "emitSplitActionX86 .slot_to_slot round-trips a 256-bit ymm through the xmm scratch with vmovups" {
    // x86, unlike aarch64's 128-bit-only NEON, has a genuine wide-vector
    // `slot_to_slot` case. An 8-lane f32 vector is a 256-bit ymm, so both
    // halves must use vmovups (32 bytes) rather than movups (16 bytes), or
    // the upper 4 lanes would be dropped.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const v8 = try func.types.intern(.{ .vector = .{ .len = 8, .elem = f32_t } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, v8);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(v) });

    var ctx = Ctx{ .func = &func };
    defer ctx.code.deinit(allocator);
    const act = SplitAction{ .at = 0, .kind = .slot_to_slot, .value = v, .is_xmm = true, .slot = 2, .move_from_slot = 4 };
    try emitSplitActionX86(allocator, &ctx, act);

    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(allocator);
    try expected.appendSlice(allocator, encode.vmovupsLoad(xmm_scratch, ctx.xmmDisp(4)).slice());
    try expected.appendSlice(allocator, encode.vmovupsStore(ctx.xmmDisp(2), xmm_scratch).slice());
    try std.testing.expectEqualSlices(u8, expected.items, ctx.code.items);
}
