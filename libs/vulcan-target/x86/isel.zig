//! x86 (32-bit, cdecl) instruction selection. Integer arguments pass on the stack
//! (`[esp+4]`, `[esp+8]`, ...) and the result returns in EAX. This covers multiple blocks
//! (high-IR if/jump with edge moves), comparison, division, shifts, immediate-operand
//! arithmetic, and register spilling.
//!
//! The prologue loads each argument into its assigned register. The body mirrors the
//! x86-64 selector. EBX and ESI are the spill reload scratch registers. EDI is the
//! parallel-move scratch register. EBX is low-byte-addressable, so a spilled boolean
//! result can use setcc (32-bit setcc only targets EAX, ECX, EDX, or EBX). EAX and EDX
//! are reserved when dividing. ECX is reserved when shifting. Entry parameters are not
//! spilled (more parameters than pool registers is unsupported). Spill slots are 4 bytes
//! wide and sit below the arguments, at `[esp + slot*4]`.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");
const wimmer = @import("../wimmer.zig");
const addrfold = @import("../addrfold.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Reg = encode.Reg;

/// A shared no-fold analysis. Its `baseOf`, `offOf`, and `isDeadAdd` behave as if nothing folded, so
/// any path that never overrides `Ctx.fold` stays byte-identical to the code from before address
/// folding existed. It holds no allocation (see `addrfold.Analysis.empty`), so it never needs `deinit`.
const empty_fold: addrfold.Analysis = addrfold.Analysis.empty;

/// The x86-32 fold predicate for `addrfold.analyze`. It folds a load or store whose pointer is an
/// `arith_imm.add(base, imm)` into a `[base + disp32]` addressing mode, for any access size (an x86
/// memory operand carries a 32-bit signed displacement regardless of width). The fold applies exactly
/// when the add's imm fits a signed 32-bit displacement. The isel already assumes an `arith_imm` imm
/// fits an i32 (`@intCast(a.imm)` when it emits the add), so this matches an existing invariant. It
/// returns the byte offset (equal to the add's imm) when in range, else null. `analyze` calls this
/// only after confirming the pointer is an `arith_imm.add`, so the unwraps below are guaranteed, but
/// still asserted. The fold analysis covers all loads and stores, including float and vector ones, but
/// x86-32 rejects those at emit time anyway (`error.Unsupported`), so folding their address is
/// harmless: the whole compile fails regardless.
fn x86FoldOffset(_: void, func: *const Function, mem_inst: ir.function.Inst) ?i64 {
    const ptr = switch (func.opcode(mem_inst)) {
        .load => |l| l.ptr,
        .store => |st| st.ptr,
        else => unreachable, // analyze hands foldOffset only a load or store
    };
    const def = func.definingInst(ptr).?; // analyze confirmed that ptr is defined by an arith_imm.add
    const add = switch (func.opcode(def)) {
        .arith_imm => |a| a,
        else => unreachable,
    };
    std.debug.assert(add.op == .add);
    if (std.math.cast(i32, add.imm) == null) return null;
    return add.imm;
}

pub const Error = std.mem.Allocator.Error || error{Unsupported};

const ret_reg: Reg = .eax;
const scratch1: Reg = .ebx; // one of the low 4 registers: holds a left operand or a spilled destination (setcc-able)
const scratch2: Reg = .edi; // right-operand reload scratch (same as the move scratch, never overlaps)
const move_scratch: Reg = .edi;

/// Whether `r` has a low-byte (al/cl/dl/bl) form, that is, is one of eax/ecx/edx/ebx (indices 0..3).
/// 32-bit mode has no REX prefix, so esi, edi, esp, and ebp have no 8-bit form: encoding index 6 (esi)
/// as an 8-bit register names DH, a different register. A `setcc` or other 8-bit op must therefore
/// target a byte-addressable register. A boolean value homed in esi is staged through the reserved
/// byte scratch register ebx (see the icmp arm in `lowerInst`).
fn isByteReg(r: Reg) bool {
    return @intFromEnum(r) < 4;
}

const Loc = union(enum) { reg: Reg, spill: u32 };

/// One piece of a live-range-split value's life, on the shared Wimmer path. The value lives in `loc`
/// from position `from` until the next segment. `segments[0].from` is the value's definition position,
/// so a lookup at any position at or after the definition resolves to some segment. A whole-life value
/// (one segment) has no `Ctx.segments` entry, so `loc` falls back to the whole-life `loc_of` map that
/// `translateAllocationX86` fills for it.
const Segment = struct { from: u32, loc: Loc };

/// An intra-block re-home that the emitter drains at position `at`, on the shared Wimmer path. The
/// shared allocator already ordered these, per same-position cluster, into a hazard-free parallel
/// move, so draining them in order (see `emitFromAllocation`) never overwrites a live value. A
/// `slot_to_slot` action re-homes a spilled value from `move_from_slot` to `slot` without ever holding
/// it in a value register: `emitSplitActionX86` expands it into a reload, then a store, through the
/// gpr class scratch register (edi), so `reg` and `move_from` stay at their defaults for that kind.
const SplitAction = struct {
    at: u32,
    kind: enum { store, reload, move, slot_to_slot },
    value: Value,
    slot: u32 = 0,
    reg: Reg = .eax,
    move_from: Reg = .eax, // `.move` source (a register-to-register re-home)
    move_from_slot: u32 = 0, // `.slot_to_slot` source slot (`slot` is the destination)
};

/// One ordered control-flow-edge move, on the shared Wimmer path: a class-relative register index or a
/// spill slot. The shared allocator already ordered each edge into a valid parallel-move sequence
/// (it reads sources before it overwrites them, and it breaks cycles through the class scratch
/// register), so the emitter replays each move as a primitive op.
const EdgeLoc = union(enum) { reg: u16, slot: u32 };
const EdgeMove = struct { src: EdgeLoc, dst: EdgeLoc };
const EdgeMoveSet = struct { pred: Block, succ: Block, moves: []EdgeMove };

const Fixup = struct { at: usize, target: u32 };

/// Which relocation shape a `Reloc` records. `.call` is a `call rel32` displacement field
/// (addend -4 from the E8 opcode, resolved inside the module by the linker). `.abs32` is a
/// `global_addr`'s `mov rd, imm32` immediate field, patched to the symbol's absolute
/// runtime address once known (i386 has no rip-relative addressing mode, see
/// `link.zig`'s `applyGlobalReloc`). `.got_abs` is a GOT-indirect `global_addr`'s
/// (`via_got`) `mov rd, [abs32]` field, emitting `R_386_GOT32`: the abs32 field addresses
/// the symbol's `.got` slot (which a real `ld.so` fills through `R_386_GLOB_DAT`), so the
/// `mov` loads the symbol's address rather than embedding it as an immediate. This is
/// used for a data import of a shared object's global.
pub const Kind = enum { call, abs32, got_abs };

/// A relocation of `kind`, targeting `symbol`, at byte `offset` within the compiled
/// function's code. For `.call` that offset is the rel32 displacement field (PC-relative,
/// resolved at link time inside the module). For `.abs32` it is the `mov`'s imm32 field
/// (absolute, patched once the symbol's runtime address is known). For `.got_abs` it is
/// the `mov`'s abs32 field, patched to the GOT slot address the dynamic linker
/// assigns the symbol.
pub const Reloc = struct { offset: usize, symbol: []const u8, kind: Kind = .call };

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
    alloca_base: i32 = 0, // esp offset of the alloca region (sits above the spill slots)
    alloca_off: std.AutoHashMapUnmanaged(Value, u32) = .{}, // each alloca result's byte offset in that region
    // Address-mode-fold analysis, read by the load/store emit arms and the dead-add skip.
    // Defaults to the empty analysis (nothing folds). `compile` overrides it with a real one.
    fold: *const addrfold.Analysis = &empty_fold,
    // Filled by `translateAllocationX86` from the shared allocation (these Ctx defaults are the
    // pre-fill empty state). `segments` holds each live-range-split value's list, ascending by `from`.
    // `actions` holds the ordered intra-block re-homes to drain. `def_pos` holds each value's
    // definition position in the shared numbering (the emission assert reads it). `pos` is the current
    // emission position, threaded per instruction so `loc` picks the active segment. `edge_moves` holds
    // the precomputed, already-ordered control-flow-edge moves that `emitMoves` replays when
    // `edge_move_driven` is set.
    segments: std.AutoHashMapUnmanaged(Value, []Segment) = .{},
    actions: std.ArrayList(SplitAction) = .empty,
    def_pos: []u32 = &.{},
    pos: u32 = 0,
    edge_moves: []EdgeMoveSet = &.{},
    edge_move_driven: bool = false,
    // Threaded by `emitFromAllocation`'s prologue so `va_start` can find the first incoming
    // stack argument (only meaningful when `func.is_variadic`; both default to 0, and stay 0
    // for a non-variadic function, where no `va_*` op is ever emitted). `va_frame` is the `sub
    // esp, frame` amount. `va_saved` is the count of callee-saved gprs the prologue pushed (each
    // push shifts the incoming stack args one 4-byte slot higher), mirroring the same formula
    // the entry-parameter-load loop below already uses.
    va_frame: i32 = 0,
    va_saved: u32 = 0,

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

/// Compile `func` to machine code plus its call relocations. The caller owns the result.
///
/// The production register allocator is the shared Wimmer-Franz linear-scan-on-SSA allocator.
/// There is no fallback to a native linear scan (the retired `assignRegs`). The pipeline mirrors
/// `compileFunctionWimmerX86Fold` (splitCriticalEdges, then addrfold.analyze, then
/// applyFoldRewriteX86, then x86_32RegDescription, then wimmer.allocate, then
/// translateAllocationX86, then emitFromAllocation), but runs on an independently owned deep
/// `clone`, so the public `*const Function` entry points never touch a caller's function (a caller
/// may reuse or share its function across backends). The real fold analysis is threaded into emission.
pub fn compile(allocator: std.mem.Allocator, func: *const Function) Error!Compiled {
    // This backend does not yet lower f16; reject it cleanly rather than
    // silently treat it as f64.
    if (ir.function.functionUsesF16(func)) return error.Unsupported;
    if (ir.function.functionUsesF128(func)) return error.Unsupported;
    if (func.blockCount() == 0) return error.Unsupported;

    // The Wimmer pipeline mutates the function (splitCriticalEdges appends forwarding blocks,
    // applyFoldRewriteX86 repoints folded pointers and removes dead adds), but the public entry points
    // take `*const Function`, so work on an independently owned deep `clone` and leave the caller's
    // function byte-for-byte pristine. This keeps every public `*const` signature unchanged.
    var work = try func.clone(allocator);
    defer work.deinit();

    // Split critical edges first (the shared resolver needs a block on every critical edge to place
    // its shuffle moves), before any numbering is built.
    try ir.critical_edge.splitCriticalEdges(allocator, &work);

    // Neutralize every block unreachable from the entry before any numbering is built. A pass that
    // raises a loop nest into one op could orphan its blocks (the IR has no block-delete), leaving
    // instructions that use values defined in reachable blocks. The shared `buildIntervals` walks
    // every block, so such a value would get a live range built entirely from the dead region and trip
    // the allocator's SSA definition-in-range assert. Emptying every unreachable block's params,
    // instructions, and terminator removes those spurious uses. This backend emits every block, so it
    // does not consume the returned reachable set: `emitFromAllocation` detects a neutralized block by
    // its emptied shape (no instructions and a null terminator) and emits nothing for it. This is a
    // no-op for a function whose blocks are all reachable, so output stays byte-identical for every
    // existing caller.
    const reachable = try ir.reachable.neutralizeUnreachable(allocator, &work);
    allocator.free(reachable);

    // Address-mode folding is a pre-allocation IR rewrite, so it is sound under the fold-blind
    // shared Wimmer allocator (which reads only the actual IR operands). `analyze` recognizes each
    // foldable `p = arith_imm.add(base, imm); load/store(p)`. `applyFoldRewriteX86` then repoints
    // each folded memory op's `ptr` to `base` and drops the dead adds in the clone, so the allocator
    // keeps `base` live to the load or store. The same analysis threads into emission through
    // `ctx.fold`: `folds` is keyed by the surviving memory inst, so `baseOf`/`offOf` stay consistent
    // with the IR.
    var fold = try addrfold.analyze(allocator, &work, {}, x86FoldOffset);
    defer fold.deinit(allocator);
    applyFoldRewriteX86(&work, &fold);

    var desc = try x86_32RegDescription(allocator, &work);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, &work, &desc);
    defer walloc.deinit(allocator);

    var ctx = Ctx{ .func = &work, .fold = &fold };
    defer ctx.loc_of.deinit(allocator);
    defer ctx.code.deinit(allocator);
    defer ctx.fixups.deinit(allocator);
    defer ctx.relocs.deinit(allocator);
    defer ctx.alloca_off.deinit(allocator);
    defer {
        var seg_it = ctx.segments.valueIterator();
        while (seg_it.next()) |s| allocator.free(s.*);
        ctx.segments.deinit(allocator);
    }
    defer ctx.actions.deinit(allocator);
    // `def_pos` is a heap-owned copy (translateAllocationX86 always allocates it). The `&.{}` default
    // is a zero-length slice whose free is a no-op, so an unconditional free is safe on any early exit.
    defer allocator.free(ctx.def_pos);
    defer {
        for (ctx.edge_moves) |es| allocator.free(es.moves);
        allocator.free(ctx.edge_moves);
    }

    var saved: std.ArrayList(Reg) = .empty;
    defer saved.deinit(allocator);
    var num_slots: u32 = 0;
    try translateAllocationX86(allocator, &work, &walloc, &ctx, &num_slots, &saved);
    const frame = try frameLayoutW(allocator, &ctx, &work, num_slots);
    var compiled = try emitFromAllocation(allocator, &ctx, &work, frame, saved.items);
    errdefer compiled.deinit(allocator);

    // Every `Reloc.symbol` is a borrowed slice into the emitting function's symbol storage, and the
    // contract is that those names outlive `Compiled` (the caller's function does). Emission borrowed
    // them from the clone, whose storage `work.deinit` frees on return, so re-point each name to the
    // original `func`'s identical, longer-lived symbol string (the clone re-interned symbols 1:1, so
    // the same name exists there). This keeps the borrowed-name contract intact.
    for (compiled.relocs) |*r| r.symbol = rebindSymbolName(func, r.symbol);
    return compiled;
}

/// The `func`-owned symbol string equal to `name`. Every emitted relocation names a callee the
/// function interned (a `call`'s `symbol` indexes `func`'s symbol table), so a match always exists.
/// A miss would be a codegen bug, not a runtime condition.
fn rebindSymbolName(func: *const Function, name: []const u8) []const u8 {
    var i: u32 = 0;
    while (i < func.symbolCount()) : (i += 1) {
        const s = func.symbolName(i);
        if (std.mem.eql(u8, s, name)) return s;
    }
    unreachable;
}

fn lowerInst(allocator: std.mem.Allocator, ctx: *Ctx, inst: ir.function.Inst) Error!void {
    const func = ctx.func;
    // A folded address-add is dead: every use of its result was rerouted to the base by the fold,
    // so the add itself must not be emitted (its result is never read). This is a no-op when
    // nothing folded.
    if (ctx.fold.isDeadAdd(inst)) return;
    if (func.opcode(inst) == .store) {
        // `store` produces no result, so handle it before the result-unwrap below.
        const st = func.opcode(inst).store;
        if (func.types.type_kind(func.valueType(st.value)) != .int) return error.Unsupported; // x86-32 is integer-only
        const bits = intBits(func, st.value);
        if (bits > 32) return error.Unsupported; // no multi-register wide-int support here
        // A folded store addresses `[base + disp32]`: `baseOf` yields the fold base (the add's
        // left operand) and `offOf` the displacement. Both are the raw ptr and 0 when unfolded, so
        // the non-folding case is byte-identical.
        const base = try ctx.use(allocator, ctx.fold.baseOf(func, inst), scratch2);
        const disp: i32 = @intCast(ctx.fold.offOf(inst));
        if (bits <= 16) {
            // movToMem8/movToMem16 need a byte-addressable source (al/cl/dl/bl). edi/esi have
            // no 8-bit form, so unconditionally stage the value through scratch1 (ebx, which
            // does have one) rather than special-casing which registers happen to qualify.
            var val = try ctx.use(allocator, st.value, scratch1);
            if (val != scratch1) {
                try ctx.put(allocator, encode.movReg(scratch1, val));
                val = scratch1;
            }
            std.debug.assert(base != scratch1); // ptr (scratch2) and the staged value never collide
            try ctx.put(allocator, if (bits <= 8) encode.movToMem8(base, disp, val) else encode.movToMem16(base, disp, val));
        } else {
            const val = try ctx.use(allocator, st.value, scratch1);
            try ctx.put(allocator, encode.movToMem32(base, disp, val));
        }
        return;
    }
    if (func.opcode(inst) == .va_start) {
        // Initialize the `va_list` object. It has no result, so (like `store` above) this is
        // handled before the result unwrap below.
        return emitVaStart(allocator, ctx, func, func.opcode(inst).va_start.list);
    }
    if (func.opcode(inst) == .va_end) {
        // cdecl's `va_list` (a bare pointer walking the caller's stack) owns no
        // resource, so `va_end` is a no-op.
        return;
    }
    const result = func.instResult(inst).?;
    switch (func.opcode(inst)) {
        .iconst => |c| {
            const rd = ctx.dst(result, scratch1);
            try ctx.put(allocator, encode.movImm(rd, @intCast(c)));
            try ctx.store(allocator, result, rd);
        },
        .fconst => {
            // x86-32 has no float registers, so a float constant cannot be held in a register. The
            // only supported use is a `double` argument pushed to a variadic call, which reads the
            // constant's bits straight from this `fconst` at the call site (see the `.call` arm). The
            // definition therefore emits nothing. A float constant reaching any other (integer) use
            // is rejected by that use's own integer-only guard, so a no-op here causes no miscompile.
        },
        .arith => |a| {
            const signed = isSigned(func, a.lhs);
            switch (a.op) {
                .div, .rem => {
                    // idiv/div destroys EDX (sign extension or remainder) and then reads the
                    // divisor, so a divisor allocated to EAX or EDX must be copied out before the
                    // mov-eax/cdq-xor below overwrites it. The shared allocator models div as an
                    // eax+edx clobber at this position, so a live divisor is normally kept clear of
                    // them, but the divisor itself may still be homed in eax or edx (it is consumed
                    // here, not live-through), so the guard is needed. `ctx.loc` reads the location
                    // without emitting, so the reload order below is unchanged when it is false. This
                    // mirrors x86_64's `rhs_in_clobber` guard.
                    const rhs_in_clobber = switch (ctx.loc(a.rhs)) {
                        .reg => |r| r == .eax or r == .edx,
                        else => false,
                    };
                    var divisor: ?Reg = null;
                    if (rhs_in_clobber) {
                        const rr = try ctx.use(allocator, a.rhs, scratch2);
                        try ctx.put(allocator, encode.movReg(scratch2, rr));
                        divisor = scratch2;
                    }
                    try ctx.put(allocator, encode.movReg(.eax, try ctx.use(allocator, a.lhs, scratch1)));
                    try ctx.put(allocator, if (signed) encode.cdq() else encode.xorr(.edx, .edx));
                    const rr = divisor orelse try ctx.use(allocator, a.rhs, scratch2);
                    try ctx.put(allocator, if (signed) encode.idiv(rr) else encode.divu(rr));
                    const rd = ctx.dst(result, scratch1);
                    const res: Reg = if (a.op == .div) .eax else .edx;
                    if (rd != res) try ctx.put(allocator, encode.movReg(rd, res));
                    try ctx.store(allocator, result, rd);
                },
                .shl, .shr => {
                    // The shift count goes in ECX, so `lhs` allocated to ECX must be copied out
                    // before ECX is overwritten with the count below. The shared allocator models a
                    // variable shift as an ecx clobber at this position, but `lhs` is consumed here
                    // (not live-through), so it may still be homed in ecx and the guard is needed.
                    // This mirrors x86_64's `lhs_in_rcx` guard.
                    const lhs_in_ecx = switch (ctx.loc(a.lhs)) {
                        .reg => |r| r == .ecx,
                        else => false,
                    };
                    var rl = try ctx.use(allocator, a.lhs, scratch1);
                    if (lhs_in_ecx) {
                        try ctx.put(allocator, encode.movReg(scratch1, rl));
                        rl = scratch1;
                    }
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
                .mulh => return error.Unsupported, // no immediate form; expanded before isel
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
            try ctx.put(allocator, encode.cmp(rl, rr));
            // `setcc` and `movzx` need a byte-addressable (ABCD) register. The result's home may be
            // esi (which has no low byte, so `setcc esi` would wrongly encode DH) or a spill slot, so
            // materialize the boolean in the reserved byte scratch ebx (scratch1) and move it to the
            // home. This is the same staging the 8-bit store uses. When the home is already
            // byte-addressable (eax, ecx, edx, or ebx itself when spilled), `staged == home`, so
            // nothing extra is emitted and the output stays byte-identical to the direct form. The
            // operands `rl` and `rr` are dead after `cmp`, so reusing ebx here never overwrites a live
            // value.
            const home = ctx.dst(result, scratch1);
            const staged: Reg = if (isByteReg(home)) home else scratch1;
            try ctx.put(allocator, encode.setcc(staged, condOf(cmp.op, isSigned(func, cmp.lhs))));
            try ctx.put(allocator, encode.movzxByte(staged, staged));
            if (staged != home) try ctx.put(allocator, encode.movReg(home, staged));
            try ctx.store(allocator, result, home);
        },
        .call => |c| {
            // cdecl: push arguments right to left, `call` (relocated), clean the stack, and
            // return the result in EAX. Each spilled argument's slot offset accounts for the
            // pushes already done (ESP has moved). The call clobbers the caller-saved registers
            // {eax,ecx,edx}. The shared allocator keeps a cross-call value in callee-saved esi
            // (or spills it) through the per-call clobber site, so it survives the call.
            const args = func.valueList(c.args);
            var pushed: usize = 0; // 32-bit words already pushed (corrects a spilled arg's ESP offset)
            var cleanup: usize = 0; // total bytes to reclaim after the call (4 per int, 8 per double)
            var j = args.len;
            while (j > 0) {
                j -= 1;
                const a = args[j];
                if (func.types.type_kind(func.valueType(a)) == .float) {
                    // A `double` (variadic float args are default-promoted to double) is pushed as
                    // 8 bytes: two 32-bit pushes, the high word first, so the low word ends at the
                    // lower address (a little-endian 8-byte value). x86-32 has no float registers, so
                    // only a compile-time float constant is supported here. Its bits come straight
                    // from the defining `fconst`. This is byte-identical for integer-only calls (no
                    // float argument).
                    const bits = floatConstBits(func, a) orelse return error.Unsupported;
                    try ctx.put(allocator, encode.pushImm(@bitCast(@as(u32, @truncate(bits >> 32)))));
                    try ctx.put(allocator, encode.pushImm(@bitCast(@as(u32, @truncate(bits)))));
                    pushed += 2;
                    cleanup += 8;
                } else {
                    switch (ctx.loc(a)) {
                        .reg => |r| try ctx.put(allocator, encode.pushReg(r)),
                        .spill => |slot| {
                            try ctx.put(allocator, encode.stackLoad(scratch1, slotDisp(slot) + @as(i32, @intCast(pushed)) * 4));
                            try ctx.put(allocator, encode.pushReg(scratch1));
                        },
                    }
                    pushed += 1;
                    cleanup += 4;
                }
            }
            try ctx.put(allocator, encode.callRel(0));
            try ctx.relocs.append(allocator, .{ .offset = ctx.code.items.len - 4, .symbol = func.symbolName(c.symbol) });
            if (cleanup > 0) try ctx.put(allocator, encode.aluImm(0, .esp, @intCast(cleanup))); // add esp, cleanup
            const rd = ctx.dst(result, scratch1);
            if (rd != .eax) try ctx.put(allocator, encode.movReg(rd, .eax));
            try ctx.store(allocator, result, rd);
        },
        .call_indirect => |c| {
            // Indirect call through a function pointer: identical cdecl argument-push setup
            // to `.call` above, except the target is a runtime value (`c.target`), not a
            // relocatable symbol, and the branch is `call edi` (through a register) instead of
            // `call rel32`.
            //
            // Stage the target into edi (`scratch2`/`move_scratch`). The push loop below
            // reuses ebx (`scratch1`) to reload a spilled argument right before pushing it, so
            // the call target must live somewhere that loop never touches. edi, reserved out
            // of the allocatable pool exactly like ebx, fits (this mirrors x86-64's r10
            // staging, which has no such reload-scratch collision, since its `.call` moves args
            // through registers, not the stack).
            const tgt = try ctx.use(allocator, c.target, scratch2);
            if (tgt != scratch2) try ctx.put(allocator, encode.movReg(scratch2, tgt));
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
            try ctx.put(allocator, encode.callReg(scratch2)); // call edi
            if (args.len > 0) try ctx.put(allocator, encode.aluImm(0, .esp, @intCast(args.len * 4))); // add esp, n*4
            const rd = ctx.dst(result, scratch1);
            if (rd != .eax) try ctx.put(allocator, encode.movReg(rd, .eax));
            try ctx.store(allocator, result, rd);
        },
        .alloca => {
            // The result is the address of its reserved stack slot: lea it from esp. `compile`
            // assigned slot offsets through `computeAllocaSlots` before any code was emitted,
            // including the `sub esp, frame` that reserves the region.
            const off = ctx.alloca_base + @as(i32, @intCast(ctx.alloca_off.get(result).?));
            const rd = ctx.dst(result, scratch1);
            try ctx.put(allocator, encode.leaFromStack(rd, off));
            try ctx.store(allocator, result, rd);
        },
        .load => {
            // x86-32 is integer-only: reject a float, vector, bool, or pointer load result
            // cleanly rather than misread its bytes.
            if (func.types.type_kind(func.valueType(result)) != .int) return error.Unsupported;
            const bits = intBits(func, result);
            if (bits > 32) return error.Unsupported; // no multi-register wide-int support here
            // A folded load addresses `[base + disp32]`: `baseOf` yields the fold base (the add's
            // left operand) and `offOf` the displacement. Both are the raw ptr and 0 when
            // unfolded, so the non-folding case is byte-identical.
            const base = try ctx.use(allocator, ctx.fold.baseOf(func, inst), scratch2);
            const disp: i32 = @intCast(ctx.fold.offOf(inst));
            const rd = ctx.dst(result, scratch1);
            // Load exactly the value's own width, so no bytes beyond the object are read (a
            // wider load would pull garbage from the next array element into the register). A
            // narrow load sign-extends a signed value and zero-extends an unsigned one into the
            // full 32-bit destination, matching x86-64's isel.
            const signed = isSigned(func, result);
            try ctx.put(allocator, switch (bits) {
                0...8 => if (signed) encode.movsxByteFromMem(rd, base, disp) else encode.movzxByteFromMem(rd, base, disp),
                9...16 => if (signed) encode.movsxWordFromMem(rd, base, disp) else encode.movzxWordFromMem(rd, base, disp),
                else => encode.movFromMem32(rd, base, disp),
            });
            try ctx.store(allocator, result, rd);
        },
        .global_addr => |ga| {
            const rd = ctx.dst(result, scratch1);
            if (ga.via_got) {
                // GOT-indirect symbol address (a data import): `mov rd, [abs32]`, which loads
                // the symbol's address from its `.got` slot (the real `ld.so` fills that slot
                // through `R_386_GLOB_DAT`). abs32 is a 0 placeholder here. The `.got_abs` reloc
                // (emitting `R_386_GOT32`) records the abs32 field's byte offset, patched to the
                // GOT slot's absolute address once placed (i386 has no rip-relative form, unlike
                // x86-64's `mov rd, [rip+disp32]`). The result register then holds the same
                // runtime address the direct `mov rd, imm32` path would, so a downstream
                // `load` through it is identical.
                try ctx.put(allocator, encode.movFromAbs32(rd, 0));
                try ctx.relocs.append(allocator, .{ .offset = ctx.code.items.len - 4, .symbol = func.symbolName(ga.symbol), .kind = .got_abs });
            } else {
                // Absolute symbol address: `mov rd, imm32`, with imm32 = 0 here (a reloc records
                // the imm32 field's byte offset, patched to the symbol's runtime address once
                // known, see link.zig's `applyGlobalReloc`). i386 has no rip-relative
                // addressing mode, unlike x86-64's `lea rd, [rip+disp32]`.
                try ctx.put(allocator, encode.movImm(rd, 0));
                try ctx.relocs.append(allocator, .{ .offset = ctx.code.items.len - 4, .symbol = func.symbolName(ga.symbol), .kind = .abs32 });
            }
            try ctx.store(allocator, result, rd);
        },
        // Fetch the next variadic argument from the `va_list`. `va_start` and `va_end`
        // are handled earlier in this function (before the result unwrap above), since they
        // have no result.
        .va_arg => |va| try emitVaArg(allocator, ctx, func, va.list, result),
        .convert => |cv| {
            // x86-32 is integer-only in registers (the only float value this backend ever
            // touches is a compile-time `fconst` staged straight into a variadic call push,
            // see its comment above), so a convert touching a float operand or result never
            // reaches here: reject it cleanly rather than misread its bits.
            if (func.types.type_kind(func.valueType(cv.value)) != .int) return error.Unsupported;
            if (func.types.type_kind(func.valueType(result)) != .int) return error.Unsupported;
            const src_bits = intBits(func, cv.value);
            const dst_bits = intBits(func, result);
            // Neither width fits a single x86-32 register beyond 32 bits: this backend has no
            // multi-register wide-int support, so it fails closed rather than mislower it.
            if (src_bits > 32 or dst_bits > 32) return error.Unsupported;
            if (dst_bits > src_bits) {
                // Widening must extend by the source's own signedness (sign-extend a signed
                // source, zero-extend an unsigned one). A 32-bit x86 register holds a narrower
                // value zero-extended in its upper bits regardless of that value's own sign
                // (this is architectural), so a bare move would carry that zero-extended pattern
                // into a wider destination unchanged, silently discarding a negative sign. This
                // mirrors x86_64's convert arm (aarch64 sbfm/ubfm, riscv64 slli+srai/srli).
                // Same-width or narrowing (the `else` below) keeps the low bits already in place.
                const signed = isSigned(func, cv.value);
                const rd = ctx.dst(result, scratch1);
                if (src_bits == 8) {
                    // The register-to-register movzx/movsx-byte forms read `src`'s low byte
                    // directly, so without a REX prefix `src` must be eax, ecx, edx, or ebx.
                    // Stage it through the byte-addressable scratch1 when its home is not, the
                    // same staging the icmp/8-bit-store arms above use.
                    var val = try ctx.use(allocator, cv.value, scratch1);
                    if (!isByteReg(val)) {
                        try ctx.put(allocator, encode.movReg(scratch1, val));
                        val = scratch1;
                    }
                    try ctx.put(allocator, if (signed) encode.movsxByte(rd, val) else encode.movzxByte(rd, val));
                } else {
                    std.debug.assert(src_bits == 16); // src_bits < dst_bits <= 32 leaves only 8/16 here
                    const val = try ctx.use(allocator, cv.value, scratch2);
                    try ctx.put(allocator, if (signed) encode.movsxWord(rd, val) else encode.movzxWord(rd, val));
                }
                try ctx.store(allocator, result, rd);
            } else {
                const src = try ctx.use(allocator, cv.value, scratch2);
                const rd = ctx.dst(result, scratch1);
                if (rd != src) try ctx.put(allocator, encode.movReg(rd, src));
                try ctx.store(allocator, result, rd);
            }
        },
        else => return error.Unsupported,
    }
}

fn emitIf(allocator: std.mem.Allocator, ctx: *Ctx, cf: ir.function.If, pred: Block) Error!void {
    const cond = try ctx.use(allocator, cf.cond, scratch1);
    try ctx.put(allocator, encode.testReg(cond, cond));
    const jnz = try emitBranch(allocator, ctx, encode.jcc(.ne, 0));
    try emitMoves(allocator, ctx, cf.@"else", pred);
    try emitBranchTo(allocator, ctx, encode.jmp(0), @intFromEnum(cf.@"else".target));
    const then_start = ctx.code.items.len;
    const rel: i32 = @intCast(@as(i64, @intCast(then_start)) - @as(i64, @intCast(jnz + 4)));
    std.mem.writeInt(u32, ctx.code.items[jnz..][0..4], @bitCast(rel), .little);
    try emitMoves(allocator, ctx, cf.then, pred);
    try emitBranchTo(allocator, ctx, encode.jmp(0), @intFromEnum(cf.then.target));
}

fn emitJump(allocator: std.mem.Allocator, ctx: *Ctx, jump: ir.function.Jump, pred: Block) Error!void {
    try emitMoves(allocator, ctx, jump, pred);
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

fn emitMoves(allocator: std.mem.Allocator, ctx: *Ctx, jump: ir.function.Jump, pred: Block) Error!void {
    // Shared Wimmer path: the allocator already resolved this edge into an ordered parallel-move
    // sequence (params, live-through values, spills, and cycles broken through the class scratch), so
    // replay it op by op and derive nothing. `translateAllocationX86` always runs before emission and
    // unconditionally sets `edge_move_driven`, so there is no other producer left to derive from.
    std.debug.assert(ctx.edge_move_driven);
    try emitEdgeMovesX86(allocator, ctx, pred, jump.target);
}

/// The precomputed edge-move set for `pred -> succ`, or null when the edge needs no shuffle.
fn findEdgeMovesX86(ctx: *const Ctx, pred: Block, succ: Block) ?*const EdgeMoveSet {
    for (ctx.edge_moves) |*set| {
        if (set.pred == pred and set.succ == succ) return set;
    }
    return null;
}

/// Replay the precomputed, already-ordered edge moves for `pred -> succ` op by op (the Wimmer path).
/// The shared allocator resolved the parallel move (it reads sources before it overwrites them, and
/// it routes any slot-to-slot shuffle through the class scratch), so each move is a primitive
/// register or slot op.
fn emitEdgeMovesX86(allocator: std.mem.Allocator, ctx: *Ctx, pred: Block, succ: Block) Error!void {
    const set = findEdgeMovesX86(ctx, pred, succ) orelse return;
    for (set.moves) |m| try emitOneEdgeMoveX86(allocator, ctx, m);
}

/// Emit one ordered edge move. A register-to-register move emits a `mov` (skipped when equal), a
/// register-to-slot move emits a spill store, a slot-to-register move emits a reload. A slot-to-slot
/// move never appears (the shared ordering expanded it through the class scratch), so it is
/// unreachable.
fn emitOneEdgeMoveX86(allocator: std.mem.Allocator, ctx: *Ctx, m: EdgeMove) Error!void {
    switch (m.src) {
        .reg => |si| {
            const sr: Reg = @enumFromInt(@as(u3, @intCast(si)));
            switch (m.dst) {
                .reg => |di| {
                    const dr: Reg = @enumFromInt(@as(u3, @intCast(di)));
                    if (sr != dr) try ctx.put(allocator, encode.movReg(dr, sr));
                },
                .slot => |ds| try ctx.put(allocator, encode.stackStore(slotDisp(ds), sr)),
            }
        },
        .slot => |ss| switch (m.dst) {
            .reg => |di| try ctx.put(allocator, encode.stackLoad(@enumFromInt(@as(u3, @intCast(di))), slotDisp(ss))),
            .slot => unreachable, // slot->slot was expanded through the class scratch
        },
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

fn isSigned(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.signedness == .signed,
        else => true,
    };
}

/// A value's bit width. The caller has already checked its type is `.int`.
fn intBits(func: *const Function, v: Value) u16 {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.bits,
        else => unreachable, // caller checked type_kind == .int first
    };
}

/// The 64-bit IEEE-754 pattern of `v` when `v` is a compile-time float constant, else `null`.
/// x86-32 has no float registers, so a `double` variadic-call argument is pushed by reading the
/// constant's bits straight from its defining `fconst` (a non-constant float arg is unsupported).
fn floatConstBits(func: *const Function, v: Value) ?u64 {
    const inst = func.definingInst(v) orelse return null;
    return switch (func.opcode(inst)) {
        .fconst => |val| @bitCast(val),
        else => null,
    };
}

/// Lay out the alloca region: each `alloca` result gets a naturally-aligned byte offset
/// (relative to the region base), recorded in `map`. Returns the region's total size. x86-32
/// is integer-only, so a non-integer `elem` (float/vector/aggregate) is rejected.
fn computeAllocaSlots(allocator: std.mem.Allocator, func: *const Function, map: *std.AutoHashMapUnmanaged(Value, u32)) Error!u32 {
    var cur: u32 = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .alloca => |al| {
                    const size = try typeSize(func, al.elem);
                    cur = alignUp(cur, typeAlign(size));
                    try map.put(allocator, func.instResult(inst).?, cur);
                    cur += size;
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

/// The storage size of an alloca's element type, in bytes. x86-32 is integer-only.
fn typeSize(func: *const Function, ty: ir.types.Type) Error!u32 {
    return switch (func.types.type_kind(ty)) {
        .int => |i| (@as(u32, i.bits) + 7) / 8,
        // A blob-typed alloca, sized for its stack slot only. `cc.compile` always lowers
        // `__builtin_va_list` through the host layout (aarch64), never the eventual x86-32
        // target's own `.ptr` shape (see `ctype.builtinVaList`). So a variadic function's
        // `va_list` object arrives as an aggregate array of 64-bit words. x86-32's own
        // define-side lowering only ever reads or writes the first 4 bytes of it (a bare
        // `char*` walking the caller's stack), so the larger blob is a harmless over-allocation.
        // This is gated on `is_variadic`, so a non-variadic function keeps rejecting aggregates
        // exactly as before (its frame stays byte-identical). This mirrors riscv64's identical fix.
        .array => |a| if (func.is_variadic) @as(u32, @intCast(a.len)) * try typeSize(func, a.elem) else error.Unsupported,
        else => error.Unsupported,
    };
}

/// The natural alignment for a storage size, capped at 4 (the widest x86-32 GPR load/store).
fn typeAlign(size: u32) u32 {
    return if (size <= 1) 1 else if (size <= 2) 2 else 4;
}

// ===========================================================================
// This is the i386 (cdecl) variadic define side. cdecl passes every argument on the caller's
// stack (no register-passed arguments), so `va_list` is a plain `char*` walking that stack:
// there is no register-save-area and no prologue spill. `va_start` seeds the pointer just past
// the last named (fixed) parameter. `va_arg` reads through it and bumps it by the argument's
// own 4-byte-rounded size. `va_end` is a no-op (see `emitVaStart`/`emitVaArg` below).
// ===========================================================================

/// The cdecl stack-slot size (4-byte rounded) of one variadic-argument type. An `int` or `ptr`
/// takes one 4-byte slot. A `double` (the only shape a variadic `float` argument reaches, after
/// C's default argument promotion) takes two. This is shared by `va_start` (summing the fixed
/// parameters' slots to find the first unnamed one) and `va_arg` (the walked type's own
/// stride).
fn vaSlotSize(func: *const Function, ty: ir.types.Type) Error!u32 {
    return switch (func.types.type_kind(ty)) {
        .int => |i| alignUp((@as(u32, i.bits) + 7) / 8, 4),
        .float => |f| alignUp(@as(u32, switch (f) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
            .f128 => 16,
        }), 4),
        .ptr => 4,
        else => error.Unsupported,
    };
}

/// The total 4-byte-rounded byte size of this variadic function's fixed (named) parameters, in
/// their declared order. `va_start` uses this to find the address just past them, the first
/// unnamed argument. This backend only ever homes an `int` or `ptr` fixed parameter (x86-32 is
/// integer-only, see the module header), so in every case this backend actually reaches, each
/// contributes exactly 4 bytes. `vaSlotSize` is still used (rather than a hardcoded 4), so this
/// stays correct if a wider fixed-parameter shape is ever supported.
fn fixedNamedBytes(func: *const Function) Error!u32 {
    const eparams = func.blockParams(@enumFromInt(0));
    const n_fixed = @min(func.num_fixed_params, @as(u32, @intCast(eparams.len)));
    var total: u32 = 0;
    for (eparams[0..n_fixed]) |p| total += try vaSlotSize(func, func.valueType(p));
    return total;
}

/// Expand `va_start(list)` (cdecl): `*list = ` the address of the first unnamed incoming stack
/// argument. cdecl has no register-save-area, so that address is simply `esp + va_frame +
/// 4*va_saved + 4 + (the fixed parameters' total slot size)`. `va_frame` is the prologue's `sub
/// esp` amount. `4*va_saved` steps over the callee-saved registers the prologue pushed above the
/// frame (each shifts the incoming stack args one slot higher). The `+4` steps over the
/// return address. Together this is the same base the entry-parameter-load loop in
/// `emitFromAllocation` already uses for argument `i`, at `+ 4*i`. `list` is the `va_list`
/// object's address (an alloca slot), not the `va_list` value itself.
fn emitVaStart(allocator: std.mem.Allocator, ctx: *Ctx, func: *const Function, list: Value) Error!void {
    const named_bytes = try fixedNamedBytes(func);
    const first_off: i32 = ctx.va_frame + @as(i32, @intCast(4 * ctx.va_saved)) + 4 + @as(i32, @intCast(named_bytes));
    const lp = try ctx.use(allocator, list, scratch2);
    try ctx.put(allocator, encode.leaFromStack(scratch1, first_off)); // scratch1 = &(first vararg)
    try ctx.put(allocator, encode.movToMem32(lp, 0, scratch1)); // *list = scratch1
}

/// Expand `va_arg(list, ty)` (cdecl): `p = *list; *list = p + roundUp(sizeof(ty),4); result =
/// *p`. Every variadic argument sits in one contiguous, ascending run of 4-byte-rounded slots on
/// the caller's stack (cdecl has no register-save-area), so the walk is a plain pointer bump.
/// `aluMemImm8` bumps the `*list` field directly in memory, so only one scratch register (the
/// loaded pointer `p`) is ever needed besides `list`'s own location.
///
/// x86-32 is integer-only (see the module header): an int or pointer result loads normally, but a
/// float result cannot be held in any register on this backend (there is no float register
/// class at all, this is not specific to variadic handling). Mirroring `fconst`'s own established
/// convention for this exact situation, only the pointer advance is emitted and no result-defining
/// code is produced. A later use of that (unrepresentable) value is rejected by that use's own
/// integer-only guard, exactly like a float constant reaching a non-call use. The advance itself
/// is still fully correct for a `double` (an 8-byte stride), so a caller that discards the
/// result (just to skip over it) still walks the list correctly.
fn emitVaArg(allocator: std.mem.Allocator, ctx: *Ctx, func: *const Function, list: Value, result: Value) Error!void {
    const rty = func.valueType(result);
    const is_int = func.types.type_kind(rty) == .int;
    if (is_int and intBits(func, result) > 32) return error.Unsupported; // no multi-register wide-int support here
    const size = try vaSlotSize(func, rty);
    const lp = try ctx.use(allocator, list, scratch2);
    try ctx.put(allocator, encode.movFromMem32(scratch1, lp, 0)); // scratch1 = p = *list
    try ctx.put(allocator, encode.aluMemImm8(0, lp, 0, @intCast(size))); // *list += size
    if (!is_int) return; // float result: advance only (see doc comment above)
    const signed = isSigned(func, result);
    const rd = ctx.dst(result, scratch1); // may legally coincide with scratch1 (base==dst is a plain `mov reg,[reg]`)
    try ctx.put(allocator, switch (intBits(func, result)) {
        0...8 => if (signed) encode.movsxByteFromMem(rd, scratch1, 0) else encode.movzxByteFromMem(rd, scratch1, 0),
        9...16 => if (signed) encode.movsxWordFromMem(rd, scratch1, 0) else encode.movzxWordFromMem(rd, scratch1, 0),
        else => encode.movFromMem32(rd, scratch1, 0),
    });
    try ctx.store(allocator, result, rd);
}

// ===========================================================================
// The shared Wimmer-Franz allocator (`wimmer.zig`) drives the production
// `compile`/`selectFunction`, emitting through the per-opcode arms above through
// a `RegDescription` and a split allocate, translate, emit pipeline (no native
// allocator, no fallback). The `compileFunctionWimmerX86`/`...Fold` entries below
// are test-only thin wrappers the differential exercises. Byte-addressability of
// `setcc`/8-bit ops (esi has no low byte) is resolved at emit time by staging
// through the reserved ebx scratch (the icmp arm above), needing no new Wimmer
// capability. There is one register class (integer gpr); x86-32 has no floats.
// ===========================================================================

/// The caller-saved gpr set {eax,ecx,edx}. A call clobbers exactly these; esi survives it. ebx and
/// edi are reserved scratch and never enter the pool.
const caller_saved_gpr = [_]Reg{ .eax, .ecx, .edx };
/// The callee-saved gpr set {esi}. It is allocatable (so a cross-call value can live there instead
/// of spilling) but survives a call, so no call clobbers it. It is saved and restored in the
/// prologue when used.
const callee_saved_gpr = [_]Reg{.esi};

// The backend context the shared allocator threads through `classOf`/`useKind`. x86-32 needs no extra
// state (its decisions read only the function, passed separately), so this is a zero-field singleton
// whose address is a stable, non-owned `ctx` pointer.
const X86RegCtx = struct {};
const x86_reg_ctx: X86RegCtx = .{};

/// `RegDescription.classOf` for x86-32: every value lives in the single gpr class (0). x86-32 has no
/// floats or vectors, so there is no second class.
fn x86ClassOf(ctx: *const anyopaque, func: *const Function, v: Value) u16 {
    _ = ctx;
    _ = func;
    _ = v;
    return 0;
}

/// `RegDescription.useKind` for x86-32: every gpr operand is `must_have_register` (mirroring x86_64's
/// gpr rule), so a cross-call value escapes into callee-saved esi rather than a slot, and the
/// register-reload-at-use behavior is preserved. The emitter's `use` still reloads a spilled operand
/// into a scratch as a fallback. `inst` and `operand` are unused (the kind is uniform for the one
/// class).
fn x86UseKind(ctx: *const anyopaque, func: *const Function, inst: ir.function.Inst, operand: Value) wimmer.UseKind {
    _ = ctx;
    _ = func;
    _ = inst;
    _ = operand;
    return .must_have_register;
}

/// Append a `u16` index for every register in `regs` to `list` (via `@intFromEnum`).
fn appendRegIndices(allocator: std.mem.Allocator, list: *std.ArrayList(u16), regs: []const Reg) Error!void {
    for (regs) |r| try list.append(allocator, @intFromEnum(r));
}

/// The kind of fixed-register clobber a single instruction contributes at its position. A call
/// clobbers all caller-saved registers, a div/rem needs eax and edx, a variable shift needs ecx,
/// everything else clobbers nothing. This mirrors x86_64's `clobberKindOf`.
const ClobberKind = enum { none, call, div, shift };

/// Which fixed-register clobber `inst` contributes. A `div`/`rem` (arith or arith_imm) uses eax and
/// edx. A `shl`/`shr` `arith` uses ecx (an `arith_imm` shift has an immediate count, so it needs no
/// ecx). A call clobbers every caller-saved register.
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

/// Free a per-class clobber list (each class's `regs`, then the outer slice).
fn freeClassRegs(allocator: std.mem.Allocator, cr: []const wimmer.ClassRegs) void {
    for (cr) |c| allocator.free(c.regs);
    allocator.free(cr);
}

/// Build the per-class clobber list for a clobber `kind`, or null when the instruction clobbers
/// nothing (so no site is recorded). A call clobbers {eax,ecx,edx}. A div clobbers {eax,edx}. A shift
/// clobbers {ecx}. The caller owns the returned slices and must free them through `RegDescription.deinit`.
fn buildClobber(allocator: std.mem.Allocator, kind: ClobberKind) Error!?[]wimmer.ClassRegs {
    switch (kind) {
        .none => return null,
        .call => {
            const regs = try allocator.alloc(u16, caller_saved_gpr.len);
            errdefer allocator.free(regs);
            for (caller_saved_gpr, 0..) |r, i| regs[i] = @intFromEnum(r);
            const clob = try allocator.alloc(wimmer.ClassRegs, 1);
            clob[0] = .{ .class = 0, .regs = regs };
            return clob;
        },
        .div => {
            const regs = try allocator.alloc(u16, 2);
            errdefer allocator.free(regs);
            regs[0] = @intFromEnum(Reg.eax);
            regs[1] = @intFromEnum(Reg.edx);
            const clob = try allocator.alloc(wimmer.ClassRegs, 1);
            clob[0] = .{ .class = 0, .regs = regs };
            return clob;
        },
        .shift => {
            const regs = try allocator.alloc(u16, 1);
            errdefer allocator.free(regs);
            regs[0] = @intFromEnum(Reg.ecx);
            const clob = try allocator.alloc(wimmer.ClassRegs, 1);
            clob[0] = .{ .class = 0, .regs = regs };
            return clob;
        },
    }
}

/// Build the per-function x86-32 `RegDescription` the shared Wimmer-Franz allocator consumes. It has
/// one class (gpr), where the physical-register index equals the register's own enum value.
/// Allocatable over {eax,ecx,edx,esi}; esi is callee-saved (a cross-call value can live there
/// instead of spilling). ebx and edi are reserved scratch (ebx byte-addressable for setcc/8-bit
/// staging, edi the parallel-move scratch) and never enter the pool. `entry_fixed` is empty: cdecl
/// passes every argument on the stack, so no parameter pins an ABI register (the prologue loads each
/// from its stack slot into the allocator-chosen home). Each call, div/rem, and shl/shr becomes a
/// per-position clobber site: a call clobbers {eax,ecx,edx}, a div clobbers {eax,edx}, a shift
/// clobbers {ecx}. The caller owns the result and must call `deinit` on it.
pub fn x86_32RegDescription(allocator: std.mem.Allocator, func: *const Function) Error!wimmer.RegDescription {
    // --- Class 0 (gpr): caller-saved and callee-saved, 4-byte slots. ---
    var gpr_alloc: std.ArrayList(u16) = .empty;
    errdefer gpr_alloc.deinit(allocator);
    try appendRegIndices(allocator, &gpr_alloc, &caller_saved_gpr);
    try appendRegIndices(allocator, &gpr_alloc, &callee_saved_gpr);
    const gpr_alloc_owned = try gpr_alloc.toOwnedSlice(allocator);
    errdefer allocator.free(gpr_alloc_owned);

    var gpr_cs: std.ArrayList(u16) = .empty;
    errdefer gpr_cs.deinit(allocator);
    try appendRegIndices(allocator, &gpr_cs, &callee_saved_gpr);
    const gpr_cs_owned = try gpr_cs.toOwnedSlice(allocator);
    errdefer allocator.free(gpr_cs_owned);

    const classes = try allocator.alloc(wimmer.RegClass, 1);
    errdefer allocator.free(classes);
    classes[0] = .{ .name = "gpr", .allocatable = gpr_alloc_owned, .callee_saved = gpr_cs_owned, .slot_bytes = 4 };

    // --- Entry params: empty. cdecl args arrive on the stack, so no parameter pins an ABI register. ---
    const entry_fixed = try allocator.alloc(wimmer.FixedAssign, 0);
    errdefer allocator.free(entry_fixed);

    // --- Clobber sites: one per call, div/rem, and shl/shr position, in the same single-step numbering
    // `buildIntervals` uses (block-param row, one position per instruction, one terminator slot, over
    // every block), so the positions line up with the intervals. ---
    var sites: std.ArrayList(wimmer.CallSite) = .empty;
    var built: usize = 0;
    errdefer {
        for (sites.items[0..built]) |cs| freeClassRegs(allocator, cs.clobbered);
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
        for (call_sites) |cs| freeClassRegs(allocator, cs.clobbered);
        allocator.free(call_sites);
    }

    // --- Scratch: exactly one per class (the shared allocator asserts `scratch.len == classes.len`),
    // the register it routes parallel-move cycles and slot-to-slot shuffles through. That is edi
    // (`move_scratch`), matching `emitSplitActionX86`/`emitOneEdgeMoveX86`. The other reserved scratch
    // register ebx (byte-addressable staging and operand reload) is kept out of play purely by its
    // absence from the allocatable set, exactly as x86_64 reserves r10 without listing it here. ---
    const scratch = try allocator.alloc(u16, 1);
    errdefer allocator.free(scratch);
    scratch[0] = @intFromEnum(move_scratch); // edi

    return .{
        .classes = classes,
        .classOf = x86ClassOf,
        .useKind = x86UseKind,
        .coalesce_spill_slots = true,
        .entry_fixed = entry_fixed,
        .call_sites = call_sites,
        .scratch = scratch,
        .ctx = &x86_reg_ctx,
    };
}

/// Map a shared gpr `wimmer.Location` to this backend's `Loc` (register index to the Reg enum, or a
/// per-class slot to a spill slot).
fn wimmerGprLoc(loc: wimmer.Location) Loc {
    return switch (loc) {
        .reg => |ri| .{ .reg = @enumFromInt(@as(u3, @intCast(ri))) },
        .slot => |s| .{ .spill = s },
    };
}

/// Build the gpr drain action realizing `src -> dst` for `value` at `at` (already translated into
/// this backend's `Loc`). Register to slot produces a `store`, slot to register a `reload`, register
/// to register a `move`, and slot to slot a `slot_to_slot` (`emitSplitActionX86` expands it into a
/// reload-then-store pair through the class scratch register edi, so it never needs a value register
/// of its own).
fn wimmerGprTransition(value: Value, src: Loc, dst: Loc, at: u32) SplitAction {
    return switch (src) {
        .reg => |sr| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .move, .value = value, .reg = dr, .move_from = sr },
            .spill => |ds| .{ .at = at, .kind = .store, .value = value, .reg = sr, .slot = ds },
        },
        .spill => |ss| switch (dst) {
            .reg => |dr| .{ .at = at, .kind = .reload, .value = value, .reg = dr, .slot = ss },
            .spill => |ds| .{ .at = at, .kind = .slot_to_slot, .value = value, .slot = ds, .move_from_slot = ss },
        },
    };
}

/// Map a shared edge-move `wimmer.Location` to an `EdgeLoc` (class-relative register index or slot).
fn edgeLocX86(loc: wimmer.Location) EdgeLoc {
    return switch (loc) {
        .reg => |ri| .{ .reg = ri },
        .slot => |s| .{ .slot = s },
    };
}

fn regLessThanX86(_: void, a: Reg, b: Reg) bool {
    return @intFromEnum(a) < @intFromEnum(b);
}

/// Translate a finished shared `wimmer.Allocation` into a filled `ctx` (loc_of, segments, actions,
/// edge_moves, def_pos) plus the spill-slot count and the callee-saved gpr push set. A whole-life
/// value (one segment) lands in `loc_of`. A genuinely split value lands in `segments`, and the shared
/// allocator's already-ordered `walloc.actions` is consumed verbatim into `ctx.actions` (one
/// store, reload, move, or slot_to_slot per intra-block boundary, in hazard-free order). The
/// entry-param moves are handled by the `emitFromAllocation` prologue (it loads each stack arg into
/// the param's location, whatever the allocator chose), so no ABI-register requirement is imposed here.
fn translateAllocationX86(
    allocator: std.mem.Allocator,
    func: *const Function,
    walloc: *const wimmer.Allocation,
    ctx: *Ctx,
    num_slots_out: *u32,
    saved: *std.ArrayList(Reg),
) Error!void {
    // def_pos uses the same single-step numbering the shared allocator and `emitFromAllocation` use
    // (block-param row, one position per instruction, one terminator slot, over every block). It is
    // owned by `ctx` immediately, so the caller's `defer` frees it on any later failure.
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

    std.debug.assert(walloc.slot_count_per_class.len == 1);
    num_slots_out.* = walloc.slot_count_per_class[0];

    var it = walloc.segments.iterator();
    while (it.next()) |e| {
        const value = e.key_ptr.*;
        const wsegs = e.value_ptr.*;
        std.debug.assert(wsegs.len > 0);
        if (wsegs.len == 1) {
            try ctx.loc_of.put(allocator, value, wimmerGprLoc(wsegs[0].loc));
            continue;
        }
        const segs = try allocator.alloc(Segment, wsegs.len);
        for (wsegs, 0..) |ws, i| segs[i] = .{ .from = ws.from, .loc = wimmerGprLoc(ws.loc) };
        ctx.segments.put(allocator, value, segs) catch |err| {
            allocator.free(segs);
            return err;
        };
    }

    // Consume the shared allocator's already-ordered intra-block actions verbatim: each is one
    // primitive transfer at its position, ascending by `at`, with each same-position cluster in
    // hazard-free order (`orderIntraActions`), so appending it and draining in order never overwrites
    // a live value. A cross-block location change is not here (it is an edge move, translated below).
    for (walloc.actions) |wa| {
        std.debug.assert(wa.class == 0); // x86-32 has one class
        const src = wimmerGprLoc(wa.src);
        const dst = wimmerGprLoc(wa.dst);
        try ctx.actions.append(allocator, wimmerGprTransition(wa.value, src, dst, wa.at));
    }

    // Callee-saved gprs the allocation used become the prologue push set.
    for (walloc.used_callee_saved) |us| {
        std.debug.assert(us.class == 0);
        try saved.append(allocator, @enumFromInt(@as(u3, @intCast(us.reg))));
    }
    std.mem.sort(Reg, saved.items, {}, regLessThanX86);

    // Control-flow-edge moves: translate each ordered `wimmer.Move` into an `EdgeMove`, keyed by
    // (pred, succ). `emitMoves` replays them when `edge_move_driven` is set.
    var edge_sets: std.ArrayList(EdgeMoveSet) = .empty;
    errdefer {
        for (edge_sets.items) |es| allocator.free(es.moves);
        edge_sets.deinit(allocator);
    }
    for (walloc.edge_moves) |wem| {
        const moves = try allocator.alloc(EdgeMove, wem.moves.len);
        errdefer allocator.free(moves);
        for (wem.moves, 0..) |wm, i| {
            std.debug.assert(wm.class == 0);
            moves[i] = .{ .src = edgeLocX86(wm.src), .dst = edgeLocX86(wm.dst) };
        }
        try edge_sets.append(allocator, .{ .pred = wem.pred, .succ = wem.succ, .moves = moves });
    }
    ctx.edge_moves = try edge_sets.toOwnedSlice(allocator);
    ctx.edge_move_driven = true;
}

/// Emit one split-boundary drain action. A `store` writes `reg` to its slot. A `reload` brings a slot
/// back into `reg`. A `move` copies `move_from` into `reg` (an identity move emits nothing). A
/// `slot_to_slot` action re-homes a spilled value from `move_from_slot` to `slot` through the gpr class
/// scratch register edi (reserved out of every pool, so this touch never conflicts with a live value).
fn emitSplitActionX86(allocator: std.mem.Allocator, ctx: *Ctx, act: SplitAction) Error!void {
    switch (act.kind) {
        .store => try ctx.put(allocator, encode.stackStore(slotDisp(act.slot), act.reg)),
        .reload => try ctx.put(allocator, encode.stackLoad(act.reg, slotDisp(act.slot))),
        .move => if (act.reg != act.move_from) try ctx.put(allocator, encode.movReg(act.reg, act.move_from)),
        .slot_to_slot => {
            try ctx.put(allocator, encode.stackLoad(move_scratch, slotDisp(act.move_from_slot)));
            try ctx.put(allocator, encode.stackStore(slotDisp(act.slot), move_scratch));
        },
    }
}

/// Compute the stack frame and fill `ctx.alloca_base`. The frame is the spill slots (4 bytes each),
/// then the alloca region, 16-aligned. Callee-saved pushes sit above the
/// frame (higher addresses than the `sub esp` region), so spill and alloca offsets are independent of
/// how many registers are pushed; the stack-arg loads in `emitFromAllocation` add the push delta.
fn frameLayoutW(allocator: std.mem.Allocator, ctx: *Ctx, func: *const Function, num_slots: u32) Error!i32 {
    ctx.alloca_base = @intCast(num_slots * 4);
    const alloca_bytes = try computeAllocaSlots(allocator, func, &ctx.alloca_off);
    return @intCast((@as(u64, num_slots) * 4 + alloca_bytes + 15) & ~@as(u64, 15));
}

/// Emit machine code from a finished, filled `ctx` (segments, actions, edge moves, def_pos) plus the
/// computed `frame` and the callee-saved gprs `saved` to preserve. This walks blocks in order,
/// threading `ctx.pos` per instruction (so `loc` picks the active segment), draining `ctx.actions` at
/// each position before the instruction, and reusing the same per-opcode arms (`lowerInst`, `emitIf`,
/// `emitJump`) the whole backend shares. `saved` is pushed at the prologue (before `sub esp`) and popped
/// in reverse order at each epilogue (after `add esp`).
fn emitFromAllocation(allocator: std.mem.Allocator, ctx: *Ctx, func: *const Function, frame: i32, saved: []const Reg) Error!Compiled {
    const nblocks = func.blockCount();
    const block_start = try allocator.alloc(usize, nblocks);
    defer allocator.free(block_start);

    // Prologue: push the used callee-saved gprs (above the frame), reserve the spill frame, then load
    // each cdecl argument from its stack slot into its allocator-chosen home. After the pushes and the
    // `sub esp, frame`, an argument that sat at `[esp_entry + 4 + 4*i]` is at `[esp + frame +
    // 4*num_pushed + 4 + 4*i]` (the pushes moved esp down by 4 each). A register parameter goes to its
    // register, a spilled one is staged through ebx into its slot.
    for (saved) |r| try ctx.put(allocator, encode.pushReg(r));
    if (frame > 0) try ctx.put(allocator, encode.aluImm(5, .esp, frame)); // sub esp, frame
    ctx.va_frame = frame;
    ctx.va_saved = @intCast(saved.len);
    ctx.pos = 0;
    const eparams = func.blockParams(@enumFromInt(0));
    for (eparams, 0..) |p, i| {
        const src: i32 = frame + @as(i32, @intCast(4 * saved.len)) + 4 + @as(i32, @intCast(4 * i));
        switch (ctx.loc(p)) {
            .reg => |r| try ctx.put(allocator, encode.stackLoad(r, src)),
            .spill => |slot| {
                try ctx.put(allocator, encode.stackLoad(scratch1, src));
                try ctx.put(allocator, encode.stackStore(slotDisp(slot), scratch1));
            },
        }
    }

    var pos_base: u32 = 0;
    var action_cursor: usize = 0;
    for (0..nblocks) |bi| {
        const block: Block = @enumFromInt(bi);
        block_start[bi] = ctx.code.items.len;
        // A neutralized (unreachable) block: `neutralizeUnreachable` emptied its params, instructions,
        // and terminator, so downstream passes see only the reachable CFG. It carries no code, and no
        // reachable branch targets it (this is valid SSA), so emit nothing. The shared Wimmer allocator
        // still numbers it (param row + 0 instructions + terminator slot = 2 positions), so advance
        // `pos_base` over that span and discard any split actions the allocator recorded inside a block
        // that is never entered. When every block is reachable this branch is never taken, so output
        // stays byte-identical for every normal function. The detector (no instructions and a null
        // terminator) is unambiguous: a block ending in a structured `if` has instructions, and a
        // forwarding block has a non-null jump terminator, so only a neutralized block matches both.
        if (func.blockInsts(block).len == 0 and func.terminator(block) == null) {
            const dead_end = pos_base + 1; // last position this empty block owns (param row, terminator slot)
            while (action_cursor < ctx.actions.items.len and ctx.actions.items[action_cursor].at <= dead_end) {
                action_cursor += 1;
            }
            pos_base += 2;
            continue;
        }
        var terminated = false;
        const insts = func.blockInsts(block);
        for (insts, 0..) |inst, inst_idx| {
            // Derive the position from the block base plus the instruction index (continue-safe), which
            // equals `translateAllocationX86`'s numbering for this instruction.
            ctx.pos = pos_base + 1 + @as(u32, @intCast(inst_idx));
            // An instruction with a result must be emitted at exactly that result's definition
            // position. This pins the emission numbering to the allocator's and must never trip.
            if (func.instResult(inst)) |r| std.debug.assert(ctx.pos == ctx.def_pos[@intFromEnum(r)]);
            // Drain re-home actions landing at this position before emitting the instruction.
            while (action_cursor < ctx.actions.items.len and ctx.actions.items[action_cursor].at <= ctx.pos) {
                const act = ctx.actions.items[action_cursor];
                std.debug.assert(act.at == ctx.pos); // actions land on instruction positions only
                try emitSplitActionX86(allocator, ctx, act);
                action_cursor += 1;
            }
            if (func.opcode(inst) == .@"if") {
                try emitIf(allocator, ctx, func.opcode(inst).@"if", block);
                terminated = true;
            } else {
                try lowerInst(allocator, ctx, inst);
            }
        }
        // The terminator shares the block-end position. Drain any actions recorded at it (for example a
        // value re-homed only for a `ret` or edge use) before the terminator emits.
        ctx.pos = pos_base + 1 + @as(u32, @intCast(insts.len));
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
                        const src = try ctx.use(allocator, value, ret_reg);
                        if (src != ret_reg) try ctx.put(allocator, encode.movReg(ret_reg, src));
                    },
                    else => return error.Unsupported, // multi-value struct return not yet lowered
                }
                if (frame > 0) try ctx.put(allocator, encode.aluImm(0, .esp, frame)); // add esp, frame
                // Epilogue: restore callee-saved gprs in reverse push order, then return.
                var si: usize = saved.len;
                while (si > 0) {
                    si -= 1;
                    try ctx.put(allocator, encode.popReg(saved[si]));
                }
                try ctx.put(allocator, encode.ret());
            },
            .jump => |j| try emitJump(allocator, ctx, j, block),
        };
        pos_base = pos_base + 2 + @as(u32, @intCast(insts.len));
    }
    // Every action must have drained: an action lands at an instruction or terminator position, and
    // all of those are visited above.
    std.debug.assert(action_cursor == ctx.actions.items.len);

    for (ctx.fixups.items) |f| {
        const rel: i32 = @intCast(@as(i64, @intCast(block_start[f.target])) - @as(i64, @intCast(f.at + 4)));
        std.mem.writeInt(u32, ctx.code.items[f.at..][0..4], @bitCast(rel), .little);
    }
    return .{ .code = try ctx.code.toOwnedSlice(allocator), .relocs = try ctx.relocs.toOwnedSlice(allocator) };
}

/// Visit every operand value read by `inst`, calling `f(ctx, value, is_edge_arg)`. The block
/// arguments of an `if` are edge args (they move along a control edge); every other operand is an
/// ordinary use. A folded load or store attributes its pointer use to the fold base (`baseOf`), not
/// the dead add's own result, so the base stays live to the memory op and the dead add's result gets
/// no use. With the empty analysis, `baseOf` returns the raw ptr, so the non-folding case is
/// unchanged. This mirrors the x86_64 `forEachOperand`, kept local so `countUses` can score a value
/// the exact way the fold rewrite repoints operands.
fn forEachOperand(func: *const Function, inst: ir.function.Inst, fold: *const addrfold.Analysis, ctx: anytype, comptime f: fn (@TypeOf(ctx), Value, bool) void) void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .fconst128, .alloca, .global_addr => {},
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
        .load => f(ctx, fold.baseOf(func, inst), false),
        .store => |st| {
            f(ctx, st.value, false);
            f(ctx, fold.baseOf(func, inst), false);
        },
        .prefetch => |pf| f(ctx, pf.ptr, false),
        // Not yet expanded by this backend, but still visited here so
        // regalloc sees `list` as a use ahead of that.
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

/// Terminator analogue of `forEachOperand`. The `jump` arguments are edge args. The `ret` value is
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

/// Total operand uses of `v` across the whole function (instruction operands, if/jump edge args, and
/// terminators). This counts raw operands (`empty_fold`), so after `applyFoldRewriteX86` repoints a
/// folded memory op's pointer to its base, a dead address-add's result scores zero. This backs the
/// dead-code-elimination assert in `applyFoldRewriteX86`.
fn countUses(func: *const Function, v: Value) usize {
    var count: usize = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| forEachOperand(func, inst, &empty_fold, CountCtx{ .target = v, .count = &count }, countOperand);
        if (func.terminator(block)) |term| forEachTermOperand(func, term, CountCtx{ .target = v, .count = &count }, countOperand);
    }
    return count;
}

/// Make each address fold visible to the fold-blind shared allocator. For every folded memory op the
/// analysis found, repoint its pointer operand to the fold base (the add's left operand), then delete
/// each now-dead address-add. `folds` stays consistent for emission: it is keyed by the surviving
/// memory inst and holds base+off, so `baseOf` returns the (now raw) ptr, equal to base, and `offOf`
/// the displacement. Only dead adds are removed, never a memory inst, so the offsets survive. This
/// runs on the caller's function (a Wimmer caller passes a throwaway copy), after critical-edge
/// splitting and before `wimmer.allocate`. It is sound across blocks: base dominated the add, and the
/// add dominated the load, so base dominates the load.
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
    // Drop the dead adds. A dead add's every use was a folded ptr use now repointed to the base, so its
    // result is unused. Assert that before removing it (a surviving use would mean dropping a live
    // definition, a miscompile). Removal order is irrelevant: no dead add's result feeds another
    // instruction.
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

/// Compile `func` through the shared Wimmer-Franz allocator, then emit through the same per-opcode
/// arms as `compile`. Test-only: it is exactly the production `compile` pipeline without the `clone`
/// and `rebindSymbolName` wrapping, run directly on the caller's mutable `func` so the differential
/// can compile a throwaway copy. It runs the shared scan, translates its target-independent
/// `Allocation` into a filled `Ctx`, and reuses the existing emission. It bails with `error.Unsupported`
/// on anything not faithfully translatable, never a silent miscompile. It takes `func` by mutable
/// pointer because `splitCriticalEdges` inserts forwarding blocks in place; a differential caller
/// builds two identical functions and compiles one each way. Address folding is off here (the
/// `...Fold` entry exercises it), so `ctx.fold` stays the empty analysis, matching the pre-fold
/// behavior.
pub fn compileFunctionWimmerX86(allocator: std.mem.Allocator, func: *Function) Error!Compiled {
    if (ir.function.functionUsesF16(func)) return error.Unsupported;
    if (ir.function.functionUsesF128(func)) return error.Unsupported;
    if (func.blockCount() == 0) return error.Unsupported;

    // Split critical edges first (mutating `func`), so the shared resolver's no-critical-edge
    // precondition holds and the RegDescription, scan, and emission all see one CFG.
    try ir.critical_edge.splitCriticalEdges(allocator, func);

    var desc = try x86_32RegDescription(allocator, func);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, func, &desc);
    defer walloc.deinit(allocator);

    var ctx = Ctx{ .func = func };
    defer ctx.loc_of.deinit(allocator);
    defer ctx.code.deinit(allocator);
    defer ctx.fixups.deinit(allocator);
    defer ctx.relocs.deinit(allocator);
    defer ctx.alloca_off.deinit(allocator);
    defer {
        var seg_it = ctx.segments.valueIterator();
        while (seg_it.next()) |s| allocator.free(s.*);
        ctx.segments.deinit(allocator);
    }
    defer ctx.actions.deinit(allocator);
    // `def_pos` is a heap-owned copy (translateAllocationX86 always allocates it). The `&.{}` default
    // is a zero-length slice whose free is a no-op, so an unconditional free is safe on any early exit.
    defer allocator.free(ctx.def_pos);
    defer {
        for (ctx.edge_moves) |es| allocator.free(es.moves);
        allocator.free(ctx.edge_moves);
    }

    var saved: std.ArrayList(Reg) = .empty;
    defer saved.deinit(allocator);
    var num_slots: u32 = 0;
    try translateAllocationX86(allocator, func, &walloc, &ctx, &num_slots, &saved);
    const frame = try frameLayoutW(allocator, &ctx, func, num_slots);
    return emitFromAllocation(allocator, &ctx, func, frame, saved.items);
}

/// Like `compileFunctionWimmerX86`, but with address-mode folding on: the exact pipeline the
/// production flip will use. It analyzes the folds, then `applyFoldRewriteX86` repoints each folded
/// memory op's `ptr` to its base and removes the dead adds in place, so the fold is visible to the
/// fold-blind shared allocator (which reads only raw operands), and `base` stays live to the load or
/// store. The same analysis threads into emission through `ctx.fold`: `folds` is keyed by the
/// surviving memory inst, so `baseOf` returns the (now raw) ptr, equal to base, and `offOf` the
/// displacement, consistent with the rewritten IR (the memory inst survives, only the add is removed,
/// so the offset side-table stays valid). Test-only here (the fold-under-pressure differential
/// exercises the rewrite). It takes `func` by mutable pointer: `splitCriticalEdges` and
/// `applyFoldRewriteX86` mutate it in place, so a differential caller builds two identical functions
/// and compiles one each way.
pub fn compileFunctionWimmerX86Fold(allocator: std.mem.Allocator, func: *Function) Error!Compiled {
    if (ir.function.functionUsesF16(func)) return error.Unsupported;
    if (ir.function.functionUsesF128(func)) return error.Unsupported;
    if (func.blockCount() == 0) return error.Unsupported;

    // Split critical edges first (mutating `func`), matching `compileFunctionWimmerX86`.
    try ir.critical_edge.splitCriticalEdges(allocator, func);

    // Analyze before the rewrite (it reads the `arith_imm.add` each fold rests on), then rewrite the
    // IR so the fold is visible to the fold-blind shared allocator. `analyze` yields an empty analysis
    // when nothing folds, so this path degrades to `compileFunctionWimmerX86`'s behavior on such a
    // function.
    var fold = try addrfold.analyze(allocator, func, {}, x86FoldOffset);
    defer fold.deinit(allocator);
    applyFoldRewriteX86(func, &fold);

    var desc = try x86_32RegDescription(allocator, func);
    defer desc.deinit(allocator);
    var walloc = try wimmer.allocate(allocator, func, &desc);
    defer walloc.deinit(allocator);

    var ctx = Ctx{ .func = func, .fold = &fold };
    defer ctx.loc_of.deinit(allocator);
    defer ctx.code.deinit(allocator);
    defer ctx.fixups.deinit(allocator);
    defer ctx.relocs.deinit(allocator);
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
    try translateAllocationX86(allocator, func, &walloc, &ctx, &num_slots, &saved);
    const frame = try frameLayoutW(allocator, &ctx, func, num_slots);
    return emitFromAllocation(allocator, &ctx, func, frame, saved.items);
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
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(prod) });
    const code = try selectFunction(allocator, &func);
    defer allocator.free(code);
    try std.testing.expectEqual(@as(u8, 0xC3), code[code.len - 1]); // ret
}

test "global_addr emits mov rd, imm32 and an .abs32 reloc at the imm32 field" {
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

    // Exactly one reloc, naming "g", kind `.abs32` (not the default `.call`).
    try std.testing.expectEqual(@as(usize, 1), compiled.relocs.len);
    const r = compiled.relocs[0];
    try std.testing.expectEqualStrings("g", r.symbol);
    try std.testing.expectEqual(Kind.abs32, r.kind);

    // The reloc offset lands on a real `mov rd, imm32` (B8+r), 1 byte before the
    // recorded imm32 field.
    try std.testing.expect(r.offset >= 1);
    const opcode = compiled.code[r.offset - 1];
    try std.testing.expectEqual(@as(u8, 0xB8), opcode & 0xF8); // B8+r short-form mov-immediate
    // The placeholder imm32 is zero (patched later by the linker).
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, compiled.code[r.offset..][0..4], .little));
}

test "a via_got global_addr emits mov rd, [abs32] and a .got_abs reloc (GOT-indirect data import)" {
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

    // Exactly one reloc, naming "g", kind `.got_abs` (the GOT-indirect data import).
    try std.testing.expectEqual(@as(usize, 1), compiled.relocs.len);
    const r = compiled.relocs[0];
    try std.testing.expectEqualStrings("g", r.symbol);
    try std.testing.expectEqual(Kind.got_abs, r.kind);

    // The reloc offset lands on a real `mov rd, [abs32]` (opcode 8B, ModRM mod=00 rm=101),
    // 2 bytes before the recorded abs32 field. The opcode is `0x8B` (mov, a LOAD from the GOT
    // slot's absolute address), NOT `0xB8` (mov rd, imm32, which would embed the address).
    try std.testing.expect(r.offset >= 2);
    const opcode = compiled.code[r.offset - 2];
    const modrm = compiled.code[r.offset - 1];
    try std.testing.expectEqual(@as(u8, 0x8B), opcode);
    try std.testing.expectEqual(@as(u8, 0x05), modrm & 0xC7); // mod=00, rm=101 (disp32 absolute)
    // The placeholder abs32 is zero (patched later by the linker to the GOT slot vaddr).
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

test "an f16 function is rejected cleanly, not miscompiled as f64" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .float = .f16 });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, t);
    const y = try func.appendBlockParam(b, t);
    const s = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(s) });

    try std.testing.expectError(error.Unsupported, selectFunction(allocator, &func));
}
