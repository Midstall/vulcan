//! SASS scoreboard scheduler. Fixed-latency instructions (ALU) are covered by the
//! per-instruction stall delay, but variable-latency instructions (global loads
//! LDG, special-register reads S2R) finish an unknown number of cycles after they
//! issue. The hardware tracks completion with six scoreboards: a producer sets a
//! write barrier on issue, and any later consumer must wait on that scoreboard
//! before reading the register.
//!
//! Walks the instruction stream and, using a conservative model of each
//! instruction's register reads, assigns a write barrier to every variable-latency
//! producer and a wait mask to its consumers. A scoreboard is freed once a consumer
//! has waited on it (the data is then ready for any later read). Correctness pass:
//! without it, a consumer could read a register before the load filling it completes.
//!
//! Register reads come from `readsSrc`. For an ALU op that is a conservative superset
//! (the three ALU source fields), which only ever adds an unnecessary wait. For a memory
//! op, an atomic or a barrier the superset is WRONG in both directions, because those
//! opcodes put other things in the ALU source fields, so each names its own sources.
//!
//! Limits: read barriers (protecting a variable-latency op's source registers from being
//! overwritten before it consumes them) are not assigned. Stall delays are left as the isel
//! set them.

const std = @import("std");
const encode = @import("encode.zig");

const Inst = encode.Inst;
const RZ = encode.RZ;

/// Whether `opcode` is a bindless texturing op that writes the split 4-register RGBA result block
/// and reads its coordinate at bit 24 + bindless HANDLE at bit 32: TEX / TEX.LL (0xd61) and the
/// TLD4 gather (0xd64). The scheduler treats them identically - variable latency, a source read at
/// bit 32/64, and a write barrier that spans all four result registers.
fn isTexResult(opcode: u32) bool {
    return opcode == encode.TEX_OPCODE or opcode == encode.TLD4_OPCODE or opcode == encode.TLD_OPCODE;
}

/// The variable-latency opcodes whose results need a scoreboard. LDC (constant-
/// bank load) is variable-latency too: on a const-cache miss its result lands an
/// unknown number of cycles later, so a consumer that does not wait on it reads a
/// stale register (e.g. a zero kernel-parameter). It needs a scoreboard like LDG.
fn isVariableLatency(opcode: u32) bool {
    // LDG, S2R, LDC, plus the graphics attribute fetch (ALD) and fragment input
    // interpolation (IPA): their results land an unknown number of cycles after
    // issue, so a consumer (the AST that stores them, the MOV to a color register)
    // must wait on the scoreboard or it reads a stale/zero register.
    return opcode == 0x981 or opcode == 0x919 or opcode == 0xb82 or // LDG, S2R, LDC
        opcode == 0x321 or opcode == 0x326 or // ALD, IPA
        // SHFL (all-immediate quad form 0xf89, the lane/c register forms 0x389/0x589/
        // 0x989): on Blackwell sm120 SHFL is DECOUPLED (NAK sm120_instr_latencies:
        // Op::Shfl => DecoupledAgu), i.e. variable-latency. Its result lands an unknown
        // number of cycles after issue, so a consumer (FSWZADD, i2f) MUST wait on its
        // scoreboard or it reads the register before the cross-lane shuffle completes -
        // exactly the garbage-derivative wall (a freshly-IPA'd varying shuffled through a
        // SHFL whose result was consumed too early).
        opcode == 0xf89 or opcode == 0x389 or opcode == 0x589 or opcode == 0x989 or
        isTexResult(opcode) or // TEX / TLD4 (the GPU texture sample or gather, RGBA result)
        // MUFU (the multifunction/special-function unit: RCP/RSQ/SQRT/SIN/COS/EX2/LG2,
        // opcode 0x108): on Blackwell sm120 MUFU is DECOUPLED (NAK sm120_instr_latencies:
        // Op::MuFu(_) => Decoupled), i.e. variable-latency on the SFU pipe. Its result
        // lands an unknown number of cycles after issue, so a consumer that does not wait
        // on its scoreboard reads the destination register STALE. `normalize(v)` lowers to
        // MUFU.SQRT then MUFU.RCP then a chain of FMULs - all dependent - so without a
        // scoreboard the RCP reads the SQRT result early and every FMUL reads the RCP
        // early: the normalized vector comes out garbage AND varies run-to-run (the dFdx
        // wall's final symptom: correct per-axis derivatives but a noisy normal). The SFU
        // ops the deriv FS uses (normalize) all go through MUFU, so a scoreboard here is
        // the fix. (NAK's default delay for a coupled op would NOT cover the SFU latency.)
        opcode == encode.MUFU_OPCODE or
        // LDS (0x984), the shared-memory load, and BAR (0xb1d), the workgroup barrier. NAK
        // sm120_instr_latencies classes Op::Ld as DecoupledAgu for EVERY memory space, not
        // only the global one, and Op::Bar the same way. So an LDS result lands an unknown
        // number of cycles after issue exactly as an LDG result does. Without a scoreboard
        // here a consumer reads the destination register STALE, which on a staged shared
        // tile means each thread reads whatever the register held before the tile load.
        opcode == 0x984 or opcode == 0xb1d or
        // The atomics that GIVE BACK the old value: ATOMG (0x9a8), ATOMS (0x98c) and the
        // two compare-and-swap forms (0x3a9 global, 0x38d shared). NAK
        // sm120_instr_latencies classes `Op::Atom(_) => DecoupledAgu`, the same class as a
        // load, so the old value lands an unknown number of cycles after issue. A
        // compare-and-swap loop reads that value to decide whether to retry, so without a
        // scoreboard it tests a STALE register and either spins forever or accepts a swap
        // that never happened. RED (0x98e) is deliberately absent: it writes no register,
        // so there is nothing for a consumer to wait on.
        opcode == encode.ATOMG_OPCODE or opcode == encode.ATOMS_OPCODE or
        opcode == encode.ATOMG_CAS_OPCODE or opcode == encode.ATOMS_CAS_OPCODE;
}

/// Whether `opcode` writes a destination GPR at bits 16..23 (so the scheduler can
/// associate a scoreboard with it). Stores and control flow do not.
fn writesDst(opcode: u32) bool {
    return switch (opcode) {
        0x986, 0x947, 0x94d => false, // STG, BRA, EXIT
        // STS (0x988), the shared store, and BAR (0xb1d), the workgroup barrier. Neither
        // writes a GPR, and both leave bits 16..23 at zero. Reading that as a write to R0
        // is worse than a wasted entry: if R0 has an in-flight producer, the scheduler
        // makes the store or the barrier wait on that scoreboard and CLEARS its tag, so
        // the instruction that really consumes R0 never waits and reads it stale. STG is
        // excluded just above for this same reason.
        0x988, 0xb1d => false,
        // Convergence barriers operate on the Bar register file, not GPRs: BCLEAR
        // (0x355), BSSY (0x945), BSYNC (0x941). Their bits 16..23 encode a barrier
        // register (or RZ), NOT a GPR dst - excluding them keeps the GPR scoreboard
        // map from being polluted by a phantom "R0/R1" write.
        0x355, 0x945, 0x941 => false,
        // RED (0x98e), the global atomic reduction. It applies the operation to memory and
        // gives back nothing, so it is a store, not a load. The encoder writes RZ into bits
        // 16..24, following NAK's set_dst(&Dst::None), and the RZ guard below would already
        // skip it, but naming it here does not depend on that: if the destination field ever
        // read as R0 the scheduler would record a phantom write, steal a live scoreboard and
        // clear it, and the real consumer of R0 would read stale. That is the STS and BAR
        // bug of 6e09607. The atomics that DO give back a value (ATOMG, ATOMS and the two
        // compare-and-swap forms) write a real destination at 16..24 and stay out of this
        // list.
        encode.RED_OPCODE => false,
        else => true,
    };
}

/// Whether `opcode` reads a source GPR in the field at bit `pos` (24, 32 or 64).
///
/// The `else` arm is the conservative ALU superset the scheduler started with: srcA at 24,
/// srcB at 32 but only in the register source form, srcC at 64. A memory or atomic op is not
/// an ALU op, so that rule is wrong for it in BOTH directions, and both directions corrupt
/// the scoreboard state:
///
///   - Too FEW reads. A store's data register sits at bit 32, but every store opcode carries
///     form 4, not form 1, so the ALU rule skips it. The store then issues with no wait on
///     the in-flight load that fills its data register and writes a STALE value to memory.
///   - Too MANY reads. LDS leaves bits 64..71 at zero, which the ALU rule reads as R0. If R0
///     has a live producer, the load waits on that scoreboard and CLEARS its tag, so the
///     instruction that really consumes R0 never waits and reads stale. That is the
///     phantom-read half of the STS and BAR bug fixed in 6e09607, on the source side.
///
/// So each memory and atomic opcode names its own source fields instead.
fn readsSrc(opcode: u32, form: u32, pos: usize) bool {
    return switch (opcode) {
        // Convergence barriers (BCLEAR/BSSY/BSYNC) and BAR read no GPR at all: their
        // bit-24/16 fields hold a Bar register, and BAR has no operand.
        0x355, 0x945, 0x941, 0xb1d => false,
        // LDG and LDS: the address at 24, nothing else. LDG holds URZ in bits 64..71 and LDS
        // leaves them zero, so neither field is a GPR source.
        0x981, 0x984 => pos == 24,
        // STG, STS and RED: the address at 24 and the data at 32.
        0x986, 0x988, encode.RED_OPCODE => pos == 24 or pos == 32,
        // ATOMG and ATOMS: the address at 24 and the data at 32. Bits 64..71 hold URZ.
        encode.ATOMG_OPCODE, encode.ATOMS_OPCODE => pos == 24 or pos == 32,
        // Compare-and-swap: the address at 24, the compare operand at 32 and the swap data
        // at 64. All three are real registers, so all three must be waited on.
        encode.ATOMG_CAS_OPCODE, encode.ATOMS_CAS_OPCODE => true,
        else => pos != 32 or form == 1 or isTexResult(opcode),
    };
}

/// Whether `opcode` addresses memory through a 64-bit GPR PAIR at bit 24, so it reads
/// `addr + 1` as well as `addr`. Every GLOBAL access does. The shared ones take a 32-bit
/// offset into the CTA window in ONE register, so reading `addr + 1` for them would wait on
/// and free an unrelated register's scoreboard.
fn readsAddrPair(opcode: u32) bool {
    return opcode == 0x981 or opcode == 0x986 or // LDG, STG
        opcode == encode.ATOMG_OPCODE or opcode == encode.RED_OPCODE or
        opcode == encode.ATOMG_CAS_OPCODE;
}

/// How many consecutive destination registers one instruction writes under its single write
/// barrier. Every consumer of ANY register in the block must wait on that barrier, so the
/// producer has to tag the whole block, not only the first register.
///
/// A TEX writes one register per channel of its channel mask (bits 72..76). A B64 load fills
/// (dst, dst+1) and a B128 load fills (dst .. dst+3), from the memory type at bits 73..76. A
/// 64-bit atomic gives back a value in (dst, dst+1), from the WIDER 4-bit atomic type at bits
/// 73..77. Everything else writes one register.
fn dstSpan(opcode: u32, inst: Inst) u32 {
    if (isTexResult(opcode)) return @popCount(getField(inst, 72, 4));
    if (opcode == 0x981 or opcode == 0x984) return switch (getField(inst, 73, 3)) { // LDG, LDS
        @intFromEnum(encode.MemType.b64) => 2,
        @intFromEnum(encode.MemType.b128) => 4,
        else => 1,
    };
    if (opcode == encode.ATOMG_OPCODE or opcode == encode.ATOMS_OPCODE or
        opcode == encode.ATOMG_CAS_OPCODE or opcode == encode.ATOMS_CAS_OPCODE)
        return switch (getField(inst, 73, 4)) {
            @intFromEnum(encode.AtomType.u64), @intFromEnum(encode.AtomType.i64) => 2,
            else => 1,
        };
    return 1;
}

fn getField(inst: Inst, comptime lo: usize, comptime width: usize) u32 {
    // The fields this pass touches never span a 32-bit word boundary.
    const word = lo / 32;
    const off: u5 = @intCast(lo % 32);
    const mask: u32 = (@as(u32, 1) << width) - 1;
    return (inst[word] >> off) & mask;
}

fn setField(inst: *Inst, comptime lo: usize, comptime width: usize, val: u32) void {
    const word = lo / 32;
    const off: u5 = @intCast(lo % 32);
    const mask: u32 = ((@as(u32, 1) << width) - 1) << off;
    inst[word] = (inst[word] & ~mask) | ((val << off) & mask);
}

const num_scoreboards = 6;

/// Assign scoreboards and wait masks across `insts` so every variable-latency
/// result is awaited before it is consumed. Rewrites the scheduling control fields
/// in place. `block_starts`, if given, are the instruction indices at which a basic
/// block begins (a branch target). The scheduler drains all scoreboards at each so
/// the LINEAR scoreboard model stays sound across control flow (a producer in one
/// path must not have its barrier waited-on along a different path).
pub fn schedule(insts: []Inst) void {
    scheduleBlocks(insts, &.{});
}

pub fn scheduleBlocks(insts: []Inst, block_starts: []const usize) void {
    // The ENTRY block start (the smallest one - the isel emits a straight-line PROLOGUE, e.g. the
    // IPA varying-fetches + the LDC uniform-block-base loads, then records block 0 starting AFTER
    // it) has a SINGLE linear predecessor (the prologue), not multiple branch predecessors. Draining
    // there is not only unnecessary, it is HARMFUL: the drain @memsets the scoreboard map, dropping
    // the prologue LDC/IPA tags, so a later consumer (the uniform LDG) emits NO wait and relies on
    // the drain's 0x3f wait having landed the LDC - which a cold constant-cache miss at high
    // occupancy (a tall render target lighting up more TPCs) does NOT satisfy, so the LDG reads a
    // stale (zero) base and faults Xid 31 @ 0x0. Skip the entry; real branch targets still drain.
    var entry_start: usize = std.math.maxInt(usize);
    for (block_starts) |b| entry_start = @min(entry_start, b);
    // scoreboard_of[reg] = scoreboard index + 1 (0 = the register has no in-flight
    // variable-latency producer).
    var scoreboard_of = [_]u8{0} ** 256;
    var free_mask: u8 = (1 << num_scoreboards) - 1; // scoreboards 0..5 free

    for (insts, 0..) |*inst, idx| {
        const opcode = getField(inst.*, 0, 12);

        // At a basic-block boundary (a branch target, idx > 0), drain every in-flight
        // scoreboard before issuing the block's first instruction. The scheduler walks
        // the instruction stream LINEARLY, but control flow means the producer that
        // tagged a scoreboard may belong to a DIFFERENT predecessor path than the one
        // actually taken at run time - so a cross-block wait could reference a barrier
        // never set on the live path (a hang or a missed wait). Draining at the boundary
        // makes every variable-latency result land before the block runs, so the linear
        // model is exact. (A shader with multiple branch-helpers - e.g. vkcube's per-
        // channel sRGB curve over a derivative-lit textured fragment - exposes this. A
        // single straight-line body never crosses a boundary so this is a no-op there.)
        if (idx > 0 and free_mask != (1 << num_scoreboards) - 1) {
            for (block_starts) |bs| if (bs == idx and bs != entry_start) {
                setField(inst, 116, 6, getField(inst.*, 116, 6) | ((1 << num_scoreboards) - 1));
                @memset(&scoreboard_of, 0);
                free_mask = (1 << num_scoreboards) - 1;
                break;
            };
        }

        // Which scoreboards must this instruction wait on? Conservatively, any of
        // its source-register fields whose register has an in-flight producer.
        //
        // The bit-32 field is srcB only in the register source form (form bits 9..11
        // == 1). In the 32-bit immediate form (form == 4, e.g. MOV imm) those bits hold
        // the immediate rather than a register. Reading them as a register both adds a
        // spurious wait and, worse, frees that scoreboard, since the freeing below treats
        // the false read as a real consume. The immediate's stray register number then
        // collides with a live load's destination and steals its scoreboard, so the real
        // consumer never waits and reads a stale value, like a UBO member whose LDG had
        // not landed. Skip bit 32 unless the op is in register-source form.
        const form = getField(inst.*, 9, 3);
        // TEX is not an ALU op and has no source-form bits, but it reads a coordinate
        // register at bit 24 and the bindless texture handle register at bit 32. The
        // handle comes from an LDC with variable latency, so TEX must wait on its
        // scoreboard, meaning its bit-32 source is a real register read.
        //
        // The memory ops, the atomics and the barriers each name their own source fields.
        // See `readsSrc` for why the ALU rule is wrong for them in both directions.
        const is_barrier = opcode == 0x355 or opcode == 0x945 or opcode == 0x941 or opcode == 0xb1d;
        var wait: u32 = 0;
        inline for (.{ 24, 32, 64 }) |pos| {
            if (readsSrc(opcode, form, pos)) {
                const reg = getField(inst.*, pos, 8);
                if (reg != RZ and scoreboard_of[reg] != 0) wait |= @as(u32, 1) << @intCast(scoreboard_of[reg] - 1);
            }
        }
        // A global load, store or atomic addresses memory through a 64-bit REGISTER PAIR at
        // the bit-24 source: it reads both `addr` (lo) and `addr+1` (hi). The hi half is
        // not an explicit source field, so wait on its in-flight producer too -
        // otherwise the load issues with a stale high address dword (e.g. a UBO base
        // pointer whose hi LDC has not landed), reading garbage and faulting the GR
        // front-end (an "illegal instruction encoding" scoreboard hazard on Blackwell).
        if (readsAddrPair(opcode)) {
            const addr_lo = getField(inst.*, 24, 8);
            if (addr_lo != RZ) {
                const addr_hi = addr_lo + 1;
                if (addr_hi != RZ and scoreboard_of[addr_hi] != 0)
                    wait |= @as(u32, 1) << @intCast(scoreboard_of[addr_hi] - 1);
            }
        }
        if (wait != 0) {
            setField(inst, 116, 6, getField(inst.*, 116, 6) | wait);
            // Freeing: clear the scoreboard tag from the registers THIS instruction read
            // (the ones the wait covers), then return a scoreboard to the free pool only
            // once none of its registers remain tagged. A multi-register producer (TEX,
            // which writes a 4-register RGBA result block under ONE write barrier) keeps
            // its scoreboard tagged on the result registers NOT yet consumed, so each
            // later reload of an un-read channel ALSO waits on the same scoreboard.
            // For a single-register producer (LDG/S2R/LDC/ALD/IPA) the one read register
            // clears and the scoreboard frees immediately - identical to the old free-on-
            // first-wait behavior.
            clearReadRegs(inst.*, opcode, form, &scoreboard_of, wait);
            var sb: u3 = 0;
            while (sb < num_scoreboards) : (sb += 1) {
                if (wait & (@as(u32, 1) << sb) != 0) {
                    var still_used = false;
                    for (scoreboard_of) |s| if (s == @as(u8, sb) + 1) {
                        still_used = true;
                    };
                    if (!still_used) free_mask |= @as(u8, 1) << sb;
                }
            }
        }

        // WRITE-AFTER-WRITE hazard: if this instruction WRITES a register that still has
        // an in-flight variable-latency producer (a decoupled LDG/LDC/etc. whose result
        // has not landed), it must wait on that producer's scoreboard FIRST. Otherwise the
        // two writes race: the linear-scan allocator legitimately reuses a register for a
        // synchronous write (e.g. MOV.imm) once the prior value's SSA live range ends, but
        // the prior value was loaded by a DECOUPLED LDG that completes an unknown number of
        // cycles later - so the async load lands AFTER the synchronous write and clobbers
        // it. Symptom (the EGL/GLES uniform-block bug): the GLSL front end eagerly loads
        // every default-block float (incl. unused ones like LightSourcePosition.w) into a
        // small register pool. A dead/unread LDG's destination is reused by a later
        // MOV.imm address offset, the dead load lands late and overwrites the offset, and
        // the next member's LDG reads from a garbage address -> a uniform reads the wrong
        // value (MaterialColor came back (1,1,0)). The RAW path above only protects READS.
        // This protects WRITES. Wait on the scoreboard, then clear its tag so the register
        // is reusable. (DrainAll's boundary drain is per-block, not per-register.)
        if (writesDst(opcode) and !is_barrier) {
            const wdst = getField(inst.*, 16, 8);
            if (wdst != RZ and scoreboard_of[wdst] != 0) {
                const sb_idx = scoreboard_of[wdst] - 1;
                const wbit: u32 = @as(u32, 1) << @intCast(sb_idx);
                setField(inst, 116, 6, getField(inst.*, 116, 6) | wbit);
                // Clear this register's tag. Free the scoreboard if no other register
                // (a multi-register TEX block) still holds it.
                scoreboard_of[wdst] = 0;
                var still_used = false;
                for (scoreboard_of) |s| if (s == sb_idx + 1) {
                    still_used = true;
                };
                if (!still_used) free_mask |= @as(u8, 1) << @intCast(sb_idx);
            }
        }

        // A variable-latency producer claims a scoreboard for its destination.
        if (isVariableLatency(opcode) and writesDst(opcode)) {
            const dst = getField(inst.*, 16, 8);
            if (dst != RZ) {
                if (free_mask == 0) drainAll(inst, &scoreboard_of, &free_mask);
                const sb: u3 = @intCast(@ctz(free_mask));
                free_mask &= ~(@as(u8, 1) << sb);
                setField(inst, 110, 3, sb); // write barrier
                // TEX writes a 4-register RGBA result BLOCK (dst..dst+3), all gated by
                // the one write barrier. Every consumer of ANY of the four must wait on
                // the scoreboard, so claim it for all four registers - not just dst. With
                // only dst tagged, the scoreboard freed after the first reload (R) waits,
                // so the G/B/A reloads would race the still-in-flight TEX and read stale
                // registers (a sporadic wrong channel). LDG/S2R/LDC/ALD/IPA write a single
                // register, so they tag only dst. A DEPTH-COMPARE TEX (z_cmpr, sampler2DShadow) writes a
                // SINGLE scalar (channel_mask = R only), so span it by the channel-mask popcount (bits
                // 72..76): 4 for an RGBA sample/gather/fetch, 1 for the shadow scalar - tagging only the
                // registers the TEX actually writes, so a later ALU write to dst+1..dst+3 is not
                // needlessly gated on the shadow scoreboard. A B64 or B128 load and a 64-bit
                // atomic write a register block for the same reason, so `dstSpan` covers
                // all of them.
                const span: u32 = dstSpan(opcode, inst.*);
                var k: u32 = 0;
                while (k < span and dst + k < RZ) : (k += 1) {
                    scoreboard_of[dst + k] = @as(u8, sb) + 1;
                }
            }
        }
    }
}

/// Clear the in-flight-scoreboard tag from each register THIS instruction read whose
/// producer is among `wait`. Mirrors the source-register set the wait computation
/// scans (srcA@24, srcB@32 in register form or for TEX, srcC@64, plus the LDG/STG
/// 64-bit address pair's high half). A register that was NOT read keeps its tag, so a
/// later instruction reading it re-waits (the multi-register TEX-result case).
fn clearReadRegs(inst: Inst, opcode: u32, form: u32, scoreboard_of: *[256]u8, wait: u32) void {
    inline for (.{ 24, 32, 64 }) |pos| {
        if (readsSrc(opcode, form, pos)) {
            const reg = getField(inst, pos, 8);
            if (reg != RZ and scoreboard_of[reg] != 0 and
                (wait & (@as(u32, 1) << @intCast(scoreboard_of[reg] - 1))) != 0)
                scoreboard_of[reg] = 0;
        }
    }
    if (readsAddrPair(opcode)) { // a global access: clear the address-hi half too
        const addr_lo = getField(inst, 24, 8);
        if (addr_lo != RZ) {
            const addr_hi = addr_lo + 1;
            if (addr_hi != RZ and scoreboard_of[addr_hi] != 0 and
                (wait & (@as(u32, 1) << @intCast(scoreboard_of[addr_hi] - 1))) != 0)
                scoreboard_of[addr_hi] = 0;
        }
    }
}

/// When all six scoreboards are in flight, make this instruction wait on every one
/// (a full drain) so a scoreboard can be reused.
fn drainAll(inst: *Inst, scoreboard_of: *[256]u8, free_mask: *u8) void {
    setField(inst, 116, 6, (1 << num_scoreboards) - 1);
    @memset(scoreboard_of, 0);
    free_mask.* = (1 << num_scoreboards) - 1;
}

test "a load's consumer waits on the load's scoreboard" {
    // S2R R4, IMAD R5, R4, R6, RZ (reads the S2R result), LDG R7, [R8:R9],
    // IMAD R10, R7, R6, RZ (reads the LDG result), EXIT.
    var insts = [_]Inst{
        encode.s2r(4, encode.SR_TID_X, .{}),
        encode.imad(5, 4, 6, RZ, .{}),
        encode.ldgU32(7, 8, .{}),
        encode.imad(10, 7, 6, RZ, .{}),
        encode.exit(.{}),
    };
    schedule(&insts);

    // S2R (inst 0) gets a write barrier. The IMAD reading R4 (inst 1) waits on it.
    const s2r_bar = getField(insts[0], 110, 3);
    try std.testing.expect(s2r_bar < 6); // a real scoreboard, not 7 (none)
    try std.testing.expect((getField(insts[1], 116, 6) & (@as(u32, 1) << @intCast(s2r_bar))) != 0);

    // LDG (inst 2) gets a write barrier. The IMAD reading R7 (inst 3) waits on it.
    const ldg_bar = getField(insts[2], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expect((getField(insts[3], 116, 6) & (@as(u32, 1) << @intCast(ldg_bar))) != 0);

    // The S2R scoreboard was freed at inst 1 and reused for the LDG.
    try std.testing.expectEqual(s2r_bar, ldg_bar);
}

test "an independent instruction adds no wait" {
    // S2R R4, IADD3 R5, R6, R7 (independent of R4), EXIT.
    var insts = [_]Inst{
        encode.s2r(4, encode.SR_TID_X, .{}),
        encode.iadd3(5, 6, 7, .{}),
        encode.exit(.{}),
    };
    schedule(&insts);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // no wait
}

test "fixed-latency instructions get no write barrier" {
    var insts = [_]Inst{
        encode.iadd3(4, 5, 6, .{}),
        encode.imad(7, 4, 8, RZ, .{}),
        encode.exit(.{}),
    };
    schedule(&insts);
    try std.testing.expectEqual(@as(u32, 7), getField(insts[0], 110, 3)); // none (7)
    try std.testing.expectEqual(@as(u32, 7), getField(insts[1], 110, 3));
}

test "an LDG consumer FAR from its LDC-address producer still waits (uniform-block base)" {
    // Reproduce the glmark2 bump-normals fault: a mat4 uniform's base address is loaded via LDC
    // into R6:R7, then MANY unrelated MOV.imm (dead constant materialization) intervene, then an
    // LDG uses R6:R7 as the address. The LDG MUST wait on the LDC scoreboards - else on a cold
    // constant-cache miss (high occupancy / a tall render target using more TPCs) the LDG reads a
    // stale (zero) base and faults Xid 31 @ 0x0 (GPCCLIENT_T1). See [[prism-glmark2-perf-cliff]].
    var insts: [28]Inst = undefined;
    insts[0] = encode.ipa(4, 0x80, .{}); // fragment-input interpolation (variable latency)
    insts[1] = encode.ipa(5, 0x84, .{});
    insts[2] = encode.ldc(6, 0, 0x148, .{}); // R6 = uniform-block base lo
    insts[3] = encode.ldc(7, 0, 0x14c, .{}); // R7 = base hi
    for (4..26) |i| insts[i] = encode.movImm(8, @intCast(i), .{}); // 22 dead MOV.imm R8
    insts[26] = encode.ldgU32(8, 6, .{}); // LDG R8 <- [R6:R7]
    insts[27] = encode.exit(.{});
    // block 0 STARTS at index 4 (the IPA/LDC prologue is emitted before it) - the isel's real
    // shape. The entry drain must be SKIPPED here (single linear predecessor) so the LDG still
    // waits on the LDC scoreboards rather than relying on a dropped drain wait.
    scheduleBlocks(&insts, &.{4});

    const ldc_lo_bar = getField(insts[2], 110, 3);
    const ldc_hi_bar = getField(insts[3], 110, 3);
    try std.testing.expect(ldc_lo_bar < 6 and ldc_hi_bar < 6); // both got a real scoreboard
    const ldg_wait = getField(insts[26], 116, 6);
    // The LDG reading R6:R7 must wait on BOTH LDC scoreboards.
    try std.testing.expect((ldg_wait & (@as(u32, 1) << @intCast(ldc_lo_bar))) != 0);
    try std.testing.expect((ldg_wait & (@as(u32, 1) << @intCast(ldc_hi_bar))) != 0);
}

test "an LDS result gets a scoreboard and its consumer waits on it" {
    // Regression: isVariableLatency listed LDG but not LDS, so a shared-memory load got no
    // write barrier and its consumer read the destination register STALE. NAK
    // sm120_instr_latencies classes Op::Ld as DecoupledAgu for EVERY memory space, not only
    // the global one. On a staged shared tile the symptom is each thread reading whatever the
    // register held before the tile load.
    var insts: [3]Inst = undefined;
    insts[0] = encode.ldsU32(4, 2, .{}); // R4 <- shared[R2]
    insts[1] = encode.iadd3(5, 4, 4, .{}); // consumes R4
    insts[2] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const lds_bar = getField(insts[0], 110, 3);
    try std.testing.expect(lds_bar < 6); // a real scoreboard, not 7 = none
    const consumer_wait = getField(insts[1], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(lds_bar))) != 0);
}

test "an STS does not claim a phantom write to R0" {
    // Regression: writesDst returned true for STS, whose bits 16..23 are zero, so the
    // scheduler recorded a write to R0. When R0 had a live in-flight producer the store waited
    // on that scoreboard and CLEARED its tag, so the instruction that really consumed R0 never
    // waited and read it stale. STG is excluded for this same reason.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.stsU32(6, 8, .{}); // shared[R6] = R8, touches no GPR dst
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    // The consumer must still wait on the LDG. If the STS stole and cleared the tag, this is 0.
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "a BAR neither claims a destination register nor reads three phantom R0 sources" {
    // Regression: BAR (0xb1d) leaves bits 16..23, 24, 32 and 64 at zero. Without the writesDst
    // and is_barrier exclusions the scheduler read those as a write to R0 plus three reads of
    // R0, which both freed a live scoreboard and added a spurious wait.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.barSync(.{}); // the workgroup barrier
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "an STG waits on the in-flight producer of its DATA register" {
    // Regression: every store opcode carries source form 4, not form 1, so the ALU source
    // rule skipped the bit-32 field and the store issued with no wait on the load filling
    // its data register. The store then wrote a STALE value to memory, with no diagnostic.
    var insts: [3]Inst = undefined;
    insts[0] = encode.ldgU32(8, 2, .{}); // R8 <- global, variable latency
    insts[1] = encode.stgU32(4, 8, .{}); // global[R4:R5] = R8
    insts[2] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    const store_wait = getField(insts[1], 116, 6);
    try std.testing.expect((store_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "an LDS does not read a phantom R0 at bit 64" {
    // Regression: LDS leaves bits 64..71 at zero, and the ALU source rule read that as a
    // read of R0. With a live producer for R0 the load waited on its scoreboard and CLEARED
    // the tag, so the instruction that really consumes R0 never waited and read it stale.
    // Same shape as the STS and BAR phantom-write bug of 6e09607, on the source side.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.ldsU32(4, 6, .{}); // R4 <- shared[R6], touches no other GPR
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the LDS waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "a 64-bit load tags its WHOLE destination register pair" {
    // A B64 load fills (dst, dst+1) under one write barrier. Tagging only dst leaves a
    // consumer of dst+1 with no wait, so it reads the high half before the load lands.
    var insts: [3]Inst = undefined;
    insts[0] = encode.ldg(6, 4, .b64, .{}); // R6:R7 <- global[R4:R5]
    insts[1] = encode.iadd3(9, 7, 7, .{}); // consumes the HIGH half only
    insts[2] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    const consumer_wait = getField(insts[1], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "an ATOMG result gets a scoreboard and its consumer waits on it" {
    // NAK sm120_instr_latencies classes Op::Atom as DecoupledAgu, the same class as a load,
    // so the old value the atomic gives back lands an unknown number of cycles after issue.
    // A consumer that does not wait reads the destination register STALE.
    var insts: [3]Inst = undefined;
    insts[0] = encode.atomg(6, 4, 8, .add, .u32, .{}); // R6 <- old value at global[R4:R5]
    insts[1] = encode.iadd3(9, 6, 6, .{}); // consumes the returned old value
    insts[2] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const atom_bar = getField(insts[0], 110, 3);
    try std.testing.expect(atom_bar < 6); // a real scoreboard, not 7 = none
    const consumer_wait = getField(insts[1], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(atom_bar))) != 0);
}

test "a global atomic waits on the producers of its ADDRESS PAIR and its DATA register" {
    // The address is a 64-bit pair (R4, R5), and only its low half is an explicit source
    // field. The data register sits at bit 32 under source form 4, which the ALU rule skips.
    // Missing either wait sends the atomic at a stale address or with stale data, and an
    // atomic writes memory, so both are silent corruption on real silicon.
    var insts: [5]Inst = undefined;
    insts[0] = encode.ldc(4, 0, 0x10, .{}); // R4 = address lo, variable latency
    insts[1] = encode.ldc(5, 0, 0x14, .{}); // R5 = address hi
    insts[2] = encode.ldgU32(8, 2, .{}); // R8 = the amount to add
    insts[3] = encode.atomg(6, 4, 8, .add, .u32, .{});
    insts[4] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const lo_bar = getField(insts[0], 110, 3);
    const hi_bar = getField(insts[1], 110, 3);
    const data_bar = getField(insts[2], 110, 3);
    try std.testing.expect(lo_bar < 6 and hi_bar < 6 and data_bar < 6);
    const wait = getField(insts[3], 116, 6);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(lo_bar))) != 0);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(hi_bar))) != 0);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(data_bar))) != 0);
}

test "a RED claims no scoreboard but still waits on its data producer" {
    // RED gives back no value, so it must NOT take a write barrier: a scoreboard nobody ever
    // waits on is one of six lost for the rest of the block. It DOES read memory operands,
    // so it must still wait on the load that fills its data register, or it reduces a stale
    // value into memory. Its destination field holds RZ (NAK's set_dst(&Dst::None)), so the
    // scheduler must not record a write either.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(8, 2, .{}); // R8 = the amount to add, variable latency
    insts[1] = encode.redg(4, 8, .add, .u32, .{}); // global[R4:R5] += R8
    insts[2] = encode.iadd3(9, 8, 8, .{}); // a later reader of R8
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    try std.testing.expectEqual(@as(u32, 7), getField(insts[1], 110, 3)); // no write barrier
    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    const red_wait = getField(insts[1], 116, 6);
    try std.testing.expect((red_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
    // The RED read R8, so the scoreboard is spent and the later reader adds no wait.
    try std.testing.expectEqual(@as(u32, 0), getField(insts[2], 116, 6));
}

test "a compare-and-swap waits on its compare operand and on its swap data at bit 64" {
    // The CAS forms put the compare operand at bit 32 and the swap data at bit 64. A stale
    // compare operand makes the swap take or miss when it should not, and stale swap data
    // writes the wrong value into memory.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(8, 2, .{}); // R8 = the expected value
    insts[1] = encode.ldsU32(10, 12, .{}); // R10 = the value to swap in
    insts[2] = encode.atomgCas(6, 4, 8, 10, .u32, .{});
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const cmp_bar = getField(insts[0], 110, 3);
    const data_bar = getField(insts[1], 110, 3);
    try std.testing.expect(cmp_bar < 6 and data_bar < 6);
    const wait = getField(insts[2], 116, 6);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(cmp_bar))) != 0);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(data_bar))) != 0);
    // The result is variable-latency, so the CAS itself takes a write barrier.
    try std.testing.expect(getField(insts[2], 110, 3) < 6);
}

test "a shared atomic reads ONE address register, not a pair" {
    // A shared address is a 32-bit offset into the CTA window. Reading `addr + 1` as the high
    // half of a pair would wait on an unrelated register's scoreboard and CLEAR it, so the
    // instruction that really consumes that register would read it stale.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(5, 2, .{}); // R5 is unrelated to the shared address in R4
    insts[1] = encode.atoms(6, 4, 8, .add, .u32, .{}); // R6 <- old value at shared[R4]
    insts[2] = encode.iadd3(9, 5, 5, .{}); // the REAL consumer of R5
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the atomic waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
    // The shared atomic gives back a value, so it is variable-latency like ATOMG.
    try std.testing.expect(getField(insts[1], 110, 3) < 6);
}
