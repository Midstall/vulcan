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
//! Limits: register reads are a conservative superset (the three ALU source
//! fields), which only ever adds an unnecessary wait, never misses a real one. Read
//! barriers (protecting a variable-latency op's source registers from being
//! overwritten) are unnecessary here because the naive allocator never reuses a
//! register. Stall delays are left as the isel set them.

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
        opcode == encode.MUFU_OPCODE;
}

/// Whether `opcode` writes a destination GPR at bits 16..23 (so the scheduler can
/// associate a scoreboard with it). Stores and control flow do not.
fn writesDst(opcode: u32) bool {
    return switch (opcode) {
        0x986, 0x947, 0x94d => false, // STG, BRA, EXIT
        // Convergence barriers operate on the Bar register file, not GPRs: BCLEAR
        // (0x355), BSSY (0x945), BSYNC (0x941). Their bits 16..23 encode a barrier
        // register (or RZ), NOT a GPR dst - excluding them keeps the GPR scoreboard
        // map from being polluted by a phantom "R0/R1" write.
        0x355, 0x945, 0x941 => false,
        else => true,
    };
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
        const is_tex = isTexResult(opcode);
        // Convergence barriers (BCLEAR/BSSY/BSYNC) read NO GPRs - their bit-24/16 fields
        // encode a Bar register, not a source GPR. Skip the GPR source-wait/free scan so
        // a barrier reg value (e.g. B0..B15) is not misread as "R0..R15" and made to
        // spuriously wait on or free a live scoreboard.
        const is_barrier = opcode == 0x355 or opcode == 0x945 or opcode == 0x941;
        var wait: u32 = 0;
        if (!is_barrier) inline for (.{ 24, 32, 64 }) |pos| {
            if (pos != 32 or form == 1 or is_tex) {
                const reg = getField(inst.*, pos, 8);
                if (reg != RZ and scoreboard_of[reg] != 0) wait |= @as(u32, 1) << @intCast(scoreboard_of[reg] - 1);
            }
        };
        // A global load/store addresses memory through a 64-bit REGISTER PAIR at the
        // bit-24 source: it reads both `addr` (lo) and `addr+1` (hi). The hi half is
        // not an explicit source field, so wait on its in-flight producer too -
        // otherwise the load issues with a stale high address dword (e.g. a UBO base
        // pointer whose hi LDC has not landed), reading garbage and faulting the GR
        // front-end (an "illegal instruction encoding" scoreboard hazard on Blackwell).
        if (opcode == 0x981 or opcode == 0x986) { // LDG, STG
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
                // needlessly gated on the shadow scoreboard.
                const span: u32 = if (isTexResult(opcode)) @popCount(getField(inst.*, 72, 4)) else 1;
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
    const is_tex = isTexResult(opcode);
    inline for (.{ 24, 32, 64 }) |pos| {
        if (pos != 32 or form == 1 or is_tex) {
            const reg = getField(inst, pos, 8);
            if (reg != RZ and scoreboard_of[reg] != 0 and
                (wait & (@as(u32, 1) << @intCast(scoreboard_of[reg] - 1))) != 0)
                scoreboard_of[reg] = 0;
        }
    }
    if (opcode == 0x981 or opcode == 0x986) { // LDG, STG: clear the address-hi half too
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
