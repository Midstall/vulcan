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
//! (the three ALU source fields), which only ever adds an unnecessary wait. For every
//! other opcode the superset is WRONG in both directions, because those opcodes put other
//! things in the ALU source fields: an attribute address, a branch offset, a lookup table,
//! a second destination, or nothing at all. So each of them names its own sources.
//!
//! `readsSrc` names WHICH field is a register. `srcSpan` says HOW MANY consecutive
//! registers that field covers, because a source field often names only the FIRST register
//! of a run: a 64-bit address pair, a wide store's data block, a texture coordinate block,
//! or one lane's slice of a matrix fragment. `dstSpan` is the same idea on the destination
//! side.
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
        opcode == encode.ATOMG_CAS_OPCODE or opcode == encode.ATOMS_CAS_OPCODE or
        // The tensor-core ops that need a scoreboard. NAK sm120_instr_latencies.rs is exact
        // about which ones do: `SM120Latency::needs_scoreboards` returns true for the
        // latency classes Dmma, Hmma, RedirectedFp64, Branch, Decoupled and DecoupledAgu.
        //   - HMMA (0x23c) has class `Hmma`, which IS in that list.
        //   - LDSM (0x83b) and MOVM (0x23a) both have class `DecoupledAgu`, the same class
        //     as an LDS, so their results land an unknown number of cycles after issue.
        // Without a scoreboard the HMMA that consumes an LDSM fragment reads the whole
        // register run STALE, which means each lane multiplies whatever its registers held
        // before the tile load. IMMA (0x237) is DELIBERATELY absent: its class is `Imma`,
        // which is NOT in that list, so it is a COUPLED (fixed-latency) op that the stall
        // delay covers. A write barrier on an op the hardware never signals would hang the
        // consumer forever, which is worse than the missing wait. See `dstSpan` for the
        // IMMA arm that this exclusion does not remove.
        opcode == encode.HMMA_OPCODE or opcode == encode.LDSM_OPCODE or
        opcode == encode.MOVM_OPCODE;
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
        // AST (0x322), the graphics attribute store, and KIL (0x95b), the fragment
        // discard. Neither writes a GPR, and neither encoder writes bits 16..23 at all, so
        // the field reads as R0. That is the STS and BAR bug of 6e09607: if R0 has a live
        // in-flight producer, the scheduler makes the AST or the KIL wait on that
        // scoreboard and CLEARS its tag, so the instruction that really consumes R0 emits
        // no wait of its own and reads stale.
        0x322, 0x95b => false,
        // PLOP3 (0x81c), the predicate-logic op. Its result is a PREDICATE at bits 81..83.
        // Bits 16..23 hold the SECOND lookup table, which the encoder sets to 0, so the
        // field reads as R0 with the same phantom-write result as AST and KIL.
        0x81c => false,
        // The tensor-core ops HMMA (0x23c), IMMA (0x237), LDSM (0x83b) and MOVM (0x23a) each
        // write a REAL GPR at bits 16..23, so they belong to the `else` arm and need no arm
        // here. Each writes a RUN of registers rather than one, and `dstSpan` gives the
        // length of that run so the write-after-write scan covers all of it.
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
/// The same rule holds for every OTHER opcode that is not an ALU op. An instruction whose
/// bits 24..31 or 64..71 hold something that is not a register has that field misread as a
/// GPR number by the ALU rule. Those bits can hold an attribute address, a branch offset, a
/// lookup table, a second destination, or simply nothing at all. So each such opcode names
/// its own source fields instead, and only the true ALU family falls to the `else` arm.
///
/// `disasm.zig` decodes the same instruction set independently and lists the same source
/// fields per opcode. The two models agree, opcode for opcode.
fn readsSrc(opcode: u32, form: u32, pos: usize) bool {
    return switch (opcode) {
        // Convergence barriers (BCLEAR/BSSY/BSYNC) and BAR read no GPR at all: their
        // bit-24/16 fields hold a Bar register, and BAR has no operand.
        0x355, 0x945, 0x941, 0xb1d => false,
        // IPA (0x326), the fragment-input interpolation, in both the perspective and the
        // CONSTANT form. Bits 64..71 hold the ATTRIBUTE ADDRESS divided by 4, not a
        // register: ATTR_GENERIC0 (0x80) reads back as "R32" and ATTR_POSITION (0x70) as
        // "R28". Bits 24..31 are left at zero, which reads as "R0". The only register
        // field, the interpolation offset at bit 32, is always RZ. So an IPA reads no GPR.
        //
        // Left as the ALU rule had it, an IPA into a shader that keeps a live value in R32
        // makes the IPA wait on that value's scoreboard and CLEAR its tag, so the
        // instruction that really consumes R32 emits no wait and reads it stale. Every
        // fragment shader with a generic varying issues this instruction.
        0x326 => false,
        // S2R (0x919), the special-register read. The system value is an 8-bit selector at
        // bits 72..79, and the destination is the only register field. Bits 24..31 and
        // 64..71 stay zero, so the ALU rule read them as two reads of R0.
        0x919 => false,
        // BRA (0x947), EXIT (0x94d) and KIL (0x95b) read no GPR. The BRA taken
        // condition is a PREDICATE at bits 87..89, and its relative offset occupies bits
        // 16..23 plus 34..81, so bits 64..71 hold OFFSET BITS that the ALU rule read as a
        // GPR number. A forward branch puts zeros there and so reads "R0". EXIT and KIL
        // leave bits 24..31 and 64..71 at zero for the same two phantom reads of R0.
        0x947, 0x94d, 0x95b => false,
        // PLOP3 (0x81c), the predicate-logic op. Its operands are three PREDICATES, at bits
        // 68..70, 77..79 and 87..89. Bits 24..31 are zero, so the ALU rule read them as a
        // read of R0. Bits 64..71 are worse: they hold the LOW THREE BITS OF THE LOOKUP
        // TABLE at 64..66 and the third predicate source (PT, which is 7) at 68..70, so the
        // field reads as 0x70 plus the low lookup-table bits. LUT_AND names R112, and
        // LUT_OR and LUT_XOR both name R116. Which register the phantom read hit therefore
        // depended on the boolean operation being compiled.
        0x81c => false,
        // MOV (ALU base 0x002), in the register form 0x202 and the 32-bit immediate form
        // 0x802. Its only operand is at bit 32, and only in the register form. The
        // immediate form holds the value there. Bits 24..31 and 64..71 stay at zero in
        // BOTH forms, so the ALU rule read two phantom R0 sources on the most frequently
        // emitted instruction in the backend.
        0x202, 0x802 => pos == 32 and form == 1,
        // LDC (0xb82), the constant-bank load. Bits 24..31 hold the DYNAMIC offset
        // register, which the encoder always sets to RZ, and it is a real register field.
        // The 16-bit static offset lives at bits 38..53 and the bank at 54..58, so bits
        // 64..71 are zero and the ALU rule read them as R0.
        0xb82 => pos == 24,
        // ALD (0x321), the vertex attribute load: the dynamic offset at 24 and the
        // per-vertex index at 32, both RZ as the encoder emits them. The attribute address
        // is an immediate at bits 40..49, and bits 64..71 are zero, so the ALU rule read
        // them as R0.
        0x321 => pos == 24 or pos == 32,
        // AST (0x322), the output attribute store: the dynamic offset at 24, the data at
        // 32 and the per-vertex index at 64. All three are real register fields.
        0x322 => true,
        // SHFL (0xf89), the all-immediate quad butterfly shuffle: the shuffled value at
        // 24. The lane mask and the segment/clamp are immediates at bits 40..59, and bits
        // 64..71 are zero, so the ALU rule read them as R0.
        0xf89 => pos == 24,
        // FSWZADD (0x822), the quad swizzle-add that finishes a derivative: the shuffled
        // neighbour at 24 and self at 64. Bits 32..39 hold the PACKED LANE OPERATIONS, not
        // a register. Source form 4 already kept the ALU rule off that field, so this arm
        // states the layout rather than changing it.
        0x822 => pos == 24 or pos == 64,
        // TEX, TLD4 and TLD: the coordinate at 24 and the bindless handle at 32. Bits
        // 64..71 hold the SECOND DESTINATION (dst + 2, the B and A channels), not a
        // source. Reading it as a source made the texture op wait on, and clear the tag
        // of, whatever still had an in-flight producer in dst + 2. That accidental
        // write-after-write wait is now kept, and extended to the whole result block, by
        // the destination-span scan in `scheduleBlocks`.
        //
        // Both source fields name the FIRST register of a RUN: the coordinate BLOCK at 24
        // is 1 to 4 registers by the dimension, and src1 at 32 holds the handle plus any
        // explicit LOD or depth-compare reference. `srcSpan` gives both lengths.
        encode.TEX_OPCODE, encode.TLD4_OPCODE, encode.TLD_OPCODE => pos == 24 or pos == 32,
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
        // HMMA and IMMA: the A fragment at 24, the B fragment at 32 and the C fragment at
        // 64. All three are real registers, so all three must be waited on.
        //
        // The ALU rule already gives that answer, because bits 9..11 of both opcodes read
        // as form 1. That is an ACCIDENT of the opcode numbers, not a rule, and the same
        // accident is what made LDC, BRA and S2R look safe before they were audited. The
        // arm states the layout so a later opcode change cannot silently drop a source.
        //
        // Each of the three is the FIRST register of a RUN that one lane holds. `srcSpan`
        // gives the length of that run, so the wait covers every register of it. A wait on
        // the first register alone let the tensor core read the rest of the fragment STALE.
        encode.HMMA_OPCODE, encode.IMMA_OPCODE => true,
        // LDSM (0x83b) and MOVM (0x23a): the address, or the fragment to transpose, at 24.
        // Nothing else. LDSM holds URZ at bits 32..39 and its 24-bit immediate offset stops
        // at bit 63, so bits 64..71 are ZERO and the ALU rule read them as a read of R0.
        // MOVM writes nothing above bit 31 except its mode field at 78..80, so the ALU rule
        // read TWO phantom R0 sources on it. Both are the phantom-read half of the STS and
        // BAR bug of 6e09607: a phantom read on a register with a live producer makes the
        // load wait on that scoreboard and CLEARS its tag, so the instruction that really
        // consumes the register never waits and reads it stale.
        encode.LDSM_OPCODE, encode.MOVM_OPCODE => pos == 24,
        else => pos != 32 or form == 1,
    };
}

/// Whether `opcode` addresses memory through a 64-bit GPR PAIR at bit 24, so it reads
/// `addr + 1` as well as `addr`. Every GLOBAL access does. The shared ones take a 32-bit
/// offset into the CTA window in ONE register, so reading `addr + 1` for them would wait on
/// and free an unrelated register's scoreboard. `srcSpan` turns this into the run length of
/// the bit-24 operand.
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
    // ALD (0x321) fills `comps` CONSECUTIVE registers, and bits 74..75 hold comps - 1. A
    // one-component fetch is the only shape the isel emits today, so this is not live, but
    // a two-component varying fetch tagged only the first register and let a consumer of
    // the second read it before the attribute landed. `srcSpan` covers the AST that stores
    // the same block back.
    if (opcode == 0x321) return getField(inst, 74, 2) + 1;
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
    // A tensor op writes a RUN of registers per lane, and the run length is in the
    // instruction. Every tile HMMA names has a 16 by 8 result, so one lane holds 4 of its
    // 128 values. An fp32 result takes one register per value, and an fp16 result packs
    // two values per register. The result type is bit 76.
    if (opcode == encode.HMMA_OPCODE)
        return if (getField(inst, 76, 1) == @intFromEnum(encode.HmmaDstType.f32)) 4 else 2;
    // An IMMA result is int32 whatever the input width is, so one register per value. An
    // m8n8 tile holds 64 values over 32 lanes, that is 2 per lane, and an m16n8 tile holds
    // 128, that is 4. The tile selector is split across bit 75 and bits 85..87.
    //
    // The LOW bit of the selector cannot change the answer today: it separates m16n8k32
    // from m16n8k16, and both of those write 4 registers. The whole selector is rebuilt
    // anyway, because the arms below then name the NAK tiles directly and stay correct if
    // a later tile with a different result height reuses the low bit. A mutation that drops
    // bit 75 here is therefore EQUIVALENT, not a gap in the tests.
    if (opcode == encode.IMMA_OPCODE) {
        const tile: u3 = @intCast(getField(inst, 75, 1) | (getField(inst, 85, 2) << 1));
        return switch (tile) {
            @intFromEnum(encode.ImmaSize.m8n8k16), @intFromEnum(encode.ImmaSize.m8n8k32) => 2,
            @intFromEnum(encode.ImmaSize.m16n8k16),
            @intFromEnum(encode.ImmaSize.m16n8k32),
            @intFromEnum(encode.ImmaSize.m16n8k64),
            => 4,
            // 1, 3 and 7 are gaps in the NAK table and the encoder cannot produce them.
            // The smallest span never tags a register the op does not write.
            1, 3, 7 => 1,
        };
    }
    // LDSM gives each lane 32 bits per fragment, so the fragment count at bits 72..74 is
    // also the number of destination registers.
    if (opcode == encode.LDSM_OPCODE) {
        const count: u2 = @intCast(getField(inst, 72, 2));
        return switch (count) {
            @intFromEnum(encode.LdsmCount.x1) => 1,
            @intFromEnum(encode.LdsmCount.x2) => 2,
            @intFromEnum(encode.LdsmCount.x4) => 4,
            // NAK panics on any other count and the encoder cannot produce one.
            3 => 1,
        };
    }
    // MOVM transposes ONE 8 by 8 fragment, which gives each lane 32 bits, so it writes a
    // single register. It falls to the span of 1 below and needs no arm here.
    return 1;
}

/// How many CONSECUTIVE registers the source operand at bit `pos` covers. The mirror of
/// `dstSpan`.
///
/// Most operands are one register. Some name only the FIRST register of a RUN: a 64-bit
/// global address, the data block of a wide store or atomic, a multi-component attribute,
/// a texture coordinate block, or one lane's slice of a matrix fragment. For those, a
/// producer that fills a register INSIDE the run but not at its first position must be
/// waited on too. A wait on the first register alone lets the instruction read the rest of
/// the run STALE, which is the source-side mirror of the destination-block bug `dstSpan`
/// closes.
///
/// The run length is DATA-DEPENDENT: the memory type, the atomic type, the component count,
/// the texture dimension and the tensor tile each change it. So each length is DECODED out
/// of the instruction word, the way `dstSpan` decodes the tile selector. A fixed worst case
/// is wrong in both directions. Too SMALL reads a stale register. Too LARGE makes the
/// instruction wait on a scoreboard it does not need AND FREE it, so the real consumer of
/// that register emits no wait of its own and reads stale. There are only six scoreboards.
///
/// Audited and confirmed to be ONE register, so they need no arm: the LDSM address (a
/// 32-bit offset into the shared window, one register, the same form LDS and STS take), the
/// MOVM source (one 8 by 8 fragment of 16-bit elements gives each lane 32 bits), the LDS,
/// STS and shared-atomic addresses, the ALD dynamic offset and vertex index, the LDC
/// dynamic offset, the SHFL value, the FSWZADD operands, and every ALU source.
fn srcSpan(opcode: u32, inst: Inst, pos: usize) u32 {
    // A GLOBAL access addresses memory through an aligned GPR PAIR at bit 24: it reads both
    // `addr` (lo) and `addr+1` (hi). The hi half is not an explicit source field, so without
    // this the load issues with a stale high address dword (a UBO base pointer whose hi LDC
    // has not landed), reads garbage and faults the GR front-end.
    if (pos == 24 and readsAddrPair(opcode)) return 2;
    // STG and STS read the data BLOCK that starts at bit 32. Its length is the register
    // count of the 3-bit memory type at bits 73..75, the same field `dstSpan` reads for a
    // load: a b64 store reads (data, data+1) and a b128 store (data .. data+3). The isel
    // emits a b64 store for every POINTER value, and a pointer pair whose two halves come
    // from two separate LDCs has a live producer on the HIGH half only, so a wait on `data`
    // alone stores a stale address dword.
    if ((opcode == 0x986 or opcode == 0x988) and pos == 32)
        return switch (getField(inst, 73, 3)) {
            @intFromEnum(encode.MemType.b64) => 2,
            @intFromEnum(encode.MemType.b128) => 4,
            else => 1,
        };
    // The atomics take their operands at the atomic width, which is the WIDER 4-bit field
    // at bits 73..76, not the 3-bit memory type. ATOMG, ATOMS and RED read the data at bit
    // 32. Compare-and-swap reads the compare operand at 32 and the swap data at 64. A
    // 64-bit atomic reads a register PAIR for each of them.
    if (opcode == encode.ATOMG_OPCODE or opcode == encode.ATOMS_OPCODE or
        opcode == encode.RED_OPCODE or opcode == encode.ATOMG_CAS_OPCODE or
        opcode == encode.ATOMS_CAS_OPCODE)
    {
        if (pos == 32 or pos == 64) return switch (getField(inst, 73, 4)) {
            @intFromEnum(encode.AtomType.u64), @intFromEnum(encode.AtomType.i64) => 2,
            else => 1,
        };
    }
    // AST (0x322) stores `comps` CONSECUTIVE registers starting at bit 32, and bits 74..75
    // hold comps - 1. Every call site passes 1 today, so this is not live, but a
    // multi-component output store waited only on the producer of the first register and
    // wrote the rest of the block to the attribute STALE.
    if (opcode == 0x322 and pos == 32) return getField(inst, 74, 2) + 1;
    if (isTexResult(opcode)) {
        // The COORDINATE BLOCK at bit 24, whose length the dimension field at bits 61..63
        // selects. Not live today: the isel materializes every register of the block with a
        // MOV, an FADD, an F2I or an FFMA immediately before the texture op, and each of
        // those is fixed-latency, so no register of the block carries an in-flight tag by
        // the time the TEX issues. The span costs nothing and removes the dependence on
        // that isel property.
        if (pos == 24) return texCoordRegs(@intCast(getField(inst, 61, 3)));
        if (pos == 32) {
            // A GATHER reads its src1 as the bindless HANDLE alone. Its LOD is implicit,
            // and bits 87..88 hold the gather COMPONENT, not a LOD mode, so the lod-mode
            // decode below would misread the component and over-wait.
            if (opcode == encode.TLD4_OPCODE) return 1;
            // src1 starts with the handle. NAK `nak_nir_lower_tex.c` packs the explicit
            // LOD and the depth-compare reference into the SAME run, right after it, so a
            // TEX.LL reads (handle, lod) and a depth-compare TEX reads (handle, dref).
            var regs: u32 = 1;
            if (getField(inst, 78, 1) == 1) regs += 1; // z_cmpr: the dref
            regs += texLodRegs(@intCast(getField(inst, 59, 1) | (getField(inst, 87, 3) << 1)));
            return regs;
        }
    }
    // The tensor fragments: A at bit 24, B at bit 32 and C at bit 64. Each names the first
    // register of one LANE's slice of a matrix.
    if (opcode == encode.HMMA_OPCODE) {
        const tile: u2 = @intCast(getField(inst, 75, 1) | (getField(inst, 78, 1) << 1));
        if (pos == 24) return hmmaOperandRegs(tile, false);
        if (pos == 32) return hmmaOperandRegs(tile, true);
        // C has the shape AND the element type of D, both selected by bit 76, so its run is
        // exactly as long.
        if (pos == 64) return dstSpan(opcode, inst);
    }
    if (opcode == encode.IMMA_OPCODE) {
        const tile: u3 = @intCast(getField(inst, 75, 1) | (getField(inst, 85, 2) << 1));
        // The width flags are INDEPENDENT: bit 83 is "A is 4-bit" and bit 84 is "B is
        // 4-bit", and `m16n8k32` accepts either for either operand. So each operand's run
        // is read from its OWN flag.
        if (pos == 24) return immaOperandRegs(tile, getField(inst, 83, 1) == 1, false);
        if (pos == 32) return immaOperandRegs(tile, getField(inst, 84, 1) == 1, true);
        // The accumulator is int32 whatever the input width is, like the result.
        if (pos == 64) return dstSpan(opcode, inst);
    }
    return 1;
}

/// How many CONSECUTIVE coordinate registers a texture op reads at bit 24, from the
/// dimension field at bits 61..63. The values are NAK `SM70Encoder::set_tex_dim`: 1D = 0,
/// 2D = 1, 3D = 2, Cube = 3, Array1D = 4, Array2D = 5, ArrayCube = 7. An ARRAY target reads
/// the layer index as an extra FIRST register, which is how the isel packs `sampler2DArray`.
///
/// The encoders reach 1 (2D), 2 (3D) and 5 (2D array). `TexDim.cube` is an internal marker
/// the isel lowers to a 2D atlas sample, so no TEX carries it, but the coordinate count of a
/// native cube target is the 3-register direction vector and the arm says so.
fn texCoordRegs(dim: u3) u32 {
    return switch (dim) {
        0 => 1, // 1D: x
        1 => 2, // 2D: u, v
        2 => 3, // 3D: u, v, w
        3 => 3, // Cube: the x, y, z direction vector
        4 => 2, // Array1D: layer, x
        5 => 3, // Array2D: layer, u, v
        6 => 1, // a gap in the NAK table
        7 => 4, // ArrayCube: layer, x, y, z
    };
}

/// How many registers of a texture op's src1 run the LOD mode takes AFTER the handle. The
/// mode is split by NAK `set_tex_lod_mode2(59..60, 87..90)`: the low bit at 59 and the three
/// high bits at 87..89.
///
/// NAK `TexLodMode` and the src1 packing in `nak_nir_lower_tex.c`: Auto (0) and Zero (1)
/// name no register, Bias (2), Lod (3) and Clamp (4) each take one, and BiasClamp (5) takes
/// two. Only Auto and Lod are reachable: `tex`, `tex2d` and `tld4` write Auto, and `texLod`
/// and `tld` write Lod.
fn texLodRegs(mode: u4) u32 {
    return switch (mode) {
        0, 1 => 0, // Auto, Zero
        2, 3, 4 => 1, // Bias, Lod, Clamp
        5 => 2, // BiasClamp
        6...15 => 0, // gaps in the NAK table
    };
}

/// How many CONSECUTIVE registers one HMMA input fragment covers in one lane. `is_b`
/// selects the B operand, which is `k` by `n`, where A is `m` by `k`.
///
/// A matrix of `rows` by `cols` elements is spread over the 32 lanes of the warp and packed
/// into 32-bit registers, so one lane holds `rows * cols * element_bits / (32 * 32)` of
/// them. The A and B elements are fp16, the only source type the encoder writes (bits 82..83
/// stay 0, and NAK confirms no other value). A bf16 source would pack the same way; a tf32
/// source would not, and it is unreachable for the same reason.
///
/// The `m16n8k4` tile is the tf32 shape, so with fp16 elements its B operand is HALF a
/// register. The floor to one register never names a register the tile does not read.
fn hmmaOperandRegs(tile: u2, is_b: bool) u32 {
    const shape: struct { m: u32, n: u32, k: u32 } = switch (tile) {
        @intFromEnum(encode.HmmaSize.m16n8k8) => .{ .m = 16, .n = 8, .k = 8 },
        @intFromEnum(encode.HmmaSize.m16n8k16) => .{ .m = 16, .n = 8, .k = 16 },
        @intFromEnum(encode.HmmaSize.m16n8k4) => .{ .m = 16, .n = 8, .k = 4 },
        // 3 is a gap in the NAK table and the encoder cannot produce it. The smallest span
        // never names a register the op does not read.
        3 => return 1,
    };
    const elems = if (is_b) shape.k * shape.n else shape.m * shape.k;
    return @max(1, elems * 16 / (32 * 32));
}

/// How many CONSECUTIVE registers one IMMA input fragment covers in one lane. `is_b` selects
/// the B operand and `four_bit` is THAT operand's own width flag (bit 83 for A, bit 84 for
/// B).
///
/// The same per-lane count as `hmmaOperandRegs`, with 8-bit or 4-bit elements. The tile
/// shapes come from NAK's `ImmaSize` table, and the counts reproduce the PTX "Matrix
/// Fragments" register vectors: `m8n8k16` reads 1 and 1, `m16n8k16` reads 2 and 1,
/// `m16n8k32` reads 4 and 2 at 8 bits or 2 and 1 at 4 bits, and `m16n8k64` reads 4 and 2.
fn immaOperandRegs(tile: u3, four_bit: bool, is_b: bool) u32 {
    const shape: struct { m: u32, n: u32, k: u32 } = switch (tile) {
        @intFromEnum(encode.ImmaSize.m8n8k16) => .{ .m = 8, .n = 8, .k = 16 },
        @intFromEnum(encode.ImmaSize.m8n8k32) => .{ .m = 8, .n = 8, .k = 32 },
        @intFromEnum(encode.ImmaSize.m16n8k16) => .{ .m = 16, .n = 8, .k = 16 },
        @intFromEnum(encode.ImmaSize.m16n8k32) => .{ .m = 16, .n = 8, .k = 32 },
        @intFromEnum(encode.ImmaSize.m16n8k64) => .{ .m = 16, .n = 8, .k = 64 },
        // 1, 3 and 7 are gaps in the NAK table and the encoder cannot produce them.
        1, 3, 7 => return 1,
    };
    const elems = if (is_b) shape.k * shape.n else shape.m * shape.k;
    const bits: u32 = if (four_bit) 4 else 8;
    return @max(1, elems * bits / (32 * 32));
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
        // Every opcode that is not an ALU op names its own source fields. See `readsSrc`
        // for why the ALU rule is wrong for them in both directions.
        //
        // Each source field names only the FIRST register of its operand. An operand that
        // is a RUN - a 64-bit global address pair, a wide store's data block, a texture
        // coordinate block, one lane's slice of a matrix fragment - reads every register of
        // the run, so `srcSpan` gives the length and the scan covers all of it. Waiting on
        // the first register alone let a producer of `reg + 1` and beyond go unawaited, and
        // the instruction then read those registers STALE.
        var wait: u32 = 0;
        inline for (.{ 24, 32, 64 }) |pos| {
            if (readsSrc(opcode, form, pos)) {
                const reg = getField(inst.*, pos, 8);
                if (reg != RZ) {
                    const rspan = srcSpan(opcode, inst.*, pos);
                    var r: u32 = 0;
                    while (r < rspan and reg + r < RZ) : (r += 1) {
                        const sreg = reg + r;
                        if (scoreboard_of[sreg] != 0)
                            wait |= @as(u32, 1) << @intCast(scoreboard_of[sreg] - 1);
                    }
                }
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
        //
        // The scan covers the WHOLE destination block, not only the first register. A TEX
        // writes up to four result registers, and a B64 or B128 load two or four, all
        // under one write barrier. Each of them can clobber a different in-flight
        // producer, so each needs its own wait. Before this scan spanned the block, the
        // texture ops got the dst + 2 half of that protection by accident, through the
        // second-destination field at bit 64 that `readsSrc` used to misread as a source.
        if (writesDst(opcode)) {
            const wdst = getField(inst.*, 16, 8);
            const wspan = dstSpan(opcode, inst.*);
            var w: u32 = 0;
            while (w < wspan and wdst + w < RZ) : (w += 1) {
                const wreg = wdst + w;
                if (scoreboard_of[wreg] == 0) continue;
                const sb_idx = scoreboard_of[wreg] - 1;
                const wbit: u32 = @as(u32, 1) << @intCast(sb_idx);
                setField(inst, 116, 6, getField(inst.*, 116, 6) | wbit);
                // Clear this register's tag. Free the scoreboard if no other register
                // (a multi-register TEX block) still holds it.
                scoreboard_of[wreg] = 0;
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
/// producer is among `wait`. Mirrors the source-register set the wait computation scans,
/// EXACTLY: `readsSrc` for the three operand fields, and `srcSpan` for the whole run each
/// field names. A register that was NOT read keeps its tag, so a later instruction reading
/// it re-waits (the multi-register TEX-result case).
///
/// The two scans must stay identical. A register that is waited on but not cleared holds a
/// scoreboard that never returns to the free pool, and a register that is cleared but not
/// waited on loses the protection of a producer that is still in flight.
fn clearReadRegs(inst: Inst, opcode: u32, form: u32, scoreboard_of: *[256]u8, wait: u32) void {
    inline for (.{ 24, 32, 64 }) |pos| {
        if (readsSrc(opcode, form, pos)) {
            const reg = getField(inst, pos, 8);
            if (reg != RZ) {
                const rspan = srcSpan(opcode, inst, pos);
                var r: u32 = 0;
                while (r < rspan and reg + r < RZ) : (r += 1) {
                    const sreg = reg + r;
                    if (scoreboard_of[sreg] != 0 and
                        (wait & (@as(u32, 1) << @intCast(scoreboard_of[sreg] - 1))) != 0)
                        scoreboard_of[sreg] = 0;
                }
            }
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

test "an IPA does not read its ATTRIBUTE ADDRESS as a source register" {
    // Regression: `encode.ipa` writes the attribute address divided by 4 into bits 64..71,
    // and the ALU source rule read those bits as a GPR number. ATTR_GENERIC0 (0x80) came
    // back as "R32". With a live in-flight producer for R32 the IPA waited on that
    // scoreboard and CLEARED its tag, so the instruction that really consumes R32 emitted
    // no wait and read it stale. An IPA reads no GPR at all: its one register field, the
    // interpolation offset at bit 32, is always RZ.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(32, 2, .{}); // R32 <- global, variable latency
    insts[1] = encode.ipa(4, encode.ATTR_GENERIC0, .{}); // bits 64..71 = 0x80 >> 2 = 32
    insts[2] = encode.iadd3(5, 32, 32, .{}); // the REAL consumer of R32
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the IPA waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "an S2R reads no phantom R0 sources" {
    // Regression: S2R takes its system value from a selector at bits 72..79 and leaves bits
    // 24..31 and 64..71 at zero, which the ALU rule read as two reads of R0.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.s2r(4, encode.SR_TID_X, .{});
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the S2R waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "a BRA does not read its BRANCH OFFSET as a source register" {
    // Regression: a BRA holds its relative offset in bits 16..23 plus 34..81, so bits
    // 64..71 are OFFSET BITS, and bits 24..31 are zero. The ALU rule read both as GPR
    // numbers. A forward branch has zeros in both, so it phantom-read R0 twice.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.bra(4, .{}); // a forward branch: bits 24..31 and 64..71 are zero
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the BRA waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "an EXIT reads no phantom R0 sources" {
    // Regression: EXIT leaves bits 24..31 and 64..71 at zero, which the ALU rule read as
    // two reads of R0. A dead load into R0 just before the EXIT then made the EXIT wait on
    // a scoreboard it has no use for.
    var insts: [2]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency, never read
    insts[1] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    try std.testing.expect(getField(insts[0], 110, 3) < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the EXIT waits on nothing
}

test "a KIL neither claims a destination register nor reads phantom sources" {
    // Regression: KIL discards the fragment and touches no GPR, but it leaves bits 16..23,
    // 24..31 and 64..71 at zero. The scheduler read that as a write to R0 plus two reads of
    // R0, so a live producer for R0 was waited on and its tag CLEARED, and the instruction
    // that really consumes R0 read stale. Same shape as the STS and BAR bug of 6e09607.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.kil(.{});
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the KIL waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "a PLOP3 does not read its LOOKUP TABLE as a source register" {
    // Regression: PLOP3 combines two PREDICATES into a predicate and touches no GPR. It
    // leaves bits 16..23 at zero (the second lookup table) and bits 24..31 at zero, and
    // bits 64..71 hold the low three lookup-table bits at 64..66 over the third predicate
    // source (PT = 7) at 68..70. The scheduler therefore read a phantom write to R0, a
    // phantom read of R0, and a phantom read of R112 or R116 depending on which boolean
    // operation was being compiled. LUT_XOR (0x3C) names R116.
    var insts: [5]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0: the phantom write and the bit-24 phantom read
    insts[1] = encode.ldgU32(116, 2, .{}); // R116 = 0x70 | (LUT_XOR & 7), the bit-64 phantom read
    insts[2] = encode.plop3(1, 2, 3, encode.LUT_XOR, .{});
    insts[3] = encode.iadd3(5, 0, 116, .{}); // the REAL consumers of R0 and R116
    insts[4] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const r0_bar = getField(insts[0], 110, 3);
    const r116_bar = getField(insts[1], 110, 3);
    try std.testing.expect(r0_bar < 6 and r116_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[2], 116, 6)); // the PLOP3 waits on nothing
    const consumer_wait = getField(insts[3], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(r0_bar))) != 0);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(r116_bar))) != 0);
}

test "an AST does not claim a phantom write to R0" {
    // Regression: AST stores to an output attribute and writes no GPR, but the encoder
    // leaves bits 16..23 at zero, so the scheduler recorded a write to R0. With a live
    // producer for R0 the store waited on that scoreboard and CLEARED its tag, so the
    // instruction that really consumes R0 read stale. This is the STS bug of 6e09607 in
    // the graphics path.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.ast(encode.ATTR_POSITION, 8, 1, .{}); // o[POSITION] = R8
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the AST waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "an ALD does not read a phantom R0 at bit 64" {
    // Regression: ALD takes its attribute address from an immediate at bits 40..49 and
    // leaves bits 64..71 at zero, which the ALU rule read as a read of R0.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.ald(4, encode.ATTR_GENERIC0, 1, .{});
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the ALD waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "an LDC does not read a phantom R0 at bit 64" {
    // Regression: LDC holds its static offset at bits 38..53 and its bank at 54..58, so
    // bits 64..71 are zero and the ALU rule read them as a read of R0. Every uniform load
    // in the backend is an LDC, so this stole a scoreboard on nearly every shader.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.ldc(4, 0, 0x10, .{});
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the LDC waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "a MOV reads no phantom R0 sources in either form" {
    // Regression: MOV takes its one operand from bit 32 in the register form and holds its
    // immediate there in the immediate form. Bits 24..31 and 64..71 are zero in BOTH forms,
    // so the ALU rule read two phantom R0 sources on the most frequently emitted
    // instruction in the backend.
    var insts: [5]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.movImm(4, 0x1234, .{});
    insts[2] = encode.movReg(5, 6, .{});
    insts[3] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[4] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // MOV.imm waits on nothing
    try std.testing.expectEqual(@as(u32, 0), getField(insts[2], 116, 6)); // MOV.reg waits on nothing
    const consumer_wait = getField(insts[3], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "a SHFL does not read a phantom R0 at bit 64" {
    // Regression: the all-immediate quad SHFL holds its lane mask and its segment/clamp in
    // bits 40..59 and leaves bits 64..71 at zero, which the ALU rule read as a read of R0.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 2, .{}); // R0 <- global, variable latency
    insts[1] = encode.shflBflyQuad(4, 6, 1, .{}); // R4 <- R6 from the horizontal neighbour
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // the SHFL waits on nothing
    const consumer_wait = getField(insts[2], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "a TEX waits before it overwrites ANY register of its result block" {
    // A TEX writes R4..R7 under one write barrier, and each of the four can clobber a
    // different in-flight producer. The write-after-write scan used to look only at bits
    // 16..23, so only R4 was protected. R6 was covered by accident, because bits 64..71
    // hold the SECOND DESTINATION and the ALU rule misread them as a source. R5 and R7 were
    // covered by nothing, so a decoupled load into either landed after the TEX and
    // clobbered a result channel.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(5, 2, .{}); // R5 = result dst + 1, in flight
    insts[1] = encode.ldgU32(6, 2, .{}); // R6 = result dst + 2, in flight
    insts[2] = encode.tex2d(4, 8, 10, .{}); // writes R4..R7
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const r5_bar = getField(insts[0], 110, 3);
    const r6_bar = getField(insts[1], 110, 3);
    try std.testing.expect(r5_bar < 6 and r6_bar < 6);
    const tex_wait = getField(insts[2], 116, 6);
    try std.testing.expect((tex_wait & (@as(u32, 1) << @intCast(r5_bar))) != 0);
    try std.testing.expect((tex_wait & (@as(u32, 1) << @intCast(r6_bar))) != 0);
}

test "an HMMA result gets a scoreboard and its consumer waits on it" {
    // NAK sm120_instr_latencies.rs gives HMMA the latency class `Hmma`, which
    // `SM120Latency::needs_scoreboards` accepts, so an HMMA result lands an unknown number
    // of cycles after issue. Without a barrier the instruction that reads the result gets
    // whatever the register held before the multiply.
    var insts: [3]Inst = undefined;
    insts[0] = encode.hmma(8, 0, 4, 8, .m16n8k16, .f32, .{}); // D fragment into R8..R11
    insts[1] = encode.iadd3(20, 8, 8, .{}); // consumes R8
    insts[2] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const hmma_bar = getField(insts[0], 110, 3);
    try std.testing.expect(hmma_bar < 6); // a real scoreboard, not 7 = none
    const consumer_wait = getField(insts[1], 116, 6);
    try std.testing.expect((consumer_wait & (@as(u32, 1) << @intCast(hmma_bar))) != 0);
}

test "an HMMA tags its WHOLE fp32 destination run, and only that run" {
    // The D fragment of a 16 by 8 fp32 result is 4 registers per lane, all under ONE write
    // barrier. A consumer of ANY of the four must wait. With only the first register tagged,
    // a read of R9, R10 or R11 races the still-running multiply and gets a stale value.
    var insts: [6]Inst = undefined;
    insts[0] = encode.hmma(8, 0, 4, 8, .m16n8k16, .f32, .{}); // R8..R11
    insts[1] = encode.iadd3(20, 9, 9, .{}); // consumes R9
    insts[2] = encode.iadd3(21, 10, 10, .{}); // consumes R10
    insts[3] = encode.iadd3(22, 11, 11, .{}); // consumes R11
    insts[4] = encode.iadd3(23, 12, 12, .{}); // R12 is PAST the run
    insts[5] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const hmma_bar = getField(insts[0], 110, 3);
    try std.testing.expect(hmma_bar < 6);
    const bit = @as(u32, 1) << @intCast(hmma_bar);
    try std.testing.expect((getField(insts[1], 116, 6) & bit) != 0);
    try std.testing.expect((getField(insts[2], 116, 6) & bit) != 0);
    try std.testing.expect((getField(insts[3], 116, 6) & bit) != 0);
    // R12 is outside the run, so gating it would be a wasted wait on a register the tensor
    // op never writes.
    try std.testing.expectEqual(@as(u32, 0), getField(insts[4], 116, 6));
}

test "an fp16 HMMA result packs two elements per register, so its run is half as long" {
    // The same 16 by 8 result written as fp16 is 2 registers, not 4, because each register
    // holds two elements. A span read off the tile shape alone would tag R10 and R11 as
    // well and gate a later write to them for no reason.
    var insts: [4]Inst = undefined;
    insts[0] = encode.hmma(8, 0, 4, 8, .m16n8k16, .f16, .{}); // R8..R9 only
    insts[1] = encode.iadd3(20, 9, 9, .{}); // consumes R9, inside the run
    insts[2] = encode.iadd3(21, 10, 10, .{}); // R10 is PAST the run
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const hmma_bar = getField(insts[0], 110, 3);
    try std.testing.expect(hmma_bar < 6);
    try std.testing.expect((getField(insts[1], 116, 6) & (@as(u32, 1) << @intCast(hmma_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[2], 116, 6));
}

test "an HMMA waits on the in-flight producer of EACH of its three fragment bases" {
    // A, B and C sit at bits 24, 32 and 64. Each is the first register of one lane's run.
    // A missed wait here feeds the tensor core a register the load has not filled yet.
    var insts: [5]Inst = undefined;
    insts[0] = encode.ldgU32(0, 30, .{}); // R0 = the A base
    insts[1] = encode.ldgU32(4, 30, .{}); // R4 = the B base
    insts[2] = encode.ldgU32(12, 30, .{}); // R12 = the C base
    insts[3] = encode.hmma(16, 0, 4, 12, .m16n8k16, .f32, .{});
    insts[4] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const wait = getField(insts[3], 116, 6);
    for (insts[0..3]) |p| {
        const bar = getField(p, 110, 3);
        try std.testing.expect(bar < 6);
        try std.testing.expect((wait & (@as(u32, 1) << @intCast(bar))) != 0);
    }
}

test "an IMMA claims NO scoreboard but still waits on its fragment producers" {
    // NAK sm120_instr_latencies.rs gives IMMA the latency class `Imma`, which
    // `SM120Latency::needs_scoreboards` does NOT accept, so IMMA is a coupled op that the
    // stall delay covers. A write barrier on an op the hardware never signals would hang
    // every consumer, so the exclusion is deliberate. Its SOURCE waits still matter.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 30, .{}); // R0 = the A base
    insts[1] = encode.ldgU32(4, 30, .{}); // R4 = the B base
    insts[2] = encode.imma(8, 0, 4, 8, .m16n8k32, .{ .signed = true }, .{ .signed = true }, false, .{});
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    try std.testing.expectEqual(@as(u32, 7), getField(insts[2], 110, 3)); // no write barrier
    const wait = getField(insts[2], 116, 6);
    for (insts[0..2]) |p| {
        const bar = getField(p, 110, 3);
        try std.testing.expect(bar < 6);
        try std.testing.expect((wait & (@as(u32, 1) << @intCast(bar))) != 0);
    }
}

test "an IMMA waits before it overwrites ANY register of its int32 destination run" {
    // IMMA gets no scoreboard of its own, but it still WRITES a run of registers. If one of
    // them still has an in-flight load, the load lands after the multiply and destroys the
    // result. An m16n8 tile writes 4 int32 registers per lane.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(11, 30, .{}); // R11 is the LAST register of the run below
    insts[1] = encode.iadd3(20, 21, 22, .{}); // filler, reads nothing in flight
    insts[2] = encode.imma(8, 0, 4, 8, .m16n8k32, .{ .signed = true }, .{ .signed = true }, false, .{});
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expect((getField(insts[2], 116, 6) & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "an m8n8 IMMA writes half the run of an m16n8 IMMA" {
    // 64 result values over 32 lanes is 2 registers per lane, and 128 is 4. The tile
    // selector is split across bit 75 and bits 85..87, so a span that read only one half of
    // it would give the wrong run for two of the five tiles.
    const plain: encode.ImmaOperand = .{ .signed = true };
    var small: [4]Inst = undefined;
    small[0] = encode.ldgU32(10, 30, .{}); // R10 is PAST an m8n8 run of R8..R9
    small[1] = encode.iadd3(20, 21, 22, .{});
    small[2] = encode.imma(8, 0, 4, 8, .m8n8k16, plain, plain, false, .{});
    small[3] = encode.exit(.{});
    scheduleBlocks(&small, &.{0});
    try std.testing.expectEqual(@as(u32, 0), getField(small[2], 116, 6));

    var large: [4]Inst = undefined;
    large[0] = encode.ldgU32(10, 30, .{}); // R10 is INSIDE an m16n8 run of R8..R11
    large[1] = encode.iadd3(20, 21, 22, .{});
    large[2] = encode.imma(8, 0, 4, 8, .m16n8k16, plain, plain, false, .{});
    large[3] = encode.exit(.{});
    scheduleBlocks(&large, &.{0});
    const bar = getField(large[0], 110, 3);
    try std.testing.expect(bar < 6);
    try std.testing.expect((getField(large[2], 116, 6) & (@as(u32, 1) << @intCast(bar))) != 0);
}

test "an LDSM result gets a scoreboard and tags every fragment register it fills" {
    // NAK sm120_instr_latencies.rs gives LDSM the class DecoupledAgu, the same class as an
    // LDS, so its result lands an unknown number of cycles after issue. It fills one
    // register per fragment, so an x4 load fills four and a consumer of ANY of them waits.
    var insts: [5]Inst = undefined;
    insts[0] = encode.ldsm(8, 2, .x4, false, .{}); // R8..R11 <- shared[R2]
    insts[1] = encode.iadd3(20, 11, 11, .{}); // consumes R11, the last of the run
    insts[2] = encode.iadd3(21, 12, 12, .{}); // R12 is PAST the run
    insts[3] = encode.iadd3(22, 23, 24, .{});
    insts[4] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldsm_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldsm_bar < 6);
    try std.testing.expect((getField(insts[1], 116, 6) & (@as(u32, 1) << @intCast(ldsm_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[2], 116, 6));
}

test "the LDSM destination run follows the fragment count, for EVERY count" {
    // x1 fills R8, x2 fills R8..R9 and x4 fills R8..R11. A consumer of the LAST register of
    // the run must wait, and a consumer of the register just PAST it must not. The middle
    // case earns its place: with only x1 and x4 checked, a span that collapsed x2 to one
    // register passed every test.
    const cases = [_]struct { count: encode.LdsmCount, last: u8, past: u8 }{
        .{ .count = .x1, .last = 8, .past = 9 },
        .{ .count = .x2, .last = 9, .past = 10 },
        .{ .count = .x4, .last = 11, .past = 12 },
    };
    for (cases) |c| {
        var insts: [4]Inst = undefined;
        insts[0] = encode.ldsm(8, 2, c.count, false, .{});
        insts[1] = encode.iadd3(20, c.last, c.last, .{}); // inside the run
        insts[2] = encode.iadd3(21, c.past, c.past, .{}); // past the run
        insts[3] = encode.exit(.{});
        scheduleBlocks(&insts, &.{0});
        const bar = getField(insts[0], 110, 3);
        try std.testing.expect(bar < 6);
        try std.testing.expect((getField(insts[1], 116, 6) & (@as(u32, 1) << @intCast(bar))) != 0);
        try std.testing.expectEqual(@as(u32, 0), getField(insts[2], 116, 6));
    }
}

test "an LDSM does not read a phantom R0 at bit 64" {
    // Regression shape: LDSM's 24-bit immediate offset stops at bit 63, so bits 64..71 are
    // ZERO and the ALU source rule read them as a read of R0. With a live producer in R0 the
    // load waited on that scoreboard and CLEARED its tag, so the real consumer of R0 never
    // waited and read it stale.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 20, .{}); // R0 <- global, variable latency
    insts[1] = encode.ldsm(8, 2, .x2, false, .{}); // reads R2 only
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // no phantom wait
    try std.testing.expect((getField(insts[2], 116, 6) & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "an LDSM waits on the in-flight producer of its ADDRESS register" {
    // Every lane gives its own shared-window offset, and that offset normally comes from an
    // S2R lane id. Without the wait the load reads whatever the register held before.
    var insts: [3]Inst = undefined;
    insts[0] = encode.s2r(2, encode.SR_LANEID, .{});
    insts[1] = encode.ldsm(8, 2, .x2, false, .{});
    insts[2] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const s2r_bar = getField(insts[0], 110, 3);
    try std.testing.expect(s2r_bar < 6);
    try std.testing.expect((getField(insts[1], 116, 6) & (@as(u32, 1) << @intCast(s2r_bar))) != 0);
}

test "a MOVM reads only its source and gets its own scoreboard" {
    // MOVM writes nothing above bit 31 except its mode field, so the ALU rule read TWO
    // phantom R0 sources on it. Its class is DecoupledAgu, so its single result register
    // needs a barrier of its own.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(0, 20, .{}); // R0 <- global, variable latency
    insts[1] = encode.movm(8, 4, .{}); // transposes R4 into R8
    insts[2] = encode.iadd3(1, 0, 0, .{}); // the REAL consumer of R0
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldg_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldg_bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6)); // no phantom wait
    try std.testing.expect(getField(insts[1], 110, 3) < 6); // MOVM got a real scoreboard
    try std.testing.expect((getField(insts[2], 116, 6) & (@as(u32, 1) << @intCast(ldg_bar))) != 0);
}

test "a MOVM waits on the producer of the fragment it transposes" {
    var insts: [3]Inst = undefined;
    insts[0] = encode.ldsm(4, 2, .x1, false, .{}); // R4 <- one fragment
    insts[1] = encode.movm(8, 4, .{}); // transposes R4
    insts[2] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ldsm_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ldsm_bar < 6);
    try std.testing.expect((getField(insts[1], 116, 6) & (@as(u32, 1) << @intCast(ldsm_bar))) != 0);
}

// The SOURCE-SPAN tests. Each puts a variable-latency producer into a register that is
// INSIDE the operand's run but NOT at its first position, then the instruction under test,
// then asserts the instruction waits on that producer's scoreboard. A test that only fills
// the first register of a run proves nothing: `readsSrc` already named that one, which is
// exactly how the gap survived the phantom-register audit.
//
// Each test also puts a second producer ONE PAST the run and asserts the instruction does
// NOT wait on it. Over-waiting is not merely wasteful: the wait FREES that scoreboard, so
// the instruction that really consumes the register emits no wait of its own and reads it
// stale. There are only six scoreboards.

test "a b64 STG waits on the producer of the HIGH half of its data pair" {
    // `memTypeOf` gives every POINTER value a b64 store, so this shape is emitted today. A
    // pointer pair whose two halves come from two separate LDCs has a live producer on the
    // HIGH half alone, and the data field at bit 32 names only the LOW half. The store then
    // wrote a stale address dword to memory.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldc(9, 0, 0x14c, .{}); // R9 = the hi half of the data pair R8:R9
    insts[1] = encode.ldc(11, 0, 0x150, .{}); // R11 is PAST the pair
    insts[2] = encode.stg(4, 8, .b64, .{}); // [R4:R5] = R8:R9
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const hi_bar = getField(insts[0], 110, 3);
    const past_bar = getField(insts[1], 110, 3);
    try std.testing.expect(hi_bar < 6 and past_bar < 6);
    const wait = getField(insts[2], 116, 6);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(hi_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(past_bar)));
}

test "the STG data block follows the memory type, from b32 through b128" {
    // A b128 store reads (data .. data+3) and a b32 store reads data alone. Reading the
    // width off the 3-bit memory type at bits 73..75 is what keeps both exact: a fixed
    // span of 4 would make every 32-bit store wait on three registers it never reads.
    var wide: [4]Inst = undefined;
    wide[0] = encode.ldgU32(11, 4, .{}); // the LAST register of the b128 block R8..R11
    wide[1] = encode.ldgU32(12, 4, .{}); // R12 is PAST the block
    wide[2] = encode.stg(6, 8, .b128, .{});
    wide[3] = encode.exit(.{});
    scheduleBlocks(&wide, &.{0});
    const last_bar = getField(wide[0], 110, 3);
    const past_bar = getField(wide[1], 110, 3);
    try std.testing.expect(last_bar < 6 and past_bar < 6);
    const wide_wait = getField(wide[2], 116, 6);
    try std.testing.expect((wide_wait & (@as(u32, 1) << @intCast(last_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wide_wait & (@as(u32, 1) << @intCast(past_bar)));

    var narrow: [3]Inst = undefined;
    narrow[0] = encode.ldgU32(9, 4, .{}); // R9 is PAST a b32 store's single data register
    narrow[1] = encode.stgU32(6, 8, .{});
    narrow[2] = encode.exit(.{});
    scheduleBlocks(&narrow, &.{0});
    try std.testing.expectEqual(@as(u32, 0), getField(narrow[1], 116, 6));
}

test "consuming a whole run FREES the scoreboard, not only its first register" {
    // The wait scan and `clearReadRegs` must cover the SAME registers. A register that is
    // waited on but never cleared keeps its tag, so its scoreboard never returns to the
    // free pool and the next producer has to take a different one. With only six
    // scoreboards, a block that leaks them this way drains far more often than it needs to.
    //
    // The b64 store below waits on the producer of R9, the HIGH half of its data pair, so
    // the tag on R9 must clear and the scoreboard must be free for the next load to reuse.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(9, 30, .{}); // R9 = the hi half of the data pair R8:R9
    insts[1] = encode.stg(4, 8, .b64, .{}); // [R4:R5] = R8:R9, and it waits on R9
    insts[2] = encode.ldgU32(12, 30, .{}); // the next producer, which should REUSE it
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const first = getField(insts[0], 110, 3);
    try std.testing.expect(first < 6);
    try std.testing.expect((getField(insts[1], 116, 6) & (@as(u32, 1) << @intCast(first))) != 0);
    try std.testing.expectEqual(first, getField(insts[2], 110, 3));
}
test "a b64 STS waits on the producer of the HIGH half of its data pair" {
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(9, 30, .{}); // R9 = the hi half of the data pair R8:R9
    insts[1] = encode.ldgU32(10, 30, .{}); // R10 is PAST the pair
    insts[2] = encode.sts(4, 8, .b64, .{}); // shared[R4] = R8:R9
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const hi_bar = getField(insts[0], 110, 3);
    const past_bar = getField(insts[1], 110, 3);
    const wait = getField(insts[2], 116, 6);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(hi_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(past_bar)));
}

test "a 64-bit atomic reads a register PAIR for its data, a 32-bit one reads one register" {
    // The atomic width is the WIDER 4-bit field at bits 73..76, not the memory type a load
    // or a store carries, and the two hold different values. A u64 add reads (data,
    // data+1).
    var wide: [4]Inst = undefined;
    wide[0] = encode.ldgU32(9, 30, .{}); // the hi half of the data pair R8:R9
    wide[1] = encode.ldgU32(10, 30, .{}); // R10 is PAST the pair
    wide[2] = encode.atomg(20, 4, 8, .add, .u64, .{});
    wide[3] = encode.exit(.{});
    scheduleBlocks(&wide, &.{0});
    const hi_bar = getField(wide[0], 110, 3);
    const past_bar = getField(wide[1], 110, 3);
    const wide_wait = getField(wide[2], 116, 6);
    try std.testing.expect((wide_wait & (@as(u32, 1) << @intCast(hi_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wide_wait & (@as(u32, 1) << @intCast(past_bar)));

    var narrow: [3]Inst = undefined;
    narrow[0] = encode.ldgU32(9, 30, .{}); // R9 is PAST a 32-bit atomic's single data register
    narrow[1] = encode.atomg(20, 4, 8, .add, .u32, .{});
    narrow[2] = encode.exit(.{});
    scheduleBlocks(&narrow, &.{0});
    try std.testing.expectEqual(@as(u32, 0), getField(narrow[1], 116, 6));
}

test "a 64-bit compare-and-swap reads a PAIR for both its compare operand and its swap data" {
    var insts: [5]Inst = undefined;
    insts[0] = encode.ldgU32(9, 30, .{}); // the hi half of the compare pair R8:R9
    insts[1] = encode.ldgU32(13, 30, .{}); // the hi half of the swap pair R12:R13
    insts[2] = encode.ldgU32(14, 30, .{}); // R14 is PAST the swap pair
    insts[3] = encode.atomgCas(20, 4, 8, 12, .u64, .{});
    insts[4] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const cmp_bar = getField(insts[0], 110, 3);
    const swap_bar = getField(insts[1], 110, 3);
    const past_bar = getField(insts[2], 110, 3);
    const wait = getField(insts[3], 116, 6);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(cmp_bar))) != 0);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(swap_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(past_bar)));
}

test "a multi-component AST waits on a producer INSIDE its data block" {
    // AST stores `comps` consecutive registers from the data field at bit 32, and bits
    // 74..75 hold comps - 1. Every call site passes 1 today, so this is not live, but a
    // three-component output store waited only on the producer of the first register and
    // sent the other two to the attribute STALE.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(10, 30, .{}); // the LAST register of the data block R8..R10
    insts[1] = encode.ldgU32(11, 30, .{}); // R11 is PAST the block
    insts[2] = encode.ast(encode.ATTR_POSITION, 8, 3, .{});
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const last_bar = getField(insts[0], 110, 3);
    const past_bar = getField(insts[1], 110, 3);
    try std.testing.expect(last_bar < 6 and past_bar < 6);
    const wait = getField(insts[2], 116, 6);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(last_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(past_bar)));
}

test "a multi-component ALD tags EVERY register it fills" {
    // The destination side of the same field. An ALD of three components fills R4..R6 under
    // ONE write barrier, so a consumer of R5 or R6 must wait on it too.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ald(4, encode.ATTR_GENERIC0, 3, .{}); // R4..R6
    insts[1] = encode.iadd3(20, 6, 6, .{}); // consumes R6, the LAST of the run
    insts[2] = encode.iadd3(21, 7, 7, .{}); // R7 is PAST the run
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const ald_bar = getField(insts[0], 110, 3);
    try std.testing.expect(ald_bar < 6);
    try std.testing.expect((getField(insts[1], 116, 6) & (@as(u32, 1) << @intCast(ald_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[2], 116, 6));
}

test "a TEX waits on a producer INSIDE its coordinate block" {
    // A 2D sample reads (u, v) at coord and coord+1, and the source field names only coord.
    // NOT live today: the isel materializes every register of the block with a MOV, an
    // FADD, an F2I or an FFMA right before the sample, and all of those are fixed-latency,
    // so no register of the block carries a tag when the TEX issues. The span removes the
    // dependence on that isel property.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(9, 30, .{}); // v, inside the coordinate block R8..R9
    insts[1] = encode.ldgU32(10, 30, .{}); // R10 is PAST the block
    insts[2] = encode.tex2d(4, 8, 20, .{}); // dst R4..R7, coord R8:R9, handle R20
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const v_bar = getField(insts[0], 110, 3);
    const past_bar = getField(insts[1], 110, 3);
    try std.testing.expect(v_bar < 6 and past_bar < 6);
    const wait = getField(insts[2], 116, 6);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(v_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(past_bar)));
}

test "the TEX coordinate block follows the DIMENSION field" {
    // A 3D target reads (u, v, w) and a 2D ARRAY target reads (layer, u, v), both three
    // registers, where a 2D target reads two. The count comes from the dimension at bits
    // 61..63, so a 2D sample is not made to wait on a third register it never reads.
    const dims = [_]u8{ encode.TexDim.dim_3d, encode.TexDim.array_2d };
    for (dims) |dim| {
        var insts: [4]Inst = undefined;
        insts[0] = encode.ldgU32(10, 30, .{}); // the THIRD coordinate register R8..R10
        insts[1] = encode.ldgU32(11, 30, .{}); // R11 is PAST the block
        insts[2] = encode.tex(4, 8, 20, dim, .{});
        insts[3] = encode.exit(.{});
        scheduleBlocks(&insts, &.{0});

        const third_bar = getField(insts[0], 110, 3);
        const past_bar = getField(insts[1], 110, 3);
        const wait = getField(insts[2], 116, 6);
        try std.testing.expect((wait & (@as(u32, 1) << @intCast(third_bar))) != 0);
        try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(past_bar)));
    }
}

test "a depth-compare TEX waits on the producer of the DREF beside its handle" {
    // NAK packs src1 as [handle, dref], so a z_cmpr TEX reads handle + 1 as well. Bit 78
    // selects the depth compare, and the span reads it there.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(21, 30, .{}); // the dref at handle + 1
    insts[1] = encode.ldgU32(22, 30, .{}); // R22 is PAST src1
    insts[2] = encode.texShadow(4, 8, 20, encode.TexDim.dim_2d, .{});
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const dref_bar = getField(insts[0], 110, 3);
    const past_bar = getField(insts[1], 110, 3);
    try std.testing.expect(dref_bar < 6 and past_bar < 6);
    const wait = getField(insts[2], 116, 6);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(dref_bar))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(past_bar)));
}

test "an explicit-LOD texture op waits on the producer of the LOD beside its handle" {
    // TEX.LL and TLD both read src1 as [handle, lod]. The LOD mode is split across bit 59
    // and bits 87..89, and only the Lod value adds that second register: an implicit-LOD
    // TEX must not wait on handle + 1.
    const ops = [_]Inst{
        encode.texLod(4, 8, 20, encode.TexDim.dim_2d, .{}),
        encode.tld(4, 8, 20, encode.TexDim.dim_2d, .{}),
    };
    for (ops) |op| {
        var insts: [4]Inst = undefined;
        insts[0] = encode.ldgU32(21, 30, .{}); // the LOD at handle + 1
        insts[1] = encode.ldgU32(22, 30, .{}); // R22 is PAST src1
        insts[2] = op;
        insts[3] = encode.exit(.{});
        scheduleBlocks(&insts, &.{0});

        const lod_bar = getField(insts[0], 110, 3);
        const past_bar = getField(insts[1], 110, 3);
        const wait = getField(insts[2], 116, 6);
        try std.testing.expect((wait & (@as(u32, 1) << @intCast(lod_bar))) != 0);
        try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(past_bar)));
    }

    var auto: [3]Inst = undefined;
    auto[0] = encode.ldgU32(21, 30, .{}); // an implicit-LOD TEX reads its handle ALONE
    auto[1] = encode.tex2d(4, 8, 20, .{});
    auto[2] = encode.exit(.{});
    scheduleBlocks(&auto, &.{0});
    try std.testing.expectEqual(@as(u32, 0), getField(auto[1], 116, 6));
}

test "a TLD4 reads its handle ALONE, because bits 87..88 hold the gather component" {
    // A gather has no LOD mode: bits 87..88 hold the component select instead, and the
    // component values 1 and 2 read back as the Bias and Clamp lod modes. Decoding a LOD
    // mode there would make the gather wait on handle + 1, free that scoreboard, and leave
    // the real consumer of the register with no wait of its own.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(21, 30, .{}); // handle + 1, which a gather does NOT read
    insts[1] = encode.tld4(4, 8, 20, encode.TexDim.dim_2d, 1, .{}); // component 1 = "Bias"
    insts[2] = encode.iadd3(24, 21, 21, .{}); // the REAL consumer of R21
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const bar = getField(insts[0], 110, 3);
    try std.testing.expect(bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6));
    try std.testing.expect((getField(insts[2], 116, 6) & (@as(u32, 1) << @intCast(bar))) != 0);
}

test "every HMMA tile reads its OWN A and B fragment run lengths" {
    // A fragment is one LANE's slice of a matrix: `rows * cols` fp16 elements over the 32
    // lanes of the warp, two of them packed into each 32-bit register. So A is `m` by `k`
    // and B is `k` by `n`, and the tile changes both lengths. The counts below reproduce
    // the register vectors the PTX "Matrix Fragments" section gives for each shape.
    //
    // Each case puts a producer on the LAST register of the A run and another ONE PAST it,
    // and the same pair around the B run. A run that is too SHORT fails the first assert
    // and one that is too LONG fails the second, so neither direction can slip through.
    // `m16n8k4` is the tf32 shape and the encoder writes an fp16 source type, so its B
    // operand is half a register and floors to one.
    const cases = [_]struct { size: encode.HmmaSize, a: u8, b: u8 }{
        .{ .size = .m16n8k8, .a = 2, .b = 1 },
        .{ .size = .m16n8k16, .a = 4, .b = 2 },
        .{ .size = .m16n8k4, .a = 1, .b = 1 },
    };
    const a_base: u8 = 8;
    const b_base: u8 = 4;
    for (cases) |c| {
        var insts: [6]Inst = undefined;
        insts[0] = encode.ldgU32(a_base + c.a - 1, 30, .{}); // the LAST register of the A run
        insts[1] = encode.ldgU32(a_base + c.a, 30, .{}); // one PAST the A run
        insts[2] = encode.ldgU32(b_base + c.b - 1, 30, .{}); // the LAST register of the B run
        insts[3] = encode.ldgU32(b_base + c.b, 30, .{}); // one PAST the B run
        insts[4] = encode.hmma(24, a_base, b_base, 16, c.size, .f32, .{});
        insts[5] = encode.exit(.{});
        scheduleBlocks(&insts, &.{0});

        const wait = getField(insts[4], 116, 6);
        for ([_]usize{ 0, 2 }) |inside| {
            const bar = getField(insts[inside], 110, 3);
            try std.testing.expect(bar < 6);
            try std.testing.expect((wait & (@as(u32, 1) << @intCast(bar))) != 0);
        }
        for ([_]usize{ 1, 3 }) |past| {
            const bar = getField(insts[past], 110, 3);
            try std.testing.expect(bar < 6);
            try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(bar)));
        }
    }
}

test "an HMMA waits on a producer INSIDE its C fragment run, at the length of the RESULT type" {
    // C has the shape and the element type of D, both selected by bit 76, so an fp32
    // accumulator is 4 registers per lane and an fp16 one is 2. The destination is kept
    // clear of the C run here, so only the SOURCE scan can produce the wait.
    var wide: [4]Inst = undefined;
    wide[0] = encode.ldgU32(19, 30, .{}); // the LAST register of the fp32 C run R16..R19
    wide[1] = encode.ldgU32(20, 30, .{}); // R20 is PAST the run
    wide[2] = encode.hmma(24, 8, 4, 16, .m16n8k16, .f32, .{});
    wide[3] = encode.exit(.{});
    scheduleBlocks(&wide, &.{0});
    const wide_last = getField(wide[0], 110, 3);
    const wide_past = getField(wide[1], 110, 3);
    const wide_wait = getField(wide[2], 116, 6);
    try std.testing.expect((wide_wait & (@as(u32, 1) << @intCast(wide_last))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wide_wait & (@as(u32, 1) << @intCast(wide_past)));

    var packed_c: [4]Inst = undefined;
    packed_c[0] = encode.ldgU32(17, 30, .{}); // the LAST register of the fp16 C run R16..R17
    packed_c[1] = encode.ldgU32(18, 30, .{}); // R18 is PAST the run
    packed_c[2] = encode.hmma(24, 8, 4, 16, .m16n8k16, .f16, .{});
    packed_c[3] = encode.exit(.{});
    scheduleBlocks(&packed_c, &.{0});
    const packed_last = getField(packed_c[0], 110, 3);
    const packed_past = getField(packed_c[1], 110, 3);
    const packed_wait = getField(packed_c[2], 116, 6);
    try std.testing.expect((packed_wait & (@as(u32, 1) << @intCast(packed_last))) != 0);
    try std.testing.expectEqual(@as(u32, 0), packed_wait & (@as(u32, 1) << @intCast(packed_past)));
}

test "every IMMA tile reads its OWN A and B fragment run lengths, at its own operand width" {
    // The same per-lane count as an HMMA fragment, with 8-bit or 4-bit elements: four or
    // eight of them pack into a register. The tile shapes are NAK's `ImmaSize` table and
    // the counts reproduce the PTX "Matrix Fragments" register vectors. `m16n8k32` appears
    // twice because it is the one tile that accepts either width, and the width HALVES both
    // runs.
    //
    // Each case puts a producer on the LAST register of each run and another ONE PAST it,
    // so a length that is too short and one that is too long both fail.
    const cases = [_]struct { size: encode.ImmaSize, four_bit: bool, a: u8, b: u8 }{
        .{ .size = .m8n8k16, .four_bit = false, .a = 1, .b = 1 },
        .{ .size = .m8n8k32, .four_bit = true, .a = 1, .b = 1 },
        .{ .size = .m16n8k16, .four_bit = false, .a = 2, .b = 1 },
        .{ .size = .m16n8k32, .four_bit = false, .a = 4, .b = 2 },
        .{ .size = .m16n8k32, .four_bit = true, .a = 2, .b = 1 },
        .{ .size = .m16n8k64, .four_bit = true, .a = 4, .b = 2 },
    };
    const a_base: u8 = 8;
    const b_base: u8 = 16;
    for (cases) |c| {
        var insts: [6]Inst = undefined;
        insts[0] = encode.ldgU32(a_base + c.a - 1, 30, .{}); // the LAST register of the A run
        insts[1] = encode.ldgU32(a_base + c.a, 30, .{}); // one PAST the A run
        insts[2] = encode.ldgU32(b_base + c.b - 1, 30, .{}); // the LAST register of the B run
        insts[3] = encode.ldgU32(b_base + c.b, 30, .{}); // one PAST the B run
        insts[4] = encode.imma(40, a_base, b_base, 24, c.size, .{ .signed = true, .four_bit = c.four_bit }, .{ .signed = true, .four_bit = c.four_bit }, false, .{});
        insts[5] = encode.exit(.{});
        scheduleBlocks(&insts, &.{0});

        const wait = getField(insts[4], 116, 6);
        for ([_]usize{ 0, 2 }) |inside| {
            const bar = getField(insts[inside], 110, 3);
            try std.testing.expect(bar < 6);
            try std.testing.expect((wait & (@as(u32, 1) << @intCast(bar))) != 0);
        }
        for ([_]usize{ 1, 3 }) |past| {
            const bar = getField(insts[past], 110, 3);
            try std.testing.expect(bar < 6);
            try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(bar)));
        }
    }
}

test "the IMMA width flags are read PER OPERAND, so a mixed-width multiply gets two run lengths" {
    // Bit 83 is "A is 4-bit" and bit 84 is "B is 4-bit". An `m16n8k32` tile accepts either
    // width for either operand, so a multiply of 8-bit activations by 4-bit weights sets
    // ONE of the two bits. Reading the wrong bit for an operand gives it the other
    // operand's run length: here A is 8-bit and reads FOUR registers while B is 4-bit and
    // reads ONE, and no test with a single shared width can tell the two bits apart.
    var insts: [5]Inst = undefined;
    insts[0] = encode.ldgU32(11, 30, .{}); // the LAST register of the 8-bit A run R8..R11
    insts[1] = encode.ldgU32(12, 30, .{}); // R12 is PAST the A run
    insts[2] = encode.ldgU32(17, 30, .{}); // R17 is PAST the single-register 4-bit B run
    insts[3] = encode.imma(40, 8, 16, 20, .m16n8k32, .{ .signed = false }, .{ .signed = true, .four_bit = true }, false, .{});
    insts[4] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const a_last = getField(insts[0], 110, 3);
    const a_past = getField(insts[1], 110, 3);
    const b_past = getField(insts[2], 110, 3);
    try std.testing.expect(a_last < 6 and a_past < 6 and b_past < 6);
    const wait = getField(insts[3], 116, 6);
    try std.testing.expect((wait & (@as(u32, 1) << @intCast(a_last))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(a_past)));
    try std.testing.expectEqual(@as(u32, 0), wait & (@as(u32, 1) << @intCast(b_past)));
}

test "an IMMA waits on a producer INSIDE its int32 accumulator run" {
    // The accumulator is int32 whatever the input width is, so an m16n8 tile reads 4
    // registers and an m8n8 tile reads 2. The destination is kept clear of the C run, so
    // only the SOURCE scan can produce the wait.
    var wide: [4]Inst = undefined;
    wide[0] = encode.ldgU32(23, 30, .{}); // the LAST register of the m16n8 C run R20..R23
    wide[1] = encode.ldgU32(24, 30, .{}); // R24 is PAST the run
    wide[2] = encode.imma(40, 8, 16, 20, .m16n8k16, .{ .signed = true }, .{ .signed = true }, false, .{});
    wide[3] = encode.exit(.{});
    scheduleBlocks(&wide, &.{0});
    const wide_last = getField(wide[0], 110, 3);
    const wide_past = getField(wide[1], 110, 3);
    const wide_wait = getField(wide[2], 116, 6);
    try std.testing.expect((wide_wait & (@as(u32, 1) << @intCast(wide_last))) != 0);
    try std.testing.expectEqual(@as(u32, 0), wide_wait & (@as(u32, 1) << @intCast(wide_past)));

    var small: [4]Inst = undefined;
    small[0] = encode.ldgU32(21, 30, .{}); // the LAST register of the m8n8 C run R20..R21
    small[1] = encode.ldgU32(22, 30, .{}); // R22 is PAST the run
    small[2] = encode.imma(40, 8, 16, 20, .m8n8k16, .{ .signed = true }, .{ .signed = true }, false, .{});
    small[3] = encode.exit(.{});
    scheduleBlocks(&small, &.{0});
    const small_last = getField(small[0], 110, 3);
    const small_past = getField(small[1], 110, 3);
    const small_wait = getField(small[2], 116, 6);
    try std.testing.expect((small_wait & (@as(u32, 1) << @intCast(small_last))) != 0);
    try std.testing.expectEqual(@as(u32, 0), small_wait & (@as(u32, 1) << @intCast(small_past)));
}

test "an LDSM reads ONE address register, not a run" {
    // Every other operand of a tensor op is a run, but the LDSM address is a 32-bit offset
    // into the shared window in ONE register, the same form LDS and STS take. A span of 2
    // here would wait on an unrelated register's producer AND free its scoreboard, so the
    // instruction that really consumes that register would emit no wait and read stale.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(9, 30, .{}); // addr + 1, which the LDSM does NOT read
    insts[1] = encode.ldsm(12, 8, .x4, false, .{});
    insts[2] = encode.iadd3(24, 9, 9, .{}); // the REAL consumer of R9
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const bar = getField(insts[0], 110, 3);
    try std.testing.expect(bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6));
    try std.testing.expect((getField(insts[2], 116, 6) & (@as(u32, 1) << @intCast(bar))) != 0);
}

test "a MOVM reads ONE source register, not a run" {
    // MOVM transposes one 8 by 8 fragment of 16-bit elements, which gives each lane 32
    // bits: one register in and one out.
    var insts: [4]Inst = undefined;
    insts[0] = encode.ldgU32(9, 30, .{}); // src + 1, which the MOVM does NOT read
    insts[1] = encode.movm(12, 8, .{});
    insts[2] = encode.iadd3(24, 9, 9, .{}); // the REAL consumer of R9
    insts[3] = encode.exit(.{});
    scheduleBlocks(&insts, &.{0});

    const bar = getField(insts[0], 110, 3);
    try std.testing.expect(bar < 6);
    try std.testing.expectEqual(@as(u32, 0), getField(insts[1], 116, 6));
    try std.testing.expect((getField(insts[2], 116, 6) & (@as(u32, 1) << @intCast(bar))) != 0);
}
