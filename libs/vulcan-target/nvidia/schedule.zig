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
        opcode == 0x321 or opcode == 0x326; // ALD, IPA
}

/// Whether `opcode` writes a destination GPR at bits 16..23 (so the scheduler can
/// associate a scoreboard with it). Stores and control flow do not.
fn writesDst(opcode: u32) bool {
    return switch (opcode) {
        0x986, 0x947, 0x94d => false, // STG, BRA, EXIT
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
/// in place.
pub fn schedule(insts: []Inst) void {
    // scoreboard_of[reg] = scoreboard index + 1 (0 = the register has no in-flight
    // variable-latency producer).
    var scoreboard_of = [_]u8{0} ** 256;
    var free_mask: u8 = (1 << num_scoreboards) - 1; // scoreboards 0..5 free

    for (insts) |*inst| {
        const opcode = getField(inst.*, 0, 12);

        // Which scoreboards must this instruction wait on? Conservatively, any of
        // its source-register fields whose register has an in-flight producer.
        var wait: u32 = 0;
        inline for (.{ 24, 32, 64 }) |pos| {
            const reg = getField(inst.*, pos, 8);
            if (reg != RZ and scoreboard_of[reg] != 0) wait |= @as(u32, 1) << @intCast(scoreboard_of[reg] - 1);
        }
        if (wait != 0) {
            setField(inst, 116, 6, getField(inst.*, 116, 6) | wait);
            // Freeing: once awaited, the result is ready, so the scoreboard is free
            // and no register depending on it needs to wait again.
            var sb: u3 = 0;
            while (sb < num_scoreboards) : (sb += 1) {
                if (wait & (@as(u32, 1) << sb) != 0) {
                    free_mask |= @as(u8, 1) << sb;
                    for (&scoreboard_of) |*s| if (s.* == @as(u8, sb) + 1) {
                        s.* = 0;
                    };
                }
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
                scoreboard_of[dst] = @as(u8, sb) + 1;
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
