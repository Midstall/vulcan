//! aarch64 ldr/str -> ldp/stp peephole, decode + fusion-predicate stage (Task 2 of the ldp/stp
//! peephole). These two functions are PURE: `decodeMem` recognizes the bounded set of
//! offset-form ldr/str encodings the isel emits (see encode.zig), and `tryFuse` decides whether
//! two adjacent decoded memory ops may be combined into a single ldp/stp. Neither function
//! walks or mutates a code buffer. Wiring this into an actual pass over emitted code is Task 3.
//!
//! A wrong decode or a wrong fuse here would silently miscompile once Task 3's pass trusts it,
//! so `decodeMem` must be the EXACT inverse of the isel's encoders (verified against
//! strOff/ldrOff/strW/ldrW/strFp/ldrFp/strQ/ldrQ in encode.zig, see the tests), and anything
//! that is not one of those exact forms (a different addressing mode, a pre/post-index form, a
//! non-memory op) must decode to null rather than guess.

const std = @import("std");
const encode = @import("encode.zig");
const isel = @import("isel.zig");
const Reg = encode.Reg;

/// The side-table row types the fusion pass remaps. Taken directly from the isel so a fusion
/// updates the exact structures `compileFunction` resolves branches and emits debug info from.
const Fixup = isel.Fixup;
const Reloc = isel.Reloc;
const LineEntry = isel.LineEntry;

/// The offset-form ldr/str encodings the aarch64 isel emits. Anything else is opaque to this
/// peephole and decodes to null.
pub const MemKind = enum { ldr_x, str_x, ldr_w, str_w, ldr_s, str_s, ldr_d, str_d, ldr_q, str_q };

/// A decoded memory instruction. `off` is the BYTE offset: the raw u12 scaled-imm field already
/// multiplied out by the element size, so two decoded insns can be compared directly regardless
/// of kind.
pub const MemInsn = struct { kind: MemKind, rt: u5, rn: u5, off: u32 };

/// The element size in bytes for `kind` (also the scale of its imm12 field).
fn sizeOf(kind: MemKind) u32 {
    return switch (kind) {
        .ldr_x, .str_x, .ldr_d, .str_d => 8,
        .ldr_w, .str_w, .ldr_s, .str_s => 4,
        .ldr_q, .str_q => 16,
    };
}

/// True for the `ldr_*` variants. A load writes `rt` from memory, whereas a store reads `rt`
/// into memory and never overwrites a register.
fn isLoad(kind: MemKind) bool {
    return switch (kind) {
        .ldr_x, .ldr_w, .ldr_s, .ldr_d, .ldr_q => true,
        .str_x, .str_w, .str_s, .str_d, .str_q => false,
    };
}

/// The top-10-bit class selector every recognized offset-form encoding uses (bits [31:22]);
/// the low 22 bits carry imm12/rn/rt and are zero in each base opcode below. Pre/post-index and
/// register-offset addressing modes for the same size/class live at a different bit-24 value
/// and so never collide with these classes.
const class_mask: u32 = 0xFFC00000;

/// Decode `word` if it is one of the recognized offset-form ldr/str encodings, else null.
/// Classes below are cross-checked one-for-one against encode.zig:
///   ldr_x/str_x -> ldrOff/strOff (imm12 scaled by 8)
///   ldr_w/str_w -> ldrW/strW     (imm12 scaled by 4)
///   ldr_s/str_s -> ldrFp/strFp(dbl=false) (imm12 scaled by 4)
///   ldr_d/str_d -> ldrFp/strFp(dbl=true)  (imm12 scaled by 8)
///   ldr_q/str_q -> ldrQ/strQ     (imm12 scaled by 16)
pub fn decodeMem(word: u32) ?MemInsn {
    const rt: u5 = @truncate(word & 0x1F);
    const rn: u5 = @truncate((word >> 5) & 0x1F);
    const imm12: u32 = (word >> 10) & 0xFFF;
    const kind: MemKind = switch (word & class_mask) {
        0xF9400000 => .ldr_x,
        0xF9000000 => .str_x,
        0xB9400000 => .ldr_w,
        0xB9000000 => .str_w,
        0xBD400000 => .ldr_s,
        0xBD000000 => .str_s,
        0xFD400000 => .ldr_d,
        0xFD000000 => .str_d,
        0x3DC00000 => .ldr_q,
        0x3D800000 => .str_q,
        else => return null,
    };
    return MemInsn{ .kind = kind, .rt = rt, .rn = rn, .off = imm12 * sizeOf(kind) };
}

/// Given two adjacent decoded mem insns (`a` then `b` in program order), return the fused
/// ldp/stp word if they may be paired, else null. Requires ALL of:
///  - `a.kind == b.kind` (same load/store, class, and size).
///  - `a.rn == b.rn` (same base register).
///  - `b.off == a.off + sizeOf(a.kind)` (`b` sits exactly one element above `a`).
///  - if `a`/`b` are loads: `a.rt != b.rt` (an ldp with equal destinations is UNPREDICTABLE) and
///    `a.rt != a.rn` (the first load must not overwrite the base register the second load still
///    needs). A fused ldp computes both addresses from the pre-instruction base, so this also
///    keeps the fused form execution-equivalent to the original two-instruction sequence.
///  - `a.off / sizeOf(a.kind) <= 63` (the ldp/stp imm7 is a signed 7-bit field). `a.off` is
///    derived from an unsigned u12 scaled-imm so it is always >= 0, making the range check just
///    this upper bound.
///
/// Task 1 only added GPR (x/w) offset-form ldp/stp encoders (`ldpOffX`/`stpOffX`/`ldpOffW`/
/// `stpOffW`). The scalar-float (s/d) and 128-bit vector (q) ldp/stp forms live at different,
/// not-yet-added-or-disasm-confirmed base opcodes (the SIMD&FP register-pair family), so
/// pairing those kinds is deliberately deferred: they still decode correctly above (so a caller
/// walking code with `decodeMem` treats them as opaque single instructions, not garbage), but
/// `tryFuse` returns null for them rather than emit an unvalidated encoding.
pub fn tryFuse(a: MemInsn, b: MemInsn) ?u32 {
    if (a.kind != b.kind) return null;
    if (a.rn != b.rn) return null;
    const size = sizeOf(a.kind);
    if (b.off != a.off + size) return null;
    if (isLoad(a.kind)) {
        if (a.rt == b.rt) return null;
        if (a.rt == a.rn) return null;
    }
    const imm7: u32 = a.off / size;
    if (imm7 > 63) return null;

    const rt1: Reg = @enumFromInt(a.rt);
    const rt2: Reg = @enumFromInt(b.rt);
    const rn: Reg = @enumFromInt(a.rn);
    const imm: i16 = @intCast(a.off);
    return switch (a.kind) {
        .ldr_x => encode.ldpOffX(rt1, rt2, rn, imm),
        .str_x => encode.stpOffX(rt1, rt2, rn, imm),
        .ldr_w => encode.ldpOffW(rt1, rt2, rn, imm),
        .str_w => encode.stpOffW(rt1, rt2, rn, imm),
        .ldr_s, .str_s, .ldr_d, .str_d, .ldr_q, .str_q => null,
    };
}

/// Fuse adjacent ldr/str pairs into ldp/stp in `words`, updating the side tables to the new
/// layout. Only fuses IMMEDIATELY-ADJACENT words where both decode as mem, `tryFuse` succeeds,
/// AND the SECOND word is not a block start (control flow must not be able to enter between the
/// pair). Fusion deletes the second word (a load/store, never a branch or reloc site) and
/// replaces the first with the fused ldp/stp. Mutates `words` (shrinks it) and remaps
/// `block_start`, `fixups`, `relocs`, `lines` in place.
///
/// Miscompile surface: the fixup-resolution loop in `compileFunction` runs AFTER this pass over
/// the remapped tables, so branch displacements are recomputed from the shrunk layout. A word
/// referenced by a block start, a fixup, or a reloc is never the deleted (second) member of a
/// pair, so its remapped index is always well-defined and points at a surviving word.
pub fn pairMemory(
    allocator: std.mem.Allocator,
    words: *std.ArrayList(u32),
    block_start: []usize, // WORD indices, remapped in place
    fixups: []Fixup, // .at WORD indices, remapped in place (.target block index unchanged)
    relocs: []Reloc, // .offset WORD indices, remapped in place
    lines: []LineEntry, // .offset BYTE offsets, remapped in place
) std.mem.Allocator.Error!void {
    const len = words.items.len;

    // Which word indices begin a block. A word is fusable-as-second only if it is NOT a block
    // start, so that control flow can never enter between the two members of a pair.
    var is_block_start = try allocator.alloc(bool, len);
    defer allocator.free(is_block_start);
    @memset(is_block_start, false);
    for (block_start) |bs| {
        std.debug.assert(bs <= len);
        if (bs < len) is_block_start[bs] = true; // a block start == len emits nothing to fuse into
    }

    // `removed_before[w]` = number of words deleted with index strictly before `w`, with the twist
    // that a deleted word's own slot records the count INCLUDING its own deletion. That makes a
    // surviving word `w` map to `w - removed_before[w]` and a deleted (second-of-pair) word collapse
    // onto the fused instruction that replaced its pair's first word. Length is `len + 1` so a
    // block start sitting exactly at the end (`== len`) still has a defined slot.
    var removed_before = try allocator.alloc(u32, len + 1);
    defer allocator.free(removed_before);

    var write: usize = 0; // in-place compaction cursor (always <= the read cursor `i`)
    var removed: u32 = 0;
    var i: usize = 0;
    while (i < len) {
        var did_fuse = false;
        // A pair needs a following word that is not a block start, and both members must decode as
        // memory ops that `tryFuse` accepts. Never chain a third word into the same pair (i += 2).
        if (i + 1 < len and !is_block_start[i + 1]) {
            if (decodeMem(words.items[i])) |a| {
                if (decodeMem(words.items[i + 1])) |b| {
                    if (tryFuse(a, b)) |fused| {
                        removed_before[i] = removed;
                        removed += 1;
                        removed_before[i + 1] = removed; // deleted slot collapses onto the fused word
                        words.items[write] = fused;
                        write += 1;
                        i += 2;
                        did_fuse = true;
                    }
                }
            }
        }
        if (!did_fuse) {
            removed_before[i] = removed;
            words.items[write] = words.items[i];
            write += 1;
            i += 1;
        }
    }
    removed_before[len] = removed;

    // The surviving count is exactly the original minus every deleted second-of-pair word.
    std.debug.assert(write == len - removed);

    // Remap the side tables into the shrunk layout. Each keyed index is a surviving word, so its
    // new index is `w - removed_before[w]` and lands below the new word count.
    for (block_start) |*bs| {
        bs.* -= removed_before[bs.*];
        std.debug.assert(bs.* <= write);
    }
    for (fixups) |*f| {
        f.at -= removed_before[f.at];
        std.debug.assert(f.at < write);
    }
    for (relocs) |*r| {
        r.offset -= removed_before[r.offset];
        std.debug.assert(r.offset < write);
    }
    for (lines) |*l| {
        // Line offsets are BYTES; a line that sat on a deleted word collapses onto the fused
        // instruction, which is acceptable for debug info.
        const w = l.offset / 4;
        l.offset = @intCast((w - removed_before[w]) * 4);
    }

    words.shrinkRetainingCapacity(write);
}

test "decodeMem recognizes each ldr/str offset form and rejects a non-memory word" {
    const t = std.testing;

    try t.expectEqual(MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 1, .off = 16 }, decodeMem(encode.ldrOff(.x0, .x1, 16)).?);
    try t.expectEqual(MemInsn{ .kind = .str_x, .rt = 2, .rn = 3, .off = 8 }, decodeMem(encode.strOff(.x2, .x3, 8)).?);
    try t.expectEqual(MemInsn{ .kind = .ldr_w, .rt = 4, .rn = 5, .off = 12 }, decodeMem(encode.ldrW(.x4, .x5, 12)).?);
    try t.expectEqual(MemInsn{ .kind = .str_w, .rt = 6, .rn = 7, .off = 4 }, decodeMem(encode.strW(.x6, .x7, 4)).?);
    try t.expectEqual(MemInsn{ .kind = .ldr_s, .rt = 0, .rn = 1, .off = 8 }, decodeMem(encode.ldrFp(.x0, .x1, 8, false)).?);
    try t.expectEqual(MemInsn{ .kind = .str_s, .rt = 2, .rn = 3, .off = 4 }, decodeMem(encode.strFp(.x2, .x3, 4, false)).?);
    try t.expectEqual(MemInsn{ .kind = .ldr_d, .rt = 4, .rn = 5, .off = 16 }, decodeMem(encode.ldrFp(.x4, .x5, 16, true)).?);
    try t.expectEqual(MemInsn{ .kind = .str_d, .rt = 6, .rn = 7, .off = 8 }, decodeMem(encode.strFp(.x6, .x7, 8, true)).?);
    try t.expectEqual(MemInsn{ .kind = .ldr_q, .rt = 0, .rn = 1, .off = 32 }, decodeMem(encode.ldrQ(.x0, .x1, 32)).?);
    try t.expectEqual(MemInsn{ .kind = .str_q, .rt = 2, .rn = 3, .off = 16 }, decodeMem(encode.strQ(.x2, .x3, 16)).?);

    // Non-memory ops: rejected outright.
    try t.expectEqual(@as(?MemInsn, null), decodeMem(encode.add64(.x0, .x1, .x2)));
    try t.expectEqual(@as(?MemInsn, null), decodeMem(encode.nop()));
    try t.expectEqual(@as(?MemInsn, null), decodeMem(0x14000000)); // b #0 (unconditional branch)

    // A different addressing mode for the same class: pre/post-index and unscaled-immediate
    // loads/stores clear bit 24, landing outside every class above (e.g. 0xF8400000 is the
    // 64-bit pre/post-index/unscaled family, distinct from ldr_x's 0xF9400000).
    try t.expectEqual(@as(?MemInsn, null), decodeMem(0xF8400400));
    try t.expectEqual(@as(?MemInsn, null), decodeMem(0xB8400400));
}

test "tryFuse pairs two adjacent consecutive-offset x loads into ldp" {
    const a = MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 2, .off = 0 };
    const b = MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 2, .off = 8 };
    const fused = tryFuse(a, b);
    try std.testing.expectEqual(encode.ldpOffX(.x0, .x1, .x2, 0), fused.?);

    const disasm = @import("disasm.zig");
    const alloc = std.testing.allocator;
    const s = try disasm.one(alloc, fused.?);
    defer alloc.free(s);
    try std.testing.expectEqualStrings("ldp x0, x1, [x2]", s);
}

test "tryFuse pairs two adjacent consecutive-offset w stores into stp" {
    const a = MemInsn{ .kind = .str_w, .rt = 4, .rn = 5, .off = 0 };
    const b = MemInsn{ .kind = .str_w, .rt = 6, .rn = 5, .off = 4 };
    try std.testing.expectEqual(encode.stpOffW(.x4, .x6, .x5, 0), tryFuse(a, b).?);
}

test "tryFuse allows a store pair even when rt equals rn (the load-only base-clobber rule)" {
    // Storing x2 through base x2 is fine (the store never writes a register), unlike the load
    // case where a.rt == a.rn would mean the first load overwrites the shared base.
    const a = MemInsn{ .kind = .str_x, .rt = 2, .rn = 2, .off = 0 };
    const b = MemInsn{ .kind = .str_x, .rt = 3, .rn = 2, .off = 8 };
    try std.testing.expectEqual(encode.stpOffX(.x2, .x3, .x2, 0), tryFuse(a, b).?);
}

test "tryFuse returns null for scalar-float and vector kinds (no validated fp/q ldp/stp encoder yet)" {
    const s1 = MemInsn{ .kind = .ldr_s, .rt = 0, .rn = 1, .off = 0 };
    const s2 = MemInsn{ .kind = .ldr_s, .rt = 2, .rn = 1, .off = 4 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(s1, s2));

    const d1 = MemInsn{ .kind = .ldr_d, .rt = 0, .rn = 1, .off = 0 };
    const d2 = MemInsn{ .kind = .ldr_d, .rt = 2, .rn = 1, .off = 8 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(d1, d2));

    const q1 = MemInsn{ .kind = .ldr_q, .rt = 0, .rn = 1, .off = 0 };
    const q2 = MemInsn{ .kind = .ldr_q, .rt = 2, .rn = 1, .off = 16 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(q1, q2));
}

/// Build an owned `std.ArrayList(u32)` from a literal slice so a test can hand `pairMemory` a
/// word buffer it will shrink in place. The caller deinits.
fn wordsFrom(allocator: std.mem.Allocator, init: []const u32) !std.ArrayList(u32) {
    var w: std.ArrayList(u32) = .empty;
    try w.appendSlice(allocator, init);
    return w;
}

test "pairMemory fuses an adjacent x-load pair and shrinks the word count by one" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.ldrOff(.x0, .x2, 0),
        encode.ldrOff(.x1, .x2, 8),
        encode.ret(),
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{0};

    try pairMemory(alloc, &words, &block_start, &.{}, &.{}, &.{});

    try std.testing.expectEqual(@as(usize, 2), words.items.len);
    try std.testing.expectEqual(encode.ldpOffX(.x0, .x1, .x2, 0), words.items[0]);
    try std.testing.expectEqual(encode.ret(), words.items[1]);
    // The fused word is an ldp, which is opaque to the single-op decoder.
    try std.testing.expectEqual(@as(?MemInsn, null), decodeMem(words.items[0]));
}

test "pairMemory does not fuse across a block boundary" {
    const alloc = std.testing.allocator;
    const original = [_]u32{
        encode.ldrOff(.x0, .x2, 0),
        encode.ldrOff(.x1, .x2, 8),
        encode.ret(),
    };
    var words = try wordsFrom(alloc, &original);
    defer words.deinit(alloc);
    // The second load starts block 1, so control flow can enter between the pair: no fusion.
    var block_start = [_]usize{ 0, 1 };

    try pairMemory(alloc, &words, &block_start, &.{}, &.{}, &.{});

    try std.testing.expectEqualSlices(u32, &original, words.items);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, &block_start);
}

test "pairMemory remaps a branch fixup and block_start after a fused pair" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.ldrOff(.x0, .x2, 0),
        encode.ldrOff(.x1, .x2, 8),
        encode.b(0), // fixup site at word index 2
        encode.ret(), // block 1 starts here (word index 3)
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{ 0, 3 };
    var fixups = [_]Fixup{.{ .at = 2, .target = 1 }};

    try pairMemory(alloc, &words, &block_start, &fixups, &.{}, &.{});

    try std.testing.expectEqual(@as(usize, 3), words.items.len);
    try std.testing.expectEqual(@as(usize, 1), fixups[0].at); // 2 - 1 deleted word
    try std.testing.expectEqual(@as(u32, 1), fixups[0].target); // block index unchanged
    try std.testing.expectEqual(@as(usize, 0), block_start[0]);
    try std.testing.expectEqual(@as(usize, 2), block_start[1]); // 3 - 1 deleted word
}

test "pairMemory is a no-op leaving words byte-identical when nothing is fusable" {
    const alloc = std.testing.allocator;
    // Two x-loads at NON-consecutive offsets (0 and 16): not a fusable pair.
    const original = [_]u32{
        encode.ldrOff(.x0, .x2, 0),
        encode.ldrOff(.x1, .x2, 16),
        encode.ret(),
    };
    var words = try wordsFrom(alloc, &original);
    defer words.deinit(alloc);
    var block_start = [_]usize{0};
    var fixups = [_]Fixup{.{ .at = 2, .target = 0 }};
    var relocs = [_]Reloc{.{ .offset = 2, .symbol = "x" }};
    var lines = [_]LineEntry{.{ .offset = 8, .line = 7 }};

    try pairMemory(alloc, &words, &block_start, &fixups, &relocs, &lines);

    try std.testing.expectEqualSlices(u32, &original, words.items);
    try std.testing.expectEqualSlices(usize, &.{0}, &block_start);
    try std.testing.expectEqual(@as(usize, 2), fixups[0].at);
    try std.testing.expectEqual(@as(usize, 2), relocs[0].offset);
    try std.testing.expectEqual(@as(u32, 8), lines[0].offset);
}

test "tryFuse rejects non-consecutive, different-base, rt1==rn, rt1==rt2 (load), mismatched-size, out-of-range" {
    const base = MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 2, .off = 0 };

    // Non-consecutive offset: b sits two elements above a, not one.
    try std.testing.expectEqual(@as(?u32, null), tryFuse(base, MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 2, .off = 16 }));

    // Different base register.
    try std.testing.expectEqual(@as(?u32, null), tryFuse(base, MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 3, .off = 8 }));

    // a.rt == a.rn: the first load would overwrite the base the second load still needs.
    const clobbers_base = MemInsn{ .kind = .ldr_x, .rt = 2, .rn = 2, .off = 0 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(clobbers_base, MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 2, .off = 8 }));

    // a.rt == b.rt: an ldp with equal destinations is UNPREDICTABLE.
    try std.testing.expectEqual(@as(?u32, null), tryFuse(base, MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 2, .off = 8 }));

    // Mismatched kind/size: an x-load cannot fuse with a w-load even at consecutive offsets.
    try std.testing.expectEqual(@as(?u32, null), tryFuse(base, MemInsn{ .kind = .ldr_w, .rt = 1, .rn = 2, .off = 8 }));

    // Out of range: imm7 = a.off / size = 512 / 8 = 64, over the 63 upper bound.
    const far = MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 2, .off = 512 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(far, MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 2, .off = 520 }));
}
