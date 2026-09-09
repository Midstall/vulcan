//! Per-target tensor capability data: what one target's `matmul` lowering requires of the
//! operation and of the memory it reads. `ir.function.MatMul` describes WHAT to compute. This
//! file describes what each target needs before it can compute it, because the three lowerings
//! that exist do not agree on a single answer.
//!
//! The et-soc tensor unit is cache-line addressed, owns a fixed set of scratch registers for the
//! whole operation, and only accepts tiles its descriptor fields can encode. The scalar loop nest
//! in `vulcan-ir.expand` has none of those needs: it reads packed row-major memory at any
//! alignment and it owns no register. An NVIDIA tensor core is different again, because one HMMA
//! is warp-collective and each of the 32 lanes holds a piece of the tile in its own registers.
//! One IR op, three sets of preconditions. Prose in a doc comment cannot express that, and a pass
//! that raises a loop nest to a `matmul` cannot read prose. This is the data form of it.
//!
//! WHY THIS MODULE. `vulcan-ir` owns `MatMul` but cannot hold the answer, because a target
//! capability is not IR and `vulcan-ir` imports nothing. `vulcan-opt/microarch` owns the per-part
//! `Model`, but that registry is a CPU scheduling model whose `Arch` is aarch64, riscv64 and
//! x86_64, so an NVIDIA descriptor has no place in it. `vulcan-gpu` already owns `Abi`, the other
//! per-target capability descriptor, it depends only on the IR, and both `vulcan-opt` and
//! `vulcan-target` can import it without a cycle. So the descriptor lives here beside `Abi`.
//!
//! The three descriptors below sit in one file on purpose. Their VALUE is the comparison: a
//! reader must see, in one place, that et-soc needs 64-byte alignment and the scalar nest needs
//! none. Split across three backends, the contrast that this file exists to show disappears.

const std = @import("std");
const ir = @import("vulcan-ir");

const MatMul = ir.function.MatMul;
const MatMulType = ir.function.MatMulType;

/// The A/B element size in bytes of one `matmul` dtype. C is always a 32-bit element whatever the
/// A/B dtype is, so this covers the inputs only. Same table `vulcan-ir.expand` builds its scalar
/// nest from and the riscv64 backend derives its packing factor from.
pub fn elemBytes(dtype: MatMulType) u8 {
    return switch (dtype) {
        .fp32 => 4,
        .fp16 => 2,
        .int8, .uint8 => 1,
    };
}

/// One exact tile a target lowers, in the `MatMul` field names. A tensor core accepts a closed
/// list of these, because each shape is a different instruction.
pub const Tile = struct { m: u16, n: u16, k: u16 };

/// A native output tile that the lowering repeats over a grid of sub-tiles, so it accepts many
/// shapes but not every shape. This is the et-soc form: the tensor unit computes one `rows` by
/// `cols` output tile per pass, and the backend unrolls a grid of passes at compile time to reach
/// a larger `m`, `n` and `k`.
pub const Tiling = struct {
    /// Output rows in one native tile.
    rows: u16,
    /// Output columns in one native tile.
    cols: u16,
    /// `n` must be a multiple of this many columns, because the pass descriptor encodes the
    /// column count in groups of this size and a part group has no encoding.
    col_group: u16,
    /// The bytes of one contraction slot. `slot_bytes / elemBytes(dtype)` A/B elements pack into
    /// one slot, so `k` must be a multiple of that count, and one pass contracts `rows` slots of
    /// them. The count is 1 for a 4-byte element, 2 for a 2-byte element and 4 for a 1-byte one.
    slot_bytes: u16,
    /// The most sub-tile passes the lowering emits, that is the product of the tile counts along
    /// m, n and k. The grid is unrolled at compile time, so this bounds the code size.
    max_passes: u32,

    /// How many A/B elements of `dtype` pack into one contraction slot.
    pub fn pack(self: Tiling, dtype: MatMulType) u16 {
        return self.slot_bytes / elemBytes(dtype);
    }

    /// The contraction elements one pass reads, that is `rows` slots of packed elements.
    pub fn contraction(self: Tiling, dtype: MatMulType) u16 {
        return self.rows * self.pack(dtype);
    }

    /// Whether the grid can reach an `m` by `n` output with a `k` contraction at `dtype`.
    pub fn accepts(self: Tiling, dtype: MatMulType, m: u16, n: u16, k: u16) bool {
        // A zero dimension has no tile grid at all. The lowering rejects it rather than emit an
        // empty pass sequence.
        if (m == 0 or n == 0 or k == 0) return false;
        if (n % self.col_group != 0) return false;
        const group = self.pack(dtype);
        if (k % group != 0) return false; // a part-filled slot has no encoding
        const k_tile = self.contraction(dtype);
        const m_tiles = (@as(u32, m) + self.rows - 1) / self.rows;
        const n_tiles = (@as(u32, n) + self.cols - 1) / self.cols;
        const k_tiles = (@as(u32, k) + k_tile - 1) / k_tile;
        // u64 so an absurd shape cannot overflow before the cap refuses it.
        return @as(u64, m_tiles) * n_tiles * k_tiles <= self.max_passes;
    }
};

/// The tiles one target lowers.
pub const TileShapes = union(enum) {
    /// Every shape. A scalar expansion into a loop nest reads `m`, `n` and `k` as loop bounds, so
    /// no shape is out of reach.
    any,
    /// A closed list of exact tiles. A tensor core takes only these, because one tile shape is
    /// one instruction with one fragment layout.
    list: []const Tile,
    /// A native tile repeated over a compile-time grid. See `Tiling`.
    tiling: Tiling,

    /// Whether an `m` by `n` by `k` tile at `dtype` is a shape this target lowers.
    pub fn accepts(self: TileShapes, dtype: MatMulType, m: u16, n: u16, k: u16) bool {
        return switch (self) {
            .any => true,
            .list => |tiles| for (tiles) |t| {
                if (t.m == m and t.n == n and t.k == k) break true;
            } else false,
            .tiling => |t| t.accepts(dtype, m, n, k),
        };
    }
};

/// A run of registers one target's `matmul` lowering owns for the whole operation. A value that
/// is live across the op must not sit in this run, or the lowering destroys it.
///
/// `first` is the target's own register name. Nothing in this module reads it: the field records
/// an ownership contract for a person and for the backend that must honor it, and a target-neutral
/// module has no register vocabulary to interpret it with.
pub const Clobber = struct {
    /// The name of the first register in the run.
    first: []const u8,
    /// Registers in the run. When `per_row` is set this is the count for ONE output row, and the
    /// run then holds `count * @min(tiling.rows, m)` registers.
    count: u16 = 1,
    /// Whether the run grows with the output rows of one native tile. False for a fixed run.
    per_row: bool = false,
};

/// Why one target cannot lower one `matmul`. Each value names the field that does not fit, so a
/// caller can say which precondition failed rather than only that one did.
pub const Reject = enum {
    /// The target lowers no `matmul` at all.
    unsupported,
    /// The A/B element type is not one this target lowers.
    dtype,
    /// The `m` by `n` by `k` tile is not a shape this target lowers.
    shape,
    /// The fused quantize epilogue is not one this target lowers.
    quant,
};

/// What one target's `matmul` lowering can take. Construct one per target, then ask it with
/// `rejects` before you build or keep a `matmul` for that target.
pub const Tensor = struct {
    /// The alignment in bytes the lowering needs of the `a`, `b` and `c` pointers. 1 when it needs
    /// none.
    ///
    /// `rejects` does NOT read this field, and it cannot: an IR `ptr` carries no alignment, so no
    /// query over a `MatMul` can decide it. It stays a contract on whoever builds the op. A pass
    /// that raises a loop nest to a `matmul` must prove the bases already meet it, or repack the
    /// tiles into memory that does.
    operand_align: u32,
    /// The A/B element types the lowering accepts. Empty when the target lowers no `matmul`.
    dtypes: std.EnumSet(MatMulType),
    /// Whether one lowered `matmul` is executed by a whole subgroup together, with each lane
    /// holding a piece of the tile in its own registers. An NVIDIA tensor core works this way, so
    /// its lowering cannot appear on a path only some lanes of the warp take. A per-thread
    /// lowering (the scalar nest, and the et-soc tensor unit, which one minion drives alone) is
    /// false and has no such rule.
    warp_collective: bool,
    /// The tiles the lowering accepts.
    shapes: TileShapes,
    /// Whether the lowering accepts the fused quantize epilogue (`MatMulQuant`).
    quant: bool,
    /// The registers the lowering owns for the whole operation. Empty when it owns none.
    clobbers: []const Clobber,

    /// Whether this target lowers any `matmul` at all.
    pub fn lowersAny(self: *const Tensor) bool {
        return self.dtypes.count() != 0;
    }

    /// Why this target cannot lower `mm`, or null when it can.
    ///
    /// The query covers what a `MatMul` states: its dtype, its tile, and its epilogue. It cannot
    /// cover `operand_align`, because the IR does not carry pointer alignment. See that field.
    pub fn rejects(self: *const Tensor, mm: MatMul) ?Reject {
        if (!self.lowersAny()) return .unsupported;
        if (!self.dtypes.contains(mm.dtype)) return .dtype;
        if (mm.quant != null and !self.quant) return .quant;
        if (!self.shapes.accepts(mm.dtype, mm.m, mm.n, mm.k)) return .shape;
        return null;
    }

    /// Whether this target lowers `mm`. The negation of `rejects`, for a caller that only needs
    /// the answer.
    pub fn canLower(self: *const Tensor, mm: MatMul) bool {
        return self.rejects(mm) == null;
    }
};

/// Every dtype `MatMul` can hold. The scalar nest and the et-soc tensor unit both take all four.
const all_dtypes: std.EnumSet(MatMulType) = .initFull();

/// The scalar loop nest `vulcan-ir.expand.expandMatmul` writes. It is the reference answer every
/// tensor lowering gets checked against, so it must be the least demanding descriptor here.
///
/// It reads tightly packed row-major memory through ordinary loads at any alignment, so it needs
/// no alignment. It is plain IR, so the register allocator places its loads and stores like any
/// others and it owns no register. Its loop bounds are `m`, `n` and `k` themselves, so no shape is
/// out of reach. One thread runs the whole nest, so it is not warp-collective.
///
/// It refuses only the quantize epilogue, and `expand.matmulUnsupported` gives the reason:
/// requantization is a rounding decision, and an expansion that rounds differently from the tensor
/// unit answers a different question than the op asked.
pub const scalar: Tensor = .{
    .operand_align = 1,
    .dtypes = all_dtypes,
    .warp_collective = false,
    .shapes = .any,
    .quant = false,
    .clobbers = &.{},
};

/// The et-soc tensor unit, as the riscv64 backend lowers it today. These were the preconditions in
/// `MatMul`'s doc comment. They are et-soc hardware facts, not properties of the operation, which
/// is why they are here and no longer there.
///
/// ALIGNMENT, 64 bytes. `tensor_load` addresses 64-byte-aligned lines with a 64-byte-granular
/// stride, and the hardware masks both the descriptor address and the stride with `~0x3f`. The
/// backend builds the descriptor by an OR of static field bits into the low 6 bits of the pointer,
/// so an unaligned base would corrupt the descriptor instead of moving the load. The same 64 bytes
/// hold one matrix row. The backend stages a row whose pitch is not a multiple of 64 through an
/// aligned scratch buffer, which is what lets an arbitrary `k` and `n` work, but the BASE pointer
/// is never staged.
///
/// SHAPE. The native output tile is 16 rows by up to 16 columns, and one pass contracts one
/// 64-byte scratchpad line, which holds 16 fp32, 32 fp16 or 64 int8. `n` must be a multiple of 4,
/// because the pass descriptor encodes the column count as `cols / 4 - 1`. `k` must fill whole
/// packed slots. The grid of passes is unrolled at compile time, so the pass count is capped at 64
/// to bound the code size.
///
/// CLOBBERS. The lowering owns x5 (the sub-tile pointer), x6 (the descriptor scratch), x7 (the
/// staging word copy), x28 (the aligned staging base) and x31 (the load stride and the store
/// address), plus the TenC accumulator registers. TenC output row i lives in f(2i) and f(2i+1),
/// and the widest tile has `@min(16, m)` rows, so the run is f0 to f(2 * @min(16, m) - 1). The
/// doc comment this replaces named only x31, x6 and the even TenC registers. The lowering has
/// grown since, and the wider set above is what the backend writes today.
///
/// An `embedded` matmul additionally owns x9, x29 and x30 as the a/b/c holder registers, and it
/// saves and restores every register in this whole set around the operation. A standalone matmul
/// does neither, so the set above is what a live value must stay out of.
pub const et_soc: Tensor = .{
    .operand_align = 64,
    .dtypes = all_dtypes,
    .warp_collective = false,
    .shapes = .{ .tiling = .{ .rows = 16, .cols = 16, .col_group = 4, .slot_bytes = 4, .max_passes = 64 } },
    .quant = true,
    .clobbers = &.{
        .{ .first = "x5" },
        .{ .first = "x6" },
        .{ .first = "x7" },
        .{ .first = "x28" },
        .{ .first = "x31" },
        .{ .first = "f0", .count = 2, .per_row = true },
    },
};

/// NVIDIA, which lowers no `matmul` today. The backend has no HMMA and no IMMA case, so an empty
/// capability is the correct entry, not a missing one: a pass that asks this descriptor gets a
/// definite "no" and leaves the loop nest alone, rather than raising an op that fails to lower.
///
/// `warp_collective` is already true, because it is a fact about the hardware and not about the
/// state of the backend. When HMMA arrives, one instruction is executed by all 32 lanes of a warp
/// together and each lane holds a fragment of the tile in its own registers. That is a different
/// rule from et-soc's fixed scratch registers, and it is the reason this field exists. The dtype
/// set and the tile list stay empty until the lowering that fills them is written.
pub const nvidia: Tensor = .{
    .operand_align = 1,
    .dtypes = .initEmpty(),
    .warp_collective = true,
    .shapes = .{ .list = &.{} },
    .quant = false,
    .clobbers = &.{},
};

/// Build a bare `MatMul` for the tests below. The a/b/c operands are never read by any query here,
/// so placeholder handles are enough.
fn testMatmul(dtype: MatMulType, m: u16, n: u16, k: u16) MatMul {
    return .{
        .a = @enumFromInt(0),
        .b = @enumFromInt(1),
        .c = @enumFromInt(2),
        .m = m,
        .n = n,
        .k = k,
        .dtype = dtype,
        .accumulate = false,
    };
}

test "the scalar expansion needs no alignment and owns no register, the et-soc unit needs both" {
    try std.testing.expectEqual(@as(u32, 1), scalar.operand_align);
    try std.testing.expectEqual(@as(usize, 0), scalar.clobbers.len);
    try std.testing.expectEqual(@as(u32, 64), et_soc.operand_align);
    try std.testing.expectEqual(@as(usize, 6), et_soc.clobbers.len);
    // The TenC run is the only shape-dependent one: 2 registers per output row.
    try std.testing.expect(et_soc.clobbers[5].per_row);
    try std.testing.expectEqual(@as(u16, 2), et_soc.clobbers[5].count);
    try std.testing.expectEqualStrings("f0", et_soc.clobbers[5].first);
    for (et_soc.clobbers[0..5]) |c| try std.testing.expect(!c.per_row);
}

test "the scalar expansion takes a tile no tensor unit takes" {
    // 3 x 5 x 7 breaks the et-soc column group and the packed slot at once. The loop nest takes it.
    const odd = testMatmul(.fp32, 3, 5, 7);
    try std.testing.expect(scalar.canLower(odd));
    try std.testing.expectEqual(Reject.shape, et_soc.rejects(odd).?);
}

test "et-soc refuses a column count that is not a multiple of four" {
    try std.testing.expect(et_soc.canLower(testMatmul(.fp32, 16, 16, 16)));
    try std.testing.expectEqual(Reject.shape, et_soc.rejects(testMatmul(.fp32, 16, 14, 16)).?);
    // 12 columns is a multiple of 4 and not of 8, so it pins the group at 4 in both directions. A
    // group of 8 would refuse this shape and the backend accepts it.
    try std.testing.expect(et_soc.canLower(testMatmul(.fp32, 16, 12, 16)));
}

test "et-soc refuses a contraction that leaves a part-filled slot, per dtype" {
    // fp32 packs one element per slot, so every k passes. fp16 packs two and int8 packs four.
    try std.testing.expect(et_soc.canLower(testMatmul(.fp32, 16, 16, 17)));
    try std.testing.expectEqual(Reject.shape, et_soc.rejects(testMatmul(.fp16, 16, 16, 17)).?);
    try std.testing.expect(et_soc.canLower(testMatmul(.fp16, 16, 16, 18)));
    try std.testing.expectEqual(Reject.shape, et_soc.rejects(testMatmul(.int8, 16, 16, 18)).?);
    try std.testing.expect(et_soc.canLower(testMatmul(.int8, 16, 16, 20)));
}

test "et-soc refuses a tile grid above its pass cap" {
    // 64 by 64 by 64 fp32 is 4 x 4 x 4 = 64 passes, exactly the cap. One more tile along m is 80.
    try std.testing.expect(et_soc.canLower(testMatmul(.fp32, 64, 64, 64)));
    try std.testing.expectEqual(Reject.shape, et_soc.rejects(testMatmul(.fp32, 80, 64, 64)).?);
    // The same shape at int8 packs 4 elements per slot, so k costs a quarter of the tiles and it
    // fits again. The cap is per dtype, not per element count.
    try std.testing.expect(et_soc.canLower(testMatmul(.int8, 80, 64, 64)));
}

test "a zero dimension has no et-soc tile grid but is a valid empty scalar nest" {
    const empty = testMatmul(.fp32, 16, 16, 0);
    try std.testing.expect(scalar.canLower(empty));
    try std.testing.expectEqual(Reject.shape, et_soc.rejects(empty).?);
}

test "only et-soc takes the quantize epilogue" {
    var mm = testMatmul(.int8, 16, 16, 16);
    mm.quant = .{ .scale = .{ .scalar = 0x3f800000 }, .relu = false };
    try std.testing.expect(et_soc.canLower(mm));
    try std.testing.expectEqual(Reject.quant, scalar.rejects(mm).?);
}

test "nvidia reports unsupported for every matmul, not a wrong dtype or shape" {
    try std.testing.expect(!nvidia.lowersAny());
    try std.testing.expectEqual(Reject.unsupported, nvidia.rejects(testMatmul(.fp16, 16, 16, 16)).?);
    try std.testing.expectEqual(Reject.unsupported, nvidia.rejects(testMatmul(.fp32, 3, 5, 7)).?);
    // The warp-collective fact is recorded even though no lowering uses it yet.
    try std.testing.expect(nvidia.warp_collective);
    try std.testing.expect(!et_soc.warp_collective and !scalar.warp_collective);
}

test "a fixed tile list takes its own shapes and nothing else" {
    // The shape an NVIDIA HMMA descriptor will hold once the lowering exists.
    const hmma: Tensor = .{
        .operand_align = 1,
        .dtypes = .initOne(.fp16),
        .warp_collective = true,
        .shapes = .{ .list = &.{ .{ .m = 16, .n = 8, .k = 8 }, .{ .m = 16, .n = 8, .k = 16 } } },
        .quant = false,
        .clobbers = &.{},
    };
    try std.testing.expect(hmma.canLower(testMatmul(.fp16, 16, 8, 16)));
    try std.testing.expectEqual(Reject.shape, hmma.rejects(testMatmul(.fp16, 16, 8, 32)).?);
    try std.testing.expectEqual(Reject.dtype, hmma.rejects(testMatmul(.fp32, 16, 8, 16)).?);
}

test "the packing count follows the element size of the dtype" {
    const t: Tiling = .{ .rows = 16, .cols = 16, .col_group = 4, .slot_bytes = 4, .max_passes = 64 };
    try std.testing.expectEqual(@as(u16, 1), t.pack(.fp32));
    try std.testing.expectEqual(@as(u16, 2), t.pack(.fp16));
    try std.testing.expectEqual(@as(u16, 4), t.pack(.int8));
    try std.testing.expectEqual(@as(u16, 4), t.pack(.uint8));
    // One pass reads one 64-byte scratchpad line: 16 fp32, 32 fp16 or 64 int8.
    try std.testing.expectEqual(@as(u16, 16), t.contraction(.fp32));
    try std.testing.expectEqual(@as(u16, 32), t.contraction(.fp16));
    try std.testing.expectEqual(@as(u16, 64), t.contraction(.int8));
}

test "elemBytes gives the A/B element size of every dtype" {
    try std.testing.expectEqual(@as(u8, 4), elemBytes(.fp32));
    try std.testing.expectEqual(@as(u8, 2), elemBytes(.fp16));
    try std.testing.expectEqual(@as(u8, 1), elemBytes(.int8));
    try std.testing.expectEqual(@as(u8, 1), elemBytes(.uint8));
}
