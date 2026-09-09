//! IR-to-IR expansions of operations a backend cannot lower, each run once before instruction
//! selection so that backend's isel never meets one. `expandMulh` rewrites the `mulh` BinOp into
//! half-width limbs. `expandMatmul` rewrites the et-soc tensor-tile `matmul` into a scalar loop
//! nest, which is what lets a matmul execute anywhere at all.
//!
//! `expandMulh` expands the high half of a full-width product into plain multiplies, shifts, and
//! masks, for backends that have no high-multiply instruction (wasm/spirv/x86/x86_64/nvidia). The
//! two native scalar backends (aarch64 smulh/umulh, riscv64 mulh/mulhu) lower `mulh` directly and
//! never call this. Producing `mulh` is the magic-number divide lowering's job (`strength.zig`); a
//! backend without native support runs this once before instruction selection so its isel never
//! meets a `mulh`.
//!
//! The high half is computed from half-width limbs. For a W-bit value the limbs are H = W/2 bits:
//! splitting a = ahi*2^H + alo and b likewise, the full product's high W bits are
//!   hihi + (lohi >> H) + (hilo >> H) + (((lolo >> H) + (lohi & m) + (hilo & m)) >> H)
//! where lolo = alo*blo, lohi = alo*bhi, hilo = ahi*blo, hihi = ahi*bhi, m = 2^H - 1. Every shift is
//! masked back to H bits, so an arithmetic right shift is fine even on a signed type (the sign fill
//! lands above bit H and is masked away), which is why no unsigned reinterpret is needed. For a
//! signed `mulh` the unsigned high half is corrected by `- (a<0 ? b : 0) - (b<0 ? a : 0)`.

const std = @import("std");
const function = @import("function.zig");
const types = @import("types.zig");

const Function = function.Function;
const Value = function.Value;
const Block = function.Block;
const Inst = function.Inst;
const BinOp = function.BinOp;
const MatMul = function.MatMul;

/// Rewrite every `arith` with op `.mulh` in `func` into an equivalent limb sequence. Returns
/// whether anything was rewritten.
pub fn expandMulh(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        // Does this block hold a mulh? Rebuild its instruction list only if so.
        var has = false;
        for (func.blockInsts(block)) |inst| {
            if (isMulh(func, inst)) {
                has = true;
                break;
            }
        }
        if (!has) continue;
        changed = true;

        var out: std.ArrayList(Inst) = .empty;
        defer out.deinit(allocator);
        // Snapshot the original list: appending new insts to the function's pool must not perturb
        // the sequence we are iterating.
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        for (original) |inst| {
            if (!isMulh(func, inst)) {
                try out.append(allocator, inst);
                continue;
            }
            const a = func.opcode(inst).arith;
            const result = func.instResult(inst).?;
            const high = try emitLimbs(func, &out, allocator, a.lhs, a.rhs, func.valueType(result));
            func.replaceAllUses(result, high);
        }
        try func.setBlockInsts(block, out.items);
    }
    return changed;
}

fn isMulh(func: *const Function, inst: Inst) bool {
    return switch (func.opcode(inst)) {
        .arith => |a| a.op == .mulh,
        else => false,
    };
}

/// Emit the limb sequence for `high half of (lhs * rhs)` at type `ty`, appending each instruction
/// to `out`, and return the value holding the high half.
fn emitLimbs(func: *Function, out: *std.ArrayList(Inst), allocator: std.mem.Allocator, lhs: Value, rhs: Value, ty: types.Type) std.mem.Allocator.Error!Value {
    const info = switch (func.types.type_kind(ty)) {
        .int => |i| i,
        else => unreachable, // mulh is integer-only (verify/strength guarantee it)
    };
    const w: u16 = info.bits;
    const h: i64 = @intCast(w / 2);
    const mask_h: i64 = (@as(i64, 1) << @intCast(w / 2)) - 1;

    const b = struct {
        f: *Function,
        o: *std.ArrayList(Inst),
        a: std.mem.Allocator,
        ty: types.Type,
        fn konst(self: @This(), c: i64) std.mem.Allocator.Error!Value {
            const v = try self.f.createInst(self.ty, .{ .iconst = c });
            try self.o.append(self.a, self.f.definingInst(v).?);
            return v;
        }
        fn op(self: @This(), o: BinOp, x: Value, y: Value) std.mem.Allocator.Error!Value {
            const v = try self.f.createInst(self.ty, .{ .arith = .{ .op = o, .lhs = x, .rhs = y } });
            try self.o.append(self.a, self.f.definingInst(v).?);
            return v;
        }
    }{ .f = func, .o = out, .a = allocator, .ty = ty };

    const m = try b.konst(mask_h);
    const hs = try b.konst(h);

    const alo = try b.op(.bit_and, lhs, m);
    const ahi_s = try b.op(.shr, lhs, hs);
    const ahi = try b.op(.bit_and, ahi_s, m);
    const blo = try b.op(.bit_and, rhs, m);
    const bhi_s = try b.op(.shr, rhs, hs);
    const bhi = try b.op(.bit_and, bhi_s, m);

    const lolo = try b.op(.mul, alo, blo);
    const lohi = try b.op(.mul, alo, bhi);
    const hilo = try b.op(.mul, ahi, blo);
    const hihi = try b.op(.mul, ahi, bhi);

    const lolo_hi_s = try b.op(.shr, lolo, hs);
    const lolo_hi = try b.op(.bit_and, lolo_hi_s, m);
    const lohi_lo = try b.op(.bit_and, lohi, m);
    const hilo_lo = try b.op(.bit_and, hilo, m);
    const cross0 = try b.op(.add, lolo_hi, lohi_lo);
    const cross = try b.op(.add, cross0, hilo_lo);

    const lohi_hi_s = try b.op(.shr, lohi, hs);
    const lohi_hi = try b.op(.bit_and, lohi_hi_s, m);
    const hilo_hi_s = try b.op(.shr, hilo, hs);
    const hilo_hi = try b.op(.bit_and, hilo_hi_s, m);
    const cross_hi_s = try b.op(.shr, cross, hs);
    const cross_hi = try b.op(.bit_and, cross_hi_s, m);

    const s0 = try b.op(.add, hihi, lohi_hi);
    const s1 = try b.op(.add, s0, hilo_hi);
    const unsigned_high = try b.op(.add, s1, cross_hi);
    if (info.signedness == .unsigned) return unsigned_high;

    // Signed correction: subtract b where a is negative, and a where b is negative. The sign mask is
    // an arithmetic shift of the ORIGINAL signed operand by W-1 (all ones when negative, else zero).
    const wm1 = try b.konst(@as(i64, @intCast(w - 1)));
    const amask = try b.op(.shr, lhs, wm1);
    const bmask = try b.op(.shr, rhs, wm1);
    const ca = try b.op(.bit_and, amask, rhs);
    const cb = try b.op(.bit_and, bmask, lhs);
    const c0 = try b.op(.sub, unsigned_high, ca);
    return b.op(.sub, c0, cb);
}

/// Why `expandMatmul` leaves one `matmul` in place. Each value names a feature the scalar nest
/// does not model, so a caller that wants a hard failure instead of a surviving `matmul` can ask
/// for the reason and raise its own error. A `null` reason means the expansion handles the op.
pub const Unsupported = enum {
    /// A `quant` epilogue. Requantization is a rounding decision: the fp32 scale multiply, the
    /// relu, the saturation to int8 or uint8, and the zero-point add must happen in the et-soc
    /// order and with the et-soc rounding mode, or the expansion answers a different question than
    /// the tensor unit does. A wrong requantization is worse than no expansion, so this rejects.
    quant,
};

/// The reason `expandMatmul` cannot rewrite `mm`, or null when it can.
///
/// The epilogue is the only rejection. All four dtypes expand: `fp32` multiplies its loaded
/// elements directly, and `fp16`, `int8` and `uint8` widen each element to the 32-bit accumulator
/// first. `input_signs` needs no entry of its own either: `verify` accepts it only when
/// `dtype == .int8`, and the int8 nest reads the per-operand signedness straight out of it. Nor
/// does `embedded`. That flag tells the et-soc backend to save and restore the registers its tensor
/// lowering clobbers, and a scalar nest clobbers nothing, so it is not addressed to this pass.
pub fn matmulUnsupported(mm: MatMul) ?Unsupported {
    if (mm.quant != null) return .quant;
    return switch (mm.dtype) {
        .fp32, .fp16, .int8, .uint8 => null,
    };
}

/// The element shape one `matmul` dtype gives the nest.
const Shape = struct {
    /// The A element type, as loaded from memory.
    a_elem: types.Type,
    /// The B element type, as loaded from memory.
    b_elem: types.Type,
    /// The accumulator and C element type. Always 32 bits: f32 for `fp32`, i32 for `int8`/`uint8`.
    acc: types.Type,
    /// The A/B element size in bytes, which is every A/B pointer stride's unit.
    elem_bytes: i64,
    /// Whether each loaded element passes through a `convert` to the accumulator type first. The
    /// 8-bit dtypes convert, fp32 multiplies the loaded values directly.
    convert: bool,
    /// Whether the accumulator is a float, which picks `fconst` over `iconst` for its zero.
    acc_is_float: bool,
};

/// C is always a 32-bit element, whatever the A/B dtype is (see `MatMul`'s doc comment).
const c_elem_bytes: i64 = 4;

/// The element shape of `mm`. Only called after `matmulUnsupported` returned null.
fn shapeOf(func: *Function, mm: MatMul) std.mem.Allocator.Error!Shape {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    switch (mm.dtype) {
        .fp32 => return .{
            .a_elem = f32_t,
            .b_elem = f32_t,
            .acc = f32_t,
            .elem_bytes = 4,
            .convert = false,
            .acc_is_float = true,
        },
        .int8, .uint8 => {
            // `input_signs`, when set, is AUTHORITATIVE per operand and `dtype` then only names the
            // hardware element type (see `InputSigns`). Without it the dtype decides both operands.
            const default_unsigned = mm.dtype == .uint8;
            const a_unsigned = if (mm.input_signs) |s| s.a_unsigned else default_unsigned;
            const b_unsigned = if (mm.input_signs) |s| s.b_unsigned else default_unsigned;
            return .{
                .a_elem = try func.types.intern(.{ .int = .{ .signedness = if (a_unsigned) .unsigned else .signed, .bits = 8 } }),
                .b_elem = try func.types.intern(.{ .int = .{ .signedness = if (b_unsigned) .unsigned else .signed, .bits = 8 } }),
                .acc = i32_t,
                .elem_bytes = 1,
                .convert = true,
                .acc_is_float = false,
            };
        },
        // fp16 loads half-width elements and widens each to the f32 accumulator, which is the
        // et-soc tensor unit's own `fp16 -> fp32` type. C stays 32-bit, so only A and B narrow.
        .fp16 => return .{
            .a_elem = try func.types.intern(.{ .float = .f16 }),
            .b_elem = try func.types.intern(.{ .float = .f16 }),
            .acc = f32_t,
            .elem_bytes = 2,
            .convert = true,
            .acc_is_float = true,
        },
    }
}

/// One `matmul` to rewrite: where it sits and what it says.
const Site = struct { block: Block, index: usize, mm: MatMul };

/// The first `matmul` this pass can rewrite, in block order then instruction order, or null when
/// the function holds none.
fn findSite(func: *const Function) ?Site {
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        var branched = false;
        for (func.blockInsts(block), 0..) |inst, index| {
            switch (func.opcode(inst)) {
                // An `if` does not terminate its block, so both of its edges leave the block and
                // any instruction after it is unreachable. Splitting such a block would move the
                // `if` into the continuation and change where those edges are taken from, so a
                // matmul behind one is left alone. No builder emits this shape.
                .@"if" => branched = true,
                .matmul => |mm| {
                    if (branched) continue;
                    if (matmulUnsupported(mm) != null) continue;
                    return .{ .block = block, .index = index, .mm = mm };
                },
                else => {},
            }
        }
    }
    return null;
}

/// Whether any attribute is keyed by a block. `reorderBlocks` does not remap a block id held in an
/// attribute payload, and this pass finishes with `reorderBlocks`, so a function carrying such an
/// attribute has no safe rewrite here. `vulcan-opt.blocklayout` skips the same functions for the
/// same reason.
fn hasBlockAttribute(func: *const Function) bool {
    for (func.attributeEntries()) |entry| {
        if (entry.target == .block) return true;
    }
    return false;
}

/// Rewrite every `matmul` in `func` that this pass supports into a scalar loop nest over plain
/// loads, multiplies, adds and one store. Returns whether anything was rewritten.
///
/// `matmul` is an et-soc tensor-tile op that only the et-soc VPU backend lowers, so a function
/// holding one cannot execute anywhere else. After this pass runs, every op left in the function is
/// an ordinary scalar op, so any backend compiles it. That makes the expansion the reference answer
/// a tensor lowering gets checked against.
///
/// The nest is the row-major definition of the product, with `a` an m by k matrix, `b` a k by n
/// matrix, and `c` an m by n matrix of 32-bit elements:
///
/// ```
/// for (i in 0..m)
///   for (j in 0..n) {
///     acc = if (accumulate) c[i][j] else 0;
///     for (p in 0..k) acc += a[i][p] * b[p][j];
///     c[i][j] = acc;
///   }
/// ```
///
/// Every pointer walks by a constant stride rather than by a computed index, which is the same
/// shape `vulcan-opt.microarch.matmul_recog` reads when it raises a nest back to a `matmul`: A
/// steps one element per p and one k-element row per i, B steps one n-element row per p and one
/// element per j, and C steps one 32-bit element per j across the whole nest.
///
/// PRECONDITIONS: none. `MatMul`'s doc comment lists 64-byte alignment of `a`, `b` and `c`, one
/// matrix row per cache line, and backend ownership of x31, x6 and the TenC registers across the
/// operation. Those are et-soc HARDWARE requirements of the tensor unit, not properties of the
/// operation. This expansion is plain IR: it reads and writes tightly packed row-major memory at
/// any alignment, it clobbers no register, and the register allocator sees the loads and stores it
/// emits like any others. That difference is the point. A matmul that only runs where those
/// hardware preconditions hold cannot be checked against anything.
///
/// A `matmul` this pass does not support is LEFT IN PLACE and the function is unchanged around it,
/// so a backend that cannot lower it still reports its own error. `matmulUnsupported` names the
/// reason for a caller that wants to raise that error earlier.
pub fn expandMatmul(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    // Guard first: the layout step at the end of every rewrite cannot keep a block-keyed attribute
    // valid, so such a function keeps its matmul rather than getting a stale attribute.
    if (hasBlockAttribute(func)) return false;

    var changed = false;
    // Each rewrite renumbers the blocks, so the next site is found from the top. A rewrite always
    // removes the matmul it was found for, so this terminates.
    while (findSite(func)) |site| {
        try expandSite(allocator, func, site);
        changed = true;
    }
    return changed;
}

/// The blocks one rewrite adds, in the order they are laid out.
const Nest = struct {
    i_head: Block,
    j_head: Block,
    j_body: Block,
    p_head: Block,
    p_body: Block,
    j_latch: Block,
    i_latch: Block,
    /// What the matmul's own block held after it, plus that block's terminator.
    cont: Block,
};

/// Rewrite one `matmul` into the nest. The block holding it becomes the preheader: the
/// instructions before the matmul stay there, and the instructions after it, along with the
/// block's terminator, move to the continuation the nest falls out to.
fn expandSite(allocator: std.mem.Allocator, func: *Function, site: Site) std.mem.Allocator.Error!void {
    const mm = site.mm;
    const shape = try shapeOf(func, mm);
    const preheader = site.block;
    const old_block_count = func.blockCount();

    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    // Snapshot the block's instruction list: `setBlockInsts` clears the list it would then read
    // from, so the two halves must be copied out first.
    const original = try allocator.dupe(Inst, func.blockInsts(preheader));
    defer allocator.free(original);
    const before = original[0..site.index];
    const after = original[site.index + 1 ..];

    var nest: Nest = undefined;
    nest.i_head = try func.appendBlock();
    nest.j_head = try func.appendBlock();
    nest.j_body = try func.appendBlock();
    nest.p_head = try func.appendBlock();
    nest.p_body = try func.appendBlock();
    nest.j_latch = try func.appendBlock();
    nest.i_latch = try func.appendBlock();
    nest.cont = try func.appendBlock();

    try func.setBlockInsts(nest.cont, after);
    func.terminatorPtr(nest.cont).* = func.terminator(preheader);
    try func.setBlockInsts(preheader, before);
    func.terminatorPtr(preheader).* = null;

    // The preheader: the loop bounds and the fresh accumulator, all loop-invariant, in the one
    // block that dominates the whole nest.
    const zero = try func.appendInst(preheader, i32_t, .{ .iconst = 0 });
    const m_bound = try func.appendInst(preheader, i32_t, .{ .iconst = mm.m });
    const n_bound = try func.appendInst(preheader, i32_t, .{ .iconst = mm.n });
    const k_bound = try func.appendInst(preheader, i32_t, .{ .iconst = mm.k });
    const acc_zero = if (shape.acc_is_float)
        try func.appendInst(preheader, shape.acc, .{ .fconst = 0.0 })
    else
        try func.appendInst(preheader, shape.acc, .{ .iconst = 0 });
    try func.setJump(preheader, nest.i_head, &.{ zero, mm.a, mm.c });

    // The i-loop carries the row of A and the running write pointer into C. C is one advance
    // across the whole nest, so the i-loop never resets it.
    const i = try func.appendBlockParam(nest.i_head, i32_t);
    const a_row = try func.appendBlockParam(nest.i_head, ptr_t);
    const c_row = try func.appendBlockParam(nest.i_head, ptr_t);
    const i_lt = try func.appendInst(nest.i_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = m_bound } });
    try func.appendIf(
        nest.i_head,
        i_lt,
        .{ .target = nest.j_head, .args = &.{ zero, mm.b, c_row } },
        .{ .target = nest.cont },
    );

    // The j-loop carries the column pointer into B and the element pointer into C. The A row is
    // invariant across j, so it is read straight from the i-header, which dominates every block
    // below.
    const j = try func.appendBlockParam(nest.j_head, i32_t);
    const b_col = try func.appendBlockParam(nest.j_head, ptr_t);
    const c_elem = try func.appendBlockParam(nest.j_head, ptr_t);
    const j_lt = try func.appendInst(nest.j_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = j, .rhs = n_bound } });
    try func.appendIf(nest.j_head, j_lt, .{ .target = nest.j_body }, .{ .target = nest.i_latch });

    // The p-loop preheader. The `accumulate` load lives HERE and not in the j-header, because the
    // j-header also runs with `j == n`, where `c_elem` is one past the end of the C row.
    const acc_init = if (mm.accumulate)
        try func.appendInst(nest.j_body, shape.acc, .{ .load = .{ .ptr = c_elem } })
    else
        acc_zero;
    try func.setJump(nest.j_body, nest.p_head, &.{ zero, acc_init, a_row, b_col });

    // The p-loop: the reduction. Its accumulator param holds the finished sum on the exit edge, and
    // the p-header dominates the j-latch, so the store reads it there with no block argument.
    const p = try func.appendBlockParam(nest.p_head, i32_t);
    const acc = try func.appendBlockParam(nest.p_head, shape.acc);
    const a_elem = try func.appendBlockParam(nest.p_head, ptr_t);
    const b_elem = try func.appendBlockParam(nest.p_head, ptr_t);
    const p_lt = try func.appendInst(nest.p_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = p, .rhs = k_bound } });
    try func.appendIf(nest.p_head, p_lt, .{ .target = nest.p_body }, .{ .target = nest.j_latch });

    const a_val = try func.appendInst(nest.p_body, shape.a_elem, .{ .load = .{ .ptr = a_elem } });
    const b_val = try func.appendInst(nest.p_body, shape.b_elem, .{ .load = .{ .ptr = b_elem } });
    // The 8-bit dtypes widen each element to the accumulator before the multiply, which is where
    // the per-operand signedness of `input_signs` takes effect: the load type decides whether the
    // convert extends the sign or zero-fills.
    const a_op = if (shape.convert) try func.appendInst(nest.p_body, shape.acc, .{ .convert = .{ .value = a_val } }) else a_val;
    const b_op = if (shape.convert) try func.appendInst(nest.p_body, shape.acc, .{ .convert = .{ .value = b_val } }) else b_val;
    const product = try func.appendInst(nest.p_body, shape.acc, .{ .arith = .{ .op = .mul, .lhs = a_op, .rhs = b_op } });
    const next_acc = try func.appendInst(nest.p_body, shape.acc, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = product } });
    const next_p = try func.appendArithImm(nest.p_body, i32_t, .add, p, 1);
    const next_a_elem = try func.appendArithImm(nest.p_body, ptr_t, .add, a_elem, shape.elem_bytes);
    const next_b_elem = try func.appendArithImm(nest.p_body, ptr_t, .add, b_elem, @as(i64, mm.n) * shape.elem_bytes);
    try func.setJump(nest.p_body, nest.p_head, &.{ next_p, next_acc, next_a_elem, next_b_elem });

    // The j-latch: write the element, then step j, the B column, and C by one element each.
    try func.appendStore(nest.j_latch, acc, c_elem);
    const next_j = try func.appendArithImm(nest.j_latch, i32_t, .add, j, 1);
    const next_b_col = try func.appendArithImm(nest.j_latch, ptr_t, .add, b_col, shape.elem_bytes);
    const next_c_elem = try func.appendArithImm(nest.j_latch, ptr_t, .add, c_elem, c_elem_bytes);
    try func.setJump(nest.j_latch, nest.j_head, &.{ next_j, next_b_col, next_c_elem });

    // The i-latch: step i and advance A by one k-element row. C carries on from where the j-loop
    // left it, which is the first element of the next C row.
    const next_i = try func.appendArithImm(nest.i_latch, i32_t, .add, i, 1);
    const next_a_row = try func.appendArithImm(nest.i_latch, ptr_t, .add, a_row, @as(i64, mm.k) * shape.elem_bytes);
    try func.setJump(nest.i_latch, nest.i_head, &.{ next_i, next_a_row, c_elem });

    try layOutNest(allocator, func, @intFromEnum(preheader), old_block_count, nest);
}

/// Put the blocks in an order where every block follows its immediate dominator.
///
/// The nest is built by appending, so its eight blocks land after every block the function already
/// had. That order is legal IR and `ir.verify` accepts it, but the machine backends number
/// linear-scan liveness by block INDEX and need a definition's block to come before every block it
/// dominates. A continuation placed after the blocks it dominates makes the register allocator read
/// a use before its definition and produce an unsound allocation, which surfaces far away as
/// `wimmer.zig`'s allocation verifier firing. `vulcan-opt.blocklayout` states the same rule for the
/// same reason.
///
/// The nest is spliced in where its preheader already was: the blocks before the preheader keep
/// their places, then the preheader, the i-header, the j-header, the j-body, the p-header, the
/// p-body, the j-latch, the i-latch, the continuation, then the blocks that followed the preheader.
/// Each header's immediate dominator is the header outside it, each latch's is a header, and the
/// continuation's is the i-header, so every added block follows the block that dominates it. Every
/// block the function came in with keeps its relative place, and every one of those the preheader
/// dominated is now dominated by the continuation, which still precedes it.
fn layOutNest(
    allocator: std.mem.Allocator,
    func: *Function,
    preheader_index: usize,
    old_block_count: usize,
    nest: Nest,
) std.mem.Allocator.Error!void {
    const order = try allocator.alloc(Block, func.blockCount());
    defer allocator.free(order);

    for (0..preheader_index + 1) |bi| order[bi] = @enumFromInt(bi);
    const added = [_]Block{
        nest.i_head, nest.j_head,  nest.j_body,  nest.p_head,
        nest.p_body, nest.j_latch, nest.i_latch, nest.cont,
    };
    for (added, 0..) |block, offset| order[preheader_index + 1 + offset] = block;

    var next = preheader_index + 1 + added.len;
    for (preheader_index + 1..old_block_count) |bi| {
        order[next] = @enumFromInt(bi);
        next += 1;
    }
    std.debug.assert(next == order.len); // every block placed exactly once
    try func.reorderBlocks(allocator, order);
}

const testing = std.testing;

fn intTy(func: *Function, bits: u16, signedness: std.builtin.Signedness) !types.Type {
    return func.types.intern(.{ .int = .{ .signedness = signedness, .bits = bits } });
}

/// The i128 oracle: the true high `bits` of the full-width product, matching `mulh` semantics.
fn oracleHigh(a: i64, b: i64, bits: u16, signedness: std.builtin.Signedness) i64 {
    const shift: u7 = @intCast(bits);
    return switch (signedness) {
        .signed => @truncate(@as(i128, a) * @as(i128, b) >> shift),
        .unsigned => blk: {
            const au: u128 = @as(u64, @bitCast(a)) & maskBits(bits);
            const bu: u128 = @as(u64, @bitCast(b)) & maskBits(bits);
            break :blk @bitCast(@as(u64, @truncate((au * bu) >> shift)));
        },
    };
}

fn maskBits(bits: u16) u64 {
    return if (bits >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(bits)) - 1;
}

/// Sign- or zero-extend the low `bits` of `v` to a canonical i64, matching how a W-bit register
/// value reads back. Used by the test evaluator to model each op at its declared width.
fn wrapTo(v: i64, bits: u16, signedness: std.builtin.Signedness) i64 {
    if (bits >= 64) return v;
    const low: u64 = @as(u64, @bitCast(v)) & maskBits(bits);
    return switch (signedness) {
        .unsigned => @bitCast(low),
        .signed => blk: {
            const sign = @as(u64, 1) << @intCast(bits - 1);
            break :blk @bitCast(if (low & sign != 0) low | ~maskBits(bits) else low);
        },
    };
}

/// Evaluate a value whose whole dataflow is constants (iconst leaves, arith nodes), modelling each
/// op at its result type's width and signedness. Only for the tests below (inputs are constants).
fn evalConst(func: *const Function, v: Value) i64 {
    const inst = func.definingInst(v).?;
    const info = switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i,
        else => unreachable,
    };
    return switch (func.opcode(inst)) {
        .iconst => |c| wrapTo(c, info.bits, info.signedness),
        .arith => |a| blk: {
            const l = evalConst(func, a.lhs);
            const r = evalConst(func, a.rhs);
            const raw: i64 = switch (a.op) {
                .add => l +% r,
                .sub => l -% r,
                .mul => l *% r,
                .bit_and => l & r,
                .shr => switch (info.signedness) {
                    .signed => l >> @intCast(@as(u64, @bitCast(r)) & 63),
                    .unsigned => @bitCast((@as(u64, @bitCast(l)) & maskBits(info.bits)) >> @intCast(@as(u64, @bitCast(r)) & 63)),
                },
                else => unreachable, // the expansion only emits the ops above
            };
            break :blk wrapTo(raw, info.bits, info.signedness);
        },
        else => unreachable,
    };
}

fn expectMulhExpands(bits: u16, signedness: std.builtin.Signedness, a: i64, b: i64) !void {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, bits, signedness);
    const e = try func.appendBlock();
    const av = try func.appendInst(e, t, .{ .iconst = a });
    const bv = try func.appendInst(e, t, .{ .iconst = b });
    const r = try func.appendInst(e, t, .{ .arith = .{ .op = .mulh, .lhs = av, .rhs = bv } });
    func.setTerminator(e, .{ .ret = function.Ret.one(r) });

    try testing.expect(try expandMulh(allocator, &func));
    for (func.blockInsts(e)) |inst| try testing.expect(!isMulh(&func, inst)); // no mulh survives
    const got = evalConst(&func, func.terminator(e).?.ret.values[0]);
    try testing.expectEqual(oracleHigh(a, b, bits, signedness), got);
}

test "expandMulh matches the i128 oracle for signed 64-bit" {
    try expectMulhExpands(64, .signed, 0x123456789, 0x9876543);
    try expectMulhExpands(64, .signed, -0x123456789, 0x9876543);
    try expectMulhExpands(64, .signed, -3, -7);
    try expectMulhExpands(64, .signed, std.math.maxInt(i64), std.math.maxInt(i64));
    try expectMulhExpands(64, .signed, std.math.minInt(i64), 2);
}

test "expandMulh matches the i128 oracle for unsigned 64-bit" {
    try expectMulhExpands(64, .unsigned, @bitCast(@as(u64, 0xFFFFFFFF00000000)), @bitCast(@as(u64, 0x2)));
    try expectMulhExpands(64, .unsigned, @bitCast(~@as(u64, 0)), @bitCast(~@as(u64, 0)));
    try expectMulhExpands(64, .unsigned, 0x123456789, 0x9876543);
}

test "expandMulh matches the i128 oracle for 32-bit widths" {
    try expectMulhExpands(32, .signed, 100000, 100000);
    try expectMulhExpands(32, .signed, -100000, 100000);
    try expectMulhExpands(32, .unsigned, @bitCast(@as(u64, 0xFFFF0000)), 0x30000);
}
