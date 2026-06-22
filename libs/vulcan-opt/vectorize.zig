//! SLP auto-vectorizer: fuses runs of `lanes` parallel scalar f32 arith ops into one vector
//! arith (the shape the GLSL/SPIR-V frontends emit when scalarizing a vecN op). Chain reuse
//! keeps intermediates in vector registers via a scalar-to-lane map, so a chain like (a+b)*c
//! does not re-pack between groups. Per block, contiguous same-op runs only. lanes is 4 for
//! NEON/SSE/RVV, 8 for AVX. `run` defaults to 4.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const BinOp = ir.function.BinOp;

pub const Error = std.mem.Allocator.Error;

/// The widest lane count any target asks for (AVX `<8 x f32>`), for fixed-size scratch buffers.
const MAX_LANES = 8;

/// Which lane of which vector a scalar value holds (recorded on extract).
const LaneOf = struct { vec: Value, lane: u8 };
const VMap = std.AutoHashMapUnmanaged(Value, LaneOf);

/// Vectorize every eligible group in `func` at the NEON/SSE/RVV width (4). Returns true if
/// anything changed.
pub fn run(allocator: std.mem.Allocator, func: *Function) Error!bool {
    return runLanes(allocator, func, 4);
}

/// Vectorize `func` fusing runs of `lanes` scalars (4 for 128-bit SIMD, 8 for AVX). Returns
/// true if anything changed.
pub fn runLanes(allocator: std.mem.Allocator, func: *Function, lanes: u8) Error!bool {
    std.debug.assert(lanes >= 2 and lanes <= MAX_LANES);
    var vmap: VMap = .empty;
    defer vmap.deinit(allocator);
    var changed = false;
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        while (try vectorizeOne(allocator, func, @enumFromInt(bi), &vmap, lanes)) changed = true;
    }
    return changed;
}

fn isScalarF32(func: *const Function, v: Value) bool {
    return switch (func.types.type_kind(func.valueType(v))) {
        .float => |f| f == .f32,
        else => false,
    };
}

/// Find and fuse the first eligible group of `lanes` instructions in `block`. Returns true if
/// it did.
fn vectorizeOne(allocator: std.mem.Allocator, func: *Function, block: Block, vmap: *VMap, lanes: u8) Error!bool {
    // Scan (read-only) for `lanes` contiguous f32 `arith` instructions sharing a BinOp.
    var pos: usize = 0;
    var op: BinOp = undefined;
    var group: [MAX_LANES]Inst = undefined;
    var found = false;
    {
        const insts = func.blockInsts(block);
        scan: for (0..(if (insts.len >= lanes) insts.len - lanes + 1 else 0)) |g| {
            const head = func.opcodeMut(insts[g]).*;
            if (head != .arith or !isScalarF32(func, func.instResult(insts[g]).?)) continue;
            for (1..lanes) |k| {
                const o = func.opcodeMut(insts[g + k]).*;
                if (o != .arith or o.arith.op != head.arith.op) continue :scan;
                if (!isScalarF32(func, func.instResult(insts[g + k]).?)) continue :scan;
            }
            pos = g;
            op = head.arith.op;
            for (0..lanes) |k| group[k] = insts[g + k];
            found = true;
            break;
        }
    }
    if (!found) return false;

    // Capture each lane's operands and result before appending.
    var a: [MAX_LANES]Value = undefined;
    var b: [MAX_LANES]Value = undefined;
    var c: [MAX_LANES]Value = undefined;
    for (0..lanes) |k| {
        const arith = func.opcodeMut(group[k]).*.arith;
        a[k] = arith.lhs;
        b[k] = arith.rhs;
        c[k] = func.instResult(group[k]).?;
    }

    const f32_t = try func.types.intern(.{ .float = .f32 });
    const vt = try func.types.intern(.{ .vector = .{ .len = lanes, .elem = f32_t } });

    const old_len = func.blockInsts(block).len; // boundary between old and appended insts

    const va = try operandVector(func, block, vt, vmap, a[0..lanes]);
    const vb = try operandVector(func, block, vt, vmap, b[0..lanes]);
    const vc = try func.appendInst(block, vt, .{ .arith = .{ .op = op, .lhs = va, .rhs = vb } });
    var x: [MAX_LANES]Value = undefined;
    try unpack(allocator, func, block, f32_t, vc, x[0..lanes], vmap);

    // Redirect every downstream use of the scalar results to the extracted lanes.
    for (0..lanes) |k| func.replaceAllUses(c[k], x[k]);

    // Move the appended sequence to the group's position and drop the now-dead scalar ops.
    try splice(allocator, func, block, pos, old_len, lanes);
    return true;
}

/// The vector to feed as an operand: if `scalars` are exactly lanes 0..N-1 of one existing
/// vector (a chain from an earlier group), reuse it directly. Otherwise pack them.
fn operandVector(func: *Function, block: Block, vt: ir.types.Type, vmap: *VMap, scalars: []const Value) Error!Value {
    if (vmap.get(scalars[0])) |l0| {
        if (l0.lane == 0) {
            var same = true;
            for (1..scalars.len) |k| {
                const lk = vmap.get(scalars[k]) orelse {
                    same = false;
                    break;
                };
                if (lk.vec != l0.vec or lk.lane != k) {
                    same = false;
                    break;
                }
            }
            if (same) return l0.vec; // the chain stays in a vector register
        }
    }
    return pack(func, block, vt, scalars);
}

/// Pack scalars into a vector with a `struct_new` (lowered to one insert per lane). A pure
/// register build, so a pack rendered dead by chain reuse falls to DCE.
fn pack(func: *Function, block: Block, vt: ir.types.Type, scalars: []const Value) Error!Value {
    const list = try func.internValueList(scalars);
    return func.appendInst(block, vt, .{ .struct_new = .{ .fields = list } });
}

/// Extract `out.len` scalars from `vec` (one `extract` op per lane), recording each as a known
/// lane of `vec` so a later group can reuse it. The extracts are pure, so chain-dead ones DCE.
fn unpack(allocator: std.mem.Allocator, func: *Function, block: Block, f32_t: ir.types.Type, vec: Value, out: []Value, vmap: *VMap) Error!void {
    for (0..out.len) |k| {
        out[k] = try func.appendInst(block, f32_t, .{ .extract = .{ .aggregate = vec, .index = @intCast(k) } });
        try vmap.put(allocator, out[k], .{ .vec = vec, .lane = @intCast(k) });
    }
}

/// Reorder the block so the appended vector sequence (insts[old_len..]) sits where the
/// scalar group was, and the `lanes` dead scalar instructions at `pos` are removed.
fn splice(allocator: std.mem.Allocator, func: *Function, block: Block, pos: usize, old_len: usize, lanes: u8) Error!void {
    const insts = func.blockInstsMut(block);
    var rebuilt: std.ArrayList(Inst) = .empty;
    defer rebuilt.deinit(allocator);
    try rebuilt.appendSlice(allocator, insts.items[0..pos]); // before the group
    try rebuilt.appendSlice(allocator, insts.items[old_len..]); // the vectorized sequence
    try rebuilt.appendSlice(allocator, insts.items[pos + lanes .. old_len]); // after the group
    insts.clearRetainingCapacity();
    try insts.appendSlice(allocator, rebuilt.items);
}
