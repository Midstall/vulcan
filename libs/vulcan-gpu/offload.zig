//! Rewrites a kernel into an ordinary function that runs the whole grid in a loop nest, so a
//! kernel can EXECUTE on the host through the CPU backends and the JIT. The result is the
//! reference answer a GPU backend gets checked against.
//!
//! The pass is target-neutral. It reads the same `Builtin` tags and the same declared
//! workgroup size the backends read, so the oracle and the GPU path agree on the ABI by
//! construction and not by comment.

const std = @import("std");
const ir = @import("vulcan-ir");
const attrs = @import("attrs.zig");
const builtin_mod = @import("builtin.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Type = ir.types.Type;
const Builtin = builtin_mod.Builtin;

/// What lowering can fail with. `Unsupported` covers every kernel shape this pass refuses to
/// rewrite. A refusal is deliberate: a half-right nest computes wrong answers quietly, which
/// is exactly what the oracle exists to catch.
pub const Error = std.mem.Allocator.Error || error{Unsupported};

/// The entry block of every function. Its parameters are the function's parameters.
const entry_block: Block = @enumFromInt(0);

/// The number of axes a grid has. Index 0 is x, index 1 is y, index 2 is z.
const axes: usize = 3;

/// The number of blocks the nest adds to the kernel's own block count: the spliced body
/// entry, one header and one latch per axis for the workgroup loops and again for the thread
/// loops, and the exit.
const nest_blocks: usize = 1 + 4 * axes + 1;

/// Which value of the nest a builtin reads.
const SourceKind = enum { thread_id, block_id, block_dim, grid_dim, global_id };

/// A builtin this pass can supply: the nest value it reads, and the axis it selects.
const NestSource = struct { kind: SourceKind, axis: u2 };

/// The nest source for `b`, or null when this pass cannot supply it. The three grid axes are
/// all in scope. The subgroup builtins are not: a subgroup is a hardware partition of a
/// workgroup, and the host nest runs one thread at a time, so it has no honest lane index.
/// The graphics builtins are not in scope either, because this pass lowers compute kernels.
fn nestSource(b: Builtin) ?NestSource {
    return switch (b) {
        .thread_id_x => .{ .kind = .thread_id, .axis = 0 },
        .thread_id_y => .{ .kind = .thread_id, .axis = 1 },
        .thread_id_z => .{ .kind = .thread_id, .axis = 2 },
        .block_id_x => .{ .kind = .block_id, .axis = 0 },
        .block_id_y => .{ .kind = .block_id, .axis = 1 },
        .block_id_z => .{ .kind = .block_id, .axis = 2 },
        .block_dim_x => .{ .kind = .block_dim, .axis = 0 },
        .block_dim_y => .{ .kind = .block_dim, .axis = 1 },
        .block_dim_z => .{ .kind = .block_dim, .axis = 2 },
        .grid_dim_x => .{ .kind = .grid_dim, .axis = 0 },
        .grid_dim_y => .{ .kind = .grid_dim, .axis = 1 },
        .grid_dim_z => .{ .kind = .grid_dim, .axis = 2 },
        .global_id_x => .{ .kind = .global_id, .axis = 0 },
        .global_id_y => .{ .kind = .global_id, .axis = 1 },
        .global_id_z => .{ .kind = .global_id, .axis = 2 },
        .lane_id,
        .warp_id,
        .subgroup_size,
        .vertex_index,
        .instance_index,
        .frag_coord,
        .point_coord,
        .front_facing,
        => null,
    };
}

/// One entry parameter the nest computes instead of passing.
const BuiltinParam = struct { value: Value, source: NestSource };

/// Whether any block returns a value. A value-returning kernel needs the output-pointer
/// convention, which belongs with the parameter block and not with the nest.
fn returnsValue(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        const term = func.terminator(@enumFromInt(bi)) orelse continue;
        switch (term) {
            .ret => |r| if (r.count != 0) return true,
            .jump => {},
        }
    }
    return false;
}

/// Whether any edge branches back to the entry block. The entry becomes the nest's preheader,
/// so a kernel that jumps back to it would re-enter the preheader once per iteration.
fn entryIsBranchTarget(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .@"if" => |cf| {
                    if (cf.then.target == entry_block) return true;
                    if (cf.@"else".target == entry_block) return true;
                },
                else => {},
            }
        }
        const term = func.terminator(block) orelse continue;
        switch (term) {
            .jump => |j| if (j.target == entry_block) return true,
            .ret => {},
        }
    }
    return false;
}

/// Whether any attribute is keyed by a block. `reorderBlocks` does not remap a block id that
/// an attribute payload holds, and this pass lays the result out with `reorderBlocks`, so a
/// kernel that carries such an attribute has no safe lowering here.
fn hasBlockAttribute(func: *const Function) bool {
    for (func.attributeEntries()) |entry| {
        switch (entry.target) {
            .block => return true,
            .func, .inst, .value => {},
        }
    }
    return false;
}

/// Rewrite `func`, a kernel, into an ordinary function that executes the whole grid.
///
/// The result's entry block takes the grid size in workgroups as THREE leading i32
/// parameters, `grid_x`, `grid_y` and `grid_z`, then the kernel's non-builtin parameters in
/// their original order. Every builtin parameter becomes a value computed from the loop
/// induction variables, so the caller supplies only real data.
///
/// The nest is six loops deep:
///
/// ```
/// for (block_id_z in 0..grid_z)
///   for (block_id_y in 0..grid_y)
///     for (block_id_x in 0..grid_x)
///       for (thread_id_z in 0..block[2])
///         for (thread_id_y in 0..block[1])
///           for (thread_id_x in 0..block[0]) body
/// ```
///
/// The x axis is the innermost loop of each pair, so the nest visits x fastest and z slowest.
/// That is the order every GPU numbers its lanes in. `block` is the declared workgroup size,
/// which the caller reads from `attrs.localSize`.
///
/// The blocks come out in an order where every block follows its immediate dominator, which is
/// what the machine backends' linear-scan liveness needs. See `layOutNest`.
///
/// The pass supplies `thread_id_*`, `block_id_*`, `block_dim_*`, `grid_dim_*` and
/// `global_id_*` on all three axes. A kernel that reads a subgroup builtin, `lane_id`,
/// `warp_id` or `subgroup_size`, returns `error.Unsupported`: the host nest runs one thread at
/// a time and has no subgroup to index. So does a kernel that reads a graphics builtin, a
/// kernel whose builtin parameter is not a signed 32-bit integer, a kernel whose entry block
/// is a branch target, and a kernel that carries a block-keyed attribute, which the layout
/// step cannot keep valid.
///
/// A kernel that RETURNS a value returns `error.Unsupported` too. Such a kernel writes its
/// result through an implicit output pointer, and that convention belongs with the parameter
/// block. This pass is about the nest.
///
/// The caller OWNS the result and must `deinit` it.
pub fn lowerToLoopNest(
    allocator: std.mem.Allocator,
    func: *const Function,
    block: [3]u32,
) Error!Function {
    if (func.blockCount() == 0) return error.Unsupported;
    if (returnsValue(func)) return error.Unsupported;
    if (entryIsBranchTarget(func)) return error.Unsupported;
    if (hasBlockAttribute(func)) return error.Unsupported;

    // Clone first: `clone` re-interns every type kind in order, so the n-th kind keeps handle
    // n and every Type in the copied arrays stays valid. The kernel's own blocks keep their
    // indices too, so its internal branch targets need no remap.
    var out = try func.clone(allocator);
    errdefer out.deinit();

    const i32_t = try out.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try out.types.intern(.bool);

    // Split the kernel's entry parameters. A builtin parameter leaves the signature and the
    // nest computes it; every other parameter keeps its place, three slots further along.
    var real: std.ArrayList(Value) = .empty;
    defer real.deinit(allocator);
    var builtins: std.ArrayList(BuiltinParam) = .empty;
    defer builtins.deinit(allocator);

    for (out.blockParams(entry_block)) |p| {
        const tag = attrs.builtinOf(&out, p) orelse {
            try real.append(allocator, p);
            continue;
        };
        const source = nestSource(tag) orelse return error.Unsupported;
        // The induction variables are i32, and the IR requires both operands of an
        // arithmetic op to share a type, so a builtin of another width has no correct
        // lowering here.
        if (out.valueType(p) != i32_t) return error.Unsupported;
        try builtins.append(allocator, .{ .value = p, .source = source });
    }

    const kernel_blocks = out.blockCount();

    // Move the kernel's entry body out of block 0, which becomes the preheader.
    const body_entry = try out.appendBlock();
    const entry_insts = try allocator.dupe(Inst, out.blockInsts(entry_block));
    defer allocator.free(entry_insts);
    try out.setBlockInsts(body_entry, entry_insts);
    out.terminatorPtr(body_entry).* = out.terminator(entry_block);
    out.blockInstsMut(entry_block).clearRetainingCapacity();
    out.terminatorPtr(entry_block).* = null;

    // One header and one latch per axis, twice: the workgroup loops outside, the thread loops
    // inside. The headers get made from z inward, which is the order they nest in.
    var nest: NestBlocks = undefined;
    nest.body_entry = body_entry;
    for (0..axes) |i| nest.block_head[axes - 1 - i] = try out.appendBlock();
    for (0..axes) |i| nest.thread_head[axes - 1 - i] = try out.appendBlock();
    for (&nest.thread_latch) |*slot| slot.* = try out.appendBlock();
    for (&nest.block_latch) |*slot| slot.* = try out.appendBlock();
    nest.exit = try out.appendBlock();

    // The preheader: the grid size in workgroups on all three axes, then the kernel's real
    // parameters.
    var grid: [axes]Value = undefined;
    for (&grid) |*slot| slot.* = try out.newParam(entry_block, i32_t);
    var params: std.ArrayList(Value) = .empty;
    defer params.deinit(allocator);
    try params.appendSlice(allocator, &grid);
    try params.appendSlice(allocator, real.items);
    try out.setBlockParams(entry_block, params.items);

    const zero = try out.appendInst(entry_block, i32_t, .{ .iconst = 0 });
    try out.setJump(entry_block, nest.block_head[axes - 1], &.{zero});

    // The workgroup loops, from the slowest axis inward. `if` does not terminate a block, so a
    // header needs no terminator of its own: both of its edges leave the block. The z header
    // falls out of the whole nest, and every other header falls out to the latch of the axis
    // outside it.
    var block_id: [axes]Value = undefined;
    for (0..axes) |i| {
        const a = axes - 1 - i;
        const head = nest.block_head[a];
        block_id[a] = try out.appendBlockParam(head, i32_t);
        const in_grid = try out.appendInst(head, bool_t, .{
            .icmp = .{ .op = .lt, .lhs = block_id[a], .rhs = grid[a] },
        });
        const inward: Block = if (a == 0) nest.thread_head[axes - 1] else nest.block_head[a - 1];
        const outward: Block = if (a == axes - 1) nest.exit else nest.block_latch[a + 1];
        try out.appendIf(head, in_grid, .{ .target = inward, .args = &.{zero} }, .{ .target = outward });
    }

    // The thread loops, again from the slowest axis inward. The x header dominates the body,
    // every latch, and itself, so the values it defines reach every use with no block
    // argument. It is where the builtins get supplied, and by the time this loop reaches it
    // every axis already has its induction variable and its size constant.
    var thread_id: [axes]Value = undefined;
    var block_dim: [axes]Value = undefined;
    for (0..axes) |i| {
        const a = axes - 1 - i;
        const head = nest.thread_head[a];
        thread_id[a] = try out.appendBlockParam(head, i32_t);
        block_dim[a] = try out.appendInst(head, i32_t, .{ .iconst = block[a] });
        if (a == 0) try supplyBuiltins(&out, head, i32_t, builtins.items, .{
            .thread_id = thread_id,
            .block_id = block_id,
            .block_dim = block_dim,
            .grid_dim = grid,
        });
        const in_block = try out.appendInst(head, bool_t, .{
            .icmp = .{ .op = .lt, .lhs = thread_id[a], .rhs = block_dim[a] },
        });
        const inward: ir.function.EdgeDesc = if (a == 0)
            .{ .target = nest.body_entry }
        else
            .{ .target = nest.thread_head[a - 1], .args = &.{zero} };
        const outward: Block = if (a == axes - 1) nest.block_latch[0] else nest.thread_latch[a + 1];
        try out.appendIf(head, in_block, inward, .{ .target = outward });
    }

    // One thread of the kernel finishes where the kernel returned. A block with no terminator
    // is an implicit `ret void`, so it ends the thread too and takes the same edge.
    try retireThread(&out, body_entry, nest.thread_latch[0]);
    for (1..kernel_blocks) |bi| try retireThread(&out, @enumFromInt(bi), nest.thread_latch[0]);

    for (0..axes) |a| {
        const next = try out.appendArithImm(nest.thread_latch[a], i32_t, .add, thread_id[a], 1);
        try out.setJump(nest.thread_latch[a], nest.thread_head[a], &.{next});
    }
    for (0..axes) |a| {
        const next = try out.appendArithImm(nest.block_latch[a], i32_t, .add, block_id[a], 1);
        try out.setJump(nest.block_latch[a], nest.block_head[a], &.{next});
    }

    out.setTerminator(nest.exit, .{ .ret = ir.function.Ret.none() });

    try layOutNest(allocator, &out, kernel_blocks, nest);
    return out;
}

/// The blocks the nest adds. The per-axis arrays are indexed by axis, so index 0 is x.
const NestBlocks = struct {
    body_entry: Block,
    block_head: [axes]Block,
    thread_head: [axes]Block,
    thread_latch: [axes]Block,
    block_latch: [axes]Block,
    exit: Block,
};

/// Put the blocks in an order where every block follows its immediate dominator.
///
/// The nest is built by appending, so the kernel's body keeps the low block indices and the
/// six loop headers land after it. That order is legal IR, and `ir.verify` accepts it, but the
/// machine backends number linear-scan liveness by block index. A body block that comes before
/// the header which defines the induction variables makes the register allocator read a use
/// before its definition, and the allocation it produces is unsound. `vulcan-opt.blocklayout`
/// states the same rule for the same reason.
///
/// The order is the preheader, the three workgroup headers from z inward, the three thread
/// headers from z inward, the body, the three thread latches from x outward, the three
/// workgroup latches from x outward, then the exit. Each header's immediate dominator is the
/// header outside it, and each latch's immediate dominator is a header, so every block in this
/// order follows the block that dominates it. The kernel's own blocks keep the relative order
/// they came in with, so a kernel that was itself laid out this way stays laid out this way.
fn layOutNest(
    allocator: std.mem.Allocator,
    out: *Function,
    kernel_blocks: usize,
    nest: NestBlocks,
) std.mem.Allocator.Error!void {
    const order = try allocator.alloc(Block, out.blockCount());
    defer allocator.free(order);
    order[0] = entry_block;
    for (0..axes) |i| order[1 + i] = nest.block_head[axes - 1 - i];
    for (0..axes) |i| order[1 + axes + i] = nest.thread_head[axes - 1 - i];

    const body_at = 1 + 2 * axes;
    order[body_at] = nest.body_entry;
    for (1..kernel_blocks) |bi| order[body_at + bi] = @enumFromInt(bi);

    const latches_at = body_at + kernel_blocks;
    for (0..axes) |a| order[latches_at + a] = nest.thread_latch[a];
    for (0..axes) |a| order[latches_at + axes + a] = nest.block_latch[a];
    order[latches_at + 2 * axes] = nest.exit;
    try out.reorderBlocks(allocator, order);
}

/// The values one iteration of the nest exposes, one per axis.
const Induction = struct {
    thread_id: [axes]Value,
    block_id: [axes]Value,
    block_dim: [axes]Value,
    grid_dim: [axes]Value,
};

/// Compute each builtin from `ind` in `head` and replace every use of the parameter it stood
/// for. `global_id_a` is `block_id_a * block_dim_a + thread_id_a`, per axis, which is the same
/// fusion the hardware builtin names.
fn supplyBuiltins(
    out: *Function,
    head: Block,
    i32_t: Type,
    builtins: []const BuiltinParam,
    ind: Induction,
) std.mem.Allocator.Error!void {
    for (builtins) |bp| {
        const a = bp.source.axis;
        const supplied: Value = switch (bp.source.kind) {
            .thread_id => ind.thread_id[a],
            .block_id => ind.block_id[a],
            .block_dim => ind.block_dim[a],
            .grid_dim => ind.grid_dim[a],
            .global_id => blk: {
                const scaled = try out.appendInst(head, i32_t, .{
                    .arith = .{ .op = .mul, .lhs = ind.block_id[a], .rhs = ind.block_dim[a] },
                });
                break :blk try out.appendInst(head, i32_t, .{
                    .arith = .{ .op = .add, .lhs = scaled, .rhs = ind.thread_id[a] },
                });
            },
        };
        out.replaceAllUses(bp.value, supplied);
    }
}

/// Send a kernel block that ends a thread to the innermost latch instead. A `ret` and an unset
/// terminator both end the thread, and neither may leave the nest.
fn retireThread(out: *Function, block: Block, latch: Block) std.mem.Allocator.Error!void {
    const term = out.terminator(block);
    if (term) |t| switch (t) {
        .ret => {},
        .jump => return,
    };
    try out.setJump(block, latch, &.{});
}

const i32_kind: ir.types.TypeKind = .{ .int = .{ .signedness = .signed, .bits = 32 } };

/// A kernel `fn(a: i32, b: ptr) void` that stores `a` through `b`. No builtins.
fn plainKernel(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i32_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    try func.appendStore(entry, a, b);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// A kernel `fn(v: i32 [tag], buf: ptr) void` that stores its builtin through `buf`.
fn builtinKernel(allocator: std.mem.Allocator, tag: Builtin) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const v = try func.appendBlockParam(entry, i32_t);
    const buf = try func.appendBlockParam(entry, ptr_t);
    try attrs.setBuiltin(&func, v, tag);
    try func.appendStore(entry, v, buf);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// A kernel that takes one i32 parameter per entry of `tags`, then a pointer, and stores every
/// one of those parameters through that pointer in order. The n-th store carries the n-th tag,
/// so a test reads a whole set of builtins from one lowering.
fn multiBuiltinKernel(allocator: std.mem.Allocator, tags: []const Builtin) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i32_t = try func.types.intern(i32_kind);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();

    var values: std.ArrayList(Value) = .empty;
    defer values.deinit(allocator);
    for (tags) |tag| {
        const v = try func.appendBlockParam(entry, i32_t);
        try attrs.setBuiltin(&func, v, tag);
        try values.append(allocator, v);
    }
    const buf = try func.appendBlockParam(entry, ptr_t);
    for (values.items) |v| try func.appendStore(entry, v, buf);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// The opcode that defines `v`, or null when `v` is a block parameter.
fn defOpcode(func: *const Function, v: Value) ?ir.function.Opcode {
    const inst = func.definingInst(v) orelse return null;
    return func.opcode(inst);
}

/// The value operand of the `n`-th `store` in `func`, counted over the blocks in order, or
/// null when there are fewer stores than that.
fn nthStoredValue(func: *const Function, n: usize) ?Value {
    var seen: usize = 0;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .store => |st| {
                    if (seen == n) return st.value;
                    seen += 1;
                },
                else => {},
            }
        }
    }
    return null;
}

/// The value operand of the first `store` in `func`, or null when there is none.
fn firstStoredValue(func: *const Function) ?Value {
    return nthStoredValue(func, 0);
}

test "a kernel with no builtins keeps its parameters and gains the three grid parameters" {
    // Expected case. The nest still runs, it just has nothing to feed the body.
    const allocator = std.testing.allocator;
    var kernel = try plainKernel(allocator);
    defer kernel.deinit();

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 4, 1, 1 });
    defer lowered.deinit();

    const params = lowered.blockParams(@enumFromInt(0));
    try std.testing.expectEqual(@as(usize, 5), params.len);
    for (params[0..4]) |p| {
        try std.testing.expectEqual(i32_kind, lowered.types.type_kind(lowered.valueType(p)));
    }
    try std.testing.expect(lowered.types.type_kind(lowered.valueType(params[4])) == .ptr);
}

test "a global_id_x kernel drops the builtin parameter from its signature" {
    const allocator = std.testing.allocator;
    var kernel = try builtinKernel(allocator, .global_id_x);
    defer kernel.deinit();

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 4, 1, 1 });
    defer lowered.deinit();

    const params = lowered.blockParams(@enumFromInt(0));
    try std.testing.expectEqual(@as(usize, 4), params.len);
    for (params[0..3]) |p| {
        try std.testing.expectEqual(i32_kind, lowered.types.type_kind(lowered.valueType(p)));
    }
    try std.testing.expect(lowered.types.type_kind(lowered.valueType(params[3])) == .ptr);
    // The builtin is computed, never passed, so no surviving parameter carries the tag.
    for (params) |p| try std.testing.expectEqual(@as(?Builtin, null), attrs.builtinOf(&lowered, p));
}

test "the lowered nest computes global_id_x as block_id * block_dim + thread_id" {
    const allocator = std.testing.allocator;
    var kernel = try builtinKernel(allocator, .global_id_x);
    defer kernel.deinit();

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 4, 1, 1 });
    defer lowered.deinit();

    const stored = firstStoredValue(&lowered) orelse return error.MissingStore;
    const sum = defOpcode(&lowered, stored) orelse return error.NotAnInstruction;
    try std.testing.expect(sum == .arith);
    try std.testing.expectEqual(ir.function.BinOp.add, sum.arith.op);

    const product = defOpcode(&lowered, sum.arith.lhs) orelse return error.NotAnInstruction;
    try std.testing.expect(product == .arith);
    try std.testing.expectEqual(ir.function.BinOp.mul, product.arith.op);

    // The multiply is by the declared workgroup size, and the addend is the inner induction
    // variable, which is a block parameter and so has no defining instruction.
    const size = defOpcode(&lowered, product.arith.rhs) orelse return error.NotAnInstruction;
    try std.testing.expect(size == .iconst);
    try std.testing.expectEqual(@as(i64, 4), size.iconst);
    try std.testing.expectEqual(@as(?ir.function.Opcode, null), defOpcode(&lowered, sum.arith.rhs));
}

test "global_id_y and global_id_z scale by the size of their own axis" {
    // The axis sizes are all different, so a lowering that reused the x values would show the
    // wrong constant here rather than pass by coincidence.
    const allocator = std.testing.allocator;
    const block: [3]u32 = .{ 2, 4, 8 };
    const cases = [_]struct { tag: Builtin, want: i64 }{
        .{ .tag = .global_id_x, .want = 2 },
        .{ .tag = .global_id_y, .want = 4 },
        .{ .tag = .global_id_z, .want = 8 },
    };
    for (cases) |case| {
        var kernel = try builtinKernel(allocator, case.tag);
        defer kernel.deinit();
        var lowered = try lowerToLoopNest(allocator, &kernel, block);
        defer lowered.deinit();

        const stored = firstStoredValue(&lowered) orelse return error.MissingStore;
        const sum = defOpcode(&lowered, stored) orelse return error.NotAnInstruction;
        try std.testing.expectEqual(ir.function.BinOp.add, sum.arith.op);
        const product = defOpcode(&lowered, sum.arith.lhs) orelse return error.NotAnInstruction;
        try std.testing.expectEqual(ir.function.BinOp.mul, product.arith.op);
        const size = defOpcode(&lowered, product.arith.rhs) orelse return error.NotAnInstruction;
        try std.testing.expectEqual(case.want, size.iconst);
    }
}

test "block_dim_x, block_dim_y and block_dim_z become the three declared sizes" {
    const allocator = std.testing.allocator;
    var kernel = try multiBuiltinKernel(allocator, &.{ .block_dim_x, .block_dim_y, .block_dim_z });
    defer kernel.deinit();

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 2, 4, 8 });
    defer lowered.deinit();

    const want = [_]i64{ 2, 4, 8 };
    for (want, 0..) |n, i| {
        const stored = nthStoredValue(&lowered, i) orelse return error.MissingStore;
        const size = defOpcode(&lowered, stored) orelse return error.NotAnInstruction;
        try std.testing.expect(size == .iconst);
        try std.testing.expectEqual(n, size.iconst);
    }
}

test "grid_dim_x, grid_dim_y and grid_dim_z are the three leading parameters in order" {
    const allocator = std.testing.allocator;
    var kernel = try multiBuiltinKernel(allocator, &.{ .grid_dim_x, .grid_dim_y, .grid_dim_z });
    defer kernel.deinit();

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 2, 4, 8 });
    defer lowered.deinit();

    // The grid size is not known until the call, so each one must be the parameter itself and
    // not a constant. Parameter 0 is grid_x, parameter 1 is grid_y, parameter 2 is grid_z.
    const params = lowered.blockParams(@enumFromInt(0));
    for (0..3) |i| {
        const stored = nthStoredValue(&lowered, i) orelse return error.MissingStore;
        try std.testing.expectEqual(params[i], stored);
    }
}

test "the six thread and block indices are six distinct induction variables" {
    // Each is a block parameter of one of the nest's headers, so none has a defining
    // instruction, and no two of the six may be the same value.
    const allocator = std.testing.allocator;
    const tags = [_]Builtin{
        .thread_id_x, .thread_id_y, .thread_id_z,
        .block_id_x,  .block_id_y,  .block_id_z,
    };
    var kernel = try multiBuiltinKernel(allocator, &tags);
    defer kernel.deinit();

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 2, 4, 8 });
    defer lowered.deinit();

    var seen: [tags.len]Value = undefined;
    for (0..tags.len) |i| {
        const stored = nthStoredValue(&lowered, i) orelse return error.MissingStore;
        try std.testing.expectEqual(@as(?ir.function.Opcode, null), defOpcode(&lowered, stored));
        seen[i] = stored;
    }
    for (seen, 0..) |a, i| {
        for (seen[i + 1 ..]) |b| try std.testing.expect(a != b);
    }

    // The thread loops sit inside the workgroup loops, so every thread index is created after
    // every workgroup index.
    for (seen[0..3]) |t| {
        for (seen[3..]) |b| try std.testing.expect(@intFromEnum(t) > @intFromEnum(b));
    }
}

test "a multi-block kernel body is spliced into the nest and keeps its control flow" {
    const allocator = std.testing.allocator;
    var kernel = Function.init(allocator);
    defer kernel.deinit();
    const i32_t = try kernel.types.intern(i32_kind);
    const bool_t = try kernel.types.intern(.bool);
    const ptr_t = try kernel.types.ptrGlobal();

    const entry = try kernel.appendBlock();
    const then_block = try kernel.appendBlock();
    const merge = try kernel.appendBlock();
    const gid = try kernel.appendBlockParam(entry, i32_t);
    const buf = try kernel.appendBlockParam(entry, ptr_t);
    try attrs.setBuiltin(&kernel, gid, .global_id_x);
    const odd = try kernel.appendArithImm(entry, i32_t, .bit_and, gid, 1);
    const zero = try kernel.appendInst(entry, i32_t, .{ .iconst = 0 });
    const is_odd = try kernel.appendInst(entry, bool_t, .{
        .icmp = .{ .op = .ne, .lhs = odd, .rhs = zero },
    });
    try kernel.appendIf(entry, is_odd, .{ .target = then_block }, .{ .target = merge });
    try kernel.appendStore(then_block, gid, buf);
    try kernel.setJump(then_block, merge, &.{});
    kernel.setTerminator(merge, .{ .ret = ir.function.Ret.none() });

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 4, 1, 1 });
    defer lowered.deinit();

    try std.testing.expectEqual(@as(usize, 3 + nest_blocks), lowered.blockCount());

    var diags = try ir.verify.verify(allocator, &lowered, .high);
    defer diags.deinit();
    try std.testing.expectEqual(@as(usize, 0), diags.count());
}

test "every axis of every supported grid builtin lowers" {
    // The full supported set, in one place, so a builtin that quietly stops lowering is caught
    // here rather than in whichever execution test happened to use it.
    const allocator = std.testing.allocator;
    const supported = [_]Builtin{
        .thread_id_x, .thread_id_y, .thread_id_z,
        .block_id_x,  .block_id_y,  .block_id_z,
        .block_dim_x, .block_dim_y, .block_dim_z,
        .grid_dim_x,  .grid_dim_y,  .grid_dim_z,
        .global_id_x, .global_id_y, .global_id_z,
    };
    try std.testing.expectEqual(@as(usize, 15), supported.len);
    for (supported) |tag| {
        var kernel = try builtinKernel(allocator, tag);
        defer kernel.deinit();
        var lowered = try lowerToLoopNest(allocator, &kernel, .{ 2, 4, 8 });
        defer lowered.deinit();

        // The builtin left the signature, so only the three grid values and the buffer remain.
        try std.testing.expectEqual(@as(usize, 4), lowered.blockParams(@enumFromInt(0)).len);
        var diags = try ir.verify.verify(allocator, &lowered, .high);
        defer diags.deinit();
        try std.testing.expectEqual(@as(usize, 0), diags.count());
    }
}

test "the subgroup builtins are rejected rather than guessed at" {
    // A subgroup is a hardware partition of a workgroup. The host nest runs one thread at a
    // time, so any value it invented for these would be a quiet wrong answer.
    const allocator = std.testing.allocator;
    const rejected = [_]Builtin{ .lane_id, .warp_id, .subgroup_size };
    for (rejected) |tag| {
        var kernel = try builtinKernel(allocator, tag);
        defer kernel.deinit();
        try std.testing.expectError(
            error.Unsupported,
            lowerToLoopNest(allocator, &kernel, .{ 1, 1, 1 }),
        );
    }
}

test "the graphics builtins are rejected because this pass lowers compute kernels" {
    const allocator = std.testing.allocator;
    const rejected = [_]Builtin{
        .vertex_index, .instance_index, .frag_coord, .point_coord, .front_facing,
    };
    for (rejected) |tag| {
        var kernel = try builtinKernel(allocator, tag);
        defer kernel.deinit();
        try std.testing.expectError(
            error.Unsupported,
            lowerToLoopNest(allocator, &kernel, .{ 1, 1, 1 }),
        );
    }
}

test "a value-returning kernel is rejected" {
    const allocator = std.testing.allocator;
    var kernel = Function.init(allocator);
    defer kernel.deinit();
    const i32_t = try kernel.types.intern(i32_kind);
    const entry = try kernel.appendBlock();
    const a = try kernel.appendBlockParam(entry, i32_t);
    kernel.setTerminator(entry, .{ .ret = ir.function.Ret.one(a) });

    try std.testing.expectError(
        error.Unsupported,
        lowerToLoopNest(allocator, &kernel, .{ 1, 1, 1 }),
    );
}

test "a builtin parameter that is not a 32-bit integer is rejected" {
    // The induction variables are i32, and an arithmetic op needs both operands to share a
    // type, so a wider builtin has no correct lowering here.
    const allocator = std.testing.allocator;
    var kernel = Function.init(allocator);
    defer kernel.deinit();
    const i64_t = try kernel.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try kernel.types.ptrGlobal();
    const entry = try kernel.appendBlock();
    const gid = try kernel.appendBlockParam(entry, i64_t);
    const buf = try kernel.appendBlockParam(entry, ptr_t);
    try attrs.setBuiltin(&kernel, gid, .global_id_x);
    try kernel.appendStore(entry, gid, buf);
    kernel.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(
        error.Unsupported,
        lowerToLoopNest(allocator, &kernel, .{ 4, 1, 1 }),
    );
}

test "a kernel whose entry block is a branch target is rejected" {
    // The entry becomes the preheader, so an edge back to it would re-enter the preheader once
    // per iteration and reset the nest.
    const allocator = std.testing.allocator;
    var kernel = Function.init(allocator);
    defer kernel.deinit();
    const i32_t = try kernel.types.intern(i32_kind);
    const bool_t = try kernel.types.intern(.bool);
    const entry = try kernel.appendBlock();
    const done = try kernel.appendBlock();
    const gid = try kernel.appendBlockParam(entry, i32_t);
    try attrs.setBuiltin(&kernel, gid, .global_id_x);
    const zero = try kernel.appendInst(entry, i32_t, .{ .iconst = 0 });
    const again = try kernel.appendInst(entry, bool_t, .{
        .icmp = .{ .op = .ne, .lhs = gid, .rhs = zero },
    });
    try kernel.appendIf(entry, again, .{ .target = entry }, .{ .target = done });
    kernel.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(
        error.Unsupported,
        lowerToLoopNest(allocator, &kernel, .{ 4, 1, 1 }),
    );
}

test "a kernel that carries a block attribute is rejected" {
    // `reorderBlocks` does not remap a block id inside an attribute payload, and the layout
    // step reorders, so such a kernel has no safe lowering here.
    const allocator = std.testing.allocator;
    var kernel = try builtinKernel(allocator, .global_id_x);
    defer kernel.deinit();
    try kernel.addAttr(.{ .block = @enumFromInt(0) }, .cold);

    try std.testing.expectError(
        error.Unsupported,
        lowerToLoopNest(allocator, &kernel, .{ 4, 1, 1 }),
    );
}

test "the lowered function passes the IR verifier" {
    // The strongest structural check available: a missing block argument, a dangling edge, or
    // a bad terminator all show up here.
    const allocator = std.testing.allocator;
    var kernel = Function.init(allocator);
    defer kernel.deinit();
    const i32_t = try kernel.types.intern(i32_kind);
    const ptr_t = try kernel.types.ptrGlobal();
    const entry = try kernel.appendBlock();
    const gid = try kernel.appendBlockParam(entry, i32_t);
    const buf = try kernel.appendBlockParam(entry, ptr_t);
    try attrs.setBuiltin(&kernel, gid, .global_id_x);
    try attrs.setLocalSize(&kernel, .{ 32, 1, 1 });
    const offset = try kernel.appendArithImm(entry, i32_t, .mul, gid, 4);
    const slot = try kernel.appendInst(entry, ptr_t, .{
        .arith = .{ .op = .add, .lhs = buf, .rhs = offset },
    });
    const tripled = try kernel.appendArithImm(entry, i32_t, .mul, gid, 3);
    try kernel.appendStore(entry, tripled, slot);
    kernel.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    var lowered = try lowerToLoopNest(allocator, &kernel, attrs.localSize(&kernel));
    defer lowered.deinit();

    var diags = try ir.verify.verify(allocator, &lowered, .high);
    defer diags.deinit();
    try std.testing.expectEqual(@as(usize, 0), diags.count());
    // The workgroup size came off the kernel's own attribute, so the nest's x bound is 32.
    try std.testing.expectEqual(@as(usize, 1 + nest_blocks), lowered.blockCount());
}
