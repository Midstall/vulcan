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

/// The number of blocks the nest adds to the kernel's own block count: the spliced body
/// entry, the two loop headers, the two latches, and the exit.
const nest_blocks: usize = 6;

/// A builtin this pass can supply from the nest's induction variables.
const NestSource = enum { thread_id, block_id, block_dim, global_id };

/// The nest source for `b`, or null when this pass cannot supply it. Only the x axis is in
/// scope. The y and z axes, the grid size, and the subgroup builtins are a follow-up.
fn nestSource(b: Builtin) ?NestSource {
    return switch (b) {
        .thread_id_x => .thread_id,
        .block_id_x => .block_id,
        .block_dim_x => .block_dim,
        .global_id_x => .global_id,
        .thread_id_y,
        .thread_id_z,
        .block_id_y,
        .block_id_z,
        .block_dim_y,
        .block_dim_z,
        .grid_dim_x,
        .grid_dim_y,
        .grid_dim_z,
        .global_id_y,
        .global_id_z,
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

/// Rewrite `func`, a kernel, into an ordinary function that executes the whole grid.
///
/// The result's entry block takes the grid size in workgroups as ONE leading i32 parameter,
/// then the kernel's non-builtin parameters in their original order. Every builtin parameter
/// becomes a value computed from the loop induction variables, so the caller supplies only
/// real data.
///
/// The nest is `for (block_id in 0..grid_x) for (thread_id in 0..block[0])`, with the kernel
/// body as the inner body. `block` is the declared workgroup size, which the caller reads from
/// `attrs.localSize`.
///
/// Only the x axis is in scope. A kernel that reads a y or z axis builtin, the grid size, or a
/// subgroup builtin returns `error.Unsupported`, as does a kernel whose builtin parameter is
/// not a signed 32-bit integer, and a kernel whose entry block is a branch target.
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

    // Clone first: `clone` re-interns every type kind in order, so the n-th kind keeps handle
    // n and every Type in the copied arrays stays valid. The kernel's own blocks keep their
    // indices too, so its internal branch targets need no remap.
    var out = try func.clone(allocator);
    errdefer out.deinit();

    const i32_t = try out.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try out.types.intern(.bool);

    // Split the kernel's entry parameters. A builtin parameter leaves the signature and the
    // nest computes it; every other parameter keeps its place, one slot further along.
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

    const outer_head = try out.appendBlock();
    const inner_head = try out.appendBlock();
    const inner_latch = try out.appendBlock();
    const outer_latch = try out.appendBlock();
    const exit = try out.appendBlock();

    // The preheader: the grid size in workgroups, then the kernel's real parameters.
    const grid_x = try out.newParam(entry_block, i32_t);
    var params: std.ArrayList(Value) = .empty;
    defer params.deinit(allocator);
    try params.append(allocator, grid_x);
    try params.appendSlice(allocator, real.items);
    try out.setBlockParams(entry_block, params.items);

    const zero = try out.appendInst(entry_block, i32_t, .{ .iconst = 0 });
    try out.setJump(entry_block, outer_head, &.{zero});

    // The outer loop over workgroups. `if` does not terminate a block, so the header needs no
    // terminator of its own: both of its edges leave the block.
    const block_id = try out.appendBlockParam(outer_head, i32_t);
    const outer_test = try out.appendInst(outer_head, bool_t, .{
        .icmp = .{ .op = .lt, .lhs = block_id, .rhs = grid_x },
    });
    try out.appendIf(
        outer_head,
        outer_test,
        .{ .target = inner_head, .args = &.{zero} },
        .{ .target = exit },
    );

    // The inner loop over the threads of one workgroup. The header dominates the body, both
    // latches, and itself, so the induction variables reach every use without block arguments.
    const thread_id = try out.appendBlockParam(inner_head, i32_t);
    const block_dim = try out.appendInst(inner_head, i32_t, .{ .iconst = block[0] });
    try supplyBuiltins(&out, inner_head, i32_t, builtins.items, .{
        .thread_id = thread_id,
        .block_id = block_id,
        .block_dim = block_dim,
    });
    const inner_test = try out.appendInst(inner_head, bool_t, .{
        .icmp = .{ .op = .lt, .lhs = thread_id, .rhs = block_dim },
    });
    try out.appendIf(
        inner_head,
        inner_test,
        .{ .target = body_entry },
        .{ .target = outer_latch },
    );

    // One thread of the kernel finishes where the kernel returned. A block with no terminator
    // is an implicit `ret void`, so it ends the thread too and takes the same edge.
    try retireThread(&out, body_entry, inner_latch);
    for (1..kernel_blocks) |bi| try retireThread(&out, @enumFromInt(bi), inner_latch);

    const next_thread = try out.appendArithImm(inner_latch, i32_t, .add, thread_id, 1);
    try out.setJump(inner_latch, inner_head, &.{next_thread});

    const next_block = try out.appendArithImm(outer_latch, i32_t, .add, block_id, 1);
    try out.setJump(outer_latch, outer_head, &.{next_block});

    out.setTerminator(exit, .{ .ret = ir.function.Ret.none() });

    return out;
}

/// The induction values one iteration of the nest exposes.
const Induction = struct { thread_id: Value, block_id: Value, block_dim: Value };

/// Compute each builtin from `ind` in `head` and replace every use of the parameter it stood
/// for. `global_id_x` is `block_id * block_dim + thread_id`, the same fusion the hardware
/// builtin names.
fn supplyBuiltins(
    out: *Function,
    head: Block,
    i32_t: Type,
    builtins: []const BuiltinParam,
    ind: Induction,
) std.mem.Allocator.Error!void {
    for (builtins) |bp| {
        const supplied: Value = switch (bp.source) {
            .thread_id => ind.thread_id,
            .block_id => ind.block_id,
            .block_dim => ind.block_dim,
            .global_id => blk: {
                const scaled = try out.appendInst(head, i32_t, .{
                    .arith = .{ .op = .mul, .lhs = ind.block_id, .rhs = ind.block_dim },
                });
                break :blk try out.appendInst(head, i32_t, .{
                    .arith = .{ .op = .add, .lhs = scaled, .rhs = ind.thread_id },
                });
            },
        };
        out.replaceAllUses(bp.value, supplied);
    }
}

/// Send a kernel block that ends a thread to the inner latch instead. A `ret` and an unset
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

/// The opcode that defines `v`, or null when `v` is a block parameter.
fn defOpcode(func: *const Function, v: Value) ?ir.function.Opcode {
    const inst = func.definingInst(v) orelse return null;
    return func.opcode(inst);
}

/// The value operand of the first `store` in `func`, or null when there is none.
fn firstStoredValue(func: *const Function) ?Value {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            switch (func.opcode(inst)) {
                .store => |st| return st.value,
                else => {},
            }
        }
    }
    return null;
}

test "a kernel with no builtins keeps its parameters and gains the grid parameter" {
    // Expected case. The nest still runs, it just has nothing to feed the body.
    const allocator = std.testing.allocator;
    var kernel = try plainKernel(allocator);
    defer kernel.deinit();

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 4, 1, 1 });
    defer lowered.deinit();

    const params = lowered.blockParams(@enumFromInt(0));
    try std.testing.expectEqual(@as(usize, 3), params.len);
    try std.testing.expectEqual(i32_kind, lowered.types.type_kind(lowered.valueType(params[0])));
    try std.testing.expectEqual(i32_kind, lowered.types.type_kind(lowered.valueType(params[1])));
    try std.testing.expect(lowered.types.type_kind(lowered.valueType(params[2])) == .ptr);
}

test "a global_id_x kernel drops the builtin parameter from its signature" {
    const allocator = std.testing.allocator;
    var kernel = try builtinKernel(allocator, .global_id_x);
    defer kernel.deinit();

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 4, 1, 1 });
    defer lowered.deinit();

    const params = lowered.blockParams(@enumFromInt(0));
    try std.testing.expectEqual(@as(usize, 2), params.len);
    try std.testing.expectEqual(i32_kind, lowered.types.type_kind(lowered.valueType(params[0])));
    try std.testing.expect(lowered.types.type_kind(lowered.valueType(params[1])) == .ptr);
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

test "block_dim_x becomes the declared workgroup size as a constant" {
    const allocator = std.testing.allocator;
    var kernel = try builtinKernel(allocator, .block_dim_x);
    defer kernel.deinit();

    var lowered = try lowerToLoopNest(allocator, &kernel, .{ 64, 1, 1 });
    defer lowered.deinit();

    const stored = firstStoredValue(&lowered) orelse return error.MissingStore;
    const size = defOpcode(&lowered, stored) orelse return error.NotAnInstruction;
    try std.testing.expect(size == .iconst);
    try std.testing.expectEqual(@as(i64, 64), size.iconst);
}

test "thread_id_x and block_id_x come from the two induction variables" {
    // Both are block parameters of the nest's headers, so neither has a defining instruction,
    // and the two must be different values.
    const allocator = std.testing.allocator;
    var tid_kernel = try builtinKernel(allocator, .thread_id_x);
    defer tid_kernel.deinit();
    var tid_lowered = try lowerToLoopNest(allocator, &tid_kernel, .{ 8, 1, 1 });
    defer tid_lowered.deinit();
    const tid = firstStoredValue(&tid_lowered) orelse return error.MissingStore;
    try std.testing.expectEqual(@as(?ir.function.Opcode, null), defOpcode(&tid_lowered, tid));

    var bid_kernel = try builtinKernel(allocator, .block_id_x);
    defer bid_kernel.deinit();
    var bid_lowered = try lowerToLoopNest(allocator, &bid_kernel, .{ 8, 1, 1 });
    defer bid_lowered.deinit();
    const bid = firstStoredValue(&bid_lowered) orelse return error.MissingStore;
    try std.testing.expectEqual(@as(?ir.function.Opcode, null), defOpcode(&bid_lowered, bid));

    // The inner induction variable is created after the outer one, so the thread index is the
    // later value of the two.
    try std.testing.expect(@intFromEnum(tid) > @intFromEnum(bid));
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

test "a y-axis builtin is rejected rather than silently lowered as x" {
    // Suspicious case. This is the failure that would produce wrong answers quietly.
    const allocator = std.testing.allocator;
    var kernel = try builtinKernel(allocator, .global_id_y);
    defer kernel.deinit();
    try std.testing.expectError(
        error.Unsupported,
        lowerToLoopNest(allocator, &kernel, .{ 1, 1, 1 }),
    );
}

test "the out-of-scope compute builtins are all rejected" {
    const allocator = std.testing.allocator;
    const rejected = [_]Builtin{
        .thread_id_y, .thread_id_z, .block_id_y, .block_id_z,    .block_dim_y,
        .block_dim_z, .grid_dim_x,  .grid_dim_y, .grid_dim_z,    .global_id_y,
        .global_id_z, .lane_id,     .warp_id,    .subgroup_size,
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
    // The workgroup size came off the kernel's own attribute, so the nest's bound is 32.
    try std.testing.expectEqual(@as(usize, 1 + nest_blocks), lowered.blockCount());
}
