//! Integer register allocation under loop pressure, executed on qemu-riscv64 (the oracle). Two
//! independent properties the riscv64 allocator gained:
//!
//!   1. INTEGER BLOCK-PARAM SPILL. A loop that carries more simultaneously-live integer block params
//!      than the 17 allocatable integer registers used to fail `allocateRegisters` with
//!      `error.Unsupported` (only instruction results spilled; block params did not). Now a non-entry
//!      int block param that cannot get a register spills to a stack slot, and the block-edge parallel
//!      move stores/reloads it through the int spill scratch registers, exactly as the float/vector
//!      classes already do. The first test carries 26 loop-live int params and checks the spilled
//!      round-trip computes the right sum.
//!
//!   2. ACROSS-BACK-EDGE LIVENESS. A value defined in the entry block and read inside a loop body,
//!      but NOT threaded as a loop block param, is live across the loop's back-edge. The old naive
//!      forward liveness pass ended that value's live range at its textual use in the body, so the
//!      register was freed and immediately reused by the next body temp (the free list is LIFO); the
//!      next iteration then read that temp instead of the invariant, a silent miscompile.
//!      `extendLiveRanges` (a live-in/out CFG fixpoint) now keeps the value live across the whole
//!      body. The second test builds exactly that shape and checks it computes correctly - before the
//!      fix it read garbage.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

/// Loop-carried accumulators. 24 accumulators plus the induction variable `i` and the bound `n`
/// gives 26 integer values live at once across the loop, well past the 17 allocatable integer
/// registers, so at least nine of them must spill to the stack.
const n_acc = 24;

/// Build a single-loop function that threads `i`, `n`, and `n_acc` accumulators as block params
/// (26 simultaneously-live integers) so the integer file is exhausted and the surplus params spill:
///
///   f(n): a[k] = k+1 for k in 0..n_acc; for i in 0..n: a[k] += i; return sum(a)
///
/// Every accumulator is summed at the end, so all stay live across the loop (nothing can be dropped).
/// The updates and the final reduction are pure `arith`/`icmp`, whose operands reload from spill slots
/// transparently, and the back-edge jump carries all 26 params so the edge-move exercises the spilled
/// param store/reload path on both sides.
fn buildAccumNest(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i64_t);

    // entry: initial i = 0 and a[k] = k+1, then jump to the header with the whole carried set.
    const iv0 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    var init_acc: [n_acc]Value = undefined;
    for (0..n_acc) |k| init_acc[k] = try func.appendInst(entry, i64_t, .{ .iconst = @intCast(k + 1) });
    {
        var edge: [2 + n_acc]Value = undefined;
        edge[0] = iv0;
        edge[1] = n;
        for (0..n_acc) |k| edge[2 + k] = init_acc[k];
        try func.setJump(entry, header, &edge);
    }

    // header params: i, n, a[0..n_acc].
    const h_i = try func.appendBlockParam(header, i64_t);
    const h_n = try func.appendBlockParam(header, i64_t);
    var h_acc: [n_acc]Value = undefined;
    for (0..n_acc) |k| h_acc[k] = try func.appendBlockParam(header, i64_t);
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = h_i, .rhs = h_n } });
    {
        var then_args: [2 + n_acc]Value = undefined;
        then_args[0] = h_i;
        then_args[1] = h_n;
        for (0..n_acc) |k| then_args[2 + k] = h_acc[k];
        var else_args: [n_acc]Value = undefined;
        for (0..n_acc) |k| else_args[k] = h_acc[k];
        try func.appendIf(header, cond, .{ .target = body, .args = &then_args }, .{ .target = exit, .args = &else_args });
    }

    // body params: i, n, a[0..n_acc]. Update a[k] += i, i += 1, jump back to the header.
    const b_i = try func.appendBlockParam(body, i64_t);
    const b_n = try func.appendBlockParam(body, i64_t);
    var b_acc: [n_acc]Value = undefined;
    for (0..n_acc) |k| b_acc[k] = try func.appendBlockParam(body, i64_t);
    var next_acc: [n_acc]Value = undefined;
    for (0..n_acc) |k| next_acc[k] = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = b_acc[k], .rhs = b_i } });
    const next_i = try func.appendArithImm(body, i64_t, .add, b_i, 1);
    {
        var edge: [2 + n_acc]Value = undefined;
        edge[0] = next_i;
        edge[1] = b_n;
        for (0..n_acc) |k| edge[2 + k] = next_acc[k];
        try func.setJump(body, header, &edge);
    }

    // exit params: a[0..n_acc]. Sum them and return.
    var e_acc: [n_acc]Value = undefined;
    for (0..n_acc) |k| e_acc[k] = try func.appendBlockParam(exit, i64_t);
    var sum = e_acc[0];
    for (1..n_acc) |k| sum = try func.appendInst(exit, i64_t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = e_acc[k] } });
    func.setTerminator(exit, .{ .ret = sum });

    return func;
}

/// The scalar reference for `buildAccumNest`, computed exactly as the IR does.
fn accumReference(n: i64) i64 {
    var a: [n_acc]i64 = undefined;
    for (0..n_acc) |k| a[k] = @intCast(k + 1);
    var i: i64 = 0;
    while (i < n) : (i += 1) {
        for (0..n_acc) |k| a[k] += i;
    }
    var sum: i64 = 0;
    for (a) |v| sum += v;
    return sum;
}

test "int-spill: a 26-live-int-param loop nest spills its surplus params and computes correctly (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = try buildAccumNest(allocator);
    defer func.deinit();

    const n: i64 = 7;
    const got = harness.runFunc(std.testing.io, allocator, &func, &.{n}, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(accumReference(n), got);
}

/// Build a loop whose invariant `C` is defined in the entry block and read inside the body but is NOT
/// threaded as a block param, so it is live across the back-edge:
///
///   f(n): C = n + n; acc = 0; for i in 0..n: acc = (acc + C) + i; return acc
///
/// `C` derives from the runtime argument (so it cannot be folded into an immediate or a constant, and
/// stays a real register value), and its only textual use is `acc + C` at the top of the body. Without
/// live-range extension the allocator frees `C`'s register right after that use, and the very next body
/// temp (`(acc+C) + i`) reuses it (the free list is LIFO), so the next iteration reads the temp instead
/// of `C`. `extendLiveRanges` keeps `C` live to the end of the body, so its register is not reused and
/// every iteration reads the true invariant.
fn buildInvariantLoop(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const exit = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i64_t);
    // C = n + n: a genuine SSA value (not a constant), used only inside the body.
    const c = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = n, .rhs = n } });
    const iv0 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const acc0 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ iv0, n, acc0 });

    const h_i = try func.appendBlockParam(header, i64_t);
    const h_n = try func.appendBlockParam(header, i64_t);
    const h_acc = try func.appendBlockParam(header, i64_t);
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = h_i, .rhs = h_n } });
    try func.appendIf(header, cond, .{ .target = body, .args = &.{ h_i, h_n, h_acc } }, .{ .target = exit, .args = &.{h_acc} });

    const b_i = try func.appendBlockParam(body, i64_t);
    const b_n = try func.appendBlockParam(body, i64_t);
    const b_acc = try func.appendBlockParam(body, i64_t);
    // s = acc + C reads the across-back-edge invariant; acc1 = s + i is the temp that would reuse
    // C's register without the live-range extension.
    const s = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = b_acc, .rhs = c } });
    const acc1 = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = b_i } });
    const next_i = try func.appendArithImm(body, i64_t, .add, b_i, 1);
    try func.setJump(body, header, &.{ next_i, b_n, acc1 });

    const e_acc = try func.appendBlockParam(exit, i64_t);
    func.setTerminator(exit, .{ .ret = e_acc });

    return func;
}

/// The scalar reference for `buildInvariantLoop`, computed exactly as the IR does.
fn invariantReference(n: i64) i64 {
    const c = n + n;
    var acc: i64 = 0;
    var i: i64 = 0;
    while (i < n) : (i += 1) acc = (acc + c) + i;
    return acc;
}

test "int-spill: an entry value read across a loop back-edge keeps its register (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = try buildInvariantLoop(allocator);
    defer func.deinit();

    const n: i64 = 5;
    const got = harness.runFunc(std.testing.io, allocator, &func, &.{n}, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(invariantReference(n), got);
}
