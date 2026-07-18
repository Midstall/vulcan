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

/// Straight-line integer values whose result depends on far more simultaneously-live values than the
/// 17 allocatable integer registers, so instruction results must spill. 30 keeps roughly half of them
/// in memory at once and drives the eviction path (a defined result evicts an active value whose next
/// use is furthest, or spills itself).
const n_fan = 30;

/// Build a single-block function whose result depends on `n_fan` simultaneously-live integer values,
/// exhausting the integer file at instruction-RESULT sites (not block params) so eviction drives which
/// value spills:
///
///   f(n): a[k] = n*(k+1) + k for k in 0..n_fan; return sum_k a[k]
///
/// Every `a[k]` is created before any is consumed (so all `n_fan` are live at once), then reduced in a
/// staggered order that gives each a distinct next-use position, so the Belady victim genuinely differs
/// as the reduction proceeds. Whether a value lands in a register or a spill slot, the reduction must
/// reload every operand correctly and return the same total the Zig reference computes.
fn buildFanReduce(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });

    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);

    // a[k] = n*(k+1) + k. All created first, so all n_fan are live simultaneously at the last one.
    var a: [n_fan]Value = undefined;
    for (0..n_fan) |k| {
        const coeff = try func.appendInst(entry, i64_t, .{ .iconst = @intCast(k + 1) });
        const prod = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = n, .rhs = coeff } });
        a[k] = try func.appendArithImm(entry, i64_t, .add, prod, @intCast(k));
    }

    // Reduce in a staggered order (evens ascending, then odds descending) so the reduction consumes
    // the fan with varied, non-monotonic next-use distances. Addition is associative and commutative,
    // so the total matches the reference regardless of order.
    var sum = a[0];
    var k: usize = 2;
    while (k < n_fan) : (k += 2) sum = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = a[k] } });
    var j: usize = if (n_fan % 2 == 0) n_fan - 1 else n_fan - 2;
    while (j >= 1) : (j -= 2) {
        sum = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = a[j] } });
        if (j == 1) break;
    }
    func.setTerminator(entry, .{ .ret = sum });

    return func;
}

/// The scalar reference for `buildFanReduce`, computed with wrapping arithmetic to match the hardware.
fn fanReference(n: i64) i64 {
    var sum: i64 = 0;
    for (0..n_fan) |k| {
        const term = n *% @as(i64, @intCast(k + 1)) +% @as(i64, @intCast(k));
        sum +%= term;
    }
    return sum;
}

test "int-spill: a result depending on many spilled/evicted integer values reduces correctly (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = try buildFanReduce(allocator);
    defer func.deinit();

    // Sweep several inputs (including negatives and a large magnitude) so a wrong eviction decision
    // that reloads the wrong value would diverge from the reference on at least one of them.
    const inputs = [_]i64{ 0, 1, 2, 7, -3, 100, -1000, 123456 };
    for (inputs) |n| {
        const got = harness.runFunc(std.testing.io, allocator, &func, &.{n}, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(fanReference(n), got);
    }
}

const n_split = 24;

/// Build a single-block function that forces an intra-block TAIL SPLIT: `n_split` values are all
/// created first (so all are live at once, exhausting the integer file), then reduced in REVERSE
/// creation order so the earliest-defined values have the FARTHEST next use. Belady evicts those
/// far-next-use values, and since each is an intra-block value with a live register prefix, eviction
/// tail-splits it (register prefix `[def, p)`, stack slot `[p, end)`) rather than whole-spilling. The
/// late reduction then reads the tail from its slot, so a wrong store/reload diverges from the sum.
///
///   f(n): a[k] = n*(k+1) + k for k in 0..n_split; return a[n_split-1] + a[n_split-2] + ... + a[0]
fn buildTailSplit(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });

    const entry = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);

    var a: [n_split]Value = undefined;
    for (0..n_split) |k| {
        const coeff = try func.appendInst(entry, i64_t, .{ .iconst = @intCast(k + 1) });
        const prod = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = n, .rhs = coeff } });
        a[k] = try func.appendArithImm(entry, i64_t, .add, prod, @intCast(k));
    }

    // Reduce from the last-defined value down to the first. The first-defined `a[0]` is used last, so
    // its next use is the farthest at every eviction point: exactly the Belady victim to tail-split.
    var sum = a[n_split - 1];
    var k: usize = n_split - 1;
    while (k > 0) {
        k -= 1;
        sum = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = a[k] } });
    }
    func.setTerminator(entry, .{ .ret = sum });

    return func;
}

/// The scalar reference for `buildTailSplit`, computed with wrapping arithmetic to match the hardware.
fn tailSplitReference(n: i64) i64 {
    var sum: i64 = 0;
    for (0..n_split) |k| {
        const term = n *% @as(i64, @intCast(k + 1)) +% @as(i64, @intCast(k));
        sum +%= term;
    }
    return sum;
}

test "int-spill: intra-block tail split reloads the correct int value (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = try buildTailSplit(allocator);
    defer func.deinit();

    // First prove a REAL split happens: allocation must produce at least one segmented int value
    // (a register prefix plus a stack-slot tail). Without tail-splitting this is 0 and the test is
    // meaningless, so assert it before executing.
    const splits = try isel.splitCountForTest(allocator, &func);
    try std.testing.expect(splits > 0);

    // Then execute on qemu across a sweep (negatives and a large magnitude included): a store/reload
    // that saved or reloaded the wrong register diverges from the reference on at least one input.
    const inputs = [_]i64{ 0, 1, 2, 7, -3, 100, -1000, 123456 };
    for (inputs) |n| {
        const got = harness.runFunc(std.testing.io, allocator, &func, &.{n}, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(tailSplitReference(n), got);
    }
}

/// Number of pressure terms in the re-home kernel. Twenty independent `a*k + b` terms exhaust the 17
/// allocatable integer registers, so the early-defined, late-used `t0` tail-splits under pressure.
const n_rehome = 20;

/// Build a single-block function whose first value `t0 = a*b` is used ONLY at the very end, spanning a
/// 20-term pressure block that tail-splits it (register prefix + slot tail). As the reduction consumes
/// the terms the register pressure drops, so by `t0`'s late use a register is free again and
/// second-chance RE-HOMES `t0` into it: its final use reads that register instead of reloading the slot.
///
///   f(a, b): t0 = a*b; for k in 1..=20: term[k] = a*k + b; return sum(term) + t0
fn buildReHome(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });

    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i64_t);
    const b = try func.appendBlockParam(entry, i64_t);

    // t0 is defined first and used only at the end: a long intra live range across the pressure.
    const t0 = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    var term: [n_rehome]Value = undefined;
    for (0..n_rehome) |k| {
        const kc = try func.appendInst(entry, i64_t, .{ .iconst = @intCast(k + 1) });
        const ak = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        term[k] = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = b } });
    }
    var acc = term[0];
    for (1..n_rehome) |k| acc = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = term[k] } });
    // The late use of t0: with second-chance it reads a re-homed register, not a per-use slot reload.
    const result = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = t0 } });
    func.setTerminator(entry, .{ .ret = result });

    return func;
}

/// The scalar reference for `buildReHome`, wrapping arithmetic to match the target's 64-bit ops.
fn reHomeReference(a: i64, b: i64) i64 {
    var sum: i64 = 0;
    var k: i64 = 1;
    while (k <= n_rehome) : (k += 1) sum +%= (a *% k) +% b;
    return sum +% (a *% b);
}

test "int-spill: second-chance reload re-homes a spilled int value (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = try buildReHome(allocator);
    defer func.deinit();

    // Meaningful-differential gate: a second-chance re-home MUST have fired (a `.reg` segment after a
    // `.slot` segment). Without it this would only exercise the Task 6c tail-split + per-use reload.
    try std.testing.expect(try isel.debugReHomeCount(allocator, &func) > 0);

    // Then execute on qemu across a sweep (zero, unit, negatives, and large magnitudes): a re-home
    // that reloaded the wrong bits, or failed to reload at all, diverges from the reference.
    const inputs = [_][2]i64{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const got = harness.runFunc(std.testing.io, allocator, &func, &.{ in[0], in[1] }, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(reHomeReference(in[0], in[1]), got);
    }
}

/// Build the terminator-drain regression: `t0 = a*b` is defined first and used ONLY by the `ret` (a
/// non-edge-arg operand, so t0 is intra-splittable). The 20-term pressure block tail-splits t0, then
/// as the reduction drains the terms a register frees and second-chance RE-HOMES t0 with a `.reload`
/// recorded AT the terminator position (block_end). The per-instruction drain never reaches that
/// position, so before the terminator drain the reload was dropped and `ret` returned a stale register.
/// The reduction's final `acc` is deliberately unreturned: it exists only to raise then relieve pressure.
///
///   f(a, b): t0 = a*b; for k in 1..=20: term[k] = a*k + b; return t0
fn buildTerminatorReHome(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });

    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i64_t);
    const b = try func.appendBlockParam(entry, i64_t);

    const t0 = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    var term: [n_rehome]Value = undefined;
    for (0..n_rehome) |k| {
        const kc = try func.appendInst(entry, i64_t, .{ .iconst = @intCast(k + 1) });
        const ak = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = kc } });
        term[k] = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = ak, .rhs = b } });
    }
    var acc = term[0];
    for (1..n_rehome) |k| acc = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = term[k] } });
    // Return t0 directly, so its sole use is the terminator and any re-home of it lands there.
    func.setTerminator(entry, .{ .ret = t0 });

    return func;
}

test "int-spill: second-chance re-homes a ret-only int value AT the terminator (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = try buildTerminatorReHome(allocator);
    defer func.deinit();

    // Meaningful-differential gate: a second-chance re-home MUST have fired. t0's only use is the
    // terminator, so a re-home of it records its `.reload` AT the terminator position, exactly the
    // action the new drain must emit.
    try std.testing.expect(try isel.debugReHomeCount(allocator, &func) > 0);

    // Compile through `selectFunction` directly (NOT the scheduler harness): the scheduler would
    // reorder the block and perturb the exact terminator re-home this regression depends on. The
    // allocation that runs must be the one whose second-chance reload lands on the terminator.
    const code = try isel.selectFunction(allocator, &func);
    defer allocator.free(code);

    // Execute on qemu across a sweep (zero, unit, negatives, large magnitudes): before the terminator
    // drain, the dropped re-home reload makes `ret` return a stale register instead of a*b.
    const inputs = [_][2]i64{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const got = harness.runCode(std.testing.io, allocator, code, &.{ in[0], in[1] }, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(in[0] *% in[1], got);
    }
}

/// Number of early split products in the decline kernel. Six `a*(k+1)` products are defined first and
/// used only in the middle window, so they tail-split under the peak pressure.
const n_decline_split = 6;

/// Number of twice-used residents in the decline kernel. Each `b + const` resident is used in a first
/// reduction AND in a second reduction, so it stays live across the middle split-value window and
/// keeps the register file busy there. Twelve residents plus the accumulator sustain enough pressure
/// that, at many second-chance points, no integer register is free to re-home a pending split value
/// (it reloads per use until pressure eases). riscv64 runs second-chance at EVERY position, so a
/// starved value re-homes as soon as a register does free, but the DECLINE path is exercised richly
/// meanwhile.
const n_decline_res = 12;

/// Build a single-block kernel that drives the register file hard enough to exercise BOTH the
/// second-chance re-home path AND its decline path (a pending split value with no free register at a
/// given point). The split products are used late, bracketed by two reductions over the twice-used
/// residents, so the file stays busy across their uses:
///
///   f(a, b): sp[k] = a*(k+1); res[k] = b + (100+k); return sum(res) + sum(sp) + sum(res)
fn buildDecline(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });

    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, i64_t);
    const b = try func.appendBlockParam(entry, i64_t);

    var sp: [n_decline_split]Value = undefined;
    for (0..n_decline_split) |k| {
        const c = try func.appendInst(entry, i64_t, .{ .iconst = @intCast(k + 1) });
        sp[k] = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    }
    var res: [n_decline_res]Value = undefined;
    for (0..n_decline_res) |k| res[k] = try func.appendArithImm(entry, i64_t, .add, b, @intCast(100 + k));

    var acc = res[0];
    for (1..n_decline_res) |k| acc = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = res[k] } });
    for (0..n_decline_split) |k| acc = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = sp[k] } });
    for (0..n_decline_res) |k| acc = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = res[k] } });
    func.setTerminator(entry, .{ .ret = acc });

    return func;
}

/// The scalar reference for `buildDecline`, wrapping arithmetic to match the target's 64-bit ops.
fn declineReference(a: i64, b: i64) i64 {
    var acc: i64 = 0;
    for (0..n_decline_res) |k| acc +%= b +% @as(i64, @intCast(100 + k));
    for (0..n_decline_split) |k| acc +%= a *% @as(i64, @intCast(k + 1));
    for (0..n_decline_res) |k| acc +%= b +% @as(i64, @intCast(100 + k));
    return acc;
}

test "int-spill: second-chance declines when no register is free (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    var func = try buildDecline(allocator);
    defer func.deinit();

    // Meaningful gate: the kernel exercises BOTH paths. Re-homes fire (so the re-home code runs) AND
    // the decline path fires (at least one second-chance point had a pending split value but NO free
    // register, so it reloaded from its slot). Unlike aarch64 (which re-homes only at interval
    // starts), riscv64 runs second-chance at every position, so a starved value re-homes once a
    // register frees. What is asserted here is that the "no register free -> decline" branch is taken.
    const rehomes = try isel.debugReHomeCount(allocator, &func);
    const declines = try isel.debugDeclineCount(allocator, &func);
    try std.testing.expect(rehomes > 0);
    try std.testing.expect(declines > 0);

    // Then execute on qemu across a sweep (zero, unit, negatives, and large magnitudes): a per-use
    // slot reload (the decline path) that loaded the wrong bits diverges from the reference.
    const inputs = [_][2]i64{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ -1, -1 }, .{ 3, 5 }, .{ -2, 1 }, .{ 7, -9 }, .{ 100, 25 }, .{ -37, 41 } };
    for (inputs) |in| {
        const got = harness.runFunc(std.testing.io, allocator, &func, &.{ in[0], in[1] }, harness.qemu_user) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(declineReference(in[0], in[1]), got);
    }
}
