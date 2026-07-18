//! Differential JIT oracle for the microarch loop unroller. For each loop shape,
//! we build two identical functions, unroll one under a wide out-of-order model
//! (ampere-altra), JIT both on the host, and require bit-identical results for a
//! spread of inputs. The unrolled function must compute exactly what the original
//! does; any divergence is a miscompile in the transform, not a test to relax.
//!
//! This runs only where the native JIT has a backend (aarch64/x86_64/riscv64/x86).

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;

/// The wide, out-of-order model the unroller actually transforms for.
fn ampere() *const opt.microarch.Model {
    return opt.microarch.modelFor(.@"ampere-altra");
}

/// A builder writes one loop shape into a fresh function: a single i32 parameter
/// `n`, an i32 result.
const Builder = *const fn (*Function) anyerror!void;

/// Shape 1: `for (i = 0; i < n; i += 1) {}  return i` (returns n for n >= 0).
/// The induction variable escapes the loop and is read directly at the exit,
/// which is exactly the stale-value hazard loop-closed SSA has to fix.
fn buildCounted(func: *Function) anyerror!void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const next = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{next});
    func.setTerminator(done, .{ .ret = i });
}

/// Shape 2: `s = 0; for (i = 0; i < n; i += 1) s += i;  return s`. Two carried
/// values (i and s); the accumulator escapes and is read directly at the exit.
fn buildSum(func: *Function) anyerror!void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const s = try func.appendBlockParam(loop, i32_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const bs = try func.appendBlockParam(body, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done });
    const ns = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = bi } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    func.setTerminator(done, .{ .ret = s });
}

/// Shape 3: a Fibonacci-style recurrence with two carried values that both
/// update (a' = b, b' = a + b), one of which (a) escapes and is returned.
fn buildFib(func: *Function) anyerror!void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const av = try func.appendBlockParam(loop, i32_t);
    const bv = try func.appendBlockParam(loop, i32_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const ba = try func.appendBlockParam(body, i32_t);
    const bb = try func.appendBlockParam(body, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const one = try func.appendInst(entry, i32_t, .{ .iconst = 1 });
    try func.setJump(entry, loop, &.{ zero, zero, one });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, av, bv } }, .{ .target = done });
    const nb = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = ba, .rhs = bb } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, bb, nb });
    func.setTerminator(done, .{ .ret = av });
}

/// Shape 4: a side-effecting body. `slot` is a stack cell (alloca) holding the
/// running total; the body loads it, adds `i`, and stores it back, so the
/// carried state lives in memory rather than in an SSA-carried block param.
/// This is the eligibility-accepted "side-effecting body" shape: the header
/// stays pure (load/store only appear in the body), so the loop is still
/// eligible, and unrolling must preserve the load/store count and order.
fn buildMemAccum(func: *Function) anyerror!void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const slot = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i32_t } });
    try func.appendStore(entry, zero, slot);
    try func.setJump(entry, loop, &.{zero});
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const acc = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = slot } });
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = bi } });
    try func.appendStore(body, nacc, slot);
    const next = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{next});
    const final = try func.appendInst(done, i32_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(done, .{ .ret = final });
}

/// Shape 5: a multi-block body with an internal diamond. The body entry tests
/// `bi < bacc` (a value only known at runtime, so both arms run across the
/// input spread) and branches to a then-block and an else-block that add
/// different amounts before both jumping to a merge block, which is the
/// latch closing the back-edge. This proves multi-block bodies and an
/// internal `@"if"` clone and unroll correctly, not just straight-line ones.
fn buildDiamond(func: *Function) anyerror!void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body_entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const merge = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const acc = try func.appendBlockParam(loop, i32_t);
    const bi = try func.appendBlockParam(body_entry, i32_t);
    const bacc = try func.appendBlockParam(body_entry, i32_t);
    const ti = try func.appendBlockParam(then_b, i32_t);
    const tacc = try func.appendBlockParam(then_b, i32_t);
    const ei = try func.appendBlockParam(else_b, i32_t);
    const eacc = try func.appendBlockParam(else_b, i32_t);
    const mi = try func.appendBlockParam(merge, i32_t);
    const macc = try func.appendBlockParam(merge, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body_entry, .args = &.{ i, acc } }, .{ .target = done });
    const next_i = try func.appendArithImm(body_entry, i32_t, .add, bi, 1);
    const split = try func.appendInst(body_entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = bi, .rhs = bacc } });
    try func.appendIf(
        body_entry,
        split,
        .{ .target = then_b, .args = &.{ next_i, bacc } },
        .{ .target = else_b, .args = &.{ next_i, bacc } },
    );
    try func.setJump(then_b, merge, &.{ ti, tacc }); // then-arm: pass the accumulator through unchanged
    const else_acc = try func.appendArithImm(else_b, i32_t, .add, eacc, 3);
    try func.setJump(else_b, merge, &.{ ei, else_acc }); // else-arm: add a different amount
    try func.setJump(merge, loop, &.{ mi, macc });
    func.setTerminator(done, .{ .ret = acc });
}

/// Build two copies of `build`, unroll one under ampere-altra, JIT both, and
/// require identical results across a spread of inputs.
fn expectUnrollMatches(build: Builder) !void {
    const allocator = std.testing.allocator;
    const inputs = [_]i64{ 0, 1, 2, 3, 5, 8, 16 };

    var orig = Function.init(allocator);
    defer orig.deinit();
    try build(&orig);

    var unrolled = Function.init(allocator);
    defer unrolled.deinit();
    try build(&unrolled);

    const changed = try opt.microarch.unroll.run(allocator, &unrolled, ampere());
    try std.testing.expect(changed); // every shape here is eligible

    // The transform must preserve well-formedness for codegen.
    var diags = try ir.verify.verify(allocator, &unrolled, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_o = try target.native.jitFunction(allocator, &orig);
    defer buf_o.deinit();
    var buf_u = try target.native.jitFunction(allocator, &unrolled);
    defer buf_u.deinit();

    const Fn = *const fn (i64) callconv(.c) i64;
    const f_o = buf_o.entry(Fn, 0);
    const f_u = buf_u.entry(Fn, 0);

    for (inputs) |n| {
        try std.testing.expectEqual(f_o(n), f_u(n));
    }
}

test "unroll differential: counted loop (escaping induction variable)" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectUnrollMatches(buildCounted);
}

test "unroll differential: sum reduction (two carried values, accumulator escapes)" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectUnrollMatches(buildSum);
}

test "unroll differential: fibonacci recurrence (both carried values update)" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectUnrollMatches(buildFib);
}

test "unroll differential: side-effecting body (accumulator carried through memory)" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectUnrollMatches(buildMemAccum);
}

test "unroll differential: multi-block body with an internal diamond" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectUnrollMatches(buildDiamond);
}

/// Whether the native JIT has a backend for the host architecture.
fn hasJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .x86, .riscv64 => true,
        else => false,
    };
}
