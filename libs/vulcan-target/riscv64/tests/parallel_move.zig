//! Jump-edge block-argument parallel-move regression tests, executed on qemu-riscv64 (the oracle).
//!
//! A loop back-edge that carries its loop-carried values in a *permuted* order (a swap of two, or a
//! 3-way rotation) turns the edge's (arg_reg -> param_reg) moves into a register permutation cycle:
//! the loop header's params sit in fixed registers, and the (critical-edge-split) back-edge forwards
//! those very registers back into the header permuted. The naive in-order emit that lowered these
//! edges before clobbered a value mid-cycle (`mv x7,x28` then `mv x28,x7` collapses a swap so both
//! registers end up holding the second value), so the loop computed the wrong result. These build
//! such loops in IR, run them under qemu, and assert the architecturally-correct result - which only
//! holds if the edge moves are done in parallel (cycle-broken through a scratch), exactly what
//! `parallelMove{Int,Float,Vector}` now do.

const std = @import("std");
const ir = @import("vulcan-ir");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };

/// A self-loop whose back-edge swaps two loop-carried i32 values each iteration and decrements a
/// counter, returning the first carried value on exit. The header's two carried params sit in fixed
/// registers; the back-edge passes them swapped, so lowering must emit a 2-cycle register swap.
/// After `n` iterations the first value is the original first value when `n` is even and the original
/// second value when `n` is odd - so a correct result depends on that swap being a genuine parallel
/// (cycle-broken) move.
///
/// entry(n, a, b) -> header(n, a, b)
/// header(i, va, vb): inext = i - 1; if i > 0 then header(inext, vb, va) else done(va)   // the swap
/// done(ra): ret ra
fn buildSwapLoop(allocator: std.mem.Allocator) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const t = try f.types.intern(i32k);
    const bool_t = try f.types.intern(.bool);

    const entry = try f.appendBlock();
    const header = try f.appendBlock();
    const done = try f.appendBlock();

    const n = try f.appendBlockParam(entry, t);
    const ea = try f.appendBlockParam(entry, t);
    const eb = try f.appendBlockParam(entry, t);
    const i = try f.appendBlockParam(header, t);
    const va = try f.appendBlockParam(header, t);
    const vb = try f.appendBlockParam(header, t);
    const ra = try f.appendBlockParam(done, t);

    try f.setJump(entry, header, &.{ n, ea, eb });
    const zero = try f.appendInst(header, t, .{ .iconst = 0 });
    const inext = try f.appendArithImm(header, t, .add, i, -1);
    const cmp = try f.appendInst(header, bool_t, .{ .icmp = .{ .op = .gt, .lhs = i, .rhs = zero } });
    try f.appendIf(header, cmp, .{ .target = header, .args = &.{ inext, vb, va } }, .{ .target = done, .args = &.{va} });
    f.setTerminator(done, .{ .ret = ra });
    return f;
}

/// A self-loop whose back-edge rotates three loop-carried i32 values (a,b,c) -> (c,a,b) each
/// iteration, returning the first on exit - a 3-cycle, not just a 2-cycle. After `n` iterations the
/// first value is c/b/a for n mod 3 == 1/2/0 respectively (one rotation sends c to the front).
fn buildRotateLoop(allocator: std.mem.Allocator) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const t = try f.types.intern(i32k);
    const bool_t = try f.types.intern(.bool);

    const entry = try f.appendBlock();
    const header = try f.appendBlock();
    const done = try f.appendBlock();

    const n = try f.appendBlockParam(entry, t);
    const ea = try f.appendBlockParam(entry, t);
    const eb = try f.appendBlockParam(entry, t);
    const ec = try f.appendBlockParam(entry, t);
    const i = try f.appendBlockParam(header, t);
    const va = try f.appendBlockParam(header, t);
    const vb = try f.appendBlockParam(header, t);
    const vc = try f.appendBlockParam(header, t);
    const ra = try f.appendBlockParam(done, t);

    try f.setJump(entry, header, &.{ n, ea, eb, ec });
    const zero = try f.appendInst(header, t, .{ .iconst = 0 });
    const inext = try f.appendArithImm(header, t, .add, i, -1);
    const cmp = try f.appendInst(header, bool_t, .{ .icmp = .{ .op = .gt, .lhs = i, .rhs = zero } });
    try f.appendIf(header, cmp, .{ .target = header, .args = &.{ inext, vc, va, vb } }, .{ .target = done, .args = &.{va} });
    f.setTerminator(done, .{ .ret = ra });
    return f;
}

/// The f32 analogue of `buildSwapLoop`. Uses float loop-carried values so the back-edge moves are
/// `fmv.s` (float register file) rather than `mv` - a cycle that `parallelMoveFloat` must break. The
/// counter and its `1`/`0` operands are passed as floats too (fa0..fa4) to avoid needing float
/// constants, and are carried unchanged around the loop.
///
/// entry(cnt, a, b, one, zero) -> header(cnt, a, b, one, zero)
/// header(c, va, vb, one, zero): c2 = c - one; if c > zero then header(c2, vb, va, one, zero) else done(va)
/// done(ra): ret ra
fn buildFloatSwapLoop(allocator: std.mem.Allocator) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const ft = try f.types.intern(.{ .float = .f32 });
    const bool_t = try f.types.intern(.bool);

    const entry = try f.appendBlock();
    const header = try f.appendBlock();
    const done = try f.appendBlock();

    const cnt = try f.appendBlockParam(entry, ft);
    const ea = try f.appendBlockParam(entry, ft);
    const eb = try f.appendBlockParam(entry, ft);
    const eone = try f.appendBlockParam(entry, ft);
    const ezero = try f.appendBlockParam(entry, ft);
    const c = try f.appendBlockParam(header, ft);
    const va = try f.appendBlockParam(header, ft);
    const vb = try f.appendBlockParam(header, ft);
    const one = try f.appendBlockParam(header, ft);
    const zero = try f.appendBlockParam(header, ft);
    const ra = try f.appendBlockParam(done, ft);

    try f.setJump(entry, header, &.{ cnt, ea, eb, eone, ezero });
    const c2 = try f.appendInst(header, ft, .{ .arith = .{ .op = .sub, .lhs = c, .rhs = one } });
    const cmp = try f.appendInst(header, bool_t, .{ .icmp = .{ .op = .gt, .lhs = c, .rhs = zero } });
    try f.appendIf(header, cmp, .{ .target = header, .args = &.{ c2, vb, va, one, zero } }, .{ .target = done, .args = &.{va} });
    f.setTerminator(done, .{ .ret = ra });
    return f;
}

/// A <4 x f32> analogue of `buildSwapLoop`: two vectors are built (via `struct_new` over `fconst`
/// lanes) in the entry block and carried around a self-loop whose back-edge swaps them, so the edge
/// moves are `vmv.v.v` on the RVV register file - a cycle that `parallelMoveVector` must break
/// through `vector_scratch` (v31). Lane 0 of the surviving vector is returned as a scalar f32 (the
/// user-mode float ABI returns fa0, not a whole vector). The counter is baked in as an `iconst`, so
/// the function takes no arguments. After `n` iterations the survivor is the first vector when `n` is
/// even and the second when odd.
fn buildVectorSwapLoop(allocator: std.mem.Allocator, n: i64, a: [4]f32, b: [4]f32) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const t = try f.types.intern(i32k);
    const ft = try f.types.intern(.{ .float = .f32 });
    const v4 = try f.types.intern(.{ .vector = .{ .len = 4, .elem = ft } });
    const bool_t = try f.types.intern(.bool);

    const entry = try f.appendBlock();
    const header = try f.appendBlock();
    const done = try f.appendBlock();

    const i = try f.appendBlockParam(header, t);
    const ha = try f.appendBlockParam(header, v4);
    const hb = try f.appendBlockParam(header, v4);
    const rv = try f.appendBlockParam(done, v4);

    var al: [4]Value = undefined;
    var bl: [4]Value = undefined;
    for (0..4) |k| al[k] = try f.appendInst(entry, ft, .{ .fconst = a[k] });
    for (0..4) |k| bl[k] = try f.appendInst(entry, ft, .{ .fconst = b[k] });
    const va = try f.appendInst(entry, v4, .{ .struct_new = .{ .fields = try f.internValueList(&al) } });
    const vb = try f.appendInst(entry, v4, .{ .struct_new = .{ .fields = try f.internValueList(&bl) } });
    const cnt = try f.appendInst(entry, t, .{ .iconst = n });
    try f.setJump(entry, header, &.{ cnt, va, vb });

    const zero = try f.appendInst(header, t, .{ .iconst = 0 });
    const inext = try f.appendArithImm(header, t, .add, i, -1);
    const cmp = try f.appendInst(header, bool_t, .{ .icmp = .{ .op = .gt, .lhs = i, .rhs = zero } });
    try f.appendIf(header, cmp, .{ .target = header, .args = &.{ inext, hb, ha } }, .{ .target = done, .args = &.{ha} });

    const lane = try f.appendInst(done, ft, .{ .extract = .{ .aggregate = rv, .index = 0 } });
    f.setTerminator(done, .{ .ret = lane });
    return f;
}

test "parallel-move: <4 x f32> back-edge vector swap (2-cycle) computes the correct value (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bits = struct {
        fn f(x: f32) u64 {
            return @as(u32, @bitCast(x));
        }
    }.f;
    const a = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [4]f32{ 5.0, 6.0, 7.0, 8.0 };
    // n=2 (even) -> first vector survives, lane 0 = 1.0.
    var f2 = try buildVectorSwapLoop(allocator, 2, a, b);
    defer f2.deinit();
    try std.testing.expectEqual(bits(1.0), try harness.runFuncFloat(io, allocator, &f2, false, &.{}, harness.qemu_user));
    // n=3 (odd) -> second vector survives, lane 0 = 5.0.
    var f3 = try buildVectorSwapLoop(allocator, 3, a, b);
    defer f3.deinit();
    try std.testing.expectEqual(bits(5.0), try harness.runFuncFloat(io, allocator, &f3, false, &.{}, harness.qemu_user));
}

test "parallel-move: i32 back-edge swap (2-cycle) computes the correct value (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // n even -> first value survives (a); n odd -> swapped (b). The even cases are the ones the naive
    // sequential emit got wrong (collapsing both carried values to b).
    var f2 = try buildSwapLoop(allocator);
    defer f2.deinit();
    try harness.expectRun(io, allocator, &f2, &.{ 2, 111, 222 }, 111, harness.qemu_user); // even -> a
    var f4 = try buildSwapLoop(allocator);
    defer f4.deinit();
    try harness.expectRun(io, allocator, &f4, &.{ 4, 111, 222 }, 111, harness.qemu_user); // even -> a
    var f3 = try buildSwapLoop(allocator);
    defer f3.deinit();
    try harness.expectRun(io, allocator, &f3, &.{ 3, 111, 222 }, 222, harness.qemu_user); // odd -> b
}

test "parallel-move: i32 back-edge 3-way rotation (3-cycle) computes the correct value (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // One rotation (a,b,c)->(c,a,b) sends c to the front: n=1 -> c, n=2 -> b, n=3 -> a.
    var f1 = try buildRotateLoop(allocator);
    defer f1.deinit();
    try harness.expectRun(io, allocator, &f1, &.{ 1, 10, 20, 30 }, 30, harness.qemu_user);
    var f2 = try buildRotateLoop(allocator);
    defer f2.deinit();
    try harness.expectRun(io, allocator, &f2, &.{ 2, 10, 20, 30 }, 20, harness.qemu_user);
    var f3 = try buildRotateLoop(allocator);
    defer f3.deinit();
    try harness.expectRun(io, allocator, &f3, &.{ 3, 10, 20, 30 }, 10, harness.qemu_user);
}

test "parallel-move: f32 back-edge swap (2-cycle) computes the correct value (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bits = struct {
        fn f(x: f32) u64 {
            return @as(u32, @bitCast(x));
        }
    }.f;
    // cnt=2 (even) -> first value (a) survives. a=1.5, b=2.5, one=1.0, zero=0.0.
    var f2 = try buildFloatSwapLoop(allocator);
    defer f2.deinit();
    const args2: [5]u64 = .{ bits(2.0), bits(1.5), bits(2.5), bits(1.0), bits(0.0) };
    try std.testing.expectEqual(bits(1.5), try harness.runFuncFloat(io, allocator, &f2, false, &args2, harness.qemu_user));
    // cnt=3 (odd) -> second value (b) ends up first.
    var f3 = try buildFloatSwapLoop(allocator);
    defer f3.deinit();
    const args3: [5]u64 = .{ bits(3.0), bits(1.5), bits(2.5), bits(1.0), bits(0.0) };
    try std.testing.expectEqual(bits(2.5), try harness.runFuncFloat(io, allocator, &f3, false, &args3, harness.qemu_user));
}
