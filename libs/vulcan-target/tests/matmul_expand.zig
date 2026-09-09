//! Execution tests for `vulcan-ir.expand.expandMatmul`. Each test builds a function holding one
//! `matmul`, expands it into a scalar loop nest, verifies the result, JITs it for the host CPU,
//! calls it over real A, B and C buffers, and asserts every element of C against a matrix worked
//! out by hand.
//!
//! `matmul` is an et-soc tensor-tile op, and the et-soc VPU is the only backend that lowers it, so
//! before this pass a matmul could not run anywhere except under `sw-sysemu`. There was no answer
//! to check a tensor lowering against. These tests are that answer.
//!
//! Structural tests cannot do this job. A nest that swaps two indices, drops the accumulate, or
//! reads B transposed has exactly the same opcodes as a correct one, so only the numbers tell them
//! apart. Every C matrix below is derived by hand in the comment above the test.
//!
//! Shapes are deliberately non-square, with m, n and k all different, so a transposition cannot
//! pass by coincidence.
//!
//! Runs where the native JIT has a backend for the host architecture, and skips elsewhere.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const MatMul = ir.function.MatMul;
const MatMulType = ir.function.MatMulType;

/// Whether the native JIT has a backend for the host architecture.
fn hasJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .x86, .riscv64 => true,
        else => false,
    };
}

/// The signature every function these tests build compiles to: the three tile pointers, in the
/// order `matmul` names them.
const MatMulFn = *const fn (a: [*]const u8, b: [*]const u8, c: [*]u8) callconv(.c) void;

/// Build a function whose whole body is one `matmul` over its three pointer parameters.
fn matmulFunc(allocator: std.mem.Allocator, mm: MatMul) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);

    var op = mm;
    op.a = a;
    op.b = b;
    op.c = c;
    _ = try func.appendStmtRaw(entry, .{ .matmul = op });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// Expand `func` in place, check that the expansion reported a change, and verify the result in
/// the codegen profile the backends read.
fn expandAndVerify(allocator: std.mem.Allocator, func: *Function) !void {
    try std.testing.expect(try ir.expand.expandMatmul(allocator, func));
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            try std.testing.expect(func.opcode(inst) != .matmul); // no matmul survives
        }
    }
    var diags = try ir.verify.verify(allocator, func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

/// Build, expand, verify, JIT and run one matmul over the three buffers. `a`, `b` and `c` are raw
/// element slices of whatever type the dtype names.
fn runMatmul(mm: MatMul, a: []const u8, b: []const u8, c: []u8) !void {
    const allocator = std.testing.allocator;
    var func = try matmulFunc(allocator, mm);
    defer func.deinit();
    try expandAndVerify(allocator, &func);

    var code = try target.native.jitFunction(allocator, &func);
    defer code.deinit();
    code.entry(MatMulFn, 0)(a.ptr, b.ptr, c.ptr);
}

/// Reinterpret an f32 slice as the raw bytes the JITed function reads and writes.
fn bytesOf(comptime T: type, slice: []T) []u8 {
    return std.mem.sliceAsBytes(slice);
}

/// A base `matmul`, with the three pointer operands filled in by `matmulFunc`.
fn tile(m: u16, n: u16, k: u16, dtype: MatMulType, accumulate: bool) MatMul {
    return .{
        .a = @enumFromInt(0),
        .b = @enumFromInt(0),
        .c = @enumFromInt(0),
        .m = m,
        .n = n,
        .k = k,
        .dtype = dtype,
        .accumulate = accumulate,
    };
}

// A = [[1, 2, 3],        B = [[1,  2,  3,  4],
//      [4, 5, 6]]             [5,  6,  7,  8],
//                             [9, 10, 11, 12]]
//
// C[0][0] = 1*1 + 2*5  + 3*9  =  1 + 10 + 27 =  38
// C[0][1] = 1*2 + 2*6  + 3*10 =  2 + 12 + 30 =  44
// C[0][2] = 1*3 + 2*7  + 3*11 =  3 + 14 + 33 =  50
// C[0][3] = 1*4 + 2*8  + 3*12 =  4 + 16 + 36 =  56
// C[1][0] = 4*1 + 5*5  + 6*9  =  4 + 25 + 54 =  83
// C[1][1] = 4*2 + 5*6  + 6*10 =  8 + 30 + 60 =  98
// C[1][2] = 4*3 + 5*7  + 6*11 = 12 + 35 + 66 = 113
// C[1][3] = 4*4 + 5*8  + 6*12 = 16 + 40 + 72 = 128
test "expandMatmul: fp32 2x3 times 3x4 computes the hand-worked product" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    // Four spare slots that no store may reach, so an overrun shows up here too.
    var c = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

    try runMatmul(tile(2, 4, 3, .fp32, false), bytesOf(f32, &a), bytesOf(f32, &b), bytesOf(f32, &c));

    const want = [_]f32{ 38, 44, 50, 56, 83, 98, 113, 128, -1, -1, -1, -1 };
    try std.testing.expectEqualSlices(f32, &want, &c);
}

// The same A and B, over a C preloaded with known non-zero values, so an expansion that ignores
// `accumulate` writes the plain product and fails on every element:
//
//   [100, 200, 300, 400]   [ 38, 44,  50,  56]   [138, 244, 350, 456]
//   [500, 600, 700, 800] + [ 83, 98, 113, 128] = [583, 698, 813, 928]
test "expandMatmul: fp32 accumulate adds into a preloaded C" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f32{ 100, 200, 300, 400, 500, 600, 700, 800, -1, -1, -1, -1 };

    try runMatmul(tile(2, 4, 3, .fp32, true), bytesOf(f32, &a), bytesOf(f32, &b), bytesOf(f32, &c));

    const want = [_]f32{ 138, 244, 350, 456, 583, 698, 813, 928, -1, -1, -1, -1 };
    try std.testing.expectEqualSlices(f32, &want, &c);
}

// The k = 1 boundary: the reduction runs exactly one step, so an off-by-one bound is visible.
//
// A = [[2],     B = [[5, 6, 7]]
//      [3]]
//
// C = [[10, 12, 14],
//      [15, 18, 21]]
test "expandMatmul: fp32 k = 1 runs the reduction exactly once" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]f32{ 2, 3 };
    var b = [_]f32{ 5, 6, 7 };
    var c = [_]f32{ -1, -1, -1, -1, -1, -1, -1 };

    try runMatmul(tile(2, 3, 1, .fp32, false), bytesOf(f32, &a), bytesOf(f32, &b), bytesOf(f32, &c));

    const want = [_]f32{ 10, 12, 14, 15, 18, 21, -1 };
    try std.testing.expectEqualSlices(f32, &want, &c);
}

// The m = 1 boundary: one output row, so the i-loop runs once and the A row never steps.
//
// A = [[1, 2, 3]]    B = [[1, 2],
//                         [3, 4],
//                         [5, 6]]
//
// C[0][0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
// C[0][1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
test "expandMatmul: fp32 m = 1 writes one row" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]f32{ 1, 2, 3 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var c = [_]f32{ -1, -1, -1 };

    try runMatmul(tile(1, 2, 3, .fp32, false), bytesOf(f32, &a), bytesOf(f32, &b), bytesOf(f32, &c));

    const want = [_]f32{ 22, 28, -1 };
    try std.testing.expectEqualSlices(f32, &want, &c);
}

// A 3x2 times 2x4, the mirror of the first test's shape. A pass that reads m where it should read
// k, or that strides A by n, gets a different answer for one of the two shapes.
//
// A = [[1, 2],     B = [[1, 2, 3, 4],
//      [3, 4],          [5, 6, 7, 8]]
//      [5, 6]]
//
// C[0] = [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
// C[1] = [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
// C[2] = [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
test "expandMatmul: fp32 3x2 times 2x4 pins the m, n and k roles" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var c = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

    try runMatmul(tile(3, 4, 2, .fp32, false), bytesOf(f32, &a), bytesOf(f32, &b), bytesOf(f32, &c));

    const want = [_]f32{ 11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68, -1 };
    try std.testing.expectEqualSlices(f32, &want, &c);
}

// An asymmetric B, so reading B transposed answers a different question. B[0][1] is 100 and
// B[1][0] is 1, and nothing else in B is non-zero.
//
// A = [[1, 2]]    B = [[0, 100],
//                      [1,   0]]
//
// C[0][0] = 1*0 + 2*1   = 2
// C[0][1] = 1*100 + 2*0 = 100
//
// A transposed read of B answers [200, 1] instead.
test "expandMatmul: fp32 reads B row-major, not transposed" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]f32{ 1, 2 };
    var b = [_]f32{ 0, 100, 1, 0 };
    var c = [_]f32{ -1, -1 };

    try runMatmul(tile(1, 2, 2, .fp32, false), bytesOf(f32, &a), bytesOf(f32, &b), bytesOf(f32, &c));

    const want = [_]f32{ 2, 100 };
    try std.testing.expectEqualSlices(f32, &want, &c);
}

// int8 inputs with an int32 C, including negative products, so a zero-extending load is visible.
//
// A = [[1, -2, 3],       B = [[ 1,  2,   3,  4],
//      [4,  5, -6]]           [-5,  6,   7,  8],
//                             [ 9, 10, -11, 12]]
//
// C[0][0] = 1*1 + (-2)(-5) + 3*9    =  1 + 10 + 27  =  38
// C[0][1] = 1*2 + (-2)(6)  + 3*10   =  2 - 12 + 30  =  20
// C[0][2] = 1*3 + (-2)(7)  + 3*(-11)=  3 - 14 - 33  = -44
// C[0][3] = 1*4 + (-2)(8)  + 3*12   =  4 - 16 + 36  =  24
// C[1][0] = 4*1 + 5*(-5)   + (-6)*9 =  4 - 25 - 54  = -75
// C[1][1] = 4*2 + 5*6      + (-6)*10=  8 + 30 - 60  = -22
// C[1][2] = 4*3 + 5*7      + (-6)(-11) = 12+35+66   = 113
// C[1][3] = 4*4 + 5*8      + (-6)*12=  16 + 40 - 72 = -16
test "expandMatmul: int8 2x3 times 3x4 accumulates into int32" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]i8{ 1, -2, 3, 4, 5, -6 };
    var b = [_]i8{ 1, 2, 3, 4, -5, 6, 7, 8, 9, 10, -11, 12 };
    var c = [_]i32{ -1, -1, -1, -1, -1, -1, -1, -1, -1 };

    try runMatmul(tile(2, 4, 3, .int8, false), bytesOf(i8, &a), bytesOf(i8, &b), bytesOf(i32, &c));

    const want = [_]i32{ 38, 20, -44, 24, -75, -22, 113, -16, -1 };
    try std.testing.expectEqualSlices(i32, &want, &c);
}

// uint8 inputs above 127, so a sign-extending load would answer negatives.
//
// A = [[200, 100],      B = [[3, 4],
//      [ 50, 255]]           [5, 6]]
//
// C[0][0] = 200*3 + 100*5 =  600 +  500 = 1100
// C[0][1] = 200*4 + 100*6 =  800 +  600 = 1400
// C[1][0] =  50*3 + 255*5 =  150 + 1275 = 1425
// C[1][1] =  50*4 + 255*6 =  200 + 1530 = 1730
test "expandMatmul: uint8 zero-extends both operands" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]u8{ 200, 100, 50, 255 };
    var b = [_]u8{ 3, 4, 5, 6 };
    var c = [_]i32{ -1, -1, -1, -1, -1 };

    try runMatmul(tile(2, 2, 2, .uint8, false), bytesOf(u8, &a), bytesOf(u8, &b), bytesOf(i32, &c));

    const want = [_]i32{ 1100, 1400, 1425, 1730, -1 };
    try std.testing.expectEqualSlices(i32, &want, &c);
}

// Mixed signedness: uint8 activations times int8 weights, which `input_signs` spells as
// `dtype == .int8` with `a_unsigned` set.
//
// A = [[200, 100]] (u8)    B = [[-3,  4],   (i8)
//                               [ 5, -6]]
//
// C[0][0] = 200*(-3) + 100*5    = -600 + 500 = -100
// C[0][1] = 200*4    + 100*(-6) =  800 - 600 =  200
test "expandMatmul: input_signs gives each operand its own signedness" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]u8{ 200, 100 };
    var b = [_]i8{ -3, 4, 5, -6 };
    var c = [_]i32{ -1, -1, -1 };

    var mm = tile(1, 2, 2, .int8, false);
    mm.input_signs = .{ .a_unsigned = true, .b_unsigned = false };
    try runMatmul(mm, bytesOf(u8, &a), bytesOf(i8, &b), bytesOf(i32, &c));

    const want = [_]i32{ -100, 200, -1 };
    try std.testing.expectEqualSlices(i32, &want, &c);
}

// `embedded` is an et-soc lowering directive: it tells that backend to save and restore the
// registers its tensor unit clobbers. A scalar nest clobbers nothing, so the expansion handles an
// embedded matmul exactly like any other and the answer is the same as the first test's.
test "expandMatmul: embedded is expanded, not rejected" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };

    var mm = tile(2, 4, 3, .fp32, false);
    mm.embedded = true;
    try runMatmul(mm, bytesOf(f32, &a), bytesOf(f32, &b), bytesOf(f32, &c));

    const want = [_]f32{ 38, 44, 50, 56, 83, 98, 113, 128 };
    try std.testing.expectEqualSlices(f32, &want, &c);
}

/// The signature of the surrounded-matmul function: the three tile pointers, a scratch output, and
/// one integer the surrounding code multiplies out.
const SurroundedFn = *const fn (a: [*]const f32, b: [*]const f32, c: [*]f32, out: [*]i32, x: i32) callconv(.c) void;

/// The multipliers the surrounded-matmul function computes before the matmul and stores after
/// it, in a later block. Sixteen values live at once is more integer registers than the host has,
/// so a nest allowed to reuse their registers corrupts at least one.
const carried = [_]i64{ 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59 };

/// Build `entry: v[i] = x * carried[i]; matmul(a, b, c); jump tail` and `tail: out[i] = v[i]`.
///
/// The matmul sits in the MIDDLE of a block, with sixteen values defined before it and used after
/// it in a later block, so the split has to keep every one of them reaching its use across a nest
/// that did not exist when they were defined. `layoutRespectsDominance` below is what pins the
/// order those blocks end up in.
fn surroundedFunc(allocator: std.mem.Allocator, mm: MatMul) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    const entry = try func.appendBlock();
    const tail = try func.appendBlock();

    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    const out = try func.appendBlockParam(entry, ptr_t);
    const x = try func.appendBlockParam(entry, i32_t);

    var live: [carried.len]Value = undefined;
    for (carried, 0..) |factor, i| live[i] = try func.appendArithImm(entry, i32_t, .mul, x, factor);

    var op = mm;
    op.a = a;
    op.b = b;
    op.c = c;
    _ = try func.appendStmtRaw(entry, .{ .matmul = op });
    try func.setJump(entry, tail, &.{});

    for (live, 0..) |value, i| {
        const slot = try func.appendArithImm(tail, ptr_t, .add, out, @as(i64, @intCast(i)) * 4);
        try func.appendStore(tail, value, slot);
    }
    func.setTerminator(tail, .{ .ret = ir.function.Ret.none() });
    return func;
}

test "expandMatmul: values live across the matmul survive the nest" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var func = try surroundedFunc(allocator, tile(2, 4, 3, .fp32, false));
    defer func.deinit();
    try expandAndVerify(allocator, &func);

    var code = try target.native.jitFunction(allocator, &func);
    defer code.deinit();

    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    var out = [_]i32{-1} ** carried.len;
    code.entry(SurroundedFn, 0)(&a, &b, &c, &out, 2);

    const want_c = [_]f32{ 38, 44, 50, 56, 83, 98, 113, 128 };
    try std.testing.expectEqualSlices(f32, &want_c, &c);
    const want_out = [_]i32{ 6, 10, 14, 22, 26, 34, 38, 46, 58, 62, 74, 82, 86, 94, 106, 118 };
    try std.testing.expectEqualSlices(i32, &want_out, &out);
}

/// The successors of `block`: both edges of every `if` in it, plus its terminator's jump target.
fn successorsOf(func: *const Function, block: Block, out: *std.ArrayList(u32), allocator: std.mem.Allocator) !void {
    for (func.blockInsts(block)) |inst| {
        switch (func.opcode(inst)) {
            .@"if" => |cf| {
                try out.append(allocator, @intFromEnum(cf.then.target));
                try out.append(allocator, @intFromEnum(cf.@"else".target));
            },
            else => {},
        }
    }
    switch (func.terminator(block) orelse return) {
        .jump => |j| try out.append(allocator, @intFromEnum(j.target)),
        .ret => {},
    }
}

/// Check that every block follows every block that dominates it.
///
/// This is the invariant `vulcan-opt.blocklayout` states and the machine backends depend on: they
/// number linear-scan liveness by block INDEX, so a definition's block must come before every block
/// it dominates. `ir.verify` does NOT check it, and a pass that appends blocks breaks it by
/// construction, so the expansion ends with a layout step and this test is what holds that step in
/// place. The dominator sets come from the same iterative fixpoint `verify` uses internally.
fn layoutRespectsDominance(allocator: std.mem.Allocator, func: *const Function) !void {
    const n = func.blockCount();
    if (n == 0) return;

    var preds = try allocator.alloc(std.ArrayList(u32), n);
    defer {
        for (preds) |*list| list.deinit(allocator);
        allocator.free(preds);
    }
    for (preds) |*list| list.* = .empty;

    var succ: std.ArrayList(u32) = .empty;
    defer succ.deinit(allocator);
    for (0..n) |bi| {
        succ.clearRetainingCapacity();
        try successorsOf(func, @enumFromInt(bi), &succ, allocator);
        for (succ.items) |s| try preds[s].append(allocator, @intCast(bi));
    }

    // dom[b * n + a] is true while block `a` may still dominate block `b`. The entry is dominated
    // only by itself, every other block starts dominated by everything, and the sets shrink to a
    // fixpoint against the intersection over each block's predecessors.
    const dom = try allocator.alloc(bool, n * n);
    defer allocator.free(dom);
    @memset(dom, true);
    for (0..n) |a| dom[a] = (a == 0);

    var changed = true;
    while (changed) {
        changed = false;
        for (1..n) |b| {
            for (0..n) |a| {
                if (!dom[b * n + a]) continue;
                if (a == b) continue; // a block always dominates itself
                var all = preds[b].items.len != 0;
                for (preds[b].items) |p| {
                    if (!dom[@as(usize, p) * n + a]) {
                        all = false;
                        break;
                    }
                }
                if (!all) {
                    dom[b * n + a] = false;
                    changed = true;
                }
            }
        }
    }

    for (0..n) |b| {
        for (0..n) |a| {
            if (!dom[b * n + a]) continue;
            try std.testing.expect(a <= b); // a dominator must precede the block it dominates
        }
    }
}

test "expandMatmul: the expanded function is laid out for the backends" {
    const allocator = std.testing.allocator;

    // The surrounded shape is the one that needs the layout step: the block after the matmul's own
    // block is dominated by the continuation the nest falls out to, which is appended after it.
    var surrounded = try surroundedFunc(allocator, tile(2, 4, 3, .fp32, false));
    defer surrounded.deinit();
    try expandAndVerify(allocator, &surrounded);
    try layoutRespectsDominance(allocator, &surrounded);

    var plain = try matmulFunc(allocator, tile(2, 4, 3, .fp32, true));
    defer plain.deinit();
    try expandAndVerify(allocator, &plain);
    try layoutRespectsDominance(allocator, &plain);
}

/// Whether `func` still holds a `matmul`.
fn holdsMatmul(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@enumFromInt(bi))) |inst| {
            if (func.opcode(inst) == .matmul) return true;
        }
    }
    return false;
}

// fp16 A and B, widened to an f32 accumulator, over the same numbers as the first test. Every
// value here is a small integer, which f16 holds exactly, so the answer is the fp32 answer and a
// lost widening or a 4-byte A stride shows up as a wrong element rather than as rounding.
test "expandMatmul: fp16 inputs widen to an f32 C" {
    if (comptime !hasJit()) return error.SkipZigTest;

    var a = [_]f16{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f16{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1, -1 };

    try runMatmul(tile(2, 4, 3, .fp16, false), bytesOf(f16, &a), bytesOf(f16, &b), bytesOf(f32, &c));

    const want = [_]f32{ 38, 44, 50, 56, 83, 98, 113, 128, -1 };
    try std.testing.expectEqualSlices(f32, &want, &c);
}

test "expandMatmul: a quant epilogue is left in place and named" {
    const allocator = std.testing.allocator;
    var mm = tile(2, 4, 3, .int8, false);
    // A scalar requantize scale of 1.0, which is enough to make the epilogue present. The
    // expansion refuses on presence, not on the scale's value.
    mm.quant = .{ .scale = .{ .scalar = @bitCast(@as(f32, 1.0)) }, .relu = false };

    var func = try matmulFunc(allocator, mm);
    defer func.deinit();

    try std.testing.expect(!try ir.expand.expandMatmul(allocator, &func));
    try std.testing.expect(holdsMatmul(&func));
    try std.testing.expectEqual(@as(?ir.expand.Unsupported, .quant), ir.expand.matmulUnsupported(mm));
}

test "expandMatmul: a supported matmul is named as supported" {
    try std.testing.expectEqual(@as(?ir.expand.Unsupported, null), ir.expand.matmulUnsupported(tile(2, 4, 3, .fp32, true)));
    try std.testing.expectEqual(@as(?ir.expand.Unsupported, null), ir.expand.matmulUnsupported(tile(2, 4, 3, .fp16, false)));
    try std.testing.expectEqual(@as(?ir.expand.Unsupported, null), ir.expand.matmulUnsupported(tile(2, 4, 3, .int8, false)));
    try std.testing.expectEqual(@as(?ir.expand.Unsupported, null), ir.expand.matmulUnsupported(tile(2, 4, 3, .uint8, false)));
}

test "expandMatmul: a function carrying a block attribute is left alone" {
    const allocator = std.testing.allocator;
    var func = try matmulFunc(allocator, tile(2, 4, 3, .fp32, false));
    defer func.deinit();
    // `reorderBlocks` does not remap a block id held in an attribute payload, and the layout step
    // uses it, so such a function keeps its matmul rather than getting a stale reference.
    try func.addAttr(.{ .block = @enumFromInt(0) }, .cold);

    try std.testing.expect(!try ir.expand.expandMatmul(allocator, &func));
    try std.testing.expect(holdsMatmul(&func));
}

test "expandMatmul: two matmuls in one function both expand" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    // The same tile twice, the second accumulating into the first's answer, so C ends up doubled.
    var first = tile(2, 4, 3, .fp32, false);
    first.a = a;
    first.b = b;
    first.c = c;
    var second = first;
    second.accumulate = true;
    _ = try func.appendStmtRaw(entry, .{ .matmul = first });
    _ = try func.appendStmtRaw(entry, .{ .matmul = second });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    try expandAndVerify(allocator, &func);

    var code = try target.native.jitFunction(allocator, &func);
    defer code.deinit();

    var av = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var bv = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var cv = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    code.entry(MatMulFn, 0)(@ptrCast(&av), @ptrCast(&bv), @ptrCast(&cv));

    const want = [_]f32{ 76, 88, 100, 112, 166, 196, 226, 256 };
    try std.testing.expectEqualSlices(f32, &want, &cv);
}
