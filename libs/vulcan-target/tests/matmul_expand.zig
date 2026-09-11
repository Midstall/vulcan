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

// ---------------------------------------------------------------------------------------------
// Induction-variable strength reduction (`vulcan-opt.ivsr`) over the same nests.
//
// The pass rewrites `k*n` and every address the loop rebuilds from `k` into loop-carried variables
// the back edge advances by a constant. That is a rewrite of the loop body, so a mistake in it
// computes the wrong matrix while keeping exactly the same opcodes. Only running it tells them
// apart, which is the same reason the tests above exist.
//
// Two shapes are covered. The first is `expandMatmul`'s own nest, put through the whole optimizer
// and checked against the hand-worked matrices above. The second is the NAIVE index-arithmetic
// nest, `c[i*n + j] += a[i*k + p] * b[p*n + j]`, which is what a frontend writes and what the
// pass was built for: it recomputes `p*n` and both addresses on every trip.

const opt = @import("vulcan-opt");

/// A measurement of the innermost natural loop of `func`: how many instructions it holds, and
/// whether any of them is an integer multiply or shift.
///
/// Total instructions is the wrong measure for this pass. It MOVES work out of the loop into the
/// preheader, so a nest can come out the same size overall and still run far less code: the inner
/// body of a matmul runs `m*n*k` times where its preheader runs `m*n` times. The innermost loop is
/// what decides the running time, and lowering its instruction count is what this pass is for.
///
/// `scaling` counts only INTEGER multiplies and shifts, which is what the index arithmetic this
/// pass removes is made of. The floating-point multiply of the product itself is not one, so it
/// cannot hide a failure.
const LoopSize = struct { instructions: usize, scaling: bool };

fn innermostLoop(allocator: std.mem.Allocator, func: *const Function) !LoopSize {
    var info = try opt.loops.analyze(allocator, func);
    defer info.deinit(allocator);

    var best_blocks: ?usize = null;
    var best: LoopSize = .{ .instructions = 0, .scaling = false };
    for (info.loops) |loop| {
        var blocks: usize = 0;
        var size: LoopSize = .{ .instructions = 0, .scaling = false };
        for (0..func.blockCount()) |bi| {
            if (!loop.contains(bi)) continue;
            blocks += 1;
            const insts = func.blockInsts(@enumFromInt(bi));
            size.instructions += insts.len;
            for (insts) |inst| {
                const result = func.instResult(inst) orelse continue;
                if (func.types.type_kind(func.valueType(result)) != .int) continue;
                if (isScaling(func.opcode(inst))) size.scaling = true;
            }
        }
        // Fewest blocks wins: in a nest, the innermost loop is the one no other loop sits inside.
        if (best_blocks == null or blocks < best_blocks.?) {
            best_blocks = blocks;
            best = size;
        }
    }
    return best;
}

/// Exhaustive with no `else` prong: a new opcode that scales a value and is not listed here would
/// read as "no scaling left" and let this test pass on a loop the pass never touched.
fn isScaling(opcode: ir.function.Opcode) bool {
    return switch (opcode) {
        .arith => |a| a.op == .mul or a.op == .shl,
        .arith_imm => |a| a.op == .mul or a.op == .shl,
        .iconst,
        .fconst,
        .fconst128,
        .icmp,
        .select,
        .struct_new,
        .extract,
        .convert,
        .unary,
        .alloca,
        .call,
        .call_indirect,
        .global_addr,
        .load,
        .store,
        .prefetch,
        .va_start,
        .va_arg,
        .va_end,
        .dot,
        .matmul,
        .barrier,
        .atomic_rmw,
        .@"if",
        => false,
    };
}

test "ivsr: the expanded matmul nest still computes the hand-worked product after the optimizer" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var func = try matmulFunc(allocator, tile(2, 4, 3, .fp32, false));
    defer func.deinit();
    try expandAndVerify(allocator, &func);
    _ = try opt.optimize(allocator, &func);

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var code = try target.native.jitFunction(allocator, &func);
    defer code.deinit();

    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f32{ -1, -1, -1, -1, -1, -1, -1, -1 };
    code.entry(MatMulFn, 0)(bytesOf(f32, &a).ptr, bytesOf(f32, &b).ptr, bytesOf(f32, &c).ptr);

    const want = [_]f32{ 38, 44, 50, 56, 83, 98, 113, 128 };
    try std.testing.expectEqualSlices(f32, &want, &c);
}

/// Build the NAIVE matmul nest, the one a frontend writes:
///
/// ```
/// for (i in 0..m) for (j in 0..n) { acc = 0; for (p in 0..k) acc += a[i*k+p] * b[p*n+j];
///                                   c[i*n+j] = acc; }
/// ```
///
/// Every address is rebuilt from the loop indices, which is exactly the shape `ivsr` reduces. The
/// row offset `i*k` sits in the inner loop's PREHEADER, where loop-invariant code motion would put
/// it, so the inner loop is left with the two addresses and `p*n`.
///
/// The block order is the same dominance-respecting one `expandMatmul.layOutNest` produces:
/// preheader, i-header, j-header, j-body, p-header, p-body, j-latch, i-latch, continuation.
fn naiveNest(allocator: std.mem.Allocator, m: i64, n: i64, k: i64) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);

    const entry = try func.appendBlock();
    const i_head = try func.appendBlock();
    const j_head = try func.appendBlock();
    const j_body = try func.appendBlock();
    const p_head = try func.appendBlock();
    const p_body = try func.appendBlock();
    const j_latch = try func.appendBlock();
    const i_latch = try func.appendBlock();
    const cont = try func.appendBlock();

    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const fzero = try func.appendInst(entry, f32_t, .{ .fconst = 0.0 });
    const m_bound = try func.appendInst(entry, i32_t, .{ .iconst = m });
    const n_bound = try func.appendInst(entry, i32_t, .{ .iconst = n });
    const k_bound = try func.appendInst(entry, i32_t, .{ .iconst = k });
    try func.setJump(entry, i_head, &.{zero});

    const i = try func.appendBlockParam(i_head, i32_t);
    const i_lt = try func.appendInst(i_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = m_bound } });
    try func.appendIf(i_head, i_lt, .{ .target = j_head, .args = &.{ i, zero } }, .{ .target = cont });

    const ji = try func.appendBlockParam(j_head, i32_t);
    const j = try func.appendBlockParam(j_head, i32_t);
    const j_lt = try func.appendInst(j_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = j, .rhs = n_bound } });
    try func.appendIf(j_head, j_lt, .{ .target = j_body, .args = &.{ ji, j } }, .{ .target = i_latch, .args = &.{ji} });

    // The inner loop's preheader: the A row offset, which does not change with p.
    const bi = try func.appendBlockParam(j_body, i32_t);
    const bj = try func.appendBlockParam(j_body, i32_t);
    const a_row = try func.appendInst(j_body, i32_t, .{ .arith = .{ .op = .mul, .lhs = bi, .rhs = k_bound } });
    try func.setJump(j_body, p_head, &.{ bi, bj, zero, fzero });

    const pi = try func.appendBlockParam(p_head, i32_t);
    const pj = try func.appendBlockParam(p_head, i32_t);
    const p = try func.appendBlockParam(p_head, i32_t);
    const acc = try func.appendBlockParam(p_head, f32_t);
    const p_lt = try func.appendInst(p_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = p, .rhs = k_bound } });
    try func.appendIf(
        p_head,
        p_lt,
        .{ .target = p_body, .args = &.{ pi, pj, p, acc } },
        .{ .target = j_latch, .args = &.{ pi, pj, acc } },
    );

    const qi = try func.appendBlockParam(p_body, i32_t);
    const qj = try func.appendBlockParam(p_body, i32_t);
    const q = try func.appendBlockParam(p_body, i32_t);
    const qacc = try func.appendBlockParam(p_body, f32_t);
    // a[i*k + p]
    const a_index = try func.appendInst(p_body, i32_t, .{ .arith = .{ .op = .add, .lhs = a_row, .rhs = q } });
    const a_off = try func.appendArithImm(p_body, i32_t, .mul, a_index, 4);
    const a_ptr = try func.appendInst(p_body, ptr_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = a_off } });
    const a_val = try func.appendInst(p_body, f32_t, .{ .load = .{ .ptr = a_ptr } });
    // b[p*n + j]. `p*n` is the product the baseline measurement caught vulcan recomputing.
    const b_row = try func.appendInst(p_body, i32_t, .{ .arith = .{ .op = .mul, .lhs = q, .rhs = n_bound } });
    const b_index = try func.appendInst(p_body, i32_t, .{ .arith = .{ .op = .add, .lhs = b_row, .rhs = qj } });
    const b_off = try func.appendArithImm(p_body, i32_t, .mul, b_index, 4);
    const b_ptr = try func.appendInst(p_body, ptr_t, .{ .arith = .{ .op = .add, .lhs = b, .rhs = b_off } });
    const b_val = try func.appendInst(p_body, f32_t, .{ .load = .{ .ptr = b_ptr } });
    const product = try func.appendInst(p_body, f32_t, .{ .arith = .{ .op = .mul, .lhs = a_val, .rhs = b_val } });
    const next_acc = try func.appendInst(p_body, f32_t, .{ .arith = .{ .op = .add, .lhs = qacc, .rhs = product } });
    const next_p = try func.appendArithImm(p_body, i32_t, .add, q, 1);
    try func.setJump(p_body, p_head, &.{ qi, qj, next_p, next_acc });

    // c[i*n + j] = acc
    const li = try func.appendBlockParam(j_latch, i32_t);
    const lj = try func.appendBlockParam(j_latch, i32_t);
    const lacc = try func.appendBlockParam(j_latch, f32_t);
    const c_row = try func.appendInst(j_latch, i32_t, .{ .arith = .{ .op = .mul, .lhs = li, .rhs = n_bound } });
    const c_index = try func.appendInst(j_latch, i32_t, .{ .arith = .{ .op = .add, .lhs = c_row, .rhs = lj } });
    const c_off = try func.appendArithImm(j_latch, i32_t, .mul, c_index, 4);
    const c_ptr = try func.appendInst(j_latch, ptr_t, .{ .arith = .{ .op = .add, .lhs = c, .rhs = c_off } });
    try func.appendStore(j_latch, lacc, c_ptr);
    const next_j = try func.appendArithImm(j_latch, i32_t, .add, lj, 1);
    try func.setJump(j_latch, j_head, &.{ li, next_j });

    const ii = try func.appendBlockParam(i_latch, i32_t);
    const next_i = try func.appendArithImm(i_latch, i32_t, .add, ii, 1);
    try func.setJump(i_latch, i_head, &.{next_i});

    func.setTerminator(cont, .{ .ret = ir.function.Ret.none() });
    return func;
}

/// The row-major product, worked out in the test rather than by the compiler under test.
fn referenceProduct(allocator: std.mem.Allocator, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) ![]f32 {
    const out = try allocator.alloc(f32, m * n);
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |p| sum += a[i * k + p] * b[p * n + j];
            out[i * n + j] = sum;
        }
    }
    return out;
}

/// Build the naive nest at one shape, run it unoptimized and optimized, and require both to match
/// a product worked out here. The unoptimized run is the control: it proves the nest itself is
/// right, so a mismatch can only come from the optimizer.
fn expectNaiveNest(m: usize, n: usize, k: usize) !void {
    const allocator = std.testing.allocator;

    const a = try allocator.alloc(f32, m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, k * n);
    defer allocator.free(b);
    // Values that make a swapped index or a dropped term show up: no two entries are equal, and
    // the fractions keep every product distinct without leaving the exact f32 range.
    for (a, 0..) |*v, x| v.* = @as(f32, @floatFromInt(x)) * 0.5 - 3.0;
    for (b, 0..) |*v, x| v.* = @as(f32, @floatFromInt(x)) * 0.25 + 1.0;

    const want = try referenceProduct(allocator, a, b, m, n, k);
    defer allocator.free(want);

    const shape = .{ @as(i64, @intCast(m)), @as(i64, @intCast(n)), @as(i64, @intCast(k)) };

    const plain_c = try allocator.alloc(f32, m * n);
    defer allocator.free(plain_c);
    @memset(plain_c, -1);
    {
        var func = try naiveNest(allocator, shape[0], shape[1], shape[2]);
        defer func.deinit();
        var code = try target.native.jitFunction(allocator, &func);
        defer code.deinit();
        code.entry(MatMulFn, 0)(bytesOf(f32, a).ptr, bytesOf(f32, b).ptr, bytesOf(f32, plain_c).ptr);
    }
    try std.testing.expectEqualSlices(f32, want, plain_c);

    const tuned_c = try allocator.alloc(f32, m * n);
    defer allocator.free(tuned_c);
    @memset(tuned_c, -1);
    {
        var func = try naiveNest(allocator, shape[0], shape[1], shape[2]);
        defer func.deinit();
        _ = try opt.optimize(allocator, &func);
        var diags = try ir.verify.verify(allocator, &func, .low);
        defer diags.deinit();
        try std.testing.expect(diags.ok());
        var code = try target.native.jitFunction(allocator, &func);
        defer code.deinit();
        code.entry(MatMulFn, 0)(bytesOf(f32, a).ptr, bytesOf(f32, b).ptr, bytesOf(f32, tuned_c).ptr);
    }
    try std.testing.expectEqualSlices(f32, want, tuned_c);
}

test "ivsr: the naive matmul nest computes the same matrix optimized as unoptimized" {
    if (comptime !hasJit()) return error.SkipZigTest;
    // Non-square, all three extents different, so a transposition cannot pass by coincidence.
    try expectNaiveNest(2, 4, 3);
    try expectNaiveNest(3, 5, 7);
    try expectNaiveNest(5, 2, 6);
}

test "ivsr: the naive matmul nest is right at trip counts of zero and one" {
    if (comptime !hasJit()) return error.SkipZigTest;
    // k == 0 leaves the reduction loop with no trip at all, so every derived variable is read
    // exactly zero times and C must be the accumulator's initial value.
    try expectNaiveNest(2, 3, 0);
    // k == 1 runs the body once and takes the back edge once, which is where an initial value
    // computed as if the loop had already stepped would show up.
    try expectNaiveNest(2, 3, 1);
    try expectNaiveNest(1, 1, 1);
    // A zero extent on either output axis skips the inner loops entirely.
    try expectNaiveNest(0, 3, 4);
    try expectNaiveNest(3, 0, 4);
}

test "ivsr: the naive nest's inner loop loses its index arithmetic" {
    // Without this the tests above would still pass if `ivsr` never fired. The reduction loop must
    // come out smaller, and with no integer scaling left in it at all: both addresses and `p*n`
    // become variables the back edge advances, so every multiply and shift the body used to run on
    // every trip is gone.
    const allocator = std.testing.allocator;

    // `optimizeEarly` is the whole pipeline except the late step, so it is exactly the shape the
    // pass is handed. Nothing else in the optimizer removes this arithmetic.
    var without = try naiveNest(allocator, 2, 4, 3);
    defer without.deinit();
    _ = try opt.optimizeEarly(allocator, &without);
    const before = try innermostLoop(allocator, &without);

    var with = try naiveNest(allocator, 2, 4, 3);
    defer with.deinit();
    _ = try opt.optimize(allocator, &with);
    const after = try innermostLoop(allocator, &with);

    try std.testing.expect(before.scaling); // the rest of the pipeline leaves it there
    try std.testing.expect(!after.scaling); // and this pass takes it out
    try std.testing.expect(after.instructions < before.instructions);

    // And the pass itself reports a change on the shape the rest of the pipeline leaves behind.
    var analyses = opt.pass.Analyses{ .allocator = allocator, .func = &without };
    defer analyses.deinit();
    try std.testing.expect(try opt.ivsr.run(allocator, &without, &analyses));
}
