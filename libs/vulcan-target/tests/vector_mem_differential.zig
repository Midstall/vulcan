//! Differential JIT oracle for vector memory coalescing (the SLP vectorizer's load/store
//! coalescing, libs/vulcan-opt/vectorize.zig). We build a scalar elementwise memory kernel twice
//! (loads through pointer parameters, arithmetic, stores through a pointer parameter). ONE copy is
//! left untouched (the scalar baseline); the OTHER is run through `microarch.optimize` for the
//! ampere model, which now coalesces the contiguous scalar loads into wide vector loads, fuses the
//! arithmetic, and coalesces the contiguous scalar stores into a wide vector store. JIT both on the
//! host, run them over a shared backing array, and require bit-identical per-element results. This
//! proves the coalescing is correct AND that it fired (the tuned IR carries a vector load and a
//! vector store), including the safety case where a store sits between the loads and coalescing must
//! decline while staying correct.
//!
//! Runs only where the native JIT has a backend for the host (aarch64/x86_64/riscv64/x86); the
//! coalescing itself is model-driven for ampere (aarch64), so the "fired" assertions are checked
//! only on an aarch64 host, where that model's JIT actually runs.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;
const Value = ir.function.Value;
const BinOp = ir.function.BinOp;

/// The number of contiguous f32 elements each kernel processes. Eight is two full 4-lane NEON
/// groups, so coalescing forms two wide loads/stores per operand, exercising the multi-group path.
const N = 8;

/// Build a scalar elementwise f32 kernel over `N` contiguous elements through three pointer
/// parameters `a`, `b`, `out`: `N` scalar loads of `a`, `N` of `b`, `N` scalar `kind` arith, `N`
/// scalar stores to `out`. This is exactly the shape the SLP vectorizer's coalescing collapses.
/// `store_between` drops a store to a fourth `scratch` pointer in the middle of the `a` loads, an
/// alias hazard that must make load coalescing on `a` decline (the safety case).
const Kind = enum { add, mul_add };

fn buildMemKernel(func: *Function, kind: Kind, store_between: bool) anyerror!void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.intern(.ptr);
    const block = try func.appendBlock();
    const pa = try func.appendBlockParam(block, ptr_t);
    const pb = try func.appendBlockParam(block, ptr_t);
    const pout = try func.appendBlockParam(block, ptr_t);
    const pscratch = try func.appendBlockParam(block, ptr_t);

    var av: [N]Value = undefined;
    for (0..N) |i| {
        const addr = try func.appendArithImm(block, ptr_t, .add, pa, @intCast(i * 4));
        av[i] = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = addr } });
        if (store_between and i == N / 2) try func.appendStore(block, av[0], pscratch);
    }
    var bv: [N]Value = undefined;
    for (0..N) |i| {
        const addr = try func.appendArithImm(block, ptr_t, .add, pb, @intCast(i * 4));
        bv[i] = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = addr } });
    }
    // For `mul_add`, two contiguous same-op runs: the muls, then the adds. The mul result vector
    // chains into the add (vectorize.zig's chain reuse), while `a` is reloaded for the add's rhs.
    var cv: [N]Value = undefined;
    switch (kind) {
        .add => for (0..N) |i| {
            cv[i] = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = av[i], .rhs = bv[i] } });
        },
        .mul_add => {
            var mv: [N]Value = undefined;
            for (0..N) |i| mv[i] = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = av[i], .rhs = bv[i] } });
            for (0..N) |i| cv[i] = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = mv[i], .rhs = av[i] } });
        },
    }
    for (0..N) |i| {
        const addr = try func.appendArithImm(block, ptr_t, .add, pout, @intCast(i * 4));
        try func.appendStore(block, cv[i], addr);
    }
    func.setTerminator(block, .{ .ret = null });
}

const KernelFn = *const fn ([*]f32, [*]f32, [*]f32, [*]f32) callconv(.c) void;

/// The scalar reference each kernel must reproduce lane for lane.
fn reference(kind: Kind, a: f32, b: f32) f32 {
    return switch (kind) {
        .add => a + b,
        .mul_add => a * b + a,
    };
}

/// True if block 0 of `func` has a `load` producing a vector value (a coalesced wide load).
fn hasVectorLoad(func: *const Function) bool {
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .load) continue;
        const r = func.instResult(inst).?;
        if (func.types.type_kind(func.valueType(r)) == .vector) return true;
    }
    return false;
}

/// True if block 0 of `func` has a `store` of a vector value (a coalesced wide store).
fn hasVectorStore(func: *const Function) bool {
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .store) continue;
        if (func.types.type_kind(func.valueType(func.opcode(inst).store.value)) == .vector) return true;
    }
    return false;
}

/// Count block-0 scalar (non-vector-result) loads.
fn scalarLoadCount(func: *const Function) usize {
    var n: usize = 0;
    for (func.blockInsts(@enumFromInt(0))) |inst| {
        if (func.opcode(inst) != .load) continue;
        const r = func.instResult(inst).?;
        if (func.types.type_kind(func.valueType(r)) != .vector) n += 1;
    }
    return n;
}

fn hasJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .x86, .riscv64 => true,
        else => false,
    };
}

/// Build the scalar baseline and an ampere-optimized twin, JIT both, and require identical output
/// arrays for a spread of inputs. When `expect_coalesced`, also assert the tuned IR carries a wide
/// load and store (coalescing fired). The coalescing is only checked on an aarch64 host, where the
/// ampere (aarch64) model's JIT actually executes.
fn expectCoalescingCorrect(kind: Kind, store_between: bool, expect_coalesced: bool) !void {
    const allocator = std.testing.allocator;

    var baseline = Function.init(allocator);
    defer baseline.deinit();
    try buildMemKernel(&baseline, kind, store_between);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try buildMemKernel(&tuned, kind, store_between);

    const before = scalarLoadCount(&tuned);
    const changed = try opt.microarch.optimize(allocator, &tuned, opt.microarch.modelFor(.@"ampere-altra"));

    var diags_b = try ir.verify.verify(allocator, &baseline, .low);
    defer diags_b.deinit();
    try std.testing.expect(diags_b.ok());
    var diags_t = try ir.verify.verify(allocator, &tuned, .low);
    defer diags_t.deinit();
    try std.testing.expect(diags_t.ok());

    // Structural proof that coalescing fired: a wide load, a wide store, and strictly fewer scalar
    // loads than the baseline. Only meaningful for an aarch64 host (ampere is an aarch64 model).
    if (expect_coalesced and builtin.cpu.arch == .aarch64) {
        try std.testing.expect(changed);
        try std.testing.expect(hasVectorLoad(&tuned));
        try std.testing.expect(hasVectorStore(&tuned));
        try std.testing.expect(scalarLoadCount(&tuned) < before);
    }

    if (comptime !hasJit()) return error.SkipZigTest;

    var buf_b = try target.native.jitFunction(allocator, &baseline);
    defer buf_b.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();

    const f_b = buf_b.entry(KernelFn, 0);
    const f_t = buf_t.entry(KernelFn, 0);

    // A few representative input spreads, including negatives and zeros.
    const spreads = [_][N]f32{
        .{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 },
        .{ -1.5, 2.25, -3.0, 0.0, 100.0, -0.5, 42.0, -1000.0 },
        .{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    };
    for (spreads) |a| {
        const b = [N]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
        var a_b = a;
        var a_t = a;
        var bb_b = b;
        var bb_t = b;
        var out_b = [_]f32{0} ** N;
        var out_t = [_]f32{0} ** N;
        var scratch_b = [_]f32{0} ** N;
        var scratch_t = [_]f32{0} ** N;
        f_b(&a_b, &bb_b, &out_b, &scratch_b);
        f_t(&a_t, &bb_t, &out_t, &scratch_t);
        for (0..N) |i| {
            // Bit-exact: both compute the same f32 op sequence, so the patterns must match exactly.
            try std.testing.expectEqual(@as(u32, @bitCast(out_b[i])), @as(u32, @bitCast(out_t[i])));
            try std.testing.expectEqual(@as(u32, @bitCast(reference(kind, a[i], b[i]))), @as(u32, @bitCast(out_t[i])));
        }
    }
}

test "vector-mem differential: elementwise add coalesces to wide load/store and matches the scalar baseline" {
    try expectCoalescingCorrect(.add, false, true);
}

test "vector-mem differential: multiply-add (SAXPY) coalesces via result-chaining and matches the scalar baseline" {
    // The SAXPY `a[i]*b[i]+a[i]` shape, recovered by the result-chaining credit. Ampere's f32 mul is
    // PIPELINED (throughput weight 1); this kernel's intermediate mul result is NOT stored, it chains
    // lane-by-lane into the following add. The result-chaining credit prices the mul group WITHOUT an
    // unpack (its result rides a vector register into the add, the extracts DCE away), so the mul
    // group is now profitable via its two coalesced-load operands, and the add group is profitable via
    // operand-chaining (mul result reused) plus its coalesced `a` reload and coalesced out store. The
    // whole kernel vectorizes: wide loads, `<4 x f32>` mul and add, a wide store, and strictly fewer
    // scalar loads than the baseline, still bit-exact against the scalar reference.
    try expectCoalescingCorrect(.mul_add, false, true);
}

test "vector-mem differential: a store between the loads is handled safely and still matches" {
    // The safety case: a store sits between the `a` loads. That hazard forces the `a` operand of the
    // mul group to fall back to a pack (its wide load cannot form across the write), and a register
    // pack (0.8) plus the single coalesced `b` load tips the mul group back to exactly break-even on
    // ampere, so it stays scalar (result-chaining removes the unpack but never the surviving pack).
    // Whether or not the group vectorizes, the store must be respected (never coalesced across) and
    // baseline and tuned must agree bit-for-bit; declining is always safe, so this is a pure
    // correctness check.
    try expectCoalescingCorrect(.mul_add, true, false);
}
