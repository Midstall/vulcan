//! Differential JIT oracle for the IR `prefetch` hint. We build two identical
//! functions that load through a pointer parameter and do arithmetic; ONE copy
//! gets a hand-inserted `prefetch` of that same pointer right after it becomes
//! available. JIT both on the host and require bit-identical results for every
//! input, proving the PRFM (aarch64) / dropped-hint (every other backend)
//! lowering has no observable effect on the function's result.
//!
//! This runs only where the native JIT has a backend (aarch64/x86_64/riscv64/x86).

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;

/// `fn(p: *i64) i64 { return *p + 10; }`. `with_prefetch` hand-inserts a
/// `prefetch` of `p` right after the block parameter (the pointer) is
/// available, before the load reads through it.
fn buildLoadAdd(func: *Function, with_prefetch: bool) anyerror!void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    if (with_prefetch) try func.appendPrefetch(entry, p);
    const val = try func.appendInst(entry, i64_t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendArithImm(entry, i64_t, .add, val, 10);
    func.setTerminator(entry, .{ .ret = sum });
}

/// `fn(p: *i64) i64 { *p = *p * 2; return *p + *p; }`. A second shape that also
/// stores back through the pointer, so the prefetched build and the plain build
/// exercise both a load and a store on the same address, with the prefetch
/// (of a distinct, harmlessly-computed address `p+8`) sitting between them.
/// `p+8` is never dereferenced, only prefetched, so it is fine that it may
/// point one i64 past the caller's real cell: a prefetch never reads memory
/// architecturally, it's a hint only.
fn buildStoreDouble(func: *Function, with_prefetch: bool) anyerror!void {
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const val = try func.appendInst(entry, i64_t, .{ .load = .{ .ptr = p } });
    const doubled = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = val, .rhs = val } });
    try func.appendStore(entry, doubled, p);
    if (with_prefetch) {
        const ahead = try func.appendArithImm(entry, ptr_t, .add, p, 8);
        try func.appendPrefetch(entry, ahead);
    }
    const reloaded = try func.appendInst(entry, i64_t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(entry, i64_t, .{ .arith = .{ .op = .add, .lhs = reloaded, .rhs = reloaded } });
    func.setTerminator(entry, .{ .ret = sum });
}

const Builder = *const fn (*Function, bool) anyerror!void;

/// Build a plain copy and a prefetch-hinted copy of `build`, JIT both, and
/// require identical results (both the returned value AND the pointee left
/// behind in the caller's cell) across a spread of inputs.
fn expectPrefetchIsNoOp(build: Builder) !void {
    const allocator = std.testing.allocator;
    const inputs = [_]i64{ 0, 1, -1, 7, 42, -100, 1 << 40 };

    var plain = Function.init(allocator);
    defer plain.deinit();
    try build(&plain, false);

    var hinted = Function.init(allocator);
    defer hinted.deinit();
    try build(&hinted, true);

    var diags_p = try ir.verify.verify(allocator, &plain, .low);
    defer diags_p.deinit();
    try std.testing.expect(diags_p.ok());
    var diags_h = try ir.verify.verify(allocator, &hinted, .low);
    defer diags_h.deinit();
    try std.testing.expect(diags_h.ok());

    var buf_p = try target.native.jitFunction(allocator, &plain);
    defer buf_p.deinit();
    var buf_h = try target.native.jitFunction(allocator, &hinted);
    defer buf_h.deinit();

    const Fn = *const fn (*i64) callconv(.c) i64;
    const f_p = buf_p.entry(Fn, 0);
    const f_h = buf_h.entry(Fn, 0);

    for (inputs) |n| {
        var cell_p: i64 = n;
        var cell_h: i64 = n;
        const r_p = f_p(&cell_p);
        const r_h = f_h(&cell_h);
        try std.testing.expectEqual(r_p, r_h);
        try std.testing.expectEqual(cell_p, cell_h); // the prefetch changed no memory either
    }
}

test "prefetch differential: load-and-add through a pointer parameter" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectPrefetchIsNoOp(buildLoadAdd);
}

test "prefetch differential: store-double-reload with a prefetch of a computed nearby address" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectPrefetchIsNoOp(buildStoreDouble);
}

test "prefetch differential: the hinted build's aarch64 machine code actually contains a PRFM" {
    if (builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var hinted = Function.init(allocator);
    defer hinted.deinit();
    try buildLoadAdd(&hinted, true);

    const code = try target.native.compile(allocator, &hinted);
    defer allocator.free(code);
    try std.testing.expect(containsPrfm(code));

    // The un-hinted twin must NOT contain one (nothing accidentally fired).
    var plain = Function.init(allocator);
    defer plain.deinit();
    try buildLoadAdd(&plain, false);
    const plain_code = try target.native.compile(allocator, &plain);
    defer allocator.free(plain_code);
    try std.testing.expect(!containsPrfm(plain_code));
}

/// The wide, out-of-order model the prefetch-insertion pass actually transforms for.
fn ampere() *const opt.microarch.Model {
    return opt.microarch.modelFor(.@"ampere-altra");
}

/// `fn(n: i32, arr: *i64) i64`: `s = 0; p = arr; for (i = 0; i < n; i += 1) { s += *p; p += 8; }
/// return s`. `p` is a pointer-typed loop-carried value that steps by a constant 8-byte stride each
/// iteration and is dereferenced directly each time: exactly the affine-strided-load shape
/// `microarch.prefetch.run` looks for. This is the real oracle for Task 3: the pass reads real
/// loop/load structure (not a hand-inserted prefetch like the tests above), so it proves both that
/// eligibility fires on a genuine strided loop AND that the inserted hint changes nothing.
fn buildStridedSum(func: *Function) anyerror!void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i32_t);
    const arr = try func.appendBlockParam(entry, ptr_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const p = try func.appendBlockParam(loop, ptr_t);
    const s = try func.appendBlockParam(loop, i64_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const bp = try func.appendBlockParam(body, ptr_t);
    const bs = try func.appendBlockParam(body, i64_t);
    const ds = try func.appendBlockParam(done, i64_t);

    const zero32 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const zero64 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero32, arr, zero64 });

    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, p, s } }, .{ .target = done, .args = &.{s} });

    const val = try func.appendInst(body, i64_t, .{ .load = .{ .ptr = bp } });
    const ns = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = val } });
    const np = try func.appendArithImm(body, ptr_t, .add, bp, 8);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, np, ns });

    func.setTerminator(done, .{ .ret = ds });
}

test "prefetch differential: strided-reduction loop over a real array (sum a[i] for i in 0..n)" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var plain = Function.init(allocator);
    defer plain.deinit();
    try buildStridedSum(&plain);

    var hinted = Function.init(allocator);
    defer hinted.deinit();
    try buildStridedSum(&hinted);

    // The pass, not the test, decides where a prefetch goes: this is the real insertion pass
    // running over genuine loop/load structure, not a hand-placed prefetch like the tests above.
    const changed = try opt.microarch.prefetch.run(allocator, &hinted, ampere());
    try std.testing.expect(changed); // the loop and its load are eligible; something must fire

    var diags_p = try ir.verify.verify(allocator, &plain, .low);
    defer diags_p.deinit();
    try std.testing.expect(diags_p.ok());
    var diags_h = try ir.verify.verify(allocator, &hinted, .low);
    defer diags_h.deinit();
    try std.testing.expect(diags_h.ok());

    var buf_p = try target.native.jitFunction(allocator, &plain);
    defer buf_p.deinit();
    var buf_h = try target.native.jitFunction(allocator, &hinted);
    defer buf_h.deinit();

    const Fn = *const fn (i64, [*]i64) callconv(.c) i64;
    const f_p = buf_p.entry(Fn, 0);
    const f_h = buf_h.entry(Fn, 0);

    // A backing array long enough for the largest n exercised below (16).
    var backing = [_]i64{ 10, -3, 42, 7, -100, 1000, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const inputs = [_]i64{ 0, 1, 2, 5, 16 };
    for (inputs) |n| {
        const r_p = f_p(n, &backing);
        const r_h = f_h(n, &backing);
        try std.testing.expectEqual(r_p, r_h);
    }
}

/// Whether `code` (host machine bytes) contains a `PRFM (immediate)` word. The
/// encoding is `0xF9800000 | (Rn << 5)`; every bit outside the 5-bit Rn field
/// (bits [9:5]) is fixed, so masking those bits out and comparing the rest
/// finds the instruction regardless of which register the allocator chose.
fn containsPrfm(code: []const u8) bool {
    const rn_field_mask: u32 = 0x1F << 5;
    var i: usize = 0;
    while (i + 4 <= code.len) : (i += 4) {
        const word = std.mem.readInt(u32, code[i..][0..4], .little);
        if ((word & ~rn_field_mask) == 0xF9800000) return true;
    }
    return false;
}

/// Whether the native JIT has a backend for the host architecture.
fn hasJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .x86, .riscv64 => true,
        else => false,
    };
}
