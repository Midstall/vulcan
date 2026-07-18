//! Zicbop `prefetch.r` differential oracle, executed under real qemu-riscv64 (not the native
//! JIT: qemu runs the compiled code in its own child process, so unlike
//! `libs/vulcan-target/tests/prefetch_differential.zig` this cannot hand a host pointer across
//! the process boundary; the "backing array" instead lives in the executed program's own stack,
//! see `build` below).
//!
//! A prefetch is a hint with no observable effect, and Zicbop's `prefetch.r` is encoded
//! ORI-shaped (opcode 0b0010011, funct3 0b110, rd=x0; see encode.zig), so it decodes as a
//! harmless `ori x0, rs1, imm` no-op on any hardware, Zicbop or not. That makes this the one
//! prefetch-lowering path that is actually execution-validatable: we build a strided
//! pointer-walking i64 reduction loop, model-drive-prefetch ONE copy under river-rc1.f (an
//! RV64GC application tier that now carries Zicbop, see registry.zig), compile the tuned copy
//! through `isel.selectFunctionForModel` (which lowers the inserted `.prefetch` hint to a real
//! `prefetch.r`) and the plain copy through `isel.selectFunction`, run BOTH under qemu-riscv64,
//! and require bit-identical results. We also assert structurally that the tuned code actually
//! contains a `prefetch_r` word, so the lowering is proven to have fired, not merely to be
//! harmless if it never does.
//!
//! Skips when qemu-riscv64 is not on PATH (same policy as every other qemu_user* test here).

const std = @import("std");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const isel = @import("../isel.zig");
const schedule = @import("../schedule.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;

/// The RV64GC application tier this differential targets: the one river profile (along with
/// river-rc1.ma) that carries Zicbop, per River's RVA22/RVA23 profile
/// (river/packages/river/lib/src/profiles.dart: `rvZicbop`).
fn riverF() *const opt.microarch.Model {
    return opt.microarch.modelFor(.@"river-rc1.f");
}

/// A 16-element i64 array's worth of known values, matching
/// `prefetch_differential.zig`'s backing array so the two oracles exercise the same data shape.
const backing_vals = [_]i64{ 10, -3, 42, 7, -100, 1000, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

/// `fn(n: i32) i64`: `arr = {backing_vals embedded on the stack}; s = 0; p = &arr[0];
/// for (i = 0; i < n; i += 1) { s += *p; p += 8; } return s`.
///
/// There is no host process for a qemu-run child to reach into, so the "real backing array" the
/// strided load walks has to live inside the compiled program's own stack memory rather than
/// being passed in as a pointer argument (contrast `prefetch_differential.zig`'s
/// `buildStridedSum`, which is JIT'd in-process and can just take a host array's address).
/// `alloca`'s frame slots are assigned in declaration order (see `isel.zig`'s frame-layout scan),
/// so declaring one `alloca` per element, all before anything else, lays them out at consecutive
/// 8-byte offsets: exactly a real, 16-element i64 array in the executed program's own memory.
/// Only the first alloca's pointer is kept (`base`); the rest are addressed via `base + i*8`.
fn build(func: *Function) anyerror!void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);

    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(loop, i32_t);
    const p = try func.appendBlockParam(loop, ptr_t);
    const s = try func.appendBlockParam(loop, i64_t);
    const bi = try func.appendBlockParam(body, i32_t);
    const bp = try func.appendBlockParam(body, ptr_t);
    const bs = try func.appendBlockParam(body, i64_t);
    const ds = try func.appendBlockParam(done, i64_t);

    const base = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    for (backing_vals[1..]) |_| _ = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    for (backing_vals, 0..) |v, idx| {
        const cv = try func.appendInst(entry, i64_t, .{ .iconst = v });
        const addr = if (idx == 0) base else try func.appendArithImm(entry, ptr_t, .add, base, @as(i64, @intCast(idx * 8)));
        try func.appendStore(entry, cv, addr);
    }
    const zero32 = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const zero64 = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero32, base, zero64 });

    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, p, s } }, .{ .target = done, .args = &.{s} });

    const val = try func.appendInst(body, i64_t, .{ .load = .{ .ptr = bp } });
    const ns = try func.appendInst(body, i64_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = val } });
    const np = try func.appendArithImm(body, ptr_t, .add, bp, 8);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, np, ns });

    func.setTerminator(done, .{ .ret = ds });
}

/// Compile `func` through the same legalize/split/schedule pipeline `harness.compileFunc` uses,
/// but through `isel.selectFunctionForModel` instead of plain `isel.selectFunction`, so a model's
/// capabilities (here, Zicbop) actually reach codegen. Caller owns the result.
fn compileForModel(allocator: std.mem.Allocator, func: *Function, model: *const opt.microarch.Model) !std.ArrayList(u32) {
    try ir.legalize.legalize(allocator, func);
    try isel.splitCriticalEdges(allocator, func);
    try schedule.scheduleFunction(allocator, func);
    const code = try isel.selectFunctionForModel(allocator, func, model);
    defer allocator.free(code);
    var list: std.ArrayList(u32) = .empty;
    try list.appendSlice(allocator, code);
    return list;
}

/// Whether `code` contains a Zicbop `prefetch.r` word: opcode 0b0010011, funct3 0b110, rd x0, and
/// the rs2/variant field (bits [24:20]) set to `00001` (prefetch.r, not `.i`/`.w`). Every bit
/// outside rs1 (bits [19:15]) and the offset (bits [31:25]) is fixed, so masking those two fields
/// out and comparing the rest finds the instruction regardless of which register/offset codegen
/// chose.
fn containsPrefetchR(code: []const u32) bool {
    const rs1_and_offset_mask: u32 = (0x1f << 15) | (0x7f << 25);
    for (code) |word| {
        if ((word & ~rs1_and_offset_mask) == 0x00106013) return true;
    }
    return false;
}

test "zicbop differential: strided-reduction loop under river-rc1.f gets a prefetch.r that changes nothing under qemu-riscv64" {
    const allocator = std.testing.allocator;
    const model = riverF();
    // Sanity: river-rc1.f is the model this whole oracle is about (registry.zig sets zicbop only
    // on it and river-rc1.ma), and prefetches() is what actually gates the insertion pass.
    try std.testing.expect(model.features.riscv64.zicbop);
    try std.testing.expect(model.prefetches());

    var plain = Function.init(allocator);
    defer plain.deinit();
    try build(&plain);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try build(&tuned);

    // The insertion pass, not the test, decides where a prefetch goes: this is the real pass
    // running over genuine loop/load structure (the strided `p += 8` walk over `bp`), not a
    // hand-placed prefetch.
    const changed = try opt.microarch.prefetch.run(allocator, &tuned, model);
    try std.testing.expect(changed); // the loop and its load are eligible; something must fire

    var diags_p = try ir.verify.verify(allocator, &plain, .low);
    defer diags_p.deinit();
    try std.testing.expect(diags_p.ok());
    var diags_t = try ir.verify.verify(allocator, &tuned, .low);
    defer diags_t.deinit();
    try std.testing.expect(diags_t.ok());

    var baseline_words = try harness.compileFunc(allocator, &plain);
    defer baseline_words.deinit(allocator);
    var tuned_words = try compileForModel(allocator, &tuned, model);
    defer tuned_words.deinit(allocator);

    // Structural proof the lowering actually fired (and that nothing fires without the model).
    try std.testing.expect(containsPrefetchR(tuned_words.items));
    try std.testing.expect(!containsPrefetchR(baseline_words.items));

    // If qemu-riscv64 is absent, `runCode` returns `error.SkipZigTest` (see `runProgram`'s
    // `FileNotFound` handling), which propagates straight through `try` and skips this test,
    // exactly like every other qemu_user* test in this directory.
    const inputs = [_]i64{ 0, 1, 2, 5, 16 };
    for (inputs) |n| {
        const r_p = try harness.runCode(std.testing.io, allocator, baseline_words.items, &.{n}, harness.qemu_user);
        const r_t = try harness.runCode(std.testing.io, allocator, tuned_words.items, &.{n}, harness.qemu_user);
        try std.testing.expectEqual(r_p, r_t);
    }
}
