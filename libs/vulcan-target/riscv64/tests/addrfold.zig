//! Address-mode folding (Task 4), executed on qemu-riscv64 (the oracle). A load or store whose
//! pointer is a foldable `arith_imm.add(base, imm)` addresses `imm(base)` directly: the address-add
//! is dropped and the mem op carries a signed-12 displacement. riscv64 has no load-pair, so the win
//! is add-elision plus the folded displacement (no ldp/stp).
//!
//! Every buffer lives IN THE GUEST as a run of consecutive `alloca` slots. `computeAllocaSlots` lays
//! same-size, same-alignment allocas out back-to-back, so `buf0 + size*j` is exactly `buf_j`. Each
//! alloca is kept alive by an instruction that references it (an init store, or the address-add that
//! the base feeds), so none is dropped as dead before the frame is sized. The scheduler pins loads and
//! stores in program order, so an init store always precedes the folded load that reads it back.
//!
//! Each test also proves the fold FIRED: `foldedMemOps` decodes the compiled words and counts loads
//! and stores that carry a nonzero displacement off a base other than sp. A spill addresses off sp, so
//! filtering base != x2 isolates the folded memory ops. Without folding every mem op addresses `0(base)`
//! and the count is zero, so a stuck assertion catches a fold that silently stopped firing.

const std = @import("std");
const ir = @import("vulcan-ir");
const isel = @import("../isel.zig");
const harness = @import("harness.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

const i64k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 64 } };
const i32k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 32 } };
const i16k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 16 } };
const i8k = ir.types.TypeKind{ .int = .{ .signedness = .signed, .bits = 8 } };

/// Count loads and stores in `func`'s compiled code that carry a NONZERO immediate displacement off a
/// base register other than sp (x2). A folded constant-index load/store is exactly that: `ld rd,
/// off(base)` / `sd rs2, off(base)` (and the fp/sub-word forms) with off != 0 and base != sp. Spill
/// slots address off sp, so base != x2 excludes them. Zero unless address folding fired.
fn foldedMemOps(allocator: std.mem.Allocator, func: *Function) !usize {
    var words = try harness.compileFunc(allocator, func);
    defer words.deinit(allocator);
    var count: usize = 0;
    for (words.items) |w| {
        const opcode = w & 0x7f;
        const rs1 = (w >> 15) & 0x1f;
        if (rs1 == 2) continue; // sp-relative: a spill slot, not a fold
        const is_load = opcode == 0x03 or opcode == 0x07; // LOAD (int) / LOAD-FP
        const is_store = opcode == 0x23 or opcode == 0x27; // STORE (int) / STORE-FP
        if (is_load) {
            const imm: i32 = @as(i32, @bitCast(w)) >> 20; // sign-extended I-immediate
            if (imm != 0) count += 1;
        } else if (is_store) {
            const hi: i32 = @as(i32, @bitCast(w)) >> 25; // sign-extended imm[11:5]
            const lo: i32 = @intCast((w >> 7) & 0x1f); // imm[4:0]
            if (((hi << 5) | lo) != 0) count += 1;
        }
    }
    return count;
}

/// Run `func` under qemu-riscv64, skipping cleanly when qemu is not installed.
fn run(allocator: std.mem.Allocator, func: *Function, args: []const i64) !i64 {
    return harness.runFunc(std.testing.io, allocator, func, args, harness.qemu_user) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
}

// --- test 1: an i64 copy folds both loads and stores -----------------------------------------------

const copy_n = 4;

/// f(arg): src[j] = arg + j for j in 0..N; dst[j] = src[j] via folded loads/stores; return sum(dst).
/// The copy reads `src0 + 8j` and writes `dst0 + 8j` (both fold for j >= 1), so it exercises a folded
/// load AND a folded store, and the readback re-folds the loads. Expected = sum(arg + j) = 4*arg + 6.
fn buildCopy(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(i64k);
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const arg = try func.appendBlockParam(e, i64_t);

    var src: [copy_n]Value = undefined;
    var dst: [copy_n]Value = undefined;
    for (0..copy_n) |j| src[j] = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    for (0..copy_n) |j| dst[j] = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    // Initialize each src slot (references every src alloca so none is DCE'd) to arg + j.
    for (0..copy_n) |j| {
        const jv = try func.appendInst(e, i64_t, .{ .iconst = @intCast(j) });
        const v = try func.appendInst(e, i64_t, .{ .arith = .{ .op = .add, .lhs = arg, .rhs = jv } });
        try func.appendStore(e, v, src[j]); // own pointer, off 0 (not folded)
    }
    // Zero each dst slot (references every dst alloca).
    for (0..copy_n) |j| {
        const z = try func.appendInst(e, i64_t, .{ .iconst = 0 });
        try func.appendStore(e, z, dst[j]);
    }
    // Copy via folded addresses: v = load(src0 + 8j); store v to (dst0 + 8j).
    for (0..copy_n) |j| {
        const sp = if (j == 0) src[0] else try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = src[0], .imm = @intCast(j * 8) } });
        const v = try func.appendInst(e, i64_t, .{ .load = .{ .ptr = sp } });
        const dp = if (j == 0) dst[0] else try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = dst[0], .imm = @intCast(j * 8) } });
        try func.appendStore(e, v, dp);
    }
    // Read the copy back through folded loads and sum it.
    var acc: ?Value = null;
    for (0..copy_n) |j| {
        const rp = if (j == 0) dst[0] else try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = dst[0], .imm = @intCast(j * 8) } });
        const w = try func.appendInst(e, i64_t, .{ .load = .{ .ptr = rp } });
        acc = if (acc) |a| try func.appendInst(e, i64_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = w } }) else w;
    }
    func.setTerminator(e, .{ .ret = acc.? });
    return func;
}

test "riscv64 addrfold: constant-index i64 copy folds and computes correctly" {
    const allocator = std.testing.allocator;
    var func = try buildCopy(allocator);
    defer func.deinit();

    // The copy and readback fold at least the three nonzero-offset loads and three nonzero-offset
    // stores, so many folded mem ops survive (a zero here would mean folding never fired).
    try std.testing.expect(try foldedMemOps(allocator, &func) >= 3);

    for ([_]i64{ 0, 1, 7, -3, 1000, -12345 }) |arg| {
        const expected: i64 = 4 * arg + 6; // sum(arg + j) for j in 0..4
        const got = run(allocator, &func, &.{arg}) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(expected, got);
    }
}

// --- test 2: byte and halfword folds ---------------------------------------------------------------

/// f(arg): buf[j] = arg + j (as an `elem`-typed slot) for j in 0..4; return sum of folded loads
/// buf0 + size*j. `elem` is i8 (size 1) or i16 (size 2), so a wrong displacement scale would read the
/// wrong element. The sum is taken in i32 to avoid narrow overflow. Expected = sum(trunc(arg + j)).
fn buildNarrowSum(allocator: std.mem.Allocator, comptime kind: ir.types.TypeKind, comptime size: usize) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(i64k);
    const i32_t = try func.types.intern(i32k);
    const elem_t = try func.types.intern(kind);
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const arg = try func.appendBlockParam(e, i64_t);

    var buf: [4]Value = undefined;
    for (0..4) |j| buf[j] = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = elem_t } });
    for (0..4) |j| {
        const jv = try func.appendInst(e, i64_t, .{ .iconst = @intCast(j) });
        const sum = try func.appendInst(e, i64_t, .{ .arith = .{ .op = .add, .lhs = arg, .rhs = jv } });
        const narrow = try func.appendInst(e, elem_t, .{ .convert = .{ .value = sum } }); // truncate to elem
        try func.appendStore(e, narrow, buf[j]); // own pointer, off 0
    }
    var acc: ?Value = null;
    for (0..4) |j| {
        const p = if (j == 0) buf[0] else try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf[0], .imm = @intCast(j * size) } });
        const v = try func.appendInst(e, elem_t, .{ .load = .{ .ptr = p } });
        const w = try func.appendInst(e, i32_t, .{ .convert = .{ .value = v } }); // sign-extend to i32
        acc = if (acc) |a| try func.appendInst(e, i32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = w } }) else w;
    }
    func.setTerminator(e, .{ .ret = acc.? });
    return func;
}

test "riscv64 addrfold: byte and halfword constant-index loads fold and compute" {
    const allocator = std.testing.allocator;

    var byte_fn = try buildNarrowSum(allocator, i8k, 1);
    defer byte_fn.deinit();
    try std.testing.expect(try foldedMemOps(allocator, &byte_fn) >= 1);
    for ([_]i64{ 0, 1, 10, 100, -5 }) |arg| {
        var expected: i32 = 0;
        for (0..4) |j| expected += @as(i8, @truncate(arg + @as(i64, @intCast(j))));
        const got = run(allocator, &byte_fn, &.{arg}) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(@as(i64, expected), got);
    }

    var half_fn = try buildNarrowSum(allocator, i16k, 2);
    defer half_fn.deinit();
    try std.testing.expect(try foldedMemOps(allocator, &half_fn) >= 1);
    for ([_]i64{ 0, 1, 1000, 30000, -7 }) |arg| {
        var expected: i32 = 0;
        for (0..4) |j| expected += @as(i16, @truncate(arg + @as(i64, @intCast(j))));
        const got = run(allocator, &half_fn, &.{arg}) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(@as(i64, expected), got);
    }
}

// --- test 3: word (i32) and floating-point (f32/f64) folds -----------------------------------------

test "riscv64 addrfold: word and fp constant-index loads fold" {
    const allocator = std.testing.allocator;

    // i32 words: buf[j] = arg + j; return sum of folded loads buf0 + 4j.
    var word_fn = try buildNarrowSum(allocator, i32k, 4);
    defer word_fn.deinit();
    try std.testing.expect(try foldedMemOps(allocator, &word_fn) >= 1);
    for ([_]i64{ 0, 3, 55, -9, 100000 }) |arg| {
        var expected: i32 = 0;
        for (0..4) |j| expected +%= @as(i32, @truncate(arg + @as(i64, @intCast(j))));
        const got = run(allocator, &word_fn, &.{arg}) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(@as(i64, expected), got);
    }

    // Floating point: store (f32/f64)arg into s1, read it back through a folded fp load s0 + size, and
    // convert to i64. A wrong displacement would read s0's slot (a different value) or fault.
    var f32_fn = try buildFpField(allocator, .{ .float = .f32 }, 4);
    defer f32_fn.deinit();
    try std.testing.expect(try foldedMemOps(allocator, &f32_fn) >= 1);
    for ([_]i64{ 0, 1, 42, -17, 1234 }) |arg| {
        const got = run(allocator, &f32_fn, &.{arg}) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(arg, got);
    }

    var f64_fn = try buildFpField(allocator, .{ .float = .f64 }, 8);
    defer f64_fn.deinit();
    try std.testing.expect(try foldedMemOps(allocator, &f64_fn) >= 1);
    for ([_]i64{ 0, 1, 42, -17, 987654 }) |arg| {
        const got = run(allocator, &f64_fn, &.{arg}) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(arg, got);
    }
}

/// f(arg): s0, s1 are `float`-typed slots; store (float)arg into s1 (own pointer); load it back
/// through the folded fp address `s0 + size` (= s1); convert back to an integer and return. s0 is kept
/// alive by the address-add that reads it. The int<->float converts route through i32 (the riscv64
/// isel only converts a 32-bit source/destination). Expected = arg (exact for the small sweep values).
fn buildFpField(allocator: std.mem.Allocator, comptime kind: ir.types.TypeKind, comptime size: usize) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(i64k);
    const i32_t = try func.types.intern(i32k);
    const flt_t = try func.types.intern(kind);
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const arg = try func.appendBlockParam(e, i64_t);

    const s0 = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = flt_t } });
    const s1 = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = flt_t } });
    const argw = try func.appendInst(e, i32_t, .{ .convert = .{ .value = arg } }); // i64 -> i32 (narrow)
    const xf = try func.appendInst(e, flt_t, .{ .convert = .{ .value = argw } }); // i32 -> float
    try func.appendStore(e, xf, s1); // own pointer, off 0 (references s1)
    const p = try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = s0, .imm = @intCast(size) } }); // = s1 addr
    const v = try func.appendInst(e, flt_t, .{ .load = .{ .ptr = p } }); // folds to size(s0)
    const r = try func.appendInst(e, i32_t, .{ .convert = .{ .value = v } }); // float -> i32
    func.setTerminator(e, .{ .ret = r });
    return func;
}

// --- test 4: a CROSS-BLOCK folded load (the correctness invariant) ---------------------------------

const xblock_pressure = 8;

/// f(arg, cond): in ENTRY, buf1 = arg and `p = buf0 + 8` (= buf1's address, a DEAD add: its only use
/// is the load in the successor). On cond > 0, `then_b` first builds several live values (so the free
/// list would hand a temp `buf0`'s register if `buf0` were treated as dead after entry), then loads
/// [p] (folds to `8(buf0)`) and reduces. The load's pointer use must be attributed to `buf0` in the
/// SUCCESSOR block, so `buf0` stays live across the branch; miss the reroute and a temp steals its
/// register and the folded load reads a garbage address. Expected(cond>0) = arg + sum(cond + k),
/// k in 1..P; Expected(cond<=0) = cond.
fn buildCrossBlock(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(i64k);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.intern(.ptr);
    const entry = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const arg = try func.appendBlockParam(entry, i64_t);
    const cond = try func.appendBlockParam(entry, i64_t);

    const buf0 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    const buf1 = try func.appendInst(entry, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    try func.appendStore(entry, arg, buf1); // buf1 = arg (own pointer, references buf1)
    // buf0 is referenced only by this address-add, whose result feeds solely the successor's load: a
    // dead add, so buf0's ONLY in-loop-successor liveness flows through the folded load's baseOf.
    const p = try func.appendInst(entry, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf0, .imm = 8 } });
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    const c = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = cond, .rhs = zero } });
    try func.appendIf(entry, c, .{ .target = then_b, .args = &.{} }, .{ .target = else_b, .args = &.{} });

    // then_b: build the pressure BEFORE the folded load so a stolen base register is read by the load.
    var vals: [xblock_pressure]Value = undefined;
    for (0..xblock_pressure) |k| {
        vals[k] = try func.appendInst(then_b, i64_t, .{ .arith_imm = .{ .op = .add, .lhs = cond, .imm = @intCast(k + 1) } });
    }
    const w = try func.appendInst(then_b, i64_t, .{ .load = .{ .ptr = p } }); // folds to 8(buf0) = buf1 = arg
    var acc = w;
    for (0..xblock_pressure) |k| acc = try func.appendInst(then_b, i64_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = vals[k] } });
    func.setTerminator(then_b, .{ .ret = acc });
    func.setTerminator(else_b, .{ .ret = cond });
    return func;
}

test "riscv64 addrfold: a CROSS-BLOCK folded load computes correctly" {
    const allocator = std.testing.allocator;
    var func = try buildCrossBlock(allocator);
    defer func.deinit();

    try std.testing.expect(try foldedMemOps(allocator, &func) >= 1); // the successor load folded

    for ([_]i64{ 1, 0, -1, 5, 100, -7, 42 }) |cond| {
        // Vary arg per case so a stale (wrong) base address could not accidentally match.
        const arg: i64 = 20 + cond;
        var expected: i64 = undefined;
        if (cond > 0) {
            expected = arg;
            var k: i64 = 1;
            while (k <= xblock_pressure) : (k += 1) expected += cond + k;
        } else {
            expected = cond;
        }
        const got = run(allocator, &func, &.{ arg, cond }) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(expected, got);
    }
}

// --- test 5: a base whose add has a NON-folded consumer keeps the add ------------------------------

/// f(arg): buf1 = arg; `p = buf0 + 8` (= buf1's address). `p` feeds the folded load [p] AND a plain
/// `p - buf0` subtraction (a non-folded use), so `p` is NOT dead and the address-add MUST still be
/// emitted. `p - buf0 = 8`, so return `load(8(buf0)) + 8 = arg + 8`. If the add were wrongly elided,
/// the subtraction would read a garbage `p` and the result would diverge.
fn buildLiveAdd(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const i64_t = try func.types.intern(i64k);
    const ptr_t = try func.types.intern(.ptr);
    const e = try func.appendBlock();
    const arg = try func.appendBlockParam(e, i64_t);

    const buf0 = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    const buf1 = try func.appendInst(e, ptr_t, .{ .alloca = .{ .elem = i64_t } });
    try func.appendStore(e, arg, buf1); // buf1 = arg
    const p = try func.appendInst(e, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = buf0, .imm = 8 } }); // = buf1 addr
    const v = try func.appendInst(e, i64_t, .{ .load = .{ .ptr = p } }); // folds to 8(buf0) = arg
    const d = try func.appendInst(e, i64_t, .{ .arith = .{ .op = .sub, .lhs = p, .rhs = buf0 } }); // = 8, keeps p live
    const r = try func.appendInst(e, i64_t, .{ .arith = .{ .op = .add, .lhs = v, .rhs = d } }); // arg + 8
    func.setTerminator(e, .{ .ret = r });
    return func;
}

test "riscv64 addrfold: a base used by a folded load AND another consumer keeps the add and still computes" {
    const allocator = std.testing.allocator;
    var func = try buildLiveAdd(allocator);
    defer func.deinit();

    try std.testing.expect(try foldedMemOps(allocator, &func) >= 1); // the load still folded

    for ([_]i64{ 0, 1, 9, -4, 5000 }) |arg| {
        const got = run(allocator, &func, &.{arg}) catch |e| switch (e) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return e,
        };
        try std.testing.expectEqual(arg + 8, got);
    }
}
